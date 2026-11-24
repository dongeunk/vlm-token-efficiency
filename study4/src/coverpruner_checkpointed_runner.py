"""Stagewise orchestrator; callbacks reuse immutable selection/Decoder engines."""
from coverpruner_checkpoint import AlreadyCompleted,AmbiguousExecution


def execute_method(store,key,select,prefill,decode=None,*,generation=False,optional=None,resume=False):
    if key not in store.config['methods']:raise KeyError(key)
    # OS lease is released on a hard process exit, never on a second active caller.
    with store.lock('method-'+key+'.lock'):
        item=store.snapshot()['methods'][key]
        if item['status']=='completed':raise AlreadyCompleted(key)
        if resume and item['status']=='running':store.fail(key,'Explicit recovery after interrupted process')
        store.begin(key) # durable reservation also blocks corrupted-tail retries
        try:
            item=store.snapshot()['methods'][key]
            if 'selection' in item['stages']:selected=store.data(key,'selection')
            else:selected=select();store.record(key,'selection',selected)
            if 'prefill' in item['stages']:
                if decode is not None:raise AmbiguousExecution('KV tensors are not persisted; never resume partial generation automatically')
                session=None # metadata can complete; no hidden-state/cache reconstruction
            else:
                if generation:store.reserve_generation(key)
                session,prefill_record=prefill(selected);store.record(key,'prefill',prefill_record)
                if decode is not None:
                    def progress(record):store.record(key,'decode',record)
                    final=decode(session,progress);store.record(key,'decode',final)
            store.complete(key)
        except BaseException as exc:
            store.fail(key,repr(exc));raise
        if optional is not None:
            try:
                name,data,passed=optional(session)
                store.diagnostic(key,name,data,passed)
                if not passed:raise RuntimeError('Optional diagnostic failed: '+name)
            except BaseException as exc:
                store.diagnostic(key,'optional_exception',dict(error=repr(exc)),False)
                store.fail(key,repr(exc));raise
        return session