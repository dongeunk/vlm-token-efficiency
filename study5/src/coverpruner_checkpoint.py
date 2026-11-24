"""Atomic append-only checkpoints. No model imports; uncertain generation never repeats."""
from contextlib import contextmanager
from pathlib import Path
import copy,hashlib,json,math,os,re,uuid

class AlreadyCompleted(RuntimeError):pass
class AmbiguousExecution(RuntimeError):pass
class Busy(RuntimeError):pass
class CorruptCheckpoint(RuntimeError):pass


def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(encode(value)).hexdigest()


def atomic(path,value,limit=32768):
    data=encode(value)
    if len(data)>limit:raise ValueError('Record exceeds32KiB')
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with open(tmp,'xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    if path.read_bytes()!=data:raise CorruptCheckpoint('Atomic write read-back failed')


def envelope(payload):return dict(payload=payload,sha256=digest(payload))
def read_envelope(path):
    if path.stat().st_size>32768:raise CorruptCheckpoint('Oversize record')
    obj=json.loads(path.read_text(encoding='utf-8'))
    if set(obj)!={'payload','sha256'} or digest(obj['payload'])!=obj['sha256']:raise CorruptCheckpoint('Invalid checksum')
    return obj['payload'],obj['sha256']


class CheckpointStore:
    @classmethod
    def create(cls,parent,run_id,config):
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}',run_id):raise ValueError('Invalid Run ID')
        if not config.get('methods') or type(config.get('generation_budget')) is not int or config['generation_budget']<0:raise ValueError('Invalid run config')
        if type(config.get('layers')) is not int or config['layers']<1 or type(config.get('max_new_tokens')) is not int or not 1<=config['max_new_tokens']<=10:raise ValueError('Invalid runtime limits')
        if any(not re.fullmatch(r'[A-Za-z0-9_-]{1,48}',k) or v.get('stages') not in (['selection','prefill'],['selection','prefill','decode']) for k,v in config['methods'].items()):raise ValueError('Invalid method key/stage plan')
        path=Path(parent).resolve()/run_id;path.mkdir(exist_ok=False)
        for name in ('journal','objects','reservations'):(path/name).mkdir()
        payload=dict(run_id=run_id,config=config)
        atomic(path/'CONFIG.json',envelope(payload))
        store=cls(path)
        state=dict(run_id=run_id,config_sha256=digest(payload),status='pending',methods={
            key:dict(status='pending',stages={},diagnostics={},generation_reserved=False) for key in config['methods']})
        with store.lock():store._commit(state,None)
        return store

    def __init__(self,path,expected_config=None):
        self.path=Path(path).resolve();self.header,self.config_hash=read_envelope(self.path/'CONFIG.json')
        if self.header['run_id']!=self.path.name:raise CorruptCheckpoint('Run identity mismatch')
        self.config=self.header['config'];self.recovery_issues=[]
        if expected_config is not None and expected_config!=self.config:raise CorruptCheckpoint('Resume configuration differs')

    @contextmanager
    def lock(self,name='writer.lock'):
        if name!='writer.lock' and not re.fullmatch(r'method-[A-Za-z0-9_-]{1,48}\.lock',name):raise ValueError('Invalid lock path')
        with open(self.path/name,'a+b') as f:
            f.seek(0,2)
            if f.tell()==0:f.write(b'0');f.flush()
            f.seek(0)
            try:
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as exc:raise Busy('Checkpoint writer locked') from exc
            try:yield
            finally:
                f.seek(0)
                if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
                else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

    def _object(self,ref):
        name=ref['file']
        if not re.fullmatch(r'[a-f0-9]{32}\.json',name):raise CorruptCheckpoint('Invalid object path')
        payload,checksum=read_envelope(self.path/'objects'/name)
        if checksum!=ref['sha256'] or payload['run_id']!=self.header['run_id']:raise CorruptCheckpoint('Object identity/hash mismatch')
        return payload['data']

    def _load(self):
        good=None;previous=None;issues=[]
        for file in sorted((self.path/'journal').glob('cp-*.json')):
            try:
                payload,h=read_envelope(file)
                if payload['previous']!=previous:raise CorruptCheckpoint('Unlinked journal record')
                state=payload['state']
                if state['run_id']!=self.header['run_id'] or state['config_sha256']!=self.config_hash:raise CorruptCheckpoint('Config drift')
                for item in state['methods'].values():
                    for ref in item['stages'].values():self._object(ref)
                    for ref in item['diagnostics'].values():self._object(ref)
                good=state;previous=h
            except (OSError,ValueError,KeyError,TypeError,CorruptCheckpoint) as exc:issues.append(file.name+': '+str(exc))
        self.recovery_issues=issues[-16:]
        if good is None:raise CorruptCheckpoint('No valid checkpoint')
        return copy.deepcopy(good),previous

    def _status(self,state):
        items=list(state['methods'].values())
        if any(m['status']=='failed' or any(self._object(r)['status']=='failed' for r in m['diagnostics'].values()) for m in items):return 'failed'
        if all(m['status']=='completed' for m in items):
            required_ok=all(all(name in m['diagnostics'] and self._object(m['diagnostics'][name])['status']=='completed' for name in self.config['methods'][key].get('required_diagnostics',[])) for key,m in state['methods'].items())
            return 'completed' if required_ok else 'running'
        return 'running' if any(m['status']=='running' for m in items) else 'pending'

    def _commit(self,state,previous):
        state['status']=self._status(state)
        indices=[int(p.stem[3:]) for p in (self.path/'journal').glob('cp-*.json') if re.fullmatch(r'cp-\d{8}',p.stem)]
        number=max(indices,default=0)+1
        atomic(self.path/'journal'/f'cp-{number:08d}.json',envelope(dict(previous=previous,state=state)))
        # Verify referenced data before acknowledging durable state.
        verified,_=self._load()
        if verified!=state:raise CorruptCheckpoint('Checkpoint verification failed')
        if sum(p.stat().st_size for p in self.path.rglob('*') if p.is_file())>512*1024**2:raise ValueError('Run exceeds512MiB')

    def snapshot(self):
        with self.lock():return self._load()[0]

    def data(self,key,stage):
        with self.lock():return self._object(self._load()[0]['methods'][key]['stages'][stage])

    def _change(self,callback):
        with self.lock():
            state,previous=self._load();callback(state);self._commit(state,previous)

    def begin(self,key):
        def change(state):
            item=state['methods'][key]
            if item['status']=='completed':raise AlreadyCompleted(key)
            if (self.path/'reservations'/(key+'.json')).exists() or item['generation_reserved']:raise AmbiguousExecution('Generation was reserved; never auto-repeat: '+key)
            if item['status']=='running':raise Busy('Interrupted/running method requires explicit failure recovery')
            item.update(status='running');state['status']='running'
        self._change(change)

    def reserve_generation(self,key):
        with self.lock():
            state,previous=self._load();item=state['methods'][key]
            if item['status']!='running' or item['generation_reserved']:raise AmbiguousExecution('Invalid/reserved generation')
            reservations=list((self.path/'reservations').glob('*.json'))
            if len(reservations)>=self.config['generation_budget']:raise RuntimeError('Generation budget exhausted/forbidden')
            target=self.path/'reservations'/(key+'.json')
            # Exclusive durable tombstone: even a torn reservation blocks repeat.
            with open(target,'xb') as f:f.write(encode(envelope(dict(run_id=self.path.name,method=key))));f.flush();os.fsync(f.fileno())
            read_envelope(target)
            item['generation_reserved']=True;self._commit(state,previous)

    def _validate(self,key,stage,data,state):
        required=self.config['methods'][key]['stages'];item=state['methods'][key]
        if stage not in required:raise ValueError('Unrequested stage')
        previous=required[:required.index(stage)]
        if any(s not in item['stages'] for s in previous):raise ValueError('Stage order mismatch')
        if stage=='selection':
            ids=data['selected_ids'];seed=data['seed_ids']
            if len(ids)!=64 or len(set(ids))!=64 or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in ids):raise ValueError('Unique K64 required')
            if len(seed) not in (0,25) or len(set(seed))!=len(seed) or not set(seed)<=set(ids):raise ValueError('Invalid seed IDs')
            if not math.isfinite(data['coverage']) or data['coverage']<0:raise ValueError('Invalid Coverage')
        elif stage=='prefill':
            expected=0 if data.get('cache_mode')=='none' else data['input_length']
            if data['visual_tokens']!=64 or data['cache_before']!=0 or data['cache_after']!=expected or data['cache_lengths']!=[expected]*self.config['layers']:raise ValueError('Cold-KV/no-cache prefill mismatch')
        elif stage=='decode':
            tokens=data['generated_ids'];length=self._object(item['stages']['prefill'])['input_length']
            if not 1<=len(tokens)<=self.config['max_new_tokens'] or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in tokens):raise ValueError('Invalid generated token IDs')
            if data['cache_length']!=length+len(tokens)-1:raise ValueError('Decode cache length mismatch')
            if stage in item['stages']:
                old=self._object(item['stages'][stage]);prefix=old['generated_ids']
                if tokens[:len(prefix)]!=prefix or old.get('complete'):raise ValueError('Decode prefix overwrite/repeat')

    def record(self,key,stage,data):
        encode(data) # reject NaN, tensors and unsupported types BEFORE any write
        def change(state):
            item=state['methods'][key]
            if item['status']!='running':raise ValueError('Method not running')
            if stage in item['stages'] and stage!='decode':raise ValueError('Immutable stage already saved')
            self._validate(key,stage,data,state)
            name=uuid.uuid4().hex+'.json';obj=envelope(dict(run_id=self.path.name,data=data))
            atomic(self.path/'objects'/name,obj);item['stages'][stage]=dict(file=name,sha256=obj['sha256'])
        self._change(change)

    def complete(self,key):
        def change(state):
            item=state['methods'][key];required=self.config['methods'][key]['stages']
            if item['status']!='running' or any(s not in item['stages'] for s in required):raise ValueError('Incomplete method cannot complete')
            if 'decode' in required and (not item['generation_reserved'] or not (self.path/'reservations'/(key+'.json')).exists()):raise ValueError('Decode lacks durable Generation reservation')
            if 'decode' in required and (not item['generation_reserved'] or not (self.path/'reservations'/(key+'.json')).exists()):raise AmbiguousExecution('Decode completion requires durable generation reservation')
            for stage in required:
                data=self._object(item['stages'][stage])
                if stage=='decode' and not data.get('complete'):raise ValueError('Partial decode cannot complete')
                if stage!='decode':self._validate(key,stage,data,state)
            item['status']='completed'
            state['status']='completed' if all(m['status']=='completed' for m in state['methods'].values()) else 'running'
        self._change(change)

    def fail(self,key,error):
        def change(state):
            item=state['methods'][key]
            if item['status']!='completed':item['status']='failed'
            item['error']=str(error)[:1200];state['status']='failed'
        self._change(change)

    def diagnostic(self,key,name,data,passed):
        # Persist errors/top1/etc BEFORE the caller raises a parity exception.
        value=dict(status='completed' if passed else 'failed',data=data)
        def change(state):
            item=state['methods'][key]
            if name in item['diagnostics']:raise ValueError('Diagnostic immutable/already saved')
            obj=envelope(dict(run_id=self.path.name,data=value));filename=uuid.uuid4().hex+'.json'
            atomic(self.path/'objects'/filename,obj);item['diagnostics'][name]=dict(file=filename,sha256=obj['sha256'])
            if not passed:state['status']='failed'
        self._change(change)

    def diagnostic_data(self,key,name):
        with self.lock():return self._object(self._load()[0]['methods'][key]['diagnostics'][name])
    def consume_legacy(self,key,evidence):
        """Import only a prior consumed attempt; never fabricate lost stage outputs."""
        if not evidence.get('path') or not re.fullmatch(r'[a-f0-9]{64}',evidence.get('sha256','')):raise ValueError('Legacy evidence required')
        with self.lock():
            state,previous=self._load();item=state['methods'][key]
            if item['status']!='pending' or item['stages']:raise ValueError('Legacy method already used')
            target=self.path/'reservations'/(key+'.json')
            with open(target,'xb') as f:f.write(encode(envelope(dict(run_id=self.path.name,method=key,legacy=evidence))));f.flush();os.fsync(f.fileno())
            item.update(status='failed',generation_reserved=True,legacy_evidence=evidence,error='Prior execution consumed; artifacts unavailable; do not regenerate')
            state['status']='failed';self._commit(state,previous)