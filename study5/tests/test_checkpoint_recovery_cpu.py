"""Fault-injection checkpoints, abrupt exits, corruption and at-most-once execution."""
from pathlib import Path
import copy,json,os,subprocess,sys,tempfile,unittest
from unittest.mock import patch
from coverpruner_checkpoint import CheckpointStore,AlreadyCompleted,AmbiguousExecution,Busy,CorruptCheckpoint,atomic,envelope
from coverpruner_checkpointed_runner import execute_method
ROOT=Path(__file__).resolve().parents[1]


def config(generation=False,budget=1,methods=('A',),diagnostics=()):
    return dict(generation_budget=budget,layers=2,max_new_tokens=10,methods={m:dict(stages=['selection','prefill']+(['decode'] if generation else []),required_diagnostics=list(diagnostics)) for m in methods})
def selected():return dict(selected_ids=list(range(64)),seed_ids=[],coverage=12.5,reference_coverage=12.5000001,problem_fingerprints={'A':'a'*64,'w':'b'*64})
def prefill():return dict(visual_tokens=64,input_length=70,cache_before=0,cache_after=70,cache_lengths=[70,70])
def decode(tokens=(1,2),complete=True):return dict(generated_ids=list(tokens),cache_length=70+len(tokens)-1,complete=complete)


class CheckpointRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.parent=Path(tempfile.mkdtemp(prefix='checkpoint-',dir=tempfile.gettempdir()))
    def store(self,**kwargs):return CheckpointStore.create(self.parent,'test-run',config(**kwargs))
    def child(self,body):
        pre="import sys,os;sys.path.insert(0,"+repr(str(ROOT/'src'))+");from coverpruner_checkpoint import *;s=CheckpointStore("+repr(str(self.parent/'test-run'))+");"
        return subprocess.run([sys.executable,'-I','-B','-c',pre+body],capture_output=True,timeout=10)
    def finish(self,s,key='A'):
        s.begin(key);s.record(key,'selection',selected());s.record(key,'prefill',prefill());s.complete(key)

    def test_unique_run_config_recorded_pending(self):
        s=self.store();self.assertEqual(s.snapshot()['status'],'pending')
        self.assertTrue((s.path/'CONFIG.json').exists())
        with self.assertRaises(FileExistsError):CheckpointStore.create(self.parent,'test-run',config())
    def test_invalid_run_method_paths_rejected(self):
        with self.assertRaises(ValueError):CheckpointStore.create(self.parent,'../escape',config())
        bad=config();bad['methods']={'../escape':dict(stages=['selection','prefill'])}
        with self.assertRaises(ValueError):CheckpointStore.create(self.parent,'bad',bad)
    def test_resume_configuration_drift_rejected(self):
        s=self.store();different=config();different['max_new_tokens']=9
        with self.assertRaises(CorruptCheckpoint):CheckpointStore(s.path,expected_config=different)
    def test_pending_running_completed_after_verified_records(self):
        s=self.store();s.begin('A');self.assertEqual(s.snapshot()['status'],'running')
        s.record('A','selection',selected());s.record('A','prefill',prefill());s.complete('A')
        self.assertEqual(s.snapshot()['status'],'completed');self.assertEqual(s.data('A','selection'),selected())
    def test_selection_saved_before_prefill_failure(self):
        s=self.store()
        def crash(_):raise RuntimeError('prefill failed')
        with self.assertRaises(RuntimeError):execute_method(s,'A',selected,crash)
        self.assertEqual(s.data('A','selection'),selected());self.assertEqual(s.snapshot()['methods']['A']['status'],'failed')
    def test_prefill_saved_before_decode_failure(self):
        s=self.store(generation=True)
        def crash(*a):raise RuntimeError('decode failed')
        with self.assertRaises(RuntimeError):execute_method(s,'A',selected,lambda _:('session',prefill()),crash,generation=True)
        self.assertEqual(s.data('A','prefill'),prefill())
        with self.assertRaises(AmbiguousExecution):s.begin('A')
    def test_partial_decode_survives_exception(self):
        s=self.store(generation=True)
        def crash(_,progress):progress(decode((1,),False));raise RuntimeError('partial stop')
        with self.assertRaises(RuntimeError):execute_method(s,'A',selected,lambda _:('session',prefill()),crash,generation=True)
        self.assertEqual(s.data('A','decode')['generated_ids'],[1]);self.assertFalse(s.data('A','decode')['complete'])
    def test_optional_failure_keeps_completed_core_and_error(self):
        s=self.store(diagnostics=('padding',))
        with self.assertRaises(RuntimeError):execute_method(s,'A',selected,lambda _:('session',prefill()),optional=lambda _:('padding',{'mean_abs':.02,'top1':True},False))
        self.assertEqual(s.snapshot()['methods']['A']['status'],'completed');self.assertEqual(s.snapshot()['status'],'failed')
        self.assertEqual(s.diagnostic_data('A','padding')['data']['mean_abs'],.02)
        with self.assertRaises(AlreadyCompleted):s.begin('A')
    def test_completed_method_never_calls_callbacks_again(self):
        s=self.store();calls=[]
        def callback():calls.append(1);return selected()
        execute_method(s,'A',callback,lambda _:('session',prefill()))
        with self.assertRaises(AlreadyCompleted):execute_method(s,'A',callback,lambda _:('session',prefill()))
        self.assertEqual(calls,[1])
    def test_missing_stage_cannot_complete(self):
        s=self.store();s.begin('A');s.record('A','selection',selected())
        with self.assertRaises(ValueError):s.complete('A')
        self.assertNotEqual(s.snapshot()['methods']['A']['status'],'completed')
    def test_partial_decode_cannot_complete(self):
        s=self.store(generation=True);s.begin('A');s.record('A','selection',selected());s.reserve_generation('A');s.record('A','prefill',prefill());s.record('A','decode',decode((1,),False))
        with self.assertRaises(ValueError):s.complete('A')
        s.record('A','decode',decode((1,2),True));s.complete('A');self.assertEqual(s.snapshot()['status'],'completed')
    def test_immutable_selection_and_decode_prefix(self):
        s=self.store(generation=True);s.begin('A');s.record('A','selection',selected())
        with self.assertRaises(ValueError):s.record('A','selection',selected())
        s.reserve_generation('A');s.record('A','prefill',prefill());s.record('A','decode',decode((1,),False))
        with self.assertRaises(ValueError):s.record('A','decode',decode((3,4),False))
    def test_nan_or_duplicate_selection_rejected_without_state_loss(self):
        s=self.store();s.begin('A');bad=selected();bad['coverage']=float('nan')
        with self.assertRaises(ValueError):s.record('A','selection',bad)
        bad=selected();bad['selected_ids'][1]=0
        with self.assertRaises(ValueError):s.record('A','selection',bad)
        self.assertEqual(s.snapshot()['methods']['A']['stages'],{})
    def test_zero_generation_budget_enforced(self):
        s=self.store(budget=0);s.begin('A')
        with self.assertRaises(RuntimeError):s.reserve_generation('A')
        self.assertEqual(list((s.path/'reservations').glob('*')),[])
    def test_generation_budget_counts_failed_reserved_attempt(self):
        s=self.store(generation=True,methods=('A','B'),budget=1);s.begin('A');s.reserve_generation('A');s.fail('A','interrupted');s.begin('B')
        with self.assertRaises(RuntimeError):s.reserve_generation('B')
    def test_hard_exit_after_reservation_before_journal_blocks_retry(self):
        s=self.store(generation=True);s.begin('A')
        child=self.child("s._commit=lambda *a:os._exit(23);s.reserve_generation('A')")
        self.assertEqual(child.returncode,23)
        reopened=CheckpointStore(s.path)
        with self.assertRaises(AmbiguousExecution):reopened.begin('A')
    def test_hard_exit_after_prefill_keeps_prior_stages(self):
        s=self.store();s.begin('A');s.record('A','selection',selected())
        child=self.child("s.record('A','prefill',"+repr(prefill())+");os._exit(17)")
        self.assertEqual(child.returncode,17);reopened=CheckpointStore(s.path)
        self.assertEqual(reopened.data('A','prefill'),prefill());self.assertEqual(reopened.snapshot()['methods']['A']['status'],'running')
        reopened.complete('A');self.assertEqual(reopened.snapshot()['status'],'completed')
    def test_hard_exit_during_atomic_replace_keeps_old_checkpoint(self):
        s=self.store();s.begin('A');s.record('A','selection',selected())
        child=self.child("import coverpruner_checkpoint as c;c.os.replace=lambda *a:os._exit(33);s.record('A','prefill',"+repr(prefill())+")")
        self.assertEqual(child.returncode,33);reopened=CheckpointStore(s.path)
        self.assertEqual(reopened.data('A','selection'),selected());self.assertNotIn('prefill',reopened.snapshot()['methods']['A']['stages'])
    def test_corrupt_tail_recovers_good_checkpoint_and_new_branch(self):
        s=self.store();s.begin('A');s.record('A','selection',selected());s.record('A','prefill',prefill())
        tail=sorted((s.path/'journal').glob('cp-*.json'))[-1];tail.write_text('{broken',encoding='utf-8')
        reopened=CheckpointStore(s.path);self.assertEqual(reopened.data('A','selection'),selected());self.assertTrue(reopened.recovery_issues)
        self.assertNotIn('prefill',reopened.snapshot()['methods']['A']['stages'])
        reopened.record('A','prefill',prefill());reopened.complete('A');self.assertEqual(CheckpointStore(s.path).snapshot()['status'],'completed')
    def test_corrupt_stage_object_rolls_back_and_no_generation_repeat(self):
        s=self.store(generation=True);s.begin('A');s.record('A','selection',selected());s.reserve_generation('A');s.record('A','prefill',prefill())
        ref=s.snapshot()['methods']['A']['stages']['prefill'];(s.path/'objects'/ref['file']).write_text('broken')
        reopened=CheckpointStore(s.path);self.assertEqual(reopened.data('A','selection'),selected())
        with self.assertRaises(AmbiguousExecution):reopened.begin('A')
    def test_corrupt_configuration_fails_closed(self):
        s=self.store();(s.path/'CONFIG.json').write_text('{}')
        with self.assertRaises((CorruptCheckpoint,KeyError)):CheckpointStore(s.path)
    def test_legacy_consumed_run_not_reconstructed_or_regenerated(self):
        s=self.store(generation=True);s.consume_legacy('A',{'path':'runs/p2b-native-20261117_01/LEDGER.json','sha256':'a'*64})
        item=s.snapshot()['methods']['A'];self.assertEqual(item['stages'],{});self.assertEqual(item['status'],'failed')
        with self.assertRaises(AmbiguousExecution):s.begin('A')
    def test_process_writer_lock_prevents_concurrent_begin(self):
        s=self.store()
        with s.lock():
            child=self.child("\ntry:s.begin('A')\nexcept Busy:sys.exit(49)\nsys.exit(0)")
        self.assertEqual(child.returncode,49)
    def test_native_no_cache_forward_record_is_not_fake_KV(self):
        s=self.store();s.begin('A');s.record('A','selection',selected());p=prefill();p.update(cache_mode='none',cache_after=0,cache_lengths=[0,0]);s.record('A','prefill',p);s.complete('A')
        self.assertEqual(s.data('A','prefill')['cache_after'],0)
    def test_diagnostic_saved_before_assert_and_not_overwritten(self):
        s=self.store();self.finish(s);s.diagnostic('A','padding',{'max_abs':.06,'mean_abs':.016,'top1':True},False)
        with self.assertRaises(ValueError):s.diagnostic('A','padding',{'mean_abs':0},True)
        self.assertEqual(s.diagnostic_data('A','padding')['data']['mean_abs'],.016)
    def test_required_diagnostic_missing_prevents_run_completion(self):
        s=self.store(diagnostics=('padding',));self.finish(s)
        self.assertEqual(s.snapshot()['methods']['A']['status'],'completed');self.assertEqual(s.snapshot()['status'],'running')
        s.diagnostic('A','padding',{'mean_abs':0},True);self.assertEqual(s.snapshot()['status'],'completed')
    def test_failed_diagnostic_not_erased_by_later_method_completion(self):
        s=self.store(methods=('A','B'));self.finish(s,'A');s.diagnostic('A','padding',{'mean_abs':.016},False);self.finish(s,'B')
        self.assertEqual(s.snapshot()['status'],'failed')
    def test_torn_reservation_is_conservatively_consumed(self):
        s=self.store(generation=True);s.begin('A');(s.path/'reservations'/'A.json').write_text('{torn')
        with self.assertRaises(AmbiguousExecution):CheckpointStore(s.path).begin('A')
    def test_partial_temporary_file_is_ignored(self):
        s=self.store();self.finish(s);(s.path/'journal'/'cp-99999999.json.tmp-partial').write_text('{torn')
        self.assertEqual(CheckpointStore(s.path).snapshot()['status'],'completed')
    def test_resume_saved_selection_does_not_repeat_selection(self):
        s=self.store();s.begin('A');s.record('A','selection',selected());s.fail('A','before prefill')
        calls=[]
        execute_method(s,'A',lambda:self.fail('selection repeated'),lambda data:(calls.append(1),prefill()),resume=True)
        self.assertEqual(calls,[1]);self.assertEqual(s.snapshot()['status'],'completed')

    def test_resume_finished_forward_does_not_repeat_prefill(self):
        s=self.store();s.begin('A');s.record('A','selection',selected());s.record('A','prefill',prefill())
        execute_method(s,'A',lambda:self.fail('selection repeated'),lambda _:self.fail('prefill repeated'),resume=True)
        self.assertEqual(s.snapshot()['status'],'completed')

    def test_diagnostics_required_and_failure_not_erased_by_other_method(self):
        s=self.store(methods=('A','B'),diagnostics=('pad',));self.finish(s,'A')
        self.assertNotEqual(s.snapshot()['status'],'completed')
        s.diagnostic('A','pad',{'mean':.02},False);self.finish(s,'B');s.diagnostic('B','pad',{'mean':0},True)
        self.assertEqual(s.snapshot()['status'],'failed');self.assertEqual(s.snapshot()['methods']['A']['status'],'completed')

    def test_native_no_cache_forward_record_is_not_fabricated_KV(self):
        s=self.store();s.begin('A');s.record('A','selection',selected());data=prefill();data.update(cache_mode='none',cache_after=0,cache_lengths=[0,0])
        s.record('A','prefill',data);s.complete('A');self.assertEqual(s.data('A','prefill')['cache_after'],0)