from collections import Counter
from contextlib import contextmanager
import random
from pathlib import Path
import tempfile
import unittest

from study5_eval import CONFIG, Ledger, InterruptedGeneration, execute, jobs
from study5_analysis import analyse, CONTRASTS


class BulkMockLedger(Ledger):
    # The small disk-recovery cases use FULL. Bulk mocks exercise the same transactions
    # and persisted state without paying thousands of durable syncs for disposable data.
    @contextmanager
    def connect(self):
        with super().connect() as db:
            db.execute("PRAGMA synchronous=OFF")
            yield db


class CandidateResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        images = []
        for index in range(79):
            questions = [
                {"question_id": str(index * 6 + slot), "text": "Synthetic question",
                 "label": "yes" if slot < 3 else "no"}
                for slot in range(6)
            ]
            images.append({"image_id": index, "image_path": "synthetic.jpg",
                           "image_sha256": "synthetic", "questions": questions})
        cls.plan = list(jobs({"images": images}))
        cls.identity = {"kind": "cpu_mock", "config": CONFIG, "manifest": "synthetic_candidate_reference"}

    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.calls=Counter()

    def generate(self,job):
        self.calls[job["id"]]+=1
        return dict(response=job["label"],selection=dict(seed_cell_counts=[1]*25,final_cell_counts=[40]+[1]*24,
                    initial_coverage=5.,final_coverage=10.))

    def test_plan_balance_unique_order(self):
        self.assertEqual(len({j["id"] for j in self.plan}),2370)
        self.assertEqual(Counter(j["method"] for j in self.plan),dict.fromkeys("ADGHI",474))
        self.assertEqual(Counter(j["label"] for j in self.plan),{"yes":1185,"no":1185})

    def test_uninterrupted_vs_partial_chunk_resume_reordered(self):
        verifier=BulkMockLedger(self.root/"full.db",self.plan,self.identity)
        execute(verifier,self.plan,lambda j:j,self.generate)
        expected=verifier.finish()
        self.assertEqual(len(self.calls),2370)
        self.calls.clear()
        path=self.root/"interrupted.db"
        ledger=BulkMockLedger(path,self.plan,self.identity)
        execute(ledger,self.plan,lambda j:j,self.generate,limit=317)
        records,pending,complete=ledger.inspect()
        self.assertEqual((len(records),len(pending),complete),(317,2053,None))
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            ledger.finish()
        with ledger.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM chunks").fetchone()[0],2)
        shuffled=list(self.plan)
        random.Random(4).shuffle(shuffled)
        reopened=BulkMockLedger(path,shuffled,self.identity)
        execute(reopened,shuffled,lambda j:j,self.generate)
        self.assertEqual(reopened.finish(),expected)
        self.assertEqual(set(self.calls.values()),{1})
        self.assertEqual(len(self.calls),2370)
        execute(reopened,self.plan,lambda j:self.fail("Completed work scheduled"),self.generate)
        self.assertEqual(len(reopened.inspect()[0]),2370)
        summary=analyse(self.plan,expected)
        self.assertEqual(summary["repetitions"],10000)
        self.assertEqual(set(summary["comparisons"]),{c[0]+"-"+c[1] for c in CONTRASTS})
        for pair in summary["comparisons"].values():
            self.assertEqual(pair["metrics"]["accuracy"],{"difference":0.,"ci95":[0.,0.]})

    def test_safe_pre_generation_retry_only_missing(self):
        path=self.root/"before.db"
        ledger=Ledger(path,self.plan,self.identity)
        execute(ledger,self.plan,lambda j:j,self.generate,limit=3)
        with self.assertRaisesRegex(RuntimeError,"before"):
            execute(ledger,self.plan,lambda j:(_ for _ in ()).throw(RuntimeError("before")),self.generate)
        reopened=Ledger(path,self.plan,self.identity)
        execute(reopened,self.plan,lambda j:j,self.generate,limit=1)
        self.assertEqual(set(self.calls),{j["id"] for j in self.plan[:4]})
        self.assertEqual(set(self.calls.values()),{1})
        with reopened.connect() as db:
            self.assertEqual(db.execute("SELECT stage FROM events").fetchone()[0],"before_generation")

    def test_uncertain_generation_reuse_and_error_logging(self):
        path=self.root/"uncertain.db"
        ledger=Ledger(path,self.plan,self.identity)
        def failed(_):
            raise KeyboardInterrupt("Mock worker interrupted")
        with self.assertRaises(KeyboardInterrupt):
            execute(ledger,self.plan,lambda j:j,failed)
        with self.assertRaises(InterruptedGeneration):
            Ledger(path,self.plan,self.identity)
        with ledger.connect() as db:
            self.assertEqual(db.execute("SELECT stage FROM events").fetchone()[0],"after_reservation")

    def test_identity_and_duplicate_guard(self):
        path=self.root/"identity.db"
        ledger=Ledger(path,self.plan,self.identity)
        for key in ("config","manifest"):
            with self.assertRaisesRegex(ValueError,"mismatch"):
                Ledger(path,self.plan,{**self.identity,key:"drift"})
        with self.assertRaisesRegex(ValueError,"duplicate"):
            Ledger(path,[*self.plan[:-1],self.plan[0]],self.identity)
        first=self.plan[0]
        ledger.reserve(first)
        ledger.commit(first,{"response":"yes"})
        self.assertFalse(ledger.reserve(first))
        with self.assertRaisesRegex(ValueError,"overwrite"):
            ledger.commit(first,{"response":"no"})

    def test_chunk_damage_and_missing_completed_output_block(self):
        path=self.root/"damage.db"
        ledger=Ledger(path,self.plan,self.identity)
        execute(ledger,self.plan,lambda j:j,self.generate,limit=150)
        with ledger.connect() as db,db:
            db.execute("UPDATE chunks SET sha='corrupt'")
        with self.assertRaisesRegex(ValueError,"Chunk"):
            Ledger(path,self.plan,self.identity)
        with ledger.connect() as db,db:
            db.execute("UPDATE jobs SET result=NULL WHERE id=?",(self.plan[0]["id"],))
        with self.assertRaisesRegex(ValueError,"Committed result missing"):
            Ledger(path,self.plan,self.identity)
