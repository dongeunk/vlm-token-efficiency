"""Fixed Study5 candidate contract and atomic, at-most-once job ledger."""
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from coverpruner_checkpoint import digest
from study5_scoring import parse_answer, official_answer

ROOT = Path(__file__).resolve().parents[1]
S4 = ROOT.parent/"study4"
RESERVE = S4/"runs/p3b0-dev-20261124/RESERVE79.json"
DEV = RESERVE.with_name("DEV100.json")
POPE = ROOT.parent/"archive/study1-2/ai_project/data/pope/coco_pope_random.json"
METHODS = tuple("ADGHI")
SUFFIX = "\nAnswer with Yes or No only."
REVISION = "7e3e67edbbed1bf9888184d9df282b700a323964"
PROTOCOL_SHA = "9d97a841904a18d4df8902325d593466b2cb5e53756a492d973240b025166351"
CONFIG = dict(methods=list(METHODS), snapshot=REVISION, K=64, batch=1, padding=False,
              backend="sdpa",dtype="float16", decoding="greedy",max_new_tokens=10,prompt_suffix=SUFFIX,
              parser="strict_answer_only_v1", auxiliary_parser="official_compatible",
              torch="2.13.0+cu126",transformers="5.15.1",gpu="GTX 1660 SUPER",protocol_sha256=PROTOCOL_SHA,
              bootstrap_seed=20261126,bootstrap_repetitions=10000,images_per_chunk=5,
              scope="holdout79_candidate_exploratory",independent_test=False,tensor_dumps=0)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(condition,message):
    if not condition:
        raise ValueError(message)


def audited_reference():
    audit=read(ROOT/"evidence/HOLDOUT79_AUDIT.json")
    reserve_sha=sha(RESERVE)
    dev_sha=sha(DEV)
    pope_sha=sha(POPE)
    require(reserve_sha==audit["reserve_sha256"]=="a80c6741782c7ef13b9f3dfd4ce15fbe8697222335c60b02346964b04744c1f8",
            "RESERVE79 changed")
    require(dev_sha==audit["dev_sha256"]=="ab748187e49cf6f1963fd7cf65727ee41f1b858d5a27e35e8892b235a73bf4ee"
            and pope_sha==audit["official_pope_sha256"]=="ac25245170b975a5bdf9080b23fd431dfe6be458bc038259c1f4f09a6bef7994",
            "Dev or official annotation changed")
    require(sha(ROOT/"docs/STUDY5_PROTOCOL.md")==PROTOCOL_SHA,"P1 protocol changed")
    images=read(RESERVE)["images"]
    dev=read(DEV)
    ids=[r["image_id"] for r in images]
    require(len(ids)==len(set(ids))==79 and ids==audit["audited_scope_image_ids"] and not audit["exclusions"],
            "Candidate set/order changed")
    require(not set(ids)&({r["image_id"] for r in dev["images"]}|set(dev["exposed_ids"])),"Recorded exposure overlap")
    official={}
    for line in POPE.read_bytes().splitlines():
        if line.strip():
            row=json.loads(line)
            qid=str(row["question_id"])
            require(qid not in official,"Duplicate official question ID")
            official[qid]=row
    counts=Counter()
    seen=set()
    for image in images:
        path=Path(image["image_path"]).resolve()
        name=f"COCO_val2014_{image['image_id']:012d}.jpg"
        require(path.name==name and path.is_relative_to(POPE.parent.parent/"coco")
                and sha(path)==image["image_sha256"],"Image path or bytes drift")
        questions=image["questions"]
        require(Counter(q["label"] for q in questions)=={"yes":3,"no":3},"Class counts changed")
        for q in questions:
            qid=q["question_id"]
            row=official[qid]
            require(qid not in seen and (row["image"],row["text"],row["label"])==(name,q["text"],q["label"]),
                    "Official POPE question drift")
            seen.add(qid)
            counts[q["label"]]+=1
    require(len(seen)==474 and counts=={"yes":237,"no":237},"Candidate counts drift")
    return dict(reserve_sha256=reserve_sha,dev_sha256=dev_sha,official_sha256=pope_sha,
                audit_sha256=sha(ROOT/"evidence/HOLDOUT79_AUDIT.json"),images=images,
                independent_test=False,unknown_history=audit["unknown_history"])


def jobs(ref):
    for idx,image in enumerate(ref["images"]):
        for q in image["questions"]:
            for arm in METHODS:
                yield dict(id=f'{image["image_id"]}/{q["question_id"]}/{arm}', image_id=image["image_id"],
                           question_id=q["question_id"],method=arm,text=q["text"],label=q["label"],
                           image_path=image["image_path"], image_sha256=image["image_sha256"],
                           chunk=idx//CONFIG["images_per_chunk"]+1)


class InterruptedGeneration(RuntimeError):
    pass


class Ledger:
    def __init__(self,path,plan,identity):
        self.plan={j["id"]:j for j in plan}
        require(len(self.plan)==len(plan)==2370,"Unexpected or duplicate job plan")
        self.identity=json.dumps(identity,sort_keys=True,separators=(",",":"))
        self.path=path
        with self.connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='meta'").fetchone():
                require(not db.execute("SELECT 1 FROM sqlite_master WHERE type='table'").fetchone(),"Unknown ledger")
                db.execute("CREATE TABLE meta (identity TEXT NOT NULL, complete TEXT)")
                db.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, chunk INTEGER NOT NULL, state TEXT NOT NULL, result TEXT, sha TEXT)")
                db.execute("CREATE TABLE events (id TEXT NOT NULL, stage TEXT NOT NULL, error TEXT NOT NULL)")
                db.execute("CREATE TABLE chunks (chunk INTEGER PRIMARY KEY, sha TEXT NOT NULL)")
                db.execute("INSERT INTO meta VALUES (?,NULL)",(self.identity,))
                db.executemany("INSERT INTO jobs VALUES (?,?,'pending',NULL,NULL)",
                               ((j["id"],j["chunk"]) for j in plan))
            require(db.execute("SELECT identity FROM meta").fetchall()==[(self.identity,)],"Settings or Manifest mismatch")
        self.inspect()

    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.execute("PRAGMA synchronous=FULL")
        return closing(db)

    def inspect(self):
        with self.connect() as db:
            db.execute("BEGIN")
            require(db.execute("PRAGMA integrity_check").fetchall()==[("ok",)],"SQLite integrity failure")
            rows=db.execute("SELECT id,chunk,state,result,sha FROM jobs").fetchall()
            require({r[0] for r in rows}==set(self.plan),"Job inventory mismatch")
            records={}
            groups={}
            pending=[]
            for jid,chunk,state,raw,signature in rows:
                require(chunk==self.plan[jid]["chunk"],"Chunk membership mismatch")
                if state=="reserved":
                    raise InterruptedGeneration(f"Uncertain GPU generation for {jid}; preserve ledger and investigate")
                if state=="complete":
                    require(raw is not None and signature is not None,"Committed result missing")
                    row=json.loads(raw)
                    require(digest(row)==signature and row["id"]==jid,"Job checksum mismatch")
                    records[jid]=row
                    groups.setdefault(chunk,[]).append((jid,signature))
                else:
                    require(state=="pending" and raw is None and signature is None,"Invalid job state")
                    pending.append(jid)
            chunk_counts=Counter(j["chunk"] for j in self.plan.values())
            sealed={c:digest(sorted(rows)) for c,rows in groups.items() if len(rows)==chunk_counts[c]}
            saved=dict(db.execute("SELECT chunk,sha FROM chunks"))
            require(sealed==saved,"Chunk missing, damaged, or inconsistent")
            complete=db.execute("SELECT complete FROM meta").fetchone()[0]
            require(complete is None or (not pending and complete==digest(sorted(sealed.items()))),"Completion checksum mismatch")
            return records,pending,complete

    def event(self,jid,stage,error):
        with self.connect() as db,db:
            db.execute("INSERT INTO events VALUES (?,?,?)",(jid,stage,str(error)[:2000]))

    def reserve(self,job):
        with self.connect() as db,db:
            db.execute("BEGIN IMMEDIATE")
            state=db.execute("SELECT state FROM jobs WHERE id=?",(job["id"],)).fetchone()[0]
            if state=="complete":
                return False
            if state!="pending":
                raise InterruptedGeneration(f"Generation already reserved for {job['id']}")
            db.execute("UPDATE jobs SET state='reserved' WHERE id=?",(job["id"],))
            return True

    def commit(self,job,result):
        row={**result,"id":job["id"]}
        with self.connect() as db,db:
            db.execute("BEGIN IMMEDIATE")
            changed=db.execute("UPDATE jobs SET state='complete',result=?,sha=? WHERE id=? AND state='reserved'",
                               (json.dumps(row,allow_nan=False),digest(row),job["id"])).rowcount
            require(changed==1,"Cannot overwrite a completed or unreserved job")
            rows=db.execute("SELECT id,state,sha FROM jobs WHERE chunk=? ORDER BY id",(job["chunk"],)).fetchall()
            if all(state=="complete" for _,state,_ in rows):
                db.execute("INSERT INTO chunks VALUES (?,?)",(job["chunk"],digest([(jid,sha) for jid,_,sha in rows])))
        return row

    def finish(self,validate=None):
        records,pending,complete=self.inspect()
        require(not pending and len(records)==2370,"Incomplete result set")
        require(Counter(self.plan[jid]["method"] for jid in records)==dict.fromkeys(METHODS,474),"Per-method counts incorrect")
        if validate is not None:
            validate(records)
        if complete is None:
            with self.connect() as db,db:
                db.execute("BEGIN IMMEDIATE")
                checksum=digest(db.execute("SELECT chunk,sha FROM chunks ORDER BY chunk").fetchall())
                db.execute("UPDATE meta SET complete=? WHERE complete IS NULL",(checksum,))
        return self.inspect()[0]


def execute(ledger,plan,prepare,generate,*,limit=None):
    records,_,_=ledger.inspect()
    count=0
    for job in plan:
        if job["id"] in records:
            continue
        if limit is not None and count>=limit:
            return
        try:
            packet=prepare(job)
        except Exception as error:
            ledger.event(job["id"],"before_generation",error)
            raise
        if not ledger.reserve(job):
            continue
        try:
            result=generate(packet)
            raw=result["response"]
            require(isinstance(raw,str),"Missing raw response")
            result.update(strict=parse_answer(raw),official=official_answer(raw))
            ledger.commit(job,result)
            count+=1
        except (Exception,KeyboardInterrupt) as error:
            ledger.event(job["id"],"after_reservation",error)
            raise

