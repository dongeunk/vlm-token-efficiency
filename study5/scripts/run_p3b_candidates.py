"""Holdout79 candidate evaluation. Default is a read-only planning check."""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from coverpruner_checkpoint import atomic, digest, envelope
from study5_eval import CONFIG, Ledger, audited_reference, execute, jobs, read, require, sha
from study5_analysis import analyse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--run-candidate-gpu",action="store_true",help="Separately authorized exploratory candidate run")
    args=parser.parse_args()
    snapshot=args.snapshot.resolve()
    require(snapshot.name==CONFIG["snapshot"] and (snapshot/"model.safetensors").is_file(),"Expected pinned local snapshot")
    run=args.run_dir.resolve()
    require(run.parent==ROOT/"runs","Run directory must be a direct child of Study5/runs")
    ref=audited_reference()
    plan=list(jobs(ref))
    source={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/"src").glob("*.py"))}
    source["configs/layout_geometry.json"]=sha(ROOT/"configs/layout_geometry.json")
    source["scripts/run_p3b_candidates.py"]=sha(Path(__file__))
    metadata={p.name:sha(p) for p in sorted(snapshot.glob("*")) if p.is_file() and p.suffix in {".json",".txt"}}
    identity=dict(config=CONFIG,manifest_sha256=digest(ref),job_plan_sha256=digest(plan),
                  snapshot_metadata=metadata,snapshot_weight_bytes=(snapshot/"model.safetensors").stat().st_size,source=source)
    if not args.run_candidate_gpu:
        print(json.dumps(dict(mode="plan_only",jobs=len(plan),config_sha256=digest(CONFIG),
                              manifest_sha256=identity["manifest_sha256"],identity_sha256=digest(identity),gpu_jobs=0)))
        return
    evidence=read(ROOT/"evidence/P3A_CPU.json")
    require(evidence["status"]=="PASS" and evidence["identity_sha256"]==digest(identity),"CPU gate or locked identity drift")
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1")
    run.mkdir(parents=True,exist_ok=True)
    # The identity is in SQLite before model imports. A reserved job blocks resume.
    ledger=Ledger(run/"jobs.sqlite3",plan,identity)
    _,pending,_=ledger.inspect()
    if pending:
        from study5_eval_gpu import Backend
        backend=Backend(snapshot)
        execute(ledger,plan,backend.prepare,backend.generate)
    def validate_outputs(records):
        score_hashes={}
        for j in plan:
            row=records[j["id"]]
            profile=row["profile"]
            require(1<=len(row["token_ids"])<=10 and profile["visual_tokens"]==64
                    and profile["padding_rows"]==0 and profile["backend"]=="sdpa",
                    "Output K64, token count or profile drift")
            require(row["selection"]["method"]==j["method"],"Output method mismatch")
            key=j["image_id"],j["question_id"]
            require(score_hashes.setdefault(key,row["query_score_sha256"])==row["query_score_sha256"],
                    "Query score mismatched across arms")
    records=ledger.finish(validate_outputs)
    report=analyse(plan,records)
    target=run/"candidate_summary.json"
    packet=envelope(report)
    if target.exists():
        require(read(target)==packet,"Existing candidate result differs")
    else:
        temporary=run/f".summary-{uuid.uuid4().hex}.json"
        atomic(temporary,packet,limit=256*1024)
        try:
            os.link(temporary,target)
        except FileExistsError:
            require(read(target)==packet,"Existing candidate result differs")
        finally:
            temporary.unlink()
    print("COMPLETE: 2,370 candidate responses; exploratory, independence unconfirmed.")


if __name__=="__main__":
    main()

