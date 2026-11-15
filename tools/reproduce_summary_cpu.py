# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dongeunk (original project contributions).
# See LICENSE. Public aggregate-only adapter; no pruning algorithm is modified.
"""Verify stored main aggregate counts on CPU; never load raw responses/models."""
from __future__ import annotations
import argparse
import csv
import importlib.abc
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES="-1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in {"torch","transformers","accelerate","huggingface_hub"}:
            raise ImportError("Aggregate reproduction forbids model/GPU imports")
        return None
sys.meta_path.insert(0,NoModels())
import numpy as np
from src.smol_s3_eval.statistics import metrics_from_counts,paired_bootstrap
from src.smol_s3_eval.lock import METHODS

def read(path):return json.loads(path.read_text(encoding="utf-8"))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap",action="store_true",help="Optionally recompute the frozen10000-resample seed42 CI on CPU")
    args=parser.parse_args();directory=ROOT/"research/second_study/main"
    rows=[json.loads(s) for s in (directory/"per_image.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(rows)!=200 or len({r["image_id"] for r in rows})!=200 or any(r["questions"]!=6 for r in rows):raise ValueError("Exactly200 six-question image clusters required")
    counts=np.asarray([[r["counts"][m] for m in METHODS] for r in rows],dtype=np.float64)
    values=metrics_from_counts(counts);totals=metrics_from_counts(counts.sum(axis=0))
    if not np.all(counts[:,:,:6].sum(axis=-1)==6) or not np.all(counts[:,:,[0,2,4]].sum(axis=-1)==3) or not np.all(counts[:,:,[1,3,5]].sum(axis=-1)==3):raise ValueError("Frozen yes3/no3 support differs")
    with (directory/"METHOD_COMPARISON.csv").open(encoding="utf-8",newline="") as stream:expected={r["method"]:r for r in csv.DictReader(stream)}
    for i,m in enumerate(METHODS):
        target=expected[m]
        actual=[totals[i,0],totals[i,1],totals[i,2],totals[i,3]]
        golden=[float(target["semantic_f1"]),float(target["semantic_accuracy"]),int(target["invalid"])/1200,float(target["strict_accuracy"])]
        if not np.allclose(actual,golden,rtol=0,atol=1e-12):raise ValueError("Frozen metric mismatch")
    frozen=read(directory/"PAIRED_BOOTSTRAP_10000.json")
    primary=totals[METHODS.index("s2_c"),0]-totals[METHODS.index("s2_l"),0]
    if abs(primary-frozen["comparisons"]["s2_c-s2_l"]["pope_f1"]["estimate"])>1e-12:raise ValueError("Primary estimate differs")
    if args.bootstrap:
        boot=paired_bootstrap(rows,resamples=10000,seed=42)
        for pair,v in frozen["comparisons"].items():
            for metric,t in v.items():
                got=boot["comparisons"][pair][metric]
                if not np.allclose([got["estimate"]]+got["ci95"],[t["estimate"]]+t["ci95"],rtol=0,atol=1e-12):raise ValueError("Frozen CI mismatch")
    forbidden=[n for n in sys.modules if n.startswith(("src.s2_coverage_study","src.smol_s2_study","torch","transformers","_s2_coverage_study_pinned_reference"))]
    if forbidden:raise RuntimeError("Forbidden helper/model module imported")
    print(json.dumps(dict(status="PASS",images=200,questions=1200,methods=list(METHODS),
        metrics={m:dict(F1=float(totals[i,0]),accuracy=float(totals[i,1]),invalid_rate=float(totals[i,2]),strict_accuracy=float(totals[i,3])) for i,m in enumerate(METHODS)},
        optional_bootstrap_recomputed=args.bootstrap,raw_response_parser_replayed=False,held_helper_imported=False,model_loads=0,GPU_operations=0,new_generation=0,files_written=0),indent=2))
if __name__=="__main__":main()
