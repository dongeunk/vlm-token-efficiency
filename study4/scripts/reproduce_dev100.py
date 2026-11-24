"""Recompute Study4 Dev100 classification metrics and optional paired bootstrap.

Uses the released, anonymized image-cluster confusion counts. No images,
model, raw answers, network connection or third-party packages are needed.
"""
import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import random

RESULTS = Path(__file__).resolve().parents[1] / "results"
ARMS = ("Full", *"ABCDEF")
FIELDS = ("tp", "fp", "tn", "fn")
SEED = 20261124
DRAWS = 10_000


def records(file):
    with (RESULTS / (file+".csv")).open(encoding="utf-8",newline="") as src:
        return list(csv.DictReader(src))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, label):
    require(math.isclose(actual,float(expected),rel_tol=1e-12,abs_tol=1e-12),label)


def ratio(num,denom):
    return num/denom if denom else 0.0


def metrics(counts):
    tp,fp,tn,fn = counts
    return dict(accuracy=ratio(tp+tn,tp+fp+tn+fn),f1=ratio(2*tp,2*tp+fp+fn),
                precision=ratio(tp,tp+fp),yes_recall=ratio(tp,tp+fn),no_recall=ratio(tn,tn+fp))


def counts_by_arm():
    groups = {arm:{} for arm in ARMS}
    for row in records("image_cluster_counts"):
        arm, index = row["method"],int(row["cluster_index"])
        require(arm in groups and index not in groups[arm],"Duplicate or unknown group")
        values = tuple(int(row[f]) for f in FIELDS)
        tp,fp,tn,fn=values
        require(min(values)>=0 and tp+fn==3 and tn+fp==3,"Each image must have three Yes and three No")
        groups[arm][index] = values
    require(all(set(g)==set(range(1,101)) for g in groups.values()),"Each arm requires the same 100 clusters")
    return {arm:[g[i] for i in range(1,101)] for arm,g in groups.items()}


def validate_scores(groups):
    totals = {arm:tuple(map(sum,zip(*rows,strict=True))) for arm,rows in groups.items()}
    scores = {arm:metrics(row) for arm,row in totals.items()}
    rows = records("performance")
    require(tuple(r["method"] for r in rows)==ARMS,"Unexpected performance rows")
    for row in rows:
        arm = row["method"]
        require(int(row["n"])==600 and tuple(int(row[f]) for f in FIELDS)==totals[arm],f"{arm} counts")
        for field,key in (("accuracy","strict_accuracy"),("f1","strict_f1"),("precision","strict_precision"),
                          ("yes_recall","yes_recall"),("yes_recall","strict_recall"),("no_recall","no_recall"),
                          ("accuracy","official_accuracy"),("f1","official_f1")):
            compare(scores[arm][field],row[key],f"{arm} {key}")
        require(all(float(row[f])==0 for f in ("parser_disagreements","strict_invalid_rate","invalid_yes","invalid_no")),
                "Anonymized count assumptions failed")
    paired = records("paired_vs_A")
    require({r["method"] for r in paired}==set(ARMS)-{"A"} and len(paired)==6,"Unexpected paired rows")
    for row in paired:
        arm = row["method"]
        require(row["reference"]=="A","Unexpected paired baseline")
        for metric in ("accuracy","f1","yes_recall"):
            actual=scores[arm][metric]-scores["A"][metric]
            compare(actual,row[metric+"_difference"],f"{arm} paired {metric}")
        net=(int(row["wrong_to_correct"])-int(row["correct_to_wrong"]))/600
        compare(net,row["accuracy_difference"],f"{arm} paired net change")
    return scores


def validate_bootstrap(groups,scores):
    entries = records("cluster_bootstrap")
    keys = {(r["method"],r["reference"],r["metric"]) for r in entries}
    expected = {(arm,ref,metric) for arm in ARMS[1:] for ref in ("A","Full") if arm!=ref
                for metric in ("accuracy","f1","yes_recall")}
    require(len(entries)==len(expected) and keys==expected,"Unexpected bootstrap comparisons")
    vectors = {key:[] for key in keys}
    rng = random.Random(SEED)
    for _ in range(DRAWS):
        multiplicity = Counter(rng.randrange(100) for _ in range(100))
        sample = {}
        for arm,rows in groups.items():
            counts = tuple(sum(rows[i][j]*weight for i,weight in multiplicity.items()) for j in range(4))
            sample[arm] = metrics(counts)
        for (arm,ref,metric), vec in vectors.items():
            vec.append(sample[arm][metric]-sample[ref][metric])
    for row in entries:
        arm,ref,metric = (row[k] for k in ("method","reference","metric"))
        require((int(row["seed"]),int(row["repetitions"]),int(row["clusters"]))==(SEED,DRAWS,100),"Bootstrap settings differ")
        compare(scores[arm][metric]-scores[ref][metric],row["difference"],f"{arm} vs {ref} {metric}")
        values = sorted(vectors[arm,ref,metric])
        for p,key in ((.025,"lower95"),(.975,"upper95")):
            pos = p*(DRAWS-1)
            i = int(pos)
            compare(values[i]+(pos-i)*(values[i+1]-values[i]),row[key],f"{arm} vs {ref} {metric} {key}")
    print(f"Verified {len(entries)} paired 95% CIs with {DRAWS:,} shared image-cluster resamples.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap",action="store_true")
    args=parser.parse_args()
    groups=counts_by_arm()
    scores=validate_scores(groups)
    print("Dev100 • 100 image clusters • 600 questions/arm • 4,200 historical responses")
    for arm,row in scores.items():
        print(f"{arm:>4}  Accuracy {row['accuracy']:.2%}  F1 {row['f1']:.2%}  Yes Recall {row['yes_recall']:.2%}")
    if args.bootstrap:
        validate_bootstrap(groups,scores)


if __name__=="__main__":
    main()
