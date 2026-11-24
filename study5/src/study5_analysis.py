"""Paired cluster summaries for a fully completed candidate run."""
from collections import defaultdict
import random

from study5_eval import METHODS, CONFIG, require
from study5_scoring import metrics

CONTRASTS=("DG","DH","DI","GH","IH","DA","GA")
FIELDS=("accuracy","f1","precision","recall","no_recall")
LABEL_KEYS=(("yes","yes"),("no","yes"),("no","no"),("yes","no"),("yes","invalid"),("no","invalid"))


def analyse(plan,records):
    import numpy as np
    require(len(plan)==len(records)==2370 and set(records)=={j["id"] for j in plan},"Analysis requires all jobs")
    images=list(dict.fromkeys(j["image_id"] for j in plan))
    require(len(images)==79,"Expected 79 paired clusters")
    slots={x:i for i,x in enumerate(images)}
    arms={x:i for i,x in enumerate(METHODS)}
    counts=np.zeros((79,5,6),dtype=np.int64)
    by_question=defaultdict(dict)
    occupancy=defaultdict(list)
    strict=defaultdict(list)
    auxiliary=defaultdict(list)
    for j in plan:
        row=records[j["id"]]
        arm,label,pred=j["method"],j["label"],row["strict"]
        counts[slots[j["image_id"]],arms[arm],LABEL_KEYS.index((label,pred))]+=1
        by_question[j["question_id"]][arm]=pred==label
        strict[arm].append(dict(label=label,prediction=pred))
        auxiliary[arm].append(dict(label=label,prediction=row["official"]))
        sel=row["selection"]
        occupancy[arm].append([sum(x>0 for x in sel["seed_cell_counts"]),sum(x>0 for x in sel["final_cell_counts"]),
                               max(sel["final_cell_counts"]),sel["initial_coverage"],sel["final_coverage"]])
    require((counts.sum(axis=2)==6).all(),"Incomplete image clusters")
    def measure(v):
        tp,fp,tn,fn,iy,ino=np.moveaxis(v,-1,0)
        def rate(u,w):
            return np.divide(u,w,out=np.zeros_like(u,dtype=float),where=w!=0)
        return dict(accuracy=rate(tp+tn,v.sum(axis=-1)),f1=rate(2*tp,2*tp+fp+fn+iy),
                    precision=rate(tp,tp+fp),recall=rate(tp,tp+fn+iy),no_recall=rate(tn,tn+fp+ino))
    rng=random.Random(CONFIG["bootstrap_seed"])
    reps=CONFIG["bootstrap_repetitions"]
    sample=[]
    for _ in range(reps):
        sample.append(np.bincount([rng.randrange(79) for _ in range(79)],minlength=79))
    sample_measure=measure((np.asarray(sample)@counts.reshape(79,-1)).reshape(reps,5,6))
    estimates=measure(counts.sum(axis=0))
    comparisons={}
    for contrast in CONTRASTS:
        left,right=contrast
        a,b=arms[left],arms[right]
        comparisons[left+"-"+right]=dict(
            primary_metrics=["accuracy"] if contrast=="DG" else [],
            correct_to_wrong=sum(v[right] and not v[left] for v in by_question.values()),
            wrong_to_correct=sum(not v[right] and v[left] for v in by_question.values()),
            metrics={m:dict(difference=float(estimates[m][a]-estimates[m][b]),
                 ci95=np.quantile(sample_measure[m][:,a]-sample_measure[m][:,b],[.025,.975],method="linear").tolist()) for m in FIELDS})
    def by_method(rows):
        values=metrics(rows)
        total=values["tn"]+values["fp"]+values["invalid_no"]
        values["no_recall"]=values["tn"]/total if total else 0.
        return values
    cells={}
    for arm in METHODS:
        seed,final,largest,start,total=np.mean(occupancy[arm],axis=0).tolist()
        cells[arm]=dict(mean_seed_occupied_cells=seed if arm in "DGI" else None,mean_final_occupied_cells=final,
                      mean_final_occupied_fraction=final/25,mean_max_final_cell_tokens=largest,
                      mean_seed_coverage=start if arm in "DGI" else None,mean_final_coverage=total)
    return dict(scope=CONFIG["scope"],independent_test=False,bootstrap_seed=CONFIG["bootstrap_seed"],repetitions=reps,
                paired_image_clusters=79,interval="linear percentile 95%",secondary_interpretation="exploratory_unadjusted",
                strict={a:by_method(strict[a]) for a in METHODS},official={a:by_method(auxiliary[a]) for a in METHODS},
                comparisons=comparisons,occupancy_coverage=cells)

