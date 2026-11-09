"""Run the specified multi-seed, equal-budget active-learning experiment."""
import csv
import importlib.metadata as package_metadata
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits
from acquisition import select_batch
from encoder import Encoder, ROOT
from prepare import load_clean_data

SEEDS=[42,43,44,45,46]
BUDGETS=[450,900,1500,3000]
STRATEGIES=["random","entropy","balanced_margin"]


def new_head(seed):
    return LogisticRegression(C=10,max_iter=500,tol=1e-4,solver="lbfgs",random_state=seed)


def classification_metrics(labels,probabilities):
    predicted=probabilities.argmax(axis=1)
    return {"accuracy":float(accuracy_score(labels,predicted)),
            "macro_f1":float(f1_score(labels,predicted,average="macro",zero_division=0))}


def write_csv(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    return {"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1))}


def main():
    data,quality=load_clean_data()
    artifact=ROOT/"artifacts"
    modeldir=artifact/"checkpoints"
    modeldir.mkdir(exist_ok=True)
    cache=np.load(artifact/"embeddings.npz",allow_pickle=False)
    Xtrain,Xval=cache["train"],cache["val"]
    labels=quality["label_order"]
    label_id={label:i for i,label in enumerate(labels)}
    ytrain=np.array([label_id[label] for _,label in data["train"]])
    yval=np.array([label_id[label] for _,label in data["val"]])
    rows=[]
    trajectories={}
    with threadpool_limits(limits=2):
        for seed in SEEDS:
            starter_rng=np.random.default_rng(seed)
            starter=np.concatenate([starter_rng.choice(np.flatnonzero(ytrain==label),3,replace=False)
                                    for label in range(len(labels))])
            for strategy in STRATEGIES:
                selected=starter.copy()
                rng=np.random.default_rng(seed+2026)
                previous=None
                for budget in BUDGETS:
                    if len(selected)<budget:
                        available=np.setdiff1d(np.arange(len(ytrain)),selected,assume_unique=True)
                        p=previous.predict_proba(Xtrain[available]) if strategy!="random" else None
                        batch=select_batch(p,available,budget-len(selected),strategy,rng)
                        assert not set(batch)&set(selected)
                        selected=np.concatenate([selected,batch])
                    assert len(selected)==budget and len(np.unique(selected))==budget
                    model=new_head(seed)
                    start=perf_counter()
                    model.fit(Xtrain[selected],ytrain[selected])
                    fit_seconds=perf_counter()-start
                    counts=np.bincount(ytrain[selected],minlength=len(labels))
                    values=classification_metrics(yval,model.predict_proba(Xval))
                    row={"seed":seed,"strategy":strategy,"labels":budget,
                         "validation_accuracy":values["accuracy"],"validation_macro_f1":values["macro_f1"],
                         "fit_seconds":fit_seconds,"optimizer_iterations":int(model.n_iter_.max()),
                         "minimum_class_count":int(counts.min()),"maximum_class_count":int(counts.max())}
                    if row["optimizer_iterations"]>=500:
                        raise RuntimeError("Optimizer iteration limit reached; do not report unconverged fits")
                    key=f"{seed}_{strategy}_{budget}"
                    joblib.dump(model,modeldir/f"{key}.joblib",compress=3)
                    trajectories[key]=selected.tolist()
                    rows.append(row)
                    previous=model
                    print("FIT",seed,strategy,budget,"VAL_F1",round(values["macro_f1"],4),flush=True)
        validation_means={strategy:float(np.mean([r["validation_macro_f1"] for r in rows
                                                 if r["strategy"]==strategy and r["labels"]==BUDGETS[-1]]))
                          for strategy in STRATEGIES}
        chosen=max(STRATEGIES,key=lambda strategy:validation_means[strategy])
        selection={"strategy":chosen,"labels":BUDGETS[-1],"deployment_seed":42,
                   "validation_means_at_final_budget":validation_means,
                   "selection_policy":"Mean validation macro-F1; seed fixed in advance"}
        (artifact/"frozen_selection.json").write_text(json.dumps(selection,indent=2))
        (artifact/"acquisition_indices.json").write_text(json.dumps(trajectories))
        print("FROZEN_SELECTION",json.dumps(selection),flush=True)
        # Test labels enter the experiment only after every trajectory and selection choice is frozen.
        Xtest=cache["test"]
        ytest=np.array([label_id[label] for _,label in data["test"]])
        for row in rows:
            key=f"{row['seed']}_{row['strategy']}_{row['labels']}"
            model=joblib.load(modeldir/f"{key}.joblib")
            values=classification_metrics(ytest,model.predict_proba(Xtest))
            row["test_accuracy"],row["test_macro_f1"]=values["accuracy"],values["macro_f1"]
        write_csv(artifact/"learning_curves.csv",rows)
        summary=[]
        for strategy in STRATEGIES:
            for budget in BUDGETS:
                block=[r for r in rows if r["strategy"]==strategy and r["labels"]==budget]
                entry={"strategy":strategy,"labels":budget}
                for metric in ["validation_macro_f1","test_accuracy","test_macro_f1"]:
                    values=mean_std([r[metric] for r in block])
                    entry[metric+"_mean"],entry[metric+"_std"]=values["mean"],values["std"]
                summary.append(entry)
        write_csv(artifact/"learning_curve_summary.csv",summary)
        comparisons={}
        random_by_seed={r["seed"]:r["test_macro_f1"] for r in rows
                        if r["strategy"]=="random" and r["labels"]==BUDGETS[-1]}
        for strategy in STRATEGIES[1:]:
            delta=[r["test_macro_f1"]-random_by_seed[r["seed"]] for r in rows
                   if r["strategy"]==strategy and r["labels"]==BUDGETS[-1]]
            half=float(t.ppf(.975,len(delta)-1)*np.std(delta,ddof=1)/np.sqrt(len(delta)))
            center=float(np.mean(delta))
            comparisons[strategy]={"paired_differences":delta,"mean_difference":center,
                                   "ci95_low":center-half,"ci95_high":center+half,
                                   "scope":"Seed variability on the fixed test set; not new-dataset uncertainty"}
        full=new_head(42)
        full.fit(Xtrain,ytrain)
        if full.n_iter_.max()>=500:
            raise RuntimeError("Full-data reference did not converge")
        full_metrics=classification_metrics(ytest,full.predict_proba(Xtest))
        deployed=joblib.load(modeldir/f"42_{chosen}_{BUDGETS[-1]}.joblib")
        validation_probability=deployed.predict_proba(Xval)
        threshold=float(np.quantile(validation_probability.max(axis=1),.05,method="lower"))
        known_p=deployed.predict_proba(Xtest)
        unknown_p=deployed.predict_proba(cache["oos_test"])
        known_accept=known_p.max(axis=1)>=threshold
        unknown_reject=unknown_p.max(axis=1)<threshold
        unknown_scores=np.concatenate([1-known_p.max(axis=1),1-unknown_p.max(axis=1)])
        unknown_truth=np.concatenate([np.zeros(len(known_p)),np.ones(len(unknown_p))])
        rejection={"threshold":threshold,"known_acceptance_rate":float(known_accept.mean()),
                   "unknown_rejection_rate":float(unknown_reject.mean()),
                   "unknown_false_acceptance_rate":float(1-unknown_reject.mean()),
                   "unknown_detection_roc_auc":float(roc_auc_score(unknown_truth,unknown_scores)),
                   "accepted_known_accuracy":float(accuracy_score(ytest[known_accept],known_p[known_accept].argmax(axis=1))),
                   "known_test_rows":len(known_p),"unknown_test_rows":len(unknown_p),
                   "threshold_policy":"Known validation fifth percentile; no unknown examples used for gate selection"}
        for split,proba in [("test",known_p),("oos_test",unknown_p)]:
            records=[]
            for i,p in enumerate(proba):
                records.append({"row":i,"true_intent":data[split][i][1],"predicted_intent":labels[int(p.argmax())],
                                "maximum_probability":float(p.max()),"rejected":bool(p.max()<threshold)})
            write_csv(artifact/f"deployment_{split}_predictions.csv",records)
        bundle={"classifier":deployed,"labels":labels,"threshold":threshold,
                "selection":selection,"dataset_sha256":quality["dataset_sha256"],
                "model_revision":json.loads((ROOT/"model_revision.json").read_text())}
        joblib.dump(bundle,artifact/"classifier.joblib",compress=3)
        encoder=Encoder()
        probes=[text for text,_ in data["val"][:50]]
        for text in probes[:3]:
            deployed.predict_proba(encoder.encode([text]))
        elapsed=[]
        for text in probes:
            start=perf_counter()
            deployed.predict_proba(encoder.encode([text]))
            elapsed.append((perf_counter()-start)*1000)
    final=next(r for r in summary if r["strategy"]==chosen and r["labels"]==BUDGETS[-1])
    report={"seeds":SEEDS,"strategies":STRATEGIES,"budgets":BUDGETS,"fitted_budget_models":len(rows),
            "clean_training_pool":len(ytrain),"classes":len(labels),"validation_rows":len(yval),
            "test_rows":len(ytest),"selection":selection,"selected_policy_final_budget":final,
            "paired_vs_random_at_3000":comparisons,"full_data_reference":full_metrics,
            "deployed_seed42_known_test":classification_metrics(ytest,known_p),"rejection":rejection,
            "local_latency":{"queries":len(elapsed),"median_ms":float(np.median(elapsed)),
                             "p95_ms":float(np.percentile(elapsed,95)),
                             "scope":"Sequential local encoding plus head scoring; excludes startup and HTTP overhead"},
            "versions":{name:package_metadata.version(name) for name in
                        ["numpy","scipy","scikit-learn","onnxruntime","tokenizers","joblib","threadpoolctl"]}}
    (artifact/"metrics.json").write_text(json.dumps(report,indent=2))
    print("RESULT",json.dumps(report),flush=True)


if __name__=="__main__":
    main()
