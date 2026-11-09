"""Render measured learning curves and a readable experiment report."""
import csv
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"artifacts/.mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    r=json.loads((ROOT/"artifacts/metrics.json").read_text())
    q=json.loads((ROOT/"artifacts/data_quality.json").read_text())
    with (ROOT/"artifacts/learning_curve_summary.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    names={"random":"Random","entropy":"Entropy","balanced_margin":"Balanced margin"}
    colors={"random":"#536170","entropy":"#C65351","balanced_margin":"#1768AC"}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.labelcolor":"#253242","text.color":"#253242"})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),gridspec_kw={"width_ratios":[1.4,1]})
    ax=axes[0]
    for strategy in r["strategies"]:
        block=sorted([row for row in rows if row["strategy"]==strategy],key=lambda x:int(x["labels"]))
        x=np.array([int(row["labels"]) for row in block])
        y=np.array([float(row["test_macro_f1_mean"]) for row in block])
        sd=np.array([float(row["test_macro_f1_std"]) for row in block])
        ax.plot(x,y,"o-",color=colors[strategy],label=names[strategy],linewidth=2,markersize=4)
        ax.fill_between(x,y-sd,y+sd,color=colors[strategy],alpha=.12)
    ax.axhline(r["full_data_reference"]["macro_f1"],color="#7A7A7A",linestyle="--",linewidth=1,
               label=f"Full data ({r['clean_training_pool']:,} labels)")
    ax.set(xlabel="Acquired training labels",ylabel="Known-intent test macro-F1",ylim=(.78,.97),
           xticks=r["budgets"],yticks=[.8,.85,.9,.95],title="Learning curves across five seeds")
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.grid(axis="y",alpha=.18)
    ax.legend(loc="lower right",frameon=False,fontsize=9)
    ax=axes[1]
    for i,strategy in enumerate(["entropy","balanced_margin"]):
        result=r["paired_vs_random_at_3000"][strategy]
        mean=result["mean_difference"]*100
        lower,upper=result["ci95_low"]*100,result["ci95_high"]*100
        ax.errorbar(mean,i,xerr=[[mean-lower],[upper-mean]],fmt="o",capsize=5,
                    color=colors[strategy],markersize=6,linewidth=2)
        ax.annotate(f"{mean:+.2f} pp",(mean,i),xytext=(0,13),textcoords="offset points",ha="center",fontsize=9)
    ax.axvline(0,color="#8F98A2",linestyle="--",linewidth=1)
    ax.set(yticks=[0,1],yticklabels=[names["entropy"],names["balanced_margin"]],ylim=(-.55,1.55),
           xlabel="Macro-F1 difference versus random (pp)",title="Equal-budget comparison at 3,000 labels")
    ax.grid(axis="x",alpha=.18)
    fig.suptitle("Active learning on CLINC150",fontsize=15,fontweight="bold",y=.98)
    fig.text(.5,.015,"Left: mean ± one standard deviation. Right: paired 95% t intervals across five seeds on the fixed test set.",
             ha="center",fontsize=9,color="#536170")
    fig.tight_layout(rect=(0,.06,1,.94),w_pad=2.5)
    figures=ROOT/"figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures/"learning_curves.png",dpi=180,facecolor="white")
    plt.close(fig)
    lines=["# Measured active-learning results","","Generated on 9 November 2026 from the bundled implementation. Results describe an offline simulation on a fixed public benchmark.","",
           f"Training pool: {r['clean_training_pool']:,} records after removing {len(q['removed_train_source_rows'])} normalized train/evaluation overlaps. Classes: {r['classes']}. Validation: {r['validation_rows']:,} known-intent examples. Test: {r['test_rows']:,} known-intent and {r['rejection']['unknown_test_rows']:,} unknown examples.","",
           "![Learning curves and paired comparisons](figures/learning_curves.png)","","## Complete equal-budget results","",
           "| Strategy | Training labels | Mean test macro-F1 | SD across seeds | Mean test accuracy |",
           "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {names[row['strategy']]} | {int(row['labels']):,} | {float(row['test_macro_f1_mean']):.4f} | {float(row['test_macro_f1_std']):.4f} | {float(row['test_accuracy_mean']):.4f} |")
    lines += ["",f"Selected policy: **{r['selection']['strategy']}**, by mean validation macro-F1 at 3,000 labels. Deployment seed was fixed at 42. All trajectories and this choice were frozen before test learning-curve metrics were calculated.","",
              "## Paired differences versus random at 3,000 labels","","| Strategy | Mean difference in percentage points | 95% interval across seeds |","|---|---:|---|"]
    for strategy,values in r["paired_vs_random_at_3000"].items():
        lines.append(f"| {names[strategy]} | {values['mean_difference']*100:+.2f} | [{values['ci95_low']*100:+.2f}, {values['ci95_high']*100:+.2f}] |")
    lines += ["","These intervals quantify paired seed variability on this one fixed test set. They do not establish performance on new datasets. The entropy result is retained even though it is worse than random.","",
              f"Full-data reference ({r['clean_training_pool']:,} training labels): macro-F1 **{r['full_data_reference']['macro_f1']:.4f}**, accuracy **{r['full_data_reference']['accuracy']:.4f}**.","",
              "The budget comparison excludes the separate 3,000 labeled validation examples. Each strategy begins from an oracle-assisted, class-balanced 450-label seed set. No actual human annotation costs, time savings, or production outcomes were measured.","",
              "## Deployment model and rejection gate","",
              f"Seed-42 deployment model known-intent macro-F1: **{r['deployed_seed42_known_test']['macro_f1']:.4f}**. The gate threshold is the fifth percentile of maximum probabilities on known validation examples, not a formal coverage guarantee.","",
              "| Gate metric | Value |","|---|---:|"]
    for metric in ["threshold","known_acceptance_rate","unknown_rejection_rate","unknown_false_acceptance_rate","unknown_detection_roc_auc","accepted_known_accuracy"]:
        lines.append(f"| {metric} | {r['rejection'][metric]:.4f} |")
    latency=r["local_latency"]
    lines += ["","## Local timing","",f"{latency['queries']} sequential validation queries: median **{latency['median_ms']:.2f} ms**, p95 **{latency['p95_ms']:.2f} ms**. Includes encoding and classifier scoring; excludes loading, startup, and HTTP overhead. Machine-dependent, single-process measurements are not production throughput evidence.","",
              "## Reproducibility","",f"Dataset SHA-256: `{q['dataset_sha256']}`.","",
              "See `EXPERIMENT_SPEC.md`, `artifacts/learning_curves.csv`, `artifacts/acquisition_indices.json`, and the pinned dataset/model revisions for full protocol and audit details.","",
              "## Attribution","","CLINC150: Larson et al. (2019), https://aclanthology.org/D19-1131/. Dataset: https://github.com/clinc/oos-eval, identified as CC BY 3.0 by https://huggingface.co/datasets/clinc/clinc_oos. Encoder: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, Apache 2.0."]
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print("Wrote RESULTS.md and figures/learning_curves.png")


if __name__=="__main__":
    main()
