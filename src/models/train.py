"""Fit a detector, evaluate it with the shared harness next to the length baseline, and log the
run to MLflow.

    PYTHONPATH=src/data:src/eval:src/models uv run python src/models/train.py --detector tfidf

Pass --tracking-uri (or set MLFLOW_TRACKING_URI, e.g. http://127.0.0.1:5000 with the team
server running) to log; otherwise the report is printed and not logged.
"""

import argparse
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import LengthBaseline
from dataset import DEFAULT_PATH
from harness import evaluate, load_data
from metrics import auroc
from tfidf import TfidfDetector

EXPERIMENT = "phase3-detectors"
TFIDF_C_GRID = (0.1, 1.0, 10.0)
SHOWN = ["eval_set", "slice", "n_benign", "n_adversarial", "auroc", "auroc_lo", "auroc_hi", "auroc_length",
         "tpr", "tpr_length", "fpr"]


def fit_detector(name, data):
    """Fit on train_pool train. TF-IDF picks its regularization C by val AUROC."""
    if name == "length":
        return LengthBaseline().fit(data.train.text, data.train.label), {}
    fitted = [TfidfDetector(C=c).fit(data.train.text, data.train.label) for c in TFIDF_C_GRID]
    val_aurocs = [auroc(model.predict_proba(data.val.text), data.val.label) for model in fitted]
    best = int(np.argmax(val_aurocs))
    return fitted[best], {"C": TFIDF_C_GRID[best], "val_auroc": val_aurocs[best]}


def metric_items(report):
    """MLflow names as {metric}/{eval_set}/{slice}; MLflow keys cannot contain "=". NaN is dropped."""
    items = {}
    for row in report.itertuples():
        for metric in ("auroc", "tpr", "fpr"):
            value = getattr(row, metric)
            if not math.isnan(value):
                items[f"{metric}/{row.eval_set}/{row.slice}".replace("=", "-")] = float(value)
    return items


def compare(report, reference):
    """The detector's report with the length baseline's AUROC and TPR alongside."""
    keep = reference[["eval_set", "slice", "auroc", "tpr"]].rename(columns={"auroc": "auroc_length", "tpr":
                                                                          "tpr_length"})
    return report.merge(keep, on=["eval_set", "slice"], how="left")


def log_run(uri, name, params, threshold, args, report, detector):
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "detector": name,
                "target_fpr": args.target_fpr,
                "threshold": threshold,
                "n_boot": args.n_boot,
                "dataset_md5": hashlib.md5(Path(args.data).read_bytes()).hexdigest(),
                **params,
            }
        )
        mlflow.log_metrics(metric_items(report))
        mlflow.log_text(report.to_csv(index=False), "report.csv")
        if hasattr(detector, "top_features"):
            features = detector.top_features(50)
            mlflow.log_text("\n".join(f"{k}\t{f}" for k, fs in features.items() for f in fs), "top_features.tsv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--detector", choices=["length", "tfidf"], required=True)
    parser.add_argument("--data", default=str(DEFAULT_PATH), help="dataset.parquet to read")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    args = parser.parse_args(argv)

    data = load_data(args.data)
    detector, params = fit_detector(args.detector, data)
    threshold, report = evaluate(detector, data, args.target_fpr, args.n_boot)
    if args.detector == "length":
        shown = report.assign(auroc_length=report.auroc, tpr_length=report.tpr)
    else:
        _, reference = evaluate(fit_detector("length", data)[0], data, args.target_fpr, args.n_boot)
        shown = compare(report, reference)

    print(f"{detector.name}: threshold {threshold:.4f} at {args.target_fpr:.0%} FPR on val; {params or 'no hyperparameters'}")
    print(shown[SHOWN].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.tracking_uri:
        log_run(args.tracking_uri, detector.name, params, threshold, args, report, detector)
        print(f"logged to MLflow experiment {EXPERIMENT!r} at {args.tracking_uri}")
    else:
        print("not logged to MLflow: pass --tracking-uri or set MLFLOW_TRACKING_URI")
    return report


if __name__ == "__main__":
    main()