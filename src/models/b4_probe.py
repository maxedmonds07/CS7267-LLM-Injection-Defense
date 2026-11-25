"""Measure how much signal the two explicit relevance features carry on validation."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from b4_data import DEFAULT_CONTEXTS, load_frames, pairs
from b4_features import FEATURE_NAMES, RelevanceFeatures
from b4_run import input_identity
from dataset import DEFAULT_PATH
from metrics import auroc, bootstrap_auroc_ci, rates_at_threshold, threshold_at_fpr


def run_probe(data, contexts, out):
    frames = load_frames(data, contexts, evaluation=False)
    train, val = frames["train"], frames["val"]
    train_pairs = pairs(train)
    extractor = RelevanceFeatures().fit(train_pairs)
    model = LogisticRegression(C=1.0, random_state=7267).fit(extractor.transform(train_pairs), train.label)
    scores = model.predict_proba(extractor.transform(pairs(val)))[:, 1]
    labels = val.label.to_numpy()
    threshold = threshold_at_fpr(scores[labels == 0], 0.01)
    tpr, fpr = rates_at_threshold(scores, labels, threshold)
    low, high = bootstrap_auroc_ci(scores, labels, n_boot=1000)
    result = {"detector": "logistic_regression_relevance_features_only", "eval_set": "train_pool_val",
              "feature_names": list(FEATURE_NAMES), "n_train": len(train), "n_val": len(val),
              "val_auroc": auroc(scores, labels), "bootstrap_95": [low, high],
              "val_threshold": threshold, "val_tpr": tpr, "val_fpr": fpr,
              "weights": model.coef_[0].tolist(), **input_identity(data, contexts)}
    if not np.isfinite(scores).all():
        raise FloatingPointError("Non-finite relevance probe predictions")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Relevance alone: validation AUROC {result['val_auroc']:.4f} [{low:.4f}, {high:.4f}], "
          f"TPR {tpr:.4f}, FPR {fpr:.4f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_PATH))
    parser.add_argument("--contexts", default=str(DEFAULT_CONTEXTS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_probe(args.data, args.contexts, args.out)
