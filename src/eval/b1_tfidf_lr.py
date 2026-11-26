"""Family-grouped cross-validation of B1 (TF-IDF + logistic regression) on train_pool train.

Each fold holds out whole groups, and in BIPIA whole attack families (family_cv.family_folds), so
the out-of-fold scores measure transfer to attack families the model was not fit on. Reports
AUROC, TPR at 5% FPR and length-binned AUROC with 95% bootstrap intervals, overall and per source.
Reads only train_pool train: val, test and the holdouts are never loaded.

    make b1-family-cv  ->  report/baselines/b1_family_cv.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import DEFAULT_PATH, load
from family_cv import family_folds
from metrics import auroc, bootstrap_ci, length_binned_auroc, tpr_at_fpr
from tfidf import TfidfDetector

COLUMNS = ["text", "label", "source", "group", "attack_family"]
DEFAULT_OUT = Path("report/baselines/b1_family_cv.csv")


def load_train(path=DEFAULT_PATH) -> pd.DataFrame:
    """train_pool train only; dataset.load("fit") refuses holdouts and the test split."""
    return load("fit", roles=["train_pool"], splits=["train"], path=path).select(COLUMNS).to_pandas()


def out_of_fold(frame, k=5, seed=7267, make_detector=TfidfDetector):
    """(scores, folds): each row scored by the model fit on the other k-1 folds."""
    folds = family_folds(frame, k=k, seed=seed)
    scores = np.full(len(frame), np.nan)
    for fold in range(k):
        held = folds == fold
        detector = make_detector().fit(frame.text[~held], frame.label[~held])
        scores[held] = detector.predict_proba(frame.text[held])
    return scores, folds


def _binned(scores, texts, labels):
    # bootstrap_ci calls statistic(*columns, labels)
    return length_binned_auroc(scores, labels, texts)


def _row(name, scores, labels, texts, n_boot, seed):
    labels, texts = np.asarray(labels), np.asarray(texts, dtype=object)
    row = {"slice": name, "rows": len(labels), "attacks": int(labels.sum())}
    for metric, statistic, columns in (
        ("auroc", auroc, (scores,)),
        ("tpr_at_5fpr", tpr_at_fpr, (scores,)),
        ("length_binned_auroc", _binned, (scores, texts)),
    ):
        row[metric] = statistic(*columns, labels)
        row[f"{metric}_low"], row[f"{metric}_high"] = bootstrap_ci(statistic, labels, *columns,
                                                                    n_boot=n_boot, seed=seed)
    return row


def summarize(frame, scores, folds, n_boot=1000, seed=7267) -> pd.DataFrame:
    """One row per fold (point estimates), then all rows and each source with 95% intervals."""
    labels, texts = frame.label.to_numpy(), frame.text.to_numpy()
    rows = [{"slice": f"fold {f}", "rows": int((folds == f).sum()), "attacks": int(labels[folds == f].sum()),
             "auroc": auroc(scores[folds == f], labels[folds == f]),
             "tpr_at_5fpr": tpr_at_fpr(scores[folds == f], labels[folds == f]),
             "length_binned_auroc": length_binned_auroc(scores[folds == f], labels[folds == f], texts[folds == f])}
            for f in np.unique(folds)]
    rows.append(_row("all", scores, labels, texts, n_boot, seed))
    for source in sorted(frame.source.unique()):
        mask = (frame.source == source).to_numpy()
        rows.append(_row(source, scores[mask], labels[mask], texts[mask], n_boot, seed))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=DEFAULT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7267)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)

    frame = load_train(args.data)
    scores, folds = out_of_fold(frame, k=args.folds, seed=args.seed)
    table = summarize(frame, scores, folds, n_boot=args.n_boot, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Wrote {args.out}")
    return table


if __name__ == "__main__":
    main()
