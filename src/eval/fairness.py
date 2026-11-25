"""Disaggregated metrics: AUROC, TPR and FPR per group, at each detector's deployed threshold.

    PYTHONPATH=src/data:src/eval:src/models uv run python src/eval/fairness.py

Reads report/phase5/scores.parquet and thresholds.json (phase5_scores.py) and writes
report/phase5/fairness.csv. The data contains no people, so the groups are the ones a deployed
detector treats differently: source, attack surface, attack kind, text length quartile and
script. FPR, the rate at which benign content is blocked, carries a Wilson interval.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import auroc, bootstrap_auroc_ci

ROOT = Path(__file__).resolve().parents[2]
PHASE5 = ROOT / "report" / "phase5"
EVAL_SETS = ("test", "mcptox", "msb")
COLUMNS = ["detector", "eval_set", "group_by", "group", "n_benign", "n_adversarial", "auroc",
           "auroc_lo", "auroc_hi", "tpr", "fpr", "fpr_lo", "fpr_hi"]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials; (nan, nan) when n is 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def length_quartiles(frame: pd.DataFrame) -> pd.Series:
    """Q1..Q4 by character length, with edges from the benign rows of this frame, so each benign
    quartile holds about a quarter of benign rows and attacks fall wherever their length puts them."""
    benign = frame.loc[frame.label == 0, "n_chars"]
    if benign.empty:
        return pd.Series("all", index=frame.index)
    edges = np.quantile(benign, [0.25, 0.5, 0.75])
    return pd.Series(np.searchsorted(edges, frame.n_chars, side="right"), index=frame.index).map(
        lambda q: f"Q{q + 1}")


def groups(frame: pd.DataFrame):
    """(group_by, group, rows). attack_kind pairs that kind's attacks with every benign row,
    because benign rows have no kind (as in harness.slices). A grouping with one value is kept
    only as 'all'."""
    yield "all", "all", frame
    for column in ("source", "attack_surface", "script"):
        values = sorted(frame[column].dropna().unique())
        if len(values) > 1:
            for value in values:
                yield column, value, frame[frame[column] == value]
    benign = frame[frame.label == 0]
    kinds = sorted(frame.loc[frame.label == 1, "attack_kind"].dropna().unique())
    if len(kinds) > 1:
        for kind in kinds:
            yield "attack_kind", kind, pd.concat([benign, frame[frame.attack_kind == kind]])
    quartile = length_quartiles(frame)
    for value in sorted(quartile.unique()):
        yield "length", value, frame[quartile == value]


def disaggregate(scores: pd.DataFrame, thresholds: dict, n_boot: int = 1000) -> pd.DataFrame:
    records = []
    for detector, threshold in thresholds.items():
        for eval_set in EVAL_SETS:
            frame = scores[scores.eval_set == eval_set]
            for group_by, group, rows in groups(frame):
                labels, values = rows.label.to_numpy(), rows[detector].to_numpy()
                flagged = values >= threshold
                n_ben, n_adv = int((labels == 0).sum()), int((labels == 1).sum())
                false_pos = int((flagged & (labels == 0)).sum())
                both = n_ben > 0 and n_adv > 0
                low, high = bootstrap_auroc_ci(values, labels, n_boot=n_boot) if both else (np.nan, np.nan)
                fpr_lo, fpr_hi = wilson(false_pos, n_ben)
                records.append({
                    "detector": detector, "eval_set": eval_set, "group_by": group_by, "group": group,
                    "n_benign": n_ben, "n_adversarial": n_adv,
                    "auroc": auroc(values, labels) if both else np.nan, "auroc_lo": low, "auroc_hi": high,
                    "tpr": (flagged & (labels == 1)).sum() / n_adv if n_adv else np.nan,
                    "fpr": false_pos / n_ben if n_ben else np.nan, "fpr_lo": fpr_lo, "fpr_hi": fpr_hi,
                })
    return pd.DataFrame(records, columns=COLUMNS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=str(PHASE5))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    folder = Path(args.dir)
    scores = pd.read_parquet(folder / "scores.parquet")
    thresholds = json.loads((folder / "thresholds.json").read_text())
    table = disaggregate(scores, thresholds, args.n_boot)
    table.to_csv(folder / "fairness.csv", index=False)
    print(f"wrote {len(table)} rows to {folder / 'fairness.csv'}")


if __name__ == "__main__":
    main()
