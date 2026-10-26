"""Evaluate every detector the same way.
Fit on train_pool train, choose hyperparameters and the threshold on train_pool val, then report
on the train_pool test split and on every row of MCPTox (LOSO) and MSB (transfer).
"""
import pandas as pd
from metrics import auroc, bootstrap_auroc_ci, length_separability, rates_at_threshold, threshold_at_fpr
from dataclasses import dataclass
from dataset import DEFAULT_PATH, load

COLUMNS = ["text", "label", "source", "attack_kind", "attack_surface"]

@dataclass
class Data:
    train: pd.DataFrame
    val: pd.DataFrame
    evals: dict[str, pd.DataFrame]


def load_data(path=DEFAULT_PATH) -> Data:
    def rows(purpose, **filters):
        return load(purpose, path=path, **filters).select(COLUMNS).to_pandas()

    return Data(
        train=rows("fit", roles=["train_pool"], splits=["train"]),
        val=rows("fit", roles=["train_pool"], splits=["val"]),
        evals={
            "test": rows("eval", roles=["train_pool"], splits=["test"]),
            # Held out whole: never filter these by split.
            "mcptox": rows("eval", sources=["mcptox"]),
            "msb": rows("eval", sources=["msb"]),
        },
    )

def slices(frame):
    """(name, rows) pairs. Source and surface slices keep both labels from that group. An
    attack_kind slice pairs that kind's attacks with every benign row, because benign rows
    have no attack_kind. A column with one value adds nothing to "all" and is skipped."""
    yield "all", frame
    for column in ("source", "attack_surface"):
        values = sorted(frame[column].unique())
        if len(values) > 1:
            for value in values:
                yield f"{column}={value}", frame[frame[column] == value]
    benign = frame[frame.label == 0]
    kinds = sorted(frame.loc[frame.label == 1, "attack_kind"].dropna().unique())
    if len(kinds) > 1:
        for kind in kinds:
            yield f"attack_kind={kind}", pd.concat([benign, frame[frame.attack_kind == kind]])

REPORT_COLUMNS = [
    "detector", "eval_set", "slice", "n_benign", "n_adversarial",
    "auroc", "auroc_lo", "auroc_hi", "tpr", "fpr", "length_separability",
]

def evaluate(detector, data, target_fpr=0.01, n_boot=1000):
    """(threshold, report) for a fitted detector. The threshold is set once, on val benign scores,
    and applied unchanged to every eval set and slice."""
    val_scores = detector.predict_proba(data.val.text)
    threshold = threshold_at_fpr(val_scores[data.val.label.to_numpy() == 0], target_fpr)
    records = []
    for set_name, frame in data.evals.items():
        frame = frame.assign(score=detector.predict_proba(frame.text))
        for slice_name, rows in slices(frame):
            low, high = bootstrap_auroc_ci(rows.score, rows.label, n_boot=n_boot)
            tpr, fpr = rates_at_threshold(rows.score, rows.label, threshold)
            both_labels = rows.label.nunique() == 2
            records.append(
                {
                    "detector": detector.name,
                    "eval_set": set_name,
                    "slice": slice_name,
                    "n_benign": int((rows.label == 0).sum()),
                    "n_adversarial": int((rows.label == 1).sum()),
                    "auroc": auroc(rows.score, rows.label),
                    "auroc_lo": low,
                    "auroc_hi": high,
                    "tpr": tpr,
                    "fpr": fpr,
                    "length_separability": (
                        length_separability(rows.text, rows.label) if both_labels else float("nan")
                    ),
                }
            )
    return threshold, pd.DataFrame(records, columns=REPORT_COLUMNS)