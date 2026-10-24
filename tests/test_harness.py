"""The evaluation harness: which rows each step sees, and how eval sets are sliced."""

from harness import COLUMNS, REPORT_COLUMNS, evaluate, load_data, slices
import numpy as np
from metrics import threshold_at_fpr
import math

def test_train_and_val_come_from_the_training_pool(dataset_path):
    data = load_data(dataset_path)
    assert set(data.train.source) == set(data.val.source) == {"poisonedrag", "bipia"}
    assert len(data.train) == len(data.val) == 48
    assert list(data.train.columns) == COLUMNS


def test_eval_sets_in_order(dataset_path):
    data = load_data(dataset_path)
    assert list(data.evals) == ["test", "mcptox", "msb"]
    assert set(data.evals["test"].source) == {"poisonedrag", "bipia"}


def test_holdouts_keep_every_split(dataset_path):
    # MCPtox rows carry train/val/test values; all of them are held out.
    data = load_data(dataset_path)
    assert len(data.evals["mcptox"]) == 72
    assert len(data.evals["msb"]) == 16


def test_slices_skip_columns_with_one_value(dataset_path):
    names = [name for name, _ in slices(load_data(dataset_path).evals["mcptox"])]
    assert names == ["all"]


def test_source_and_surface_slices(dataset_path):
    names = [name for name, _ in slices(load_data(dataset_path).evals["msb"])]
    assert names == ["all", "attack_surface=tool_description", "attack_surface=tool_output"]


def test_attack_kind_slice_keeps_every_benign_row(dataset_path):
    test = load_data(dataset_path).evals["test"]
    named = dict(slices(test))
    for kind in ("instruction", "poisoning"):
        part = named[f"attack_kind={kind}"]
        assert (part.label == 0).sum() == (test.label == 0).sum()
        assert set(part.loc[part.label == 1, "attack_kind"]) == {kind}

class Oracle:
    name = "oracle"

    def predict_proba(self, texts):
        return np.array([1.0 if "ignore previous" in text else 0.0 for text in texts])

def test_report_shape(dataset_path):
    _, report = evaluate(Oracle(), load_data(dataset_path), n_boot=20)
    assert list(report.columns) == REPORT_COLUMNS
    assert set(report.eval_set) == {"test", "mcptox", "msb"}
    assert set(report.detector) == {"oracle"}

def test_perfect_detector(dataset_path):
    _, report = evaluate(Oracle(), load_data(dataset_path), n_boot=20)
    whole = report[report.slice == "all"]
    assert (whole.auroc == 1.0).all()
    assert (whole.tpr == 1.0).all()
    assert (whole.fpr == 0.0).all()

def test_threshold_comes_from_val_benign_scores(dataset_path):
    data = load_data(dataset_path)
    threshold, _ = evaluate(Oracle(), data, target_fpr=0.05, n_boot=20)
    val_scores = Oracle().predict_proba(data.val.text)
    assert threshold == threshold_at_fpr(val_scores[data.val.label.to_numpy() == 0], 0.05)

def test_single_label_slice_reports_tpr_only(dataset_path):
    _, report = evaluate(Oracle(), load_data(dataset_path), n_boot=20)
    row = report[(report.eval_set == "msb") & (report.slice == "attack_surface=tool_output")].iloc[0]
    assert row.n_benign == 0 and row.n_adversarial == 4
    assert row.tpr == 1.0
    for column in ("auroc", "auroc_lo", "auroc_hi", "fpr", "length_separability"):
        assert math.isnan(row[column])
