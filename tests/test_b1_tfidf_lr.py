"""Family-grouped CV of B1: reads only train_pool train, scores every row out of fold, and
reports finite intervals."""

import numpy as np
import pandas as pd

from b1_tfidf_lr import load_train, main, out_of_fold


def test_reads_only_train_pool_train(dataset_path):
    frame = load_train(dataset_path)
    rows = pd.read_parquet(dataset_path)
    expected = rows[(rows.role == "train_pool") & (rows.split == "train")]
    assert sorted(frame.text) == sorted(expected.text)


def test_each_row_is_scored_by_a_model_that_did_not_see_its_group(dataset_path):
    frame = load_train(dataset_path)
    seen = []

    class Recorder:
        def fit(self, texts, labels):
            seen.append(set(texts.index))
            return self

        def predict_proba(self, texts):
            return np.full(len(texts), 0.5)

    scores, folds = out_of_fold(frame, k=3, make_detector=Recorder)
    assert not np.isnan(scores).any()
    assert frame.assign(fold=folds).groupby("group").fold.nunique().max() == 1
    for fold, fitted in enumerate(seen):
        assert fitted.isdisjoint(frame.index[folds == fold])


def test_writes_intervals_for_all_rows_and_each_source(dataset_path, tmp_path):
    out = tmp_path / "cv.csv"
    table = main(["--data", str(dataset_path), "--out", str(out), "--folds", "3", "--n-boot", "50"])
    assert out.exists()
    assert {"all", "bipia", "poisonedrag"} <= set(table.slice)
    overall = table.set_index("slice").loc["all"]
    for metric in ("auroc", "tpr_at_5fpr"):
        assert overall[f"{metric}_low"] <= overall[metric] <= overall[f"{metric}_high"]
