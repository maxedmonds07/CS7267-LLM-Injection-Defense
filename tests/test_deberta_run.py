"""The Colab entry point: train a DeBERTa run, then write its score file."""

import json

import pyarrow.parquet as pq
import pytest

import deberta_run


def test_train_refuses_holdout_sources(dataset_path, tmp_path):
    with pytest.raises(SystemExit):
        deberta_run.main(["train", "--row", "row1", "--seed", "1", "--model-out", str(tmp_path / "m"),
                          "--train-sources", "mcptox", "--data", str(dataset_path)])


def test_train_then_score_val_only(tiny, dataset_path, tmp_path):
    model = tmp_path / "model"
    deberta_run.main(["train", "--row", "row2", "--seed", "7267", "--model-out", str(model),
                      "--pairwise-weight", "0", "--limit", "16", "--epochs", "1", "--no-fp16",
                      "--device", "cpu", "--data", str(dataset_path)])
    run = json.loads((model / "run.json").read_text())
    assert (run["run_name"], run["row"], run["seed"]) == ("row2-s7267", "row2", 7267)
    assert run["train_sources"] == ["bipia", "poisonedrag"]

    out = tmp_path / "scores.parquet"
    deberta_run.main(["score", "--model", str(model), "--out", str(out), "--val-only",
                      "--device", "cpu", "--data", str(dataset_path)])
    table = pq.read_table(out)
    assert set(table.column("eval_set").to_pylist()) == {"val"}
    meta = json.loads(table.schema.metadata[b"deberta"])
    assert meta["val_only"] is True and meta["run_name"] == "row2-s7267"
    assert meta["selected_epoch"] == 0


def test_bipia_only_trains_on_bipia_rows(tiny, dataset_path, tmp_path, monkeypatch):
    seen = {}
    from deberta import DebertaDetector

    original = DebertaDetector.fit

    def spy(self, texts, labels, groups=None, val=None):
        seen["groups"], seen["val_n"] = list(groups), len(val[0])
        return original(self, texts, labels, groups=groups, val=val)

    monkeypatch.setattr(DebertaDetector, "fit", spy)
    deberta_run.main(["train", "--row", "row3", "--seed", "1", "--model-out", str(tmp_path / "m"),
                      "--train-sources", "bipia", "--epochs", "1", "--no-fp16", "--device", "cpu",
                      "--data", str(dataset_path)])
    assert all(g.startswith("bipia") for g in seen["groups"])
    assert seen["val_n"] == 48  # val is the full train_pool val, both sources (spec decision 13)


def test_pairwise_run_records_its_twin_pairs(tiny, dataset_path, tmp_path):
    model = tmp_path / "model"
    deberta_run.main(["train", "--row", "row2", "--seed", "7267", "--model-out", str(model),
                      "--pairwise-weight", "0.5", "--epochs", "1", "--no-fp16",
                      "--device", "cpu", "--data", str(dataset_path)])
    run = json.loads((model / "run.json").read_text())
    assert run["pairwise_weight"] == 0.5
    assert run["twin_pairs"] > 0
