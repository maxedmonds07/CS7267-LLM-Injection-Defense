"""Evaluating B2 from score files: Colab scores, the unchanged harness evaluates locally."""

import pyarrow.parquet as pq
import pytest

from deberta_run import scores_table
from harness import load_data
from precomputed import PrecomputedScores
from train import main


def _write(path, data, *, sets=("val", "test", "mcptox", "msb"), seed=7267):
    frames = {"val": data.val, **data.evals}
    frames = {k: v for k, v in frames.items() if k in sets}
    scores = {k: v.text.str.contains("ignore previous").to_numpy(float) for k, v in frames.items()}
    info = {"run_name": path.stem, "row": "row1", "seed": seed, "config": {"lr": 2e-5}}
    pq.write_table(scores_table(frames, scores, info, val_only=sets == ("val",)), path)
    return path


def test_harness_runs_on_a_score_file(dataset_path, tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    path = _write(tmp_path / "row1-s7267.parquet", load_data(dataset_path))
    report = main(["--detector", "deberta", "--scores", str(path), "--data", str(dataset_path),
                   "--n-boot", "20"])
    assert set(report.detector) == {"deberta_row1"}
    assert report.query("eval_set == 'mcptox' and slice == 'all'").auroc.item() == 1.0


def test_metadata_becomes_name_seed_and_params(dataset_path, tmp_path):
    scores = PrecomputedScores(_write(tmp_path / "row1-s7268.parquet", load_data(dataset_path), seed=7268))
    assert (scores.name, scores.seed) == ("deberta_row1", 7268)
    assert scores.params["run_name"] == "row1-s7268"
    assert scores.params["config.lr"] == 2e-5
    assert scores.val_only is False


def test_missing_rows_fail_loudly(dataset_path, tmp_path):
    data = load_data(dataset_path)
    path = _write(tmp_path / "sweep.parquet", data, sets=("val",))
    assert PrecomputedScores(path).val_only is True
    # The fixture repeats texts across sources, so add one that no file row covers.
    texts = ["a tool description nobody scored", *data.evals["mcptox"].text]
    with pytest.raises(ValueError, match=r"1 of 73 rows have no score"):
        PrecomputedScores(path).predict_proba(texts)


def test_deberta_needs_a_score_file(dataset_path):
    with pytest.raises(SystemExit):
        main(["--detector", "deberta", "--data", str(dataset_path)])


def test_conflicting_duplicate_scores_rejected(dataset_path, tmp_path):
    data = load_data(dataset_path)
    frames = {"val": data.val.iloc[:2], "test": data.val.iloc[:2]}  # same texts twice
    scores = {"val": [0.1, 0.2], "test": [0.9, 0.2]}
    path = tmp_path / "bad.parquet"
    pq.write_table(scores_table(frames, scores, {"run_name": "bad", "row": "row1", "seed": 1,
                                                 "config": {}}, val_only=False), path)
    with pytest.raises(ValueError, match="different scores"):
        PrecomputedScores(path)
