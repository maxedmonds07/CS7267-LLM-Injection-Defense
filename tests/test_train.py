"""The training CLI: fit, compare with the length baseline, log to MLflow when asked."""

import dataclasses
import math

import mlflow
import numpy as np
import pandas as pd
import pytest

import train
from train import EXPERIMENT, TFIDF_C_GRID, compare, fit_detector, main, metric_items
from harness import load_data


def test_tfidf_picks_c_from_the_grid(dataset_path):
    detector, params = fit_detector("tfidf", load_data(dataset_path))
    assert detector.name == "tfidf"
    assert params["C"] in TFIDF_C_GRID
    assert 0.0 <= params["val_auroc"] <= 1.0


def test_tfidf_refuses_a_single_label_val_split(dataset_path):
    data = load_data(dataset_path)
    data = dataclasses.replace(data, val=data.val[data.val.label == 0])
    with pytest.raises(ValueError, match="both labels"):
        fit_detector("tfidf", data)


def test_perplexity_uses_frozen_model_without_fit(dataset_path):
    model, params = fit_detector("perplexity", load_data(dataset_path))
    assert model.name == "perplexity_b0"
    assert model._model is None
    assert params["model_name"] == "distilbert/distilgpt2"


def test_perplexity_uses_the_same_harness(dataset_path, monkeypatch):
    # Stub the expensive model inference; this verifies training/evaluation plumbing.
    monkeypatch.setattr(
        train.PerplexityDetector,
        "predict_proba",
        lambda self, texts: np.array([float("ignore previous" in t) for t in texts]),
    )
    report = main(["--detector", "perplexity", "--data", str(dataset_path), "--n-boot", "20"])
    assert set(report.detector) == {"perplexity_b0"}
    assert set(report.eval_set) == {"test", "mcptox", "msb"}


def test_metric_keys_skip_nan():
    report = pd.DataFrame(
        [{"eval_set": "msb", "slice": "attack_surface=tool_output", "auroc": math.nan, "tpr": 0.8, "fpr":
          math.nan}]
    )
    assert metric_items(report) == {"tpr/msb/attack_surface-tool_output": 0.8}


def test_compare_puts_length_next_to_the_detector():
    report = pd.DataFrame([{"eval_set": "mcptox", "slice": "all", "auroc": 0.9, "tpr": 0.5}])
    reference = pd.DataFrame([{"eval_set": "mcptox", "slice": "all", "auroc": 0.77, "tpr": 0.1}])
    row = compare(report, reference).iloc[0]
    assert (row.auroc, row.auroc_length, row.tpr, row.tpr_length) == (0.9, 0.77, 0.5, 0.1)


def test_runs_without_tracking(dataset_path, capsys, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    report = main(["--detector", "length", "--data", str(dataset_path), "--n-boot", "20"])
    assert set(report.detector) == {"length_only"}
    assert "not logged to MLflow" in capsys.readouterr().out


def test_logs_a_run(dataset_path, tmp_path, monkeypatch):
    # With a SQLite backend MLflow writes artifacts to ./mlruns; keep them out of the repo.
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    main(["--detector", "tfidf", "--data", str(dataset_path), "--n-boot", "20", "--tracking-uri", uri])
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    assert len(runs) == 1
    assert runs.loc[0, "params.detector"] == "tfidf"
    assert runs.loc[0, "metrics.auroc/mcptox/all"] == 1.0
