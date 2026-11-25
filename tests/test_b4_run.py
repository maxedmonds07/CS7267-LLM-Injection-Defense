"""Real small-model end-to-end training, scoring, evaluation and experiment orchestration."""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_b4 import tiny_b4
from test_b4_data import contexts_path
from b4_data import digest, load_frames
from b4_run import B4Scores, evaluate_scores, input_identity, score_model, train_model
from b4_experiments import freeze_manifest, read_config, run_experiments


def test_same_candidate_gets_its_own_score_for_each_context(tmp_path):
    rows = pd.DataFrame({"id": ["first", "second"], "text": ["same candidate"] * 2,
                         "context": ["task A", "task B"], "label": [0, 1]})
    scored = rows[["id", "label"]].assign(text_sha256=rows.text.map(digest),
        context_sha256=rows.context.map(digest), score=[0.2, 0.9], eval_set="val")
    table = pa.Table.from_pandas(scored, preserve_index=False).replace_schema_metadata({
        b"b4": json.dumps({"run_name": "fixture"}).encode()})
    path = tmp_path / "scores.parquet"
    pq.write_table(table, path)
    detector = B4Scores(path, {"val": rows}, {})
    texts = rows.set_index("id").text.iloc[::-1]
    assert detector.predict_proba(texts).tolist() == [0.9, 0.2]


def test_train_score_evaluate_and_tamper_detection(tiny_b4, dataset_path, contexts_path, tmp_path):
    config = {k: v for k, v in tiny_b4.items() if k != "device"}
    config["epochs"] = 1
    model = tmp_path / "model"
    train_model(dataset_path, contexts_path, model, model_config=config, device="cpu")
    out = tmp_path / "scores.parquet"
    score_model(dataset_path, contexts_path, model, out, device="cpu")
    report = evaluate_scores(dataset_path, contexts_path, out, tmp_path / "report.csv", n_boot=3)
    assert set(report.eval_set) == {"test", "mcptox", "msb"}
    frames = load_frames(dataset_path, contexts_path, evaluation=True)
    frames["mcptox"].loc[0, "context"] = "changed task"
    with pytest.raises(ValueError, match="context_sha256"):
        B4Scores(out, frames, input_identity(dataset_path, contexts_path))


def test_smoke_cannot_score_holdouts(tiny_b4, dataset_path, contexts_path, tmp_path):
    config = {k: v for k, v in tiny_b4.items() if k != "device"}
    train_model(dataset_path, contexts_path, tmp_path / "model", model_config=config, limit=8, device="cpu")
    with pytest.raises(ValueError, match="validation only"):
        score_model(dataset_path, contexts_path, tmp_path / "model", tmp_path / "scores.parquet", device="cpu")


def test_full_grid_and_reports(tiny_b4, dataset_path, contexts_path, tmp_path):
    config = read_config("configs/b4.yaml")
    config["model"].update({k: v for k, v in tiny_b4.items() if k != "device"})
    config["model"]["epochs"] = 1
    config["seeds"] = [7267, 7268]
    config["evaluation"]["n_boot"] = 2
    config["search"]["enabled"] = True
    result = run_experiments(config, tmp_path / "grid", dataset_path, contexts_path, mode="full", device="cpu")
    assert set(result.experiment) == {"text_control", "pair_only", "relevance", "relevance_bipia"}
    assert result.n_seeds.max() == 2
    reports = tmp_path / "grid/reports"
    for name in ("seed_summary.csv", "ablation_deltas.csv", "reference_length.csv", "reference_tfidf.csv",
                 "summary.md", "auroc.png", "selection.json"):
        assert (reports / name).exists()
    for path in (tmp_path / "grid/sweep").glob("*/scores.parquet"):
        assert set(pq.read_table(path)["eval_set"].to_pylist()) == {"val"}
    assert set(pd.read_csv(reports / "ablation_deltas.csv").seed) == {7267, 7268}


def test_manifest_refuses_reuse_with_changed_config(tmp_path):
    freeze_manifest(tmp_path / "manifest.json", {"lr": 0.01})
    with pytest.raises(ValueError, match="changed"):
        freeze_manifest(tmp_path / "manifest.json", {"lr": 0.02})
