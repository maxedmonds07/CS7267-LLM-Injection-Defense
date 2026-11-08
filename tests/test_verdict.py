"""B2 verdict: pick the configuration on val only, then judge every seed against B1 on MCPTox."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import verdict as verdict_module
from deberta_run import scores_table
from harness import REPORT_COLUMNS, load_data
from verdict import main, seed_passes, select_configuration, val_selection_score, verdict


def _write(path, data, row, seed, score_fn, *, val_only=False):
    """A score file for `row`/`seed`. score_fn(set_name, frame) -> scores."""
    frames = {"val": data.val} if val_only else {"val": data.val, **data.evals}
    scores = {name: np.asarray(score_fn(name, frame), dtype=float) for name, frame in frames.items()}
    info = {"run_name": path.stem, "row": row, "seed": seed, "config": {}}
    pq.write_table(scores_table(frames, scores, info, val_only=val_only), path)
    return path


def _oracle(name, frame):
    return frame.label.to_numpy(float)


def _inverted(name, frame):
    return 1 - frame.label.to_numpy(float)


def test_verdict_values():
    assert verdict([True, True, True]) == "pass"
    assert verdict([False, False, False]) == "fail"
    assert verdict([True, False, True]) == "mixed"


def test_verdict_needs_at_least_one_seed():
    with pytest.raises(ValueError, match="no seeds"):
        verdict([])


def test_seed_passes_needs_both_conditions():
    row = dict(eval_set="mcptox", slice="all")
    ok = pd.DataFrame([{**row, "tpr_at_matched_fpr_lo": 0.25, "length_binned_auroc_lo": 0.55}])
    assert seed_passes(ok, b1_tpr=0.19)
    assert not seed_passes(ok.assign(tpr_at_matched_fpr_lo=0.18), b1_tpr=0.19)
    assert not seed_passes(ok.assign(length_binned_auroc_lo=0.49), b1_tpr=0.19)


def test_seed_passes_is_strict_and_nan_fails():
    row = dict(eval_set="mcptox", slice="all")
    ok = pd.DataFrame([{**row, "tpr_at_matched_fpr_lo": 0.25, "length_binned_auroc_lo": 0.55}])
    assert not seed_passes(ok.assign(tpr_at_matched_fpr_lo=0.19), b1_tpr=0.19)
    assert not seed_passes(ok.assign(length_binned_auroc_lo=0.5), b1_tpr=0.19)
    assert not seed_passes(ok.assign(length_binned_auroc_lo=float("nan")), b1_tpr=0.19)


def test_seed_passes_reads_only_mcptox_all():
    other = dict(tpr_at_matched_fpr_lo=0.9, length_binned_auroc_lo=0.9)
    report = pd.DataFrame([
        {"eval_set": "test", "slice": "all", **other},
        {"eval_set": "mcptox", "slice": "all", "tpr_at_matched_fpr_lo": 0.1, "length_binned_auroc_lo": 0.9},
    ])
    assert not seed_passes(report, b1_tpr=0.19)
    with pytest.raises(ValueError, match="mcptox"):
        seed_passes(report[report.eval_set == "test"], b1_tpr=0.19)


def test_val_selection_score_uses_val_rows(dataset_path, tmp_path):
    data = load_data(dataset_path)
    # Perfect on val, inverted everywhere else: only val may count.
    path = _write(tmp_path / "a.parquet", data, "row1", 7267,
                  lambda name, frame: _oracle(name, frame) if name == "val" else _inverted(name, frame))
    assert val_selection_score(path) == (1.0, 1.0)


def test_selection_never_reads_holdout_rows(dataset_path, tmp_path):
    # Row A wins on val and loses on MCPTox; row B the reverse. Selection must pick A.
    data = load_data(dataset_path)
    a = _write(tmp_path / "a.parquet", data, "row1", 7267,
               lambda name, frame: _oracle(name, frame) if name == "val" else _inverted(name, frame))
    b = _write(tmp_path / "b.parquet", data, "row2", 7267,
               lambda name, frame: _inverted(name, frame) if name == "val" else _oracle(name, frame))
    assert select_configuration({"row1": [a], "row2": [b]}) == "row1"


def test_selection_uses_the_mean_over_seeds_then_auroc(monkeypatch):
    fake = {"r1a": (1.0, 0.90), "r1b": (0.2, 0.90),   # mean TPR 0.6
            "r2a": (0.7, 0.80), "r2b": (0.7, 0.80),   # mean TPR 0.7: wins on TPR
            "r3a": (0.7, 0.95), "r3b": (0.7, 0.95)}   # ties row2 on TPR, wins on AUROC
    monkeypatch.setattr(verdict_module, "val_selection_score", lambda path: fake[path])
    assert select_configuration({"row1": ["r1a", "r1b"], "row2": ["r2a", "r2b"]}) == "row2"
    assert select_configuration({"row1": ["r1a", "r1b"], "row2": ["r2a", "r2b"],
                                 "row3": ["r3a", "r3b"]}) == "row3"


def _b1_csv(path, tpr=0.19):
    pd.DataFrame([{"detector": "tfidf", "eval_set": "mcptox", "slice": "all",
                   "tpr_at_matched_fpr": tpr}]).to_csv(path, index=False)
    return path


def test_cli_writes_seed_column_and_verdict(dataset_path, tmp_path):
    data = load_data(dataset_path)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    for seed in (7267, 7268, 7269):
        _write(scores_dir / f"row1-s{seed}.parquet", data, "row1", seed, _oracle)
    _write(scores_dir / "sweep-row2-w0.5-s7267.parquet", data, "row2", 7267, _oracle, val_only=True)
    out = tmp_path / "out"

    main(["--scores-dir", str(scores_dir), "--b1", str(_b1_csv(tmp_path / "b1.csv")),
          "--out-dir", str(out), "--data", str(dataset_path), "--n-boot", "20"])

    report = pd.read_csv(out / "b2_deberta_row1.csv")
    assert set(report.columns) == set(REPORT_COLUMNS) | {"seed"}
    assert sorted(report.seed.unique()) == [7267, 7268, 7269]
    assert not (out / "b2_deberta_row2.csv").exists()          # the sweep is val-only
    text = (out / "b2_verdict.md").read_text()
    assert "row1" in text and "0.19" in text
    assert "sweep-row2-w0.5-s7267" in text                      # appendix, val numbers only
    assert "**Verdict: pass**" in text


def test_cli_refuses_without_reported_runs(dataset_path, tmp_path):
    data = load_data(dataset_path)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    _write(scores_dir / "sweep-row2-w0.5-s7267.parquet", data, "row2", 7267, _oracle, val_only=True)
    with pytest.raises(SystemExit):
        main(["--scores-dir", str(scores_dir), "--b1", str(_b1_csv(tmp_path / "b1.csv")),
              "--out-dir", str(tmp_path / "out"), "--data", str(dataset_path), "--n-boot", "20"])


def test_b1_bar_must_be_present(dataset_path, tmp_path):
    data = load_data(dataset_path)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    _write(scores_dir / "row1-s7267.parquet", data, "row1", 7267, _oracle)
    old_b1 = tmp_path / "b1.csv"
    pd.DataFrame([{"eval_set": "mcptox", "slice": "all", "auroc": 0.675}]).to_csv(old_b1, index=False)
    with pytest.raises(ValueError, match="tpr_at_matched_fpr"):
        main(["--scores-dir", str(scores_dir), "--b1", str(old_b1),
              "--out-dir", str(tmp_path / "out"), "--data", str(dataset_path), "--n-boot", "20"])
