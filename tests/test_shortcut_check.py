"""The shortcut check runs end to end on the fixture dataset and returns a tri-state verdict."""
import shortcut_check
from shortcut_probe import bootstrap_ci
import numpy as np
from sklearn.metrics import roc_auc_score


def test_check_runs_on_fixture(dataset_path, tmp_path):
    report = shortcut_check.main(["--detector", "tfidf", "--data", str(dataset_path), "--out", str(tmp_path / "o")])
    assert report.verdict in {"PASS", "FAIL", "INCONCLUSIVE"}
    assert (tmp_path / "o" / "verdict.txt").read_text().startswith(report.verdict)
    assert set(report.gate_table.eval_set) == {"test", "mcptox", "msb"}


def test_group_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 100)
    s = y + rng.normal(0, 1, 200)
    g = np.repeat(np.arange(50), 4)
    point, lo, hi = bootstrap_ci(roc_auc_score, y, s, g)
    assert lo <= point <= hi


def _score_file(path, data):
    """A B2-style score file covering val and the three eval sets, and nothing else."""
    import pyarrow.parquet as pq
    from deberta_run import scores_table

    frames = {"val": data.val, **data.evals}
    scores = {k: v.text.str.contains("ignore previous").to_numpy(float) for k, v in frames.items()}
    info = {"run_name": path.stem, "row": "row1", "seed": 7267, "config": {}}
    pq.write_table(scores_table(frames, scores, info, val_only=False), path)
    return path


def test_deberta_score_file_runs_and_skips_rescoring_checks(dataset_path, tmp_path):
    """Regression (PR #40 review): a score file cannot score padded texts, so padding must be
    skipped and said so, instead of crashing with 'rows have no score'."""
    from harness import load_data

    path = _score_file(tmp_path / "row1-s7267.parquet", load_data(dataset_path))
    report = shortcut_check.main(["--detector", "deberta", "--scores", str(path), "--data", str(dataset_path)])
    assert report.verdict in {"PASS", "FAIL", "INCONCLUSIVE"}
    assert report.padding == {}
    assert any(r.startswith("padding check skipped") for r in report.reasons)
    assert any(r.startswith("label shuffle skipped") for r in report.reasons)
