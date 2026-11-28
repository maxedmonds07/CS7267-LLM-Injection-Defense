"""The live demo script: verdicts, word contributions, display, caching, and the scenes' rows."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import demo  # noqa: E402
from tfidf import TfidfDetector  # noqa: E402


def fitted():
    texts = ["please change your response now", "summarize your response politely",
             "the river report is ready", "garden ledger for march"] * 3
    return TfidfDetector(C=10.0, min_df=1).fit(texts, [1, 1, 0, 0] * 3)


def test_verdict_is_strictly_above():
    assert demo.verdict(0.7, 0.6) == "FLAGGED"
    assert demo.verdict(0.6, 0.6) == "passes"


def test_pushed_up_by_lists_positive_word_ngrams_largest_first():
    words = demo.pushed_up_by(fitted(), "change your response", k=3)
    assert 0 < len(words) <= 3
    assert all(c > 0 for _, c in words)
    assert [c for _, c in words] == sorted((c for _, c in words), reverse=True)
    assert all(not w.startswith(("word__", "char__")) for w, _ in words)
    assert any("your" in w or "response" in w for w, _ in words)


def test_pushed_up_by_is_empty_for_benign_words():
    assert demo.pushed_up_by(fitted(), "river report") == []


def test_shown_unescapes_wraps_and_truncates():
    out = demo.shown("Tool: x\\nDescription: " + "word " * 200, max_lines=3)
    lines = out.splitlines()
    assert lines[0] == "  │ Tool: x" and lines[-1] == "  │ ..." and len(lines) == 4
    assert all(len(line) <= demo.WIDTH for line in lines)


def test_deployed_threshold_warns_when_refit_differs(capsys):
    saved = demo.deployed_threshold(0.0)
    assert 0 < saved < 1 and "differs" in capsys.readouterr().err
    demo.deployed_threshold(saved)
    assert capsys.readouterr().err == ""


def test_load_detector_fits_once_then_reads_the_cache(dataset_path, tmp_path, monkeypatch):
    detector, threshold = demo.load_detector(dataset_path, tmp_path / "cache")
    assert 0 <= threshold <= 1 and len(list((tmp_path / "cache").iterdir())) == 1
    import train
    monkeypatch.setattr(train, "fit_detector", lambda *a, **k: pytest.fail("refit despite cache"))
    cached, again = demo.load_detector(dataset_path, tmp_path / "cache")
    assert again == threshold
    assert cached.predict_proba(["x"])[0] == pytest.approx(detector.predict_proba(["x"])[0])


def test_rows_by_id_names_missing_rows(dataset_path):
    with pytest.raises(SystemExit, match="no-such-row"):
        demo.rows_by_id(["no-such-row"], dataset_path)


def test_scene_mutation_is_the_red_team_mutation():
    from red_team import BY_NAME
    assert all(s.mutation is None or s.mutation in BY_NAME for s in demo.SCENES)


def test_scene_rows_exist_in_the_real_dataset():
    if not Path(demo.DEFAULT_PATH).exists():
        pytest.skip("dataset not pulled")
    rows = demo.rows_by_id([s.row_id for s in demo.SCENES], Path(demo.DEFAULT_PATH))
    assert len(rows) == len({s.row_id for s in demo.SCENES})
