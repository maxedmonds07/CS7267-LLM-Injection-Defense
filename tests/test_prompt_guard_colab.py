"""Prompt Guard 2 scoring script: chunking, batching and a score file the harness can read."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from harness import Data  # noqa: E402
from precomputed import PrecomputedScores  # noqa: E402
from prompt_guard_colab import (  # noqa: E402
    NAME, PromptGuardDetector, malicious_index, max_by_owner, max_pool_scores, native_summary, score_file,
    windows,
)


class Stub:
    """Scores len/100; pretends texts longer than 10 characters were split into chunks."""

    def __init__(self):
        self.calls, self.chunked = 0, 0

    def __call__(self, texts):
        self.calls += 1
        self.chunked += sum(len(t) > 10 for t in texts)
        return np.array([min(len(t) / 100, 1.0) for t in texts])


def frame(prefix, n):
    return pd.DataFrame({"text": [f"{prefix} {'x' * i}" for i in range(n)], "label": [i % 2 for i in range(n)]})


def data():
    return Data(train=frame("train", 4), val=frame("val", 6),
                evals={"mcptox": frame("mcp", 5), "bipia_test": frame("bipia", 30)})


def test_predict_proba_batches():
    stub = Stub()
    scores = PromptGuardDetector(scorer=stub, batch_size=4).predict_proba([f"t{i}" for i in range(10)])
    assert stub.calls == 3 and len(scores) == 10


def wrap(ids):
    return [0, *ids, 2]  # [CLS] ids [SEP]


def test_windows_cover_long_texts_and_keep_short_ones_whole():
    out, owners, multi = windows([list(range(5)), list(range(1000))], wrap, chunk_tokens=510, stride=384)
    assert out[0] == wrap(list(range(5))) and owners[0] == 0 and multi == 1
    long = [w[1:-1] for w, o in zip(out, owners) if o == 1]
    assert len(long) == 3 and all(len(w) == 510 for w in long)
    assert set().union(*map(set, long)) == set(range(1000))  # an injection at the end is still scored
    assert long[-1][-1] == 999


def window_scorer(batch):
    """A window is 'malicious' exactly when it contains token 99."""
    return [1.0 if 99 in w else 0.1 for w in batch]


def test_max_pool_finds_an_injection_past_the_first_window():
    late = [1] * 30 + [99]  # token 99 sits after the first 10-token window
    scores, multi = max_pool_scores([late, [1, 3]], window_scorer, wrap, chunk_tokens=10, stride=8)
    assert scores.tolist() == [1.0, 0.1] and multi == 1


def test_truncate_scores_only_the_first_window():
    late = [1] * 30 + [99]
    scores, multi = max_pool_scores([late], window_scorer, wrap, chunk_tokens=10, stride=8, truncate=True)
    assert scores.tolist() == [0.1] and multi == 1  # still counted: truncation cut it


def test_max_pool_wraps_windows_in_special_tokens_and_respects_the_window_batch():
    seen = []

    def spy(batch):
        seen.append(batch)
        return [0.5] * len(batch)

    max_pool_scores([[5] * 25], spy, wrap, chunk_tokens=10, stride=10, window_batch=2)
    assert [len(b) for b in seen] == [2, 1]
    assert all(w[0] == 0 and w[-1] == 2 and len(w) == 12 for b in seen for w in b)


def test_max_by_owner_keeps_each_texts_highest_window():
    np.testing.assert_allclose(max_by_owner([0.1, 0.9, 0.2, 0.3], [0, 0, 1, 1], 2), [0.9, 0.3])


def test_malicious_index_finds_the_label_and_refuses_a_three_class_model():
    assert malicious_index({0: "LABEL_0", 1: "LABEL_1"}) == 1
    assert malicious_index({"0": "MALICIOUS", "1": "BENIGN"}) == 0
    with pytest.raises(ValueError, match="2 classes"):
        malicious_index({0: "benign", 1: "injection", 2: "jailbreak"})  # Prompt Guard 1


def test_score_file_is_named_apart_from_the_deberta_runs(tmp_path):
    d = data()
    path = tmp_path / "prompt-guard-2.parquet"
    score_file(PromptGuardDetector(scorer=Stub(), batch_size=4), d, path)
    read = PrecomputedScores(path)
    assert read.name == NAME and not read.name.startswith("deberta")
    np.testing.assert_allclose(read.predict_proba(d.evals["mcptox"].text),
                               [len(t) / 100 for t in d.evals["mcptox"].text])


def test_deberta_score_files_keep_their_name(tmp_path):
    from deberta_run import scores_table
    d = data()
    pq.write_table(scores_table({"val": d.val}, {"val": np.zeros(len(d.val))}, {"row": "row1", "seed": 1},
                                val_only=True), tmp_path / "b2.parquet")
    assert PrecomputedScores(tmp_path / "b2.parquet").name == "deberta_row1"


def test_score_file_records_chunked_texts_per_eval_set(tmp_path):
    d = data()
    score_file(PromptGuardDetector(scorer=Stub(), batch_size=4), d, tmp_path / "pg.parquet")
    meta = json.loads(pq.read_schema(tmp_path / "pg.parquet").metadata[b"deberta"])
    expected = {name: int((f.text.str.len() > 10).sum()) for name, f in {"val": d.val, **d.evals}.items()}
    assert meta["chunked"] == expected and expected["bipia_test"] > expected["mcptox"]


def test_native_summary_reports_the_off_the_shelf_cut_and_exposure():
    frames = {"val": frame("val", 4), "test": frame("t", 4), "mcptox": frame("m", 4)}
    scores = {"val": np.array([0.1, 0.9, 0.6, 0.2]), "test": np.zeros(4), "mcptox": np.array([0.0, 0.4, 0.0, 0.6])}
    out = native_summary(frames, scores, {"val": 0, "test": 1, "mcptox": 0}).set_index("eval_set")
    assert out.loc["val", "recall_at_0.5"] == 0.5 and out.loc["val", "fpr_at_0.5"] == 0.5
    assert out.loc["mcptox", "recall_at_0.5"] == 0.5 and out.loc["mcptox", "fpr_at_0.5"] == 0.0
    assert out.loc["test", "exposure"] == "unknown" and out.loc["mcptox", "chunked"] == 0
    assert not out["truncate"].any()


def test_truncated_run_is_named_and_written_apart(tmp_path):
    stub = Stub()
    stub.truncate = True
    path = tmp_path / "prompt-guard-2-truncated.parquet"
    score_file(PromptGuardDetector(scorer=stub, batch_size=4), data(), path)
    meta = json.loads(pq.read_schema(path).metadata[b"deberta"])
    assert PrecomputedScores(path).name == f"{NAME}_truncated" and meta["truncate"] is True
    native = pd.read_csv(tmp_path / "prompt-guard-2-truncated_native.csv")
    assert native["truncate"].all()


def test_default_device_prefers_cuda_then_apple_gpu_then_cpu():
    from prompt_guard_colab import default_device
    assert default_device(cuda=True, mps=True) == "cuda"
    assert default_device(cuda=False, mps=True) == "mps"
    assert default_device(cuda=False, mps=False) == "cpu"
