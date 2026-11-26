"""Prompt Guard 2 scoring script: batching and a score file the harness can read."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from harness import Data  # noqa: E402
from precomputed import PrecomputedScores  # noqa: E402
from prompt_guard_colab import PromptGuardDetector, score_file  # noqa: E402


class Stub:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return np.array([len(t) / 100 for t in texts])


def frame(prefix, n):
    return pd.DataFrame({"text": [f"{prefix} {'x' * i}" for i in range(n)], "label": [i % 2 for i in range(n)]})


def test_predict_proba_batches():
    stub = Stub()
    scores = PromptGuardDetector(scorer=stub, batch_size=4).predict_proba([f"t{i}" for i in range(10)])
    assert stub.calls == 3 and len(scores) == 10


def test_score_file_round_trips_through_precomputed_scores(tmp_path):
    data = Data(train=frame("train", 4), val=frame("val", 6), evals={"mcptox": frame("mcp", 5), "msb": frame("msb", 3)})
    path = tmp_path / "prompt-guard-2.parquet"
    score_file(PromptGuardDetector(scorer=Stub(), batch_size=4), data, path)
    read = PrecomputedScores(path)
    assert read.name == "deberta_prompt-guard-2"
    np.testing.assert_allclose(read.predict_proba(data.evals["mcptox"].text),
                               [len(t) / 100 for t in data.evals["mcptox"].text])


def test_score_file_records_truncation_after_scoring(tmp_path):
    import json
    import pyarrow.parquet as pq

    def scorer(texts):  # pretends every text was truncated
        scorer.truncated += len(texts)
        return np.zeros(len(texts))

    scorer.truncated = 0
    data = Data(train=frame("train", 2), val=frame("val", 3), evals={"mcptox": frame("mcp", 2)})
    score_file(PromptGuardDetector(scorer=scorer, batch_size=4), data, tmp_path / "pg.parquet")
    meta = json.loads(pq.read_schema(tmp_path / "pg.parquet").metadata[b"deberta"])
    assert meta["truncated"] == 5
