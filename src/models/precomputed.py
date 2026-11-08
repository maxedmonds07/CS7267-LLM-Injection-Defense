"""Scores computed elsewhere (B2 on Colab), served to the harness like a fitted detector.

The harness only calls predict_proba(texts). Rows are matched by sha256 of the text: texts
are unique in dataset.parquet (DATA_CARD.md), so no id column is needed.
"""

import json

import numpy as np
import pyarrow.parquet as pq

from deberta_run import METADATA_KEY, text_hash


class PrecomputedScores:
    def __init__(self, path):
        self.path = str(path)
        table = pq.read_table(path)
        meta = json.loads(table.schema.metadata[METADATA_KEY])
        frame = table.to_pandas()
        conflicts = frame.groupby("text_sha256").score.nunique()
        if (conflicts > 1).any():
            raise ValueError(f"{int((conflicts > 1).sum())} texts have different scores in {path}")
        self._scores = dict(zip(frame.text_sha256, frame.score))
        self.name = f"deberta_{meta['row']}"
        self.seed = meta["seed"]
        self.val_only = bool(meta.get("val_only", False))
        self.params = {
            "scores_file": self.path,
            **{k: v for k, v in meta.items() if k != "config"},
            **{f"config.{k}": v for k, v in meta.get("config", {}).items()},
        }

    def fit(self, texts, labels):
        return self

    def predict_proba(self, texts):
        keys = [text_hash(t) for t in texts]
        missing = sum(k not in self._scores for k in keys)
        if missing:
            raise ValueError(f"{missing} of {len(keys)} rows have no score in {self.path}")
        return np.array([self._scores[k] for k in keys], dtype=float)
