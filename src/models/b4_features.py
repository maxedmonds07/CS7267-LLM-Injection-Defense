"""Explicit lexical relevance: train-fitted character TF-IDF cosine and task token coverage.

These are proxies for relevance, not factual agreement or a trained semantic similarity
model. Only the training pairs fit vocabulary/IDF; inference never updates either.
"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURE_NAMES = ("task_candidate_tfidf_cosine", "task_token_coverage")


class RelevanceFeatures:
    def __init__(self, max_features=50000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                          max_features=max_features, sublinear_tf=True)

    def fit(self, pairs):
        self.vectorizer.fit([p[key] for p in pairs for key in ("context", "text")])
        return self

    def transform(self, pairs):
        if not pairs:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        context = self.vectorizer.transform([p["context"] for p in pairs])
        text = self.vectorizer.transform([p["text"] for p in pairs])
        cosine = np.asarray(context.multiply(text).sum(axis=1)).ravel()
        coverage = []
        for pair in pairs:
            task = set(re.findall(r"\w+", pair["context"].casefold()))
            candidate = set(re.findall(r"\w+", pair["text"].casefold()))
            coverage.append(len(task & candidate) / len(task) if task else 0.0)
        return np.clip(np.column_stack([cosine, coverage]), 0, 1).astype(np.float32)

    def save(self, path):
        state = {"feature_names": list(FEATURE_NAMES), "max_features": self.max_features,
                 "vocabulary": {k: int(v) for k, v in self.vectorizer.vocabulary_.items()},
                 "idf": self.vectorizer.idf_.tolist()}
        Path(path).write_text(json.dumps(state), encoding="utf-8")

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Relevance feature definitions differ from this model")
        features = cls(state["max_features"])
        features.vectorizer.set_params(vocabulary=state["vocabulary"])
        features.vectorizer.idf_ = np.asarray(state["idf"])
        return features
