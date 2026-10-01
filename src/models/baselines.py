"""Shortcut baselines that every detector must beat.

LengthBaseline sees only log character length. Report it next to every detector, on the same
eval sets: a detector that barely beats it may be classifying length, not injection.

  PYTHONPATH=src/data:src/eval uv run python src/models/baselines.py
"""

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class LengthBaseline:
    """Logistic regression on log(1 + character length)."""

    name = "length_only"

    def __init__(self):
        self.model = LogisticRegression()

    @staticmethod
    def _features(texts):
        # The log keeps a few 7,000-character BIPIA documents from swamping 88-character
        # MCPTox descriptions.
        return np.log1p([len(text) for text in texts]).reshape(-1, 1)

    def fit(self, texts, labels):
        self.model.fit(self._features(texts), labels)
        return self

    def predict_proba(self, texts):
        """Probability that each text is adversarial, one value per text."""
        return self.model.predict_proba(self._features(texts))[:, 1]


def main():
    from dataset import DEFAULT_PATH, load
    from metrics import length_separability

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=DEFAULT_PATH, help="dataset.parquet to read")
    args = parser.parse_args()

    def rows(purpose, **filters):
        return load(purpose, path=args.data, **filters).select(["text", "label", "source"]).to_pandas()

    train = rows("fit", roles=["train_pool"], splits=["train"])
    model = LengthBaseline().fit(train.text, train.label)
    test = rows("eval", roles=["train_pool"], splits=["test"])
    eval_sets = {
        "in-distribution test": test,
        **{f"  {source}": part for source, part in test.groupby("source")},
        "LOSO (MCPTox)": rows("eval", sources=["mcptox"]),
        "transfer (MSB)": rows("eval", sources=["msb"]),
    }

    print(f"{'eval set':24} {'fitted AUROC':>12} {'separability':>12}")
    for name, part in eval_sets.items():
        auroc = roc_auc_score(part.label, model.predict_proba(part.text))
        print(f"{name:24} {auroc:12.3f} {length_separability(part.text, part.label):12.3f}")


if __name__ == "__main__":
    main()
