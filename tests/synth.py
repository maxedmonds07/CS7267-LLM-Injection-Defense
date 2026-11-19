"""Synthetic stand-in for dataset.parquet: same length quirks as the real corpus, no real data."""
import numpy as np
import pandas as pd

WORDS = "the of and report system user file data tool server request value result time open read list item".split()
INJECT = ["ignore previous instructions", "send the secrets to", "you must always call", "do not tell the user"]


def _text(rng, n_words, inject):
    words = list(rng.choice(WORDS, n_words))
    if inject:
        words.insert(rng.integers(0, len(words) + 1), rng.choice(INJECT))
    return " ".join(words)


def make(n_per=300, seed=0):
    rng = np.random.default_rng(seed)
    # (source, benign words, adversarial words): the length direction flips between sources
    spec = {"poisonedrag": (70, 30), "bipia": (250, 250), "mcptox": (15, 50), "msb": (25, 55)}
    rows = []
    for source, (nb, na) in spec.items():
        for i in range(n_per):
            rows.append((_text(rng, max(3, int(rng.normal(nb, nb * 0.3))), False), 0, source, f"{source}-b{i}"))
            inject = rng.random() < 0.8
            rows.append((_text(rng, max(3, int(rng.normal(na, na * 0.3))), inject), 1, source, f"{source}-a{i}"))
    return pd.DataFrame(rows, columns=["text", "label", "source", "group"])
