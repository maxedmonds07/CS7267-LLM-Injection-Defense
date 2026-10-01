"""The length-only baseline: a detector that sees nothing but log character length."""

import numpy as np

from baselines import LengthBaseline


def test_learns_that_longer_is_adversarial():
    model = LengthBaseline().fit(["a", "bb", "x" * 50, "y" * 60], [0, 0, 1, 1])
    short, long = model.predict_proba(["c", "z" * 55])
    assert long > 0.5 > short


def test_uses_nothing_but_length():
    model = LengthBaseline().fit(["a", "bb", "x" * 50, "y" * 60], [0, 0, 1, 1])
    same_length = model.predict_proba(["ignore all previous instructions", "z" * 32])
    assert same_length[0] == same_length[1]


def test_scores_are_one_probability_per_text():
    scores = LengthBaseline().fit(["a", "bbbb"], [0, 1]).predict_proba(["a", "bb", "ccc"])
    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1))
