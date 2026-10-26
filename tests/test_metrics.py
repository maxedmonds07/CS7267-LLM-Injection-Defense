"""Length separability: how much of a label text length alone could explain, in either direction."""

import pytest
import math
import numpy as np
from metrics import length_separability, auroc, bootstrap_auroc_ci, rates_at_threshold, threshold_at_fpr

def test_longer_adversarial_is_fully_separable():
    assert length_separability(["a", "bb", "cccc", "ddddd"], [0, 0, 1, 1]) == 1.0


def test_shorter_adversarial_counts_the_same():
    # PoisonedRAG's direction: poisoned passages are shorter than retrieved ones.
    assert length_separability(["a", "bb", "cccc", "ddddd"], [1, 1, 0, 0]) == 1.0


def test_equal_lengths_carry_no_signal():
    assert length_separability(["ab", "cd", "ef", "gh"], [0, 1, 0, 1]) == 0.5


def test_single_class_is_undefined():
    # MSB tool_output has adversarial rows only.
    with pytest.raises(ValueError, match="both labels"):
        length_separability(["a", "bb"], [1, 1])

def test_auroc_perfect_and_single_label():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert math.isnan(auroc([0.8, 0.9], [1, 1]))


def test_threshold_allows_exactly_the_target_share():
    benign = [i / 10 for i in range(1, 11)]  # 0.1 .. 1.0
    threshold = threshold_at_fpr(benign, 0.1)
    assert threshold == 0.9
    assert rates_at_threshold(benign, [0] * 10, threshold)[1] == 0.1


def test_threshold_rounds_toward_fewer_false_positives():
    benign = [i / 10 for i in range(1, 11)]
    # 1% of 10 rows is 0.1 rows, so no benign row may be flagged.
    assert rates_at_threshold(benign, [0] * 10, threshold_at_fpr(benign, 0.01))[1] == 0.0


def test_threshold_with_ties_never_exceeds_target():
    benign = [0.5] * 10 + [0.9]
    threshold = threshold_at_fpr(benign, 0.2)
    assert rates_at_threshold(benign, [0] * 11, threshold)[1] <= 0.2


def test_threshold_rejects_bad_input():
    with pytest.raises(ValueError, match="benign"):
        threshold_at_fpr([], 0.01)
    with pytest.raises(ValueError, match="target_fpr"):
        threshold_at_fpr([0.1], 1.0)


def test_rates_count_scores_strictly_above_threshold():
    tpr, fpr = rates_at_threshold([0.2, 0.5, 0.5, 0.9], [0, 0, 1, 1], 0.5)
    assert (tpr, fpr) == (0.5, 0.0)


def test_rates_with_no_benign_rows():
    # M5B tool_output: TPR is defined, FPR is not.
    tpr, fpr = rates_at_threshold([0.9, 0.1], [1, 1], 0.5)
    assert tpr == 0.5
    assert math.isnan(fpr)


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(0)
    scores = np.r_[rng.normal(0, 1, 200), rng.normal(1, 1, 200)]
    labels = np.r_[np.zeros(200), np.ones(200)]
    lo, hi = bootstrap_auroc_ci(scores, labels, n_boot=200)
    assert lo < auroc(scores, labels) < hi
    assert bootstrap_auroc_ci(scores, labels, n_boot=200) == (lo, hi)  # seeded


def test_bootstrap_with_one_label():
    lo, hi = bootstrap_auroc_ci([0.9, 0.1], [1, 1], n_boot=10)
    assert math.isnan(lo) and math.isnan(hi)