"""Length separability: how much of a label text length alone could explain, in either direction."""

import pytest
import math
import numpy as np
from metrics import (
    MATCHED_FPR, auroc, bootstrap_auroc_ci, bootstrap_ci, length_binned_auroc, length_separability,
    rates_at_threshold, threshold_at_fpr, tpr_at_fpr,
)

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


def test_matched_fpr_is_five_percent():
    assert MATCHED_FPR == 0.05


def test_tpr_at_fpr_sets_its_threshold_on_these_rows():
    benign = np.linspace(0, 1, 100)          # 5 of 100 lie above 0.95
    adversarial = np.array([0.97, 0.99, 0.5, 0.2])
    scores = np.r_[benign, adversarial]
    labels = np.r_[np.zeros(100), np.ones(4)]
    assert tpr_at_fpr(scores, labels, fpr=0.05) == 0.5


def test_tpr_at_fpr_is_nan_with_one_label():
    assert math.isnan(tpr_at_fpr([0.1, 0.9], [1, 1]))


def test_length_binned_auroc_ignores_a_pure_length_score():
    # Long texts are mostly adversarial; the score only says "long".
    texts = ["x" * n for n in range(1, 201)]
    labels = np.array([int(n > 100 and n % 4 != 0) for n in range(1, 201)])
    scores = np.array([float(len(t) > 100) for t in texts])
    assert auroc(scores, labels) > 0.7
    assert length_binned_auroc(scores, labels, texts) == 0.5


def test_length_binned_auroc_rewards_a_real_signal():
    texts = ["x" * n for n in range(1, 201)]
    labels = np.array([n % 2 for n in range(1, 201)])
    assert length_binned_auroc(labels.astype(float), labels, texts) == 1.0


def test_length_binned_auroc_nan_below_min_rows():
    assert math.isnan(length_binned_auroc([0.1, 0.9] * 10, [0, 1] * 10, ["a", "bb"] * 10))


def test_length_binned_auroc_equal_bins_when_lengths_tie():
    # Every text has the same length: ranking first still makes four equal bins.
    texts = ["same"] * 80
    labels = np.array([i % 2 for i in range(80)])
    assert length_binned_auroc(labels.astype(float), labels, texts) == 1.0


def test_bootstrap_ci_recomputes_the_statistic_per_replicate():
    scores = np.r_[np.linspace(0, 1, 200), np.linspace(0.5, 1.5, 200)]
    labels = np.r_[np.zeros(200), np.ones(200)]
    low, high = bootstrap_ci(lambda s, y: tpr_at_fpr(s, y, 0.05), labels, scores, n_boot=200)
    point = tpr_at_fpr(scores, labels, 0.05)
    assert low < point < high


def test_bootstrap_ci_nan_with_one_label():
    low, high = bootstrap_ci(auroc, [1, 1, 1], [0.1, 0.2, 0.3], n_boot=20)
    assert math.isnan(low) and math.isnan(high)


def test_bootstrap_auroc_ci_unchanged_after_refactor():
    scores = np.r_[np.linspace(0, 1, 50), np.linspace(0.3, 1.3, 50)]
    labels = np.r_[np.zeros(50), np.ones(50)]
    assert bootstrap_auroc_ci(scores, labels, n_boot=200) == pytest.approx(
        bootstrap_ci(auroc, labels, scores, n_boot=200))
