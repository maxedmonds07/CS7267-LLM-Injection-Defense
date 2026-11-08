"""Evaluation metrics shared by every detector."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def length_separability(texts, labels) -> float:
    """max(AUROC, 1 - AUROC) of raw character length: how well length alone separates the
    labels in either direction. 0.5 means no signal; 1.0 means length decides the label.
    Needs no training, so it bounds what any length shortcut could earn on these rows."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        raise ValueError("length separability needs both labels in the rows scored")
    auroc = roc_auc_score(labels, [len(text) for text in texts])
    return float(max(auroc, 1 - auroc))

def auroc(scores, labels) -> float:
    """AUROC, or NaN when the rows hold one label (MSB tool_output has no benign rows)."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))

def threshold_at_fpr(benign_scores, target_fpr) -> float:
    """Score above which a text is called adversarial, chosen so that at most target_fpr of these
    benign scores lie above it. Set it on in-distribution val, then apply it unchanged to every
    eval set: a holdout with 21 benign rows cannot support its own 1% threshold."""
    benign_scores = np.asarray(benign_scores, dtype=float)
    if benign_scores.size == 0:
        raise ValueError("threshold_at_fpr needs at least one benign score")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    allowed = int(np.floor(target_fpr * benign_scores.size + 1e-9))
    return float(np.sort(benign_scores)[::-1][allowed])

def rates_at_threshold(scores, labels, threshold) -> tuple[float, float]:
    """(TPR, FPR) when score > threshold means adversarial. A rate with no rows to measure is NaN."""
    flagged = np.asarray(scores, dtype=float) > threshold
    labels = np.asarray(labels)
    
    def rate(mask):
        return float(flagged[mask].mean()) if mask.any() else float("nan")
    
    return rate(labels == 1), rate(labels == 0)

def bootstrap_ci(statistic, labels, *columns, n_boot=1000, seed=7267, level=0.95) -> tuple[float, float]:
    """Percentile interval for statistic(*columns, labels). Benign and adversarial rows are
    resampled separately, so every replicate keeps both labels and the label mix of these rows.
    The statistic runs in full on each replicate, so anything it derives from the rows (a
    threshold, length bins) is re-derived per replicate. NaN replicates are ignored; NaN pair
    when the rows hold one label or every replicate is NaN."""
    labels = np.asarray(labels)
    columns = [np.asarray(column) for column in columns]
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        index = np.r_[rng.choice(positive, len(positive)), rng.choice(negative, len(negative))]
        stats.append(statistic(*(column[index] for column in columns), labels[index]))
    stats = np.asarray(stats, dtype=float)
    stats = stats[~np.isnan(stats)]
    if not stats.size:
        return float("nan"), float("nan")
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(stats, [tail, 100 - tail])
    return float(low), float(high)

def bootstrap_auroc_ci(scores, labels, n_boot=1000, seed=7267, level=0.95) -> tuple[float, float]:
    """Percentile interval for AUROC (see bootstrap_ci). NaN when the rows hold one label."""
    return bootstrap_ci(auroc, labels, np.asarray(scores, dtype=float),
                        n_boot=n_boot, seed=seed, level=level)

MATCHED_FPR = 0.05
"""FPR at which every detector's TPR is compared for the B2 verdict. Measurement only: the
threshold comes from the eval slice's own benign rows, so it is never a deployable threshold."""

def tpr_at_fpr(scores, labels, fpr=MATCHED_FPR) -> float:
    """TPR at the threshold that lets `fpr` of these rows' own benign scores through. Compares
    detectors at the same FPR on one slice; not a deployed threshold (that comes from val).
    NaN when the rows hold one label."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if not (labels == 1).any() or not (labels == 0).any():
        return float("nan")
    tpr, _ = rates_at_threshold(scores, labels, threshold_at_fpr(scores[labels == 0], fpr))
    return tpr

def length_binned_auroc(scores, labels, texts, n_bins=4, min_rows=40) -> float:
    """AUROC within character-length quantile bins, averaged with weights by bin size. A score
    that only tracks length earns 0.5 here even where length separates the labels. Bins come
    from length ranks, so ties still give equal-sized bins. Bins holding one label are skipped.
    NaN with fewer than min_rows rows, one label overall, or fewer than two usable bins."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if len(labels) < min_rows or len(np.unique(labels)) < 2:
        return float("nan")
    lengths = pd.Series([len(text) for text in texts])
    bins = pd.qcut(lengths.rank(method="first"), n_bins, labels=False).to_numpy()

    values, weights = [], []
    for b in np.unique(bins):
        in_bin = bins == b
        if len(np.unique(labels[in_bin])) == 2:
            values.append(auroc(scores[in_bin], labels[in_bin]))
            weights.append(in_bin.sum())
    if len(values) < 2:
        return float("nan")
    return float(np.average(values, weights=weights))
