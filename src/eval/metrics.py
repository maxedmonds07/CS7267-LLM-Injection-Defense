"""Evaluation metrics shared by every detector."""

import numpy as np
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

def bootstrap_auroc_ci(scores, labels, n_boot=1000, seed=7267, level=0.95) -> tuple[float, float]:
    """Percentile interval for AUROC. Benign and adversarial rows are resampled separately, so
    every replicate keeps both labels. NaN when the rows hold one label."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    positive, negative = scores[labels == 1], scores[labels == 0]
    
    if not len(positive) or not len(negative):
        return float("nan"), float("nan")
    
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    
    stats = [
    roc_auc_score(y, np.r_[rng.choice(positive, len(positive)), rng.choice(negative, len(negative))])
    for _ in range(n_boot)
    ]
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(stats, [tail, 100 - tail])
    return float(low), float(high)
