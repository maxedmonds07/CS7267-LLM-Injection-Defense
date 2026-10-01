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
