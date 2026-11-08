"""B2: fine-tuned DeBERTa-v3 injection detector with chunked max-pool scoring.

A text is split into overlapping token windows; each window gets one logit and the text's
score is the sigmoid of the highest one, so an injected span is never truncated away or
diluted by a long benign document. Training uses the same max (multiple-instance learning),
optionally with a margin ranking term on BIPIA clean/attacked twins.

Design: docs/superpowers/specs/2026-11-07-deberta-detector-design.md
Install the optional dependencies with ``uv sync --group deberta``. This module imports
without torch; torch and transformers are imported where they are used.
"""

from collections import defaultdict


def chunk_spans(n_tokens: int, chunk_tokens: int = 256, stride: int = 192) -> list[tuple[int, int]]:
    """Content-token windows [start, end) covering every token. The last window is shifted to
    end at the last token, so a text never ends in a short tail window. Zero tokens still get
    one (empty) window, which becomes [CLS] [SEP]."""
    if not 1 <= stride <= chunk_tokens:
        raise ValueError("stride must be in [1, chunk_tokens]")
    if n_tokens <= chunk_tokens:
        return [(0, n_tokens)]
    spans = []
    for start in range(0, n_tokens, stride):
        span = (min(start, n_tokens - chunk_tokens), min(start, n_tokens - chunk_tokens) + chunk_tokens)
        if not spans or span != spans[-1]:
            spans.append(span)
        if span[1] == n_tokens:
            break
    return spans


def twin_pairs(groups, labels) -> list[tuple[int, int]]:
    """(attacked_index, clean_index) for every group holding exactly one row of each label.
    BIPIA contexts qualify; PoisonedRAG questions (5 attacks, 10 benign) and singletons don't."""
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    pairs = []
    for indices in members.values():
        if len(indices) != 2:
            continue
        first, second = indices
        if {labels[first], labels[second]} == {0, 1}:
            pairs.append((first, second) if labels[first] == 1 else (second, first))
    return pairs


def select_epoch(history: list[dict]) -> int:
    """Index of the best epoch: highest val TPR at 1% val FPR, ties broken by val AUROC."""
    return max(range(len(history)), key=lambda k: (history[k]["val_tpr"], history[k]["val_auroc"]))
