"""B2 DeBERTa detector. Helper tests need no torch; model tests use a tiny random model."""

from deberta import chunk_spans, select_epoch, twin_pairs


def test_zero_tokens_gives_one_empty_window():
    assert chunk_spans(0) == [(0, 0)]


def test_short_text_is_one_window():
    assert chunk_spans(256) == [(0, 256)]
    assert chunk_spans(10) == [(0, 10)]


def test_windows_cover_every_token_and_end_at_the_last():
    for n in (257, 448, 1000, 1700):
        spans = chunk_spans(n)
        covered = set().union(*(range(s, e) for s, e in spans))
        assert covered == set(range(n))
        assert spans[-1][1] == n
        assert all(e - s == 256 for s, e in spans)
        assert all(b[0] - a[0] <= 192 for a, b in zip(spans, spans[1:]))


def test_twin_pairs_only_one_clean_one_attacked():
    groups = ["b1", "b1", "b2", "b2", "p", "p", "p", "solo"]
    labels = [1, 0, 0, 1, 1, 0, 0, 1]
    assert sorted(twin_pairs(groups, labels)) == [(0, 1), (3, 2)]


def test_select_epoch_prefers_tpr_then_auroc():
    history = [
        {"val_tpr": 0.4, "val_auroc": 0.95},
        {"val_tpr": 0.5, "val_auroc": 0.90},
        {"val_tpr": 0.5, "val_auroc": 0.93},
    ]
    assert select_epoch(history) == 2
