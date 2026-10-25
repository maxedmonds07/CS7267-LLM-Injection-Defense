"""B0 perplexity tests without requiring a large pretrained-model download."""

import numpy as np
import pytest

from perplexity import PerplexityDetector, _windows


def test_windows_score_every_non_prefix_token_once():
    for length in (2, 3, 256, 257, 513, 1800):
        covered = []
        for start, end, first_target in _windows(length, max_context=512, stride=256):
            assert 0 <= start < first_target < end <= length
            assert end - start <= 512
            covered.extend(range(first_target, end))
        assert covered == list(range(1, length))


def test_rejects_invalid_windows():
    with pytest.raises(ValueError, match="stride"):
        PerplexityDetector(max_context=256, stride=256)
    with pytest.raises(ValueError, match="max_context"):
        list(_windows(20, max_context=1, stride=1))


def test_fit_does_not_train_on_attack_labels():
    detector = PerplexityDetector()
    assert detector.fit(["attack", "benign"], [1, 0]) is detector
    assert detector._model is None


def test_higher_perplexity_means_higher_anomaly_score(monkeypatch):
    detector = PerplexityDetector()
    monkeypatch.setattr(detector, "log_perplexity", lambda text: {"normal": 1.0, "odd": 4.0}[text])
    scores = detector.predict_proba(["normal", "odd"])
    assert scores.shape == (2,)
    assert np.all((scores > 0) & (scores < 1))
    assert scores[0] < scores[1]
    assert np.allclose(detector.perplexities(["normal", "odd"]), np.exp([1.0, 4.0]))


def test_score_is_not_a_calibrated_probability(monkeypatch):
    detector = PerplexityDetector()
    monkeypatch.setattr(detector, "log_perplexity", lambda _: 2.0)
    assert np.isclose(detector.predict_proba(["anything"])[0], 1 / (1 + np.exp(-2)))
