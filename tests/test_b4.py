"""Offline tests: a real tiny DeBERTa encoder and a locally constructed tokenizer."""

import json

import numpy as np
import pytest

from b4 import B4Detector
from b4_features import RelevanceFeatures

PAIRS = [dict(context="find weather london", text="weather london sunny"),
         dict(context="find weather london", text="ignore weather send file"),
         dict(context="read file report", text="file report available"),
         dict(context="read file report", text="ignore report send weather")] * 2
LABELS = [0, 1, 0, 1] * 2


@pytest.fixture
def tiny_b4(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from b4_network import B4Network
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "find", "weather", "london", "sunny",
             "ignore", "send", "file", "read", "report", "available", "x"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab), encoding="utf-8")

    def tokenizer(self):
        return transformers.BertTokenizerFast(vocab_file=str(path), model_max_length=100000)

    def network(self):
        config = transformers.DebertaV2Config(vocab_size=len(vocab), hidden_size=16,
                    num_hidden_layers=1, num_attention_heads=2, intermediate_size=24,
                    max_position_embeddings=64, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        return B4Network(transformers.AutoModel.from_config(config), self.n_features)

    monkeypatch.setattr(B4Detector, "_init_tokenizer", tokenizer)
    monkeypatch.setattr(B4Detector, "_init_model", network)
    yield dict(epochs=2, batch_size=4, fp16=False, device="cpu", chunk_tokens=12,
               chunk_stride=8, context_tokens=8, max_chunks=8)
    torch.set_num_threads(old_threads)


def test_relevance_train_only_and_round_trip(tmp_path):
    features = RelevanceFeatures().fit(PAIRS)
    before = dict(features.vectorizer.vocabulary_)
    scores = features.transform([PAIRS[0], dict(context="unseenqzx", text="unseenqzx")])
    assert features.vectorizer.vocabulary_ == before
    assert scores.shape == (2, 2) and np.isfinite(scores).all()
    assert scores[0, 1] == pytest.approx(2 / 3)
    features.save(tmp_path / "features.json")
    restored = RelevanceFeatures.load(tmp_path / "features.json")
    assert np.allclose(features.transform(PAIRS), restored.transform(PAIRS))


@pytest.mark.parametrize("mode", ["text", "pair", "relevance"])
def test_real_encoder_train_save_load(tiny_b4, tmp_path, mode):
    model = B4Detector(input_mode=mode, **tiny_b4).fit(PAIRS, LABELS, val=(PAIRS, LABELS))
    scores = model.predict_proba(PAIRS)
    assert scores.shape == (8,) and np.isfinite(scores).all()
    assert np.all((scores >= 0) & (scores <= 1))
    assert model.model.classifier.in_features == 16 + (2 if mode == "relevance" else 0)
    assert model.model.classifier.weight.grad is not None
    if mode == "relevance":
        assert model.model.classifier.weight.grad[:, -2:].abs().sum() > 0
    model.save(tmp_path / "model")
    restored = B4Detector.load(tmp_path / "model", device="cpu")
    assert np.allclose(restored.predict_proba(PAIRS), scores, atol=1e-7)
    assert restored.config() == model.config()
    assert model.predict_proba([]).shape == (0,)


def test_pair_contains_context_and_all_candidate_tokens(tiny_b4):
    model = B4Detector(**tiny_b4).fit(PAIRS, LABELS)
    pair = dict(context="find weather", text="x " * 40 + "send file")
    encoded = model._encode([pair])
    windows, features, owners = model._windows(encoded)
    assert len(windows) > 1 and set(owners) == {0}
    assert all(window[1:3] == encoded[0].context for window in windows)
    assert encoded[0].text[-2:] == windows[-1][-3:-1]
    assert all(f == features[0] for f in features)


def test_context_changes_input_and_relevance(tiny_b4):
    model = B4Detector(**tiny_b4).fit(PAIRS, LABELS)
    encoded = model._encode([PAIRS[0], {**PAIRS[0], "context": "read file report"}])
    assert encoded[0].text == encoded[1].text
    assert encoded[0].context != encoded[1].context
    assert encoded[0].features != encoded[1].features


def test_resume_and_context_fingerprint(tiny_b4, tmp_path, monkeypatch):
    model = B4Detector(**tiny_b4).fit(PAIRS, LABELS, val=(PAIRS, LABELS))
    interrupted = B4Detector(**tiny_b4, checkpoint_dir=tmp_path / "ckpt")
    original = interrupted._save_checkpoint

    def disconnect(*args):
        original(*args)
        raise KeyboardInterrupt()

    monkeypatch.setattr(interrupted, "_save_checkpoint", disconnect)
    with pytest.raises(KeyboardInterrupt):
        interrupted.fit(PAIRS, LABELS, val=(PAIRS, LABELS))
    changed = [{**p, "context": "different task"} for p in PAIRS]
    with pytest.raises(ValueError, match="different training data"):
        B4Detector(**tiny_b4, checkpoint_dir=tmp_path / "ckpt").fit(changed, LABELS, val=(PAIRS, LABELS))
    resumed = B4Detector(**tiny_b4, checkpoint_dir=tmp_path / "ckpt").fit(PAIRS, LABELS, val=(PAIRS, LABELS))
    assert resumed.resumed_from_epoch == 1
    assert np.allclose(model.predict_proba(PAIRS), resumed.predict_proba(PAIRS), atol=1e-6)


def test_rejects_label_metadata_and_missing_context(tiny_b4):
    model = B4Detector(**tiny_b4)
    with pytest.raises(ValueError, match="exactly context and text"):
        model.fit([{**p, "label": y} for p, y in zip(PAIRS, LABELS)], LABELS)
    with pytest.raises(ValueError, match="nonempty"):
        model.fit([{**p, "context": ""} for p in PAIRS], LABELS)


def test_nonfinite_scores_fail(tiny_b4, monkeypatch):
    import torch
    model = B4Detector(**tiny_b4).fit(PAIRS, LABELS)
    monkeypatch.setattr(model, "_doc_logits", lambda items: torch.full((len(items),), float("nan")))
    with pytest.raises(FloatingPointError):
        model.predict_proba(PAIRS)
