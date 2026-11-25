"""B4: paired DeBERTa-v3 with explicit relevance, plus matched text/pair ablations.

Uses B2's tested optimization, gradient clipping and resumable epoch checkpoints.
Schema agreement is intentionally deferred: no schema-rich training data is available.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from b4_features import FEATURE_NAMES, RelevanceFeatures
from deberta import DebertaDetector, chunk_spans


@dataclass
class EncodedPair:
    context: list
    text: list
    features: list


class B4Detector(DebertaDetector):
    def __init__(self, *, input_mode="relevance", context_tokens=128, relevance_max_features=50000,
                 **kwargs):
        if input_mode not in ("text", "pair", "relevance"):
            raise ValueError("input_mode must be text, pair, or relevance")
        if "pairwise_weight" in kwargs:
            raise ValueError("B4 uses paired INPUTS, not B2's clean/attacked ranking loss")
        super().__init__(pairwise_weight=0.0, **kwargs)
        if context_tokens < 1 or self.epochs < 1 or self.batch_size < 1 or self.max_chunks < 1:
            raise ValueError("Token, epoch and batch budgets must be positive")
        self.input_mode = input_mode
        self.context_tokens = context_tokens
        self.relevance_max_features = relevance_max_features
        self.relevance = None
        self.name = f"b4_{input_mode}"

    @property
    def n_features(self):
        return len(FEATURE_NAMES) if self.input_mode == "relevance" else 0

    def config(self):
        config = super().config()
        config.pop("pairwise_weight")
        return {**config, "input_mode": self.input_mode, "context_tokens": self.context_tokens,
                "relevance_max_features": self.relevance_max_features}

    def _init_model(self):
        from transformers import AutoModel
        from b4_network import B4Network
        return B4Network(AutoModel.from_pretrained(self.model_name), self.n_features)

    @staticmethod
    def _validate_pairs(pairs):
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) != {"context", "text"}:
                raise ValueError("Each B4 input needs exactly context and text; no labels or metadata")
            if not all(isinstance(pair[k], str) for k in pair) or not pair["context"].strip():
                raise ValueError("Each pair needs nonempty trusted context and string candidate text")

    def fit(self, pairs, labels, groups=None, val=None):
        pairs, labels = list(pairs), list(labels)
        self._validate_pairs(pairs)
        if not pairs or len(pairs) != len(labels) or set(labels) != {0, 1}:
            raise ValueError("Training needs matching pairs and binary labels of both classes")
        if val is not None:
            val = (list(val[0]), list(val[1]))
            self._validate_pairs(val[0])
            if len(val[0]) != len(val[1]) or set(val[1]) != {0, 1}:
                raise ValueError("Validation needs matching pairs and both labels")
        if self.n_features:
            self.relevance = RelevanceFeatures(self.relevance_max_features).fit(pairs)
        return super().fit(pairs, labels, groups=groups, val=val)

    def _encode(self, pairs):
        pairs = list(pairs)
        self._validate_pairs(pairs)
        if not pairs:
            return []
        texts = self.tokenizer([p["text"] for p in pairs], add_special_tokens=False)["input_ids"]
        if self.input_mode == "text":
            contexts = [[] for _ in pairs]
        else:
            contexts = self.tokenizer([p["context"] for p in pairs], add_special_tokens=False,
                                      truncation=True, max_length=self.context_tokens)["input_ids"]
        features = self.relevance.transform(pairs).tolist() if self.n_features else [[] for _ in pairs]
        return [EncodedPair(c, t, f) for c, t, f in zip(contexts, texts, features)]

    def _micro_batches(self, units, encoded):
        batch, size = [], 0
        for unit in units:
            n = sum(len(chunk_spans(len(encoded[i].text), self.chunk_tokens, self.chunk_stride)) for i in unit)
            if batch and size + n > self.max_chunks:
                yield batch
                batch, size = [], 0
            batch.append(unit)
            size += n
        if batch:
            yield batch

    def _windows(self, encoded):
        windows, features, owners = [], [], []
        limit = getattr(self.model.encoder.config, "max_position_embeddings", 512)
        for owner, pair in enumerate(encoded):
            for start, end in chunk_spans(len(pair.text), self.chunk_tokens, self.chunk_stride):
                ids = (self.tokenizer.build_inputs_with_special_tokens(pair.text[start:end])
                       if self.input_mode == "text" else
                       self.tokenizer.build_inputs_with_special_tokens(pair.context, pair.text[start:end]))
                if len(ids) > limit:
                    raise ValueError(f"B4 pair has {len(ids)} tokens, exceeds encoder limit {limit}")
                windows.append(ids)
                features.append(pair.features)
                owners.append(owner)
        return windows, features, owners

    def _doc_logits(self, encoded):
        import torch
        windows, features, owners = self._windows(encoded)
        device = self._torch_device()
        chunks = []
        # Every candidate token is scored; repeat the trusted context for each chunk.
        for start in range(0, len(windows), self.max_chunks):
            batch = windows[start:start + self.max_chunks]
            width = max(map(len, batch))
            ids = torch.tensor([w + [self.tokenizer.pad_token_id] * (width - len(w)) for w in batch], device=device)
            mask = torch.tensor([[1] * len(w) + [0] * (width - len(w)) for w in batch], device=device)
            extra = torch.tensor(features[start:start + len(batch)], dtype=torch.float32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self._use_amp()):
                chunks.append(self.model(ids, mask, extra))
        chunk = torch.cat(chunks)
        index = torch.tensor(owners, device=device)
        out = torch.full((len(encoded),), float("-inf"), device=device, dtype=chunk.dtype)
        return out.scatter_reduce(0, index, chunk, reduce="amax")

    def _val_metrics(self, val):
        import numpy as np
        from metrics import auroc, rates_at_threshold, threshold_at_fpr
        labels = np.asarray(val[1])
        scores = self.predict_proba(val[0])
        threshold = threshold_at_fpr(scores[labels == 0], 0.01)
        tpr, _ = rates_at_threshold(scores, labels, threshold)
        clipped = np.clip(scores, 1e-12, 1 - 1e-12)
        loss = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean()
        return {"val_tpr": tpr, "val_auroc": auroc(scores, labels), "val_log_loss": float(loss)}

    def _save_checkpoint(self, epoch, optimizer, scheduler, scaler):
        super()._save_checkpoint(epoch, optimizer, scheduler, scaler)
        print(f"{self.name} seed {self.seed}, epoch {epoch + 1}/{self.epochs}: "
              + json.dumps(self.history[-1]), flush=True)

    def save(self, path):
        import torch
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.encoder.save_pretrained(path / "encoder")
        torch.save(self.model.classifier.state_dict(), path / "classifier.pt")
        self.tokenizer.save_pretrained(path / "tokenizer")
        if self.n_features:
            self.relevance.save(path / "relevance.json")
        (path / "detector.json").write_text(json.dumps({
            "format": "b4-v1", "config": self.config(), "selected_epoch": self.selected_epoch,
            "history": self.history, "feature_names": list(FEATURE_NAMES) if self.n_features else []
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from b4_network import B4Network
        path = Path(path)
        saved = json.loads((path / "detector.json").read_text(encoding="utf-8"))
        if saved.get("format") != "b4-v1":
            raise ValueError("Not a B4 model")
        detector = cls(**saved["config"], device=device)
        detector.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        detector.tokenizer.model_max_length = 10**9
        detector.model = B4Network(AutoModel.from_pretrained(path / "encoder"), detector.n_features)
        detector.model.classifier.load_state_dict(torch.load(path / "classifier.pt", map_location="cpu", weights_only=True))
        detector.model.to(detector._torch_device())
        if detector.n_features:
            detector.relevance = RelevanceFeatures.load(path / "relevance.json")
        detector.history, detector.selected_epoch = saved["history"], saved["selected_epoch"]
        return detector
