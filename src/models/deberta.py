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


PAIRWISE_MARGIN = 1.0


def pairwise_margin_loss(attacked_logits, clean_logits, margin=PAIRWISE_MARGIN):
    """Mean of relu(margin - (attacked - clean)) over twin pairs: each attacked text should
    out-score its own clean twin by at least `margin` logits. 0.0 when there are no pairs."""
    import torch

    if attacked_logits.numel() == 0:
        return torch.zeros((), device=attacked_logits.device, dtype=attacked_logits.dtype)
    return torch.relu(margin - (attacked_logits - clean_logits)).mean()


def select_epoch(history: list[dict]) -> int:
    """Index of the best epoch: highest val TPR at 1% val FPR, ties broken by val AUROC."""
    return max(range(len(history)), key=lambda k: (history[k]["val_tpr"], history[k]["val_auroc"]))


CONFIG_FIELDS = (
    "model_name", "chunk_tokens", "chunk_stride", "pairwise_weight", "lr", "epochs",
    "batch_size", "seed", "fp16", "max_chunks",
)


class DebertaDetector:
    """Fine-tuned DeBERTa-v3, one logit per chunk, text score = sigmoid(max chunk logit)."""

    name = "deberta"

    def __init__(self, model_name="microsoft/deberta-v3-base", chunk_tokens=256, chunk_stride=192,
                 pairwise_weight=0.0, lr=2e-5, epochs=3, batch_size=16, seed=7267,
                 fp16=True, device=None, checkpoint_dir=None, max_chunks=32):
        if not 1 <= chunk_stride <= chunk_tokens:
            raise ValueError("chunk_stride must be in [1, chunk_tokens]")
        self.model_name = model_name
        self.chunk_tokens = chunk_tokens
        self.chunk_stride = chunk_stride
        self.pairwise_weight = pairwise_weight
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.fp16 = fp16
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        # Windows per forward pass. Training splits a batch into micro-batches under this budget
        # and accumulates gradients; no window is ever dropped.
        self.max_chunks = max_chunks
        self.tokenizer = None
        self.model = None
        self.history = []
        self.selected_epoch = None
        self.resumed_from_epoch = 0
        self.n_twin_pairs = 0

    # --- building blocks ---------------------------------------------------------------

    def _init_tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.model_max_length = 10**9  # we chunk ourselves; silence the length warning
        return tokenizer

    def _init_model(self):
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1)

    def _torch_device(self):
        import torch

        if self.device:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _use_amp(self):
        return bool(self.fp16) and self._torch_device().type == "cuda"

    def _encode(self, texts) -> list[list[int]]:
        return self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]

    def _windows(self, token_lists):
        """Every chunk of every text as [CLS] + ids + [SEP], with the index of its text."""
        cls, sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        windows, owners = [], []
        for owner, ids in enumerate(token_lists):
            for start, end in chunk_spans(len(ids), self.chunk_tokens, self.chunk_stride):
                windows.append([cls, *ids[start:end], sep])
                owners.append(owner)
        return windows, owners

    def _chunk_logits(self, windows):
        """One logit per window, as a 1-D tensor on the model's device."""
        import torch

        device = self._torch_device()
        width = max(len(w) for w in windows)
        pad = self.tokenizer.pad_token_id
        ids = torch.tensor([w + [pad] * (width - len(w)) for w in windows], device=device)
        mask = torch.tensor([[1] * len(w) + [0] * (width - len(w)) for w in windows], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self._use_amp()):
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        return logits.squeeze(-1).float()

    def _doc_logits(self, token_lists):
        """Max over each text's chunk logits, one value per text."""
        import torch

        windows, owners = self._windows(token_lists)
        chunk = self._chunk_logits(windows)
        index = torch.tensor(owners, device=chunk.device)
        out = torch.full((len(token_lists),), float("-inf"), device=chunk.device, dtype=chunk.dtype)
        return out.scatter_reduce(0, index, chunk, reduce="amax")

    def _loss(self, units, doc_logits, labels, n_rows=None, n_pairs=None):
        """This micro-batch's share of the batch loss: BCE on text-level max logits, plus
        pairwise_weight x the margin loss over twin pairs. `units` are lists of row indices
        (a pair is [attacked, clean]); doc_logits and labels follow the flattened units.
        n_rows and n_pairs are the whole batch's counts, so summing the shares over
        micro-batches gives the batch's mean BCE plus the weighted mean margin over its pairs.
        Left out, they default to this micro-batch's own counts (its plain mean loss)."""
        import torch

        rows = len(labels)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(doc_logits, labels)
        loss = bce * (rows / (n_rows or rows))
        if self.pairwise_weight > 0:
            starts, offset = [], 0
            for unit in units:
                if len(unit) == 2:
                    starts.append(offset)
                offset += len(unit)
            if starts:
                index = torch.tensor(starts, device=doc_logits.device)
                margin = pairwise_margin_loss(doc_logits[index], doc_logits[index + 1])
                loss = loss + self.pairwise_weight * margin * (len(starts) / (n_pairs or len(starts)))
        return loss

    def _micro_batches(self, units, token_lists):
        """Split units into groups whose window count stays under max_chunks (a unit is never
        split; a unit over the budget forms its own group)."""
        groups, current, used = [], [], 0
        for unit in units:
            size = sum(len(chunk_spans(len(token_lists[i]), self.chunk_tokens, self.chunk_stride))
                       for i in unit)
            if current and used + size > self.max_chunks:
                groups.append(current)
                current, used = [], 0
            current.append(unit)
            used += size
        if current:
            groups.append(current)
        return groups

    def _train_step(self, units, token_lists, labels, optimizer, scheduler, scaler, epoch):
        """One optimizer step over a batch of units, accumulated across micro-batches."""
        import torch

        device = self._torch_device()
        n_rows = sum(len(u) for u in units)
        n_pairs = sum(len(u) == 2 for u in units)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for group in self._micro_batches(units, token_lists):
            rows = [i for unit in group for i in unit]
            doc_logits = self._doc_logits([token_lists[i] for i in rows])
            target = torch.tensor([float(labels[i]) for i in rows], device=device)
            share = self._loss(group, doc_logits, target, n_rows=n_rows, n_pairs=n_pairs)
            if not torch.isfinite(share):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}; rerun with fp16=False")
            (scaler.scale(share) if scaler else share).backward()
            total += share.item()
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        return total

    def _units(self, n_rows, groups, labels):
        """Training units. With pairwise_weight 0, one row each. Otherwise every clean/attacked
        twin pair (BIPIA) is one unit [attacked, clean], so the twins always share a batch and
        a micro-batch, and every other row (PoisonedRAG, unpaired rows) is a unit of one."""
        if self.pairwise_weight <= 0:
            self.n_twin_pairs = 0
            return [[i] for i in range(n_rows)]
        pairs = twin_pairs(list(groups), [int(label) for label in labels])
        if not pairs:
            raise ValueError("pairwise_weight > 0 but the training rows hold no clean/attacked "
                             "twin pairs (groups with exactly one row of each label)")
        self.n_twin_pairs = len(pairs)
        unit_at = {min(pair): list(pair) for pair in pairs}
        paired = {i for pair in pairs for i in pair}
        return [unit_at[i] if i in unit_at else [i]
                for i in range(n_rows) if i in unit_at or i not in paired]

    def _val_metrics(self, val):
        from metrics import auroc, rates_at_threshold, threshold_at_fpr
        import numpy as np

        texts, labels = val
        labels = np.asarray(labels)
        scores = self.predict_proba(texts)
        threshold = threshold_at_fpr(scores[labels == 0], 0.01)
        tpr, _ = rates_at_threshold(scores, labels, threshold)
        return {"val_tpr": tpr, "val_auroc": auroc(scores, labels)}

    # --- public interface --------------------------------------------------------------

    def fit(self, texts, labels, groups=None, val=None):
        """Train on train_pool train. `groups` enables the pairwise term (needed when
        pairwise_weight > 0). `val` = (texts, labels) chooses the epoch (best val TPR at 1% val
        FPR, ties by val AUROC); without it the last epoch is kept."""
        import copy

        import numpy as np
        import torch
        from transformers import get_linear_schedule_with_warmup

        labels = list(labels)
        texts = list(texts)
        if self.pairwise_weight > 0 and groups is None:
            raise ValueError("pairwise_weight > 0 needs groups to find the clean/attacked twins")
        units = self._units(len(texts), groups, labels)
        if self.tokenizer is None:
            self.tokenizer = self._init_tokenizer()
        torch.manual_seed(self.seed)
        self.model = self._init_model().to(self._torch_device())
        token_lists = self._encode(texts)
        self._fingerprint = self._data_fingerprint(texts, labels, groups, val)

        # Each epoch's shuffle depends only on seed + epoch, so the exact number of optimizer
        # steps is known up front (pair units make batches uneven) and a resumed run replays
        # exactly the batches an uninterrupted run would have seen.
        orders = [np.random.default_rng(self.seed + epoch).permutation(len(units))
                  for epoch in range(self.epochs)]
        total_steps = sum(len(list(self._batches([units[k] for k in order]))) for order in orders)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
        scaler = torch.amp.GradScaler("cuda") if self._use_amp() else None

        self.history, best_state = [], None
        start = 0
        if self.checkpoint_dir:
            start, best_state = self._load_checkpoint(optimizer, scheduler, scaler)
        self.resumed_from_epoch = start
        for epoch in range(start, self.epochs):
            torch.manual_seed(self.seed + epoch)  # dropout, likewise per epoch
            self.model.train()
            losses = []
            for batch in self._batches([units[k] for k in orders[epoch]]):
                losses.append(self._train_step(batch, token_lists, labels, optimizer, scheduler,
                                               scaler, epoch))
            entry = {"epoch": epoch, "loss": float(np.mean(losses))}
            if val is not None:
                entry.update(self._val_metrics(val))
            self.history.append(entry)
            if val is not None and select_epoch(self.history) == epoch:
                best_state = copy.deepcopy({k: v.cpu() for k, v in self.model.state_dict().items()})
            if self.checkpoint_dir:
                self._save_checkpoint(epoch, optimizer, scheduler, scaler)

        if val is not None:
            self.selected_epoch = select_epoch(self.history)
            self.model.load_state_dict(best_state)
        else:
            self.selected_epoch = self.epochs - 1
        return self

    # --- checkpoints ------------------------------------------------------------------

    @staticmethod
    def _data_fingerprint(texts, labels, groups, val) -> str:
        """sha256 of exactly what a run trains and selects on, so a checkpoint from another run
        (a --limit smoke run, another row's train sources, a run without val) is never resumed."""
        import hashlib
        import json

        payload = {
            "texts": list(texts),
            "labels": [int(label) for label in labels],
            "groups": None if groups is None else [str(g) for g in groups],
            "val": None if val is None else [list(val[0]), [int(label) for label in val[1]]],
        }
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    def _epoch_dirs(self):
        """{epoch: directory} for every completed checkpoint (temp directories excluded)."""
        from pathlib import Path

        root = Path(self.checkpoint_dir)
        if not root.exists():
            return {}
        return {int(d.name.split("_")[1]): d for d in root.iterdir()
                if d.is_dir() and d.name.startswith("epoch_") and not d.name.endswith(".tmp")}

    def _save_checkpoint(self, epoch, optimizer, scheduler, scaler):
        """Write epoch_{epoch}/ atomically (temp dir, then rename), so a disconnect mid-save never
        leaves a checkpoint that looks complete. Keeps only the latest and the best epoch."""
        import shutil
        from pathlib import Path

        import torch

        root = Path(self.checkpoint_dir)
        root.mkdir(parents=True, exist_ok=True)
        tmp, final = root / f"epoch_{epoch}.tmp", root / f"epoch_{epoch}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        torch.save(self.model.state_dict(), tmp / "model.pt")
        torch.save({
            "config": self.config(),
            "data": self._fingerprint,
            "history": self.history,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
        }, tmp / "state.pt")
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)

        keep = {epoch}
        if self.history and "val_tpr" in self.history[0]:
            keep.add(self.history[select_epoch(self.history)]["epoch"])
        for old, directory in self._epoch_dirs().items():
            if old not in keep:
                shutil.rmtree(directory)
            elif old != epoch:
                # Only the latest epoch needs optimizer state to resume; the best keeps weights.
                (directory / "state.pt").unlink(missing_ok=True)

    def _load_checkpoint(self, optimizer, scheduler, scaler):
        """Restore the latest completed epoch. Returns (next epoch, best weights or None)."""
        import torch

        dirs = self._epoch_dirs()
        if not dirs:
            return 0, None
        latest = max(dirs)
        state = torch.load(dirs[latest] / "state.pt", map_location="cpu", weights_only=True)
        if state["config"] != self.config():
            raise ValueError(
                f"{self.checkpoint_dir} holds a run with a different configuration: "
                f"{state['config']} (this run: {self.config()}); use a fresh checkpoint_dir")
        if state.get("data") != self._fingerprint:
            raise ValueError(
                f"{self.checkpoint_dir} holds a run with different training data (rows, labels, "
                f"groups or val set; for example a --limit smoke run); use a fresh checkpoint_dir")
        device = self._torch_device()
        self.model.load_state_dict(torch.load(dirs[latest] / "model.pt", map_location=device, weights_only=True))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        self.history = state["history"]
        best_state = None
        if self.history and "val_tpr" in self.history[0]:
            best = self.history[select_epoch(self.history)]["epoch"]
            best_state = torch.load(dirs[best] / "model.pt", map_location="cpu", weights_only=True)
        return latest + 1, best_state

    def _batches(self, units):
        """Consecutive units, up to batch_size rows per batch; a unit is never split."""
        batch, rows = [], 0
        for unit in units:
            if batch and rows + len(unit) > self.batch_size:
                yield batch
                batch, rows = [], 0
            batch.append(unit)
            rows += len(unit)
        if batch:
            yield batch

    def predict_proba(self, texts):
        """One score per text: sigmoid of the max chunk logit. Ordering is all the harness uses."""
        import numpy as np
        import torch

        token_lists = self._encode(texts)
        self.model.eval()
        scores = []
        with torch.no_grad():
            for group in self._micro_batches([[i] for i in range(len(token_lists))], token_lists):
                rows = [i for unit in group for i in unit]
                # Sigmoid in float64: in float32 every logit above ~16.6 becomes exactly 1.0,
                # and the ties would cost AUROC and matched-FPR TPR their ordering.
                logits = self._doc_logits([token_lists[i] for i in rows]).detach().cpu().double()
                scores.append(torch.sigmoid(logits))
        return torch.cat(scores).numpy().astype(np.float64) if scores else np.zeros(0)

    def config(self) -> dict:
        return {field: getattr(self, field) for field in CONFIG_FIELDS}

    def save(self, path):
        import json
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "model")
        self.tokenizer.save_pretrained(path / "tokenizer")
        (path / "detector.json").write_text(json.dumps(
            {"config": self.config(), "selected_epoch": self.selected_epoch, "history": self.history},
            indent=2))

    @classmethod
    def load(cls, path, device=None):
        import json
        from pathlib import Path

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        saved = json.loads((path / "detector.json").read_text())
        detector = cls(**saved["config"], device=device)
        detector.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        detector.tokenizer.model_max_length = 10**9
        detector.model = AutoModelForSequenceClassification.from_pretrained(path / "model")
        detector.model.to(detector._torch_device())
        detector.selected_epoch = saved["selected_epoch"]
        detector.history = saved["history"]
        return detector
