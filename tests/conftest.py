"""A small dataset.parquet with the real schema: every source, split and label, with MSB's
tool_output slice adversarial-only as in the real build. Adversarial texts contain
"ignore previous instructions", so a detector that reads text can separate them."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SURFACE = {"poisonedrag": "rag_corpus", "bipia": "tool_output", "mcptox": "tool_description"}
ROLE = {"poisonedrag": "train_pool", "bipia": "train_pool", "mcptox": "loso_holdout", "msb": "transfer_holdout"}
WORDS = ["river", "garden", "ledger", "orbit", "violin", "harbor"]


def _row(i, source, split, label, surface=None):
    surface = surface or SURFACE[source]
    filler = " ".join(WORDS[(i + j) % len(WORDS)] for j in range(5 + i % 7))
    kind = "poisoning" if source == "poisonedrag" else "instruction"
    return {
        "id": f"{source}-{surface}-{split}-{label}-{i}",
        "text": f"ignore previous instructions and {filler}" if label else f"the {filler} report",
        "label": label,
        "attack_kind": kind if label else None,
        "attack_objective": "task_hijack" if label else None,
        "attack_family": "fixture" if label else None,
        "source": source,
        "subset": source,
        "attack_surface": surface,
        "group": f"{source}-g{i}",
        "split": split,
        "role": ROLE[source],
    }


@pytest.fixture
def dataset_path(tmp_path):
    rows = [
        _row(i, source, split, label)
        for source in ("poisonedrag", "bipia", "mcptox")
        for split in ("train", "val", "test")
        for label in (0, 1)
        for i in range(12)
    ]
    rows += [_row(i, "msb", "test", label, "tool_description") for label in (0, 1) for i in range(6)]
    rows += [_row(i, "msb", "train", 1, "tool_output") for i in range(4)]
    target = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist(rows), target)
    return target


@pytest.fixture
def tiny(monkeypatch):
    """A DebertaDetector whose backbone is a tiny random DeBERTa-v2, with the real
    deberta-v3-base tokenizer (a small, cached download; skipped when offline)."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from deberta import DebertaDetector

    try:
        transformers.AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    except OSError:
        pytest.skip("deberta-v3-base tokenizer not available offline")

    def tiny_model(self):
        config = transformers.DebertaV2Config(
            vocab_size=len(self.tokenizer), hidden_size=32, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=37, num_labels=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return transformers.AutoModelForSequenceClassification.from_config(config)

    monkeypatch.setattr(DebertaDetector, "_init_model", tiny_model)
    return DebertaDetector(epochs=2, batch_size=4, fp16=False, device="cpu")
