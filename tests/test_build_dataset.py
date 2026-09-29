"""Unit tests for build_dataset, on small fixtures in each source's own format."""

import json
import random
from pathlib import Path

import pytest
import yaml

import build_dataset as bd

ROOT = Path(__file__).resolve().parents[1]
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def ben(row_id="b", text="good", group="g", split=None):
    return bd.make_row(row_id, text, source="s", subset="x", surface="rag_corpus", group=group, split=split)


def adv(row_id="a", text="bad", group="g", split=None, key=("m", "cat")):
    return bd.make_row(
        row_id, text, source="s", subset="x", surface="rag_corpus", group=group, split=split,
        attack=("instruction", "cat", key),
    )


def valid_rows():
    rows = [ben(), adv()]
    bd.apply_objectives(rows, {"m": {"cat": "task_hijack"}})
    for row in rows:
        row["split"], row["role"] = "train", "train_pool"
    return rows


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_benign_row_has_schema_columns_and_no_attack_fields():
    row = ben()
    assert list(row) == bd.COLUMNS
    assert row["label"] == 0
    assert row["attack_kind"] is row["attack_objective"] is row["attack_family"] is None


def test_apply_objectives_maps_and_removes_private_key():
    rows = [adv(), ben()]
    bd.apply_objectives(rows, {"m": {"cat": "task_hijack"}})
    assert rows[0]["attack_objective"] == "task_hijack"
    assert list(rows[0]) == bd.COLUMNS


def test_apply_objectives_lists_every_missing_category():
    rows = [adv(key=("m", "one")), adv(row_id="a2", key=("n", "two"))]
    with pytest.raises(bd.BuildError, match=r"m: one[\s\S]*n: two"):
        bd.apply_objectives(rows, {"m": {}})


def test_apply_objectives_rejects_values_outside_the_matrix():
    with pytest.raises(bd.BuildError, match="not a coding-matrix objective"):
        bd.apply_objectives([adv()], {"m": {"cat": "world_domination"}})


def test_dedupe_keeps_first_copy_and_counts_drops():
    kept, dropped = bd.dedupe([ben("b1", "same"), ben("b2", "same"), ben("b3", "other")])
    assert [r["id"] for r in kept] == ["b1", "b3"]
    assert dropped == 1


def test_dedupe_fails_when_a_text_has_both_labels():
    with pytest.raises(bd.BuildError, match="both labels"):
        bd.dedupe([ben("b", "same"), adv("a", "same")])


def test_splits_are_per_group_and_follow_ratios():
    rows = [ben(f"b{i}", f"t{i}", group=f"g{i % 400}") for i in range(2000)]
    bd.assign_splits(rows, seed=1, ratios=RATIOS)
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_group.values())
    assert 0.6 < sum(r["split"] == "train" for r in rows) / len(rows) < 0.8


def test_preset_splits_are_respected():
    rows = [ben(f"t{i}", f"t{i}", group=f"t{i}", split="test") for i in range(50)]
    rows += [ben(f"v{i}", f"v{i}", group=f"v{i}", split="trainval") for i in range(200)]
    bd.assign_splits(rows, seed=1, ratios=RATIOS)
    assert {r["split"] for r in rows[:50]} == {"test"}
    assert {r["split"] for r in rows[50:]} == {"train", "val"}


def test_splits_are_deterministic_and_seeded():
    def splits(seed):
        rows = [ben(f"b{i}", group=f"g{i}") for i in range(200)]
        bd.assign_splits(rows, seed=seed, ratios=RATIOS)
        return [r["split"] for r in rows]

    assert splits(1) == splits(1)
    assert splits(1) != splits(2)


def test_validate_accepts_valid_rows():
    bd.validate(valid_rows())


@pytest.mark.parametrize(
    "break_rows, message",
    [
        (lambda rows: rows[1].update(id="b"), "duplicate id"),
        (lambda rows: rows[1].update(split="test"), "more than one split"),
        (lambda rows: rows[0].update(attack_kind="instruction"), "benign row with attack fields"),
        (lambda rows: rows[1].update(attack_family=None), "adversarial row missing attack fields"),
        (lambda rows: rows[0].update(text="  "), "empty text"),
        (lambda rows: rows[0].update(split="holdout"), "unknown split"),
        (lambda rows: rows[0].update(role="whatever"), "unknown role"),
    ],
)
def test_validate_rejects_broken_rows(break_rows, message):
    rows = valid_rows()
    break_rows(rows)
    with pytest.raises(bd.BuildError, match=message):
        bd.validate(rows)


def poisonedrag_fixture(root):
    write_json(root / "results/adv_targeted_results/nq.json",
               {"q1": {"id": "q1", "question": "?", "adv_texts": ["fake one", "fake two"]}})
    write_json(root / "results/beir_results/nq-contriever.json", {"q1": {"d1": 0.5, "d2": 0.9, "d3": 0.7}})
    write_jsonl(root / "beir/nq/corpus.jsonl",
                [{"_id": d, "title": "Title", "text": f"real {d}"} for d in ("d1", "d2", "d3")])


def test_poisonedrag_rows(tmp_path):
    poisonedrag_fixture(tmp_path)
    rows = bd.rows_poisonedrag(tmp_path, ["nq"], "rag_corpus", top_k=2)
    adversarial = [r for r in rows if r["label"] == 1]
    benign = [r for r in rows if r["label"] == 0]
    assert [r["text"] for r in adversarial] == ["fake one", "fake two"]
    # Highest-scoring passages first, text only (PoisonedRAG never shows the LLM a title).
    assert [r["text"] for r in benign] == ["real d2", "real d3"]
    assert {r["group"] for r in rows} == {"poisonedrag:nq:q1"}
    assert adversarial[0]["attack_kind"] == "poisoning"
    assert adversarial[0]["attack_surface"] == "rag_corpus"
    assert adversarial[0]["_objective"] == ("poisonedrag", "poisonedrag")


def test_missing_input_points_to_make_data(tmp_path):
    with pytest.raises(bd.BuildError, match="make data"):
        bd.rows_poisonedrag(tmp_path, ["nq"], "rag_corpus", top_k=2)
