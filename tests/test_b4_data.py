"""Task recovery and training/evaluation boundaries for the B4 context sidecar."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import b4_data
from b4_data import POLICY, attach_contexts, context_index, digest, file_digest, load_frames


@pytest.fixture
def contexts_path(dataset_path, tmp_path):
    original = pq.read_table(dataset_path).to_pylist()
    records = [dict(id=r["id"], text_sha256=digest(r["text"]), context="find weather london",
                    context_kind="fixture", context_count=1, supported=True) for r in original]
    table = pa.Table.from_pylist(records).replace_schema_metadata({
        b"b4_contexts": json.dumps({"policy": POLICY, "dataset_sha256": file_digest(dataset_path)}).encode()})
    path = tmp_path / "contexts.parquet"
    pq.write_table(table, path)
    return path


def test_train_loader_never_requests_test_or_holdouts(dataset_path, contexts_path, monkeypatch):
    real = b4_data.load
    calls = []

    def guarded(purpose, **kwargs):
        calls.append((purpose, kwargs))
        assert purpose == "fit" and kwargs["roles"] == ["train_pool"]
        assert kwargs["splits"] in (["train"], ["val"])
        return real(purpose, **kwargs)

    monkeypatch.setattr(b4_data, "load", guarded)
    frames = load_frames(dataset_path, contexts_path)
    assert set(frames) == {"train", "val"} and len(calls) == 2


def test_evaluation_preserves_all_holdout_splits(dataset_path, contexts_path):
    frames = load_frames(dataset_path, contexts_path, evaluation=True)
    assert set(frames["mcptox"].split) == {"train", "val", "test"}
    assert set(frames["train"].role) == {"train_pool"}


def test_context_hash_guards_and_missing_rows(dataset_path, contexts_path):
    frame = pq.read_table(dataset_path).to_pandas()
    contexts = pq.read_table(contexts_path).to_pandas()
    with pytest.raises(ValueError, match="missing"):
        attach_contexts(frame, contexts.iloc[1:])
    contexts.loc[0, "text_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hashes"):
        attach_contexts(frame, contexts)


def test_context_recovery_never_uses_answers_or_attack_specific_queries(tmp_path):
    raw, generated = tmp_path / "raw", tmp_path / "generated"

    def write(path, obj, jsonl=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj) + ("\n" if jsonl else ""), encoding="utf-8")

    write(raw / "poisonedrag/results/adv_targeted_results/nq.json", {
        "one": {"id": "q1", "question": "trusted question", "correct answer": "SECRET_ANSWER"}})
    for task in ("email", "table", "code", "qa", "abstract"):
        base = generated / "bipia" if task in ("qa", "abstract") else raw / "bipia/benchmark"
        for split in ("train", "test"):
            write(base / task / f"{split}.jsonl", dict(context="candidate", question="trusted question",
                  ideal="SECRET_ANSWER", code=["code"], error=["error"]), jsonl=True)
    write(raw / "mcptox/response_all.json", {"servers": {
        "tools": {"clean_querys": ["legitimate task"], "malicious_instance": ["SECRET_ATTACK"]}}})
    write(raw / "mcptox/pure_tool.json", [{"attack": {"query": "SECRET_ATTACK"}}])
    write(raw / "msb/data/agent_task.jsonl", {"task_tool": [{"tool": "tools", "task": "legitimate task"}]}, True)
    index = context_index(raw, generated)
    assert index["poisonedrag:nq:q1"][0] == "trusted question"
    assert index["mcptox:tools"][0] == index["msb:tools"][0]
    assert "SECRET" not in str(index)
    assert index[f"bipia:abstract:train:{digest('candidate')[:16]}"][0].startswith("Summarize")
