"""Sanity checks on data/raw as produced by the `fetch` stage (`make data`)."""

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
RAW = ROOT / CONFIG["paths"]["raw_dir"]
DATASETS = CONFIG["datasets"]


def dataset_dir(name: str) -> Path:
    path = RAW / name
    if not (path / "SOURCE.json").exists():
        pytest.skip(f"{path} not fetched; run `make data`")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("name", DATASETS)
def test_source_matches_config(name):
    source = json.loads((dataset_dir(name) / "SOURCE.json").read_text())
    assert source["repo"] == DATASETS[name]["repo"]
    assert source["commit"] == DATASETS[name]["commit"], "stale data; run `make data`"


@pytest.mark.parametrize("name", DATASETS)
def test_configured_paths_present_and_nonempty(name):
    root = dataset_dir(name)
    for rel in DATASETS[name]["paths"]:
        path = root / rel
        assert path.exists(), f"missing {rel}"
        files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
        assert files, f"{rel} is empty"
        assert all(f.stat().st_size > 0 for f in files if f.suffix in {".json", ".jsonl"})


class TestPoisonedRAG:
    beir = DATASETS["poisonedrag"]["beir"]

    @pytest.mark.parametrize("subset", beir["subsets"])
    def test_attack_cases(self, subset):
        root = dataset_dir("poisonedrag")
        cases = json.loads((root / f"results/adv_targeted_results/{subset}.json").read_text())
        assert len(cases) == 100
        for case in cases.values():
            assert {"id", "question", "correct answer", "incorrect answer", "adv_texts"} <= case.keys()
            assert case["adv_texts"] and all(isinstance(t, str) and t for t in case["adv_texts"])

    @pytest.mark.parametrize("subset", beir["subsets"])
    def test_beir_subset_covers_targets(self, subset):
        root = dataset_dir("poisonedrag")
        cases = json.loads((root / f"results/adv_targeted_results/{subset}.json").read_text())
        retrieved = json.loads((root / f"results/beir_results/{subset}-contriever.json").read_text())
        out = root / "beir" / subset

        target_qids = {c["id"] for c in cases.values()}
        assert {q["_id"] for q in read_jsonl(out / "queries.jsonl")} == target_qids

        corpus_ids = {d["_id"] for d in read_jsonl(out / "corpus.jsonl")}
        for qid in target_qids:
            assert set(list(retrieved[qid])[: self.beir["top_k"]]) <= corpus_ids

        qrels = (out / "qrels.tsv").read_text().splitlines()
        assert qrels[0].split("\t") == ["query-id", "corpus-id", "score"]
        assert {row.split("\t")[1] for row in qrels[1:]} <= corpus_ids


class TestBIPIA:
    @pytest.mark.parametrize("task", ["email", "table", "code"])
    @pytest.mark.parametrize("split", ["train", "test"])
    def test_contexts(self, task, split):
        rows = read_jsonl(dataset_dir("bipia") / f"benchmark/{task}/{split}.jsonl")
        assert rows
        assert all({"context", "ideal"} <= row.keys() for row in rows)

    @pytest.mark.parametrize("kind", ["text", "code"])
    @pytest.mark.parametrize("split", ["train", "test"])
    def test_attacks(self, kind, split):
        attacks = json.loads(
            (dataset_dir("bipia") / f"benchmark/{kind}_attack_{split}.json").read_text()
        )
        assert attacks
        for family, prompts in attacks.items():
            assert prompts and all(isinstance(p, str) for p in prompts), family


class TestMCPTox:
    def test_responses(self):
        data = json.loads((dataset_dir("mcptox") / "response_all.json").read_text())
        assert {"data_length", "attack_scopes", "servers"} <= data.keys()
        assert len(data["servers"]) == 45

    def test_tools(self):
        tools = json.loads((dataset_dir("mcptox") / "pure_tool.json").read_text())
        assert len(tools) == 45


class TestMSB:
    def test_agent_tasks(self):
        rows = read_jsonl(dataset_dir("msb") / "data/agent_task.jsonl")
        assert rows
        assert all({"agent_name", "system_prompt", "task_tool"} <= r.keys() for r in rows)

    def test_attack_tasks(self):
        rows = read_jsonl(dataset_dir("msb") / "data/attack_task.jsonl")
        assert rows
        assert all({"attack_task", "implementation"} <= r.keys() for r in rows)

    def test_attack_types(self):
        (row,) = read_jsonl(dataset_dir("msb") / "data/attack_type.jsonl")
        assert "prompt_injection" in row["attack_type"]

    def test_no_upstream_secrets(self):
        assert not list(dataset_dir("msb").rglob(".env"))
