"""Recover label-independent task context without changing the original dataset or splits.

The sidecar includes every original id, including unsupported rows. MCPTox/MSB use a
server-level catalogue of legitimate tasks, NOT attack-specific queries, clean text twins,
ideal answers, or attack objectives. See docs/b4-relevance.md for the resulting estimand.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataset import DEFAULT_PATH, load

POLICY = "b4-task-context-v1"
DEFAULT_CONTEXTS = DEFAULT_PATH.with_name("b4_contexts.parquet")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def as_text(value):
    return "\n".join(value) if isinstance(value, list) else value


def context_index(raw, generated):
    """group -> (context, provenance, number of available contexts); never reads labels."""
    raw, generated = Path(raw), Path(generated)
    contexts = {}
    for path in sorted((raw / "poisonedrag/results/adv_targeted_results").glob("*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")).values():
            if item.get("question", "").strip():
                contexts[f"poisonedrag:{path.stem}:{item['id']}"] = (
                    item["question"].strip(), "original_question", 1)

    for task in ("email", "table", "qa", "code", "abstract"):
        base = generated / "bipia" if task in ("qa", "abstract") else raw / "bipia/benchmark"
        for split in ("train", "test"):
            choices = defaultdict(set)
            for item in read_jsonl(base / task / f"{split}.jsonl"):
                # Match the original builder's hash BEFORE stripping text.
                group = f"bipia:{task}:{split}:{digest(as_text(item['context']))[:16]}"
                if task == "abstract":
                    context = "Summarize the following article."
                elif task == "code":
                    context = ("Help debug the following code and error.\nCode:\n"
                               + as_text(item.get("code", "")) + "\nError:\n"
                               + as_text(item.get("error", "")))
                else:
                    context = item.get("question", "")
                if context.strip():
                    choices[group].add(context.strip())
            kind = {"abstract": "fixed_summary_task", "code": "code_and_error"}.get(task, "original_question")
            for group, options in choices.items():
                # Original build deduplicated contexts with multiple questions. Keep one
                # deterministic question for BOTH clean and attacked rows of that group.
                contexts[group] = (sorted(options)[0], kind, len(options))

    servers = json.loads((raw / "mcptox/response_all.json").read_text(encoding="utf-8"))["servers"]
    for server, spec in servers.items():
        queries = sorted({q.strip() for q in spec.get("clean_querys", []) if q.strip()})
        if queries:
            contexts[f"mcptox:{server}"] = (
                "Supported user tasks:\n" + "\n".join(queries), "server_task_catalogue", len(queries))

    tasks = defaultdict(set)
    for agent in read_jsonl(raw / "msb/data/agent_task.jsonl"):
        for item in agent["task_tool"]:
            if item["task"].strip():
                tasks[item["tool"]].add(item["task"].strip())
    for server, queries in tasks.items():
        contexts[f"msb:{server}"] = (
            "Supported user tasks:\n" + "\n".join(sorted(queries)), "server_task_catalogue", len(queries))
    return contexts


def build_contexts(data_path=DEFAULT_PATH, raw="data/raw", generated="data/generated"):
    index = context_index(raw, generated)
    original = load("eval", path=data_path).to_pandas()
    if original.id.duplicated().any():
        raise ValueError("B4 expects unique original row ids")
    rows = []
    for row in original.itertuples():
        context, kind, count = index.get(row.group, ("", "unavailable", 0))
        rows.append(dict(id=row.id, text_sha256=digest(row.text), context=context,
                         context_kind=kind, context_count=count, supported=bool(context)))
    table = pa.Table.from_pylist(rows).replace_schema_metadata({
        b"b4_contexts": json.dumps({"policy": POLICY, "dataset_sha256": file_digest(data_path)}).encode()})
    coverage = original[["id", "source", "subset", "role", "split", "label"]].merge(
        pd.DataFrame(rows)[["id", "supported", "context_kind", "context_count"]], on="id", validate="one_to_one")
    coverage = coverage.groupby(["source", "subset", "role", "split", "label", "context_kind"],
                                dropna=False).agg(total=("id", "size"), supported=("supported", "sum"),
                                                ambiguous=("context_count", lambda v: int((v > 1).sum()))).reset_index()
    coverage["excluded"] = coverage.total - coverage.supported
    return table, coverage


def read_contexts(path):
    table = pq.read_table(path)
    meta = json.loads((table.schema.metadata or {}).get(b"b4_contexts", b"{}"))
    if meta.get("policy") != POLICY:
        raise ValueError("Unknown B4 context policy; rebuild contexts with b4_data.py")
    frame = table.to_pandas()
    required = {"id", "text_sha256", "context", "context_kind", "supported"}
    if not required <= set(frame) or frame.id.duplicated().any():
        raise ValueError("Context sidecar needs unique ids and the B4 context columns")
    if frame[list(required)].isna().any().any() or not frame.context.map(lambda s: isinstance(s, str)).all():
        raise ValueError("Null or invalid context values")
    if not (frame.supported == frame.context.str.strip().ne("")).all():
        raise ValueError("Context supported flags disagree with context availability")
    return frame, meta


def attach_contexts(frame, contexts):
    merged = frame.merge(contexts, on="id", how="left", validate="one_to_one", sort=False)
    if merged.supported.isna().any():
        raise ValueError("Context sidecar is missing original dataset ids")
    if not (merged.text.map(digest) == merged.text_sha256).all():
        raise ValueError("Context sidecar does not match candidate text hashes")
    return merged[merged.supported].copy().reset_index(drop=True)


def load_frames(data_path=DEFAULT_PATH, contexts_path=DEFAULT_CONTEXTS, *, evaluation=False):
    """Training reads ONLY train_pool train/val; evaluation additionally loads full holdouts."""
    contexts, meta = read_contexts(contexts_path)
    if meta.get("dataset_sha256") != file_digest(data_path):
        raise ValueError("Context sidecar belongs to a different dataset; rebuild it")

    def rows(purpose, **filters):
        return attach_contexts(load(purpose, path=data_path, **filters).to_pandas(), contexts)

    frames = {name: rows("fit", roles=["train_pool"], splits=[name]) for name in ("train", "val")}
    if evaluation:
        frames.update(test=rows("eval", roles=["train_pool"], splits=["test"]),
                      mcptox=rows("eval", sources=["mcptox"]), msb=rows("eval", sources=["msb"]))
    for name, frame in frames.items():
        if frame.empty or set(frame.label) != {0, 1}:
            raise ValueError(f"B4 {name} needs supported rows of both labels")
    return frames


def pairs(frame):
    return frame[["context", "text"]].to_dict("records")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=str(DEFAULT_PATH))
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--generated", default="data/generated")
    parser.add_argument("--out", default=str(DEFAULT_CONTEXTS))
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.resolve() == Path(args.data).resolve():
        parser.error("The sidecar must not overwrite the original dataset")
    table, coverage = build_contexts(args.data, args.raw, args.generated)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    coverage.to_csv(out.with_suffix(".coverage.csv"), index=False)
    print(coverage.groupby("source")[["total", "supported", "excluded"]].sum().to_string())
    print(f"Wrote {out}; original dataset and splits unchanged")


if __name__ == "__main__":
    main()
