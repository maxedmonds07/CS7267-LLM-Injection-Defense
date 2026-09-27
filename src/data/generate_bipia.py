"""Rebuild BIPIA's WebQA and Summarization contexts, which BIPIA does not redistribute.

Reimplements BIPIA's benchmark/{qa,abstract}/process.py without the `datasets` library:
rows are selected by BIPIA's index.json and the output must match its md5.txt byte for byte,
or the task's `md5` in config.yaml where BIPIA's can't be reproduced (WebQA).

  bipia_summarization: XSum parquet from Hugging Face at a pinned revision.
  bipia_webqa:         combined-newsqa-data-v1.csv built by `make newsqa` and tracked by DVC.

Usage: python src/data/generate_bipia.py --task {bipia_summarization,bipia_webqa} [--config config.yaml]
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import string
import sys
from itertools import chain
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from fetch import download

HF_PARQUET_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/data/{split}-00000-of-00001.parquet"
SPLITS = ("train", "test")


def xsum_rows(spec: dict, indexes: dict, cache_dir: Path) -> dict[str, list[dict]]:
    """abstract/process.py: XSum train/test rows at BIPIA's indexes."""
    rows = {}
    for split in SPLITS:
        url = HF_PARQUET_URL.format(repo=spec["source"], revision=spec["revision"], split=split)
        path = download(url, cache_dir / spec["revision"] / f"{split}.parquet")
        table = pq.read_table(path, columns=["document", "summary"]).take(indexes[split])
        rows[split] = [{"ideal": r["summary"], "context": r["document"]} for r in table.to_pylist()]
    return rows


def read_newsqa_csv(path: Path) -> list[list[str]]:
    """Parse the combined CSV exactly as the Hugging Face `newsqa` loader (combined-csv) does."""
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(reader)
        return [row for row in reader if row]


def newsqa_answers(story: str, char_ranges: str) -> list[str]:
    """Answer spans from `answer_char_ranges`, e.g. "196:228|196:202,217:228|None"."""
    answers = set()
    for char_range in chain(*(part.split("|") for part in char_ranges.split(","))):
        if char_range != "None":
            start, end = map(int, char_range.split(":"))
            answers.add(story[start:end].strip(string.punctuation + string.whitespace))
    return sorted(answers)


def normalize_story(story: str) -> str:
    story = re.sub(r"\n\s*\n", "\n\n", story)
    return re.sub(r"\n{3,}", "\n\n", story)


def newsqa_rows(spec: dict, indexes: dict, cache_dir: Path) -> dict[str, list[dict]]:
    """qa/process.py: both splits are drawn from the single combined "train" split."""
    source = Path(spec["source"])
    if not source.exists():
        raise FileNotFoundError(f"{source} missing; run `make newsqa` or `dvc pull {source}`")
    table = read_newsqa_csv(source)

    rows = {}
    for split in SPLITS:
        rows[split] = []
        for i in indexes[split]:
            # Columns: story_id, question, answer_char_ranges, ..., story_text
            row = table[i]
            rows[split].append(
                {
                    "ideal": newsqa_answers(row[-1], row[2]),
                    "context": normalize_story(row[-1]),
                    "question": row[1],
                }
            )
    # Upstream's one manual correction.
    rows["test"][87]["ideal"] = ["Janine Sligar"]
    return rows


BUILDERS = {"bipia_summarization": xsum_rows, "bipia_webqa": newsqa_rows}


def generate(task: str, config: dict) -> None:
    spec = config["generated"][task]
    bipia_dir = Path(config["paths"]["raw_dir"]) / "bipia" / "benchmark" / spec["task_dir"]
    dest = Path(config["paths"]["generated_dir"]) / "bipia" / spec["task_dir"]
    cache_dir = Path(".cache") / task

    indexes = json.loads((bipia_dir / "index.json").read_text())
    bipia_md5 = dict(
        reversed(line.split("  ")) for line in (bipia_dir / "md5.txt").read_text().splitlines()
    )
    expected = spec.get("md5", bipia_md5)

    print(f"{task}: {spec['source']} -> {dest}", file=sys.stderr)
    rows = BUILDERS[task](spec, indexes, cache_dir)

    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for split in SPLITS:
        # Matches jsonlines' default encoder, which BIPIA's process.py writes with.
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[split]).encode()
        name = f"{split}.jsonl"
        if (actual := hashlib.md5(data).hexdigest()) != expected[name]:
            shutil.rmtree(dest)
            raise ValueError(f"{task}/{name}: md5 {actual} != expected {expected[name]}")
        (dest / name).write_bytes(data)
        print(f"  {name}: {len(rows[split])} rows, md5 ok", file=sys.stderr)

    source = {k: v for k, v in spec.items() if k != "task_dir"}
    source["bipia_commit"] = config["datasets"]["bipia"]["commit"]
    source["md5"] = expected
    source["matches_bipia_md5"] = expected == bipia_md5
    (dest / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--task", required=True, choices=BUILDERS)
    args = parser.parse_args()
    generate(args.task, yaml.safe_load(args.config.read_text()))


if __name__ == "__main__":
    main()
