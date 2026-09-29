"""Build one labeled table of untrusted text segments from every fetched injection source.

A row is text that reaches the model without the user writing it (a retrieved passage, a
fetched document, a tool description or a tool response), labeled adversarial (1) or
benign (0). Design: docs/superpowers/specs/2026-09-29-build-dataset-design.md

Usage: python src/data/build_dataset.py [--config config.yaml]
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

COLUMNS = [
    "id", "text", "label", "attack_kind", "attack_objective", "attack_family",
    "source", "subset", "attack_surface", "group", "split", "role",
]
OBJECTIVES = {
    "task_hijack", "data_exfiltration", "misinformation", "denial_of_service",
    "tool_selection_manipulation",
}
ROLES = {"train_pool", "loso_holdout", "transfer_holdout"}
SPLITS = ("train", "val", "test")
ATTACK_FIELDS = ("attack_kind", "attack_objective", "attack_family")


class BuildError(Exception):
    """An input or invariant problem: the build stops and writes nothing."""


def make_row(row_id, text, *, source, subset, surface, group, split=None, attack=None) -> dict:
    """A benign row, or an adversarial one when attack=(kind, family, (mapping, category)).

    `split` may preset "test" or "trainval" for sources that ship their own split (BIPIA).
    The objective is looked up later by apply_objectives, from config.yaml.
    """
    row = dict.fromkeys(COLUMNS)
    row.update(
        id=row_id, text=text, label=0, source=source, subset=subset,
        attack_surface=surface, group=group, split=split,
    )
    if attack:
        kind, family, objective_key = attack
        row.update(label=1, attack_kind=kind, attack_family=family, _objective=objective_key)
    return row


def apply_objectives(rows: list[dict], objectives: dict) -> None:
    """Fill attack_objective from config.yaml's dataset.objectives, listing every gap at once."""
    missing, invalid = set(), set()
    for row in rows:
        key = row.pop("_objective", None)
        if key is None:
            continue
        mapping, category = key
        objective = (objectives.get(mapping) or {}).get(category)
        if objective is None:
            missing.add(f"{mapping}: {category}")
        elif objective not in OBJECTIVES:
            invalid.add(f"{mapping}: {category} -> {objective}")
        row["attack_objective"] = objective
    problems = []
    if missing:
        problems.append("no dataset.objectives mapping in config.yaml for:\n  " + "\n  ".join(sorted(missing)))
    if invalid:
        problems.append(
            f"not a coding-matrix objective ({', '.join(sorted(OBJECTIVES))}):\n  " + "\n  ".join(sorted(invalid))
        )
    if problems:
        raise BuildError("\n".join(problems))


def dedupe(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop repeated texts with the same label (first copy kept); fail on a text with both labels."""
    label_of, kept, dropped, conflicts = {}, [], 0, []
    for row in rows:
        if row["text"] not in label_of:
            label_of[row["text"]] = row["label"]
            kept.append(row)
        elif label_of[row["text"]] == row["label"]:
            dropped += 1
        else:
            conflicts.append(row["id"])
    if conflicts:
        raise BuildError(f"same text with both labels: {conflicts[:5]}")
    return kept, dropped


def unit_interval(seed: int, group: str) -> float:
    """A stable pseudo-random number in [0, 1) for a group, independent of row order."""
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_splits(rows: list[dict], seed: int, ratios: dict) -> None:
    """Split by group so near-duplicates never straddle splits.

    A preset "test" is kept; a preset "trainval" chooses train or val in proportion;
    no preset uses all three ratios.
    """
    val_share = ratios["val"] / (ratios["train"] + ratios["val"])
    for row in rows:
        u = unit_interval(seed, row["group"])
        preset = row["split"]
        if preset == "test":
            continue
        if preset == "trainval":
            row["split"] = "val" if u < val_share else "train"
        elif preset is None:
            if u < ratios["train"]:
                row["split"] = "train"
            elif u < ratios["train"] + ratios["val"]:
                row["split"] = "val"
            else:
                row["split"] = "test"
        else:
            raise BuildError(f"unexpected preset split {preset!r} on {row['id']}")


def validate(rows: list[dict]) -> None:
    """Check the spec's invariants; report every kind of failure at once."""
    problems = []
    duplicates = [i for i, n in Counter(r["id"] for r in rows).items() if n > 1]
    if duplicates:
        problems.append(f"duplicate id: {duplicates[:5]}")
    group_splits = {}
    for row in rows:
        group_splits.setdefault(row["group"], set()).add(row["split"])
    spanning = sorted(g for g, splits in group_splits.items() if len(splits) > 1)
    if spanning:
        problems.append(f"group in more than one split: {spanning[:5]}")
    checks = {
        "benign row with attack fields": lambda r: r["label"] == 0 and any(r[f] is not None for f in ATTACK_FIELDS),
        "adversarial row missing attack fields": lambda r: r["label"] == 1 and any(r[f] is None for f in ATTACK_FIELDS),
        "empty text": lambda r: not r["text"] or not r["text"].strip(),
        "unknown split": lambda r: r["split"] not in SPLITS,
        "unknown role": lambda r: r["role"] not in ROLES,
    }
    for message, failed in checks.items():
        bad = [r["id"] for r in rows if failed(r)]
        if bad:
            problems.append(f"{message}: {bad[:5]}")
    if problems:
        raise BuildError("\n".join(problems))


def read_json(path: Path):
    if not path.exists():
        raise BuildError(f"missing {path}; run `make data` (or `dvc pull`)")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise BuildError(f"missing {path}; run `make data` (or `dvc pull`)")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rows_poisonedrag(raw: Path, subsets: list[str], surface: str, top_k: int) -> list[dict]:
    """5 adversarial passages per target question, plus its top-k real retrieved passages."""
    rows = []
    for subset in subsets:
        targets = read_json(raw / "results/adv_targeted_results" / f"{subset}.json")
        retrieved = read_json(raw / "results/beir_results" / f"{subset}-contriever.json")
        corpus = {doc["_id"]: doc["text"] for doc in read_jsonl(raw / "beir" / subset / "corpus.jsonl")}
        for target in targets.values():
            qid = target["id"]
            group = f"poisonedrag:{subset}:{qid}"
            common = dict(source="poisonedrag", subset=subset, surface=surface, group=group)
            for j, text in enumerate(target["adv_texts"]):
                rows.append(make_row(
                    f"{group}:adv:{j}", text,
                    attack=("poisoning", "poisonedrag", ("poisonedrag", "poisonedrag")), **common,
                ))
            scores = retrieved[qid]
            for doc_id in sorted(scores, key=lambda d: (-scores[d], d))[:top_k]:
                rows.append(make_row(f"{group}:doc:{doc_id}", corpus[doc_id], **common))
    return rows
