"""Build one labeled table of untrusted text segments from every fetched injection source.

A row is text that reaches the model without the user writing it (a retrieved passage, a
fetched document, a tool description or a tool response), labeled adversarial (1) or
benign (0). Design: docs/superpowers/specs/2026-09-29-build-dataset-design.md

Usage: python src/data/build_dataset.py [--config config.yaml]
"""

import ast
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from nltk.tokenize.punkt import PunktSentenceTokenizer

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


POSITIONS = ("start", "middle", "end")
# task -> attack file prefix (text_attack_*.json or code_attack_*.json)
BIPIA_TASKS = {"email": "text", "table": "text", "qa": "text", "abstract": "text", "code": "code"}
BIPIA_GENERATED = {"qa", "abstract"}  # rebuilt by the bipia_* stages into data/generated


def insert_attack(context: str, attack: str, position: str, rng: random.Random) -> str:
    """BIPIA's insert_start / insert_end / insert_middle (bipia/data/utils.py)."""
    if position == "start":
        return "\n".join([attack, context])
    if position == "end":
        return "\n".join([context, attack])
    spans = list(PunktSentenceTokenizer().span_tokenize(context))
    start, _ = rng.sample(spans, k=1)[0]
    return "\n".join([context[:start], attack, context[start:]])


def flatten_attacks(attacks: dict) -> dict:
    """Name each attack string "{category}-{i}", as BIPIA's load_attack does."""
    return {f"{category}-{i}": text for category, texts in attacks.items() for i, text in enumerate(texts)}


def rows_bipia(raw: Path, generated: Path, surface: str, seed: int, attacks_per_context: int) -> list[dict]:
    """Every clean context, plus sampled attacks inserted at sampled positions.

    BIPIA's own train/test split is kept: its train contexts and attacks become our
    train/val ("trainval"), its test ones our test.
    """
    rows = []
    for task, attack_file in BIPIA_TASKS.items():
        for bipia_split in ("train", "test"):
            attacks = flatten_attacks(read_json(raw / "benchmark" / f"{attack_file}_attack_{bipia_split}.json"))
            names = sorted(attacks)
            base = generated if task in BIPIA_GENERATED else raw / "benchmark"
            contexts = read_jsonl(base / task / f"{bipia_split}.jsonl")
            preset = "trainval" if bipia_split == "train" else "test"
            for i, sample in enumerate(contexts):
                context = sample["context"]
                if isinstance(context, list):  # code contexts are lists of lines
                    context = "\n".join(context)
                group = f"bipia:{task}:{bipia_split}:{i}"
                common = dict(source="bipia", subset=task, surface=surface, group=group, split=preset)
                rows.append(make_row(f"{group}:clean", context, **common))
                rng = random.Random(f"{seed}:{group}")
                for name in rng.sample(names, k=min(attacks_per_context, len(names))):
                    position = rng.choice(POSITIONS)
                    category = name.rsplit("-", 1)[0]
                    rows.append(make_row(
                        f"{group}:{name}:{position}", insert_attack(context, attacks[name], position, rng),
                        attack=("instruction", category, ("bipia", category)), **common,
                    ))
    return rows


# One "Tool: …\nDescription: …\nArguments:" block of an MCPTox clean system prompt.
MCPTOX_TOOL_BLOCK = re.compile(r"Tool: (.+?)\nDescription: (.*?)\nArguments:", re.S)


def tool_text(name: str, description: str) -> str:
    """The one layout for every tool_description row, taken from MCPTox's system prompts."""
    return f"Tool: {name}\nDescription: {description.strip()}"


def rows_mcptox(raw: Path, surface: str) -> list[dict]:
    """Each server's clean tool descriptions, plus the 485 poisoned tools."""
    rows = []
    for server, spec in read_json(raw / "response_all.json")["servers"].items():
        group = f"mcptox:{server}"
        for name, description in MCPTOX_TOOL_BLOCK.findall(spec["clean_system_promot"]):
            rows.append(make_row(
                f"{group}:clean:{name}", tool_text(name, description),
                source="mcptox", subset=server, surface=surface, group=group,
            ))
    for entry in read_json(raw / "pure_tool.json"):
        for key, tool in entry.items():
            server, risk = tool["server_name"], tool["security risk"]
            rows.append(make_row(
                f"mcptox:{key}", tool_text(tool["tool_name"], tool["tool_content"]),
                source="mcptox", subset=server, surface=surface, group=f"mcptox:{server}",
                attack=("instruction", f"{risk} / {tool['paradigm']}", ("mcptox", risk)),
            ))
    return rows


# MSB response attacks (its main.py): attack type -> template constant, None = the task alone.
MSB_RESPONSE_TEMPLATES = {
    "false_error": "TOOL_RESPONSE_ATTACK_TEMPLATE",
    "simulated_user": "SIMULATED_USER_TEMPLATE",
    "search_term_deception": None,
}
MSB_PID = "4242"  # stands in for the live PID MSB substitutes into kill_process


def is_tool_decorator(node: ast.expr) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    return isinstance(target, ast.Attribute) and target.attr == "tool"


def tool_docstrings(path: Path) -> dict[str, str]:
    """Docstrings of @mcp.tool() functions, read with ast so MSB's servers never run."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: ast.get_docstring(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and ast.get_docstring(node)
        and any(is_tool_decorator(d) for d in node.decorator_list)
    }


def string_constants(path: Path) -> dict[str, str]:
    """Top-level NAME = "..." assignments (MSB's data/prompt_template.py)."""
    if not path.exists():
        raise BuildError(f"missing {path}; run `make data` (or `dvc pull`)")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def rows_msb(raw: Path, seed: int) -> list[dict]:
    """The static texts MSB's main.py assembles: tool descriptions and tool responses."""
    data = raw / "data"
    templates = string_constants(data / "prompt_template.py")
    tasks = {
        t["attack_task"]: t["implementation"].replace("{pid}", MSB_PID)
        for t in read_jsonl(data / "attack_task.jsonl")
    }
    rows = []
    for server_dir in sorted(p for p in (data / "tools/attack_tools").glob("*/*") if p.is_dir()):
        server = server_dir.name
        group = f"msb:{server}"
        common = dict(source="msb", subset=server, surface="tool_description", group=group)
        clean = {}
        for path in sorted(server_dir.glob("*.py")):
            if "-" not in path.stem:
                clean.update(tool_docstrings(path))
        for name, doc in clean.items():
            rows.append(make_row(f"{group}:clean:{name}", tool_text(name, doc), **common))
            # main.py's prompt_injection: description.rstrip() + "\n" + the filled template.
            task = random.Random(f"{seed}:{group}:{name}").choice(sorted(tasks))
            injection = templates["PROMPT_INJECTION_TEMPLATE"].replace("{goal}", tasks[task])
            rows.append(make_row(
                f"{group}:prompt_injection:{name}", tool_text(name, doc.rstrip() + "\n" + injection),
                attack=("instruction", "prompt_injection", ("msb_tasks", task)), **common,
            ))
        for path in sorted(server_dir.glob("*-*.py")):
            attack_type = path.stem.split("-", 1)[1]
            for name, doc in tool_docstrings(path).items():
                if clean.get(name) != doc:
                    rows.append(make_row(
                        f"{group}:{attack_type}:{name}", tool_text(name, doc),
                        attack=("instruction", attack_type, ("msb_types", attack_type)), **common,
                    ))
    for task, implementation in sorted(tasks.items()):
        group = f"msb:task:{task}"
        for attack_type, template in MSB_RESPONSE_TEMPLATES.items():
            prefix = templates[template] if template else ""
            rows.append(make_row(
                f"{group}:{attack_type}", prefix + implementation,
                source="msb", subset="tool_response", surface="tool_output", group=group,
                attack=("instruction", attack_type, ("msb_tasks", task)),
            ))
    return rows
