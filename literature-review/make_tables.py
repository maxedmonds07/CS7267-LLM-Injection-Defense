# /// script
# requires-python = ">=3.11"
# dependencies = ["openpyxl"]
# ///
"""Build the Phase 2 report tables from the literature coding matrix.

  uv run literature-review/make_tables.py

Reads CS7267_Phase2_LitReview_CodingMatrix.xlsx and writes tables.md next to it. Rerun after
every coding change; edit the matrix, never tables.md.
"""

from collections import defaultdict
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
MATRIX = HERE / "CS7267_Phase2_LitReview_CodingMatrix.xlsx"
OUT = HERE / "tables.md"

SURFACES = [
    "RAG corpus / retriever",
    "Tool output (indirect, at runtime)",
    "Tool description / metadata (MCP)",
    "Direct user input",
    "Multiple (note which)",
]
PAYLOADS = ["Instruction injection", "Content poisoning", "Both (note which)"]
DEFENSES = [
    "Detection (classifier / probe)",
    "Input transformation / prompting",
    "Training-time (fine-tune / alignment)",
    "Architectural isolation",
    "Runtime policy / capability enforcement",
    "Certified / provable",
    "Forensics / traceback",
]
NO_DEFENSE = "N/A (attack- or benchmark-only)"

# Papers whose benchmark or released data other work evaluates on. PoisonedRAG is an attack
# paper, but its targeted questions are the de facto RAG-poisoning benchmark.
BENCHMARKS = ["P03", "P08", "P09", "P12", "P14", "P16", "P18", "P20"]
# Benchmarks this project's dataset or harness is built from (config.yaml).
IN_OUR_DATA = {
    "P03": "harness (AgentDojo 0.1.35)",
    "P12": "LOSO holdout",
    "P14": "train pool",
    "P18": "transfer holdout",
    "P20": "train pool",
}


def load_rows():
    sheet = openpyxl.load_workbook(MATRIX, read_only=True)["Coding Matrix"]
    rows = sheet.iter_rows(values_only=True)
    header = next(rows)
    return [dict(zip(header, row)) for row in rows if row[0]]


def cite(row):
    return f"{row['ID']} {row['Short name']}"


def cell(rows):
    return "<br>".join(cite(r) for r in rows) or "—"


def table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def short(text):
    """Drop the dropdown's '(note which)' hint in table labels."""
    return text.replace(" (note which)", "")


def threat_taxonomy(rows):
    grid = defaultdict(list)
    for r in rows:
        grid[r["Attack surface"], r["Payload"]].append(r)
    body = [[short(s)] + [cell(grid[s, p]) for p in PAYLOADS] for s in SURFACES]
    return table(["Attack surface \\ Payload"] + [short(p) for p in PAYLOADS], body)


def defense_taxonomy(rows):
    body = []
    for d in DEFENSES:
        papers = [r for r in rows if r["Defense class"] == d]
        points = sorted({r["Enforcement point"] for r in papers})
        surfaces = sorted({short(r["Attack surface"]) for r in papers})
        body.append([d, cell(papers), ", ".join(points) or "—", ", ".join(surfaces) or "—"])
    return table(["Defense class", "Papers", "Enforcement point", "Surfaces defended"], body)


def benchmarks(rows):
    by_id = {r["ID"]: r for r in rows}
    body = []
    for pid in BENCHMARKS:
        r = by_id[pid]
        body.append(
            [
                cite(r),
                short(r["Attack surface"]),
                short(r["Payload"]),
                r["Benchmarks / datasets used (free text)"],
                r["Primary metric"],
                r["Key quantitative result + page/table (free text)"],
                r["Addresses MCP explicitly"],
                IN_OUR_DATA.get(pid, "—"),
            ]
        )
    return table(
        ["Benchmark", "Surface", "Payload", "Scale", "Metric", "Headline result", "MCP", "In our data"],
        body,
    )


def coverage(rows):
    grid = defaultdict(list)
    for r in rows:
        grid[r["Attack surface"], r["Defense class"]].append(r)
    columns = DEFENSES + [NO_DEFENSE]
    body = [
        [short(s)] + [", ".join(r["ID"] for r in grid[s, d]) or "**0**" for d in columns]
        for s in SURFACES
    ]
    header = ["Surface \\ Defense"] + [d.split(" (")[0] for d in DEFENSES] + ["None (attack / benchmark)"]
    return table(header, body)


def gaps(rows):
    """Per surface: how much attack evidence exists versus how much defense and detection."""
    body = []
    for s in SURFACES:
        here = [r for r in rows if r["Attack surface"] == s]
        threats = [r for r in here if r["Defense class"] == NO_DEFENSE]
        defenses = [r for r in here if r["Defense class"] != NO_DEFENSE]
        detectors = [r for r in defenses if r["Defense class"] == DEFENSES[0]]
        body.append([short(s), len(here), cell(threats), cell(defenses), cell(detectors)])
    return table(["Surface", "Papers", "Attacks / benchmarks only", "Defenses", "Detectors"], body)


def main():
    rows = load_rows()
    verified = [r for r in rows if "Claude" not in str(r["Coded by (initials)"])]
    venue_checked = [r for r in rows if r["Venue verified? (Y/N + where)"]]
    sections = [
        "# Phase 2 literature tables",
        f"Generated by `make_tables.py` from `{MATRIX.name}`; do not edit by hand.\n\n"
        f"**Coding status:** {len(verified)} of {len(rows)} rows confirmed by a team member, "
        f"{len(venue_checked)} of {len(rows)} venues verified. Rows coded by Claude are drafts: "
        "do not cite anything below until its row is confirmed.",
        "## Table 1. Threat taxonomy: attack surface by payload\n\n"
        "Where adversarial content enters the system, and how it works on the model. "
        "Defense papers are placed by the threat they defend against.\n\n" + threat_taxonomy(rows),
        "## Table 2. Defense taxonomy\n\n" + defense_taxonomy(rows),
        "## Table 3. Benchmarks\n\n" + benchmarks(rows),
        "## Table 4. Coverage: attack surface by defense class\n\n"
        "**0** marks an empty cell: a candidate gap in the papers held, to be confirmed by a "
        "targeted search before it is claimed.\n\n" + coverage(rows),
        "## Table 5. Gaps by surface\n\n" + gaps(rows),
    ]
    OUT.write_text("\n\n".join(sections) + "\n")
    print(f"wrote {OUT.relative_to(HERE.parent)} ({len(rows)} papers, {len(verified)} confirmed)")


if __name__ == "__main__":
    main()
