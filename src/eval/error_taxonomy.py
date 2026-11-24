"""Failure-mode taxonomy: assign each error at the deployed threshold to a category with a fixed,
rule-based codebook, and count categories per detector.

    PYTHONPATH=src/data:src/eval:src/models uv run python src/eval/error_taxonomy.py

Reads report/phase5/scores.parquet and thresholds.json, and the texts from the dataset. Writes
report/phase5/errors/taxonomy.csv (counts), cue_rates.csv (how often each codebook cue appears
in benign vs adversarial rows) and sample.csv (row ids and categories of a fixed-seed sample,
for manual checking; no text, so no attack strings enter git).

The codebook was written after reading a sample of MCPTox errors, revised once after hand-checking
29 coded rows (missing verbs such as fetch, prepend and modify), and then checked on a fresh
sample (CHECK_SEED) that was not used to write it. Rules are
checked in order and the first match wins, so every error gets exactly one category.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from dataset import DEFAULT_PATH

ROOT = Path(__file__).resolve().parents[2]
PHASE5 = ROOT / "report" / "phase5"
DETECTORS = ("b1_optuna", "b2_row2_s7267")
EVAL_SETS = ("test", "mcptox", "msb")
SAMPLE_PER_CELL = 10
SEED = 7267
CHECK_SEED = 7268

# Cues. Each is a regular expression over the lower-cased text.
CUES = {
    "prerequisite_call": r"\b(before|prior to)\b[^.]{0,120}\b(call|run|use|invoke|query|read|execute|fetch|"
                         r"search|request|retrieve|get|check|open|list|show|make sure|verify)\b"
                         r"|\b(must|should|need to|required to)\s+(first\s+)?(call|run|invoke|execute|query|fetch)\b"
                         r"|\bfirst (call|run|invoke|execute|query|fetch)\b",
    "argument_override": r"\b(override|overwrite|replace|change|set|redirect|substitute|modify|prepend|append|"
                         r"rewrite|alter|route)\b[^.]{0,60}\b(parameter|argument|param|query|recipient|value|"
                         r"content|message|to\s+'|to\s+\"|to\s+`)|\balways (use|set|send|pass|rewrite|override)\b"
                         r"|\badd the parameter\b",
    "authority_claim": r"\b(priority|higher than|security alert|mandate|mandatory|compliance|policy|"
                       r"protocol|regulat\w*|required by|system notice|important:)\b",
    "usage_imperative": r"\b(must|always|never|only works|do not|don't|should|important|ensure|make sure)\b",
    "sensitive_vocab": r"\b(password|credential|token|secret|api key|env\w*|delete|drop|reset|"
                       r"overwrite|security|permission|auth\w*|private)\b",
    "repo_or_account_action": r"\b(repositor\w*|fork|push|commit|branch|account|namespace)\b",
}
COMPILED = {name: re.compile(pattern) for name, pattern in CUES.items()}

# Codebook: (error type, category, cues that must all match). First match wins; "other" catches
# the rest. Missed attacks are described by how they are phrased; false alarms by what in a benign
# text looks like an attack.
CODEBOOK = [
    ("false_negative", "procedural prerequisite (\"before X, call Y\")", ["prerequisite_call"]),
    ("false_negative", "argument or parameter override", ["argument_override"]),
    ("false_negative", "authority or policy framing", ["authority_claim"]),
    ("false_negative", "plain imperative, no injection vocabulary", ["usage_imperative"]),
    ("false_positive", "usage instructions addressed to the model", ["usage_imperative"]),
    ("false_positive", "sensitive or destructive vocabulary", ["sensitive_vocab"]),
    ("false_positive", "repository or account action", ["repo_or_account_action"]),
]
OTHER = "other"


def cues(text: str) -> dict:
    lowered = text.lower()
    return {name: bool(rx.search(lowered)) for name, rx in COMPILED.items()}


def categorize(text: str, error: str) -> str:
    found = cues(text)
    for kind, category, needed in CODEBOOK:
        if kind == error and all(found[c] for c in needed):
            return category
    return OTHER


def errors(rows: pd.DataFrame, detector: str, threshold: float) -> pd.DataFrame:
    """False positives and false negatives of one detector at its deployed threshold."""
    flagged = rows[detector] >= threshold
    kind = pd.Series(pd.NA, index=rows.index, dtype="object")
    kind[flagged & (rows.label == 0)] = "false_positive"
    kind[~flagged & (rows.label == 1)] = "false_negative"
    out = rows.assign(error=kind).dropna(subset=["error"])
    return out.assign(category=[categorize(t, e) for t, e in zip(out.text, out.error)])


def taxonomy(rows: pd.DataFrame, thresholds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables, samples = [], []
    for detector in DETECTORS:
        for eval_set in EVAL_SETS:
            part = errors(rows[rows.eval_set == eval_set], detector, thresholds[detector])
            totals = part.groupby("error").size()
            counts = part.groupby(["error", "category"]).size().rename("n").reset_index()
            counts["share"] = counts.n / counts.error.map(totals)
            tables.append(counts.assign(detector=detector, eval_set=eval_set))
            for (_, _), cell in part.groupby(["error", "category"]):
                pick = cell.sample(min(SAMPLE_PER_CELL, len(cell)), random_state=SEED)
                samples.append(pick.assign(detector=detector)[
                    ["detector", "eval_set", "id", "error", "category", "label", detector]
                ].rename(columns={detector: "score"}))
    columns = ["detector", "eval_set", "error", "category", "n", "share"]
    return pd.concat(tables)[columns], pd.concat(samples, ignore_index=True)


def cue_rates(rows: pd.DataFrame) -> pd.DataFrame:
    """Share of benign and adversarial rows in each eval set that contain each cue."""
    found = pd.DataFrame([cues(t) for t in rows.text], index=rows.index)
    found[["eval_set", "label"]] = rows[["eval_set", "label"]]
    rates = found.groupby(["eval_set", "label"]).mean().reset_index()
    return rates.melt(id_vars=["eval_set", "label"], var_name="cue", value_name="share")


def load_rows(folder: Path, data_path=DEFAULT_PATH) -> tuple[pd.DataFrame, dict]:
    scores = pd.read_parquet(folder / "scores.parquet")
    texts = pd.read_parquet(data_path, columns=["id", "text"])
    thresholds = json.loads((folder / "thresholds.json").read_text())
    return scores.merge(texts, on="id", how="left", validate="many_to_one"), thresholds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=str(PHASE5))
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    args = ap.parse_args(argv)
    folder = Path(args.dir)
    rows, thresholds = load_rows(folder, args.data)
    rows = rows[rows.eval_set.isin(EVAL_SETS)]
    table, sample = taxonomy(rows, thresholds)
    out = folder / "errors"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "taxonomy.csv", index=False)
    sample.to_csv(out / "sample.csv", index=False)
    cue_rates(rows).to_csv(out / "cue_rates.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
