"""Score every val and eval row once per frozen detector, for the Phase 5 analyses.

    PYTHONPATH=src/data:src/eval:src/models uv run python src/eval/phase5_scores.py

Writes report/phase5/scores.parquet (one row per dataset row and eval set, one score column per
detector, no text) and report/phase5/thresholds.json (each detector's deployed threshold: 1% FPR
on val benign rows, as in Phase 4). Nothing is refit except B1 and B1-Optuna, which are
deterministic refits of their committed settings; B2 is read from its Colab score files.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import DEFAULT_PATH, load
from harness import load_data
from metrics import threshold_at_fpr
from precomputed import PrecomputedScores
from train import fit_detector

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "report" / "phase5"
SCORES_DIR = ROOT / "report" / "deberta_scores"
TARGET_FPR = 0.01
B2_RUNS = ("row2-s7267", "row2-s7268", "row2-s7269")
META = ["id", "label", "source", "subset", "attack_kind", "attack_objective", "attack_family",
        "attack_surface", "group"]
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def script_of(text: str) -> str:
    """'cjk' if the text contains Chinese, Japanese or Korean characters, else 'latin'."""
    return "cjk" if CJK.search(text) else "latin"


def detectors(data, scores_dir=SCORES_DIR, b2_runs=B2_RUNS):
    """(column name, fitted detector) for every detector Phase 5 analyzes."""
    yield "length", fit_detector("length", data)[0]
    yield "b1", fit_detector("tfidf", data)[0]
    yield "b1_optuna", fit_detector("tfidf-optuna", data)[0]
    for run in b2_runs:
        yield f"b2_{run.replace('-', '_')}", PrecomputedScores(Path(scores_dir) / f"{run}.parquet")


# The same filters as harness.load_data, so the rows match the Phase 4 evaluation exactly.
SETS = {
    "val": ("fit", {"roles": ["train_pool"], "splits": ["val"]}),
    "test": ("eval", {"roles": ["train_pool"], "splits": ["test"]}),
    "mcptox": ("eval", {"sources": ["mcptox"]}),
    "msb": ("eval", {"sources": ["msb"]}),
}


def frames(path=DEFAULT_PATH) -> pd.DataFrame:
    """Val plus every eval set, with each row's id and metadata."""
    parts = [load(purpose, path=path, **filters).select(["text", *META]).to_pandas().assign(eval_set=name)
             for name, (purpose, filters) in SETS.items()]
    rows = pd.concat(parts, ignore_index=True)
    rows["n_chars"] = rows.text.str.len()
    rows["script"] = rows.text.map(script_of)
    return rows


def score_all(data, path=DEFAULT_PATH, scores_dir=SCORES_DIR, b2_runs=B2_RUNS):
    rows = frames(path)
    thresholds = {}
    is_val_benign = (rows.eval_set == "val") & (rows.label == 0)
    for name, detector in detectors(data, scores_dir, b2_runs):
        rows[name] = detector.predict_proba(rows.text)
        thresholds[name] = float(threshold_at_fpr(rows.loc[is_val_benign, name].to_numpy(), TARGET_FPR))
    return rows.drop(columns="text"), thresholds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    data = load_data(args.data)
    rows, thresholds = score_all(data, args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / "scores.parquet", index=False)
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    print(f"wrote {len(rows):,} rows x {len(thresholds)} detectors to {out}/scores.parquet")
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
