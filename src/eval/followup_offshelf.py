"""Follow-up: Prompt Guard 2 (off the shelf, zero-shot) against length and B1-Optuna on MCPTox
(outside the pre-registered protocol).

    PYTHONPATH=src/data:src/eval:src/models uv run python src/eval/followup_offshelf.py

Reads report/phase5/scores.parquet and report/offshelf_scores/prompt-guard-2.parquet (written by
scripts/prompt_guard_colab.py; `dvc pull report/offshelf_scores.dvc`), matches rows by text hash,
and writes report/followup/offshelf.json: AUROC differences with 95% intervals from resampling
MCPTox servers, and Prompt Guard's median score by label.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import DEFAULT_PATH, load
from deberta_run import text_hash
from followup_length import SCORES, paired_server_bootstrap
from metrics import auroc

ROOT = Path(__file__).resolve().parents[2]
PROMPT_GUARD = ROOT / "report" / "offshelf_scores" / "prompt-guard-2.parquet"
OUT = ROOT / "report" / "followup" / "offshelf.json"
COLUMN = "prompt_guard_2"


def attach(scores: pd.DataFrame, texts: pd.DataFrame, offshelf: pd.DataFrame, eval_set: str = "mcptox") -> pd.DataFrame:
    """The eval set's rows of `scores` with the off-the-shelf score as COLUMN. `texts` maps id to
    text; `offshelf` is a score file (text_sha256, eval_set, score). Every row must match."""
    rows = scores[scores.eval_set == eval_set].copy()
    hashes = texts.set_index("id").text.map(text_hash)
    lookup = offshelf[offshelf.eval_set == eval_set].set_index("text_sha256").score
    rows[COLUMN] = rows.id.map(hashes).map(lookup).to_numpy()
    if rows[COLUMN].isna().any():
        raise ValueError(f"{int(rows[COLUMN].isna().sum())} {eval_set} rows have no {COLUMN} score")
    return rows


def compare(rows: pd.DataFrame, n_boot: int = 1000, seed: int = 7267) -> dict:
    labels = rows.label.to_numpy()
    return {
        f"{COLUMN}_auroc": auroc(rows[COLUMN].to_numpy(), labels),
        f"{COLUMN}_minus_length": paired_server_bootstrap(rows, COLUMN, "length", n_boot=n_boot, seed=seed),
        f"{COLUMN}_minus_b1_optuna": paired_server_bootstrap(rows, COLUMN, "b1_optuna", n_boot=n_boot, seed=seed),
        "median_score": {"benign": float(np.median(rows[COLUMN][labels == 0])),
                         "attack": float(np.median(rows[COLUMN][labels == 1]))},
        "servers": int(rows.subset.nunique()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    ap.add_argument("--scores", default=str(SCORES))
    ap.add_argument("--offshelf", default=str(PROMPT_GUARD))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    texts = load("eval", sources=["mcptox"], path=args.data).select(["id", "text"]).to_pandas()
    rows = attach(pd.read_parquet(args.scores), texts, pd.read_parquet(args.offshelf))
    result = compare(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
