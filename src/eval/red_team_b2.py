"""Red-team mutations for B2: score the mutated test rows with a saved DeBERTa checkpoint.

    PYTHONPATH=src/data:src/eval:src/models uv run --group deberta python src/eval/red_team_b2.py \
        --model models/deberta/row2-s7267

B2 cannot score new text from its Colab score file, so this loads the checkpoint (extracted from
models/deberta/row2-s7267.tar) and scores val, test and every mutated row locally. Before using
the local scores it checks that they agree with the committed score file on val and test, then
reuses red_team.evaluate and red_team.summarize unchanged, so the summary matches B1's format.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dataset import DEFAULT_PATH
from deberta_run import text_hash
from harness import Data, evaluate, load_data
from precomputed import PrecomputedScores
from red_team import DEFAULT_OUT, RED_TEAM_FPR, build_mutations, eval_sets, load_test, summarize

ROOT = Path(__file__).resolve().parents[2]
MIN_SPEARMAN = 0.99


class ScoreLookup:
    """Scores keyed by text hash, served like a fitted detector."""

    def __init__(self, name, scores: dict):
        self.name, self._scores = name, scores

    def predict_proba(self, texts):
        return np.array([self._scores[text_hash(t)] for t in texts], dtype=float)


def agreement(local: dict, committed: PrecomputedScores, texts) -> float:
    """Spearman correlation between local and committed scores on the same texts."""
    keys = [text_hash(t) for t in texts]
    return float(spearmanr([local[k] for k in keys], committed.predict_proba(texts)).statistic)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=str(ROOT / "models/deberta/row2-s7267"))
    ap.add_argument("--scores", default=str(ROOT / "report/deberta_scores/row2-s7267.parquet"),
                    help="committed Colab score file for the same run, used as a consistency check")
    ap.add_argument("--device", help="cuda, mps or cpu (default: auto)")
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args(argv)

    from deberta import DebertaDetector

    detector = DebertaDetector.load(args.model, device=args.device)
    run = json.loads((Path(args.model) / "run.json").read_text())
    name = f"deberta_{run['run_name'].replace('-', '_')}"
    data = load_data(args.data)
    test = load_test(args.data)
    mutated = build_mutations(test)

    texts = pd.unique(pd.concat([data.val.text, test.text, mutated.text], ignore_index=True))
    print(f"scoring {len(texts):,} unique texts on {detector._torch_device()}", flush=True)
    local = dict(zip(map(text_hash, texts), detector.predict_proba(list(texts))))

    committed = PrecomputedScores(args.scores)
    rho = {split: agreement(local, committed, frame.text) for split, frame in (("val", data.val), ("test", test))}
    print(f"agreement with {args.scores}: " + ", ".join(f"{k} Spearman {v:.4f}" for k, v in rho.items()))
    if min(rho.values()) < MIN_SPEARMAN:
        raise SystemExit(f"local scores disagree with the committed score file (< {MIN_SPEARMAN})")

    lookup = ScoreLookup(name, local)
    _, report = evaluate(lookup, Data(train=data.train, val=data.val, evals=eval_sets(test, mutated)),
                         target_fpr=RED_TEAM_FPR, n_boot=args.n_boot)
    summary = summarize(report, mutated)
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "report.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "agreement.json").write_text(json.dumps(rho, indent=2) + "\n")
    pd.set_option("display.width", 220)
    print(summary[["mutation", "family", "auroc", "tpr_at_val_threshold", "delta_tpr",
                   "ctrl_fpr_at_val_threshold"]].round(3).to_string(index=False))
    print(f"wrote {out}/summary.csv and report.csv")


if __name__ == "__main__":
    main()
