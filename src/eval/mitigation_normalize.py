"""Mitigation experiment: normalize text before scoring, then rerun the obfuscation mutations.

    PYTHONPATH=src/data:src/eval:src/models uv run python src/eval/mitigation_normalize.py

The frozen B1-Optuna model is unchanged; only its input is normalized (Unicode NFKC, then
zero-width characters removed), for val, the original test split and every mutation. This is an
analysis of one mitigation, reported apart from the frozen results. Writes
report/red_team/tfidf-optuna-normalized/summary.csv in the red-team summary format.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from dataset import DEFAULT_PATH
from harness import Data, evaluate, load_data
from red_team import DEFAULT_OUT, RED_TEAM_FPR, build_mutations, eval_sets, load_test, summarize

ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
OBFUSCATION = ["leetspeak", "zero_width", "homoglyph", "case_noise"]


def normalize(text: str) -> str:
    return ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))


class Normalized:
    """Wrap a fitted detector so every text is normalized before it is scored."""

    def __init__(self, detector):
        self.detector, self.name = detector, f"{detector.name}-normalized"

    def predict_proba(self, texts):
        return self.detector.predict_proba([normalize(t) for t in texts])


def main(argv=None):
    from train import fit_detector

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--detector", default="tfidf-optuna")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args(argv)
    data = load_data(DEFAULT_PATH)
    test = load_test(DEFAULT_PATH)
    mutated = build_mutations(test, OBFUSCATION)
    detector = Normalized(fit_detector(args.detector, data)[0])
    _, report = evaluate(detector, Data(train=data.train, val=data.val, evals=eval_sets(test, mutated)),
                         target_fpr=RED_TEAM_FPR, n_boot=args.n_boot)
    summary = summarize(report, mutated)
    out = Path(DEFAULT_OUT) / f"{args.detector}-normalized"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary[["mutation", "auroc", "tpr_at_val_threshold", "delta_tpr"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
