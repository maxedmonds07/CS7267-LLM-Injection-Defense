"""Is a fitted detector cheating? Runs the shortcut probe battery and gate on top of the team harness.

    uv run python src/eval/shortcut_check.py --detector tfidf [--data PATH] [--out DIR]

Fits the detector exactly as train.py does (train.fit_detector on train_pool train), scores the
same three eval sets (train_pool test, MCPTox LOSO, MSB transfer), then runs check_detector:
probes it must beat, label shuffle, length matching, padding invariance, split leaks.
For --detector deberta (a score file) label shuffle and padding are skipped: both need new scores.
Prints PASS / FAIL / INCONCLUSIVE with reasons. Nothing is logged to MLflow here.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from baselines import LengthBaseline
from dataset import DEFAULT_PATH
from harness import load_data
from precomputed import PrecomputedScores
from shortcut_probe import check_detector
from tfidf import TfidfDetector
from train import fit_detector

def factory(name, params):
    """Unfitted copy for the label-shuffle check; None when refitting would be too costly."""
    if name == "tfidf":
        return lambda: TfidfDetector(C=params["C"])
    if name == "length":
        return LengthBaseline
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--detector", choices=["length", "tfidf", "perplexity", "deberta"], required=True)
    ap.add_argument("--scores", default=None, help="B2 score file (report/deberta_scores/<run>.parquet), for --detector deberta")
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    ap.add_argument("--out", default=None, help="folder for gate.csv, probes.csv, verdict.txt")
    args = ap.parse_args(argv)

    data = load_data(args.data)
    detector, params = fit_detector(args.detector, data, scores=args.scores)
    report = check_detector(detector, data.train, data.evals, make_detector=factory(args.detector, params),
                            leak_against="test", can_rescore=not isinstance(detector, PrecomputedScores))

    pd.set_option("display.width", 220)
    print(f"{args.detector}: {report.verdict}")
    for reason in report.reasons:
        print(" -", reason)
    print("\nGate table:\n", report.gate_table.round(3).to_string(index=False))
    print("\nProbe battery:\n", report.probe_table.round(3).to_string(index=False))
    print("\nlabel shuffle AUROCs:", report.shuffle_aurocs, "\nsplit leaks:", report.leaks)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        report.gate_table.to_csv(out / "gate.csv", index=False)
        report.probe_table.to_csv(out / "probes.csv", index=False)
        (out / "verdict.txt").write_text(report.verdict + "\n" + "\n".join(report.reasons) + "\n")
    return report


if __name__ == "__main__":
    main()
