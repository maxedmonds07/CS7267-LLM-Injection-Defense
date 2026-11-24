"""Inference latency, model size and fit time for each detector, on this machine.

    PYTHONPATH=src/data:src/eval:src/models uv run --group perplexity --group deberta \
        python scripts/bench_latency.py

Scores the same N test rows with each detector: one warm-up pass, then the median of REPEATS
timed passes. Writes report/phase5/latency.csv. Run it on an otherwise idle machine; the numbers
are wall-clock and depend on the hardware, which is recorded in the file.
"""

import argparse
import platform
import statistics
import subprocess
import time
import warnings
from pathlib import Path

import pandas as pd

from harness import load_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "phase5" / "latency.csv"
N_ROWS = 1000
REPEATS = 5
SEED = 7267
B2_MODEL = ROOT / "models" / "deberta" / "row2-s7267"


def machine() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        chip = platform.processor() or platform.machine()
    return f"{chip}, {platform.system()} {platform.release()}"


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return time.perf_counter() - start, result


def bench(predict, texts, repeats=REPEATS) -> float:
    """Median seconds for one pass over texts, after one untimed warm-up pass."""
    predict(texts)
    return statistics.median(timed(predict, texts)[0] for _ in range(repeats))


def n_params(detector) -> int:
    if hasattr(detector, "pipeline"):
        return int(detector.pipeline.steps[1][1].coef_.size + 1)
    if hasattr(detector, "model") and hasattr(detector.model, "parameters"):
        return int(sum(p.numel() for p in detector.model.parameters()))
    if hasattr(detector, "_model") and detector._model is not None:
        return int(sum(p.numel() for p in detector._model.parameters()))
    return 2  # length baseline: one weight and a bias


def detectors(data, include_torch=True):
    """(name, device, fit seconds or None, detector). Fit time is wall time on train_pool train."""
    from train import fit_detector

    for label, name in (("length", "length"), ("b1", "tfidf"), ("b1_optuna", "tfidf-optuna")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seconds, (detector, _) = timed(fit_detector, name, data)
        yield label, "cpu", seconds, detector
    if not include_torch:
        return
    from deberta import DebertaDetector
    from perplexity import PerplexityDetector

    for device in ("cpu", "mps"):
        yield "b0_perplexity", device, None, PerplexityDetector(device=device).fit([], [])
        yield "b2_row2_s7267", device, None, DebertaDetector.load(B2_MODEL, device=device)


def run(data, n_rows=N_ROWS, repeats=REPEATS, include_torch=True) -> pd.DataFrame:
    test = data.evals["test"].sample(min(n_rows, len(data.evals["test"])), random_state=SEED)
    texts = list(test.text)
    records = []
    for name, device, fit_seconds, detector in detectors(data, include_torch):
        seconds = bench(detector.predict_proba, texts, repeats)
        records.append({
            "detector": name, "device": device, "n_rows": len(texts), "repeats": repeats,
            "seconds_per_pass": seconds, "ms_per_row": 1000 * seconds / len(texts),
            "rows_per_second": len(texts) / seconds, "parameters": n_params(detector),
            "fit_seconds": fit_seconds, "machine": machine(),
            "mean_chars": float(test.text.str.len().mean()),
        })
        print(f"{name:15s} {device:4s} {1000 * seconds / len(texts):9.3f} ms/row", flush=True)
    return pd.DataFrame(records)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-rows", type=int, default=N_ROWS)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-torch", action="store_true", help="skip B0 and B2 (no torch install)")
    args = ap.parse_args(argv)
    table = run(load_data(), args.n_rows, args.repeats, include_torch=not args.no_torch)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
