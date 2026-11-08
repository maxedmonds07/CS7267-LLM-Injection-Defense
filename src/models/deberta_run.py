"""Train a B2 DeBERTa run and write its score file. The Colab notebook calls this; all logic
lives here and in deberta.py.

    PYTHONPATH=src/data:src/eval:src/models uv run --group deberta python src/models/deberta_run.py \
        train --row row1 --seed 7267 --model-out RUN/model --checkpoint-dir RUN/ckpt
    PYTHONPATH=... python src/models/deberta_run.py score --model RUN/model --out RUN/scores.parquet

A score file holds one score per row of val (and, unless --val-only, test, MCPTox and MSB).
`make evaluate DETECTOR=deberta SCORES=...` then runs the unchanged harness on it locally.
Design: docs/superpowers/specs/2026-11-07-deberta-detector-design.md
"""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataset import DEFAULT_PATH
from harness import load_data

TRAIN_SOURCES = ("bipia", "poisonedrag")
METADATA_KEY = b"deberta"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scores_table(frames: dict, scores: dict, run_info: dict, val_only: bool) -> pa.Table:
    """One row per scored text: text_sha256, eval_set, label, score; run_info in the schema
    metadata. Texts are unique in dataset.parquet, so the hash identifies the row."""
    parts = [
        pd.DataFrame({
            "text_sha256": [text_hash(t) for t in frame.text],
            "eval_set": name,
            "label": frame.label.to_numpy().astype("int8"),
            "score": np.asarray(scores[name], dtype="float64"),
        })
        for name, frame in frames.items()
    ]
    table = pa.Table.from_pandas(pd.concat(parts, ignore_index=True), preserve_index=False)
    metadata = {METADATA_KEY: json.dumps({**run_info, "val_only": val_only}).encode()}
    return table.replace_schema_metadata(metadata)


def write_scores(detector, data, path, run_info: dict, val_only: bool) -> None:
    frames = {"val": data.val} if val_only else {"val": data.val, **data.evals}
    scores = {name: detector.predict_proba(frame.text) for name, frame in frames.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(scores_table(frames, scores, run_info, val_only), path)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _limit(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """About n rows, sampled per label so both labels survive (smoke runs only)."""
    parts = [
        rows.sample(n=max(1, min(len(rows), round(n * len(rows) / len(frame)))), random_state=seed)
        for _, rows in frame.groupby("label")
    ]
    return pd.concat(parts).sort_index()


def train(args) -> None:
    from deberta import DebertaDetector

    data = load_data(args.data)
    train_rows = data.train[data.train.source.isin(args.train_sources)]
    val_rows = data.val  # always the full train_pool val (spec decision 13)
    if args.limit:
        train_rows = _limit(train_rows, args.limit, args.seed)
        val_rows = _limit(val_rows, math.ceil(args.limit / 4), args.seed)

    detector = DebertaDetector(
        pairwise_weight=args.pairwise_weight, epochs=args.epochs, seed=args.seed,
        fp16=not args.no_fp16, device=args.device, checkpoint_dir=args.checkpoint_dir,
    )
    detector.fit(train_rows.text, train_rows.label, groups=train_rows.group,
                 val=(val_rows.text, val_rows.label))
    detector.save(args.model_out)
    run = {
        "run_name": args.run_name or f"{args.row}-s{args.seed}",
        "row": args.row,
        "seed": args.seed,
        "train_sources": list(args.train_sources),
        "limit": args.limit,
        "git_commit": git_commit(),
    }
    (Path(args.model_out) / "run.json").write_text(json.dumps(run, indent=2))
    print(f"trained {run['run_name']}: selected epoch {detector.selected_epoch}, "
          f"history {detector.history}")


def score(args) -> None:
    from deberta import DebertaDetector

    detector = DebertaDetector.load(args.model, device=args.device)
    run = json.loads((Path(args.model) / "run.json").read_text())
    run_info = {**run, "selected_epoch": detector.selected_epoch, "config": detector.config()}
    write_scores(detector, load_data(args.data), args.out, run_info, args.val_only)
    print(f"wrote {args.out} ({'val only' if args.val_only else 'val, test, mcptox, msb'})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("train", help="fit on train_pool train, choose the epoch on val")
    fit.add_argument("--row", required=True, help="run-plan row, e.g. row1")
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--model-out", required=True)
    fit.add_argument("--run-name", help="default: {row}-s{seed}; sweeps use sweep-row2-w{weight}-s{seed}")
    fit.add_argument("--checkpoint-dir")
    fit.add_argument("--pairwise-weight", type=float, default=0.0)
    fit.add_argument("--train-sources", nargs="+", choices=TRAIN_SOURCES, default=list(TRAIN_SOURCES))
    fit.add_argument("--limit", type=int, help="smoke runs: about N train rows and N/4 val rows")
    fit.add_argument("--epochs", type=int, default=3)
    fit.add_argument("--no-fp16", action="store_true")
    fit.add_argument("--device", help="cuda, mps or cpu (default: auto)")
    fit.add_argument("--data", default=str(DEFAULT_PATH))

    sc = commands.add_parser("score", help="write a score file from a trained model")
    sc.add_argument("--model", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--val-only", action="store_true", help="sweeps: never score the holdouts")
    sc.add_argument("--device")
    sc.add_argument("--data", default=str(DEFAULT_PATH))

    args = parser.parse_args(argv)
    {"train": train, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
