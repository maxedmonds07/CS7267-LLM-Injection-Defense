"""Train, score and evaluate B4 using the original split assignments and paired-data cohort."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from b4_data import DEFAULT_CONTEXTS, digest, file_digest, load_frames, pairs
from dataset import DEFAULT_PATH
from deberta_run import TRAIN_SOURCES, git_commit
from harness import Data, evaluate


def input_identity(data, contexts):
    return {"dataset_sha256": file_digest(data), "contexts_sha256": file_digest(contexts)}


def code_digest():
    """Also detects uncommitted changes and source bundles without a .git directory."""
    import hashlib
    source = Path(__file__).resolve().parents[1]
    hasher = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        hasher.update(path.relative_to(source).as_posix().encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def train_model(data, contexts, model_out, *, model_config, train_sources=TRAIN_SOURCES,
                checkpoint_dir=None, limit=None, device=None, run_name="b4"):
    from b4 import B4Detector
    if not train_sources or not set(train_sources) <= set(TRAIN_SOURCES):
        raise ValueError("B4 training sources must be bipia and/or poisonedrag")
    frames = load_frames(data, contexts, evaluation=False)
    train, val = frames["train"], frames["val"]
    train = train[train.source.isin(train_sources)]
    if limit is not None:
        if limit < 2:
            raise ValueError("Smoke limit must be at least two")
        # Sample whole groups; the same seed gives the same rows for every input ablation.
        order = np.random.default_rng(model_config.get("seed", 7267)).permutation(sorted(train.group.unique()))
        chosen, n = [], 0
        for group in order:
            chosen.append(group)
            n += int((train.group == group).sum())
            if n >= limit:
                break
        train = train[train.group.isin(chosen)]
    detector = B4Detector(**model_config, checkpoint_dir=checkpoint_dir, device=device)
    detector.fit(pairs(train), train.label, groups=train.group, val=(pairs(val), val.label))
    out = Path(model_out)
    detector.save(out)
    run = {"run_name": run_name, "train_sources": list(train_sources), "limit": limit,
           "n_train": len(train), "n_val": len(val), "seed": detector.seed,
           "git_commit": git_commit(), "code_sha256": code_digest(),
           **input_identity(data, contexts), "config": detector.config(),
           "selected_epoch": detector.selected_epoch, "history": detector.history}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"Trained {run_name}, {len(train)} rows, selected epoch {detector.selected_epoch}", flush=True)
    return run


def score_model(data, contexts, model_path, out, *, val_only=False, device=None):
    from b4 import B4Detector
    model_path = Path(model_path)
    run = json.loads((model_path / "run.json").read_text(encoding="utf-8"))
    if any(run[k] != v for k, v in input_identity(data, contexts).items()):
        raise ValueError("Data or paired contexts changed since training")
    if run["limit"] is not None and not val_only:
        raise ValueError("Smoke runs may score validation only")
    frames = load_frames(data, contexts, evaluation=not val_only)
    detector = B4Detector.load(model_path, device=device)
    tables = []
    for name, frame in frames.items():
        if name == "train":
            continue
        scored = frame[["id", "label"]].copy()
        scored["text_sha256"] = frame.text.map(digest)
        scored["context_sha256"] = frame.context.map(digest)
        scored["score"] = detector.predict_proba(pairs(frame))
        scored["eval_set"] = name
        tables.append(scored)
    table = pa.Table.from_pandas(pd.concat(tables, ignore_index=True), preserve_index=False)
    table = table.replace_schema_metadata({b"b4": json.dumps({**run, "val_only": val_only}).encode()})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(out).with_suffix(".parquet.tmp")
    pq.write_table(table, temp)
    temp.replace(out)
    print(f"Scored {run['run_name']} -> {out}", flush=True)


class B4Scores:
    """Context-aware score validation before adapting to the existing text-only harness."""
    def __init__(self, path, frames, identity):
        table = pq.read_table(path)
        self.meta = json.loads((table.schema.metadata or {}).get(b"b4", b"{}"))
        if any(self.meta.get(k) != v for k, v in identity.items()):
            raise ValueError("Score file has different data/context fingerprints")
        frame = table.to_pandas()
        if frame.id.duplicated().any():
            raise ValueError("Duplicate score identities")
        if not np.isfinite(frame.score).all() or not frame.score.between(0, 1).all():
            raise ValueError("B4 score file contains invalid probabilities")
        expected = []
        for name, part in frames.items():
            if name == "train":
                continue
            part = part[["id", "label", "text", "context"]].copy()
            part["eval_set"] = name
            part["text_sha256"] = part.text.map(digest)
            part["context_sha256"] = part.context.map(digest)
            expected.append(part.drop(columns=["text", "context"]))
        expected = pd.concat(expected, ignore_index=True)
        if set(frame.id) != set(expected.id):
            raise ValueError("Score file must cover exactly the requested evaluation cohort")
        joined = expected.merge(frame, on="id", suffixes=("_expected", "_saved"), validate="one_to_one")
        for key in ("label", "eval_set", "text_sha256", "context_sha256"):
            if not (joined[f"{key}_expected"] == joined[f"{key}_saved"]).all():
                raise ValueError(f"Score file disagrees with {key}")
        self.name = self.meta["run_name"]
        self.scores = dict(zip(frame.id, frame.score))

    def predict_proba(self, texts):
        """The harness passes a text Series. Its index identifies the already validated pair."""
        if not isinstance(texts, pd.Series) or texts.index.name != "id":
            raise ValueError("B4 score lookup needs a text Series indexed by dataset row id")
        return np.asarray([self.scores[row_id] for row_id in texts.index])


def harness_data(frames):
    return Data(frames["train"], frames["val"],
                {k: v for k, v in frames.items() if k not in ("train", "val")})


def evaluate_scores(data, contexts, scores, out, *, n_boot=1000, target_fpr=0.01, tracking_uri=None):
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    frames = load_frames(data, contexts, evaluation=True)
    detector = B4Scores(scores, frames, input_identity(data, contexts))
    # Keep the actual candidate text for length-based metrics and its row id for B4 score lookup.
    indexed = {name: frame.set_index("id") for name, frame in frames.items()}
    threshold, report = evaluate(detector, harness_data(indexed), target_fpr=target_fpr, n_boot=n_boot)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    metrics = {"threshold": threshold, "target_fpr": target_fpr, "n_boot": n_boot,
               "run_name": detector.name, "scope": "B4 supported-context cohort"}
    Path(out).with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if tracking_uri:
        import mlflow
        from train import metric_items
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("b4-relevance")
        with mlflow.start_run(run_name=detector.name):
            mlflow.log_params({**metrics, **detector.meta["config"],
                               **input_identity(data, contexts)})
            mlflow.log_metrics(metric_items(report))
            mlflow.log_dict(detector.meta, "run.json")
            mlflow.log_artifact(str(out))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--model-out", required=True)
    fit.add_argument("--checkpoint-dir")
    fit.add_argument("--input-mode", choices=["text", "pair", "relevance"], default="relevance")
    fit.add_argument("--train-sources", nargs="+", choices=TRAIN_SOURCES, default=list(TRAIN_SOURCES))
    fit.add_argument("--run-name", default="b4")
    fit.add_argument("--seed", type=int, default=7267)
    fit.add_argument("--epochs", type=int, default=3)
    fit.add_argument("--lr", type=float, default=2e-5)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--max-chunks", type=int, default=16)
    fit.add_argument("--context-tokens", type=int, default=128)
    fit.add_argument("--limit", type=int)
    fit.add_argument("--no-fp16", action="store_true")
    sc = commands.add_parser("score")
    sc.add_argument("--model", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--val-only", action="store_true")
    ev = commands.add_parser("evaluate")
    ev.add_argument("--scores", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--n-boot", type=int, default=1000)
    ev.add_argument("--target-fpr", type=float, default=0.01)
    ev.add_argument("--tracking-uri")
    for command in (fit, sc, ev):
        command.add_argument("--data", default=str(DEFAULT_PATH))
        command.add_argument("--contexts", default=str(DEFAULT_CONTEXTS))
    for command in (fit, sc):
        command.add_argument("--device")
    args = parser.parse_args(argv)
    if args.command == "train":
        config = {k: getattr(args, k) for k in ("input_mode", "seed", "epochs", "lr", "batch_size",
                                               "max_chunks", "context_tokens")}
        train_model(args.data, args.contexts, args.model_out, model_config={**config, "fp16": not args.no_fp16},
                    train_sources=args.train_sources, checkpoint_dir=args.checkpoint_dir, limit=args.limit,
                    device=args.device, run_name=args.run_name)
    elif args.command == "score":
        score_model(args.data, args.contexts, args.model, args.out, val_only=args.val_only, device=args.device)
    else:
        evaluate_scores(args.data, args.contexts, args.scores, args.out, n_boot=args.n_boot,
                        target_fpr=args.target_fpr, tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
