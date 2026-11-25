"""Run the Colab B4 experiment grid, then produce same-cohort reports and seed intervals."""

import argparse
import gc
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from b4_data import DEFAULT_CONTEXTS, load_frames
from b4_run import code_digest, evaluate_scores, harness_data, input_identity, score_model, train_model
from dataset import DEFAULT_PATH

METRICS = ("auroc", "tpr", "fpr", "tpr_at_matched_fpr", "length_binned_auroc")


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"version", "seeds", "model", "experiments", "search", "evaluation", "smoke"}
    if not isinstance(config, dict) or set(config) != required or config["version"] != 1:
        raise ValueError("Unknown or missing B4 experiment configuration keys/version")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Experiment seeds must be nonempty and unique")
    model_keys = {"model_name", "epochs", "lr", "batch_size", "chunk_tokens", "chunk_stride",
                  "context_tokens", "max_chunks", "max_grad_norm", "relevance_max_features", "fp16"}
    if set(config["model"]) - model_keys:
        raise ValueError("Unknown model configuration key")
    names = set()
    for exp in config["experiments"]:
        if set(exp) != {"name", "input_mode", "train_sources"} or not re.fullmatch(r"[a-z][a-z0-9_]*", exp["name"]):
            raise ValueError("Invalid experiment definition")
        if exp["name"] in names or exp["input_mode"] not in ("text", "pair", "relevance"):
            raise ValueError("Duplicate experiment or unknown input mode")
        if not exp["train_sources"] or not set(exp["train_sources"]) <= {"bipia", "poisonedrag"}:
            raise ValueError("Only train_pool sources can train B4")
        names.add(exp["name"])
    if not names:
        raise ValueError("No experiments configured")
    if set(config["search"]) != {"enabled", "learning_rates"} or not isinstance(config["search"]["enabled"], bool):
        raise ValueError("Invalid search configuration")
    if not config["search"]["learning_rates"] or any(v <= 0 for v in config["search"]["learning_rates"]):
        raise ValueError("Search learning rates must be positive")
    if set(config["evaluation"]) != {"target_fpr", "n_boot"} or not 0 <= config["evaluation"]["target_fpr"] < 1:
        raise ValueError("Invalid evaluation configuration")
    if config["evaluation"]["n_boot"] < 1 or set(config["smoke"]) != {"limit", "epochs"}:
        raise ValueError("Invalid bootstrap or smoke configuration")
    return config


def freeze_manifest(path, manifest):
    path = Path(path)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError(f"Experiment configuration/data/code changed: use a new output directory ({path})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def selected_metrics(run):
    return run["history"][run["selected_epoch"]]


def release_memory():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one(root, name, data, contexts, model_config, train_sources, device, limit=None):
    folder = Path(root) / name
    manifest = {"name": name, "config": model_config, "train_sources": list(train_sources),
                "limit": limit, **input_identity(data, contexts), "code_sha256": code_digest()}
    freeze_manifest(folder / "manifest.json", manifest)
    run_path = folder / "model/run.json"
    if run_path.exists():
        print(f"Reuse finished training: {name}", flush=True)
        return json.loads(run_path.read_text(encoding="utf-8"))
    run = train_model(data, contexts, folder / "model", model_config=model_config,
                      train_sources=train_sources, checkpoint_dir=folder / "ckpt", limit=limit,
                      device=device, run_name=name)
    # Drive only needs optimizer state while the run is unfinished. Keep the selected model.
    shutil.rmtree(folder / "ckpt")
    release_memory()
    return run


def run_sweep(config, root, data, contexts, device):
    trials = []
    for i, lr in enumerate(config["search"]["learning_rates"]):
        name = f"sweep-{i}-s{config['seeds'][0]}"
        run = run_one(root / "sweep", name, data, contexts,
                      {**config["model"], "input_mode": "relevance", "seed": config["seeds"][0], "lr": lr},
                      ["bipia", "poisonedrag"], device)
        scores = root / "sweep" / name / "scores.parquet"
        if not scores.exists():
            score_model(data, contexts, scores.parent / "model", scores, val_only=True, device=device)
            release_memory()
        trials.append({"name": name, "lr": lr, **selected_metrics(run)})
    # Validation only; no test/holdout prediction or performance enters selection.
    best = max(trials, key=lambda r: (r["val_tpr"], -r["val_log_loss"]))
    pd.DataFrame(trials).to_csv(root / "sweep/trials.csv", index=False)
    (root / "sweep/selection.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best["lr"]


def summarize(reports, runs, output):
    from scipy.stats import t
    output = Path(output)
    report = pd.concat(reports, ignore_index=True)
    report.to_csv(output / "per_run.csv", index=False)
    rows = []
    for (experiment, eval_set, slice_name), part in report.groupby(["experiment", "eval_set", "slice"]):
        for metric in METRICS:
            values = part[metric].dropna().to_numpy()
            n = len(values)
            mean = float(values.mean()) if n else float("nan")
            std = float(values.std(ddof=1)) if n > 1 else float("nan")
            margin = float(t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else float("nan")
            rows.append(dict(experiment=experiment, eval_set=eval_set, slice=slice_name, metric=metric,
                             n_seeds=n, mean=mean, std=std, ci95_low=mean - margin, ci95_high=mean + margin))
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "seed_summary.csv", index=False)
    val_rows = [{"experiment": exp, "seed": run["seed"], **selected_metrics(run)} for exp, run in runs]
    val = pd.DataFrame(val_rows).groupby("experiment")[["val_tpr", "val_log_loss"]].mean().reset_index()
    val = val.sort_values(["val_tpr", "val_log_loss", "experiment"], ascending=[False, True, True])
    val.to_csv(output / "validation_selection.csv", index=False)
    selection = {"selected_experiment": val.iloc[0].experiment,
                 "rule": "mean validation TPR at 1% FPR; ties by mean validation log loss",
                 "scope": "Supported-context cohort; no B2 verdict is replaced"}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    headline = summary[(summary.slice == "all") & (summary.metric == "auroc")]
    lines = ["# B4 relevance experiments", "", "All comparisons use the supported-context cohort.",
             "Schema agreement is deferred. Context catalogues are not observed runtime requests.",
             "Seed intervals are 95% t-intervals over independent seeds (not clipped to [0,1]).",
             "Per-run CSVs also contain bootstrap intervals from the shared harness.", "",
             f"Configuration selected on validation only: **{selection['selected_experiment']}**.", "",
             "| Experiment | Eval set | AUROC mean | Seed SD | 95% seed CI |", "|---|---|---:|---:|---|"]
    for row in headline.itertuples():
        lines.append(f"| {row.experiment} | {row.eval_set} | {row.mean:.4f} | {row.std:.4f} | [{row.ci95_low:.4f}, {row.ci95_high:.4f}] |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Fixed comparisons, paired by seed; no selecting the best holdout seed.
    effects = []
    for before, after in (("text_control", "pair_only"), ("pair_only", "relevance"),
                          ("relevance", "relevance_bipia")):
        a, b = report[report.experiment == before], report[report.experiment == after]
        for row in a.merge(b, on=["seed", "eval_set", "slice"], suffixes=("_before", "_after")).itertuples():
            for metric in METRICS:
                effects.append(dict(comparison=f"{before} -> {after}", seed=row.seed, eval_set=row.eval_set,
                                    slice=row.slice, metric=metric,
                                    delta=getattr(row, metric + "_after") - getattr(row, metric + "_before")))
    pd.DataFrame(effects).to_csv(output / "ablation_deltas.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, name in zip(axes, ("test", "mcptox", "msb")):
        part = headline[headline.eval_set == name]
        axis.bar(part.experiment, part["mean"], yerr=part["std"].fillna(0), capsize=3)
        axis.axhline(0.5, linestyle="--", color="gray")
        axis.set_title(name)
        axis.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("AUROC (mean ± seed SD)")
    fig.tight_layout()
    fig.savefig(output / "auroc.png", dpi=160)
    plt.close(fig)
    return summary


def run_experiments(config, root, data, contexts, *, mode="smoke", device=None, tracking_uri=None):
    root = Path(root)
    freeze_manifest(root / "experiment.json", {"config": config, **input_identity(data, contexts),
                                              "code_sha256": code_digest()})
    if mode == "smoke":
        name = f"smoke-s{config['seeds'][0]}"
        run_one(root / "smoke", name, data, contexts,
                {**config["model"], "epochs": config["smoke"]["epochs"], "seed": config["seeds"][0],
                 "input_mode": "relevance"}, ["bipia", "poisonedrag"], device, config["smoke"]["limit"])
        folder = root / "smoke" / name
        score_model(data, contexts, folder / "model", folder / "scores.parquet", val_only=True, device=device)
        return None
    if mode == "sweep":
        return run_sweep(config, root, data, contexts, device)
    if mode != "full":
        raise ValueError("mode must be smoke, sweep or full")
    lr = run_sweep(config, root, data, contexts, device) if config["search"]["enabled"] else config["model"]["lr"]
    runs, reports = [], []
    for exp in config["experiments"]:
        for seed in config["seeds"]:
            name = f"b4-{exp['name']}-s{seed}"
            run = run_one(root / "full", name, data, contexts,
                          {**config["model"], "lr": lr, "seed": seed, "input_mode": exp["input_mode"]},
                          exp["train_sources"], device)
            runs.append((exp["name"], run))
    # Finish the ENTIRE training grid before scoring any holdout.
    for experiment, run in runs:
        folder = root / "full" / run["run_name"]
        if not (folder / "scores.parquet").exists():
            score_model(data, contexts, folder / "model", folder / "scores.parquet", device=device)
            release_memory()
        report = evaluate_scores(data, contexts, folder / "scores.parquet", folder / "report.csv",
                                  **config["evaluation"], tracking_uri=tracking_uri)
        reports.append(report.assign(experiment=experiment, seed=run["seed"]))
    output = root / "reports"
    output.mkdir(exist_ok=True)
    # Refit cheap references on the same supported training/validation cohort.
    from train import fit_detector
    from harness import evaluate
    cohort = harness_data(load_frames(data, contexts, evaluation=True))
    for name in ("length", "tfidf"):
        model, _ = fit_detector(name, cohort)
        _, reference = evaluate(model, cohort, **config["evaluation"])
        reference.to_csv(output / f"reference_{name}.csv", index=False)
    summary = summarize(reports, runs, output)
    print(f"Reports and ablations: {output}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/b4.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["smoke", "sweep", "full"], default="smoke")
    parser.add_argument("--data", default=str(DEFAULT_PATH))
    parser.add_argument("--contexts", default=str(DEFAULT_CONTEXTS))
    parser.add_argument("--device")
    parser.add_argument("--tracking-uri")
    args = parser.parse_args(argv)
    run_experiments(read_config(args.config), args.out, args.data, args.contexts,
                    mode=args.mode, device=args.device, tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
