"""B1-Optuna: search TF-IDF features and regularization by mean held-out AUROC over family folds
of train_pool train, then refit the best settings on train and evaluate once with the harness.

    PYTHONPATH=src/data:src/eval:src/models uv run python src/models/tfidf_optuna.py

Spec: docs/superpowers/specs/2026-11-18-b1-optuna-design.md. Trials only ever see train_pool
train rows; val sets the deployed threshold, and test, MCPTox and MSB are scored once at the end.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from dataset import DEFAULT_PATH, HOLDOUT_ROLES, HoldoutError, load
from family_cv import family_folds
from metrics import auroc
from tfidf import TfidfDetector

N_TRIALS = 100
SAMPLER_SEED = 7267
FOLDS = 5
FOLD_SEED = 7267
CHAR_RANGES = {"2-4": (2, 4), "3-5": (3, 5), "3-6": (3, 6)}
BEST_PARAMS_PATH = Path(__file__).resolve().parents[2] / "report/phase4/b1_optuna/best_params.json"


class TunedTfidfDetector(TfidfDetector):
    """TfidfDetector with its feature settings exposed. The defaults are B1's exact pipeline."""

    name = "tfidf-optuna"

    def __init__(self, C=1.0, features="both", word_max_n=2, char_analyzer="char_wb",
                 char_range=(3, 5), min_df=2, max_features=None, sublinear_tf=True,
                 class_weight=None):
        if features not in ("word", "char", "both"):
            raise ValueError(f"features must be word, char or both, not {features!r}")
        self.C = C
        common = {"min_df": min_df, "max_features": max_features, "sublinear_tf": sublinear_tf}
        parts = []
        if features in ("word", "both"):
            parts.append(("word", TfidfVectorizer(ngram_range=(1, word_max_n), **common)))
        if features in ("char", "both"):
            parts.append(("char", TfidfVectorizer(analyzer=char_analyzer, ngram_range=tuple(char_range),
                                                  **common)))
        self.pipeline = make_pipeline(
            FeatureUnion(parts),
            LogisticRegression(C=C, max_iter=2000, class_weight=class_weight),
        )


def suggest(trial) -> dict:
    """One point of the search space (spec decision 6), as TunedTfidfDetector arguments."""
    params = {"C": trial.suggest_float("C", 1e-2, 1e3, log=True)}
    features = trial.suggest_categorical("features", ["word", "char", "both"])
    params["features"] = features
    if features in ("word", "both"):
        params["word_max_n"] = trial.suggest_int("word_max_n", 1, 3)
    if features in ("char", "both"):
        params["char_analyzer"] = trial.suggest_categorical("char_analyzer", ["char", "char_wb"])
        params["char_range"] = CHAR_RANGES[trial.suggest_categorical("char_range", list(CHAR_RANGES))]
    params["min_df"] = trial.suggest_categorical("min_df", [1, 2, 5])
    params["max_features"] = trial.suggest_categorical("max_features", [None, 50_000, 200_000])
    params["sublinear_tf"] = trial.suggest_categorical("sublinear_tf", [True, False])
    params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
    return params


def detector_params(optuna_params: dict) -> dict:
    """A finished trial's params (as Optuna stores them) as TunedTfidfDetector arguments."""
    params = dict(optuna_params)
    if "char_range" in params:
        params["char_range"] = CHAR_RANGES[params["char_range"]]
    return params


def check_fit_rows(frame):
    """Trials may only read train_pool train: val is kept for the threshold, and test and the
    holdouts are scored once at the end."""
    bad = sorted(set(frame.role) & HOLDOUT_ROLES) + sorted(set(frame.split) - {"train"})
    if bad:
        raise HoldoutError(f"the study may only read train_pool train rows, not {', '.join(bad)}")


def objective(trial, texts, labels, folds):
    """Mean held-out AUROC over the folds. Reports the running mean after each fold so the
    pruner can stop a weak trial early; fold order is fixed, so running means are comparable."""
    params = suggest(trial)
    scores = []
    for f in range(folds.max() + 1):
        held = folds == f
        model = TunedTfidfDetector(**params).fit(texts[~held], labels[~held])
        scores.append(auroc(model.predict_proba(texts[held]), labels[held]))
        trial.set_user_attr(f"auroc_fold{f}", scores[-1])
        trial.report(float(np.mean(scores)), step=f)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))


def progress_line(study, trial, n_trials) -> str:
    """One line per finished trial: its state and score, the best so far, and its duration."""
    width = len(str(n_trials))
    if trial.state == optuna.trial.TrialState.COMPLETE:
        result = f"COMPLETE {trial.value:.4f}"
    elif trial.state == optuna.trial.TrialState.PRUNED:
        folds_run = sum(key.startswith("auroc_fold") for key in trial.user_attrs)
        result = f"PRUNED after {folds_run} fold{'s' if folds_run != 1 else ''}"
    else:
        result = trial.state.name
    try:
        best = f"best {study.best_value:.4f} (#{study.best_trial.number})"
    except ValueError:  # no trial has completed yet
        best = "best -"
    seconds = trial.duration.total_seconds() if trial.duration else float("nan")
    return f"trial {trial.number + 1:>{width}}/{n_trials}  {result:<22} {best}  {seconds:5.1f}s"


def run_study(frame, n_trials=N_TRIALS, seed=SAMPLER_SEED, folds=None, progress=True):
    """The study over `frame` (train_pool train only), with folds from family_folds. With
    progress, prints one line per finished trial."""
    check_fit_rows(frame)
    folds = family_folds(frame, k=FOLDS, seed=FOLD_SEED) if folds is None else folds
    texts, labels = frame.text.to_numpy(), frame.label.to_numpy()
    study = optuna.create_study(
        study_name="b1-tfidf-optuna", direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        callbacks = [lambda s, t: print(progress_line(s, t, n_trials), flush=True)] if progress else []
        study.optimize(lambda t: objective(t, texts, labels, folds), n_trials=n_trials,
                       callbacks=callbacks)
    return study, folds


ALWAYS_SAMPLED_NONE = ("max_features", "class_weight")  # sampled every trial; None is a value


def trials_table(trials) -> pd.DataFrame:
    """Optuna's trials_dataframe (in memory, or read back from trials.csv) as one row per trial:
    number, state, value, duration_s, the parameters without the params_ prefix and the fold
    AUROCs. None is written as "none" for parameters sampled in every trial, so it cannot be
    confused with a parameter the trial did not use (left empty)."""
    table = pd.DataFrame({"number": trials["number"], "state": trials["state"], "value": trials["value"]})
    table["duration_s"] = pd.to_timedelta(trials["duration"]).dt.total_seconds()
    for column in sorted(c for c in trials.columns if c.startswith("params_")):
        name = column.removeprefix("params_")
        values = trials[column]
        if name in ALWAYS_SAMPLED_NONE:
            values = values.astype(object).where(values.notna(), "none")
        if name in ("max_features", "min_df", "word_max_n"):
            values = values.where(values.isin(["none"]), pd.to_numeric(values, errors="coerce")).astype(object)
            values = values.map(lambda v: int(v) if isinstance(v, float) and v == v else v)
        table[name] = values
    for column in sorted(c for c in trials.columns if c.startswith("user_attrs_auroc_fold")):
        table[column.removeprefix("user_attrs_")] = trials[column]
    return table


def trial_params(row) -> dict:
    """One trials_table row's parameters as logged: unused ones dropped, "none" kept."""
    skip = {"number", "state", "value", "duration_s"}
    return {k: v for k, v in row.items()
            if k not in skip and not k.startswith("auroc_fold") and not (isinstance(v, float) and v != v)}


def save_study(study, frame, folds, out_dir):
    """best_params.json, trials.csv, folds.csv and the history and importance plots."""
    import matplotlib

    matplotlib.use("Agg")
    from optuna.visualization import matplotlib as plots

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = study.best_trial
    (out_dir / "best_params.json").write_text(json.dumps(
        {"trial": best.number, "mean_cv_auroc": best.value, "params": best.params,
         "fold_auroc": {k: v for k, v in best.user_attrs.items() if k.startswith("auroc_fold")},
         "n_trials": len(study.trials), "sampler_seed": SAMPLER_SEED, "folds": FOLDS,
         "fold_seed": FOLD_SEED}, indent=2))
    trials_table(study.trials_dataframe()).to_csv(out_dir / "trials.csv", index=False)
    (frame.assign(fold=folds).groupby(["fold", "source", "label"]).size()
     .rename("rows").reset_index().to_csv(out_dir / "folds.csv", index=False))
    for name, draw in (("history", plots.plot_optimization_history),
                       ("importance", plots.plot_param_importances)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                draw(study).figure.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        except (ValueError, RuntimeError) as exc:  # importance needs enough finished trials
            print(f"skipped {name}.png: {exc}")


def log_study(uri, trials, params, threshold, report, args):
    """One tfidf-optuna run like every other detector's, with one nested run per trial.
    `trials` is a trials_table."""
    import mlflow

    from train import EXPERIMENT, metric_items

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=TunedTfidfDetector.name):
        mlflow.log_params({"detector": TunedTfidfDetector.name, "target_fpr": args.target_fpr,
                           "threshold": threshold, "n_boot": args.n_boot, **params})
        mlflow.log_metrics(metric_items(report))
        mlflow.log_text(report.to_csv(index=False), "report.csv")
        mlflow.log_artifacts(args.out_dir, "study")
        for row in trials.to_dict("records"):
            with mlflow.start_run(run_name=f"trial-{row['number']}", nested=True):
                mlflow.log_params({**trial_params(row), "state": row["state"]})
                if row["value"] == row["value"]:  # not NaN
                    mlflow.log_metric("mean_cv_auroc", row["value"])


def run_params(best: dict, n_trials: int) -> dict:
    """The tfidf-optuna run's params, from best_params.json's fields."""
    return {**best["params"], "best_trial": best["trial"], "mean_cv_auroc": best["mean_cv_auroc"],
            "n_trials": n_trials, "sampler_seed": SAMPLER_SEED, "folds": FOLDS, "fold_seed": FOLD_SEED}


def fit_best(params: dict, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return TunedTfidfDetector(**detector_params(params)).fit(data.train.text, data.train.label)


def log_or_explain(args, trials, params, threshold, report):
    """Log to MLflow if asked. A failed upload never loses a finished study: the files are
    already written, and --log-only uploads them later without re-running it."""
    if not args.tracking_uri:
        print("not logged to MLflow: pass --tracking-uri or set MLFLOW_TRACKING_URI")
        return
    try:
        log_study(args.tracking_uri, trials, params, threshold, report, args)
    except Exception as exc:  # noqa: BLE001 - any upload failure gets the same advice
        raise SystemExit(f"MLflow logging failed ({type(exc).__name__}: {exc}).\nThe study and report "
                         f"are saved; upload them later with: make b1-optuna ARGS=--log-only") from exc
    print(f"logged to MLflow at {args.tracking_uri}")


def log_only(args):
    """Upload a finished study's saved files. The threshold is recomputed by refitting the best
    settings on train (seconds, deterministic) instead of re-running the study."""
    from harness import load_data
    from metrics import threshold_at_fpr

    out_dir = Path(args.out_dir)
    best = json.loads((out_dir / "best_params.json").read_text())
    trials = pd.read_csv(out_dir / "trials.csv")
    if "params_C" in trials.columns:  # written before trials_table existed
        trials = trials_table(trials)
        trials.to_csv(out_dir / "trials.csv", index=False)
    report = pd.read_csv(args.out)
    data = load_data(args.data)
    detector = fit_best(best["params"], data)
    val_scores = detector.predict_proba(data.val.text)
    threshold = threshold_at_fpr(val_scores[data.val.label.to_numpy() == 0], args.target_fpr)
    print(f"{detector.name}: threshold {threshold:.6g} at {args.target_fpr:.0%} FPR on val (refit)")
    log_or_explain(args, trials, run_params(best, len(trials)), threshold, report)


def main(argv=None):
    from harness import evaluate, load_data
    from train import SHOWN, compare, fit_detector

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=str(DEFAULT_PATH))
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out-dir", default="report/phase4/b1_optuna")
    parser.add_argument("--out", default="report/baselines/b1_tfidf_lr_optuna.csv")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--log-only", action="store_true",
                        help="Upload a finished study's saved files to MLflow; do not re-run it")
    args = parser.parse_args(argv)
    if args.log_only:
        return log_only(args)

    frame = load("fit", path=args.data, roles=["train_pool"], splits=["train"]).to_pandas()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study, folds = run_study(frame, n_trials=args.n_trials)
    save_study(study, frame, folds, args.out_dir)
    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    print(f"study: {len(study.trials)} trials {states}; best #{study.best_trial.number} "
          f"mean CV AUROC {study.best_value:.4f} {study.best_params}")

    # Refit the best settings on all of train; the harness sets the threshold on val.
    data = load_data(args.data)
    detector = fit_best(study.best_params, data)
    threshold, report = evaluate(detector, data, args.target_fpr, args.n_boot)
    best = {"params": study.best_params, "trial": study.best_trial.number,
            "mean_cv_auroc": study.best_value}
    params = run_params(best, len(study.trials))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    _, reference = evaluate(fit_detector("length", data)[0], data, args.target_fpr, args.n_boot)
    print(f"{detector.name}: threshold {threshold:.6g} at {args.target_fpr:.0%} FPR on val; wrote {args.out}")
    print(compare(report, reference)[SHOWN].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log_or_explain(args, trials_table(study.trials_dataframe()), params, threshold, report)
    return study, report


if __name__ == "__main__":
    main()
