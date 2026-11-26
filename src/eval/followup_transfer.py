"""Follow-up transfer experiments with B1-Optuna's committed settings (outside the pre-registered
protocol: every one of them trains on MCPTox or MSB rows).

    PYTHONPATH=src/data:src/eval:src/models uv run python src/eval/followup_transfer.py

Writes to report/followup/:
- few_shot.csv: B1-Optuna refit with the rows of k MCPTox servers added to training, tested on the
  remaining servers, over random server draws. Threshold: 1% FPR on in-distribution val benign.
- in_surface.csv: trained and tested inside MCPTox, leave-one-server-out and leave-one-template-out,
  and trained on MSB, tested on MCPTox. Threshold: 1% FPR on each fold's training benign rows.
- few_shot.png: the few-shot curve.
Folds and draws are by server (`subset`), never by row.
"""

import argparse
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd

from dataset import DEFAULT_PATH, load
from followup_length import template_of
from metrics import auroc, length_binned_auroc, rates_at_threshold, threshold_at_fpr

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "report" / "followup"
TARGET_FPR = 0.01
KS = (0, 1, 2, 5, 10, 20)
COLUMNS = ["text", "label", "subset", "attack_family"]

Fit = Callable[[pd.DataFrame], object]


def b1_optuna_fit(train: pd.DataFrame):
    """B1-Optuna's committed best settings, fitted on the given rows."""
    from tfidf_optuna import BEST_PARAMS_PATH, fit_best
    params = json.loads(Path(BEST_PARAMS_PATH).read_text())["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_best(params, SimpleNamespace(train=train))


def score_block(scores, labels, texts, threshold) -> dict:
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    recall, fpr = rates_at_threshold(scores, labels, threshold)
    return dict(auroc=auroc(scores, labels), binned_auroc=length_binned_auroc(scores, labels, list(texts)),
                recall=recall, fpr=fpr, n=len(labels))


def _threshold(model, benign_texts) -> float:
    return threshold_at_fpr(model.predict_proba(benign_texts), TARGET_FPR)


def few_shot(data, mcptox: pd.DataFrame, fit: Fit = b1_optuna_fit, ks=KS, draws=10, seed=7267) -> pd.DataFrame:
    """For each draw and k, train on data.train plus k random MCPTox servers, test on the rest."""
    rng = np.random.default_rng(seed)
    servers = np.array(sorted(mcptox.subset.unique()))
    base = data.train[["text", "label"]]
    val_benign = data.val.text[data.val.label == 0]
    rows, k0 = [], None
    for draw in range(draws):
        order = rng.permutation(servers)
        for k in ks:
            k_used = min(k, len(servers) - 1)
            if k_used == 0:
                if k0 is None:
                    model = fit(base)
                    k0 = score_block(model.predict_proba(mcptox.text), mcptox.label, mcptox.text,
                                     _threshold(model, val_benign))
                result = k0
            else:
                chosen = set(order[:k_used])
                added = mcptox[mcptox.subset.isin(chosen)]
                test = mcptox[~mcptox.subset.isin(chosen)]
                model = fit(pd.concat([base, added[["text", "label", "subset"]]], ignore_index=True))
                result = score_block(model.predict_proba(test.text), test.label, test.text,
                                     _threshold(model, val_benign))
            rows.append(dict(k=k, draw=draw, k_used=k_used, **result))
    return pd.DataFrame(rows)


def _folds(mcptox: pd.DataFrame, by: str, seed: int):
    """(train, test) frames. 'subset': leave one server out. 'template': servers split into three
    seeded groups; fold t tests template t's attacks and the benign rows of group t's servers,
    and trains on the other groups' servers without template t's attacks."""
    if by == "subset":
        for server in sorted(mcptox.subset.unique()):
            yield mcptox[mcptox.subset != server], mcptox[mcptox.subset == server]
        return
    if by != "template":
        raise ValueError(f"by must be 'subset' or 'template', not {by!r}")
    template = template_of(mcptox.attack_family)
    servers = np.random.default_rng(seed).permutation(sorted(mcptox.subset.unique()))
    groups = dict(zip(sorted(template.dropna().unique()), np.array_split(servers, 3)))
    for t, group in groups.items():
        in_group = mcptox.subset.isin(group)
        is_t = (template == t).fillna(False).to_numpy(dtype=bool)  # benign rows have no template
        yield mcptox[~in_group & ~is_t], mcptox[in_group & ((mcptox.label == 0) | is_t)]


def in_surface(mcptox: pd.DataFrame, fit: Fit = b1_optuna_fit, by: str = "subset", seed: int = 7267) -> dict:
    """Out-of-fold scores pooled across folds; recall and FPR pooled from per-fold thresholds."""
    scores, labels, texts, flagged, folds = [], [], [], [], 0
    for train, test in _folds(mcptox, by, seed):
        if test.empty:
            continue
        model = fit(train[["text", "label", "subset"]])
        threshold = _threshold(model, train.text[train.label == 0])
        s = model.predict_proba(test.text)
        scores.append(s); labels.append(test.label.to_numpy()); texts += list(test.text)
        flagged.append(s > threshold)
        folds += 1
    scores, labels, flagged = map(np.concatenate, (scores, labels, flagged))
    return dict(experiment=f"in_surface_by_{by}", auroc=auroc(scores, labels),
                binned_auroc=length_binned_auroc(scores, labels, texts),
                recall=float(flagged[labels == 1].mean()), fpr=float(flagged[labels == 0].mean()),
                n=len(labels), folds=folds)


def msb_to_mcptox(msb: pd.DataFrame, mcptox: pd.DataFrame, fit: Fit = b1_optuna_fit) -> dict:
    """Train on MSB tool descriptions, test on MCPTox; threshold from MSB's (few) benign rows."""
    model = fit(msb[["text", "label"]])
    result = score_block(model.predict_proba(mcptox.text), mcptox.label, mcptox.text,
                         _threshold(model, msb.text[msb.label == 0]))
    return dict(experiment="msb_to_mcptox", folds=1, **result)


def figure(curve: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = curve.groupby("k")[["binned_auroc", "auroc", "recall"]].quantile([0.25, 0.5, 0.75]).unstack()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for metric, name, color in (("auroc", "AUROC", "#2b6cb0"), ("binned_auroc", "length-binned AUROC", "#dd6b20"),
                                ("recall", "recall at 1% val FPR", "#4a5568")):
        ax.plot(stats.index, stats[(metric, 0.5)], marker="o", color=color, label=name)
        ax.fill_between(stats.index, stats[(metric, 0.25)], stats[(metric, 0.75)], color=color, alpha=0.15)
    ax.axhline(0.5, color="#a0aec0", linestyle="--", linewidth=1)
    ax.set(xscale="symlog", xlabel="MCPTox servers added to training (k)", ylabel="held-out MCPTox servers", ylim=(0, 1))
    ax.set_xticks(sorted(curve.k.unique()), [str(k) for k in sorted(curve.k.unique())])
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    from harness import load_data

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7267)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data)
    mcptox = load("eval", sources=["mcptox"], path=args.data).select(COLUMNS).to_pandas()
    msb = load("eval", sources=["msb"], path=args.data).select(COLUMNS).to_pandas()

    curve = few_shot(data, mcptox, draws=args.draws, seed=args.seed)
    curve.to_csv(out / "few_shot.csv", index=False, float_format="%.3f")
    figure(curve, out / "few_shot.png")
    print(curve.groupby("k")[["k_used", "auroc", "binned_auroc", "recall", "fpr", "n"]].median().to_string(float_format="%.3f"))

    table = pd.DataFrame([in_surface(mcptox, by="subset", seed=args.seed),
                          in_surface(mcptox, by="template", seed=args.seed),
                          msb_to_mcptox(msb, mcptox)])
    table.to_csv(out / "in_surface.csv", index=False, float_format="%.3f")
    print(table.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
