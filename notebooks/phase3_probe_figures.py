"""Figures 08-10 for the Phase 3 visual-data section: shortcut probes with confidence intervals.

08 ROC of the length cue (fitted on train vs fit inside each set), with TPR@5%FPR
09 AUROC of every trivial cue with 95% group-bootstrap CIs
10 Spearman(log length, label) per source with 95% CIs

Called from phase3_eda.py, so one script writes every figure. Uses src/eval/shortcut_probe.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))

from metrics import MATCHED_FPR  # noqa: E402
from shortcut_probe import CueProbe, bootstrap_ci, out_of_fold, probe_battery, tpr_at_fpr  # noqa: E402

PROBE_FITTED = "#4C78A8"
PROBE_ORACLE = "#F58518"
NEUTRAL = "#4C566A"
FPR_BUDGET = MATCHED_FPR  # 5%, the team's operating point (Nov 9 sync)

# Set by make_probe_figures(); phase3_eda passes its own saver so figures land in the run folder.
_save = None


def probe_frames(df: pd.DataFrame):
    """Training rows and the five evaluation sets used by the shortcut figures.

    Training rows are train_pool / train only, the same rows ``dataset.load("fit", ...)`` allows.
    """
    train = df[(df["role"] == "train_pool") & (df["split"] == "train")].reset_index(drop=True)
    assert not train["role"].isin(["loso_holdout", "transfer_holdout"]).any()
    test = df[(df["role"] == "train_pool") & (df["split"] == "test")].reset_index(drop=True)
    sets = {
        "in-dist test": test,
        "BIPIA": test[test["source"] == "bipia"].reset_index(drop=True),
        "PoisonedRAG": test[test["source"] == "poisonedrag"].reset_index(drop=True),
        "LOSO (MCPTox)": df[df["source"] == "mcptox"].reset_index(drop=True),
        "transfer (MSB)": df[df["source"] == "msb"].reset_index(drop=True),
    }
    return train, sets


def plot_roc_length(train: pd.DataFrame, sets: dict) -> pd.DataFrame:
    """ROC of the length cue, fitted on train versus fit inside each set (oracle)."""
    probe = CueProbe("length_only", ["log_length"]).fit(train["text"], train["label"])
    fig, axes = plt.subplots(1, len(sets), figsize=(3.6 * len(sets), 3.8), sharey=True)
    rows = []
    for ax, (name, part) in zip(axes, sets.items()):
        fitted = probe.predict_proba(part["text"])
        folds = max(2, min(5, part["group"].nunique()))
        oracle = out_of_fold(CueProbe("length_only", ["log_length"]), part, folds)
        tprs = []
        for label, scores, color in (("fitted on train", fitted, PROBE_FITTED), ("oracle (fit inside set)", oracle, PROBE_ORACLE)):
            fpr, tpr, _ = roc_curve(part["label"], scores)
            ax.plot(fpr, tpr, color=color, label=label)
            tprs.append(tpr_at_fpr(part["label"], scores, FPR_BUDGET))
            rows.append({"eval_set": name, "mode": label, "tpr": tprs[-1], "n_benign": int((part["label"] == 0).sum())})
        ax.axvline(FPR_BUDGET, color="grey", linestyle=":", linewidth=1)
        ax.plot([0, 1], [0, 1], color="lightgrey", linewidth=1)
        ax.set_title(f"{name}\n{tprs[0]:.0%} / {tprs[1]:.0%} TPR at 5% FPR", fontsize=10)
        ax.set_xlabel("false-positive rate")
    axes[0].set_ylabel("true-positive rate")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("ROC of the length cue alone (panel titles: TPR at 5% FPR, fitted / oracle)", y=1.03)
    fig.tight_layout()
    _save(fig, "08_roc_length_probe.png")
    return pd.DataFrame(rows)


def plot_probe_forest(train: pd.DataFrame, sets: dict) -> pd.DataFrame:
    """AUROC of each trivial cue with a 95% group-bootstrap CI, fitted versus oracle."""
    table = probe_battery(train, sets, fpr=FPR_BUDGET)
    cues = table[table["family"] == "cue"]
    probes = list(cues["probe"].unique())
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(sets) * len(probes) + 1.5))
    y, ticks, labels = 0, [], []
    for name in sets:
        for probe in probes:
            for mode, color, offset in (("fitted", PROBE_FITTED, 0.12), ("oracle", PROBE_ORACLE, -0.12)):
                r = cues[(cues["eval_set"] == name) & (cues["probe"] == probe) & (cues["mode"] == mode)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(
                    r["auroc"], y + offset, xerr=[[r["auroc"] - r["auroc_lo"]], [r["auroc_hi"] - r["auroc"]]],
                    fmt="o", color=color, capsize=3,
                    label=mode if (name, probe) == (list(sets)[0], probes[0]) else None,
                )
            ticks.append(y)
            labels.append(f"{name} · {probe}")
            y += 1
    ax.axvline(0.5, color="lightgrey", linestyle="--")
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC with 95% CI (resampling whole groups)")
    ax.set_title("What trivial cues earn. A detector's CI-lower must clear the fitted CI-upper by 0.05")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
    fig.tight_layout()
    _save(fig, "09_auroc_ci_forest.png")
    return table


def plot_spearman_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman(log length, label) per source with a 95% group-bootstrap CI."""
    rows = []
    for source, part in df.groupby("source"):
        if part["label"].nunique() < 2:
            continue
        stat = lambda labels, scores: spearmanr(scores, labels).statistic  # noqa: E731
        rho, lo, hi = bootstrap_ci(stat, part["label"].to_numpy(), part["log_char_len"].to_numpy(), part["group"].to_numpy(), n_boot=500)
        rows.append({"source": source, "rho": rho, "lo": lo, "hi": hi, "n": len(part)})
    out = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.errorbar(out["rho"], range(len(out)), xerr=[out["rho"] - out["lo"], out["hi"] - out["rho"]], fmt="o", color=NEUTRAL, capsize=3)
    ax.axvline(0, color="lightgrey", linestyle="--")
    ax.set_yticks(range(len(out)), [f"{s} (n={n:,})" for s, n in zip(out["source"], out["n"])])
    ax.set_xlabel("Spearman correlation of log length with the label, 95% CI")
    ax.set_title("Length points opposite ways by source")
    fig.tight_layout()
    _save(fig, "10_spearman_ci.png")
    return out


def probe_markdown(roc: pd.DataFrame, forest: pd.DataFrame, spearman: pd.DataFrame) -> str:
    """The write-up section for figures 08-10, built from the numbers just computed."""
    pct = lambda v: f"{v:.1%}"  # noqa: E731
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    roc_rows = [[r.eval_set, r.mode, pct(r.tpr), f"{r.n_benign:,}", "yes" if r.n_benign * FPR_BUDGET >= 1 else "no"] for r in roc.itertuples()]
    cues = forest[forest["family"] == "cue"]
    cue_rows = [
        [r.probe, r.eval_set, r.mode, f"{r.n:,}", f"{r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]", pct(r.tpr_at_fpr)]
        for r in cues.rename(columns={"tpr@fpr": "tpr_at_fpr"}).itertuples()
    ]
    sp_rows = [[r.source, f"{r.n:,}", f"{r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]"] for r in spearman.itertuples()]
    return f"""## Length shortcut with confidence intervals and TPR@FPR

TPR@5%FPR is the share of adversarial texts caught when at most 5% of benign texts are wrongly flagged. It is the operating point that matters for a filter that runs on every retrieved passage or tool result. Measuring it takes at least 20 benign rows; MSB has 21, so its budget allows a single false positive and the number is very noisy. The last column marks this. "Oracle" means the cue was fit inside that evaluation set (grouped 5-fold), which is the most the cue could earn there whatever direction the training pool taught it. All intervals are 95% and resample whole groups.

![ROC of the length cue](figures/08_roc_length_probe.png)

{_md_table(["Eval set", "Fit", "TPR at 5% FPR", "Benign rows", "5% FPR measurable"], roc_rows)}

![AUROC with confidence intervals](figures/09_auroc_ci_forest.png)

{_md_table(["Cue", "Eval set", "Fit", "Rows", "AUROC [95% CI]", "TPR at 5% FPR"], cue_rows)}

![Spearman with confidence intervals](figures/10_spearman_ci.png)

{_md_table(["Source", "Rows", "Spearman(log length, label) [95% CI]"], sp_rows)}
"""


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])




def make_probe_figures(df: pd.DataFrame, save) -> str:
    """Draw figures 08-10 with phase3_eda's saver and return the write-up section."""
    global _save
    _save = save
    train, sets = probe_frames(df)
    roc = plot_roc_length(train, sets)
    forest = plot_probe_forest(train, sets)
    spearman = plot_spearman_ci(df)
    return probe_markdown(roc, forest, spearman)
