"""Follow-up length analyses on the frozen Phase 5 scores (outside the pre-registered protocol).

    PYTHONPATH=src/data:src/eval:src/models uv run python src/eval/followup_length.py

Reads report/phase5/scores.parquet and thresholds.json and writes to report/followup/:
- b2_length.csv: Spearman correlation of each B2 seed's score with text length, per eval set and
  source, on all rows and on benign rows only; plus B2's MCPTox AUROC when benign rows shorter
  than the shortest attack are dropped (`long_benign` rows).
- templates.csv: MCPTox attacks by MCPTox template, each scored against every MCPTox benign row.
- paired.json: paired server bootstrap of B1-Optuna minus length on MCPTox, and the rank average.
- b2_length.png: B2 score against length on MCPTox, by label.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from metrics import auroc

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "report" / "phase5" / "scores.parquet"
THRESHOLDS = ROOT / "report" / "phase5" / "thresholds.json"
OUT_DIR = ROOT / "report" / "followup"
B2 = ("b2_row2_s7267", "b2_row2_s7268", "b2_row2_s7269")
FIGURE_SEED = "b2_row2_s7268"


def template_of(attack_family: pd.Series) -> pd.Series:
    """'Template-1'..'Template-3' from an MCPTox attack family name; NaN for benign rows."""
    return attack_family.astype("string").str.extract(r"(Template-\d+)")[0]


def b2_length_spearman(scores: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each B2 seed with n_chars, by eval set and source, on all rows and
    on benign rows. Ties (most MCPTox scores are near zero) get average ranks."""
    rows = []
    for (eval_set, source), part in scores.groupby(["eval_set", "source"]):
        for subset, sub in (("all", part), ("benign", part[part.label == 0])):
            for detector in (c for c in B2 if c in scores):
                if len(sub) > 2 and sub[detector].nunique() > 1:
                    rho = float(spearmanr(sub[detector], sub.n_chars).statistic)
                    rows.append(dict(detector=detector, eval_set=eval_set, source=source,
                                     rows=subset, n=len(sub), rho=rho))
    return pd.DataFrame(rows)


def long_benign_auroc(scores: pd.DataFrame, column: str) -> float:
    """MCPTox AUROC after dropping benign rows shorter than the shortest attack."""
    mcp = scores[scores.eval_set == "mcptox"]
    shortest = mcp.loc[mcp.label == 1, "n_chars"].min()
    kept = mcp[(mcp.label == 1) | (mcp.n_chars >= shortest)]
    return auroc(kept[column].to_numpy(), kept.label.to_numpy())


def per_template(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Each MCPTox template's attacks against all MCPTox benign rows: length and B1-Optuna AUROC,
    and B1-Optuna recall at the given (deployed) threshold."""
    mcp = scores[scores.eval_set == "mcptox"].assign(template=lambda f: template_of(f.attack_family))
    benign = mcp[mcp.label == 0]
    rows = []
    for template, attacks in mcp[mcp.label == 1].groupby("template"):
        both = pd.concat([attacks, benign])
        rows.append(dict(
            template=template, attacks=len(attacks),
            length_auroc=auroc(both.length.to_numpy(), both.label.to_numpy()),
            b1_optuna_auroc=auroc(both.b1_optuna.to_numpy(), both.label.to_numpy()),
            b1_optuna_recall=float((attacks.b1_optuna >= threshold).mean()),
        ))
    return pd.DataFrame(rows)


def rank_average(*columns: pd.Series) -> np.ndarray:
    """Mean of the columns' ranks: a parameter-free combination of detectors."""
    return np.mean([rankdata(np.asarray(c)) for c in columns], axis=0)


def paired_server_bootstrap(frame: pd.DataFrame, a: str, b: str, group: str = "subset",
                            n_boot: int = 1000, seed: int = 7267) -> dict:
    """AUROC(a) - AUROC(b), with a 95% interval from resampling servers with replacement. A
    resample holding one label is redrawn."""
    diff = auroc(frame[a].to_numpy(), frame.label.to_numpy()) - auroc(frame[b].to_numpy(), frame.label.to_numpy())
    rng = np.random.default_rng(seed)
    parts = [g for _, g in frame.groupby(group)]
    diffs = []
    while len(diffs) < n_boot:
        sample = pd.concat([parts[i] for i in rng.integers(0, len(parts), len(parts))])
        if sample.label.nunique() < 2:
            continue
        labels = sample.label.to_numpy()
        diffs.append(auroc(sample[a].to_numpy(), labels) - auroc(sample[b].to_numpy(), labels))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(diff), "lo": float(lo), "hi": float(hi)}


def figure(scores: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mcp = scores[scores.eval_set == "mcptox"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for label, name, color, marker in ((0, "benign", "#2b6cb0", "o"), (1, "attack", "#dd6b20", "^")):
        part = mcp[mcp.label == label]
        ax.scatter(part.n_chars, part[FIGURE_SEED].clip(lower=1e-6), s=10, alpha=0.6,
                   color=color, marker=marker, label=name)
    ax.set(xscale="log", yscale="log", xlabel="characters", ylabel="B2 score (seed 7268)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scores", default=str(SCORES))
    ap.add_argument("--thresholds", default=str(THRESHOLDS))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    scores = pd.read_parquet(args.scores)
    thresholds = json.loads(Path(args.thresholds).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spearman = b2_length_spearman(scores)
    long_rows = [dict(detector=d, eval_set="mcptox", source="mcptox", rows="long_benign",
                      n=int(((scores.eval_set == "mcptox") & (scores.label == 0)
                             & (scores.n_chars >= scores[(scores.eval_set == "mcptox") & (scores.label == 1)].n_chars.min())).sum()),
                      auroc=long_benign_auroc(scores, d)) for d in B2]
    b2 = pd.concat([spearman, pd.DataFrame(long_rows)], ignore_index=True)
    b2.to_csv(out / "b2_length.csv", index=False, float_format="%.3f")

    templates = per_template(scores, thresholds["b1_optuna"])
    templates.to_csv(out / "templates.csv", index=False, float_format="%.3f")

    mcp = scores[scores.eval_set == "mcptox"].copy()
    mcp["rank_avg"] = rank_average(mcp.length, mcp.b1_optuna)
    paired = {
        "b1_optuna_minus_length": paired_server_bootstrap(mcp, "b1_optuna", "length"),
        "rank_avg_minus_length": paired_server_bootstrap(mcp, "rank_avg", "length"),
        "rank_avg_auroc": auroc(mcp.rank_avg.to_numpy(), mcp.label.to_numpy()),
        "servers": int(mcp.subset.nunique()),
    }
    (out / "paired.json").write_text(json.dumps(paired, indent=2) + "\n")
    figure(scores, out / "b2_length.png")

    print(b2.to_string(index=False, float_format="%.3f"))
    print(templates.to_string(index=False, float_format="%.3f"))
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
