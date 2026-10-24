"""Phase 3 visual profile of the unified injection dataset.

The course checklist (distributions, correlation, missingness, outliers, target,
one non-obvious insight) is written for a table of numeric columns. This corpus
is text plus categories, so the plots use the 12 stored columns and a small set
of text statistics computed from ``text``. Nothing here is a model feature the
detector will train on; ``attack_kind``, ``attack_objective`` and ``attack_family``
are null on every benign row and must stay out of the model.

Run from the repository root, after ``data/processed/dataset.parquet`` exists:

    uv run python notebooks/phase3_eda.py

Figures land in ``notebooks/figures/``. The write-up is
``notebooks/PHASE3_VISUAL_DATA.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))

from dataset import load  # noqa: E402

FIG_DIR = ROOT / "notebooks" / "figures"
WRITEUP = ROOT / "notebooks" / "PHASE3_VISUAL_DATA.md"

BENIGN = "#4C78A8"
ADVERSARIAL = "#E45756"
NEUTRAL = "#4C566A"
EXPECTED_ROWS = 11_005

CATEGORICAL = [
    "source",
    "subset",
    "attack_surface",
    "role",
    "split",
    "attack_kind",
    "attack_objective",
    "attack_family",
]
ATTACK_FIELDS = ("attack_kind", "attack_objective", "attack_family")
NUMERIC = [
    "log_char_len",
    "word_count",
    "avg_word_len",
    "punct_rate",
    "upper_rate",
    "digit_rate",
]
NUMERIC_LABELS = {
    "log_char_len": "log character length",
    "word_count": "word count",
    "avg_word_len": "average word length",
    "punct_rate": "punctuation rate",
    "upper_rate": "uppercase rate",
    "digit_rate": "digit rate",
}


def load_frame() -> pd.DataFrame:
    df = load("eval").to_pandas()
    if len(df) != EXPECTED_ROWS:
        raise SystemExit(
            f"expected {EXPECTED_ROWS} rows (the Phase 2 build); found {len(df)}. "
            "Refusing to draw figures from a different table."
        )
    text = df["text"].fillna("")
    words = text.str.split()
    word_count = words.str.len()
    df["char_len"] = text.str.len()
    df["log_char_len"] = np.log1p(df["char_len"])
    df["word_count"] = word_count
    df["avg_word_len"] = [
        (sum(len(w) for w in ws) / len(ws)) if ws else 0.0 for ws in words
    ]
    letters = text.str.len().replace(0, np.nan)
    df["punct_rate"] = text.str.count(r"[^\w\s]") / letters
    df["upper_rate"] = text.str.count(r"[A-Z]") / letters
    df["digit_rate"] = text.str.count(r"\d") / letters
    df["label_name"] = df["label"].map({0: "benign", 1: "adversarial"})
    return df


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 12,
        }
    )


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _counts(series: pd.Series) -> pd.Series:
    return series.fillna("(null)").astype(str).value_counts()


def plot_categorical(df: pd.DataFrame) -> None:
    fields = ["source", "attack_surface", "role", "split", "attack_kind", "attack_objective"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, column in zip(axes.ravel(), fields):
        counts = _counts(df[column]).sort_values()
        ax.barh(counts.index, counts.values, color=NEUTRAL)
        ax.set_title(column)
        ax.set_xlabel("rows")
    fig.suptitle("How often each categorical column occurs", y=1.02)
    fig.tight_layout()
    _save(fig, "01_categorical_distributions.png")

    # subset has 60 values: 8 large collections and 52 small MCP tool servers.
    large = _counts(df["subset"])
    large = large[large >= 100].sort_values()
    small_rows = int((~df["subset"].isin(large.index)).sum())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(large.index, large.values, color=NEUTRAL)
    ax.set_xlabel("rows")
    ax.set_title(f"Subsets with at least 100 rows. The other {small_rows:,} rows are spread across MCP tool servers.")
    fig.tight_layout()
    _save(fig, "01b_subset.png")

    family = _counts(df.loc[df["label"].eq(1), "attack_family"])
    top = family.head(15).sort_values()
    other = int(family.iloc[15:].sum())
    top = pd.concat([pd.Series({f"other ({len(family) - 15} families)": other}), top])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top.index, top.values, color=ADVERSARIAL)
    ax.set_xlabel("adversarial rows")
    ax.set_title("Attack family, adversarial rows only")
    fig.tight_layout()
    _save(fig, "01c_attack_family.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, column in zip(axes.ravel(), NUMERIC):
        for label, color in (("benign", BENIGN), ("adversarial", ADVERSARIAL)):
            part = df.loc[df["label_name"].eq(label), column].dropna()
            ax.hist(part, bins=40, density=True, alpha=0.55, color=color, label=label)
        ax.set_title(NUMERIC_LABELS[column])
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Distribution of text statistics, pooled across sources", y=1.02)
    fig.tight_layout()
    _save(fig, "02_numeric_distributions.png")


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    rows, cols = table.shape
    return float(np.sqrt(chi2 / (n * (min(rows, cols) - 1))))


def plot_correlation(df: pd.DataFrame) -> dict:
    stats = df[NUMERIC]
    matrix = stats.corr(method="spearman")
    with_label = {
        column: float(spearmanr(df[column], df["label"], nan_policy="omit").statistic)
        for column in NUMERIC
    }
    by_source = pd.DataFrame(
        {
            source: {
                column: float(spearmanr(part[column], part["label"], nan_policy="omit").statistic)
                for column in NUMERIC
            }
            for source, part in df.groupby("source")
            if part["label"].nunique() == 2
        }
    ).T

    filled = df.copy()
    for column in ATTACK_FIELDS:
        filled[column] = filled[column].fillna("(null on benign rows)")
    cats = ["source", "subset", "attack_surface", "role", "split", *ATTACK_FIELDS, "label_name"]
    cramer = pd.DataFrame(index=cats, columns=cats, dtype=float)
    for left in cats:
        for right in cats:
            cramer.loc[left, right] = _cramers_v(filled[left].astype(str), filled[right].astype(str))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, grid, title in (
        (axes[0], matrix, "Spearman correlation of text statistics"),
        (axes[1], by_source, "Spearman correlation with the label, by source"),
    ):
        image = ax.imshow(grid.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(grid.shape[1]), [NUMERIC_LABELS.get(c, c) for c in grid.columns], rotation=40, ha="right")
        ax.set_yticks(range(grid.shape[0]), [NUMERIC_LABELS.get(c, c) for c in grid.index])
        ax.set_title(title)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid.to_numpy(dtype=float)[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, "03_correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(cramer.to_numpy(dtype=float), cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(cats)), cats, rotation=40, ha="right")
    ax.set_yticks(range(len(cats)), cats)
    ax.set_title("Cramér's V between categorical columns")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, "03b_categorical_association.png")
    return {"with_label": with_label, "by_source": by_source}


def plot_missing(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        if column in NUMERIC or column in {"char_len", "label_name"}:
            continue
        null = df[column].isna()
        if column == "text":
            null = null | df[column].fillna("").str.strip().eq("")
        rows.append(
            {
                "column": column,
                "nulls": int(null.sum()),
                "null_share": float(null.mean()),
                "nulls_on_benign": int(null[df["label"].eq(0)].sum()),
                "nulls_on_adversarial": int(null[df["label"].eq(1)].sum()),
            }
        )
    missing = pd.DataFrame(rows).sort_values("nulls")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [ADVERSARIAL if n else NEUTRAL for n in missing["nulls"]]
    ax.barh(missing["column"], missing["null_share"] * 100, color=colors)
    ax.set_xlabel("percent of rows that are null")
    ax.set_xlim(0, 100)
    ax.set_title("Missing values. Red bars are null on every benign row and on no adversarial row.")
    fig.tight_layout()
    _save(fig, "04_missing_data.png")
    return missing


def plot_outliers(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    flagged = pd.Series(False, index=df.index)
    for source, part in df.groupby("source"):
        q1, q3 = part["char_len"].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (part["char_len"] < low) | (part["char_len"] > high)
        flagged.loc[part.index] = mask.to_numpy()
        records.append(
            {
                "source": source,
                "rows": int(len(part)),
                "outliers": int(mask.sum()),
                "outlier_share": float(mask.mean()),
                "fence_low": float(low),
                "fence_high": float(high),
                "min": int(part["char_len"].min()),
                "max": int(part["char_len"].max()),
            }
        )
    summary = pd.DataFrame(records).sort_values("source")

    order = ["bipia", "poisonedrag", "mcptox", "msb"]
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [df.loc[df["source"].eq(source), "char_len"] for source in order]
    box = ax.boxplot(data, tick_labels=order, showfliers=True, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#D8DEE9")
    ax.set_yscale("log")
    ax.set_ylabel("characters (log scale)")
    ax.set_title("Character length by source. Points past the whiskers are 1.5-IQR outliers.")
    fig.tight_layout()
    _save(fig, "05_length_outliers.png")
    return summary


def plot_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, column, title in (
        (axes[0], "source", "By source"),
        (axes[1], "attack_surface", "By attack surface"),
        (axes[2], "split", "By split"),
    ):
        counts = (
            df.groupby([column, "label_name"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["benign", "adversarial"], fill_value=0)
        )
        bottom = np.zeros(len(counts))
        for name, color in (("benign", BENIGN), ("adversarial", ADVERSARIAL)):
            ax.bar(counts.index.astype(str), counts[name], bottom=bottom, color=color, label=name)
            bottom = bottom + counts[name].to_numpy()
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("rows")
    axes[0].legend(frameon=False)
    fig.suptitle("Target label: benign versus adversarial", y=1.03)
    fig.tight_layout()
    _save(fig, "06_target_balance.png")


def plot_insight(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Length separates the classes in opposite directions on different sources."""
    order = ["poisonedrag", "bipia", "mcptox", "msb"]
    medians = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    bins = np.linspace(np.log10(df["char_len"].clip(lower=1)).min(), np.log10(df["char_len"]).max(), 35)
    for ax, source in zip(axes.ravel(), order):
        part = df.loc[df["source"].eq(source)]
        medians[source] = {}
        for label, color in (("benign", BENIGN), ("adversarial", ADVERSARIAL)):
            values = part.loc[part["label_name"].eq(label), "char_len"]
            if values.empty:
                continue
            ax.hist(np.log10(values), bins=bins, density=True, alpha=0.55, color=color, label=label)
            median = float(values.median())
            medians[source][label] = median
            ax.axvline(np.log10(median), color=color, linestyle="--", linewidth=1)
        ax.set_title(source)
        ax.set_xlabel("log10(characters)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "The same cue points opposite ways: poisoned passages are shorter, injected tool text is longer",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "07_length_reversal.png")
    return medians


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def write_report(df: pd.DataFrame, corr: dict, missing: pd.DataFrame, outliers: pd.DataFrame, medians: dict) -> None:
    n_adv = int(df["label"].sum())
    n_ben = int((df["label"] == 0).sum())
    by_source_rows = []
    for source, stats in corr["by_source"].iterrows():
        by_source_rows.append([source, f"{stats['log_char_len']:+.2f}", f"{medians[source].get('benign', float('nan')):,.0f}", f"{medians[source].get('adversarial', float('nan')):,.0f}"])
    outlier_rows = [
        [r.source, f"{r.outliers:,}", f"{r.outlier_share:.1%}", f"{r.min:,}", f"{r.max:,.0f}"]
        for r in outliers.itertuples()
    ]
    missing_rows = [
        [r.column, f"{r.nulls:,}", f"{r.nulls_on_benign:,}", f"{r.nulls_on_adversarial:,}"]
        for r in missing.sort_values("column").itertuples()
        if r.nulls
    ]
    text = f"""# Phase 3 visual data

Figures for the unified corpus in `data/processed/dataset.parquet`: {len(df):,} rows, {n_adv:,} adversarial and {n_ben:,} benign. Regenerated by `uv run python notebooks/phase3_eda.py`. The stored columns are categories plus the raw text. The numeric plots use statistics computed from `text` (length, word count, punctuation, uppercase, digits). Those statistics are for exploration. They are not extra columns in the dataset.

## Distribution of each column

`source`, `attack_surface`, `role`, `split`, `attack_kind` and `attack_objective`:

![Categorical distributions](figures/01_categorical_distributions.png)

`subset` has 60 values. Eight collections hold almost all of the rows. The rest are individual MCP tool servers, each with a few dozen rows, so they are counted in the caption rather than drawn as sixty bars.

![Large subsets](figures/01b_subset.png)

Attack family, counted on adversarial rows only. Benign rows leave this field empty on purpose. PoisonedRAG is one family of 1,500 rows; the long tail is BIPIA's attack categories.

![Attack family](figures/01c_attack_family.png)

Text statistics, pooled. Pooling hides the pattern in the last figure: a single histogram of length looks almost the same for both classes because PoisonedRAG and the tool sources pull in opposite directions.

![Numeric distributions](figures/02_numeric_distributions.png)

## Correlation

Left: Spearman correlation among the text statistics. Length, word count and log length move together, which is expected. Right: the same statistics' correlation with `label`, computed separately inside each source. Log length correlates about {corr['by_source'].loc['poisonedrag', 'log_char_len']:+.2f} with the label on PoisonedRAG and about {corr['by_source'].loc['mcptox', 'log_char_len']:+.2f} on MCPTox.

![Correlation heatmaps](figures/03_correlation_heatmap.png)

{_md_table(["Source", "Spearman(log length, label)", "Benign median characters", "Adversarial median characters"], by_source_rows)}

Cramér's V measures association between the categorical columns. `source` and `attack_surface` are nearly the same fact: only MSB uses more than one surface. `attack_kind`, `attack_objective` and `attack_family` associate perfectly with `label` once their empty benign cells are filled in, because those cells are empty exactly when the row is benign.

![Categorical association](figures/03b_categorical_association.png)

## Missing data

Blank text would count as missing. There is none. The only nulls are the three attack fields, and every one of them sits on a benign row.

![Missing data](figures/04_missing_data.png)

{_md_table(["Column", "Nulls", "On benign rows", "On adversarial rows"], missing_rows)}

A model that is allowed to see `attack_kind` can read the label off the missingness. Training uses `text` and `label` only.

## Outliers

An outlier is a character length outside the 1.5-IQR fences of its own source. The long BIPIA documents are real emails, articles and tables, not broken rows. PoisonedRAG's adversarial texts are short because the attack writes a short passage, not because those rows are corrupt.

![Length outliers](figures/05_length_outliers.png)

{_md_table(["Source", "Outlier rows", "Share", "Shortest", "Longest"], outlier_rows)}

## Target

The label is close to balanced overall ({n_adv / len(df):.1%} adversarial) and inside the train, validation and test splits. It is not balanced inside a source. MSB `tool_output` has adversarial rows and no benign rows, so a detector can look accurate on that slice by always saying adversarial.

![Target balance](figures/06_target_balance.png)

## Non-obvious insight: length reverses direction

A length histogram of the whole corpus suggests length does not separate the classes. Split by source, it does, and the direction flips.

On PoisonedRAG the adversarial passages sit to the left of the benign retrieved passages: the attack text is shorter than a real passage. On MCPTox and MSB the adversarial tool text sits to the right: the injection makes the description longer. On BIPIA the two histograms lie on top of each other. The dashed lines are medians.

![Length reversal](figures/07_length_reversal.png)

That is why a length-only model fit on the pooled training set looks useless in-distribution and then looks strong on MCPTox and MSB. It learned one direction and the held-out tool descriptions happen to match that direction, while PoisonedRAG matches the other one. Any detector score has to be compared with length inside each source. The numbers for that comparison are already in `DATA_CARD.md`, from `src/models/baselines.py`.
"""
    WRITEUP.write_text(text)


def main() -> None:
    _style()
    df = load_frame()
    plot_categorical(df)
    plot_numeric_distributions(df)
    corr = plot_correlation(df)
    missing = plot_missing(df)
    outliers = plot_outliers(df)
    plot_target(df)
    medians = plot_insight(df)
    write_report(df, corr, missing, outliers, medians)
    print(f"wrote figures to {FIG_DIR}")
    print(f"wrote {WRITEUP}")


if __name__ == "__main__":
    main()
