"""Build the Phase 4 report's tables and AUROC figure from the committed result files.

  uv run python report/phase4/make_tables.py           # write tables.md and fig_auroc.png
  uv run python report/phase4/make_tables.py --check   # fail if report.md is out of date

Reads report/baselines/*.csv and report/phase4/b1_optuna/, and computes the length-only
reference with the shared harness (its bootstrap is seeded, so reruns match). Edit the sources
and rerun; never edit tables.md, and paste its tables into report.md unchanged.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINES = ROOT / "report/baselines"
STUDY = HERE / "b1_optuna"
OUT, FIGURE, REPORT = HERE / "tables.md", HERE / "fig_auroc.png", HERE / "report.md"
sys.path[:0] = [str(ROOT / "src" / part) for part in ("data", "eval", "models")]

SINGLE = {"B0 perplexity": "b0_perplexity", "B1 TF-IDF": "b1_tfidf_lr",
          "B1-Optuna": "b1_tfidf_lr_optuna"}
B2_ROWS = {"B2 row 1": "row1", "B2 row 2 (selected)": "row2", "B2 row 3": "row3"}
B2_NOTE = ("B2 rows (mean ± standard deviation over 3 seeds): row 1 plain, row 2 adds the pairwise "
           "loss, row 3 is row 2 trained on BIPIA only. Per-seed intervals are in Appendix A.")
EVAL_SETS = {"test": "train_pool test (in distribution)", "mcptox": "MCPTox (unseen surface, LOSO)",
             "msb": "MSB (transfer, indicative)"}
SLICE_NAMES = {"source=bipia": "test: BIPIA", "source=poisonedrag": "test: PoisonedRAG",
               "attack_kind=instruction": "test: instruction", "attack_kind=poisoning": "test: poisoning",
               "attack_surface=tool_description": "MSB: tool descriptions",
               "attack_surface=tool_output": "MSB: tool outputs"}
TABLE_IDS = {"test": "tbl:test", "mcptox": "tbl:mcptox", "msb": "tbl:msb"}


def f3(value):
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def ci(value, low, high):
    return f"{f3(value)} [{f3(low)}, {f3(high)}]" if not math.isnan(low) else f3(value)


def wilson(count, n, z=1.959964):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return math.nan, math.nan
    p = count / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def fpr_cell(fpr, n_benign):
    if n_benign == 0 or math.isnan(fpr):
        return "–"
    low, high = wilson(round(fpr * n_benign), n_benign)
    return f"{fpr:.3f} [{low:.3f}, {high:.3f}]"


def mean_sd(values):
    values = pd.Series(values, dtype=float).dropna()
    return "–" if values.empty else f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"


def table(header, rows, caption, ident, text_columns=1):
    """A pandoc pipe table: the first text_columns left-aligned, the rest (numbers) right-aligned.
    Each separator's dash count is its column's longest cell (header words count singly, since
    headers wrap), which pandoc turns into relative column widths."""
    widths = [max([len(word) for word in str(h).split()] + [len(str(row[i])) for row in rows])
              for i, h in enumerate(header)]
    aligns = [":" + "-" * max(3, w) if i < text_columns else "-" * max(3, w) + ":"
              for i, w in enumerate(widths)]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(aligns) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + f"\n\nTable: {caption} {{#{ident}}}"


def load_reports():
    """{name: report frame} for single runs (length computed), and {row: frame with seed} for B2."""
    from harness import evaluate, load_data
    from train import fit_detector

    data = load_data()
    _, length = evaluate(fit_detector("length", data)[0], data)
    single = {"Length only": length}
    single.update({name: pd.read_csv(BASELINES / f"{stem}.csv") for name, stem in SINGLE.items()})
    b2 = {name: pd.read_csv(BASELINES / f"b2_deberta_{row}.csv") for name, row in B2_ROWS.items()}
    return data, single, b2


def at(frame, eval_set, slice_="all"):
    rows = frame[(frame.eval_set == eval_set) & (frame.slice == slice_)]
    return rows.iloc[0] if len(rows) == 1 else rows


def split_table(data):
    rows = [("train_pool train", "fit", data.train, "fitting; Optuna folds"),
            ("train_pool val", "fit", data.val, "selection; deployed threshold (1% FPR)"),
            ("train_pool test", "eval", data.evals["test"], "in-distribution results; decision rules"),
            ("MCPTox", "LOSO holdout", data.evals["mcptox"], "unseen surface; scored once per run"),
            ("MSB", "transfer holdout", data.evals["msb"], "cross-benchmark; indicative")]
    body = [(name, role, use, f"{(f.label == 0).sum():,}", f"{(f.label == 1).sum():,}")
            for name, role, f, use in rows]
    return table(["Split", "Role", "Used for", "Benign", "Adversarial"], body,
                 "Splits. MCPTox and MSB are held out whole and never used to choose anything.", "tbl:splits",
                 text_columns=3)


def eval_table(eval_set, single, b2):
    header = ["Detector", "AUROC [95% CI]", "Length-binned AUROC", "TPR (deployed)",
              "FPR (deployed) [Wilson]", "TPR at 5% FPR"]
    body = []
    for name, frame in single.items():
        r = at(frame, eval_set)
        body.append((name, ci(r.auroc, r.auroc_lo, r.auroc_hi), f3(r.length_binned_auroc), f3(r.tpr),
                     fpr_cell(r.fpr, int(r.n_benign)), f3(r.tpr_at_matched_fpr)))
    for name, frame in b2.items():
        r = at(frame, eval_set)
        body.append((name, mean_sd(r.auroc), mean_sd(r.length_binned_auroc), mean_sd(r.tpr),
                     mean_sd(r.fpr), mean_sd(r.tpr_at_matched_fpr)))
    first = at(single["B1 TF-IDF"], eval_set)
    caption = (f"{EVAL_SETS[eval_set]}: {int(first.n_benign)} benign, {int(first.n_adversarial)} adversarial. "
               + B2_NOTE)
    return table(header, body, caption, TABLE_IDS[eval_set])


def verdict_table(b2, bar):
    frame = b2["B2 row 2 (selected)"]
    body = []
    for _, r in at(frame, "mcptox").sort_values("seed").iterrows():
        passes = r.tpr_at_matched_fpr_lo > bar and r.length_binned_auroc_lo > 0.5
        body.append((int(r.seed), ci(r.tpr_at_matched_fpr, r.tpr_at_matched_fpr_lo, r.tpr_at_matched_fpr_hi),
                     ci(r.length_binned_auroc, r.length_binned_auroc_lo, r.length_binned_auroc_hi),
                     "yes" if passes else "no"))
    return table(["Seed", "TPR at matched 5% FPR [95% CI]", "Length-binned AUROC [95% CI]", "Passes"], body,
                 f"B2 verdict on MCPTox for the selected configuration (row 2). A seed passes when both lower "
                 f"bounds clear their bars: B1's TPR at 5% FPR ({bar:.4f}) and 0.5.", "tbl:verdict")


def ablation_table(single, b2):
    def b2_mean(name, eval_set, column):
        return at(b2[name], eval_set)[column].mean()

    def one(name, eval_set, column):
        return float(at(single[name], eval_set)[column])

    columns = [("test", "auroc"), ("mcptox", "auroc"), ("mcptox", "length_binned_auroc"),
               ("mcptox", "tpr_at_matched_fpr")]
    rows = []
    pairs = [("Pairwise loss on BIPIA twins", "B2 row 1 → row 2", "B2 row 1", "B2 row 2 (selected)"),
             ("PoisonedRAG in training", "B2 row 3 → row 2", "B2 row 3", "B2 row 2 (selected)")]
    for choice, label, base, variant in pairs:
        rows.append((choice, label, *[f"{b2_mean(base, e, c):.3f} → {b2_mean(variant, e, c):.3f}" for e, c in columns]))
    rows.append(("Search space and family-CV objective", "B1 → B1-Optuna",
                 *[f"{one('B1 TF-IDF', e, c):.3f} → {one('B1-Optuna', e, c):.3f}" for e, c in columns]))
    return table(["Design choice", "Comparison", "Test AUROC", "MCPTox AUROC", "MCPTox length-binned",
                  "MCPTox TPR at 5% FPR"], rows,
                 "Ablation: each row changes one design choice. B2 cells are means over 3 seeds. "
                 "Word vs character features is in Table \\ref{tbl:features}.", "tbl:ablation", text_columns=2)


def features_table():
    trials = pd.read_csv(STUDY / "trials.csv")
    rows = []
    for column, label in (("features", "Feature set"), ("char_analyzer", "Char analyzer")):
        for value, group in trials.dropna(subset=[column]).groupby(column):
            rows.append((label, value, len(group), f"{group.value.max():.4f}", f"{group.value.median():.4f}"))
    return table(["Setting", "Value", "Trials", "Best CV AUROC", "Median CV AUROC"], rows,
                 "B1-Optuna trials grouped by feature setting (mean held-out AUROC over 5 family folds of train). "
                 "TPE samples promising settings more often, so trial counts differ.", "tbl:features", text_columns=2)


def seeds_table(b2):
    body = []
    for name, frame in b2.items():
        for seed in sorted(frame.seed.unique()):
            run = frame[frame.seed == seed]
            t, m, s = at(run, "test"), at(run, "mcptox"), at(run, "msb")
            body.append((name.split(" (")[0].removeprefix("B2 row "), int(seed), ci(t.auroc, t.auroc_lo, t.auroc_hi),
                         ci(m.auroc, m.auroc_lo, m.auroc_hi),
                         ci(m.tpr_at_matched_fpr, m.tpr_at_matched_fpr_lo, m.tpr_at_matched_fpr_hi),
                         ci(s.auroc, s.auroc_lo, s.auroc_hi)))
    return table(["B2 row", "Seed", "Test AUROC", "MCPTox AUROC", "MCPTox TPR at 5% FPR", "MSB AUROC"], body,
                 "Every reported B2 run, with 95% bootstrap intervals. Row 2 was selected on validation.",
                 "tbl:seeds")


def slices_table(single, b2):
    names = ["Length only", "B1 TF-IDF", "B1-Optuna"]
    slices = [("test", "source=bipia"), ("test", "source=poisonedrag"), ("test", "attack_kind=instruction"),
              ("test", "attack_kind=poisoning"), ("msb", "attack_surface=tool_description"),
              ("msb", "attack_surface=tool_output")]
    body = []
    for eval_set, slice_ in slices:
        row = [SLICE_NAMES[slice_], int(at(single['B1 TF-IDF'], eval_set, slice_).n_benign),
               int(at(single['B1 TF-IDF'], eval_set, slice_).n_adversarial)]
        row += [f3(float(at(single[n], eval_set, slice_).auroc)) for n in names]
        row += [f3(at(b2[n], eval_set, slice_).auroc.mean()) for n in b2]
        body.append(row)
    return table(["Slice", "Benign", "Adv.", "Length", "B1", "B1-Optuna", "B2 row 1", "B2 row 2",
                  "B2 row 3"], body,
                 "AUROC by slice (B2: mean over seeds). Test-split slices pair each attack kind with every benign "
                 "row. MSB tool outputs have no benign rows, so AUROC is undefined.",
                 "tbl:slices")


def figure(single, b2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = [(name, frame) for name, frame in single.items()]
    for name, frame in b2.items():
        short = name.split(" (")[0]
        entries += [(f"{short} s{seed}", frame[frame.seed == seed]) for seed in sorted(frame.seed.unique())]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True)
    for ax, eval_set in zip(axes, EVAL_SETS):
        for y, (name, frame) in enumerate(entries):
            r = at(frame, eval_set)
            colour = "0.45" if name == "Length only" else ("C3" if name.startswith("B2") else "C0")
            ax.errorbar(r.auroc, y, xerr=[[r.auroc - r.auroc_lo], [r.auroc_hi - r.auroc]], fmt="o",
                        color=colour, ms=4, capsize=2)
        ax.axvline(0.5, color="0.6", ls="--", lw=0.8)
        ax.axvline(float(at(single["Length only"], eval_set).auroc), color="0.45", ls=":", lw=1)
        ax.set_title({"test": "train_pool test", "mcptox": "MCPTox", "msb": "MSB"}[eval_set], fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel("AUROC (95% CI)")
    axes[0].set_yticks(range(len(entries)), [name for name, _ in entries], fontsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=200)


def build():
    data, single, b2 = load_reports()
    bar = float(at(single["B1 TF-IDF"], "mcptox").tpr_at_matched_fpr)
    best = json.loads((STUDY / "best_params.json").read_text())
    blocks = [
        "<!-- Generated by report/phase4/make_tables.py; do not edit. -->",
        split_table(data),
        *[eval_table(e, single, b2) for e in EVAL_SETS],
        verdict_table(b2, bar),
        ablation_table(single, b2),
        features_table(),
        seeds_table(b2),
        slices_table(single, b2),
        f"<!-- B1-Optuna best trial #{best['trial']}, mean CV AUROC {best['mean_cv_auroc']:.4f} -->",
    ]
    OUT.write_text("\n\n".join(blocks) + "\n")
    figure(single, b2)
    print(f"wrote {OUT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")


def check():
    """Every table in tables.md must appear verbatim in report.md."""
    report = REPORT.read_text()
    tables = [b for b in OUT.read_text().split("\n\n") if b.startswith("|")]
    captions = [b for b in OUT.read_text().split("\n\n") if b.startswith("Table:")]
    missing = [c for t, c in zip(tables, captions) if f"{t}\n\n{c}" not in report]
    for caption in missing:
        print(f"out of date in report.md: {caption[:90]}")
    if missing:
        raise SystemExit(1)
    print(f"report.md matches all {len(tables)} tables in tables.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="Check report.md against tables.md")
    if parser.parse_args().check:
        check()
    else:
        build()
