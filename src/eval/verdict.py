"""Choose the B2 configuration on train_pool val, then judge every seed of it against B1 on MCPTox.

    PYTHONPATH=src/data:src/eval:src/models uv run python src/eval/verdict.py \\
        --scores-dir report/deberta_scores --b1 report/baselines/b1_tfidf_lr.csv \\
        --out-dir report/baselines

The success criterion is fixed in docs/superpowers/specs/2026-11-07-deberta-detector-design.md:
every seed of the selected configuration must have, on MCPTox, a TPR-at-5%-FPR lower bound above
B1's TPR at 5% FPR and a length-binned AUROC lower bound above 0.5. Selection reads only the
score files' val rows; MCPTox and MSB are scored once per reported run and never choose anything.
Val-only (sweep) files are listed in an appendix with val numbers only.
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dataset import DEFAULT_PATH
from harness import REPORT_COLUMNS, evaluate, load_data
from metrics import auroc, rates_at_threshold, threshold_at_fpr
from precomputed import PrecomputedScores

SELECTION_FPR = 0.01
LENGTH_FLOOR = 0.5


def val_selection_score(score_file) -> tuple[float, float]:
    """(val TPR at 1% val FPR, val AUROC) from the file's val rows only, the same rule
    DebertaDetector uses to choose an epoch."""
    frame = pq.read_table(score_file, columns=["eval_set", "label", "score"]).to_pandas()
    val = frame[frame.eval_set == "val"]
    scores, labels = val.score.to_numpy(float), val.label.to_numpy()
    if not (labels == 0).any() or not (labels == 1).any():
        raise ValueError(f"{score_file}: val rows need both labels to select on")
    threshold = threshold_at_fpr(scores[labels == 0], SELECTION_FPR)
    tpr, _ = rates_at_threshold(scores, labels, threshold)
    return tpr, auroc(scores, labels)


def val_means(files_by_row: dict) -> dict:
    """{row: (mean val TPR, mean val AUROC)} across each row's seeds. Reads only val rows."""
    means = {}
    for row, files in files_by_row.items():
        values = np.array([val_selection_score(f) for f in files], dtype=float)
        means[row] = (float(values[:, 0].mean()), float(values[:, 1].mean()))
    return means


def select_configuration(files_by_row: dict) -> str:
    """The row with the highest mean val TPR across its seeds, ties broken by mean val AUROC."""
    means = val_means(files_by_row)
    return max(sorted(means), key=lambda row: means[row])


def _mcptox_all(report: pd.DataFrame) -> pd.Series:
    rows = report[(report.eval_set == "mcptox") & (report.slice == "all")]
    if len(rows) != 1:
        raise ValueError(f"expected one mcptox/all row in the report, found {len(rows)}")
    return rows.iloc[0]


def seed_passes(report: pd.DataFrame, b1_tpr: float) -> bool:
    """Both lower bounds on mcptox/all clear their bars, strictly. A NaN bound fails."""
    row = _mcptox_all(report)
    return bool(row.tpr_at_matched_fpr_lo > b1_tpr and row.length_binned_auroc_lo > LENGTH_FLOOR)


def verdict(passes: list) -> str:
    if not passes:
        raise ValueError("no seeds to judge")
    if all(passes):
        return "pass"
    return "fail" if not any(passes) else "mixed"


def b1_bar(path) -> float:
    report = pd.read_csv(path)
    if "tpr_at_matched_fpr" not in report.columns:
        raise ValueError(f"{path} has no tpr_at_matched_fpr column: regenerate it with "
                         "make evaluate DETECTOR=tfidf OUT=...")
    value = float(_mcptox_all(report).tpr_at_matched_fpr)
    if math.isnan(value):
        raise ValueError(f"{path}: B1's tpr_at_matched_fpr on mcptox/all is NaN")
    return value


def _fmt(value) -> str:
    return "NaN" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"


def write_verdict(path, *, selected, bar, b1_path, selection, seed_rows, result, sweeps):
    lines = [
        "# B2 DeBERTa verdict",
        "",
        f"**Verdict: {result}**",
        "",
        f"- Selected configuration: **{selected}**, chosen on train_pool val only (highest mean val "
        f"TPR at {SELECTION_FPR:.0%} val FPR across seeds, ties by mean val AUROC).",
        f"- B1 bar: TPR at 5% FPR on MCPTox = **{_fmt(bar)}** (`{b1_path}`).",
        f"- A seed passes when, on MCPTox, its TPR-at-5%-FPR 95% lower bound is above the B1 bar "
        f"and its length-binned AUROC 95% lower bound is above {LENGTH_FLOOR}.",
        "- Pass needs every seed to pass; fail means none did; mixed is anything between. "
        "Nothing is rerun after this verdict.",
        "",
        "## Selection on val",
        "",
        "| row | seeds | mean val TPR @1% FPR | mean val AUROC |",
        "|---|---|---|---|",
    ]
    for row, (n, tpr, roc) in sorted(selection.items()):
        mark = " (selected)" if row == selected else ""
        lines.append(f"| {row}{mark} | {n} | {_fmt(tpr)} | {_fmt(roc)} |")
    lines += [
        "",
        f"## {selected} on MCPTox, every seed",
        "",
        "| seed | TPR @5% FPR | lower | upper | length-binned AUROC | lower | upper | passes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in seed_rows:
        lines.append(
            f"| {r['seed']} | {_fmt(r['tpr_at_matched_fpr'])} | {_fmt(r['tpr_at_matched_fpr_lo'])} | "
            f"{_fmt(r['tpr_at_matched_fpr_hi'])} | {_fmt(r['length_binned_auroc'])} | "
            f"{_fmt(r['length_binned_auroc_lo'])} | {_fmt(r['length_binned_auroc_hi'])} | "
            f"{'yes' if r['passes'] else 'no'} |"
        )
    if sweeps:
        lines += [
            "",
            "## Appendix: val-only sweep runs",
            "",
            "Not reported runs: val numbers only, never scored on the holdouts.",
            "",
            "| run | row | seed | val TPR @1% FPR | val AUROC |",
            "|---|---|---|---|---|",
        ]
        for s in sweeps:
            lines.append(f"| {s['run']} | {s['row']} | {s['seed']} | {_fmt(s['tpr'])} | {_fmt(s['auroc'])} |")
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scores-dir", required=True, help="Directory of B2 score files (*.parquet)")
    parser.add_argument("--b1", required=True, help="B1 report CSV with tpr_at_matched_fpr")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data", default=str(DEFAULT_PATH), help="dataset.parquet to read")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)

    files = sorted(Path(args.scores_dir).glob("*.parquet"))
    reported, sweeps = defaultdict(list), []
    for path in files:
        scores = PrecomputedScores(path)
        if scores.val_only:
            tpr, roc = val_selection_score(path)
            sweeps.append({"run": path.stem, "row": scores.params["row"], "seed": scores.seed,
                           "tpr": tpr, "auroc": roc})
        else:
            reported[scores.params["row"]].append(path)
    if not reported:
        parser.error(f"no reported (non val-only) score files in {args.scores_dir}")

    bar = b1_bar(args.b1)
    selected = select_configuration(reported)
    selection = {row: (len(reported[row]), *means) for row, means in val_means(reported).items()}

    data = load_data(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for row, paths in sorted(reported.items()):
        reports = []
        for path in paths:
            scores = PrecomputedScores(path)
            _, report = evaluate(scores, data, n_boot=args.n_boot)
            report.insert(1, "seed", scores.seed)
            reports.append(report)
            if row == selected:
                best = _mcptox_all(report)
                seed_rows.append({"seed": scores.seed, "passes": seed_passes(report, bar),
                                  **{k: best[k] for k in REPORT_COLUMNS if k.startswith(
                                      ("tpr_at_matched_fpr", "length_binned_auroc"))}})
        pd.concat(reports, ignore_index=True).to_csv(out_dir / f"b2_deberta_{row}.csv", index=False)

    seed_rows.sort(key=lambda r: r["seed"])
    result = verdict([r["passes"] for r in seed_rows])
    write_verdict(out_dir / "b2_verdict.md", selected=selected, bar=bar, b1_path=args.b1,
                  selection=selection, seed_rows=seed_rows, result=result,
                  sweeps=sorted(sweeps, key=lambda s: s["run"]))
    print(f"selected {selected}; B1 bar {bar:.4f}; verdict: {result}")
    print(f"wrote {out_dir / 'b2_verdict.md'} and b2_deberta_<row>.csv for {sorted(reported)}")
    return result


if __name__ == "__main__":
    main()
