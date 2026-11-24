"""SHAP feature attribution for the TF-IDF detectors (B1 and B1-Optuna).

    PYTHONPATH=src/data:src/eval:src/models uv run python src/eval/explain_tfidf.py

For a linear model, SHAP values with an interventional background are exact: feature i of row x
contributes coef_i * (x_i - mean_i), in log-odds. We use shap.LinearExplainer with 200 train_pool
train rows as the background, computed in batches because the feature space is large (200,000
features for B1-Optuna). Writes to report/phase5/shap/:

  global.csv      mean |SHAP| per feature on test and on MCPTox, top features only
  presence.csv    share of rows, by eval set and label, that contain each of the model's top
                  adversarial word n-grams (ranked by mean |SHAP| on test)
  local.csv       top contributions for six MCPTox rows (three false alarms, three misses)
  *.png           the figures used in the report
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from harness import load_data
from train import fit_detector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "phase5" / "shap"
PHASE5 = ROOT / "report" / "phase5"
BACKGROUND = 200
BATCH = 64
SEED = 7267
TOP = 50
TOP_WORDS = 10
DETECTORS = {"b1": "tfidf", "b1_optuna": "tfidf-optuna"}


def parts(detector):
    """(vectorizer union, logistic regression, feature names) of a fitted TF-IDF detector."""
    union, model = detector.pipeline.steps[0][1], detector.pipeline.steps[1][1]
    return union, model, np.asarray(union.get_feature_names_out())


def explainer(detector, background_texts):
    import shap

    union, model, _ = parts(detector)
    return shap.LinearExplainer(model, union.transform(list(background_texts)))


def shap_batches(explain, detector, texts, batch=BATCH):
    """Yield dense SHAP arrays (rows x features) for consecutive batches of texts."""
    union, _, _ = parts(detector)
    texts = list(texts)
    for start in range(0, len(texts), batch):
        values = explain.shap_values(union.transform(texts[start:start + batch]))
        yield np.asarray(values.toarray() if hasattr(values, "toarray") else values)


def mean_abs(explain, detector, texts) -> np.ndarray:
    total, n = None, 0
    for values in shap_batches(explain, detector, texts):
        part = np.abs(values).sum(axis=0)
        total = part if total is None else total + part
        n += len(values)
    return total / max(n, 1)


def local_top(explain, detector, text, k=8) -> pd.DataFrame:
    _, _, names = parts(detector)
    values = next(shap_batches(explain, detector, [text]))[0]
    order = np.argsort(-np.abs(values))[:k]
    return pd.DataFrame({"feature": names[order], "shap": values[order]})


def pick_rows(folder: Path, column: str, n=3):
    """Three MCPTox false alarms (highest scores) and three misses (lowest scores) at the
    deployed threshold, by row id."""
    scores = pd.read_parquet(folder / "scores.parquet")
    threshold = json.loads((folder / "thresholds.json").read_text())[column]
    m = scores[scores.eval_set == "mcptox"]
    fp = m[(m.label == 0) & (m[column] >= threshold)].nlargest(n, column)
    fn = m[(m.label == 1) & (m[column] < threshold)].nsmallest(n, column)
    return pd.concat([fp.assign(error="false_positive"), fn.assign(error="false_negative")])[["id", "error", column]]


def bar(ax, labels, values, title):
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in values]
    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(title, fontsize=9)


def show(name: str) -> str:
    kind, _, gram = name.partition("__")
    return f"{kind}: {gram!r}"


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_data()
    texts = pd.read_parquet(ROOT / "data/processed/dataset.parquet", columns=["id", "text"]).set_index("id").text
    background = data.train.text.sample(BACKGROUND, random_state=SEED)

    global_rows, mass_rows, local_rows = [], [], []
    for column, name in DETECTORS.items():
        detector, _ = fit_detector(name, data)
        explain = explainer(detector, background)
        _, model, names = parts(detector)
        coef = model.coef_[0]
        per_set = {k: mean_abs(explain, detector, data.evals[k].text) for k in ("test", "mcptox", "msb")}
        top = np.argsort(-per_set["test"])[:TOP]
        for rank, i in enumerate(top):
            global_rows.append({"detector": column, "rank": rank + 1, "feature": names[i], "coef": coef[i],
                                **{f"mean_abs_shap_{k}": v[i] for k, v in per_set.items()}})
        adversarial_words = [i for i in top if coef[i] > 0 and names[i].startswith("word__")][:TOP_WORDS]
        union, _, _ = parts(detector)
        for k, frame in data.evals.items():
            present = (union.transform(list(frame.text))[:, adversarial_words] > 0).toarray()
            for label in (0, 1):
                mask = frame.label.to_numpy() == label
                for j, i in enumerate(adversarial_words):
                    mass_rows.append({"detector": column, "eval_set": k, "label": label, "n": int(mask.sum()),
                                      "word": names[i].removeprefix("word__"), "coef": float(coef[i]),
                                      "share_present": float(present[mask, j].mean()) if mask.any() else float("nan")})
        for row in pick_rows(PHASE5, column).itertuples():
            top_local = local_top(explain, detector, texts[row.id])
            local_rows.append(top_local.assign(detector=column, id=row.id, error=row.error))

        # Figure: top 15 test features, signed by coefficient, test vs MCPTox.
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
        idx = top[:15]
        signed = np.sign(coef[idx])
        bar(axes[0], [show(names[i]) for i in idx], signed * per_set["test"][idx], "test split")
        bar(axes[1], [show(names[i]) for i in idx], signed * per_set["mcptox"][idx], "MCPTox")
        fig.supxlabel("mean |SHAP| (log-odds), signed toward adversarial (red) or benign (blue)", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{column}_global.png", dpi=200)
        plt.close(fig)

    pd.DataFrame(global_rows).to_csv(out / "global.csv", index=False)
    pd.DataFrame(mass_rows).to_csv(out / "presence.csv", index=False)
    local = pd.concat(local_rows, ignore_index=True)
    local.to_csv(out / "local.csv", index=False)

    # Figure: local explanations for B1-Optuna's six rows.
    b1o = local[local.detector == "b1_optuna"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    for ax, ((row_id, error), part) in zip(axes.flat, b1o.groupby(["id", "error"], sort=False)):
        bar(ax, [show(f) for f in part.feature], part.shap.to_numpy(),
            f"{'false alarm' if error == 'false_positive' else 'missed attack'}")
    fig.tight_layout()
    fig.savefig(out / "b1_optuna_local.png", dpi=200)
    plt.close(fig)
    print(pd.DataFrame(mass_rows).pivot_table(index=["detector", "word"], columns=["eval_set", "label"],
                                              values="share_present").round(3).to_string())


if __name__ == "__main__":
    main()
