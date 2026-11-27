"""Follow-up: an off-the-shelf detector's scores joined to the phase-5 scores by text hash."""

import pandas as pd
import pytest

from deberta_run import text_hash
from followup_offshelf import COLUMN, attach, compare


def tables():
    rows = []
    for s in range(4):
        for j, label in enumerate((0, 0, 1, 1)):
            rows.append(dict(id=f"r{s}{j}", text=f"text {s} {j}", eval_set="mcptox", label=label,
                             subset=f"srv{s}", length=float(j), b1_optuna=float(s)))  # server only: AUROC 0.5
    scores = pd.DataFrame(rows)
    texts = scores[["id", "text"]]
    offshelf = pd.DataFrame({"text_sha256": scores.text.map(text_hash), "eval_set": "mcptox",
                             "score": scores.label * 0.9 + 0.01})
    # a row from another eval set with the same hash must not be picked up
    offshelf = pd.concat([offshelf, offshelf.head(1).assign(eval_set="val", score=-1.0)], ignore_index=True)
    return scores.drop(columns="text"), texts, offshelf


def test_attach_matches_by_hash_within_eval_set():
    scores, texts, offshelf = tables()
    rows = attach(scores, texts, offshelf)
    assert len(rows) == 16 and (rows[COLUMN] == rows.label * 0.9 + 0.01).all()


def test_attach_refuses_missing_scores():
    scores, texts, offshelf = tables()
    with pytest.raises(ValueError, match="1 mcptox rows"):
        attach(scores, texts, offshelf.iloc[1:])


def test_compare_reports_differences_and_medians():
    scores, texts, offshelf = tables()
    out = compare(attach(scores, texts, offshelf), n_boot=50)
    assert out[f"{COLUMN}_auroc"] == 1.0 and out["servers"] == 4
    assert out[f"{COLUMN}_minus_length"]["diff"] == 0.0  # length also separates perfectly here
    assert out[f"{COLUMN}_minus_b1_optuna"]["diff"] == pytest.approx(0.5)
    assert out["median_score"] == {"benign": pytest.approx(0.01), "attack": pytest.approx(0.91)}
