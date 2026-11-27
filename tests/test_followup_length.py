"""Follow-up length analyses: B2 against length, per-template results, paired server bootstrap."""

import numpy as np
import pandas as pd
import pytest

from followup_length import (
    b2_length_spearman, long_benign_auroc, paired_server_bootstrap, per_template, rank_average,
    template_of,
)


def frame():
    """Two servers, both labels each. Attacks are longer; b2 falls with length."""
    rows = []
    for server in ("s1", "s2", "s3", "s4"):
        for n, label, family in [(50, 0, None), (100, 0, None), (200, 0, None),
                                 (150, 1, "x-Template-1"), (300, 1, "y-Template-2"), (400, 1, "y-Template-2")]:
            rows.append(dict(eval_set="mcptox", source="mcptox", subset=server, label=label,
                             attack_family=family, n_chars=n + len(server)))
    f = pd.DataFrame(rows)
    f["length"] = f.n_chars.astype(float)
    f["b1_optuna"] = f.label * 0.5 + f.n_chars / 1000
    for seed in ("s7267", "s7268", "s7269"):
        f[f"b2_row2_{seed}"] = 1 / f.n_chars
    return f


def test_template_of():
    s = template_of(pd.Series(["x Template-2", None, "y-Template-3"]))
    assert s[0] == "Template-2" and s[2] == "Template-3" and pd.isna(s[1])


def test_spearman_negative_when_score_falls_with_length():
    out = b2_length_spearman(frame())
    assert set(out.rows) == {"all", "benign"}
    assert (out.rho < -0.9).all()
    assert len(out[out.rows == "all"]) == 3  # one row per seed on the single source


def test_long_benign_auroc_drops_short_benign_rows():
    f = frame()
    # All attacks longer than all benign: AUROC 0 for b2 on the full set.
    # Keeping benign rows >= shortest attack (150+2) leaves the 200-char ones, which are
    # longer than the 150 attacks only, so the inverse-length score ranks 4/12 pairs correctly.
    assert long_benign_auroc(f, "b2_row2_s7267") == pytest.approx(4 / 12)


def test_per_template_counts_and_recall():
    out = per_template(frame(), threshold=0.5).set_index("template")
    assert out.loc["Template-1", "attacks"] == 4 and out.loc["Template-2", "attacks"] == 8
    assert out.loc["Template-1", "length_auroc"] == pytest.approx(8 / 12)
    # b1_optuna on attacks = 0.5 + n/1000 >= 0.5 always
    assert out.loc["Template-2", "b1_optuna_recall"] == 1.0


def test_paired_bootstrap_identical_columns_is_zero():
    out = paired_server_bootstrap(frame(), "length", "length", n_boot=50)
    assert out == {"diff": 0.0, "lo": 0.0, "hi": 0.0}


def test_paired_bootstrap_skips_one_label_resamples():
    f = frame()
    f = pd.concat([f, f.assign(subset="benign-only", label=0)[f.label == 0]], ignore_index=True)
    out = paired_server_bootstrap(f, "b1_optuna", "length", n_boot=200)
    assert np.isfinite([out["lo"], out["hi"]]).all() and out["lo"] <= out["diff"] <= out["hi"]


def test_rank_average_of_identical_columns_preserves_order():
    a = pd.Series([3.0, 1.0, 2.0])
    assert np.argsort(rank_average(a, a)).tolist() == [1, 2, 0]
