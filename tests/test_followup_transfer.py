"""Follow-up transfer experiments: few-shot MCPTox servers, in-surface folds, MSB to MCPTox."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from followup_transfer import few_shot, in_surface, msb_to_mcptox, score_block  # noqa: E402
from harness import Data  # noqa: E402
from synth import make  # noqa: E402
from tfidf import TfidfDetector  # noqa: E402


class Spy:
    """Fast fit that records the servers it was trained on."""

    def __init__(self):
        self.trained = []

    def __call__(self, train):
        self.trained.append(set(train.get("subset", pd.Series(dtype=str)).dropna()))
        return TfidfDetector(C=1.0, min_df=1).fit(train.text, train.label)


def synth():
    rows = make(n_per=60, seed=0)
    rows["subset"] = rows.group.str.extract(r"-[ab](\d+)")[0].astype(int).mod(12).astype(str).radd("srv")
    rows["attack_family"] = None
    attacks = rows.label == 1
    rows.loc[attacks, "attack_family"] = [f"x-Template-{i % 3 + 1}" for i in range(attacks.sum())]
    train = rows[rows.source.isin(["bipia", "poisonedrag"])].drop(columns="subset")
    half = train.sample(frac=0.5, random_state=0)
    data = Data(train=half, val=train.drop(half.index), evals={})
    return data, rows[rows.source == "mcptox"].reset_index(drop=True), rows[rows.source == "msb"].reset_index(drop=True)


def test_few_shot_never_trains_on_test_servers():
    data, mcp, _ = synth()
    spy = Spy()
    out = few_shot(data, mcp, spy, ks=(0, 2, 5), draws=3, seed=1)
    assert len(out) == 9
    # fits run in row order, except k=0, which is fitted once (first) and cached
    assert spy.trained[0] == set()
    for row, servers in zip(out[out.k > 0].itertuples(), spy.trained[1:], strict=True):
        assert len(servers) == row.k_used
        assert row.n == (~mcp.subset.isin(servers)).sum()  # tested only on servers it never saw


def test_few_shot_caps_k():
    data, mcp, _ = synth()
    out = few_shot(data, mcp, Spy(), ks=(999,), draws=1, seed=1)
    assert out.k_used.iloc[0] == mcp.subset.nunique() - 1


def test_few_shot_k0_is_constant_across_draws():
    data, mcp, _ = synth()
    spy = Spy()
    out = few_shot(data, mcp, spy, ks=(0,), draws=4, seed=1)
    assert out.auroc.nunique() == 1 and len(spy.trained) == 1


def test_in_surface_by_subset_scores_every_row_once():
    _, mcp, _ = synth()
    out = in_surface(mcp, Spy(), by="subset")
    assert out["n"] == len(mcp) and out["folds"] == mcp.subset.nunique()


def test_in_surface_by_template_has_three_folds_and_no_server_overlap():
    _, mcp, _ = synth()
    spy = Spy()
    out = in_surface(mcp, spy, by="template", seed=3)
    assert out["folds"] == 3 and 0 < out["n"] < len(mcp)
    assert 0 <= out["auroc"] <= 1
    # each fold trains on a third-disjoint set of servers: no fold sees every server
    assert all(len(servers) < mcp.subset.nunique() for servers in spy.trained)


def test_msb_to_mcptox_reports_all_mcptox_rows():
    _, mcp, msb = synth()
    out = msb_to_mcptox(msb, mcp, Spy())
    assert out["n"] == len(mcp)


def test_score_block_binned_nan_on_tiny_set():
    scores, labels = np.linspace(0, 1, 10), np.array([0, 1] * 5)
    out = score_block(scores, labels, ["x" * i for i in range(10)], threshold=0.5)
    assert np.isnan(out["binned_auroc"]) and np.isfinite(out["auroc"])
    assert out["recall"] == 0.6 and out["fpr"] == 0.4
