import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "eval"))

from shortcut_probe import bootstrap_ci, tpr_at_fpr  # noqa: E402
from shortcut_probe import (BagOfWordsProbe, CueProbe, gate, label_shuffle, length_matched_auroc,  # noqa: E402
                            probe_battery, source_identity, split_leaks)
from sklearn.metrics import roc_auc_score  # noqa: E402
from synth import make  # noqa: E402


@pytest.fixture(scope="module")
def data():
    df = make()
    g = df.group.str.extract(r"-[ab](\d+)")[0].astype(int)
    split = np.where(g % 5 == 0, "test", "train")        # group-disjoint, same index shared by a/b twins
    return df[split == "train"].reset_index(drop=True), df[split == "test"].reset_index(drop=True)


# --- metrics ---
def test_tpr_at_fpr_perfect_and_random():
    y = np.array([0] * 100 + [1] * 100)
    assert tpr_at_fpr(y, np.r_[np.zeros(100), np.ones(100)], 0.01) == 1.0
    rng = np.random.default_rng(0)
    assert tpr_at_fpr(y, rng.random(200), 0.01) < 0.2


def test_tpr_at_fpr_never_exceeds_budget():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.95])  # one benign scores above one adversarial
    assert tpr_at_fpr(y, s, 0.0) == pytest.approx(0.5)  # only the 0.95 clears every benign score


def test_group_bootstrap_wider_than_row_bootstrap():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 10)
    y = np.repeat(np.tile([0, 1], 20), 10)                      # each group has one label
    s = y * 0.3 + np.repeat(rng.normal(0, 1, 40), 10) + rng.normal(0, 0.1, 400)  # group-shared noise
    _, lo_g, hi_g = bootstrap_ci(roc_auc_score, y, s, groups, n_boot=300)
    _, lo_r, hi_r = bootstrap_ci(roc_auc_score, y, s, None, n_boot=300)
    assert hi_g - lo_g > hi_r - lo_r


# --- probes ---
def test_length_probe_learns_wrong_direction_on_reversed_source(data):
    train, test = data
    table = probe_battery(train, {"pr": test[test.source == "poisonedrag"]}, [CueProbe("length_only", ["log_length"])])
    fitted = table[table["mode"] == "fitted"].iloc[0]
    oracle = table[table["mode"] == "oracle"].iloc[0]
    assert fitted.auroc < 0.4           # pooled fit points the wrong way here (as in the data card: 0.138)
    assert oracle.auroc > 0.85          # but length does separate this source when fit inside it


def test_label_shuffle_is_chance(data):
    train, test = data
    aurocs = label_shuffle(BagOfWordsProbe, train, test, n_rounds=3)
    assert 0.35 < np.mean(aurocs) < 0.65


def test_label_shuffle_flags_a_label_derived_feature(data):
    """A detector reading a label-derived marker keeps its AUROC when training labels are
    shuffled, because its score never depended on what it learned. The control must flag that."""
    train, test = data

    class Peeker:
        def fit(self, texts, labels):
            return self

        def predict_proba(self, texts):
            return np.array([float("[ADV]" in t) for t in texts])

    marked = test.assign(text=np.where(test.label == 1, test.text + " [ADV]", test.text))
    assert np.mean(label_shuffle(Peeker, train, marked, n_rounds=2)) > 0.9


# --- length matching ---
def test_length_matched_exposes_a_cheater_and_spares_an_honest_model(data):
    train, test = data
    cheat = CueProbe("len", ["log_length"]).fit(train.text, train.label)
    honest = BagOfWordsProbe().fit(train.text, train.label)
    c_full, c_match, *_ = length_matched_auroc(test, cheat.predict_proba(test.text))
    h_full, h_match, kept, resid = length_matched_auroc(test, honest.predict_proba(test.text))
    assert resid < 0.62                     # matching actually removed the length signal
    assert c_match < 0.6                    # cheater collapses toward chance
    assert h_match > 0.75                   # honest model keeps the content signal


# --- leaks ---
def test_split_leaks_zero_on_clean_split_and_flagged_on_dirty(data):
    train, test = data
    clean = split_leaks(train, test.iloc[:50])
    assert clean["shared_groups"] == 0 and clean["exact_text_overlap"] == 0
    dirty = split_leaks(train, pd.concat([test.iloc[:20], train.iloc[:5]]))
    assert dirty["exact_text_overlap"] >= 5 and dirty["shared_groups"] >= 5


def test_source_identity_runs(data):
    train, test = data
    out = source_identity(pd.concat([train, test]))
    assert 0 <= out["accuracy"] <= 1 and out["majority_rate"] == pytest.approx(0.25, abs=0.01)


# --- the gate ---
def test_gate_separates_cheater_from_honest_model(data):
    train, test = data
    sets = {"in-dist": test}
    table = probe_battery(train, sets, [CueProbe("length_only", ["log_length"])])
    cheat = CueProbe("len2", ["log_length", "word_count"]).fit(train.text, train.label)
    honest = BagOfWordsProbe().fit(train.text, train.label)
    verdict_c, _ = gate({"in-dist": (test, cheat.predict_proba(test.text))}, table)
    verdict_h, detail = gate({"in-dist": (test, honest.predict_proba(test.text))}, table)
    assert verdict_c == "FAIL"
    assert verdict_h == "PASS", detail


def test_gate_refuses_to_pass_when_length_already_saturates_the_eval_set(data):
    """Synthetic MCPTox is separable by length alone (probe ~1.0). No detector can beat that by a
    margin, and length matching has almost nothing to match. The gate must say so, not pass."""
    train, test = data
    part = test[test.source == "mcptox"]
    table = probe_battery(train, {"mcptox": part}, [CueProbe("length_only", ["log_length"])])
    honest = BagOfWordsProbe().fit(train.text, train.label)
    verdict, detail = gate({"mcptox": (part, honest.predict_proba(part.text))}, table)
    assert verdict == "FAIL" and detail.rows_matched.iloc[0] < 20


# ---- PR #40 review fixes -----------------------------------------------------------------------

def _probe_table(eval_set, his):
    return pd.DataFrame([{"probe": f"p{i}", "family": "cue", "eval_set": eval_set, "mode": "fitted",
                          "auroc": 0.6, "auroc_lo": 0.5, "auroc_hi": hi} for i, hi in enumerate(his)])


def _separable_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    text = ["x" * int(rng.integers(20, 200)) for _ in y]          # length independent of label
    frame = pd.DataFrame({"text": text, "label": y, "source": "s", "group": np.arange(n)})
    return frame, y + rng.normal(0, 0.3, n)


def test_gate_ignores_probe_with_nan_ci():
    """A NaN auroc_hi used to sort last and be picked as 'best', turning into a misleading FAIL."""
    from shortcut_probe import gate
    frame, scores = _separable_frame()
    verdict, table = gate({"t": (frame, scores)}, _probe_table("t", [0.6, np.nan]))
    assert table.best_probe.item() == "p0" and table.beats_probe.item() is True
    assert verdict == "PASS"


def test_gate_is_inconclusive_when_no_probe_ci_is_usable():
    from shortcut_probe import gate
    frame, scores = _separable_frame()
    verdict, table = gate({"t": (frame, scores)}, _probe_table("t", [np.nan]))
    assert table.beats_probe.item() is None and verdict == "INCONCLUSIVE"


def test_length_match_needs_min_rows():
    from shortcut_probe import MIN_MATCHED_ROWS, length_matched_auroc
    frame, scores = _separable_frame(n=MIN_MATCHED_ROWS - 20)
    assert np.isnan(length_matched_auroc(frame, scores)[1])
    frame, scores = _separable_frame(n=MIN_MATCHED_ROWS * 3)
    assert not np.isnan(length_matched_auroc(frame, scores)[1])
