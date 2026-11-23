"""Shortcut probe: is a detector cheating, and can this report be trusted?

Five checks, each answering one question. `gate()` combines them into the Phase 4 go/no-go.

  1. probe_battery      How much do trivial cues (length, punctuation, ...) and bag-of-words earn
                        on each eval set? This is the score a detector must beat.
  2. label_shuffle      Does the pipeline score ~0.5 when labels are random? (leak / bug detector)
  3. length_matched     Does a detector's AUROC survive when length no longer separates classes?
  4. split_leaks        Do groups, exact texts or near-duplicates cross train and test?
  5. source_identity    Can a model tell WHICH SOURCE a text is from? (Fomin 2026 dataset-identity probe)

Frames are pandas DataFrames with columns: text, label, source, group (and split/role when used).
Run on the real table:  PYTHONPATH=src/data:src/eval uv run python src/eval/shortcut_probe.py
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def bootstrap_ci(metric, y, scores, groups, n_boot=300, seed=7267, level=0.95):
    """(point, lo, hi): percentile CI that resamples whole GROUPS (rows from one group are
    near-duplicates, so resampling rows would make the interval too narrow). Replicates with
    one label are skipped. NaN interval if too few valid replicates."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)   # None = plain row bootstrap
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(metric(y, scores))
    ids = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = np.concatenate([index[g] for g in rng.choice(ids, len(ids))])
        if len(np.unique(y[pick])) == 2:
            stats.append(metric(y[pick], scores[pick]))
    if len(stats) < 20:
        return point, float("nan"), float("nan")
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(stats, [tail, 100 - tail])
    return point, float(lo), float(hi)


def tpr_at_fpr(y, scores, fpr=0.05):
    """TPR when the threshold is set on THESE rows so at most `fpr` of benign rows are flagged.
    (The team harness uses a val-calibrated threshold instead; this one is only a probe
    comparison number.)"""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    benign = np.sort(scores[y == 0])[::-1]
    if len(benign) == 0 or (y == 1).sum() == 0:
        return float("nan")
    thr = benign[int(np.floor(fpr * len(benign) + 1e-9))] if int(np.floor(fpr * len(benign) + 1e-9)) < len(benign) else benign[-1]
    return float((scores[y == 1] > thr).mean())

# ---- 1. trivial-cue probes -------------------------------------------------------------------

CUES = {
    "log_length": lambda t: np.log1p(len(t)),
    "word_count": lambda t: np.log1p(len(t.split())),
    "punct_rate": lambda t: len(re.findall(r"[^\w\s]", t)) / max(len(t), 1),
    "upper_rate": lambda t: len(re.findall(r"[A-Z]", t)) / max(len(t), 1),
    "digit_rate": lambda t: len(re.findall(r"\d", t)) / max(len(t), 1),
    "newline_count": lambda t: np.log1p(t.count("\n")),
    "ends_with_period": lambda t: float(t.rstrip().endswith((".", "!", "?"))),
}


def cue_matrix(texts, names=None):
    names = names or list(CUES)
    return np.array([[CUES[n](t) for n in names] for t in texts], dtype=float)


class CueProbe:
    """Logistic regression on a handful of hand-made cues. `names` picks which cues it may see."""

    def __init__(self, name, names):
        self.name, self.names = name, names
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    def fit(self, texts, labels):
        self.model.fit(cue_matrix(texts, self.names), labels)
        return self

    def predict_proba(self, texts):
        return self.model.predict_proba(cue_matrix(texts, self.names))[:, 1]


class BagOfWordsProbe:
    """TF-IDF word 1-2grams + logistic regression. Not a 'cheat' by itself, but if it already
    matches your fancy model, the benchmark does not need the fancy model."""

    name = "bag_of_words"
    family = "lexical"   # context for the reader; NOT a shortcut, so the gate does not compare against it

    def __init__(self):
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50_000),
            LogisticRegression(max_iter=1000, C=1.0),
        )

    def fit(self, texts, labels):
        self.model.fit(list(texts), labels)
        return self

    def predict_proba(self, texts):
        return self.model.predict_proba(list(texts))[:, 1]


def default_probes():
    return [
        CueProbe("length_only", ["log_length"]),
        CueProbe("all_cues", list(CUES)),
        BagOfWordsProbe(),
    ]


def _score(part, scores, fpr):
    auroc, low, high = bootstrap_ci(roc_auc_score, part.label, scores, part.group, n_boot=300)
    return {
        "auroc": auroc, "auroc_lo": low, "auroc_hi": high,
        "tpr@fpr": tpr_at_fpr(part.label, scores, fpr), "n": len(part),
    }


def probe_battery(train, eval_sets, probes=None, fpr=0.05):
    """Fit every probe on `train`, score each eval set. Returns a long DataFrame.

    Also reports `oracle` rows: the same probe fit INSIDE each eval set's own rows (5-fold by
    group). That is an upper bound on what the cue could ever earn there, whatever direction
    the training pool taught it."""
    probes = probes or default_probes()
    rows = []
    for probe in probes:
        probe.fit(train.text, train.label)
        for set_name, part in eval_sets.items():
            if part.label.nunique() < 2:
                continue
            rows.append({"probe": probe.name, "family": getattr(probe, "family", "cue"), "eval_set": set_name, "mode": "fitted",
                         **_score(part, probe.predict_proba(part.text), fpr)})
            folds = min(5, part.group.nunique())
            if folds >= 2:
                oof = out_of_fold(probe, part, folds)
                rows.append({"probe": probe.name, "family": getattr(probe, "family", "cue"), "eval_set": set_name, "mode": "oracle",
                             **_score(part, oof, fpr)})
    return pd.DataFrame(rows)


def out_of_fold(probe, part, folds):
    """Out-of-fold scores: refit a copy of the probe on each group-disjoint fold."""
    text, y = np.asarray(part.text), part.label.to_numpy()
    out = np.zeros(len(part))
    for tr, te in GroupKFold(folds).split(text, y, part.group.to_numpy()):
        if len(np.unique(y[tr])) < 2:
            out[te] = 0.5
            continue
        out[te] = copy.deepcopy(probe).fit(text[tr], y[tr]).predict_proba(text[te])
    return out


# ---- 2. label-shuffle control ----------------------------------------------------------------

def label_shuffle(make_detector, train, test, n_rounds=5, seed=7267):
    """Refit on train with labels permuted across individual ROWS, score test with the true labels.
    A sound pipeline gives AUROC ~ 0.5. Much higher means test info leaks into training or
    the evaluation code is wrong. Returns the list of AUROCs, one per round."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rounds):
        shuffled = train.label.to_numpy().copy()
        rng.shuffle(shuffled)
        det = make_detector().fit(train.text, shuffled)
        out.append(float(roc_auc_score(test.label, det.predict_proba(test.text))))
    return out


# ---- 3. length-matched evaluation -------------------------------------------------------------

MIN_MATCHED_ROWS = 100   # below this, a length-matched AUROC is too noisy to call pass or fail

def length_matched_mask(df, n_bins=10, seed=7267):
    """Boolean mask keeping, inside each source, equal numbers of benign and adversarial rows
    per length decile (deciles of that source's lengths). Length then carries ~no signal, so a
    detector's AUROC on the kept rows is what it earns from content."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(df), dtype=bool)
    lengths = df.text.str.len().to_numpy()
    for _, idx in df.groupby("source").indices.items():
        sub = df.iloc[idx]
        if sub.label.nunique() < 2:
            continue
        edges = np.unique(np.quantile(lengths[idx], np.linspace(0, 1, n_bins + 1)))
        bins = np.clip(np.searchsorted(edges, lengths[idx], side="right") - 1, 0, max(len(edges) - 2, 0))
        for b in np.unique(bins):
            in_bin = idx[bins == b]
            pos = in_bin[df.label.to_numpy()[in_bin] == 1]
            neg = in_bin[df.label.to_numpy()[in_bin] == 0]
            k = min(len(pos), len(neg))
            if k:
                keep[rng.choice(pos, k, replace=False)] = True
                keep[rng.choice(neg, k, replace=False)] = True
    return keep


def length_matched_auroc(df, scores, seed=7267, min_rows=MIN_MATCHED_ROWS):
    """(AUROC on all rows, AUROC on length-matched rows, rows kept, residual length separability).
    The matched AUROC is NaN when fewer than `min_rows` rows can be matched."""
    scores = np.asarray(scores)
    mask = length_matched_mask(df, seed=seed)
    if mask.sum() < min_rows or df.label[mask].nunique() < 2:
        return float("nan"), float("nan"), int(mask.sum()), float("nan")
    full = roc_auc_score(df.label, scores)
    matched = roc_auc_score(df.label[mask], scores[mask])
    resid = roc_auc_score(df.label[mask], df.text[mask].str.len())
    return float(full), float(matched), int(mask.sum()), float(max(resid, 1 - resid))


# ---- 4. split leaks ---------------------------------------------------------------------------

def split_leaks(train, test, near_threshold=0.9):
    """Counts of ways test rows can be seen in training. All should be 0."""
    exact = int(test.text.isin(set(train.text)).sum())
    shared_groups = len(set(train.group) & set(test.group))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=1, sublinear_tf=True)
    a = vec.fit_transform(train.text)
    b = vec.transform(test.text)
    dist, _ = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(a).kneighbors(b)
    near = int((1 - dist[:, 0] >= near_threshold).sum())
    return {"exact_text_overlap": exact, "shared_groups": shared_groups,
            f"near_duplicates(cos>={near_threshold})": near}


# ---- 5. source-identity probe -----------------------------------------------------------------

def source_identity(df, seed=7267):
    """Grouped 5-fold accuracy of predicting `source` from text. Near 1.0 means a detector can
    shortcut through 'which benchmark is this?' (Fomin 2026 reports 96.6% on theirs). Compared
    with the majority-class rate so the number is interpretable."""
    folds = min(5, df.group.nunique())
    pred = cross_val_predict(
        BagOfWordsProbeMulti(), list(df.text), df.source.to_numpy(),
        groups=df.group.to_numpy(), cv=GroupKFold(folds),
    )
    acc = float((pred == df.source.to_numpy()).mean())
    return {"accuracy": acc, "majority_rate": float(df.source.value_counts(normalize=True).iloc[0])}


class BagOfWordsProbeMulti(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.m_ = make_pipeline(TfidfVectorizer(min_df=2, sublinear_tf=True), LogisticRegression(max_iter=1000))
        self.m_.fit(list(X), y)
        self.classes_ = self.m_[-1].classes_
        return self

    def predict(self, X):
        return self.m_.predict(list(X))


# ---- the gate -------------------------------------------------------------------------------

def gate(detector_scores: dict, probe_table: pd.DataFrame, margin=0.05, min_matched_ratio=0.8):
    """Phase 4 go/no-go for ONE detector. Returns ("PASS" | "FAIL" | "INCONCLUSIVE", table).

    detector_scores: {eval_set_name: (frame, scores)}. On every eval set:
      (a) detector AUROC CI-lower must exceed the best FITTED cue probe's CI-upper by `margin`
          (bag-of-words is reported but is not a shortcut, so it is not compared against);
      (b) on length-matched rows the detector must keep >= min_matched_ratio of its excess
          over 0.5. If too few rows can be matched (classes barely overlap in length) the
          check cannot be run: that is INCONCLUSIVE, not a pass, and not silently a fail.
    A probe whose CI is NaN (too few valid bootstrap replicates) is not used as the bar; if no
    probe or the detector itself has a usable CI, check (a) cannot run and is INCONCLUSIVE.
    FAIL if any check fails; else INCONCLUSIVE if any could not run; else PASS."""
    rows = []
    for name, (frame, scores) in detector_scores.items():
        auroc, lo, hi = bootstrap_ci(roc_auc_score, frame.label, scores, frame.group, n_boot=300)
        fitted = probe_table[(probe_table.eval_set == name) & (probe_table["mode"] == "fitted")
                             & (probe_table.family == "cue")]
        usable = fitted.dropna(subset=["auroc_hi"])
        if len(usable) and not np.isnan(lo):
            best = usable.sort_values("auroc_hi").iloc[-1]
            best_probe, best_hi, beats = best.probe, best.auroc_hi, bool(lo > best.auroc_hi + margin)
        else:
            best_probe, best_hi, beats = None, float("nan"), None
        full, matched, kept, resid = length_matched_auroc(frame, scores)
        keeps = None if np.isnan(matched) else bool((matched - 0.5) >= min_matched_ratio * (full - 0.5))
        rows.append({"eval_set": name, "detector_auroc": auroc, "ci_lo": lo,
                     "best_probe": best_probe, "probe_auroc_hi": best_hi,
                     "beats_probe": beats,
                     "length_matched_auroc": matched, "rows_matched": kept, "survives_length_match": keeps})
    table = pd.DataFrame(rows).astype({"beats_probe": object, "survives_length_match": object})
    if (table.beats_probe == False).any() or (table.survives_length_match == False).any():  # noqa: E712
        return "FAIL", table
    if table.beats_probe.isna().any() or table.survives_length_match.isna().any():
        return "INCONCLUSIVE", table
    return "PASS", table


# ---- 6. padding invariance ---------------------------------------------------------------------

def padding_invariance(detector, frame, seed=7267):
    """Counterfactual length test. Append a different benign text from the SAME source to every
    benign row (about doubling its length, adding no injection content) and re-score. A detector
    reading content should not care; one reading length moves. Returns
    (AUROC original, AUROC padded, |difference|)."""
    rng = np.random.default_rng(seed)
    padded = frame.copy()
    for _, idx in frame.groupby("source").indices.items():
        sub = frame.iloc[idx]
        benign = idx[sub.label.to_numpy() == 0]
        if len(benign) < 2:
            continue
        filler = frame.text.to_numpy()[rng.permutation(benign)]
        for row, extra in zip(benign, filler):
            padded.iat[row, frame.columns.get_loc("text")] = frame.text.iat[row] + "\n" + extra
    before = roc_auc_score(frame.label, detector.predict_proba(frame.text))
    after = roc_auc_score(padded.label, detector.predict_proba(padded.text))
    return float(before), float(after), float(abs(before - after))


# ---- 7. one call for everything ---------------------------------------------------------------

# Defaults agreed for now (MARGIN / MIN_RETAINED from the gate; the other three are new, see notes).
SHUFFLE_MAX = 0.60      # mean AUROC allowed after label shuffling
PAD_MAX_DELTA = 0.10    # max |AUROC change| when benign rows are lengthened
LEAK_MAX = 0            # exact-text or shared-group overlap allowed between train and test


@dataclass
class ShortcutReport:
    verdict: str                      # PASS | FAIL | INCONCLUSIVE
    reasons: list                     # why, in plain words
    gate_table: pd.DataFrame
    probe_table: pd.DataFrame
    shuffle_aurocs: list | None
    padding: dict                     # eval set -> (before, after, delta)
    leaks: dict

    def metrics(self) -> dict:
        """Flat numeric dict, ready for mlflow.log_metrics."""
        out = {"shortcut/shuffle_auroc_mean": float(np.mean(self.shuffle_aurocs)) if self.shuffle_aurocs else np.nan,
               "shortcut/exact_text_overlap": self.leaks["exact_text_overlap"],
               "shortcut/shared_groups": self.leaks["shared_groups"]}
        for row in self.gate_table.itertuples():
            out[f"shortcut/{row.eval_set.strip()}/length_matched_auroc"] = row.length_matched_auroc
            out[f"shortcut/{row.eval_set.strip()}/best_probe_auroc_hi"] = row.probe_auroc_hi
        for name, (_, _, delta) in self.padding.items():
            out[f"shortcut/{name.strip()}/padding_delta"] = delta
        return out

    def summary(self) -> str:
        lines = [f"Shortcut check: {self.verdict}", *[f"  - {r}" for r in self.reasons],
                 "A PASS means these checks found no shortcut. It is not proof that none exists."]
        return "\n".join(lines)


def check_detector(detector, train, eval_sets, make_detector=None, probe_table=None,
                   leak_against="in-dist", shuffle_rounds=3, can_rescore=True) -> ShortcutReport:
    """Run every shortcut check on one fitted detector.

    detector      fitted object with predict_proba(texts) -> P(adversarial)
    train         training frame (text, label, source, group), the rows the detector was fit on
    eval_sets     {name: frame}; the frame named `leak_against` is also used for the leak and
                  label-shuffle checks (it must be a train_pool test split, not a holdout)
    make_detector zero-argument factory for an UNFITTED copy, needed for the label-shuffle check
    can_rescore   False for precomputed scores (B2 score files): they only cover the dataset's own
                  texts, so the padding check, which scores new padded texts, is skipped and said so
    """
    probe_table = probe_table if probe_table is not None else probe_battery(train, eval_sets)
    scored = {k: (v, detector.predict_proba(v.text)) for k, v in eval_sets.items() if v.label.nunique() == 2}
    verdict, gate_table = gate(scored, probe_table)
    reasons = []
    for r in gate_table.itertuples():
        if r.beats_probe is None:
            reasons.append(f"{r.eval_set}: no usable bootstrap CI for the detector or the cue probes; "
                           "probe comparison not run")
        elif not r.beats_probe:
            reasons.append(f"{r.eval_set}: AUROC CI-lower {r.ci_lo:.3f} does not beat best probe "
                           f"({r.best_probe}, CI-upper {r.probe_auroc_hi:.3f}) by the margin")
        if r.survives_length_match is False:
            reasons.append(f"{r.eval_set}: AUROC falls to {r.length_matched_auroc:.3f} on length-matched rows")
        if r.survives_length_match is None:
            reasons.append(f"{r.eval_set}: only {r.rows_matched} rows could be length-matched "
                           f"(need {MIN_MATCHED_ROWS}); check not run")

    shuffle = None
    test = eval_sets.get(leak_against)
    if make_detector is not None and test is not None:
        shuffle = label_shuffle(make_detector, train, test, n_rounds=shuffle_rounds)
        if np.mean(shuffle) > SHUFFLE_MAX:
            verdict = "FAIL"
            reasons.append(f"label shuffle: mean AUROC {np.mean(shuffle):.3f} > {SHUFFLE_MAX} "
                           "(score does not depend on learned labels)")
    elif make_detector is None:
        reasons.append("label shuffle skipped: no make_detector given")

    padding = {}
    if not can_rescore:
        reasons.append("padding check skipped: precomputed scores cannot score new (padded) texts")
    for name, frame in (scored.items() if can_rescore else ()):
        padding[name] = padding_invariance(detector, frame[0])
        if padding[name][2] > PAD_MAX_DELTA:
            verdict = "FAIL"
            reasons.append(f"{name}: AUROC moves {padding[name][2]:.3f} when benign rows are lengthened")

    leaks = split_leaks(train, test) if test is not None else {"exact_text_overlap": 0, "shared_groups": 0}
    if leaks["exact_text_overlap"] > LEAK_MAX or leaks["shared_groups"] > LEAK_MAX:
        verdict = "FAIL"
        reasons.append(f"train/test leak: {leaks}")

    if not reasons:
        reasons.append("every check within thresholds")
    return ShortcutReport(verdict, reasons, gate_table, probe_table, shuffle, padding, leaks)


# ---- command line on the real table ----------------------------------------------------------

def main(argv=None):
    import argparse
    from dataset import DEFAULT_PATH, load
    ap = argparse.ArgumentParser(description="Probe battery on the real table (no detector needed).")
    ap.add_argument("--data", default=DEFAULT_PATH)
    path = ap.parse_args(argv).data
    cols = ["text", "label", "source", "group"]

    def rows(purpose, **kw):
        return load(purpose, path=path, **kw).select(cols).to_pandas()

    train = rows("fit", roles=["train_pool"], splits=["train"])
    test = rows("eval", roles=["train_pool"], splits=["test"])
    eval_sets = {"in-dist test": test, "LOSO mcptox": rows("eval", sources=["mcptox"]),
                 "transfer msb": rows("eval", sources=["msb"])}
    eval_sets.update({f"  {s}": p for s, p in test.groupby("source")})

    table = probe_battery(train, eval_sets)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print("\nlabel shuffle AUROCs:", np.round(label_shuffle(BagOfWordsProbe, train, test), 3))
    print("split leaks:", split_leaks(train, test))
    print("source identity:", source_identity(pd.concat([train, test])))


if __name__ == "__main__":
    main()
