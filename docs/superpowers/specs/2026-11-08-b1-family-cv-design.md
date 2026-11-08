# B1-famcv: choosing B1's settings by how well it handles attack families it has not seen

Date: 2026-11-08
Status: draft, awaiting team review
Lead: Adedapo (Seun) Odeyemi

## Goal

B1 TF-IDF picks its regularization `C` by AUROC on the train_pool val split, and sets its
deployed threshold at 1% FPR on val's 801 benign rows. Val holds the **same attack
families** as train, so this rewards memorizing each family's wording. The deployed
threshold also does not hold up:

| Eval set | Benign rows | FPR at the val threshold | 95% Wilson interval |
|---|---|---|---|
| train_pool test | 827 | 1.9% | 1.2–3.1% |
| MCPTox | 362 | 5.0% | 3.2–7.7% |
| MSB | 21 | 9.5% | 2.6–28.9% |

This spec adds a second way to select B1, **family-grouped cross-validation** ("famcv"):
choose `C` by AUROC on attack families left out of fitting. One variant also sets the
threshold from out-of-fold benign scores, about 4,900 rows instead of 801. The result is
two new B1 runs, reported next to the current one with the same harness and eval sets. Only
the selection method changes.

**What this is expected to fix, and what it is not:**
- **Expected:** an in-distribution test FPR closer to 1%, and possibly better transfer to
  BIPIA's test-only attack families.
- **Not expected:** MCPTox's 5% FPR. The 2026-11-08 analysis traced it to a genre gap. Train
  holds no benign tool descriptions, and words like "your", "reply" and "response" appear
  almost only in train attacks (43% vs 6% of rows for "your"). MCPTox benign rows score
  higher across the whole distribution (median 0.20 vs 0.06 on val), so no threshold set on
  train_pool data can bring it back to 1%. B1-famcv reports MCPTox FPR, but success is not
  judged on it.

**An honest null result is fine.** If famcv picks the same `C` as B1 (C=10 today), B1-famcv's
model is the same as B1's, and the write-up says so.

## Dates

| Date | Milestone |
|---|---|
| Nov 9–11 | Fold assignment, CV loop, harness and CLI changes, with tests |
| Nov 12 | Three B1 runs, comparison table, decision under the rule below |
| Nov 30 | Numbers frozen (shared with B2) |

This work is small (TF-IDF fits in seconds) and must not delay B2. If it slips past Nov 15,
cut it and report B1 as it is.

## Goals, non-goals, anti-goals

**Goals**
- A `--selection` option on `train.py` with three values, so all three runs go through
  the same `evaluate` and the same report columns.
- Fold assignment that never splits a BIPIA clean/attacked twin pair across folds, and never
  puts one attack family in two folds.
- A new report slice that separates seen from unseen attack families, so "does it transfer
  to unseen families" has its own number for every detector, B0 and B2 included.
- A decision rule fixed before any run (below).

**Non-goals**
- **Changing B2 DeBERTa's protocol.** The DeBERTa spec's decision 13 (selection and threshold
  from train_pool val only) stays as it is. Whether B2 adopts famcv is a separate decision,
  taken after this work reports.
- **A full LOFO table** (one model per held-out family, reported per family). It may come
  later. This spec uses family folds for selection only.
- **Extending B1's `C` grid.** C=10 sits at the top edge of `(0.1, 1.0, 10.0)`. Changing the
  grid would be a second variable. Note it in the write-up instead.
- **Fixing MCPTox FPR** (see Goal). That needs benign tool descriptions in training, for
  example B2's stream 5.
- **Confidence intervals on FPR as report columns.** Useful, but new `REPORT_COLUMNS`
  would change every baseline CSV and the B2 files `verdict.py` writes. Wilson intervals go in
  the write-up table instead.

**Anti-goals: these would make the work a failure even if the numbers look good**
- **MCPTox or MSB rows in any fold**, or used to choose `C`, the threshold, `k`, the
  bucketing or the selection method. They are scored once per reported run.
- **Choosing between selection methods by MCPTox or MSB numbers.** Picking the method that
  looks best on a holdout is the same as tuning on it. The rule below uses train_pool test
  only.
- **Adjusting `k`, the bucketing or the decision rule after seeing test numbers.**
- **Reporting only the better run.** All three runs are reported.

## The three runs

| Run (`detector` column) | `--selection` | Choose `C` by | Final model fit on | Deployed threshold from |
|---|---|---|---|---|
| `tfidf` (current B1) | `val` | val AUROC | train | val benign scores (801) |
| `tfidf-famcv` | `famcv` | mean held-out AUROC over 5 family folds **of train** | train | val benign scores (801) |
| `tfidf-famcv-oof` | `famcv-oof` | mean held-out AUROC over 5 family folds **of train + val** | train + val | pooled out-of-fold benign scores of the chosen `C` (about 4,900) |

`tfidf-famcv` changes only how `C` is chosen. Val stays untouched, so its threshold is set
exactly as in B1. `tfidf-famcv-oof` also changes the threshold and the fit data. The three
runs separate the two effects.

## Decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Folds | `k = 5` |
| 2 | Unit of a fold | The `group`. Every row in a group gets the same fold. Every BIPIA train_pool group holds one clean and one attacked row, so a held-out family's benign twins are held out with it. No train_pool group holds more than one attack family (checked against the parquet) |
| 3 | Sources with more than one attack family (BIPIA: 25 families in train + val) | Families are put in buckets greedily. Take families largest first (attack row count, ties by name), and put each in the bucket with the fewest attack rows so far (ties to the lowest bucket index). Deterministic, with no seed. About 486 BIPIA attacks per bucket; the 15 large families (143–173 rows each) land 3 per bucket. The 10 small families (2–9 rows each) fill the gaps |
| 4 | Sources with one attack family (PoisonedRAG, 254 groups) | Groups are shuffled with seed `7267` and dealt round-robin over the 5 folds. Every fold then also holds out some PoisonedRAG, so every row gets an out-of-fold score |
| 5 | Groups with no attack row | Treated like decision 4. None exist in train_pool today |
| 6 | Selection metric | Mean of per-fold held-out AUROC over the 5 folds. This is B1's metric (AUROC), so the only change is which rows decide. Ties go to the smaller `C`, as `np.nanargmax` does in B1 today (Claude's call) |
| 7 | Out-of-fold threshold (`famcv-oof`) | Collect the chosen `C`'s held-out scores from all 5 folds, keep the benign ones, and apply the existing `threshold_at_fpr(…, 0.01)`. Fold models are trained on about 80% of the data, so their scores can sit slightly lower than the refit model's. Accepted: logistic regression on the same features shifts little, and the in-distribution test FPR shows any effect |
| 8 | Seen and unseen family slices | For each eval set and source whose attacks include **both** a family seen in train + val and one not seen, add `<source>_family=seen` and `<source>_family=unseen` (for example `bipia_family=unseen`), each paired with that source's benign rows. No comma in the name, because MLflow metric keys cannot contain one. "Seen" means in any train or val attack row, for every run, so the slices match across runs. Today this adds two rows on the test split for BIPIA: 357 unseen-family attacks and 20 seen (Language Translation), against 377 benign. The seen slice is small and labeled indicative. MCPTox and MSB have no seen families, so they get no new slice |
| 9 | Detector name | Set on the instance: `tfidf`, `tfidf-famcv`, `tfidf-famcv-oof`. It fills the report's `detector` column and the MLflow run name |
| 10 | What is logged | Params: `selection`, `k`, `fold_seed`, the chosen `C`, its mean CV AUROC, and the threshold. Artifacts: `cv_table.csv` (C × fold: held-out AUROC, benign and attack counts) and `folds.csv` (family or group → fold). Fold models are not saved (Claude's call) |
| 11 | Detectors that support famcv | TF-IDF only. `--selection famcv*` with any other detector is an argparse error |
| 12 | Report files | `report/baselines/b1_tfidf_lr_famcv.csv` and `report/baselines/b1_tfidf_lr_famcv_oof.csv`, with the same columns as `b1_tfidf_lr.csv`. B0 and B1 CSVs are regenerated once, to gain the seen/unseen slices |

## Decision rule (fixed before any run)

Measured on the **train_pool test split only**:

1. **Adopt famcv for choosing `C`** if `tfidf-famcv`'s AUROC on `bipia_family=unseen`
   is at least B1's, and its test `all` AUROC is no more than 0.01 below B1's.
2. **Adopt the out-of-fold threshold** if `tfidf-famcv-oof`'s test `all` FPR is closer to 1%
   than `tfidf-famcv`'s, and rule 1's AUROC conditions also hold for `tfidf-famcv-oof`
   against B1.
3. Anything else: B1 stays as it is, and the result is reported as a negative or mixed finding.

These are point-estimate comparisons. The CIs will overlap, because the test split has 827
benign rows and the unseen slice has 357 attacks. The write-up states that "adopted" means
"preferred for this project", not "significantly better". MCPTox and MSB numbers are
reported for every run and never enter the rule.

Using train_pool test to choose between selection methods spends a little of its
independence. That is accepted because the holdouts stay clean, and it is stated in the
write-up.

## Interface

`src/eval/family_cv.py` (new):

```python
def family_folds(frame: pd.DataFrame, k: int = 5, seed: int = 7267) -> np.ndarray:
    """One fold index in [0, k) per row of frame, following decisions 2–5. Needs the
    `source`, `label`, `attack_family` and `group` columns. Raises ValueError if a group
    holds two attack families, or if any fold ends up with only one label."""

@dataclass
class CVResult:
    best: float               # the chosen parameter value
    mean_auroc: float         # its mean held-out AUROC
    table: pd.DataFrame       # param, fold, auroc, n_benign, n_adversarial
    oof_scores: np.ndarray    # the chosen parameter's held-out score for every row

def cross_validate(make_detector, grid, frame, folds) -> CVResult:
    """Fit make_detector(value) on every fold's training rows for every grid value, and score
    that fold's held-out rows. Pick by decision 6."""
```

`src/eval/harness.py`:
- `COLUMNS` gains `attack_family`.
- `slices(frame, seen_families=None)` yields the decision 8 slices when `seen_families` is given.
- `evaluate(detector, data, target_fpr=0.01, n_boot=1000, threshold=None)` uses `threshold`
  when given, and otherwise sets it on val as today. It passes the train + val attack
  families to `slices`.

`src/models/train.py`:
- `fit_detector(name, data, selection="val", ...)` gains the `famcv` and `famcv-oof` branches
  for TF-IDF. For `famcv-oof`, `params["threshold"]` holds the out-of-fold threshold, and
  `main` passes it to `evaluate`.
- `--selection {val,famcv,famcv-oof}` (default `val`) and `--folds` (default 5).
- `log_run` writes decision 10's params and artifacts.

`Makefile`: `make evaluate DETECTOR=tfidf SELECTION=famcv-oof OUT=…`.

## Known risks

- **famcv picks C=10, the same as B1.** Then `tfidf-famcv` equals B1, and only
  `tfidf-famcv-oof`'s threshold differs. That is still a result. Report it.
- **One fold has few benign rows of one source.** Folds balance attack rows, and benign twins
  follow them, so BIPIA benign counts match BIPIA attack counts per fold. PoisonedRAG is dealt
  by group. `family_folds` raises if a fold lacks a label.
- **The out-of-fold threshold is biased low** (decision 7). If test FPR under `famcv-oof` comes
  out *above* `famcv`'s, this is the first suspect. Report it rather than tune it.
- **B2's verdict reads `b1_tfidf_lr.csv`.** `src/eval/verdict.py` takes B1's bar from its
  `mcptox`/`all` row. Regenerating the file adds slice rows but must leave that row unchanged,
  and the famcv runs are written to their own files, so B1's bar for B2 does not move.

## Write-up deliverables

- One table: B1, B1-famcv and B1-famcv-oof on test (`all`, BIPIA unseen and seen families),
  MCPTox and MSB. Columns: AUROC with CI, TPR, and FPR with a Wilson interval.
- The chosen `C` and mean CV AUROC per run, and the per-fold table as an appendix.
- The outcome under the decision rule: adopted, not adopted, or mixed.
- One sentence on MCPTox FPR: still about 5% (if so), with the genre-gap explanation.
