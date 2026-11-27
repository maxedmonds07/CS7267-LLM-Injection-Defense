# Review Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the deck-v2 review to the paper and deck, and add the follow-up analyses
(review experiments 1, 2, 3, 4, 5-script, 7) before 2026-11-30.

**Architecture:** Two new analysis scripts in `src/eval/` following the Phase 5 pattern (pure
functions + `main()`, a `make` target, outputs in `report/followup/`), one Colab scoring script,
then text edits to `report/final/report.md` and the Slides artifact. Experiments take the fit
function as a parameter so tests run on `tests/synth.py` data with a fast detector.

**Tech Stack:** Python via `uv`, pandas, numpy, scipy (Spearman), scikit-learn, matplotlib,
pandoc/tectonic for the PDF, Claude Slides artifact for the deck.

**Spec:** `docs/superpowers/specs/2026-11-26-review-followup-design.md`; the review itself is
`docs/Review Detectors Learn a Voice, Not a Behavior (deck v2).md` (numbers and slide fixes
are quoted from it).

## Global Constraints

- One branch per task. Task 1 branches from `origin/main`; each later task branches from the
  previous task's branch until that one is merged (all touch `report/final/report.md`).
- No `Co-Authored-By` or other AI trailers in commits. Never push, open PRs or `dvc push`; the
  user does.
- Run Python as `PYTHONPATH=src/data:src/eval:src/models uv run …` (the Makefile's `SRC_PATH`).
- Every MCPTox-reading result is labelled "follow-up analysis outside the pre-registered protocol".
- MCPTox sampling and folds are by server (`subset` column; 45 servers), never by row.
- Deployed threshold = 1% FPR on val benign rows (`report/phase5/thresholds.json`), unless a
  step says otherwise.
- Wording only for metrics: do not re-sort tables around length-binned AUROC.

## Review Focus

1. Servers with one label (MCPTox has servers with only benign or only attack rows): fold metrics
   must not crash; pooled out-of-fold scores are scored once at the end.
2. Few-shot draws with k larger than the number of servers left (k=20 of 45 is fine; synthetic
   data has fewer): cap k at `n_servers - 1` and record the cap, don't raise.
3. `length_binned_auroc` returning NaN on small test sets (fewer than 40 rows or <2 usable bins):
   keep NaN in CSVs, write "n/a" in paper tables, never 0.
4. B2 score columns with ties at 0 (98–100% of MCP rows below 0.01): Spearman uses average ranks;
   the long-benign restriction keeps benign rows with `n_chars >= min(attack n_chars)`.
5. Prompt Guard texts longer than 512 tokens: truncate, and record `truncated` per row.

---

### Task 1: Paper wording fixes

**Files:**
- Modify: `report/final/report.md`
- Test: `tests/test_final_report.py` (existing), `make check-final`

No code. Each item below is a find-and-rewrite in `report.md` (grep the quoted phrase):

- [ ] **Step 1:** B2 "noise" (abstract line ~25 and every other occurrence: `grep -n "noise among\|is noise\|below chance" report/final/report.md`) → B2 "ranks MCP rows by inverse length". Keep a placeholder sentence "(Spearman with length: see §follow-up)" to be filled by Task 2.
- [ ] **Step 2:** Capacity claim ("reverse of capacity", "more capacity") → "the highest in-distribution score did not predict transfer"; add one clause that B1 and B2 differ in loss, pooling, tokenization and case handling, so capacity is not isolated.
- [ ] **Step 3:** Taxonomy share: replace the "59% of misses" framing with "the cue is in 50.7% of attacks; B1-Optuna misses 75% of procedural-prerequisite attacks (185 of 246) against 55% of the rest (131 of 239)". Add that the explanatory regexes were written after reading holdout errors and feed no detector.
- [ ] **Step 4:** Relevance features "lower" AUROC → "do not help" (two of three seeds; s.d. 0.099 around 0.454).
- [ ] **Step 5:** "Fomin et al." → "Fomin"; add "measured on classifiers over LLM activations". Check `report/final/extra.bib` author field has one author.
- [ ] **Step 6:** "Strongest detector" claims → qualify "by raw AUROC"; add one sentence on perplexity's length-binned AUROC 0.739 (vs 0.725 B1-Optuna; raw 0.430): its ranking within length bins is better than its raw ordering, which length inverts. B1-Optuna phrasing: "the only trained detector that matches length".
- [ ] **Step 7:** Deployment/model card: B1-Optuna recommendation states "not on MCP tool descriptions: there it catches 35% of attacks at 6.9% FPR".
- [ ] **Step 8:** Robustness table/text (`tbl:noise`): name model (B1-Optuna; B2 seed 7267) and "5% validation-FPR threshold, in-distribution attacks only". MSB 0.866 marked "indicative (21 benign rows)" wherever it appears.
- [ ] **Step 9:** Limitations: add (a) MCPTox attacks come from three templates (77/183/225), so the holdout tests three phrasings; (b) no off-the-shelf detector or LLM-judge baseline; (c) regexes written after reading holdout errors; (d) Wilson intervals assume independent rows while descriptions cluster by server. Related work/intro: add MCPTox's finding that indirect-injection payloads transplanted into tool descriptions succeed ~0% of the time (MCPTox §4.3), and o1-mini's 72.8% ASR if not present.
- [ ] **Step 10:** Team contributions: Mohammed Adil Ahmed row includes B4 and Phase 6 (check it already does; the deck is what lags).
- [ ] **Step 11: Verify**

Run: `make report-final && make check-final && uv run pytest tests/test_final_report.py -q`
Expected: PDF builds, check-final passes (page limit included), tests pass.

- [ ] **Step 12: Commit** on branch `docs/review-wording`:

```bash
git add report/final/report.md report/final/extra.bib report/final/report.pdf
git commit -m "Bring the final report's claims in line with the deck-v2 review."
```

### Task 2: Length analyses (experiments 3, 4, paired test)

**Files:**
- Create: `src/eval/followup_length.py`
- Create: `tests/test_followup_length.py`
- Modify: `Makefile` (target `followup-length`), `report/final/report.md`
- Outputs: `report/followup/b2_length.csv`, `report/followup/templates.csv`, `report/followup/paired.json`, `report/followup/b2_length.png`

**Interfaces:**
- Consumes: `report/phase5/scores.parquet` (columns `label, source, subset, attack_family, eval_set, n_chars, length, b1_optuna, b2_row2_s7267, b2_row2_s7268, b2_row2_s7269`), `report/phase5/thresholds.json`, `metrics.auroc`.
- Produces:
  - `template_of(attack_family: pd.Series) -> pd.Series` (`"Template-1".."Template-3"`, NaN for benign)
  - `b2_length_spearman(scores: pd.DataFrame) -> pd.DataFrame` columns `[detector, eval_set, source, rows, rho]`, `rows ∈ {"all","benign"}`
  - `long_benign_auroc(scores: pd.DataFrame, column: str) -> float`
  - `per_template(scores: pd.DataFrame, threshold: float) -> pd.DataFrame` columns `[template, attacks, length_auroc, b1_optuna_auroc, b1_optuna_recall]`; each template's attacks scored against all MCPTox benign rows
  - `paired_server_bootstrap(frame, a: str, b: str, group="subset", n_boot=1000, seed=7267) -> dict` keys `diff, lo, hi` (AUROC(a) − AUROC(b), resampling servers with replacement)
  - `rank_average(*columns: pd.Series) -> np.ndarray`

- [ ] **Step 1: Write failing tests** on a hand-built frame:

```python
def test_template_of():
    s = pd.Series(["x Template-2", None, "y-Template-3"])
    assert template_of(s).tolist()[0::2] == ["Template-2", "Template-3"] and pd.isna(template_of(s)[1])

def test_spearman_negative_when_score_falls_with_length(): ...  # rho < -0.9 for score = -n_chars
def test_long_benign_auroc_drops_short_benign_rows(): ...  # short benign rows excluded changes AUROC to a known value
def test_per_template_counts_and_recall():  # 2 templates, known attacks counts; recall = share >= threshold
def test_paired_bootstrap_identical_columns_is_zero():
    out = paired_server_bootstrap(frame, "length", "length")
    assert out["diff"] == 0 and out["lo"] == 0 and out["hi"] == 0
def test_paired_bootstrap_skips_one_label_resamples(): ...  # no NaN in lo/hi
```

- [ ] **Step 2:** Run `PYTHONPATH=src/data:src/eval:src/models uv run pytest tests/test_followup_length.py -q` → FAIL (module missing).
- [ ] **Step 3:** Implement the functions plus `main()` writing the four outputs. The figure: B2 seed 7268 score (log-scaled y, clipped at 1e-6) vs `n_chars` (log x) on MCPTox, coloured by label. Use `scipy.stats.spearmanr`. Bootstrap resamples where one label is absent are redrawn.
- [ ] **Step 4:** Tests pass (same command).
- [ ] **Step 5: Reproduce the review's numbers.** Add `make followup-length` (`PYTHONPATH="$(SRC_PATH)" uv run python src/eval/followup_length.py`), run it, and check against the review: MCPTox B2 Spearman −0.45/−0.77/−0.73; long-benign AUROC 0.727/0.595/0.616 on 121 benign rows (≥133 chars); template table rows 77/0.776/0.733/0.299, 183/0.761/0.736/0.213, 225/0.784/0.805/0.476; paired diff −0.008 [−0.103, +0.099]; rank average 0.821. A mismatch beyond rounding: stop and report, don't tune.
- [ ] **Step 6:** Paper: fill Task 1's B2 placeholder with the correlations and long-benign AUROCs; add the per-template table (backup-style, in §why or appendix); replace "ties length" support with the paired interval; add the figure.
- [ ] **Step 7:** `make report-final && make check-final` pass.
- [ ] **Step 8: Commit** on `analysis/followup-length`: script, test, Makefile, `report/followup/*`, report.md, report.pdf.

### Task 3: Transfer experiments (experiments 1, 2, 7)

**Files:**
- Create: `src/eval/followup_transfer.py`, `tests/test_followup_transfer.py`
- Modify: `Makefile` (target `followup-transfer`), `report/final/report.md`
- Outputs: `report/followup/few_shot.csv`, `report/followup/in_surface.csv`, `report/followup/msb_to_mcptox.json`, `report/followup/few_shot.png`

**Interfaces:**
- Consumes: `harness.load_data() -> Data(train, val, evals)`, `dataset.load("eval", sources=[...])` for MCPTox/MSB with columns `text, label, subset, attack_family`; `metrics.auroc, length_binned_auroc, threshold_at_fpr, rates_at_threshold`; `tfidf_optuna.fit_best(params, data)`; `followup_length.template_of`.
- Produces:
  - `Fit = Callable[[pd.DataFrame], Detector]` (takes a train frame with `text, label`; returns an object with `predict_proba(texts)`); default `b1_optuna_fit` wraps `fit_best` with the committed params from `BEST_PARAMS_PATH`.
  - `score_block(scores, labels, texts, threshold) -> dict` keys `auroc, binned_auroc, recall, fpr, n`
  - `few_shot(data, mcptox, fit, ks=(0,1,2,5,10,20), draws=10, seed=7267) -> pd.DataFrame` columns `[k, draw, k_used, auroc, binned_auroc, recall, fpr, n]`. For each draw, sample k servers; train = `data.train` ∪ their rows; test = remaining MCPTox servers; threshold = 1% FPR on `data.val` benign scores of that fit. k=0 trains on `data.train` alone and tests on all MCPTox servers, so it is fit once and repeated in every draw's rows.
  - `in_surface(mcptox, fit, by: str) -> dict` with `by ∈ {"subset","template"}`: out-of-fold scores pooled across folds, then `score_block`; per-fold threshold = 1% FPR on that fold's training benign rows, recall/fpr computed per fold then pooled by counts. For `"template"`: three folds; test = that template's attacks + benign rows of a seeded third of servers; train = other templates' attacks + other servers' benign rows.
  - `msb_to_mcptox(msb, mcptox, fit) -> dict` (`score_block` result; threshold 1% FPR on MSB benign, noted as 21 rows).

- [ ] **Step 1: Write failing tests** with `tests/synth.make()` (rename `group` → `subset`, add `attack_family` cycling `Template-1..3` on attacks) and a fast fit (`TfidfDetector(C=1.0)`):

```python
def test_few_shot_never_trains_on_test_servers(): ...  # spy fit records train subsets; disjoint from test subsets each draw
def test_few_shot_caps_k(): ...  # ks=(0, 999) -> k_used == n_servers - 1, no exception
def test_few_shot_k0_is_constant_across_draws(): ...
def test_in_surface_by_subset_scores_every_row_once(): ...  # n == len(mcptox)
def test_in_surface_by_template_has_three_folds_and_no_server_overlap(): ...
def test_score_block_binned_nan_on_tiny_set(): ...  # 10 rows -> binned_auroc is NaN, auroc finite
```

- [ ] **Step 2:** Run `PYTHONPATH=src/data:src/eval:src/models uv run pytest tests/test_followup_transfer.py -q` → FAIL.
- [ ] **Step 3:** Implement the functions and `main()` (`--draws`, `--seed`, `--out`). Figure: median and IQR of binned AUROC and recall vs k, with the length baseline's binned AUROC (0.5 by construction) and k=0 marked.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Add `make followup-transfer`, run it (expect minutes on CPU). Sanity: k=0 AUROC ≈ 0.776 (B1-Optuna MCPTox, Phase 5) within draw noise.
- [ ] **Step 6:** Paper: new subsection at the end of the "Why the Detectors Fail" section, "Follow-up analyses outside the pre-registered protocol", with the few-shot figure, an in-surface table (LOSO, leave-one-template-out, MSB→MCPTox), and one sentence per result on what it says about "the bottleneck is data". Update the conclusion's data claim to match the result, whichever way it goes.
- [ ] **Step 7:** `make report-final && make check-final` pass.
- [ ] **Step 8: Commit** on `analysis/followup-transfer`.

### Task 4: Prompt Guard 2 Colab script

**Files:**
- Create: `scripts/prompt_guard_colab.py`, `tests/test_prompt_guard_colab.py`

**Interfaces:**
- Consumes: the layout `src/models/precomputed.py::PrecomputedScores` reads (check its required columns before writing).
- Produces: `score_rows(texts: list[str], scorer: Callable[[list[str]], np.ndarray], batch_size=32) -> np.ndarray`; `write_scores(rows: pd.DataFrame, scores, truncated, path)`; `main()` that loads `meta-llama/Llama-Prompt-Guard-2-86M` with `transformers` (token from `HF_TOKEN`), truncates at 512 tokens, scores the MCPTox, MSB and test rows from `dataset.parquet`, writes `report/deberta_scores/prompt-guard-2.parquet`.

- [ ] **Step 1:** Failing test: `write_scores` output round-trips through `PrecomputedScores(path).predict_proba(texts)` with a stub scorer; `score_rows` batches correctly (stub called ceil(n/batch) times).
- [ ] **Step 2:** Run it → FAIL. **Step 3:** Implement. **Step 4:** Pass.
- [ ] **Step 5:** Docstring gives the Colab cells (install, `HF_TOKEN`, run, download). Commit on `analysis/prompt-guard-script`. Tell the user a teammate must accept the Llama license on Hugging Face first.

### Task 5: Deck and outline

**Files:**
- Modify: Slides artifact `https://claude.ai/artifact/PRfbv6Dr8tuHKvFMebN2JT` (files under `project/slides/`), `report/final/presentation-outline.md`

- [ ] **Step 1:** `list` the artifact's files, read `project/deck.json`, map slide numbers 3, 5, 6, 8, 9, 11–17 to ids; read those slide files.
- [ ] **Step 2:** Apply, using numbers from Tasks 1–3: slide 3 "Fomin"; slide 5 MSB tile "indicative"; slide 6 title "four detector families" (or list them); slide 8 add binned-AUROC column and MCPTox recall (0.348); slide 9 75% vs 55%; slide 11 "B2 ranks MCP rows by inverse length" with the Spearman numbers; slide 12 "does not help"; slides 13–14 label model and threshold on each recall ("B2 seed 7267, 5% FPR", "B1-Optuna, 5%", "B1-Optuna, deployed 1%"), slide 13 "transfers best" → "the only trained detector that matches length"; slide 15 Ahmed row adds B4 and Phase 6; slide 16 capacity wording and the few-shot result replacing the "next: data" prediction; slide 17 add the four limitations from Task 1 step 9.
- [ ] **Step 3:** Publish only changed slide files in one call; then preview slides 8 and 16 (dense tables/figures).
- [ ] **Step 4:** Update `presentation-outline.md` to deck v2's slide list and new titles; commit on `docs/deck-v2-outline`.

## Execution order and time

Task 1 (~1 h) → Task 2 (~1.5 h) → Task 3 (~2 h incl. runs) → Task 5 (~1 h) ; Task 4 (~30 min) can go any time.
