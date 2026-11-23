# Phase 5: evaluation, interpretability and ethics, required tasks

Date: 2026-11-23
Deadlines: Phase 4 numbers frozen **2026-11-30**, write-up due **2026-12-07** (same pandoc → PDF
format as Phase 4).

The course asks Phase 5 to cover:

| Evaluation | Interpretability and ethics |
|---|---|
| Held-out test set performance | SHAP or LIME feature attribution |
| Robustness: noise, distribution shift | Attention / saliency maps if deep model |
| Fairness: disaggregated metrics by group | Model card |
| Latency and compute cost | Identify potential harms and misuse |
| Error analysis: failure mode taxonomy | Propose mitigation strategies |

This document checks each item against the repository as of `origin/main` at `3c4a825` (PR #40
merged), and lists the tasks that remain.

**Scope rule.** Phase 5 adds analysis only. It uses the frozen models and the committed score
files, runs on a laptop CPU, and retrains nothing. No Phase 4 number changes after Nov 30. Every
new analysis reads the holdouts only to *explain* a frozen model, never to select or tune one.

## Status at a glance

| Requirement | Status | What exists | Gap |
|---|---|---|---|
| Held-out test performance | **Met** | Phase 4 §Results: test split, MCPTox (LOSO) and MSB with bootstrap AUROC CIs, Wilson FPR intervals and length-binned AUROC for every detector (`report/baselines/*.csv`, `report/phase4/tables.md`) | Only needs a short headline table that names the interval method per row |
| Robustness: distribution shift | **Met** | The design is leave-one-surface-out: MCPTox and MSB are surfaces never seen in training. Shortcut gate (`src/eval/shortcut_check.py`, PR #40) | Restate it as the robustness result; no new runs |
| Robustness: noise | **Partial** | 11 red-team mutations of the test split (`src/eval/red_team.py`, `make red-team`); results for length and B1 in `report/red_team/{length,tfidf}/` | Not run for B1-Optuna or B2. `make red-team-data` already writes the mutated rows for B2 |
| Fairness: disaggregated by group | **Partial** | Per-slice rows in every baseline CSV (`source=`, `attack_surface=`, `attack_kind=`), with AUROC, TPR and FPR; Phase 4 Appendix B | No Wilson FPR interval per slice; no length-decile breakdown of FPR; no paragraph on what "group" means here |
| Latency and compute cost | **Missing** | Nothing measured. Colab T4 runs are logged to MLflow | Inference time per 1k rows, model size, training compute |
| Error analysis: failure taxonomy | **Missing** | Only aggregate statements (genre gap, length cue, obfuscation) | No sampled errors, no categories, no counts |
| SHAP or LIME | **Missing** | No dependency, no code | Everything |
| Attention / saliency (B2) | **Missing** | Checkpoint `models/deberta/row2-s7267.tar.dvc` | Everything |
| Model card | **Missing** | `DATA_CARD.md` (Gebru et al. datasheet) | A model card for the detectors |
| Harms and misuse | **Partial** | `DATA_CARD.md` "Uses" and ethics answers (no IRB needed; public benchmarks) | Harms of *deploying a detector*, and dual use of this work |
| Mitigation strategies | **Partial** | Phase 4 §Robustness suggests Unicode normalization; §Discussion suggests training on tool descriptions | One mitigation per identified harm |

## Required tasks

Tasks are ordered by how much new work they need. Each lists the requirement it serves and a done
check. The B1 detector we recommend, and use for the deep dives, is **B1-Optuna**: it is the
strongest trained detector on the unseen surface (MCPTox AUROC 0.767). B2 row 2 seed 7267 is the
DeBERTa checkpoint we have locally, and its own MCPTox AUROC is the best of the three seeds.

### 1. Held-out performance (evaluation)

- [ ] 1.1 One headline table: length, B0, B1, B1-Optuna, B2 row 2, on test, MCPTox and MSB.
  Columns: AUROC with interval, TPR and FPR at the deployed threshold, TPR at 5% FPR. Copy the
  numbers from `report/baselines/*.csv`; generate the table with `report/phase4/make_tables.py`
  rather than by hand.
- [ ] 1.2 Name the interval method on each row: bootstrap over groups (B0, B1), mean ± SD over 3
  seeds (B2), Wilson (FPR).

**Done when:** the table builds from committed CSVs and matches Phase 4 exactly.

### 2. Robustness: noise and distribution shift (evaluation)

- [ ] 2.1 `make red-team DETECTOR=tfidf-optuna` → `report/red_team/tfidf-optuna/`. This needs
  `tfidf-optuna` added to the `--detector` choices in `src/eval/red_team.py` (today: `length`,
  `tfidf`, `perplexity`), loading B1-Optuna's `best_params.json`.
- [ ] 2.2 B2 on the mutations: `make red-team-data`, then score
  `data/red_team/test_mutations.parquet` with the row2-s7267 checkpoint **on CPU or MPS**
  (about 607 × 12 rows). If it takes more than about an hour, score only the three obfuscation
  mutations plus the original, and say so.
- [ ] 2.3 One robustness table: Δ TPR at the 5% val threshold for each mutation × {B1,
  B1-Optuna, B2}. Note that B2 uses a subword tokenizer, so we expect it to resist leetspeak and
  homoglyphs differently from character n-grams. That is a prediction to state before scoring.
- [ ] 2.4 A distribution-shift paragraph from the existing results: in-distribution vs MCPTox vs
  MSB AUROC, the shortcut gate verdicts, and the MCPTox FPR at the deployed threshold (5.0% B1,
  6.9% B1-Optuna against a 1% target).

**Done when:** every rubric word (noise, distribution shift) has a table or paragraph with numbers.

### 3. Fairness: disaggregated metrics by group (evaluation)

There are no people in this data, so demographic fairness does not apply, and the write-up should
say so plainly. The groups that matter are the ones a deployed detector treats differently:
**attack surface, source benchmark, attack kind and text length**. The harm that maps onto
fairness is *benign text that gets blocked*. That falls on legitimate tool authors and content,
so FPR by group is the headline column, not TPR.

- [ ] 3.1 A script `src/eval/fairness.py` that reads score files and the baseline CSVs and
  writes `report/phase5/fairness.csv`: per detector × eval set × group, n benign, n adversarial,
  AUROC [CI], TPR, FPR [Wilson]. Groups: the existing `source`, `attack_surface` and
  `attack_kind` slices, plus length deciles computed on benign rows.
- [ ] 3.2 Report the largest FPR gap between groups for each detector, with intervals. Flag any
  group with fewer than 30 benign rows as indicative (MSB has 21; MSB tool outputs have none).
- [ ] 3.3 State the language limitation: every source is English, so behavior on other languages
  is unknown. Check this against `DATA_CARD.md` §Composition before writing it.
- [ ] 3.4 Tests: Wilson interval on known counts, group with no benign rows gives no FPR, length
  deciles computed on benign rows only.

**Done when:** `fairness.csv` exists, is generated by one command, and the write-up names the
largest gap.

### 4. Latency and compute cost (evaluation)

- [ ] 4.1 `scripts/bench_latency.py`: for length, B0, B1, B1-Optuna and B2 row 2, time scoring
  of 1,000 test rows on a laptop CPU (and MPS for B0 and B2, if present). Fixed batch size,
  one warm-up, median of 5 runs. Writes `report/phase5/latency.csv` with rows/s, ms per row,
  peak memory, and the machine spec.
- [ ] 4.2 Model size: parameter count and on-disk size for each detector (B1: vocabulary size ×
  1 coefficient; B0: DistilGPT2 82M; B2: DeBERTa-v3-base).
- [ ] 4.3 Training compute: wall time per run from MLflow (B1, B1-Optuna's 100 trials, B2's 9
  runs plus sweeps on a T4). Total GPU-hours for the project. If MLflow lacks a duration, use the
  run plan's estimate and mark it as such.
- [ ] 4.4 A test for the benchmark script on a 10-row fixture (runs, writes the expected
  columns).

**Done when:** one table gives accuracy (MCPTox AUROC) next to cost (ms per row, size, training
hours), so the trade-off is visible: B1-Optuna is both the cheapest and the best transfer.

### 5. Error analysis: failure-mode taxonomy (evaluation)

- [ ] 5.1 Sample errors at the deployed threshold, with a fixed seed: 30 false positives and 30
  false negatives each for B1-Optuna and B2 row 2 seed 7267 on MCPTox, plus 20 of each on the
  test split. Script: `src/eval/error_sample.py` → `report/phase5/errors/*.csv` (row id,
  text, score, label, source).
- [ ] 5.2 Draft the codebook *before* coding the rows. Starting categories, to be revised on a
  first pass of 20 rows:
  - FP: benign tool description written as an instruction ("always call this tool before …");
    benign text mentioning security or prompts; long benign text.
  - FN: attack phrased as tool documentation; indirect or polite attack with no injection
    vocabulary; short attack; obfuscated attack (test split only).
  - Other / unclear.
- [ ] 5.3 Two team members code an overlap of 30 rows independently; report Cohen's κ. Then
  split the rest.
- [ ] 5.4 A taxonomy table: category, count per detector and eval set, and one short example
  per category (attack strings shortened, see risk R4).

**Done when:** the table exists and the categories explain the Phase 4 claim that MCPTox is a
genre gap (expect the "benign description reads as instruction" FP category to dominate).

### 6. SHAP / LIME feature attribution (interpretability)

- [ ] 6.1 `uv add shap`. For B1 and B1-Optuna, use `shap.LinearExplainer` on the fitted
  pipeline. For a linear model on TF-IDF this is exact and fast. Use a background sample of 200
  train_pool rows.
- [ ] 6.2 Global: the top 20 n-grams pushing toward "adversarial" and toward "benign", as a bar
  chart. Compare B1 (word n-grams) with B1-Optuna (its selected features).
- [ ] 6.3 Global, by eval set: mean |SHAP| per feature on test vs MCPTox. This tests whether the
  features the model relies on are even present in tool descriptions.
- [ ] 6.4 Local: force or waterfall plots for 3 MCPTox false positives and 3 false negatives
  taken from task 5's sample.
- [ ] 6.5 Optional: LIME (`uv add lime`) on B2 for the same 6 rows, for a cross-family
  comparison.
- [ ] 6.6 Script `src/eval/explain_tfidf.py` → `report/phase5/shap/`. Tests: attributions
  sum to the model's logit minus the base value on a tiny fixture; refuses a fit on holdout data.

**Done when:** the figures are committed, and the write-up says which tokens drive the MCPTox
false positives.

### 7. Attention / saliency maps for B2 (interpretability)

- [ ] 7.1 Pull `models/deberta/row2-s7267.tar` (`dvc pull`). Integrated gradients on the
  embedding layer with Captum (`uv add captum`), 50 steps, a pad-token baseline, on CPU.
- [ ] 7.2 About 20 rows: the 6 from task 6.4, plus 7 correctly caught test attacks and 7
  correctly passed benign test rows for contrast. Render token heatmaps (HTML → PNG) to
  `report/phase5/saliency/`.
- [ ] 7.3 Quantify, so the result is not anecdote: on MCPTox benign rows, the share of
  attribution mass on imperative verbs and second-person words ("you", "must", "always",
  "ignore"). Compare with test benign rows. This tests the open Phase 4 hypothesis that benign
  tool descriptions read like instructions.
- [ ] 7.4 Raw attention weights are optional and go in an appendix only: attention is not a
  faithful explanation (Jain and Wallace 2019), which the write-up should cite.

**Done when:** heatmaps exist for every sampled row and the 7.3 number answers the hypothesis
yes or no.

### 8. Model card (ethics)

- [ ] 8.1 `MODEL_CARD.md` at the repo root, following Mitchell et al. (2019): model details,
  intended use, **out-of-scope use**, factors (the groups from task 3), metrics with intervals,
  evaluation data, training data (link to `DATA_CARD.md`), quantitative analyses (tasks 1–4),
  ethical considerations (task 9), and caveats and recommendations (task 10).
- [ ] 8.2 One card for B1-Optuna (the detector we would ship), with B2 row 2 as a section stating
  why it is *not* recommended (fails the pre-registered criterion on every seed).
- [ ] 8.3 State the deployment boundary in the card's first screen: validated on RAG passages and
  tool outputs only. Not validated for MCP tool descriptions, where it ties text length.

**Done when:** the card links to every number it quotes and the report includes it as an
appendix.

### 9. Potential harms and misuse (ethics)

Write one paragraph each, tied to evidence from this project:

- [ ] 9.1 **False sense of security.** A detector that scores 0.92 in distribution and below
  chance on a new surface (B2 on MCPTox) invites deployment where it does not work.
- [ ] 9.2 **Blocking legitimate content.** MCPTox FPR is 5 to 7% at a 1% target, so honest tool
  authors are flagged. The harm falls on a group (tool authors whose style is imperative),
  which links to task 3.
- [ ] 9.3 **Dual use of this work.** The red-team results, SHAP features and saliency maps tell
  an attacker which tokens to avoid and which obfuscations work (leetspeak drops B1's TPR from
  0.647 to 0.280).
- [ ] 9.4 **The dataset as an attack corpus.** It collects working injection strings in one
  place. They are already public, but aggregation lowers the cost.
- [ ] 9.5 **Over-reliance in place of architecture.** A detector used instead of privilege
  separation or tool allow-lists. Detection is probabilistic, and one miss is enough.
- [ ] 9.6 **Automation bias and accountability.** Who reviews a flag, and what happens to a
  blocked tool?

### 10. Mitigation strategies (ethics)

One mitigation per harm, each stating which result motivates it:

| Harm | Mitigation | Evidence |
|---|---|---|
| 9.1 False security | Gate deployment by surface: refuse to score surfaces not in the validation set; publish the model card's out-of-scope list | MCPTox and MSB results |
| 9.2 Blocking legitimate content | Surface-specific thresholds set on benign data from that surface; flag-and-review rather than block; collect benign tool descriptions for training | MCPTox FPR, task 3 |
| 9.3 Dual use | Release aggregate red-team numbers, not a ready-made evasion recipe; character normalization (NFKC, strip zero-width) before scoring, which removes the largest losses | `report/red_team/tfidf/summary.csv` |
| 9.4 Attack corpus | Keep raw data behind the existing DVC/R2 access; release only derived scores and IDs | `DATA_CARD.md` §Distribution |
| 9.5 Over-reliance | Defense in depth: detector plus privilege separation, tool allow-lists and human confirmation for side-effecting calls; measure end-to-end attack success (AgentDojo) | Phase 4 §Discussion |
| 9.6 Accountability | Log every flag with score and threshold; human appeal path; monitor score drift per surface | — |

- [ ] 10.1 Optional, cheap, and the only one we can measure: add NFKC normalization plus
  zero-width stripping as a preprocessing step *in an analysis script only* (not in the frozen
  detector), and re-run the obfuscation mutations for B1-Optuna. Report it as a mitigation
  experiment, separate from the frozen results.

### 11. Report and housekeeping

- [x] 11.1 `report/phase5/` with `report.md`, `header.tex`, `ieee.csl`, `repo-links.lua` and
  `extra.bib` copied from `report/phase4/`, and a `make report-phase5` target. Sections map 1:1
  to the rubric's ten items.
- [ ] 11.2 Add the new analyses as DVC stages (`fairness`, `latency`, `explain_tfidf`,
  `saliency`) so `dvc repro` regenerates `report/phase5/`.
- [x] 11.3 Add to `extra.bib`: Lundberg and Lee 2017 (SHAP), Ribeiro et al. 2016 (LIME),
  Sundararajan et al. 2017 (integrated gradients), Mitchell et al. 2019 (model cards), Jain and
  Wallace 2019 (attention is not explanation).
- [ ] 11.4 README: a "Phase 5 analyses" section with one command per task.

## Risks

- **R1: B2 on CPU is slow.** DeBERTa-v3-base with integrated gradients at 50 steps is about 50
  forward and backward passes per row. 20 rows is fine; the 7,000 mutated rows for task 2.2 may
  not be. Cut order: 2.2 down to obfuscation only, then 6.5 (LIME on B2), then 7.4.
- **R2: explaining on holdouts.** Tasks 5–7 read MCPTox to *explain* frozen models. That is
  allowed, but nothing learned there may feed back into a reported detector. Write this in the
  report's method section, and keep task 10.1 labeled as a separate experiment.
- **R3: the error taxonomy is subjective.** Mitigate with a codebook written before coding and
  κ on an overlap set (task 5.3). Report κ even if it is low.
- **R4: printing attacks.** The SHAP features, error examples and saliency figures show working
  injection strings. Shorten or redact them in the report, and do not print full payloads.
- **R5: one week.** Seven new analysis tasks before Nov 30. Tasks 1, 2.4, 3.3 and 8–10 are
  writing only and can run in parallel with the analysis.

## Suggested schedule

| Dates | Work |
|---|---|
| Nov 23–25 | Task 3 (fairness script), task 4 (latency), task 6 (SHAP on B1), task 2.1 |
| Nov 24–27 | Task 5 (sample, codebook, coding), task 7 (saliency), task 2.2 |
| Nov 26–30 | Task 8 model card, tasks 9–10 harms and mitigations, task 10.1. **Analyses frozen Nov 30** |
| Dec 1–7 | Task 11 report, figures and README |

## Owner sign-up

| Task | Owner |
|---|---|
| 1 Held-out table | |
| 2 Robustness | |
| 3 Fairness | |
| 4 Latency and compute | |
| 5 Error taxonomy (two coders) | |
| 6 SHAP / LIME | |
| 7 Saliency | |
| 8 Model card | |
| 9–10 Harms and mitigations | |
| 11 Report | |
