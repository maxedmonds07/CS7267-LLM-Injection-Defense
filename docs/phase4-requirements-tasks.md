# Phase 4: model development and experimentation, required tasks

Date: 2026-11-09
Deadlines: numbers frozen **2026-11-30**, write-up due **2026-12-07**.

The course requires that Phase 4 "systematically develop, tune, and compare multiple model
architectures", with every experiment tracked and reproducible from a single config file.
This document checks each requirement against the repository as of `main` at `add8a49`, and
lists the tasks that remain.

## Status at a glance

| Requirement | Status | What exists | Gap |
|---|---|---|---|
| At least 3 distinct model families | **Mostly met** | B0 perplexity (DistilGPT2), B1 TF-IDF + logistic regression, B2 DeBERTa (code merged, 1 of 9 runs done), plus the length-only reference | B2 runs not finished; B0 is not trained or tuned (see risk R1) |
| Hyperparameter search (Optuna / Ray) | **Missing** | Hand-written grids: B1 `C ∈ {0.1, 1, 10}`, B2 `pairwise_weight ∈ {0.5, 1.0}` | No Optuna or Ray anywhere; not in `pyproject.toml` |
| Cross-validation with confidence intervals | **Partial** | 95% bootstrap CIs on AUROC and TPR at matched FPR, for every eval set and slice; B1-famcv spec and plan merged (#36) | No CV implemented yet; no fold-level mean ± CI; B2 has 1 seed of 3 |
| Ablation study on key design choices | **Planned** | B2 run plan rows 2 (pairwise loss) and 3 (BIPIA only); B1-famcv (selection method) | None of the ablation runs have been done |
| Negative results documented honestly | **Partial** | Anti-goals and pass/fail rule fixed before training; `row1-s7267` FAILS (MCPTox AUROC 0.319); B0 below chance on holdouts | Not written up anywhere |
| Reproducible via a single script | **Missing** | `make data` rebuilds the dataset; `make evaluate DETECTOR=...` runs one detector; `make b2-report` | No single entry point for all experiments; B2 training runs in Colab |
| Config-driven (YAML) | **Partial** | `config.yaml` drives the data pipeline only | Detector and experiment settings are `argparse` defaults in `train.py` and `deberta_run.py` |
| Version-controlled on GitHub | **Met** | PR workflow, DVC for data on R2, MLflow for runs | B2 score files not yet tracked (`report/deberta_scores/` is untracked) |
| Unit-tested data and model components | **Mostly met** | 205 tests across data, metrics, harness, TF-IDF, perplexity, DeBERTa, verdict | New code (Optuna, CV, config loader) needs tests; three deferred minors |
| README with full setup instructions | **Partial** | Setup, data, DVC, B0 and MLflow sections | Status line is stale; no B1 or B2 sections, no Colab steps, no "reproduce everything" section |

## Required tasks

Tasks are ordered by deadline risk. Each lists the requirement it serves and a done check.

### 1. Finish the B2 DeBERTa run plan (model families, ablation, CIs)

Plan: `docs/superpowers/plans/2026-11-07-deberta-detector.md`, Task 11.

- [ ] 1.1 Sweep row 2's `pairwise_weight` on val only: `sweep-row2-w0.5-s7267`, `sweep-row2-w1.0-s7267`.
- [ ] 1.2 Row 1 seeds 7268 and 7269 (7267 is done).
- [ ] 1.3 Row 2, three seeds, at the chosen weight.
- [ ] 1.4 Row 3 (BIPIA only), three seeds. This is the main ablation.
- [ ] 1.5 Track the score files: `dvc add report/deberta_scores`, commit the `.dvc` file, then `make push-data`.
- [ ] 1.6 Log every reported run to MLflow (`make evaluate DETECTOR=deberta SCORES=...`).
- [ ] 1.7 `make b2-report`: writes `b2_deberta_row{1,2,3}.csv` and `b2_verdict.md`. Do not rerun anything after reading the verdict.
- [ ] 1.8 DVC-track seed 7267 of the selected row (`make push-large`).

**Done when:** 9 reported runs plus the sweeps are in `report/deberta_scores/`, and the verdict file states pass, fail or mixed with every seed shown.

### 2. Add a hyperparameter search with Optuna (hyperparameter search)

This is the largest gap: the requirement names Optuna or Ray, and the project has neither.

- [ ] 2.1 Add `optuna` as a dependency (`uv add optuna`) and an `optuna` storage that MLflow can see (Optuna's MLflow callback, or log each trial as a nested run).
- [ ] 2.2 **B1 study** (cheap, the main search): search `C` (log-uniform, wider than today's grid), word vs character n-grams, n-gram range, `min_df`, `max_features`, `sublinear_tf` and `class_weight`. Objective: mean AUROC over the family-grouped folds from task 3, on `train_pool` only. Use a fixed sampler seed and a fixed trial budget (for example 100 trials).
- [ ] 2.3 **B2 study** (small, Colab budget): 6 to 10 one-seed, val-only trials over learning rate, epochs and `pairwise_weight`, using Optuna's ask-and-tell interface so each Colab run reports one trial. This replaces the two-point `pairwise_weight` sweep in task 1.1 if it is ready in time. Otherwise keep the sweep and report the B2 search as limited.
- [ ] 2.4 **B0:** search `max_context` and `stride` on val, or state in the write-up that B0 is an untuned reference and why.
- [ ] 2.5 Save each study's best parameters, the trial table and Optuna's importance and history plots under `report/phase4/`.

**Guard:** no study may read MCPTox or MSB. The objective must load data only through `dataset.load("fit", ...)`, which raises `HoldoutError` otherwise. Add a test for this.

**Spec conflict to resolve:** the B1-famcv spec lists "extending B1's `C` grid" as a non-goal. The course requirement overrides it. Either amend the spec, or run the Optuna study as a separate, clearly named B1 variant (`b1_optuna`) next to B1 and B1-famcv.

### 3. Implement cross-validation with confidence intervals (CV)

Plan: `docs/superpowers/plans/2026-11-08-b1-family-cv.md`.

- [ ] 3.1 Implement B1-famcv as planned: family-grouped folds that never split a BIPIA twin pair, the `--selection` option on `train.py`, and the seen vs unseen family report slice.
- [ ] 3.2 Report a per-fold metric table for B1 and B0 (AUROC, TPR at matched FPR) with the mean and a 95% interval across folds (t-interval over k folds).
- [ ] 3.3 For B2, k-fold training is too expensive on a T4. State that its variance comes from 3 seeds plus bootstrap CIs on each eval set, and report mean ± interval across seeds.
- [ ] 3.4 Add Wilson intervals for FPR to the write-up tables (already planned in the famcv spec).

**Done when:** every detector's headline number in the write-up has an interval, and the method used for each (fold CI, seed spread, bootstrap) is named.

### 4. Ablation study (ablation)

The key design choices already identified, and the run that tests each:

| Design choice | Comparison | Source |
|---|---|---|
| Pairwise loss on BIPIA clean/attacked twins | B2 row 1 vs row 2 | Task 1 |
| PoisonedRAG in training (reversed length cue) | B2 row 2 (or 1) vs row 3 | Task 1 |
| Model selection on val vs family-grouped CV | B1 vs B1-famcv | Task 3 |
| Threshold from 801 val benign rows vs out-of-fold benign rows | B1-famcv variants | Task 3 |
| Features: word vs character n-grams | Optuna B1 study, or a fixed pair of runs | Task 2 |
| Synthetic tool descriptions (stretch) | B2 row 4 | Plan Task 9; cut first if behind |

- [ ] 4.1 One ablation table in the write-up: each row changes one thing, with the same eval sets and interval columns.
- [ ] 4.2 Report the length-binned AUROC for each row, so a gain that comes only from length is visible.

### 5. Make all experiments reproducible from one config and one script (engineering)

- [ ] 5.1 Add an `experiments:` section to `config.yaml` (or `configs/experiments/*.yaml`, one file per run) holding every detector setting now set by `argparse` defaults: TF-IDF grid or Optuna search space, perplexity model and window, DeBERTa learning rate, epochs, batch size, seeds and `pairwise_weight`, plus `target_fpr` and `n_boot`.
- [ ] 5.2 Give `train.py` and `deberta_run.py` a `--config` option that reads that section. Command-line flags may override it, and the resolved config is logged to MLflow with each run.
- [ ] 5.3 Add a single entry point, `make reproduce` (or `scripts/reproduce.sh`), that runs: `make data`, B0, B1, B1-famcv, the Optuna B1 study, B2 evaluation from the DVC-tracked score files, `make b2-report`, and the write-up tables and figures.
- [ ] 5.4 Add the evaluation steps as DVC stages (`evaluate_b0`, `evaluate_b1`, `b2_report`, and so on) with `params:` pointing at `config.yaml`, so `dvc repro` reruns only what changed and `dvc.lock` records every result.
- [ ] 5.5 For B2 training, which cannot run in the single script on a laptop: document the Colab notebook (`notebooks/deberta_colab.ipynb`) as the one manual step, driven by the same config entry. The reproduce script consumes its score files.

**Done when:** a fresh clone with R2 keys runs `make install && make reproduce` and regenerates every CSV and table in the write-up.

### 6. Tests (engineering)

- [ ] 6.1 Tests for the config loader: defaults, overrides, and an unknown key is an error.
- [ ] 6.2 Tests for the Optuna objective on a tiny fixture: fixed seed gives the same best trial; the objective refuses holdout data.
- [ ] 6.3 Tests for the fold assignment (planned in the famcv plan): twins stay together, no family spans two folds.
- [ ] 6.4 Fix the three deferred B2 minors: catch NaN scores where they are produced, add gradient clipping, and check score-file labels at evaluate time. Add a test for each.
- [ ] 6.5 Run the full suite in CI on each PR (a GitHub Actions workflow running `uv run pytest`), if the team does not have one yet.

### 7. Write up negative results (negative results)

Each of these is already known and must appear in the write-up, with numbers:

- [ ] 7.1 **B2 row 1 fails the success criterion** on MCPTox: AUROC 0.319, TPR at 5% FPR 0.033 (CI 0.016–0.051) against B1's 0.2247. Test AUROC is 0.994, so the model fits the training surfaces but does not transfer. Report all seeds and rows, whatever the final verdict.
- [ ] 7.2 **Val saturation:** B2 reaches val AUROC 1.0 from epoch 1, so val cannot rank configurations. This is why the log-loss tie-break was added (amendment, 2026-11-08).
- [ ] 7.3 **B0 perplexity is below chance on the holdouts** (MCPTox AUROC 0.43, MSB 0.13) and near chance in distribution (0.515).
- [ ] 7.4 **MCPTox FPR is a genre gap**, not a threshold problem: train has no benign tool descriptions, so no threshold set on train_pool data reaches 1% FPR on MCPTox.
- [ ] 7.5 **Small holdouts:** MSB has 21 benign rows, so its intervals are wide; MSB `tool_output` has no benign rows and no FPR.
- [ ] 7.6 Any null result from B1-famcv (for example, it picks the same `C`) and any Optuna study that does not beat the hand grid.

### 8. README and repository housekeeping (engineering)

- [ ] 8.1 Update the README status line (it still says detectors are "the next phase").
- [ ] 8.2 Add README sections for B1 TF-IDF and B2 DeBERTa: how to train, the Colab steps and the R2 token, and how to evaluate a score file.
- [ ] 8.3 Add a "Reproduce all results" section pointing at task 5's entry point, and an "Experiments" section with the config schema and the Optuna studies.
- [ ] 8.4 Add a results table (B0, B1, B1-famcv, B1-Optuna, B2 rows) with intervals, linking to `report/baselines/*.csv`.
- [ ] 8.5 Commit the move of `report/future-work.md` to `report/future-work/future-work.md` on the open `report/future-work-notes` branch.

## Risks

- **R1: does B0 count as a model family?** B0 is a frozen language model with no training on labels, and it is not tuned. If a grader reads "model family" as a trained classifier, the project has only two (linear and transformer). A cheap fix is a third trained family, for example a gradient-boosted tree (LightGBM) on TF-IDF and length features, through the same harness and Optuna. That costs about a day; decide by Nov 12.
- **R2: Colab time.** 9 reported B2 runs plus sweeps and a B2 Optuna study is about 6 to 9 hours of T4 time. Cut order if behind: row 4, then the B2 Optuna study (keep the two-point sweep), then row 3 seeds 2 and 3.
- **R3: tuning on holdouts.** More search means more chances to leak. Every new objective and selection step must read `train_pool` only, enforced by `dataset.load` and a test.

## Suggested schedule

| Dates | Work |
|---|---|
| Nov 9–12 | Task 3 (famcv), task 5.1–5.2 (config), task 2.1–2.2 (B1 Optuna), decide R1 |
| Nov 10–22 | Task 1 on Colab, in parallel; task 2.3 if ready before row 2 |
| Nov 16–25 | Task 5.3–5.5 (single script, DVC stages), task 6 |
| Nov 23–30 | Task 4 ablation table, task 1.7 verdict. **Numbers frozen Nov 30** |
| Dec 1–7 | Task 7 write-up, task 8 README |
