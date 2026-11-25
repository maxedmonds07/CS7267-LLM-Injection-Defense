# B4 relevance experiments

**B4 does not transfer.** Every configuration scored about 0.99 AUROC in distribution and below 0.5 on both holdouts, where the length-only baseline did better. Adding task context moved holdout AUROC toward chance without improving detection at a fixed false-positive rate. The two explicit relevance features, the question B4 was designed to answer, lowered holdout AUROC.

`relevance` is the configuration reported as B4 because the preset selection rule picked it on validation. Validation was saturated, so this choice says nothing about transfer (see [Selection and results](#selection-and-results)). [Selection record](../report/b4/selection.json)

## What we trained

Each configuration started from `microsoft/deberta-v3-base` for each seed (7267, 7268, 7269), giving 12 independent fine-tunes. The candidate was the passage, tool output, or tool description being classified as benign or adversarial. A pair consisted of the trusted task context and that candidate.

| Configuration | Encoder input | Explicit features | Training sources | Comparison |
|---|---|---|---|---|
| `text_control` | Candidate alone | None | BIPIA + PoisonedRAG | Baseline for adding context |
| `pair_only` | Task context + candidate | None | BIPIA + PoisonedRAG | Compare with `text_control` to measure the contribution of context |
| **`relevance`** | Task context + candidate | TF-IDF cosine and task-token coverage | BIPIA + PoisonedRAG | Compare with `pair_only` to measure the contribution of the two features |
| `relevance_bipia` | Task context + candidate | Same two features | BIPIA only | Compare with `relevance` to measure the effect of removing PoisonedRAG from training |

The paired encoder read up to 128 task tokens alongside each overlapping 256-token candidate window (stride 192). In `pair_only`, the linear classifier received the first token’s hidden state. In `relevance` and `relevance_bipia`, we appended two numbers to that representation.

1. **TF-IDF cosine similarity** between the full task and candidate strings, using character 3–5-grams within word boundaries. Each relevance run fitted its vocabulary and IDF weights on its own training pairs.
2. **Task-token coverage**, the fraction of distinct, case-folded words in the full task text that also appeared in the candidate.

The encoder and classifier were trained jointly with binary cross-entropy on the highest window logit for each candidate. We deferred schema agreement because the training pairs lacked consistent trusted schemas. [Feature definitions](../src/models/b4_features.py) · [Classification head](../src/models/b4_network.py)

A separate logistic regression used only the two numerical features and scored 0.5828 validation AUROC. Its predictions played no part in training or selecting the 12 DeBERTa models. Length-only and candidate-text TF-IDF baselines were also evaluated on the B4 rows. [Feature-only diagnostic](../report/b4/relevance_probe_validation.json) · [Length baseline](../report/b4/reference_length.csv) · [TF-IDF baseline](../report/b4/reference_tfidf.csv)

## Examples and task context

We kept the original labels and split assignments, joined context by row ID, and verified each candidate’s text hash. PoisonedRAG supplied original questions. For BIPIA we used the question, code and error text, or a fixed instruction for the summarization task. Where a BIPIA group had several questions, we used the first in lexical order for both clean and attacked rows. For MCPTox and MSB tool descriptions, the server’s catalogue of legitimate tasks served as a proxy for an individual request. Clean and attacked descriptions from a server received the same catalogue. [Context construction](../src/data/b4_data.py)

The mixed-source runs used 7,217 training rows. Each BIPIA-only run used 4,008. All configurations used the same 1,416 validation rows and the same evaluation rows, including for the text-only control. Coverage was 1,434/1,434 for the in-distribution test, 672/847 for MCPTox (servers with a task catalogue), and 76/91 for MSB (tool descriptions). The 15 MSB tool-output rows lacked a task mapping. For a direct comparison with the earlier candidate-text DeBERTa (B2 in the Phase 4 and final reports), match the holdout row IDs first. [Per-run evaluation counts](../report/b4/per_run.csv) · [Earlier DeBERTa study](../report/phase4/report.md)

## Selection and results

The reported study used **three epochs per run**, batch size 16, and a base learning rate of `2e-5` with the optional sweep disabled. Within a run, we selected the epoch with the highest validation true-positive rate (TPR) at a threshold targeting a 1% false-positive rate (FPR) on benign validation examples. Ties used validation AUROC. Across configurations, we selected by mean validation TPR over the three seeds, breaking ties by lower mean validation log loss. The selected `relevance` checkpoints had mean validation TPR 1.000 and mean validation log loss 0.0090. We scored the test and holdout examples after all 12 fine-tunes finished. [Training configuration](../configs/b4.yaml) · [Validation results](../report/b4/validation_selection.csv)

**Validation could not separate the leading configurations.** `relevance`, `pair_only` and `text_control` all reached mean validation TPR of 0.998 or higher, so the choice fell to the log-loss tiebreak. In the second run described below, `relevance` and `pair_only` both reached TPR 1.000, and their mean log losses differed by 0.00004 (0.01273 vs 0.01277). Validation is in distribution, so it cannot predict which configuration transfers.

| Evaluation set | Length only | TF-IDF | `pair_only` mean AUROC | **B4 `relevance` mean AUROC** | 95% interval for the B4 mean across seeds |
|---|---:|---:|---:|---:|---|
| In-distribution test | 0.491 | 0.902 | 0.9910 | 0.9900 | 0.9856–0.9944 |
| MCPTox (unseen surface) | 0.760 | 0.700 | 0.5056 | 0.4536 | 0.2079–0.6993 |
| MSB (unseen benchmark) | 0.907 | 0.481 | 0.3348 | 0.2573 | 0.0495–0.4651 |

The length-only and TF-IDF references were refitted once on the same rows, so they have no seed interval. [Length baseline](../report/b4/reference_length.csv) · [TF-IDF baseline](../report/b4/reference_tfidf.csv)

Adding context (`text_control → pair_only`) raised mean holdout AUROC by 0.166 on MCPTox and 0.175 on MSB, for all three seeds. This moved the scores from well below chance to about chance. Mean TPR at a matched 5% FPR on MCPTox rose only from 0.019 to 0.044, so context removed some misleading signal rather than adding detection. Adding explicit features (`pair_only → relevance`) lowered mean AUROC by 0.0520 on MCPTox and 0.0775 on MSB. One MCPTox seed improved. All three MSB seeds declined. Mean TPR at a matched 5% FPR also fell on both holdouts. That metric sets a threshold on each evaluation set’s benign rows. Training on BIPIA only (`relevance → relevance_bipia`) lowered PoisonedRAG test AUROC to about 0.75 and lowered mean AUROC on both holdouts. [All four configurations](../report/b4/summary.md) · [Per-seed differences](../report/b4/ablation_deltas.csv)

At the threshold set on validation, `relevance` flagged 17.3% of MCPTox attacks and 20.0% of its benign rows, which is worse than flagging rows at random. It flagged no MSB attacks.

### Why the holdout scores fall below chance

The cause is not settled. One candidate is a length shortcut that reverses between training and the holdouts. In training, PoisonedRAG attacks are shorter than benign passages: on its test rows, the length-only score has AUROC 0.138. On the holdouts, attacks are longer, with length-only AUROC 0.760 on MCPTox and 0.907 on MSB. On MCPTox, `text_control` has a raw AUROC of 0.339 but scores 0.672 when AUROC is computed within length bins, which fits a learned "shorter means attack" rule.

Two results go against length being the main cause. Training on BIPIA only, which removes PoisonedRAG's short attacks, lowered MCPTox AUROC both here (0.454 to 0.397) and for B2 in the final report (0.346 to 0.199). MSB does not show the within-bin gain either (0.333), although it has only 21 benign rows. The final report's saliency analysis offers another account: B2 scores 98–100% of MCPTox rows below 0.01, so the detector barely responds to tool descriptions at all. [Length baseline by source](../report/b4/reference_length.csv) · [Length-binned AUROC per run](../report/b4/per_run.csv) · [Final report](../report/final/report.md)

**MSB results are noisy.** With 21 benign and 55 attacked rows, seed standard deviations reach 0.15 AUROC. The t-intervals in `summary.md` are not clipped to [0, 1], so some fall below zero. Report the direction of MSB effects, not their size.

### Second independent run

A second team member repeated the full grid with the same configuration, dataset, task contexts and code. The refitted length-only and TF-IDF references were byte-identical, so both runs evaluated the same rows. GPU training is not bit-for-bit reproducible, so the same seed gave different models. In-distribution test AUROC agreed within 0.006 per seed. Holdout AUROC for the same configuration and seed differed by up to 0.10 on MCPTox (`text_control` seed 7267: 0.321 vs 0.419) and 0.21 on MSB (`pair_only` seed 7269: 0.471 vs 0.265). Pooling the six seed-runs from both runs:

| Comparison | MCPTox mean ΔAUROC | MSB mean ΔAUROC | Seed-runs improved (MCPTox / MSB) |
|---|---:|---:|---|
| `text_control → pair_only` | +0.146 | +0.142 | 5/6 / 5/6 |
| `pair_only → relevance` | −0.056 | −0.047 | 1/6 / 1/6 |

The conclusions are the same in both runs. Seed 7267 in the second run is the exception to the context gain on both holdouts. [Second-run summary](../report/b4/rerun/summary.md) · [Per-run metrics](../report/b4/rerun/per_run.csv) · [Per-seed differences](../report/b4/rerun/ablation_deltas.csv) · [Validation results](../report/b4/rerun/validation_selection.csv)

To retrieve the B4 values in the table, filter `seed_summary.csv` to `experiment = relevance`, `slice = all`, and `metric = auroc`. Its 95% t-intervals describe variation in the mean across training seeds. The AUROC intervals in `per_run.csv` use 1,000 bootstrap resamples of evaluation rows within each label for a single trained model. [Metrics across seeds](../report/b4/seed_summary.csv) · [Per-run metrics](../report/b4/per_run.csv)

## Reproducing the completed study

1. With the local processed, raw, and generated data available, run `python scripts/package_b4_colab.py --inputs` in the project environment. Upload `dist/cs7267-b4/b4-relevance-v1/` at the same relative path inside your Drive project. It contains the source ZIP and the three input files (`dataset.parquet`, `contexts.parquet`, and `contexts.coverage.csv`).
2. Open the notebook with a Colab GPU runtime. Use `STUDY = "b4-relevance-v1"`, `DATA_MODE = "upload"`, and set `DRIVE_PROJECT` to the Drive project folder containing `dist`. The reported study used `MODE = "full"` and `INCLUDE_SWEEP = False`. The optional smoke mode checks a small training run on validation. [Colab notebook](../notebooks/b4_colab.ipynb)
3. Find the results in `My Drive/cs7267-b4/<STUDY>/experiments-fixed/reports/`. Keep the same study and settings to resume at an epoch boundary. For different code, data, or settings, package a new study using `--study <name>` and set `STUDY` accordingly. Allow Drive space for all 12 models and the active checkpoint.

For the alternative DVC workflow, package the source without `--inputs`, leave `DRIVE_PROJECT` blank, and use `DATA_MODE = "dvc"`. The notebook can prompt for the source ZIP and cache it in the study folder. DVC uses the read-only `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` in Colab Secrets to fetch the inputs. Keep data, paired contexts, and models in team-authorized storage.
