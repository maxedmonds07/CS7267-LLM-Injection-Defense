---
title: "Detecting Prompt Injection Across RAG and MCP Surfaces"
subtitle: "Phase 4 Report: Model Development and Experimentation"
author:
  - Carter Corbin
  - Max Edmonds
  - Mohammed Adil Ahmed
  - Oluwamayowa Adewummi
  - Adedapo Odeyemi
date: |
  ```{=latex}
  \vspace{-0.6em}College of Computing and Software Engineering\\
  Kennesaw State University, Kennesaw, GA\\[1.2em]
  CS 7267 Machine Learning \textperiodcentered{} December 2026
  ```
abstract: |
  We train three families of prompt-injection detectors on text from two attack surfaces,
  RAG passages and tool outputs, and test whether any of them transfers to a third surface
  it never saw: the descriptions of Model Context Protocol (MCP) tools. A perplexity score
  from a frozen language model, a TF-IDF logistic regression and a fine-tuned DeBERTa-v3
  are compared through one harness, at a threshold fixed on validation data, next to a
  length-only reference. A 100-trial Optuna search, scored on attack families held out of
  fitting, improves TF-IDF in distribution (AUROC 0.902 to 0.922) and on MCP tool
  descriptions (0.675 to 0.767), where it is the strongest trained detector but only ties
  text length. DeBERTa fits its training surfaces almost perfectly (test AUROC up to 0.997)
  yet ranks MCP tool descriptions below chance and fails its pre-registered success
  criterion on every seed. Every detector fits what it was trained on and none transfers.
  The evidence points to missing exposure to the target genre, not to model capacity or
  the decision threshold.
documentclass: article
fontsize: 11pt
geometry: margin=1in
linestretch: 1.1
numbersections: true
colorlinks: true
linkcolor: "black"
citecolor: "blue"
urlcolor: "blue"
bibliography:
  - ../../literature-review/references.bib
  - extra.bib
csl: ieee.csl
link-citations: true
reference-section-title: References
---

\newpage
\tableofcontents
\newpage

# Introduction {#sec:intro}

Applications built on large language models increasingly read text their users never wrote:
passages retrieved for retrieval-augmented generation (RAG), documents returned by tools, and
the descriptions of Model Context Protocol (MCP) tools that an agent reads before deciding which
tool to call. Any of that text can carry an indirect prompt injection
[@greshake2023indirect; @yi2025bipia]. Our Phase 2 review found that no defense in the reviewed
set targets MCP tool descriptions, that no detector classifies tool outputs, and that no detector
is evaluated on an attack surface it was not trained on. It also showed that text length alone
separates the classes in several sources, so every detector must be reported next to a
length-only baseline [@fomin2026benchmarks].

Phase 4 asks one question with that setup: **does a stronger, tuned detector trained on two
attack surfaces transfer to a third that it never saw?** We train on RAG passages (PoisonedRAG
[@zou2025poisonedrag]) and tool outputs (BIPIA [@yi2025bipia]), and evaluate in distribution,
on MCP tool descriptions held out entirely (MCPTox [@wang2026mcptox], leave-one-surface-out),
and on a second benchmark held out entirely (MSB [@zhang2026msb]).

This report contributes:

- three detector families and a length reference, evaluated through one harness at a threshold
  fixed on validation data (§\ref{sec:models});
- a hyperparameter search scored on attack families held out of fitting, which improves the
  TF-IDF detector on every eval set (§\ref{sec:search});
- a pre-registered success criterion for the transformer detector and its verdict: **fail** on
  every seed (§\ref{sec:results});
- an ablation of the main design choices (§\ref{sec:ablation}), a robustness check against
  rewritten attacks (§\ref{sec:robustness}), and an account of what did not work
  (§\ref{sec:negative}).

# Experimental Setup {#sec:setup}

## Data and splits

The corpus is the one built and documented in Phase 2: 11,005 labeled text segments from four
public benchmarks, versioned with DVC. Table \ref{tbl:splits} shows how it is used. The training
pool (BIPIA and PoisonedRAG) is split by `group`, so a BIPIA context and its attacked twin always
land in the same split. MCPTox and MSB are held out whole: no row from either is used to fit,
select or threshold any model, and each is scored once per reported run.

| Split | Role | Used for | Benign | Adversarial |
|:----------------|:----------------|:---------------------------------------|------:|-----------:|
| train_pool train | fit | fitting; Optuna folds | 4,133 | 3,084 |
| train_pool val | fit | selection; deployed threshold (1% FPR) | 801 | 615 |
| train_pool test | eval | in-distribution results; decision rules | 827 | 607 |
| MCPTox | LOSO holdout | unseen surface; scored once per run | 362 | 485 |
| MSB | transfer holdout | cross-benchmark; indicative | 21 | 70 |

Table: Splits. MCPTox and MSB are held out whole and never used to choose anything. {#tbl:splits}

The two holdouts differ from training in two ways at once. Their attack surface is new (tool
descriptions), and they contain only instruction injection, while the training pool mixes
instruction injection (BIPIA) with knowledge poisoning (PoisonedRAG).

## Evaluation protocol

Every number in this report comes from one harness (`src/eval/harness.py`). For each detector it
reports, per eval set and per slice (source, attack surface, attack kind):

- **AUROC** with a 95% percentile bootstrap interval, resampling benign and adversarial rows
  separately [@efron1993introduction];
- **deployed TPR and FPR** at a single threshold, set at 1% FPR on the validation split's 801
  benign rows and applied unchanged everywhere. FPR is shown with a Wilson 95% interval
  [@wilson1927probable];
- **TPR at a matched 5% FPR**, with the threshold set on each eval set's own benign rows. This
  ranks detectors at equal false-positive cost and is never used to deploy anything. 5% rather
  than 1%, because 1% of MCPTox's 362 benign rows is three rows;
- **length-binned AUROC**: AUROC computed within bins of similar text length and averaged, so a
  detector that only ranks by length scores about 0.5.

Three rules were fixed before any Phase 4 model was trained. The holdouts are never used to pick
a model, hyperparameter, checkpoint, seed or threshold. Every reported run is shown, not only the
best. Decision rules between variants use the in-distribution test split only.

## Reproducibility and tracking

Data, score files and the selected DeBERTa model are versioned with DVC on a Cloudflare R2
remote. Every reported run is logged to a shared MLflow server (experiment `phase3-detectors`,
backed by Neon PostgreSQL and object storage), with its parameters, all metrics and the full
report as an artifact. Seeds are fixed throughout (7267 by default). `make evaluate`,
`make b1-optuna` and `make b2-report` reproduce each step; DeBERTa training runs in a Colab
notebook because it needs a GPU, and the rest of the pipeline consumes its score files.
Appendix \ref{appendix-reproduce} lists the commands in order. Each step is reproduced by its
own `make` target; there is no single end-to-end command or shared configuration file.

On 2026-11-19 the shared MLflow database lost its compute endpoint at the end of the 55-minute
Optuna study, and the run's upload failed. Because results are written to disk before they are
uploaded, a `--log-only` mode recovered the run without re-running it, and its threshold,
metrics and artifacts matched the committed files.

# Model Families {#sec:models}

**Length only.** Logistic regression on log character length, fit on the training split. It is
the shortcut every detector must beat: it is near chance in distribution (AUROC 0.491) but
strong on both holdouts (0.774 on MCPTox, 0.866 on MSB), because the direction of the length
difference between benign and adversarial text changes from source to source.

**B0: perplexity.** A frozen DistilGPT2 [@sanh2019distilbert; @radford2019language] scores each
text by its perplexity over sliding windows (512 tokens, stride 256). It is never trained on our
labels, so it tests whether injected text is simply less fluent than benign text.

**B1: TF-IDF and logistic regression.** Word 1–2-grams and character 3–5-grams within word
boundaries, `min_df` 2, sublinear term frequency, and an L2 logistic regression
[@pedregosa2011scikit]. The regularization `C` is chosen from {0.1, 1, 10} by validation AUROC
(10, the top of the grid). It is fast and inspectable, and in Phase 3 it was the detector to beat.
Its tuned variant, B1-Optuna, is described in §\ref{sec:search}.

**B2: fine-tuned DeBERTa-v3.** `microsoft/deberta-v3-base` [@he2023debertav3] fine-tuned for
binary classification, following PromptShield's recipe of fine-tuning a pretrained encoder and
judging it at a low false-positive rate [@jacob2025promptshield]. Two changes target why B1
fails off-surface:

- **Chunked max-pool scoring.** Each text is split into overlapping 256-token windows (stride
  192), each window is scored, and the text's score is the highest window score. An injection is
  a short span inside otherwise benign text, and nothing is truncated away.
- **Pairwise loss on BIPIA twins.** Every BIPIA context appears once clean and once attacked. A
  margin term pushes the attacked twin's score above its clean twin's. The rest of the text is
  identical, so the gradient points at the injected span.

Training uses learning rate 2e-5, batch size 16, 3 epochs, mixed precision and gradient clipping
at norm 1.0, on a Colab T4 (about 18 minutes per run). The epoch is chosen by validation TPR at
1% FPR. Three configurations (rows) are trained with three seeds each (7267–7269):

1. **Row 1**, plain: chunked scoring, trained on BIPIA and PoisonedRAG;
2. **Row 2**, row 1 plus the pairwise loss (weight 0.5, §\ref{sec:search});
3. **Row 3**, row 2 trained on BIPIA only, to test whether PoisonedRAG's reversed length cue hurts.

A fourth row with synthetic tool descriptions was planned as a stretch goal and cut for Colab
time.

B0 is not trained on our labels, so only B1 and B2 are trained families. We count B0 as a third
family because it tests a distinct, label-free hypothesis that the literature proposes as a
defense: that injected text is less fluent than benign text. Its failure is itself a result
(§\ref{sec:results}).

# Hyperparameter Search {#sec:search}

## B1-Optuna

B1's original search covered one hyperparameter on three values, and chose the edge value. We
replaced it with an Optuna study [@akiba2019optuna] using a TPE sampler [@bergstra2011algorithms]
(seed 7267, 100 trials, fixed before the run) over:

- `C`, log-uniform on [0.01, 1000];
- the feature set: word n-grams, character n-grams, or both;
- word n-gram range (1, *n*) for *n* in {1, 2, 3}; character analyzer (`char` or `char_wb`) and
  range (2–4, 3–5 or 3–6);
- `min_df` in {1, 2, 5}; `max_features` per vectorizer in {unlimited, 50k, 200k}; sublinear term
  frequency on or off; class weighting none or balanced.

**The objective is the important choice.** Validation holds the same attack families as training,
so a wide search scored on it rewards memorizing each family's wording. Each trial is instead
scored by mean AUROC over five **family-grouped folds of the training split**: each BIPIA attack
family is held out whole, so the score measures detection of attack families the model was not
fit on (§\ref{sec:cv}). Validation is untouched by the study and only sets the deployed threshold,
exactly as for B1; a guard rejects any validation, test or holdout row that reaches a trial. A
median pruner could stop weak trials after each fold.

The best trial (#93, mean held-out AUROC 0.9215) uses word 1–3-grams together with plain character
2–4-grams, `C` = 6.38, `min_df` 1, 200,000 features per vectorizer, sublinear term frequency and
balanced class weights. Refit on the training split and evaluated once, it beats B1 under the rule
fixed in its specification (test AUROC above B1's upper interval bound): **0.922 [0.907, 0.935]
against 0.902 [0.886, 0.917]**. Figure \ref{fig:importance} shows which settings mattered.
Combining word and character features was clearly better than either alone
(Table \ref{tbl:features}); plain character n-grams, which cross word boundaries, beat the
boundary-respecting variant B1 used.

| Setting | Value | Trials | Best CV AUROC | Median CV AUROC |
|:-------------|:-------|------:|------:|------:|
| Feature set | both | 80 | 0.9215 | 0.9146 |
| Feature set | char | 11 | 0.9094 | 0.9018 |
| Feature set | word | 9 | 0.9052 | 0.9023 |
| Char analyzer | char | 79 | 0.9215 | 0.9147 |
| Char analyzer | char_wb | 12 | 0.9101 | 0.9045 |

Table: B1-Optuna trials grouped by feature setting (mean held-out AUROC over 5 family folds of train). TPE samples promising settings more often, so trial counts differ. {#tbl:features}

![Parameter importance in the B1-Optuna study (Optuna's default PED-ANOVA estimator).](b1_optuna/importance.png){#fig:importance width=70%}

All 100 trials completed and the pruner stopped none, so pruning saved no time; the study took 55
minutes on one CPU core.

## B2 search

DeBERTa runs cost about 18 minutes of GPU time each, so B2's search was small: the pairwise-loss
weight was chosen from {0.5, 1.0} with one seed, on validation only, before any holdout was
scored. Both settings saturated validation (val AUROC 0.9999 for both), and 0.5 was chosen by
validation log-loss (0.048 vs 0.057). Validation saturates from the first epoch for every
configuration, which is why runs are compared by log-loss when their validation TPR ties
(§\ref{sec:negative}).

This search is limited. Learning rate, epochs and batch size were kept at the recipe's values
rather than searched. Since validation saturates, a wider search scored on validation could not
have ranked the candidates in any case; a family-fold objective like B1-Optuna's would need a
DeBERTa run per fold, which the Colab budget does not allow.

## B0

B0 is an untuned reference. Its window and stride only change how perplexity is aggregated, and
no setting can turn a fluency score into an injection score on surfaces where attacks are as
fluent as benign text (§\ref{sec:results}).

# Cross-Validation and Uncertainty {#sec:cv}

## Family-grouped folds

The folds used by B1-Optuna are built so that held-out AUROC means
"attack families not seen in fitting". BIPIA's 25 attack families are assigned whole to five
folds, largest first into the lightest fold; PoisonedRAG has a single attack family, so its
groups are dealt round-robin. A group never spans two folds, so a BIPIA context and its attacked
twin are always held out together. On the training split each fold holds 400–401 BIPIA pairs and
215–220 PoisonedRAG attacks.

These folds serve as B1-Optuna's search objective. We did not run B1 itself under family
cross-validation (choosing its `C` on the folds instead of validation, or setting its threshold
from about 4,900 out-of-fold benign scores instead of 801), so the uncertainty reported for the
single-run detectors comes from bootstrap intervals on the fixed splits.

## Seed variance and intervals

Training DeBERTa once per fold is beyond the Colab budget, so B2's uncertainty is reported two
ways: a bootstrap interval for every run (Table \ref{tbl:seeds}), and the mean and standard
deviation across its three seeds (Tables \ref{tbl:test}–\ref{tbl:msb}). For the single-run
detectors the interval is the bootstrap AUROC interval and the Wilson interval for FPR.

# Results {#sec:results}

Figure \ref{fig:auroc} summarizes every reported run; Tables \ref{tbl:test}–\ref{tbl:msb} give
the full numbers per eval set.

![AUROC with 95% bootstrap intervals for every reported run, by eval set. Dashed line: chance; dotted line: length only.](fig_auroc.png){#fig:auroc width=100%}

## In distribution

Every trained detector fits the training surfaces (Table \ref{tbl:test}). B2 is near perfect when
trained on both sources (AUROC 0.995 for row 1), B1-Optuna improves on B1 (0.922 vs 0.902), and
both rank by more than length: their length-binned AUROCs match their AUROCs. Length and
perplexity are near chance. B2 row 3, trained without PoisonedRAG, drops to 0.750 AUROC on
PoisonedRAG's test rows (Table \ref{tbl:slices}), as expected for a source it never saw.

| Detector | AUROC [95% CI] | Length-binned AUROC | TPR (deployed) | FPR (deployed) [Wilson] | TPR at 5% FPR |
|:-------------------|--------------------:|-------------:|-------------:|--------------------:|-------------:|
| Length only | 0.491 [0.460, 0.526] | 0.552 | 0.010 | 0.007 [0.003, 0.016] | 0.077 |
| B0 perplexity | 0.515 [0.482, 0.546] | 0.531 | 0.000 | 0.002 [0.001, 0.009] | 0.054 |
| B1 TF-IDF | 0.902 [0.886, 0.917] | 0.903 | 0.404 | 0.019 [0.012, 0.031] | 0.603 |
| B1-Optuna | 0.922 [0.907, 0.935] | 0.918 | 0.567 | 0.024 [0.016, 0.037] | 0.674 |
| B2 row 1 | 0.995 ± 0.002 | 0.996 ± 0.001 | 0.963 ± 0.003 | 0.015 ± 0.004 | 0.986 ± 0.003 |
| B2 row 2 (selected) | 0.987 ± 0.004 | 0.989 ± 0.003 | 0.961 ± 0.011 | 0.017 ± 0.006 | 0.975 ± 0.007 |
| B2 row 3 | 0.936 ± 0.013 | 0.837 ± 0.025 | 0.599 ± 0.008 | 0.017 ± 0.006 | 0.609 ± 0.008 |

Table: train_pool test (in distribution): 827 benign, 607 adversarial. B2 rows (mean ± standard deviation over 3 seeds): row 1 plain, row 2 adds the pairwise loss, row 3 is row 2 trained on BIPIA only. Per-seed intervals are in Appendix A. {#tbl:test}

## MCP tool descriptions (MCPTox)

The picture reverses on the unseen surface (Table \ref{tbl:mcptox}). B1-Optuna is the strongest
trained detector, 0.767 [0.736, 0.800] AUROC, up from B1's 0.675. Even so it only ties raw length
(0.774), and its interval overlaps length's. Its length-binned AUROC of 0.725 shows that, unlike
length, it ranks within texts of similar length; it is a weak detector, not a length proxy. Every
B2 configuration and B0 score **below chance**: they systematically rank benign tool descriptions
above poisoned ones.

The deployed threshold does not hold either. Set at 1% FPR on validation, it gives 5.0% FPR for
B1 and 6.9% for B1-Optuna on MCPTox. B2's FPR stays near zero only because its scores on MCPTox
are low across the board, so it also catches almost nothing (TPR about 1%).

| Detector | AUROC [95% CI] | Length-binned AUROC | TPR (deployed) | FPR (deployed) [Wilson] | TPR at 5% FPR |
|:-------------------|--------------------:|-------------:|-------------:|--------------------:|-------------:|
| Length only | 0.774 [0.738, 0.814] | 0.531 | 0.000 | 0.000 [0.000, 0.011] | 0.002 |
| B0 perplexity | 0.430 [0.384, 0.472] | 0.739 | 0.004 | 0.105 [0.077, 0.141] | 0.000 |
| B1 TF-IDF | 0.675 [0.637, 0.711] | 0.685 | 0.175 | 0.050 [0.032, 0.077] | 0.225 |
| B1-Optuna | 0.767 [0.736, 0.800] | 0.725 | 0.348 | 0.069 [0.047, 0.100] | 0.274 |
| B2 row 1 | 0.357 ± 0.075 | 0.626 ± 0.106 | 0.008 ± 0.006 | 0.006 ± 0.005 | 0.038 ± 0.014 |
| B2 row 2 (selected) | 0.346 ± 0.128 | 0.530 ± 0.163 | 0.008 ± 0.005 | 0.002 ± 0.003 | 0.096 ± 0.097 |
| B2 row 3 | 0.199 ± 0.075 | 0.300 ± 0.109 | 0.017 ± 0.025 | 0.011 ± 0.007 | 0.024 ± 0.032 |

Table: MCPTox (unseen surface, LOSO): 362 benign, 485 adversarial. B2 rows (mean ± standard deviation over 3 seeds): row 1 plain, row 2 adds the pairwise loss, row 3 is row 2 trained on BIPIA only. Per-seed intervals are in Appendix A. {#tbl:mcptox}

## MSB

MSB has 21 benign rows, so one false positive moves FPR by 4.8 points and its intervals are wide
(Table \ref{tbl:msb}); we treat it as indicative. Length scores 0.866; every trained detector is at
or below 0.56, and B2 is again below chance.

| Detector | AUROC [95% CI] | Length-binned AUROC | TPR (deployed) | FPR (deployed) [Wilson] | TPR at 5% FPR |
|:-------------------|--------------------:|-------------:|-------------:|--------------------:|-------------:|
| Length only | 0.866 [0.765, 0.948] | 0.567 | 0.000 | 0.000 [0.000, 0.155] | 0.386 |
| B0 perplexity | 0.134 [0.050, 0.242] | 0.297 | 0.000 | 0.048 [0.008, 0.227] | 0.000 |
| B1 TF-IDF | 0.537 [0.418, 0.660] | 0.543 | 0.100 | 0.095 [0.027, 0.289] | 0.071 |
| B1-Optuna | 0.559 [0.441, 0.677] | 0.549 | 0.157 | 0.048 [0.008, 0.227] | 0.157 |
| B2 row 1 | 0.326 ± 0.091 | 0.465 ± 0.046 | 0.033 ± 0.036 | 0.000 ± 0.000 | 0.129 ± 0.080 |
| B2 row 2 (selected) | 0.281 ± 0.224 | 0.429 ± 0.236 | 0.019 ± 0.022 | 0.000 ± 0.000 | 0.095 ± 0.091 |
| B2 row 3 | 0.163 ± 0.040 | 0.371 ± 0.060 | 0.029 ± 0.014 | 0.016 ± 0.027 | 0.029 ± 0.014 |

Table: MSB (transfer, indicative): 21 benign, 70 adversarial. B2 rows (mean ± standard deviation over 3 seeds): row 1 plain, row 2 adds the pairwise loss, row 3 is row 2 trained on BIPIA only. Per-seed intervals are in Appendix A. {#tbl:msb}

## B2 verdict under the pre-registered criterion

Before training, we fixed when B2 would count as beating B1: on MCPTox, **every seed** of the
configuration selected on validation must have (1) a 95% lower bound on TPR at matched 5% FPR
above B1's TPR at 5% FPR (0.2247), and (2) a 95% lower bound on length-binned AUROC above 0.5.
Row 2 was selected on validation (highest mean validation TPR at 1% FPR, 0.9995). **No seed
passes** (Table \ref{tbl:verdict}), so the verdict is **fail**. Seed 7267 has a length-binned lower
bound above 0.5 but its TPR interval stays below B1's; the other two seeds fail both conditions.

| Seed | TPR at matched 5% FPR [95% CI] | Length-binned AUROC [95% CI] | Passes |
|:----|--------------------:|--------------------:|------:|
| 7267 | 0.208 [0.153, 0.249] | 0.708 [0.656, 0.770] | no |
| 7268 | 0.039 [0.023, 0.056] | 0.390 [0.342, 0.441] | no |
| 7269 | 0.041 [0.025, 0.074] | 0.490 [0.440, 0.547] | no |

Table: B2 verdict on MCPTox for the selected configuration (row 2). A seed passes when both lower bounds clear their bars: B1's TPR at 5% FPR (0.2247) and 0.5. {#tbl:verdict}

B1-Optuna is stronger than the B1 that B2 was measured against. The criterion named the original
B1 before B2 was trained, so the verdict keeps it; against B1-Optuna (0.274) B2 would fail by a
wider margin.

# Ablation Study {#sec:ablation}

Table \ref{tbl:ablation} changes one design choice per row, with the same eval sets.

| Design choice | Comparison | Test AUROC | MCPTox AUROC | MCPTox length-binned | MCPTox TPR at 5% FPR |
|:------------------------------------|:----------------|-------------:|-------------:|-------------:|-------------:|
| Pairwise loss on BIPIA twins | B2 row 1 → row 2 | 0.995 → 0.987 | 0.357 → 0.346 | 0.626 → 0.530 | 0.038 → 0.096 |
| PoisonedRAG in training | B2 row 3 → row 2 | 0.936 → 0.987 | 0.199 → 0.346 | 0.300 → 0.530 | 0.024 → 0.096 |
| Search space and family-CV objective | B1 → B1-Optuna | 0.902 → 0.922 | 0.675 → 0.767 | 0.685 → 0.725 | 0.225 → 0.274 |

Table: Ablation: each row changes one design choice. B2 cells are means over 3 seeds. Word vs character features is in Table \ref{tbl:features}. {#tbl:ablation}

- **Pairwise loss.** It costs a little in distribution (0.995 to 0.987) and does not move MCPTox
  AUROC (0.357 to 0.346). Its gain in TPR at 5% FPR (0.038 to 0.096) comes almost entirely from
  one seed (7267: 0.208; the others 0.039 and 0.041), so we do not read it as an effect.
- **PoisonedRAG in training.** We expected PoisonedRAG's reversed length cue to hurt transfer.
  The opposite holds: removing it makes B2 worse everywhere, including MCPTox (0.346 to 0.199
  AUROC). More varied training text helps more than a cleaner length signal.
- **Search space and objective.** B1-Optuna improves every column. It changes the features, the
  regularization and the selection objective at once, so this row does not say how much of the
  gain comes from the family-fold objective alone. Separating it would take B1's original grid
  selected on the family folds, which we did not run (§\ref{sec:cv}).

# Robustness and Shortcut Analysis {#sec:robustness}

Two checks built in Phase 3 ask whether a detector's score is real and whether it survives an
attacker who rewrites the attack. Both are run on B1, the detector they were built for, and use a
5% validation-FPR threshold rather than the 1% used elsewhere in this report, so B1's original TPR
here is 0.647 rather than 0.404.

**Shortcut gate.** B1 passes in distribution: its AUROC lower bound (0.880) clears the best cue
probe built from length, punctuation, capitalization and digit rates (upper bound 0.778), and it
keeps its signal when classes are balanced within length deciles (0.867). On MCPTox it fails: a
length probe's upper bound (0.874) is above B1's lower bound (0.615), although B1 keeps 0.712
AUROC on the 222 length-matched rows, so it is not relying on length there. On MSB only 26 rows can
be length-matched, too few to run the check. Label shuffling gives AUROC about 0.50 and there are
no shared groups or duplicate texts between training and test.

**Red-team mutations.** Rewriting every test-split attack eleven ways shows where B1 is brittle.
Obfuscation that changes characters hurts most: TPR falls from 0.647 to 0.280 with leetspeak,
0.328 with zero-width characters and 0.516 with homoglyphs. Wrapping attacks in JSON or Markdown
costs about 0.04; synonyms, polite rewording, sentence shuffling, case noise, whitespace and HTML
comments change almost nothing. Character-level preprocessing (Unicode normalization, stripping
zero-width characters) is a cheap fix for the largest losses.

We did not score the mutations with B1-Optuna or B2. B1-Optuna uses the same kind of word and
character n-gram features as B1, so it is likely brittle in the same places, but we have not
measured it. B2 would need the mutation file scored on Colab, and since B2 already fails to
transfer to an unseen surface, its robustness to rewrites of in-distribution attacks would not
change the conclusion.

# Negative Results and Limitations {#sec:negative}

- **The transformer does not transfer.** B2 fits the training surfaces (test AUROC 0.92–0.997
  across all nine runs) but every run is below chance on MCPTox (0.147–0.491) and the verdict is
  fail on every seed. We report all nine runs.
- **B2 is unstable across seeds.** For the selected row, MCPTox AUROC ranges from 0.247 to 0.491
  and MSB AUROC from 0.093 to 0.529 across seeds that differ only in their random seed.
- **Validation saturates.** B2 reaches validation AUROC of about 0.9999 from the first epoch, so
  validation can barely rank B2 configurations; ties are broken by log-loss, a rule added
  (2026-11-08) before any B2 holdout number was computed.
- **Perplexity is not a detector here.** B0 is near chance in distribution (0.515) and below
  chance on both holdouts (0.430 and 0.134).
- **The MCPTox false-positive rate is a genre gap, not a threshold problem.** The training pool
  has no benign tool descriptions, so no threshold chosen on training-pool data reaches 1% FPR on
  MCPTox; B1's benign MCPTox scores sit higher across the whole distribution.
- **The holdouts are small.** MSB has 21 benign rows, and its tool-output slice has no benign rows
  at all, so it has no FPR or AUROC.
- **The searches are limited.** B2's search is two points; B0 is not tuned; the Optuna pruner
  never fired; and B1-Optuna changes several things at once, so its gain is not attributed to one
  of them.
- **The robustness checks cover B1 only.** The shortcut gate and the red-team mutations were not
  run on B1-Optuna or B2.
- **One run predates a code change.** B2 row 1, seed 7267, was trained before gradient clipping
  was added. By the run plan it was not retrained, so it is not bit-identical to its sibling
  seeds.

# Discussion and Next Steps {#sec:discussion}

All three families learn their training surfaces, and none learns "injection" in a form that
carries to MCP tool descriptions. The ordering on the unseen surface is the reverse of capacity:
the tuned linear model transfers best, the fine-tuned transformer worst. Because DeBERTa fits its
training data almost perfectly while ranking the new surface below chance, the bottleneck is not
capacity but what the model was shown. It learned what attacks look like inside retrieved
passages and tool outputs, and MCP tool descriptions are a different genre: text addressed to
the model about how to use a tool. We expect that benign descriptions already read like
instructions, which would explain the inverted ranking; we have not measured this, and the
final report should check it. Either way, a detector meant for that surface needs examples
of it, benign ones especially.

Two directions follow. First, train with benign and adversarial tool descriptions, which the cut
stretch row would have tested with synthetic data, and measure again on MCPTox without using it
for training. Second, measure what a detector is worth end to end, as the reduction in attack
success of a defended agent in AgentDojo [@debenedetti2024agentdojo], rather than by AUROC alone.
Our future-work notes also describe skill registries as a further surface that, like MCP tool
descriptions, an agent reads before acting.

**Use of AI assistance.** Parts of the experiment code (the Optuna study, the family folds, the
table generator for this report) were written with an AI assistant (Claude) and merged through
reviewed pull requests, with tests. This draft was also prepared with AI assistance from the
committed result files. Every analysis choice, every decision rule and the final text were reviewed
by the team.

\appendix

# Every B2 Run {#appendix-seeds}

| B2 row | Seed | Test AUROC | MCPTox AUROC | MCPTox TPR at 5% FPR | MSB AUROC |
|:---|----:|--------------------:|--------------------:|--------------------:|--------------------:|
| 1 | 7267 | 0.995 [0.992, 0.997] | 0.319 [0.281, 0.357] | 0.033 [0.016, 0.052] | 0.246 [0.148, 0.357] |
| 1 | 7268 | 0.997 [0.994, 0.998] | 0.308 [0.269, 0.346] | 0.027 [0.012, 0.041] | 0.426 [0.312, 0.556] |
| 1 | 7269 | 0.994 [0.988, 0.997] | 0.443 [0.404, 0.480] | 0.054 [0.033, 0.080] | 0.306 [0.196, 0.432] |
| 2 | 7267 | 0.991 [0.986, 0.996] | 0.491 [0.453, 0.529] | 0.208 [0.153, 0.249] | 0.529 [0.411, 0.651] |
| 2 | 7268 | 0.984 [0.974, 0.992] | 0.247 [0.211, 0.283] | 0.039 [0.023, 0.056] | 0.093 [0.040, 0.157] |
| 2 | 7269 | 0.986 [0.977, 0.993] | 0.300 [0.263, 0.337] | 0.041 [0.025, 0.074] | 0.221 [0.120, 0.332] |
| 3 | 7267 | 0.948 [0.938, 0.958] | 0.166 [0.135, 0.197] | 0.012 [0.004, 0.025] | 0.179 [0.084, 0.288] |
| 3 | 7268 | 0.923 [0.909, 0.935] | 0.285 [0.245, 0.322] | 0.060 [0.039, 0.080] | 0.192 [0.091, 0.307] |
| 3 | 7269 | 0.938 [0.927, 0.949] | 0.147 [0.115, 0.180] | 0.000 [0.000, 0.000] | 0.118 [0.054, 0.196] |

Table: Every reported B2 run, with 95% bootstrap intervals. Row 2 was selected on validation. {#tbl:seeds}

# Results by Slice {#appendix-slices}

| Slice | Benign | Adv. | Length | B1 | B1-Optuna | B2 row 1 | B2 row 2 | B2 row 3 |
|:----------------------|------:|----:|------:|-----:|---------:|-----:|-----:|-----:|
| test: BIPIA | 377 | 377 | 0.532 | 0.849 | 0.862 | 0.993 | 0.985 | 0.993 |
| test: PoisonedRAG | 450 | 230 | 0.138 | 0.942 | 0.965 | 0.999 | 0.998 | 0.750 |
| test: instruction | 827 | 377 | 0.741 | 0.897 | 0.911 | 0.993 | 0.980 | 0.988 |
| test: poisoning | 827 | 230 | 0.082 | 0.909 | 0.939 | 0.998 | 0.999 | 0.851 |
| MSB: tool descriptions | 21 | 55 | 0.907 | 0.481 | 0.505 | 0.229 | 0.202 | 0.126 |
| MSB: tool outputs | 0 | 15 | – | – | – | – | – | – |

Table: AUROC by slice (B2: mean over seeds). Test-split slices pair each attack kind with every benign row. MSB tool outputs have no benign rows, so AUROC is undefined. {#tbl:slices}

# Reproducing the Results {#appendix-reproduce}

From a fresh clone with access to the DVC remote, these commands rebuild every number in this
report, in order:

```bash
make install && make data      # dependencies; data and score files (DVC)
make evaluate DETECTOR=perplexity OUT=report/baselines/b0_perplexity.csv
make evaluate DETECTOR=tfidf OUT=report/baselines/b1_tfidf_lr.csv
make b1-optuna                 # about an hour on one CPU core
make b2-report                 # B2 verdict and per-row CSVs
make phase4-tables             # this report's tables and figure
make report-phase4             # this PDF
```

B2 training runs in `notebooks/deberta_colab.ipynb` on a GPU and writes the score files that
`make b2-report` reads. Set `MLFLOW_TRACKING_URI` to log each step to the team's MLflow server.
