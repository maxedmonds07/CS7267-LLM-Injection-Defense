---
title: "Detecting Prompt Injection Across RAG and MCP Surfaces"
subtitle: "Phase 5 Report: Evaluation, Interpretability, and Ethics"
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
  Phase 4 found that prompt-injection detectors trained on RAG passages and tool outputs do not
  transfer to Model Context Protocol (MCP) tool descriptions. Here we evaluate and explain the
  frozen detectors to find out why, and what that means for deploying one. Beyond held-out
  AUROC we measure robustness to eleven rewrites of each attack, error rates by group, latency
  and cost, and a failure-mode taxonomy; we explain the TF-IDF model with exact SHAP values and
  the fine-tuned DeBERTa with integrated gradients. The errors have one main cause. Half of the
  MCP attacks are phrased as ordinary procedure ("before using X, you must first call Y"), a
  pattern found in 0.6% of benign tool descriptions and 1% of training-pool test rows; the
  detectors never saw it, and they miss these attacks. Meanwhile the benign descriptions they
  flag are long and half of them give the agent usage instructions; B1-Optuna's false-positive
  rate on the longest quarter of benign descriptions is 14.3% against a 1% target. SHAP and integrated gradients
  show that both trained models learned the same cue, second-person phrasing about the model's
  response ("in your response", "reply"), which MCP attacks rarely use and benign tool
  documentation sometimes does. DeBERTa scores 98–100% of MCP tool descriptions below 0.01, so its
  below-chance ranking is noise among near-zero scores, not the confident inversion hypothesized
  in Phase 4. Character obfuscation is the cheapest evasion: leetspeak cuts B1-Optuna's recall from 0.725 to 0.213,
  and input normalization repairs zero-width attacks but not leetspeak or homoglyphs. We
  document the recommended detector in a model card, set out six harms of deploying a
  prompt-injection detector, and pair each with a mitigation.
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
  - ../phase4/extra.bib
  - extra.bib
csl: ieee.csl
link-citations: true
reference-section-title: References
---

\newpage
\tableofcontents
\newpage

# Introduction {#sec:intro}

Phase 4 trained three detector families (a perplexity score, TF-IDF logistic regression, and a
fine-tuned DeBERTa-v3) on two attack surfaces, RAG passages and tool outputs, and tested them on
a third surface they never saw: the descriptions of MCP tools. Every detector fitted its training
surfaces and none transferred. This report asks *why*, and what that means for anyone who would
deploy such a detector.

Phase 5 analyzes the frozen Phase 4 detectors: nothing is retrained and no Phase 4 number
changes. Sections \ref{sec:heldout}–\ref{sec:errors} evaluate the detectors (held-out
performance, robustness, fairness, cost, and errors). Sections \ref{sec:shap}–\ref{sec:saliency}
explain them. Sections \ref{sec:harms}–\ref{sec:mitigation} and the model card in
Appendix \ref{appendix-model-card} cover ethics. Our contributions are:

- a failure-mode taxonomy showing that the commonest MCP attack phrasing is absent from the
  training data, with a rule-based codebook anyone can rerun;
- disaggregated error rates showing who bears the false alarms: authors of long,
  instruction-style tool documentation;
- SHAP and integrated-gradients evidence on what each model relies on, and a direct test of
  the Phase 4 hypothesis that benign tool descriptions "read like instructions";
- a model card and a harm-by-harm mitigation plan, including one measured mitigation.

# Evaluation Setup {#sec:setup}

**Detectors.** The length-only reference, B0 perplexity (DistilGPT2), B1 TF-IDF, B1-Optuna (the
100-trial search), and B2 DeBERTa row 2 (pairwise loss, selected on validation), as frozen at the
end of Phase 4. B1-Optuna is the detector we would deploy and the one we explain in most depth.
For B2 the analyses that need the model itself (noise, saliency, latency) use seed 7267, the
checkpoint tracked in DVC; it is also the seed with the highest MCPTox AUROC (0.491).
Score-based analyses report all three seeds.

**Eval sets.** The train_pool test split (827 benign, 607 adversarial), MCPTox (362 benign, 485
adversarial; never seen in training), and MSB (21 benign, 70 adversarial; indicative only).

**Thresholds and intervals.** Each detector is deployed at the score that gives 1% FPR on
validation benign rows. AUROC intervals are 95% bootstrap intervals that resample whole groups,
so a BIPIA twin pair never splits; FPR intervals are Wilson intervals [@wilson1927probable]; B2
values are mean ± standard deviation over three seeds.

**Explaining on holdouts.** Sections \ref{sec:errors}–\ref{sec:saliency} read MCPTox to explain
the frozen models. Nothing learned there feeds back into a reported detector; the mitigation we
test (Section \ref{sec:mitigation}) is reported separately from the frozen results.

**Reproducibility.** Every analysis is a script with a make target (Appendix
\ref{appendix-reproduce}). A shared step scores every row once per detector
(`report/phase5/scores.parquet`); it reproduces every Phase 4 AUROC and deployed FPR exactly.

# Held-Out Test Set Performance {#sec:heldout}

| Detector | Test AUROC | MCPTox AUROC | MSB AUROC | MCPTox FPR |
|:--------------------|:------------------|:------------------|:------------------|-------:|
| Length only | 0.491 [0.460, 0.526] | 0.774 [0.738, 0.814] | 0.866 [0.765, 0.948] | 0.000 |
| B0 perplexity | 0.515 [0.482, 0.546] | 0.430 [0.384, 0.472] | 0.134 [0.050, 0.242] | 0.105 |
| B1 TF-IDF | 0.902 [0.886, 0.917] | 0.675 [0.637, 0.711] | 0.537 [0.418, 0.660] | 0.050 |
| B1-Optuna | 0.922 [0.907, 0.935] | 0.767 [0.736, 0.800] | 0.559 [0.441, 0.677] | 0.069 |
| B2 row 2 (3 seeds) | 0.987 ± 0.004 | 0.346 ± 0.128 | 0.281 ± 0.224 | 0.002 |

Table: AUROC on each held-out set (bootstrap 95% CI; B2: mean ± SD over seeds) and the false-positive rate on MCPTox at the deployed threshold (target 1%). From `report/baselines/*.csv`. {#tbl:heldout}

In distribution every trained detector separates attacks from benign text, B2 almost
perfectly. On MCP tool descriptions the ordering reverses: B1-Optuna is the strongest trained
detector but only ties text length, and B2 ranks tool descriptions below chance
(Table \ref{tbl:heldout}). The rest of this report explains these two facts.

# Robustness {#sec:robustness}

## Distribution shift {#sec:shift}

The leave-one-surface-out design is a distribution-shift test, and every detector fails it.
Three results show that the failure is in the ranking itself, not in calibration:

- **Ranking collapses.** B2 drops from 0.987 to 0.346 AUROC, below chance, so no threshold can
  rescue it.
- **The deployed threshold does not carry over.** Set at 1% FPR on validation, it gives 5.0%
  FPR for B1 and 6.9% for B1-Optuna on MCPTox.
- **Length explains as much as the detectors.** B1 beats every cheap-cue probe in distribution,
  but on MCPTox a length probe's upper bound (0.874) exceeds B1's lower bound (0.615). MCPTox
  attacks are longer than its benign descriptions: every attack falls in the top half of benign
  lengths.

Section \ref{sec:errors} shows what changed between the surfaces.

## Noise: rewritten attacks {#sec:noise}

Each test-split attack is rewritten eleven ways (`src/eval/red_team.py`): character
obfuscation, formatting wrappers, paraphrase and reordering. Benign rows receive the same
rewrite as a control for false positives. Each detector is scored at its 5% validation-FPR
threshold, the operating point of the red-team tool, so the original recall differs from the 1%
threshold used elsewhere.

| Mutation | Family | B1 | B1-Optuna | B2 (seed 7267) |
|:-----------------|:-------------|---------:|---------:|---------:|
| (original) | – | 0.647 | 0.725 | 0.984 |
| leetspeak | obfuscation | 0.280 (−0.367) | 0.213 (−0.512) | 0.110 (−0.873) |
| case noise | obfuscation | 0.647 (+0.000) | 0.725 (+0.000) | 0.124 (−0.860) |
| zero-width characters | obfuscation | 0.328 (−0.320) | 0.483 (−0.242) | 0.237 (−0.746) |
| homoglyphs | obfuscation | 0.516 (−0.132) | 0.614 (−0.110) | 0.766 (−0.217) |
| JSON wrapper | format | 0.606 (−0.041) | 0.669 (−0.056) | 0.814 (−0.170) |
| Markdown wrapper | format | 0.614 (−0.033) | 0.713 (−0.012) | 0.781 (−0.203) |
| HTML comment | format | 0.647 (+0.000) | 0.730 (+0.005) | 0.941 (−0.043) |
| whitespace | format | 0.647 (+0.000) | 0.778 (+0.053) | 0.984 (+0.000) |
| synonyms | paraphrase | 0.646 (−0.002) | 0.727 (+0.002) | 0.984 (+0.000) |
| polite, indirect | paraphrase | 0.647 (+0.000) | 0.679 (−0.046) | 0.984 (+0.000) |
| sentence shuffle | reorder | 0.647 (+0.000) | 0.689 (−0.036) | 0.985 (+0.002) |
| benign FPR, original | – | 0.056 | 0.074 | 0.074 |

Table: Recall on rewritten test-split attacks at each detector's 5% validation-FPR threshold. From `report/red_team/*/summary.csv`. {#tbl:noise}

B2's scores come from the checkpoint, run locally on 16,627 unique texts; before using them we
checked that they reproduce the committed Colab score file (Spearman 0.9999 on val and test).

- **Meaning-preserving rewrites are harmless; character-level ones are not.** Synonyms, polite
  rewording and sentence order cost at most 0.05 recall for any detector. Character obfuscation
  is the cheapest evasion for all three.
- **The most accurate detector is the most brittle.** B2 loses almost all recall to leetspeak
  (0.984 to 0.110), random capitalization (0.124) and zero-width characters (0.237), and 0.17 to
  0.20 to a JSON or Markdown wrapper. DeBERTa's tokenizer is case-sensitive and splits obfuscated
  words into unfamiliar subword pieces, so a rewrite that leaves the text readable to a person
  moves it far from anything seen in training. The TF-IDF models lowercase their input, so case
  noise cannot touch them.
- **B1-Optuna's character n-grams cut both ways.** It resists zero-width characters better than
  B1 (0.483 vs 0.328) but loses more to leetspeak (0.213 vs 0.280), and whitespace noise raises
  its recall only because it raises its false-positive rate on rewritten benign text as well
  (0.074 to 0.131).

The fix we can measure, input normalization, is in Section \ref{sec:mitigation}.

# Fairness: Metrics by Group {#sec:fairness}

The data contains no people, so demographic fairness does not apply. The groups that matter
are the ones a deployed detector treats differently: **attack surface and source, attack kind,
text length, and script**. The fairness harm is benign content that gets blocked, which falls on
legitimate tool authors and document owners, so FPR is the headline column. Length quartiles
use edges from each eval set's benign rows.

| Eval set | Group | Benign | Adv. | B1-Optuna FPR [Wilson] | B1-Optuna TPR | B2 FPR (mean) | B2 TPR (mean) |
|:------|:-----------------|----:|----:|:-------------------|-----:|-----:|-----:|
| test | BIPIA (tool output) | 377 | 377 | 0.034 [0.020, 0.058] | 0.501 | 0.008 | 0.946 |
| test | PoisonedRAG (RAG) | 450 | 230 | 0.016 [0.008, 0.032] | 0.674 | 0.024 | 0.986 |
| test | length Q1 (short) | 207 | 240 | 0.034 [0.016, 0.068] | 0.679 | 0.050 | 0.986 |
| test | length Q4 (long) | 207 | 212 | 0.010 [0.003, 0.035] | 0.425 | 0.002 | 0.940 |
| MCPTox | length Q1 (short) | 91 | 0 | 0.000 [0.000, 0.041] | – | 0.000 | – |
| MCPTox | length Q2 | 88 | 0 | 0.034 [0.012, 0.095] | – | 0.000 | – |
| MCPTox | length Q3 | 92 | 230 | 0.098 [0.052, 0.176] | 0.252 | 0.000 | 0.004 |
| MCPTox | length Q4 (long) | 91 | 255 | 0.143 [0.085, 0.229] | 0.435 | 0.007 | 0.010 |
| MCPTox | CJK script | 23 | 2 | 0.000 [0.000, 0.143] | 0.000 | 0.000 | 0.000 |
| MCPTox | Latin script | 339 | 483 | 0.074 [0.050, 0.107] | 0.350 | 0.002 | 0.008 |

Table: Error rates by group at the deployed threshold (1% validation FPR). Every group, detector and eval set, with AUROC intervals, is in `report/phase5/fairness.csv`. {#tbl:fairness}

- **Long tool documentation bears the false alarms.** On MCPTox, B1-Optuna flags none of the
  shortest quarter of benign descriptions but 14.3% [8.5%, 22.9%] of the longest, fourteen
  times its target. Its intervals for Q1 and Q4 do not overlap. B1 shows the same gradient (2.2%
  to 7.7%).
- **In distribution the gradient reverses.** On the test split, short benign texts are flagged
  most (B1-Optuna 3.4% for Q1, 1.0% for Q4; B2 5.0% vs 0.2%), and long attacks are missed most
  (B1-Optuna recall 0.425 on Q4). The direction of the length bias depends on the surface, so a
  single global threshold cannot be fair to both.
- **Tool outputs are harder than RAG passages.** BIPIA's benign tool outputs are flagged twice as
  often by B1-Optuna (3.4% vs 1.6%), and its attacks are caught less often (0.501 vs 0.674).
- **B2's low MCPTox FPR is not fairness.** B2 flags almost nothing on MCPTox, benign or not
  (recall 0.008), so its near-zero FPR comes from not working there.
- **Script.** Only 25 MCPTox rows contain Chinese, Japanese or Korean text (23 benign), and no
  detector flags them; with 23 benign rows the FPR interval reaches 14%, so this is a gap in
  the data rather than evidence of fairness. All training text is English.

# Latency and Compute Cost {#sec:cost}

| Detector | Device | ms per row | Rows per second | Parameters | Fit or training time |
|:----------------|:----|------:|------:|------:|:------------------------|
| Length only | CPU | <0.001 | > 10^6^ | 2 | 0.02 s |
| B1 TF-IDF | CPU | 0.42 | 2,382 | 585,055 | 18 s (three values of `C`) |
| B1-Optuna | CPU | 0.58 | 1,727 | 377,215 | 7 s refit; 55 min for the 100-trial search |
| B0 perplexity | CPU | 57.6 | 17 | 81.9 M | none (frozen DistilGPT2) |
| B0 perplexity | Apple GPU (MPS) | 26.6 | 38 | 81.9 M | none |
| B2 DeBERTa row 2 | CPU | 56.2 | 18 | 184.4 M | not recorded; 6–9 T4 GPU-hours for all B2 runs (estimate) |
| B2 DeBERTa row 2 | Apple GPU (MPS) | 44.0 | 23 | 184.4 M | as above |

Table: Inference cost on the same 1,000 test rows (mean 1,056 characters), median of five passes after one warm-up, on an Apple M4 Max laptop. Fit time is wall time on the training split; the search time is the sum of trial durations in `report/phase4/b1_optuna/trials.csv`. From `report/phase5/latency.csv`. {#tbl:cost}

- **The best transferring detector is also the cheapest.** B1-Optuna scores a text in 0.58 ms on
  one CPU core, about 100 times faster than B2 on the same CPU (56.2 ms) and 76 times faster than
  B2 on the laptop GPU (44.0 ms). At B1-Optuna's speed, screening every retrieved passage or tool
  output adds well under a millisecond per item to an agent's step; B2 adds tens of
  milliseconds per item, more for long documents, which B2 splits into 256-token windows.
- **Size.** B1-Optuna is a 377,215-weight linear model; B2 has 184 million parameters and a 704 MB
  checkpoint. B2's extra capacity bought in-distribution accuracy (test AUROC 0.987 vs 0.922) and
  nothing on the unseen surface.
- **Training cost.** B1-Optuna's whole search took 55 minutes on a laptop CPU and its final fit
  7 seconds. B2's training time was not recorded during the Colab runs; the run plan estimated
  6 to 9 T4 GPU-hours for the nine reported runs and the sweeps.

# Error Analysis: A Failure-Mode Taxonomy {#sec:errors}

**Method.** We read a sample of MCPTox errors, wrote a codebook of six textual cues, and
assigned every error at the deployed threshold to one category: the first rule whose cues match,
or "other" (`src/eval/error_taxonomy.py`). Missed attacks are categorized by how they are
phrased; false alarms by what in a benign text resembles an attack. The codebook was revised
once, after hand-checking 29 coded rows showed missing verbs (fetch, prepend, modify). It was
then checked on a fresh sample of 40 errors, drawn with a different seed and not used to write
it: the codebook's category agreed with a reading of the full text for 36 of 40 rows (90%; 21
of 24 misses and 15 of 16 false alarms). All four disagreements were value substitutions filed
under a neighboring category. The check was made by the AI assistant that wrote the codebook,
not by a second human coder, so we report agreement rather than Cohen's $\kappa$
[@cohen1960coefficient] (`report/phase5/errors/check.csv`).

| Error | Category | B1-Optuna | B2 (seed 7267) |
|:------|:----------------------------------------------|------:|------:|
| Missed attack | procedural prerequisite ("before X, call Y") | 185 (59%) | 246 (51%) |
| Missed attack | argument or parameter override | 76 (24%) | 162 (34%) |
| Missed attack | authority or policy framing | 20 (6%) | 31 (6%) |
| Missed attack | plain imperative, no injection vocabulary | 20 (6%) | 25 (5%) |
| Missed attack | other | 15 (5%) | 17 (4%) |
| False alarm | usage instructions addressed to the model | 13 (52%) | 2 (100%) |
| False alarm | repository or account action | 5 (20%) | 0 |
| False alarm | sensitive or destructive vocabulary | 1 (4%) | 0 |
| False alarm | other | 6 (24%) | 0 |

Table: MCPTox errors at the deployed threshold, by category. B1-Optuna misses 316 of 485 attacks and raises 25 false alarms; B2 misses 481 and raises 2. From `report/phase5/errors/taxonomy.csv`. {#tbl:taxonomy}

| Cue | MCPTox benign | MCPTox attack | Test benign | Test attack |
|:--------------------------------|------:|------:|------:|------:|
| procedural prerequisite | 0.006 | 0.507 | 0.008 | 0.010 |
| argument override | 0.041 | 0.373 | 0.001 | 0.002 |
| authority or policy claim | 0.025 | 0.379 | 0.034 | 0.040 |
| usage imperative (must, always, only…) | 0.163 | 0.866 | 0.169 | 0.206 |

Table: Share of rows containing each cue. From `report/phase5/errors/cue_rates.csv`. {#tbl:cues}

Table \ref{tbl:cues} explains Table \ref{tbl:taxonomy}. MCPTox attacks do not look like the
attacks in the training pool. Half of them are written as a **procedural prerequisite**: a
fake rule, inside a tool's description, that tells the agent to call another tool first, or to
change an argument. Neither phrasing occurs in the training pool: 1.0% and 0.2% of test-split
attacks, against 50.7% and 37.3% of MCPTox attacks. The detectors learned what injected
instructions look like inside retrieved passages and tool outputs ("ignore previous
instructions", requests to exfiltrate or answer differently), and MCPTox attacks contain little
of that vocabulary. The same cues separate MCPTox's own classes well (a prerequisite cue occurs
in 50.7% of attacks and 0.6% of benign descriptions), so the surface is learnable; the
detectors were never shown it.

The false alarms are the mirror image: half of B1-Optuna's are benign descriptions that give the
model usage instructions ("IMPORTANT: always use absolute paths", "only works within allowed
directories"). Such imperatives occur in 16% of benign descriptions; in the training pool they
are only slightly more common in attacks than in benign text (21% vs 17%), so the imperative
words alone do not explain the alarms. Section \ref{sec:shap} shows which features do.

**Test split.** In distribution the errors follow the source rather than the phrasing: 71% of
B1-Optuna's 263 missed test attacks are BIPIA instructions, and the missed attacks are long
(median 1,214 characters against 716 for all attacks), where one injected sentence is diluted
in a passage. B2 misses only 17 test attacks.

# Feature Attribution with SHAP {#sec:shap}

For a linear model on TF-IDF features, SHAP values [@lundberg2017unified] are exact: feature $i$
of a text contributes $w_i (x_i - \bar{x}_i)$ to the log-odds, and the contributions add up to
the model's logit. We compute them with `shap.LinearExplainer` against a background of 200
training rows (`src/eval/explain_tfidf.py`; a test checks the additivity).

![B1-Optuna's 15 features with the largest mean |SHAP| on the test split, signed by direction, and the same features on MCPTox.](shap/b1_optuna_global.png){#fig:shap-global width=95%}

**What B1-Optuna learned.** Its strongest attack features are second-person phrasing addressed
to the model about its *output*: *your*, *in your*, *response*, *your response*, *reply*
(Figure \ref{fig:shap-global}). This is the voice of BIPIA's injected instructions ("in your
response, include…"). The strongest benign features are function words (*and*, *to*, *of*),
which mark ordinary prose. B1 learned the same list (`report/phase5/shap/global.csv`).

| Word n-gram | Coef. | Test benign | Test attack | MCPTox benign | MCPTox attack |
|:---------------|------:|------:|------:|------:|------:|
| your | 13.4 | 0.127 | 0.445 | 0.017 | 0.027 |
| in your | 10.7 | 0.013 | 0.166 | 0.008 | 0.002 |
| response | 7.9 | 0.015 | 0.171 | 0.025 | 0.004 |
| your response | 9.6 | 0.000 | 0.150 | 0.000 | 0.000 |
| reply | 7.6 | 0.034 | 0.145 | 0.003 | 0.002 |

Table: Share of rows containing B1-Optuna's top attack word n-grams. From `report/phase5/shap/presence.csv`. {#tbl:presence}

**Why it misses MCP attacks.** These words are almost absent from MCPTox attacks: *your*
appears in 44.5% of test attacks but 2.7% of MCPTox attacks, and *your response* in 15% against
none (Table \ref{tbl:presence}). An MCP attack instructs the agent about *tools* ("before using
X, call Y"), not about its response. With no learned feature present, the score of a missed
attack is the sum of many small contributions: in Figure \ref{fig:shap-local} no feature moves
a missed attack by more than 0.33 log-odds, and *must* and *you* push toward benign.

![Local SHAP explanations for B1-Optuna's three highest-scoring false alarms (top) and three lowest-scoring missed attacks (bottom) on MCPTox. Red pushes toward "attack".](shap/b1_optuna_local.png){#fig:shap-local width=100%}

**Why it flags benign descriptions.** The same features, when they do appear in benign text,
are decisive. "Create a new GitHub repository **in your** account" gains 2.4 log-odds from
*in your* alone; a Slack tool named for replying gains 1.5 from *reply*; a database tool that
asks the agent to "**provide** the current working directory of the **users** project" is
flagged on *in your*, *your* and *provide*. The detector reacts to ordinary second-person
phrasing in tool documentation, which explains both the false alarms in
Table \ref{tbl:taxonomy} and why longer descriptions, with more chances to contain such
phrasing, are flagged more (Table \ref{tbl:fairness}).

# Saliency Maps for DeBERTa {#sec:saliency}

We use integrated gradients [@sundararajan2017axiomatic] rather than raw attention weights,
which are not a faithful explanation of a prediction [@jain2019attention]. For each text we take
the 256-token window that sets its score (B2 scores a text by its highest-scoring window) and
attribute that window's logit to its tokens with Captum [@kokhlikyan2020captum]: 300 steps, a
baseline that replaces every token except [CLS] and [SEP] with [PAD]
(`src/eval/saliency_b2.py`). We attribute 30 randomly chosen rows from each of four groups:
MCPTox benign, MCPTox attack, test benign and test attack. To test the Phase 4 hypothesis, that
B2 misranks MCPTox because benign tool descriptions read like instructions, we measure the
share of each row's positive (toward "attack") attribution that falls on instruction words:
modals and imperatives (*must, always, never, should, ensure, use, call, run, first, before,
important*) and second person (*you, your*).

We first checked convergence. At 50 steps, the completeness error (attributions should sum to
the logit difference from the baseline) reached about 20% of the logit on confident rows; at
300 steps its median is at most 0.0013 log-odds in every group, so all results below use 300
steps.

| Group | Rows | Median logit | Instruction-word share of positive attribution [95% CI] | Instruction words, share of tokens |
|:----------------|----:|------:|:-------------------|------:|
| Test benign | 30 | −8.96 | 0.000 [0.000, 0.001] | 0.006 |
| Test attack | 30 | +8.80 | 0.060 [0.035, 0.090] | 0.014 |
| MCPTox benign | 30 | −8.61 | 0.039 [0.001, 0.111] | 0.011 |
| MCPTox attack | 30 | −8.61 | 0.062 [0.027, 0.104] | 0.054 |

Table: Integrated-gradients attribution for B2 (seed 7267) by group. From `report/phase5/saliency/summary.csv`. {#tbl:saliency}

**The Phase 4 hypothesis is not supported.** Instruction words receive about 6% of B2's
attack-ward attribution in attacks and 4% in benign tool descriptions (Table \ref{tbl:saliency}),
with overlapping intervals. Instruction words are almost four times as common in MCPTox attacks as in
test attacks (5.4% vs 1.4% of tokens), yet they do not raise the score: B2 has not learned that
imperatives addressed to the agent signal an attack.

**What B2 attends to instead.** On test attacks, the tokens with the highest attribution are
*your*, *response*, *reply* and *in*, the same second-person output phrasing that SHAP found in
B1-Optuna. Two very different models learned the same surface cue from BIPIA. On MCPTox, the
logits of attacks and benign descriptions have the same median (−8.61) and no sampled attack
scores above −6.4; attribution falls on formatting tokens (`Tool:`, `Description`, punctuation,
quotes), and in the heatmaps (Figure \ref{fig:saliency}) almost every token in an attack pushes
toward benign, including "you MUST first use". Only *Before* draws a weak push toward attack.
Across all three seeds, 98–100% of MCPTox rows, benign and attack alike, score below 0.01.

![Integrated-gradients heatmaps for three benign and three attack MCPTox tool descriptions (B2, seed 7267). Red pushes toward "attack", blue toward "benign"; the first 60 tokens are shown.](saliency/heatmaps.png){#fig:saliency width=100%}

So B2's below-chance AUROC on MCPTox (Table \ref{tbl:heldout}) is not a confident inversion.
B2 treats the whole surface as benign, and its ranking is an ordering of near-zero scores, set
by formatting rather than by content. The fix is the one the error analysis points to: training
data from the surface.

# Potential Harms and Misuse {#sec:harms}

Each harm below is grounded in a result from this project.

1. **A false sense of security.** A detector that scores 0.99 AUROC in distribution and below
   chance on a new surface invites deployment where it does not work. Aggregate in-distribution
   metrics hide exactly the failure that matters.
2. **Blocking legitimate content.** At a 1% target, B1-Optuna flags 14.3% of long benign MCP tool
   descriptions (Table \ref{tbl:fairness}). The cost falls on honest tool authors, and most on
   those who write careful usage instructions, the documentation practice we would want to
   encourage.
3. **Dual use of this work.** The red-team results, SHAP features and taxonomy tell an attacker
   which tokens to avoid, which obfuscations work (leetspeak alone cuts B1-Optuna's recall from
   0.725 to 0.213) and which phrasing goes unnoticed (procedural prerequisites).
4. **The dataset as an attack corpus.** It collects working injection strings in one place.
   Each is already public, but aggregation lowers the cost of reuse.
5. **Over-reliance in place of architecture.** A probabilistic detector used instead of
   privilege separation or tool allow-lists fails open on a single miss, and the best detector
   here misses 65% of MCP attacks.
6. **Automation bias and accountability.** If a flag blocks a tool automatically, someone must own
   the review and the appeal; a tool author whose description is blocked has no way to learn why.

# Mitigation Strategies {#sec:mitigation}

| Harm | Mitigation | Evidence |
|:---------------------|:----------------------------------------------|:------------------|
| 1. False security | Score only surfaces in the validation set; publish the out-of-scope list (model card) | Section \ref{sec:shift} |
| 2. Blocking legitimate content | Thresholds per surface, set on benign data from that surface; flag for review instead of blocking; train on benign tool descriptions | Sections \ref{sec:fairness}, \ref{sec:errors} |
| 3. Dual use | Release aggregate red-team numbers and the codebook, not an evasion recipe; normalize input before scoring | Table \ref{tbl:noise} |
| 4. Attack corpus | Keep raw data behind DVC/R2 access; commit scores and row ids, not text (as this phase does) | `DATA_CARD.md` |
| 5. Over-reliance | Defense in depth: detector plus privilege separation, allow-lists and user confirmation of side-effecting calls; measure end-to-end attack success [@debenedetti2024agentdojo] | Section \ref{sec:errors} |
| 6. Accountability | Log every flag with its score and threshold; give tool authors an appeal path; monitor score drift per surface | – |

Table: One mitigation per harm in Section \ref{sec:harms}. {#tbl:mitigation}

**A measured mitigation: input normalization.** We rescored the obfuscated test attacks after
applying Unicode NFKC normalization and removing zero-width characters before B1-Optuna sees
the text (`src/eval/mitigation_normalize.py`; the model itself is unchanged). Zero-width
attacks are fully repaired: recall returns from 0.483 to 0.725, the unmutated level.
Homoglyphs (0.614) and leetspeak (0.213) are unchanged, because NFKC does not map Cyrillic
letters or digits to Latin letters. Closing those gaps needs a confusables map
(Unicode TS 39) or training on obfuscated text; normalization alone is necessary but not
sufficient.

**The mitigation the taxonomy points to.** The most effective fix is data, not modeling: the
training pool needs benign and adversarial tool descriptions, especially procedural
prerequisites and argument overrides, and benign descriptions that contain usage instructions.
Since MCPTox is our only held-out tool-description set, such data must come from a different
source (or be synthesized) so that MCPTox remains a fair test.

# Limitations {#sec:limitations}

- The analyses explain frozen detectors; they cannot test fixes that need retraining, such as
  adding tool descriptions to training.
- The error taxonomy is rule-based. It was checked on 40 held-out rows (90% agreement) by the AI
  assistant that wrote it, not by independent human coders.
- MSB has 21 benign rows and MCPTox has 23 benign CJK rows, so those groups are indicative.
- B2's noise, saliency and latency results use one seed (7267); its seeds differ widely on
  MCPTox (AUROC 0.247 to 0.491).
- Attribution methods explain a model's computation, not the cause of an attack's success. SHAP
  on a linear model is exact; integrated gradients depend on the baseline, and we use one.
- Latency was measured on one laptop; B2's training time was not recorded during the Colab
  runs, so its training cost is the run plan's estimate.

# Conclusion {#sec:conclusion}

The detectors do not transfer to MCP tool descriptions because they learned a voice, not a
behavior. Both the TF-IDF model and DeBERTa key on second-person phrasing about the model's
*response*, the way BIPIA's attacks are written. MCP attacks are written as tool procedure ("before
using X, call Y"; "change the parameter to …"), a phrasing that half of them use and that the
training data almost never contains, so the detectors miss them; and benign tool documentation
that happens to say "in your account" or "reply" is flagged, more often the longer it is.
DeBERTa does not invert the new surface so much as fail to respond to it: it scores nearly every
MCP description as benign. The same analysis shows that the most accurate model in distribution
is the most fragile: character obfuscation and even random capitalization remove most of its
recall.

We would deploy B1-Optuna, as documented in the model card, only on the surfaces it was
validated on, with input normalization, per-surface thresholds, review instead of automatic
blocking, and architectural defenses around it. It is also the cheapest detector by two orders
of magnitude. The next step is not a larger model but different data: benign and adversarial
tool descriptions from a source other than MCPTox, with MCPTox kept as the held-out test, and
an end-to-end measure of how much a detector reduces attack success in an agent.

**Use of AI assistance.** The Phase 5 analysis scripts, the error codebook and its 40-row check,
the model card and this report were prepared with an AI assistant (Claude) working from the
committed result files, and merged through reviewed pull requests with tests. Every number in
this report is generated by a committed script. Analysis choices, decision rules and the final
text were reviewed by the team.

\appendix

# Reproducing the Results {#appendix-reproduce}

```bash
make install && make data      # dependencies; data and B2 score files (DVC)
make b2-checkpoint             # B2 row2-s7267 checkpoint, for noise, saliency and latency
make phase5-fairness           # scores.parquet, thresholds.json, fairness.csv
make phase5-errors             # errors/taxonomy.csv, cue_rates.csv, sample.csv
make red-team DETECTOR=tfidf-optuna
make red-team-b2               # B2 on the mutations (GPU or Apple MPS recommended)
uv run python src/eval/mitigation_normalize.py
make phase5-shap               # shap/
make phase5-saliency           # saliency/
make phase5-latency            # latency.csv (idle machine)
make report-phase5             # this PDF
```

# Model Card {#appendix-model-card}

The model card below is `MODEL_CARD.md` from the repository root [@mitchell2019model]. It
complements the dataset's datasheet [@gebru2021datasheets].
