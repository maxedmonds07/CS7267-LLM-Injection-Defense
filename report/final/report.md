---
title: "Detecting Prompt Injection Across RAG and MCP Surfaces: Detectors Learn a Voice, Not a Behavior"
author: |
  ```{=latex}
  Carter Corbin\qquad Max Edmonds\qquad Mohammed Adil Ahmed\\
  Oluwamayowa Adewummi\qquad Adedapo Odeyemi\\[0.4em]
  College of Computing and Software Engineering, Kennesaw State University, Kennesaw, GA\\
  \url{https://github.com/maxedmonds07/CS7267-LLM-Injection-Defense}
  ```
abstract: |
  LLM agents read text their users never wrote: retrieved passages, tool outputs, and the
  descriptions of Model Context Protocol (MCP) tools. Any of it can carry an indirect prompt
  injection. We ask whether a detector trained on two of these surfaces transfers to a third it
  never saw. We build an 11,005-row corpus from four public benchmarks, train a perplexity
  score, a TF-IDF logistic regression (tuned with a family-grouped Optuna search) and a
  fine-tuned DeBERTa-v3 on RAG passages and tool outputs, and hold out MCP tool descriptions
  entirely. In distribution the trained detectors reach 0.90–0.99 AUROC; on MCP tool descriptions the
  best of them only ties text length (0.767 vs 0.774) and DeBERTa falls below chance (0.346).
  Letting DeBERTa read the user's task alongside the text lifts it only to about chance (0.45–0.51).
  An error taxonomy, exact SHAP values and integrated gradients explain why: both trained models
  learned the *voice* of the training attacks, second-person phrasing about the model's
  response ("in your response", "reply"); the word *your* alone appears in 44.5% of
  in-distribution test attacks but 2.7% of MCP attacks. Half of MCP attacks are instead written as tool procedure ("before
  using X, you must first call Y"), a phrasing found in 1% of in-distribution test attacks. DeBERTa
  scores 98–100% of MCP tool descriptions below 0.01, so its below-chance AUROC is noise
  among near-zero scores, and it is also the most brittle model: leetspeak cuts its recall
  from 0.984 to 0.110. False alarms fall on long benign tool documentation (14.3% false-positive
  rate on the longest quarter against a 1% target). We release the code, a model card, and a
  harm-by-harm mitigation plan, and argue that the bottleneck is training data from the
  surface, not model capacity.
documentclass: article
numbersections: true
colorlinks: true
linkcolor: "black"
citecolor: "blue"
urlcolor: "blue"
bibliography:
  - ../../literature-review/references.bib
  - ../phase2/extra.bib
  - ../phase4/extra.bib
  - ../phase5/extra.bib
  - extra.bib
csl: ieee.csl
link-citations: true
---

# Introduction {#sec:intro}

Applications built on large language models (LLMs) increasingly place text in the model's
context that the user never wrote: passages retrieved for retrieval-augmented generation (RAG),
documents returned by tools, and the descriptions of Model Context Protocol (MCP) tools that an
agent reads before it decides which tool to call. An attacker who controls any of that text can
plant instructions without interacting with the user, an attack Greshake et al. named *indirect
prompt injection* [@greshake2023indirect]. MCP widens the surface: a poisoned tool description
enters the context when a server is registered, before any tool runs, and reaches 72.8% attack
success on o1-mini [@wang2026mcptox].

A cheap defense is a detector that screens each untrusted segment before the model reads it.
Detectors are usually evaluated on the same benchmarks they were trained on, where standard
cross-validation overstates AUROC by 8.0–16.5 points relative to leave-one-dataset-out evaluation
[@fomin2026benchmarks]. An attacker, however, chooses the surface. We therefore ask: **does a
detector trained on RAG passages and tool outputs transfer to MCP tool descriptions, a surface
it never saw, and if not, why not?**

Our contributions are:

- **A leave-one-surface-out benchmark** (§\ref{sec:data}): 11,005 labeled segments from four
  public benchmarks, split by group, with MCP tool descriptions held out whole and a length-only
  reference that every detector must beat.
- **Three detector families** under one harness at a deployed 1% false-positive threshold
  (§\ref{sec:method}), including a TF-IDF model tuned on family-grouped folds and a fine-tuned
  DeBERTa-v3 with a pre-registered success criterion, plus a DeBERTa variant that reads the
  user's task with each text. The trained detectors fit their training surfaces; none transfers
  (§\ref{sec:results}).
- **An explanation of the failure** (§\ref{sec:why}): a rule-based failure taxonomy, exact SHAP
  values and integrated gradients show that the trained detectors key on the second-person voice
  of the training attacks, while MCP attacks are written as tool procedure.
- **An assessment for deployment** (§\ref{sec:deploy}): robustness to eleven rewrites, error
  rates by group, latency and cost, six harms with a mitigation for each (one measured), and a
  model card (Appendix \ref{app:card}).

Section \ref{sec:contrib} lists the method each team member developed.

# Related Work {#sec:related}

Our Phase 2 review coded 20 papers by attack surface, payload and defense class (its tables are
in the repository). For this report we extended it with a targeted search on injection detectors,
MCP tool poisoning and shortcut learning. Table \ref{tbl:coverage} places the work by surface.

## Attacks by surface {#sec:rw-attacks}

Prompt injection was first shown in its *direct* form, where the user's own input overrides the
developer's instructions [@perez2022ignore]. Greshake et al. showed the *indirect* form, in which
the instruction arrives in data the application retrieves [@greshake2023indirect], and Liu et al.
formalize the attack space and find that none of ten defenses suffices [@liu2024formalizing].
Indirect attacks differ by the surface they enter through. On the *RAG corpus*, PoisonedRAG
reaches 90% attack success with five malicious passages per target question
[@zou2025poisonedrag], and adversarial passages optimized against a dense retriever transfer
across corpora [@zhong2023poisoning]. In *tool outputs*, BIPIA [@yi2025bipia], InjecAgent
[@zhan2024injecagent] and AgentDojo [@debenedetti2024agentdojo] plant instructions in the emails,
tables, web pages and documents a tool returns. In *MCP tool descriptions*, the attack sits in
metadata the agent reads when a server is registered: MCPTox poisons tools on 45 real servers
[@wang2026mcptox], ToolHijacker crafts documents that win tool selection [@shi2026toolhijacker],
MSB covers every stage of the MCP pipeline [@zhang2026msb], and MCP-ITP optimizes *implicit*
poisoned descriptions that steer the agent to a different, legitimate tool while evading an
LLM-based detector [@li2026mcpitp]. The goal is the same on every surface, but the text is not:
a RAG attack is a short passage that answers a question wrongly, a tool-output attack addresses
the model from inside a document, and a tool-description attack is written as part of the tool's
own documentation (§\ref{sec:taxonomy}).

## Defenses that change the model or the system {#sec:rw-defenses}

Model-level defenses retrain the LLM to ignore instructions found in data: structured instruction
tuning separates the prompt and data channels [@chen2025struq], preference optimization rewards
following the intended instruction [@chen2025secalign], task-specific fine-tuning removes general
instruction following [@piet2024jatmo], and an instruction hierarchy trains the model to rank
privileged instructions above tool and user text [@wallace2024instruction]. Prompt-level defenses
mark untrusted input so the model can tell it apart [@hines2024spotlighting]. System-level
defenses enforce control- and data-flow policies outside the model [@debenedetti2026camel] or
certify RAG answers by isolate-then-aggregate [@xiang2026robustrag], and RAG forensics traces a
poisoned answer back to the passages that caused it [@zhang2025traceback]. Each of these needs
access to the model's weights, its prompt or the agent's architecture. A detector needs none: it
screens each untrusted segment before the model reads it, which is why it is often the first
defense a deployment adds, and why its failure modes matter.

## Injection detectors {#sec:rw-detectors}

Detectors fall into four families. *Fluency filters* flag text with high perplexity under a
language model. They catch the gibberish adversarial suffixes found by gradient search
[@zou2023universal] but not human-written attacks, whose perplexity resembles ordinary text
[@alon2023perplexity; @jain2023baseline]; our B0 tests whether the idea carries over to indirect
injection. *Fine-tuned classifiers* train an encoder on labeled injections. PromptShield
fine-tunes a pretrained encoder and argues that detectors must be judged at very low
false-positive rates [@jacob2025promptshield], and InjecGuard shows that such guardrails
*over-defend*, flagging benign text that contains trigger words common in attacks
[@li2024injecguard], the failure we observe on benign tool documentation (§\ref{sec:shap}). Our
B1 and B2 belong to this family. *LLM-based detectors* use a language model as the judge;
DataSentinel fine-tunes one in a minimax game against adaptive attacks [@liu2025datasentinel].
*Internal-signal detectors* read the protected model's own state: Attention Tracker flags inputs
that pull specific attention heads away from the original instruction [@hung2025attention], and
RevPRAG detects RAG poisoning from activations [@tan2025revprag]. The last family needs
white-box access to the protected LLM, and the third an LLM call per segment. We study the first
two, which are cheap enough to run on every segment and independent of the model they protect.

## Evaluation under distribution shift {#sec:rw-shift}

Classifiers often learn *shortcuts*: features that predict the label in the training distribution
but are not the intended concept [@geirhos2020shortcut]. Models trained on natural-language
inference data learn annotation artifacts such as negation words [@gururangan2018annotation] and
syntactic heuristics such as word overlap [@mccoy2019right], and score well until a test set
breaks the artifact. Domain-generalization work therefore holds out whole domains at test time
[@gulrajani2021search; @koh2021wilds]. Injection detection has only recently adopted this
practice. Fomin shows that injection classifiers learn dataset identity, with 28–44% of their top
features being dataset shortcuts, and that leave-one-dataset-out testing lowers AUROC
[@fomin2026benchmarks]. Concurrently with our work, Liu replays benign tool calls from two agent
benchmarks, finds that detector rankings on BIPIA do not carry over to agent tool outputs, and
attributes the gap to the form of the training inputs [@liu2026passing]. Both hold out
*datasets*; neither holds out a *surface*, and neither traces the failure to the features the
detector uses.

| Surface | Attacks and benchmarks | Detector evaluations | Held out of training |
|:----------------|:--------------------------|:----------------------------------|:--------------------|
| RAG corpus | [@zou2025poisonedrag; @zhong2023poisoning] | activations, white-box [@tan2025revprag] | no |
| Tool output | [@yi2025bipia; @zhan2024injecagent; @debenedetti2024agentdojo] | general classifiers [@jacob2025promptshield; @fomin2026benchmarks; @liu2026passing] | datasets [@fomin2026benchmarks; @liu2026passing] |
| MCP tool description | [@wang2026mcptox; @shi2026toolhijacker; @zhang2026msb; @li2026mcpitp] | a detector as the target of evasion [@li2026mcpitp] | whole surface (this work) |

Table: Prior work by the surface an attack enters through. *Held out of training*: whether a detector is tested on data excluded from its training, and at what level. From our 20-paper review and a targeted search. {#tbl:coverage}

## The research gap {#sec:rw-gap}

Three gaps follow, and each maps to a contribution. *(1) No detector has been tested on MCP tool
descriptions held out of training.* MCP attacks are benchmarked [@wang2026mcptox; @zhang2026msb],
and in the work we found a detector appears on this surface mainly as something an attack is
optimized to evade [@li2026mcpitp]. We hold the surface out whole and report every detector next
to a length baseline (§§\ref{sec:data}–\ref{sec:results}). *(2) Transfer has been measured across
datasets, not surfaces.* Leave-one-dataset-out testing [@fomin2026benchmarks] and cross-benchmark
replay [@liu2026passing] score new data from a familiar surface; we score a surface whose attacks
are written differently. *(3) The failure has not been explained by the features a detector
uses.* Shortcut learning predicts that detectors key on incidental cues [@geirhos2020shortcut],
and over-defense shows that trigger words cause false alarms [@li2024injecguard]. We name the cue
for both a linear and a transformer detector with exact SHAP values and integrated gradients, and
measure how often it appears on each surface (§\ref{sec:why}). Throughout, we follow Fomin's
advice to report a length baseline and PromptShield's to report low-FPR operating points.

# Data {#sec:data}

The corpus unifies four public benchmarks into one table of *untrusted text segments*, any text
that reaches the model without the user writing it (Table \ref{tbl:data}). PoisonedRAG
contributes its malicious passages for 300 target questions in NQ, HotpotQA and MS MARCO, with
the top-10 real passages a Contriever retriever returns for the same question as hard benign
negatives [@zou2025poisonedrag; @thakur2021beir]. BIPIA contributes every clean context (emails,
tables, code, news [@trischler2017newsqa] and summaries [@narayan2018xsum]) as a benign row and
one attacked copy, its *twin*, as an adversarial row. MCPTox contributes 485 poisoned tool
descriptions and the clean descriptions of the same servers; MSB contributes tool descriptions
and outputs rebuilt from its templates.

| Source | Surface | Role | Benign | Adv. | Median chars |
|:---------------|:---------------|:-------------------|------:|------:|:---------------|
| PoisonedRAG | RAG passage | train pool | 2,955 | 1,500 | 389 / 183 |
| BIPIA | tool output | train pool | 2,806 | 2,806 | 1,752 / 1,816 |
| MCPTox | tool description | LOSO holdout | 362 | 485 | 88 / 292 |
| MSB | tool desc./output | transfer holdout | 21 | 70 | 156 / 314 |

Table: Corpus sources. Median length in characters, benign / adversarial. The training pool is split 70/15/15 by group into train (4,133 benign / 3,084 adversarial), validation (801 / 615) and test (827 / 607). {#tbl:data}

**Splits.** Rows are split by *group* (a question, a BIPIA context with its twin, an MCP server),
so a group never spans two splits; BIPIA's own train/test boundary is kept, so most of its test
attack families are unseen in training. MCPTox and MSB are held out whole: no row from either is
used to fit, select or threshold any model. A guard in the data loader raises an error if a
held-out source or the test split reaches fitting.

**Cleaning and the length shortcut.** We removed formatting cues that reveal the label
(surrounding whitespace from BIPIA's insertion, passage titles only benign rows had,
source-specific tool layouts) and 47 duplicate texts. Length remains a shortcut, and its
direction flips between sources (Table \ref{tbl:data}): poisoned passages are shorter than real
ones, poisoned tool descriptions longer than clean ones. A logistic regression on log length is
near chance in distribution (AUROC 0.491) but reaches 0.774 on MCPTox and 0.866 on MSB, so every
detector is reported next to it.

**Ethics of the data.** All rows come from public benchmarks; the project collects nothing from
people, so it is not human-subjects research under 45 CFR 46.102. Some upstream sources forbid
redistribution and MCPTox states no license, so the rebuilt rows stay in a private DVC remote
and we release the pipeline that rebuilds them, documented in a datasheet-style data card
[@gebru2021datasheets; @pushkarna2022datacards].

# Detectors and Evaluation Protocol {#sec:method}

**Length only.** Logistic regression on log character length: the shortcut to beat.

**B0: perplexity.** Frozen DistilGPT2 [@sanh2019distilbert; @radford2019language] perplexity over
512-token sliding windows. It is not trained on our labels and tests the hypothesis that
injected text is less fluent than benign text.

**B1: TF-IDF + logistic regression** [@pedregosa2011scikit] with word 1–2-grams and
character 3–5-grams; `C` chosen from {0.1, 1, 10} on validation. **B1-Optuna** replaces that
search with a 100-trial TPE study [@akiba2019optuna; @bergstra2011algorithms] over the feature
set, n-gram ranges, `C`, `min_df`, vocabulary size, sublinear term frequency and class weights.
Its objective is the key choice: validation holds the same attack families as training, so each
trial is scored by mean AUROC over five **family-grouped folds** of the training split, in which
each of BIPIA's 25 attack families is held out whole. The best trial uses word 1–3-grams plus
character 2–4-grams, `C` = 6.38 and balanced class weights (mean held-out-family AUROC 0.921).

**B2: fine-tuned DeBERTa-v3-base** [@he2023debertav3], following PromptShield's recipe
[@jacob2025promptshield], with two changes aimed at transfer: *chunked max-pool scoring* (a text's
score is the highest score among its overlapping 256-token windows, since an injection is a short
span in benign text) and a *pairwise margin loss* that pushes each BIPIA attacked twin above its
clean twin, pointing the gradient at the injected span. Three configurations (plain; + pairwise
loss; + pairwise loss trained on BIPIA only) were trained with three seeds each on a Colab T4
(learning rate 2e-5, batch 16, 3 epochs). Row 2 (pairwise loss, both sources) was selected on
validation.

**B4: DeBERTa that reads the task.** B2 sees only the candidate text, so it cannot tell whether
an instruction belongs to what the user asked for. B4 pairs each candidate with trusted task
context: the original question for PoisonedRAG and BIPIA (code and error text, or a fixed
summarization instruction, for BIPIA's code and abstract tasks), and for MCPTox and MSB the
server's catalogue of legitimate user tasks, a proxy for an individual request. DeBERTa-v3-base
reads up to 128 task tokens with each 256-token candidate window, without B2's pairwise loss.
Four configurations isolate each change: the candidate alone (`text`), the task and candidate
together (`pair`), the pair plus two lexical relevance features, character TF-IDF cosine and
task-word coverage (`relevance`), and `relevance` trained on BIPIA only. Each was trained with
three seeds under B2's settings, and `relevance` was selected on validation. Holdout rows with no
task context (175 of 847 on MCPTox, 15 of 91 on MSB) are excluded, so B4 is compared with length
and B1 refitted on the same rows rather than with Table \ref{tbl:main}.

**Protocol.** One harness scores every detector. The **deployed threshold** is the score that
gives 1% FPR on the validation split's benign rows, applied unchanged to every eval set. We
report AUROC with a 95% bootstrap interval that resamples whole groups [@efron1993introduction];
deployed TPR and FPR, with a Wilson interval for FPR [@wilson1927probable]; and *length-binned
AUROC*, AUROC within bins of similar length, averaged, so that a pure length ranker scores 0.5.
B2 is reported as mean ± SD over seeds. Three rules were fixed before training: holdouts never
choose a model, hyperparameter, seed or threshold; every run is reported; and B2 counts as beating
B1 only if, on MCPTox, *every* seed of the selected configuration has 95% lower bounds above B1's
TPR at 5% FPR (0.225) and above 0.5 length-binned AUROC.

# Results: Fit In Distribution, Fail Off Surface {#sec:results}

| Detector | Test AUROC | MCPTox AUROC | binned | MSB | MCPTox FPR |
|:-----------------|:---------------|:---------------|------:|------:|:---------------|
| Length only | 0.491 [0.460, 0.526] | 0.774 [0.738, 0.814] | 0.531 | 0.866 | 0.000 [0.000, 0.011] |
| B0 perplexity | 0.515 [0.482, 0.546] | 0.430 [0.384, 0.472] | 0.739 | 0.134 | 0.105 [0.077, 0.141] |
| B1 TF-IDF | 0.902 [0.886, 0.917] | 0.675 [0.637, 0.711] | 0.685 | 0.537 | 0.050 [0.032, 0.077] |
| B1-Optuna | 0.922 [0.907, 0.935] | 0.767 [0.736, 0.800] | 0.725 | 0.559 | 0.069 [0.047, 0.100] |
| B2 row 2 | 0.987 ± 0.004 | 0.346 ± 0.128 | 0.530 | 0.281 | 0.002 ± 0.003 |

Table: Held-out results. Test: 827 benign / 607 adversarial; MCPTox: 362 / 485; MSB: 21 / 70. *binned*: MCPTox length-binned AUROC; *MSB*: MSB AUROC (indicative; intervals about ±0.12). Brackets: 95% bootstrap (AUROC) or Wilson (FPR) intervals; B2: mean (± SD) over three seeds. FPR at the deployed threshold (target 1%). {#tbl:main}

![AUROC with 95% bootstrap intervals for every reported run (B2: each of three seeds per configuration), by eval set. Dashed line: chance; dotted line: length only.](../phase4/fig_auroc.png){#fig:auroc width=95%}

**In distribution, every trained detector fits** (Table \ref{tbl:main}, Figure \ref{fig:auroc}). B2 is near perfect,
B1-Optuna improves on B1 (0.922 vs 0.902, above B1's interval), and both rank by more than length
(length-binned AUROC 0.989 ± 0.003 and 0.918). Length and perplexity are near chance.

**On MCP tool descriptions the order reverses.** B1-Optuna is the strongest trained detector but
only ties raw length, with overlapping intervals; its length-binned AUROC of 0.725 shows that it
does rank within texts of similar length, so it is a weak detector rather than a length proxy.
B2 and B0 fall *below chance*. The deployed threshold does not carry over either: 1% FPR on
validation becomes 6.9% for B1-Optuna on MCPTox, and B2's near-zero FPR comes with a recall of
0.008. **B2 fails its pre-registered criterion on every seed** (Table \ref{tbl:verdict}): no
seed's lower bound on TPR at 5% FPR clears B1's 0.225, and two of three seeds also fail the
length-binned condition. MSB (21 benign rows) is indicative only, but shows the same pattern.

| B2 row 2 seed | TPR at matched 5% FPR [95% CI] | Length-binned AUROC [95% CI] | Passes |
|:----|:--------------------|:--------------------|:----|
| 7267 | 0.208 [0.153, 0.249] | 0.708 [0.656, 0.770] | no |
| 7268 | 0.039 [0.023, 0.056] | 0.390 [0.342, 0.441] | no |
| 7269 | 0.041 [0.025, 0.074] | 0.490 [0.440, 0.547] | no |

Table: Pre-registered verdict for B2 on MCPTox. A seed passes when both lower bounds clear their bars: B1's TPR at 5% FPR (0.225) and 0.5. Row 2 was selected on validation before any holdout was scored. {#tbl:verdict}

**Ablations** (three seeds each) point at data rather than architecture. The pairwise loss costs
a little in distribution (0.995 → 0.987) and does not move MCPTox AUROC (0.357 → 0.346).
Removing PoisonedRAG from B2's training, which we expected to help by removing its reversed
length cue, makes MCPTox *worse* (0.346 → 0.199): more varied training text helps more than a
cleaner length signal. Across all nine B2 runs, test AUROC is 0.92–0.997 and MCPTox AUROC
0.147–0.491. The ordering on the unseen surface is the reverse of capacity.

| Detector (B4 rows) | Test AUROC | MCPTox AUROC | binned | MSB | MCPTox TPR at 5% FPR |
|:-----------------|:---------------|:---------------|:---------------|:---------------|:---------------|
| Length only | 0.491 | 0.760 | 0.472 | 0.907 | 0.002 |
| B1 TF-IDF | 0.902 | 0.700 | 0.710 | 0.481 | 0.181 |
| B4 `text` | 0.993 ± 0.002 | 0.339 ± 0.049 | 0.672 ± 0.070 | 0.160 ± 0.022 | 0.019 ± 0.014 |
| B4 `pair` | 0.991 ± 0.002 | 0.506 ± 0.012 | 0.455 ± 0.021 | 0.335 ± 0.151 | 0.044 ± 0.050 |
| B4 `relevance` | 0.990 ± 0.002 | 0.454 ± 0.099 | 0.486 ± 0.105 | 0.257 ± 0.084 | 0.012 ± 0.011 |
| B4 `relevance`, BIPIA only | 0.927 ± 0.008 | 0.397 ± 0.061 | 0.520 ± 0.119 | 0.175 ± 0.065 | 0.032 ± 0.014 |

Table: B4 on the rows that have task context. Test: 827 benign / 607 adversarial; MCPTox: 269 / 403 (of 362 / 485); MSB: 21 / 55 (of 21 / 70). Length and B1 were refitted once on these rows; B4: mean ± SD over three seeds. *binned*: MCPTox length-binned AUROC. `relevance` was selected on validation. {#tbl:b4}

**Reading the task does not make DeBERTa transfer** (Table \ref{tbl:b4}). Every B4 configuration
fits in distribution and stays below chance on both holdouts, where length alone scores 0.760 and
0.907 on the same rows. Adding the task (`text` → `pair`) raises MCPTox AUROC from 0.339 to 0.506
on every seed, but its length-binned AUROC falls from 0.672 to 0.455 and TPR at matched 5% FPR
stays below 0.05: the model becomes less wrong, not a detector. The relevance features, the
change B4 was built to test, lower MCPTox AUROC to 0.454 (two of three seeds) and MSB AUROC on
every seed. Training on BIPIA only again makes MCPTox worse (0.454 → 0.397), as it did for B2.
Validation cannot choose among these: three configurations reach validation TPR of at least 0.998
at 1% FPR. A second, independent run of all twelve fine-tunes reached the same conclusions,
although single-seed holdout AUROC moved by up to 0.21 between runs because GPU training is not
bit-for-bit reproducible.

# Why the Detectors Fail {#sec:why}

Sections \ref{sec:taxonomy}–\ref{sec:saliency} read MCPTox to *explain* the frozen detectors.
Nothing learned there feeds back into a reported detector.

## A failure-mode taxonomy {#sec:taxonomy}

We read a sample of MCPTox errors, wrote a codebook of six regular-expression cues, and assigned
every error at the deployed threshold to the first matching category. The codebook was revised
once, then checked on 40 fresh errors drawn with a different seed: its category matched a reading
of the full text for 36 of 40 rows (90%). The check was made by the AI assistant that wrote the
codebook, not by a second human coder, so we report agreement rather than Cohen's $\kappa$
[@cohen1960coefficient].

| Missed attacks, by phrasing | B1-Optuna | B2 | MCPTox cue | Test cue |
|:--------------------------------|------:|------:|------:|------:|
| procedural prerequisite | 185 (59%) | 246 (51%) | 0.507 | 0.010 |
| argument or parameter override | 76 (24%) | 162 (34%) | 0.373 | 0.002 |
| authority or policy framing | 20 (6%) | 31 (6%) | 0.379 | 0.040 |
| plain imperative or other | 35 (11%) | 42 (9%) | 0.866 | 0.206 |

Table: Left: missed MCPTox attacks at the deployed threshold by category (B1-Optuna misses 316 of 485; B2, seed 7267, misses 481). A *procedural prerequisite* is a fake rule such as "before using X, you must first call Y". Right: share of adversarial rows containing the category's cue (for the last row, a usage imperative such as *must, always, only*). {#tbl:taxonomy}

**MCP attacks are written as tool procedure** (Table \ref{tbl:taxonomy}). Half of them are a
fake rule inside a tool's description telling the agent to call another tool first; a third
change an argument. These phrasings occur in 1.0% and 0.2% of training-pool test attacks against
50.7% and 37.3% of MCPTox attacks. The surface is learnable, since the prerequisite cue occurs in
only 0.6% of MCPTox's *benign* descriptions, but the detectors were never shown it. The false
alarms mirror this: half of B1-Optuna's 25 are benign descriptions that give the model usage
instructions ("IMPORTANT: always use absolute paths").

## Feature attribution with SHAP {#sec:shap}

For a linear model on TF-IDF features, SHAP values [@lundberg2017unified] are exact: feature $i$
contributes $w_i(x_i - \bar{x}_i)$ to the log-odds. We compute them with `LinearExplainer`
against 200 training rows (a test checks additivity), which makes model-agnostic approximations
such as LIME [@ribeiro2016why] unnecessary for B1.

![B1-Optuna's 15 features with the largest mean |SHAP| on the test split, signed toward "attack" (red) or "benign" (blue), and the same features on MCPTox.](../phase5/shap/b1_optuna_global.png){#fig:shap width=88%}

| Word n-gram | Coef. | Test benign | Test adv. | MCPTox benign | MCPTox adv. |
|:-------------|------:|------:|------:|------:|------:|
| your | 13.4 | 0.127 | 0.445 | 0.017 | 0.027 |
| in your | 10.7 | 0.013 | 0.166 | 0.008 | 0.002 |
| your response | 9.6 | 0.000 | 0.150 | 0.000 | 0.000 |
| reply | 7.6 | 0.034 | 0.145 | 0.003 | 0.002 |

Table: Share of rows containing B1-Optuna's top attack word n-grams. {#tbl:presence}

**B1-Optuna learned a voice.** Its strongest attack features are second-person phrasing about
the model's *output*: *your*, *in your*, *response*, *your response*, *reply*
(Figure \ref{fig:shap}), the voice of BIPIA's injected instructions ("in your response,
include…"). B1 learned the same list. These words are almost absent from MCPTox attacks: *your*
appears in 44.5% of test attacks but 2.7% of MCPTox attacks (Table \ref{tbl:presence}). With no
learned feature present, a missed attack's score is a sum of small contributions: in the three
lowest-scoring misses no feature moves the score by more than 0.33 log-odds, and *must* and *you*
push toward benign (Figure \ref{fig:shap-local}).
When the features *do* appear in benign documentation they are decisive: "Create a new GitHub
repository **in your** account" gains 2.4 log-odds from *in your* alone. This explains the false
alarms, and why longer descriptions, with more chances to contain such phrasing, are flagged more
(§\ref{sec:deploy}).

![Local SHAP explanations for B1-Optuna's three highest-scoring false alarms (top) and three lowest-scoring missed attacks (bottom) on MCPTox. Red pushes toward "attack".](../phase5/shap/b1_optuna_local.png){#fig:shap-local width=92%}

## Saliency for DeBERTa {#sec:saliency}

Raw attention weights are not a faithful explanation [@jain2019attention], so we use integrated
gradients [@sundararajan2017axiomatic] via Captum [@kokhlikyan2020captum]: for each text, the
logit of its highest-scoring window is attributed to the window's tokens over 300 steps from a
baseline of [PAD] tokens (median completeness error ≤ 0.0013 log-odds; 50 steps left errors near
20% of the logit). We attribute 30 random rows from each of four groups and measure the share of
attack-ward attribution on *instruction words* (modals, imperatives such as *must, always, call,
first, before*, and *you, your*). Phase 4 hypothesized that B2 inverts MCPTox because benign tool
descriptions "read like instructions".

| Group (30 rows each) | Median logit | Attribution share [95% CI] | Token share |
|:----------------|------:|:------------------------|------:|
| Test benign | −8.96 | 0.000 [0.000, 0.001] | 0.006 |
| Test attack | +8.80 | 0.060 [0.035, 0.090] | 0.014 |
| MCPTox benign | −8.61 | 0.039 [0.001, 0.111] | 0.011 |
| MCPTox attack | −8.61 | 0.062 [0.027, 0.104] | 0.054 |

Table: Integrated gradients for B2 (seed 7267). *Attribution share*: share of attack-ward attribution on instruction words; *token share*: share of tokens that are instruction words. {#tbl:saliency}

**The hypothesis is not supported** (Table \ref{tbl:saliency}). Instruction words draw a similar
small share of attribution in MCP attacks and benign descriptions, with overlapping intervals,
although they are almost four times as common in MCP attacks as in test attacks. On test attacks
the top-attributed tokens are *your*, *response*, *reply* and *in*: two very different models
learned the same cue from BIPIA. On MCPTox, attacks and benign descriptions have the same median
logit, attribution falls on formatting tokens, and even "you MUST first use" pushes toward
benign (Figure \ref{fig:saliency}). Across all three seeds B2 scores 98–100% of MCPTox rows below 0.01. Its below-chance
AUROC is therefore not a confident inversion but an ordering of near-zero scores set by
formatting: **B2 does not respond to the surface at all.**

# Deployment: Robustness, Fairness, Cost and Harms {#sec:deploy}

**Robustness to rewrites.** Each test-split attack is rewritten eleven ways (character
obfuscation, format wrappers, paraphrase, reordering), and recall is measured at each detector's
5% validation-FPR threshold (Table \ref{tbl:noise}). Meaning-preserving rewrites (synonyms,
polite rewording, sentence shuffling) cost every detector at most 0.05. Character obfuscation is
the cheapest evasion, and **the most accurate detector is the most brittle**: B2's case-sensitive
subword tokenizer turns a still-readable rewrite into unfamiliar pieces, so random capitalization
alone cuts its recall to 0.124; the TF-IDF models lowercase their input.

| Mutation | B1 | B1-Optuna | B2 (seed 7267) |
|:----------------|---------:|---------:|---------:|
| (original) | 0.647 | 0.725 | 0.984 |
| leetspeak | 0.280 | 0.213 | 0.110 |
| case noise | 0.647 | 0.725 | 0.124 |
| zero-width characters | 0.328 | 0.483 | 0.237 |
| homoglyphs | 0.516 | 0.614 | 0.766 |
| JSON wrapper | 0.606 | 0.669 | 0.814 |
| paraphrase or reorder (worst of 3) | 0.646 | 0.679 | 0.984 |

Table: Recall on rewritten test-split attacks at each detector's 5% validation-FPR threshold. {#tbl:noise}

![Integrated-gradients heatmaps for three benign and three attack MCPTox tool descriptions (B2, seed 7267; first 60 tokens). Red pushes toward "attack", blue toward "benign".](../phase5/saliency/heatmaps.png){#fig:saliency width=100%}

**Error rates by group.** The corpus contains no people, so demographic fairness does not apply;
the groups a deployed detector treats differently are surface, source, length and script, and the
harm is benign content blocked, so FPR is the key metric (Table \ref{tbl:fairness}). On MCPTox, B1-Optuna flags **0.0%
[0.0, 4.1]** of the shortest quarter of benign descriptions and **14.3% [8.5, 22.9]** of the
longest, fourteen times its target; long, careful tool documentation bears the false alarms. In
distribution the gradient reverses (3.4% for the shortest quarter, 1.0% for the longest), so no
single global threshold is fair to both surfaces. BIPIA tool outputs are flagged twice as often as
RAG passages (3.4% vs 1.6%). Only 25 MCPTox rows contain CJK text, too few to measure.

| Set | Group | Benign | Adv. | B1-Optuna FPR [Wilson] | B1-Optuna TPR | B2 FPR |
|:------|:------------------|----:|----:|:--------------|-----:|-----:|
| test | BIPIA | 377 | 377 | 0.034 [0.020, 0.058] | 0.501 | 0.008 |
| test | PoisonedRAG | 450 | 230 | 0.016 [0.008, 0.032] | 0.674 | 0.024 |
| test | length Q1 | 207 | 240 | 0.034 [0.016, 0.068] | 0.679 | 0.050 |
| test | length Q4 | 207 | 212 | 0.010 [0.003, 0.035] | 0.425 | 0.002 |
| MCPTox | length Q1 | 91 | 0 | 0.000 [0.000, 0.041] | – | 0.000 |
| MCPTox | length Q2 | 88 | 0 | 0.034 [0.012, 0.095] | – | 0.000 |
| MCPTox | length Q3 | 92 | 230 | 0.098 [0.052, 0.176] | 0.252 | 0.000 |
| MCPTox | length Q4 | 91 | 255 | 0.143 [0.085, 0.229] | 0.435 | 0.007 |
| MCPTox | CJK script | 23 | 2 | 0.000 [0.000, 0.143] | 0.000 | 0.000 |

Table: Error rates by group at the deployed threshold (target FPR 1%). Length quartiles (Q1 shortest) use edges from each eval set's benign rows. B2: mean over three seeds; its recall is at least 0.94 on the test groups shown and at most 0.011 on MCPTox. {#tbl:fairness}

**Latency and cost** (Table \ref{tbl:cost}; Apple M4 Max laptop, 1,000 test rows, median of five passes). B1-Optuna
scores a text in **0.58 ms** on one CPU core with 377,215 weights; B2 takes 56.2 ms on the same
CPU and 44.0 ms on the laptop GPU, with 184.4 M parameters; B0 takes 57.6 ms. B1-Optuna's whole
search took 55 minutes on a CPU and its refit 7 seconds; B2's nine runs and sweeps took an
estimated 6–9 T4 GPU-hours (run times were not recorded). The detector that transfers best is also
about 100 times cheaper. B2's near-zero MCPTox FPR is not fairness: it flags almost nothing there,
attack or benign.

| Detector | Device | ms/row | Rows/s | Params | Fit or training time |
|:----------------|:----|------:|------:|------:|:------------------------|
| Length only | CPU | <0.001 | >10^6^ | 2 | 0.02 s |
| B1 TF-IDF | CPU | 0.42 | 2,382 | 585,055 | 18 s |
| B1-Optuna | CPU | 0.58 | 1,727 | 377,215 | 7 s refit; 55 min search |
| B0 perplexity | CPU / MPS | 57.6 / 26.6 | 17 / 38 | 81.9 M | none (frozen) |
| B2 DeBERTa row 2 | CPU / MPS | 56.2 / 44.0 | 18 / 23 | 184.4 M | 6–9 T4 GPU-h, all runs (est.) |

Table: Inference cost on the same 1,000 test rows (mean 1,056 characters) after one warm-up; MPS is the laptop's GPU. {#tbl:cost}

| Harm | Evidence | Mitigation |
|:-----------------------|:-------------------------|:----------------------------------------|
| False sense of security | 0.99 AUROC in distribution, below chance off surface | Score only validated surfaces; publish the out-of-scope list (model card) |
| Blocking legitimate content | 14.3% FPR on long benign tool docs | Per-surface thresholds; review instead of block; train on benign tool docs |
| Dual use of this work | Leetspeak: 0.725 → 0.213 recall | Release aggregates, not an evasion recipe; normalize input |
| Dataset as attack corpus | Working injections in one place | Rows stay in a private remote; commit scores and ids, not text |
| Over-reliance | Best detector misses 65% of MCP attacks | Privilege separation, tool allow-lists, confirmation of side effects |
| Opaque blocking | Authors cannot see why they are flagged | Log score and threshold; appeal path; per-surface drift monitoring |

Table: Harms of deploying an injection detector, the result that grounds each, and a mitigation. {#tbl:harms}

**One measured mitigation.** Applying Unicode NFKC normalization and stripping zero-width
characters before B1-Optuna (model unchanged) fully repairs zero-width attacks (recall 0.483 →
0.725) but leaves homoglyphs (0.614) and leetspeak (0.213) unchanged, since NFKC maps neither
Cyrillic look-alikes nor digits to Latin letters. Normalization is necessary, not sufficient.

# Limitations and Conclusion {#sec:conclusion}

**Limitations.** We explain frozen detectors. The one retrained variant, B4, adds task context,
not new data, so we could not test the fix we recommend: adding tool descriptions to training.
B4's MCP task context is a server-level catalogue rather than an observed request, and it
excludes holdout rows without one. The holdouts are small (MSB has 21 benign rows) and
come from single benchmarks. B2's search covered two values of one hyperparameter, its seeds
disagree widely on MCPTox (0.247–0.491), and its robustness, saliency and latency results use one
seed. The error codebook was validated by its author, an AI assistant, not by independent human
coders. Integrated gradients depend on the baseline, and we use one. We measure detection, not
the end-to-end reduction in attack success of a defended agent [@debenedetti2024agentdojo].

**Conclusion.** Detectors trained on RAG passages and tool outputs do not transfer to MCP tool
descriptions because they learned a *voice*, not a *behavior*. Both the TF-IDF model and DeBERTa
key on second-person phrasing about the model's response, the way BIPIA's attacks are written;
MCP attacks are written as tool procedure, a phrasing the training data almost never contains,
and benign tool documentation that happens to say "in your account" is flagged. More capacity
made this worse, not better: DeBERTa fits its training surfaces almost perfectly, does not
respond to the new one, and is the easiest to evade. Letting it read the user's task does not fix
this: the paired model moves toward chance, not above it. We would deploy B1-Optuna, as documented in
its model card, only on validated surfaces, with input normalization, per-surface thresholds,
review rather than automatic blocking, and architectural defenses around it. The next step is not
a larger model but different data: benign and adversarial tool descriptions from a source other
than MCPTox, with MCPTox kept as the held-out test.

**Use of AI assistance.** Parts of the experiment and analysis code, the error codebook and its
check, the model card and the drafts of this report were prepared with an AI assistant (Claude),
working from committed result files and merged through reviewed pull requests with tests. Every
number is produced by a committed script; analysis choices and the final text were reviewed by the
team.

# Team Contributions {#sec:contrib}

Each team member designed and wrote the code for at least one method used in this paper
(Table \ref{tbl:contrib}). Each contribution was made through a pull request (PR) to the public
repository; the PR numbers below refer to it.

| Member | Method developed and its code | Phase | Used in |
|:------------------|:-----------------------------------------------------------|:------|:--------|
| Mohammed Adil Ahmed | B0 perplexity and B1 TF-IDF detectors and MLflow experiment tracking (`src/models/perplexity.py`, `src/models/tfidf.py`, `mlflow-server/`; PRs #5, #22); B4 task-paired DeBERTa and its Colab experiment grid (`src/models/b4*.py`, `src/data/b4_data.py`, `notebooks/b4_colab.ipynb`; PR #53) | 3–4, 6 | §§4–5 |
| Carter Corbin | Length-shortcut gate, probe figures and the red-team mutation generator (`notebooks/phase3_probe_figures.py` and the red-team stage; PR #40) | 3 | §§3, 7 |
| Max Edmonds | Visual profile of the unified corpus (`notebooks/phase3_eda.py`; PRs #20, #26); DeBERTa score guards and gradient clipping (`src/models/deberta.py`, `src/models/precomputed.py`; PR #39); the B2 training runs and verdict (PR #43) | 3–4 | §§3, 5 |
| Oluwamayowa Adewummi | Family-grouped cross-validation of B1 with 95% intervals and length-binned AUROC (`src/eval/b1_tfidf_lr.py`; PR #46) | 4 | §4 |
| Adedapo Odeyemi | Data pipeline (DVC stages, unified dataset build) and literature review; B2 DeBERTa model with chunked scoring and pairwise loss; B1-Optuna; Phase 5 SHAP, integrated gradients, error taxonomy and red-team scoring; reports (PRs #2, #7, #13, #30, #35, #45, #49) | 1–6 | §§2–7 |

Table: The method each team member developed and coded, the project phases it belongs to, and the sections of this paper that use it. {#tbl:contrib}

# References {-}

::: {#refs}
:::

\appendix

# Reproducing the Results {#app:reproduce}

Every number in this paper is produced by a committed script with a `make` target, except B4's;
the PDF is built by `make report-final`. B4's twelve fine-tunes run in `notebooks/b4_colab.ipynb`
on a Colab GPU, and Table \ref{tbl:b4} is read from its committed result files in `report/b4/`
(setup in `docs/b4-relevance.md`). The rebuilt dataset, the B2 score files and the B2 checkpoint are
versioned with DVC in a team-only remote, so `make data`, `make b2-report` and `make b2-checkpoint`
need its credentials. Outside readers can rebuild the dataset with `dvc repro` from the pinned
public upstream sources (NewsQA must first be built under its own terms), retrain B2 with the
Colab notebook (`notebooks/deberta_colab.ipynb`), and re-check every table from the committed
per-row scores of all detectors (`report/phase5/scores.parquet`).

```bash
make install && make data      # dependencies (uv); dataset and B2 score files (DVC)
make evaluate                  # length, B0, B1 through one harness -> report/baselines/
make b1-optuna                 # 100-trial family-fold search -> report/phase4/b1_optuna/
make b2-report                 # B2 tables and pre-registered verdict from the score files
make b2-checkpoint             # B2 row2-s7267 checkpoint (DVC)
make phase5                    # fairness, error taxonomy, SHAP, saliency, latency
make red-team DETECTOR=tfidf-optuna && make red-team-b2
make report-final check-final  # this PDF and its rubric check
```

Seeds are fixed (7267 by default; B2 seeds 7267–7269). Score files for every eval row are
committed (`report/phase5/scores.parquet`, no text) and reproduce every AUROC and deployed FPR in
Table \ref{tbl:main} exactly.

# Model Card {#app:card}

The card below is `MODEL_CARD.md` from the repository [@mitchell2019model].
