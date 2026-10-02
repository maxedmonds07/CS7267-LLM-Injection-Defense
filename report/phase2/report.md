---
title: "Detecting Prompt Injection Across RAG and MCP Surfaces"
subtitle: "Phase 2 Report: Literature Review and Data Acquisition"
author:
  - Carter Corbin
  - Max Edmonds
  - Mohammed Adil Admed
  - Oluwamayowa Adewummi
  - Adedapo Odeyemi
date: "CS 7267 Machine Learning · October 2026"
abstract: |
  LLM applications increasingly read text that their users never wrote: passages retrieved
  for retrieval-augmented generation (RAG), documents fetched by tools, and the descriptions
  and responses of Model Context Protocol (MCP) tools. Any of it can carry an indirect
  prompt injection. This report covers Phase 2 of our project, which builds a detector for
  such untrusted text. We review 20 papers on attacks, defenses, benchmarks and evaluation
  methodology, code them on a shared frame, and organize them into a taxonomy by attack
  surface, payload and defense class. Within the reviewed set, no defense targets MCP tool
  descriptions, no detector targets tool outputs, and no detector is evaluated on an attack
  surface it never saw in training. We then describe the corpus built to study those gaps:
  11,005 labeled text segments from four public benchmarks, versioned with DVC, split by
  group with two sources held out entirely for leave-one-surface-out and cross-benchmark
  evaluation, profiled for nulls and class balance, and documented in a data card. The
  profile also exposes a length shortcut that every detector in Phase 3 must be measured
  against.
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

# Introduction

An LLM application that retrieves documents or calls tools places untrusted text in the
model's context. An attacker who controls that text, for example a web page, an email, a
passage in a retrieval corpus or the description of a third-party tool, can plant
instructions or false content without ever interacting with the user. Greshake et al.
named this threat *indirect prompt injection* and showed it against deployed systems
[@greshake2023indirect]. The Model Context Protocol (MCP) widens the attack surface
further: tool descriptions enter the agent's context when a server is registered, before
any tool runs [@wang2026mcptox; @zhang2026msb].

**Research question.** Can a detector that classifies untrusted text segments as
adversarial or benign generalize (i) within the surfaces it is trained on, (ii) to an
attack surface it has never seen, and (iii) to a benchmark it has never seen?

Phase 2 lays the groundwork: a structured review of the state of the art, and a dataset
built and documented to answer the question. Table \ref{tbl:rubric} maps each Phase 2
requirement to the section and repository artifact that addresses it.

| Requirement | Where it is addressed | Repository artifact |
|:------------------------------|:---------------------|:--------------------------------------|
| Review 20–30 papers | §\ref{corpus-and-coding-protocol}, §\ref{coverage-of-the-reviewed-set} | Coding matrix (`literature-review/*.xlsx`) |
| Organize into a taxonomy table | §\ref{taxonomy}, Tables \ref{tbl:threats}–\ref{tbl:defenses} | `literature-review/tables.md` (generated) |
| Identify SOTA benchmarks | §\ref{state-of-the-art-benchmarks}, Table \ref{tbl:benchmarks} | `literature-review/tables.md` |
| Note gaps the project addresses | §\ref{gaps-this-project-addresses}, Table \ref{tbl:coverage} | Coverage sheet of the matrix |
| Maintain a BibTeX bibliography | References | `literature-review/references.bib` |
| Download and version raw dataset | §\ref{download-and-versioning}, Figure \ref{fig:pipeline} | `config.yaml`, `dvc.yaml`, `dvc.lock` |
| Check licensing and IRB | §\ref{licensing-privacy-and-irb} | `DATA_CARD.md` (Distribution, Collection) |
| Profile: shape, nulls, class balance | §\ref{profile} | `data/processed/profile.md`, `profile.json` |
| Write a data card | §\ref{data-card}, Appendix \ref{data-card-summary} | `DATA_CARD.md` |
| Set up data versioning | §\ref{download-and-versioning} | DVC remote on Cloudflare R2 |

Table: Phase 2 requirements and where this report and the repository address them. {#tbl:rubric}

# Literature Review

## Corpus and coding protocol

The review set holds 20 papers published between 2023 and 2026, selected for their
relevance to indirect prompt injection and retrieval poisoning against LLM applications
that use retrieval, tools or MCP. By primary contribution, 8 are defenses, 4 are
benchmarks or evaluation environments, 3 are attacks, 1 is an evaluation-methodology
paper, and 4 combine contributions (for example a benchmark and a defense). Twelve are
rated highly relevant to our research question.

<!-- TODO(team): add the search strategy here: databases and venues searched, query
strings, date range, and the inclusion and exclusion criteria. -->

Each paper is coded as one row of a shared coding matrix. The frame records, among other
dimensions:

- **Attack surface**: where adversarial content enters the system (RAG corpus, tool
  output at runtime, tool description or metadata, direct user input).
- **Payload**: how the content works on the model. *Instruction injection* issues
  commands; *content poisoning* is plausible false content that misleads by being
  believed, with no commands at all.
- **Attack objective**: what the attacker wants (task hijack, data exfiltration,
  misinformation, denial of service, tool-selection manipulation).
- **Defense class** and **enforcement point**: the defense mechanism and where in the
  stack it sits.
- **Primary metric**, **headline result**, **benchmarks used**, and whether the paper
  **addresses MCP explicitly**.

The payload dimension was added during coding: attack objective alone could not separate
PoisonedRAG-style false passages from instructions that pursue the same misinformation
goal. It corresponds one to one to the `attack_kind` column of our dataset (§\ref{construction-and-cleaning}), so
our results can be compared with prior work split the same way.

**Verification.** Coding was first drafted with an AI assistant from each paper's
abstract, introduction and threat-model section, then re-read and confirmed row by row by
a team member. Venues and years were checked against publisher records (via DOI), DBLP
and Semantic Scholar. This check found published versions of ten papers whose downloaded
copies were arXiv preprints or carried no venue. The bibliography cites the published
version, with the arXiv identifier kept as an `eprint` field.

## Coverage of the reviewed set

The papers are spread unevenly across surfaces. Nine study several surfaces at once, five
the RAG corpus, four tool outputs, and only two MCP tool descriptions [@wang2026mcptox;
@shi2026toolhijacker]. None studies direct user input, which is outside our scope.
Thirteen papers concern instruction injection, four content poisoning (all on RAG), and
three both. Attack success rate (ASR) is the headline metric of 12 papers. Only three
report a detection metric: AUROC [@fomin2026benchmarks] or true and false positive rates
[@jacob2025promptshield; @tan2025revprag].

## Taxonomy

Table \ref{tbl:threats} organizes the papers by the threat they study or defend against;
Table \ref{tbl:defenses} organizes the defenses by mechanism.

| Attack surface | Instruction injection | Content poisoning | Both |
|:-----------------------|:----------------------------------|:----------------------------|:--------------------|
| RAG corpus / retriever | — | RevPRAG [@tan2025revprag], RAG traceback [@zhang2025traceback], adversarial passages [@zhong2023poisoning], PoisonedRAG [@zou2025poisonedrag] | RobustRAG [@xiang2026robustrag] |
| Tool output (runtime) | AgentDojo [@debenedetti2024agentdojo], CaMeL [@debenedetti2026camel], Spotlighting [@hines2024spotlighting], InjecAgent [@zhan2024injecagent] | — | — |
| Tool description (MCP) | MCPTox [@wang2026mcptox] | — | ToolHijacker [@shi2026toolhijacker] |
| Several surfaces | SecAlign [@chen2025secalign], StruQ [@chen2025struq], When Benchmarks Lie [@fomin2026benchmarks], Greshake et al. [@greshake2023indirect], PromptShield [@jacob2025promptshield], Liu et al. [@liu2024formalizing], Jatmo [@piet2024jatmo], BIPIA [@yi2025bipia] | — | MSB [@zhang2026msb] |

Table: Threat taxonomy: attack surface by payload. Defense papers are placed by the threat they defend against. {#tbl:threats}

**RAG corpus.** Attacks inject a handful of passages into a knowledge base:
PoisonedRAG reaches 90% ASR with five malicious texts per target question in a database
of millions [@zou2025poisonedrag], and adversarial passages optimized against a dense
retriever transfer to unseen corpora [@zhong2023poisoning]. The RAG surface has the most
varied defenses: an activation probe [@tan2025revprag], a certified isolate-then-aggregate
scheme [@xiang2026robustrag] and post-hoc forensic traceback [@zhang2025traceback]. Every
RAG paper in the set is about content poisoning, not injected instructions.

**Tool output.** Benchmarks place instructions in emails, web pages and documents that
tools return: AgentDojo [@debenedetti2024agentdojo] and InjecAgent
[@zhan2024injecagent]. Defenses mark untrusted input in the prompt
[@hines2024spotlighting] or enforce control- and data-flow policies outside the model
[@debenedetti2026camel].

**Tool description (MCP).** MCPTox shows that tool-poisoning instructions placed in tool
metadata reach 72.8% ASR on o1-mini, with refusal rates under 3% across 20 agents
[@wang2026mcptox]. ToolHijacker crafts tool documents that win tool selection without any
access to the target model [@shi2026toolhijacker]. MSB benchmarks attacks at every stage
of the MCP pipeline: tool signatures, parameters, responses and retrieval
[@zhang2026msb].

**Several surfaces.** Model-level defenses treat any data channel as untrusted:
preference optimization [@chen2025secalign], structured instruction tuning
[@chen2025struq], and task-specific fine-tuning that removes instruction following
altogether [@piet2024jatmo]. Liu et al. formalize the attack space and show that none of
ten defenses is sufficient [@liu2024formalizing].

| Defense class | Papers | Enforcement point | Surfaces defended |
|:--------------------------|:--------------------------------------|:-----------------------|:-------------------------|
| Detection (classifier or probe) | When Benchmarks Lie [@fomin2026benchmarks], PromptShield [@jacob2025promptshield], RevPRAG [@tan2025revprag] | Prompt layer; runtime | Several; RAG |
| Input transformation / prompting | Spotlighting [@hines2024spotlighting], BIPIA [@yi2025bipia] | Prompt layer | Tool output; several |
| Training-time (fine-tuning, alignment) | SecAlign [@chen2025secalign], StruQ [@chen2025struq], Jatmo [@piet2024jatmo] | Model weights | Several |
| Runtime policy / capabilities | CaMeL [@debenedetti2026camel] | Orchestrator | Tool output |
| Certified / provable | RobustRAG [@xiang2026robustrag] | Orchestrator | RAG |
| Forensics / traceback | RAG traceback [@zhang2025traceback] | Retriever / index | RAG |
| Architectural isolation | — | — | — |

Table: Defense taxonomy by mechanism. {#tbl:defenses}

## State-of-the-art benchmarks

Table \ref{tbl:benchmarks} lists the benchmarks that the reviewed work evaluates on.
PoisonedRAG is an attack paper, but its targeted questions serve as the de facto
benchmark for RAG poisoning. Four of the eight benchmarks feed our dataset, and a fifth
is our end-to-end harness.

| Benchmark | Surface | Payload | Scale | Metric | In our project |
|:--------------------|:---------------|:------------|:----------------------------------------|:---------|:----------------|
| AgentDojo [@debenedetti2024agentdojo] | Tool output | Instruction | 97 user tasks, 629 security test cases | ASR, utility | End-to-end harness (v0.1.35) |
| InjecAgent [@zhan2024injecagent] | Tool output | Instruction | 1,054 test cases, 17 user tools, 62 attacker tools | ASR | — |
| BIPIA [@yi2025bipia] | Tool output (email, web, table, code, summaries) | Instruction | Five task types with text and code attacks | ASR | Training pool |
| Open-Prompt-Injection [@liu2024formalizing] | Several | Instruction | 5 attacks, 10 defenses, 10 LLMs, 7 tasks | Attack success score | — |
| PromptShield [@jacob2025promptshield] | Several | Instruction | Conversational and application data | TPR at low FPR | — |
| MCPTox [@wang2026mcptox] | MCP tool description | Instruction | 1,312 cases, 45 servers, 353 tools | ASR | LOSO holdout |
| MSB [@zhang2026msb] | MCP pipeline | Both | 2,000 instances, 405 tools, 12 attack types | ASR, NRP | Transfer holdout |
| PoisonedRAG [@zou2025poisonedrag] | RAG corpus | Poisoning | NQ, HotpotQA, MS MARCO targets | ASR | Training pool |

Table: Benchmarks used by the reviewed work and their role in this project. {#tbl:benchmarks}

Three observations shape our evaluation design. First, nearly every benchmark measures
*attack success against an agent*, not *detection of the injected text*, so detector
results are rarely comparable across papers. Second, the detection papers that do exist
warn that standard evaluation overstates performance. PromptShield argues that a deployed
detector must operate at very low false positive rates, because benign traffic dominates;
its detector reaches 65.3% TPR at 0.1% FPR against 9.4% for the best prior scheme
[@jacob2025promptshield]. Fomin finds that standard cross-validation reports pooled AUC
8.0–16.5 points higher than leave-one-dataset-out evaluation, and that 28–44% of a
classifier's top features are dataset shortcuts [@fomin2026benchmarks]. Third, the MCP
benchmarks report that more capable models are often *more* susceptible
[@wang2026mcptox; @zhang2026msb], so model scale is not a defense.

## Gaps this project addresses

Table \ref{tbl:coverage} crosses attack surface with defense class. An empty cell is a
candidate gap in the reviewed set.

| Surface | Detection | Prompting | Training | Runtime policy | Certified | Forensics | Attack or benchmark only |
|:-----------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| RAG corpus | 1 | 0 | 0 | 0 | 1 | 1 | 2 |
| Tool output | **0** | 1 | 0 | 1 | 0 | 0 | 2 |
| Tool description (MCP) | **0** | **0** | **0** | **0** | **0** | **0** | 2 |
| Several surfaces | 2 | 1 | 3 | 0 | 0 | 0 | 3 |

Table: Papers per attack surface and defense class. Architectural isolation (no papers) and direct user input (no papers) are omitted. {#tbl:coverage}

We draw four gaps from the review. Each is stated relative to the reviewed set.

1. **No defense for MCP tool descriptions.** Two papers show that poisoned tool
   metadata hijacks agents [@wang2026mcptox; @shi2026toolhijacker], and none defends
   against it. ToolHijacker reports that existing prevention and detection defenses fail
   on tool documents [@shi2026toolhijacker]. Neither of the two detectors coded as
   covering several surfaces addresses MCP [@fomin2026benchmarks;
   @jacob2025promptshield].
2. **No detector for tool outputs.** Tool-output defenses rewrite prompts
   [@hines2024spotlighting] or enforce policy [@debenedetti2026camel]; none classifies the
   returned text.
3. **No evaluation on a held-out surface.** The only held-out evaluation in the set
   leaves out whole *datasets* [@fomin2026benchmarks]. No paper trains a detector on some
   attack surfaces and tests it on another.
4. **Shortcut-blind detector evaluation.** Fomin shows that detectors learn dataset
   identity [@fomin2026benchmarks]; no other paper in the set reports a baseline that
   would reveal such shortcuts.

Our project addresses these gaps directly. The detector is trained on RAG passages and
tool outputs and evaluated three ways: in distribution, on MCP tool descriptions held out
entirely (leave-one-surface-out, gaps 1 and 3), and on MSB, a benchmark held out entirely
(transfer). Tool outputs are part of training and of the in-distribution test (gap 2).
Every detector will be reported next to a length-only baseline and at low false positive
rates (gap 4, §\ref{profile}).

# Data Acquisition

## Sources and licensing

The corpus unifies four public benchmarks into one table of *untrusted text segments*.
A segment is any text that reaches the model without the user writing it.

| Source | Surface | Role | Adversarial | Benign | Groups | License |
|:----------------------|:------------------|:----------------|-----------:|---------:|-------:|:----------------------------|
| PoisonedRAG [@zou2025poisonedrag] + BEIR [@thakur2021beir] | RAG corpus | Training pool | 1,500 | 2,955 | 300 | MIT; BEIR corpora per dataset |
| BIPIA [@yi2025bipia] | Tool output | Training pool | 2,806 | 2,806 | 2,806 | MIT; TableQA CC BY-SA 4.0; NewsQA and XSum terms |
| MCPTox [@wang2026mcptox] | Tool description | LOSO holdout | 485 | 362 | 45 | None stated (team-internal copy) |
| MSB [@zhang2026msb] | Tool description, tool output | Transfer holdout | 70 | 21 | 11 | MIT |
| **Total** | | | **4,861** | **6,144** | | |

Table: Sources, roles and licenses. MSB contributes 55 adversarial and 21 benign tool descriptions and 15 adversarial tool outputs. {#tbl:sources}

AgentDojo 0.1.35 (MIT) is pinned as the end-to-end attack-success harness for Phase 3;
it is not part of the table.

## Download and versioning

Every upstream repository is pinned to an exact commit in `config.yaml`, so a rebuild
fetches byte-identical inputs. The pipeline is a five-stage DVC graph
(Figure \ref{fig:pipeline}). Two stages rebuild BIPIA contexts that BIPIA does not
redistribute: summaries from XSum [@narayan2018xsum] and web QA from NewsQA
[@trischler2017newsqa]. Each output is checked against pinned MD5 hashes. NewsQA sits
behind terms of use, so it was built once with the official toolchain and is stored only
in the team's private remote.

```{=latex}
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  stage/.style={draw, rounded corners=2pt, fill=blue!6, font=\small\ttfamily, minimum height=7mm, inner sep=4pt},
  data/.style={draw, dashed, font=\footnotesize, inner sep=3pt, align=center},
  arr/.style={-{Stealth[length=2mm]}, thick}]
\node[stage] (fetch)  at (0, 0)     {fetch};
\node[stage] (sum)    at (3.9, 1.2) {bipia\_summarization};
\node[stage] (web)    at (3.9, -1.2){bipia\_webqa};
\node[data]  (newsqa) at (3.9, -2.5){NewsQA CSV (private)};
\node[stage] (build)  at (8.0, 0)   {build\_dataset};
\node[stage] (prof)   at (11.6, 0)  {profile\_dataset};
\node[data]  (parq)   at (8.7, -1.6){dataset.parquet\\summary.json};
\node[data]  (pmd)    at (11.6, -1.6){profile.json\\profile.md};
\draw[arr] (fetch) |- (sum);
\draw[arr] (fetch) |- (web);
\draw[arr] (newsqa) -- (web);
\draw[arr] (sum) -| ([xshift=-7mm]build.north);
\draw[arr] (web) -| ([xshift=-7mm]build.south);
\draw[arr] (fetch) -- (build);
\draw[arr] (build) -- (prof);
\draw[arr, dashed] ([xshift=7mm]build.south) -- (parq.north);
\draw[arr, dashed] (prof) -- (pmd);
\end{tikzpicture}
\caption{The DVC pipeline. \texttt{fetch} downloads the upstream repositories at the commits pinned in \texttt{config.yaml}. Solid arrows are stage dependencies; dashed arrows point to tracked outputs. All cached data is stored in a private Cloudflare R2 remote.}
\label{fig:pipeline}
\end{figure}
```

`dvc.lock` records the hash of every input and output. The current build (produced
2026-09-29, Git commit `af5ec37`) locks `dataset.parquet` at MD5
`a1c6426890678d576fb6231dc6050491`. Any team member can restore any committed version
with `git checkout <commit>` followed by `make data`, and `dvc metrics diff` shows how
row counts and the profile change in review.

## Construction and cleaning

**Schema and labels.** Each row carries the text, a binary label (1 adversarial,
0 benign), the payload type (`attack_kind`: instruction or poisoning), the attack
objective, the source's own attack category verbatim, the source and subset, the attack
surface, a group identifier, a split and a role. The objective is mapped from about 75
upstream categories in `config.yaml`, and changes to that mapping are reviewed like
edits to the coding matrix.

**Per-source construction.** Adversarial RAG rows are PoisonedRAG's five malicious
texts for each of 100 target questions in NQ, MS MARCO and HotpotQA; benign rows are the
top-10 real passages a Contriever retriever returns for the same question, which makes
them hard negatives. For BIPIA, every clean context is a benign row, and one sampled
attack per context, inserted at a sampled position (start, middle or end), is the
adversarial row. MCPTox contributes its 485 poisoned tool descriptions and the clean
descriptions of the same servers. MSB attack texts are assembled at run time upstream,
so we rebuild the static text the model would see from MSB's tool files and templates.

**Splits and holdouts.** Rows are split 70/15/15 into train, validation and test *by
group* (a question, a context, an MCP server or an attack task), so a group never spans
two splits. BIPIA's own train/test boundary is kept, so its test split mostly holds
attack families never seen in training. A `role` column assigns each source to the
training pool or to a holdout. The data loader raises an error if fitting data would
include a held-out source or the test split.

**Cleaning.** We removed formatting cues that would reveal the label: surrounding
whitespace (an artifact of BIPIA's insertion), passage titles that only benign
PoisonedRAG rows would have had, and source-specific tool-description layouts. All tool
descriptions now share one format. 47 exact duplicate texts with the same label were
dropped. The build stops, writing nothing, if a text appears with both labels, if a
group spans splits, if a benign row carries attack fields, or if a category lacks an
objective mapping.

## Profile

The `profile_dataset` stage recomputes the profile whenever the dataset changes.

**Shape and nulls.** The table has 11,005 rows and 12 columns. The only nulls are the
three attack fields on the 6,144 benign rows, which the schema requires. No other column
contains a null or a blank text, and there are no duplicate IDs or texts.

**Class balance.** The corpus is 44.2% adversarial, and the share is stable across
splits. It varies strongly by source and role, however (Table \ref{tbl:balance}).

| Group | Adversarial | Benign | Adversarial share |
|:----------------------------------|-------:|-------:|------:|
| All rows | 4,861 | 6,144 | 44.2% |
| Train split | 3,469 | 4,393 | 44.1% |
| Validation split | 713 | 875 | 44.9% |
| Test split | 679 | 876 | 43.7% |
| Training pool | 4,306 | 5,761 | 42.8% |
|   PoisonedRAG (RAG corpus) | 1,500 | 2,955 | 33.7% |
|   BIPIA (tool output) | 2,806 | 2,806 | 50.0% |
| LOSO holdout: MCPTox | 485 | 362 | 57.3% |
| Transfer holdout: MSB | 70 | 21 | 76.9% |

Table: Class balance by split, role and source. {#tbl:balance}

Because MSB is 76.9% adversarial, a detector that flags everything scores 77% accuracy on
it. We therefore report AUROC and the true positive rate at a fixed low false positive
rate, not accuracy.

**A length shortcut.** Text length differs sharply between classes, in opposite
directions on different sources (Table \ref{tbl:length}). Poisoned passages are shorter
than retrieved ones; injected tool descriptions are longer than clean ones.

| Source | Benign | Adversarial |
|:----------------------|-------:|-------:|
| PoisonedRAG | 389 | 183 |
| BIPIA | 1,752 | 1,816 |
| MCPTox | 88 | 292 |
| MSB | 156 | 314 |

Table: Median text length in characters, by source and label. {#tbl:length}

We measure how much length alone explains with two numbers (Table \ref{tbl:baseline}):
the AUROC of a logistic regression on log length, fit on the training pool, and length
separability, $\max(\mathrm{AUROC}, 1-\mathrm{AUROC})$ of raw length, which needs no
training and counts a cue in either direction.

| Evaluation set | Fitted AUROC | Separability |
|:-------------------------------|-------:|-------:|
| In-distribution test | 0.491 | 0.509 |
|   BIPIA only | 0.532 | 0.532 |
|   PoisonedRAG only | 0.138 | 0.862 |
| LOSO holdout (MCPTox) | 0.774 | 0.774 |
| Transfer holdout (MSB) | 0.866 | 0.866 |

Table: What text length alone achieves on each evaluation set. The fitted model learns the wrong direction for PoisonedRAG, which separability exposes. {#tbl:baseline}

Length alone reaches 0.77 AUROC on the LOSO holdout and 0.87 on the transfer holdout.
A detector scoring near those values may be classifying length, not injection. This is
the kind of shortcut Fomin warns about [@fomin2026benchmarks], and it is why every
detector in Phase 3 will be reported next to this baseline.

## Licensing, privacy and IRB

**Licensing.** PoisonedRAG, BIPIA and MSB are MIT-licensed; BIPIA's TableQA contexts are
CC BY-SA 4.0. NewsQA and the CNN stories it draws on are available under terms of use
that forbid redistribution, and XSum's BBC articles are licensed for research use. MCPTox
states no
license, so its use is team-internal. The data therefore stays in a private remote; if
it is ever shared, we will share the build pipeline, not the rebuilt rows.

**Privacy.** A spot-check of every benign BIPIA email and NewsQA context (samples plus
a pattern scan for email addresses, phone numbers, social security and card numbers)
found no full account or card numbers, phone numbers or SSNs. It did find that BIPIA's
emails, which come from OpenAI Evals, are not synthetic: they read as one company's real
inbox, with names, vendors, amounts and masked account numbers from 2022. The news
contexts name public figures and, in crime and accident reports, private individuals.
This content is already public through the upstream benchmarks. We keep it unchanged,
and we will not quote email rows in our write-ups.

**IRB.** The project does not interact with people or collect data from them; every
row comes from publicly released benchmarks. Under the U.S. Common Rule (45 CFR 46.102)
this is generally not human subjects research, so no IRB review was sought. If the work
is published beyond the course, we will request a formal "not human subjects research"
determination from the institution's IRB.

## Data card

The corpus is documented in `DATA_CARD.md` at the repository root. It opens with a
summary and per-source table in the style of Data Cards [@pushkarna2022datacards], then
answers the questions of Datasheets for Datasets [@gebru2021datasheets] on motivation,
composition, collection, preprocessing, uses, distribution and maintenance.
Appendix \ref{data-card-summary} reproduces its summary. Besides the facts in this section, the card records
the intended uses and the uses to avoid: training on held-out sources, measuring
direct-injection robustness, estimating false positive rates on MCP tool responses (no
benign ones exist), comparing payload types without stratifying, and building attack
generators.

# Summary and Next Steps

The review maps 20 papers onto a taxonomy of attack surface, payload and defense class.
Within that set, MCP tool descriptions have no defense, tool outputs have no detector,
and no detector has been evaluated on an unseen attack surface. The dataset is built to
measure exactly those cases. It is versioned end to end, split by group with two sources
fully held out, profiled, and documented.

Phase 3 will train detector baselines on the training pool and report, for each:

- AUROC and TPR at a low fixed FPR on the in-distribution test set, the LOSO holdout
  (MCPTox) and the transfer holdout (MSB);
- results stratified by `attack_kind`, because the training pool mixes poisoning and
  instruction injection while both holdouts are instruction-only;
- the length-only baseline alongside, as the score to beat.

Experiments are tracked in the team's shared MLflow server. AgentDojo will measure
whether a detector reduces end-to-end attack success in a defended agent.

**Use of AI assistance.** The coding of all 20 papers was first drafted with an AI
assistant (Claude) and then confirmed by a team member, and venues were checked against
publisher and index records. Parts of the data tooling were written with AI assistance
and merged through reviewed pull requests. All analysis choices and the text of this
report were reviewed by the team.

\appendix

# Data card summary

| Field | Value |
|:----------------|:------------------------------------------------------------------|
| Task | Binary classification of untrusted text segments: 1 adversarial (prompt injection or poisoning), 0 benign |
| Unit | One text segment that reaches an LLM without the user writing it: a retrieved passage, a fetched document, an MCP tool description or a tool response |
| Size | 11,005 rows: 4,861 adversarial, 6,144 benign (47 exact duplicates dropped) |
| Surfaces | RAG corpus, tool output, tool description |
| Sources | PoisonedRAG (+ BEIR), BIPIA, MCPTox, MSB |
| Held out | MCPTox (unseen surface) and MSB (unseen benchmark) are never trained on |
| Profile | No unexpected nulls, no duplicate IDs or texts, 44.2% adversarial |
| Version | `dvc.lock`, dataset MD5 `a1c6426890678d576fb6231dc6050491` (built 2026-09-29) |
| Storage | Private Cloudflare R2 DVC remote; not redistributed |
| Full card | `DATA_CARD.md` in the repository |

Table: Summary of the data card. {#tbl:datacard}
