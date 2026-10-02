# Data card: CS7267 Untrusted-Segment Injection Dataset

This data card describes the corpus built for the project. It opens with a summary and a
per-source table in the style of Pushkarna et al.,
[*Data Cards*](https://doi.org/10.1145/3531146.3533231) (FAccT 2022), then answers the
questions of Gebru et al., [*Datasheets for Datasets*](https://doi.org/10.1145/3458723)
(CACM 2021), section by section.

Numbers describe the build recorded in `data/processed/summary.json`,
`data/processed/profile.md` and `dvc.lock`.

## At a glance

| | |
|--------------------|--------------------------------------------------------------------------------|
| **Task** | Binary classification of untrusted text segments: `1` adversarial (prompt injection or poisoning), `0` benign |
| **Unit** | One text segment that reaches an LLM without the user writing it: a retrieved passage, a fetched document, an MCP tool description or a tool response |
| **Size** | 11,005 rows: 4,861 adversarial, 6,144 benign (47 exact duplicates dropped) |
| **Surfaces** | `rag_corpus`, `tool_output`, `tool_description` |
| **Sources** | PoisonedRAG (+ BEIR), BIPIA, MCPTox, MSB |
| **Held out** | MCPTox (`loso_holdout`, an unseen surface) and MSB (`transfer_holdout`, an unseen benchmark) are never trained on |
| **File** | `data/processed/dataset.parquet`, built by the `build_dataset` DVC stage |
| **Load with** | `src/data/dataset.py`, which raises `HoldoutError` when a fit would include held-out data |
| **Profile** | `data/processed/profile.md`: 11,005 × 12, no unexpected nulls, no duplicate ids or texts, 44.2% adversarial |
| **Design** | [`docs/superpowers/specs/2026-09-29-build-dataset-design.md`](docs/superpowers/specs/2026-09-29-build-dataset-design.md) |

### Sources

| Source | Surface | Role | Adversarial | Benign | Groups | Upstream license |
|---------------|---------------------|-------------------|-----------:|--------:|----------:|----------------|
| PoisonedRAG + BEIR | `rag_corpus` | `train_pool` | 1,500 | 2,955 | 300 | MIT (BEIR corpora: per-dataset terms) |
| BIPIA | `tool_output` | `train_pool` | 2,806 | 2,806 | 2,806 | MIT; TableQA CC BY-SA 4.0; NewsQA and XSum terms |
| MCPTox | `tool_description` | `loso_holdout` | 485 | 362 | 45 | None stated (team-internal copy) |
| MSB | `tool_description` | `transfer_holdout` | 55 | 21 | 11 (with tool output) | MIT |
| MSB | `tool_output` | `transfer_holdout` | 15 | 0 | | MIT |

Each upstream repository is pinned to a commit in `config.yaml`.

## Motivation

**For what purpose was the dataset created?**
To train and evaluate detectors for *indirect* prompt injection: classifying text that an
LLM application reads from RAG retrieval or MCP tools as adversarial or benign. Detection
had to be measurable in three settings: within a surface, on an unseen surface
(leave-one-surface-out, MCP tool descriptions), and on an unseen benchmark (MSB). No single
public source covers all three, so the sources are unified into one schema with fixed
holdouts.

**Who created the dataset, and on behalf of which entity?**
The CS 7267 (Machine Learning) project team.
- Carter Corbin, Max Edmonds, Mohammed Adil Ahmed, Oluwamayowa Adewummi, Adedapo Odeyemi

**Who funded the creation of the dataset?**
- None

## Composition

**What do the instances represent?**
Each row is one untrusted text segment with its label and provenance. Columns:

| Column | Meaning |
|----------------------------|------------------------------------------------------------------------|
| `id` | Unique; traceable to the source record (for example `mcptox:FileSystem_1`) |
| `text` | Detector input, stripped of surrounding whitespace |
| `label` | `1` adversarial, `0` benign |
| `attack_kind` | `instruction` (injected commands) or `poisoning` (false content); null if benign |
| `attack_objective` | `task_hijack`, `data_exfiltration`, `misinformation`, `denial_of_service`, `tool_selection_manipulation`; null if benign |
| `attack_family` | The source's own attack category, verbatim; null if benign |
| `source`, `subset` | Origin dataset and its sub-collection (for example `bipia` / `email`) |
| `attack_surface` | `rag_corpus`, `tool_output` or `tool_description` |
| `group` | Unit that decides the split: question, context, MCP server or attack task |
| `split` | `train`, `val` or `test`, assigned per group |
| `role` | `train_pool`, `loso_holdout` or `transfer_holdout` |

**How many instances are there?**
11,005 in total. Adversarial rows by payload and objective:

| `attack_kind` | `attack_objective` | Rows |
|---|---|---:|
| `instruction` | `task_hijack` | 2,847 |
| `instruction` | `data_exfiltration` | 214 |
| `instruction` | `denial_of_service` | 149 |
| `instruction` | `misinformation` | 129 |
| `instruction` | `tool_selection_manipulation` | 22 |
| `poisoning` | `misinformation` | 1,500 |

**Is it a sample from a larger set?**
Yes, in three places:
- **PoisonedRAG:** 100 target questions each from NQ, MS MARCO and HotpotQA. Benign rows are
  each question's top-10 Contriever-retrieved real passages (hard negatives), not a random
  corpus sample.
- **BIPIA:** one attack per context, sampled with seed `7267` and inserted at a sampled
  position (start, middle or end), rather than BIPIA's full ~600k context × attack cross
  product.
- **MSB:** only the static text an attack produces. Variants whose attack exists only at run
  time (most `name_overlap` and `tool_transfer` files reuse a clean docstring) are left
  out.

**What data does each instance consist of?**
Raw text only: no features, embeddings or tokenization.

**Is there a label or target?**
`label`. `attack_kind`, `attack_objective` and `attack_family` support per-category
analysis. `attack_objective` is the team's coding judgment, mapped from each source's own
categories in `config.yaml`. It is not an upstream label.

**Is any information missing from individual instances?**
Benign rows have null attack fields by design: 6,144 nulls in each of `attack_kind`,
`attack_objective` and `attack_family`, all on benign rows. No other column has a null or
a blank text (`data/processed/profile.md`, regenerated by the `profile_dataset` stage). MSB
contributes no benign `tool_output` rows because upstream has none.

**Are relationships between instances explicit?**
Yes, through `group`. An adversarial row and the benign context it was built from share a
group (BIPIA, MSB descriptions), as do a question's poisoned and retrieved passages
(PoisonedRAG).

**Are there recommended splits?**
Yes. Splits are 70/15/15 by group, so a group never spans two splits. BIPIA keeps its own
train/test boundary (our `val` is 15% of BIPIA train groups), so BIPIA `test` mostly holds
attack families never seen in training. `role` sits on top of `split`:

| Role | Train | Val | Test | Use |
|--------------------|---------:|---------:|---------:|-----------------------------------------------------|
| `train_pool` | 7,217 | 1,416 | 1,434 | Fit on train, tune on val, report in-distribution on test |
| `loso_holdout` | 572 | 166 | 109 | Evaluation only; score every row |
| `transfer_holdout` | 73 | 6 | 12 | Evaluation only; score every row |

Splits inside the holdouts exist only for consistency. Evaluation should use the whole
source.

**Are there errors, noise or redundancies?**
- 47 exact duplicate texts with the same label were dropped (first kept). The build fails if
  a text appears with both labels.
- Near-duplicates are not merged, for example the same BIPIA email with an extra header, or
  MCPTox attacks built from shared templates. Grouping keeps most of them inside one split.
- BIPIA WebQA contexts were rebuilt from NewsQA with the official toolchain. They match our
  pinned md5s but not BIPIA's own md5s, so story text can differ slightly from upstream's
  build.

**Known shortcuts (spurious cues a detector could learn).**
Already removed: surrounding whitespace (BIPIA insertion artifacts), passage titles that
only benign PoisonedRAG rows would have had, and source-specific tool-description layouts
(all rendered as `Tool: {name}\nDescription: {description}`).

Still present, so check before trusting a score. Median character lengths:

| Source | Benign | Adversarial |
|---|---:|---:|
| PoisonedRAG | 389 | 183 |
| BIPIA | 1,752 | 1,816 |
| MCPTox | 88 | 292 |
| MSB | 156 | 314 |

Length alone separates the classes well, and in opposite directions: poisoned passages are
shorter than retrieved ones, while injected tool descriptions are longer than clean ones.
Length separability, `max(AUROC, 1 − AUROC)` of raw length, and the AUROC of
`LengthBaseline` (logistic regression on log length, fit on `train_pool` train):

| Eval set | Fitted AUROC | Separability |
|---|---:|---:|
| In-distribution test | 0.491 | 0.509 |
|   BIPIA | 0.532 | 0.532 |
|   PoisonedRAG | 0.138 | 0.862 |
| LOSO (MCPTox) | 0.774 | 0.774 |
| Transfer (MSB) | 0.866 | 0.866 |

Report both next to every detector, so length is the score to beat. To regenerate, run
`PYTHONPATH=src/data:src/eval uv run python src/models/baselines.py`.

**Is the dataset self-contained?**
No. `dataset.parquet` is derived from upstream repositories pinned in `config.yaml`, BEIR
corpora, XSum (Hugging Face) and NewsQA. NewsQA and the CNN stories sit behind terms of use,
so that build lives only in the team's private DVC remote.

**Does the dataset contain confidential data?**
Nothing secret, but the BIPIA emails come from a real inbox. All text comes from public
benchmarks. A spot-check on 2026-10-01 (samples plus a regex scan of every benign BIPIA
email and NewsQA row for email addresses, phone numbers, SSNs and card numbers) found:

- **BIPIA emails (78 benign contexts) do not look synthetic.** BIPIA takes them from
  [OpenAI Evals](https://github.com/openai/evals) and does not say whether they are real.
  They read as one company's actual inbox: 64 of the 78 are addressed to "David" at
  Moonchaser (`david@moonchaser.io`). Most are Mercury bank notifications, plus receipts and
  invoices from Deel, PayPal, Upwork, Webflow, Eurostar and Air Canada, with real dates in
  2022, dollar amounts and order numbers. Account numbers are masked to their last four
  digits (three accounts appear). No full account or card numbers, phone numbers or SSNs
  were found.
- **NewsQA contexts are published CNN articles.** The only contact details are CNN's own
  (a fact-check address and a public phone line).

**Does it contain offensive, insulting or threatening content?**
Yes, by construction. Adversarial rows include working prompt-injection payloads, malware
and ransomware instructions (BIPIA code attacks), scam, phishing and propaganda text, and
data-exfiltration instructions aimed at agents.

**Does it identify individuals or contain sensitive personal data?**
Yes, in two ways:

- **News-derived contexts** (NewsQA, XSum, BEIR passages) name people as reported in
  published articles. Most are public figures, but crime and accident reports also name
  private individuals.
- **BIPIA emails** name private individuals in a business setting: the recipient, their
  company, and payees or senders such as contractors. They also reveal that company's
  vendors, payment amounts and transaction dates. This is financial information about
  identifiable people, already public through OpenAI Evals and BIPIA.

Nothing was collected from people by this project, and no rows were removed. Do not quote
BIPIA email rows in the paper or slides; use a BIPIA table or code example to illustrate
`tool_output`.

## Collection process

**How was the data acquired?**
Entirely from existing datasets; nothing was newly written by annotators.
1. `fetch` downloads pinned files from each upstream repository, plus the BEIR passages
   PoisonedRAG targets.
2. `bipia_summarization` and `bipia_webqa` rebuild BIPIA contexts that upstream does not
   redistribute (XSum, NewsQA).
3. `build_dataset` converts every source to the schema, assigns splits, validates
   invariants and deduplicates.

The pipeline is defined in `dvc.yaml`. `dvc.lock` records the hash of every input and
output.

**What sampling strategy was used?** See *Is it a sample* above. All randomness uses seed
`7267`, and the same seed produces an identical file (tested).

**Who was involved, and how were they compensated?**
The project team, as coursework. Upstream authors created all underlying text.

**Over what timeframe?**
The build was designed and produced on 2026-09-29. The current `dvc.lock` was committed
then in `af5ec37` ("Fix label shortcuts and split leaks found in review") and locks
`dataset.parquet` at md5 `a1c6426890678d576fb6231dc6050491`. Upstream commits are pinned in
`config.yaml`. The text itself is older: the BIPIA emails date from 2022, and the news
articles are older still.

**Were ethical review processes conducted?**
No IRB review was sought. The project does not interact with people or collect data from
them: every row comes from existing, publicly released benchmarks. Under the U.S. Common
Rule (45 CFR 46.102), that is generally not human subjects research, because no data is
obtained through interaction with people and publicly available information is not
"private information."

The closest case is the BIPIA emails, which identify private individuals (see
*Composition*). They were already published through OpenAI Evals and BIPIA, and this
project uses them unchanged, only as model inputs, without studying the people in them.
- The prohect team will engage with the University's IRB for a "not human subjects research" determination if this work is published beyond the course.

**Consent and notification of individuals:** no data was collected from people, so this
project sought no consent and sent no notifications. The people in BIPIA's emails and in
the news articles were not asked by this project; whether they consented upstream is not
documented.

## Preprocessing, cleaning and labeling

**What preprocessing was done?**
- Whitespace stripped from every `text`.
- BIPIA attacks inserted into contexts following `bipia/data/utils.py` (`\n` joins at start
  or end, a Punkt sentence boundary for middle). Code contexts joined with `\n`.
- MSB docstrings extracted with `ast` without importing upstream code. Prompt-injection
  descriptions and tool responses rebuilt from MSB's templates, with a fixed `{pid}`.
- Exact duplicates dropped. Rows that fail any invariant stop the build.

**How were labels assigned?**
`label` comes from each source's construction: an attack text is adversarial, and a clean
context or tool description is benign. `attack_objective` is a manual mapping of about 75
upstream categories, reviewed like an edit to the team's literature-review coding matrix.

**Is the raw data saved?**
Yes. Fetched and rebuilt inputs are tracked by DVC under `data/raw` and `data/generated`.

**Is the preprocessing software available?**
Yes: `src/data/`, tested in `tests/test_build_dataset.py` and `tests/test_generate_bipia.py`.

## Uses

**Intended uses**
- Training and evaluating text-level detectors for indirect prompt injection on RAG and MCP
  surfaces.
- Measuring generalization to an unseen surface (MCPTox) and an unseen benchmark (MSB).
- Per-category error analysis by `attack_kind`, `attack_objective` and `attack_family`.

**Uses to avoid**
- **Training on held-out sources.** Fitting on MCPTox or MSB invalidates the LOSO and
  transfer results. Always load through `dataset.load("fit", ...)`.
- **Measuring direct-injection or jailbreak robustness.** No row comes from the user's own
  message.
- **Estimating false-positive rates on MCP tool responses.** There are no benign MCP tool
  outputs.
- **Reporting accuracy on the holdouts.** Class balance varies by source: MSB is 76.9%
  adversarial and MCPTox 57.3%, so labeling everything adversarial already scores 77% on
  MSB. Report AUROC and TPR at a fixed low FPR instead.
- **Comparing results across payload types without stratifying.** Training mixes poisoning
  and instruction injection, but both holdouts are instruction-only. Report scores per
  `attack_kind`.
- **Building or tuning attack generators.** The adversarial rows are working payloads.

**Could the composition affect future uses?**
Benign examples come from a narrow set of domains: news, Wikipedia-style QA passages,
synthetic emails, code snippets, and the tool descriptions of a few dozen MCP servers. A
detector may flag ordinary text from other domains, or text that merely *talks about*
instructions (documentation, tutorials). False-positive rates outside these domains are
unknown.

## Distribution

**Will the dataset be distributed outside the team?**
- No. The data lives in a private Cloudflare R2
DVC remote, and teammates get access keys from the project owner.

Constraints if that changes:
- NewsQA and the CNN stories may not be redistributed. Share the build pipeline, not the
  rebuilt contexts.
- MCPTox states no license upstream, so its use here is team-internal only.
- TableQA content is CC BY-SA 4.0, so derivatives must keep that license.

- Export controls or other regulatory restrictions: None known.

## Maintenance

**Who maintains the dataset, and how can they be contacted?**
- Adedapo Odeyemi ([\@blacng](https://github.com/blacng)), for the CS7267 project team. Open an issue on this repository for questions or problems with the data.


**Will the dataset be updated?**
Through pull requests. Changes to pinned source commits, sampling parameters or the
objective mapping go in their own PR, with `dvc.lock` and `summary.json` committed and the
rebuilt data pushed. `uv run dvc metrics diff` shows how row counts changed.

**Are older versions kept?**
Yes. Every committed `dvc.lock` points to its outputs in the DVC remote, so
`git checkout <commit> && make data` restores that version.

**Can others contribute?**
Team members, through the pull-request workflow in [`README.md`](README.md#contributing).
