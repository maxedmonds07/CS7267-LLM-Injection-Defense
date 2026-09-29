# `build_dataset`: one labeled table across all injection sources

Date: 2026-09-29
Status: draft, awaiting team review

## Goal

Turn the fetched and generated sources (PoisonedRAG, BIPIA, MCPTox, MSB) into a single
labeled table of **untrusted text segments**, so every detector and evaluation reads the
same file and the leave-one-surface-out (LOSO) and transfer holdouts cannot leak into
training.

A row is one piece of text that reaches the model without the user writing it: a retrieved
passage, an email or web document, a tool description, or a tool response. The detector's
job is to classify it as adversarial or benign.

AgentDojo is out of scope: it is the end-to-end attack-success harness, not a text dataset.

## Decisions

| # | Decision | Choice |
|---|---|---|
| 1 | What `label = 1` means | Any adversarial content. `attack_kind` separates `instruction` (injected commands) from `poisoning` (false content). |
| 2 | Where holdout rules live | Grouped `train/val/test` within every source, plus a `role` column from `config.yaml`. A loader refuses held-out roles for training unless explicitly allowed. |
| 3 | Column vocabulary | `attack_surface` and `attack_objective` use the values of the team's literature-review coding matrix. |
| 4 | BIPIA's surface | `tool_output`: BIPIA contexts are documents fetched at runtime (email, web page), not a poisoned retrieval corpus. Its clean contexts are also the only benign `tool_output` rows. |
| 5 | BIPIA size | One sampled attack per context at a sampled position (start/middle/end), not the full ~600k cross product. |
| 6 | PoisonedRAG negatives | Top-10 Contriever-retrieved real passages per target question (hard negatives). |

## Schema

Output: `data/processed/dataset.parquet`, one row per text segment.

| Column | Type | Values / example | Notes |
|---|---|---|---|
| `id` | str | `mcptox:FileSystem_1`, `bipia:email:test:12:TaskAutomation-3:end` | Unique; traceable to the source record |
| `text` | str | | Detector input |
| `label` | int8 | `1` adversarial, `0` benign | |
| `attack_kind` | str, null if benign | `instruction`, `poisoning` | Payload type |
| `attack_objective` | str, null if benign | `task_hijack`, `data_exfiltration`, `misinformation`, `denial_of_service`, `tool_selection_manipulation` | Coding-matrix "Attack objective"; mapped in `config.yaml` |
| `attack_family` | str, null if benign | BIPIA category, MCPTox risk + template, MSB attack type, `poisonedrag` | The source's own category, kept verbatim |
| `source` | str | `poisonedrag`, `bipia`, `mcptox`, `msb` | |
| `subset` | str | `nq`, `email`, `FileSystem`, `PubMed_MCP_Server` | |
| `attack_surface` | str | `rag_corpus`, `tool_output`, `tool_description` | Coding-matrix "Attack surface". `direct_user_input` has no source (see Limitations) |
| `group` | str | question id, BIPIA context id, MCP server | Unit that decides the split |
| `split` | str | `train`, `val`, `test` | Assigned per group |
| `role` | str | `train_pool`, `loso_holdout`, `transfer_holdout` | Per source, from `config.yaml` |

Invariants: `label == 0` exactly when `attack_kind`, `attack_objective` and `attack_family`
are null; `id` is unique; a `group` never spans two splits.

## Per-source construction

### PoisonedRAG → `rag_corpus`, `poisoning`, `misinformation`, role `train_pool`

- **Adversarial:** the 5 `adv_texts` per target question in
  `results/adv_targeted_results/{nq,msmarco,hotpotqa}.json` (3 × 100 × 5 = 1,500).
- **Benign:** the top-`poisonedrag_benign_top_k` (default 10) passages from
  `results/beir_results/{subset}-contriever.json` for the same question, text from
  `beir/{subset}/corpus.jsonl` (title and text joined as the retriever sees them). ~3,000.
- `group` = `{subset}:{question id}`.

### BIPIA → `tool_output`, `instruction`, role `train_pool`

- **Benign:** every context in `benchmark/{email,table,code}/{train,test}.jsonl` and
  `data/generated/bipia/{qa,abstract}/{train,test}.jsonl`. ~3,100.
- **Adversarial:** for each context, `bipia_attacks_per_context` (default 1) attacks sampled
  with the seed from the matching attack file: `text_attack_{split}.json` for email, table,
  qa and abstract, `code_attack_{split}.json` for code. The attack is inserted at a sampled
  position (start, middle, end) following BIPIA's own insertion code. ~3,100.
- **Splits:** BIPIA's `train` contexts and attacks feed our `train`/`val` (val = 15% of
  train groups); BIPIA's `test` feeds our `test`. BIPIA's train and test attack categories
  are disjoint, so our test measures unseen attack families.
- `group` = `{task}:{bipia split}:{context index}`, so a context's clean and injected
  versions share a split.

### MCPTox → `tool_description`, `instruction`, role `loso_holdout`

- **Adversarial:** the 485 `tool_content` entries in `pure_tool.json`. `attack_family` =
  `{security risk} / {paradigm}`.
- **Benign:** each tool's description parsed from `servers[*].clean_system_promot` in
  `response_all.json` (the `Tool: … Description: …` blocks). ~350.
- `group` = `server_name`.

### MSB → `tool_description` and `tool_output`, `instruction`, role `transfer_holdout`

- **Descriptions:** tool docstrings from the attack-variant server files
  (`tools/attack_tools/*/*/*-{attack_type}.py`), extracted with `ast` without importing the
  files. Benign rows: docstrings from the clean `*_server.py` files in the same directories.
- **Responses:** `TOOL_RESPONSE_ATTACK_TEMPLATE` or `SIMULATED_USER_TEMPLATE` (from
  `data/prompt_template.py`) + each of the 5 `attack_task.jsonl` implementations, one row
  per response attack type × task (~15 unique texts; a small transfer check only).
- `group` = the server directory for description rows (an attack-variant docstring is the
  clean one plus an injected sentence, so both must share a split) and the attack task for
  response rows.
- Requires adding `data/prompt_template.py` to MSB's fetch `paths` in `config.yaml`.

## Configuration

New `dataset:` section in `config.yaml`:

```yaml
dataset:
  out: data/processed/dataset.parquet
  summary: data/processed/summary.json
  seed: 7267
  split_ratios: {train: 0.70, val: 0.15, test: 0.15}
  poisonedrag_benign_top_k: 10
  bipia_attacks_per_context: 1
  sources:
    poisonedrag: {role: train_pool, attack_surface: rag_corpus}
    bipia:       {role: train_pool, attack_surface: tool_output}
    mcptox:      {role: loso_holdout, attack_surface: tool_description}
    msb:         {role: transfer_holdout}   # surface set per row: description vs response
  # Source category -> coding-matrix attack objective. Every category must be mapped.
  objectives:
    bipia: {Task Automation: task_hijack, Keylogging: data_exfiltration, ...}
    mcptox: {Credential Leakage: data_exfiltration, Information Manipulation: misinformation, ...}
    msb: {name_overlap: tool_selection_manipulation, prompt_injection: task_hijack, ...}
```

The ~75 objective mappings are a coding judgment, the same one a paper coder makes. The
implementation proposes them; a teammate reviews them in the PR.

## Code layout

- `src/data/build_dataset.py`: one function per source (`rows_poisonedrag`, `rows_bipia`,
  `rows_mcptox`, `rows_msb`) returning rows in the schema, then a shared pass that assigns
  splits by group, validates the invariants, deduplicates, and writes the Parquet file and
  the summary. Usage matches the other stages:
  `python src/data/build_dataset.py --config config.yaml`.
- `src/data/dataset.py`: `load(roles=..., splits=..., allow_holdout=False)`. Raises if a
  requested `train` or `val` split would include a `loso_holdout` or `transfer_holdout` row
  and `allow_holdout` is false.

## Pipeline

New stage in `dvc.yaml`:

```yaml
build_dataset:
  cmd: python src/data/build_dataset.py --config config.yaml
  deps:
    - src/data/build_dataset.py
    - data/raw
    - data/generated/bipia
  params:
    - config.yaml:
        - dataset
  outs:
    - data/processed/dataset.parquet
  metrics:
    - data/processed/summary.json:
        cache: false
```

`summary.json` holds row counts by source × attack_surface × label × split, plus the number
of dropped duplicates. It is committed to git so `dvc metrics diff` shows data changes in
review.

## Failure handling

The build stops with a clear message, and writes nothing, when:

- a source category has no objective mapping (all missing ones are listed);
- the same text appears with both labels;
- an invariant fails (duplicate `id`, a group across splits, attack fields on a benign row);
- an expected input file is missing (points to `make data` / `dvc pull`).

Exact duplicate texts with the same label are dropped (first occurrence kept) and counted in
the summary; MCPTox templates are expected to produce some.

## Testing

`tests/test_build_dataset.py`, with small fixtures in each source's format (no real data
needed, like `test_generate_bipia.py`):

- each `rows_*` function turns its fixture into the expected rows, including BIPIA insertion
  at start/middle/end and MSB docstring extraction;
- no group spans two splits; benign rows have null attack fields;
- the same seed produces an identical file;
- every category in the real `config.yaml` has an objective mapping (reads only config and
  the category lists, not the data);
- `load` refuses held-out roles for train/val unless `allow_holdout=True`.

## Size estimate

Exact counts come from `summary.json` after the first build.

| attack_surface | adversarial | benign | role |
|---|---|---|---|
| rag_corpus (PoisonedRAG) | ~1,500 | ~3,000 | train_pool |
| tool_output (BIPIA) | ~3,100 | ~3,100 | train_pool |
| tool_description (MCPTox) | 485 | ~350 | loso_holdout |
| tool_description + tool_output (MSB) | tens to low hundreds | tens | transfer_holdout |

## Limitations

- **Indirect injection only.** No source contains `direct_user_input` attacks (jailbreaks in
  the user's own message). This matches the project scope (RAG and MCP surfaces) and should
  be stated in the write-up.
- **Few benign tool responses outside BIPIA.** MSB ships no benign tool outputs, so
  false-positive rate on MCP tool responses specifically cannot be measured.
- **BIPIA sampling.** One attack per context makes per-category statistics noisy; raise
  `bipia_attacks_per_context` if needed.
- **Payload mismatch across sources.** The training pool mixes poisoning (PoisonedRAG) and
  instruction injection (BIPIA), while both holdouts are instruction-only. Report results per
  `attack_kind` so a weak holdout score can be attributed correctly.

## Out of scope

Tokenization, embeddings, training-time sampling or weighting, AgentDojo, and MLflow logging.
These belong to the baseline and evaluation work that follows.

## Feedback for the literature-review coding matrix

Building this dataset against the coding matrix surfaced one missing dimension:

- **Payload type (proposed new column).** Dropdown: `Instruction injection; Content
  poisoning; Both`. "Attack objective" records *what* the attacker wants (for example
  misinformation), but not *how* the payload works: text that issues commands to the model
  (BIPIA, MCPTox, MSB) versus plausible false content with no commands (PoisonedRAG). The two
  call for different defenses (instruction detectors versus cross-source consistency checks),
  so papers should be coded separately on it. This is the dataset's `attack_kind` column.

A possible second addition, depending on how fine-grained the review needs to be:
**Attack technique** (for example name overlap, preference manipulation, encoding or
obfuscation), which describes how the attack evades notice rather than what it wants. The
dataset keeps this as `attack_family`.
