# CS7267 LLM Injection Defense

Detecting prompt injection across RAG and MCP surfaces.

LLM applications increasingly read text the user never wrote: passages retrieved for RAG,
emails and web pages fetched by tools, MCP tool descriptions, and tool responses. Any of it
can carry an **indirect prompt injection**: instructions or false content planted by an
attacker. This project builds a detector that classifies such untrusted text segments as
adversarial or benign, and measures how well it generalizes:

- **within a surface:** train and test on RAG passages and tool outputs;
- **to an unseen surface (LOSO):** leave MCP tool descriptions out of training entirely;
- **to an unseen benchmark (transfer):** test on MSB, which no model ever trains on.

AgentDojo is pinned as the end-to-end attack-success harness for evaluating a defended agent.

**Status:** the data pipeline and labeled dataset are in place. Detector baselines, training
and evaluation are the next phase; experiments are tracked in the team's shared MLflow
server.

## Repository layout

```text
config.yaml               Data sources (pinned commits), dataset build settings, harness versions
dvc.yaml / dvc.lock       Data pipeline stages and their locked outputs
Makefile                  Setup, data sync and one-off data tasks (`make help`)
src/data/
  fetch.py                Download pinned source files into data/raw
  generate_bipia.py       Rebuild BIPIA contexts that upstream does not redistribute
  build_dataset.py        Build the unified labeled table, data/processed/dataset.parquet
  profile_dataset.py      Profile the table (nulls, class balance, length), data/processed/profile.md
  dataset.py              Load the table, refusing held-out data when fitting
tests/                    Pytest suite (fixtures only; most tests need no real data)
mlflow-server/            Shared MLflow tracking server setup (see its README)
mlflow-demo/              Minimal example of logging a run to MLflow
docs/superpowers/         Design specs and implementation plans
scripts/                  Helpers, e.g. multipart upload for large DVC files
```

## Getting started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (installs Python 3.11 and all dependencies)
- `make`
- R2 access keys for the team's DVC remote, from the project owner
- Docker, only if you rebuild the NewsQA CSV (`make newsqa`; almost nobody needs to)

### Setup

```bash
git clone https://github.com/maxedmonds07/CS7267-LLM-Injection-Defense.git
cd CS7267-LLM-Injection-Defense

make install          # uv sync: project + dev dependencies into .venv
make r2-credentials   # store your R2 keys in .dvc/config.local (git-ignored)
make data             # dvc pull, then dvc repro for any stale stage
uv run pytest         # check the setup
```

The DVC remote location is already committed in `.dvc/config`; `make r2-credentials` only
adds your keys. Run `make check-remote` to confirm the remote is reachable.

Run every Python command through `uv run` (for example `uv run python ...`), or activate
`.venv` first.

## Datasets

### Sources

All sources are pinned to an exact commit in `config.yaml`, so `dvc repro` is reproducible.

| Source | Surface | Role in the dataset | License |
|---|---|---|---|
| [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) + BEIR passages | RAG corpus | `train_pool` | MIT |
| [BIPIA](https://github.com/microsoft/BIPIA) | Tool output (email, web, table, code, summaries) | `train_pool` | MIT; TableQA CC BY-SA 4.0 |
| [MCPTox](https://github.com/zhiqiangwang4/MCPTox-Benchmark) | MCP tool descriptions | `loso_holdout` | None stated (team-internal copy) |
| [MSB](https://github.com/dongsenzhang/MSB) | Tool descriptions and tool output | `transfer_holdout` | MIT |
| [AgentDojo](https://github.com/ethz-spylab/agentdojo) 0.1.35 | End-to-end agent harness | Not in the table | MIT |

BIPIA's WebQA (NewsQA) and Summarization (XSum) contexts are not redistributed upstream.
The `bipia_webqa` and `bipia_summarization` stages rebuild them and check the result against
pinned md5s. NewsQA sits behind Microsoft's and DeepMind's terms of use, so it was built once
with `make newsqa ACCEPT_TERMS=1` and lives only in the team's private R2 remote.

### The unified table

`build_dataset` writes `data/processed/dataset.parquet`, one row per untrusted text segment:

| Column | Meaning |
|---|---|
| `text`, `label` | Detector input; `1` adversarial, `0` benign |
| `attack_kind` | `instruction` (injected commands) or `poisoning` (false content) |
| `attack_objective` | `task_hijack`, `data_exfiltration`, `misinformation`, `denial_of_service`, `tool_selection_manipulation` |
| `attack_family` | The source's own attack category, verbatim |
| `source`, `subset`, `attack_surface` | Where the text comes from: `rag_corpus`, `tool_output`, `tool_description` |
| `group`, `split` | `train`/`val`/`test` assigned per group, so a group never spans two splits |
| `role` | `train_pool`, `loso_holdout` or `transfer_holdout` |

Current build (`data/processed/summary.json`, 11,005 rows, 47 exact duplicates dropped):

| Source | Surface | Adversarial | Benign |
|---|---|---:|---:|
| PoisonedRAG | `rag_corpus` | 1,500 | 2,955 |
| BIPIA | `tool_output` | 2,806 | 2,806 |
| MCPTox | `tool_description` | 485 | 362 |
| MSB | `tool_description` | 55 | 21 |
| MSB | `tool_output` | 15 | 0 |

The full design, including per-source construction, the attack-objective mapping and known
limitations, is in
[`docs/superpowers/specs/2026-09-29-build-dataset-design.md`](docs/superpowers/specs/2026-09-29-build-dataset-design.md).
[`DATA_CARD.md`](DATA_CARD.md) is the data card: intended uses, known
shortcuts and distribution constraints.

### Loading data

Always load through `src/data/dataset.py`. For `purpose="fit"` it raises `HoldoutError` if the
result would include a held-out source or the test split:

```python
from dataset import load  # run with PYTHONPATH=src/data (pytest sets it already)

train = load("fit", roles=["train_pool"], splits=["train"])
val = load("fit", roles=["train_pool"], splits=["val"])
test = load("eval", roles=["train_pool"], splits=["test"])
loso = load("eval", sources=["mcptox"])     # every MCPTox row
transfer = load("eval", sources=["msb"])    # every MSB row
```

`load` returns a `pyarrow.Table`; call `.to_pandas()` if you need a DataFrame.

## Data pipeline (DVC)

```text
fetch ──────┬──────────────> bipia_summarization ──┐
            ├──────────────> bipia_webqa ──────────┼──> build_dataset ──> profile_dataset
            │  NewsQA CSV ──────┘                  │
            └──────────────────────────────────────┘
```

| Command | What it does |
|---|---|
| `make data` | Pull everything from R2, then rebuild stale stages |
| `uv run dvc repro` | Rebuild stages whose code, params or inputs changed |
| `uv run dvc metrics diff` | Compare `summary.json` row counts and `profile.json` against `main` |
| `make push-data` | Upload new DVC outputs to R2 |
| `make push-large DVC_FILE=path.dvc` | Upload one large file in small parts (for slow uplinks), then `make push-data` |

After changing a stage, commit `dvc.lock`, `data/processed/summary.json` and
`data/processed/profile.{json,md}` with the code,
and push the data so teammates can `make data`.

## Experiment tracking (MLflow)

Everyone runs MLflow locally, backed by one shared Neon PostgreSQL database and one object
storage bucket, so all runs appear for the whole team. `mlflow` and `boto3` are main
dependencies; the server's extras are in the `mlflow-server` dependency group.

```bash
uv sync --group mlflow-server
cp mlflow-server/.env.example mlflow-server/.env   # fill in the credentials from the project owner
cd mlflow-server && uv run --group mlflow-server python start_mlflow.py
```

Then open <http://127.0.0.1:5000> and point training code at it:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("your-experiment")
```

[`mlflow-server/README.md`](mlflow-server/README.md) covers the architecture, what to log,
team rules and deleting experiments; `uv run python mlflow-demo/example_mlflow.py` logs a
small example run. Never commit `.env`.

## Tests

```bash
uv run pytest
```

Most tests use small fixtures. `tests/test_harnesses.py` checks that AgentDojo is installed at
the version pinned in `config.yaml`.

## Contributing

- Never commit to `main`. Work on a branch and open a pull request.
- Changes to the attack-objective mapping in `config.yaml` are coding judgments: review them
  like an edit to the literature-review coding matrix.
- Pinned source commits change only on purpose, in their own PR, with the rebuilt data
  pushed.
