# Model card: B1-Optuna prompt-injection detector

A model card in the format of Mitchell et al. (2019), for the detector this project would
deploy, with a section on the fine-tuned transformer (B2) and why it is not recommended. It
complements the dataset's datasheet, [DATA_CARD.md](DATA_CARD.md). Every number below is copied
from a committed result file, linked next to it.

> **Deployment boundary.** Validated on RAG passages and tool outputs only. It is **not
> validated for MCP tool descriptions**: there it ranks attacks only as well as text length
> does (AUROC 0.767 vs 0.774) and flags 6.9% of benign descriptions at a 1% target.

## Model details

| | |
|---|---|
| Name | B1-Optuna (`tfidf-optuna`) |
| Type | Logistic regression on TF-IDF features: word 1–3 grams plus character 2–4 grams (up to 200,000 of each; 377,214 in the fitted model), `C` = 6.38, balanced class weights |
| Selection | 100-trial Optuna TPE search, objective = mean AUROC over 5 family-grouped folds of the training split (trial 93, mean CV AUROC 0.921); settings in [best_params.json](report/phase4/b1_optuna/best_params.json) |
| Output | A score in [0, 1]; flagged at or above 0.6245, the score at 1% FPR on validation benign rows ([thresholds.json](report/phase5/thresholds.json)) |
| Code | [src/models/tfidf_optuna.py](src/models/tfidf_optuna.py); fit with `fit_detector("tfidf-optuna", data)` in [src/models/train.py](src/models/train.py) |
| Developers | CS 7267 team, Kennesaw State University, December 2026 |
| License | Course project; code under the repository's license. The training data keeps each source's license ([DATA_CARD.md](DATA_CARD.md)) |

## Intended use

- **Primary use:** a first-pass filter that scores untrusted text before an LLM agent reads it:
  retrieved RAG passages and tool outputs. A flag routes the text to review or to a restricted
  context; it is one layer of defense, not the defense.
- **Primary users:** developers of LLM agents and researchers comparing injection detectors.
- **Out of scope:**
  - MCP tool descriptions and any surface not in the validation set (Section *Quantitative
    analyses*). Skill files and agent definitions are untested.
  - Blocking without review. At its threshold the detector misses 43% of in-distribution
    attacks and 65% of MCPTox attacks.
  - Adversaries who know the model. Character obfuscation removes most of its recall.
  - Non-English text. Only 26 evaluation rows contain Chinese, Japanese or Korean text, too few to measure.

## Factors

The data contains no people, so demographic factors do not apply. The factors that change the
detector's behavior are the **attack surface** (RAG passage, tool output, tool description),
the **source benchmark**, the **attack kind** (instruction vs knowledge poisoning), **text
length**, the **script** (Latin vs CJK), and **adversarial rewriting**. Results by factor are in
[fairness.csv](report/phase5/fairness.csv) and the red-team summaries.

## Metrics

AUROC with a 95% bootstrap interval over groups (BIPIA twin pairs never split); TPR and FPR at
the deployed threshold, with Wilson intervals for FPR; TPR at 5% FPR for comparison across
detectors. FPR is the metric that matters most for fairness, because it measures benign
content that gets blocked.

## Evaluation data

- **Test split** of the training pool (BIPIA tool outputs, PoisonedRAG passages): 827 benign,
  607 adversarial.
- **MCPTox** MCP tool descriptions, never seen in training (leave-one-surface-out): 362 benign,
  485 adversarial.
- **MSB** tool descriptions and outputs: 21 benign, 70 adversarial; indicative only.

## Training data

Train split of BIPIA (tool outputs) and PoisonedRAG (RAG passages), 7,217 rows; validation
split (1,416 rows) sets the threshold. See [DATA_CARD.md](DATA_CARD.md).

## Quantitative analyses

| Eval set | AUROC [95% CI] | TPR | FPR [Wilson 95%] |
|---|---|---|---|
| Test | 0.922 [0.907, 0.935] | 0.567 | 0.024 [0.016, 0.037] |
| MCPTox | 0.767 [0.736, 0.800] | 0.348 | 0.069 [0.047, 0.100] |
| MSB | 0.559 [0.441, 0.677] | 0.157 | 0.048 [0.008, 0.227] |

Source: [b1_tfidf_lr_optuna.csv](report/baselines/b1_tfidf_lr_optuna.csv), [fairness.csv](report/phase5/fairness.csv).

- **By group.** On the test split, BIPIA tool outputs are harder than PoisonedRAG passages
  (AUROC 0.862 vs 0.965; FPR 3.4% vs 1.6%). On MCPTox, the false-positive rate rises with
  length: 0% for the shortest quarter of benign descriptions, 14.3% [8.5%, 22.9%] for the
  longest.
- **Robustness.** At a 5% validation-FPR threshold, leetspeak cuts recall on rewritten
  test attacks from 0.725 to 0.213, zero-width characters to 0.483 and homoglyphs to 0.614;
  paraphrase and formatting cost at most 0.06
  ([summary.csv](report/red_team/tfidf-optuna/summary.csv)).
- **Cost.** 0.58 ms per text on one laptop CPU core (Apple M4 Max), 377,215 weights, a 7-second
  refit; the 100-trial search took 55 minutes ([latency.csv](report/phase5/latency.csv)).

## Ethical considerations

- **False sense of security.** Strong in-distribution numbers hide failure on new surfaces.
- **Blocking legitimate content.** Long, instruction-style tool documentation is flagged most;
  the cost falls on honest tool authors.
- **Dual use.** Published attributions and red-team results show attackers what to avoid.
- **Data.** Every training text comes from public benchmarks; no personal data is collected.

## Caveats and recommendations

- Normalize text before scoring (NFKC, strip zero-width characters).
- Set thresholds per surface, on benign data from that surface, and review flags rather than
  block.
- Combine with architectural defenses: privilege separation, tool allow-lists, and user
  confirmation of side-effecting calls.
- Before using it on MCP tool descriptions, add benign and adversarial tool descriptions to
  training; their most common attack phrasing, "before X, call Y", is nearly absent from the
  current training data.

## Not recommended: B2 DeBERTa-v3 (row 2)

The fine-tuned DeBERTa-v3-base reaches test AUROC 0.987 ± 0.004 over three seeds but ranks MCP
tool descriptions below chance (0.346 ± 0.128) and fails its pre-registered success criterion
on every seed ([b2_verdict.md](report/baselines/b2_verdict.md)). On MCPTox it scores 98–100%
of rows, benign and attack alike, below 0.01, and among those scores it rates longer text as more benign, so it ranks MCP rows
by inverse length rather than by their phrasing. It
is also the most brittle detector: leetspeak cuts its recall on rewritten test attacks from
0.984 to 0.110 and random capitalization to 0.124
([summary.csv](report/red_team/deberta_row2_s7267/summary.csv)). It is about 100 times slower than B1-Optuna (56 ms per text on the
same CPU, 44 ms on the laptop GPU) and has 184 million parameters. We document it for comparison
only.
