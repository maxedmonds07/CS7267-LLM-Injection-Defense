# B2 `deberta`: a fine-tuned transformer detector that must beat B1 and length off-surface

Date: 2026-11-07
Status: draft, awaiting team review
Lead: Adedapo (Seun) Odeyemi; work streams below are split across the team

## Goal

Build a detector that beats B1 TF-IDF, and so also length, on the holdouts. On the in-distribution
test split, B1 TF-IDF reaches 0.902 AUROC, but it drops to 0.675 on MCPTox and 0.537 on
MSB, where raw length scores 0.774 and 0.866. The phase 2 report (gap 4) commits every
detector to being reported next to that baseline.

This detector follows PromptShield (P08): fine-tune a pretrained encoder and judge it at a
low false positive rate. It adds two changes aimed at why B1 fails:

- **(A) Chunked max-pool scoring.** Split the text into overlapping token windows, score
  each one, and use the highest window score as the text's score. An injection is a short
  span inside otherwise benign text, and nothing is truncated away.
- **(B) Pairwise loss on BIPIA twins.** Every BIPIA context appears once clean and once
  attacked, in the same `group`. A ranking term pushes the attacked twin above its clean
  twin. The rest of the text is identical, so the gradient points at the injected span.

**An honest negative result is an acceptable outcome.** If B2 does not beat B1 under the
criterion below, the write-up says so and shows why. The project succeeds either way, as
long as the evaluation is clean.

## Dates

| Date | Milestone |
|---|---|
| Nov 9–15 | Detector module and tests, Colab notebook, fp16 smoke test, new metrics (streams 1, 2, 4) |
| Nov 16–22 | Full training runs and ablation; selected model in R2. Main numbers should exist by **Nov 22** |
| Nov 23–30 | Evaluation, per-slice tables, stretch stream 5 if on time. **Numbers frozen Nov 30** |
| Dec 1–7 | Write-up only, due **Dec 7**. No new experiments after Nov 30 |

Cut order if behind: stream 5 (synthetic tool descriptions), then the ablation, then (B).
The floor is plain fine-tuning with (A).

## Goals, non-goals, anti-goals

**Goals**
- A detector with the same interface as `TfidfDetector`, evaluated by the existing harness
  (`src/eval/harness.py`) and `make evaluate`, so its numbers sit beside length, B0 and B1.
- A success criterion fixed before any training (next section).
- A clear answer to "does it learn injection or length?", from the length-binned metric and
  the BIPIA-only ablation.

**Non-goals**
- Off-the-shelf zero-shot classifiers (Prompt Guard, ProtectAI DeBERTa). Worth trying, but as
  a separate small task.
- Known-answer detection and activation probes (other review-derived ideas). Not this spec.
- AgentDojo end-to-end runs. Wiring the detector into AgentDojo is a later step.
- Changing `data/processed/dataset.parquet`. B0 and B1 are already reported on it.

**Anti-goals: these would make the work a failure even if the numbers look good**
- Using MCPTox or MSB to pick a model, a hyperparameter, a checkpoint, a seed or a
  threshold. Selection uses train_pool val only. The holdouts are scored once per run that
  is reported.
- Reporting the best seed or run and hiding the rest. Every run in the run plan is reported.
- Claiming generalization when length-binned AUROC says the signal is length.
- Letting MCPTox or MSB text shape the stream 5 templates.

## Success criterion (fixed before training)

The bar is **B1 TF-IDF**, not only length. A "stronger detector" has to beat the strongest
detector already reported. Length is beaten automatically by anything that clears B1 here:
at 5% FPR, length's TPR on MCPTox is 0.002.

**B2 beats B1** when, on **MCPTox** (LOSO holdout, 362 benign / 485 adversarial),
**every seed** of the selected configuration meets both conditions:

1. **TPR at a matched 5% FPR**: TPR at the threshold that gives 5% FPR *on MCPTox's own
   benign rows*. Its bootstrap 95% CI lower bound must be above B1's TPR at 5% FPR, measured
   the same way. B1 scores 0.190 today; recompute it with the same code when the numbers
   freeze, and use that value. This threshold is computed for **measurement only**: it ranks
   detectors at equal false-positive cost and is never used to select or deploy anything.
   5% rather than 1%, because 1% of 362 benign rows is 3 rows.
2. **Length-binned AUROC** (defined in stream 4): the bootstrap 95% CI lower bound is
   above 0.5, so the signal is not length.

The bootstrap resamples MCPTox benign and adversarial rows separately (as
`bootstrap_auroc_ci` does) and recomputes the 5% FPR threshold inside each replicate.

Also reported, but not part of the verdict:
- the **deployed** numbers: TPR and FPR at the threshold set at 1% FPR on train_pool val
  and applied unchanged, as in the harness today. MCPTox FPR here shows how far the val
  threshold drifts
- AUROC with CI, next to length's 0.774 and B1's 0.675
- every MSB number

If some seeds pass and some fail, the result is reported as **mixed**, with every seed
shown.

Also reported, but not part of the verdict:
- AUROC with CI next to length's 0.774
- FPR on MCPTox, compared with the 1% target (how far the val threshold drifts)
- every MSB number

MSB has 21 benign rows, so it is too small to decide anything: one false positive moves FPR
by 4.8%. It is reported with CIs and labeled as indicative.

## Decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Base model | `microsoft/deberta-v3-base` (184M parameters), with a classification head that outputs one logit per chunk |
| 2 | Where training runs | Google Colab, free tier (T4, 16 GB). University cluster access is not available |
| 3 | Where evaluation runs | Colab writes a **score file** per run (one score per row id, for val, test, MCPTox and MSB). `make evaluate DETECTOR=deberta SCORES=...` then runs the unchanged harness locally on those scores, so MLflow logging works through each member's local server as it does today. No model download is needed to evaluate (Claude's call, after the builder check) |
| 4 | Chunking | Windows of 256 tokens with a stride of 192 (64 tokens of overlap). A text with 256 tokens or fewer is one chunk. Window and stride may be tuned on val |
| 5 | Text score | Maximum of the chunk logits, passed through a sigmoid. Training uses the same max (multiple-instance learning), so training and inference score text the same way |
| 6 | Loss | Binary cross-entropy on the text-level max logit, plus `pairwise_weight` × margin ranking loss on BIPIA twin pairs (attacked max logit > clean max logit) |
| 7 | Model selection | Highest val TPR at 1% val FPR, ties broken by val AUROC, chosen per epoch and per configuration. Train_pool val only |
| 8 | Seeds | 3 per configuration in the run plan |
| 9 | Precision | fp16 mixed precision if the smoke test passes, otherwise fp32. DeBERTa-v3 is known to overflow in fp16 and the T4 has no bf16 |
| 10 | Model and score storage | Score files (small) for every reported run are DVC-tracked under `report/deberta_scores/<run_name>.parquet`. Only **one** model is DVC-tracked: seed 1 of the selected configuration, under `models/deberta/<run_name>/` (safetensors and tokenizer, about 370–740 MB), kept for AgentDojo later. The lead moves it from Drive and uploads it with `make push-large`; plain `dvc push` fails above about 50 MB on the lead's uplink (Claude's call) |
| 11 | Dependencies | A new `deberta` uv dependency group: `torch`, `transformers`, `sentencepiece` (needed by the DeBERTa-v3 tokenizer). The `perplexity` group stays as it is |
| 12 | Training-data groups | Add `group` to the harness's `COLUMNS` so `fit` can find BIPIA twins. Every BIPIA train_pool group holds exactly one clean and one attacked row (checked against the parquet). Eval sets are unaffected |
| 13 | Val set for selection and threshold | The full train_pool val split (BIPIA + PoisonedRAG, 801 benign rows), for **every** row of the run plan, including row 3 (trained on BIPIA only) and row 4. Synthetic val rows from stream 5 are used only for early stopping inside row 4, never for the threshold or for choosing between rows, so rows stay comparable (Claude's call) |

## Interface

`src/models/deberta.py`, matching `TfidfDetector` and `PerplexityDetector`:

```python
class DebertaDetector:
    name = "deberta"

    def __init__(self, model_name="microsoft/deberta-v3-base", chunk_tokens=256, chunk_stride=192,
                 pairwise_weight=0.0, lr=2e-5, epochs=3, batch_size=16, seed=7267,
                 fp16=True, device=None, checkpoint_dir=None): ...

    def fit(self, texts, labels, groups=None, val=None):
        """Train on train_pool train. `groups` enables the pairwise term (needed when
        pairwise_weight > 0). `val` = (texts, labels) for per-epoch selection (decision 7).
        With checkpoint_dir set, saves every epoch and resumes from the last save."""

    def predict_proba(self, texts):
        """One score per text: sigmoid of the max chunk logit. Ordering is all the harness uses."""

    def save(self, path): ...

    @classmethod
    def load(cls, path, device=None): ...
```

`src/models/train.py` gets a `deberta` branch in `fit_detector` that returns a
`PrecomputedScores` detector: it reads a score file (`--scores report/deberta_scores/<run_name>.parquet`)
and its `predict_proba` looks scores up by row. It fails loudly if any requested row has no
score. Look-up by `id` is preferred, which means adding `id` to the harness's `COLUMNS`. `make
evaluate DETECTOR=deberta SCORES=...` passes it through and needs no `deberta` group locally.
The score file carries the training config, seed and git commit as metadata, and the MLflow
run logs them as params. `DebertaDetector.load` is still needed in Colab to write score files,
and later for AgentDojo.

## Work streams

Each stream gets its own branch and pull request. The lead reviews anything that touches the
interface above.

| # | Stream | Starts | Depends on | Done when |
|---|---|---|---|---|
| 1 | **Core detector** (lead): chunking, max-pool, BCE training loop, save/load, epoch checkpoints with resume | Nov 9 | — | Unit tests pass on a tiny model (for example `hf-internal-testing/tiny-random-DebertaV2Model`) on CPU: chunk boundaries, max-pool, save/load round trip, resume after a simulated interruption |
| 2 | **Colab and data**: notebook that clones the repo, `uv sync --group deberta`, reads R2 keys from Colab Secrets, `dvc pull`s the dataset, calls `DebertaDetector.fit`, writes checkpoints and score files to Google Drive | Nov 9 (against stream 1's interface) | 1 to run for real | A 200-row fp16 smoke run finishes without NaN loss (otherwise switch to fp32 and record why); a full run survives a forced disconnect and resumes |
| 3 | **Pairwise loss (B)**: pair BIPIA rows by `group`, add the margin term weighted by `pairwise_weight` | ~Nov 12 | 1's training loop | A test shows the term is zero when each attacked twin outscores its clean twin by the margin, and non-zero otherwise |
| 4 | **New metrics**: `length_binned_auroc` and `tpr_at_fpr` (TPR at a matched FPR measured on the eval set itself, with the bootstrap described under the success criterion) in `src/eval/metrics.py`, added as report columns | Nov 9 (check against TF-IDF and length) | — | The length baseline scores about 0.5 length-binned AUROC on MCPTox (sanity check); B1 rows regenerated with the new columns |
| 5 | **Stretch: synthetic tool descriptions (C)** | Nov 9, separate branch | — | See below. Cut first if behind |

### Stream 4: length-binned AUROC

Within one eval set and slice: split rows into length quartiles by character length,
computed on that slice. Compute AUROC inside each bin that holds both labels, then take the
mean weighted by the bin's row count. A score that only encodes length gets about 0.5,
because length barely varies inside a bin. Bootstrap the CI the same way as
`bootstrap_auroc_ci`, resampling benign and adversarial rows separately. Below 40 rows, or
with fewer than two bins that hold both labels, report NaN: MSB per-surface slices will
usually be NaN, and that is expected.

### Stream 5: synthetic tool descriptions (stretch)

The training pool has no `tool_description` rows, and that is the surface MCPTox tests.
Stream 5 adds some without touching the holdouts:

- **Benign:** tool definitions that ship in the `agentdojo==0.1.35` package, rendered as
  `Tool: {name}\nDescription: {description}` (the data card's tool-description format).
- **Adversarial:** the same descriptions with an injected instruction, from templates written
  by someone who **has not opened MCPTox or MSB data**. Write templates from the attack
  objectives in the coding matrix, not from benchmark examples.
- Written to a separate file, `data/processed/synthetic_tools.parquet`, as its own DVC stage,
  with `role = train_pool` and a grouped train/val split. `dataset.parquet` does not change.
- Used only in an extra configuration (run plan row 4), never mixed silently into the others.

## Run plan

Reported runs: 3 seeds each, selection per decision 7 and 13, with score files for test,
MCPTox and MSB. **Sweeps** (row 2's `pairwise_weight`, any window/stride tuning) use 1 seed
and write **val scores only**. They are never scored on the holdouts, and their val numbers
go in an appendix table, so "every run is reported" still holds (Claude's call).

| Row | Configuration | Training data | `pairwise_weight` | Purpose |
|---|---|---|---|---|
| 1 | Plain (A) | BIPIA + PoisonedRAG | 0 | Floor |
| 2 | (A) + (B) | BIPIA + PoisonedRAG | chosen on val from {0.5, 1.0} | Does pairing help? |
| 3 | Ablation | BIPIA only | best of rows 1–2 by val | Does PoisonedRAG's reversed length cue hurt? |
| 4 | Stretch | best of rows 1–3 + synthetic tools | as row 3 | Does seeing the surface help? |

The **selected configuration** for the success criterion is the row with the highest mean
val TPR at 1% FPR across seeds, chosen before holdouts are scored. Rows 1–3 are all
reported whichever wins. A rough estimate on a T4 is 20–40 minutes per run: 9 reported
runs plus 2–4 one-seed sweeps for rows 1–3, about 6–9 hours. The builder confirms this with the first full run.

## Known risks

- **Max-pool alone does not remove the length cue.** MCPTox descriptions and PoisonedRAG
  passages are short, so they fit in one chunk, and a transformer can still read token
  count inside a chunk. (A) mainly fixes truncation and dilution for long BIPIA documents.
  The real length tests are the length-binned AUROC and the BIPIA-only ablation (row 3).
  Taking the max over more chunks also gives longer texts more chances to score high, a
  small upward length bias. The same metric watches for it.
- **BIPIA twins differ in length.** The attacked twin is always longer, so the pairwise term
  can reward length as well as content. Chunking limits this to the last chunk. Watch the
  length-binned AUROC of row 2 against row 1.
- **Colab disconnects and quotas.** Mitigated by epoch checkpoints on Drive and resume
  (stream 2). If the free tier runs out, one paid month of Colab is the fallback.
- **R2 keys in Colab.** One shared Colab token (decided by the lead), set up as described
  under "Colab R2 token".
- **fp16 NaNs.** Decision 9 smoke test, with an fp32 fallback.

## Colab R2 token

One token is shared by everyone who trains in Colab, separate from each person's own
token. Leaking it exposes the bucket's contents, including the NewsQA/CNN-derived text that
lives only in the private R2 remote under its terms of use. So:

- **Scope:** the R2 account owner creates an R2 API token with **Object Read** permission
  only, restricted to the project bucket only.
- **Expiry:** Dec 14, 2026, a week after the write-up deadline. Revoke it earlier if
  anything looks wrong.
- **Check:** `make r2-check` must pass with it, and `make r2-check-write` must **fail**.
  Run both once before handing it out.
- **Distribution:** the access key and secret go to each Colab user through a password
  manager or a private DM, never in git, an issue, a shared doc or a group channel. Colab
  Secrets belong to each person's Google account and are not shared with a notebook, so
  every user adds them to their own Colab Secrets (`R2_ACCESS_KEY_ID`,
  `R2_SECRET_ACCESS_KEY`).
- **In the notebook:** read them with `google.colab.userdata.get` into the
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` environment variables that DVC's S3 remote
  reads. Never print them, never pass them on a `!` shell command line, and never write
  `.dvc/config.local`.
- **Committed notebook:** commit it with outputs cleared, so no cell output can carry a key
  or a path from someone's Drive.
- **If it leaks:** revoke it in the Cloudflare dashboard, create a new one, and
  redistribute. Nothing in the repo changes.

## Left to the builder

- The optimizer, learning-rate schedule and warmup, and the margin value for the ranking loss.
- How chunks are batched (padding and bucketing), and a cap on chunks per text if memory
  needs one. The cap must not drop the end of a text silently, because BIPIA inserts attacks
  at the end.
- Whether to save weights in fp16 to halve the model size.
- Notebook layout, as long as logic stays in `src/models/deberta.py`.

## Write-up deliverables (Dec 1–7)

- One table per eval set: length, B0, B1, B2 (each run row), with AUROC, length-binned AUROC,
  TPR and FPR, CIs included.
- The verdict under the success criterion, stated as pass, fail or mixed, with seeds shown.
- `report/baselines/b2_deberta_<row>.csv`: the existing baseline CSV columns plus a `seed`
  column, one block of rows per seed. B0 and B1 files are unchanged (Claude's call).
- README status updated.
