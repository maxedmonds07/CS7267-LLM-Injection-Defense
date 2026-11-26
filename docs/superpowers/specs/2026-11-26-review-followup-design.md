# Review follow-up: design

Source: `docs/Review Detectors Learn a Voice, Not a Behavior (deck v2).md` (review of deck v2,
`report/final/report.md` and `report/phase5/scores.parquet` at commit `c24cd11`).
Deadline: presentation and paper on 2026-11-30.

## Intent

Make every claim in the paper and deck match the evidence, and turn the review's two untested
claims ("B2 is noise", "the bottleneck is data") into measured results. Optimize for time: wording
first, then analyses that read existing scores, then cheap CPU experiments, then the deck.

## Goals

- Paper (`report/final/report.md`) carries every fix in the review's consistency table and the
  three new limitations.
- Reproducible scripts for review experiments 1, 2, 3, 4 and 7, plus the paired server bootstrap.
- A Colab-ready Prompt Guard 2 scoring script (experiment 5) that a teammate runs on a GPU.
- Deck v2 (Claude Slides artifact `https://claude.ai/artifact/PRfbv6Dr8tuHKvFMebN2JT`) carries the
  review's slide fixes with the final numbers; `report/final/presentation-outline.md` matches it.

## Non-goals

- Making length-binned AUROC the headline metric (decided: wording only; add the binned column
  where missing, qualify "strongest" with "by raw AUROC", explain the perplexity binned result).
- Demo script, second-coder sheet (team-side; not prepared here).
- A controlled capacity experiment (wording fix only).
- Running Prompt Guard 2 ourselves.

## Anti-goals

- Any new result presented as part of the pre-registered protocol. Every MCPTox-reading experiment
  is framed as a follow-up analysis outside it, like paper §6.
- Row-level sampling of MCPTox in experiments; draws and folds are by server (`subset`), so
  per-server phrasing cannot leak.

## Constraints

- One branch + PR per unit of work, from `origin/main`; no AI trailers; the user pushes and opens PRs.
- `uv` for Python, `make` targets for entry points; run with
  `PYTHONPATH=src/data:src/eval:src/models`.
- Data is local (`data/processed/dataset.parquet`); scores in `report/phase5/scores.parquet`.

## Approaches

1. **Phase-5 pattern (chosen):** scripts in `src/eval/`, a `make` target each, outputs under
   `report/followup/`, tests with the synthetic fixtures in `tests/synth.py`. Reproducible from
   the paper's appendix, matches the codebase.
2. Notebook: fastest to write, not reproducible from the appendix.
3. DVC stages: most reproducible, pipeline overhead not worth it four days out.

## Design (PRs in time order)

1. **Paper wording** (no code). B2 "noise" → "ranks MCP rows by inverse length"; capacity claim →
   "the highest in-distribution score did not predict transfer"; 59% → B1-Optuna misses 75% of
   procedural-prerequisite attacks (185/246) vs 55% of the rest (131/239); relevance features
   "lower" → "do not help"; "Fomin et al." → "Fomin" with the activation-classifier clause;
   deployment scoped "not on MCP"; perplexity binned AUROC (0.739) acknowledged; limitations add
   three-template holdout, no off-the-shelf/LLM-judge baseline, regexes written after reading
   holdout errors; robustness recalls labelled with model and 5% threshold; MSB 0.866 marked
   indicative; MCPTox o1-mini 72.8% evidence and the ~0% transplanted-payload finding added.
2. **`src/eval/followup_length.py`** (experiments 3, 4, paired test), reads `scores.parquet` and
   text lengths: B2 Spearman with length by seed (all MCPTox, benign only, in-distribution benign
   by source), B2 AUROC on MCPTox restricted to benign rows ≥ shortest attack; per-template
   attacks / length AUROC / B1-Optuna AUROC / recall at deployed threshold; paired server
   bootstrap B1-Optuna − length and rank-average of the two. Outputs CSV/JSON + one figure
   (B2 score vs length by label). Paper numbers updated from it.
3. **`src/eval/followup_transfer.py`** (experiments 1, 2, 7), B1-Optuna's committed settings:
   few-shot curve k ∈ {0,1,2,5,10,20} MCPTox servers added to training, 10 random draws, tested on
   the remaining servers; leave-one-server-out and leave-one-template-out inside MCPTox; MSB→MCPTox.
   Each reports raw AUROC, length-binned AUROC and recall at 1% FPR (threshold from the training
   side's benign rows). Results and a figure go into a new follow-up subsection of the paper.
4. **`scripts/prompt_guard_colab.py`** (experiment 5): scores MCPTox, MSB and test rows with
   `meta-llama/Llama-Prompt-Guard-2-86M` (gated; HF token from the environment) and writes a
   parquet in the `report/deberta_scores` layout so `PrecomputedScores` can read it.
5. **Deck**: apply slide fixes 1–7 and the consistency-table rows to the Slides artifact with the
   final numbers; update `presentation-outline.md` to deck v2.

## Left to the builder

Figure styling, CSV column names, number of bootstrap resamples (default 1000 as in `metrics.py`),
exact paper placement of the follow-up subsection.
