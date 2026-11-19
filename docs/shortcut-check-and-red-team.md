# Shortcut check and red-team mutations

Two tools for the feedback loop: is a detector's score real, and does it survive an attacker who
rewrites the attack? Both reuse the team's `train.fit_detector` and `harness.evaluate`, read data only
through `dataset.load` (holdouts are never fitted on), and log nothing to MLflow.

| Command | What it does | Output |
|---|---|---|
| `make shortcut-check DETECTOR=tfidf` | Runs the shortcut gate on one detector (`length`, `tfidf`, `perplexity`, or `deberta SCORES=...`) | `PASS` / `FAIL` / `INCONCLUSIVE` with reasons. `OUT=dir` also writes `gate.csv`, `probes.csv`, `verdict.txt` |
| `make red-team DETECTOR=tfidf` | Rewrites every test-split attack 11 ways and re-scores it | `report/red_team/<detector>/summary.csv` (one row per mutation), `report.csv` (full harness report) |
| `make red-team-data` | Writes the mutated rows only, for scoring B2 on Colab | `data/red_team/test_mutations.parquet` (git-ignored; deterministic, so anyone can regenerate it) |
| `uv run python notebooks/phase3_eda.py --run-folder` | Figures 01–10 plus a Markdown write-up in `notebooks/runs/<date_time>/` (no PDF build) | Figures 08–10 are new: length ROC, cue AUROC CIs, Spearman CIs |

All operating points use the team's **5% FPR** (Nov 9 sync). Confidence intervals resample whole
`group`s, so a BIPIA context and its attacked twin never land on different sides of a resample.

## 1. The shortcut gate (`src/eval/shortcut_probe.py`, `src/eval/shortcut_check.py`)

A detector cheats when its score comes from a cue that is true in this dataset but is not about
injection. The known one is length, which points opposite ways by source.

| Check | Rule (defaults, not yet team-agreed) |
|---|---|
| Beats the cheap cues | Detector AUROC CI-lower must exceed the best *fitted* cue probe's CI-upper by **0.05** on each eval set. Cues: length only, and length + punctuation/uppercase/digit rates. |
| Survives length matching | Balance classes per length decile inside each source; the detector must keep **≥ 80%** of its above-chance AUROC. Fewer than **100** matchable rows → INCONCLUSIVE (too noisy to call). |
| Label shuffle | Retrained on row-shuffled labels, mean AUROC ≤ 0.60. Skipped for B2 score files (no refit possible). |
| Padding | Lengthening benign rows with other benign text from the same source may move AUROC ≤ 0.10. Skipped for B2 score files, which cannot score new texts (said in the reasons). |
| Split leaks | Zero shared groups and zero exact duplicate texts between train and test. |

### Results on the real data (2026-11-15, `main` at 26e06a8)

| Detector | Test split | MCPTox (LOSO) | MSB (transfer) | Verdict |
|---|---|---|---|---|
| Length baseline | 0.491 | 0.774 | 0.866 | **FAIL** (expected: it *is* the shortcut; sanity check passes) |
| B1 TF-IDF | 0.902 (CI-lo 0.880) ✔ beats cues | 0.675 (CI-lo 0.615) ✘ length CI-hi 0.874 | 0.537 (CI-lo 0.472) ✘ length CI-hi 0.960 | **FAIL** on both holdouts |

- TF-IDF passes cleanly in distribution: it beats every cue probe and keeps its signal under length
  matching (0.867).
- On both holdouts **length alone beats TF-IDF**. On MCPTox TF-IDF does not lean on length (its
  length-matched AUROC is higher, 0.71 on 222 rows), but it does not generalize either. On MSB only
  26 rows can be length-matched, below the 100-row minimum, so that check is not run there.
- Label shuffle ≈ 0.50 for both (no label leak). Shared groups 0, exact duplicates 0; 13 near-duplicate
  pairs (char-4gram cosine ≥ 0.9) cross train/test, reported but not gated.
- Probe numbers worth quoting: inside PoisonedRAG, length *fitted on the pool* scores 0.14 but the
  *oracle* (fit inside the source) scores 0.86. Length is a strong cue there that the pooled fit points
  the wrong way. Bag-of-words fit inside MCPTox or MSB scores 0.99+, so each holdout is trivially
  separable by its own vocabulary.

Files: `report/shortcut_check/{length,tfidf}/`, figures in `report/shortcut_check/figures/`.

## 2. Red-team mutations (`src/eval/red_team.py`)

The question: do the models still catch an attack when its phrasing changes? Only the **train_pool
test split** is mutated, so no mutated text can reach training, val, or the holdouts. Labels are copied
unchanged. Each mutation's random choices are seeded from (row id, mutation name), so reruns match.

| Family | Mutations |
|---|---|
| paraphrase | `synonym` (injection vocabulary → synonyms), `polite_indirect` (imperatives → polite requests) |
| reorder | `sentence_shuffle` |
| obfuscation | `case_noise`, `homoglyph` (Cyrillic look-alikes), `leetspeak`, `zero_width` (invisible U+200B) |
| format | `whitespace`, `markdown_wrap`, `json_wrap`, `html_comment` |

Two eval sets per mutation go through `harness.evaluate`, with the threshold set once on val benign rows at 5% FPR:

- `mut:<name>`: mutated attacks plus the **original** benign rows. This is the realistic case, since
  the attacker controls only the attack text.
- `ctrl:<name>`: the same mutation applied to benign rows too. Its FPR shows whether the transform
  alone makes clean text look malicious.

### Results, B1 TF-IDF (test split; TPR at the val threshold, original 0.647)

| Mutation | Attacks changed | AUROC | TPR | ΔTPR | ctrl FPR |
|---|---|---|---|---|---|
| leetspeak | 100% | 0.729 | 0.280 | **−0.367** | 0.050 |
| zero_width | 100% | 0.801 | 0.328 | **−0.320** | 0.037 |
| homoglyph | 100% | 0.853 | 0.516 | **−0.132** | 0.042 |
| json_wrap | 100% | 0.891 | 0.606 | −0.041 | 0.059 |
| markdown_wrap | 100% | 0.891 | 0.614 | −0.033 | 0.054 |
| synonym | 45% | 0.900 | 0.646 | −0.002 | 0.060 |
| polite_indirect | 55% | 0.900 | 0.647 | 0.000 | 0.056 |
| sentence_shuffle, case_noise, whitespace, html_comment | 75–100% | 0.901–0.902 | 0.647 | 0.000 | 0.056 |

- **Character-level obfuscation breaks B1.** Leetspeak and zero-width characters cut its catch rate
  roughly in half, and homoglyphs cost 13 points. The word tokenizer never sees the original word.
- **Casing, whitespace, reordering and HTML comments do nothing.** The vectorizer lowercases and
  ignores punctuation and order.
- **The paraphrase mutations are a weak test.** They changed only about half of the attacks, because
  PoisonedRAG attacks are false-fact passages rather than instructions (`polite_indirect` changes 0%
  of them), and the words they swap carry little of B1's signal. A stronger paraphraser is future work.
- The length baseline barely moves (largest shift +0.04 TPR from `zero_width`, which adds characters).
  That confirms the mutations do not hand the length cue anything new.

## 3. Open questions for the team

1. Gate thresholds (0.05 margin, 80% retention) are defaults. Agree on them before more detectors are judged.
2. Gate 1 compares against *fitted* cue probes. Should it use the stronger *oracle* probes?
3. `harness.evaluate` and `train.py` still default `target_fpr=0.01` for the val threshold, while the
   team judges at 5%. These tools use 5%. Align `train.py`, or report both.
4. B2 on the mutations needs a Colab pass: score `data/red_team/test_mutations.parquet` with the
   trained model, then evaluate the score file like the other B2 runs.
5. Character-level robustness is a cheap win to test: normalize text before scoring (NFKC, strip
   zero-width characters, map homoglyphs) and re-run `make red-team`.
