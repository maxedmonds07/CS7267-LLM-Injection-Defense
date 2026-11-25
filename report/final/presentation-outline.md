# Presentation outline: Detectors Learn a Voice, Not a Behavior

Outline of the final presentation deck (a Claude Slides artifact; ask the team for the link).
About 13.5 minutes plus Q&A: 16 main slides and 3 backup slides. Each person presents the slides
that use the method they built (paper §9). Every number comes from `report/final/report.md`;
figure paths are relative to the repo root.

| # | Slide | Speaker | Time |
|--:|:------|:--------|-----:|
| 1 | Title | Max Edmonds | 0:20 |
| 2 | The threat: LLM agents read text no user wrote | Max Edmonds | 1:00 |
| 3 | The question: does a detector transfer to a surface it never saw? | Adedapo Odeyemi | 0:40 |
| 4 | Data: 11,005 segments, with MCP held out whole | Max Edmonds | 1:00 |
| 5 | Text length alone is strong off-surface | Carter Corbin | 0:45 |
| 6 | Four detectors, one harness | Mohammed Adil Ahmed, then Oluwamayowa Adewummi | 1:00 |
| 7 | Fit in distribution, fail off surface (AUROC figure) | Oluwamayowa Adewummi | 1:00 |
| 8 | On MCP tool descriptions, the order reverses (results table) | Max Edmonds | 1:00 |
| 9 | MCP attacks are written as tool procedure | Adedapo Odeyemi | 1:00 |
| 10 | SHAP: the TF-IDF model learned a voice | Adedapo Odeyemi | 1:00 |
| 11 | Saliency: DeBERTa does not respond to the surface | Adedapo Odeyemi | 0:50 |
| 12 | Reading the task does not help (B4) | Mohammed Adil Ahmed | 0:40 |
| 13 | Deployment: brittle, noisy on long docs, and costly | Carter Corbin | 1:15 |
| 14 | Harms and mitigations | Mohammed Adil Ahmed | 0:45 |
| 15 | Team contributions: who built what | Oluwamayowa Adewummi | 0:30 |
| 16 | Conclusion | Mohammed Adil Ahmed | 0:45 |
| 17–19 | Backup: limitations, per-seed verdict, local SHAP | anyone | — |

Time per speaker: Max Edmonds 3:20, Adedapo Odeyemi 3:30, Carter Corbin 2:00, Mohammed Adil
Ahmed 2:45, Oluwamayowa Adewummi 1:55.

---

## 1. Title — Max Edmonds

- *Detectors Learn a Voice, Not a Behavior*: detecting prompt injection across RAG and MCP surfaces.
- The five authors; Kennesaw State University; repo link.

## 2. The threat — Max Edmonds

- Three surfaces: RAG passages and tool outputs (training), MCP tool descriptions (held out).
- Indirect prompt injection (Greshake et al.); poisoned MCP tool descriptions reach 72.8% attack
  success on o1-mini (MCPTox).

## 3. The question — Adedapo Odeyemi (literature review)

- Trained on RAG passages and tool outputs, does a detector transfer to MCP tool descriptions,
  a surface it never saw? If not, why not?
- In-benchmark testing overstates AUROC by 8.0–16.5 points (Fomin et al.); concurrent work holds
  out datasets, not a surface.

## 4. Data — Max Edmonds (corpus profile)

- Corpus table: PoisonedRAG 2,955 / 1,500 and BIPIA 2,806 / 2,806 (training pool); MCPTox
  362 / 485 (main holdout); MSB 21 / 70 (transfer holdout).
- Split 70/15/15 by group; a loader guard keeps holdouts out of fitting, selection and thresholds.

## 5. The length baseline — Carter Corbin (shortcut gate, probe figures)

- Length-only AUROC: 0.491 in distribution, **0.774 on MCPTox**, 0.866 on MSB.
- Length flips direction between sources; every detector is reported next to it.

## 6. Detectors — Mohammed Adil Ahmed (B0, B1, B4), then Oluwamayowa Adewummi (family-grouped CV)

- B0 perplexity (frozen DistilGPT2); B1 TF-IDF + logistic regression; B1-Optuna; B2 DeBERTa-v3;
  B4, DeBERTa that also reads the user's task (slide 12).
- Family-grouped folds hold out each attack family whole.
- Threshold at 1% FPR on validation; group-bootstrap AUROC intervals; Wilson FPR intervals.

## 7. AUROC figure — Oluwamayowa Adewummi (intervals, length-binned AUROC)

- **Visual:** `report/phase4/fig_auroc.png`.
- Test: 0.90–0.99 for trained detectors; MCPTox: nothing clears length, DeBERTa below chance.

## 8. Results table — Max Edmonds (B2 runs and verdict)

- MCPTox AUROC: length 0.774, B1-Optuna 0.767, B1 0.675, B0 0.430, B2 0.346 ± 0.128.
- B2 fails its pre-registered criterion on every seed; dropping PoisonedRAG makes MCPTox worse
  (0.346 → 0.199).

## 9. Tool procedure — Adedapo Odeyemi (error taxonomy)

- "Before using X, you must first call Y": 50.7% of MCPTox attacks vs 1.0% of test attacks.
- 59% of B1-Optuna's misses and 51% of DeBERTa's.

## 10. SHAP — Adedapo Odeyemi

- **Visual:** `report/phase5/shap/b1_optuna_global.png`.
- "your" in 44.5% of test attacks vs 2.7% of MCP attacks; "in your" adds +2.4 log-odds to a
  benign GitHub tool description.

## 11. Saliency — Adedapo Odeyemi (DeBERTa model, integrated gradients)

- **Visual:** `report/phase5/saliency/heatmaps.png`.
- Same voice on test attacks; MCP attacks and benign rows share a median logit of −8.61;
  98–100% of MCP rows score below 0.01.

## 12. Reading the task does not help (B4) — Mohammed Adil Ahmed

- B4 pairs each text with the user's task (the question, or the MCP server's task catalogue);
  four configurations, three seeds each, on the 672 MCPTox and 76 MSB rows that have a task.
- Adding the task lifts MCPTox AUROC from 0.339 to 0.506, about chance; TPR at 5% FPR stays
  below 0.05. Lexical relevance features lower it to 0.454. Length alone: 0.760 on these rows.
- Takeaway: context makes DeBERTa less wrong, not a detector; the gap is still the data.

## 13. Deployment — Carter Corbin (red-team mutation generator)

- Leetspeak cuts DeBERTa recall from 0.984 to 0.110; 14.3% FPR on the longest benign tool docs;
  B1-Optuna about 100× cheaper (0.58 vs 56.2 ms per row).

## 14. Harms and mitigations — Mohammed Adil Ahmed

- Four of the paper's six harms, each with evidence and a mitigation.
- Measured: NFKC plus zero-width stripping repairs zero-width attacks (0.483 → 0.725), not
  homoglyphs (0.614) or leetspeak (0.213).

## 15. Team contributions — Oluwamayowa Adewummi

- Table of the method each member developed and coded, with phases; paper §9 has the files and PRs.

## 16. Conclusion — Mohammed Adil Ahmed

- Detectors learned a voice, not a behavior; more capacity made it worse, and reading the
  user's task did not fix it.
- Deploy B1-Optuna only with guardrails; next step is data, not a bigger model.

## Backup slides

- **Limitations:** small holdouts; B2 seed variance (0.247–0.491) and single-seed analyses;
  frozen detectors; B4 uses server task catalogues, not observed requests, and fewer holdout
  rows; detection, not end-to-end attack success.
- **Per-seed verdict:** TPR at 5% FPR 0.208 / 0.039 / 0.041 against B1's 0.225.
- **Local SHAP:** `report/phase5/shap/b1_optuna_local.png`.
