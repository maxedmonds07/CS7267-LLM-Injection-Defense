# Final report (Phase 6) — design

Approved 2026-11-24.

## Intent
The course's final deliverable: one paper covering the whole project (data, detectors,
evaluation, interpretability, ethics). Rubric: 8–12 pages, NeurIPS/ICML format, fully cited
(APA or IEEE), reproducibility checklist completed, public GitHub repo linked.

## Decisions (user)
- **Format:** NeurIPS style through pandoc (Markdown source, as in Phases 2–5), not native LaTeX
  and not ICML.
- **Page limit:** 8–12 pages of *main text*; references, the checklist and appendices follow and
  do not count (NeurIPS convention).
- **Scope:** paper only; the public presentation is a later, separate step.
- **Authors:** the five names on the earlier reports (Corbin, Edmonds, Ahmed, Adewummi, Odeyemi).

## Decisions (mine, stated in the design)
- A condensed rewrite of Phases 2–5, not a concatenation. No new experiments; every number is
  copied from a committed result file.
- Official `neurips_2025.sty` (the 2026 style is not published at the usual URL), `preprint`
  mode with author names and no line numbers; the footer notice is replaced with the course line
  so the paper does not claim to be a NeurIPS submission or publication.
- IEEE citations via the existing `.bib` files and `ieee.csl`.
- The official 16-question NeurIPS paper checklist, answered honestly with section pointers.
- Appendix: extra tables and the model card.
- Repo link in the title-page footnote and in the checklist.
- Branch `phase6/final-report` off `phase5/evaluation-interpretability-ethics`; ask before push/PR.
- Due date assumed 2026-12-07.

## Main-text outline (~10 pages, cap 12)
1 Introduction (1) · 2 Related work (1) · 3 Data (1) · 4 Detectors and protocol (1) ·
5 Results: held-out, shift, robustness (2) · 6 Why they fail: taxonomy, SHAP, saliency (1.5) ·
7 Fairness, cost, harms, mitigations (1) · 8 Limitations and conclusion (0.5)

## Anti-goals
- Any number not traceable to a committed file; any TODO/placeholder in the PDF.
- A footer implying acceptance at NeurIPS.
- Main text over 12 pages.
