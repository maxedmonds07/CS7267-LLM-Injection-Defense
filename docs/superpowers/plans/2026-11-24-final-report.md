# Final Report (Phase 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A NeurIPS-format final paper, `report/final/report.pdf`, built by `make report-final`.

**Architecture:** Markdown source rendered by pandoc's default LaTeX template, with
`neurips_2025.sty` loaded through `-H header.tex`; citeproc with `ieee.csl`. A small checker
(`scripts/check_final_report.py`) enforces the rubric mechanically: main-text page count,
checklist completeness, no placeholders, every citation resolved.

**Tech Stack:** pandoc + tectonic, citeproc, uv/pytest, pdftotext (poppler).

**Spec:** `docs/superpowers/specs/2026-11-24-final-report-design.md`

## Global Constraints
- Main text (title page through Conclusion) ≤ 12 pages, ≥ 8.
- `\usepackage[preprint,nonatbib]{neurips_2025}`; footer notice = "CS 7267 Machine Learning, Kennesaw State University. Final project report, December 2026."
- Authors: Carter Corbin, Max Edmonds, Mohammed Adil Ahmed, Oluwamayowa Adewummi, Adedapo Odeyemi.
- Repo URL https://github.com/maxedmonds07/CS7267-LLM-Injection-Defense on the title page and in the checklist.
- Every number copied from a committed file under `report/`; no new experiments.
- No `TODO`, `TBD`, `@@`, `??` or `[TODO]` in source or PDF. No Claude attribution in commits.

## Review Focus
- An unresolved citation key renders as `(?)` / bold key silently: checker greps the PDF text.
- The NeurIPS checklist macros `\answerTODO` left in: checker counts 16 answered items.
- Main text spills past 12 pages after a figure moves: checker finds the "References" page.
- A number drifts from its source file during condensing: Task 4 cross-checks each headline number against the CSVs.
- Pandoc's default template clashing with the style (geometry, hyperref, abstract env): Task 1 builds a skeleton before any writing.

---

### Task 1: Style, build target and checker

**Files:**
- Create: `report/final/neurips_2025.sty` (official, unmodified), `report/final/header.tex`, `report/final/report.md` (skeleton), `report/final/checklist.md`, `report/final/extra.bib`, copy `ieee.csl`, `repo-links.lua` from phase5
- Create: `scripts/check_final_report.py`, `tests/test_final_report.py`
- Modify: `Makefile` (targets `report-final`, `check-final`; `.PHONY`)

**Interfaces:**
- Produces: `check(pdf: Path, sources: list[Path]) -> list[str]` (problems; empty = pass);
  `main_text_pages(pdf_text_pages: list[str]) -> int` = index of the first page whose text starts a "References" heading;
  `checklist_answers(text: str) -> int` = count of `[Yes]|[No]|[NA]` answers.

- [ ] Tests: `test_main_text_pages_finds_references`, `test_checklist_counts_answers` (16 answers → 16, a `[TODO]` → reported), `test_placeholders_flagged` (`TODO`, `??`, `(?)` in text → problem).
- [ ] Implement checker (pdftotext per page via `subprocess`), make tests pass.
- [ ] Skeleton report with headings + one citation; `make report-final check-final` builds and passes except page minimum.
- [ ] Commit.

### Task 2: Main text §1–4 (intro, related work, data, detectors/protocol)
Sources: `report/phase2/report.md`, `report/phase4/report.md`, `DATA_CARD.md`, `literature-review/references.bib`.
- [ ] Write; build; commit.

### Task 3: Main text §5–8 (results, why they fail, fairness/cost/harms, conclusion)
Sources: `report/phase4/report.md`, `report/phase5/report.md`, `report/baselines/*.csv`, `report/phase5/*`, `report/red_team/*/summary.csv`. Figures reused: `report/phase4/fig_auroc.png`, `report/phase5/shap/b1_optuna_global.png`.
- [ ] Write; build; `check-final` page count 8–12; commit.

### Task 4: Checklist, appendix, verification
- [ ] `checklist.md`: the 16 NeurIPS questions verbatim, answers with `\answerYes{}` etc. and justifications pointing to sections.
- [ ] Appendix: extra tables + model card (piped as in Phase 5).
- [ ] Cross-check every headline number against its CSV; `make check-final` passes; full `uv run pytest`.
- [ ] Commit; ask the user before push + PR.
