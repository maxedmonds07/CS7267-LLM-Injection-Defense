"""Tests for the final-report rubric checker (scripts/check_final_report.py)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_final_report", ROOT / "scripts" / "check_final_report.py")
cfr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfr)


def test_main_text_pages_finds_references():
    pages = ["Title\n1 Introduction", "body", "8 Conclusion\nend", "References\n[1] A. Author"]
    assert cfr.main_text_pages(pages) == 3


def test_main_text_pages_references_mid_page_counts_that_page():
    pages = ["Title", "Conclusion text\nReferences\n[1] A"]
    assert cfr.main_text_pages(pages) == 2


def test_main_text_pages_ignores_word_in_running_text():
    pages = ["see the references below", "References\n[1] A"]
    assert cfr.main_text_pages(pages) == 1


def test_checklist_counts_answers():
    text = "\n".join(f"Answer: [Yes] item {i}" for i in range(15)) + "\nAnswer: [NA] last"
    assert cfr.checklist_answers(text) == 16


def test_placeholders_flagged():
    for bad in ("TODO here", "see Table ??", "as shown (?)", "Answer: [TODO]", "TBD"):
        assert cfr.placeholders(bad), bad
    assert not cfr.placeholders("All done. TODOS is not a word we use; total 12.")


def test_check_reports_page_and_checklist_problems():
    pages = ["Title", "References\n[1]", "Answer: [Yes]"]
    problems = cfr.check_pages(pages, min_pages=8, max_pages=12, checklist_items=16)
    assert any("main text" in p for p in problems)
    assert any("checklist" in p for p in problems)
