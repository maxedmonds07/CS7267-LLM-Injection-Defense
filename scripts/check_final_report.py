"""Check the final report PDF against the mechanical parts of the rubric.

    uv run python scripts/check_final_report.py report/final/report.pdf

Checks: the main text (title page through the page where "References" starts) is 8-12 pages;
the NeurIPS checklist has 16 answers; no placeholders or unresolved references remain; the
public repository is linked. Needs pdftotext (poppler). Exit status 1 lists the problems.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_URL = "github.com/maxedmonds07/CS7267-LLM-Injection-Defense"
REFERENCES = re.compile(r"^\s*(\d+\s+)?References\s*$", re.MULTILINE)
ANSWER = re.compile(r"\[(Yes|No|NA)\]")
# Unresolved citations render as the bare key plus "?" (e.g. "zou2025poisonedrag?").
PLACEHOLDER = re.compile(r"\bTODO\b|\bTBD\b|\?\?|\(\?\)|@@|\b[a-z]+\d{4}[a-z]*\?")


def main_text_pages(pages: list[str]) -> int:
    """1-based number of the last main-text page (0 if no References heading): the page with the
    References heading, or the page before it when the heading is the page's first line."""
    for number, text in enumerate(pages, start=1):
        if match := REFERENCES.search(text):
            return number - 1 if not text[:match.start()].strip() else number
    return 0


def checklist_answers(text: str) -> int:
    return len(ANSWER.findall(text))


def placeholders(text: str) -> list[str]:
    return sorted(set(PLACEHOLDER.findall(text)))


def check_pages(pages: list[str], min_pages=8, max_pages=12, checklist_items=16) -> list[str]:
    problems = []
    main = main_text_pages(pages)
    # The References heading usually starts partway down the last main-text page.
    if not main:
        problems.append("main text: no References heading found")
    elif not min_pages <= main <= max_pages:
        problems.append(f"main text: References start on page {main}, outside {min_pages}-{max_pages}")
    text = "\n".join(pages)
    if (n := checklist_answers(text)) != checklist_items:
        problems.append(f"checklist: {n} answers, expected {checklist_items}")
    if found := placeholders(text):
        problems.append(f"placeholders: {found}")
    if REPO_URL not in text.replace("\n", ""):
        problems.append("repository link missing")
    return problems


def pdf_pages(pdf: Path) -> list[str]:
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True).stdout
    return text.split("\f")[:-1] if text.endswith("\f") else text.split("\f")


def main(argv=None) -> int:
    pdf = Path((argv or sys.argv[1:] or ["report/final/report.pdf"])[0])
    pages = pdf_pages(pdf)
    problems = check_pages(pages)
    print(f"{pdf}: {len(pages)} pages, main text ends on page {main_text_pages(pages)}")
    for p in problems:
        print("  -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
