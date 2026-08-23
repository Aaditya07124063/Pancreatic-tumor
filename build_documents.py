"""
build_documents.py
==================
Builds the .docx and .pdf versions of the write-ups from their Markdown sources.

Why this exists
---------------
Going straight from Markdown to .docx with pandoc produces tables with automatic
column widths. LibreOffice - which is what renders the .docx to .pdf - collapses
every column after the first to zero width, so the PDF shows only the row labels
and none of the numbers.

Routing through OpenDocument fixes it: pandoc writes the .odt with real column
widths, and LibreOffice then produces both the .pdf and the .docx from that. Both
outputs render tables correctly everywhere.

    Markdown --pandoc--> ODT --libreoffice--> PDF
                             \-libreoffice--> DOCX

Usage:
    python build_documents.py                 # build everything
    python build_documents.py report_internship_technical.md
"""

import os
import shutil
import subprocess
import sys

DOCS = [
    "report_internship_technical.md",
    "paper_ieee_conference.md",
    "manuscript_journal.md",
]

OUTDIR = "papers"


def build(md):
    stem = os.path.splitext(os.path.basename(md))[0]
    odt = os.path.join(OUTDIR, f"{stem}.odt")

    subprocess.run(
        ["pandoc", md, "-o", odt, "--toc", "--toc-depth=2", "--resource-path=."],
        check=True)

    for fmt in ("pdf", "docx"):
        subprocess.run(
            ["soffice", "--headless", "--convert-to", fmt, "--outdir", OUTDIR, odt],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.remove(odt)
    print(f"  {stem}: -> {OUTDIR}/{stem}.docx  |  {OUTDIR}/{stem}.pdf")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    targets = sys.argv[1:] or DOCS
    print("Building documents")
    for md in targets:
        if os.path.exists(md):
            build(md)
        else:
            print(f"  [skip] {md} not found")


if __name__ == "__main__":
    main()
