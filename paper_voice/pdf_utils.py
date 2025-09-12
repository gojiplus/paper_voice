"""
Utilities for extracting structured text from PDFs.

This module wraps PyPDF2 to read PDF documents and provides helpers for
locating math expressions, figure/table captions and other structural
elements. The functions are deliberately liberal in what they accept: PDFs
vary widely in how they encode text and layout, so heuristics are used to
identify elements.

The extraction functions return plain unicode strings rather than
structured objects. Higher‑level logic in other modules can decide how to
further process the raw text.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from PyPDF2 import PdfReader


def extract_raw_text(pdf_path: str) -> List[str]:
    """Extract raw text from each page of a PDF.

    Parameters
    ----------
    pdf_path: str
        Path to the PDF file on disk.

    Returns
    -------
    List[str]
        A list of strings, one per page. Pages that cannot be read will
        produce an empty string in the corresponding position.

    Notes
    -----
    PyPDF2 does not always perfectly extract text, especially from PDF
    documents created from scans or with unusual fonts. If the document
    contains images of text (e.g. scanned pages), you may need an OCR
    pipeline such as pytesseract instead.
    """
    reader = PdfReader(pdf_path)
    pages_text: List[str] = []
    for page in reader.pages:
        try:
            # Extract text for each page. The extract_text method returns a
            # string or None.
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages_text.append(text)
    return pages_text


def detect_inline_math(text: str) -> List[Tuple[int, int, str]]:
    """Detect inline LaTeX math expressions delimited by $...$ in text.

    Parameters
    ----------
    text: str
        A block of text.

    Returns
    -------
    List[Tuple[int, int, str]]
        A list of tuples (start_index, end_index, math_text). Each tuple
        records the inclusive start and exclusive end positions of the
        mathematics within the original string as well as the raw contents
        between the dollar signs. Unmatched dollar signs are ignored.

    Examples
    --------
    >>> detect_inline_math("This is $x^2$ and $\alpha$.")
    [(8, 12, 'x^2'), (18, 24, '\\alpha')]
    """
    matches: List[Tuple[int, int, str]] = []
    # Use a simple regex to find sequences between unescaped dollar signs.
    # Negative lookbehind ensures the $ isn't escaped with \\
    pattern = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")
    for match in pattern.finditer(text):
        start, end = match.span()
        math_content = match.group(1)
        matches.append((start, end, math_content))
    return matches


def extract_captions(page_text: str) -> List[Tuple[str, str]]:
    """Extract likely figure or table captions from a page of text.

    Captions in academic PDFs often start with "Figure" or "Table" followed
    by a number and optional colon/dash, then the description. This function
    uses a simple heuristic to find such patterns.

    Parameters
    ----------
    page_text: str
        Text from a single page of a PDF.

    Returns
    -------
    List[Tuple[str, str]]
        A list of (kind, caption) tuples. kind is either 'figure' or
        'table'. caption is the remainder of the line following the label.

    Notes
    -----
    The heuristic may over‑capture or miss captions depending on how the
    PDF is encoded. To improve results you might integrate a more
    sophisticated layout parser (e.g. via pdfplumber) or leverage natural
    language processing to identify caption‑like sentences.
    """
    captions: List[Tuple[str, str]] = []
    lines = page_text.splitlines()
    for line in lines:
        # Normalize whitespace and remove extra spaces
        stripped = line.strip()
        if not stripped:
            continue
        # Lowercase for detection but keep original for output
        lower = stripped.lower()
        # Simple patterns for figure/table captions
        # Accept forms like "Figure 1:", "FIGURE 2.", "Table S1 -", etc.
        fig_match = re.match(r"^(figure)\s*([\divxslc]+)[.: -]+(.*)$", lower)
        table_match = re.match(r"^(table)\s*([\divxslc]+)[.: -]+(.*)$", lower)
        if fig_match:
            kind = 'figure'
            # Use the original line for the caption text; remove the label
            # up to the first colon/dot/dash
            # Find the position of first colon or dash etc. in original line
            sep_match = re.search(r"[:.\-]", stripped, re.IGNORECASE)
            if sep_match:
                caption_text = stripped[sep_match.end():].strip()
            else:
                # Fallback: remove the first word and number
                parts = stripped.split(maxsplit=2)
                caption_text = parts[2] if len(parts) >= 3 else ''
            captions.append((kind, caption_text))
        elif table_match:
            kind = 'table'
            sep_match = re.search(r"[:.\-]", stripped, re.IGNORECASE)
            if sep_match:
                caption_text = stripped[sep_match.end():].strip()
            else:
                parts = stripped.split(maxsplit=2)
                caption_text = parts[2] if len(parts) >= 3 else ''
            captions.append((kind, caption_text))
    return captions


def extract_table_text(page_text: str) -> List[str]:
    """Heuristically extract table‑like text from a page.

    The simplest approach here is to look for lines containing multiple
    columns separated by two or more spaces or tab characters. This is
    intended as a fallback when no dedicated table extraction library is
    available. For more robust extraction consider using Camelot, Tabula or
    pdfplumber if possible (these require Java or system dependencies).

    Parameters
    ----------
    page_text: str
        The text of a single PDF page.

    Returns
    -------
    List[str]
        A list of strings representing detected table rows. Column
        separators are preserved (multiple spaces or tabs).
    """
    table_rows: List[str] = []
    for line in page_text.splitlines():
        # Consider a line as part of a table if it contains two or more
        # consecutive spaces and at least one non‑space before and after
        if re.search(r"\S\s{2,}\S", line):
            table_rows.append(line)
    return table_rows