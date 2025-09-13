"""
Simple PDF text extraction utilities.

This module just extracts raw text from PDFs. All processing (math, tables, etc.)
is now handled by the LLM, not manual parsing.
"""

from __future__ import annotations

from typing import List

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


def extract_full_document_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF as a single string.
    
    This is the main function used by the simplified pipeline.
    All processing is now done by LLM, not manual parsing.
    """
    pages = extract_raw_text(pdf_path)
    return '\n\n'.join(page for page in pages if page.strip())