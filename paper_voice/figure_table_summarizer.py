"""
LLM‑powered summarisation for figures and tables.

This module defines helper functions that invoke the OpenAI API to
generate brief, descriptive summaries for figure and table content.  The
summariser is optional: if no API key is supplied the raw caption or
table text will be returned unmodified.  These helpers are thin
wrappers; higher‑level code should cache results appropriately if
multiple captions are repeated.
"""

from __future__ import annotations

import os
from typing import List, Optional

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    # Allow import even if openai is not installed; functions will check
    OpenAI = None  # type: ignore
    openai_available = False


def _ensure_api_key(api_key: Optional[str]) -> str:
    """Return a valid OpenAI API key or raise an error.

    This helper first checks the provided argument; if missing, it falls
    back to the ``OPENAI_API_KEY`` environment variable. If still not
    found, a ``ValueError`` is raised.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("An OpenAI API key is required for summarisation.")
    return key


def summarise_caption(caption: str, kind: str = "figure", api_key: Optional[str] = None) -> str:
    """Generate a concise summary for a figure or table caption using OpenAI.

    Parameters
    ----------
    caption: str
        The raw caption text extracted from the PDF.
    kind: str, optional
        Either ``"figure"`` or ``"table"``; used to inform the prompt.
    api_key: str, optional
        An OpenAI API key. If not provided, ``OPENAI_API_KEY`` from the
        environment will be used. If neither is available, the input
        caption is returned unchanged.

    Returns
    -------
    str
        A summary suitable for reading aloud. If summarisation fails,
        returns the original caption.
    """
    if not caption.strip():
        return caption
    # Use fallback if openai or API key missing
    if not openai_available:
        return caption
    try:
        key = _ensure_api_key(api_key)
    except ValueError:
        return caption
    try:
        client = OpenAI(api_key=key)
        prompt = (
            f"You are a helpful assistant specialised in converting {kind} captions "
            "from academic papers into concise descriptions suitable for audio. "
            "Summarise the following caption in one or two sentences, focusing on what "
            "the figure or table conveys: \n\n"
            f"Caption: {caption.strip()}"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=100,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception:
        # Gracefully fall back to the original caption
        return caption


def summarise_table(rows: List[str], api_key: Optional[str] = None) -> str:
    """Generate a summary description of tabular content using OpenAI.

    Parameters
    ----------
    rows: List[str]
        A list of strings, each representing a row of the table. Column
        delimiters are preserved.
    api_key: str, optional
        OpenAI API key. Same behaviour as ``summarise_caption``.

    Returns
    -------
    str
        A summary of the table content, intended for audio narration. If
        summarisation fails, returns a concatenated string of the rows.
    """
    if not rows:
        return ""
    if not openai_available:
        return " ".join(rows)
    try:
        key = _ensure_api_key(api_key)
    except ValueError:
        return " ".join(rows)
    try:
        client = OpenAI(api_key=key)
        # Join the rows with newlines to preserve structure for the model
        table_text = "\n".join(rows)
        prompt = (
            "You are an assistant that converts tables from academic papers into brief "
            "spoken descriptions. Summarise the following table by describing what "
            "information it contains, patterns, and any notable relationships. "
            "Do not read every cell individually.\n\n"
            f"Table:\n{table_text}"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception:
        return " ".join(rows)