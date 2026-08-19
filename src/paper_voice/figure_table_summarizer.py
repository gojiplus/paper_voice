"""LLM-powered summarisation for figures and tables.

This module defines helper functions that invoke the OpenAI API to
generate brief, descriptive summaries for figure and table content.  The
summariser is optional: if no API key is supplied the raw caption or
table text will be returned unmodified.  These helpers are thin
wrappers; higher-level code should cache results appropriately if
multiple captions are repeated.
"""

from __future__ import annotations

import os

try:
    from openai import OpenAI

    openai_available = True
except ImportError:  # pragma: no cover
    # Allow import even if openai is not installed; functions will check
    OpenAI = None
    openai_available = False


def _ensure_api_key(api_key: str | None) -> str:
    """Return a valid OpenAI API key or raise an error.

    This helper first checks the provided argument; if missing, it falls
    back to the ``OPENAI_API_KEY`` environment variable. If still not
    found, a ``ValueError`` is raised.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("An OpenAI API key is required for summarisation.")
    return key


def summarise_caption(
    caption: str, kind: str = "figure", api_key: str | None = None
) -> str:
    """Generate a concise summary for a figure or table caption using OpenAI.

    Args:
        caption: The raw caption text extracted from the PDF. Blank
            captions are returned as-is without calling the API.
        kind: Either ``"figure"`` or ``"table"``; interpolated into the
            prompt so the model knows what it is describing.
        api_key: An OpenAI API key. If not provided, ``OPENAI_API_KEY``
            from the environment will be used. If neither is available,
            the input caption is returned unchanged.

    Returns:
        A one- or two-sentence summary suitable for reading aloud. If the
        ``openai`` package is missing, no key is available, or the request
        fails, the original caption is returned.
    """
    if not caption.strip():
        return caption
    # Use fallback if openai or API key missing
    if not openai_available or OpenAI is None:
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
        message = response.choices[0].message.content
        if message is None:
            # No usable completion: same fallback as the handler below.
            return caption
        return message.strip()
    except Exception:
        # Gracefully fall back to the original caption
        return caption


def summarise_table(rows: list[str], api_key: str | None = None) -> str:
    """Generate a summary description of tabular content using OpenAI.

    Args:
        rows: One string per table row, with column delimiters preserved.
            They are joined with newlines before being sent to the model.
        api_key: OpenAI API key. Same behaviour as ``summarise_caption``:
            falls back to ``OPENAI_API_KEY`` from the environment.

    Returns:
        A summary of the table content, intended for audio narration. An
        empty ``rows`` list yields an empty string; if the ``openai``
        package is missing, no key is available, or the request fails, the
        rows are returned joined by spaces.
    """
    if not rows:
        return ""
    if not openai_available or OpenAI is None:
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
        message = response.choices[0].message.content
        if message is None:
            # No usable completion: same fallback as the handler below.
            return " ".join(rows)
        return message.strip()
    except Exception:
        return " ".join(rows)


# American spelling aliases for compatibility
def summarize_figure_with_llm(caption: str, api_key: str | None = None) -> str:
    """American spelling alias for summarise_caption with figure type."""
    return summarise_caption(caption, kind="figure", api_key=api_key)


def summarize_table_with_llm(caption: str, api_key: str | None = None) -> str:
    """American spelling alias for summarise_caption with table type."""
    return summarise_caption(caption, kind="table", api_key=api_key)
