"""
paper_voice
=================================

This package provides functionality for converting academic PDFs into spoken audio.
The goal is to support reading technical papers in a way that respects complex
content such as mathematical expressions, tables, and figure captions.  The
public API exposes a set of helper functions to extract structured text from
PDFs, transform mathematics into speech-friendly phrases, summarise figures and
tables via LLMs, and synthesise audio via either offline engines or the
OpenAI Text‑to‑Speech API.

Modules
-------

`pdf_utils`:
    Helpers for extracting text, detecting math, and locating captions from PDF
    documents.

`math_to_speech`:
    Conversion utilities for mapping LaTeX/unicode mathematical expressions into
    spoken equivalents.

`figure_table_summarizer`:
    Optional integration for sending figure and table content to an LLM (OpenAI
    API) for summarisation.

`tts`:
    Abstraction layer over text‑to‑speech engines. Supports offline
    synthesis via ``pyttsx3`` and cloud synthesis via the OpenAI TTS API.
"""

from . import pdf_utils, math_to_speech, figure_table_summarizer, tts

__all__ = [
    "pdf_utils",
    "math_to_speech",
    "figure_table_summarizer",
    "tts",
]