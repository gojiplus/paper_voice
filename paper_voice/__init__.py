"""
paper_voice
=================================

This package provides functionality for converting academic PDFs into spoken audio.
The goal is to support reading technical papers by using LLM to convert mathematical 
expressions and complex content into clear, narration-ready text.

The package now uses a simplified approach: extract text, enhance with LLM, 
and synthesize audio.

Modules
-------

`pdf_utils`:
    Simple PDF text extraction utilities.

`simple_llm_enhancer`:
    LLM-based enhancement that converts math expressions and cleans LaTeX artifacts.

`selective_enhancer`:
    Main interface for content enhancement using the simple LLM approach.

`figure_table_summarizer`:
    LLM integration for figure and table summarization.

`tts`:
    Abstraction layer over text‑to‑speech engines. Supports offline
    synthesis via ``pyttsx3`` and cloud synthesis via the OpenAI TTS API.
"""

from . import pdf_utils, simple_llm_enhancer, selective_enhancer, figure_table_summarizer, tts

__all__ = [
    "pdf_utils",
    "simple_llm_enhancer", 
    "selective_enhancer",
    "figure_table_summarizer",
    "tts",
]