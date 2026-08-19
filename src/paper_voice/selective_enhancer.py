"""Simple content enhancement for academic papers.

This module provides a simplified approach that just passes all content
to the LLM with a comprehensive prompt. No manual LaTeX processing.
"""

from collections.abc import Callable


def enhance_content_selectively(
    content: str,
    api_key: str,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """Simple enhancement using just LLM + prompt approach.

    This function passes the entire document to the LLM with instructions
    to convert math, clean LaTeX, and make everything audio-ready.

    Args:
        content: Original content to enhance (PDF text, LaTeX, etc.)
        api_key: OpenAI API key for LLM processing
        progress_callback: Called with a human-readable status message
            at each step; no calls are made when it is None.

    Returns:
        Enhanced content ready for audio narration.
    """
    if progress_callback:
        progress_callback("Starting simple LLM enhancement...")

    if not api_key or api_key.strip() == "":
        if progress_callback:
            progress_callback("No API key provided, returning original content")
        return content

    try:
        from .simple_llm_enhancer import enhance_document_simple

        if progress_callback:
            progress_callback("Processing document with simple LLM approach...")

        enhanced_content = enhance_document_simple(content, api_key, progress_callback)

        if progress_callback:
            progress_callback("Simple enhancement completed!")

        return enhanced_content

    except Exception as e:
        if progress_callback:
            progress_callback(f"Enhancement failed: {e!s}, returning original content")
        return content


def fix_pdf_extraction_issues(
    content: str,
    api_key: str,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """Fix common PDF extraction issues using LLM.

    This function addresses broken words, spacing issues, and formatting
    problems that commonly occur during PDF text extraction.

    Args:
        content: Raw PDF extracted content with potential issues
        api_key: OpenAI API key for LLM processing
        progress_callback: Called with a human-readable status message
            at each step; no calls are made when it is None.

    Returns:
        Cleaned content with extraction issues fixed.
    """
    if progress_callback:
        progress_callback("Fixing PDF extraction issues...")

    if not api_key or api_key.strip() == "":
        if progress_callback:
            progress_callback("No API key provided, returning original content")
        return content

    try:
        from .simple_llm_enhancer import enhance_document_simple

        if progress_callback:
            progress_callback("Processing PDF extraction fixes with LLM...")

        # Use the same enhancement approach - it handles PDF cleanup too
        enhanced_content = enhance_document_simple(content, api_key, progress_callback)

        if progress_callback:
            progress_callback("PDF extraction fix completed!")

        return enhanced_content

    except Exception as e:
        if progress_callback:
            progress_callback(f"PDF fix failed: {e!s}, returning original content")
        return content
