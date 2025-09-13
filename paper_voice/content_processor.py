"""
Unified content processor for all input types.

This module provides a single, clean interface for processing different types
of academic content (PDF, LaTeX, Markdown, plain text) with selective LLM 
enhancement for math, figures, and tables.
"""

from typing import Optional, Union
from dataclasses import dataclass
from .latex_processor import process_latex_document, process_markdown_with_math
from .selective_enhancer import enhance_content_selectively, fix_pdf_extraction_issues
from . import pdf_utils


@dataclass
class ProcessedDocument:
    """Container for processed document content."""
    enhanced_text: str
    input_type: str
    figures: list = None
    tables: list = None
    equations: list = None
    metadata: dict = None
    has_math: bool = False
    has_figures: bool = False
    has_tables: bool = False


def process_content_unified(
    content: str, 
    input_type: str,
    api_key: Optional[str] = None,
    use_llm_enhancement: bool = True,
    progress_callback=None
) -> ProcessedDocument:
    """
    Process content with appropriate enhancement based on type.
    
    This is the main entry point for all content processing. It:
    1. Determines the appropriate processing strategy
    2. Applies selective LLM enhancement for math, figures, tables
    3. Preserves original text structure for everything else
    
    Parameters
    ----------
    content : str
        Raw content to process
    input_type : str
        Type of content: 'pdf', 'latex', 'markdown', 'text'
    api_key : str, optional
        OpenAI API key for LLM enhancement
    use_llm_enhancement : bool
        Whether to use LLM for math, figures, tables
    progress_callback : callable, optional
        Callback for progress updates
        
    Returns
    -------
    ProcessedDocument
        Processed content with metadata
    """
    
    if progress_callback:
        progress_callback(f"Processing {input_type} content...")
    
    figures = []
    tables = []
    equations = []
    metadata = {}
    enhanced_text = content
    
    if input_type.lower() in ['latex', 'tex']:
        if progress_callback:
            progress_callback("Processing LaTeX document with math, figures, tables...")
        
        # Use the comprehensive LaTeX processor
        if progress_callback:
            progress_callback(f"Launching LaTeX processor for {len(content)} characters...")
            
        processed = process_latex_document(
            content,
            summarize_figures=use_llm_enhancement and api_key is not None,
            summarize_tables=use_llm_enhancement and api_key is not None,
            api_key=api_key,
            use_llm_math=use_llm_enhancement and api_key is not None
        )
        
        enhanced_text = processed.text
        figures = processed.figures
        tables = processed.tables
        equations = processed.equations
        metadata = processed.metadata
        
        if progress_callback:
            progress_callback(f"LaTeX processing complete: {len(enhanced_text)} chars, {len(figures)} figures, {len(tables)} tables, {len(equations)} equations")
        
    elif input_type.lower() == 'markdown':
        if progress_callback:
            progress_callback("Processing Markdown with math expressions...")
        
        # Process Markdown with math
        enhanced_text = process_markdown_with_math(
            content, 
            api_key=api_key if use_llm_enhancement else None,
            use_llm=use_llm_enhancement and api_key is not None
        )
        
    elif input_type.lower() == 'pdf':
        if progress_callback:
            progress_callback("Processing PDF content...")
        
        if use_llm_enhancement and api_key:
            # For PDF, first fix extraction issues, then selective enhancement
            if progress_callback:
                progress_callback(f"Fixing PDF extraction issues for {len(content)} characters...")
            
            try:
                enhanced_text = fix_pdf_extraction_issues(content, api_key)
                if progress_callback:
                    progress_callback(f"PDF extraction fix complete: {len(enhanced_text)} characters")
                    
                if progress_callback:
                    progress_callback("Applying selective enhancement to PDF content...")
                enhanced_text = enhance_content_selectively(enhanced_text, api_key, progress_callback)
                
                if progress_callback:
                    progress_callback(f"PDF selective enhancement complete: {len(enhanced_text)} characters")
                    
            except Exception as e:
                if progress_callback:
                    progress_callback(f"PDF processing failed: {str(e)}")
                enhanced_text = content
        else:
            enhanced_text = content
            
    else:  # Plain text
        if progress_callback:
            progress_callback("Processing text content...")
        
        if use_llm_enhancement and api_key:
            # Apply selective enhancement (math, figures, tables only)
            enhanced_text = enhance_content_selectively(content, api_key, progress_callback)
        else:
            enhanced_text = content
    
    # Detect content characteristics
    has_math = '$' in enhanced_text or '\\(' in enhanced_text
    has_figures = any(word in enhanced_text.lower() for word in ['figure', 'fig.'])
    has_tables = any(word in enhanced_text.lower() for word in ['table', 'tab.'])
    
    if progress_callback:
        progress_callback("Content processing complete!")
    
    return ProcessedDocument(
        enhanced_text=enhanced_text,
        input_type=input_type,
        figures=figures,
        tables=tables,
        equations=equations,
        metadata=metadata,
        has_math=has_math,
        has_figures=has_figures,
        has_tables=has_tables
    )


def process_latex_content(
    content: str,
    api_key: Optional[str] = None,
    use_llm_enhancement: bool = True
) -> str:
    """
    Simplified interface for LaTeX content processing.
    
    This function specifically handles LaTeX files and applies LLM enhancement
    to math, figures, and tables while preserving all other text.
    """
    
    processed = process_latex_document(
        content,
        summarize_figures=use_llm_enhancement and api_key is not None,
        summarize_tables=use_llm_enhancement and api_key is not None,
        api_key=api_key,
        use_llm_math=use_llm_enhancement and api_key is not None
    )
    
    return processed.text


def process_text_with_math(
    content: str,
    api_key: Optional[str] = None,
    use_llm_enhancement: bool = True
) -> str:
    """
    Process text content that may contain mathematical expressions.
    
    This handles both Markdown-style and LaTeX-style math expressions.
    """
    
    # First try to process as Markdown with math
    if '$' in content or '\\(' in content:
        return process_markdown_with_math(
            content, 
            api_key=api_key if use_llm_enhancement else None,
            use_llm=use_llm_enhancement and api_key is not None
        )
    
    # If no math detected, apply selective enhancement
    if use_llm_enhancement and api_key:
        return enhance_content_selectively(content, api_key)
    
    return content


def get_supported_input_types() -> list:
    """Get list of supported input types."""
    return ['pdf', 'latex', 'markdown', 'text']


def detect_input_type(content: str, filename: Optional[str] = None) -> str:
    """
    Automatically detect the input type based on content and filename.
    
    Returns one of: 'pdf', 'latex', 'markdown', 'text'
    """
    
    if filename:
        filename_lower = filename.lower()
        if filename_lower.endswith('.tex') or filename_lower.endswith('.latex'):
            return 'latex'
        elif filename_lower.endswith('.md') or filename_lower.endswith('.markdown'):
            return 'markdown'
        elif filename_lower.endswith('.pdf'):
            return 'pdf'
    
    # Content-based detection
    latex_indicators = [
        '\\documentclass', '\\begin{document}', '\\section', '\\subsection',
        '\\usepackage', '\\maketitle', '\\author', '\\title'
    ]
    
    if any(indicator in content for indicator in latex_indicators):
        return 'latex'
    
    # Check for Markdown indicators
    markdown_indicators = [
        '# ', '## ', '### ', '```', '---', '* ', '- ', '1. '
    ]
    
    if any(indicator in content for indicator in markdown_indicators):
        return 'markdown'
    
    # Default to text
    return 'text'