"""
Selective content enhancement for academic papers.

This module provides selective enhancement that preserves the original text structure
while only improving specific content types (math, figures, tables) via LLM.
"""

import re
from typing import List, Tuple, Optional
from openai import OpenAI
from .llm_math_explainer import explain_math_with_llm_sync
from .figure_table_summarizer import summarize_figure_with_llm, summarize_table_with_llm


def enhance_content_selectively(content: str, api_key: str, progress_callback=None) -> str:
    """
    Selectively enhance content by improving only math, figures, and tables.
    
    This function preserves the original text structure and only enhances:
    1. Mathematical expressions (both inline and display)
    2. Figure captions and descriptions  
    3. Table content and structure
    
    All other text remains unchanged to avoid unwanted summarization.
    
    Parameters
    ----------
    content : str
        Original content to enhance
    api_key : str
        OpenAI API key for LLM processing
    progress_callback : callable, optional
        Callback function for progress updates
        
    Returns
    -------
    str
        Enhanced content with improved math, figures, and tables
    """
    
    if progress_callback:
        progress_callback("Starting selective content enhancement...")
    
    enhanced_content = content
    
    # 1. Enhance mathematical expressions
    enhanced_content = _enhance_math_expressions(enhanced_content, api_key, progress_callback)
    
    # 2. Enhance figure captions and descriptions
    enhanced_content = _enhance_figures(enhanced_content, api_key, progress_callback)
    
    # 3. Enhance table content
    enhanced_content = _enhance_tables(enhanced_content, api_key, progress_callback)
    
    if progress_callback:
        progress_callback("Selective enhancement completed!")
    
    return enhanced_content


def _enhance_math_expressions(content: str, api_key: str, progress_callback=None) -> str:
    """Enhance mathematical expressions while preserving all other text."""
    
    processed_content = content
    
    # Find and process display math ($$...$$)
    display_math_pattern = r'\$\$(.*?)\$\$'
    display_matches = list(re.finditer(display_math_pattern, content, re.DOTALL))
    
    if display_matches:
        if progress_callback:
            progress_callback(f"Enhancing {len(display_matches)} display equations...")
        
        # Process in reverse order to maintain positions
        for i, match in enumerate(reversed(display_matches)):
            math_expr = match.group(1).strip()
            
            # Get context around the expression
            context_start = max(0, match.start() - 500)
            context_end = min(len(content), match.end() + 500)
            context = content[context_start:context_end]
            
            # Get LLM explanation
            try:
                explanation = explain_math_with_llm_sync(math_expr, api_key, context)
                replacement = f" {explanation.natural_explanation} "
                processed_content = (processed_content[:match.start()] + 
                                   replacement + 
                                   processed_content[match.end():])
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Warning: Failed to enhance equation {i+1}: {str(e)}")
                # Keep original if enhancement fails
                continue
    
    # Find and process inline math ($...$) - but not display math
    inline_math_pattern = r'(?<!\$)\$([^$]+?)\$(?!\$)'
    inline_matches = list(re.finditer(inline_math_pattern, processed_content))
    
    if inline_matches:
        if progress_callback:
            progress_callback(f"Enhancing {len(inline_matches)} inline expressions...")
        
        # Process in reverse order to maintain positions
        for i, match in enumerate(reversed(inline_matches)):
            math_expr = match.group(1).strip()
            
            # Get context around the expression
            context_start = max(0, match.start() - 300)
            context_end = min(len(processed_content), match.end() + 300)
            context = processed_content[context_start:context_end]
            
            try:
                explanation = explain_math_with_llm_sync(math_expr, api_key, context)
                replacement = f" {explanation.natural_explanation} "
                processed_content = (processed_content[:match.start()] + 
                                   replacement + 
                                   processed_content[match.end():])
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Warning: Failed to enhance inline expression {i+1}: {str(e)}")
                # Keep original if enhancement fails
                continue
    
    return processed_content


def _enhance_figures(content: str, api_key: str, progress_callback=None) -> str:
    """Enhance figure captions and descriptions."""
    
    # Pattern to match figure references and captions
    figure_patterns = [
        r'Figure\s+\d+[:\.\-]\s*([^\n]+(?:\n(?!\s*\n)[^\n]+)*)',  # Figure 1: caption
        r'Fig\.\s*\d+[:\.\-]\s*([^\n]+(?:\n(?!\s*\n)[^\n]+)*)',   # Fig. 1: caption
        r'FIGURE\s+\d+[:\.\-]\s*([^\n]+(?:\n(?!\s*\n)[^\n]+)*)',  # FIGURE 1: caption
    ]
    
    enhanced_content = content
    figure_count = 0
    
    for pattern in figure_patterns:
        matches = list(re.finditer(pattern, enhanced_content, re.IGNORECASE))
        
        if matches:
            figure_count += len(matches)
            if progress_callback:
                progress_callback(f"Enhancing {len(matches)} figure captions...")
            
            # Process in reverse order to maintain positions
            for match in reversed(matches):
                full_match = match.group(0)
                caption_text = match.group(1).strip()
                
                try:
                    # Use the existing figure summarizer
                    enhanced_caption = summarize_figure_with_llm(caption_text, api_key)
                    # Replace the caption part while keeping the "Figure X:" prefix
                    prefix = full_match.replace(caption_text, "").strip()
                    replacement = f"{prefix} {enhanced_caption}"
                    
                    enhanced_content = (enhanced_content[:match.start()] + 
                                      replacement + 
                                      enhanced_content[match.end():])
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Warning: Failed to enhance figure caption: {str(e)}")
                    # Keep original if enhancement fails
                    continue
    
    if figure_count > 0 and progress_callback:
        progress_callback(f"Enhanced {figure_count} figure captions")
    
    return enhanced_content


def _enhance_tables(content: str, api_key: str, progress_callback=None) -> str:
    """Enhance table captions and content."""
    
    # Pattern to match table references and captions
    table_patterns = [
        r'Table\s+\d+[:\.\-]\s*([^\n]+(?:\n(?!\s*\n)[^\n]+)*)',   # Table 1: caption
        r'TABLE\s+\d+[:\.\-]\s*([^\n]+(?:\n(?!\s*\n)[^\n]+)*)',   # TABLE 1: caption
    ]
    
    enhanced_content = content
    table_count = 0
    
    for pattern in table_patterns:
        matches = list(re.finditer(pattern, enhanced_content, re.IGNORECASE))
        
        if matches:
            table_count += len(matches)
            if progress_callback:
                progress_callback(f"Enhancing {len(matches)} table captions...")
            
            # Process in reverse order to maintain positions
            for match in reversed(matches):
                full_match = match.group(0)
                caption_text = match.group(1).strip()
                
                try:
                    # Use the existing table summarizer
                    enhanced_caption = summarize_table_with_llm(caption_text, api_key)
                    # Replace the caption part while keeping the "Table X:" prefix
                    prefix = full_match.replace(caption_text, "").strip()
                    replacement = f"{prefix} {enhanced_caption}"
                    
                    enhanced_content = (enhanced_content[:match.start()] + 
                                      replacement + 
                                      enhanced_content[match.end():])
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Warning: Failed to enhance table caption: {str(e)}")
                    # Keep original if enhancement fails
                    continue
    
    if table_count > 0 and progress_callback:
        progress_callback(f"Enhanced {table_count} table captions")
    
    return enhanced_content


def fix_pdf_extraction_issues(content: str, api_key: str) -> str:
    """
    Fix specific PDF extraction issues without changing the overall text structure.
    
    This function targets common PDF extraction problems:
    - Garbled mathematical symbols
    - Broken Unicode characters
    - Poor spacing around mathematical expressions
    - Fragmented mathematical notation
    
    It preserves all non-mathematical content unchanged.
    """
    
    client = OpenAI(api_key=api_key)
    
    # Look for likely garbled mathematical content
    # Common patterns: corrupted Unicode, isolated Greek letters, mathematical symbols
    garbled_math_patterns = [
        r'[αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ]',  # Greek letters
        r'[∫∑∏∆∂∇±≤≥≠≈∞∝√∈∉⊂⊃∪∩]',  # Mathematical symbols
        r'\b[a-zA-Z]\s*\d+\s*[a-zA-Z]\b',  # Likely subscript/superscript issues
    ]
    
    # Find sentences that likely contain corrupted math
    sentences = content.split('. ')
    enhanced_sentences = []
    
    for sentence in sentences:
        has_garbled_math = any(re.search(pattern, sentence) for pattern in garbled_math_patterns)
        
        if has_garbled_math and len(sentence) < 500:  # Only process short sentences to avoid context loss
            try:
                prompt = f"""This sentence was extracted from a PDF and likely contains corrupted mathematical notation.
Fix ONLY the mathematical symbols and expressions while keeping all other text exactly the same.

Common issues:
- Greek letters: α, β, θ, σ, etc. should become "alpha", "beta", "theta", "sigma"
- Math symbols: ∫ → "integral", ∑ → "sum", √ → "square root"
- Subscripts/superscripts: x₁ → "x sub 1", x² → "x squared"

Original sentence: {sentence}

Fixed sentence (keep all non-mathematical text identical):"""

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                
                fixed_sentence = response.choices[0].message.content.strip()
                enhanced_sentences.append(fixed_sentence)
            
            except Exception:
                # If fixing fails, keep original
                enhanced_sentences.append(sentence)
        else:
            # No garbled math detected, keep original
            enhanced_sentences.append(sentence)
    
    return '. '.join(enhanced_sentences)