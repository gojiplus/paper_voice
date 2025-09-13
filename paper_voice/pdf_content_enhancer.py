"""
LLM-powered PDF content enhancement.

This module uses LLMs to recover mathematical content, fix formatting issues,
and enhance poorly extracted PDF text for better narration.
"""

from typing import Optional
from openai import OpenAI
import re


def enhance_pdf_content_with_llm(raw_text: str, api_key: str) -> str:
    """
    Use LLM to enhance and fix issues in extracted PDF text.
    
    This function takes raw text extracted from a PDF (which often has
    issues like garbled math symbols, poor spacing, mangled tables)
    and uses an LLM to:
    - Reconstruct mathematical expressions from garbled symbols
    - Fix formatting and spacing issues
    - Identify and properly format mathematical content
    - Clean up table structures
    
    Parameters
    ----------
    raw_text : str
        Raw text extracted from PDF
    api_key : str
        OpenAI API key
    
    Returns
    -------
    str
        Enhanced text with improved mathematical content and formatting
    """
    
    client = OpenAI(api_key=api_key)
    
    # Split text into chunks to avoid token limits
    chunks = _split_text_into_chunks(raw_text, max_chunk_size=3000)
    enhanced_chunks = []
    
    for chunk in chunks:
        enhanced_chunk = _enhance_text_chunk(client, chunk)
        enhanced_chunks.append(enhanced_chunk)
    
    return "\n\n".join(enhanced_chunks)


def _split_text_into_chunks(text: str, max_chunk_size: int = 3000) -> list[str]:
    """Split text into manageable chunks for LLM processing."""
    
    # Try to split on paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk + paragraph) <= max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _enhance_text_chunk(client: OpenAI, chunk: str) -> str:
    """Enhance a single chunk of text using LLM."""
    
    prompt = f"""You are an expert at recovering and fixing academic paper text that was poorly extracted from PDF. Your job is to:

1. **Fix mathematical content**: Look for garbled mathematical symbols and expressions and reconstruct them properly
   - Convert Unicode/symbol garbage back to readable math (e.g., "α²" becomes "alpha squared", "∫" becomes "integral")
   - Identify mathematical expressions and format them clearly
   - Explain mathematical notation in natural language suitable for audio

2. **Fix formatting issues**: 
   - Remove excessive spacing and line breaks
   - Fix word breaks and hyphenation issues
   - Normalize paragraph structure

3. **Improve table content**:
   - If you see table-like content with poor alignment, restructure it clearly
   - Describe table contents in narrative form when possible

4. **Preserve important content**:
   - Keep all technical terms and proper names
   - Maintain the academic tone and meaning
   - Don't add information that wasn't in the original

CRITICAL: Output text that would be suitable for text-to-speech narration. Make mathematical expressions speakable.

Here's the text to enhance:

{chunk}

Enhanced text:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Low temperature for consistent, factual output
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        # If LLM fails, return original text
        print(f"LLM enhancement failed: {e}")
        return chunk


def recover_math_expressions_with_llm(text: str, api_key: str) -> str:
    """
    Specifically focus on recovering mathematical expressions from garbled PDF text.
    
    This is a more targeted function that focuses specifically on mathematical
    content recovery.
    """
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""You are a mathematician and expert at recognizing garbled mathematical expressions from poorly extracted PDF text.

Your task: Find and fix mathematical expressions that were corrupted during PDF extraction.

Common issues in PDF extraction:
- Greek letters become weird Unicode: α, β, θ, σ, etc.
- Mathematical symbols get mangled: ∫, ∑, ∆, ∂, etc.  
- Superscripts/subscripts become regular text: x² becomes x2, H₀ becomes H0
- Fractions become unclear: a/b becomes "a b" or "a over b"
- Mathematical operators get spacing issues

For each mathematical expression you find:
1. Identify what it was supposed to be
2. Convert it to clear, speakable English suitable for audio narration
3. Use phrases like "alpha squared", "integral from 0 to infinity", "theta hat", etc.

Focus ONLY on mathematical content. Don't change non-mathematical text.

Text to process:
{text}

Fixed text with mathematical expressions converted to speakable form:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Math recovery failed: {e}")
        return text