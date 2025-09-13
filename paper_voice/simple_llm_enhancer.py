"""
Simple LLM enhancer - just prompt + LLM, no manual processing.

This module takes the entire document text and sends it to the LLM with 
a comprehensive prompt to convert math expressions to natural language
and provide clear audio narration. No manual LaTeX processing.
"""

from openai import OpenAI
from typing import Optional


def enhance_with_simple_llm(content: str, api_key: str, progress_callback=None) -> str:
    """
    Simple enhancement: just pass all text to LLM with comprehensive prompt.
    
    No manual LaTeX processing, no selective enhancement, no chunking.
    Just send everything to LLM and let it handle the conversion.
    
    Parameters
    ----------
    content : str
        The entire document text (LaTeX, PDF-extracted, etc.)
    api_key : str
        OpenAI API key
    progress_callback : callable, optional
        Progress callback function
        
    Returns
    -------
    str
        Enhanced text ready for audio narration
    """
    
    if not api_key or api_key.strip() == "":
        if progress_callback:
            progress_callback("No API key provided, returning original content")
        return content
    
    if progress_callback:
        progress_callback("Starting simple LLM enhancement...")
    
    client = OpenAI(api_key=api_key)
    
    # One comprehensive prompt to handle everything
    prompt = f"""You are converting an academic paper into clear, natural audio narration. Your job is to take this raw document text and make it perfect for text-to-speech audio.

CRITICAL REQUIREMENTS:
1. Convert ALL mathematical expressions to clear natural language
2. Replace complex tables with comprehensive spoken summaries
3. Clean up all LaTeX commands and formatting artifacts
4. Keep ALL content - don't summarize or skip anything
5. Make everything flow naturally for audio narration

MATHEMATICAL CONVERSION:
- Convert $x^2$ to "x squared"  
- Convert \\(\\alpha\\) to "alpha"
- Convert \\[R^2 = 1 - \\frac{{SSE}}{{SST}}\\] to "R-squared equals one minus the ratio of sum of squared errors to total sum of squares"
- Convert \\(p_C\\) to "p subscript C"
- Convert \\(\\hat{{\\theta}}\\) to "theta hat"
- Be precise: "capital X" vs "lowercase x", "theta" not "θ"

TABLE PROCESSING:
- Replace LaTeX table environments (\\begin{{tabular}}, \\begin{{threeparttable}}) with clear spoken descriptions
- Example: "Table shows regression results. The first column presents coefficients for the baseline model..."
- Include all key information from the table in narrative form

LATEX CLEANUP:
- \\section{{Introduction}} → "Introduction"
- \\subsection{{Methods}} → "Methods"  
- Remove \\cite{{}}, \\ref{{}}, \\label{{}} commands
- Remove \\toprule, \\midrule, \\bottomrule
- Clean up broken citations like "bin1996}}"
- Remove LaTeX environments but keep the content

OUTPUT REQUIREMENTS:
- Natural flowing text perfect for audio
- No LaTeX commands remaining
- All math in clear English
- All tables described comprehensively  
- Preserve the full content and meaning
- Ready for text-to-speech conversion

DOCUMENT TO CONVERT:
{content}

Convert this entire document into clear, natural audio narration text:"""

    try:
        if progress_callback:
            progress_callback(f"Sending {len(content)} chars to LLM (may take 30-60 seconds)...")
        
        # Calculate appropriate max_tokens based on input size
        # Rule of thumb: output should be similar size or slightly larger
        estimated_tokens = len(content) // 3  # Rough estimate: 3 chars per token
        max_output_tokens = max(4000, min(8000, estimated_tokens * 2))  # 2x input, capped at 8k
        
        if progress_callback:
            progress_callback(f"Using max_tokens={max_output_tokens} for this request...")
            
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at converting academic documents into clear, natural audio narration text. Convert mathematical expressions to natural language and make everything perfect for text-to-speech."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=max_output_tokens,
            timeout=120  # 2 minute timeout
        )
        
        result = response.choices[0].message.content.strip()
        
        if progress_callback:
            progress_callback(f"LLM response received: {len(result)} characters")
            
        return result
        
    except Exception as e:
        error_msg = f"Simple LLM enhancement failed: {str(e)}"
        if progress_callback:
            progress_callback(error_msg)
        
        # For debugging, show more details
        if "timeout" in str(e).lower():
            error_msg += " (Request timed out - content may be too large)"
        elif "tokens" in str(e).lower():
            error_msg += f" (Token limit issue - content was {len(content)} chars)"
        elif "rate_limit" in str(e).lower():
            error_msg += " (Rate limit exceeded - try again in a moment)"
        
        if progress_callback:
            progress_callback(f"Detailed error: {error_msg}")
        
        # Return original content if enhancement fails
        return content


def enhance_with_chunking_fallback(content: str, api_key: str, progress_callback=None) -> str:
    """
    Fallback for very large documents - split into logical chunks.
    
    Only used if the single-call approach fails due to size limits.
    """
    
    if progress_callback:
        progress_callback("Document too large, using chunking fallback...")
    
    client = OpenAI(api_key=api_key)
    
    # Split into smaller chunks (rough paragraph splits)
    chunks = content.split('\n\n')
    enhanced_chunks = []
    
    chunk_prompt = """Convert this text chunk into clear audio narration. Convert all math expressions to natural language, clean up LaTeX artifacts, and make it flow naturally for text-to-speech:

{chunk}

Enhanced version:"""

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            enhanced_chunks.append(chunk)
            continue
            
        if progress_callback:
            progress_callback(f"Processing chunk {i+1}/{len(chunks)}...")
            
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Convert academic text to clear audio narration. Convert math to natural language."
                    },
                    {
                        "role": "user", 
                        "content": chunk_prompt.format(chunk=chunk)
                    }
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            enhanced_chunk = response.choices[0].message.content.strip()
            enhanced_chunks.append(enhanced_chunk)
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Warning: Failed to enhance chunk {i+1}: {str(e)}")
            # Keep original chunk if enhancement fails
            enhanced_chunks.append(chunk)
    
    return '\n\n'.join(enhanced_chunks)


def enhance_document_simple(content: str, api_key: str, progress_callback=None) -> str:
    """
    Main entry point for simple LLM enhancement.
    
    Tries single call first, falls back to chunking if needed.
    """
    
    if progress_callback:
        progress_callback(f"Starting enhancement for {len(content)} characters...")
    
    # ALWAYS use single call - no chunking allowed per user requirement
    if progress_callback:
        progress_callback("Using SINGLE LLM call approach (no chunking)...")
    
    return enhance_with_simple_llm(content, api_key, progress_callback)