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
        
        # Calculate appropriate max_tokens based on OpenAI limits
        estimated_input_tokens = len(content) // 3  # Rough estimate: 3 chars per token
        # Reserve reasonable space for output (1.5x input, but cap at 16k which is GPT-4o max output)
        max_output_tokens = min(16384, max(4000, int(estimated_input_tokens * 1.5)))
        
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


def enhance_with_intelligent_chunking(content: str, api_key: str, progress_callback=None) -> str:
    """
    Intelligent chunking based on actual OpenAI token limits.
    
    - GPT-4o: 128,000 tokens total (input + output)
    - We reserve 50,000 tokens for output, leaving ~78,000 for input
    - This means each chunk can be ~234,000 characters (78k tokens * 3 chars/token)
    """
    
    if progress_callback:
        progress_callback("Using intelligent chunking based on OpenAI limits...")
    
    client = OpenAI(api_key=api_key)
    
    # Conservative chunking: 70,000 input tokens max per chunk (210,000 chars)
    MAX_CHUNK_TOKENS = 70000
    MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * 3  # 210,000 characters
    
    # Split content into appropriately sized chunks
    chunks = []
    current_pos = 0
    
    while current_pos < len(content):
        # Find a good breaking point near the limit
        end_pos = min(current_pos + MAX_CHUNK_CHARS, len(content))
        
        if end_pos < len(content):
            # Look for a good break point (paragraph, section, etc.)
            chunk_text = content[current_pos:end_pos]
            
            # Try to break at section boundaries first
            for break_pattern in ['\n\\section{', '\n\\subsection{', '\n\n\n', '\n\n']:
                last_break = chunk_text.rfind(break_pattern)
                if last_break > MAX_CHUNK_CHARS * 0.7:  # At least 70% of max size
                    end_pos = current_pos + last_break
                    break
        
        chunk = content[current_pos:end_pos].strip()
        if chunk:
            chunks.append(chunk)
        
        current_pos = end_pos
    
    if progress_callback:
        progress_callback(f"Split into {len(chunks)} intelligent chunks (avg {len(content)//len(chunks):,} chars each)")
    
    enhanced_chunks = []
    
    # Process each chunk with the full comprehensive prompt
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk):,} chars)...")
        
        try:
            # Use the same comprehensive prompt as single-call approach
            enhanced_chunk = enhance_with_simple_llm(chunk, api_key, None)  # No nested progress
            enhanced_chunks.append(enhanced_chunk)
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Warning: Failed to enhance chunk {i+1}: {str(e)}")
            # Keep original chunk if enhancement fails
            enhanced_chunks.append(chunk)
    
    return '\n\n'.join(enhanced_chunks)


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
    
    Based on actual OpenAI limits:
    - GPT-4o: 128,000 tokens total (input + output)
    - GPT-4.1: 1,000,000 tokens total
    - Rule of thumb: ~3-4 characters per token
    """
    
    if progress_callback:
        progress_callback(f"Starting enhancement for {len(content)} characters...")
    
    # Calculate estimated tokens (conservative estimate: 3 chars per token)
    estimated_input_tokens = len(content) // 3
    estimated_output_tokens = estimated_input_tokens * 1.5  # Assume 1.5x expansion
    total_estimated_tokens = estimated_input_tokens + estimated_output_tokens
    
    # GPT-4o limit: 128,000 tokens total
    GPT4O_TOKEN_LIMIT = 128000
    
    if total_estimated_tokens <= GPT4O_TOKEN_LIMIT:
        if progress_callback:
            progress_callback(f"Using SINGLE LLM call ({estimated_input_tokens:,} input + {estimated_output_tokens:,} estimated output = {total_estimated_tokens:,} tokens, within {GPT4O_TOKEN_LIMIT:,} limit)")
        
        return enhance_with_simple_llm(content, api_key, progress_callback)
    else:
        if progress_callback:
            progress_callback(f"Content too large ({total_estimated_tokens:,} tokens > {GPT4O_TOKEN_LIMIT:,} limit). Using intelligent chunking with SINGLE call per chunk...")
        
        return enhance_with_intelligent_chunking(content, api_key, progress_callback)