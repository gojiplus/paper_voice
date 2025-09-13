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

MATHEMATICAL CONVERSION - Speak like a professor explaining at the board:

SUBSCRIPTS AND NOTATION (explain meaning, not just symbols):
- \\(p_C\\) → "p underscore capital C, the proportion of compliers" or "the complier share"
- \\(F_{{1C}}\\) → "F underscore one capital C, the outcome distribution for treated compliers" 
- \\(F_{{0C}}\\) → "F underscore zero capital C, the outcome distribution for control compliers"
- \\(F_{{Y|Z=z}}\\) → "the distribution of Y when the instrument takes value z"
- \\(\\hat{{\\theta}}\\) → "theta-hat, our estimate of theta" 
- \\(X_i\\) → "X underscore i" or "X for individual i"
- \\(\\beta_1\\) → "beta underscore one, the treatment effect coefficient"
- \\(e(X)\\) → "e of X, the probability that the instrument equals one given covariates X"
- \\(p_z(X)\\) → "p underscore z of X, the expected treatment status when the instrument takes value z, given covariates X"

CAPITALIZATION AND VARIABLES (TTS cannot distinguish Z vs z):
- \\(Z = z\\) → "when the instrument capital-Z takes value lowercase-z" or "when the instrument takes value z"
- \\(Z \\in \\{{0,1\\}}\\) → "the instrument takes values zero or one"
- Random variables: Use descriptive names ("the instrument" not just "Z")
- Specific values: "when Z equals zero" or "under treatment assignment"

EQUATIONS AND RELATIONSHIPS (break into natural speech):
- \\(Y = \\beta X + \\epsilon\\) → "Y equals beta times X plus the error term epsilon"
- \\(\\sum_{{i=1}}^n X_i\\) → "the sum of X sub-i from i equals 1 to n"
- \\(E[Y|X]\\) → "the expected value of Y given X"
- \\(\\frac{{a}}{{b}}\\) → "the ratio of a to b" or "a over b"

PROFESSOR-STYLE NARRATION:
- First mention: "Let p_C denote the proportion of compliers"
- Subsequent: "the complier share" or just "p_C"
- Complex expressions: Break into meaningful parts
- Estimates: "our estimate of" or "the estimated value"
- Conditional notation: "when", "given that", "conditional on"

CRITICAL EXAMPLES - Transform robotic to natural:

WRONG (robotic): "F subscript 1C and the estimated F subscript 0C"
RIGHT (natural): "the outcome distributions for treated and control compliers"

WRONG: "p subscript C times the difference between F subscript 1C of y and F subscript 0C of y"  
RIGHT: "p underscore capital C, the complier share, times the difference in outcome distributions between treated and control compliers"

WRONG: "the difference between F subscript Y given Z equals 1 and F subscript Y given Z equals 0"
RIGHT: "the difference between outcome distributions under treatment assignment versus control assignment"

WRONG: "Define e of X as the probability that Z equals one given X, p subscript z of X as the expected value of D given Z equals z and X"
RIGHT: "Define e of X as the probability that the instrument equals one, conditional on covariates X. Define p underscore z of X as the expected treatment status when the instrument takes value z, conditional on covariates X"

CONTEXT-AWARE CONVERSION:
First mention: "Define p_C as the proportion of compliers in the population"
Later references: "the complier share" or "this proportion"

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
            progress_callback(f"Prompt length: {len(prompt):,} chars, Content length: {len(content):,} chars")
            progress_callback("🚀 Making API call to OpenAI...")
            
        response = client.chat.completions.create(
            model="gpt-4o",
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
            timeout=60  # 1 minute timeout
        )
        
        result = response.choices[0].message.content.strip()
        
        if progress_callback:
            progress_callback(f"LLM response received: {len(result)} characters")
        
        # Validate that enhancement actually worked
        if result == content:
            raise Exception("LLM returned identical content - enhancement failed")
        
        # Check if critical math expressions were converted
        # Count original vs remaining math expressions
        original_inline = content.count("\\(")
        original_display = content.count("\\[")
        remaining_inline = result.count("\\(")
        remaining_display = result.count("\\[")
        
        # If most math wasn't converted, that's a failure
        if original_inline > 0 and remaining_inline >= original_inline * 0.8:
            raise Exception(f"Inline math not converted - {remaining_inline}/{original_inline} expressions remain")
        if original_display > 0 and remaining_display >= original_display * 0.8:
            raise Exception(f"Display math not converted - {remaining_display}/{original_display} expressions remain")
            
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
        elif "authentication" in str(e).lower() or "api_key" in str(e).lower():
            error_msg += " (API key issue - check your OpenAI key)"
        
        if progress_callback:
            progress_callback(f"Detailed error: {error_msg}")
        
        # RAISE the error instead of hiding it
        raise Exception(error_msg)


def enhance_with_intelligent_chunking(content: str, api_key: str, progress_callback=None) -> str:
    """
    Intelligent chunking based on actual OpenAI token limits.
    
    - GPT-4o: 128,000 tokens total (input + output)  
    - Reserve ~50,000 tokens for output, leaving ~78,000 for input
    - Each chunk: ~60,000 input tokens max = ~180,000 characters
    """
    
    if progress_callback:
        progress_callback("Using intelligent chunking based on OpenAI API limits...")
    
    client = OpenAI(api_key=api_key)
    
    # Conservative chunking for GPT-4o: 60,000 input tokens max per chunk
    MAX_CHUNK_TOKENS = 60000
    MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * 3  # ~180,000 characters
    
    if progress_callback:
        progress_callback(f"Max chunk size: {MAX_CHUNK_CHARS:,} chars (~{MAX_CHUNK_TOKENS:,} tokens)")
    
    # Split content into appropriately sized chunks
    chunks = []
    current_pos = 0
    
    while current_pos < len(content):
        # Calculate end position
        end_pos = min(current_pos + MAX_CHUNK_CHARS, len(content))
        
        # If not at the end, find a good breaking point
        if end_pos < len(content):
            chunk_text = content[current_pos:end_pos]
            
            # Try to break at natural boundaries (in order of preference)
            break_patterns = [
                '\n\\section{',      # LaTeX sections
                '\n\\subsection{',  # LaTeX subsections  
                '\n\\begin{',       # LaTeX environments
                '\n\n\n',          # Triple newlines
                '\n\n',            # Double newlines (paragraphs)
                '. ',              # Sentence endings
                ', ',              # Comma breaks
            ]
            
            for pattern in break_patterns:
                last_break = chunk_text.rfind(pattern)
                if last_break > MAX_CHUNK_CHARS * 0.6:  # At least 60% of max size
                    end_pos = current_pos + last_break + len(pattern)
                    break
        
        chunk = content[current_pos:end_pos].strip()
        if chunk:
            chunks.append(chunk)
        
        current_pos = end_pos
    
    if progress_callback:
        progress_callback(f"Split into {len(chunks)} chunks (sizes: {[len(c) for c in chunks][:5]}{'...' if len(chunks) > 5 else ''})")
    
    enhanced_chunks = []
    
    # Process each chunk with comprehensive enhancement
    for i, chunk in enumerate(chunks):
        chunk_tokens = len(chunk) // 3
        if progress_callback:
            progress_callback(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk):,} chars, ~{chunk_tokens:,} tokens)...")
        
        try:
            # Each chunk gets the full enhancement treatment
            enhanced_chunk = enhance_with_simple_llm(chunk, api_key, None)
            enhanced_chunks.append(enhanced_chunk)
            
            if progress_callback:
                progress_callback(f"✅ Chunk {i+1}/{len(chunks)} completed successfully")
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Chunk {i+1}/{len(chunks)} failed: {str(e)}")
            
            # If individual chunk fails, we need to handle it
            if "context_length_exceeded" in str(e):
                # Chunk is still too big - try to split it further
                if progress_callback:
                    progress_callback(f"Chunk {i+1} too large, attempting to split further...")
                
                try:
                    # Recursively chunk this piece
                    sub_enhanced = enhance_with_intelligent_chunking(chunk, api_key, None)
                    enhanced_chunks.append(sub_enhanced)
                    
                    if progress_callback:
                        progress_callback(f"✅ Chunk {i+1} completed via sub-chunking")
                        
                except Exception as sub_e:
                    raise Exception(f"Chunk {i+1} failed even after sub-chunking: {str(sub_e)}")
            else:
                # Some other error - propagate it
                raise Exception(f"Chunk {i+1} enhancement failed: {str(e)}")
    
    final_result = '\n\n'.join(enhanced_chunks)
    
    if progress_callback:
        progress_callback(f"✅ All {len(chunks)} chunks processed successfully! Final length: {len(final_result):,} chars")
    
    return final_result


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
                model="gpt-4o",
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
            progress_callback(f"Content too large ({total_estimated_tokens:,} tokens > {GPT4O_TOKEN_LIMIT:,} limit). Using intelligent chunking...")
        
        return enhance_with_intelligent_chunking(content, api_key, progress_callback)