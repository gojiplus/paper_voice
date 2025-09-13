"""
Advanced Streamlit app with LLM-powered mathematical explanations.

This app uses GPT-4 to convert mathematical expressions, figures, and tables
into crystal-clear natural language that listeners can easily understand.
"""

import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import streamlit as st

# Filter out pydub SyntaxWarnings for Python 3.13+ compatibility
# These warnings come from invalid escape sequences in pydub's regex patterns
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub.utils")
warnings.filterwarnings("ignore", message="invalid escape sequence", category=SyntaxWarning)
warnings.filterwarnings("ignore", message=r"invalid escape sequence.*", category=SyntaxWarning)

# Also suppress at import time
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    try:
        import pydub
    except ImportError:
        pass

# Import our modules
try:
    from paper_voice.llm_math_explainer import (
        explain_math_with_llm_sync, explain_figure_with_llm_sync, 
        explain_table_with_llm_sync, get_math_explanation_prompt
    )
    from paper_voice.latex_processor import extract_figures_and_tables, extract_latex_environments
    from paper_voice.arxiv_downloader import download_arxiv_paper, ArxivPaper
    from paper_voice import pdf_utils, tts
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from paper_voice.llm_math_explainer import (
        explain_math_with_llm_sync, explain_figure_with_llm_sync,
        explain_table_with_llm_sync, get_math_explanation_prompt
    )
    from paper_voice.latex_processor import extract_figures_and_tables, extract_latex_environments
    from paper_voice.arxiv_downloader import download_arxiv_paper, ArxivPaper
    from paper_voice import pdf_utils, tts


def download_arxiv_pdf(arxiv_id: str, dest_path: str) -> Optional[str]:
    """Download PDF from arXiv."""
    import requests
    
    # Extract ID from URL if needed
    # Handle formats like: https://arxiv.org/abs/2508.21038 or https://arxiv.org/pdf/2508.21038.pdf
    id_match = re.search(r"(?:abs|pdf)/(\d+\.\d+)(?:\.pdf)?", arxiv_id)
    if id_match:
        arxiv_id = id_match.group(1)
    
    # Try multiple URLs for better compatibility
    urls_to_try = [
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        f"https://export.arxiv.org/pdf/{arxiv_id}.pdf",
    ]
    
    for url in urls_to_try:
        try:
            resp = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if resp.status_code == 200:
                # Check if it's actually a PDF (not an error page)
                content = resp.content
                if len(content) > 1000 and content.startswith(b'%PDF'):
                    with open(dest_path, 'wb') as f:
                        f.write(content)
                    return dest_path
                    
        except Exception:
            continue
    
    return None


def extract_pdf_content(pdf_path: str, use_vision: bool = False, api_key: str = None, show_debug: bool = False) -> str:
    """Extract text content from PDF."""
    try:
        if use_vision and api_key:
            # Use vision analysis for better content separation
            try:
                from paper_voice.vision_pdf_analyzer import analyze_pdf_with_vision, create_enhanced_text_from_analysis
                
                st.info("🔍 Using GPT-4V to analyze PDF structure...")
                analysis_result = analyze_pdf_with_vision(pdf_path, api_key, max_pages=10)  # Limit to 10 pages for cost
                
                enhanced_text = create_enhanced_text_from_analysis(analysis_result)
                
                # Debug: Show content length
                if show_debug:
                    st.info(f"📄 Vision analysis extracted {len(enhanced_text)} characters")
                
                # Show analysis summary
                st.success(f"✅ Vision analysis complete: {len(analysis_result.content_blocks)} content blocks found")
                with st.expander("📊 Analysis Summary", expanded=False):
                    st.write(f"**Pages analyzed:** {analysis_result.metadata['total_pages']}")
                    st.write(f"**Content blocks:** {analysis_result.metadata['total_content_blocks']}")
                    st.write(f"**Content types found:** {', '.join(analysis_result.metadata['content_types'])}")
                
                return enhanced_text
                
            except Exception as e:
                st.warning(f"⚠️ Vision analysis failed ({str(e)}), falling back to standard extraction")
                # Fall back to standard extraction
        
        # Standard PDF text extraction
        pages = pdf_utils.extract_raw_text(pdf_path)
        
        # Debug: Show raw extraction results
        total_raw_chars = sum(len(page) for page in pages)
        if show_debug:
            st.info(f"📄 Standard extraction: {len(pages)} pages, {total_raw_chars} total characters")
        
        # Clean up the text and join with proper spacing
        cleaned_pages = []
        for i, page in enumerate(pages):
            # Remove excessive whitespace and normalize line breaks
            cleaned = re.sub(r'\n+', '\n', page.strip())  # Multiple newlines to single
            cleaned = re.sub(r' +', ' ', cleaned)  # Multiple spaces to single
            cleaned_pages.append(cleaned)
            
            # Debug: Show first few characters of each page
            if show_debug and len(cleaned) > 0:
                preview = cleaned[:100] + "..." if len(cleaned) > 100 else cleaned
                st.info(f"Page {i+1} preview ({len(cleaned)} chars): {preview}")
        
        result = "\n\n".join(cleaned_pages)
        if show_debug:
            st.info(f"📄 Final cleaned content: {len(result)} characters")
        
        return result
    except Exception as e:
        error_msg = f"Error extracting PDF content: {str(e)}"
        st.error(error_msg)
        return error_msg


def enhance_content_with_llm_fast(content: str, api_key: str, input_type: str = "text", show_debug: bool = False) -> str:
    """Fast batch enhancement of content using streamlined processing."""
    import re
    from openai import OpenAI
    
    try:
        if not api_key or api_key.strip() == "":
            if show_debug:
                st.warning("⚠️ No API key provided, skipping LLM enhancement")
            return content
        
        if show_debug:
            st.info(f"🔧 Fast processing {len(content)} characters with batch LLM enhancement")
        
        client = OpenAI(api_key=api_key)
        enhanced_content = content
        
        # Find all math expressions first (batch them)
        inline_math = re.findall(r'\\\((.*?)\\\)', content, re.DOTALL)
        display_math = re.findall(r'\\\[(.*?)\\\]', content, re.DOTALL)
        
        all_math = [(expr, 'inline') for expr in inline_math] + [(expr, 'display') for expr in display_math]
        
        if show_debug:
            st.info(f"📊 Found {len(inline_math)} inline + {len(display_math)} display = {len(all_math)} total math expressions")
        
        if all_math:
            # Create a single batch prompt for all math expressions
            batch_prompt = """You are converting mathematical expressions into clear natural language for audio narration.

CRITICAL REQUIREMENTS:
- Use precise language: "capital X" vs "lowercase x"  
- Explain meaning, not just symbols
- Use "equals" not "=", "times" not "×"
- For Greek letters: "theta" not "θ", "epsilon" not "ε"
- Make it flow naturally when spoken

Convert each expression below into natural language. Return ONLY the explanations, one per line, in the same order:

"""
            
            for i, (expr, math_type) in enumerate(all_math):
                batch_prompt += f"{i+1}. {expr.strip()}\n"
            
            if show_debug:
                st.info(f"🤖 Sending batch request to GPT-4 for {len(all_math)} expressions...")
            
            try:
                # Single API call for all math
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert at converting mathematical expressions into clear, natural language for audio narration."},
                        {"role": "user", "content": batch_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                    timeout=30  # 30 second timeout
                )
                
                explanations = response.choices[0].message.content.strip().split('\n')
                explanations = [exp.strip() for exp in explanations if exp.strip()]
                
                if show_debug:
                    st.info(f"✅ Got {len(explanations)} explanations from LLM")
                
                # Replace math expressions with explanations
                if len(explanations) >= len(all_math):
                    # Replace display math first (to avoid conflicts)
                    for i, (expr, math_type) in enumerate(all_math):
                        if math_type == 'display':
                            pattern = r'\\\[' + re.escape(expr) + r'\\\]'
                            if i < len(explanations):
                                enhanced_content = re.sub(pattern, f" {explanations[i]} ", enhanced_content, count=1)
                    
                    # Then replace inline math
                    for i, (expr, math_type) in enumerate(all_math):
                        if math_type == 'inline':
                            pattern = r'\\\(' + re.escape(expr) + r'\\\)'
                            if i < len(explanations):
                                enhanced_content = re.sub(pattern, f" {explanations[i]} ", enhanced_content, count=1)
                else:
                    if show_debug:
                        st.warning(f"⚠️ Got {len(explanations)} explanations but expected {len(all_math)}")
            
            except Exception as e:
                if show_debug:
                    st.error(f"❌ Batch LLM processing failed: {str(e)}")
                # Fall back to original content
                pass
        
        # Clean up LaTeX artifacts
        enhanced_content = re.sub(r'\\section\*?\{([^}]*)\}', r'Section: \1', enhanced_content)
        enhanced_content = re.sub(r'\\subsection\*?\{([^}]*)\}', r'Subsection: \1', enhanced_content)
        enhanced_content = re.sub(r'\\begin\{threeparttable\}.*?\\end\{threeparttable\}', 
                                 '[Table content with statistical results]', enhanced_content, flags=re.DOTALL)
        enhanced_content = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', 
                                 '[Statistical table with numerical results]', enhanced_content, flags=re.DOTALL)
        enhanced_content = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', enhanced_content)  # Remove other LaTeX commands
        enhanced_content = re.sub(r'[{}\\]', '', enhanced_content)  # Remove stray LaTeX characters
        enhanced_content = re.sub(r'\s+', ' ', enhanced_content)  # Normalize whitespace
        
        if show_debug:
            st.info(f"✅ Fast processing complete: {len(enhanced_content)} characters")
        
        return enhanced_content.strip()
    
    except Exception as e:
        error_msg = f"Fast processing failed: {e}"
        st.error(error_msg)
        return content


def process_content_with_llm(content: str, api_key: str, progress_callback=None) -> str:
    """Process content using LLM for mathematical explanations."""
    
    if progress_callback:
        progress_callback("Analyzing mathematical content...")
    
    processed_content = content
    explanations_added = []
    
    # Find and process display math ($$...$$) - correct regex patterns
    display_math_pattern = r'\$\$(.*?)\$\$'
    display_matches = list(re.finditer(display_math_pattern, content, re.DOTALL))
    
    if display_matches:
        if progress_callback:
            progress_callback(f"Processing {len(display_matches)} display equations...")
        
        # Process in reverse order to maintain positions
        for i, match in enumerate(reversed(display_matches)):
            if progress_callback:
                progress_callback(f"Explaining display equation {len(display_matches) - i}/{len(display_matches)}...")
            
            math_expr = match.group(1).strip()
            
            # Get context around the expression
            context_start = max(0, match.start() - 500)
            context_end = min(len(content), match.end() + 500)
            context = content[context_start:context_end]
            
            # Get LLM explanation
            explanation = explain_math_with_llm_sync(math_expr, api_key, context)
            
            # Replace the math with explanation
            replacement = f" {explanation.natural_explanation} "
            processed_content = processed_content[:match.start()] + replacement + processed_content[match.end():]
            
            explanations_added.append((f"$${math_expr}$$", explanation.natural_explanation))
    
    # Find and process inline math ($...$) - but not display math
    inline_math_pattern = r'(?<!\$)\$([^$]+?)\$(?!\$)'
    inline_matches = list(re.finditer(inline_math_pattern, processed_content))
    
    if inline_matches:
        if progress_callback:
            progress_callback(f"Processing {len(inline_matches)} inline expressions...")
        
        # Process in reverse order
        for i, match in enumerate(reversed(inline_matches)):
            if progress_callback:
                progress_callback(f"Explaining inline expression {len(inline_matches) - i}/{len(inline_matches)}...")
            
            math_expr = match.group(1).strip()
            
            # Skip if it's very simple (just a variable)
            if len(math_expr) <= 2 and math_expr.isalpha():
                continue
            
            # Get context
            context_start = max(0, match.start() - 300)
            context_end = min(len(processed_content), match.end() + 300)
            context = processed_content[context_start:context_end]
            
            # Get LLM explanation
            explanation = explain_math_with_llm_sync(math_expr, api_key, context)
            
            # Replace with explanation
            processed_content = processed_content[:match.start()] + f" {explanation.natural_explanation} " + processed_content[match.end():]
            
            explanations_added.append((f"${math_expr}$", explanation.natural_explanation))
    
    # Process LaTeX environments (theorem, proposition, etc.)
    theorem_pattern = r'\\begin\{(theorem|proposition|lemma|corollary)\}(.*?)\\end\{\1\}'
    theorem_matches = list(re.finditer(theorem_pattern, processed_content, re.DOTALL | re.IGNORECASE))
    
    if theorem_matches and progress_callback:
        progress_callback(f"Processing {len(theorem_matches)} theorems/propositions...")
    
    # Process LaTeX structural elements
    processed_content = re.sub(r'\\section\*?\{([^}]*)\}', r'Section: \1', processed_content)
    processed_content = re.sub(r'\\subsection\*?\{([^}]*)\}', r'Subsection: \1', processed_content)
    processed_content = re.sub(r'\\subsubsection\*?\{([^}]*)\}', r'Subsubsection: \1', processed_content)
    processed_content = re.sub(r'\\title\{([^}]*)\}', r'Title: \1', processed_content)
    processed_content = re.sub(r'\\author\{([^}]*)\}', r'Author: \1', processed_content)
    
    # Handle common LaTeX environments
    processed_content = re.sub(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', r'Abstract: \1', processed_content, flags=re.DOTALL)
    processed_content = re.sub(r'\\begin\{itemize\}(.*?)\\end\{itemize\}', r'\1', processed_content, flags=re.DOTALL)
    processed_content = re.sub(r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}', r'\1', processed_content, flags=re.DOTALL)
    processed_content = re.sub(r'\\item\s*', '• ', processed_content)
    
    # Clean up remaining LaTeX commands
    processed_content = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})*', ' ', processed_content)
    processed_content = re.sub(r'[{}]', '', processed_content)
    processed_content = re.sub(r'\s+', ' ', processed_content)
    
    if progress_callback:
        progress_callback("Mathematical processing complete!")
    
    return processed_content.strip()


def process_figures_and_tables(content: str, api_key: str) -> Tuple[List[str], List[str]]:
    """Process figures and tables with LLM explanations."""
    
    figure_explanations = []
    table_explanations = []
    
    # Extract figures and tables
    figures, tables = extract_figures_and_tables(content)
    
    # Process figures
    for caption, figure_content in figures:
        explanation = explain_figure_with_llm_sync(caption, api_key, figure_content)
        figure_explanations.append(explanation)
    
    # Process tables
    for caption, table_content in tables:
        explanation = explain_table_with_llm_sync(caption, table_content, api_key)
        table_explanations.append(explanation)
    
    return figure_explanations, table_explanations


def create_comprehensive_narration_script(content: str, input_type: str, api_key: str,
                                        uploaded_images: List[Any] = None, show_debug: bool = False) -> str:
    """Create a comprehensive narration script with LLM explanations."""
    
    script_parts = []
    
    # Introduction
    intro = f"This is a narration of the uploaded {input_type.lower()} document with enhanced mathematical explanations."
    script_parts.append(intro)
    
    # Debug: Show initial content length
    if show_debug:
        st.info(f"📝 Creating script from {len(content)} characters of {input_type} content")
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(message: str):
        status_text.text(message)
    
    # First enhance the content for better narration (FAST VERSION)
    update_progress("Enhancing content for audio narration...")
    enhanced_content = enhance_content_with_llm_fast(content, api_key, input_type, show_debug)
    progress_bar.progress(80)
    
    # Debug: Check if enhancement worked
    if not enhanced_content or enhanced_content.strip() == "":
        st.error("❌ Enhanced content is empty! Check content processing pipeline.")
        st.error(f"**Debug Info:**")
        st.error(f"- Original content length: {len(content)} characters") 
        st.error(f"- Enhanced content: '{enhanced_content}'")
        st.error(f"- Enhanced content type: {type(enhanced_content)}")
        
        # Show the raw content to debug
        if show_debug:
            with st.expander("🔍 Original Content for Debugging", expanded=True):
                st.text_area("Original content that failed to enhance:", content[:2000] + "..." if len(content) > 2000 else content, height=300)
        
        return intro  # Return just the introduction
    
    # The enhanced_content already has all math, figures, tables processed
    # by the unified backend processor, so just use it directly
    update_progress("Finalizing processed content...")
    script_parts.append(enhanced_content)
    progress_bar.progress(90)
    
    # Process uploaded images
    if uploaded_images:
        script_parts.append("Additional images were provided:")
        for i, img_file in enumerate(uploaded_images):
            explanation = explain_figure_with_llm_sync(f"Uploaded image: {img_file.name}", api_key)
            script_parts.append(explanation)
    
    progress_bar.progress(100)
    update_progress("Script generation complete!")
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    final_script = "\\n\\n".join(script_parts)
    
    # Debug: Show final script statistics
    if show_debug:
        st.info(f"📋 Final script: {len(final_script)} characters, {len(script_parts)} sections")
    
    return final_script


def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Paper Voice Advanced", 
        layout="wide", 
        page_icon="🧠",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠📄🔊 Paper Voice Advanced")
    st.markdown("""
    **Paper Voice Advanced** uses GPT-4 to convert mathematical expressions, figures, and tables 
    into **crystal-clear natural language** explanations that listeners can easily understand.
    
    ### ✨ Key Features:
    - 🧠 **LLM-Powered Math Explanations**: Convert complex LaTeX into precise natural English
    - 📊 **Intelligent Figure/Table Descriptions**: Automatic visual content narration  
    - 🎯 **Context-Aware Processing**: Understands statistical, calculus, and algebraic notation
    - 🗣️ **High-Quality Audio**: OpenAI TTS or offline speech synthesis
    - 📝 **Multiple Input Formats**: PDF, LaTeX, Markdown, plain text
    """)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # API Key (required for LLM features)
    api_key = st.sidebar.text_input(
        "🔑 OpenAI API Key (Required)",
        type="password",
        help="Required for mathematical explanations and figure descriptions"
    )
    
    if not api_key:
        st.sidebar.warning("⚠️ OpenAI API key required for advanced features")
        st.sidebar.markdown("This app uses GPT-4 to create natural language explanations of mathematical expressions.")
    
    # Input method
    input_method = st.sidebar.selectbox(
        "📁 Input Method",
        ["Upload Files", "arXiv Download", "Direct Text Input"]
    )
    
    # File uploads and inputs
    uploaded_files = []
    uploaded_images = []
    arxiv_input = ""
    direct_text = ""
    
    if input_method == "Upload Files":
        uploaded_files = st.sidebar.file_uploader(
            "📄 Upload Documents",
            type=["pdf", "tex", "md", "txt"],
            accept_multiple_files=True,
            help="Upload PDF, LaTeX, Markdown, or text files"
        )
        
        uploaded_images = st.sidebar.file_uploader(
            "🖼️ Upload Images (Optional)",
            type=["png", "jpg", "jpeg", "gif", "svg"],
            accept_multiple_files=True,
            help="Upload figures referenced in your document"
        )
        
    elif input_method == "arXiv Download":
        arxiv_input = st.sidebar.text_input(
            "📚 arXiv ID or URL",
            help="e.g., 2301.12345 or https://arxiv.org/abs/2301.12345. Downloads LaTeX source + figures for best quality."
        )
        
    elif input_method == "Direct Text Input":
        direct_text = st.sidebar.text_area(
            "✏️ Paste Content",
            height=200,
            help="Paste your LaTeX, Markdown, or text with math expressions"
        )
    
    # Advanced options
    st.sidebar.header("⚙️ Advanced Options")
    
    use_vision_pdf = st.sidebar.checkbox(
        "🔍 Use Vision AI for PDF Analysis",
        value=False,
        help="Use GPT-4V to analyze PDF layout and separate figures/math/tables (slower but higher quality)"
    )
    
    show_debug_info = st.sidebar.checkbox(
        "🐛 Show Debug Information",
        value=True,
        help="Show detailed processing steps and content previews"
    )
    
    # Audio options
    st.sidebar.header("🎵 Audio Options")
    
    use_openai_tts = st.sidebar.checkbox(
        "Use OpenAI TTS (Recommended)",
        value=True,
        help="High-quality neural text-to-speech"
    )
    
    if use_openai_tts:
        openai_voice = st.sidebar.selectbox(
            "Voice",
            ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            index=0
        )
        speech_speed = st.sidebar.slider(
            "Speech Speed",
            0.25, 4.0, 1.0, 0.25
        )
    else:
        offline_voice = st.sidebar.text_input(
            "Offline Voice Filter",
            help="Leave empty for default voice"
        )
        speech_rate = st.sidebar.slider(
            "Speech Rate (WPM)",
            100, 300, 180, 10
        )
    
    # Two-step process buttons
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Two-Step Process:**")
    st.sidebar.info("⚡ **Fast Processing**: Uses batch LLM calls instead of individual requests for much faster processing (~30 seconds vs several minutes).")
    
    generate_script_button = st.sidebar.button(
        "📝 Step 1: Generate Enhanced Script (Fast)",
        type="primary",
        disabled=not api_key,
        help="Requires OpenAI API key - uses fast batch processing (~30 seconds)"
    )
    
    generate_audio_button = st.sidebar.button(
        "🎵 Step 2: Convert Script to Audio", 
        disabled=True,  # Will be enabled when script is ready
        help="Generate audio from the enhanced script"
    )
    
    # Main interface
    if not api_key:
        st.info("🔑 Please enter your OpenAI API key in the sidebar to use advanced mathematical explanations.")
        
        st.markdown("""
        ### Example of LLM-Enhanced Mathematical Explanations:
        
        **Before (Basic):**
        > "hat theta equals fraction one over n sum from i equals one to n psi W sub i semicolon hat eta superscript minus k of i"
        
        **After (LLM-Enhanced):**
        > "Theta hat, which represents our estimator, is calculated as the average over all n observations. Specifically, we take the sum from i equals 1 to n of the function psi, evaluated at W subscript i, using the auxiliary parameter eta hat that was estimated on the complement sample excluding fold k of i, then divide this sum by n."
        
        **Statistical Expression Example:**
        
        LaTeX: `$\\sqrt{n}(\\hat{\\theta} - \\theta_0) \\xrightarrow{d} N(0, \\Sigma)$`
        
        **LLM Explanation:**
        > "As the sample size grows large, the quantity square root of n times the difference between our estimator theta hat and the true parameter theta naught converges in distribution to a normal distribution with mean zero and covariance matrix capital Sigma. This is a fundamental result showing that our estimator is asymptotically normal."
        """)
        return
    
    # Initialize session state for the enhanced script
    if 'enhanced_script' not in st.session_state:
        st.session_state.enhanced_script = ""
    
    # Main content area - single column layout for better editing experience
    st.subheader("📝 Enhanced Script")
    
    if st.session_state.enhanced_script:
        # Show editable script in a large text area
        edited_script = st.text_area(
            "Review and edit the enhanced script:",
            value=st.session_state.enhanced_script,
            height=400,
            help="Edit the script as needed before converting to audio. The script has been enhanced with natural language explanations of mathematical expressions.",
            key="script_editor"
        )
        
        # Update session state if user edits
        if edited_script != st.session_state.enhanced_script:
            st.session_state.enhanced_script = edited_script
        
        # Enable audio generation button when script is available
        st.sidebar.markdown("---")
        audio_ready = st.sidebar.button(
            "🎵 Step 2: Convert Script to Audio",
            type="primary",
            help="Generate audio from the enhanced script"
        )
        
        if audio_ready:
            st.subheader("🎵 Audio Output")
            # STEP 2: Generate Audio from Script
            try:
                current_script = st.session_state.enhanced_script
                
                with st.spinner("🎵 Generating audio from enhanced script..."):
                    audio_dir = tempfile.mkdtemp(prefix="paper_voice_advanced_")
                    audio_ext = ".mp3" if use_openai_tts else ".wav"
                    audio_path = os.path.join(audio_dir, f"enhanced_narration{audio_ext}")
                    
                    if use_openai_tts:
                        output_file = tts.synthesize_speech_chunked(
                            current_script,
                            audio_path,
                            use_openai=True,
                            api_key=api_key,
                            openai_voice=openai_voice
                        )
                    else:
                        output_file = tts.synthesize_speech_chunked(
                            current_script,
                            audio_path,
                            voice=offline_voice,
                            rate=speech_rate,
                            use_openai=False
                        )
                    
                    # Display audio
                    st.audio(output_file)
                    
                    # Download button
                    with open(output_file, "rb") as f:
                        audio_bytes = f.read()
                    
                    st.download_button(
                        "⬇️ Download Enhanced Audio",
                        data=audio_bytes,
                        file_name=f"paper_voice_enhanced{audio_ext}",
                        mime=f"audio/{'mpeg' if audio_ext == '.mp3' else 'wav'}"
                    )
                    
                    st.success("✅ Enhanced narration generated successfully!")
            
            except Exception as e:
                st.error(f"❌ Audio generation failed: {str(e)}")
                import traceback
                with st.expander("🐛 Audio Debug Information"):
                    st.code(traceback.format_exc())
    else:
        st.info("👆 Use 'Step 1: Generate Enhanced Script' to create an enhanced script from your content.")
    
    # Clear button to reset script
    if st.session_state.enhanced_script:
        if st.button("🗑️ Clear Script", help="Clear the current script to start over"):
            st.session_state.enhanced_script = ""
            st.rerun()
    
    # STEP 1: Generate Enhanced Script
    if generate_script_button:
        if not api_key:
            st.error("🔑 OpenAI API key is required for advanced processing.")
            return
        
        try:
            content = ""
            input_type = ""
            
            # Get content based on input method
            if input_method == "Upload Files" and uploaded_files:
                all_content = []
                for file in uploaded_files:
                    file_ext = file.name.lower().split('.')[-1]
                    
                    if file_ext == 'pdf':
                        # Save PDF temporarily and extract text
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file.read())
                            pdf_content = extract_pdf_content(tmp.name, use_vision=use_vision_pdf, api_key=api_key, show_debug=show_debug_info)
                            all_content.append(f"=== {file.name} ===\\n{pdf_content}")
                            os.unlink(tmp.name)
                        input_type = "PDF"
                    else:
                        file_content = file.read().decode('utf-8')
                        all_content.append(f"=== {file.name} ===\\n{file_content}")
                        
                        # Map file extensions to proper input types
                        if file_ext in ['tex', 'latex']:
                            input_type = "LaTeX"
                        elif file_ext in ['md', 'markdown']:
                            input_type = "Markdown"
                        else:
                            input_type = "Text"
                
                content = "\\n\\n".join(all_content)
                
            elif input_method == "arXiv Download" and arxiv_input.strip():
                with st.spinner("📥 Downloading LaTeX source from arXiv..."):
                    paper = download_arxiv_paper(arxiv_input.strip())
                    if paper:
                        st.success(f"✅ Downloaded: {paper.title}")
                        content = paper.latex_content
                        input_type = "LaTeX"
                        
                        # Display metadata
                        if paper.metadata:
                            with st.expander("📋 Paper Information", expanded=False):
                                if 'title' in paper.metadata:
                                    st.write(f"**Title:** {paper.metadata['title']}")
                                if 'author' in paper.metadata:
                                    st.write(f"**Author(s):** {paper.metadata['author']}")
                                if 'abstract' in paper.metadata:
                                    st.write(f"**Abstract:** {paper.metadata['abstract'][:500]}...")
                                st.write(f"**Figures found:** {len(paper.figures)}")
                                if paper.figures:
                                    st.write(f"**Figure files:** {', '.join(list(paper.figures.keys())[:5])}")
                    else:
                        st.error(f"❌ Failed to download LaTeX source from arXiv.")
                        st.info("💡 This could be because:")
                        st.info("• The paper source is not available (some older papers)")
                        st.info("• The paper ID/URL is invalid")
                        st.info("• arXiv servers are temporarily unavailable")
                        st.info("🔍 You can try uploading the PDF directly or paste the paper content as text.")
                        return
                            
            elif input_method == "Direct Text Input" and direct_text.strip():
                content = direct_text
                input_type = "Text"
            else:
                st.warning("⚠️ Please provide content to process.")
                return
            
            if not content.strip():
                st.warning("⚠️ No content found to process.")
                return
            
            # Show raw content preview if debug is enabled
            if show_debug_info:
                with st.expander("📄 Raw Extracted Content Preview", expanded=False):
                    st.write(f"**Content type:** {input_type}")
                    st.write(f"**Content length:** {len(content)} characters")
                    preview_text = content[:1000] + "..." if len(content) > 1000 else content
                    st.text_area("Raw content preview:", preview_text, height=200)
            
            # Generate enhanced script and store in session state
            try:
                with st.spinner("🧠 Creating enhanced narration with LLM explanations... (should take ~30 seconds)"):
                    enhanced_script = create_comprehensive_narration_script(
                        content, input_type, api_key, uploaded_images, show_debug_info
                    )
                
                # Store script in session state
                st.session_state.enhanced_script = enhanced_script
                
                st.success("✅ Enhanced script generated! You can now review and edit it below.")
                st.rerun()  # Refresh to show the script editor
                
            except Exception as e:
                st.error(f"❌ Script generation failed: {str(e)}")
                st.info("💡 Try refreshing the page and using a shorter document, or check your API key.")
                if show_debug_info:
                    import traceback
                    with st.expander("🐛 Error Details"):
                        st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            import traceback
            with st.expander("🐛 Debug Information"):
                st.code(traceback.format_exc())
    
    # Help section
    with st.expander("💡 How It Works"):
        st.markdown("""
        ### 🧠 LLM-Enhanced Mathematical Explanations
        
        This app uses GPT-4 to convert mathematical expressions into natural, precise English:
        
        **Example Transformations:**
        
        1. **Statistical Notation:**
           - `$E[Y_i | Z_i = 1]$` → "the expected value of Y for individual i given that the instrument Z subscript i equals 1"
           - `$\\hat{\\theta}$` → "theta hat, our estimator"
           - `$\\sqrt{n}(\\hat{\\theta} - \\theta_0)$` → "square root of n times the difference between our estimator theta hat and the true parameter theta naught"
        
        2. **Complex Expressions:**
           - Fractions become "the ratio of ... to ..." or "... divided by ..."
           - Summations explain the range and meaning: "the sum over all observations of..."
           - Convergence notation gets full explanations: "converges in distribution to..."
        
        3. **Precise Variable Handling:**
           - Distinguishes "capital X" from "lowercase x"
           - Explains subscripts contextually: "Y for individual i" instead of "Y sub i"
           - Uses full Greek letter names: "theta" not "θ"
        
        ### 🎯 Key Advantages:
        - **Contextual Understanding**: Recognizes statistical vs. calculus notation
        - **Multiple Sentences**: Uses several clear sentences when needed
        - **Listener-Friendly**: Optimized for audio comprehension
        - **Mathematically Precise**: Maintains accuracy while improving clarity
        """)


if __name__ == "__main__":
    main()