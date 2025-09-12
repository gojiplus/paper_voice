"""
Advanced Streamlit app with LLM-powered mathematical explanations.

This app uses GPT-4 to convert mathematical expressions, figures, and tables
into crystal-clear natural language that listeners can easily understand.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import streamlit as st

# Import our modules
try:
    from paper_voice.llm_math_explainer import (
        explain_math_with_llm_sync, explain_figure_with_llm_sync, 
        explain_table_with_llm_sync, get_math_explanation_prompt
    )
    from paper_voice.latex_processor import extract_figures_and_tables, extract_latex_environments
    from paper_voice import pdf_utils, tts
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from paper_voice.llm_math_explainer import (
        explain_math_with_llm_sync, explain_figure_with_llm_sync,
        explain_table_with_llm_sync, get_math_explanation_prompt
    )
    from paper_voice.latex_processor import extract_figures_and_tables, extract_latex_environments
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


def extract_pdf_content(pdf_path: str) -> str:
    """Extract text content from PDF."""
    try:
        pages = pdf_utils.extract_raw_text(pdf_path)
        # Clean up the text and join with proper spacing
        cleaned_pages = []
        for page in pages:
            # Remove excessive whitespace and normalize line breaks
            cleaned = re.sub(r'\n+', '\n', page.strip())  # Multiple newlines to single
            cleaned = re.sub(r' +', ' ', cleaned)  # Multiple spaces to single
            cleaned_pages.append(cleaned)
        
        return "\n\n".join(cleaned_pages)
    except Exception as e:
        return f"Error extracting PDF content: {str(e)}"


def process_content_with_llm(content: str, api_key: str, progress_callback=None) -> str:
    """Process content using LLM for mathematical explanations."""
    
    if progress_callback:
        progress_callback("Analyzing mathematical content...")
    
    processed_content = content
    explanations_added = []
    
    # Find and process display math ($$...$$)
    display_math_pattern = r'\\$\\$(.*?)\\$\\$'
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
    inline_math_pattern = r'(?<!\\$)\\$([^$]+?)\\$(?!\\$)'
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
    theorem_pattern = r'\\\\begin\\{(theorem|proposition|lemma|corollary)\\}(.*?)\\\\end\\{\\1\\}'
    theorem_matches = list(re.finditer(theorem_pattern, processed_content, re.DOTALL | re.IGNORECASE))
    
    if theorem_matches and progress_callback:
        progress_callback(f"Processing {len(theorem_matches)} theorems/propositions...")
    
    # Process LaTeX structural elements
    processed_content = re.sub(r'\\\\section\\*?\\{([^}]*)\\}', r'Section: \\1', processed_content)
    processed_content = re.sub(r'\\\\subsection\\*?\\{([^}]*)\\}', r'Subsection: \\1', processed_content)
    processed_content = re.sub(r'\\\\title\\{([^}]*)\\}', r'Title: \\1', processed_content)
    processed_content = re.sub(r'\\\\author\\{([^}]*)\\}', r'Author: \\1', processed_content)
    
    # Clean up remaining LaTeX commands
    processed_content = re.sub(r'\\\\[a-zA-Z]+(?:\\[[^\\]]*\\])?(?:\\{[^}]*\\})*', ' ', processed_content)
    processed_content = re.sub(r'[{}]', '', processed_content)
    processed_content = re.sub(r'\\s+', ' ', processed_content)
    
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
                                        uploaded_images: List[Any] = None) -> str:
    """Create a comprehensive narration script with LLM explanations."""
    
    script_parts = []
    
    # Introduction
    script_parts.append(f"This is a narration of the uploaded {input_type.lower()} document with enhanced mathematical explanations.")
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(message: str):
        status_text.text(message)
    
    # Process mathematical content
    update_progress("Processing mathematical expressions...")
    processed_content = process_content_with_llm(content, api_key, update_progress)
    progress_bar.progress(70)
    
    # Process figures and tables
    update_progress("Processing figures and tables...")
    figure_explanations, table_explanations = process_figures_and_tables(content, api_key)
    progress_bar.progress(90)
    
    # Add processed content
    script_parts.append(processed_content)
    
    # Add figure explanations
    if figure_explanations:
        script_parts.append("The document includes the following figures:")
        script_parts.extend(figure_explanations)
    
    # Add table explanations
    if table_explanations:
        script_parts.append("The document includes the following tables:")
        script_parts.extend(table_explanations)
    
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
    
    return "\\n\\n".join(script_parts)


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
            help="e.g., 2301.12345 or https://arxiv.org/abs/2301.12345"
        )
        
    elif input_method == "Direct Text Input":
        direct_text = st.sidebar.text_area(
            "✏️ Paste Content",
            height=200,
            help="Paste your LaTeX, Markdown, or text with math expressions"
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
    
    # Generate button
    generate_button = st.sidebar.button(
        "🎯 Generate Advanced Narration",
        type="primary",
        disabled=not api_key,
        help="Requires OpenAI API key"
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
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enhanced Script")
        script_container = st.empty()
    
    with col2:
        st.subheader("🎵 Audio Output")
        audio_container = st.empty()
    
    # Process when generate button clicked
    if generate_button:
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
                            pdf_content = extract_pdf_content(tmp.name)
                            all_content.append(f"=== {file.name} ===\\n{pdf_content}")
                            os.unlink(tmp.name)
                        input_type = "PDF"
                    else:
                        file_content = file.read().decode('utf-8')
                        all_content.append(f"=== {file.name} ===\\n{file_content}")
                        input_type = file_ext.upper()
                
                content = "\\n\\n".join(all_content)
                
            elif input_method == "arXiv Download" and arxiv_input.strip():
                with st.spinner("📥 Downloading from arXiv..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        pdf_path = download_arxiv_pdf(arxiv_input.strip(), tmp.name)
                        if pdf_path:
                            content = extract_pdf_content(pdf_path)
                            input_type = "PDF"
                            os.unlink(tmp.name)
                        else:
                            # Extract ID for better error message
                            id_match = re.search(r"(?:abs|pdf)/(\d+\.\d+)(?:\.pdf)?", arxiv_input.strip())
                            arxiv_id = id_match.group(1) if id_match else arxiv_input.strip()
                            st.error(f"❌ Failed to download from arXiv. Paper ID: {arxiv_id}")
                            st.info("💡 Try again in a few minutes - arXiv servers may be temporarily unavailable.")
                            st.info("🔍 You can also try uploading the PDF directly if you have it downloaded.")
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
            
            # Generate enhanced script
            with st.spinner("🧠 Creating enhanced narration with LLM explanations..."):
                enhanced_script = create_comprehensive_narration_script(
                    content, input_type, api_key, uploaded_images
                )
            
            # Display editable script
            with col1:
                edited_script = st.text_area(
                    "✏️ Review and edit the enhanced script:",
                    value=enhanced_script,
                    height=400,
                    help="The script has been enhanced with natural language explanations of mathematical expressions"
                )
            
            # Generate audio
            if edited_script.strip():
                with st.spinner("🎵 Generating audio..."):
                    audio_dir = tempfile.mkdtemp(prefix="paper_voice_advanced_")
                    audio_ext = ".mp3" if use_openai_tts else ".wav"
                    audio_path = os.path.join(audio_dir, f"enhanced_narration{audio_ext}")
                    
                    try:
                        if use_openai_tts:
                            output_file = tts.synthesize_speech(
                                edited_script,
                                audio_path,
                                use_openai=True,
                                api_key=api_key,
                                openai_voice=openai_voice
                            )
                        else:
                            output_file = tts.synthesize_speech(
                                edited_script,
                                audio_path,
                                voice=offline_voice,
                                rate=speech_rate,
                                use_openai=False
                            )
                        
                        # Display audio
                        with col2:
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