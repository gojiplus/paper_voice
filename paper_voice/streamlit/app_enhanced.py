"""
Enhanced Streamlit application for Paper Voice.

This app uses the new modular architecture and supports:
- PDF processing with enhanced extraction
- Direct LaTeX input
- Markdown with math processing
- Comprehensive mathematical expression support
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

# Import the new architecture
try:
    from paper_voice import (
        process_pdf_to_speech,
        process_latex_to_speech,
        process_markdown_to_speech,
        DocumentProcessor,
        ProcessingOptions
    )
    NEW_API_AVAILABLE = True
except ImportError:
    # Fallback to old API if new one isn't available
    from paper_voice import pdf_utils, math_to_speech, tts
    NEW_API_AVAILABLE = False

# Convenience function to download arXiv PDFs
def _download_arxiv_pdf(arxiv_id: str, dest_path: str) -> Optional[str]:
    """Download a PDF from arXiv given an ID or URL."""
    import re
    import requests
    
    # Extract ID from possible URL
    id_match = re.search(r"(?:abs|pdf)/([\d\.]+)(?:\.pdf)?", arxiv_id)
    if id_match:
        arxiv_id = id_match.group(1)
    
    # Construct download URL
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and resp.headers.get('content-type', '').startswith('application/pdf'):
            with open(dest_path, 'wb') as f:
                f.write(resp.content)
            return dest_path
    except Exception:
        pass
    return None


def main() -> None:
    st.set_page_config(page_title="Paper Voice Enhanced", layout="wide", page_icon="📄")
    
    st.title("📄🔊 Paper Voice Enhanced")
    st.markdown("""
    **Paper Voice** transforms academic content into spoken audio with comprehensive mathematical support.
    
    🆕 **New Features:**
    - Enhanced mathematical expression processing (100+ operators, Greek letters, LaTeX environments)
    - Direct LaTeX and Markdown input support
    - Improved PDF extraction with multiple backends
    - Better TTS options and voice control
    """)
    
    # Sidebar for input method selection
    st.sidebar.header("📝 Input Method")
    input_method = st.sidebar.selectbox(
        "Choose input type:",
        ["PDF Upload/arXiv", "LaTeX Content", "Markdown with Math"],
        help="Select how you want to provide the academic content"
    )
    
    # Common settings
    st.sidebar.header("🎛️ Processing Options")
    use_openai = st.sidebar.checkbox(
        "Use OpenAI for summarization and TTS",
        value=False,
        help="Enable LLM summarization of figures/tables and high-quality TTS"
    )
    
    api_key = ""
    if use_openai:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for OpenAI features"
        )
    
    # TTS Settings
    st.sidebar.subheader("🔊 Text-to-Speech")
    tts_backend = st.sidebar.selectbox(
        "TTS Backend",
        ["offline", "openai"] if use_openai else ["offline"],
        help="Choose speech synthesis method"
    )
    
    if tts_backend == "offline":
        offline_voice = st.sidebar.text_input(
            "Voice name (optional)",
            help="Part of voice name (e.g., 'Samantha', 'David')"
        )
        speech_rate = st.sidebar.slider(
            "Speech rate (WPM)",
            min_value=100, max_value=300, value=180, step=10
        )
    else:
        openai_voice = st.sidebar.selectbox(
            "OpenAI Voice",
            ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            index=0
        )
    
    # Advanced options
    st.sidebar.subheader("⚙️ Advanced Options")
    include_equations = st.sidebar.checkbox("Include standalone equations", value=True)
    include_figures = st.sidebar.checkbox("Include figure descriptions", value=True)
    include_tables = st.sidebar.checkbox("Include table content", value=True)
    
    # PDF-specific options
    if input_method == "PDF Upload/arXiv":
        pdf_method = st.sidebar.selectbox(
            "PDF extraction method",
            ["auto", "pymupdf", "pypdf2"],
            help="PyMuPDF is better for complex layouts"
        )
    
    # Main content area
    content = None
    pdf_path = None
    
    if input_method == "PDF Upload/arXiv":
        st.header("📄 PDF Input")
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                help="Upload an academic paper in PDF format"
            )
        
        with col2:
            arxiv_input = st.text_input(
                "Or enter arXiv ID/URL",
                placeholder="e.g., 1706.03762 or https://arxiv.org/abs/1706.03762",
                help="Download directly from arXiv"
            )
        
        # Handle PDF input
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name
                st.success(f"PDF uploaded: {uploaded_file.name}")
        
        elif arxiv_input.strip():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                with st.spinner("Downloading from arXiv..."):
                    downloaded_path = _download_arxiv_pdf(arxiv_input.strip(), tmp.name)
                if downloaded_path:
                    pdf_path = downloaded_path
                    st.success(f"Downloaded arXiv paper: {arxiv_input}")
                else:
                    st.error("Failed to download PDF. Please check the arXiv ID/URL.")
    
    elif input_method == "LaTeX Content":
        st.header("📝 LaTeX Input")
        st.markdown("Enter your LaTeX content below. Mathematical expressions, equations, and environments are fully supported.")
        
        content = st.text_area(
            "LaTeX Content",
            height=300,
            placeholder="""\\documentclass{article}
\\title{My Paper}
\\author{Author Name}

\\begin{document}
\\maketitle

\\section{Introduction}
This paper discusses $E = mc^2$ and other equations.

\\begin{equation}
\\int_0^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}
\\end{equation}

\\end{document}""",
            help="Full LaTeX document or content excerpt"
        )
    
    elif input_method == "Markdown with Math":
        st.header("📝 Markdown Input")
        st.markdown("Enter Markdown content with LaTeX math expressions using `$...$` and `$$...$$`.")
        
        content = st.text_area(
            "Markdown Content",
            height=300,
            placeholder="""# My Research Paper

## Introduction

The famous equation is $E = mc^2$, which shows the mass-energy equivalence.

## Mathematical Analysis

The Gaussian integral is given by:

$$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$

This result has applications in probability theory and quantum mechanics.""",
            help="Markdown with inline ($...$) and display ($$...$$) math"
        )
    
    # Generate button and processing
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button("🎤 Generate Speech", use_container_width=True)
    
    if generate_button:
        if input_method == "PDF Upload/arXiv" and not pdf_path:
            st.error("Please upload a PDF or enter an arXiv ID first.")
        elif input_method in ["LaTeX Content", "Markdown with Math"] and not content:
            st.error(f"Please enter {input_method.lower()} content first.")
        else:
            # Set up processing options
            options = {
                'use_summarization': use_openai and bool(api_key),
                'openai_api_key': api_key if use_openai else None,
                'include_figures': include_figures,
                'include_tables': include_tables,
                'include_equations': include_equations,
                'tts_backend': tts_backend,
            }
            
            if tts_backend == "offline":
                options.update({
                    'tts_voice': offline_voice,
                    'tts_rate': speech_rate,
                })
            else:
                options['tts_voice_openai'] = openai_voice
            
            if input_method == "PDF Upload/arXiv":
                options['pdf_extraction_method'] = None if pdf_method == "auto" else pdf_method
            
            # Process the content
            with st.spinner("Processing content..."):
                try:
                    # Create temporary audio file
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
                        audio_path = tmp_audio.name
                    
                    if NEW_API_AVAILABLE:
                        # Use new API
                        if input_method == "PDF Upload/arXiv":
                            result = process_pdf_to_speech(pdf_path, audio_path, **options)
                        elif input_method == "LaTeX Content":
                            result = process_latex_to_speech(content, audio_path, **options)
                        else:  # Markdown
                            result = process_markdown_to_speech(content, audio_path, **options)
                        
                        # Display results
                        st.success("✅ Processing completed!")
                        
                        # Show processing log
                        with st.expander("📋 Processing Log"):
                            for log_entry in result.processing_log:
                                st.text(log_entry)
                        
                        # Display script preview
                        st.subheader("📝 Generated Script")
                        script_preview = result.spoken_text[:1000] + "..." if len(result.spoken_text) > 1000 else result.spoken_text
                        st.text_area("Script Preview", value=script_preview, height=200, disabled=True)
                        
                        # Audio player and download
                        st.subheader("🎧 Generated Audio")
                        if os.path.exists(audio_path):
                            st.audio(audio_path)
                            
                            with open(audio_path, "rb") as audio_file:
                                st.download_button(
                                    label="⬇️ Download Audio",
                                    data=audio_file.read(),
                                    file_name="paper_voice_output.mp3",
                                    mime="audio/mpeg"
                                )
                        
                        # Show additional info
                        if result.processed_content.equations and include_equations:
                            with st.expander(f"🧮 Equations Found ({len(result.processed_content.equations)})"):
                                for i, eq in enumerate(result.processed_content.equations, 1):
                                    st.text(f"{i}. {eq}")
                        
                        if result.processed_content.figures and include_figures:
                            with st.expander(f"🖼️ Figures Found ({len(result.processed_content.figures)})"):
                                for caption, desc in result.processed_content.figures:
                                    st.text(f"• {caption}")
                    
                    else:
                        # Fallback to old API (simplified)
                        st.error("New API not available. Please check installation.")
                
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    st.error("Please check your input and try again.")

    # Footer with info
    st.markdown("---")
    st.markdown("""
    **Paper Voice Enhanced** - Transform academic content into speech with comprehensive mathematical support.
    
    **Features:**
    - 📊 100+ mathematical operators and symbols
    - 🔤 Greek letters and special notation  
    - 🧮 LaTeX environments (equation, align, matrix, etc.)
    - 📄 Multiple input formats (PDF, LaTeX, Markdown)
    - 🎙️ Offline and cloud TTS options
    
    For more examples, see `examples/enhanced_usage.py`
    """)


if __name__ == "__main__":
    main()