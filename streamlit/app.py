#!/usr/bin/env python3
"""
Simple Streamlit app for paper_voice using the simplified LLM approach.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Any
import warnings

# Suppress pydub warnings
warnings.filterwarnings("ignore", message=".*invalid escape sequence.*")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")

try:
    import pydub
    import pydub.silence
    import pydub.utils
    import pydub.exceptions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pydub
except ImportError:
    pass

# Import our simplified modules
try:
    from paper_voice import pdf_utils, tts, simple_llm_enhancer
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from paper_voice import pdf_utils, tts, simple_llm_enhancer


def extract_pdf_content(pdf_path: str) -> str:
    """Extract text content from PDF."""
    try:
        pages = pdf_utils.extract_raw_text(pdf_path)
        return '\n\n'.join(page for page in pages if page.strip())
    except Exception as e:
        return f"Error extracting PDF content: {str(e)}"




def create_audio_from_text(text: str, output_path: str, voice: str = "alloy", 
                          api_key: str = None) -> bool:
    """Create audio file from text using TTS."""
    try:
        if not api_key:
            st.error("No API key provided for TTS")
            return False
            
        # Use chunked synthesis for long texts
        result_path = tts.synthesize_speech_chunked(
            text=text,
            output_path=output_path, 
            use_openai=True,
            api_key=api_key,
            openai_voice=voice
        )
        return result_path == output_path
    except Exception as e:
        st.error(f"TTS failed: {str(e)}")
        return False


def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Paper Voice - Simple",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Paper Voice - Simple LLM Approach")
    st.markdown("Convert academic papers to audio using simplified LLM processing")
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    # API key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password", 
        help="Required for LLM enhancement and TTS"
    )
    
    # Voice selection
    voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    selected_voice = st.sidebar.selectbox("🗣️ Voice", voice_options, index=0)
    
    # Debug mode
    show_debug = st.sidebar.checkbox("🔍 Debug Mode", value=False)
    
    # Process input first (before columns)
    content = ""
    input_type = "text"
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload PDF or text file",
        type=['pdf', 'txt', 'tex'],
        help="Upload a PDF, text file, or LaTeX file"
    )
    
    # Text input area
    text_input = st.text_area(
        "Or paste text/LaTeX directly:",
        height=200,
        placeholder="Paste your LaTeX or text content here..."
    )
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            # Handle PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file.flush()
                content = extract_pdf_content(tmp_file.name)
                input_type = "PDF"
            os.unlink(tmp_file.name)
        else:
            # Handle text/LaTeX files
            content = uploaded_file.read().decode('utf-8')
            input_type = "LaTeX" if uploaded_file.name.endswith('.tex') else "text"
    elif text_input.strip():
        content = text_input
        input_type = "LaTeX" if "\\section" in content or "\\(" in content else "text"
    
    if content:
        st.success(f"✅ Loaded {len(content)} characters of {input_type} content")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Input Status")
        if content:
            st.info(f"Content loaded: {len(content)} characters of {input_type}")
        else:
            st.warning("No content loaded yet")
    
    with col2:
        st.header("🎯 Output")
        
        # Step 1: Generate Enhanced Script
        if st.button("📝 Step 1: Generate Enhanced Script", type="primary", disabled=not api_key):
            if not content:
                st.error("Please provide content first!")
            else:
                st.info(f"🔧 Processing {len(content)} chars with API key: {api_key[:10]}...")
                
                try:
                    with st.spinner("Enhancing content with LLM..."):
                        def progress_callback(message):
                            st.info(f"📝 {message}")
                        
                        # Call the package function directly
                        enhanced_script = simple_llm_enhancer.enhance_document_simple(
                            content, api_key, progress_callback
                        )
                    
                    # Success! Store the enhanced script
                    st.session_state['enhanced_script'] = enhanced_script
                    st.success("✅ Enhancement completed successfully!")
                    st.code(f"Sample result: {enhanced_script[:300]}...")
                    
                except Exception as e:
                    st.error(f"❌ ENHANCEMENT FAILED: {str(e)}")
                    st.warning("⚠️ The original content will NOT be stored as 'enhanced'")
                    
                    # Show the error details
                    if "Math expressions not converted" in str(e):
                        st.error("The LLM failed to convert LaTeX math expressions")
                    elif "API key" in str(e).lower():
                        st.error("Check your OpenAI API key")
                    elif "rate_limit" in str(e).lower():
                        st.error("Rate limit hit - try again in a moment")
                    
                    # DO NOT store failed content
                    if 'enhanced_script' in st.session_state:
                        del st.session_state['enhanced_script']
        
        # Display enhanced script
        if 'enhanced_script' in st.session_state:
            st.subheader("📋 Enhanced Script")
            enhanced_script = st.text_area(
                "Enhanced script (editable):",
                value=st.session_state['enhanced_script'],
                height=300,
                key="script_editor"
            )
            
            # Update session state if user edits
            st.session_state['enhanced_script'] = enhanced_script
            
            # Download button for enhanced script
            st.download_button(
                label="📥 Download Enhanced Script",
                data=enhanced_script,
                file_name="enhanced_script.txt",
                mime="text/plain",
                help="Download the enhanced script as a text file"
            )
            
            # Step 2: Convert to Audio
            if st.button("🔊 Step 2: Convert to Audio", disabled=not api_key):
                if not enhanced_script.strip():
                    st.error("Enhanced script is empty!")
                else:
                    with st.spinner("Generating audio..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                            success = create_audio_from_text(
                                enhanced_script, tmp_audio.name, selected_voice, api_key
                            )
                            
                            if success:
                                # Read the audio file
                                with open(tmp_audio.name, 'rb') as audio_file:
                                    audio_bytes = audio_file.read()
                                
                                st.success("✅ Audio generated successfully!")
                                st.audio(audio_bytes, format='audio/mp3')
                                
                                # Download button
                                st.download_button(
                                    label="⬇️ Download Audio",
                                    data=audio_bytes,
                                    file_name="paper_voice_output.mp3",
                                    mime="audio/mp3"
                                )
                            
                            # Clean up
                            os.unlink(tmp_audio.name)
    
    # Footer
    st.markdown("---")
    st.markdown("**Paper Voice** - Simplified LLM approach for academic paper narration")


if __name__ == "__main__":
    main()