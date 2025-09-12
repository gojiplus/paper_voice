"""
Streamlit application for paper_voice.

This app allows users to upload a PDF (or provide an arXiv identifier),
extracts the text while respecting mathematical notation, figures and
tables, and then synthesises the narration into an audio file. Users can
optionally supply an OpenAI API key to enable LLM‑based summarisation
and text‑to‑speech.

Run this app with:

```
streamlit run paper_voice/app.py
```

"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

from . import pdf_utils, math_to_speech, figure_table_summarizer, tts

# Convenience function to download arXiv PDFs; if network unavailable, returns None
def _download_arxiv_pdf(arxiv_id: str, dest_path: str) -> Optional[str]:
    """Attempt to download a PDF from arXiv given an ID or URL.

    The function extracts the identifier from a URL if necessary, and
    downloads the file to ``dest_path``. Returns the local path on
    success, or None on failure.
    """
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
    st.set_page_config(page_title="Paper Voice", layout="wide")
    st.title("📄🔊 Paper Voice")
    st.markdown(
        """
        **Paper Voice** transforms technical papers into spoken audio. It
        understands mathematical notation, extracts figure and table
        captions, and can optionally use the OpenAI API to summarise
        complex content and synthesise natural‑sounding speech.
        """
    )

    # Sidebar inputs
    st.sidebar.header("Upload or Download PDF")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF", type=["pdf"], help="Choose a PDF file from your computer"
    )
    arxiv_input = st.sidebar.text_input(
        "Or enter an arXiv ID/URL", value="", help="e.g. 1706.03762 or https://arxiv.org/abs/1706.03762"
    )
    use_openai = st.sidebar.checkbox(
        "Use OpenAI for summarisation and TTS",
        value=False,
    )
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value="",
        type="password",
        help="Needed if using OpenAI summarisation or TTS",
    )
    offline_voice = st.sidebar.text_input(
        "Offline voice (substring)",
        value="",
        help="If using offline TTS, enter part of the voice name (optional)"
    )
    rate = st.sidebar.slider(
        "Offline speech rate (words per minute)",
        min_value=100,
        max_value=300,
        value=200,
        step=10,
    )
    generate_button = st.sidebar.button("Generate Narration")

    # Work area: placeholder for script and audio
    script_container = st.container()
    audio_container = st.container()

    # Determine PDF source
    pdf_path: Optional[str] = None
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name
    elif arxiv_input.strip():
        # Try to download from arXiv
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            local = _download_arxiv_pdf(arxiv_input.strip(), tmp.name)
            if local:
                pdf_path = local
            else:
                st.error("Failed to download the specified arXiv PDF. Please check the ID/URL.")

    # If a PDF is available and the user requested generation
    if generate_button and pdf_path:
        with st.spinner("Extracting text and generating narration script..."):
            try:
                pages = pdf_utils.extract_raw_text(pdf_path)
            except Exception as exc:
                st.error(f"Failed to read PDF: {exc}")
                return
            narration_parts = []
            # Process each page
            for page_num, page_text in enumerate(pages, start=1):
                # Summarise captions
                captions = pdf_utils.extract_captions(page_text)
                for kind, caption in captions:
                    if use_openai and api_key:
                        summary = figure_table_summarizer.summarise_caption(
                            caption, kind=kind, api_key=api_key
                        )
                    else:
                        summary = caption
                    narration_parts.append(f"{kind.title()} summary: {summary}")
                # Summarise tables
                tables = pdf_utils.extract_table_text(page_text)
                if tables:
                    if use_openai and api_key:
                        table_summary = figure_table_summarizer.summarise_table(
                            tables, api_key=api_key
                        )
                    else:
                        table_summary = " ".join(tables)
                    narration_parts.append(f"Table summary: {table_summary}")
                # Process main text with math
                processed = math_to_speech.process_text_with_math(page_text)
                narration_parts.append(processed)
            script_text = "\n\n".join(narration_parts)
        # Display script and allow editing
        script = script_container.text_area(
            "Narration Script (edit if necessary)",
            value=script_text,
            height=300,
        )
        # Synthesise speech if script provided
        if script.strip():
            with st.spinner("Synthesising speech..."):
                # Determine file path for audio in a temp directory
                tmp_dir = tempfile.mkdtemp(prefix="paper_voice_audio_")
                # Use mp3 if using openai or offline output for user convenience
                ext = ".mp3" if use_openai else ".wav"
                audio_path = os.path.join(tmp_dir, f"narration{ext}")
                try:
                    output_file = tts.synthesize_speech(
                        script,
                        audio_path,
                        voice=offline_voice,
                        rate=rate,
                        use_openai=use_openai,
                        api_key=api_key or None,
                    )
                    # Display audio player
                    audio_container.audio(output_file)
                    # Provide download
                    with open(output_file, "rb") as f:
                        audio_bytes = f.read()
                    st.download_button(
                        label="Download Audio",
                        data=audio_bytes,
                        file_name=os.path.basename(output_file),
                        mime="audio/mpeg" if ext == ".mp3" else "audio/wav",
                    )
                except Exception as exc:
                    st.error(f"Speech synthesis failed: {exc}")
    elif generate_button and not pdf_path:
        st.warning("Please upload a PDF or provide a valid arXiv ID before generating.")


if __name__ == "__main__":
    main()