"""
Abstractions for synthesising speech from text.

This module exposes a simple API over two different TTS backends:

1. **Offline TTS via `pyttsx3`**: Available out of the box and does not
   require network access. Produces a WAV file. To convert to MP3, this
   module uses `pydub`, which requires that ``ffmpeg`` be installed on
   the system. On many Linux distributions you can install ffmpeg via
   your package manager.

2. **OpenAI Text‑to‑Speech API**: Requires a valid API key. Produces an
   MP3 directly. The OpenAI API currently supports a limited set of
   voices and languages but generally yields higher quality output.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None  # type: ignore

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    OpenAI = None  # type: ignore
    openai_available = False

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None  # type: ignore


def _ensure_ffmpeg():
    """Ensure that ffmpeg is available for pydub MP3 conversion.

    If ffmpeg is not available, this function attempts to set the
    ``AudioSegment.converter`` path to a default location. If conversion
    fails later, the user should install ffmpeg.
    """
    if AudioSegment is None:
        return
    # On many systems, ffmpeg is available in PATH. If not, you can set
    # AudioSegment.converter directly. Here we simply rely on default.
    pass


def _split_text_for_tts(text: str, max_length: int = 4000) -> list[str]:
    """Split text into chunks for TTS processing.
    
    Splits text at sentence boundaries to avoid cutting off in the middle of sentences.
    OpenAI TTS has a 4096 character limit, so we use 4000 to be safe.
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by sentences first
    sentences = text.split('. ')
    
    for i, sentence in enumerate(sentences):
        # Add back the period except for the last sentence
        if i < len(sentences) - 1:
            sentence += '. '
        
        # Check if adding this sentence would exceed the limit
        if len(current_chunk + sentence) > max_length:
            if current_chunk:  # If we have content, save it
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:  # Single sentence is too long, split by words
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk + " " + word) > max_length:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            # Single word is too long, just add it
                            chunks.append(word)
                    else:
                        temp_chunk += " " + word if temp_chunk else word
                if temp_chunk:
                    current_chunk = temp_chunk
        else:
            current_chunk += sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def synthesize_speech_chunked(
    text: str,
    output_path: str,
    voice: str = "",
    rate: int = 200,
    use_openai: bool = False,
    api_key: Optional[str] = None,
    model: str = "tts-1",
    openai_voice: str = "alloy",
) -> str:
    """Synthesize speech with automatic text chunking for long texts.
    
    This function handles texts longer than the OpenAI TTS limit by splitting
    them into chunks and concatenating the audio files.
    """
    if use_openai and len(text) > 4000:
        # Split text into chunks
        chunks = _split_text_for_tts(text)
        
        if len(chunks) == 1:
            # No splitting needed, use regular function
            return synthesize_speech(text, output_path, voice, rate, use_openai, 
                                   api_key, model, openai_voice)
        
        # Generate audio for each chunk
        temp_files = []
        temp_dir = tempfile.mkdtemp(prefix="tts_chunks_")
        
        try:
            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.mp3")
                synthesize_speech(chunk, chunk_path, voice, rate, use_openai,
                               api_key, model, openai_voice)
                temp_files.append(chunk_path)
            
            # Concatenate audio files
            if AudioSegment is None:
                raise RuntimeError("pydub is required for concatenating audio chunks")
            
            combined = AudioSegment.empty()
            for temp_file in temp_files:
                chunk_audio = AudioSegment.from_mp3(temp_file)
                combined += chunk_audio
                # Add small pause between chunks
                combined += AudioSegment.silent(duration=500)  # 500ms pause
            
            # Export final file
            combined.export(output_path, format="mp3")
            
            return output_path
            
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    else:
        # Use regular synthesis
        return synthesize_speech(text, output_path, voice, rate, use_openai,
                               api_key, model, openai_voice)


def synthesize_speech(
    text: str,
    output_path: str,
    voice: str = "",
    rate: int = 200,
    use_openai: bool = False,
    api_key: Optional[str] = None,
    model: str = "tts-1",
    openai_voice: str = "alloy",
) -> str:
    """Synthesize speech from text and write it to a file.

    Parameters
    ----------
    text: str
        The input text to speak.
    output_path: str
        Path where the audio file should be written. The file extension
        determines the format. For offline synthesis this should end
        with ``.wav``; for OpenAI TTS this should end with ``.mp3``.
    voice: str, optional
        A substring of the desired voice name when using ``pyttsx3``.
        On Windows you might use 'Zira' or 'David', on macOS 'Samantha'
        etc. If empty, the default voice is used.
    rate: int, optional
        Words per minute for offline synthesis. Typical values are 150–250.
    use_openai: bool, optional
        If true, attempts to synthesise via the OpenAI API. Requires
        ``openai`` package and a valid API key. The output format will
        always be MP3 in this mode.
    api_key: str, optional
        The OpenAI API key to use. If not provided, uses the
        ``OPENAI_API_KEY`` environment variable.
    model: str, optional
        The OpenAI TTS model to use. At time of writing 'tts-1' and
        'tts-1-hd' are available.
    openai_voice: str, optional
        The voice name for OpenAI TTS (e.g. 'alloy', 'echo').

    Returns
    -------
    str
        The path to the generated audio file. In offline mode this may
        differ from ``output_path`` if an intermediate WAV is created and
        later converted.

    Raises
    ------
    RuntimeError
        If required libraries are missing or synthesis fails.
    """
    if use_openai:
        # Use OpenAI TTS
        if not openai_available:
            raise RuntimeError("openai package is not installed; cannot use OpenAI TTS.")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No OpenAI API key provided for TTS.")
        # Send the request
        try:
            client = OpenAI(api_key=api_key)
            response = client.audio.speech.create(
                input=text,
                model=model,
                voice=openai_voice,
            )
            # Save the binary MP3
            with open(output_path, "wb") as f:
                f.write(response.content)
            return output_path
        except Exception as exc:
            raise RuntimeError(f"OpenAI TTS failed: {exc}")
    else:
        # Use offline TTS
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is not installed; cannot use offline TTS.")
        engine = pyttsx3.init()
        if voice:
            # Attempt to select a voice matching the substring
            voices = engine.getProperty('voices') or []
            for v in voices:
                if hasattr(v, 'name') and voice.lower() in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break
        engine.setProperty('rate', rate)
        # Determine file extension; offline TTS only writes WAV
        ext = os.path.splitext(output_path)[1].lower()
        if ext != '.wav':
            # We'll create a temporary WAV and later convert to desired format
            tmp_dir = tempfile.mkdtemp(prefix="speech_")
            tmp_wav = os.path.join(tmp_dir, "out.wav")
            write_path = tmp_wav
        else:
            write_path = output_path
        engine.save_to_file(text, write_path)
        engine.runAndWait()
        # Convert to MP3 if needed
        if ext != '.wav':
            if AudioSegment is None:
                raise RuntimeError(
                    "pydub is not installed; cannot convert WAV to MP3. "
                    "Install pydub and ensure ffmpeg is available."
                )
            _ensure_ffmpeg()
            audio = AudioSegment.from_wav(write_path)
            audio.export(output_path, format="mp3")
        return output_path