# Paper Voice

Convert academic papers to high-quality audio narration with precise mathematical explanations using a simplified LLM-powered approach.

## Features

- 🧮 **Natural Math Narration**: Professor-style explanations of mathematical expressions
- 📄 **Multi-Format Support**: PDFs, LaTeX, Markdown, and plain text with math notation
- 🎯 **Simple LLM Enhancement**: Single comprehensive prompt for natural audio conversion
- 🗣️ **Multiple TTS Options**: OpenAI TTS (with chunking) or offline pyttsx3
- 💻 **Web Interface**: Easy-to-use Streamlit web app
- ⚡ **Intelligent Chunking**: Handles large documents with smart OpenAI API limits

## Installation

### From PyPI (Recommended)

```bash
pip install paper_voice
```

### From Source

```bash
git clone https://github.com/gojiplus/paper_voice.git
cd paper_voice
pip install -e .
```

## Usage

### Web Interface (Recommended)

```bash
streamlit run paper_voice/streamlit/app.py
```

Upload a PDF, LaTeX file, or enter text directly. Provide an OpenAI API key for LLM-enhanced natural language conversion of mathematical expressions.

### Python API

#### Simple Enhancement (New in v0.3.0)

```python
from paper_voice.simple_llm_enhancer import enhance_document_simple

# Convert any academic content with math to natural language
content = "The equation $E = mc^2$ represents energy-mass equivalence."
enhanced = enhance_document_simple(content, api_key="your-openai-key")
print(enhanced)
# Output: "The equation energy equals mass times the speed of light squared represents energy-mass equivalence."
```

#### Complete Workflow

```python
from paper_voice import pdf_utils
from paper_voice.simple_llm_enhancer import enhance_document_simple
from paper_voice import tts

# 1. Extract text from PDF
pages = pdf_utils.extract_raw_text("paper.pdf")
content = '\n\n'.join(pages)

# 2. Enhance with LLM (converts math to natural language)
enhanced_script = enhance_document_simple(content, api_key="your-openai-key")

# 3. Generate audio
tts.synthesize_speech_chunked(
    enhanced_script, 
    "output.mp3", 
    use_openai=True, 
    api_key="your-openai-key"
)
```

#### LaTeX Processing

```python
from paper_voice.content_processor import process_content_unified

latex_content = r"""
\documentclass{article}
\begin{document}
The algorithm minimizes $J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$.
\end{document}
"""

processed = process_content_unified(
    content=latex_content,
    input_type='latex',
    api_key='your-openai-key',
    use_llm_enhancement=True
)

print(processed.enhanced_text)
```

## ✨ What's New in v0.3.0

### Simplified LLM Architecture
- **Single comprehensive prompt**: Handles all math conversion in one API call
- **Professor-style narration**: Natural explanations instead of robotic "subscript" language
- **Intelligent chunking**: Automatically handles large documents within OpenAI limits
- **Better error handling**: Clear failures instead of silent returns

### Natural Mathematical Explanations

**Before**: `$p_C$` → "p subscript C"

**After**: `$p_C$` → "p underscore capital C, the proportion of compliers"

**Complex expressions**: 
- `$F_{1C}$` → "F underscore one capital C, the outcome distribution for treated compliers"
- `$E = mc^2$` → "energy equals mass times the speed of light squared"

### Key API Changes
- Main function: `simple_llm_enhancer.enhance_document_simple()`
- Smart chunking for documents > 128K tokens
- Single LLM call for most documents
- Professor-style math conversion prompt

## Requirements

- Python 3.9+ (excluding 3.9.7)
- OpenAI API key (required for LLM enhancement)
- pydub (for audio chunking)
- PyPDF2 or PyMuPDF (for PDF processing)

### Optional Dependencies

```bash
# For better PDF processing
pip install PyMuPDF

# For offline TTS
pip install pyttsx3

# For audio format conversion
# Install ffmpeg via your system package manager
```

## Architecture

Paper Voice uses a clean modular pipeline:

**PDF → LaTeX/Markdown → LLM Enhancement → TTS**

1. **PDF Extraction**: Extract text with `pdf_utils.extract_raw_text()`
2. **LLM Enhancement**: Convert math to natural language with `simple_llm_enhancer.enhance_document_simple()`  
3. **Audio Generation**: Create audio with `tts.synthesize_speech_chunked()`

## Examples

### Basic Usage

```python
from paper_voice.simple_llm_enhancer import enhance_document_simple

# Simple math conversion
text = "The learning rate α controls convergence of $\\theta^* = \\arg\\min J(\\theta)$."
enhanced = enhance_document_simple(text, "your-api-key")
# Result: Natural professor-style explanation of the math
```

### With Progress Tracking

```python
def progress_callback(message):
    print(f"Progress: {message}")

enhanced = enhance_document_simple(
    content, 
    api_key, 
    progress_callback=progress_callback
)
```

### Large Document Handling

The system automatically handles large documents:
- Documents < 128K tokens: Single LLM call
- Documents > 128K tokens: Intelligent chunking with natural breakpoints

## Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-key-here"
```

Or pass it directly to functions:

```python
enhanced = enhance_document_simple(content, api_key="your-key")
```

## License

MIT License - see LICENSE file for details.