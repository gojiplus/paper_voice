# Paper Voice

Convert academic papers to high-quality audio narration with precise mathematical explanations, enhanced figure descriptions, and intelligent content processing.

## Features

- 🧮 **Precise Math Explanations**: LLM-powered contextual explanations of mathematical expressions
- 🖼️ **Enhanced Figure Descriptions**: AI-generated audio-friendly descriptions of figures and tables  
- 📄 **Multi-Format Support**: PDFs, LaTeX, Markdown, and plain text with math notation
- 🔗 **ArXiv Integration**: Direct download and processing of LaTeX source with figures
- ⚡ **Batch Processing**: Process multiple papers simultaneously with parallel execution
- 🎯 **Selective Enhancement**: Preserves original text while enhancing only math, figures, and tables
- 🗣️ **Multiple TTS Options**: OpenAI TTS (with chunking) or offline pyttsx3
- 💻 **CLI & Web Interface**: Full command-line interface plus Streamlit web app
- 👁️ **Vision Analysis**: Optional GPT-4V integration for superior PDF content extraction

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

### Command Line Interface

Paper Voice includes a comprehensive CLI for batch processing and automation:

```bash
# Single file processing
paper_voice paper.pdf --api-key YOUR_KEY

# Process arXiv paper
paper_voice https://arxiv.org/abs/2301.12345 --api-key YOUR_KEY --output research.mp3

# LaTeX file with custom voice
paper_voice paper.tex --latex --api-key YOUR_KEY --voice nova

# Vision-enhanced PDF analysis
paper_voice paper.pdf --vision --api-key YOUR_KEY --output enhanced.mp3

# Batch processing a directory
paper_voice --batch papers/ --api-key YOUR_KEY --output-dir ./audio_output

# Batch processing multiple files
paper_voice --batch paper1.pdf paper2.pdf paper3.tex --api-key YOUR_KEY --output-dir ./batch_output
```

#### CLI Options

- `--batch`: Enable batch processing for multiple files/directories
- `--vision`: Use GPT-4V for enhanced PDF analysis
- `--voice`: Choose TTS voice (alloy, echo, fable, onyx, nova, shimmer)
- `--speed`: Adjust speech speed (0.25 to 4.0)
- `--max-workers`: Set concurrent workers for batch processing (default: 3)
- `--output-dir`: Output directory for batch processing
- `--offline`: Use offline TTS instead of OpenAI TTS
- `--no-enhancement`: Skip LLM enhancement for faster processing

### Web Interface

```bash
streamlit run streamlit/app.py
```

Upload a PDF, LaTeX file, or enter text directly. For best results with mathematical content, provide an OpenAI API key to enable LLM-powered explanations.

### Python API

```python
from paper_voice import pdf_utils, math_to_speech
from paper_voice.arxiv_downloader import download_arxiv_paper

# Extract text from PDF
pages = pdf_utils.extract_raw_text("paper.pdf")

# Process mathematical expressions
processed = math_to_speech.process_text_with_math(pages[0])
print(processed)

# Download from arXiv
paper = download_arxiv_paper("2301.12345")
if paper:
    print(f"Downloaded: {paper.title}")
    print(f"LaTeX content: {len(paper.latex_content)} characters")
```

## ✨ What's New in v0.2.0

### Precise Mathematical Explanations
Instead of basic conversions like "alpha squared plus beta", Paper Voice now generates contextual explanations:

**Before**: `$\alpha^2 + \beta = \gamma$` → "alpha squared plus beta equals gamma"

**After**: `$\alpha^2 + \beta = \gamma$` → "In machine learning context, this equation shows that the outcome gamma is determined by the sum of the learning rate alpha squared and the regularization parameter beta..."

### Selective Enhancement
- ✅ **Only enhances**: Math expressions, figure captions, table descriptions
- ✅ **Preserves exactly**: All other academic text, structure, and content
- ❌ **No summarization**: Original text remains unchanged

### Enhanced LaTeX Processing
LaTeX files now get full LLM enhancement while preserving document structure:
- Mathematical expressions → Contextual explanations
- Figure captions → Audio-friendly descriptions  
- Table content → Clear narration
- Regular text → Preserved exactly as written

## Requirements

- Python 3.9+ (excluding 3.9.7)
- OpenAI API key (optional but recommended for enhanced explanations)
- ffmpeg (for audio processing)

## Advanced Features

### Batch Processing

Process multiple papers efficiently with parallel processing:

```bash
# Process all PDFs in a directory
paper_voice --batch research_papers/ --api-key YOUR_KEY --output-dir audio_papers/

# Process specific files with custom settings
paper_voice --batch paper1.pdf paper2.tex paper3.pdf \
  --api-key YOUR_KEY \
  --output-dir batch_output/ \
  --max-workers 5 \
  --voice nova \
  --vision
```

### ArXiv Integration

Download and process papers directly from arXiv:

```bash
# Single arXiv paper
paper_voice https://arxiv.org/abs/2301.12345 --api-key YOUR_KEY

# Multiple arXiv papers
paper_voice --batch \
  https://arxiv.org/abs/2301.12345 \
  https://arxiv.org/abs/2302.67890 \
  --api-key YOUR_KEY \
  --output-dir arxiv_audio/
```

### Vision-Enhanced PDF Analysis

Use GPT-4V for superior content extraction:

```bash
paper_voice complex_paper.pdf --vision --api-key YOUR_KEY
```

## Examples

See the `demos/` directory for usage examples:
- `demos/basic_usage.py` - Simple math processing examples
- `demos/before_after_comparison.py` - Shows improvement from LLM explanations

See the `tests/` directory for comprehensive test cases including batch processing tests.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and how to contribute to the project.