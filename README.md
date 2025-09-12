# Paper Voice

Convert academic PDFs to audio with proper mathematical notation handling.

## Features

- **Math handling**: Converts LaTeX expressions to natural speech (e.g., `$\alpha^2$` → "alpha squared")
- **LLM-powered explanations**: Uses GPT-4 to explain complex mathematical expressions clearly
- **Figure/table summaries**: Extracts and summarizes figure captions and table content
- **Multiple TTS options**: Offline (pyttsx3) or OpenAI TTS
- **Direct LaTeX/Markdown support**: Process text with math notation directly
- **Web interface**: Streamlit app for easy file uploads and processing

## Installation

```bash
git clone <your-repo-url>
cd paper_voice
pip install -e .
```

## Usage

### Web Interface

```bash
streamlit run paper_voice/streamlit/app.py
```

Upload a PDF, LaTeX file, or enter text directly. For best results with mathematical content, provide an OpenAI API key to enable LLM-powered explanations.

### Command Line

```python
from paper_voice import pdf_utils, math_to_speech

# Extract text from PDF
pages = pdf_utils.extract_raw_text("paper.pdf")

# Process mathematical expressions
processed = math_to_speech.process_text_with_math(pages[0])
print(processed)
```

## Requirements

- Python 3.9+ (excluding 3.9.7)
- OpenAI API key (optional, for enhanced math explanations)
- ffmpeg (for audio processing)

## Applications

There are several Streamlit applications available:

- `paper_voice/streamlit/app.py` - Basic PDF to audio conversion
- `paper_voice/streamlit/app_with_llm.py` - Enhanced with LLM math explanations
- `paper_voice/streamlit/app_enhanced.py` - Full-featured version with LaTeX/Markdown support

## Examples

See the `demos/` directory for usage examples:
- `demos/basic_usage.py` - Simple math processing examples
- `demos/before_after_comparison.py` - Shows improvement from LLM explanations

See the `tests/` directory for test cases.