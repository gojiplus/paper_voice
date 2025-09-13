Paper Voice Documentation
========================

Paper Voice converts academic papers to high-quality audio narration with precise mathematical explanations using a simplified LLM-powered approach.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Installation
------------

.. code-block:: bash

   pip install paper_voice

Quick Start
-----------

Basic usage with the simplified API:

.. code-block:: python

   from paper_voice.simple_llm_enhancer import enhance_document_simple
   from paper_voice import pdf_utils, tts
   
   # Extract text from PDF
   pages = pdf_utils.extract_raw_text("paper.pdf")
   content = '\n\n'.join(pages)
   
   # Convert math to natural language
   enhanced_script = enhance_document_simple(content, api_key="your-openai-key")
   
   # Generate audio
   tts.synthesize_speech_chunked(
       enhanced_script, 
       "output.mp3", 
       use_openai=True, 
       api_key="your-openai-key"
   )

Web Interface
-----------

.. code-block:: bash

   streamlit run paper_voice/streamlit/app.py

Key Features
============

* **Natural Math Narration**: Professor-style explanations of mathematical expressions
* **Single LLM Enhancement**: Comprehensive prompt handles all math conversion in one API call  
* **Intelligent Chunking**: Automatically handles large documents within OpenAI API limits
* **Multi-Format Support**: PDFs, LaTeX, Markdown, and plain text with math notation
* **Multiple TTS Options**: OpenAI TTS with chunking or offline pyttsx3

API Reference
=============

Core Functions
--------------

.. automodule:: paper_voice.simple_llm_enhancer
   :members:

.. automodule:: paper_voice.pdf_utils
   :members:

.. automodule:: paper_voice.tts
   :members:

Content Processing
------------------

.. automodule:: paper_voice.content_processor
   :members:

.. automodule:: paper_voice.latex_processor
   :members:

Selective Enhancement
---------------------

.. automodule:: paper_voice.selective_enhancer
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`