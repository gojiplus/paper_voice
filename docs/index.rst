Paper Voice Documentation
========================

Paper Voice converts academic PDFs to audio with proper mathematical notation handling.

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

   pip install paper-voice

Quick Start
-----------

Basic usage:

.. code-block:: python

   from paper_voice import pdf_utils, math_to_speech
   
   # Extract text from PDF
   pages = pdf_utils.extract_raw_text("paper.pdf")
   
   # Process mathematical expressions
   processed = math_to_speech.process_text_with_math(pages[0])
   print(processed)

API Reference
=============

.. automodule:: paper_voice
   :members:

.. automodule:: paper_voice.pdf_utils
   :members:

.. automodule:: paper_voice.math_to_speech
   :members:

.. automodule:: paper_voice.llm_math_explainer
   :members:

.. automodule:: paper_voice.tts
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`