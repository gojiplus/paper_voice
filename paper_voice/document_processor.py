"""
Main document processing pipeline.

This module orchestrates the complete pipeline:
1. PDF -> LaTeX/Markdown extraction (pdf_extractor)
2. LaTeX/Markdown -> Spoken text processing (latex_processor)
3. Optional figure/table summarization
4. Text-to-speech synthesis (tts)
"""

from __future__ import annotations

from typing import Optional, Union, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from .pdf_extractor import extract_pdf_to_latex, ExtractedDocument
from .latex_processor import process_latex_document, process_markdown_with_math, ProcessedContent
from .tts import synthesize_speech

try:
    from .figure_table_summarizer import summarise_caption, summarise_table
    SUMMARIZATION_AVAILABLE = True
except ImportError:
    SUMMARIZATION_AVAILABLE = False


@dataclass
class ProcessingOptions:
    """Options for document processing."""
    use_summarization: bool = False
    openai_api_key: Optional[str] = None
    pdf_extraction_method: Optional[str] = None  # "pymupdf" or "pypdf2"
    include_figures: bool = True
    include_tables: bool = True
    include_equations: bool = True
    tts_backend: str = "offline"  # "offline" or "openai"
    tts_voice: str = ""  # Voice selection for offline TTS
    tts_rate: int = 200  # Speaking rate for offline TTS
    tts_model: str = "tts-1"  # OpenAI TTS model
    tts_voice_openai: str = "alloy"  # OpenAI voice selection


@dataclass
class ProcessingResult:
    """Result of document processing."""
    spoken_text: str
    processed_content: ProcessedContent
    audio_file: Optional[str] = None
    processing_log: list[str] = None


class DocumentProcessor:
    """Main document processor that handles the complete pipeline."""
    
    def __init__(self, options: ProcessingOptions):
        self.options = options
        self.log = []
    
    def process_pdf(self, pdf_path: Union[str, Path], output_audio_path: Optional[str] = None) -> ProcessingResult:
        """Process a PDF file through the complete pipeline.
        
        Args:
            pdf_path: Path to PDF file
            output_audio_path: Optional path for audio output
            
        Returns:
            ProcessingResult with spoken text and optional audio file
        """
        self.log = ["Starting PDF processing"]
        
        # Step 1: Extract PDF to LaTeX/Markdown
        self.log.append("Extracting PDF content...")
        extracted_doc = extract_pdf_to_latex(
            str(pdf_path), 
            method=self.options.pdf_extraction_method
        )
        self.log.append(f"Extracted using {extracted_doc.extraction_method}")
        
        # Step 2: Process LaTeX/Markdown content
        self.log.append("Processing mathematical content...")
        processed_content = process_latex_document(
            extracted_doc.content,
            summarize_figures=self.options.use_summarization and self.options.include_figures,
            summarize_tables=self.options.use_summarization and self.options.include_tables,
            api_key=self.options.openai_api_key
        )
        
        # Step 3: Create spoken narration
        self.log.append("Creating spoken narration...")
        spoken_text = self._create_spoken_narration(processed_content, extracted_doc)
        
        # Step 4: Generate audio if requested
        audio_file = None
        if output_audio_path:
            self.log.append("Synthesizing speech...")
            audio_file = self._synthesize_audio(spoken_text, output_audio_path)
            self.log.append(f"Audio saved to {audio_file}")
        
        return ProcessingResult(
            spoken_text=spoken_text,
            processed_content=processed_content,
            audio_file=audio_file,
            processing_log=self.log.copy()
        )
    
    def process_latex(self, latex_content: str, output_audio_path: Optional[str] = None) -> ProcessingResult:
        """Process LaTeX content directly.
        
        Args:
            latex_content: LaTeX document content
            output_audio_path: Optional path for audio output
            
        Returns:
            ProcessingResult with spoken text and optional audio file
        """
        self.log = ["Starting LaTeX processing"]
        
        # Process LaTeX content
        self.log.append("Processing LaTeX mathematical content...")
        processed_content = process_latex_document(
            latex_content,
            summarize_figures=self.options.use_summarization and self.options.include_figures,
            summarize_tables=self.options.use_summarization and self.options.include_tables,
            api_key=self.options.openai_api_key
        )
        
        # Create spoken narration
        self.log.append("Creating spoken narration...")
        spoken_text = self._create_spoken_narration_from_processed(processed_content)
        
        # Generate audio if requested
        audio_file = None
        if output_audio_path:
            self.log.append("Synthesizing speech...")
            audio_file = self._synthesize_audio(spoken_text, output_audio_path)
            self.log.append(f"Audio saved to {audio_file}")
        
        return ProcessingResult(
            spoken_text=spoken_text,
            processed_content=processed_content,
            audio_file=audio_file,
            processing_log=self.log.copy()
        )
    
    def process_markdown(self, markdown_content: str, output_audio_path: Optional[str] = None) -> ProcessingResult:
        """Process Markdown content with LaTeX math.
        
        Args:
            markdown_content: Markdown document with LaTeX math
            output_audio_path: Optional path for audio output
            
        Returns:
            ProcessingResult with spoken text and optional audio file
        """
        self.log = ["Starting Markdown processing"]
        
        # Process markdown with math
        self.log.append("Processing Markdown with mathematical content...")
        processed_text = process_markdown_with_math(markdown_content)
        
        # Create minimal ProcessedContent structure
        processed_content = ProcessedContent(
            text=processed_text,
            figures=[],
            tables=[],
            equations=[],
            metadata={}
        )
        
        # Generate audio if requested
        audio_file = None
        if output_audio_path:
            self.log.append("Synthesizing speech...")
            audio_file = self._synthesize_audio(processed_text, output_audio_path)
            self.log.append(f"Audio saved to {audio_file}")
        
        return ProcessingResult(
            spoken_text=processed_text,
            processed_content=processed_content,
            audio_file=audio_file,
            processing_log=self.log.copy()
        )
    
    def _create_spoken_narration(self, processed_content: ProcessedContent, 
                               extracted_doc: ExtractedDocument) -> str:
        """Create complete spoken narration from processed content."""
        parts = []
        
        # Add title if available
        if processed_content.metadata.get('title'):
            parts.append(f"Title: {processed_content.metadata['title']}")
        
        # Add author if available
        if processed_content.metadata.get('author'):
            parts.append(f"Author: {processed_content.metadata['author']}")
        
        # Add main content
        if processed_content.text:
            parts.append(processed_content.text)
        
        # Add figures if requested
        if self.options.include_figures and processed_content.figures:
            parts.append("\\n\\nFigures:")
            for caption, description in processed_content.figures:
                if caption:
                    parts.append(f"Figure: {caption}")
                if description and description != caption:
                    parts.append(f"Description: {description}")
        
        # Add tables if requested
        if self.options.include_tables and processed_content.tables:
            parts.append("\\n\\nTables:")
            for caption, content in processed_content.tables:
                if caption:
                    parts.append(f"Table: {caption}")
                if content:
                    # Summarize table content if summarization is available
                    if self.options.use_summarization and SUMMARIZATION_AVAILABLE and self.options.openai_api_key:
                        try:
                            table_rows = content.split('\\n') if isinstance(content, str) else [str(content)]
                            summary = summarise_table(table_rows, self.options.openai_api_key)
                            parts.append(f"Table content: {summary}")
                        except Exception as e:
                            self.log.append(f"Table summarization failed: {e}")
                            parts.append(f"Table content: {content}")
                    else:
                        parts.append(f"Table content: {content}")
        
        # Add standalone equations if requested
        if self.options.include_equations and processed_content.equations:
            parts.append("\\n\\nKey equations:")
            for i, equation in enumerate(processed_content.equations, 1):
                parts.append(f"Equation {i}: {equation}")
        
        return "\\n\\n".join(parts)
    
    def _create_spoken_narration_from_processed(self, processed_content: ProcessedContent) -> str:
        """Create spoken narration from ProcessedContent only."""
        parts = []
        
        # Add title if available
        if processed_content.metadata.get('title'):
            parts.append(f"Title: {processed_content.metadata['title']}")
        
        # Add author if available
        if processed_content.metadata.get('author'):
            parts.append(f"Author: {processed_content.metadata['author']}")
        
        # Add main content
        if processed_content.text:
            parts.append(processed_content.text)
        
        return "\\n\\n".join(parts)
    
    def _synthesize_audio(self, text: str, output_path: str) -> str:
        """Synthesize audio from text."""
        try:
            if self.options.tts_backend == "openai":
                return synthesize_speech(
                    text=text,
                    output_path=output_path,
                    use_openai=True,
                    api_key=self.options.openai_api_key,
                    model=self.options.tts_model,
                    openai_voice=self.options.tts_voice_openai
                )
            else:
                return synthesize_speech(
                    text=text,
                    output_path=output_path,
                    voice=self.options.tts_voice,
                    rate=self.options.tts_rate,
                    use_openai=False
                )
        except Exception as e:
            self.log.append(f"TTS synthesis failed: {e}")
            raise


# Convenience functions for easy use
def process_pdf_to_speech(pdf_path: Union[str, Path], 
                         output_audio_path: Optional[str] = None,
                         **kwargs) -> ProcessingResult:
    """Convenience function to process PDF to speech.
    
    Args:
        pdf_path: Path to PDF file
        output_audio_path: Optional path for audio output
        **kwargs: Additional options for ProcessingOptions
        
    Returns:
        ProcessingResult
    """
    options = ProcessingOptions(**kwargs)
    processor = DocumentProcessor(options)
    return processor.process_pdf(pdf_path, output_audio_path)


def process_latex_to_speech(latex_content: str,
                           output_audio_path: Optional[str] = None,
                           **kwargs) -> ProcessingResult:
    """Convenience function to process LaTeX to speech.
    
    Args:
        latex_content: LaTeX document content
        output_audio_path: Optional path for audio output
        **kwargs: Additional options for ProcessingOptions
        
    Returns:
        ProcessingResult
    """
    options = ProcessingOptions(**kwargs)
    processor = DocumentProcessor(options)
    return processor.process_latex(latex_content, output_audio_path)


def process_markdown_to_speech(markdown_content: str,
                              output_audio_path: Optional[str] = None,
                              **kwargs) -> ProcessingResult:
    """Convenience function to process Markdown to speech.
    
    Args:
        markdown_content: Markdown content with LaTeX math
        output_audio_path: Optional path for audio output
        **kwargs: Additional options for ProcessingOptions
        
    Returns:
        ProcessingResult
    """
    options = ProcessingOptions(**kwargs)
    processor = DocumentProcessor(options)
    return processor.process_markdown(markdown_content, output_audio_path)