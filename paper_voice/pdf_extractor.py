"""
PDF extraction module for converting PDFs to LaTeX/Markdown format.

This module handles the complex task of extracting structured content from PDFs
and converting it into a LaTeX/Markdown representation that can be processed
by the latex_processor module. It handles mathematical expressions, tables,
figures, and document structure.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    import fitz  # PyMuPDF - better for complex PDFs
except ImportError:
    fitz = None


@dataclass
class ExtractedDocument:
    """Container for extracted PDF content in LaTeX/Markdown format."""
    content: str                    # Main document content in LaTeX/Markdown
    figures: List[Dict[str, str]]   # Figure information
    tables: List[Dict[str, str]]    # Table information
    metadata: Dict[str, str]        # Document metadata
    extraction_method: str          # Which extraction method was used


class PDFExtractor:
    """Extract structured content from PDFs."""
    
    def __init__(self):
        self.available_extractors = []
        if fitz is not None:
            self.available_extractors.append("pymupdf")
        if PdfReader is not None:
            self.available_extractors.append("pypdf2")
    
    def extract_document(self, pdf_path: str, method: Optional[str] = None) -> ExtractedDocument:
        """Extract document content from PDF.
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ("pymupdf", "pypdf2", or None for auto)
            
        Returns:
            ExtractedDocument with content in LaTeX/Markdown format
        """
        if method is None:
            method = self.available_extractors[0] if self.available_extractors else None
        
        if method == "pymupdf" and fitz is not None:
            return self._extract_with_pymupdf(pdf_path)
        elif method == "pypdf2" and PdfReader is not None:
            return self._extract_with_pypdf2(pdf_path)
        else:
            raise RuntimeError(f"No suitable PDF extraction method available. "
                             f"Available: {self.available_extractors}")
    
    def _extract_with_pymupdf(self, pdf_path: str) -> ExtractedDocument:
        """Extract using PyMuPDF (better for complex layouts)."""
        doc = fitz.open(pdf_path)
        content_parts = []
        figures = []
        tables = []
        metadata = {}
        
        # Extract document metadata
        doc_metadata = doc.metadata
        if doc_metadata:
            metadata.update({
                'title': doc_metadata.get('title', ''),
                'author': doc_metadata.get('author', ''),
                'subject': doc_metadata.get('subject', ''),
                'creator': doc_metadata.get('creator', ''),
            })
        
        for page_num, page in enumerate(doc):
            # Extract text with structure preservation
            text = page.get_text()
            
            # Extract images/figures
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                figures.append({
                    'page': page_num + 1,
                    'index': img_index,
                    'caption': '',  # Will be detected from text
                    'bbox': img,  # Bounding box info
                })
            
            # Extract tables using text analysis
            page_tables = self._detect_tables_from_text(text, page_num + 1)
            tables.extend(page_tables)
            
            # Process text for mathematical expressions
            processed_text = self._convert_text_to_latex_markdown(text)
            content_parts.append(processed_text)
        
        doc.close()
        
        # Combine all content
        full_content = '\n\n'.join(content_parts)
        
        # Post-process to improve structure
        full_content = self._improve_document_structure(full_content)
        
        # Associate captions with figures and tables
        self._associate_captions_with_objects(full_content, figures, tables)
        
        return ExtractedDocument(
            content=full_content,
            figures=figures,
            tables=tables,
            metadata=metadata,
            extraction_method="pymupdf"
        )
    
    def _extract_with_pypdf2(self, pdf_path: str) -> ExtractedDocument:
        """Extract using PyPDF2 (simpler, more reliable for text)."""
        reader = PdfReader(pdf_path)
        content_parts = []
        figures = []
        tables = []
        metadata = {}
        
        # Extract metadata
        if reader.metadata:
            metadata.update({
                'title': reader.metadata.get('/Title', ''),
                'author': reader.metadata.get('/Author', ''),
                'subject': reader.metadata.get('/Subject', ''),
                'creator': reader.metadata.get('/Creator', ''),
            })
        
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            
            # Extract tables using heuristics
            page_tables = self._detect_tables_from_text(text, page_num + 1)
            tables.extend(page_tables)
            
            # Process text for mathematical expressions
            processed_text = self._convert_text_to_latex_markdown(text)
            content_parts.append(processed_text)
        
        # Combine content
        full_content = '\n\n'.join(content_parts)
        full_content = self._improve_document_structure(full_content)
        
        # Detect figure references (since we can't extract images with PyPDF2)
        figures = self._detect_figure_references(full_content)
        
        # Associate captions
        self._associate_captions_with_objects(full_content, figures, tables)
        
        return ExtractedDocument(
            content=full_content,
            figures=figures,
            tables=tables,
            metadata=metadata,
            extraction_method="pypdf2"
        )
    
    def _convert_text_to_latex_markdown(self, text: str) -> str:
        """Convert extracted text to LaTeX/Markdown with math expressions."""
        # Detect and wrap mathematical expressions
        
        # Common patterns that might be mathematical expressions
        math_patterns = [
            # Variables with subscripts/superscripts
            (r'([A-Za-z])\s*([₀₁₂₃₄₅₆₇₈₉ₓᵢⱼₖₘₙ]+)', r'$\\1_{\\2}$'),
            (r'([A-Za-z])\s*([⁰¹²³⁴⁵⁶⁷⁸⁹ⁿᵏⁱʲ]+)', r'$\\1^{\\2}$'),
            
            # Greek letters (if they appear as Unicode)
            (r'\\b(α|β|γ|δ|ε|ζ|η|θ|ι|κ|λ|μ|ν|ξ|π|ρ|σ|τ|υ|φ|χ|ψ|ω)\\b', r'$\\\\alpha$'),  # Example for alpha
            
            # Mathematical operators
            (r'≤', r'$\\\\leq$'),
            (r'≥', r'$\\\\geq$'),
            (r'≠', r'$\\\\neq$'),
            (r'≈', r'$\\\\approx$'),
            (r'∞', r'$\\\\infty$'),
            (r'→', r'$\\\\rightarrow$'),
            (r'∈', r'$\\\\in$'),
            (r'∑', r'$\\\\sum$'),
            (r'∏', r'$\\\\prod$'),
            (r'∫', r'$\\\\int$'),
            
            # Fractions (simple patterns)
            (r'(\\d+)/(\\d+)', r'$\\\\frac{\\1}{\\2}$'),
        ]
        
        for pattern, replacement in math_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Convert section headers to markdown/latex format
        # Look for patterns like "1. Introduction", "2.1 Background", etc.
        text = re.sub(r'^(\\d+(?:\\.\\d+)*)\\.?\\s+([A-Z][^\\n]*)', r'\\section{\\2}', text, flags=re.MULTILINE)
        
        # Handle figure and table references
        text = re.sub(r'Figure\\s+(\\d+)', r'Figure~\\\\ref{fig:\\1}', text)
        text = re.sub(r'Table\\s+(\\d+)', r'Table~\\\\ref{tab:\\1}', text)
        
        return text
    
    def _detect_tables_from_text(self, text: str, page_num: int) -> List[Dict[str, str]]:
        """Detect tables from extracted text using heuristics."""
        tables = []
        lines = text.split('\\n')
        
        # Look for lines with multiple columns (detected by multiple spaces)
        table_lines = []
        for i, line in enumerate(lines):
            # Check if line looks like a table row (multiple items separated by spaces)
            if re.search(r'\\S\\s{2,}\\S.*\\s{2,}\\S', line):
                table_lines.append((i, line))
        
        if table_lines:
            # Group consecutive table lines
            current_table = []
            for i, (line_idx, line) in enumerate(table_lines):
                if i == 0 or table_lines[i-1][0] == line_idx - 1:
                    current_table.append(line)
                else:
                    if current_table:
                        tables.append({
                            'page': page_num,
                            'content': '\\n'.join(current_table),
                            'caption': '',  # Will be filled by caption association
                        })
                    current_table = [line]
            
            # Don't forget the last table
            if current_table:
                tables.append({
                    'page': page_num,
                    'content': '\\n'.join(current_table),
                    'caption': '',
                })
        
        return tables
    
    def _detect_figure_references(self, text: str) -> List[Dict[str, str]]:
        """Detect figure references in text."""
        figures = []
        
        # Find figure mentions
        fig_mentions = re.findall(r'Figure\\s+(\\d+)', text, re.IGNORECASE)
        for fig_num in set(fig_mentions):
            figures.append({
                'number': fig_num,
                'caption': '',  # Will be filled by caption detection
                'type': 'reference'  # This is just a reference, not actual image
            })
        
        return figures
    
    def _improve_document_structure(self, text: str) -> str:
        """Improve the overall structure of the document."""
        # Clean up excessive whitespace
        text = re.sub(r'\\n{3,}', '\\n\\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Improve paragraph breaks
        # Join lines that are probably part of the same paragraph
        lines = text.split('\\n')
        improved_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                improved_lines.append('')
                continue
            
            # Check if this line should be joined with the previous one
            if (i > 0 and improved_lines and improved_lines[-1] and 
                not line[0].isupper() and not line.startswith('\\section') and
                not re.match(r'^\\d+\\.', line) and
                len(improved_lines[-1]) > 40):  # Previous line is substantial
                # Join with previous line
                improved_lines[-1] += ' ' + line
            else:
                improved_lines.append(line)
        
        text = '\\n'.join(improved_lines)
        
        # Add proper spacing around sections
        text = re.sub(r'\\n(\\section\\{[^}]+\\})\\n', r'\\n\\n\\1\\n\\n', text)
        
        return text
    
    def _associate_captions_with_objects(self, text: str, figures: List[Dict[str, str]], 
                                       tables: List[Dict[str, str]]) -> None:
        """Associate captions with figures and tables."""
        # Find figure captions
        fig_captions = re.findall(r'Figure\\s+(\\d+)[.:]*\\s*([^\\n]*(?:\\n[^\\n]*)*?)(?=\\n\\n|Figure|Table|$)', 
                                text, re.IGNORECASE | re.MULTILINE)
        
        fig_caption_dict = {}
        for num, caption in fig_captions:
            fig_caption_dict[num] = caption.strip()
        
        # Associate captions with figures
        for fig in figures:
            fig_num = fig.get('number', '')
            if fig_num in fig_caption_dict:
                fig['caption'] = fig_caption_dict[fig_num]
        
        # Find table captions
        table_captions = re.findall(r'Table\\s+(\\d+)[.:]*\\s*([^\\n]*(?:\\n[^\\n]*)*?)(?=\\n\\n|Figure|Table|$)', 
                                  text, re.IGNORECASE | re.MULTILINE)
        
        table_caption_dict = {}
        for num, caption in table_captions:
            table_caption_dict[num] = caption.strip()
        
        # Try to associate captions with tables based on position/context
        for i, table in enumerate(tables):
            # For now, assign captions in order
            if i + 1 <= len(table_caption_dict):
                table['caption'] = table_caption_dict.get(str(i + 1), '')


# Convenience function for easy use
def extract_pdf_to_latex(pdf_path: str, method: Optional[str] = None) -> ExtractedDocument:
    """Extract PDF content as LaTeX/Markdown format.
    
    Args:
        pdf_path: Path to PDF file
        method: Extraction method ("pymupdf" or "pypdf2")
        
    Returns:
        ExtractedDocument with LaTeX/Markdown content
    """
    extractor = PDFExtractor()
    return extractor.extract_document(pdf_path, method)