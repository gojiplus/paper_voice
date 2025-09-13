"""
Vision-based PDF analysis using GPT-4V for separating figures, text, math, and tables.

This module uses OpenAI's GPT-4 Vision model to analyze PDF pages and extract
structured content with better separation of different content types.
"""

import os
import base64
import tempfile
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ContentType:
    """Content type constants."""
    TEXT = "text"
    MATH = "math"
    FIGURE = "figure"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"


@dataclass
class ContentBlock:
    """Represents a block of content from a PDF page."""
    content_type: str
    text: str
    description: str  # For figures/tables
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    page_number: int
    confidence: float = 1.0


@dataclass
class PDFAnalysisResult:
    """Result of PDF analysis."""
    content_blocks: List[ContentBlock]
    page_images: List[bytes]  # PNG images of each page
    metadata: Dict[str, str]


def pdf_to_images(pdf_path: str, dpi: int = 150) -> List[bytes]:
    """Convert PDF pages to PNG images."""
    if not PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF is required for PDF to image conversion. Install with: pip install PyMuPDF")
    
    images = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Render page to image
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PNG bytes
            img_data = pix.tobytes("png")
            images.append(img_data)
        
        doc.close()
        return images
        
    except Exception as e:
        raise RuntimeError(f"Error converting PDF to images: {e}")


def encode_image_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


def analyze_page_with_vision(image_bytes: bytes, page_number: int, api_key: str) -> List[ContentBlock]:
    """Analyze a single PDF page using GPT-4V."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI library is required for vision analysis")
    
    client = OpenAI(api_key=api_key)
    
    # Encode image
    base64_image = encode_image_base64(image_bytes)
    
    prompt = """You are an expert at analyzing academic PDF pages. Examine this page and identify different content blocks with their locations and types.

For each content block, identify:
1. Content type: text, math, figure, table, header, footer, or caption
2. The actual text content (for text/math/captions)
3. A description (for figures/tables)
4. Approximate bounding box as percentages (x0, y0, x1, y1) where (0,0) is top-left and (100,100) is bottom-right

Focus on:
- **Mathematical expressions**: Both inline and display math, equations, formulas
- **Figures**: Plots, charts, diagrams, images with their captions
- **Tables**: Data tables with their captions
- **Regular text**: Paragraphs, sections, body text
- **Headers/Footers**: Page numbers, running heads
- **Captions**: Figure and table captions (identify separately from figures/tables)

Return the analysis in this JSON format:
```json
{
  "content_blocks": [
    {
      "type": "text|math|figure|table|header|footer|caption",
      "content": "actual text content or description",
      "bbox": [x0, y0, x1, y1],
      "confidence": 0.0-1.0
    }
  ]
}
```

Be precise about mathematical content - identify equations, formulas, and mathematical expressions clearly."""

    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content
        
        # Parse JSON response
        import json
        import re
        
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")
        
        try:
            analysis = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {result_text}")
            return []
        
        # Convert to ContentBlock objects
        content_blocks = []
        for block in analysis.get("content_blocks", []):
            bbox = tuple(block.get("bbox", [0, 0, 100, 100]))
            
            content_block = ContentBlock(
                content_type=block.get("type", "text"),
                text=block.get("content", ""),
                description=block.get("content", "") if block.get("type") in ["figure", "table"] else "",
                bbox=bbox,
                page_number=page_number,
                confidence=block.get("confidence", 0.8)
            )
            content_blocks.append(content_block)
        
        return content_blocks
        
    except Exception as e:
        print(f"Error analyzing page {page_number} with vision: {e}")
        return []


def analyze_pdf_with_vision(pdf_path: str, api_key: str, max_pages: Optional[int] = None) -> PDFAnalysisResult:
    """Analyze entire PDF using vision API.
    
    Parameters
    ----------
    pdf_path : str
        Path to PDF file
    api_key : str
        OpenAI API key
    max_pages : int, optional
        Maximum number of pages to analyze (None for all pages)
        
    Returns
    -------
    PDFAnalysisResult
        Structured analysis of the PDF
    """
    
    print("Converting PDF to images...")
    page_images = pdf_to_images(pdf_path)
    
    if max_pages:
        page_images = page_images[:max_pages]
    
    print(f"Analyzing {len(page_images)} pages with GPT-4V...")
    
    all_content_blocks = []
    
    for i, image_bytes in enumerate(page_images):
        print(f"Analyzing page {i+1}/{len(page_images)}...")
        
        content_blocks = analyze_page_with_vision(image_bytes, i+1, api_key)
        all_content_blocks.extend(content_blocks)
    
    # Extract metadata
    metadata = {
        "total_pages": len(page_images),
        "total_content_blocks": len(all_content_blocks),
        "content_types": list(set(block.content_type for block in all_content_blocks))
    }
    
    return PDFAnalysisResult(
        content_blocks=all_content_blocks,
        page_images=page_images,
        metadata=metadata
    )


def group_content_by_type(content_blocks: List[ContentBlock]) -> Dict[str, List[ContentBlock]]:
    """Group content blocks by type."""
    grouped = {}
    
    for block in content_blocks:
        if block.content_type not in grouped:
            grouped[block.content_type] = []
        grouped[block.content_type].append(block)
    
    return grouped


def create_enhanced_text_from_analysis(analysis_result: PDFAnalysisResult) -> str:
    """Create enhanced text from vision analysis results."""
    
    # Group content by page and then by type
    pages = {}
    for block in analysis_result.content_blocks:
        page_num = block.page_number
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(block)
    
    # Sort blocks within each page by vertical position (top to bottom)
    for page_num in pages:
        pages[page_num].sort(key=lambda b: b.bbox[1])  # Sort by y0 (top)
    
    enhanced_text_parts = []
    
    for page_num in sorted(pages.keys()):
        page_blocks = pages[page_num]
        
        enhanced_text_parts.append(f"\\n\\n=== Page {page_num} ===\\n")
        
        for block in page_blocks:
            if block.content_type == ContentType.TEXT:
                enhanced_text_parts.append(block.text)
            elif block.content_type == ContentType.MATH:
                enhanced_text_parts.append(f"[MATHEMATICAL EXPRESSION: {block.text}]")
            elif block.content_type == ContentType.FIGURE:
                enhanced_text_parts.append(f"[FIGURE: {block.description}]")
            elif block.content_type == ContentType.TABLE:
                enhanced_text_parts.append(f"[TABLE: {block.description}]")
            elif block.content_type == ContentType.CAPTION:
                enhanced_text_parts.append(f"Caption: {block.text}")
            elif block.content_type in [ContentType.HEADER, ContentType.FOOTER]:
                # Include headers/footers but mark them
                enhanced_text_parts.append(f"[{block.content_type.upper()}: {block.text}]")
            
            enhanced_text_parts.append("\\n")
    
    return "".join(enhanced_text_parts)


def extract_figures_and_tables_from_analysis(analysis_result: PDFAnalysisResult) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Extract figures and tables from vision analysis."""
    
    figures = []
    tables = []
    
    # Find captions and associate them with figures/tables
    captions = [block for block in analysis_result.content_blocks if block.content_type == ContentType.CAPTION]
    figure_blocks = [block for block in analysis_result.content_blocks if block.content_type == ContentType.FIGURE]
    table_blocks = [block for block in analysis_result.content_blocks if block.content_type == ContentType.TABLE]
    
    # Simple association: find captions near figures/tables
    for fig_block in figure_blocks:
        # Find closest caption
        caption_text = "Figure"
        min_distance = float('inf')
        
        for cap_block in captions:
            if cap_block.page_number == fig_block.page_number:
                # Calculate distance (simple vertical distance)
                distance = abs(cap_block.bbox[1] - fig_block.bbox[3])  # Caption top to figure bottom
                if distance < min_distance and distance < 20:  # Within 20% of page height
                    min_distance = distance
                    caption_text = cap_block.text
        
        figures.append((caption_text, fig_block.description))
    
    for table_block in table_blocks:
        # Find closest caption
        caption_text = "Table"
        min_distance = float('inf')
        
        for cap_block in captions:
            if cap_block.page_number == table_block.page_number:
                distance = abs(cap_block.bbox[1] - table_block.bbox[3])
                if distance < min_distance and distance < 20:
                    min_distance = distance
                    caption_text = cap_block.text
        
        tables.append((caption_text, table_block.text))  # For tables, use the text content
    
    return figures, tables


# Example usage
if __name__ == "__main__":
    # Test with a PDF file
    test_pdf = "sample.pdf"  # Replace with actual PDF path
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key and os.path.exists(test_pdf):
        result = analyze_pdf_with_vision(test_pdf, api_key, max_pages=1)
        
        print(f"Found {len(result.content_blocks)} content blocks")
        for block in result.content_blocks:
            print(f"- {block.content_type}: {block.text[:100]}...")
        
        enhanced_text = create_enhanced_text_from_analysis(result)
        print(f"\\nEnhanced text length: {len(enhanced_text)} characters")
    else:
        print("Set OPENAI_API_KEY and provide a test PDF file")