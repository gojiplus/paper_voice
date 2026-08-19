"""Vision-based PDF analysis using GPT-4V to separate figures, text, and math.

This module uses OpenAI's GPT-4 Vision model to analyze PDF pages and extract
structured content with better separation of different content types.
"""

import base64
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    fitz = None
    PYMUPDF_AVAILABLE = False

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None
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
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    page_number: int
    confidence: float = 1.0


@dataclass
class PDFAnalysisResult:
    """Result of PDF analysis."""

    content_blocks: list[ContentBlock]
    page_images: list[bytes]  # PNG images of each page
    metadata: dict[str, str]


def pdf_to_images(pdf_path: str, dpi: int = 150) -> list[bytes]:
    """Convert PDF pages to PNG images."""
    if not PYMUPDF_AVAILABLE or fitz is None:
        raise RuntimeError(
            "PyMuPDF is required for PDF to image conversion. "
            "Install with: pip install PyMuPDF"
        )

    images = []

    try:
        doc = fitz.open(pdf_path)

        for page_num in range(doc.page_count):
            page = doc[page_num]

            # Render page to image
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PNG bytes
            img_data = pix.tobytes("png")
            images.append(img_data)

        doc.close()
        return images

    except Exception as e:
        raise RuntimeError(f"Error converting PDF to images: {e}") from e


def encode_image_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def analyze_page_with_vision(
    image_bytes: bytes, page_number: int, api_key: str
) -> list[ContentBlock]:
    """Analyze a single PDF page using GPT-4V."""
    if not OPENAI_AVAILABLE or OpenAI is None:
        raise RuntimeError("OpenAI library is required for vision analysis")

    client = OpenAI(api_key=api_key)

    # Encode image
    base64_image = encode_image_base64(image_bytes)

    prompt = (
        "You are an expert at analyzing academic PDF pages. Examine this "
        "page and identify different content blocks with their locations "
        "and types.\n"
        """
For each content block, identify:
1. Content type: text, math, figure, table, header, footer, or caption
2. The actual text content (for text/math/captions)
3. A description (for figures/tables)
"""
        "4. Approximate bounding box as percentages (x0, y0, x1, y1) where "
        "(0,0) is top-left and (100,100) is bottom-right\n"
        """
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
"""
        "\nBe precise about mathematical content - identify equations, "
        "formulas, and mathematical expressions clearly."
    )

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
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        # A None content behaves like an empty one here: both fall through to
        # the "No JSON found in response" path below.
        result_text = response.choices[0].message.content or ""

        # Parse JSON response
        import json
        import re

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"```json\s*(.*?)\s*```", result_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")

        try:
            analysis = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error("JSON parsing error: %s", e)
            logger.error("Raw response: %s", result_text)
            return []

        # Convert to ContentBlock objects
        content_blocks = []
        for block in analysis.get("content_blocks", []):
            bbox = tuple(block.get("bbox", [0, 0, 100, 100]))

            content_block = ContentBlock(
                content_type=block.get("type", "text"),
                text=block.get("content", ""),
                description=block.get("content", "")
                if block.get("type") in ["figure", "table"]
                else "",
                bbox=bbox,
                page_number=page_number,
                confidence=block.get("confidence", 0.8),
            )
            content_blocks.append(content_block)

        return content_blocks

    except Exception as e:
        logger.error("Error analyzing page %s with vision: %s", page_number, e)
        return []


def analyze_pdf_with_vision(
    pdf_path: str, api_key: str, max_pages: int | None = None
) -> PDFAnalysisResult:
    """Analyze entire PDF using vision API.

    Args:
        pdf_path: Path to the PDF file, rendered to one image per page
            before analysis.
        api_key: OpenAI API key passed to the vision call for each page.
        max_pages: Maximum number of leading pages to analyze; None (or 0)
            analyzes every page.

    Returns:
        A :class:`PDFAnalysisResult` holding the content blocks found
        across all analyzed pages, the rendered page images, and metadata
        counting the pages, blocks, and distinct content types.
    """
    logger.info("Converting PDF to images...")
    page_images = pdf_to_images(pdf_path)

    if max_pages:
        page_images = page_images[:max_pages]

    logger.info("Analyzing %s pages with GPT-4V...", len(page_images))

    all_content_blocks = []

    for i, image_bytes in enumerate(page_images):
        logger.info("Analyzing page %s/%s...", i + 1, len(page_images))

        content_blocks = analyze_page_with_vision(image_bytes, i + 1, api_key)
        all_content_blocks.extend(content_blocks)

    # Extract metadata
    metadata = {
        "total_pages": len(page_images),
        "total_content_blocks": len(all_content_blocks),
        "content_types": list({block.content_type for block in all_content_blocks}),
    }

    return PDFAnalysisResult(
        content_blocks=all_content_blocks, page_images=page_images, metadata=metadata
    )


def group_content_by_type(
    content_blocks: list[ContentBlock],
) -> dict[str, list[ContentBlock]]:
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
                enhanced_text_parts.append(
                    f"[{block.content_type.upper()}: {block.text}]"
                )

            enhanced_text_parts.append("\\n")

    return "".join(enhanced_text_parts)


def extract_figures_and_tables_from_analysis(
    analysis_result: PDFAnalysisResult,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Extract figures and tables from vision analysis."""
    tables = []

    # Find captions and associate them with figures/tables
    captions = [
        block
        for block in analysis_result.content_blocks
        if block.content_type == ContentType.CAPTION
    ]
    figure_blocks = [
        block
        for block in analysis_result.content_blocks
        if block.content_type == ContentType.FIGURE
    ]
    table_blocks = [
        block
        for block in analysis_result.content_blocks
        if block.content_type == ContentType.TABLE
    ]

    def closest_caption(block: ContentBlock, default: str) -> str:
        """Find the caption vertically nearest to a block on the same page.

        A caption can sit above or below its block, so measure both gaps and
        take the smaller. Anything further than 20% of page height away is
        assumed to belong to another block.
        """
        caption_text = default
        min_distance = float("inf")

        for cap_block in captions:
            if cap_block.page_number == block.page_number:
                distance = min(
                    abs(cap_block.bbox[1] - block.bbox[3]),  # caption below
                    abs(cap_block.bbox[3] - block.bbox[1]),  # caption above
                )
                if distance < min_distance and distance < 20:
                    min_distance = distance
                    caption_text = cap_block.text

        return caption_text

    # Simple association: find captions near figures/tables
    figures = [
        (closest_caption(fig_block, "Figure"), fig_block.description)
        for fig_block in figure_blocks
    ]

    for table_block in table_blocks:
        caption_text = closest_caption(table_block, "Table")

        tables.append(
            (caption_text, table_block.text)
        )  # For tables, use the text content

    return figures, tables
