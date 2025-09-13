"""
Tests for vision-based PDF analysis functionality.
"""

import pytest
import base64
from unittest.mock import patch, Mock, MagicMock
from paper_voice.vision_pdf_analyzer import (
    ContentType,
    ContentBlock,
    PDFAnalysisResult,
    encode_image_base64,
    group_content_by_type,
    create_enhanced_text_from_analysis,
    extract_figures_and_tables_from_analysis,
    analyze_page_with_vision
)


class TestContentTypes:
    """Test content type constants."""
    
    def test_content_type_constants(self):
        """Test that content type constants are properly defined."""
        assert ContentType.TEXT == "text"
        assert ContentType.MATH == "math"
        assert ContentType.FIGURE == "figure"
        assert ContentType.TABLE == "table"
        assert ContentType.CAPTION == "caption"
        assert ContentType.HEADER == "header"
        assert ContentType.FOOTER == "footer"


class TestContentBlock:
    """Test ContentBlock data structure."""
    
    def test_content_block_creation(self):
        """Test creating ContentBlock instances."""
        block = ContentBlock(
            content_type="text",
            text="Sample text content",
            description="Description",
            bbox=(10, 20, 100, 200),
            page_number=1,
            confidence=0.95
        )
        
        assert block.content_type == "text"
        assert block.text == "Sample text content"
        assert block.description == "Description"
        assert block.bbox == (10, 20, 100, 200)
        assert block.page_number == 1
        assert block.confidence == 0.95
    
    def test_content_block_defaults(self):
        """Test ContentBlock with default values."""
        block = ContentBlock(
            content_type="math",
            text="x^2 + y^2 = r^2",
            description="Pythagorean theorem",
            bbox=(0, 0, 50, 50),
            page_number=2
        )
        
        assert block.confidence == 1.0  # Default value


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_encode_image_base64(self):
        """Test base64 encoding of image bytes."""
        test_bytes = b"fake image data"
        encoded = encode_image_base64(test_bytes)
        
        # Should be valid base64
        decoded = base64.b64decode(encoded)
        assert decoded == test_bytes
    
    def test_group_content_by_type(self):
        """Test grouping content blocks by type."""
        blocks = [
            ContentBlock("text", "Text 1", "", (0, 0, 10, 10), 1),
            ContentBlock("math", "x^2", "", (0, 20, 10, 30), 1),
            ContentBlock("text", "Text 2", "", (0, 40, 10, 50), 1),
            ContentBlock("figure", "", "A figure", (0, 60, 10, 70), 1)
        ]
        
        grouped = group_content_by_type(blocks)
        
        assert len(grouped["text"]) == 2
        assert len(grouped["math"]) == 1
        assert len(grouped["figure"]) == 1
        assert "table" not in grouped


class TestEnhancedTextCreation:
    """Test enhanced text creation from analysis."""
    
    def test_create_enhanced_text_single_page(self):
        """Test creating enhanced text from single page analysis."""
        analysis_result = PDFAnalysisResult(
            content_blocks=[
                ContentBlock("text", "Introduction", "", (0, 0, 100, 20), 1),
                ContentBlock("math", "E = mc^2", "", (0, 30, 100, 50), 1),
                ContentBlock("figure", "", "Energy diagram", (0, 60, 100, 80), 1)
            ],
            page_images=[b"fake image"],
            metadata={"total_pages": 1}
        )
        
        enhanced_text = create_enhanced_text_from_analysis(analysis_result)
        
        assert "=== Page 1 ===" in enhanced_text
        assert "Introduction" in enhanced_text
        assert "[MATHEMATICAL EXPRESSION: E = mc^2]" in enhanced_text
        assert "[FIGURE: Energy diagram]" in enhanced_text
    
    def test_create_enhanced_text_multiple_pages(self):
        """Test creating enhanced text from multiple page analysis."""
        analysis_result = PDFAnalysisResult(
            content_blocks=[
                ContentBlock("text", "Page 1 text", "", (0, 0, 100, 20), 1),
                ContentBlock("text", "Page 2 text", "", (0, 0, 100, 20), 2),
                ContentBlock("table", "", "Data table", (0, 30, 100, 60), 2)
            ],
            page_images=[b"page1", b"page2"],
            metadata={"total_pages": 2}
        )
        
        enhanced_text = create_enhanced_text_from_analysis(analysis_result)
        
        assert "=== Page 1 ===" in enhanced_text
        assert "=== Page 2 ===" in enhanced_text
        assert "Page 1 text" in enhanced_text
        assert "Page 2 text" in enhanced_text
        assert "[TABLE: Data table]" in enhanced_text
    
    def test_create_enhanced_text_content_ordering(self):
        """Test that content is ordered by vertical position."""
        # Create blocks with different y positions (should be sorted by y0)
        analysis_result = PDFAnalysisResult(
            content_blocks=[
                ContentBlock("text", "Bottom text", "", (0, 80, 100, 100), 1),  # y0=80
                ContentBlock("text", "Top text", "", (0, 10, 100, 30), 1),      # y0=10
                ContentBlock("text", "Middle text", "", (0, 40, 100, 60), 1)    # y0=40
            ],
            page_images=[b"page"],
            metadata={"total_pages": 1}
        )
        
        enhanced_text = create_enhanced_text_from_analysis(analysis_result)
        
        # Should appear in top-to-bottom order
        top_pos = enhanced_text.find("Top text")
        middle_pos = enhanced_text.find("Middle text")
        bottom_pos = enhanced_text.find("Bottom text")
        
        assert top_pos < middle_pos < bottom_pos


class TestFigureTableExtraction:
    """Test figure and table extraction from vision analysis."""
    
    def test_extract_figures_with_captions(self):
        """Test extracting figures with associated captions."""
        analysis_result = PDFAnalysisResult(
            content_blocks=[
                ContentBlock("figure", "", "Bar chart showing results", (10, 20, 90, 60), 1),
                ContentBlock("caption", "Figure 1: Results comparison", "", (10, 65, 90, 75), 1)
            ],
            page_images=[b"page"],
            metadata={}
        )
        
        figures, tables = extract_figures_and_tables_from_analysis(analysis_result)
        
        assert len(figures) == 1
        assert figures[0][0] == "Figure 1: Results comparison"  # caption
        assert figures[0][1] == "Bar chart showing results"     # description
    
    def test_extract_tables_with_captions(self):
        """Test extracting tables with associated captions."""
        analysis_result = PDFAnalysisResult(
            content_blocks=[
                ContentBlock("caption", "Table 1: Statistical summary", "", (10, 10, 90, 20), 1),
                ContentBlock("table", "Data rows and columns", "", (10, 25, 90, 70), 1)
            ],
            page_images=[b"page"],
            metadata={}
        )
        
        figures, tables = extract_figures_and_tables_from_analysis(analysis_result)
        
        assert len(tables) == 1
        assert tables[0][0] == "Table 1: Statistical summary"  # caption
        assert tables[0][1] == "Data rows and columns"        # content
    
    def test_extract_without_captions(self):
        """Test extracting figures/tables without captions."""
        analysis_result = PDFAnalysisResult(
            content_blocks=[
                ContentBlock("figure", "", "Uncaptioned chart", (10, 20, 90, 60), 1),
                ContentBlock("table", "Raw data", "", (10, 70, 90, 100), 1)
            ],
            page_images=[b"page"],
            metadata={}
        )
        
        figures, tables = extract_figures_and_tables_from_analysis(analysis_result)
        
        assert len(figures) == 1
        assert figures[0][0] == "Figure"  # default caption
        assert figures[0][1] == "Uncaptioned chart"
        
        assert len(tables) == 1
        assert tables[0][0] == "Table"  # default caption
        assert tables[0][1] == "Raw data"


class TestVisionAnalysis:
    """Test vision analysis functionality (mocked)."""
    
    @patch('paper_voice.vision_pdf_analyzer.OpenAI')
    def test_analyze_page_with_vision_success(self, mock_openai_class):
        """Test successful page analysis with vision."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''```json
        {
          "content_blocks": [
            {
              "type": "text",
              "content": "Sample text content",
              "bbox": [10, 10, 90, 30],
              "confidence": 0.9
            },
            {
              "type": "math",
              "content": "x^2 + y^2 = r^2",
              "bbox": [10, 40, 90, 60],
              "confidence": 0.95
            }
          ]
        }
        ```'''
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test the function
        image_bytes = b"fake image data"
        result = analyze_page_with_vision(image_bytes, 1, "test-api-key")
        
        assert len(result) == 2
        assert result[0].content_type == "text"
        assert result[0].text == "Sample text content"
        assert result[0].page_number == 1
        
        assert result[1].content_type == "math"
        assert result[1].text == "x^2 + y^2 = r^2"
    
    @patch('paper_voice.vision_pdf_analyzer.OpenAI')
    def test_analyze_page_with_vision_json_error(self, mock_openai_class):
        """Test handling of JSON parsing errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON response"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        result = analyze_page_with_vision(b"fake image", 1, "test-key")
        
        # Should return empty list on JSON error
        assert result == []
    
    @patch('paper_voice.vision_pdf_analyzer.OpenAI')
    def test_analyze_page_with_vision_api_error(self, mock_openai_class):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Simulate API error
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = analyze_page_with_vision(b"fake image", 1, "test-key")
        
        # Should return empty list on API error
        assert result == []


class TestPDFToImages:
    """Test PDF to images conversion (would require PyMuPDF)."""
    
    @patch('paper_voice.vision_pdf_analyzer.fitz')
    def test_pdf_to_images_success(self, mock_fitz):
        """Test successful PDF to images conversion."""
        # Mock PyMuPDF objects
        mock_doc = Mock()
        mock_page = Mock()
        mock_pix = Mock()
        
        mock_fitz.open.return_value = mock_doc
        mock_doc.page_count = 2
        mock_doc.__getitem__.return_value = mock_page
        mock_page.get_pixmap.return_value = mock_pix
        mock_pix.tobytes.return_value = b"fake png data"
        
        # Test the function
        from paper_voice.vision_pdf_analyzer import pdf_to_images
        
        with patch('paper_voice.vision_pdf_analyzer.PYMUPDF_AVAILABLE', True):
            images = pdf_to_images("/fake/path.pdf")
        
        assert len(images) == 2
        assert all(isinstance(img, bytes) for img in images)
    
    def test_pdf_to_images_no_pymupdf(self):
        """Test PDF to images when PyMuPDF not available."""
        from paper_voice.vision_pdf_analyzer import pdf_to_images
        
        with patch('paper_voice.vision_pdf_analyzer.PYMUPDF_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="PyMuPDF is required"):
                pdf_to_images("/fake/path.pdf")


if __name__ == "__main__":
    pytest.main([__file__])