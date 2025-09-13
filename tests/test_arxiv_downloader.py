"""
Tests for arXiv downloader functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, Mock
from paper_voice.arxiv_downloader import (
    extract_arxiv_id,
    download_arxiv_paper,
    find_main_tex_file,
    extract_paper_metadata,
    process_latex_inputs
)


class TestArxivIdExtraction:
    """Test arXiv ID extraction from various URL formats."""
    
    def test_extract_arxiv_id_from_abs_url(self):
        """Test extraction from abs URL."""
        url = "https://arxiv.org/abs/2301.12345"
        assert extract_arxiv_id(url) == "2301.12345"
    
    def test_extract_arxiv_id_from_pdf_url(self):
        """Test extraction from PDF URL."""
        url = "https://arxiv.org/pdf/2301.12345.pdf"
        assert extract_arxiv_id(url) == "2301.12345"
    
    def test_extract_arxiv_id_from_direct_id(self):
        """Test extraction from direct ID."""
        id_str = "2301.12345"
        assert extract_arxiv_id(id_str) == "2301.12345"
    
    def test_extract_arxiv_id_with_version(self):
        """Test extraction with version number."""
        url = "https://arxiv.org/abs/2301.12345v2"
        assert extract_arxiv_id(url) == "2301.12345v2"
    
    def test_extract_arxiv_id_invalid(self):
        """Test invalid ID returns None."""
        assert extract_arxiv_id("invalid") is None
        assert extract_arxiv_id("https://example.com") is None


class TestTexFileDetection:
    """Test main LaTeX file detection."""
    
    def test_find_main_tex_file_single(self):
        """Test finding main file when only one exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tex_file = os.path.join(temp_dir, "paper.tex")
            with open(tex_file, 'w') as f:
                f.write("\\documentclass{article}")
            
            result = find_main_tex_file(temp_dir)
            assert result == tex_file
    
    def test_find_main_tex_file_with_documentclass(self):
        """Test finding file with documentclass."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple tex files
            with open(os.path.join(temp_dir, "section1.tex"), 'w') as f:
                f.write("\\section{Introduction}")
            
            main_file = os.path.join(temp_dir, "main.tex")
            with open(main_file, 'w') as f:
                f.write("\\documentclass{article}\\begin{document}")
            
            result = find_main_tex_file(temp_dir)
            assert result == main_file
    
    def test_find_main_tex_file_no_files(self):
        """Test when no tex files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = find_main_tex_file(temp_dir)
            assert result is None


class TestMetadataExtraction:
    """Test metadata extraction from LaTeX content."""
    
    def test_extract_title(self):
        """Test title extraction."""
        content = "\\title{A Great Paper About Something}"
        metadata = extract_paper_metadata(content)
        assert metadata['title'] == "A Great Paper About Something"
    
    def test_extract_author(self):
        """Test author extraction."""
        content = "\\author{John Doe and Jane Smith}"
        metadata = extract_paper_metadata(content)
        assert metadata['author'] == "John Doe and Jane Smith"
    
    def test_extract_abstract(self):
        """Test abstract extraction."""
        content = """
        \\begin{abstract}
        This is a sample abstract with some content.
        \\end{abstract}
        """
        metadata = extract_paper_metadata(content)
        assert "sample abstract" in metadata['abstract']
    
    def test_extract_metadata_with_latex_commands(self):
        """Test metadata extraction with LaTeX formatting."""
        content = "\\title{A Paper with \\textbf{Bold} and \\emph{Emphasis}}"
        metadata = extract_paper_metadata(content)
        assert metadata['title'] == "A Paper with Bold and Emphasis"


class TestLatexInputProcessing:
    """Test LaTeX input and include processing."""
    
    def test_process_latex_inputs_simple(self):
        """Test processing simple input commands."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create included file
            section_file = os.path.join(temp_dir, "section1.tex")
            with open(section_file, 'w') as f:
                f.write("This is section 1 content.")
            
            # Main content with input
            main_content = "\\input{section1}"
            
            result = process_latex_inputs(main_content, temp_dir)
            assert "This is section 1 content." in result
    
    def test_process_latex_inputs_with_extension(self):
        """Test processing when .tex extension is already specified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            section_file = os.path.join(temp_dir, "section1.tex")
            with open(section_file, 'w') as f:
                f.write("Section content")
            
            main_content = "\\input{section1.tex}"
            
            result = process_latex_inputs(main_content, temp_dir)
            assert "Section content" in result
    
    def test_process_latex_inputs_missing_file(self):
        """Test handling of missing included files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            main_content = "\\input{nonexistent}"
            
            result = process_latex_inputs(main_content, temp_dir)
            assert "Could not include nonexistent" in result


class TestArxivDownload:
    """Test full arXiv download functionality (mocked)."""
    
    @patch('paper_voice.arxiv_downloader.requests.get')
    @patch('paper_voice.arxiv_downloader.tarfile.open')
    def test_download_arxiv_paper_success(self, mock_tarfile, mock_requests):
        """Test successful paper download."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake tarball content"
        mock_requests.return_value = mock_response
        
        # Mock tarfile extraction
        mock_tar = Mock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar
        
        with patch('paper_voice.arxiv_downloader.find_main_tex_file') as mock_find_tex:
            with patch('paper_voice.arxiv_downloader.extract_figures_from_source') as mock_figures:
                with patch('builtins.open', create=True) as mock_open:
                    # Setup mocks
                    mock_find_tex.return_value = "/tmp/main.tex"
                    mock_figures.return_value = {"fig1.png": b"image data"}
                    mock_open.return_value.__enter__.return_value.read.return_value = "\\title{Test Paper}"
                    
                    # Test download
                    result = download_arxiv_paper("2301.12345")
                    
                    # Verify result structure
                    assert result is not None
                    assert result.arxiv_id == "2301.12345"
    
    def test_download_arxiv_paper_invalid_id(self):
        """Test download with invalid arXiv ID."""
        result = download_arxiv_paper("invalid-id")
        assert result is None
    
    @patch('paper_voice.arxiv_downloader.requests.get')
    def test_download_arxiv_paper_network_error(self, mock_requests):
        """Test download with network error."""
        mock_requests.side_effect = Exception("Network error")
        
        result = download_arxiv_paper("2301.12345")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])