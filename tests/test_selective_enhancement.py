"""Tests for selective content enhancement."""

from unittest.mock import patch

from paper_voice.selective_enhancer import (
    enhance_content_selectively,
    fix_pdf_extraction_issues,
)


class TestSelectiveEnhancement:
    """Test selective content enhancement functionality."""

    def test_enhance_content_with_api_key(self):
        """Test that content enhancement works with API key."""
        content = """
        This is a research paper about artificial intelligence.
        The equation shows $x^2 + y^2 = z^2$.
        """

        enhanced_content = """
        This is a research paper about artificial intelligence.
        The equation shows x squared plus y squared equals z squared.
        """

        with patch(
            "paper_voice.simple_llm_enhancer.enhance_document_simple"
        ) as mock_llm:
            mock_llm.return_value = enhanced_content
            result = enhance_content_selectively(content, "fake-key")

            assert "x squared plus y squared equals z squared" in result
            assert "$x^2 + y^2 = z^2$" not in result

    def test_enhance_content_without_api_key(self):
        """Test that content is unchanged without API key."""
        content = "This is test content with $x^2$ equation."

        result = enhance_content_selectively(content, "")
        assert result == content

        result = enhance_content_selectively(content, None)
        assert result == content

    def test_enhance_content_error_handling(self):
        """Test that errors in enhancement preserve original content."""
        content = "This is test content."

        with patch(
            "paper_voice.simple_llm_enhancer.enhance_document_simple"
        ) as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            result = enhance_content_selectively(content, "fake-key")

            # Should return original content when enhancement fails
            assert result == content

    def test_fix_pdf_extraction_issues_with_api_key(self):
        """Test PDF extraction issue fixes with API key."""
        # Greek letters are intentional: they simulate raw PDF math extraction
        content = "The equation shows that α increases with β."  # noqa: RUF001

        fixed_content = "The equation shows that alpha increases with beta."

        with patch(
            "paper_voice.simple_llm_enhancer.enhance_document_simple"
        ) as mock_llm:
            mock_llm.return_value = fixed_content
            result = fix_pdf_extraction_issues(content, "fake-key")

            assert "alpha" in result
            assert "beta" in result
            # Intentional Greek letters: the fix should have removed them
            assert "α" not in result  # noqa: RUF001
            assert "β" not in result

    def test_fix_pdf_extraction_issues_without_api_key(self):
        """Test PDF extraction without API key returns original."""
        # Greek letters are intentional: they simulate raw PDF math extraction
        content = "Test content with symbols α and β."  # noqa: RUF001

        result = fix_pdf_extraction_issues(content, "")
        assert result == content

        result = fix_pdf_extraction_issues(content, None)
        assert result == content

    def test_fix_pdf_extraction_issues_error_handling(self):
        """Test PDF extraction error handling."""
        content = "Test content."

        with patch(
            "paper_voice.simple_llm_enhancer.enhance_document_simple"
        ) as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            result = fix_pdf_extraction_issues(content, "fake-key")

            # Should return original content when fix fails
            assert result == content


class TestIntegration:
    """Integration tests for selective enhancement."""

    def test_content_enhancement_flow(self):
        """Test the complete content enhancement flow."""
        content = """
        This paper presents a novel approach to machine learning.
        The main equation is $E = mc^2$ which shows the relationship.
        Figure 1: Energy distribution graph.
        Table 1: Experimental results data.
        Conclusion paragraph here.
        """

        enhanced_content = """
        This paper presents a novel approach to machine learning.
        The main equation is energy equals mass times the speed of light squared
        which shows the relationship.
        Figure 1: Detailed visualization of energy distribution across
        different parameters.
        Table 1: Comprehensive experimental results showing performance metrics.
        Conclusion paragraph here.
        """

        with patch(
            "paper_voice.simple_llm_enhancer.enhance_document_simple"
        ) as mock_llm:
            mock_llm.return_value = enhanced_content
            result = enhance_content_selectively(content, "fake-key")

            # Check that all non-target text is preserved
            assert "This paper presents a novel approach" in result
            assert "Conclusion paragraph here." in result

            # Check that math is enhanced
            assert "energy equals mass times the speed of light squared" in result
            assert "$E = mc^2$" not in result

    def test_progress_callback(self):
        """Test that progress callbacks work correctly."""
        content = "Test content with $x^2$ equation."
        enhanced_content = "Test content with x squared equation."

        progress_messages = []

        def capture_progress(message):
            progress_messages.append(message)

        with patch(
            "paper_voice.simple_llm_enhancer.enhance_document_simple"
        ) as mock_llm:
            mock_llm.return_value = enhanced_content
            result = enhance_content_selectively(
                content, "fake-key", progress_callback=capture_progress
            )

            # Enhancement should still return the enhanced content
            assert result == enhanced_content

            # Should have captured progress messages
            assert len(progress_messages) > 0
            assert any(
                "Starting simple LLM enhancement" in msg for msg in progress_messages
            )
