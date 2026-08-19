"""Tests for the enhanced Paper Voice functionality."""

import pytest

from paper_voice.document_processor import (
    process_latex_to_speech,
    process_markdown_to_speech,
)
from paper_voice.latex_processor import (
    latex_math_to_speech,
    process_latex_document,
    process_markdown_with_math,
)


class TestMathProcessing:
    """Test mathematical expression processing."""

    @pytest.mark.parametrize(
        ("expr", "expected_substrings"),
        [
            (r"E = mc^2", ["big e equals mc squared"]),
            (
                r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
                [
                    "fraction",
                    "minus b plus or minus",
                    "square root of b squared minus 4ac",
                    "over 2a",
                ],
            ),
            (
                r"\int_0^{\infty} e^{-x^2} dx",
                ["integral from 0 to infinity", "minus x squared dx"],
            ),
            (
                r"\sum_{n=1}^{\infty} \frac{1}{n^2}",
                ["sum from n equals 1 to infinity", "fraction 1 over n squared"],
            ),
            (r"\alpha + \beta = \gamma", ["alpha plus beta equals gamma"]),
            (
                r"A \subseteq B \cap C",
                ["big a subset or equal to big b intersection big c"],
            ),
        ],
    )
    def test_latex_math_to_speech(self, expr, expected_substrings):
        """Test that LaTeX expressions convert to the expected spoken form."""
        spoken = latex_math_to_speech(expr)
        for expected in expected_substrings:
            assert expected in spoken
        assert "\\" not in spoken


class TestLatexDocumentProcessing:
    """Test full LaTeX document processing."""

    def test_process_latex_document(self):
        """Test processing a complete LaTeX document."""
        latex_content = r"""
        \documentclass{article}
        \title{Test Document}

        \begin{document}
        \section{Introduction}
        This is a test with $E = mc^2$ and other math.

        \begin{equation}
        \int_0^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
        \end{equation}

        \end{document}
        """

        result = process_latex_document(latex_content)

        # Sections and inline math are converted to spoken form
        assert "Section: Introduction" in result.text
        assert "big e equals mc squared" in result.text
        assert "$E = mc^2$" not in result.text

        # The display equation is extracted and converted
        assert len(result.equations) == 1
        assert "integral from 0 to infinity" in result.equations[0]
        assert "square root of pi over 2" in result.equations[0]

        # Metadata captures the title
        assert result.metadata["title"] == "Test Document"


class TestMarkdownProcessing:
    """Test Markdown with math processing."""

    def test_process_markdown_with_math(self):
        """Test that inline and display math in Markdown are converted."""
        markdown_content = (
            "# Test Paper\n"
            "\n"
            "The famous equation is $E = mc^2$.\n"
            "\n"
            "Display math:\n"
            r"$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$"
            "\n\n"
            r"More inline math: $\alpha + \beta$."
            "\n"
        )

        result = process_markdown_with_math(markdown_content)

        assert "big e equals mc squared" in result
        assert "Display equation: integral from minus infinity to infinity" in result
        assert "square root of pi" in result
        assert "alpha plus beta" in result
        assert "$" not in result


class TestHighLevelApi:
    """Test the high-level document processing API."""

    def test_process_latex_to_speech(self):
        """Test the high-level LaTeX to speech pipeline."""
        latex_content = r"The equation $E = mc^2$ is fundamental."
        result = process_latex_to_speech(latex_content)
        assert (
            result.spoken_text == "The equation big e equals mc squared is fundamental."
        )

    def test_process_markdown_to_speech(self):
        """Test the high-level Markdown to speech pipeline."""
        markdown_content = r"The integral $\int x dx$ equals $x^2/2$."
        result = process_markdown_to_speech(markdown_content)
        assert (
            result.spoken_text
            == "The integral integral x dx equals x squared divided by 2."
        )
