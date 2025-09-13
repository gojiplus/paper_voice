"""
Tests for LaTeX processing and mathematical expression handling.
"""

import pytest
from paper_voice.latex_processor import (
    latex_math_to_speech,
    _speak_variable,
    _parse_superscript,
    _parse_subscript,
    process_inline_and_display_math,
    extract_figures_and_tables,
    process_latex_document,
    GREEK_MAPPING,
    MATH_OPERATORS
)


class TestVariableSpeaking:
    """Test conversion of variables to spoken form."""
    
    def test_speak_greek_letters(self):
        """Test Greek letter pronunciation."""
        assert _speak_variable("alpha") == "alpha"
        assert _speak_variable("beta") == "beta"
        assert _speak_variable("Gamma") == "big gamma"
        assert _speak_variable("Delta") == "big delta"
    
    def test_speak_regular_letters(self):
        """Test regular letter pronunciation."""
        assert _speak_variable("x") == "x"
        assert _speak_variable("X") == "big x"
        assert _speak_variable("y") == "y"
        assert _speak_variable("Z") == "big z"
    
    def test_speak_complex_variables(self):
        """Test complex variable names."""
        assert _speak_variable("var1") == "var1"
        assert _speak_variable("theta_1") == "theta_1"


class TestSuperscriptSubscript:
    """Test superscript and subscript parsing."""
    
    def test_parse_common_superscripts(self):
        """Test common superscript expressions."""
        assert _parse_superscript("2") == "squared"
        assert _parse_superscript("3") == "cubed"
        assert _parse_superscript("4") == "to the fourth power"
        assert _parse_superscript("-1") == "inverse"
        assert _parse_superscript("T") == "transpose"
    
    def test_parse_general_superscripts(self):
        """Test general superscript expressions."""
        assert _parse_superscript("n") == "to the power n"
        assert _parse_superscript("{n+1}") == "to the power n+1"
    
    def test_parse_subscripts(self):
        """Test subscript parsing."""
        assert _parse_subscript("i") == "sub i"
        assert _parse_subscript("{max}") == "sub max"
        assert _parse_subscript("0") == "sub 0"


class TestLatexMathToSpeech:
    """Test LaTeX to speech conversion."""
    
    def test_simple_expressions(self):
        """Test simple mathematical expressions."""
        assert "x squared" in latex_math_to_speech("x^2")
        assert "x sub i" in latex_math_to_speech("x_i")
        assert "alpha" in latex_math_to_speech("\\alpha")
        assert "big gamma" in latex_math_to_speech("\\Gamma")
    
    def test_complex_expressions(self):
        """Test complex mathematical expressions."""
        # Quadratic formula components
        result = latex_math_to_speech("x^2 + y^2 = z^2")
        assert "x squared" in result
        assert "y squared" in result
        assert "z squared" in result
        assert "plus" in result
        assert "equals" in result
    
    def test_operators(self):
        """Test mathematical operators."""
        assert "less than or equal to" in latex_math_to_speech("x \\leq y")
        assert "greater than or equal to" in latex_math_to_speech("a \\geq b")
        assert "not equal to" in latex_math_to_speech("p \\neq q")
        assert "approximately equal to" in latex_math_to_speech("x \\approx y")
    
    def test_functions(self):
        """Test mathematical functions."""
        assert "sine" in latex_math_to_speech("\\sin(x)")
        assert "cosine" in latex_math_to_speech("\\cos(x)")
        assert "natural logarithm" in latex_math_to_speech("\\ln(x)")
        assert "square root" in latex_math_to_speech("\\sqrt{x}")
    
    def test_parentheses_brackets(self):
        """Test parentheses and brackets."""
        result = latex_math_to_speech("(x + y)[z - w]")
        assert "open parenthesis" in result
        assert "close parenthesis" in result  
        assert "open bracket" in result
        assert "close bracket" in result
    
    def test_fractions(self):
        """Test fraction handling."""
        result = latex_math_to_speech("\\frac{x}{y}")
        assert "fraction" in result
        assert "over" in result
    
    def test_summations_integrals(self):
        """Test summations and integrals."""
        result = latex_math_to_speech("\\sum_{i=1}^{n} x_i")
        assert "sum" in result
        assert "from" in result
        assert "to" in result
        
        result = latex_math_to_speech("\\int_0^\\infty")
        assert "integral" in result
        assert "from" in result
        assert "infinity" in result


class TestInlineDisplayMath:
    """Test inline and display math processing."""
    
    def test_inline_math_processing(self):
        """Test inline math ($...$) processing."""
        text = "The equation $x^2 + y^2 = r^2$ is important."
        result = process_inline_and_display_math(text)
        
        assert "$" not in result  # Dollar signs should be removed
        assert "squared" in result
        assert "The equation" in result
        assert "is important" in result
    
    def test_display_math_processing(self):
        """Test display math ($$...$$) processing."""
        text = "Consider the equation: $$\\int_0^\\infty e^{-x} dx = 1$$"
        result = process_inline_and_display_math(text)
        
        assert "$$" not in result
        assert "Display equation:" in result
        assert "integral" in result
    
    def test_mixed_math_processing(self):
        """Test mixed inline and display math."""
        text = "We have $x = 1$ and $$y = \\sum_{i=1}^n x_i$$"
        result = process_inline_and_display_math(text)
        
        assert "$" not in result
        assert "Display equation:" in result
        assert "sum" in result
    
    def test_nested_dollar_signs(self):
        """Test handling of nested or escaped dollar signs."""
        text = "Price is \\$5 and equation is $x^2$"
        result = process_inline_and_display_math(text)
        
        # Should process the math but leave the escaped dollar
        assert "squared" in result
        assert "\\$5" in result or "$5" in result


class TestFigureTableExtraction:
    """Test figure and table extraction from LaTeX."""
    
    def test_figure_extraction(self):
        """Test extraction of figure environments."""
        latex_text = """
        \\begin{figure}
        \\includegraphics{image.png}
        \\caption{This is a test figure}
        \\end{figure}
        """
        
        figures, tables = extract_figures_and_tables(latex_text)
        
        assert len(figures) == 1
        assert "test figure" in figures[0][0]
    
    def test_table_extraction(self):
        """Test extraction of table environments."""
        latex_text = """
        \\begin{table}
        \\begin{tabular}{|c|c|}
        A & B \\\\
        1 & 2 \\\\
        \\end{tabular}
        \\caption{Test table data}
        \\end{table}
        """
        
        figures, tables = extract_figures_and_tables(latex_text)
        
        assert len(tables) == 1
        assert "Test table data" in tables[0][0]
        assert "A & B" in tables[0][1]
    
    def test_multiple_figures_tables(self):
        """Test extraction of multiple figures and tables."""
        latex_text = """
        \\begin{figure}
        \\caption{Figure 1}
        \\end{figure}
        
        \\begin{table}
        \\caption{Table 1}
        \\begin{tabular}{c}
        Data
        \\end{tabular}
        \\end{table}
        
        \\begin{figure}
        \\caption{Figure 2}
        \\end{figure}
        """
        
        figures, tables = extract_figures_and_tables(latex_text)
        
        assert len(figures) == 2
        assert len(tables) == 1
        assert "Figure 1" in figures[0][0]
        assert "Figure 2" in figures[1][0]
        assert "Table 1" in tables[0][0]


class TestLatexDocumentProcessing:
    """Test complete LaTeX document processing."""
    
    def test_document_metadata_extraction(self):
        """Test extraction of document metadata."""
        latex_doc = """
        \\title{A Test Paper}
        \\author{John Doe}
        \\begin{abstract}
        This is the abstract.
        \\end{abstract}
        \\begin{document}
        Content here.
        \\end{document}
        """
        
        result = process_latex_document(latex_doc)
        
        assert result.metadata['title'] == "A Test Paper"
        assert result.metadata['author'] == "John Doe"
        assert "abstract" in result.metadata
    
    def test_document_math_processing(self):
        """Test math processing in complete documents."""
        latex_doc = """
        \\documentclass{article}
        \\begin{document}
        We have $x^2$ and 
        \\begin{equation}
        y = \\int_0^1 f(x) dx
        \\end{equation}
        \\end{document}
        """
        
        result = process_latex_document(latex_doc)
        
        assert "squared" in result.text
        assert "integral" in result.text
    
    def test_document_structure_cleaning(self):
        """Test cleaning of LaTeX structural elements."""
        latex_doc = """
        \\section{Introduction}
        \\subsection{Background}
        This is \\textbf{bold} and \\emph{emphasized} text.
        """
        
        result = process_latex_document(latex_doc)
        
        assert "Section: Introduction" in result.text
        assert "Subsection: Background" in result.text
        assert "bold" in result.text
        assert "emphasized" in result.text
        assert "\\textbf" not in result.text  # LaTeX commands removed


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_expressions(self):
        """Test handling of empty expressions."""
        assert latex_math_to_speech("") == ""
        assert latex_math_to_speech("   ") == ""
    
    def test_unknown_commands(self):
        """Test handling of unknown LaTeX commands."""
        result = latex_math_to_speech("\\unknowncommand{x}")
        # Should handle gracefully without crashing
        assert isinstance(result, str)
    
    def test_malformed_expressions(self):
        """Test handling of malformed expressions."""
        # Unmatched braces
        result = latex_math_to_speech("x^{2")
        assert isinstance(result, str)
        
        # Multiple operators
        result = latex_math_to_speech("x ++ y")
        assert isinstance(result, str)
    
    def test_very_long_expressions(self):
        """Test handling of very long expressions."""
        long_expr = "x^2 + y^2 + " * 100 + "z^2"
        result = latex_math_to_speech(long_expr)
        
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__])