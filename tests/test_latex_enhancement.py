"""Tests for LaTeX content enhancement with LLM."""

import pytest
from unittest.mock import Mock, patch
from paper_voice.content_processor import process_content_unified, process_latex_content
from paper_voice.latex_processor import process_latex_document


class TestLatexEnhancement:
    """Test LaTeX content processing with LLM enhancement."""
    
    def test_latex_math_with_llm(self):
        """Test that LaTeX math expressions get LLM enhancement."""
        latex_content = r"""
        \documentclass{article}
        \begin{document}
        
        The main result shows that $\alpha^2 + \beta = \gamma$ where alpha is the learning rate.
        
        We also have the display equation:
        $$\int_0^{\infty} f(x) dx = \sum_{n=1}^{\infty} \frac{1}{n^2}$$
        
        This concludes our mathematical analysis.
        \end{document}
        """
        
        # Mock the simple LLM enhancer function
        enhanced_result = """
        The main result shows that alpha squared plus beta equals gamma where alpha is the learning rate.
        
        We also have the display equation: the integral from 0 to infinity of f of x dx equals the sum from n equals 1 to infinity of 1 over n squared.
        
        This concludes our mathematical analysis.
        """
        
        with patch('paper_voice.simple_llm_enhancer.enhance_document_simple', return_value=enhanced_result):
            result = process_latex_content(latex_content, api_key="fake-key", use_llm_enhancement=True)
            
            # Check that math expressions are enhanced with LLM
            assert "alpha squared plus beta equals gamma" in result
            
            # Check that non-math text is preserved
            assert "The main result shows that" in result
            assert "This concludes our mathematical analysis." in result
            
            # Original LaTeX math should be replaced
            assert r"$\alpha^2 + \beta = \gamma$" not in result
    
    def test_latex_figures_with_llm(self):
        """Test that LaTeX figure captions get LLM enhancement."""
        latex_content = r"""
        \documentclass{article}
        \begin{document}
        
        The methodology is illustrated below.
        
        \begin{figure}
        \includegraphics{network.png}
        \caption{Network architecture showing input layer, hidden layers, and output layer.}
        \end{figure}
        
        The results demonstrate effectiveness.
        \end{document}
        """
        
        enhanced_caption = "A comprehensive diagram illustrating the neural network structure with clearly defined input processing, intermediate feature extraction layers, and final output generation."
        
        with patch('paper_voice.figure_table_summarizer.summarize_figure_with_llm', return_value=enhanced_caption):
            processed = process_latex_document(
                latex_content, 
                summarize_figures=True, 
                api_key="fake-key"
            )
            
            # Check that figure caption is enhanced
            assert enhanced_caption in str(processed.figures)
            
            # Check that non-figure text is preserved
            assert "The methodology is illustrated below." in processed.text
            assert "The results demonstrate effectiveness." in processed.text
    
    def test_latex_tables_with_llm(self):
        """Test that LaTeX table captions get LLM enhancement."""
        latex_content = r"""
        \documentclass{article}
        \begin{document}
        
        The experimental results are shown below.
        
        \begin{table}
        \caption{Performance metrics for different models.}
        \begin{tabular}{|l|c|c|}
        Model & Accuracy & Speed \\
        A & 95\% & Fast \\
        B & 92\% & Slow \\
        \end{tabular}
        \end{table}
        
        These results confirm our hypothesis.
        \end{document}
        """
        
        enhanced_caption = "Comprehensive performance comparison table presenting accuracy percentages and processing speed characteristics for two distinct machine learning models, enabling direct comparison of their effectiveness and efficiency trade-offs."
        
        with patch('paper_voice.figure_table_summarizer.summarize_table_with_llm', return_value=enhanced_caption):
            processed = process_latex_document(
                latex_content, 
                summarize_tables=True, 
                api_key="fake-key"
            )
            
            # Check that table caption is enhanced
            assert enhanced_caption in str(processed.tables)
            
            # Check that non-table text is preserved
            assert "The experimental results are shown below." in processed.text
            assert "These results confirm our hypothesis." in processed.text
    
    def test_unified_processor_latex(self):
        """Test the unified processor with LaTeX content."""
        latex_content = r"""
        \documentclass{article}
        \title{Research Paper}
        \author{Test Author}
        \begin{document}
        
        This paper presents $E = mc^2$ which is Einstein's famous equation.
        
        \begin{figure}
        \caption{Energy-mass relationship diagram.}
        \end{figure}
        
        The conclusion follows from the analysis.
        \end{document}
        """
        
        # Mock the simple LLM enhancer
        enhanced_content = """
        Title: Research Paper
        Author: Test Author
        
        This paper presents energy equals mass times the speed of light squared which is Einstein's famous equation.
        
        Figure: Detailed visualization showing the fundamental relationship between energy and mass as described by Einstein's theory of special relativity.
        
        The conclusion follows from the analysis.
        """
        
        with patch('paper_voice.simple_llm_enhancer.enhance_document_simple', return_value=enhanced_content), \
             patch('paper_voice.figure_table_summarizer.summarize_figure_with_llm', return_value="Enhanced figure description"):
            
            processed_doc = process_content_unified(
                content=latex_content,
                input_type='latex',
                api_key='fake-key',
                use_llm_enhancement=True
            )
            
            # Check that content type is detected
            assert processed_doc.input_type == 'latex'
            
            # Check that math is enhanced
            assert "energy equals mass times the speed of light squared" in processed_doc.enhanced_text
            
            # Check content characteristics
            assert processed_doc.has_math == True or processed_doc.has_figures == True
    
    def test_latex_without_llm_enhancement(self):
        """Test LaTeX processing with LLM enhancement disabled."""
        latex_content = r"""
        \documentclass{article}
        \begin{document}
        The equation $x^2 + y^2 = z^2$ is the Pythagorean theorem.
        \end{document}
        """
        
        processed_doc = process_content_unified(
            content=latex_content,
            input_type='latex',
            api_key=None,
            use_llm_enhancement=False
        )
        
        # Should use basic rule-based conversion
        assert "x squared plus y squared equals z squared" in processed_doc.enhanced_text
        assert "The equation" in processed_doc.enhanced_text
        assert "is the Pythagorean theorem." in processed_doc.enhanced_text
        
        # Should not contain original LaTeX
        assert "$x^2 + y^2 = z^2$" not in processed_doc.enhanced_text
    
    def test_preserve_non_math_text_in_latex(self):
        """Test that non-mathematical LaTeX text is preserved exactly."""
        latex_content = r"""
        \documentclass{article}
        \begin{document}
        
        This research paper investigates the computational complexity of machine learning 
        algorithms in distributed systems. We propose a novel framework that optimizes
        resource allocation while maintaining high accuracy levels.
        
        The mathematical foundation relies on $\theta = \arg\min L(\theta)$ optimization.
        
        Our experimental evaluation demonstrates significant improvements over existing 
        approaches. The proposed method achieves 95% accuracy while reducing computational
        overhead by 40%. These results indicate practical advantages for real-world applications.
        
        \end{document}
        """
        
        enhanced_content = """
        This research paper investigates the computational complexity of machine learning 
        algorithms in distributed systems. We propose a novel framework that optimizes
        resource allocation while maintaining high accuracy levels.
        
        The mathematical foundation relies on theta equals the argument that minimizes the loss function of theta optimization.
        
        Our experimental evaluation demonstrates significant improvements over existing 
        approaches. The proposed method achieves 95% accuracy while reducing computational
        overhead by 40%. These results indicate practical advantages for real-world applications.
        """
        
        with patch('paper_voice.simple_llm_enhancer.enhance_document_simple', return_value=enhanced_content):
            processed_doc = process_content_unified(
                content=latex_content,
                input_type='latex',
                api_key='fake-key',
                use_llm_enhancement=True
            )
            
            result = processed_doc.enhanced_text
            
            # All non-mathematical text should be preserved exactly
            assert "This research paper investigates the computational complexity" in result
            assert "We propose a novel framework that optimizes" in result
            assert "Our experimental evaluation demonstrates significant improvements" in result
            assert "The proposed method achieves 95% accuracy" in result
            assert "These results indicate practical advantages" in result
            
            # Math should be enhanced
            assert "theta equals the argument that minimizes the loss function of theta" in result
            
            # Original math should be replaced
            assert r"$\theta = \arg\min L(\theta)$" not in result


class TestLatexIntegration:
    """Integration tests for LaTeX processing."""
    
    def test_complex_latex_document(self):
        """Test processing of a complex LaTeX document with multiple elements."""
        complex_latex = r"""
        \documentclass{article}
        \title{Machine Learning Optimization}
        \author{Research Team}
        \begin{document}
        \maketitle
        
        \section{Introduction}
        This paper presents advances in optimization theory.
        
        \section{Methodology}
        The core algorithm minimizes $J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$.
        
        \begin{figure}
        \caption{Convergence analysis of the optimization algorithm.}
        \end{figure}
        
        \begin{table}
        \caption{Comparison of different optimization methods.}
        \end{table}
        
        \section{Results}
        The algorithm converges to $\theta^* = \arg\min J(\theta)$ efficiently.
        
        \section{Conclusion}
        Our method demonstrates superior performance across all metrics.
        \end{document}
        """
        
        # Mock enhanced result
        enhanced_result = """
        Title: Machine Learning Optimization
        Author: Research Team
        
        Section: Introduction
        This paper presents advances in optimization theory.
        
        Section: Methodology
        The core algorithm minimizes the cost function J of theta equals one over two m times the sum from i equals 1 to m of the squared difference between the hypothesis function and the actual value.
        
        Figure: Detailed graph showing the convergence behavior of the optimization algorithm over iterations.
        
        Table: Comprehensive comparison of various optimization techniques.
        
        Section: Results
        The algorithm converges to theta star equals the argument that minimizes the cost function J of theta efficiently.
        
        Section: Conclusion
        Our method demonstrates superior performance across all metrics.
        """
        
        enhanced_fig_caption = "Detailed graph showing the convergence behavior of the optimization algorithm over iterations, including convergence rate and final accuracy achieved."
        enhanced_table_caption = "Comprehensive comparison table presenting performance metrics including convergence speed, final accuracy, and computational requirements for various optimization techniques."
        
        with patch('paper_voice.simple_llm_enhancer.enhance_document_simple', return_value=enhanced_result), \
             patch('paper_voice.figure_table_summarizer.summarize_figure_with_llm', return_value=enhanced_fig_caption), \
             patch('paper_voice.figure_table_summarizer.summarize_table_with_llm', return_value=enhanced_table_caption):
            
            processed_doc = process_content_unified(
                content=complex_latex,
                input_type='latex',
                api_key='fake-key',
                use_llm_enhancement=True
            )
            
            result = processed_doc.enhanced_text
            
            # Check that all sections are preserved
            assert "Section: Introduction" in result
            assert "Section: Methodology" in result  
            assert "Section: Results" in result
            assert "Section: Conclusion" in result
            
            # Check that all non-math text is preserved
            assert "This paper presents advances in optimization theory." in result
            assert "Our method demonstrates superior performance across all metrics." in result
            
            # Check that math is enhanced
            assert "the cost function J of theta equals" in result
            assert "theta star equals the argument that minimizes" in result