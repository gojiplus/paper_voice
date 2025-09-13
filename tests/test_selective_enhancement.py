"""Tests for selective content enhancement."""

import pytest
from unittest.mock import Mock, patch
from paper_voice.selective_enhancer import (
    enhance_content_selectively,
    fix_pdf_extraction_issues,
    _enhance_math_expressions,
    _enhance_figures,
    _enhance_tables
)


class TestSelectiveEnhancement:
    """Test selective content enhancement functionality."""
    
    def test_preserve_non_math_text(self):
        """Test that non-mathematical text is preserved unchanged."""
        content = """
        This is a research paper about artificial intelligence.
        The introduction discusses various approaches to machine learning.
        We present a novel algorithm that improves accuracy by 15%.
        The experimental results demonstrate significant improvements.
        """
        
        # Mock the API call to return empty since no math expressions
        with patch('paper_voice.selective_enhancer.explain_math_with_llm_sync') as mock_llm:
            result = _enhance_math_expressions(content, "fake-key")
            
            # Should be identical since no math expressions
            assert result == content
            # LLM should not be called since no math found
            mock_llm.assert_not_called()
    
    def test_enhance_only_math_expressions(self):
        """Test that only mathematical expressions are enhanced."""
        content = """
        The algorithm achieves accuracy of $\\alpha^2 + \\beta$ where alpha is the learning rate.
        We also tested with the equation $$\\int_0^\\infty f(x) dx = \\gamma$$.
        The rest of this text should remain unchanged completely.
        """
        
        # Mock the math explanation function
        mock_explanation = Mock()
        mock_explanation.natural_explanation = "alpha squared plus beta"
        
        with patch('paper_voice.selective_enhancer.explain_math_with_llm_sync', return_value=mock_explanation):
            result = _enhance_math_expressions(content, "fake-key")
            
            # Check that non-math text is preserved
            assert "The algorithm achieves accuracy of" in result
            assert "The rest of this text should remain unchanged completely." in result
            
            # Check that math expressions are replaced
            assert "alpha squared plus beta" in result
            assert "$\\alpha^2 + \\beta$" not in result
    
    def test_enhance_figure_captions_only(self):
        """Test that only figure captions are enhanced, not other text."""
        content = """
        This paper presents novel results.
        
        Figure 1: A diagram showing the network architecture with input layer, hidden layers, and output layer.
        
        The methodology section describes our approach in detail.
        We used standard machine learning techniques.
        """
        
        mock_enhanced_caption = "A clear diagram illustrating the neural network structure with three main components: an input layer for data ingestion, multiple hidden layers for feature processing, and an output layer for final predictions."
        
        with patch('paper_voice.selective_enhancer.summarize_figure_with_llm', return_value=mock_enhanced_caption):
            result = _enhance_figures(content, "fake-key")
            
            # Check that non-figure text is preserved exactly
            assert "This paper presents novel results." in result
            assert "The methodology section describes our approach in detail." in result
            assert "We used standard machine learning techniques." in result
            
            # Check that figure caption is enhanced
            assert mock_enhanced_caption in result
            assert "Figure 1:" in result  # Prefix should be preserved
            
            # Original caption should be replaced
            assert "A diagram showing the network architecture" not in result
    
    def test_enhance_table_captions_only(self):
        """Test that only table captions are enhanced."""
        content = """
        The experimental setup included multiple datasets.
        
        Table 1: Results showing accuracy, precision, and recall for different models.
        
        These results demonstrate the effectiveness of our approach.
        """
        
        mock_enhanced_caption = "Comprehensive performance metrics comparing various machine learning models across three key evaluation criteria: accuracy measuring overall correctness, precision indicating positive prediction reliability, and recall showing the ability to identify all relevant instances."
        
        with patch('paper_voice.selective_enhancer.summarize_table_with_llm', return_value=mock_enhanced_caption):
            result = _enhance_tables(content, "fake-key")
            
            # Check that non-table text is preserved
            assert "The experimental setup included multiple datasets." in result
            assert "These results demonstrate the effectiveness of our approach." in result
            
            # Check that table caption is enhanced
            assert mock_enhanced_caption in result
            assert "Table 1:" in result  # Prefix should be preserved
    
    def test_fix_garbled_symbols_only(self):
        """Test that only garbled mathematical symbols are fixed."""
        content = """
        The equation shows that α increases with β.
        Normal text should remain unchanged.
        The integral ∫f(x)dx represents the area under the curve.
        This concludes our analysis of the algorithm.
        """
        
        # Mock the OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
        The equation shows that alpha increases with beta.
        Normal text should remain unchanged.
        The integral of f(x) with respect to x represents the area under the curve.
        This concludes our analysis of the algorithm.
        """.strip()
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('paper_voice.selective_enhancer.OpenAI', return_value=mock_client):
            result = fix_pdf_extraction_issues(content, "fake-key")
            
            # Check that non-mathematical text is preserved
            assert "Normal text should remain unchanged." in result
            assert "This concludes our analysis of the algorithm." in result
            
            # Check that garbled symbols are fixed
            assert "alpha" in result and "α" not in result
            assert "beta" in result and "β" not in result
            assert "integral of f(x)" in result and "∫f(x)" not in result
    
    def test_error_handling_preserves_original(self):
        """Test that errors in enhancement preserve the original content."""
        content = "This is a test with $x^2$ equation."
        
        # Mock LLM to raise an exception
        with patch('paper_voice.selective_enhancer.explain_math_with_llm_sync', side_effect=Exception("API Error")):
            result = _enhance_math_expressions(content, "fake-key")
            
            # Should return original content when enhancement fails
            assert result == content
    
    def test_mixed_content_selective_enhancement(self):
        """Test selective enhancement with mixed content types."""
        content = """
        Introduction to the study.
        
        The main equation is $E = mc^2$ which shows the relationship.
        
        Figure 1: Energy distribution graph.
        
        Table 1: Experimental results data.
        
        Conclusion paragraph here.
        """
        
        # Mock all enhancement functions
        math_explanation = Mock()
        math_explanation.natural_explanation = "energy equals mass times the speed of light squared"
        
        with patch('paper_voice.selective_enhancer.explain_math_with_llm_sync', return_value=math_explanation), \
             patch('paper_voice.selective_enhancer.summarize_figure_with_llm', return_value="Enhanced figure description"), \
             patch('paper_voice.selective_enhancer.summarize_table_with_llm', return_value="Enhanced table description"):
            
            result = enhance_content_selectively(content, "fake-key")
            
            # Check that all non-target text is preserved
            assert "Introduction to the study." in result
            assert "Conclusion paragraph here." in result
            
            # Check that only targeted content is enhanced
            assert "energy equals mass times the speed of light squared" in result
            assert "Enhanced figure description" in result
            assert "Enhanced table description" in result
            
            # Original math should be replaced
            assert "$E = mc^2$" not in result


class TestIntegration:
    """Integration tests for selective enhancement."""
    
    def test_no_unwanted_summarization(self):
        """Test that the selective enhancer doesn't summarize or rewrite regular text."""
        # This is the key test - ensuring no summarization happens
        content = """
        This paper presents a novel approach to machine learning that combines deep neural networks 
        with reinforcement learning techniques. The proposed method achieves state-of-the-art 
        performance on multiple benchmark datasets. We introduce a new architecture that 
        incorporates attention mechanisms and memory networks to improve learning efficiency.
        
        The experimental evaluation demonstrates significant improvements over existing methods.
        Our approach reduces training time by 40% while maintaining accuracy levels above 95%.
        These results indicate that the proposed technique offers practical advantages for 
        real-world applications in natural language processing and computer vision.
        """
        
        # Mock to ensure no LLM calls are made (since no math/figures/tables)
        with patch('paper_voice.selective_enhancer.explain_math_with_llm_sync') as mock_math, \
             patch('paper_voice.selective_enhancer.summarize_figure_with_llm') as mock_fig, \
             patch('paper_voice.selective_enhancer.summarize_table_with_llm') as mock_table:
            
            result = enhance_content_selectively(content, "fake-key")
            
            # Result should be identical - no enhancement needed
            assert result == content
            
            # No LLM functions should be called
            mock_math.assert_not_called()
            mock_fig.assert_not_called() 
            mock_table.assert_not_called()