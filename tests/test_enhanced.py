#!/usr/bin/env python3
"""
Quick test of the enhanced Paper Voice functionality.
"""

def test_latex_math_processing():
    """Test mathematical expression processing."""
    print("🧮 Testing mathematical expression processing...")
    
    try:
        from paper_voice.latex_processor import latex_math_to_speech
        
        test_expressions = [
            r"E = mc^2",
            r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
            r"\int_0^{\infty} e^{-x^2} dx",
            r"\sum_{n=1}^{\infty} \frac{1}{n^2}",
            r"\alpha + \beta = \gamma",
            r"A \subseteq B \cap C",
        ]
        
        for expr in test_expressions:
            spoken = latex_math_to_speech(expr)
            print(f"  LaTeX: {expr}")
            print(f"  Speech: {spoken}")
            print()
        
        print("✅ Math processing test passed!")
        return True
    
    except Exception as e:
        print(f"❌ Math processing test failed: {e}")
        return False


def test_latex_document_processing():
    """Test full LaTeX document processing."""
    print("📄 Testing LaTeX document processing...")
    
    try:
        from paper_voice.latex_processor import process_latex_document
        
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
        print(f"  Processed text preview: {result.text[:200]}...")
        print(f"  Found {len(result.equations)} equations")
        print(f"  Metadata: {result.metadata}")
        
        print("✅ LaTeX document processing test passed!")
        return True
    
    except Exception as e:
        print(f"❌ LaTeX document processing test failed: {e}")
        return False


def test_markdown_processing():
    """Test Markdown with math processing."""
    print("📝 Testing Markdown processing...")
    
    try:
        from paper_voice.latex_processor import process_markdown_with_math
        
        markdown_content = """
        # Test Paper
        
        The famous equation is $E = mc^2$.
        
        Display math:
        $$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$
        
        More inline math: $\alpha + \beta$.
        """
        
        result = process_markdown_with_math(markdown_content)
        print(f"  Processed result preview: {result[:200]}...")
        
        print("✅ Markdown processing test passed!")
        return True
    
    except Exception as e:
        print(f"❌ Markdown processing test failed: {e}")
        return False


def test_high_level_api():
    """Test the high-level API functions."""
    print("🎯 Testing high-level API...")
    
    try:
        from paper_voice import process_latex_to_speech, process_markdown_to_speech
        
        # Test LaTeX processing
        latex_content = r"The equation $E = mc^2$ is fundamental."
        result = process_latex_to_speech(latex_content)
        print(f"  LaTeX result: {result.spoken_text[:100]}...")
        
        # Test Markdown processing  
        markdown_content = "The integral $\int x dx$ equals $x^2/2$."
        result = process_markdown_to_speech(markdown_content)
        print(f"  Markdown result: {result.spoken_text[:100]}...")
        
        print("✅ High-level API test passed!")
        return True
    
    except Exception as e:
        print(f"❌ High-level API test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Paper Voice Enhanced - Functionality Test")
    print("=" * 50)
    
    tests = [
        test_latex_math_processing,
        test_latex_document_processing,
        test_markdown_processing,
        test_high_level_api,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The enhanced functionality is working.")
        print("\n🚀 Try running:")
        print("  python examples/enhanced_usage.py")
        print("  streamlit run paper_voice/streamlit/app_enhanced.py")
    else:
        print("⚠️  Some tests failed. Check the installation and dependencies.")
    
    return passed == len(tests)


if __name__ == "__main__":
    main()