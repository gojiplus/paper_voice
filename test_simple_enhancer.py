#!/usr/bin/env python3
"""
Test script to demonstrate the simple LLM enhancer approach.
"""

import os
from paper_voice.simple_llm_enhancer import enhance_document_simple

def test_simple_enhancer():
    # Test content from earlier conversation
    test_content = '''\\section{Introduction}
When the instrument \\(Z\\) changes from \\(z_1\\) to \\(z_2\\), the explained-share index is defined as:
\\[
\\text{explained-share} = 1 - \\frac{\\text{Var}(Y|X,Z=z_2)}{\\text{Var}(Y|X,Z=z_1)}
\\]

\\begin{threeparttable}
\\caption{Regression Results}
\\begin{tabular}{lcc}
\\toprule
Variable & Model 1 & Model 2 \\\\
\\midrule  
\\(\\alpha\\) & 0.45 & 0.52 \\\\
\\(\\beta\\) & 0.23 & 0.31 \\\\
\\bottomrule
\\end{tabular}
\\end{threeparttable}'''

    print("=== ORIGINAL CONTENT ===")
    print(test_content)
    print("\n" + "="*50 + "\n")
    
    # Get API key from environment
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: No OPENAI_API_KEY found in environment")
        return
    
    print("=== ENHANCED OUTPUT (Simple LLM Approach) ===")
    
    def progress_callback(msg):
        print(f"[PROGRESS] {msg}")
    
    try:
        enhanced = enhance_document_simple(test_content, api_key, progress_callback)
        print(enhanced)
        print("\n" + "="*50)
        print(f"Original length: {len(test_content)} characters")
        print(f"Enhanced length: {len(enhanced)} characters")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_simple_enhancer()