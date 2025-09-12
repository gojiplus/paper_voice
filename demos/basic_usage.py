#!/usr/bin/env python3
"""
Basic usage examples for Paper Voice.
Shows math processing and LLM explanations.
"""

import os
from paper_voice.llm_math_explainer import explain_math_with_llm_sync

def demo_math_processing():
    """Show basic math to speech conversion."""
    print("📖 PAPER VOICE: Basic Math Processing Demo")
    print("=" * 50)
    print()
    
    examples = [
        r"$\alpha^2 + \beta$",
        r"$\sqrt{n}(\hat{\theta} - \theta_0)$",
        r"$\sum_{i=1}^n X_i$",
        r"$\frac{\partial f}{\partial x}$"
    ]
    
    for expr in examples:
        print(f"LaTeX: {expr}")
        
        # Basic conversion (without LLM)
        from paper_voice.math_to_speech import process_text_with_math
        basic = process_text_with_math(expr)
        print(f"Basic: {basic}")
        
        # LLM-enhanced explanation (if API key available)
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                enhanced = explain_math_with_llm_sync(expr, api_key)
                print(f"LLM:   {enhanced}")
            except Exception as e:
                print(f"LLM:   (Error: {e})")
        else:
            print("LLM:   (Set OPENAI_API_KEY environment variable for enhanced explanations)")
        
        print()

if __name__ == "__main__":
    demo_math_processing()