#!/usr/bin/env python3
"""Basic usage examples for Paper Voice.

Shows math processing and LLM explanations.
"""

import os

from paper_voice.latex_processor import process_inline_and_display_math
from paper_voice.simple_llm_enhancer import enhance_document_simple


def demo_math_processing():
    """Show basic math to speech conversion."""
    print("📖 PAPER VOICE: Basic Math Processing Demo")
    print("=" * 50)
    print()

    examples = [
        r"$\alpha^2 + \beta$",
        r"$\sqrt{n}(\hat{\theta} - \theta_0)$",
        r"$\sum_{i=1}^n X_i$",
        r"$\frac{\partial f}{\partial x}$",
    ]

    api_key = os.getenv("OPENAI_API_KEY")

    for expr in examples:
        print(f"LaTeX: {expr}")

        # Basic rule-based conversion (no LLM)
        basic = process_inline_and_display_math(expr, use_llm=False)
        print(f"Basic: {basic}")

        # LLM-enhanced explanation (if API key available)
        if api_key:
            try:
                enhanced = enhance_document_simple(expr, api_key)
                print(f"LLM:   {enhanced}")
            except Exception as e:
                print(f"LLM:   (Error: {e})")
        else:
            print(
                "LLM:   (Set OPENAI_API_KEY environment variable "
                "for enhanced explanations)"
            )

        print()


if __name__ == "__main__":
    demo_math_processing()
