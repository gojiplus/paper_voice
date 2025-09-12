#!/usr/bin/env python3
"""
Real demo using expressions from the user's statistical text.
Shows actual GPT-4 transformations of complex statistical expressions.
"""

import os
from openai import OpenAI

def get_math_explanation_prompt(latex_expr: str, context: str = "") -> str:
    """Create the detailed prompt for explaining mathematical expressions."""
    
    return f"""You are a world-class mathematics exposition expert. Your job is to convert mathematical expressions into crystal-clear natural language that a listener can easily understand through audio narration.

CRITICAL REQUIREMENTS:
1. Use PRECISE language that distinguishes between variables (e.g., "capital X" vs "lowercase x")
2. Explain the MEANING, not just read symbols
3. Use multiple clear sentences when needed
4. Provide context about what mathematical concepts mean
5. Make it sound natural when spoken aloud

MATHEMATICAL EXPRESSION:
{latex_expr}

DOCUMENT CONTEXT:
{context}

EXAMPLES OF EXCELLENT EXPLANATIONS:

Example 1:
LaTeX: $\\hat{{\\theta}} = \\frac{{1}}{{n}} \\sum_{{i=1}}^{{n}} \\psi(W_i; \\hat{{\\eta}}^{{(-k(i))}})$
Explanation: "Theta hat, which represents our estimator, is calculated as the average over all n observations. Specifically, we take the sum from i equals 1 to n of the function psi, evaluated at W subscript i, using the auxiliary parameter eta hat that was estimated on the complement sample excluding fold k of i, then divide this sum by n."

Example 2:
LaTeX: $\\sqrt{{n}}(\\hat{{\\theta}} - \\theta_0) \\xrightarrow{{d}} N(0, \\Sigma)$
Explanation: "As the sample size grows large, the quantity square root of n times the difference between our estimator theta hat and the true parameter theta naught converges in distribution to a normal distribution with mean zero and covariance matrix capital Sigma. This is a fundamental result showing that our estimator is asymptotically normal."

Example 3:
LaTeX: $E[Z_i \\varepsilon_i] = 0$
Explanation: "The expected value of the product of Z subscript i and epsilon subscript i equals zero. This states that the instrument Z subscript i is uncorrelated with the error term epsilon subscript i, which is the key exclusion restriction assumption in instrumental variables estimation."

KEY PRINCIPLES:
- Always distinguish "capital" vs "lowercase" for variables
- Explain subscripts clearly: "X subscript i" not "X sub i"  
- For Greek letters, use full names: "theta" not "θ", "epsilon" not "ε"
- Explain the mathematical meaning, not just the symbols
- Use "equals" instead of "="
- Use "times" instead of "×" or "·"
- For fractions, say "over" or explain as division
- Make it flow naturally when read aloud

Now explain this mathematical expression clearly and naturally:

"""

def demo_your_statistical_expressions():
    """Demo with expressions from the user's statistical text."""
    
    # Get API key from environment or user input
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            print("❌ API key required for this demo.")
            return
    
    client = OpenAI(api_key=api_key)
    
    # Expressions from your statistical text
    expressions = [
        {
            "latex": "\\|\\hat\\Delta\\|_\\infty = \\sup_y |\\hat\\Delta(y)|",
            "context": "This is from a statistical paper about testing instrument validity. The text discusses effect-size metrics including the sup-gap measure. Delta hat represents estimated deviations from the null hypothesis.",
            "description": "Supremum norm (sup-gap metric)"
        },
        {
            "latex": "\\|\\hat\\Delta\\|_2^2 = \\int \\hat\\Delta(y)^2\\,d\\hat H(y)",
            "context": "This appears in a section about effect-size metrics complementing p-values. H hat represents the empirical distribution function. This is the integrated gap measure.",
            "description": "Squared L2 norm (integrated gap)"
        },
        {
            "latex": "R^2_{\\text{CDF}} \\equiv 1 - \\frac{\\int\\!\\big(\\hat F_{Y\\mid Z=1}-\\hat F_{Y\\mid Z=0} - \\hat p_C(\\hat F_{1C}-\\hat F_{0C})\\big)^2\\,d\\hat H}{\\int\\!\\big(\\hat F_{Y\\mid Z=1}-\\hat F_{Y\\mid Z=0}\\big)^2\\,d\\hat H}",
            "context": "This is an explained-share index from instrumental variables analysis. F represents cumulative distribution functions, Z is the instrument, p_C is the complier probability. This measures how well the complier model explains CDF differences.",
            "description": "R-squared for CDFs (explained-share index)"
        }
    ]
    
    print("🧠 REAL DEMO: Your Statistical Expressions with GPT-4")
    print("=" * 80)
    print()
    print("Using expressions from your multiplicity and interpretation section...")
    print()
    
    for i, expr in enumerate(expressions, 1):
        print(f"EXPRESSION {i}: {expr['description']}")
        print("-" * 60)
        print()
        print("📝 LaTeX Expression:")
        print(f"   {expr['latex']}")
        print()
        
        print("🎧 TERRIBLE (Current approach):")
        if i == 1:
            bad_version = "pipe pipe hat Delta pipe pipe infinity equals sup y pipe hat Delta open parenthesis y close parenthesis pipe"
        elif i == 2:
            bad_version = "pipe pipe hat Delta pipe pipe 2 superscript 2 equals integral hat Delta open parenthesis y close parenthesis superscript 2 d hat H open parenthesis y close parenthesis"
        else:
            bad_version = "R superscript 2 CDF is defined as 1 minus fraction integral of big open parenthesis hat F Y pipe Z equals 1 minus hat F Y pipe Z equals 0 minus hat p C open parenthesis hat F 1 C minus hat F 0 C close parenthesis big close parenthesis superscript 2 d hat H over integral big open parenthesis hat F Y pipe Z equals 1 minus hat F Y pipe Z equals 0 big close parenthesis superscript 2 d hat H"
        
        print(f'   "{bad_version}"')
        print()
        
        print("🤖 Calling GPT-4...")
        
        try:
            # Get GPT-4 explanation
            prompt = get_math_explanation_prompt(expr['latex'], expr['context'])
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert mathematical exposition writer who converts mathematical expressions into clear, natural language explanations for audio narration."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            explanation = response.choices[0].message.content.strip()
            
            print("✅ GPT-4 Enhanced Explanation:")
            print(f'   "{explanation}"')
            print()
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print()
        
        print("=" * 80)
        print()


def main():
    """Run the real demo."""
    
    print("🎯 REAL DEMONSTRATION WITH YOUR STATISTICAL TEXT")
    print()
    print("This uses actual expressions from your multiplicity section")
    print("and shows real GPT-4 transformations...")
    print()
    
    try:
        demo_your_statistical_expressions()
        
        print("🎧 SUMMARY: WHAT YOU JUST SAW")
        print("=" * 50)
        print("✅ Complex statistical measures explained clearly")
        print("✅ Multiple sentences breaking down meaning")  
        print("✅ Precise variable handling and context")
        print("✅ Natural flow optimized for audio listeners")
        print("✅ Mathematical concepts made accessible")
        print()
        print("🚀 This is exactly what the Streamlit app does automatically!")
        print("   Upload your LaTeX/PDF and get crystal-clear narration.")
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        print()
        print("Check your API key and network connection.")

if __name__ == "__main__":
    main()