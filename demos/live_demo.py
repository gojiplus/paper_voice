#!/usr/bin/env python3
"""
Live demonstration of LLM-enhanced mathematical explanations.
This shows actual GPT-4 output for mathematical expressions.
"""

import os
import re
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
LaTeX: $Y_i(d) \\text{{ for }} d \\in \\{{0,1\\}}$
Explanation: "Y subscript i of d represents the potential outcome for individual i under treatment status d, where d can take the value 0 for the control condition or 1 for the treatment condition."

Example 4:
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

def live_demo_with_gpt4():
    """Run live demo with actual GPT-4 calls."""
    
    # Get API key from environment or user input
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            print("❌ API key required for this demo.")
            return
    
    client = OpenAI(api_key=api_key)
    
    # Test expressions from your LATE document
    expressions = [
        {
            "latex": "\\hat{\\theta} = \\frac{1}{n} \\sum_{i=1}^{n} \\psi(W_i; \\hat{\\eta}^{(-k(i))})",
            "context": "This is from a statistical paper about Local Average Treatment Effects (LATE) and cross-fitting. The document discusses instrumental variables and causal inference in econometrics.",
            "description": "Cross-fitted estimator formula"
        },
        {
            "latex": "\\sqrt{n}(\\hat{\\theta} - \\theta_0) \\xrightarrow{d} N(0, \\Sigma)",
            "context": "This appears in a theorem about asymptotic normality in a statistical estimation context. It's discussing the large-sample distribution of an estimator.",
            "description": "Asymptotic normality result"
        },
        {
            "latex": "\\beta_{LATE} = \\frac{E[Y_i | Z_i = 1] - E[Y_i | Z_i = 0]}{E[D_i | Z_i = 1] - E[D_i | Z_i = 0]}",
            "context": "This defines the Local Average Treatment Effect (LATE) parameter using instrumental variables. Y_i is the outcome, D_i is treatment status, Z_i is the instrument.",
            "description": "LATE parameter definition"
        }
    ]
    
    print("🧠 LIVE DEMO: GPT-4 Mathematical Explanations")
    print("=" * 80)
    print()
    
    for i, expr in enumerate(expressions, 1):
        print(f"EXAMPLE {i}: {expr['description']}")
        print("-" * 60)
        print()
        print("📝 LaTeX Expression:")
        print(f"   {expr['latex']}")
        print()
        
        print("🎧 BAD (Current approach):")
        # Show what current systems produce
        if i == 1:
            bad_version = "hat theta equals fraction one over n sum from i equals one to n psi open parenthesis W sub i semicolon hat eta superscript open parenthesis minus k open parenthesis i close parenthesis close parenthesis close parenthesis"
        elif i == 2:
            bad_version = "square root of n open parenthesis hat theta minus theta sub zero close parenthesis right arrow with d above N open parenthesis zero comma big sigma close parenthesis"
        else:
            bad_version = "beta sub LATE equals fraction E open bracket Y sub i pipe Z sub i equals one close bracket minus E open bracket Y sub i pipe Z sub i equals zero close bracket over E open bracket D sub i pipe Z sub i equals one close bracket minus E open bracket D sub i pipe Z sub i equals zero close bracket"
        
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
                max_tokens=500
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
    """Run the live demonstration."""
    
    print("This will make actual API calls to OpenAI GPT-4 to demonstrate")
    print("how mathematical expressions get transformed into natural language.")
    print()
    
    try:
        live_demo_with_gpt4()
        
        print("🎯 KEY IMPROVEMENTS DEMONSTRATED:")
        print("- Precise variable handling ('capital Sigma' vs 'lowercase theta')")
        print("- Multiple clear sentences for complex expressions")  
        print("- Contextual explanations (what concepts mean)")
        print("- Natural flow for audio comprehension")
        print("- Mathematical accuracy maintained")
        print()
        print("🎧 RESULT: Mathematical papers become truly listenable!")
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        print()
        print("Note: This requires a valid OpenAI API key and network connection.")

if __name__ == "__main__":
    main()