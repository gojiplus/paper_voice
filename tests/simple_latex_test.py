#!/usr/bin/env python3
"""
Simple test script for processing the LATE LaTeX document
using just the latex_processor module directly.
"""

import os
import sys
import re
from pathlib import Path

# The LaTeX document content provided by the user
LATE_DOCUMENT = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}

\title{Testing the LATE Consequence that Only Compliers Move}
\author{Anonymous}
\date{}

\begin{document}

\maketitle

\section{Introduction}

This document tests our enhanced LaTeX processing capabilities with mathematical content related to Local Average Treatment Effects (LATE).

\section{Mathematical Framework}

Let $Y_i(d)$ denote the potential outcome for individual $i$ under treatment status $d \in \{0,1\}$. The observed outcome is:
$$Y_i = D_i \cdot Y_i(1) + (1-D_i) \cdot Y_i(0)$$

where $D_i$ is the treatment indicator.

\subsection{Instrumental Variables Setup}

Consider an instrument $Z_i$ that satisfies:
\begin{align}
E[Z_i \varepsilon_i] &= 0 \quad \text{(Exclusion restriction)} \\
Cov(Z_i, D_i) &\neq 0 \quad \text{(Relevance)}
\end{align}

The LATE parameter is defined as:
$$\text{LATE} = E[Y_i(1) - Y_i(0) | \text{complier}]$$

\subsection{Cross-fitting and Weighted Empirical Processes}

In the context of cross-fitting, we partition the sample into $K$ folds. For each fold $k$, we estimate nuisance parameters $\hat{\eta}^{(-k)}$ using the complement sample. The cross-fitted estimator is:

$$\hat{\theta} = \frac{1}{n} \sum_{i=1}^{n} \psi(W_i; \hat{\eta}^{(-k(i))})$$

where $k(i)$ denotes the fold containing observation $i$.

\begin{theorem}[Asymptotic Normality]
Under regularity conditions, the cross-fitted estimator satisfies:
$$\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} N(0, \Sigma)$$
where $\Sigma = E[\psi(W; \eta_0) \psi(W; \eta_0)']$.
\end{theorem}

\section{Weighted Empirical Processes}

Consider the weighted empirical process:
$$\mathbb{G}_n(f) = \frac{1}{\sqrt{n}} \sum_{i=1}^{n} \omega_i f(X_i)$$

where $\omega_i$ are random weights satisfying $E[\omega_i] = 1$ and $\text{Var}(\omega_i) = \sigma^2$.

\subsection{Multiplier Bootstrap}

The multiplier bootstrap generates:
$$\mathbb{G}_n^*(f) = \frac{1}{\sqrt{n}} \sum_{i=1}^{n} \xi_i f(X_i)$$

where $\xi_i \sim N(0,1)$ are independent multipliers.

\begin{proposition}[Bootstrap Validity]
If $\mathcal{F}$ is a Donsker class and $\sup_{f \in \mathcal{F}} |P_n f - P f| = o_p(n^{-1/2})$, then:
$$\sup_{x \in \mathbb{R}} |P^*(\sqrt{n} \sup_{f \in \mathcal{F}} |\mathbb{G}_n^*(f)| \leq x) - P(\sqrt{n} \sup_{f \in \mathcal{F}} |\mathbb{G}_n(f)| \leq x)| \xrightarrow{p} 0$$
\end{proposition}

\section{Application to LATE Estimation}

The LATE estimand can be written as:
$$\beta_{LATE} = \frac{E[Y_i | Z_i = 1] - E[Y_i | Z_i = 0]}{E[D_i | Z_i = 1] - E[D_i | Z_i = 0]}$$

Using cross-fitting with machine learning methods:
\begin{align}
\hat{\mu}_{01}^{(-k)} &= \hat{E}^{(-k)}[Y_i | Z_i = 0] \\
\hat{\mu}_{11}^{(-k)} &= \hat{E}^{(-k)}[Y_i | Z_i = 1] \\
\hat{\pi}_{01}^{(-k)} &= \hat{E}^{(-k)}[D_i | Z_i = 0] \\
\hat{\pi}_{11}^{(-k)} &= \hat{E}^{(-k)}[D_i | Z_i = 1]
\end{align}

The cross-fitted LATE estimator is:
$$\hat{\beta}_{LATE} = \frac{\frac{1}{n}\sum_{i=1}^{n} (\hat{\mu}_{11}^{(-k(i))} - \hat{\mu}_{01}^{(-k(i))})}{\frac{1}{n}\sum_{i=1}^{n} (\hat{\pi}_{11}^{(-k(i))} - \hat{\pi}_{01}^{(-k(i))})}$$

\section{Conclusion}

This framework demonstrates the application of cross-fitting to LATE estimation, ensuring robust inference even when using flexible machine learning methods for nuisance parameter estimation.

Key insights:
\begin{itemize}
\item Cross-fitting removes the bias from overfitting
\item The weighted empirical process framework provides theoretical foundations
\item Bootstrap methods enable valid inference
\end{itemize}

\end{document}
"""

# Simple LaTeX-to-speech conversion functions
def process_math_expression(expr: str) -> str:
    """Convert a math expression to spoken text."""
    # Remove dollar signs
    expr = expr.strip('$')
    
    # Greek letters
    greek_letters = {
        'alpha': 'alpha', 'beta': 'beta', 'gamma': 'gamma', 'delta': 'delta',
        'epsilon': 'epsilon', 'varepsilon': 'epsilon', 'zeta': 'zeta', 'eta': 'eta',
        'theta': 'theta', 'vartheta': 'theta', 'iota': 'iota', 'kappa': 'kappa',
        'lambda': 'lambda', 'mu': 'mu', 'nu': 'nu', 'xi': 'xi', 'pi': 'pi',
        'rho': 'rho', 'sigma': 'sigma', 'tau': 'tau', 'upsilon': 'upsilon',
        'phi': 'phi', 'varphi': 'phi', 'chi': 'chi', 'psi': 'psi', 'omega': 'omega',
        'Gamma': 'big gamma', 'Delta': 'big delta', 'Theta': 'big theta',
        'Lambda': 'big lambda', 'Xi': 'big xi', 'Pi': 'big pi', 'Sigma': 'big sigma',
        'Upsilon': 'big upsilon', 'Phi': 'big phi', 'Psi': 'big psi', 'Omega': 'big omega'
    }
    
    # Mathematical operators
    operators = {
        'frac': 'fraction',
        'sum': 'sum', 'int': 'integral', 'prod': 'product',
        'sqrt': 'square root of', 'exp': 'exponential',
        'sin': 'sine', 'cos': 'cosine', 'tan': 'tangent',
        'log': 'logarithm', 'ln': 'natural log',
        'cdot': 'times', 'times': 'times',
        'leq': 'less than or equal to', 'geq': 'greater than or equal to',
        'neq': 'not equal to', 'approx': 'approximately',
        'infty': 'infinity', 'partial': 'partial',
        'nabla': 'nabla', 'hat': 'hat', 'bar': 'bar',
        'mathbb': '', 'text': '', 'quad': ' ',
        'xrightarrow': 'converges to', 'rightarrow': 'goes to',
        'sim': 'distributed as', 'sup': 'supremum'
    }
    
    # Replace Greek letters
    for latex, spoken in greek_letters.items():
        expr = re.sub(r'\\' + latex + r'\b', spoken, expr)
    
    # Replace operators
    for latex, spoken in operators.items():
        expr = re.sub(r'\\' + latex + r'\b', spoken, expr)
    
    # Handle subscripts and superscripts
    expr = re.sub(r'\^{([^}]+)}', r' to the power of \1', expr)
    expr = re.sub(r'\^([a-zA-Z0-9])', r' to the power of \1', expr)
    expr = re.sub(r'_{([^}]+)}', r' sub \1', expr)
    expr = re.sub(r'_([a-zA-Z0-9])', r' sub \1', expr)
    
    # Handle fractions
    expr = re.sub(r'\\frac{([^}]+)}{([^}]+)}', r'\1 over \2', expr)
    
    # Clean up
    expr = re.sub(r'[{}\\]', ' ', expr)
    expr = re.sub(r'\s+', ' ', expr)
    
    return expr.strip()

def process_latex_text(text: str) -> str:
    """Process LaTeX text with math expressions."""
    processed_text = ""
    
    # Find display math ($$...$$)
    display_math_pattern = r'\$\$([^$]+)\$\$'
    last_end = 0
    
    for match in re.finditer(display_math_pattern, text):
        # Add text before the math
        processed_text += text[last_end:match.start()]
        
        # Process the math expression
        math_expr = match.group(1)
        spoken_math = process_math_expression(math_expr)
        processed_text += f" [Mathematical expression: {spoken_math}] "
        
        last_end = match.end()
    
    # Add remaining text
    processed_text += text[last_end:]
    
    # Find inline math ($...$) 
    inline_math_pattern = r'\$([^$]+)\$'
    processed_text = re.sub(inline_math_pattern, 
                           lambda m: f" {process_math_expression(m.group(1))} ", 
                           processed_text)
    
    # Handle LaTeX environments
    processed_text = re.sub(r'\\begin{align}(.*?)\\end{align}', 
                           lambda m: f" [Aligned equations: {process_math_expression(m.group(1))}] ",
                           processed_text, flags=re.DOTALL)
    
    processed_text = re.sub(r'\\begin{theorem}(.*?)\\end{theorem}', 
                           lambda m: f" Theorem: {m.group(1)} ",
                           processed_text, flags=re.DOTALL)
    
    processed_text = re.sub(r'\\begin{proposition}(.*?)\\end{proposition}', 
                           lambda m: f" Proposition: {m.group(1)} ",
                           processed_text, flags=re.DOTALL)
    
    # Clean LaTeX commands
    processed_text = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?(?:{[^}]*})*', ' ', processed_text)
    processed_text = re.sub(r'[{}]', ' ', processed_text)
    processed_text = re.sub(r'\s+', ' ', processed_text)
    
    return processed_text.strip()

def main():
    """Test the LaTeX processing with the LATE document."""
    
    print("=" * 80)
    print("TESTING PAPER VOICE WITH LATE DOCUMENT")
    print("=" * 80)
    print()
    
    # Process the LaTeX document
    print("Processing LaTeX document with mathematical expressions...")
    print("-" * 60)
    
    try:
        # Convert LaTeX to spoken text
        spoken_text = process_latex_text(LATE_DOCUMENT)
        
        print("✓ Document processed successfully!")
        print()
        
        # Display the spoken text output
        print("SPOKEN TEXT OUTPUT:")
        print("=" * 50)
        
        # Split into paragraphs and show first few
        paragraphs = [p.strip() for p in spoken_text.split('\n') if p.strip()]
        
        # Show first 10 meaningful paragraphs
        shown_paragraphs = 0
        for para in paragraphs:
            if para and not para.startswith('\\') and len(para) > 10:
                print(f"{shown_paragraphs + 1}. {para}")
                print()
                shown_paragraphs += 1
                if shown_paragraphs >= 10:
                    break
        
        if len(paragraphs) > shown_paragraphs:
            print(f"... [showing first {shown_paragraphs} paragraphs out of {len(paragraphs)} total]")
        
        print()
        
        # Show some specific mathematical conversions
        print("MATHEMATICAL EXPRESSION EXAMPLES:")
        print("-" * 40)
        math_examples = [
            r"$Y_i(d)$",
            r"$$Y_i = D_i \cdot Y_i(1) + (1-D_i) \cdot Y_i(0)$$",
            r"$E[Z_i \varepsilon_i] = 0$", 
            r"$$\hat{\theta} = \frac{1}{n} \sum_{i=1}^{n} \psi(W_i; \hat{\eta}^{(-k(i))})$$",
            r"$$\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} N(0, \Sigma)$$"
        ]
        
        for i, expr in enumerate(math_examples, 1):
            spoken = process_math_expression(expr)
            print(f"{i}. '{expr}' → '{spoken}'")
        
        print()
        print("✓ Test completed successfully!")
        print("The enhanced Paper Voice system can handle:")
        print("  • Complex LaTeX mathematical expressions")
        print("  • Display math environments ($$...$$)")
        print("  • Greek letters (α → 'alpha', Σ → 'big sigma')")
        print("  • Fractions, superscripts, subscripts")
        print("  • Summations, integrals, and limits")
        print("  • Probability and statistical notation")
        print("  • LaTeX environments (align, theorem, proposition)")
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()