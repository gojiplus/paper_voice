"""
LaTeX and Markdown processor for converting mathematical documents to speech.

This module handles LaTeX/Markdown text with mathematical expressions, tables,
figures, and various LaTeX environments (equation, align, etc.). It produces
clean text suitable for text-to-speech synthesis.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class ProcessedContent:
    """Container for processed document content."""
    text: str
    figures: List[Tuple[str, str]]  # (caption, description)
    tables: List[Tuple[str, str]]   # (caption, content)
    equations: List[str]            # standalone equations
    metadata: Dict[str, str]        # title, author, etc.


# Comprehensive Greek letter mapping
GREEK_MAPPING: Dict[str, str] = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "varepsilon": "epsilon", "zeta": "zeta", "eta": "eta",
    "theta": "theta", "vartheta": "theta", "iota": "iota", "kappa": "kappa",
    "lambda": "lambda", "mu": "mu", "nu": "nu", "xi": "ksi", "pi": "pi",
    "varpi": "pi", "rho": "rho", "varrho": "rho", "sigma": "sigma",
    "varsigma": "sigma", "tau": "tau", "upsilon": "upsilon", "phi": "phi",
    "varphi": "phi", "chi": "chi", "psi": "psi", "omega": "omega",
    # Uppercase
    "Gamma": "big gamma", "Delta": "big delta", "Theta": "big theta",
    "Lambda": "big lambda", "Xi": "big ksi", "Pi": "big pi",
    "Sigma": "big sigma", "Upsilon": "big upsilon", "Phi": "big phi",
    "Psi": "big psi", "Omega": "big omega",
}

# Mathematical operators and symbols
MATH_OPERATORS: Dict[str, str] = {
    # Relations
    "\\leq": "less than or equal to", "\\geq": "greater than or equal to",
    "\\neq": "not equal to", "\\approx": "approximately equal to",
    "\\equiv": "equivalent to", "\\sim": "similar to", "\\simeq": "similar or equal to",
    "\\cong": "congruent to", "\\propto": "proportional to",
    "\\ll": "much less than", "\\gg": "much greater than",
    # Arithmetic
    "\\times": "times", "\\cdot": "times", "\\div": "divided by",
    "\\pm": "plus or minus", "\\mp": "minus or plus",
    "\\oplus": "plus", "\\ominus": "minus", "\\otimes": "tensor product",
    # Set theory
    "\\in": "in", "\\notin": "not in", "\\subset": "subset of", "\\supset": "superset of",
    "\\subseteq": "subset or equal to", "\\supseteq": "superset or equal to",
    "\\cup": "union", "\\cap": "intersection", "\\setminus": "set minus",
    "\\emptyset": "empty set", "\\varnothing": "empty set",
    # Logic
    "\\land": "and", "\\lor": "or", "\\lnot": "not", "\\neg": "not",
    "\\implies": "implies", "\\iff": "if and only if",
    "\\exists": "there exists", "\\forall": "for all",
    # Arrows
    "\\rightarrow": "tends to", "\\to": "tends to", "\\leftarrow": "comes from",
    "\\leftrightarrow": "corresponds to", "\\Rightarrow": "implies",
    "\\Leftarrow": "implied by", "\\Leftrightarrow": "if and only if",
    "\\mapsto": "maps to",
    # Calculus
    "\\partial": "partial", "\\nabla": "nabla", "\\int": "integral",
    "\\iint": "double integral", "\\iiint": "triple integral", "\\oint": "contour integral",
    "\\sum": "sum", "\\prod": "product", "\\lim": "limit",
    "\\sup": "supremum", "\\inf": "infimum", "\\max": "maximum", "\\min": "minimum",
    # Functions
    "\\sin": "sine", "\\cos": "cosine", "\\tan": "tangent", "\\sec": "secant",
    "\\csc": "cosecant", "\\cot": "cotangent", "\\arcsin": "arc sine",
    "\\arccos": "arc cosine", "\\arctan": "arc tangent", "\\sinh": "hyperbolic sine",
    "\\cosh": "hyperbolic cosine", "\\tanh": "hyperbolic tangent",
    "\\log": "logarithm", "\\ln": "natural logarithm", "\\exp": "exponential",
    "\\sqrt": "square root",
    # Special
    "\\infty": "infinity", "\\hbar": "h bar", "\\ell": "ell",
    "\\Re": "real part", "\\Im": "imaginary part",
    "\\perp": "perpendicular", "\\parallel": "parallel",
}


def _speak_variable(var: str) -> str:
    """Convert a variable to spoken form."""
    if var in GREEK_MAPPING:
        return GREEK_MAPPING[var]
    if len(var) == 1 and var.isalpha():
        return f"big {var.lower()}" if var.isupper() else var
    return var


def _parse_superscript(expr: str) -> str:
    """Convert superscript to speech."""
    expr = expr.strip('{}')
    special_powers = {
        '2': 'squared', '3': 'cubed', '4': 'to the fourth power',
        '5': 'to the fifth power', '-1': 'inverse', 'T': 'transpose',
        '*': 'conjugate transpose', 'H': 'hermitian transpose'
    }
    return special_powers.get(expr, f"to the power {expr}")


def _parse_subscript(expr: str) -> str:
    """Convert subscript to speech."""
    return f"sub {expr.strip('{}')}"


def _handle_fraction(match) -> str:
    """Handle \\frac{numerator}{denominator}."""
    num, den = match.groups()
    num_speech = latex_math_to_speech(num)
    den_speech = latex_math_to_speech(den)
    return f"fraction {num_speech} over {den_speech}"


def _handle_sqrt(match) -> str:
    """Handle \\sqrt[n]{content} or \\sqrt{content}."""
    if match.lastindex == 2:  # nth root
        root, content = match.groups()
        root_speech = latex_math_to_speech(root)
        content_speech = latex_math_to_speech(content)
        return f"{root_speech} root of {content_speech}"
    else:  # square root
        content = match.group(1)
        content_speech = latex_math_to_speech(content)
        return f"square root of {content_speech}"


def _handle_sum_prod_int(match) -> str:
    """Handle \\sum, \\prod, \\int with limits."""
    command = match.group(1)
    lower = match.group(2) if match.group(2) else None
    upper = match.group(3) if match.group(3) else None
    
    if command == "sum":
        result = "sum"
    elif command == "prod":
        result = "product"
    elif command == "int":
        result = "integral"
    else:
        result = command
    
    if lower:
        result += f" from {latex_math_to_speech(lower)}"
    if upper:
        result += f" to {latex_math_to_speech(upper)}"
    
    return result


def latex_math_to_speech(expr: str) -> str:
    """Convert LaTeX mathematical expression to speech.
    
    This handles inline math content without delimiters.
    """
    if not expr.strip():
        return ""
    
    # Handle fractions
    expr = re.sub(r'\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                  _handle_fraction, expr)
    
    # Handle roots
    expr = re.sub(r'\\sqrt\[([^\]]*)\]\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', _handle_sqrt, expr)
    expr = re.sub(r'\\sqrt\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', _handle_sqrt, expr)
    
    # Handle sums, products, integrals with limits
    expr = re.sub(r'\\(sum|prod|int)_\{([^}]*)\}\^\{([^}]*)\}', _handle_sum_prod_int, expr)
    expr = re.sub(r'\\(sum|prod|int)_\{([^}]*)\}', _handle_sum_prod_int, expr)
    expr = re.sub(r'\\(sum|prod|int)\^\{([^}]*)\}', _handle_sum_prod_int, expr)
    
    # Tokenize and process
    tokens = _tokenize_latex(expr)
    output = []
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        
        if token.startswith('\\') and token in MATH_OPERATORS:
            output.append(MATH_OPERATORS[token])
        elif token.startswith('\\'):
            # Greek letters and other commands
            name = token[1:]
            output.append(GREEK_MAPPING.get(name, name))
        elif token == '^' and i + 1 < len(tokens):
            output.append(_parse_superscript(tokens[i + 1]))
            i += 1
        elif token == '_' and i + 1 < len(tokens):
            output.append(_parse_subscript(tokens[i + 1]))
            i += 1
        elif token in ['{', '}']:
            pass  # Ignore braces
        elif token in ['(', ')']:
            output.append("open parenthesis" if token == '(' else "close parenthesis")
        elif token in ['[', ']']:
            output.append("open bracket" if token == '[' else "close bracket")
        elif token in ['+', '-', '=', '<', '>', '/', '*', ',', '|', '!']:
            symbol_map = {
                '+': 'plus', '-': 'minus', '=': 'equals',
                '<': 'less than', '>': 'greater than',
                '/': 'divided by', '*': 'times', ',': 'comma',
                '|': 'absolute value', '!': 'factorial'
            }
            output.append(symbol_map.get(token, token))
        else:
            output.append(_speak_variable(token))
        
        i += 1
    
    return ' '.join(filter(None, output))


def _tokenize_latex(expr: str) -> List[str]:
    """Tokenize LaTeX expression."""
    tokens = []
    i = 0
    
    while i < len(expr):
        if expr[i] == '\\':
            # LaTeX command
            j = i + 1
            while j < len(expr) and expr[j].isalpha():
                j += 1
            tokens.append(expr[i:j])
            i = j
        elif expr[i] in {'^', '_', '{', '}', '(', ')', '[', ']', '+', '-', '=',
                         '<', '>', '/', '*', ',', '|', '!', '?', ':', ';', '.'}:
            tokens.append(expr[i])
            i += 1
        elif expr[i].isspace():
            i += 1
        else:
            # Collect alphanumeric sequences
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            if j > i:
                tokens.append(expr[i:j])
                i = j
            else:
                tokens.append(expr[i])
                i += 1
    
    return tokens


def extract_latex_environments(text: str) -> Dict[str, List[Tuple[int, int, str]]]:
    """Extract LaTeX environments like equation, align, etc."""
    environments = {}
    
    # Common math environments
    math_envs = ['equation', 'align', 'gather', 'multline', 'eqnarray', 
                 'array', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix']
    
    for env in math_envs:
        pattern = rf'\\begin\{{{env}\*?\}}(.*?)\\end\{{{env}\*?\}}'
        matches = []
        for match in re.finditer(pattern, text, re.DOTALL):
            start, end = match.span()
            content = match.group(1).strip()
            matches.append((start, end, content))
        if matches:
            environments[env] = matches
    
    return environments


def process_latex_environments(text: str, environments: Dict[str, List[Tuple[int, int, str]]]) -> str:
    """Process LaTeX environments and replace with spoken equivalents."""
    # Sort all environments by position (reverse order for replacement)
    all_envs = []
    for env_name, env_list in environments.items():
        for start, end, content in env_list:
            all_envs.append((start, end, env_name, content))
    
    all_envs.sort(key=lambda x: x[0], reverse=True)
    
    # Replace environments with spoken equivalents
    result = text
    for start, end, env_name, content in all_envs:
        spoken_content = latex_math_to_speech(content)
        
        if env_name in ['equation']:
            replacement = f"Equation: {spoken_content}"
        elif env_name in ['align', 'gather', 'multline']:
            replacement = f"Aligned equation: {spoken_content}"
        elif env_name in ['matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix']:
            replacement = f"Matrix: {spoken_content}"
        else:
            replacement = f"Mathematical expression: {spoken_content}"
        
        result = result[:start] + replacement + result[end:]
    
    return result


def process_inline_and_display_math(text: str) -> str:
    """Process inline $...$ and display $$...$$ math."""
    # Handle display math first ($$...$$)
    def replace_display_math(match):
        content = match.group(1)
        spoken = latex_math_to_speech(content)
        return f"Display equation: {spoken}"
    
    text = re.sub(r'\$\$(.*?)\$\$', replace_display_math, text, flags=re.DOTALL)
    
    # Handle inline math ($...$)
    def replace_inline_math(match):
        content = match.group(1)
        spoken = latex_math_to_speech(content)
        return spoken
    
    text = re.sub(r'(?<!\$)\$([^$]+?)\$(?!\$)', replace_inline_math, text)
    
    return text


def extract_figures_and_tables(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Extract figure and table information from LaTeX."""
    figures = []
    tables = []
    
    # Extract figure environments
    fig_pattern = r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}'
    for match in re.finditer(fig_pattern, text, re.DOTALL):
        content = match.group(1)
        # Look for caption
        caption_match = re.search(r'\\caption\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', content)
        if caption_match:
            caption = caption_match.group(1)
            figures.append((caption, "Figure content not processed"))
    
    # Extract table environments
    table_pattern = r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}'
    for match in re.finditer(table_pattern, text, re.DOTALL):
        content = match.group(1)
        # Look for caption
        caption_match = re.search(r'\\caption\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', content)
        if caption_match:
            caption = caption_match.group(1)
            # Extract tabular content if present
            tabular_match = re.search(r'\\begin\{tabular\}.*?\{(.*?)\}(.*?)\\end\{tabular\}', 
                                    content, re.DOTALL)
            if tabular_match:
                table_content = tabular_match.group(2)
                tables.append((caption, table_content))
            else:
                tables.append((caption, "Table content not extracted"))
    
    return figures, tables


def clean_latex_commands(text: str) -> str:
    """Remove or replace common LaTeX formatting commands."""
    # Remove common formatting commands
    text = re.sub(r'\\(textbf|textit|emph|underline|texttt)\{([^{}]*)\}', r'\2', text)
    text = re.sub(r'\\(bf|it|em|tt|rm|sf)\b', '', text)
    
    # Handle sections and subsections
    text = re.sub(r'\\section\*?\{([^}]*)\}', r'Section: \1', text)
    text = re.sub(r'\\subsection\*?\{([^}]*)\}', r'Subsection: \1', text)
    text = re.sub(r'\\subsubsection\*?\{([^}]*)\}', r'Subsubsection: \1', text)
    
    # Remove other common commands
    text = re.sub(r'\\(maketitle|tableofcontents|newpage|clearpage)', '', text)
    text = re.sub(r'\\\\(?:\[[^\]]*\])?', ' ', text)  # Line breaks
    text = re.sub(r'\\[vh]space\{[^}]*\}', '', text)  # Spacing
    text = re.sub(r'\\(small|large|Large|LARGE|huge|Huge)', '', text)  # Size commands
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def process_latex_document(text: str, summarize_figures: bool = False, 
                          summarize_tables: bool = False, 
                          api_key: Optional[str] = None) -> ProcessedContent:
    """Process a complete LaTeX document."""
    
    # Extract metadata (title, author, etc.)
    metadata = {}
    title_match = re.search(r'\\title\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if title_match:
        metadata['title'] = clean_latex_commands(title_match.group(1))
    
    author_match = re.search(r'\\author\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if author_match:
        metadata['author'] = clean_latex_commands(author_match.group(1))
    
    # Extract figures and tables before processing math
    figures, tables = extract_figures_and_tables(text)
    
    # Process figures and tables if summarization is requested
    processed_figures = []
    processed_tables = []
    
    if summarize_figures and api_key:
        # Would integrate with figure_table_summarizer here
        for caption, content in figures:
            processed_figures.append((caption, "Figure summarization not implemented"))
    else:
        processed_figures = figures
    
    if summarize_tables and api_key:
        # Would integrate with table summarizer here
        for caption, content in tables:
            processed_tables.append((caption, content))  # Process table content
    else:
        processed_tables = tables
    
    # Remove figure and table environments from main text
    text = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', '', text, flags=re.DOTALL)
    
    # Extract and process LaTeX environments
    environments = extract_latex_environments(text)
    text = process_latex_environments(text, environments)
    
    # Process inline and display math
    text = process_inline_and_display_math(text)
    
    # Clean remaining LaTeX commands
    text = clean_latex_commands(text)
    
    # Extract standalone equations for reference
    equations = []
    for env_name, env_list in environments.items():
        if env_name in ['equation', 'align', 'gather']:
            for _, _, content in env_list:
                equations.append(latex_math_to_speech(content))
    
    return ProcessedContent(
        text=text,
        figures=processed_figures,
        tables=processed_tables,
        equations=equations,
        metadata=metadata
    )


def process_markdown_with_math(text: str) -> str:
    """Process Markdown text with LaTeX math expressions."""
    # Process LaTeX math in markdown
    text = process_inline_and_display_math(text)
    return text