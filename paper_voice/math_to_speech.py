"""
Translate mathematical expressions into spoken phrases.

This module provides helper functions to convert LaTeX math into a spoken
description. It is designed to strike a balance between accuracy and
listenability. Complex formulas are rendered as sequences of words that
preserve the intended semantics without overwhelming the listener with
unnecessary detail. The logic here is heuristic and can be extended or
customised as needed.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from . import pdf_utils


# Mapping from LaTeX commands (without backslash) to spoken names.
GREEK_MAPPING: Dict[str, str] = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "epsilon": "epsilon",
    "varepsilon": "epsilon",
    "zeta": "zeta",
    "eta": "eta",
    "theta": "theta",
    "vartheta": "theta",
    "iota": "iota",
    "kappa": "kappa",
    "lambda": "lambda",
    "mu": "mu",
    "nu": "nu",
    "xi": "ksi",
    "pi": "pi",
    "varpi": "pi",
    "rho": "rho",
    "sigma": "sigma",
    "varsigma": "sigma",
    "tau": "tau",
    "upsilon": "upsilon",
    "phi": "phi",
    "varphi": "phi",
    "chi": "chi",
    "psi": "psi",
    "omega": "omega",
    # Uppercase Greek letters
    "Gamma": "big gamma",
    "Delta": "big delta",
    "Theta": "big theta",
    "Lambda": "big lambda",
    "Xi": "big ksi",
    "Pi": "big pi",
    "Sigma": "big sigma",
    "Upsilon": "big upsilon",
    "Phi": "big phi",
    "Psi": "big psi",
    "Omega": "big omega",
}

# Simple operator mapping
OPERATOR_MAPPING: Dict[str, str] = {
    "\\leq": "less than or equal to",
    "\\geq": "greater than or equal to",
    "\\neq": "not equal to",
    "\\approx": "approximately equal to",
    "\\times": "times",
    "\\cdot": "times",
    "\\pm": "plus or minus",
    "\\mp": "minus or plus",
    "\\infty": "infinity",
    "\\rightarrow": "tends to",
    "\\to": "tends to",
    "\\in": "in",
    "\\notin": "not in",
}


def _speak_variable(var: str) -> str:
    """Convert a single variable token into spoken form.

    Uppercase letters are read as "big <letter name>" to avoid confusion with
    lowercase letters. Greek letters are mapped via ``GREEK_MAPPING``.
    """
    # Greek letters
    if var in GREEK_MAPPING:
        return GREEK_MAPPING[var]
    # Single letters
    if len(var) == 1 and var.isalpha():
        if var.isupper():
            # Uppercase letters: prefix with 'big'
            return f"big {var.lower()}"
        else:
            return var
    return var


def _parse_superscript(expr: str) -> str:
    """Parse a superscript expression and speak it appropriately."""
    # Remove braces if present
    expr = expr.strip('{}')
    # Common powers
    if expr == '2':
        return "squared"
    if expr == '3':
        return "cubed"
    return f"to the power {expr}"


def _parse_subscript(expr: str) -> str:
    """Parse a subscript expression and speak it appropriately."""
    expr = expr.strip('{}')
    return f"sub {expr}"


def latex_to_speech(expr: str) -> str:
    """Convert a LaTeX math expression into a spoken phrase.

    Parameters
    ----------
    expr: str
        A string containing LaTeX syntax without surrounding dollar signs.

    Returns
    -------
    str
        A spoken approximation of the expression.

    Notes
    -----
    The implementation is intentionally heuristic: it handles a wide range
    of simple expressions and attempts to remain robust when encountering
    unknown tokens by leaving them as‑is. You can extend the mapping
    dictionaries or implement additional parsing rules to support more
    notation.
    """
    # Tokenize by breaking LaTeX commands and special characters
    # Keep underscores and carets for subscripts/superscripts
    tokens: List[str] = []
    i = 0
    while i < len(expr):
        if expr[i] == '\\':
            # Backslash indicates a command; consume letters
            j = i + 1
            while j < len(expr) and expr[j].isalpha():
                j += 1
            cmd = expr[i+1:j]
            tokens.append(f"\\{cmd}")
            i = j
        elif expr[i] in {'^', '_', '{', '}', '(', ')', '[', ']', '+', '-', '=', ',', '|'}:
            tokens.append(expr[i])
            i += 1
        else:
            tokens.append(expr[i])
            i += 1

    output: List[str] = []
    skip_next = False
    for idx, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        # Commands: Greek letters, operators
        if tok.startswith('\\'):
            if tok in OPERATOR_MAPPING:
                output.append(OPERATOR_MAPPING[tok])
            else:
                # Remove backslash for Greek mapping lookup
                name = tok[1:]
                spoken = GREEK_MAPPING.get(name)
                if spoken:
                    output.append(spoken)
                else:
                    output.append(name)
        elif tok == '^' and idx + 1 < len(tokens):
            # Superscript
            sup = tokens[idx + 1]
            spoken = _parse_superscript(sup)
            output.append(spoken)
            skip_next = True
        elif tok == '_' and idx + 1 < len(tokens):
            # Subscript
            sub = tokens[idx + 1]
            spoken = _parse_subscript(sub)
            output.append(spoken)
            skip_next = True
        elif tok == '{' or tok == '}':
            # Ignore braces
            continue
        elif tok == '(':
            output.append("open parenthesis")
        elif tok == ')':
            output.append("close parenthesis")
        elif tok == '[':
            output.append("open bracket")
        elif tok == ']':
            output.append("close bracket")
        elif tok == '+':
            output.append("plus")
        elif tok == '-':
            output.append("minus")
        elif tok == '=':
            output.append("equals")
        elif tok == ',':
            output.append("comma")
        elif tok == '|':
            output.append("absolute value")
        else:
            # Variable or number
            output.append(_speak_variable(tok))
    return ' '.join(word for word in output if word)


def process_text_with_math(text: str) -> str:
    """Replace inline math in a block of text with spoken equivalents.

    This function searches for LaTeX math expressions delimited by $...$ and
    replaces each with the spoken version produced by ``latex_to_speech``.

    Parameters
    ----------
    text: str
        The input text containing zero or more inline math expressions.

    Returns
    -------
    str
        The text with each inline math segment replaced by its spoken
        equivalent.
    """
    parts: List[str] = []
    last_pos = 0
    for start, end, math_expr in pdf_utils.detect_inline_math(text):
        # Append the text before the math
        parts.append(text[last_pos:start])
        # Convert the math to speech
        spoken = latex_to_speech(math_expr)
        parts.append(spoken)
        last_pos = end
    # Append the remainder
    parts.append(text[last_pos:])
    # Collapse multiple spaces created by replacements
    return re.sub(r"\s+", " ", ''.join(parts)).strip()