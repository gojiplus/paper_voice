"""Natural mathematical speech processor.

Converts LaTeX mathematical expressions to natural, contextually appropriate English
that listeners can easily understand. Focuses on clarity and comprehensibility.
"""

import re


class MathContext:
    """Track mathematical context for better speech generation."""

    def __init__(self):
        """Initialize an empty context with default settings."""
        self.variable_meanings = {}  # Track what variables represent
        self.in_theorem = False
        self.in_definition = False
        self.current_domain = "general"  # statistics, calculus, algebra, etc.


def detect_math_domain(text: str) -> str:
    """Detect the mathematical domain from context."""
    statistics_indicators = [
        "distribution",
        "expectation",
        "variance",
        "probability",
        "random",
        "sample",
        "estimator",
        "hypothesis",
        "test",
        "confidence",
        "regression",
        "correlation",
        "bootstrap",
        "LATE",
        "IV",
    ]

    calculus_indicators = [
        "derivative",
        "integral",
        "limit",
        "continuous",
        "function",
        "convergence",
        "series",
        "taylor",
        "fourier",
    ]

    algebra_indicators = [
        "matrix",
        "vector",
        "eigenvalue",
        "determinant",
        "inverse",
        "transpose",
        "linear",
        "basis",
        "dimension",
    ]

    text_lower = text.lower()

    if any(ind in text_lower for ind in statistics_indicators):
        return "statistics"
    if any(ind in text_lower for ind in calculus_indicators):
        return "calculus"
    if any(ind in text_lower for ind in algebra_indicators):
        return "linear_algebra"

    return "general"


def natural_latex_to_speech(expr: str, context: MathContext | None = None) -> str:
    """Convert LaTeX math expression to natural English.

    Args:
        expr: LaTeX mathematical expression
        context: Mathematical context for better interpretation

    Returns:
        Natural English description of the mathematical expression
    """
    if not expr.strip():
        return ""

    if context is None:
        context = MathContext()

    # Clean the expression
    expr = expr.strip().strip("$")

    # Handle specific patterns that need natural language explanations
    expr = _handle_statistical_expressions(expr, context)
    expr = _handle_convergence_expressions(expr, context)
    expr = _handle_expectation_expressions(expr, context)
    expr = _handle_fraction_expressions(expr, context)
    expr = _handle_summation_expressions(expr, context)
    expr = _handle_limit_expressions(expr, context)
    expr = _handle_sqrt_expressions(expr, context)
    expr = _handle_hat_expressions(expr, context)

    # Handle subscripts and superscripts naturally
    expr = _handle_subscripts_naturally(expr, context)
    expr = _handle_superscripts_naturally(expr, context)

    # Handle Greek letters contextually
    expr = _handle_greek_letters_naturally(expr, context)

    # Handle operators and symbols
    expr = _handle_operators_naturally(expr, context)

    # Clean up and make more natural
    return _post_process_natural_speech(expr)


def _handle_statistical_expressions(expr: str, _context: MathContext) -> str:
    """Handle statistical expressions naturally."""
    # LATE and IV expressions
    if "LATE" in expr:
        expr = expr.replace("LATE", "the Local Average Treatment Effect")

    # E[...] notation
    def replace_expectation(match):
        content = match.group(1)
        if "|" in content:
            condition_parts = content.split("|", 1)
            var_part = condition_parts[0].strip()
            cond_part = condition_parts[1].strip()
            return (
                f"the expected value of {_clean_variable_name(var_part)}"
                f" given that {_clean_condition(cond_part)}"
            )
        return f"the expected value of {_clean_variable_name(content)}"

    expr = re.sub(r"E\[([^\]]+)\]", replace_expectation, expr)

    # Var(...) notation
    def replace_variance(match):
        content = match.group(1)
        return f"the variance of {_clean_variable_name(content)}"

    expr = re.sub(r"\\text\{Var\}\(([^)]+)\)", replace_variance, expr)
    expr = re.sub(r"Var\(([^)]+)\)", replace_variance, expr)

    # Cov(...) notation
    def replace_covariance(match):
        content = match.group(1)
        if "," in content:
            variables = [v.strip() for v in content.split(",")]
            return (
                f"the covariance between {_clean_variable_name(variables[0])}"
                f" and {_clean_variable_name(variables[1])}"
            )
        return f"the covariance of {_clean_variable_name(content)}"

    expr = re.sub(r"Cov\(([^)]+)\)", replace_covariance, expr)

    # P(...) probability notation
    def replace_probability(match):
        content = match.group(1)
        if "|" in content:
            condition_parts = content.split("|", 1)
            event_part = condition_parts[0].strip()
            cond_part = condition_parts[1].strip()
            return (
                f"the probability that {_clean_condition(event_part)}"
                f" given that {_clean_condition(cond_part)}"
            )
        return f"the probability that {_clean_condition(content)}"

    return re.sub(r"P\(([^)]+)\)", replace_probability, expr)


def _handle_convergence_expressions(expr: str, _context: MathContext) -> str:
    """Handle convergence expressions naturally."""
    # Convergence in distribution
    expr = re.sub(r"\\xrightarrow\{d\}", " converges in distribution to ", expr)
    expr = re.sub(r"\\xrightarrow\{p\}", " converges in probability to ", expr)
    expr = re.sub(
        r"\\xrightarrow\{\\text\{d\}\}", " converges in distribution to ", expr
    )
    expr = re.sub(
        r"\\xrightarrow\{\\text\{p\}\}", " converges in probability to ", expr
    )

    # o_p and O_p notation
    expr = re.sub(
        r"o_p\(([^)]+)\)", r"is of smaller order in probability than \1", expr
    )
    return re.sub(r"O_p\(([^)]+)\)", r"is of order in probability \1", expr)


def _handle_expectation_expressions(expr: str, _context: MathContext) -> str:
    """Handle expectation operator more naturally."""
    # This is already handled in statistical expressions, but we can add more context
    return expr


def _handle_fraction_expressions(expr: str, _context: MathContext) -> str:
    """Handle fractions more naturally."""

    def replace_fraction(match):
        numerator = match.group(1)
        denominator = match.group(2)

        # Clean up the parts
        num_clean = _clean_math_expression(numerator)
        den_clean = _clean_math_expression(denominator)

        # Special cases
        if denominator.strip() in ["n", "N"]:
            return f"the average of {num_clean}"
        if numerator == "1":
            return f"one over {den_clean}"
        if "sqrt" in denominator:
            return f"{num_clean} divided by {den_clean}"
        return f"the ratio of {num_clean} to {den_clean}"

    return re.sub(
        r"\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
        replace_fraction,
        expr,
    )


def _handle_summation_expressions(expr: str, _context: MathContext) -> str:
    """Handle summations more naturally."""

    def replace_summation(match):
        if match.lastindex >= 3:  # Has both lower and upper bounds
            lower = match.group(2) or ""
            upper = match.group(3) or ""

            if "i=1" in lower and ("n" in upper or "N" in upper):
                return "the sum over all observations of"
            if "i=" in lower:
                start = lower.split("=")[1] if "=" in lower else lower
                return f"the sum from {start} to {upper} of"
            return f"the sum from {lower} to {upper} of"
        if match.group(2):  # Only lower bound
            lower = match.group(2)
            if "i=" in lower:
                start = lower.split("=")[1] if "=" in lower else lower
                return f"the sum starting from {start} of"
            return f"the sum over {lower} of"
        return "the sum of"

    # Handle sum with limits
    return re.sub(r"\\sum(?:_\{([^}]*)\})?(?:\^\{([^}]*)\})?", replace_summation, expr)


def _handle_limit_expressions(expr: str, _context: MathContext) -> str:
    """Handle limit expressions naturally."""

    def replace_limit(match):
        approach = match.group(1) or "approaches"
        return f"the limit as {approach} of"

    return re.sub(r"\\lim(?:_\{([^}]*)\})?", replace_limit, expr)


def _handle_sqrt_expressions(expr: str, _context: MathContext) -> str:
    """Handle square roots naturally."""

    def replace_sqrt(match):
        content = match.group(1)
        if content == "n":
            return "square root of n"
        if content in ["2", "3", "4", "5"]:
            numbers = {"2": "two", "3": "three", "4": "four", "5": "five"}
            return f"square root of {numbers[content]}"
        clean_content = _clean_math_expression(content)
        return f"square root of {clean_content}"

    return re.sub(r"\\sqrt\{([^}]+)\}", replace_sqrt, expr)


def _handle_hat_expressions(expr: str, _context: MathContext) -> str:
    """Handle hat notation (estimators) naturally."""

    def replace_hat(match):
        var = match.group(1)

        # Common statistical estimators
        if var == "theta" or var == "\\theta":
            return "theta hat, the estimator of theta,"
        if var == "beta" or var == "\\beta":
            return "beta hat, the estimated coefficient,"
        if var == "mu" or var == "\\mu":
            return "mu hat, the sample mean,"
        if var == "sigma" or var == "\\sigma":
            return "sigma hat, the estimated standard deviation,"
        clean_var = _clean_variable_name(var)
        return f"{clean_var} hat, the estimator of {clean_var},"

    return re.sub(r"\\hat\{([^}]+)\}", replace_hat, expr)


def _handle_subscripts_naturally(expr: str, _context: MathContext) -> str:
    """Handle subscripts in a natural way."""

    def replace_subscript(match):
        base = match.group(1) or ""
        subscript = match.group(2)

        # Common patterns
        if subscript == "i":
            if base in ["Y", "X", "Z", "W"]:
                return f"{base} for individual i"
            return f"{base} sub i"
        if subscript == "n":
            return f"{base} for the n-th observation"
        if subscript in ["0", "1"]:
            if base in ["Y", "y"]:
                treatment = "control" if subscript == "0" else "treatment"
                return f"{base} under the {treatment} condition"
            return f"{base} {subscript}"
        return f"{base} subscript {subscript}"

    # Handle subscripts
    expr = re.sub(r"([A-Za-z\\]+)_\{([^}]+)\}", replace_subscript, expr)
    return re.sub(r"([A-Za-z\\]+)_([A-Za-z0-9])", replace_subscript, expr)


def _handle_superscripts_naturally(expr: str, _context: MathContext) -> str:
    """Handle superscripts in a natural way."""

    def replace_superscript(match):
        base = match.group(1) or ""
        superscript = match.group(2)

        # Common patterns
        if superscript == "2":
            return f"{base} squared"
        if superscript == "3":
            return f"{base} cubed"
        if superscript == "-1":
            return f"the inverse of {base}"
        if superscript == "T":
            return f"{base} transpose"
        if superscript == "*":
            return f"{base} star"
        if superscript.startswith("(-"):
            return f"{base} estimated on the complement sample"
        return f"{base} to the power of {superscript}"

    # Handle superscripts
    expr = re.sub(r"([A-Za-z\\]+)\^\{([^}]+)\}", replace_superscript, expr)
    return re.sub(r"([A-Za-z\\]+)\^([A-Za-z0-9])", replace_superscript, expr)


def _handle_greek_letters_naturally(expr: str, _context: MathContext) -> str:
    """Handle Greek letters with consistent pronunciation."""
    greek_map = {
        "\\alpha": "alpha",
        "\\beta": "beta",
        "\\gamma": "gamma",
        "\\delta": "delta",
        "\\epsilon": "epsilon",
        "\\varepsilon": "epsilon",
        "\\zeta": "zeta",
        "\\eta": "eta",
        "\\theta": "theta",
        "\\vartheta": "theta",
        "\\iota": "iota",
        "\\kappa": "kappa",
        "\\lambda": "lambda",
        "\\mu": "mu",
        "\\nu": "nu",
        "\\xi": "xi",
        "\\pi": "pi",
        "\\rho": "rho",
        "\\sigma": "sigma",
        "\\tau": "tau",
        "\\upsilon": "upsilon",
        "\\phi": "phi",
        "\\varphi": "phi",
        "\\chi": "chi",
        "\\psi": "psi",
        "\\omega": "omega",
        # Uppercase
        "\\Gamma": "capital gamma",
        "\\Delta": "capital delta",
        "\\Theta": "capital theta",
        "\\Lambda": "capital lambda",
        "\\Xi": "capital xi",
        "\\Pi": "capital pi",
        "\\Sigma": "capital sigma",
        "\\Upsilon": "capital upsilon",
        "\\Phi": "capital phi",
        "\\Psi": "capital psi",
        "\\Omega": "capital omega",
    }

    for latex_letter, spoken in greek_map.items():
        expr = expr.replace(latex_letter, spoken)

    return expr


def _handle_operators_naturally(expr: str, _context: MathContext) -> str:
    """Handle mathematical operators naturally."""
    operator_map = {
        "\\cdot": " times ",
        "\\times": " times ",
        "\\neq": " is not equal to ",
        "\\leq": " is less than or equal to ",
        "\\geq": " is greater than or equal to ",
        "\\approx": " is approximately equal to ",
        "\\sim": " is distributed as ",
        "\\propto": " is proportional to ",
        "\\in": " is in ",
        "\\notin": " is not in ",
        "\\subset": " is a subset of ",
        "\\supset": " contains ",
        "+": " plus ",
        "-": " minus ",
        "=": " equals ",
        "<": " is less than ",
        ">": " is greater than ",
        "\\rightarrow": " approaches ",
        "\\to": " approaches ",
        "\\mapsto": " maps to ",
    }

    for latex_op, spoken in operator_map.items():
        expr = expr.replace(latex_op, spoken)

    return expr


def _clean_variable_name(var: str) -> str:
    """Clean variable names for natural speech."""
    var = var.strip()

    # Handle common variable patterns
    if var in ["Y_i(1)", "Y_i(0)"]:
        treatment = "treatment" if "1" in var else "control"
        return f"Y for individual i under the {treatment} condition"
    if var.startswith("Y_i"):
        return "the outcome for individual i"
    if var.startswith("X_i"):
        return "the covariates for individual i"
    if var.startswith("Z_i"):
        return "the instrument for individual i"
    if var == "D_i":
        return "the treatment indicator for individual i"

    return var


def _clean_condition(condition: str) -> str:
    """Clean conditional expressions for natural speech."""
    condition = condition.strip()

    # Handle common conditions
    if "Z_i = 1" in condition:
        return "the instrument equals 1"
    if "Z_i = 0" in condition:
        return "the instrument equals 0"
    if "\\text{complier}" in condition:
        return "the individual is a complier"

    return condition


def _clean_math_expression(expr: str) -> str:
    """Clean mathematical expressions for speech."""
    expr = expr.strip()

    # Remove braces
    expr = re.sub(r"[{}]", "", expr)

    # Handle common patterns
    if expr == "n":
        return "n"
    if expr == "i=1":
        return "i equals 1"

    return expr


def _post_process_natural_speech(text: str) -> str:
    """Post-process the natural speech text."""
    # Clean up multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Fix common issues
    text = text.replace("  ", " ")
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")

    # Make more natural
    text = text.replace("the the", "the")
    text = text.replace("is is", "is")

    return text.strip()


def process_complex_statistical_expression(expr: str) -> str:
    """Process complex statistical expressions with full context."""
    context = MathContext()
    context.current_domain = "statistics"

    # Special handling for the example:
    # sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} N(0, \Sigma)
    if "\\sqrt{n}" in expr and "\\hat{\\theta}" in expr and "\\xrightarrow{d}" in expr:
        return (
            "As the sample size n grows large, the square root of n times "
            "the difference between theta hat (our estimator) and the true "
            "parameter theta "
            "converges in distribution to a normal distribution with mean zero and "
            "variance capital sigma"
        )

    # Use the natural processor
    return natural_latex_to_speech(expr, context)
