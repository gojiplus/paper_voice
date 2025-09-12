"""
LLM-powered mathematical expression explainer.

Uses language models to convert mathematical expressions, figures, and tables
into crystal-clear natural language explanations that listeners can easily understand.
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class MathExplanation:
    """Container for mathematical explanation."""
    original_latex: str
    natural_explanation: str
    context_type: str  # "equation", "theorem", "definition", "calculation"
    variables_explained: Dict[str, str]  # variable -> explanation


def get_math_explanation_prompt(latex_expr: str, context: str = "") -> str:
    """Create a detailed prompt for explaining mathematical expressions."""
    
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

Example 5:
LaTeX: $\\mathbb{{G}}_n(f) = \\frac{{1}}{{\\sqrt{{n}}}} \\sum_{{i=1}}^{{n}} \\omega_i f(X_i)$
Explanation: "The empirical process G subscript n of function f is defined as one over square root of n times the sum from i equals 1 to n of omega subscript i times f evaluated at capital X subscript i, where the omega terms are random weights."

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


def get_figure_explanation_prompt(caption: str, figure_content: str = "") -> str:
    """Create a prompt for explaining figures."""
    
    return f"""You are explaining academic figures for audio narration. Convert the figure caption and any available information into a clear, detailed description that helps listeners understand what the figure shows.

FIGURE CAPTION: {caption}

FIGURE CONTENT/DATA: {figure_content}

Requirements:
1. Start with what type of figure it is (plot, diagram, chart, etc.)
2. Explain what the axes represent (if applicable)
3. Describe the key patterns or findings
4. Use specific numbers when available
5. Make it clear and engaging for audio listeners

Example:
Input: "Figure 1: Treatment effects across different subgroups"
Output: "Figure 1 presents a comparison of treatment effects across different subgroups. This chart displays how the intervention's impact varies depending on participant characteristics, with separate bars or points showing the estimated effect size for each demographic group. The figure helps illustrate whether the treatment works equally well for all participants or shows differential effects."

Provide a clear, detailed audio-friendly description:

"""


def get_table_explanation_prompt(caption: str, table_content: str) -> str:
    """Create a prompt for explaining tables."""
    
    return f"""You are explaining academic tables for audio narration. Convert the table into a clear summary that highlights the key findings and patterns.

TABLE CAPTION: {caption}

TABLE CONTENT:
{table_content}

Requirements:
1. Start with what the table shows overall
2. Highlight the most important numbers and patterns
3. Explain what the rows and columns represent
4. Mention key statistics (means, standard deviations, p-values if present)
5. Make it digestible for audio listeners (don't read every number)

Example:
Input Caption: "Descriptive statistics by treatment group"
Input Table: Shows means and SDs for outcome variables
Output: "This table presents descriptive statistics comparing the treatment and control groups. The treatment group had an average outcome of 3.2 with a standard deviation of 1.1, while the control group averaged 2.8 with a standard deviation of 0.9. The table also shows that the two groups were well-balanced on baseline characteristics, with similar ages and demographic compositions."

Provide a clear, comprehensive summary for audio:

"""


def get_theorem_explanation_prompt(theorem_content: str, context: str = "") -> str:
    """Create a prompt for explaining theorems and propositions."""
    
    return f"""You are explaining a mathematical theorem or proposition for audio narration. Make it accessible while maintaining precision.

THEOREM/PROPOSITION CONTENT:
{theorem_content}

CONTEXT: {context}

Requirements:
1. Start by stating what the theorem establishes
2. Explain the conditions or assumptions clearly
3. Describe what the conclusion means in practical terms
4. Use precise mathematical language but make it conversational
5. Explain technical terms as needed

Example:
Input: "Under regularity conditions, the cross-fitted estimator satisfies asymptotic normality"
Output: "This theorem establishes that under certain technical regularity conditions, our cross-fitted estimator has the desirable property of asymptotic normality. This means that as the sample size becomes large, the distribution of our estimator approaches a normal bell curve, which is crucial for statistical inference and allows us to construct confidence intervals and perform hypothesis tests."

Explain this theorem clearly:

"""


async def explain_math_with_llm(latex_expr: str, api_key: str, context: str = "") -> MathExplanation:
    """Use LLM to explain mathematical expression."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    prompt = get_math_explanation_prompt(latex_expr, context)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert mathematical exposition writer who converts mathematical expressions into clear, natural language explanations for audio narration."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=800
        )
        
        explanation = response.choices[0].message.content.strip()
        
        # Extract any variable explanations mentioned
        variables_explained = _extract_variable_explanations(explanation)
        
        return MathExplanation(
            original_latex=latex_expr,
            natural_explanation=explanation,
            context_type="equation",
            variables_explained=variables_explained
        )
    
    except Exception as e:
        # Fallback explanation
        return MathExplanation(
            original_latex=latex_expr,
            natural_explanation=f"Mathematical expression: {latex_expr} (LLM explanation failed: {str(e)})",
            context_type="equation",
            variables_explained={}
        )


async def explain_figure_with_llm(caption: str, api_key: str, figure_content: str = "") -> str:
    """Use LLM to explain figure."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    prompt = get_figure_explanation_prompt(caption, figure_content)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at describing academic figures for audio narration, making visual content accessible through clear verbal descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Figure: {caption}. (LLM description failed: {str(e)})"


async def explain_table_with_llm(caption: str, table_content: str, api_key: str) -> str:
    """Use LLM to explain table."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    prompt = get_table_explanation_prompt(caption, table_content)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at summarizing academic tables for audio narration, highlighting key findings and patterns clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Table: {caption}. Content: {table_content}. (LLM summary failed: {str(e)})"


async def explain_theorem_with_llm(theorem_content: str, api_key: str, context: str = "") -> str:
    """Use LLM to explain theorem or proposition."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    prompt = get_theorem_explanation_prompt(theorem_content, context)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert mathematical educator who explains theorems and propositions in clear, accessible language while maintaining mathematical precision."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=600
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Theorem: {theorem_content}. (LLM explanation failed: {str(e)})"


def _extract_variable_explanations(explanation: str) -> Dict[str, str]:
    """Extract variable explanations from the natural language explanation."""
    variables = {}
    
    # Look for patterns like "X subscript i represents..." or "theta hat is..."
    patterns = [
        r'([A-Za-z]+(?:\s+subscript\s+[A-Za-z0-9]+)?)\s+(?:represents?|is|denotes?)\s+([^.]+)',
        r'([A-Za-z]+\s+hat)\s+(?:represents?|is|denotes?)\s+([^.]+)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, explanation, re.IGNORECASE)
        for match in matches:
            var_name = match.group(1).strip()
            var_explanation = match.group(2).strip()
            variables[var_name] = var_explanation
    
    return variables


def process_latex_with_llm_explanations(text: str, api_key: str) -> str:
    """Process LaTeX text by replacing math expressions with LLM explanations."""
    import asyncio
    
    async def process_async():
        # Find all math expressions
        display_math_pattern = r'\$\$(.*?)\$\$'
        inline_math_pattern = r'\$([^$]+?)\$'
        
        # Process display math
        display_matches = list(re.finditer(display_math_pattern, text, re.DOTALL))
        inline_matches = list(re.finditer(inline_math_pattern, text))
        
        # Get explanations for all math expressions
        explanations = []
        
        for match in display_math_pattern:
            math_expr = match.group(1).strip()
            explanation = await explain_math_with_llm(math_expr, api_key, text[:match.start()])
            explanations.append((match.start(), match.end(), explanation.natural_explanation))
        
        for match in inline_matches:
            math_expr = match.group(1).strip()
            explanation = await explain_math_with_llm(math_expr, api_key, text[:match.start()])
            explanations.append((match.start(), match.end(), explanation.natural_explanation))
        
        # Replace expressions (in reverse order to maintain positions)
        explanations.sort(key=lambda x: x[0], reverse=True)
        result_text = text
        
        for start, end, explanation in explanations:
            result_text = result_text[:start] + explanation + result_text[end:]
        
        return result_text
    
    # Run the async process
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(process_async())
    except RuntimeError:
        # If no event loop, create one
        return asyncio.run(process_async())


# Synchronous versions for easier integration
def explain_math_with_llm_sync(latex_expr: str, api_key: str, context: str = "") -> MathExplanation:
    """Synchronous version of math explanation."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    prompt = get_math_explanation_prompt(latex_expr, context)
    
    try:
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
        variables_explained = _extract_variable_explanations(explanation)
        
        return MathExplanation(
            original_latex=latex_expr,
            natural_explanation=explanation,
            context_type="equation",
            variables_explained=variables_explained
        )
    
    except Exception as e:
        return MathExplanation(
            original_latex=latex_expr,
            natural_explanation=f"Mathematical expression: {latex_expr} (explanation failed: {str(e)})",
            context_type="equation",
            variables_explained={}
        )


def explain_figure_with_llm_sync(caption: str, api_key: str, figure_content: str = "") -> str:
    """Synchronous version of figure explanation."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    prompt = get_figure_explanation_prompt(caption, figure_content)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at describing academic figures for audio narration."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Figure: {caption}. (Description failed: {str(e)})"


def explain_table_with_llm_sync(caption: str, table_content: str, api_key: str) -> str:
    """Synchronous version of table explanation."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    prompt = get_table_explanation_prompt(caption, table_content)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at summarizing academic tables for audio narration."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Table: {caption}. (Summary failed: {str(e)})"