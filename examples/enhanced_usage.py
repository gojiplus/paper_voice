#!/usr/bin/env python3
"""
Enhanced Paper Voice Usage Examples

This script demonstrates the new modular architecture with comprehensive
mathematical expression support.
"""

import tempfile
from pathlib import Path
from paper_voice import (
    process_pdf_to_speech,
    process_latex_to_speech, 
    process_markdown_to_speech,
    DocumentProcessor,
    ProcessingOptions
)


def example_latex_processing():
    """Example: Process LaTeX document with comprehensive math support."""
    
    latex_content = r"""
    \documentclass{article}
    \title{Mathematical Expressions in Speech}
    \author{Paper Voice}
    
    \begin{document}
    \maketitle
    
    \section{Introduction}
    This document demonstrates the enhanced mathematical expression support
    in Paper Voice, including inline math like $E = mc^2$ and display equations.
    
    \section{Mathematical Examples}
    
    \subsection{Basic Operations}
    We can handle basic arithmetic: $a + b$, $x \times y$, and $\frac{p}{q}$.
    
    \subsection{Advanced Mathematics}
    Consider the integral:
    $$\int_0^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$
    
    And the summation:
    $$\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}$$
    
    \subsection{Linear Algebra}
    Matrix operations like $A^T$, $A^{-1}$, and eigenvalues $\lambda$ are supported.
    
    \begin{align}
    \mathbf{A}\mathbf{x} &= \mathbf{b} \\
    \det(\mathbf{A}) &\neq 0
    \end{align}
    
    \subsection{Calculus}
    Derivatives: $\frac{d}{dx}f(x) = f'(x)$
    
    Partial derivatives: $\frac{\partial}{\partial x} u(x,y)$
    
    \subsection{Greek Letters and Special Symbols}
    Common Greek letters: $\alpha, \beta, \gamma, \Delta, \Omega$
    
    Set operations: $A \cup B$, $A \cap B$, $x \in S$, $\emptyset$
    
    Logic: $\forall x \exists y: P(x,y)$
    
    \end{document}
    """
    
    print("=== LaTeX Processing Example ===")
    
    # Basic processing
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        result = process_latex_to_speech(
            latex_content, 
            output_audio_path=tmp.name,
            tts_backend="offline",  # Use offline TTS
            include_equations=True
        )
    
    print("Spoken text preview:")
    print(result.spoken_text[:500] + "..." if len(result.spoken_text) > 500 else result.spoken_text)
    print(f"\\nAudio saved to: {result.audio_file}")
    print(f"Processing log: {result.processing_log}")


def example_markdown_processing():
    """Example: Process Markdown with LaTeX math."""
    
    markdown_content = """
    # Quantum Mechanics Basics
    
    ## Wave Function
    
    The time-dependent Schrödinger equation is:
    
    $$i\\hbar \\frac{\\partial}{\\partial t} \\Psi(\\mathbf{r}, t) = \\hat{H} \\Psi(\\mathbf{r}, t)$$
    
    Where:
    - $\\Psi(\\mathbf{r}, t)$ is the wave function
    - $\\hat{H}$ is the Hamiltonian operator
    - $\\hbar$ is the reduced Planck constant
    
    ## Uncertainty Principle
    
    Heisenberg's uncertainty principle states that:
    
    $$\\Delta x \\Delta p \\geq \\frac{\\hbar}{2}$$
    
    This fundamental limit applies to position $x$ and momentum $p$ measurements.
    
    ## Applications
    
    For a particle in a box, the energy levels are:
    
    $$E_n = \\frac{n^2 \\pi^2 \\hbar^2}{2mL^2}$$
    
    where $n = 1, 2, 3, \\ldots$ and $L$ is the box length.
    """
    
    print("\\n=== Markdown Processing Example ===")
    
    result = process_markdown_to_speech(
        markdown_content,
        tts_backend="offline"
    )
    
    print("Spoken text preview:")
    print(result.spoken_text[:400] + "..." if len(result.spoken_text) > 400 else result.spoken_text)


def example_advanced_options():
    """Example: Using DocumentProcessor with advanced options."""
    
    print("\\n=== Advanced Processing Options ===")
    
    # Create advanced processing options
    options = ProcessingOptions(
        use_summarization=False,  # Would require OpenAI API key
        include_figures=True,
        include_tables=True, 
        include_equations=True,
        tts_backend="offline",
        tts_rate=180,  # Slightly slower speech
        tts_voice="",  # Use default voice
    )
    
    # Mathematical content with various constructs
    latex_content = r"""
    \begin{equation}
    \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}
    \end{equation}
    
    Maxwell's equations in differential form:
    \begin{align}
    \nabla \cdot \mathbf{E} &= \frac{\rho}{\epsilon_0} \\
    \nabla \cdot \mathbf{B} &= 0 \\  
    \nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
    \nabla \times \mathbf{B} &= \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}
    \end{align}
    
    The wave equation follows:
    $$\\frac{\partial^2 \mathbf{E}}{\partial t^2} = c^2 \nabla^2 \mathbf{E}$$
    
    where $c = \\frac{1}{\sqrt{\mu_0 \epsilon_0}}$ is the speed of light.
    """
    
    processor = DocumentProcessor(options)
    result = processor.process_latex(latex_content)
    
    print("Advanced processing result:")
    print("Equations found:", len(result.processed_content.equations))
    print("\\nSpoken equations:")
    for i, eq in enumerate(result.processed_content.equations, 1):
        print(f"{i}. {eq}")


def example_pdf_processing():
    """Example: PDF processing (would require actual PDF file)."""
    
    print("\\n=== PDF Processing Example ===")
    print("To process a PDF file, you would use:")
    print("""
    result = process_pdf_to_speech(
        pdf_path="path/to/your/paper.pdf",
        output_audio_path="output.mp3",
        use_summarization=True,  # Requires OpenAI API key
        openai_api_key="your-api-key",
        pdf_extraction_method="pymupdf",  # or "pypdf2"
        tts_backend="openai",  # or "offline" 
        tts_voice_openai="alloy"  # OpenAI voice selection
    )
    """)


def demo_math_expressions():
    """Demonstrate various mathematical expression types."""
    
    print("\\n=== Mathematical Expression Support Demo ===")
    
    from paper_voice.latex_processor import latex_math_to_speech
    
    expressions = [
        r"E = mc^2",
        r"\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
        r"\\int_0^{\\infty} e^{-x^2} dx",
        r"\\sum_{n=1}^{\\infty} \\frac{1}{n^2}",
        r"\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1",
        r"\\nabla \\times \\mathbf{F} = \\left(\\frac{\\partial F_z}{\\partial y} - \\frac{\\partial F_y}{\\partial z}\\right)\\mathbf{i}",
        r"\\alpha + \\beta = \\gamma",
        r"A \\subseteq B \\cap C",
        r"\\forall x \\in \\mathbb{R}: x^2 \\geq 0",
        r"\\sqrt[3]{8} = 2",
        r"\\binom{n}{k} = \\frac{n!}{k!(n-k)!}",
        r"\\det(\\mathbf{A}) = \\sum_{\\sigma \\in S_n} \\text{sgn}(\\sigma) \\prod_{i=1}^n a_{i,\\sigma(i)}"
    ]
    
    for expr in expressions:
        spoken = latex_math_to_speech(expr)
        print(f"LaTeX: {expr}")
        print(f"Speech: {spoken}")
        print()


if __name__ == "__main__":
    print("Paper Voice Enhanced Usage Examples")
    print("=" * 50)
    
    # Run examples
    demo_math_expressions()
    example_latex_processing()
    example_markdown_processing() 
    example_advanced_options()
    example_pdf_processing()
    
    print("\\n" + "=" * 50)
    print("Examples completed!")
    print("\\nFor more advanced usage, see the DocumentProcessor class")
    print("and ProcessingOptions for full customization.")