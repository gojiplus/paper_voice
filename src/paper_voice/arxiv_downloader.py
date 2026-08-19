"""arXiv source downloader for LaTeX and figures.

This module downloads arXiv papers in their original LaTeX source format
along with figures, which provides much better quality than PDF extraction.
"""

import io
import logging
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Container for arXiv paper content."""

    arxiv_id: str
    title: str
    latex_content: str
    figures: dict[str, bytes]  # filename -> binary content
    metadata: dict[str, str]


def extract_arxiv_id(url_or_id: str) -> str | None:
    """Extract arXiv ID from URL or validate ID format."""
    # Handle various arXiv URL formats
    patterns = [
        r"arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)",
        r"arxiv\.org/pdf/(\d+\.\d+(?:v\d+)?)(?:\.pdf)?",
        r"^(\d+\.\d+(?:v\d+)?)$",  # Direct ID
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    return None


def download_arxiv_source(arxiv_id: str, extract_dir: str | None = None) -> str | None:
    """Download arXiv source tarball and extract it.

    Args:
        arxiv_id: The arXiv paper ID (e.g., "2301.12345"), used to build the
            ``https://arxiv.org/e-print/`` download URL.
        extract_dir: Directory to extract the tarball into. If None, a
            temporary directory is created.

    Returns:
        Path to the directory holding the extracted source, or None if the
        download or extraction failed.
    """
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp(prefix=f"arxiv_{arxiv_id}_")

    # arXiv source URL format
    source_url = f"https://arxiv.org/e-print/{arxiv_id}"

    try:
        # Download the source
        response = requests.get(
            source_url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PaperVoice/1.0)"},
        )

        if response.status_code != 200:
            logger.error(
                "Failed to download arXiv source: HTTP %s", response.status_code
            )
            return None

        # Extract the tarball straight from memory; the data filter guards
        # against path-traversal entries in the archive.
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            tar.extractall(extract_dir, filter="data")

        return extract_dir

    except Exception:
        logger.exception("Error downloading arXiv source for %s", arxiv_id)
        return None


def find_main_tex_file(source_dir: str) -> str | None:
    """Find the main LaTeX file in the extracted source."""
    tex_files = list(Path(source_dir).glob("*.tex"))

    if not tex_files:
        return None

    if len(tex_files) == 1:
        return str(tex_files[0])

    # Look for common main file names
    main_candidates = [
        "main.tex",
        "paper.tex",
        "manuscript.tex",
        "article.tex",
        "document.tex",
    ]

    for candidate in main_candidates:
        candidate_path = Path(source_dir) / candidate
        if candidate_path.exists():
            return str(candidate_path)

    # Look for files with \documentclass
    for tex_file in tex_files:
        try:
            with tex_file.open(encoding="utf-8", errors="ignore") as f:
                content = f.read(1000)  # Check first 1000 chars
            if "\\documentclass" in content:
                return str(tex_file)
        except OSError:
            logger.debug("Could not read %s while looking for main file", tex_file)
            continue

    # Fallback to first .tex file
    return str(tex_files[0])


def extract_figures_from_source(source_dir: str) -> dict[str, bytes]:
    """Extract figure files from the source directory."""
    figures = {}

    # Common figure extensions
    figure_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".eps",
        ".ps",
        ".svg",
        ".tiff",
        ".gif",
    }

    source_path = Path(source_dir)

    # Find all figure files
    for ext in figure_extensions:
        for fig_file in source_path.glob(f"**/*{ext}"):
            if fig_file.is_file():
                try:
                    figures[fig_file.name] = fig_file.read_bytes()
                except OSError:
                    logger.warning("Could not read figure %s", fig_file)

    return figures


def process_latex_inputs(latex_content: str, source_dir: str) -> str:
    r"""Process \input and \include commands to merge LaTeX files."""

    def replace_input(match: re.Match[str]) -> str:
        filename = match.group(1)

        # Add .tex extension if not present
        if not filename.endswith(".tex"):
            filename += ".tex"

        input_path = Path(source_dir) / filename

        try:
            input_content = input_path.read_text(encoding="utf-8", errors="ignore")
            # Recursively process inputs in the included file
            return process_latex_inputs(input_content, source_dir)
        except OSError:
            logger.warning("Could not include %s", filename)
            return f"% Could not include {filename}"

    # Process \input{filename} and \include{filename}
    latex_content = re.sub(r"\\input\{([^}]+)\}", replace_input, latex_content)
    return re.sub(r"\\include\{([^}]+)\}", replace_input, latex_content)


# Formatting commands whose braced content is human-readable text that must
# survive cleanup (\textbf{Bold} -> Bold).
_FORMATTING_COMMAND_RE = re.compile(
    r"\\(?:textbf|textit|textsc|texttt|textrm|emph|underline|mbox)\{([^{}]*)\}"
)


def _clean_latex_markup(text: str) -> str:
    """Strip LaTeX markup from a metadata field, keeping readable content."""
    text = _FORMATTING_COMMAND_RE.sub(r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})*", "", text)
    text = re.sub(r"[{}]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_paper_metadata(latex_content: str) -> dict[str, str]:
    """Extract metadata from LaTeX source."""
    metadata = {}

    # Extract title
    title_match = re.search(r"\\title\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", latex_content)
    if title_match:
        metadata["title"] = _clean_latex_markup(title_match.group(1))

    # Extract author(s)
    author_match = re.search(
        r"\\author\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", latex_content
    )
    if author_match:
        metadata["author"] = _clean_latex_markup(author_match.group(1))

    # Extract abstract
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", latex_content, re.DOTALL
    )
    if abstract_match:
        metadata["abstract"] = _clean_latex_markup(abstract_match.group(1))

    return metadata


def download_arxiv_paper(arxiv_id_or_url: str) -> ArxivPaper | None:
    """Download and process an arXiv paper from source.

    Args:
        arxiv_id_or_url: An arXiv ID (e.g., "2301.12345") or any arXiv URL
            it can be parsed out of, such as an abs, pdf, or e-print link.

    Returns:
        An :class:`ArxivPaper` with the merged LaTeX source, figure files,
        and parsed metadata, or None if the identifier is unrecognised, the
        download failed, or no LaTeX file was found in the source.
    """
    # Extract arXiv ID
    arxiv_id = extract_arxiv_id(arxiv_id_or_url)
    if not arxiv_id:
        logger.error("Invalid arXiv ID or URL: %s", arxiv_id_or_url)
        return None

    logger.info("Downloading arXiv paper %s...", arxiv_id)

    # Download source
    source_dir = download_arxiv_source(arxiv_id)
    if not source_dir:
        return None

    try:
        # Find main LaTeX file
        main_tex = find_main_tex_file(source_dir)
        if not main_tex:
            logger.error("No LaTeX files found in source")
            return None

        logger.info("Found main LaTeX file: %s", Path(main_tex).name)

        # Read main LaTeX content
        latex_content = Path(main_tex).read_text(encoding="utf-8", errors="ignore")

        # Process \input and \include commands
        latex_content = process_latex_inputs(latex_content, source_dir)

        # Extract figures
        figures = extract_figures_from_source(source_dir)
        logger.info("Found %d figure files", len(figures))

        # Extract metadata
        metadata = extract_paper_metadata(latex_content)

        # Clean up temporary directory
        shutil.rmtree(source_dir, ignore_errors=True)

        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=metadata.get("title", f"arXiv:{arxiv_id}"),
            latex_content=latex_content,
            figures=figures,
            metadata=metadata,
        )

    except Exception:
        logger.exception("Error processing arXiv paper %s", arxiv_id)
        # Clean up on error
        shutil.rmtree(source_dir, ignore_errors=True)
        return None


def save_figures_to_directory(figures: dict[str, bytes], output_dir: str) -> list[str]:
    """Save figure files to a directory.

    Returns list of saved figure filenames.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for filename, content in figures.items():
        try:
            (out_path / filename).write_bytes(content)
            saved_files.append(filename)
        except OSError:
            logger.warning("Could not save figure %s", filename)

    return saved_files
