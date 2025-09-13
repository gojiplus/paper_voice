"""
arXiv source downloader for LaTeX and figures.

This module downloads arXiv papers in their original LaTeX source format
along with figures, which provides much better quality than PDF extraction.
"""

import os
import re
import tempfile
import tarfile
import requests
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ArxivPaper:
    """Container for arXiv paper content."""
    arxiv_id: str
    title: str
    latex_content: str
    figures: Dict[str, bytes]  # filename -> binary content
    metadata: Dict[str, str]


def extract_arxiv_id(url_or_id: str) -> Optional[str]:
    """Extract arXiv ID from URL or validate ID format."""
    # Handle various arXiv URL formats
    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)',
        r'arxiv\.org/pdf/(\d+\.\d+(?:v\d+)?)(?:\.pdf)?',
        r'^(\d+\.\d+(?:v\d+)?)$'  # Direct ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    
    return None


def download_arxiv_source(arxiv_id: str, extract_dir: Optional[str] = None) -> Optional[str]:
    """Download arXiv source tarball and extract it.
    
    Parameters
    ----------
    arxiv_id : str
        The arXiv paper ID (e.g., "2301.12345")
    extract_dir : str, optional
        Directory to extract files to. If None, creates a temp directory.
        
    Returns
    -------
    str or None
        Path to extracted directory, or None if download failed
    """
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp(prefix=f"arxiv_{arxiv_id}_")
    
    # arXiv source URL format
    source_url = f"https://arxiv.org/e-print/{arxiv_id}"
    
    try:
        # Download the source
        response = requests.get(source_url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; PaperVoice/1.0)'
        })
        
        if response.status_code != 200:
            print(f"Failed to download arXiv source: HTTP {response.status_code}")
            return None
        
        # Save and extract tarball
        tarball_path = os.path.join(extract_dir, f"{arxiv_id}.tar.gz")
        
        with open(tarball_path, 'wb') as f:
            f.write(response.content)
        
        # Extract tarball
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        
        # Remove the tarball
        os.remove(tarball_path)
        
        return extract_dir
        
    except Exception as e:
        print(f"Error downloading arXiv source: {e}")
        return None


def find_main_tex_file(source_dir: str) -> Optional[str]:
    """Find the main LaTeX file in the extracted source."""
    tex_files = list(Path(source_dir).glob("*.tex"))
    
    if not tex_files:
        return None
    
    if len(tex_files) == 1:
        return str(tex_files[0])
    
    # Look for common main file names
    main_candidates = [
        "main.tex", "paper.tex", "manuscript.tex", 
        "article.tex", "document.tex"
    ]
    
    for candidate in main_candidates:
        candidate_path = Path(source_dir) / candidate
        if candidate_path.exists():
            return str(candidate_path)
    
    # Look for files with \documentclass
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # Check first 1000 chars
                if '\\documentclass' in content:
                    return str(tex_file)
        except:
            continue
    
    # Fallback to first .tex file
    return str(tex_files[0])


def extract_figures_from_source(source_dir: str) -> Dict[str, bytes]:
    """Extract figure files from the source directory."""
    figures = {}
    
    # Common figure extensions
    figure_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.eps', '.ps', '.svg', '.tiff', '.gif'}
    
    source_path = Path(source_dir)
    
    # Find all figure files
    for ext in figure_extensions:
        for fig_file in source_path.glob(f"**/*{ext}"):
            if fig_file.is_file():
                try:
                    with open(fig_file, 'rb') as f:
                        figures[fig_file.name] = f.read()
                except Exception as e:
                    print(f"Could not read figure {fig_file}: {e}")
    
    return figures


def process_latex_inputs(latex_content: str, source_dir: str) -> str:
    """Process \input and \include commands to merge LaTeX files."""
    
    def replace_input(match):
        filename = match.group(1)
        
        # Add .tex extension if not present
        if not filename.endswith('.tex'):
            filename += '.tex'
        
        input_path = Path(source_dir) / filename
        
        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                input_content = f.read()
                # Recursively process inputs in the included file
                return process_latex_inputs(input_content, source_dir)
        except Exception as e:
            print(f"Could not include {filename}: {e}")
            return f"% Could not include {filename}"
    
    # Process \input{filename} and \include{filename}
    latex_content = re.sub(r'\\input\{([^}]+)\}', replace_input, latex_content)
    latex_content = re.sub(r'\\include\{([^}]+)\}', replace_input, latex_content)
    
    return latex_content


def extract_paper_metadata(latex_content: str) -> Dict[str, str]:
    """Extract metadata from LaTeX source."""
    metadata = {}
    
    # Extract title
    title_match = re.search(r'\\title\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', latex_content)
    if title_match:
        title = title_match.group(1)
        # Clean LaTeX commands from title
        title = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})*', '', title)
        title = re.sub(r'[{}]', '', title)
        metadata['title'] = title.strip()
    
    # Extract author(s)
    author_match = re.search(r'\\author\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', latex_content)
    if author_match:
        author = author_match.group(1)
        # Clean LaTeX commands
        author = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})*', '', author)
        author = re.sub(r'[{}]', '', author)
        metadata['author'] = author.strip()
    
    # Extract abstract
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', 
                              latex_content, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        # Clean LaTeX commands
        abstract = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})*', '', abstract)
        abstract = re.sub(r'[{}]', '', abstract)
        metadata['abstract'] = abstract.strip()
    
    return metadata


def download_arxiv_paper(arxiv_id_or_url: str) -> Optional[ArxivPaper]:
    """Download and process an arXiv paper from source.
    
    Parameters
    ----------
    arxiv_id_or_url : str
        arXiv ID (e.g., "2301.12345") or URL
        
    Returns
    -------
    ArxivPaper or None
        Processed paper data, or None if download failed
    """
    # Extract arXiv ID
    arxiv_id = extract_arxiv_id(arxiv_id_or_url)
    if not arxiv_id:
        print(f"Invalid arXiv ID or URL: {arxiv_id_or_url}")
        return None
    
    print(f"Downloading arXiv paper {arxiv_id}...")
    
    # Download source
    source_dir = download_arxiv_source(arxiv_id)
    if not source_dir:
        return None
    
    try:
        # Find main LaTeX file
        main_tex = find_main_tex_file(source_dir)
        if not main_tex:
            print("No LaTeX files found in source")
            return None
        
        print(f"Found main LaTeX file: {os.path.basename(main_tex)}")
        
        # Read main LaTeX content
        with open(main_tex, 'r', encoding='utf-8', errors='ignore') as f:
            latex_content = f.read()
        
        # Process \input and \include commands
        latex_content = process_latex_inputs(latex_content, source_dir)
        
        # Extract figures
        figures = extract_figures_from_source(source_dir)
        print(f"Found {len(figures)} figure files")
        
        # Extract metadata
        metadata = extract_paper_metadata(latex_content)
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(source_dir, ignore_errors=True)
        
        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=metadata.get('title', f"arXiv:{arxiv_id}"),
            latex_content=latex_content,
            figures=figures,
            metadata=metadata
        )
        
    except Exception as e:
        print(f"Error processing arXiv paper: {e}")
        # Clean up on error
        import shutil
        shutil.rmtree(source_dir, ignore_errors=True)
        return None


def save_figures_to_directory(figures: Dict[str, bytes], output_dir: str) -> List[str]:
    """Save figure files to a directory.
    
    Returns list of saved figure filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    for filename, content in figures.items():
        output_path = os.path.join(output_dir, filename)
        try:
            with open(output_path, 'wb') as f:
                f.write(content)
            saved_files.append(filename)
        except Exception as e:
            print(f"Could not save figure {filename}: {e}")
    
    return saved_files


# Example usage and testing
if __name__ == "__main__":
    # Test with a known arXiv paper
    test_id = "2301.12345"  # Replace with actual paper ID for testing
    
    paper = download_arxiv_paper(test_id)
    if paper:
        print(f"Successfully downloaded: {paper.title}")
        print(f"LaTeX content length: {len(paper.latex_content)} characters")
        print(f"Number of figures: {len(paper.figures)}")
        print(f"Metadata: {paper.metadata}")
    else:
        print("Failed to download paper")