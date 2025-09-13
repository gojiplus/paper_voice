"""
Command-line interface for paper_voice.

This module provides a CLI for converting academic papers to audio without
requiring the Streamlit web interface.
"""

import argparse
import os
import sys
import tempfile
import glob
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import pdf_utils, tts
from .arxiv_downloader import download_arxiv_paper
from .llm_math_explainer import explain_math_with_llm_sync
from .latex_processor import process_latex_document
from .pdf_content_enhancer import enhance_pdf_content_with_llm
from .vision_pdf_analyzer import analyze_pdf_with_vision, create_enhanced_text_from_analysis


def setup_parser() -> argparse.ArgumentParser:
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog='paper_voice',
        description='Convert academic papers to high-quality audio narration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s paper.pdf --api-key YOUR_KEY
  %(prog)s https://arxiv.org/abs/2301.12345 --api-key YOUR_KEY --output research.mp3
  %(prog)s paper.tex --latex --api-key YOUR_KEY --voice nova
  %(prog)s paper.pdf --vision --api-key YOUR_KEY --output enhanced.mp3
  %(prog)s --batch papers/ --api-key YOUR_KEY --output-dir ./audio_output
  %(prog)s --batch paper1.pdf paper2.pdf paper3.tex --api-key YOUR_KEY --output-dir ./batch_output
        """
    )
    
    # Input source (required)
    parser.add_argument(
        'input',
        nargs='*',
        help='Input file path(s), directory, arXiv URL(s)/ID(s), or text content. Use --batch for multiple inputs'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        default='paper_voice_output.mp3',
        help='Output audio file path (default: paper_voice_output.mp3). For batch mode, use --output-dir'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./output',
        help='Output directory for batch processing (default: ./output)'
    )
    
    # Input type specification
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--pdf',
        action='store_true',
        help='Treat input as PDF file'
    )
    input_group.add_argument(
        '--latex',
        action='store_true', 
        help='Treat input as LaTeX file'
    )
    input_group.add_argument(
        '--arxiv',
        action='store_true',
        help='Treat input as arXiv URL or ID'
    )
    input_group.add_argument(
        '--text',
        action='store_true',
        help='Treat input as plain text with math expressions'
    )
    
    # API configuration (required for LLM features)
    parser.add_argument(
        '--api-key',
        required=True,
        help='OpenAI API key for LLM-powered enhancements'
    )
    
    # Processing options
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch processing mode for multiple inputs'
    )
    
    parser.add_argument(
        '--vision',
        action='store_true',
        help='Use GPT-4V for PDF analysis (slower but higher quality)'
    )
    
    parser.add_argument(
        '--no-enhancement',
        action='store_true',
        help='Skip LLM enhancement (faster but lower quality)'
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='Maximum pages to process (for cost control)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=3,
        help='Maximum number of concurrent workers for batch processing (default: 3)'
    )
    
    # Audio options
    audio_group = parser.add_argument_group('audio options')
    audio_group.add_argument(
        '--voice',
        choices=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
        default='alloy',
        help='OpenAI TTS voice (default: alloy)'
    )
    
    audio_group.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Speech speed multiplier (0.25 to 4.0, default: 1.0)'
    )
    
    audio_group.add_argument(
        '--offline',
        action='store_true',
        help='Use offline TTS (pyttsx3) instead of OpenAI TTS'
    )
    
    # Utility options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.2.0'
    )
    
    return parser


def collect_input_files(inputs: List[str], args: argparse.Namespace) -> List[str]:
    """Collect all input files from the provided inputs (files, directories, URLs)."""
    all_inputs = []
    
    for input_item in inputs:
        if os.path.isdir(input_item):
            # Directory: find all supported files
            patterns = ['*.pdf', '*.tex', '*.latex', '*.txt', '*.md']
            for pattern in patterns:
                found_files = glob.glob(os.path.join(input_item, pattern))
                all_inputs.extend(found_files)
            # Also check subdirectories
            for pattern in patterns:
                found_files = glob.glob(os.path.join(input_item, '**', pattern), recursive=True)
                all_inputs.extend(found_files)
        elif os.path.isfile(input_item):
            # Single file
            all_inputs.append(input_item)
        elif input_item.startswith('http') or input_item.replace('.', '').isdigit():
            # URL or arXiv ID
            all_inputs.append(input_item)
        else:
            # Might be text content or invalid
            all_inputs.append(input_item)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_inputs = []
    for item in all_inputs:
        if item not in seen:
            seen.add(item)
            unique_inputs.append(item)
    
    return unique_inputs


def determine_input_type(input_path: str, args: argparse.Namespace) -> str:
    """Determine the input type based on arguments and file inspection."""
    # If explicitly specified, use that
    if args.pdf:
        return 'pdf'
    elif args.latex:
        return 'latex'
    elif args.arxiv:
        return 'arxiv'
    elif args.text:
        return 'text'
    
    # Auto-detect based on input
    if input_path.startswith('http') and 'arxiv.org' in input_path:
        return 'arxiv'
    elif input_path.isdigit() or '.' in input_path and len(input_path.split('.')) == 2:
        # Looks like arXiv ID (e.g., "2301.12345")
        return 'arxiv'
    elif os.path.isfile(input_path):
        ext = Path(input_path).suffix.lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext in ['.tex', '.latex']:
            return 'latex'
        else:
            return 'text'
    else:
        # Assume it's text content
        return 'text'


def load_content(input_path: str, input_type: str, args: argparse.Namespace) -> tuple[str, str]:
    """Load content based on input type."""
    if args.verbose:
        print(f"Loading {input_type} content from: {input_path}")
    
    if input_type == 'arxiv':
        paper = download_arxiv_paper(input_path)
        if not paper:
            raise ValueError(f"Failed to download arXiv paper: {input_path}")
        
        if args.verbose:
            print(f"Downloaded: {paper.title}")
            print(f"LaTeX content: {len(paper.latex_content)} characters")
            print(f"Figures: {len(paper.figures)} files")
        
        return paper.latex_content, 'LaTeX'
    
    elif input_type == 'pdf':
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"PDF file not found: {input_path}")
        
        if args.vision:
            if args.verbose:
                print("Using GPT-4V for PDF analysis...")
            
            analysis_result = analyze_pdf_with_vision(
                input_path, args.api_key, max_pages=args.max_pages
            )
            content = create_enhanced_text_from_analysis(analysis_result)
            
            if args.verbose:
                print(f"Vision analysis: {len(analysis_result.content_blocks)} content blocks")
        else:
            pages = pdf_utils.extract_raw_text(input_path)
            content = '\n\n'.join(pages)
        
        return content, 'PDF'
    
    elif input_type == 'latex':
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"LaTeX file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content, 'LaTeX'
    
    elif input_type == 'text':
        if os.path.isfile(input_path):
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            # Treat as direct text input
            content = input_path
        
        return content, 'Text'
    
    else:
        raise ValueError(f"Unknown input type: {input_type}")


def enhance_content(content: str, content_type: str, args: argparse.Namespace) -> str:
    """Enhance content using LLM if requested."""
    if args.no_enhancement:
        if args.verbose:
            print("Skipping LLM enhancement")
        return content
    
    if args.verbose:
        print("Enhancing content with LLM...")
    
    try:
        from .content_processor import process_content_unified
        
        # Use unified processor for all content types
        processed_doc = process_content_unified(
            content=content,
            input_type=content_type.lower(),
            api_key=args.api_key,
            use_llm_enhancement=True
        )
        enhanced = processed_doc.enhanced_text
        
        if args.verbose:
            print(f"Enhanced content: {len(enhanced)} characters")
        
        return enhanced
    
    except Exception as e:
        if args.verbose:
            print(f"Enhancement failed: {e}, using original content")
        return content


def process_math_expressions(content: str, args: argparse.Namespace) -> str:
    """Process mathematical expressions with LLM explanations."""
    if args.no_enhancement:
        return content
    
    if args.verbose:
        print("Processing mathematical expressions...")
    
    try:
        # This would need the full streamlit app processing logic
        # For now, return content as-is
        # TODO: Implement math processing without streamlit dependency
        return content
    
    except Exception as e:
        if args.verbose:
            print(f"Math processing failed: {e}")
        return content


def generate_output_path(input_path: str, args: argparse.Namespace, batch_mode: bool = False) -> str:
    """Generate appropriate output path for the given input."""
    if batch_mode:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate filename based on input
        if input_path.startswith('http'):
            # arXiv URL - extract ID
            if 'arxiv.org' in input_path:
                arxiv_id = input_path.split('/')[-1].replace('.pdf', '')
                filename = f"arxiv_{arxiv_id}.mp3"
            else:
                filename = f"url_{hash(input_path) % 10000}.mp3"
        elif os.path.isfile(input_path):
            # File - use base name
            base_name = Path(input_path).stem
            filename = f"{base_name}.mp3"
        else:
            # Text or unknown - use hash
            filename = f"text_{hash(input_path) % 10000}.mp3"
        
        return os.path.join(args.output_dir, filename)
    else:
        return args.output


def process_single_paper(input_path: str, args: argparse.Namespace, batch_mode: bool = False) -> tuple[str, bool, str]:
    """Process a single paper and return (output_path, success, error_message)."""
    try:
        # Generate output path
        output_path = generate_output_path(input_path, args, batch_mode)
        
        # Determine input type
        input_type = determine_input_type(input_path, args)
        
        if args.verbose:
            print(f"Processing {input_type}: {input_path} -> {output_path}")
        
        # Load content
        content, content_type = load_content(input_path, input_type, args)
        
        # Enhance content
        enhanced_content = enhance_content(content, content_type, args)
        
        # Process math expressions
        processed_content = process_math_expressions(enhanced_content, args)
        
        # Generate audio with custom output path
        original_output = args.output
        args.output = output_path
        final_output_path = generate_audio(processed_content, args)
        args.output = original_output  # Restore original
        
        return final_output_path, True, ""
    
    except Exception as e:
        return input_path, False, str(e)


def process_batch(inputs: List[str], args: argparse.Namespace) -> None:
    """Process multiple papers in parallel."""
    if args.verbose:
        print(f"Batch processing {len(inputs)} inputs with {args.max_workers} workers...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_input = {
            executor.submit(process_single_paper, input_path, args, batch_mode=True): input_path
            for input_path in inputs
        }
        
        # Process completed tasks
        for future in as_completed(future_to_input):
            input_path = future_to_input[future]
            output_path, success, error_msg = future.result()
            
            if success:
                print(f"✅ {Path(input_path).name} -> {Path(output_path).name}")
                results.append((input_path, output_path, True, ""))
            else:
                print(f"❌ {Path(input_path).name}: {error_msg}")
                results.append((input_path, "", False, error_msg))
    
    # Summary
    successful = sum(1 for _, _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"\n📊 Batch processing complete:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Output directory: {args.output_dir}")
    
    if failed > 0 and args.verbose:
        print(f"\nFailed files:")
        for input_path, _, success, error_msg in results:
            if not success:
                print(f"   • {Path(input_path).name}: {error_msg}")


def generate_audio(content: str, args: argparse.Namespace) -> str:
    """Generate audio from processed content."""
    if args.verbose:
        print(f"Generating audio: {len(content)} characters")
        print(f"Output: {args.output}")
        if not args.offline:
            print(f"Voice: {args.voice}, Speed: {args.speed}")
    
    try:
        # Use chunked synthesis for long content
        output_path = tts.synthesize_speech_chunked(
            content,
            args.output,
            use_openai=not args.offline,
            api_key=args.api_key if not args.offline else None,
            openai_voice=args.voice,
            # Speed parameter would need to be added to TTS functions
        )
        
        return output_path
    
    except Exception as e:
        raise RuntimeError(f"Audio generation failed: {e}")


def main() -> int:
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    try:
        if args.verbose:
            print(f"Paper Voice CLI v0.2.0")
        
        # Validate inputs
        if not args.input:
            print("❌ Error: No input provided")
            return 1
        
        # Handle batch mode
        if args.batch or len(args.input) > 1:
            if args.verbose:
                print(f"Batch mode enabled")
            
            # Collect all input files
            all_inputs = collect_input_files(args.input, args)
            
            if not all_inputs:
                print("❌ Error: No valid inputs found")
                return 1
            
            if args.verbose:
                print(f"Found {len(all_inputs)} inputs to process")
            
            # Process in batch
            process_batch(all_inputs, args)
            return 0
        
        else:
            # Single file mode
            input_path = args.input[0]
            
            # Determine input type
            input_type = determine_input_type(input_path, args)
            
            if args.verbose:
                print(f"Single file mode")
                print(f"Input type: {input_type}")
            
            # Load content
            content, content_type = load_content(input_path, input_type, args)
            
            # Enhance content
            enhanced_content = enhance_content(content, content_type, args)
            
            # Process math expressions
            processed_content = process_math_expressions(enhanced_content, args)
            
            # Generate audio
            output_path = generate_audio(processed_content, args)
            
            print(f"✅ Audio generated successfully: {output_path}")
            
            if args.verbose:
                file_size = os.path.getsize(output_path)
                print(f"File size: {file_size / 1024 / 1024:.1f} MB")
            
            return 0
    
    except KeyboardInterrupt:
        print("\\n❌ Interrupted by user")
        return 1
    
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cli_entry_point():
    """Entry point for the CLI when installed as a package."""
    sys.exit(main())


if __name__ == '__main__':
    cli_entry_point()