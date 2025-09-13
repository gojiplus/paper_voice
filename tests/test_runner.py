"""
Test runner for paper_voice package.

Run all tests with: python -m pytest tests/
Run specific test file: python -m pytest tests/test_arxiv_downloader.py
Run with coverage: python -m pytest tests/ --cov=paper_voice
"""

import pytest
import sys
import os

# Add the parent directory to the Python path so we can import paper_voice
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_all_tests():
    """Run all tests in the tests directory."""
    test_dir = os.path.dirname(__file__)
    return pytest.main([test_dir, '-v'])


def run_with_coverage():
    """Run tests with coverage reporting."""
    test_dir = os.path.dirname(__file__)
    return pytest.main([
        test_dir,
        '--cov=paper_voice',
        '--cov-report=html',
        '--cov-report=term-missing',
        '-v'
    ])


if __name__ == "__main__":
    # Run tests based on command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--coverage":
        exit_code = run_with_coverage()
    else:
        exit_code = run_all_tests()
    
    sys.exit(exit_code)