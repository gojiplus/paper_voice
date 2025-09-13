"""Tests for batch processing functionality in paper_voice CLI."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from paper_voice.cli import (
    collect_input_files, 
    generate_output_path, 
    process_single_paper,
    process_batch
)
import argparse


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.args = argparse.Namespace(
            output='output.mp3',
            output_dir='./output',
            batch=True,
            pdf=False,
            latex=False, 
            arxiv=False,
            text=False,
            api_key='test-key',
            vision=False,
            no_enhancement=False,
            max_pages=None,
            max_workers=2,
            voice='alloy',
            speed=1.0,
            offline=False,
            verbose=False
        )
    
    def test_collect_input_files_single_file(self):
        """Test collecting input files with a single file."""
        test_file = os.path.join(self.temp_dir, 'test.pdf')
        with open(test_file, 'w') as f:
            f.write("dummy content")
        
        inputs = [test_file]
        result = collect_input_files(inputs, self.args)
        
        assert len(result) == 1
        assert result[0] == test_file
    
    def test_collect_input_files_directory(self):
        """Test collecting input files from a directory."""
        # Create test files
        test_files = ['paper1.pdf', 'paper2.tex', 'notes.txt', 'readme.md']
        for filename in test_files:
            with open(os.path.join(self.temp_dir, filename), 'w') as f:
                f.write("dummy content")
        
        inputs = [self.temp_dir]
        result = collect_input_files(inputs, self.args)
        
        # Should find PDF, TeX, TXT, and MD files
        assert len(result) >= 4
        found_files = [os.path.basename(f) for f in result]
        for expected_file in test_files:
            assert expected_file in found_files
    
    def test_collect_input_files_arxiv_url(self):
        """Test collecting arXiv URLs."""
        inputs = ['https://arxiv.org/abs/2301.12345']
        result = collect_input_files(inputs, self.args)
        
        assert len(result) == 1
        assert result[0] == 'https://arxiv.org/abs/2301.12345'
    
    def test_collect_input_files_mixed(self):
        """Test collecting mixed input types."""
        # Create test file
        test_file = os.path.join(self.temp_dir, 'paper.pdf')
        with open(test_file, 'w') as f:
            f.write("dummy content")
        
        inputs = [test_file, 'https://arxiv.org/abs/2301.12345', '2405.67890']
        result = collect_input_files(inputs, self.args)
        
        assert len(result) == 3
        assert test_file in result
        assert 'https://arxiv.org/abs/2301.12345' in result
        assert '2405.67890' in result
    
    def test_collect_input_files_duplicates(self):
        """Test that duplicate inputs are removed."""
        test_file = os.path.join(self.temp_dir, 'test.pdf')
        with open(test_file, 'w') as f:
            f.write("dummy content")
        
        inputs = [test_file, test_file, test_file]
        result = collect_input_files(inputs, self.args)
        
        assert len(result) == 1
        assert result[0] == test_file
    
    def test_generate_output_path_single_mode(self):
        """Test output path generation in single mode."""
        result = generate_output_path('test.pdf', self.args, batch_mode=False)
        assert result == 'output.mp3'
    
    def test_generate_output_path_batch_mode_file(self):
        """Test output path generation in batch mode with file."""
        test_file = os.path.join(self.temp_dir, 'research_paper.pdf')
        with open(test_file, 'w') as f:
            f.write("dummy")
        
        result = generate_output_path(test_file, self.args, batch_mode=True)
        
        assert result.endswith('research_paper.mp3')
        assert result.startswith(self.args.output_dir)
    
    def test_generate_output_path_batch_mode_arxiv(self):
        """Test output path generation in batch mode with arXiv URL."""
        arxiv_url = 'https://arxiv.org/abs/2301.12345'
        result = generate_output_path(arxiv_url, self.args, batch_mode=True)
        
        assert 'arxiv_2301.12345.mp3' in result
        assert result.startswith(self.args.output_dir)
    
    @patch('paper_voice.cli.determine_input_type')
    @patch('paper_voice.cli.load_content')
    @patch('paper_voice.cli.enhance_content')
    @patch('paper_voice.cli.process_math_expressions')
    @patch('paper_voice.cli.generate_audio')
    @patch('os.makedirs')
    def test_process_single_paper_success(self, mock_makedirs, mock_generate_audio, 
                                        mock_process_math, mock_enhance, mock_load, mock_determine):
        """Test successful single paper processing."""
        # Setup mocks
        mock_determine.return_value = 'pdf'
        mock_load.return_value = ('content', 'PDF')
        mock_enhance.return_value = 'enhanced content'
        mock_process_math.return_value = 'processed content'
        mock_generate_audio.return_value = '/path/to/output.mp3'
        
        input_path = 'test.pdf'
        result = process_single_paper(input_path, self.args, batch_mode=True)
        
        output_path, success, error_msg = result
        assert success
        assert error_msg == ""
        assert output_path == '/path/to/output.mp3'
    
    @patch('paper_voice.cli.determine_input_type')
    def test_process_single_paper_failure(self, mock_determine):
        """Test single paper processing failure."""
        # Setup mock to raise exception
        mock_determine.side_effect = Exception("Test error")
        
        input_path = 'test.pdf'
        result = process_single_paper(input_path, self.args, batch_mode=True)
        
        output_path, success, error_msg = result
        assert not success
        assert error_msg == "Test error"
        assert output_path == input_path
    
    @patch('paper_voice.cli.ThreadPoolExecutor')
    @patch('paper_voice.cli.process_single_paper')
    @patch('os.makedirs')
    def test_process_batch(self, mock_makedirs, mock_process_single, mock_executor_class):
        """Test batch processing."""
        # Setup mocks
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Mock futures
        mock_future1 = Mock()
        mock_future2 = Mock()
        mock_future1.result.return_value = ('/path/output1.mp3', True, '')
        mock_future2.result.return_value = ('/path/output2.mp3', True, '')
        
        mock_executor.submit.side_effect = [mock_future1, mock_future2]
        mock_executor.__enter__.return_value.as_completed = Mock(return_value=[mock_future1, mock_future2])
        
        # Mock as_completed
        with patch('paper_voice.cli.as_completed', return_value=[mock_future1, mock_future2]):
            inputs = ['paper1.pdf', 'paper2.pdf']
            
            # Capture print output
            with patch('builtins.print') as mock_print:
                process_batch(inputs, self.args)
            
            # Verify executor was called
            assert mock_executor.submit.call_count == 2


class TestBatchIntegration:
    """Integration tests for batch processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)
    
    def test_end_to_end_batch_workflow(self):
        """Test complete batch processing workflow."""
        # This would be a more complex integration test
        # that would require actual API keys and could be slow
        # For now, we'll just verify the structure is correct
        
        from paper_voice.cli import main, setup_parser
        
        parser = setup_parser()
        
        # Verify batch arguments exist
        args = parser.parse_args(['--batch', '--api-key', 'fake-key', 'test.pdf'])
        assert args.batch == True
        assert args.api_key == 'fake-key'
        assert args.input == ['test.pdf']