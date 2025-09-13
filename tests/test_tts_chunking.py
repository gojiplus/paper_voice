"""
Tests for TTS text chunking functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, Mock, call
from paper_voice.tts import (
    _split_text_for_tts,
    synthesize_speech_chunked,
    synthesize_speech
)


class TestTextSplitting:
    """Test text splitting for TTS chunking."""
    
    def test_split_short_text(self):
        """Test that short text is not split."""
        text = "This is a short text."
        chunks = _split_text_for_tts(text, 4000)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_long_text_sentences(self):
        """Test splitting long text at sentence boundaries."""
        sentences = ["This is sentence one. ", "This is sentence two. ", "This is sentence three. "]
        text = "".join(sentences * 100)  # Make it long enough to require splitting
        
        chunks = _split_text_for_tts(text, 1000)
        
        assert len(chunks) > 1
        # Each chunk should be under the limit
        for chunk in chunks:
            assert len(chunk) <= 1000
        
        # Joining chunks should recreate original text
        rejoined = "".join(chunks)
        assert rejoined.strip() == text.strip()
    
    def test_split_very_long_sentence(self):
        """Test splitting when individual sentence exceeds limit."""
        long_sentence = "This is a very long sentence with many words " * 100
        
        chunks = _split_text_for_tts(long_sentence, 200)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200
    
    def test_split_with_newlines(self):
        """Test that splitting preserves content structure."""
        text = "First paragraph.\n\nSecond paragraph. Third sentence.\n\nThird paragraph."
        chunks = _split_text_for_tts(text, 50)
        
        # Should be split but preserve content
        rejoined = "".join(chunks)
        assert "First paragraph" in rejoined
        assert "Second paragraph" in rejoined
        assert "Third paragraph" in rejoined
    
    def test_split_edge_cases(self):
        """Test edge cases for text splitting."""
        # Empty text
        assert _split_text_for_tts("", 1000) == [""]
        
        # Text exactly at limit
        text = "a" * 1000
        chunks = _split_text_for_tts(text, 1000)
        assert len(chunks) == 1
        
        # Text just over limit
        text = "a" * 1001
        chunks = _split_text_for_tts(text, 1000)
        assert len(chunks) > 1


class TestChunkedSynthesis:
    """Test the chunked speech synthesis function."""
    
    @patch('paper_voice.tts.synthesize_speech')
    @patch('paper_voice.tts.AudioSegment')
    def test_synthesize_short_text_no_chunking(self, mock_audio, mock_synth):
        """Test that short text bypasses chunking."""
        short_text = "Short text"
        output_path = "/tmp/test.mp3"
        
        # Mock synthesize_speech to return the output path
        mock_synth.return_value = output_path
        
        result = synthesize_speech_chunked(
            short_text, output_path, use_openai=True, api_key="test-key"
        )
        
        # Should call regular synthesize_speech once
        mock_synth.assert_called_once()
        assert result == output_path
    
    @patch('paper_voice.tts.synthesize_speech')
    @patch('paper_voice.tts.AudioSegment')
    @patch('tempfile.mkdtemp')
    @patch('shutil.rmtree')
    def test_synthesize_long_text_with_chunking(self, mock_rmtree, mock_tempdir, mock_audio, mock_synth):
        """Test chunking with long text."""
        long_text = "This is a long sentence. " * 200  # Make it long enough to chunk
        output_path = "/tmp/test.mp3"
        
        # Mock tempfile
        mock_tempdir.return_value = "/tmp/chunks"
        
        # Mock synthesize_speech calls
        mock_synth.side_effect = [
            "/tmp/chunks/chunk_000.mp3",
            "/tmp/chunks/chunk_001.mp3"
        ]
        
        # Mock AudioSegment
        mock_segment = Mock()
        mock_audio.from_mp3.return_value = mock_segment
        mock_combined = Mock()
        mock_audio.empty.return_value = mock_combined
        mock_combined.__iadd__.return_value = mock_combined
        mock_audio.silent.return_value = Mock()
        
        result = synthesize_speech_chunked(
            long_text, output_path, use_openai=True, api_key="test-key"
        )
        
        # Should call synthesize_speech multiple times
        assert mock_synth.call_count > 1
        
        # Should export combined audio
        mock_combined.export.assert_called_once_with(output_path, format="mp3")
        
        # Should clean up temp directory
        mock_rmtree.assert_called_once_with("/tmp/chunks", ignore_errors=True)
        
        assert result == output_path
    
    @patch('paper_voice.tts.synthesize_speech')
    def test_synthesize_offline_no_chunking_needed(self, mock_synth):
        """Test that offline TTS doesn't chunk unnecessarily."""
        text = "Test text for offline synthesis"
        output_path = "/tmp/test.wav"
        
        mock_synth.return_value = output_path
        
        result = synthesize_speech_chunked(
            text, output_path, use_openai=False
        )
        
        # Should just call regular synthesis
        mock_synth.assert_called_once_with(
            text, output_path, "", 200, False, None, "tts-1", "alloy"
        )
        assert result == output_path
    
    @patch('paper_voice.tts.synthesize_speech')
    @patch('paper_voice.tts.AudioSegment')
    def test_chunking_error_handling(self, mock_audio, mock_synth):
        """Test error handling in chunked synthesis."""
        long_text = "Error test. " * 500
        output_path = "/tmp/test.mp3"
        
        # Mock AudioSegment to be None (missing dependency)
        mock_audio_module = Mock()
        mock_audio_module.AudioSegment = None
        
        with patch('paper_voice.tts.AudioSegment', None):
            with pytest.raises(RuntimeError, match="pydub is required"):
                synthesize_speech_chunked(
                    long_text, output_path, use_openai=True, api_key="test-key"
                )


class TestIntegration:
    """Integration tests for TTS functionality."""
    
    def test_chunk_size_calculation(self):
        """Test that chunks are properly sized."""
        # Create text that should be split into exactly 2 chunks
        sentence = "This is a test sentence with exactly fifty characters. "
        # 51 chars * 80 = 4080 chars (just over 4000 limit)
        text = sentence * 80
        
        chunks = _split_text_for_tts(text, 4000)
        
        assert len(chunks) >= 2
        assert all(len(chunk) <= 4000 for chunk in chunks)
        
        # Verify content is preserved
        rejoined = "".join(chunks).strip()
        assert len(rejoined) == len(text.strip())
    
    def test_speech_synthesis_parameters(self):
        """Test that synthesis parameters are passed correctly."""
        with patch('paper_voice.tts.synthesize_speech') as mock_synth:
            mock_synth.return_value = "/tmp/test.mp3"
            
            synthesize_speech_chunked(
                "Test text",
                "/tmp/output.mp3",
                voice="custom_voice",
                rate=250,
                use_openai=True,
                api_key="test-key",
                model="tts-1-hd",
                openai_voice="nova"
            )
            
            # Verify parameters are passed through
            mock_synth.assert_called_once_with(
                "Test text",
                "/tmp/output.mp3", 
                "custom_voice",
                250,
                True,
                "test-key",
                "tts-1-hd",
                "nova"
            )


if __name__ == "__main__":
    pytest.main([__file__])