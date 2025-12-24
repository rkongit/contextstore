"""
Tests for tokenizer module.
"""

import pytest
from contextstore.tokenizer import (
    FallbackWordCountTokenizer,
    ModelTokenizer,
    tokenizer_from_name,
)
from contextstore.tokenizer_interfaces import TokenCountResult


class TestFallbackWordCountTokenizer:
    """Tests for FallbackWordCountTokenizer."""
    
    def test_name(self):
        tokenizer = FallbackWordCountTokenizer()
        assert tokenizer.name == "fallback_word_count"
    
    def test_empty_string(self):
        tokenizer = FallbackWordCountTokenizer()
        result = tokenizer.count_tokens("")
        assert result.count == 0
        assert result.approximate is True
    
    def test_single_word(self):
        tokenizer = FallbackWordCountTokenizer()
        result = tokenizer.count_tokens("hello")
        assert result.count == 1  # 1 word * 1.3 = 1.3 -> 1
        assert result.approximate is True
    
    def test_multiple_words(self):
        tokenizer = FallbackWordCountTokenizer()
        result = tokenizer.count_tokens("hello world foo bar")
        # 4 words * 1.3 = 5.2 -> 5
        assert result.count == 5
        assert result.approximate is True
    
    def test_custom_multiplier(self):
        tokenizer = FallbackWordCountTokenizer(tokens_per_word=2.0)
        result = tokenizer.count_tokens("hello world")
        # 2 words * 2.0 = 4
        assert result.count == 4
    
    def test_deterministic(self):
        """Same input always produces same output."""
        tokenizer = FallbackWordCountTokenizer()
        text = "The quick brown fox jumps over the lazy dog."
        
        results = [tokenizer.count_tokens(text) for _ in range(10)]
        assert all(r.count == results[0].count for r in results)
    
    def test_tokenize_returns_none(self):
        tokenizer = FallbackWordCountTokenizer()
        assert tokenizer.tokenize("hello world") is None
    
    def test_handles_punctuation(self):
        tokenizer = FallbackWordCountTokenizer()
        result = tokenizer.count_tokens("Hello, world! How are you?")
        # Splits on whitespace: ["Hello,", "world!", "How", "are", "you?"] = 5 words
        assert result.count == 6  # 5 * 1.3 = 6.5 -> 6


class TestModelTokenizer:
    """Tests for ModelTokenizer with mock tokenizer."""
    
    def test_name(self):
        mock_tokenizer = MockTiktoken()
        tokenizer = ModelTokenizer(mock_tokenizer, "test_encoding")
        assert tokenizer.name == "test_encoding"
    
    def test_count_tokens(self):
        mock_tokenizer = MockTiktoken()
        tokenizer = ModelTokenizer(mock_tokenizer, "test")
        result = tokenizer.count_tokens("hello world")
        assert result.count == 2  # Mock returns 1 token per word
        assert result.approximate is False
    
    def test_empty_string(self):
        mock_tokenizer = MockTiktoken()
        tokenizer = ModelTokenizer(mock_tokenizer, "test")
        result = tokenizer.count_tokens("")
        assert result.count == 0
        assert result.approximate is False
    
    def test_tokenize(self):
        mock_tokenizer = MockTiktoken()
        tokenizer = ModelTokenizer(mock_tokenizer, "test")
        tokens = tokenizer.tokenize("hello world")
        assert tokens == [1, 2]  # Mock token IDs


class MockTiktoken:
    """Mock tiktoken encoder for testing."""
    
    def encode(self, text: str) -> list:
        if not text:
            return []
        words = text.split()
        return list(range(1, len(words) + 1))


class TestTokenizerFactory:
    """Tests for tokenizer_from_name factory."""
    
    def test_fallback_explicit(self):
        tokenizer = tokenizer_from_name("fallback")
        assert tokenizer.name == "fallback_word_count"
    
    def test_fallback_custom_multiplier(self):
        tokenizer = tokenizer_from_name("fallback", tokens_per_word=2.0)
        result = tokenizer.count_tokens("hello world")
        assert result.count == 4  # 2 * 2.0
    
    def test_unknown_tokenizer_returns_fallback(self):
        """Unknown tokenizer name should return fallback with warning."""
        tokenizer = tokenizer_from_name("unknown_tokenizer_xyz")
        # Should return fallback (either because tiktoken not installed or unknown encoding)
        result = tokenizer.count_tokens("hello world")
        assert result.approximate is True  # Fallback always approximate
    
    def test_factory_never_raises(self):
        """Factory should never raise, always returns a usable tokenizer."""
        names = [
            "unknown",
            "not_a_real_tokenizer",
            "",
            "cl100k_base",
            "gpt-4",
        ]
        for name in names:
            tokenizer = tokenizer_from_name(name)
            # Should be able to count tokens without error
            result = tokenizer.count_tokens("test")
            assert isinstance(result.count, int)


class TestTokenizerWithTiktoken:
    """Tests that run only if tiktoken is installed."""
    
    @pytest.fixture
    def tiktoken_available(self):
        try:
            import tiktoken
            return True
        except ImportError:
            pytest.skip("tiktoken not installed")
    
    def test_cl100k_base(self, tiktoken_available):
        tokenizer = tokenizer_from_name("cl100k_base")
        assert tokenizer.name == "cl100k_base"
        
        result = tokenizer.count_tokens("hello world")
        assert result.approximate is False
        assert result.count > 0
    
    def test_gpt4_model_name(self, tiktoken_available):
        tokenizer = tokenizer_from_name("gpt-4")
        # Should resolve to cl100k_base
        result = tokenizer.count_tokens("hello world")
        assert result.approximate is False
    
    def test_tokenize_returns_ids(self, tiktoken_available):
        tokenizer = tokenizer_from_name("cl100k_base")
        tokens = tokenizer.tokenize("hello world")
        assert tokens is not None
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

