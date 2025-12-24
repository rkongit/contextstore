"""
Tests for ContextBuilder.
"""

import pytest
from contextstore.context_builder import ContextBuilder
from contextstore.tokenizer import FallbackWordCountTokenizer


@pytest.fixture
def tokenizer():
    return FallbackWordCountTokenizer(tokens_per_word=1.0)


@pytest.fixture
def builder(tokenizer):
    return ContextBuilder(tokenizer=tokenizer)


@pytest.fixture
def sample_messages():
    return [
        {'id': '1', 'role': 'user', 'content': 'hello world', 'timestamp': '2024-01-01T00:00:00Z'},
        {'id': '2', 'role': 'assistant', 'content': 'hi there', 'timestamp': '2024-01-01T00:01:00Z'},
        {'id': '3', 'role': 'user', 'content': 'how are you', 'timestamp': '2024-01-01T00:02:00Z'},
        {'id': '4', 'role': 'assistant', 'content': 'I am fine', 'timestamp': '2024-01-01T00:03:00Z'},
    ]


class TestContextBuilder:
    
    def test_under_budget_returns_all(self, builder, sample_messages):
        result = builder.build(sample_messages, max_tokens=100)
        assert len(result.messages) == 4
        assert result.strategy_used == 'none'
        assert result.metadata.get('truncation_applied') is False
    
    def test_over_budget_truncates(self, builder, sample_messages):
        # Total: 2+2+3+3 = 10 tokens
        result = builder.build(sample_messages, max_tokens=5)
        assert result.total_tokens <= 5
        assert result.strategy_used == 'truncate_oldest'
    
    def test_deterministic(self, builder, sample_messages):
        results = [builder.build(sample_messages, max_tokens=5) for _ in range(10)]
        first = results[0]
        for r in results[1:]:
            assert r.messages == first.messages
            assert r.total_tokens == first.total_tokens
    
    def test_pre_filter(self, builder, sample_messages):
        # Filter out assistant messages
        result = builder.build(
            sample_messages,
            max_tokens=100,
            pre_filter=lambda m: m['role'] == 'user',
        )
        assert all(m['role'] == 'user' for m in result.messages)
    
    def test_post_filter(self, builder, sample_messages):
        result = builder.build(
            sample_messages,
            max_tokens=100,
            post_filter=lambda m: m['role'] == 'user',
        )
        assert all(m['role'] == 'user' for m in result.messages)
    
    def test_metrics_callback(self, builder, sample_messages):
        metrics = []
        
        def callback(m):
            metrics.append(m)
        
        builder.build(sample_messages, max_tokens=100, metrics_callback=callback)
        
        assert len(metrics) == 1
        assert 'total_tokens' in metrics[0]
        assert 'message_count' in metrics[0]
    
    def test_recent_only_strategy(self, builder, sample_messages):
        result = builder.build(
            sample_messages,
            max_tokens=5,
            strategy='recent_only',
        )
        assert result.strategy_used == 'recent_only'
        assert result.total_tokens <= 5
    
    def test_summarize_strategy(self, builder, sample_messages):
        def summarizer(msgs):
            return "summary"
        
        result = builder.build(
            sample_messages,
            max_tokens=5,
            strategy='summarize_oldest',
            strategy_opts={'summarizer': summarizer},
        )
        assert result.strategy_used == 'summarize_oldest'
    
    def test_empty_messages(self, builder):
        result = builder.build([], max_tokens=100)
        assert result.messages == []
        assert result.total_tokens == 0
    
    def test_approximate_flag(self, builder, sample_messages):
        result = builder.build(sample_messages, max_tokens=100)
        assert result.approximate is True  # Fallback is always approximate


class TestContextBuilderFactory:
    
    def test_from_model(self):
        builder = ContextBuilder.from_model('gpt-4')
        assert builder._tokenizer is not None
    
    def test_from_model_fallback(self):
        builder = ContextBuilder.from_model('fallback')
        result = builder.build(
            [{'id': '1', 'role': 'user', 'content': 'test', 'timestamp': '2024-01-01T00:00:00Z'}],
            max_tokens=100,
        )
        assert result.approximate is True


class TestIdempotence:
    """Property test: running build twice yields same output."""
    
    def test_idempotent(self, builder, sample_messages):
        result1 = builder.build(sample_messages, max_tokens=5)
        result2 = builder.build(sample_messages, max_tokens=5)
        
        assert result1.messages == result2.messages
        assert result1.total_tokens == result2.total_tokens
        assert result1.strategy_used == result2.strategy_used

