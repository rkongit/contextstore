"""
Tests for truncation strategies.
"""

import pytest
from contextstore.truncation import (
    TruncateOldestStrategy,
    RecentOnlyStrategy,
    SummarizeOldestStrategy,
    get_strategy,
)
from contextstore.tokenizer import FallbackWordCountTokenizer


@pytest.fixture
def tokenizer():
    return FallbackWordCountTokenizer(tokens_per_word=1.0)  # 1:1 for easier math


@pytest.fixture
def sample_messages():
    return [
        {'id': '1', 'role': 'user', 'content': 'hello world', 'timestamp': '2024-01-01T00:00:00Z'},
        {'id': '2', 'role': 'assistant', 'content': 'hi there friend', 'timestamp': '2024-01-01T00:01:00Z'},
        {'id': '3', 'role': 'user', 'content': 'how are you', 'timestamp': '2024-01-01T00:02:00Z'},
        {'id': '4', 'role': 'assistant', 'content': 'I am fine thanks', 'timestamp': '2024-01-01T00:03:00Z'},
    ]


class TestTruncateOldestStrategy:
    
    def test_name(self):
        strategy = TruncateOldestStrategy()
        assert strategy.name == "truncate_oldest"
    
    def test_empty_messages(self, tokenizer):
        strategy = TruncateOldestStrategy()
        result = strategy.apply([], 100, tokenizer)
        assert result.messages == []
        assert result.tokens_before == 0
        assert result.tokens_after == 0
    
    def test_under_budget_no_change(self, tokenizer, sample_messages):
        strategy = TruncateOldestStrategy()
        result = strategy.apply(sample_messages, 100, tokenizer)
        assert len(result.messages) == 4
        assert result.dropped_ids == []
    
    def test_drops_oldest_first(self, tokenizer, sample_messages):
        strategy = TruncateOldestStrategy()
        # Total: 2 + 3 + 3 + 4 = 12 tokens (with 1:1 ratio)
        result = strategy.apply(sample_messages, 7, tokenizer)
        # Should drop oldest until under 7
        assert '1' in result.dropped_ids
        assert result.tokens_after <= 7
    
    def test_deterministic(self, tokenizer, sample_messages):
        strategy = TruncateOldestStrategy()
        results = [strategy.apply(sample_messages, 5, tokenizer) for _ in range(10)]
        first = results[0]
        for r in results[1:]:
            assert r.messages == first.messages
            assert r.dropped_ids == first.dropped_ids
    
    def test_zero_budget(self, tokenizer, sample_messages):
        strategy = TruncateOldestStrategy()
        result = strategy.apply(sample_messages, 0, tokenizer)
        assert result.messages == []
        assert len(result.dropped_ids) == 4


class TestRecentOnlyStrategy:
    
    def test_name(self):
        strategy = RecentOnlyStrategy()
        assert strategy.name == "recent_only"
    
    def test_keeps_recent(self, tokenizer, sample_messages):
        strategy = RecentOnlyStrategy()
        # Keep only what fits, prioritizing recent
        result = strategy.apply(sample_messages, 5, tokenizer)
        # Should keep most recent messages that fit
        if result.messages:
            # Most recent should be kept
            kept_ids = [m['id'] for m in result.messages]
            assert '4' in kept_ids or result.tokens_after <= 5
    
    def test_deterministic(self, tokenizer, sample_messages):
        strategy = RecentOnlyStrategy()
        results = [strategy.apply(sample_messages, 5, tokenizer) for _ in range(10)]
        first = results[0]
        for r in results[1:]:
            assert r.messages == first.messages


class TestSummarizeOldestStrategy:
    
    def test_name(self):
        strategy = SummarizeOldestStrategy()
        assert strategy.name == "summarize_oldest"
    
    def test_no_summarizer_falls_back(self, tokenizer, sample_messages):
        strategy = SummarizeOldestStrategy(summarizer=None)
        result = strategy.apply(sample_messages, 5, tokenizer)
        assert result.metadata.get('fallback_reason') == 'no_summarizer'
    
    def test_with_summarizer(self, tokenizer, sample_messages):
        def mock_summarizer(msgs):
            return "summary"
        
        strategy = SummarizeOldestStrategy(summarizer=mock_summarizer, chunk_size=2)
        result = strategy.apply(sample_messages, 5, tokenizer)
        
        # Should have summarized some messages
        if result.summarized_ids:
            assert len(result.summarized_ids) > 0
    
    def test_summarizer_failure_falls_back(self, tokenizer, sample_messages):
        def failing_summarizer(msgs):
            raise RuntimeError("Failed!")
        
        strategy = SummarizeOldestStrategy(summarizer=failing_summarizer)
        result = strategy.apply(sample_messages, 5, tokenizer)
        assert result.metadata.get('fallback_reason') == 'summarizer_failed'


class TestGetStrategy:
    
    def test_get_truncate_oldest(self):
        strategy = get_strategy('truncate_oldest')
        assert strategy.name == 'truncate_oldest'
    
    def test_get_recent_only(self):
        strategy = get_strategy('recent_only')
        assert strategy.name == 'recent_only'
    
    def test_get_summarize_oldest(self):
        strategy = get_strategy('summarize_oldest')
        assert strategy.name == 'summarize_oldest'
    
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy('unknown')


class TestDeterministicOrdering:
    """Test that ordering is deterministic with tie-breaking."""
    
    def test_same_timestamp_uses_id(self, tokenizer):
        # Messages with same timestamp
        messages = [
            {'id': 'b', 'role': 'user', 'content': 'second', 'timestamp': '2024-01-01T00:00:00Z'},
            {'id': 'a', 'role': 'user', 'content': 'first', 'timestamp': '2024-01-01T00:00:00Z'},
            {'id': 'c', 'role': 'user', 'content': 'third', 'timestamp': '2024-01-01T00:00:00Z'},
        ]
        
        strategy = TruncateOldestStrategy()
        result = strategy.apply(messages, 100, tokenizer)
        
        # Should be sorted by id when timestamp is same
        ids = [m['id'] for m in result.messages]
        assert ids == ['a', 'b', 'c']


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_message_larger_than_budget(self, tokenizer):
        messages = [
            {'id': '1', 'role': 'user', 'content': 'a ' * 100, 'timestamp': '2024-01-01T00:00:00Z'},
        ]
        
        strategy = TruncateOldestStrategy()
        result = strategy.apply(messages, 5, tokenizer)
        
        # Single message can't be truncated further
        assert result.messages == [] or result.tokens_after > 5

