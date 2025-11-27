"""
Truncation strategies for token-aware context management.

Provides:
- TruncateOldestStrategy: Drop oldest messages until under budget
- RecentOnlyStrategy: Keep N recent messages under budget
- SummarizeOldestStrategy: Summarize oldest messages via callback
"""

from typing import List, Dict, Any, Optional, Callable

from .tokenizer_interfaces import (
    Tokenizer,
    TruncationStrategy,
    TruncationResult,
    deterministic_sort_key,
)


def _count_message_tokens(message: Dict[str, Any], tokenizer: Tokenizer) -> int:
    """Count tokens in a message's content."""
    content = message.get('content', '')
    if not content:
        return 0
    return tokenizer.count_tokens(content).count


def _total_tokens(messages: List[Dict[str, Any]], tokenizer: Tokenizer) -> int:
    """Sum tokens across all messages."""
    return sum(_count_message_tokens(m, tokenizer) for m in messages)


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort messages deterministically by (timestamp, id)."""
    return sorted(messages, key=deterministic_sort_key)


# =============================================================================
# Truncate Oldest Strategy
# =============================================================================

class TruncateOldestStrategy(TruncationStrategy):
    """
    Drop oldest messages first until total tokens fit within budget.
    
    Messages are processed in timestamp order (oldest first).
    Deterministic: same inputs always produce identical outputs.
    """
    
    @property
    def name(self) -> str:
        return "truncate_oldest"
    
    def apply(
        self,
        messages: List[Dict[str, Any]],
        token_budget: int,
        tokenizer: Tokenizer,
    ) -> TruncationResult:
        if not messages:
            return TruncationResult(
                messages=[],
                tokens_before=0,
                tokens_after=0,
                dropped_ids=[],
                summarized_ids=[],
            )
        
        sorted_messages = _normalize_messages(messages)
        tokens_before = _total_tokens(sorted_messages, tokenizer)
        
        # Already under budget
        if tokens_before <= token_budget:
            return TruncationResult(
                messages=sorted_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                dropped_ids=[],
                summarized_ids=[],
            )
        
        # Drop oldest messages until under budget
        dropped_ids = []
        current_tokens = tokens_before
        drop_index = 0
        
        while current_tokens > token_budget and drop_index < len(sorted_messages):
            msg = sorted_messages[drop_index]
            msg_tokens = _count_message_tokens(msg, tokenizer)
            dropped_ids.append(msg.get('id', ''))
            current_tokens -= msg_tokens
            drop_index += 1
        
        remaining = sorted_messages[drop_index:]
        tokens_after = _total_tokens(remaining, tokenizer)
        
        return TruncationResult(
            messages=remaining,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            dropped_ids=dropped_ids,
            summarized_ids=[],
            metadata={'strategy': 'truncate_oldest'},
        )


# =============================================================================
# Recent Only Strategy
# =============================================================================

class RecentOnlyStrategy(TruncationStrategy):
    """
    Keep only the most recent messages that fit within budget.
    
    Works backwards from newest, keeping messages until budget exhausted.
    """
    
    @property
    def name(self) -> str:
        return "recent_only"
    
    def apply(
        self,
        messages: List[Dict[str, Any]],
        token_budget: int,
        tokenizer: Tokenizer,
    ) -> TruncationResult:
        if not messages:
            return TruncationResult(
                messages=[],
                tokens_before=0,
                tokens_after=0,
                dropped_ids=[],
                summarized_ids=[],
            )
        
        sorted_messages = _normalize_messages(messages)
        tokens_before = _total_tokens(sorted_messages, tokenizer)
        
        # Already under budget
        if tokens_before <= token_budget:
            return TruncationResult(
                messages=sorted_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                dropped_ids=[],
                summarized_ids=[],
            )
        
        # Keep recent messages until budget exhausted (work backwards)
        kept = []
        kept_tokens = 0
        dropped_ids = []
        
        for msg in reversed(sorted_messages):
            msg_tokens = _count_message_tokens(msg, tokenizer)
            if kept_tokens + msg_tokens <= token_budget:
                kept.insert(0, msg)  # Maintain order
                kept_tokens += msg_tokens
            else:
                dropped_ids.insert(0, msg.get('id', ''))
        
        return TruncationResult(
            messages=kept,
            tokens_before=tokens_before,
            tokens_after=kept_tokens,
            dropped_ids=dropped_ids,
            summarized_ids=[],
            metadata={'strategy': 'recent_only'},
        )


# =============================================================================
# Summarize Oldest Strategy
# =============================================================================

# Type for summarizer callback
SummarizerCallback = Callable[[List[Dict[str, Any]]], str]


class SummarizeOldestStrategy(TruncationStrategy):
    """
    Summarize oldest messages using a user-provided callback.
    
    If summarizer fails, falls back to dropping the messages.
    """
    
    def __init__(
        self,
        summarizer: Optional[SummarizerCallback] = None,
        chunk_size: int = 5,
    ):
        """
        Initialize strategy.
        
        Args:
            summarizer: Callback that takes messages and returns summary text.
            chunk_size: Number of oldest messages to summarize at once.
        """
        self._summarizer = summarizer
        self._chunk_size = chunk_size
    
    @property
    def name(self) -> str:
        return "summarize_oldest"
    
    def apply(
        self,
        messages: List[Dict[str, Any]],
        token_budget: int,
        tokenizer: Tokenizer,
    ) -> TruncationResult:
        if not messages:
            return TruncationResult(
                messages=[],
                tokens_before=0,
                tokens_after=0,
                dropped_ids=[],
                summarized_ids=[],
            )
        
        sorted_messages = _normalize_messages(messages)
        tokens_before = _total_tokens(sorted_messages, tokenizer)
        
        # Already under budget
        if tokens_before <= token_budget:
            return TruncationResult(
                messages=sorted_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                dropped_ids=[],
                summarized_ids=[],
            )
        
        # No summarizer - fall back to dropping
        if self._summarizer is None:
            fallback = TruncateOldestStrategy()
            result = fallback.apply(messages, token_budget, tokenizer)
            result.metadata['fallback_reason'] = 'no_summarizer'
            return result
        
        # Summarize oldest chunk
        chunk = sorted_messages[:self._chunk_size]
        remaining = sorted_messages[self._chunk_size:]
        summarized_ids = [m.get('id', '') for m in chunk]
        
        try:
            summary_text = self._summarizer(chunk)
        except Exception:
            # Summarizer failed - drop the messages instead
            fallback = TruncateOldestStrategy()
            result = fallback.apply(messages, token_budget, tokenizer)
            result.metadata['fallback_reason'] = 'summarizer_failed'
            return result
        
        # Create summary message
        summary_message = {
            'id': f"summary_{chunk[0].get('id', 'unknown')}",
            'role': 'system',
            'content': summary_text,
            'timestamp': chunk[0].get('timestamp', ''),
            'metadata': {'is_summary': True, 'summarized_count': len(chunk)},
        }
        
        new_messages = [summary_message] + remaining
        tokens_after = _total_tokens(new_messages, tokenizer)
        
        # If still over budget, recurse
        if tokens_after > token_budget:
            return self.apply(new_messages, token_budget, tokenizer)
        
        return TruncationResult(
            messages=new_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            dropped_ids=[],
            summarized_ids=summarized_ids,
            metadata={'strategy': 'summarize_oldest'},
        )


# =============================================================================
# Strategy Registry
# =============================================================================

_STRATEGIES: Dict[str, type] = {
    'truncate_oldest': TruncateOldestStrategy,
    'recent_only': RecentOnlyStrategy,
    'summarize_oldest': SummarizeOldestStrategy,
}


def get_strategy(
    name: str,
    **opts: Any,
) -> TruncationStrategy:
    """
    Get a truncation strategy by name.
    
    Args:
        name: Strategy name ('truncate_oldest', 'recent_only', 'summarize_oldest').
        **opts: Strategy-specific options.
        
    Returns:
        TruncationStrategy instance.
        
    Raises:
        ValueError: If strategy name is unknown.
    """
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(_STRATEGIES.keys())}")
    
    strategy_cls = _STRATEGIES[name]
    
    if name == 'summarize_oldest':
        return strategy_cls(
            summarizer=opts.get('summarizer'),
            chunk_size=opts.get('chunk_size', 5),
        )
    
    return strategy_cls()

