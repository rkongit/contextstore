"""
Context Builder for token-aware context window management.

Combines tokenization and truncation strategies to build context windows
that fit within token budgets.
"""

from typing import List, Dict, Any, Optional, Callable

from .tokenizer_interfaces import (
    Tokenizer,
    BuildResult,
    MessageFilter,
    MetricsCallback,
    deterministic_sort_key,
)
from .tokenizer import FallbackWordCountTokenizer, tokenizer_from_name
from .truncation import get_strategy, TruncateOldestStrategy


class ContextBuilder:
    """
    Builds token-aware context windows.
    
    Combines tokenization and truncation to ensure context fits within
    model token limits while maintaining deterministic behavior.
    """
    
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        default_strategy: str = 'truncate_oldest',
    ):
        """
        Initialize the context builder.
        
        Args:
            tokenizer: Tokenizer to use. If None, uses FallbackWordCountTokenizer.
            default_strategy: Default truncation strategy name.
        """
        self._tokenizer = tokenizer or FallbackWordCountTokenizer()
        self._default_strategy = default_strategy
    
    @classmethod
    def from_model(cls, model_name: str, **opts) -> 'ContextBuilder':
        """
        Create a ContextBuilder for a specific model.
        
        Args:
            model_name: Model name (e.g., 'gpt-4', 'cl100k_base').
            **opts: Additional options for tokenizer.
            
        Returns:
            ContextBuilder configured for the model.
        """
        tokenizer = tokenizer_from_name(model_name, **opts)
        return cls(tokenizer=tokenizer)
    
    def build(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        strategy: Optional[str] = None,
        strategy_opts: Optional[Dict[str, Any]] = None,
        pre_filter: Optional[MessageFilter] = None,
        post_filter: Optional[MessageFilter] = None,
        metrics_callback: Optional[MetricsCallback] = None,
    ) -> BuildResult:
        """
        Build a context window within the specified token budget.
        
        Args:
            messages: Input messages with id, role, content, timestamp.
            max_tokens: Maximum tokens allowed.
            strategy: Truncation strategy ('truncate_oldest', 'recent_only', 'summarize_oldest').
            strategy_opts: Strategy-specific options.
            pre_filter: Filter before processing (True = include).
            post_filter: Filter after truncation (True = include).
            metrics_callback: Optional telemetry callback.
            
        Returns:
            BuildResult with messages fitting within budget.
        """
        strategy_opts = strategy_opts or {}
        strategy_name = strategy or self._default_strategy
        
        # Step 1: Normalize and sort messages deterministically
        working_messages = self._normalize_messages(messages)
        
        # Step 2: Apply pre-filter
        if pre_filter:
            working_messages = [m for m in working_messages if pre_filter(m)]
        
        # Step 3: Count initial tokens
        initial_tokens = self._count_total_tokens(working_messages)
        approximate = self._tokenizer.count_tokens("test").approximate
        
        # Step 4: Check if already under budget
        if initial_tokens <= max_tokens:
            # Still apply post-filter if provided
            if post_filter:
                working_messages = [m for m in working_messages if post_filter(m)]
                initial_tokens = self._count_total_tokens(working_messages)
            
            result = BuildResult(
                messages=working_messages,
                total_tokens=initial_tokens,
                approximate=approximate,
                strategy_used='none',
                metadata={'truncation_applied': False},
            )
            if metrics_callback:
                metrics_callback(self._build_metrics(result))
            return result
        
        # Step 5: Apply truncation strategy
        truncation_strategy = get_strategy(strategy_name, **strategy_opts)
        truncation_result = truncation_strategy.apply(
            working_messages,
            max_tokens,
            self._tokenizer,
        )
        
        working_messages = truncation_result.messages
        
        # Step 6: Apply post-filter
        if post_filter:
            working_messages = [m for m in working_messages if post_filter(m)]
            # Recount after post-filter
            final_tokens = self._count_total_tokens(working_messages)
        else:
            final_tokens = truncation_result.tokens_after
        
        # Step 7: Re-validate and iterate if necessary
        max_iterations = 10
        iteration = 0
        while final_tokens > max_tokens and working_messages and iteration < max_iterations:
            truncation_result = truncation_strategy.apply(
                working_messages,
                max_tokens,
                self._tokenizer,
            )
            working_messages = truncation_result.messages
            final_tokens = truncation_result.tokens_after
            iteration += 1
        
        result = BuildResult(
            messages=working_messages,
            total_tokens=final_tokens,
            approximate=approximate,
            strategy_used=strategy_name,
            metadata={
                'truncation_applied': True,
                'tokens_before': initial_tokens,
                'dropped_ids': truncation_result.dropped_ids,
                'summarized_ids': truncation_result.summarized_ids,
                'iterations': iteration + 1,
            },
        )
        
        if metrics_callback:
            metrics_callback(self._build_metrics(result))
        
        return result
    
    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort messages deterministically."""
        return sorted(messages, key=deterministic_sort_key)
    
    def _count_total_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Sum tokens across all messages."""
        total = 0
        for msg in messages:
            content = msg.get('content', '')
            if content:
                total += self._tokenizer.count_tokens(content).count
        return total
    
    def _build_metrics(self, result: BuildResult) -> Dict[str, Any]:
        """Build metrics dict for callback."""
        return {
            'total_tokens': result.total_tokens,
            'message_count': len(result.messages),
            'strategy_used': result.strategy_used,
            'approximate': result.approximate,
            'metadata': result.metadata,
        }

