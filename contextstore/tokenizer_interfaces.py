"""
Token-aware context management interfaces.

This module defines the abstract interfaces for tokenization, truncation strategies,
and context building. These interfaces are frozen for version 0.3.0.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable


# =============================================================================
# Tokenizer Interfaces
# =============================================================================

@dataclass
class TokenCountResult:
    """
    Result of a token counting operation.
    
    Attributes:
        count: Number of tokens counted.
        approximate: True if count is an estimate (e.g., word-count heuristic).
    """
    count: int
    approximate: bool = False


class Tokenizer(ABC):
    """
    Abstract base class for tokenizers.
    
    Implementations must provide token counting. Tokenization (returning token IDs)
    is optional and may return None for fallback implementations.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Tokenizer identifier.
        
        Examples: 'cl100k_base', 'gpt-4', 'fallback_word_count'
        """
        ...
    
    @abstractmethod
    def count_tokens(self, text: str) -> TokenCountResult:
        """
        Count the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            TokenCountResult with count and approximate flag.
        """
        ...
    
    @abstractmethod
    def tokenize(self, text: str) -> Optional[List[int]]:
        """
        Tokenize text into token IDs.
        
        Args:
            text: The text to tokenize.
            
        Returns:
            List of token IDs if supported, None otherwise.
            Fallback tokenizers should return None.
        """
        ...


# =============================================================================
# Truncation Strategy Interfaces
# =============================================================================

@dataclass
class TruncationResult:
    """
    Result of applying a truncation strategy.
    
    Attributes:
        messages: The resulting messages after truncation.
        tokens_before: Total tokens before truncation.
        tokens_after: Total tokens after truncation.
        dropped_ids: IDs of messages that were dropped.
        summarized_ids: IDs of messages that were summarized.
        metadata: Strategy-specific metadata.
    """
    messages: List[Dict[str, Any]]
    tokens_before: int
    tokens_after: int
    dropped_ids: List[str] = field(default_factory=list)
    summarized_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TruncationStrategy(ABC):
    """
    Abstract base class for truncation strategies.
    
    Strategies must be pure functions: same inputs always produce same outputs.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'truncate_oldest', 'recent_only')."""
        ...
    
    @abstractmethod
    def apply(
        self,
        messages: List[Dict[str, Any]],
        token_budget: int,
        tokenizer: Tokenizer,
    ) -> TruncationResult:
        """
        Apply truncation strategy to fit messages within token budget.
        
        Args:
            messages: List of messages sorted by timestamp (oldest first).
                Each message must have: id, role, content, timestamp.
                Optional: metadata.
            token_budget: Maximum number of tokens allowed.
            tokenizer: Tokenizer for counting tokens.
            
        Returns:
            TruncationResult with truncated messages and metadata.
            
        Guarantees:
            - Deterministic: same inputs produce identical outputs.
            - tokens_after <= token_budget (unless single message exceeds budget).
        """
        ...


# =============================================================================
# Context Builder Interfaces
# =============================================================================

@dataclass
class BuildResult:
    """
    Result of context building.
    
    Attributes:
        messages: The resulting messages within token budget.
        total_tokens: Total token count of resulting messages.
        approximate: True if token count is approximate.
        strategy_used: Name of the truncation strategy applied.
        metadata: Additional build metadata.
    """
    messages: List[Dict[str, Any]]
    total_tokens: int
    approximate: bool
    strategy_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Type alias for filter functions
MessageFilter = Callable[[Dict[str, Any]], bool]

# Type alias for summarizer callbacks
SummarizerCallback = Callable[[List[Dict[str, Any]]], str]

# Type alias for metrics callbacks
MetricsCallback = Callable[[Dict[str, Any]], None]


class ContextBuilderInterface(ABC):
    """
    Abstract interface for context builders.
    
    Context builders combine tokenization and truncation strategies to produce
    context windows that fit within token budgets.
    """
    
    @abstractmethod
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
            messages: Input messages. Each must have:
                - id: str (unique identifier)
                - role: str (e.g., 'user', 'assistant', 'system')
                - content: str (message text)
                - timestamp: str (ISO 8601 format)
                - metadata: Optional[Dict] (additional data)
            max_tokens: Maximum tokens allowed in output.
            strategy: Truncation strategy name. Options:
                - 'truncate_oldest': Drop oldest messages first.
                - 'recent_only': Keep only recent messages.
                - 'summarize_oldest': Summarize oldest messages.
            strategy_opts: Strategy-specific options:
                - For 'summarize_oldest': {'summarizer': Callable, 'chunk_size': int}
            pre_filter: Filter messages before processing (True = include).
            post_filter: Filter messages after truncation (True = include).
            metrics_callback: Optional callback for telemetry.
            
        Returns:
            BuildResult with messages fitting within budget.
            
        Guarantees:
            - Deterministic output for identical inputs.
            - total_tokens <= max_tokens (unless single message exceeds budget).
            - Messages ordered by timestamp (oldest first).
        """
        ...


# =============================================================================
# Ordering Constants
# =============================================================================

# Deterministic ordering: messages sorted by (timestamp, id) for tie-breaking
SORT_KEY_TIMESTAMP = 'timestamp'
SORT_KEY_ID = 'id'


def deterministic_sort_key(message: Dict[str, Any]) -> tuple:
    """
    Generate a deterministic sort key for a message.
    
    Sort order: primary by timestamp, secondary by id for tie-breaking.
    
    Args:
        message: Message dict with 'timestamp' and 'id' fields.
        
    Returns:
        Tuple (timestamp, id) for sorting.
    """
    return (
        message.get(SORT_KEY_TIMESTAMP, ''),
        message.get(SORT_KEY_ID, ''),
    )

