"""
ContextStore - A generic package for persisting conversation history.

This package provides backends for storing and retrieving conversation history
across different storage mechanisms, with token-aware context management.
"""

from .interfaces import MemoryBackend
from .memory_backends import InMemoryMemory, SQLiteMemory
from .models import Interaction

# Token-aware context management (v0.3.0+)
from .tokenizer_interfaces import (
    Tokenizer,
    TokenCountResult,
    TruncationStrategy,
    TruncationResult,
    BuildResult,
)
from .tokenizer import (
    FallbackWordCountTokenizer,
    ModelTokenizer,
    tokenizer_from_name,
)
from .truncation import (
    TruncateOldestStrategy,
    RecentOnlyStrategy,
    SummarizeOldestStrategy,
    get_strategy,
)
from .context_builder import ContextBuilder

__version__ = "0.3.0"
__all__ = [
    # Memory backends
    "MemoryBackend",
    "InMemoryMemory",
    "SQLiteMemory",
    "Interaction",
    # Tokenizer
    "Tokenizer",
    "TokenCountResult",
    "FallbackWordCountTokenizer",
    "ModelTokenizer",
    "tokenizer_from_name",
    # Truncation
    "TruncationStrategy",
    "TruncationResult",
    "TruncateOldestStrategy",
    "RecentOnlyStrategy",
    "SummarizeOldestStrategy",
    "get_strategy",
    # Context Builder
    "ContextBuilder",
    "BuildResult",
]

