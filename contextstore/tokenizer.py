"""
Tokenizer implementations for token-aware context management.

Provides:
- FallbackWordCountTokenizer: Heuristic-based token estimation (no dependencies)
- ModelTokenizer: Wrapper for model-specific tokenizers (e.g., tiktoken)
- tokenizer_from_name(): Factory function to resolve tokenizers
"""

import logging
import re
from typing import List, Optional, Dict, Any

from .tokenizer_interfaces import Tokenizer, TokenCountResult

logger = logging.getLogger(__name__)


# =============================================================================
# Fallback Tokenizer (No Dependencies)
# =============================================================================

class FallbackWordCountTokenizer(Tokenizer):
    """
    Approximate token counter using word-count heuristics.
    
    Uses the common approximation: tokens ≈ words × 1.3
    This is a rough estimate suitable when exact tokenization is unavailable.
    
    Always returns approximate=True in TokenCountResult.
    """
    
    # Average ratio of tokens to words for English text
    TOKENS_PER_WORD: float = 1.3
    
    def __init__(self, tokens_per_word: float = 1.3):
        """
        Initialize fallback tokenizer.
        
        Args:
            tokens_per_word: Multiplier for word count. Default 1.3.
        """
        self._tokens_per_word = tokens_per_word
    
    @property
    def name(self) -> str:
        return "fallback_word_count"
    
    def count_tokens(self, text: str) -> TokenCountResult:
        """
        Estimate token count from word count.
        
        Args:
            text: Text to count.
            
        Returns:
            TokenCountResult with approximate=True.
        """
        if not text:
            return TokenCountResult(count=0, approximate=True)
        
        # Split on whitespace and punctuation boundaries
        words = re.findall(r'\S+', text)
        word_count = len(words)
        token_estimate = int(word_count * self._tokens_per_word)
        
        return TokenCountResult(count=token_estimate, approximate=True)
    
    def tokenize(self, text: str) -> Optional[List[int]]:
        """
        Tokenization not supported for fallback.
        
        Returns:
            None (tokenization not available).
        """
        return None


# =============================================================================
# Model Tokenizer Wrapper
# =============================================================================

class ModelTokenizer(Tokenizer):
    """
    Wrapper for model-specific BPE tokenizers.
    
    Supports tiktoken (OpenAI) tokenizers. Can be extended for other backends.
    """
    
    def __init__(self, tokenizer: Any, tokenizer_name: str):
        """
        Initialize with a tokenizer instance.
        
        Args:
            tokenizer: The underlying tokenizer (e.g., tiktoken.Encoding).
            tokenizer_name: Name identifier for this tokenizer.
        """
        self._tokenizer = tokenizer
        self._name = tokenizer_name
    
    @property
    def name(self) -> str:
        return self._name
    
    def count_tokens(self, text: str) -> TokenCountResult:
        """
        Count tokens using the model tokenizer.
        
        Args:
            text: Text to count.
            
        Returns:
            TokenCountResult with approximate=False (exact count).
        """
        if not text:
            return TokenCountResult(count=0, approximate=False)
        
        tokens = self._tokenizer.encode(text)
        return TokenCountResult(count=len(tokens), approximate=False)
    
    def tokenize(self, text: str) -> Optional[List[int]]:
        """
        Tokenize text into token IDs.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of token IDs.
        """
        if not text:
            return []
        return self._tokenizer.encode(text)


# =============================================================================
# Factory Function
# =============================================================================

# Mapping of model names to tiktoken encoding names
_MODEL_TO_ENCODING: Dict[str, str] = {
    'gpt-4': 'cl100k_base',
    'gpt-4-turbo': 'cl100k_base',
    'gpt-4o': 'o200k_base',
    'gpt-3.5-turbo': 'cl100k_base',
    'text-embedding-ada-002': 'cl100k_base',
    'text-embedding-3-small': 'cl100k_base',
    'text-embedding-3-large': 'cl100k_base',
}


def tokenizer_from_name(
    name: str,
    **opts: Any,
) -> Tokenizer:
    """
    Get a tokenizer by name.
    
    Attempts to load model-specific tokenizer (tiktoken). Falls back to
    FallbackWordCountTokenizer if unavailable, logging a warning.
    
    Args:
        name: Tokenizer or model name. Examples:
            - 'cl100k_base', 'o200k_base' (tiktoken encodings)
            - 'gpt-4', 'gpt-4o' (model names, mapped to encodings)
            - 'fallback' (explicitly request fallback)
        **opts: Additional options:
            - tokens_per_word: float (for fallback tokenizer)
            
    Returns:
        Tokenizer instance. Returns FallbackWordCountTokenizer with warning
        if requested tokenizer is unavailable.
    """
    # Explicit fallback request
    if name == 'fallback':
        return FallbackWordCountTokenizer(
            tokens_per_word=opts.get('tokens_per_word', 1.3)
        )
    
    # Resolve model name to encoding name
    encoding_name = _MODEL_TO_ENCODING.get(name, name)
    
    # Try to load tiktoken
    try:
        import tiktoken
        
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return ModelTokenizer(encoding, encoding_name)
        except ValueError:
            # Unknown encoding name, try as model name
            try:
                encoding = tiktoken.encoding_for_model(name)
                return ModelTokenizer(encoding, name)
            except KeyError:
                logger.warning(
                    f"Unknown tokenizer '{name}'. Falling back to word-count heuristic."
                )
                return FallbackWordCountTokenizer(
                    tokens_per_word=opts.get('tokens_per_word', 1.3)
                )
    except ImportError:
        logger.warning(
            f"tiktoken not installed. Falling back to word-count heuristic for '{name}'. "
            "Install tiktoken for accurate token counting: pip install tiktoken"
        )
        return FallbackWordCountTokenizer(
            tokens_per_word=opts.get('tokens_per_word', 1.3)
        )

