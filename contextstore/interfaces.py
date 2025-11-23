"""
Abstract interface for memory backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class MemoryBackend(ABC):
    """Abstract base class for conversation history storage backends."""
    
    @abstractmethod
    def load_context(self, session_id: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load context for a session.
        
        Args:
            session_id: Unique identifier for the conversation session
            k: Optional. The number of recent interactions to return. If None, returns all.
            
        Returns:
            List of conversation messages, each as a dictionary
        """
        pass
    
    @abstractmethod
    def save_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """
        Save context for a new session. Creates a new session and replaces any existing context.
        
        Args:
            session_id: Unique identifier for the conversation session
            context: List of conversation messages to save
        """
        pass
    
    @abstractmethod
    def append_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """
        Append new context to an existing session.
        
        Args:
            session_id: Unique identifier for the conversation session
            context: List of conversation messages to append
        
        Raises:
            ValueError: If the session does not exist
        """
        pass

