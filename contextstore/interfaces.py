"""
Abstract interface for memory backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class MemoryBackend(ABC):
    """Abstract base class for conversation history storage backends."""
    
    @abstractmethod
    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load conversation history for a session.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            List of conversation messages, each as a dictionary
        """
        pass
    
    @abstractmethod
    def save_history(self, session_id: str, history: List[Dict[str, Any]]) -> None:
        """
        Save conversation history for a session.
        
        Args:
            session_id: Unique identifier for the conversation session
            history: List of conversation messages to save
        """
        pass

