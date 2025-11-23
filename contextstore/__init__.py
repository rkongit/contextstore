"""
ContextStore - A generic package for persisting conversation history.

This package provides backends for storing and retrieving conversation history
across different storage mechanisms.
"""

from .interfaces import MemoryBackend
from .memory_backends import InMemoryMemory, SQLiteMemory, Interaction

__version__ = "0.1.0"
__all__ = ["MemoryBackend", "InMemoryMemory", "SQLiteMemory", "Interaction"]

