"""
Memory backend implementations.
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .interfaces import MemoryBackend
from .models import Interaction

logger = logging.getLogger(__name__)


class InMemoryMemory(MemoryBackend):
    """In-memory memory backend implementation."""
    
    def __init__(self):
        """Initialize in-memory storage."""
        self._store: Dict[str, List[Dict[str, Any]]] = {}
    
    def load_context(self, session_id: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load context for a session.
        
        Args:
            session_id: Unique identifier for the session
            k: Optional. The number of recent interactions to return. If None, returns all.
        """
        context = self._store.get(session_id, [])
        if k is not None:
            return context[-k:] if k > 0 else []
        return context
    
    def save_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """
        Save context for a new session. Creates a new session and replaces any existing context.
        
        Args:
            session_id: Unique identifier for the session
            context: List of interaction dictionaries. Each interaction will be converted
                    to an Interaction model and assigned a unique UUID if not present.
        """
        # Convert context items to Interaction models, ensuring each has a UUID
        interactions = []
        for item in context:
            if isinstance(item, Interaction):
                interaction = item
            elif isinstance(item, dict):
                interaction = Interaction.from_dict(item)
            else:
                # If it's not a dict or Interaction, wrap it in content
                interaction = Interaction(content={'data': item})
            interactions.append(interaction.to_dict())
        
        self._store[session_id] = interactions
    
    def append_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """
        Append new context to an existing session.
        
        Args:
            session_id: Unique identifier for the session
            context: List of interaction dictionaries to append. Each interaction will be converted
                    to an Interaction model and assigned a unique UUID if not present.
        
        Raises:
            ValueError: If the session does not exist
        """
        # Load existing context
        existing_context = self.load_context(session_id)
        
        if not existing_context:
            raise ValueError(f"Session '{session_id}' does not exist. Use save_context to create a new session.")
        
        # Convert new context items to Interaction models, ensuring each has a UUID
        new_interactions = []
        for item in context:
            if isinstance(item, Interaction):
                interaction = item
            elif isinstance(item, dict):
                interaction = Interaction.from_dict(item)
            else:
                # If it's not a dict or Interaction, wrap it in content
                interaction = Interaction(content={'data': item})
            new_interactions.append(interaction.to_dict())
        
        # Append new interactions to existing context
        self._store[session_id] = existing_context + new_interactions


class SQLiteMemory(MemoryBackend):
    """SQLite-based persistent memory backend implementation."""
    
    def __init__(self, db_path: str, create_db: bool = True):
        """
        Initialize SQLite memory backend.
        
        Args:
            db_path: Path to SQLite database file
            create_db: If True, automatically create the database table if it doesn't exist.
                      If False, skip table creation (assumes table already exists).
        """
        self.db_path = db_path
        if create_db:
            self._init_db()
    
    def _init_db(self):
        """Initialize database and create conversation_history table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conversation_history table with exact schema from requirements
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tb_context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            context_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        conn.close()
    
    def _connect(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def load_context(self, session_id: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load context for a session.
        
        Args:
            session_id: Unique identifier for the session
            k: Optional. The number of recent interactions to return. If None, returns all.
        """
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT context_json FROM tb_context WHERE session_id = ?", 
            (session_id,)
        )
        row = cursor.fetchone()
        
        conn.close()
        
        if row and row[0]:
            try:
                context = json.loads(row[0])
                if k is not None:
                    return context[-k:] if k > 0 else []
                return context
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing history JSON: {e}")
                return []
        return []
    
    def save_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """
        Save context for a new session. Creates a new session and replaces any existing context.
        
        Args:
            session_id: Unique identifier for the session
            context: List of interaction dictionaries. Each interaction will be converted
                    to an Interaction model and assigned a unique UUID if not present.
        """
        # Convert context items to Interaction models, ensuring each has a UUID
        interactions = []
        for item in context:
            if isinstance(item, Interaction):
                interaction = item
            elif isinstance(item, dict):
                interaction = Interaction.from_dict(item)
            else:
                # If it's not a dict or Interaction, wrap it in content
                interaction = Interaction(content={'data': item})
            interactions.append(interaction.to_dict())
        
        json_data = json.dumps(interactions, indent=2)
        conn = self._connect()
        cursor = conn.cursor()
        
        # Check if session already exists
        cursor.execute("SELECT id FROM tb_context WHERE session_id = ?", (session_id,))
        existing = cursor.fetchone()
        
        now = datetime.now(timezone.utc).isoformat()
        
        if existing:
            # Update existing session (replace context)
            cursor.execute(
                "UPDATE tb_context SET context_json = ?, updated_at = ? WHERE session_id = ?",
                (json_data, now, session_id)
            )
        else:
            # Insert new session
            cursor.execute(
                "INSERT INTO tb_context (session_id, context_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, json_data, now, now)
            )
        
        conn.commit()
        conn.close()
    
    def append_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """
        Append new context to an existing session.
        
        Args:
            session_id: Unique identifier for the session
            context: List of interaction dictionaries to append. Each interaction will be converted
                    to an Interaction model and assigned a unique UUID if not present.
        
        Raises:
            ValueError: If the session does not exist
        """
        # Load existing context
        existing_context = self.load_context(session_id)
        
        if not existing_context:
            raise ValueError(f"Session '{session_id}' does not exist. Use save_context to create a new session.")
        
        # Convert new context items to Interaction models, ensuring each has a UUID
        new_interactions = []
        for item in context:
            if isinstance(item, Interaction):
                interaction = item
            elif isinstance(item, dict):
                interaction = Interaction.from_dict(item)
            else:
                # If it's not a dict or Interaction, wrap it in content
                interaction = Interaction(content={'data': item})
            new_interactions.append(interaction.to_dict())
        
        # Append new interactions to existing context
        combined_context = existing_context + new_interactions
        json_data = json.dumps(combined_context, indent=2)
        
        conn = self._connect()
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        
        # Update existing session with combined context
        cursor.execute(
            "UPDATE tb_context SET context_json = ?, updated_at = ? WHERE session_id = ?",
            (json_data, now, session_id)
        )
        
        conn.commit()
        conn.close()

