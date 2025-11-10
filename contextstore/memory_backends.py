"""
Memory backend implementations.
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from .interfaces import MemoryBackend

logger = logging.getLogger(__name__)


class InMemoryMemory(MemoryBackend):
    """In-memory memory backend implementation."""
    
    def __init__(self):
        """Initialize in-memory storage."""
        self._store: Dict[str, List[Dict[str, Any]]] = {}
    
    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Load conversation history for a session."""
        return self._store.get(session_id, [])
    
    def save_history(self, session_id: str, history: List[Dict[str, Any]]) -> None:
        """Save conversation history for a session."""
        self._store[session_id] = history


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
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            history_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        conn.close()
    
    def _connect(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Load conversation history for a session."""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT history_json FROM conversation_history WHERE session_id = ?", 
            (session_id,)
        )
        row = cursor.fetchone()
        
        conn.close()
        
        if row and row[0]:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing history JSON: {e}")
                return []
        return []
    
    def save_history(self, session_id: str, history: List[Dict[str, Any]]) -> None:
        """Save conversation history for a session."""
        json_data = json.dumps(history, indent=2)
        conn = self._connect()
        cursor = conn.cursor()
        
        # Check if session already exists
        cursor.execute("SELECT id FROM conversation_history WHERE session_id = ?", (session_id,))
        existing = cursor.fetchone()
        
        now = datetime.now(timezone.utc).isoformat()
        
        if existing:
            # Update existing session
            cursor.execute(
                "UPDATE conversation_history SET history_json = ?, updated_at = ? WHERE session_id = ?",
                (json_data, now, session_id)
            )
        else:
            # Insert new session
            cursor.execute(
                "INSERT INTO conversation_history (session_id, history_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, json_data, now, now)
            )
        
        conn.commit()
        conn.close()

