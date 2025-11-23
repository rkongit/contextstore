"""
Memory backend implementations.
"""

import json
import sqlite3
import logging
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .interfaces import MemoryBackend

logger = logging.getLogger(__name__)


@dataclass
class Interaction:
    """
    Model for storing interaction data with a unique UUID.
    
    Attributes:
        id: Unique identifier (UUID) for the interaction
        content: The interaction content/data
        metadata: Optional additional metadata for the interaction
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert interaction to dictionary."""
        result = asdict(self)
        # Remove None values for cleaner JSON
        if result.get('metadata') is None:
            result.pop('metadata', None)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Interaction':
        """Create interaction from dictionary."""
        # Ensure UUID is present, generate if missing
        if 'id' not in data:
            data['id'] = str(uuid.uuid4())
        return cls(**data)


class InMemoryMemory(MemoryBackend):
    """In-memory memory backend implementation."""
    
    def __init__(self):
        """Initialize in-memory storage."""
        self._store: Dict[str, List[Dict[str, Any]]] = {}
    
    def load_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Load context for a session."""
        return self._store.get(session_id, [])
    
    def save_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """
        Save context for a session.
        
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
    
    def load_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Load context for a session."""
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
                return json.loads(row[0])
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing history JSON: {e}")
                return []
        return []
    
    def save_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """
        Save context for a session.
        
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
            # Update existing session
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

