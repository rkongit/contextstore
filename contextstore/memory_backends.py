"""
Memory backend implementations.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import aiosqlite

from .interfaces import MemoryBackend
from .models import Interaction

logger = logging.getLogger(__name__)


class InMemoryMemory(MemoryBackend):
    """In-memory memory backend implementation."""
    
    def __init__(self):
        """Initialize in-memory storage."""
        self._store: Dict[str, List[Dict[str, Any]]] = {}
    
    async def load_context(self, session_id: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
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
    
    async def save_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
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
    
    async def append_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """
        Append new context to an existing session.
        
        Args:
            session_id: Unique identifier for the session
            context: List of interaction dictionaries to append. Each interaction will be converted
                    to an Interaction model and assigned a unique UUID if not present.
        
        Raises:
            ValueError: If the session does not exist
        """
        # Check if session exists (distinguish from empty session)
        if session_id not in self._store:
            raise ValueError(f"Session '{session_id}' does not exist. Use save_context to create a new session.")
        
        # Load existing context
        existing_context = await self.load_context(session_id)
        
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
    
    async def delete_session(self, session_id: str) -> None:
        """
        Delete a full session.
        
        Args:
            session_id: Unique identifier for the session
        
        Raises:
            ValueError: If the session does not exist
        """
        if session_id not in self._store:
            raise ValueError(f"Session '{session_id}' does not exist.")
        del self._store[session_id]
    
    async def delete_interaction(self, session_id: str, interaction_id: str) -> None:
        """
        Delete a particular interaction and all interactions after it.
        
        Args:
            session_id: Unique identifier for the session
            interaction_id: Unique identifier for the interaction to delete
        
        Raises:
            ValueError: If the session or interaction does not exist
        """
        if session_id not in self._store:
            raise ValueError(f"Session '{session_id}' does not exist.")
        
        context = await self.load_context(session_id)
        
        # Find the index of the interaction to delete
        interaction_index = None
        for i, interaction in enumerate(context):
            if interaction.get('id') == interaction_id:
                interaction_index = i
                break
        
        if interaction_index is None:
            raise ValueError(f"Interaction '{interaction_id}' not found in session '{session_id}'.")
        
        # Keep only interactions before the found index
        self._store[session_id] = context[:interaction_index]


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
        self._create_db = create_db
    
    async def _init_db(self):
        """Initialize database and create conversation_history table."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS tb_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                context_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            await conn.commit()
    
    async def load_context(self, session_id: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load context for a session.
        
        Args:
            session_id: Unique identifier for the session
            k: Optional. The number of recent interactions to return. If None, returns all.
        """
        # Initialize DB on first use if needed
        if self._create_db:
            await self._init_db()
            self._create_db = False
        
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                "SELECT context_json FROM tb_context WHERE session_id = ?", 
                (session_id,)
            )
            row = await cursor.fetchone()
            
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
    
    async def save_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """
        Save context for a new session. Creates a new session and replaces any existing context.
        
        Args:
            session_id: Unique identifier for the session
            context: List of interaction dictionaries. Each interaction will be converted
                    to an Interaction model and assigned a unique UUID if not present.
        """
        # Initialize DB on first use if needed
        if self._create_db:
            await self._init_db()
            self._create_db = False
        
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
        now = datetime.now(timezone.utc).isoformat()
        
        async with aiosqlite.connect(self.db_path) as conn:
            # Check if session already exists
            cursor = await conn.execute("SELECT id FROM tb_context WHERE session_id = ?", (session_id,))
            existing = await cursor.fetchone()
            
            if existing:
                # Update existing session (replace context)
                await conn.execute(
                    "UPDATE tb_context SET context_json = ?, updated_at = ? WHERE session_id = ?",
                    (json_data, now, session_id)
                )
            else:
                # Insert new session
                await conn.execute(
                    "INSERT INTO tb_context (session_id, context_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, json_data, now, now)
                )
            
            await conn.commit()
    
    async def append_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """
        Append new context to an existing session.
        
        Args:
            session_id: Unique identifier for the session
            context: List of interaction dictionaries to append. Each interaction will be converted
                    to an Interaction model and assigned a unique UUID if not present.
        
        Raises:
            ValueError: If the session does not exist
        """
        # Initialize DB on first use if needed
        if self._create_db:
            await self._init_db()
            self._create_db = False
        
        async with aiosqlite.connect(self.db_path) as conn:
            # Check if session exists (distinguish from empty session)
            cursor = await conn.execute("SELECT id FROM tb_context WHERE session_id = ?", (session_id,))
            existing = await cursor.fetchone()
            
            if not existing:
                raise ValueError(f"Session '{session_id}' does not exist. Use save_context to create a new session.")
        
        # Load existing context
        existing_context = await self.load_context(session_id)
        
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
        
        now = datetime.now(timezone.utc).isoformat()
        
        async with aiosqlite.connect(self.db_path) as conn:
            # Update existing session with combined context
            await conn.execute(
                "UPDATE tb_context SET context_json = ?, updated_at = ? WHERE session_id = ?",
                (json_data, now, session_id)
            )
            await conn.commit()
    
    async def delete_session(self, session_id: str) -> None:
        """
        Delete a full session.
        
        Args:
            session_id: Unique identifier for the session
        
        Raises:
            ValueError: If the session does not exist
        """
        # Initialize DB on first use if needed
        if self._create_db:
            await self._init_db()
            self._create_db = False
        
        async with aiosqlite.connect(self.db_path) as conn:
            # Check if session exists
            cursor = await conn.execute("SELECT id FROM tb_context WHERE session_id = ?", (session_id,))
            existing = await cursor.fetchone()
            
            if not existing:
                raise ValueError(f"Session '{session_id}' does not exist.")
            
            # Delete the session
            await conn.execute("DELETE FROM tb_context WHERE session_id = ?", (session_id,))
            await conn.commit()
    
    async def delete_interaction(self, session_id: str, interaction_id: str) -> None:
        """
        Delete a particular interaction and all interactions after it.
        
        Args:
            session_id: Unique identifier for the session
            interaction_id: Unique identifier for the interaction to delete
        
        Raises:
            ValueError: If the session or interaction does not exist
        """
        # Initialize DB on first use if needed
        if self._create_db:
            await self._init_db()
            self._create_db = False
        
        async with aiosqlite.connect(self.db_path) as conn:
            # Check if session exists
            cursor = await conn.execute("SELECT id FROM tb_context WHERE session_id = ?", (session_id,))
            existing = await cursor.fetchone()
            
            if not existing:
                raise ValueError(f"Session '{session_id}' does not exist.")
        
        # Load existing context
        context = await self.load_context(session_id)
        
        # Find the index of the interaction to delete
        interaction_index = None
        for i, interaction in enumerate(context):
            if interaction.get('id') == interaction_id:
                interaction_index = i
                break
        
        if interaction_index is None:
            raise ValueError(f"Interaction '{interaction_id}' not found in session '{session_id}'.")
        
        # Keep only interactions before the found index
        remaining_context = context[:interaction_index]
        json_data = json.dumps(remaining_context, indent=2)
        
        now = datetime.now(timezone.utc).isoformat()
        
        async with aiosqlite.connect(self.db_path) as conn:
            # Update the session with remaining context
            await conn.execute(
                "UPDATE tb_context SET context_json = ?, updated_at = ? WHERE session_id = ?",
                (json_data, now, session_id)
            )
            await conn.commit()

