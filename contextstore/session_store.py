"""
SessionStore - Unified retrieval workflow integrating memory and embeddings.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Awaitable, Union

from contextstore.interfaces import MemoryBackend
from contextstore.retrieval import RetrievalBackend, RetrievedItem, retrieve_relevant
from contextstore.embedding import EmbedFn, validate_vectors
from contextstore.models import Interaction


@dataclass
class SessionStoreConfig:
    """Configuration for SessionStore."""
    auto_embed: bool = False
    embed_fn: Optional[EmbedFn] = None


class SessionStore:
    """
    Unified store that integrates memory backend with embedding-based retrieval.
    
    Provides:
    - Standard memory operations (load, save, append)
    - Automatic embedding on message save (when auto_embed=True)
    - Background embedding support for non-blocking saves
    - Semantic search over session history
    """
    
    def __init__(
        self,
        memory: MemoryBackend,
        retrieval: Optional[RetrievalBackend] = None,
        config: Optional[SessionStoreConfig] = None
    ):
        """
        Initialize SessionStore.
        
        Args:
            memory: Backend for storing conversation history.
            retrieval: Optional backend for embedding storage and search.
            config: Optional configuration for auto-embedding.
        """
        self.memory = memory
        self.retrieval = retrieval
        self.config = config or SessionStoreConfig()
        self._background_tasks: List[asyncio.Task] = []
        
        if self.config.auto_embed and not self.config.embed_fn:
            raise ValueError("auto_embed requires embed_fn to be set in config")
        if self.config.auto_embed and not self.retrieval:
            raise ValueError("auto_embed requires a retrieval backend")

    async def load_context(self, session_id: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load context from memory backend."""
        return await self.memory.load_context(session_id, k)

    async def save_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """Save context to memory backend."""
        await self.memory.save_context(session_id, context)
        
        if self.config.auto_embed:
            await self._embed_messages(session_id, context)

    async def append_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """Append context to memory backend."""
        await self.memory.append_context(session_id, context)
        
        if self.config.auto_embed:
            await self._embed_messages(session_id, context)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session from memory backend."""
        await self.memory.delete_session(session_id)

    async def delete_interaction(self, session_id: str, interaction_id: str) -> None:
        """Delete an interaction from memory backend."""
        await self.memory.delete_interaction(session_id, interaction_id)

    async def add_message_embedding(
        self,
        session_id: str,
        message_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Manually add an embedding for a message.
        
        Args:
            session_id: The session ID.
            message_id: The message ID.
            text: The text to embed.
            metadata: Optional metadata to store with the embedding.
        """
        if not self.retrieval:
            raise ValueError("No retrieval backend configured")
        if not self.config.embed_fn:
            raise ValueError("No embed_fn configured")
            
        # Check if already embedded
        if await self.retrieval.has_embedding(session_id, message_id):
            return
            
        # Embed
        if asyncio.iscoroutinefunction(self.config.embed_fn):
            embeddings = await self.config.embed_fn([text])
        else:
            embeddings = self.config.embed_fn([text])
            
        validate_vectors(embeddings)
        await self.retrieval.add(session_id, message_id, embeddings[0], metadata)

    def spawn_background_embedding(
        self,
        session_id: str,
        message_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> asyncio.Task:
        """
        Spawn a background task to embed a message (non-blocking).
        
        Args:
            session_id: The session ID.
            message_id: The message ID.
            text: The text to embed.
            metadata: Optional metadata to store with the embedding.
            
        Returns:
            The asyncio Task for the embedding operation.
        """
        task = asyncio.create_task(
            self.add_message_embedding(session_id, message_id, text, metadata)
        )
        self._background_tasks.append(task)
        # Clean up completed tasks
        self._background_tasks = [t for t in self._background_tasks if not t.done()]
        return task

    async def retrieve_relevant(
        self,
        session_id: str,
        query: str,
        k: int = 5
    ) -> List[RetrievedItem]:
        """
        Retrieve relevant messages for a query using semantic search.
        
        Args:
            session_id: The session to search in.
            query: The text query.
            k: Number of results to return.
            
        Returns:
            List of RetrievedItem with message_id, score, and metadata.
        """
        if not self.retrieval:
            raise ValueError("No retrieval backend configured")
        if not self.config.embed_fn:
            raise ValueError("No embed_fn configured")
            
        return await retrieve_relevant(
            session_id, query, self.config.embed_fn, self.retrieval, k
        )

    async def wait_for_embeddings(self) -> None:
        """Wait for all background embedding tasks to complete."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks = []

    async def _embed_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Embed a list of messages."""
        if not self.retrieval or not self.config.embed_fn:
            return
            
        for msg in messages:
            # Get message ID and text
            msg_id = msg.get('id')
            content = msg.get('content', {})
            
            # Extract text from content
            if isinstance(content, str):
                text = content
            elif isinstance(content, dict):
                text = content.get('text') or content.get('content') or str(content)
            else:
                text = str(content)
                
            if not msg_id or not text:
                continue
                
            # Skip if already embedded
            if await self.retrieval.has_embedding(session_id, msg_id):
                continue
                
            # Embed
            if asyncio.iscoroutinefunction(self.config.embed_fn):
                embeddings = await self.config.embed_fn([text])
            else:
                embeddings = self.config.embed_fn([text])
                
            validate_vectors(embeddings)
            await self.retrieval.add(session_id, msg_id, embeddings[0], {'text': text})

