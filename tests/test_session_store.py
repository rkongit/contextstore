"""
Integration tests for SessionStore - Milestone 4: Unified Retrieval Workflow.
"""

import pytest
import asyncio
from contextstore.session_store import SessionStore, SessionStoreConfig
from contextstore.memory_backends import InMemoryMemory
from contextstore.retrieval import InMemoryEmbeddingStore


# Mock embed function for testing
async def mock_embed_fn(texts):
    """Simple mock that creates distinct vectors based on text content."""
    vectors = []
    for text in texts:
        # Create a simple hash-based vector
        h = hash(text) % 1000
        vectors.append([float(h % 10) / 10, float((h // 10) % 10) / 10, float((h // 100) % 10) / 10])
    return vectors


def sync_mock_embed_fn(texts):
    """Sync version of mock embed function."""
    vectors = []
    for text in texts:
        h = hash(text) % 1000
        vectors.append([float(h % 10) / 10, float((h // 10) % 10) / 10, float((h // 100) % 10) / 10])
    return vectors


class TestSessionStoreBasics:
    """Test basic SessionStore functionality."""
    
    @pytest.mark.asyncio
    async def test_init_without_retrieval(self):
        """SessionStore works without retrieval backend."""
        memory = InMemoryMemory()
        store = SessionStore(memory)
        
        await store.save_context("s1", [{"id": "m1", "content": "Hello"}])
        result = await store.load_context("s1")
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_init_auto_embed_requires_embed_fn(self):
        """auto_embed requires embed_fn."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        config = SessionStoreConfig(auto_embed=True)
        
        with pytest.raises(ValueError, match="auto_embed requires embed_fn"):
            SessionStore(memory, retrieval, config)

    @pytest.mark.asyncio
    async def test_init_auto_embed_requires_retrieval(self):
        """auto_embed requires retrieval backend."""
        memory = InMemoryMemory()
        config = SessionStoreConfig(auto_embed=True, embed_fn=mock_embed_fn)
        
        with pytest.raises(ValueError, match="auto_embed requires a retrieval backend"):
            SessionStore(memory, config=config)


class TestAutoEmbed:
    """Test auto-embedding functionality."""
    
    @pytest.mark.asyncio
    async def test_auto_embed_on_save(self):
        """Messages are auto-embedded on save_context."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        config = SessionStoreConfig(auto_embed=True, embed_fn=mock_embed_fn)
        store = SessionStore(memory, retrieval, config)
        
        await store.save_context("s1", [
            {"id": "m1", "content": {"text": "Hello world"}},
            {"id": "m2", "content": {"text": "Goodbye world"}}
        ])
        
        # Check embeddings exist
        assert await retrieval.has_embedding("s1", "m1")
        assert await retrieval.has_embedding("s1", "m2")

    @pytest.mark.asyncio
    async def test_auto_embed_on_append(self):
        """Messages are auto-embedded on append_context."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        config = SessionStoreConfig(auto_embed=True, embed_fn=mock_embed_fn)
        store = SessionStore(memory, retrieval, config)
        
        # First save
        await store.save_context("s1", [{"id": "m1", "content": {"text": "Hello"}}])
        
        # Then append
        await store.append_context("s1", [{"id": "m2", "content": {"text": "World"}}])
        
        assert await retrieval.has_embedding("s1", "m1")
        assert await retrieval.has_embedding("s1", "m2")

    @pytest.mark.asyncio
    async def test_auto_embed_skips_already_embedded(self):
        """Auto-embed skips messages that are already embedded."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        
        embed_count = [0]
        async def counting_embed_fn(texts):
            embed_count[0] += len(texts)
            return await mock_embed_fn(texts)
        
        config = SessionStoreConfig(auto_embed=True, embed_fn=counting_embed_fn)
        store = SessionStore(memory, retrieval, config)
        
        # Save same message twice
        await store.save_context("s1", [{"id": "m1", "content": {"text": "Hello"}}])
        await store.save_context("s1", [{"id": "m1", "content": {"text": "Hello"}}])
        
        # Should only embed once
        assert embed_count[0] == 1


class TestManualEmbedding:
    """Test manual embedding methods."""
    
    @pytest.mark.asyncio
    async def test_add_message_embedding(self):
        """add_message_embedding stores embedding correctly."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        config = SessionStoreConfig(embed_fn=mock_embed_fn)
        store = SessionStore(memory, retrieval, config)
        
        await store.add_message_embedding("s1", "m1", "Hello world", {"role": "user"})
        
        assert await retrieval.has_embedding("s1", "m1")

    @pytest.mark.asyncio
    async def test_add_message_embedding_no_retrieval(self):
        """add_message_embedding fails without retrieval backend."""
        memory = InMemoryMemory()
        store = SessionStore(memory)
        
        with pytest.raises(ValueError, match="No retrieval backend"):
            await store.add_message_embedding("s1", "m1", "Hello")

    @pytest.mark.asyncio
    async def test_add_message_embedding_no_embed_fn(self):
        """add_message_embedding fails without embed_fn."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        store = SessionStore(memory, retrieval)
        
        with pytest.raises(ValueError, match="No embed_fn"):
            await store.add_message_embedding("s1", "m1", "Hello")

    @pytest.mark.asyncio
    async def test_add_message_embedding_sync_embed_fn(self):
        """add_message_embedding works with sync embed_fn."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        config = SessionStoreConfig(embed_fn=sync_mock_embed_fn)
        store = SessionStore(memory, retrieval, config)
        
        await store.add_message_embedding("s1", "m1", "Hello world")
        
        assert await retrieval.has_embedding("s1", "m1")


class TestBackgroundEmbedding:
    """Test background embedding functionality."""
    
    @pytest.mark.asyncio
    async def test_spawn_background_embedding(self):
        """spawn_background_embedding creates a task."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        config = SessionStoreConfig(embed_fn=mock_embed_fn)
        store = SessionStore(memory, retrieval, config)
        
        task = store.spawn_background_embedding("s1", "m1", "Hello world")
        
        assert isinstance(task, asyncio.Task)
        await task
        assert await retrieval.has_embedding("s1", "m1")

    @pytest.mark.asyncio
    async def test_wait_for_embeddings(self):
        """wait_for_embeddings waits for all background tasks."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        config = SessionStoreConfig(embed_fn=mock_embed_fn)
        store = SessionStore(memory, retrieval, config)
        
        # Spawn multiple background tasks
        store.spawn_background_embedding("s1", "m1", "Hello")
        store.spawn_background_embedding("s1", "m2", "World")
        store.spawn_background_embedding("s1", "m3", "Test")
        
        # Wait for all
        await store.wait_for_embeddings()
        
        # All should be embedded
        assert await retrieval.has_embedding("s1", "m1")
        assert await retrieval.has_embedding("s1", "m2")
        assert await retrieval.has_embedding("s1", "m3")


class TestRetrieval:
    """Test retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant(self):
        """retrieve_relevant returns relevant messages."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        config = SessionStoreConfig(embed_fn=mock_embed_fn)
        store = SessionStore(memory, retrieval, config)
        
        # Add some embeddings
        await store.add_message_embedding("s1", "m1", "Hello world")
        await store.add_message_embedding("s1", "m2", "Goodbye world")
        await store.add_message_embedding("s1", "m3", "Hello world")  # Same as m1
        
        # Search
        results = await store.retrieve_relevant("s1", "Hello world", k=2)
        
        assert len(results) == 2
        # m1 and m3 have same text, should be top results
        result_ids = {r.message_id for r in results}
        assert "m1" in result_ids or "m3" in result_ids

    @pytest.mark.asyncio
    async def test_retrieve_relevant_no_retrieval(self):
        """retrieve_relevant fails without retrieval backend."""
        memory = InMemoryMemory()
        store = SessionStore(memory)
        
        with pytest.raises(ValueError, match="No retrieval backend"):
            await store.retrieve_relevant("s1", "Hello")

    @pytest.mark.asyncio
    async def test_retrieve_relevant_no_embed_fn(self):
        """retrieve_relevant fails without embed_fn."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        store = SessionStore(memory, retrieval)
        
        with pytest.raises(ValueError, match="No embed_fn"):
            await store.retrieve_relevant("s1", "Hello")


class TestFullFlow:
    """Integration tests for full workflow."""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test complete conversation with auto-embed and retrieval."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        config = SessionStoreConfig(auto_embed=True, embed_fn=mock_embed_fn)
        store = SessionStore(memory, retrieval, config)
        
        session_id = "conversation1"
        
        # Save initial messages
        await store.save_context(session_id, [
            {"id": "m1", "content": {"text": "What is Python?"}},
            {"id": "m2", "content": {"text": "Python is a programming language."}}
        ])
        
        # Append more messages
        await store.append_context(session_id, [
            {"id": "m3", "content": {"text": "What is JavaScript?"}},
            {"id": "m4", "content": {"text": "JavaScript is a scripting language."}}
        ])
        
        # Load context
        context = await store.load_context(session_id)
        assert len(context) == 4
        
        # Search for relevant messages
        results = await store.retrieve_relevant(session_id, "programming language", k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_background_embed_flow(self):
        """Test non-blocking embedding flow."""
        memory = InMemoryMemory()
        retrieval = InMemoryEmbeddingStore()
        config = SessionStoreConfig(embed_fn=mock_embed_fn)
        store = SessionStore(memory, retrieval, config)
        
        session_id = "session1"
        
        # Save without auto-embed
        await store.save_context(session_id, [
            {"id": "m1", "content": "Message 1"},
            {"id": "m2", "content": "Message 2"}
        ])
        
        # Spawn background embeddings
        store.spawn_background_embedding(session_id, "m1", "Message 1")
        store.spawn_background_embedding(session_id, "m2", "Message 2")
        
        # Do other work here (non-blocking)
        
        # Wait when ready
        await store.wait_for_embeddings()
        
        # Now can search
        results = await store.retrieve_relevant(session_id, "Message 1", k=1)
        assert len(results) == 1

