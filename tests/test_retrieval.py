import pytest
import numpy as np
from contextstore.retrieval import InMemoryEmbeddingStore, retrieve_relevant, RetrievedItem

class TestRetrieval:
    @pytest.mark.asyncio
    async def test_store_add_and_search(self):
        store = InMemoryEmbeddingStore()
        session_id = "sess1"
        
        # Add vectors: [1, 0] and [0, 1]
        await store.add(session_id, "msg1", [1.0, 0.0], {"text": "A"})
        await store.add(session_id, "msg2", [0.0, 1.0], {"text": "B"})
        
        # Search for [1, 0] -> should match msg1 perfectly
        results = await store.search(session_id, [1.0, 0.0], k=2)
        
        assert len(results) == 2
        assert results[0].message_id == "msg1"
        assert results[0].score > 0.99
        assert results[1].message_id == "msg2"
        assert results[1].score < 0.1 # Orthogonal

    @pytest.mark.asyncio
    async def test_retrieve_relevant_integration(self):
        store = InMemoryEmbeddingStore()
        session_id = "sess1"
        
        # Mock embed_fn: simple mapping
        # "A" -> [1, 0], "B" -> [0, 1]
        async def mock_embed(texts):
            vecs = []
            for t in texts:
                if t == "A": vecs.append([1.0, 0.0])
                elif t == "B": vecs.append([0.0, 1.0])
                else: vecs.append([0.0, 0.0])
            return vecs
            
        await store.add(session_id, "m1", [1.0, 0.0]) # "A"
        await store.add(session_id, "m2", [0.0, 1.0]) # "B"
        
        # Query "A"
        results = await retrieve_relevant(session_id, "A", mock_embed, store, k=1)
        assert results[0].message_id == "m1"
        
    @pytest.mark.asyncio
    async def test_has_embedding(self):
        store = InMemoryEmbeddingStore()
        await store.add("s1", "m1", [1.0])
        
        assert await store.has_embedding("s1", "m1")
        assert not await store.has_embedding("s1", "m2")
        assert not await store.has_embedding("s2", "m1")

    @pytest.mark.asyncio
    async def test_empty_search(self):
        store = InMemoryEmbeddingStore()
        results = await store.search("empty", [1.0], k=5)
        assert results == []
