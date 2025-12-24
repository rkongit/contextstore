"""
Unit tests for memory backends.
"""

import asyncio
import json
import os
import tempfile
import pytest
from typing import Generator

from contextstore.memory_backends import InMemoryMemory, SQLiteMemory
from contextstore.models import Interaction

pytest_plugins = ('pytest_asyncio',)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def in_memory_backend() -> InMemoryMemory:
    """Fixture for in-memory backend."""
    return InMemoryMemory()


@pytest.fixture
def sqlite_backend() -> Generator[SQLiteMemory, None, None]:
    """Fixture for SQLite backend with temp file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    backend = SQLiteMemory(db_path)
    yield backend
    if os.path.exists(db_path):
        os.unlink(db_path)


# ============================================================================
# InMemory Backend Tests
# ============================================================================

class TestInMemoryBackend:
    """Tests for InMemoryMemory backend."""

    @pytest.mark.asyncio
    async def test_save_and_load_context(self, in_memory_backend):
        """Test basic save and load functionality."""
        session_id = "test-session"
        context = [{"content": {"role": "user", "message": "Hello"}}]
        
        await in_memory_backend.save_context(session_id, context)
        loaded = await in_memory_backend.load_context(session_id)
        
        assert len(loaded) == 1
        assert loaded[0]["content"]["message"] == "Hello"
        assert "id" in loaded[0]

    @pytest.mark.asyncio
    async def test_load_nonexistent_session(self, in_memory_backend):
        """Test loading a session that doesn't exist returns empty list."""
        loaded = await in_memory_backend.load_context("nonexistent-session")
        assert loaded == []

    @pytest.mark.asyncio
    async def test_load_with_k_limit(self, in_memory_backend):
        """Test loading with k parameter limits results."""
        session_id = "test-session"
        context = [{"content": {"message": f"msg-{i}"}} for i in range(10)]
        
        await in_memory_backend.save_context(session_id, context)
        loaded = await in_memory_backend.load_context(session_id, k=3)
        
        assert len(loaded) == 3
        assert loaded[0]["content"]["message"] == "msg-7"
        assert loaded[2]["content"]["message"] == "msg-9"

    @pytest.mark.asyncio
    async def test_load_with_k_zero(self, in_memory_backend):
        """Test k=0 returns empty list."""
        session_id = "test-session"
        await in_memory_backend.save_context(session_id, [{"content": {"msg": "test"}}])
        
        loaded = await in_memory_backend.load_context(session_id, k=0)
        assert loaded == []

    @pytest.mark.asyncio
    async def test_append_context(self, in_memory_backend):
        """Test appending context to existing session."""
        session_id = "test-session"
        
        await in_memory_backend.save_context(session_id, [{"content": {"msg": "first"}}])
        await in_memory_backend.append_context(session_id, [{"content": {"msg": "second"}}])
        
        loaded = await in_memory_backend.load_context(session_id)
        assert len(loaded) == 2
        assert loaded[0]["content"]["msg"] == "first"
        assert loaded[1]["content"]["msg"] == "second"

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_session_fails(self, in_memory_backend):
        """Test appending to non-existent session raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            await in_memory_backend.append_context("nonexistent", [{"content": {"msg": "test"}}])

    @pytest.mark.asyncio
    async def test_append_preserves_order(self, in_memory_backend):
        """Test that append preserves message ordering."""
        session_id = "test-session"
        
        await in_memory_backend.save_context(session_id, [{"content": {"idx": 0}}])
        
        for i in range(1, 5):
            await in_memory_backend.append_context(session_id, [{"content": {"idx": i}}])
        
        loaded = await in_memory_backend.load_context(session_id)
        assert len(loaded) == 5
        for i, item in enumerate(loaded):
            assert item["content"]["idx"] == i

    @pytest.mark.asyncio
    async def test_delete_session(self, in_memory_backend):
        """Test deleting a session."""
        session_id = "test-session"
        await in_memory_backend.save_context(session_id, [{"content": {"msg": "test"}}])
        
        await in_memory_backend.delete_session(session_id)
        
        loaded = await in_memory_backend.load_context(session_id)
        assert loaded == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session_fails(self, in_memory_backend):
        """Test deleting non-existent session raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            await in_memory_backend.delete_session("nonexistent")

    @pytest.mark.asyncio
    async def test_delete_interaction(self, in_memory_backend):
        """Test deleting interaction removes it and all after."""
        session_id = "test-session"
        
        interactions = [
            Interaction(id=f"id-{i}", content={"idx": i}) for i in range(5)
        ]
        await in_memory_backend.save_context(session_id, interactions)
        
        await in_memory_backend.delete_interaction(session_id, "id-2")
        
        loaded = await in_memory_backend.load_context(session_id)
        assert len(loaded) == 2
        assert loaded[0]["content"]["idx"] == 0
        assert loaded[1]["content"]["idx"] == 1

    @pytest.mark.asyncio
    async def test_delete_nonexistent_interaction_fails(self, in_memory_backend):
        """Test deleting non-existent interaction raises ValueError."""
        session_id = "test-session"
        await in_memory_backend.save_context(session_id, [{"content": {"msg": "test"}}])
        
        with pytest.raises(ValueError, match="not found"):
            await in_memory_backend.delete_interaction(session_id, "nonexistent-id")

    @pytest.mark.asyncio
    async def test_metadata_preserved(self, in_memory_backend):
        """Test that metadata is preserved through save/load cycle."""
        session_id = "test-session"
        interaction = Interaction(
            content={"msg": "test"},
            metadata={"source": "api", "version": 2}
        )
        
        await in_memory_backend.save_context(session_id, [interaction])
        loaded = await in_memory_backend.load_context(session_id)
        
        assert loaded[0]["metadata"]["source"] == "api"
        assert loaded[0]["metadata"]["version"] == 2

    @pytest.mark.asyncio
    async def test_timestamp_auto_generated(self, in_memory_backend):
        """Test that timestamp is auto-generated if not provided."""
        session_id = "test-session"
        await in_memory_backend.save_context(session_id, [{"content": {"msg": "test"}}])
        
        loaded = await in_memory_backend.load_context(session_id)
        assert "timestamp" in loaded[0]

    @pytest.mark.asyncio
    async def test_id_auto_generated(self, in_memory_backend):
        """Test that ID is auto-generated if not provided."""
        session_id = "test-session"
        await in_memory_backend.save_context(session_id, [{"content": {"msg": "test"}}])
        
        loaded = await in_memory_backend.load_context(session_id)
        assert "id" in loaded[0]
        assert len(loaded[0]["id"]) == 36

    @pytest.mark.asyncio
    async def test_save_non_dict_content(self, in_memory_backend):
        """Test saving non-dict content wraps it properly."""
        session_id = "test-session"
        await in_memory_backend.save_context(session_id, ["raw string"])
        
        loaded = await in_memory_backend.load_context(session_id)
        assert loaded[0]["content"]["data"] == "raw string"

    @pytest.mark.asyncio
    async def test_save_empty_context(self, in_memory_backend):
        """Test saving empty context list."""
        session_id = "test-session"
        await in_memory_backend.save_context(session_id, [])
        
        loaded = await in_memory_backend.load_context(session_id)
        assert loaded == []

    @pytest.mark.asyncio
    async def test_append_empty_context(self, in_memory_backend):
        """Test appending empty context list."""
        session_id = "test-session"
        await in_memory_backend.save_context(session_id, [{"content": {"msg": "first"}}])
        await in_memory_backend.append_context(session_id, [])
        
        loaded = await in_memory_backend.load_context(session_id)
        assert len(loaded) == 1

    @pytest.mark.asyncio
    async def test_message_ordering_preserved(self, in_memory_backend):
        """Test message ordering is strictly preserved after multiple appends."""
        session_id = "test-session"
        
        await in_memory_backend.save_context(session_id, [
            {"content": {"order": 0}},
            {"content": {"order": 1}},
        ])
        
        await in_memory_backend.append_context(session_id, [
            {"content": {"order": 2}},
            {"content": {"order": 3}},
        ])
        
        await in_memory_backend.append_context(session_id, [
            {"content": {"order": 4}},
        ])
        
        loaded = await in_memory_backend.load_context(session_id)
        
        for i, item in enumerate(loaded):
            assert item["content"]["order"] == i

    @pytest.mark.asyncio
    async def test_k_returns_most_recent(self, in_memory_backend):
        """Test that k parameter returns the most recent k messages."""
        session_id = "test-session"
        
        context = [{"content": {"idx": i}} for i in range(100)]
        await in_memory_backend.save_context(session_id, context)
        
        loaded = await in_memory_backend.load_context(session_id, k=5)
        
        assert len(loaded) == 5
        assert [item["content"]["idx"] for item in loaded] == [95, 96, 97, 98, 99]

    @pytest.mark.asyncio
    async def test_save_replaces_existing_context(self, in_memory_backend):
        """Test that save_context replaces existing context entirely."""
        session_id = "test-session"
        
        await in_memory_backend.save_context(session_id, [{"content": {"msg": "first"}}])
        await in_memory_backend.save_context(session_id, [{"content": {"msg": "replaced"}}])
        
        loaded = await in_memory_backend.load_context(session_id)
        assert len(loaded) == 1
        assert loaded[0]["content"]["msg"] == "replaced"

    @pytest.mark.asyncio
    async def test_interaction_object_handling(self, in_memory_backend):
        """Test that Interaction objects are handled correctly."""
        session_id = "test-session"
        interaction = Interaction(
            id="custom-id",
            content={"msg": "test"},
            metadata={"key": "value"}
        )
        
        await in_memory_backend.save_context(session_id, [interaction])
        loaded = await in_memory_backend.load_context(session_id)
        
        assert loaded[0]["id"] == "custom-id"
        assert loaded[0]["content"]["msg"] == "test"
        assert loaded[0]["metadata"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_concurrent_appends(self, in_memory_backend):
        """Test concurrent appends to in-memory backend."""
        session_id = "test-session"
        
        await in_memory_backend.save_context(session_id, [])
        
        async def append_item(idx):
            await in_memory_backend.append_context(session_id, [{"content": {"idx": idx}}])
        
        await asyncio.gather(*[append_item(i) for i in range(10)])
        
        loaded = await in_memory_backend.load_context(session_id)
        assert len(loaded) == 10
        indices = {item["content"]["idx"] for item in loaded}
        assert indices == set(range(10))


# ============================================================================
# SQLite Backend Tests
# ============================================================================

class TestSQLiteBackend:
    """Tests for SQLiteMemory backend."""

    @pytest.mark.asyncio
    async def test_save_and_load_context(self, sqlite_backend):
        """Test basic save and load functionality."""
        session_id = "test-session"
        context = [{"content": {"role": "user", "message": "Hello"}}]
        
        await sqlite_backend.save_context(session_id, context)
        loaded = await sqlite_backend.load_context(session_id)
        
        assert len(loaded) == 1
        assert loaded[0]["content"]["message"] == "Hello"
        assert "id" in loaded[0]

    @pytest.mark.asyncio
    async def test_load_nonexistent_session(self, sqlite_backend):
        """Test loading a session that doesn't exist returns empty list."""
        loaded = await sqlite_backend.load_context("nonexistent-session")
        assert loaded == []

    @pytest.mark.asyncio
    async def test_load_with_k_limit(self, sqlite_backend):
        """Test loading with k parameter limits results."""
        session_id = "test-session"
        context = [{"content": {"message": f"msg-{i}"}} for i in range(10)]
        
        await sqlite_backend.save_context(session_id, context)
        loaded = await sqlite_backend.load_context(session_id, k=3)
        
        assert len(loaded) == 3
        assert loaded[0]["content"]["message"] == "msg-7"
        assert loaded[2]["content"]["message"] == "msg-9"

    @pytest.mark.asyncio
    async def test_append_context(self, sqlite_backend):
        """Test appending context to existing session."""
        session_id = "test-session"
        
        await sqlite_backend.save_context(session_id, [{"content": {"msg": "first"}}])
        await sqlite_backend.append_context(session_id, [{"content": {"msg": "second"}}])
        
        loaded = await sqlite_backend.load_context(session_id)
        assert len(loaded) == 2
        assert loaded[0]["content"]["msg"] == "first"
        assert loaded[1]["content"]["msg"] == "second"

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_session_fails(self, sqlite_backend):
        """Test appending to non-existent session raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            await sqlite_backend.append_context("nonexistent", [{"content": {"msg": "test"}}])

    @pytest.mark.asyncio
    async def test_delete_session(self, sqlite_backend):
        """Test deleting a session."""
        session_id = "test-session"
        await sqlite_backend.save_context(session_id, [{"content": {"msg": "test"}}])
        
        await sqlite_backend.delete_session(session_id)
        
        loaded = await sqlite_backend.load_context(session_id)
        assert loaded == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session_fails(self, sqlite_backend):
        """Test deleting non-existent session raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            await sqlite_backend.delete_session("nonexistent")

    @pytest.mark.asyncio
    async def test_delete_interaction(self, sqlite_backend):
        """Test deleting interaction removes it and all after."""
        session_id = "test-session"
        
        interactions = [
            Interaction(id=f"id-{i}", content={"idx": i}) for i in range(5)
        ]
        await sqlite_backend.save_context(session_id, interactions)
        
        await sqlite_backend.delete_interaction(session_id, "id-2")
        
        loaded = await sqlite_backend.load_context(session_id)
        assert len(loaded) == 2

    @pytest.mark.asyncio
    async def test_delete_nonexistent_interaction_fails(self, sqlite_backend):
        """Test deleting non-existent interaction raises ValueError."""
        session_id = "test-session"
        await sqlite_backend.save_context(session_id, [{"content": {"msg": "test"}}])
        
        with pytest.raises(ValueError, match="not found"):
            await sqlite_backend.delete_interaction(session_id, "nonexistent-id")

    @pytest.mark.asyncio
    async def test_metadata_preserved(self, sqlite_backend):
        """Test that metadata is preserved through save/load cycle."""
        session_id = "test-session"
        interaction = Interaction(
            content={"msg": "test"},
            metadata={"source": "api", "version": 2}
        )
        
        await sqlite_backend.save_context(session_id, [interaction])
        loaded = await sqlite_backend.load_context(session_id)
        
        assert loaded[0]["metadata"]["source"] == "api"
        assert loaded[0]["metadata"]["version"] == 2

    @pytest.mark.asyncio
    async def test_corrupted_json(self):
        """Test SQLite handles corrupted JSON gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            backend = SQLiteMemory(db_path)
            session_id = "test-session"
            
            await backend.save_context(session_id, [{"content": {"msg": "test"}}])
            
            import aiosqlite
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute(
                    "UPDATE tb_context SET context_json = ? WHERE session_id = ?",
                    ("{invalid json", session_id)
                )
                await conn.commit()
            
            loaded = await backend.load_context(session_id)
            assert loaded == []
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_persistence(self):
        """Test SQLite data persists across backend instances."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            backend1 = SQLiteMemory(db_path)
            await backend1.save_context("session1", [{"content": {"msg": "persistent"}}])
            
            backend2 = SQLiteMemory(db_path)
            loaded = await backend2.load_context("session1")
            
            assert len(loaded) == 1
            assert loaded[0]["content"]["msg"] == "persistent"
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self):
        """Test concurrent operations on different sessions in SQLite."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            backend = SQLiteMemory(db_path)
            
            async def create_session(idx):
                session_id = f"session-{idx}"
                await backend.save_context(session_id, [{"content": {"session": idx}}])
                return session_id
            
            session_ids = await asyncio.gather(*[create_session(i) for i in range(5)])
            
            for i, session_id in enumerate(session_ids):
                loaded = await backend.load_context(session_id)
                assert len(loaded) == 1
                assert loaded[0]["content"]["session"] == i
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_sequential_appends(self):
        """Test sequential appends work correctly (avoiding race conditions)."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            backend = SQLiteMemory(db_path)
            session_id = "test-session"
            await backend.save_context(session_id, [{"content": {"initial": True}}])
            
            # Sequential appends (not concurrent due to SQLite limitations)
            for i in range(5):
                await backend.append_context(session_id, [{"content": {"appended": i}}])
            
            loaded = await backend.load_context(session_id)
            assert len(loaded) == 6
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_save_replaces_existing_context(self, sqlite_backend):
        """Test that save_context replaces existing context entirely."""
        session_id = "test-session"
        
        await sqlite_backend.save_context(session_id, [{"content": {"msg": "first"}}])
        await sqlite_backend.save_context(session_id, [{"content": {"msg": "replaced"}}])
        
        loaded = await sqlite_backend.load_context(session_id)
        assert len(loaded) == 1
        assert loaded[0]["content"]["msg"] == "replaced"

    @pytest.mark.asyncio
    async def test_message_ordering_preserved(self, sqlite_backend):
        """Test message ordering is strictly preserved."""
        session_id = "test-session"
        
        await sqlite_backend.save_context(session_id, [
            {"content": {"order": 0}},
            {"content": {"order": 1}},
        ])
        
        await sqlite_backend.append_context(session_id, [
            {"content": {"order": 2}},
        ])
        
        loaded = await sqlite_backend.load_context(session_id)
        
        for i, item in enumerate(loaded):
            assert item["content"]["order"] == i
