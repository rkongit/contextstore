# ContextStore

A persistence layer that reliably stores and retrieves LLM interaction context, with pluggable backends and both sync/async APIs.

## Features

- **Pluggable Backends**: In-memory and SQLite storage, easily extensible
- **Sync/Async APIs**: Full async support with sync wrappers
- **Token-Aware Context**: Automatic truncation to fit model token limits
- **Session Management**: Organize interactions by session ID
- **Semantic Retrieval**: Embedding-based search over conversation history
- **Unified SessionStore**: Integrated memory + retrieval with auto-embedding
- **Deterministic**: Same inputs always produce identical outputs
- **Type Hints**: Full type annotation support

## Installation

```bash
pip install contextstore
```

## Quick Start

### Async API (Recommended)

```python
import asyncio
from contextstore import SQLiteMemory

async def main():
    memory = SQLiteMemory("chat.db")
    
    # Save context
    await memory.save_context("session-1", [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ])
    
    # Load context
    history = await memory.load_context("session-1")
    print(history)

asyncio.run(main())
```

### In-Memory Storage

```python
from contextstore import InMemoryMemory

memory = InMemoryMemory()
await memory.save_context("session-1", [{"role": "user", "content": "Hello!"}])
history = await memory.load_context("session-1")
```

## Token-Aware Context Building

Automatically truncate context to fit within model token limits:

```python
from contextstore import ContextBuilder

builder = ContextBuilder.from_model('gpt-4')

messages = [
    {'id': '1', 'role': 'user', 'content': 'Hello!', 'timestamp': '2024-01-01T00:00:00Z'},
    {'id': '2', 'role': 'assistant', 'content': 'Hi!', 'timestamp': '2024-01-01T00:01:00Z'},
    # ... many more messages
]

result = builder.build(messages, max_tokens=4000)
print(result.messages)       # Messages that fit
print(result.total_tokens)   # Token count
print(result.approximate)    # True if using fallback tokenizer
```

### Truncation Strategies

```python
# Drop oldest messages first (default)
result = builder.build(messages, max_tokens=4000, strategy='truncate_oldest')

# Keep only recent messages
result = builder.build(messages, max_tokens=4000, strategy='recent_only')

# Summarize oldest messages
result = builder.build(
    messages,
    max_tokens=4000,
    strategy='summarize_oldest',
    strategy_opts={'summarizer': lambda msgs: "Summary: ...", 'chunk_size': 5},
)
```

### Tokenizer Options

```python
from contextstore import tokenizer_from_name

# Model-specific (requires tiktoken)
tokenizer = tokenizer_from_name('gpt-4')

# Explicit fallback (no dependencies)
tokenizer = tokenizer_from_name('fallback')
```

## Full Example with OpenAI

```python
from uuid import uuid4
from datetime import datetime
from openai import OpenAI
from contextstore import ContextBuilder, SQLiteMemory

client = OpenAI()
memory = SQLiteMemory("chat.db")
builder = ContextBuilder.from_model('gpt-4')

async def chat(session_id: str, user_message: str):
    # Load existing context
    history = await memory.load_context(session_id)
    
    # Add new message
    history.append({
        'id': str(uuid4()),
        'role': 'user',
        'content': user_message,
        'timestamp': datetime.now().isoformat(),
    })
    
    # Truncate to fit token limit
    result = builder.build(history, max_tokens=4000)
    
    # Call OpenAI
    response = client.chat.completions.create(
        model='gpt-4',
        messages=[{'role': m['role'], 'content': m['content']} for m in result.messages],
    )
    
    # Save updated context
    assistant_msg = {
        'id': str(uuid4()),
        'role': 'assistant',
        'content': response.choices[0].message.content,
        'timestamp': datetime.now().isoformat(),
    }
    history.append(assistant_msg)
    await memory.save_context(session_id, history)
    
    return assistant_msg['content']
```

## Custom Backends

Extend `MemoryBackend` to create custom storage:

```python
from contextstore import MemoryBackend
from typing import List, Dict, Any, Optional

class RedisBackend(MemoryBackend):
    async def load_context(self, session_id: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        # Your Redis load logic
        pass
    
    async def save_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        # Your Redis save logic
        pass
    
    async def append_context(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        # Your Redis append logic
        pass
    
    async def delete_session(self, session_id: str) -> None:
        # Your Redis delete logic
        pass
    
    async def delete_interaction(self, session_id: str, interaction_id: str) -> None:
        # Your Redis delete interaction logic
        pass
```

## API Reference

### MemoryBackend Methods

| Method | Description |
|--------|-------------|
| `load_context(session_id, k=None)` | Load context (optionally last k interactions) |
| `save_context(session_id, context)` | Save/replace context |
| `append_context(session_id, context)` | Append to existing context |
| `delete_session(session_id)` | Delete entire session |
| `delete_interaction(session_id, interaction_id)` | Delete specific interaction |

### ContextBuilder

```python
ContextBuilder(tokenizer=None, default_strategy='truncate_oldest')
ContextBuilder.from_model(model_name)  # Factory method

builder.build(
    messages,           # List of message dicts
    max_tokens,         # Token budget
    strategy=None,      # See strategies below
    strategy_opts=None, # Strategy-specific options
    pre_filter=None,    # Filter before processing
    post_filter=None,   # Filter after truncation
) -> BuildResult
```

### BuildResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `messages` | `List[Dict]` | Messages within budget |
| `total_tokens` | `int` | Token count |
| `approximate` | `bool` | True if using fallback tokenizer |
| `strategy_used` | `str` | Strategy applied |
| `metadata` | `Dict` | Additional info (dropped_ids, etc.) |

### Truncation Strategies

| Strategy | Description |
|----------|-------------|
| `truncate_oldest` | Drop oldest messages until under budget (default) |
| `recent_only` | Keep only recent messages that fit within budget |
| `summarize_oldest` | Summarize oldest messages via user-provided callback |

### SessionStore

```python
SessionStore(memory, retrieval=None, config=None)

# Methods
store.load_context(session_id, k=None)
store.save_context(session_id, context)
store.append_context(session_id, context)
store.retrieve_relevant(session_id, query, k=5)
store.spawn_background_embedding(session_id, message_id, text, metadata=None)
store.wait_for_embeddings()
```

### RetrievalBackend

| Method | Description |
|--------|-------------|
| `add(session_id, message_id, vector, metadata)` | Add embedding vector |
| `search(session_id, query_vector, k)` | Search for similar vectors |
| `has_embedding(session_id, message_id)` | Check if embedding exists |

## Semantic Retrieval (v0.4.0+)

Retrieve relevant messages from conversation history using embeddings:

```python
from contextstore import InMemoryEmbeddingStore, retrieve_relevant

# Your embedding function (sync or async)
def embed_fn(texts: list[str]) -> list[list[float]]:
    # Use OpenAI, sentence-transformers, etc.
    return [[0.1, 0.2, ...] for _ in texts]

store = InMemoryEmbeddingStore()

# Add embeddings
await store.add("session-1", "msg-1", embed_fn(["Hello!"])[0], {"text": "Hello!"})
await store.add("session-1", "msg-2", embed_fn(["How are you?"])[0], {"text": "How are you?"})

# Search for relevant messages
results = await retrieve_relevant("session-1", "greeting", embed_fn, store, k=5)
for item in results:
    print(f"{item.message_id}: {item.score:.3f} - {item.metadata}")
```

## SessionStore - Unified Workflow

`SessionStore` combines memory storage with embedding-based retrieval:

```python
from contextstore import SessionStore, SessionStoreConfig, SQLiteMemory, InMemoryEmbeddingStore

memory = SQLiteMemory("chat.db")
retrieval = InMemoryEmbeddingStore()

config = SessionStoreConfig(
    auto_embed=True,
    embed_fn=your_embed_function,
)

store = SessionStore(memory, retrieval, config)

# Save context (automatically embeds when auto_embed=True)
await store.save_context("session-1", [
    {"id": "1", "role": "user", "content": "What is Python?"},
    {"id": "2", "role": "assistant", "content": "Python is a programming language."},
])

# Semantic search over history
relevant = await store.retrieve_relevant("session-1", "programming languages", k=3)

# Background embedding (non-blocking)
store.spawn_background_embedding("session-1", "msg-3", "Some text to embed")
await store.wait_for_embeddings()  # Wait for completion
```

## Requirements

- Python 3.8+
- **Optional**: `tiktoken` for accurate token counting
- **Optional**: `numpy` for embedding-based retrieval

## License

MIT License
