# ContextStore

A generic Python package for persisting conversation history across different storage backends. ContextStore provides a simple, extensible interface for saving and loading conversation histories, making it easy to maintain context across sessions in chat applications, LLM integrations, and conversational AI systems.

## Features

- **Multiple Backend Support**: Choose from in-memory or SQLite storage backends
- **Simple API**: Easy-to-use interface for saving and loading conversation history
- **Extensible**: Implement custom backends by extending the `MemoryBackend` abstract class
- **Session Management**: Organize conversations by session ID
- **Type Hints**: Full type annotation support for better IDE integration

## Installation

```bash
pip install contextstore
```


## Quick Start

### Using In-Memory Storage

```python
from contextstore import InMemoryMemory

# Create an in-memory backend
memory = InMemoryMemory()

# Save conversation history
session_id = "user-123"
history = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help you?"}
]
memory.save_history(session_id, history)

# Load conversation history
loaded_history = memory.load_history(session_id)
print(loaded_history)
```

### Using SQLite Storage

```python
from contextstore import SQLiteMemory

# Create a SQLite backend (automatically creates table)
memory = SQLiteMemory("conversations.db")

# Or use an existing database without creating the table
# memory = SQLiteMemory("existing.db", create_db=False)

# Save conversation history
session_id = "user-123"
history = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help you?"}
]
memory.save_history(session_id, history)

# Load conversation history (persists across sessions)
loaded_history = memory.load_history(session_id)
print(loaded_history)
```

## Usage Examples

### Basic Conversation Management

```python
from contextstore import SQLiteMemory

# Initialize the backend
memory = SQLiteMemory("chat_history.db")

# Start a new conversation
session_id = "session-001"
conversation = []

# Add messages to the conversation
conversation.append({"role": "user", "content": "What is Python?"})
conversation.append({"role": "assistant", "content": "Python is a programming language."})

# Save the conversation
memory.save_history(session_id, conversation)

# Later, retrieve the conversation
retrieved = memory.load_history(session_id)
print(retrieved)
```

### Integrating with Chat Applications

```python
from contextstore import SQLiteMemory

class ChatBot:
    def __init__(self, db_path: str):
        self.memory = SQLiteMemory(db_path)
    
    def chat(self, session_id: str, user_message: str):
        # Load existing history
        history = self.memory.load_history(session_id)
        
        # Add user message
        history.append({"role": "user", "content": user_message})
        
        # Generate response (your LLM logic here)
        response = self.generate_response(history)
        
        # Add assistant response
        history.append({"role": "assistant", "content": response})
        
        # Save updated history
        self.memory.save_history(session_id, history)
        
        return response
    
    def generate_response(self, history):
        # Your LLM integration here
        return "This is a placeholder response"
```

## Creating Custom Backends

You can create custom storage backends by extending the `MemoryBackend` abstract class:

```python
from contextstore import MemoryBackend
from typing import List, Dict, Any

class CustomBackend(MemoryBackend):
    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        # Your custom load logic
        pass
    
    def save_history(self, session_id: str, history: List[Dict[str, Any]]) -> None:
        # Your custom save logic
        pass
```

## API Reference

### MemoryBackend

Abstract base class for all memory backends.

#### Methods

- `load_history(session_id: str) -> List[Dict[str, Any]]`
  - Load conversation history for a given session ID
  - Returns an empty list if no history exists

- `save_history(session_id: str, history: List[Dict[str, Any]]) -> None`
  - Save conversation history for a given session ID
  - Overwrites existing history for the same session ID

### InMemoryMemory

In-memory storage backend. Data is lost when the process ends.

#### Constructor

```python
InMemoryMemory()
```

### SQLiteMemory

SQLite-based persistent storage backend.

#### Constructor

```python
SQLiteMemory(db_path: str, create_db: bool = True)
```

**Parameters:**
- `db_path`: Path to the SQLite database file (will be created if it doesn't exist)
- `create_db`: If `True` (default), automatically create the database table if it doesn't exist. If `False`, skip table creation (assumes table already exists).

## Requirements

- Python 3.8 or higher
- No external dependencies (uses only standard library)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

