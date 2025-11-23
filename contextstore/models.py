"""
Data models for context storage.
"""

import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional


@dataclass
class Interaction:
    """
    Model for storing interaction data with a unique UUID.
    
    Attributes:
        id: Unique identifier (UUID) for the interaction
        content: The interaction content/data
        metadata: Optional additional metadata for the interaction
        timestamp: ISO format timestamp of when the interaction was created
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
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
        # Ensure timestamp is present, generate if missing
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now(timezone.utc).isoformat()
        return cls(**data)

