"""
Data models for context storage.
"""

import copy
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
        """Create interaction from dictionary.
        
        Extra fields (like 'role') are preserved in the 'content' dict.
        """
        # Create a deep copy to avoid mutating the input dictionary and nested structures
        data_copy = copy.deepcopy(data)
        
        # Ensure UUID is present, generate if missing
        if 'id' not in data_copy:
            data_copy['id'] = str(uuid.uuid4())
        # Ensure timestamp is present, generate if missing
        if 'timestamp' not in data_copy:
            data_copy['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Separate valid fields from extra fields
        valid_fields = {'id', 'content', 'metadata', 'timestamp'}
        extra_fields = {k: v for k, v in data_copy.items() if k not in valid_fields}
        
        # Merge extra fields into content to preserve them
        content = data_copy.get('content', {})
        if not isinstance(content, dict):
            content = {'data': content}
        for key, value in extra_fields.items():
            if key not in content:
                content[key] = value
        data_copy['content'] = content
        
        # Filter to only include valid Interaction fields
        filtered = {k: v for k, v in data_copy.items() if k in valid_fields}
        return cls(**filtered)

