import asyncio
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Protocol, Union
from contextstore.embedding import EmbedFn, validate_vectors

@dataclass
class RetrievedItem:
    """A retrieved item from the embedding store."""
    message_id: str
    score: float
    metadata: Dict[str, Any]

class RetrievalBackend(Protocol):
    """Protocol for embedding storage backends."""
    async def add(self, session_id: str, message_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        ...
    
    async def search(self, session_id: str, query_vector: List[float], k: int) -> List[RetrievedItem]:
        ...
        
    async def has_embedding(self, session_id: str, message_id: str) -> bool:
        ...

class InMemoryEmbeddingStore:
    """
    In-memory implementation of RetrievalBackend using numpy.
    Stores vectors in a dict: {session_id: {message_id: vector}}
    """
    def __init__(self):
        # {session_id: {message_id: vector}}
        self._vectors: Dict[str, Dict[str, np.ndarray]] = {}
        # {session_id: {message_id: metadata}}
        self._metadata: Dict[str, Dict[str, Dict[str, Any]]] = {}

    async def add(self, session_id: str, message_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        if session_id not in self._vectors:
            self._vectors[session_id] = {}
            self._metadata[session_id] = {}
            
        self._vectors[session_id][message_id] = np.array(vector, dtype=np.float32)
        if metadata:
            self._metadata[session_id][message_id] = metadata

    async def has_embedding(self, session_id: str, message_id: str) -> bool:
        return session_id in self._vectors and message_id in self._vectors[session_id]

    async def search(self, session_id: str, query_vector: List[float], k: int) -> List[RetrievedItem]:
        if session_id not in self._vectors or not self._vectors[session_id]:
            return []

        # Prepare data for vectorized calculation
        message_ids = list(self._vectors[session_id].keys())
        vectors = list(self._vectors[session_id].values())
        matrix = np.stack(vectors) # (N, D)
        
        q = np.array(query_vector, dtype=np.float32) # (D,)
        
        # Cosine similarity: (A . B) / (|A| * |B|)
        # Normalize matrix rows
        norm_matrix = np.linalg.norm(matrix, axis=1, keepdims=True)
        norm_matrix[norm_matrix == 0] = 1e-10 # Avoid division by zero
        normalized_matrix = matrix / norm_matrix
        
        # Normalize query
        norm_q = np.linalg.norm(q)
        if norm_q == 0:
             return [] # Query is zero vector
        normalized_q = q / norm_q
        
        # Dot product
        scores = np.dot(normalized_matrix, normalized_q) # (N,)
        
        # Get top k
        # argsort gives indices of sorted elements (ascending)
        # We want descending, so we take from end
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            mid = message_ids[idx]
            score = float(scores[idx])
            meta = self._metadata[session_id].get(mid, {})
            results.append(RetrievedItem(message_id=mid, score=score, metadata=meta))
            
        return results

async def retrieve_relevant(
    session_id: str,
    query: str,
    embed_fn: EmbedFn,
    store: RetrievalBackend,
    k: int = 5
) -> List[RetrievedItem]:
    """
    Retrieve relevant messages for a query.
    
    Args:
        session_id: The session to search in.
        query: The text query.
        embed_fn: Function to embed the query.
        store: The embedding store to search.
        k: Number of results to return.
        
    Returns:
        List of RetrievedItem.
    """
    # Embed query
    if asyncio.iscoroutinefunction(embed_fn):
        embeddings = await embed_fn([query])
    else:
        embeddings = embed_fn([query])
        
    validate_vectors(embeddings)
    query_vector = embeddings[0]
    
    # Search
    return await store.search(session_id, query_vector, k)
