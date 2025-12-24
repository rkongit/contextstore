import numpy as np
import os
from typing import List, Tuple, Optional

try:
    import faiss
except ImportError:
    faiss = None

class FAISSIndex:
    """
    A simple wrapper around FAISS for demonstration purposes.
    Requires 'faiss-cpu' or 'faiss-gpu' to be installed.
    """
    def __init__(self, dimension: int, index_path: Optional[str] = None):
        if faiss is None:
            raise ImportError("FAISS is not installed. Please install 'faiss-cpu' or 'faiss-gpu'.")
        
        self.dimension = dimension
        self.index_path = index_path
        
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(dimension)

    def add(self, vectors: List[List[float]]) -> None:
        """Add vectors to the index."""
        if not vectors:
            return
        
        np_vectors = np.array(vectors).astype('float32')
        self.index.add(np_vectors)

    def search(self, query_vector: List[float], k: int = 5) -> Tuple[List[float], List[int]]:
        """
        Search for nearest neighbors.
        Returns (distances, indices).
        """
        np_query = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(np_query, k)
        return distances[0].tolist(), indices[0].tolist()

    def save(self, path: Optional[str] = None) -> None:
        """Save the index to disk."""
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No path specified for saving index.")
        faiss.write_index(self.index, save_path)

def build_faiss_index(dimension: int) -> FAISSIndex:
    """Factory function to create a new FAISS index."""
    return FAISSIndex(dimension)
