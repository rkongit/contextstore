from typing import List, Union, Callable, Awaitable, Optional

# EmbedFn can be synchronous or asynchronous
# Input: List[str] (texts to embed)
# Output: List[List[float]] (list of vectors)
EmbedFn = Union[
    Callable[[List[str]], List[List[float]]],
    Callable[[List[str]], Awaitable[List[List[float]]]]
]

def validate_vectors(vectors: List[List[float]], expected_dim: Optional[int] = None) -> None:
    """
    Validates that the input is a list of numeric vectors with consistent dimensions.
    
    Args:
        vectors: The list of vectors to validate.
        expected_dim: Optional expected dimension to enforce.
        
    Raises:
        ValueError: If validation fails.
    """
    if not isinstance(vectors, list):
        raise ValueError("Output must be a list of vectors.")
    
    if not vectors:
        return

    first_dim = len(vectors[0])
    if expected_dim is not None and first_dim != expected_dim:
        raise ValueError(f"Expected embedding dimension {expected_dim}, got {first_dim}.")

    for i, vec in enumerate(vectors):
        if not isinstance(vec, list):
            raise ValueError(f"Item at index {i} is not a list.")
        
        if len(vec) != first_dim:
            raise ValueError(f"Vector at index {i} has dimension {len(vec)}, expected {first_dim}.")
            
        # Basic type check for floats (or ints that can be floats)
        if not all(isinstance(x, (int, float)) for x in vec):
             raise ValueError(f"Vector at index {i} contains non-numeric values.")

def infer_embedding_dim(vectors: List[List[float]]) -> int:
    """
    Infers the embedding dimension from a list of vectors.
    
    Args:
        vectors: List of vectors.
        
    Returns:
        The dimension of the vectors.
        
    Raises:
        ValueError: If the list is empty or invalid.
    """
    if not vectors:
        raise ValueError("Cannot infer dimension from empty list.")
    
    # We rely on validate_vectors to ensure consistency, but here we just check the first one
    # assuming validation happens elsewhere or we just need a quick check.
    # For safety, let's just look at the first one.
    first_vec = vectors[0]
    if not isinstance(first_vec, list):
         raise ValueError("First item is not a list.")
         
    return len(first_vec)
