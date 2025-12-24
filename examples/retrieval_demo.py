import asyncio
import numpy as np
from typing import List
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.faiss_index import FAISSIndex

# Mock embedding function (deterministic random for demo)
def mock_embed_fn(texts: List[str]) -> List[List[float]]:
    """
    Generates deterministic random vectors for demonstration.
    In a real app, this would call OpenAI/Azure/etc.
    """
    np.random.seed(42) # Fixed seed for reproducibility
    dim = 768
    embeddings = []
    for text in texts:
        # Generate a vector based on hash of text to be somewhat consistent
        # This is just for the demo to run without API keys
        val = hash(text) % 1000 / 1000.0
        vec = np.random.rand(dim).astype('float32') + val
        vec = vec / np.linalg.norm(vec) # Normalize
        embeddings.append(vec.tolist())
    return embeddings

async def main():
    print("--- Vector Retrieval Demo ---")
    
    # 1. Setup
    try:
        index = FAISSIndex(dimension=768)
    except ImportError:
        print("FAISS not installed. Skipping demo.")
        return

    # 2. Data: A fake conversation history
    messages = [
        "I forgot my password, how do I reset it?",
        "You can reset your password by clicking 'Forgot Password' on the login page.",
        "What is the weather like today?",
        "It is sunny and 25 degrees.",
        "My account is locked out.",
        "Please contact support to unlock your account."
    ]
    print(f"Indexing {len(messages)} messages...")

    # 3. Embed
    vectors = mock_embed_fn(messages)
    
    # 4. Index
    index.add(vectors)
    print("Indexing complete.")

    # 5. Search
    query = "login help"
    print(f"\nQuery: '{query}'")
    
    query_vec = mock_embed_fn([query])[0]
    distances, indices = index.search(query_vec, k=3)

    print("\nTop 3 Results:")
    for rank, (dist, idx) in enumerate(zip(distances, indices)):
        if idx < len(messages):
            print(f"{rank+1}. [Score: {dist:.4f}] {messages[idx]}")
        else:
            print(f"{rank+1}. [Score: {dist:.4f}] (Index {idx} out of bounds)")

if __name__ == "__main__":
    asyncio.run(main())
