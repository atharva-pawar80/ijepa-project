"""
rag_prototype.py -- hand-built RAG over my own I-JEPA project files.
No LangChain yet. Understand the four steps before wrapping them.
"""
import os
import glob
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, runs locally

def load_and_chunk(folder, chunk_size=500):
    chunks = []
    filepaths = glob.glob(folder + "/*.md")
    for filepath in filepaths:
        with open(filepath, "r") as f:
            text = f.read()
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
    return chunks

def embed_chunks(chunks):
        chunks = load_and_chunk("../../..")  # or wherever you ended up after the multi-folder fix
        print(f"Got {len(chunks)} chunks")

        embeddings = embed_chunks(chunks)
        print(f"Embeddings shape: {embeddings.shape}")

def cosine_sim(a, b):
    dot = 0
    normA = 0
    normB = 0
    dot = np.dot(a, b)
    normA = np.linalg.norm(a)
    normB = np.linalg.norm(b)
    return dot / (normA * normB) if normA > 0 and normB > 0 else 0.0

def retrieve(query, chunks, chunk_embeddings, top_k=3):
    """TODO: embed the query (embedder.encode([query])), compute cosine_sim
    against every chunk_embedding, and return the top_k chunks with the
    highest similarity. Hint: np.argsort() is useful here."""
    pass

if __name__ == "__main__":
    chunks = load_and_chunk("../../../")  # adjust path to your project root
    print(f"Loaded {len(chunks)} chunks")

    chunk_embeddings = embed_chunks(chunks)
    print(f"Embeddings shape: {chunk_embeddings.shape}")

    query = "Why does the target encoder use EMA instead of gradient descent?"
    top_chunks = retrieve(query, chunks, chunk_embeddings)
    for i, c in enumerate(top_chunks):
        print(f"\n--- retrieved chunk {i+1} ---\n{c[:200]}...")