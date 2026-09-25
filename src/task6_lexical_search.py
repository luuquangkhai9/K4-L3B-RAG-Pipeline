"""
Task 6 — Lexical Search bằng BM25.
"""

import numpy as np
from rank_bm25 import BM25Okapi, BM25Plus
from src.task4_chunking_indexing import load_documents, chunk_documents

try:
    _docs = load_documents()
    CORPUS = chunk_documents(_docs)
except Exception:
    CORPUS = []

def build_bm25_index(corpus: list[dict]):
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    # Dùng BM25Plus để tránh điểm âm khi IDF âm trên tập nhỏ
    return BM25Plus(tokenized_corpus)

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    import src.task6_lexical_search as self_mod
    corpus = getattr(self_mod, "CORPUS", CORPUS)
    if not corpus:
        return []
        
    bm25 = build_bm25_index(corpus)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    # Sắp xếp index theo score giảm dần
    scored_indices = sorted(
        range(len(corpus)),
        key=lambda i: scores[i],
        reverse=True
    )
    
    results = []
    for idx in scored_indices[:top_k]:
        item = corpus[idx]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[idx]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
    return results

if __name__ == "__main__":
    res = lexical_search("tuition fee", top_k=2)
    print(f"BM25 results: {res}")