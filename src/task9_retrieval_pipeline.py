"""
Task 9 — Retrieval Pipeline kết hợp Dense + Sparse + RRF + Fallback.
"""

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf

DEFAULT_TOP_K = 5
SCORE_THRESHOLD = 0.5

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Fallback search (mặc định nếu có module pageindex)."""
    return []

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    dense = semantic_search(query, top_k=top_k * 2)
    sparse = lexical_search(query, top_k=top_k * 2)
    
    hybrid = (
        rerank_rrf([dense, sparse], top_k=top_k)
        if use_reranking else dense[:top_k]
    )

    best_dense_score = dense[0]["score"] if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception:
            pass
            
    return hybrid[:top_k]

if __name__ == "__main__":
    output = retrieve("assessment criteria", top_k=3)
    print(f"Retrieved {len(output)} items")