"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(score_threshold, (float, int)):
        raise ValueError("score_threshold must be numeric")

    dense = semantic_search(query.strip(), top_k=top_k * 2)
    if use_reranking:
        sparse = lexical_search(query.strip(), top_k=top_k * 2)
        hybrid = rerank_rrf([dense, sparse], top_k=top_k)
    else:
        hybrid = dense[:top_k]

    # A/B evaluation can set a negative threshold to disable fallback and isolate retrieval.
    best_dense_score = max((float(item["score"]) for item in dense), default=0.0)
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query.strip(), top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception:
            # An unavailable external service must not take down the local search path.
            pass
    return hybrid[:top_k]


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)
