"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

-> Dùng Jina hoặc self host hoặc bất cứ công cụ nào bạn quen
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    ranks: dict[str, dict[str, int]] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
            # Giữ bản đầu tiên (dense đứng trước) làm nội dung/metadata.
            items.setdefault(item_id, item)
            ranks.setdefault(item_id, {})[item.get("retrieval_method", "?")] = rank

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    results = []
    for item_id in ranked_ids[:top_k]:
        result = dict(items[item_id])
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        # Thứ hạng gốc trong từng danh sách, để UI/eval giải thích kết quả fuse.
        result["source_ranks"] = ranks[item_id]
        results.append(result)
    return results


if __name__ == "__main__":
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search

    query = "How is Coherence and Cohesion assessed?"
    for result in rerank_rrf([semantic_search(query, 10), lexical_search(query, 10)]):
        print(f"{result['score']:.4f} {result['source_ranks']} {result['id']}")
