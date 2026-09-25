"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score (hai thang đo khác nhau, cộng lại là sai). Công thức:

    RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1

Lưu ý: RRF score chỉ phản ánh thứ hạng, KHÔNG dùng để quyết định fallback —
Task 9 so threshold với cosine score gốc của dense search.
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            # Giữ bản xuất hiện đầu tiên; chỉ id và score được dùng để xếp hạng.
            items.setdefault(item_id, item)

    ranked_ids = sorted(scores, key=lambda item_id: scores[item_id], reverse=True)

    results = []
    for item_id in ranked_ids[:top_k]:
        result = dict(items[item_id])
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)
    return results


if __name__ == "__main__":
    print("RRF chỉ có ý nghĩa khi fuse kết quả thật; xem Task 9.")