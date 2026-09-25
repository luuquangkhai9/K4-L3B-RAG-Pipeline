"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    collection = get_collection()
    # Some lightweight collection adapters (including tests) expose query only.
    count_method = getattr(collection, "count", None)
    count = count_method() if callable(count_method) else None
    if count == 0:
        return []

    query_vector = embed_texts([query.strip()])[0]
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count) if count is not None else top_k,
        include=["documents", "metadatas", "distances"],
    )
    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    results = []
    for item_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
        if content is None or distance is None:
            continue
        results.append(
            {
                "id": item_id,
                "content": content,
                "score": max(0.0, min(1.0, 1.0 - float(distance))),
                "metadata": metadata or {},
                "retrieval_method": "dense",
            }
        )
    results.sort(key=lambda item: (-item["score"], item["id"]))
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
