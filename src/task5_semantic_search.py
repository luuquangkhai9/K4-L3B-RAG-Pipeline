"""
Task 5 — Semantic Search dùng ChromaDB collection và embed_texts chung.
"""

from src.task4_chunking_indexing import embed_texts, get_collection

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    if response["ids"] and response["ids"][0]:
        for item_id, content, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            results.append({
                "id": item_id,
                "content": content,
                "score": max(0.0, 1.0 - distance),
                "metadata": metadata,
                "retrieval_method": "dense",
            })
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]

if __name__ == "__main__":
    test_res = semantic_search("IELTS Writing criteria", top_k=3)
    print(f"Dense search found {len(test_res)} items")