"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import re


# Để trống thì tự nạp từ ChromaDB (cùng chunks với dense search).
CORPUS: list[dict] = []

_TOKEN = re.compile(r"\w+", re.UNICODE)
_index_cache: dict = {"key": None, "bm25": None, "terms": []}


def tokenize(text: str) -> list[str]:
    """Lowercase và tách theo ký tự chữ/số; giữ nguyên dấu tiếng Việt."""
    return _TOKEN.findall(text.lower())


def load_corpus() -> list[dict]:
    """Đọc toàn bộ chunks đã index trong ChromaDB."""
    from .task4_chunking_indexing import get_collection

    response = get_collection().get(include=["documents", "metadatas"])
    return [
        {"id": item_id, "content": content, "metadata": {"url": None, **metadata}}
        for item_id, content, metadata in zip(
            response["ids"], response["documents"], response["metadatas"]
        )
    ]


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(item["content"]) for item in corpus])


def _get_index(corpus: list[dict]):
    key = (id(corpus), len(corpus))
    if _index_cache["key"] != key:
        _index_cache["key"] = key
        _index_cache["bm25"] = build_bm25_index(corpus)
        _index_cache["terms"] = [set(tokenize(item["content"])) for item in corpus]
    return _index_cache["bm25"], _index_cache["terms"]


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    global CORPUS
    if not CORPUS:
        CORPUS = load_corpus()
    tokens = tokenize(query)
    if not CORPUS or not tokens or top_k <= 0:
        return []

    bm25, doc_terms = _get_index(CORPUS)
    scores = bm25.get_scores(tokens)
    # Lọc theo trùng từ khóa thay vì score > 0: với corpus nhỏ, BM25Okapi có
    # thể cho IDF = 0 (term xuất hiện ở đúng nửa số chunk) dù chunk khớp query.
    query_terms = set(tokens)
    matched = [i for i, terms in enumerate(doc_terms) if query_terms & terms]
    ranked = sorted(matched, key=lambda i: scores[i], reverse=True)
    results = []
    for index in ranked[:top_k]:
        item = CORPUS[index]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
    return results


if __name__ == "__main__":
    for result in lexical_search("Task Achievement band 7", top_k=3):
        print(f"{result['score']:.3f} {result['id']} | {result['content'][:80]!r}")
