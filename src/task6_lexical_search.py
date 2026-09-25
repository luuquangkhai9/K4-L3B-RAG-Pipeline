"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu (ví dụ "5445/QĐ-ĐHBK", "179/2026/NĐ-CP") và tên riêng.

Corpus được tái tạo từ chính Task 4 (load_documents + chunk_documents) nên ID
và nội dung khớp đúng với những gì đã index vào ChromaDB.
"""

import re


CORPUS: list[dict] = []

# Giữ chữ cái (kể cả tiếng Việt có dấu) và chữ số, tách phần còn lại.
# Không dùng .split() trần vì "học phí," và "học phí" sẽ thành hai token khác nhau.
TOKEN_PATTERN = re.compile(r"[0-9a-zà-ỹđ]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản, đủ tốt cho BM25 trên văn bản tiếng Việt."""
    return TOKEN_PATTERN.findall(text.lower())


def tokenize_corpus(corpus: list[dict]) -> list[list[str]]:
    return [tokenize(item["content"]) for item in corpus]


def load_corpus() -> list[dict]:
    """Tái tạo corpus chunks giống hệt lúc index (không cần gọi embedding)."""
    from .task4_chunking_indexing import chunk_documents, load_documents

    return chunk_documents(load_documents())


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi(tokenize_corpus(corpus))


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    from rank_bm25 import BM25Okapi

    # Đọc CORPUS tại thời điểm gọi để test có thể monkeypatch.
    corpus = CORPUS or load_corpus()
    if not corpus or top_k <= 0:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # Tokenize một lần rồi dùng lại cho cả BM25 và bước lọc bên dưới.
    tokenized = tokenize_corpus(corpus)
    scores = BM25Okapi(tokenized).get_scores(query_tokens)
    query_set = set(query_tokens)

    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)

    results = []
    for index in ranked:
        # Lọc theo token trùng query, KHÔNG lọc theo score > 0:
        # BM25Okapi trả IDF = 0 khi một term có mặt ở đúng nửa số tài liệu
        # (corpus 2 tài liệu -> log(1.5/1.5) = 0), lọc theo score sẽ làm mất
        # kết quả đúng. Cách này cũng loại tài liệu không liên quan gì tới query.
        if not query_set & set(tokenized[index]):
            continue
        item = corpus[index]
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": float(scores[index]),
                "metadata": dict(item["metadata"]),
                "retrieval_method": "bm25",
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("coherence and cohesion", top_k=3):
        print(f"{result['score']:.3f}  {result['metadata']['title']}  [{result['id']}]")