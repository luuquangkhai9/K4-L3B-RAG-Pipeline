"""Task 6 — lexical BM25 search over the same chunk corpus as Task 4."""

import re

from .task4_chunking_indexing import chunk_documents, load_documents

CORPUS: list[dict] = []
_bm25_index = None


def _tokenize(text: str) -> list[str]:
    """Keep Unicode words (including Vietnamese diacritics) and normalize case."""
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    if not corpus:
        raise ValueError("Cannot build a BM25 index from an empty corpus")
    tokenized = [_tokenize(item["content"]) for item in corpus]
    if any(not tokens for tokens in tokenized):
        raise ValueError("BM25 corpus contains an empty or non-tokenizable chunk")
    return BM25Okapi(tokenized)


def _ensure_index() -> None:
    """Build the BM25 corpus lazily from Task 4's standardized documents/chunks."""
    global CORPUS, _bm25_index
    if _bm25_index is None:
        if not CORPUS:
            CORPUS = chunk_documents(load_documents())
        _bm25_index = build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    _ensure_index()
    scores = _bm25_index.get_scores(query_tokens)
    query_token_set = set(query_tokens)
    ranked_indices = sorted(
        range(len(CORPUS)),
        key=lambda index: (-float(scores[index]), CORPUS[index]["id"]),
    )
    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if not query_token_set.intersection(_tokenize(CORPUS[index]["content"])):
            continue
        item = CORPUS[index]
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": score,
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
