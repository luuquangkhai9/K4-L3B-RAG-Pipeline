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

import os
import re
from functools import lru_cache

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


load_dotenv()

# Hiệu chỉnh bằng calibrate_threshold() + golden dataset: in-domain thấp nhất 0.38
# ("Why can a response be penalised..."), out-of-domain 0.12–0.51 → hai vùng chồng nhau.
# 0.35 chặn chắc câu ngoài domain rõ ràng (≤ 0.28); phần chồng lấn do LLM từ chối
# (NO_EVIDENCE), đã đạt 5/5 safe refusal trong evaluation.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or 0.35)
DEFAULT_TOP_K = 5
# Lấy rộng hơn top_k cho mỗi nhánh để RRF có đủ ứng viên.
CANDIDATE_MULTIPLIER = 2
# Query expansion bằng bản dịch tiếng Anh; tắt để chạy A/B (config C trong evaluation).
QUERY_TRANSLATION = os.getenv("QUERY_TRANSLATION", "1") != "0"


_VIETNAMESE = re.compile(r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.I)
TRANSLATE_PROMPT = (
    "Translate the user's question into English for searching IELTS documents. "
    "Keep IELTS terms (Task 1, Task 2, band, Coherence and Cohesion...). Output only the translation."
)


@lru_cache(maxsize=256)
def translate_query(query: str) -> str | None:
    """Dịch query tiếng Việt sang tiếng Anh: phần lớn tài liệu gốc là tiếng Anh,
    BM25 không khớp được từ khóa khác ngôn ngữ và dense cũng yếu đi."""
    if not _VIETNAMESE.search(query):
        return None
    try:
        from .task10_generation import call_llm

        translated = call_llm(TRANSLATE_PROMPT, query).strip()
    except Exception:
        return None  # dịch lỗi thì vẫn tìm bằng query gốc
    return translated if translated and translated.lower() != query.lower() else None


def retrieve_with_trace(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> tuple[list[dict], dict]:
    """Như retrieve() nhưng trả thêm trace để UI/eval giải thích quyết định."""
    candidates = top_k * CANDIDATE_MULTIPLIER
    translated = translate_query(query) if QUERY_TRANSLATION else None
    queries = [query] + ([translated] if translated else [])

    dense_lists = [semantic_search(q, top_k=candidates) for q in queries]
    sparse_lists = [lexical_search(q, top_k=candidates) for q in queries] if use_reranking else []
    if use_reranking:
        # Một lần RRF duy nhất cho tất cả danh sách (dense/BM25 x query gốc/bản dịch).
        hybrid = rerank_rrf([dense_lists[0], sparse_lists[0], *dense_lists[1:], *sparse_lists[1:]], top_k=top_k)
    else:
        merged = {}
        for item in sorted((i for d in dense_lists for i in d), key=lambda i: i["score"], reverse=True):
            merged.setdefault(item["id"], item)
        hybrid = list(merged.values())[:top_k]

    # Fallback dùng cosine gốc cao nhất của dense (không dùng RRF score).
    best_dense_score = max((d[0]["score"] for d in dense_lists if d), default=0.0)
    trace = {
        "translated_query": translated,
        "best_dense_score": best_dense_score,
        "score_threshold": score_threshold,
        "dense_count": sum(len(d) for d in dense_lists),
        "bm25_count": sum(len(s) for s in sparse_lists),
        "mode": "hybrid" if use_reranking else "dense",
        "low_confidence": best_dense_score < score_threshold,
        "fallback_attempted": False,
        "fallback_error": None,
        "used": "hybrid" if use_reranking else "dense",
    }
    if trace["low_confidence"]:
        trace["fallback_attempted"] = True
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                trace["used"] = "pageindex"
                return fallback[:top_k], trace
            trace["fallback_error"] = "PageIndex returned no results"
        except Exception as error:
            trace["fallback_error"] = str(error)
    return hybrid[:top_k], trace


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    results, _ = retrieve_with_trace(query, top_k, score_threshold, use_reranking)
    return results


def calibrate_threshold() -> None:
    """In best dense score của query trong/ngoài domain để chọn threshold."""
    in_domain = [
        "What does a band 7 response need for Task Achievement in Task 1?",
        "How many words must I write for IELTS Writing Task 2?",
        "What is assessed under Coherence and Cohesion?",
        "Lexical Resource band descriptor for band 6",
        "Will IELTS still offer a paper-based test?",
        "Mẹo tăng điểm IELTS Writing là gì?",
        "Làm sao viết IELTS Writing rõ ràng và súc tích?",
        "Tiêu chí chấm điểm Task Response trong Task 2",
        "How are responses penalised for plagiarism or bullet points?",
        "What happens if a Task 2 response is under 250 words?",
    ]
    out_domain = [
        "What is the capital of Australia?",
        "Học phí Đại học Bách khoa Hà Nội năm 2025 là bao nhiêu?",
        "How do I bake sourdough bread?",
        "Giá vàng hôm nay bao nhiêu?",
        "Who won the 2022 FIFA World Cup?",
        "Cách đăng ký học phần tại trường đại học",
        "Explain the rules of chess castling",
        "What is the TOEFL iBT speaking section format?",
    ]
    for label, queries in (("IN ", in_domain), ("OUT", out_domain)):
        for query in queries:
            dense = semantic_search(query, top_k=1)
            score = dense[0]["score"] if dense else 0.0
            print(f"{label} {score:.3f}  {query}")


if __name__ == "__main__":
    calibrate_threshold()
