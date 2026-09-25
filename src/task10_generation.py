"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve_with_trace


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
}
LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(LLM_PROVIDER, "")

REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
REFUSAL_MARKER = "NO_EVIDENCE"

SYSTEM_PROMPT = f"""Bạn là trợ lý về IELTS Writing. Trả lời chỉ từ context được cung cấp.
- Mỗi khẳng định phải có citation dạng [n], n là số của Document trong context; có thể ghép [1][3].
- Không dùng kiến thức ngoài context, không đoán.
- Nếu context không chứa thông tin để trả lời, chỉ trả lời đúng một từ: {REFUSAL_MARKER}
- Trả lời bằng ngôn ngữ của câu hỏi, ngắn gọn, rõ ràng; dùng gạch đầu dòng khi liệt kê."""

_CITATION = re.compile(r"\[(\d+)\]")


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict], numbers: list[int] | None = None) -> str:
    """Tạo context có title và source label.

    numbers: số Document của từng chunk (mặc định 1..n). Generation truyền vào
    thứ hạng gốc để citation [n] luôn trỏ tới sources[n-1] dù đã reorder.
    """
    numbers = numbers or list(range(1, len(chunks) + 1))
    parts = []
    for number, chunk in zip(numbers, chunks):
        metadata = chunk["metadata"]
        parts.append(
            f"[Document {number} | Title: {metadata['title']} | "
            f"Source: {metadata['source']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""
    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types

        response = genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models.generate_content(
            model=LLM_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt, temperature=TEMPERATURE, top_p=TOP_P
            ),
        )
        return response.text or ""
    if LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic

        response = Anthropic().messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
    raise ValueError(f"LLM_PROVIDER must be openai, gemini or anthropic, got {LLM_PROVIDER!r}")


def _refusal(reason: str, trace: dict) -> tuple[dict, dict]:
    trace["refusal_reason"] = reason
    return {"answer": REFUSAL, "sources": [], "retrieval_source": "none"}, trace


def generate_with_trace(
    query: str, top_k: int = TOP_K, use_reranking: bool = True
) -> tuple[dict, dict]:
    """generate_with_citation() kèm trace (retrieval, citation, lý do từ chối) cho UI/eval."""
    try:
        chunks, trace = retrieve_with_trace(query, top_k=top_k, use_reranking=use_reranking)
    except Exception as error:
        return _refusal(f"Retrieval lỗi: {error}", {})
    trace["refusal_reason"] = None
    if not chunks:
        return _refusal("Không tìm thấy chunk nào.", trace)
    # Dense không đủ tự tin và fallback không dùng được: không đủ evidence.
    if trace["low_confidence"] and trace["used"] != "pageindex":
        return _refusal(
            f"Cosine cao nhất {trace['best_dense_score']:.2f} < threshold "
            f"{trace['score_threshold']:.2f} và không có fallback.",
            trace,
        )

    order = reorder_for_llm(list(range(len(chunks))))
    context = format_context([chunks[i] for i in order], numbers=[i + 1 for i in order])
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    try:
        answer = call_llm(SYSTEM_PROMPT, user_message).strip()
    except Exception as error:
        return _refusal(f"LLM provider lỗi: {error}", trace)
    if not answer or REFUSAL_MARKER in answer:
        return _refusal("LLM xác định context không có thông tin trả lời.", trace)

    # Chỉ giữ citation map được về sources; bỏ số không tồn tại.
    answer = _CITATION.sub(
        lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(chunks) else "", answer
    )
    trace["cited"] = sorted({int(n) for n in _CITATION.findall(answer)})
    retrieval_source = "pageindex" if trace["used"] == "pageindex" else "hybrid"
    return {"answer": answer, "sources": chunks, "retrieval_source": retrieval_source}, trace


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    result, _ = generate_with_trace(query, top_k=top_k)
    return result


if __name__ == "__main__":
    for question in (
        "What does a band 7 response need for Task Achievement in Task 1?",
        "Giá vàng hôm nay bao nhiêu?",
    ):
        result = generate_with_citation(question)
        print(f"Q: {question}\nA: {result['answer']}\n[{result['retrieval_source']}] "
              f"{[s['id'] for s in result['sources']]}\n")
