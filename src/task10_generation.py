"""
Task 10 — Generation có citation.

Luồng: retrieve -> reorder (giảm lost-in-the-middle) -> format context ->
gọi LLM -> trả answer kèm sources.

Gateway api.ai-box.vn là OpenAI-compatible nên nhánh "openai" dùng cho cả
qwen3.8-flash. Thiếu evidence hoặc provider lỗi thì trả safe refusal,
không bịa thông tin.
"""

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation dạng [Document N] khớp với nhãn trong context.
Nếu context không chứa đủ evidence, hãy từ chối xác minh thay vì suy đoán.
Trả lời bằng tiếng Việt, ngắn gọn và nêu rõ nguồn."""

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context.

    LLM chú ý nhất ở đầu và cuối prompt (lost-in-the-middle), nên chunk quan
    trọng nhất đặt ở đầu, các chunk giữa bị đẩy vào giữa. Không sửa list gốc.
    """
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label để citation kiểm chứng được."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        parts.append(
            f"[Document {index} | Title: {metadata.get('title', '')} | "
            f"Source: {metadata.get('source', '')} | URL: {metadata.get('url') or 'n/a'}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình trong .env."""
    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return (response.choices[0].message.content or "").strip()

    if LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=LLM_MODEL,
            system=system_prompt,
            max_tokens=1024,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()

    if LLM_PROVIDER == "gemini":
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=user_message,
            config={
                "system_instruction": system_prompt,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            },
        )
        return (response.text or "").strip()

    raise ValueError(f"LLM_PROVIDER không hỗ trợ: {LLM_PROVIDER}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }

    # retrieval_source chỉ nhận hybrid | pageindex | none.
    method = chunks[0].get("retrieval_method")
    retrieval_source = "pageindex" if method == "pageindex" else "hybrid"

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception as error:
        print(f"LLM lỗi: {type(error).__name__}: {error}")
        return {
            "answer": SAFE_REFUSAL,
            "sources": chunks,
            "retrieval_source": retrieval_source,
        }

    if not answer:
        answer = SAFE_REFUSAL

    return {
        "answer": answer,
        # Giữ nguyên thứ tự score giảm dần để khớp contract, dù context đã reorder.
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    result = generate_with_citation("Band 7 yêu cầu gì ở tiêu chí Task Response?")
    print(result["answer"])
    print("\n--- Sources ---")
    for source in result["sources"]:
        print(f"  {source['score']:.3f}  {source['metadata']['title']}  [{source['id']}]")