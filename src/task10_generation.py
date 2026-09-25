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

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời bằng tiếng Việt chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation theo dạng [ID], trong đó ID là giá trị sau 'Source ID:' trong context.
Không viết tiền tố 'chunk-id:' bên trong citation; chép chính xác ID, gồm dấu gạch chéo và dấu hai chấm nếu có.
Không suy diễn ngoài nguồn. Nếu context không đủ để trả lời, hãy nói rõ rằng bạn không thể xác minh."""
SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    return list(chunks[::2]) + list(chunks[1::2][::-1])


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        parts.append(
            f"[Source ID: {chunk['id']} | Title: {metadata.get('title', 'Unknown')} | "
            f"Source: {metadata.get('source', 'Unknown')}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = LLM_PROVIDER.strip().lower()
    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI generation")
        model = LLM_MODEL.strip() or "gpt-4o-mini"
        model = {"gpt4o": "gpt-4o", "gpt4o-mini": "gpt-4o-mini"}.get(model.lower(), model)
        response = OpenAI(api_key=api_key).chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return (response.choices[0].message.content or "").strip()
    if provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini generation")
        model = LLM_MODEL.strip() or "gemini-2.5-flash"
        response = genai.Client(api_key=api_key).models.generate_content(
            model=model, contents=f"{system_prompt}\n\n{user_message}"
        )
        return (response.text or "").strip()
    if provider == "anthropic":
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic generation")
        model = LLM_MODEL.strip() or "claude-3-5-haiku-latest"
        response = anthropic.Anthropic(api_key=api_key).messages.create(
            model=model, max_tokens=1200, system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE, top_p=TOP_P,
        )
        return "\n".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()
    raise ValueError("Unsupported LLM_PROVIDER; use openai, gemini, or anthropic")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    chunks = retrieve(query.strip(), top_k=top_k)
    if not chunks:
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}

    context = format_context(reorder_for_llm(chunks))
    try:
        answer = call_llm(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {query.strip()}")
    except Exception:
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}
    valid_ids = {str(chunk["id"]) for chunk in chunks}
    cited_ids = set(re.findall(r"\[([^\[\]]+)\]", answer)) & valid_ids
    if not answer or not cited_ids:
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}
    sources = [chunk for chunk in chunks if chunk["id"] in cited_ids]
    method = chunks[0].get("retrieval_method")
    retrieval_source = "pageindex" if method == "pageindex" else "hybrid"
    return {"answer": answer, "sources": sources, "retrieval_source": retrieval_source}


if __name__ == "__main__":
    print(generate_with_citation("test query"))
