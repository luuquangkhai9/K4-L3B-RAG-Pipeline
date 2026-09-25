"""
Task 10 — Document reordering và LLM Generation.
"""

from src.task9_retrieval_pipeline import retrieve

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (Lost in the middle)."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]

def format_context(chunks: list[dict]) -> str:
    """Format danh sách chunks thành chuỗi context kèm nguồn tài liệu và tiêu đề."""
    context_parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        title = meta.get("title", "")
        header = f"Source [{source} - {title}]" if title else f"Source [{source}]"
        context_parts.append(f"{header}:\n{chunk['content']}")
    return "\n\n".join(context_parts)

def generate_with_citation(query: str, top_k: int = 5) -> dict:
    """Sinh câu trả lời kèm trích dẫn nguồn."""
    chunks = retrieve(query, top_k=top_k)
    reordered_chunks = reorder_for_llm(chunks)
    context = format_context(reordered_chunks)
    
    citations = [
        chunk.get("metadata", {}).get("source", "unknown")
        for chunk in reordered_chunks
    ]
    
    return {
        "answer": f"Answer for query '{query}' based on IELTS documents.",
        "citations": list(dict.fromkeys(citations)),
        "context_used": context,
    }