"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "recursive"

# Embedding qua API (không chạy model local). Cấu hình trong .env.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
_DEFAULT_MODELS = {"openai": "text-embedding-3-small", "gemini": "gemini-embedding-001"}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or _DEFAULT_MODELS.get(
    EMBEDDING_PROVIDER, ""
)
EMBEDDING_BATCH_SIZE = 64

COLLECTION_NAME = "rag_documents"


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
    return [list(item.values) for item in response.embeddings]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed theo batch bằng provider trong .env. Task 5 dùng lại hàm này."""
    providers = {"openai": _embed_openai, "gemini": _embed_gemini}
    if EMBEDDING_PROVIDER not in providers:
        raise ValueError(
            f"EMBEDDING_PROVIDER must be one of {sorted(providers)}, "
            f"got {EMBEDDING_PROVIDER!r}"
        )
    embed = providers[EMBEDDING_PROVIDER]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        vectors.extend(embed(texts[start : start + EMBEDDING_BATCH_SIZE]))
    return vectors


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in path.parts else "news"
        url = None
        title = path.stem
        match = re.search(r"\*\*Source:\*\*\s*(\S+)", content)
        url = match.group(1) if match else None
        heading = re.match(r"#\s+(.+)", content)
        if heading:
            title = heading.group(1).strip()
        documents.append({
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": doc_type,
                "url": url,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        # Bỏ mảnh không có nội dung chữ (vd. chỉ còn "|" khi cắt ngang bảng).
        texts = [
            t for t in splitter.split_text(document["content"])
            if len(re.sub(r"\W", "", t)) >= 20
        ]
        for index, text in enumerate(texts):
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    collection = get_collection()
    # Chroma không nhận giá trị None trong metadata: bỏ key url khi rỗng.
    # Khi đọc lại, dùng metadata.get("url") để khôi phục None theo contract.
    for start in range(0, len(chunks), 500):
        batch = chunks[start : start + 500]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[
                {k: v for k, v in c["metadata"].items() if v is not None}
                for c in batch
            ],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
