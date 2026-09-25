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

import numpy as np


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 16

COLLECTION_NAME = "rag_documents"

_embedding_model = None


def _load_environment() -> None:
    """Load project settings from .env without replacing shell variables."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env", override=False)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text with the configured provider; Task 5 reuses this function."""
    global _embedding_model
    if not texts:
        return []

    _load_environment()
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip().lower()
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL).strip() or EMBEDDING_MODEL

    if provider == "sentence_transformers":
        if _embedding_model is None:
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer(model_name)
        vectors = _embedding_model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > EMBEDDING_BATCH_SIZE,
        )
    elif provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        client = OpenAI(api_key=api_key)
        configured_model = os.getenv("EMBEDDING_MODEL", "").strip()
        model_name = (
            configured_model
            if configured_model and configured_model != EMBEDDING_MODEL
            else "text-embedding-3-small"
        )
        batches = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            response = client.embeddings.create(
                model=model_name,
                input=texts[start : start + EMBEDDING_BATCH_SIZE],
            )
            batches.extend(item.embedding for item in response.data)
        vectors = np.asarray(batches, dtype=np.float32)
    elif provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings")
        client = genai.Client(api_key=api_key)
        configured_model = os.getenv("EMBEDDING_MODEL", "").strip()
        model_name = (
            configured_model
            if configured_model and configured_model != EMBEDDING_MODEL
            else "gemini-embedding-001"
        )
        batches = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            response = client.models.embed_content(
                model=model_name,
                contents=texts[start : start + EMBEDDING_BATCH_SIZE],
            )
            batches.extend(item.values for item in response.embeddings)
        vectors = np.asarray(batches, dtype=np.float32)
    else:
        raise ValueError(
            "Unsupported EMBEDDING_PROVIDER {!r}; use sentence_transformers, openai, or gemini".format(provider)
        )

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise RuntimeError(
            f"Embedding provider returned shape {vectors.shape} for {len(texts)} texts"
        )
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embedding provider returned non-finite values")
    return vectors.tolist()


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
    if not STANDARDIZED_DIR.is_dir():
        raise FileNotFoundError(f"Standardized directory not found: {STANDARDIZED_DIR}")

    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        relative_path = path.relative_to(STANDARDIZED_DIR)
        if not relative_path.parts or relative_path.parts[0] not in {"legal", "news"}:
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        heading = re.search(r"(?m)^#\s+(.+?)\s*$", content)
        title = heading.group(1).strip() if heading else path.stem
        source_match = re.search(r"(?m)^\*\*Source:\*\*\s*(\S+)\s*$", content)
        documents.append(
            {
                "id": relative_path.as_posix(),
                "content": content,
                "metadata": {
                    "source": path.name,
                    "title": title,
                    "doc_type": relative_path.parts[0],
                    "url": source_match.group(1) if source_match else None,
                },
            }
        )
    if not documents:
        raise ValueError(f"No non-empty legal/news Markdown documents found in {STANDARDIZED_DIR}")
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    chunks = []
    seen_ids = set()
    for document in documents:
        if not document.get("content", "").strip():
            continue
        text = document["content"].strip()
        pieces = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):
                # Prefer paragraph, line, sentence, then word boundaries.
                window_start = start + CHUNK_SIZE // 2
                for separator in ("\n\n", "\n", ". ", " "):
                    boundary = text.rfind(separator, window_start, end)
                    if boundary >= window_start:
                        end = boundary + len(separator)
                        break
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= len(text):
                break
            next_start = max(start + 1, end - CHUNK_OVERLAP)
            while next_start < end and text[next_start].isspace():
                next_start += 1
            start = next_start
        for index, text in enumerate(pieces):
            content = text.strip()
            if not content:
                continue
            chunk_id = f"{document['id']}::chunk-{index}"
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk ID generated: {chunk_id}")
            seen_ids.add(chunk_id)
            chunks.append(
                {
                    "id": chunk_id,
                    "content": content,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
    if not chunks:
        raise ValueError("No non-empty chunks were produced")
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    embedded = []
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        vectors = embed_texts([chunk["content"] for chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} embeddings, received {len(vectors)}")
        for chunk, vector in zip(batch, vectors):
            embedded.append({**chunk, "embedding": vector})
    return embedded


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        raise ValueError("Cannot index an empty chunk list")
    ids = [chunk["id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs must be unique before indexing")

    collection = get_collection()
    expected_ids = set(ids)
    existing_ids = set(collection.get(include=[])['ids'])
    stale_ids = existing_ids - expected_ids
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))

    for start in range(0, len(chunks), 500):
        batch = chunks[start : start + 500]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[
                {key: value for key, value in chunk["metadata"].items() if value is not None}
                for chunk in batch
            ],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks from {len(documents)} documents")


if __name__ == "__main__":
    run_pipeline()
