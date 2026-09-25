"""
Task 4 — Chunking, embedding và indexing.

Chủ đề nhóm: IELTS Writing — band descriptors, tiêu chí chấm điểm, bài mẫu.

Nguồn dữ liệu: data/standardized/ (Markdown do Task 3 tạo, có khối metadata
ở đầu file nên đọc lại được source/title/doc_type/url).

Embedding: gateway OpenAI-compatible (api.ai-box.vn), model text-embedding-v4,
mặc định 1024 chiều. Task 5 dùng lại đúng hàm embed_texts() này.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from .task3_convert_markdown import parse_metadata


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# Gateway giới hạn số input mỗi request; 10 là mức an toàn cho text-embedding-v4.
EMBED_BATCH_SIZE = 10

COLLECTION_NAME = "rag_documents"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed văn bản theo EMBEDDING_PROVIDER trong .env.

    Task 5 gọi lại hàm này cho query, nên model và dimension luôn khớp với lúc index.
    """
    if not texts:
        return []

    if EMBEDDING_PROVIDER == "sentence_transformers":
        # Cần cài extra: pip install -e ".[local-embedding]"
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        return model.encode(texts, normalize_embeddings=True).tolist()

    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                dimensions=EMBEDDING_DIM,
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    if EMBEDDING_PROVIDER == "gemini":
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        vectors = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            response = client.models.embed_content(model=EMBEDDING_MODEL, contents=batch)
            vectors.extend(item.values for item in response.embeddings)
        return vectors

    raise ValueError(f"EMBEDDING_PROVIDER không hỗ trợ: {EMBEDDING_PROVIDER}")


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
        if path.name.startswith("."):
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        # Thư mục con quyết định doc_type: standardized/legal/... hoặc news/...
        doc_type = "legal" if "legal" in path.parts else "news"
        header = parse_metadata(content)
        documents.append(
            {
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": content,
                "metadata": {
                    "source": header.get("source") or path.name,
                    "title": header.get("title") or path.stem,
                    "doc_type": doc_type,
                    "url": header.get("url") or None,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index.

    Chia theo heading Markdown trước, rồi mới cắt nhỏ phần thân nếu quá dài.
    Mỗi chunk được gắn lại đường dẫn heading của nó (ví dụ
    "Writing Task 2 — Band Descriptors > Band 7") vì RecursiveCharacterTextSplitter
    cắt rời tiêu đề khỏi nội dung: chunk mô tả đúng tiêu chí nhưng mất nhãn band
    thì retrieval không trả lời được câu hỏi dạng "Band 7 yêu cầu gì?".
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
        strip_headers=True,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for document in documents:
        index = 0
        for section in header_splitter.split_text(document["content"]):
            prefix = " > ".join(
                section.metadata[key] for key in ("H1", "H2", "H3") if section.metadata.get(key)
            )
            heading = f"[{prefix}]\n" if prefix else ""

            # Chừa chỗ cho phần heading để chunk cuối vẫn không vượt CHUNK_SIZE.
            body_splitter = (
                splitter
                if not heading
                else RecursiveCharacterTextSplitter(
                    chunk_size=max(CHUNK_SIZE - len(heading), 100),
                    chunk_overlap=CHUNK_OVERLAP,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
            )

            # Lọc chunk rỗng trước khi đánh số để chunk_index liên tục từ 0.
            pieces = [text for text in body_splitter.split_text(section.page_content) if text.strip()]
            for text in pieces:
                chunks.append(
                    {
                        # ID ổn định theo vị trí -> upsert lại không tạo bản trùng.
                        "id": f"{document['id']}::chunk-{index}",
                        "content": heading + text,
                        "metadata": {**document["metadata"], "chunk_index": index},
                    }
                )
                index += 1
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Nhận {len(vectors)} vector cho {len(chunks)} chunk")
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return
    collection = get_collection()
    for start in range(0, len(chunks), 100):
        batch = chunks[start : start + 100]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            # Chroma chỉ nhận metadata kiểu scalar; None không hợp lệ.
            metadatas=[
                {k: (v if v is not None else "") for k, v in chunk["metadata"].items()}
                for chunk in batch
            ],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(
        f"Indexed {len(embedded_chunks)} chunks "
        f"từ {len(documents)} tài liệu ({EMBEDDING_MODEL}, {EMBEDDING_DIM} chiều)"
    )


if __name__ == "__main__":
    run_pipeline()