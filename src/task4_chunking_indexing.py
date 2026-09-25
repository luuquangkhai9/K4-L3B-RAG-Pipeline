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
import threading
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


_client_lock = threading.Lock()
_client = None


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    global _client
    # Dùng chung một client: tạo PersistentClient đồng thời từ nhiều thread
    # (Streamlit, evaluation song song) làm Chroma lỗi "Could not connect to tenant".
    with _client_lock:
        if _client is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client.get_or_create_collection(
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


def _cells(line: str) -> list[str]:
    return [c.strip().replace("<br>", " ") for c in line.strip().strip("|").split("|")]


def expand_tables(content: str) -> str:
    """Đổi bảng Markdown thành các đoạn tự đủ ngữ cảnh.

    Mỗi ô (hàng x cột) thành "<heading> — <cột 1> <giá trị> — <cột>: <nội dung>".
    Hàng có ô đầu trống là phần tiếp của hàng trên. Nhờ vậy chunk cắt ở đâu
    cũng còn biết Task/Band/tiêu chí (hàng bảng band descriptors dài > CHUNK_SIZE).
    """
    output: list[str] = []
    heading = ""
    header: list[str] = []
    rows: list[list[str]] = []

    def flush() -> None:
        for row in rows:
            label = f"{header[0]} {row[0]}".strip()
            for column, text in zip(header[1:], row[1:]):
                if text.strip():
                    output.extend([f"{heading} — {label} — {column}: {text.strip()}", ""])
        rows.clear()

    for line in content.splitlines():
        if line.lstrip().startswith("|"):
            cells = _cells(line)
            if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
                continue
            if not header:
                header = cells
            elif cells == header:
                continue  # header lặp lại khi bảng sang trang
            elif cells[0] or not rows:
                rows.append(cells + [""] * (len(header) - len(cells)))
            else:
                for i, text in enumerate(cells[1:], 1):
                    if text and i < len(rows[-1]):
                        rows[-1][i] = f"{rows[-1][i]} {text}".strip()
            continue
        if header:
            flush()
            header = []
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            # Heading phụ ("Scoring criteria...") không thay heading có tên Task.
            if "task" in text.lower() or not heading:
                heading = text
        output.append(line)
    if header:
        flush()
    # Bảng rỗng không sinh đoạn nào nhưng để lại dòng trống liên tiếp.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output))


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunks = []
    for document in documents:
        # Contextual chunk header: chunk giữa tài liệu vẫn biết mình thuộc tài
        # liệu nào (vd. đề thi mẫu) để khớp các câu hỏi nhắc tới tài liệu đó.
        header = f"[{document['metadata']['title']}]\n"
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE - len(header),
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )
        # Bỏ mảnh không có nội dung chữ (vd. chỉ còn "|" khi cắt ngang bảng).
        texts = [
            t for t in splitter.split_text(expand_tables(document["content"]))
            if len(re.sub(r"\W", "", t)) >= 20
        ]
        for index, text in enumerate(texts):
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                # Chunk đầu đã bắt đầu bằng "# <title>" nên không lặp lại.
                "content": text if text.startswith("# ") else header + text,
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
