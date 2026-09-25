"""
Task 4 — Chunking, embedding và indexing.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

COLLECTION_NAME = "rag_documents"


# Khởi tạo embedding function 1 lần duy nhất ở mức module
_default_ef = None

def get_default_ef():
    global _default_ef
    if _default_ef is None:
        from chromadb.utils import embedding_functions
        print("Đang khởi tạo model embedding onnx (chỉ tải 1 lần)...")
        _default_ef = embedding_functions.DefaultEmbeddingFunction()
    return _default_ef

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Tạo embedding cho danh sách văn bản."""
    # 1. Dùng Gemini nếu có key
    if GOOGLE_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        embeddings = []
        for text in texts:
            res = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(res["embedding"])
        return embeddings

    # 2. Dùng OpenAI nếu có key
    if OPENAI_API_KEY and EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        res = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in res.data]

    # 3. Dùng Default ONNX của ChromaDB
    ef = get_default_ef()
    return ef(texts)


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
    
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        # Xác định doc_type dựa vào tên file hoặc nội dung
        doc_type = "legal" if "ielts" in path.name.lower() and "article" not in path.name.lower() else "news"
        
        documents.append({
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": path.read_text(encoding="utf-8"),
            "metadata": {
                "source": path.name,
                "title": path.stem,
                "doc_type": doc_type,
                "url": "",
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        split_texts = splitter.split_text(document["content"])
        for index, text in enumerate(split_texts):
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk theo batch."""
    contents = [chunk["content"] for chunk in chunks]
    
    # Chia nhỏ batch 50 để tránh lỗi giới hạn payload của API
    batch_size = 50
    all_vectors = []
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        all_vectors.extend(embed_texts(batch))
        
    for chunk, vector in zip(chunks, all_vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    collection = get_collection()
    
    # Upsert theo batch để đảm bảo ổn định
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    print("Loading documents from data/standardized...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks...")
    embedded_chunks = embed_chunks(chunks)

    print("Upserting into ChromaDB...")
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks successfully!")


if __name__ == "__main__":
    run_pipeline()