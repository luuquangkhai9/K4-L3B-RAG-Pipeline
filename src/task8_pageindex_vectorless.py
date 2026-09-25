"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
# SDK chỉ nhận PDF: dùng bản PDF gốc của tài liệu legal.
LEGAL_PDF_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_ID_CACHE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

SEARCH_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 1.5


def _client():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not set")
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_doc_ids() -> dict[str, str]:
    if DOC_ID_CACHE.exists():
        return json.loads(DOC_ID_CACHE.read_text(encoding="utf-8"))
    return {}


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    client = _client()
    doc_ids = _load_doc_ids()
    for path in sorted(LEGAL_PDF_DIR.glob("*.pdf")):
        if path.name in doc_ids:
            print(f"Skip (cached): {path.name}")
            continue
        response = client.submit_document(str(path))
        doc_ids[path.name] = response["doc_id"]
        DOC_ID_CACHE.write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")
        print(f"Uploaded: {path.name} -> {response['doc_id']}")

    for name, doc_id in doc_ids.items():
        print(f"{name}: retrieval_ready={client.is_retrieval_ready(doc_id)}")


def _node_text(node: dict) -> str:
    contents = node.get("relevant_contents") or []
    parts = [c.get("relevant_content", "") for c in contents if isinstance(c, dict)]
    return "\n\n".join(p for p in parts if p).strip() or str(node.get("text", "")).strip()


def _query_document(client, source: str, doc_id: str, query: str, deadline: float) -> list[dict]:
    retrieval_id = client.submit_query(doc_id, query)["retrieval_id"]
    while True:
        response = client.get_retrieval(retrieval_id)
        if response.get("status") == "completed":
            break
        if response.get("status") == "failed" or time.monotonic() > deadline:
            raise RuntimeError(f"PageIndex retrieval {response.get('status')} for {source}")
        time.sleep(POLL_INTERVAL_SECONDS)

    nodes = []
    for node in response.get("retrieved_nodes") or []:
        content = _node_text(node)
        if not content:
            continue
        pages = [c.get("page_index") for c in node.get("relevant_contents") or [] if isinstance(c, dict)]
        nodes.append({
            "id": f"pageindex::{source}::{node.get('node_id', len(nodes))}",
            "content": content,
            "metadata": {
                "source": source,
                "title": node.get("title") or Path(source).stem,
                "doc_type": "legal",
                "url": None,
                "chunk_index": len(nodes),
                "pages": [p for p in pages if p is not None],
            },
        })
    return nodes


def _search(query: str, top_k: int) -> list[dict]:
    doc_ids = _load_doc_ids()
    if not doc_ids:
        raise RuntimeError("No PageIndex documents uploaded; run python -m src.task8_pageindex_vectorless")
    client = _client()
    deadline = time.monotonic() + SEARCH_TIMEOUT_SECONDS
    nodes = []
    for source, doc_id in doc_ids.items():
        nodes.extend(_query_document(client, source, doc_id, query, deadline))
    # API không trả score: gán score giảm dần theo thứ tự trả về.
    return [
        {**node, "score": 1.0 / rank, "retrieval_method": "pageindex"}
        for rank, node in enumerate(nodes[:top_k], 1)
    ]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    _client()  # báo lỗi ngay nếu thiếu key, không tạo thread
    # SDK không có timeout cho HTTP request nên chặn toàn bộ lời gọi bằng thread.
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        return executor.submit(_search, query, top_k).result(timeout=SEARCH_TIMEOUT_SECONDS + 5)
    except FutureTimeout as error:
        raise RuntimeError("PageIndex search timed out") from error
    finally:
        executor.shutdown(wait=False)


if __name__ == "__main__":
    upload_documents()
