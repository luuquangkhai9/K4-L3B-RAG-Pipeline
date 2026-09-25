"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LEGAL_DIR = PROJECT_ROOT / "data" / "landing" / "legal"
CACHE_PATH = PROJECT_ROOT / "pageindex_doc_ids.json"
API_BASE_URL = "https://api.pageindex.ai"
REQUEST_TIMEOUT = (10, 120)
READY_TIMEOUT_SECONDS = 300


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    from dotenv import load_dotenv
    import requests

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PAGEINDEX_API_KEY is required to upload documents to PageIndex")

    sources = sorted(LEGAL_DIR.glob("*.pdf"))
    if not sources:
        raise FileNotFoundError(f"No PDF documents found in {LEGAL_DIR}")
    cache = _load_cache()
    headers = {"api_key": api_key}
    for source in sources:
        fingerprint = _file_fingerprint(source)
        cached = cache.get(source.name, {})
        if cached.get("fingerprint") == fingerprint and cached.get("doc_id"):
            print(f"PageIndex cached: {source.name}")
            continue

        with source.open("rb") as file_handle:
            response = requests.post(
                f"{API_BASE_URL}/doc/",
                headers=headers,
                files={"file": (source.name, file_handle, "application/pdf")},
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        payload = response.json()
        doc_id = payload.get("doc_id") or payload.get("id")
        if not isinstance(doc_id, str) or not doc_id:
            raise RuntimeError(f"PageIndex upload did not return a document ID for {source.name}")
        cache[source.name] = {"fingerprint": fingerprint, "doc_id": doc_id}
        _save_cache(cache)
        _wait_for_document(doc_id, headers)
        print(f"PageIndex uploaded: {source.name}")


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        value = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    temporary_path = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".tmp")
    temporary_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(CACHE_PATH)


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wait_for_document(doc_id: str, headers: dict) -> None:
    import requests

    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = requests.get(
            f"{API_BASE_URL}/doc/{doc_id}/metadata/",
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        status = str(response.json().get("status", "")).lower()
        if status == "completed":
            return
        if status in {"failed", "error"}:
            raise RuntimeError(f"PageIndex failed to process document {doc_id}")
        time.sleep(5)
    raise TimeoutError(f"PageIndex document processing timed out for {doc_id}")


def _extract_text_nodes(value: object) -> list[dict]:
    """Extract text-bearing nodes from differing PageIndex retrieval payload shapes."""
    found = []
    if isinstance(value, dict):
        text = value.get("text") or value.get("content") or value.get("markdown")
        if isinstance(text, str) and text.strip():
            found.append({**value, "_text": text.strip()})
        for key in ("nodes", "results", "retrieved_nodes", "passages", "items"):
            child = value.get(key)
            if child is not None:
                found.extend(_extract_text_nodes(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_extract_text_nodes(child))
    return found


def _chat_search(doc_ids: list[str], query: str, headers: dict) -> dict:
    import requests

    with requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json={
            "doc_id": doc_ids,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Find the most relevant evidence in the indexed documents for this query. "
                        "Answer with concise, factual evidence and cite every claim to its source page. "
                        "Do not invent details; if the documents do not support an answer, say so.\n\n"
                        f"Query: {query}"
                    ),
                }
            ],
            "stream": True,
            "temperature": 0.0,
            "enable_citations": True,
        },
        stream=True,
        timeout=(30, READY_TIMEOUT_SECONDS),
    ) as response:
        response.raise_for_status()
        answer_parts = []
        citations = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    answer_parts.append(content)
            if isinstance(event.get("citations"), list):
                citations.extend(event["citations"])

    answer = "".join(answer_parts).strip()
    return {"choices": [{"message": {"content": answer, "citations": citations}}]}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    from dotenv import load_dotenv
    import requests

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not api_key:
        return []

    # Ensure IDs exist and changed source PDFs get re-uploaded once.
    upload_documents()
    cache = _load_cache()
    headers = {"api_key": api_key}
    sources = sorted(LEGAL_DIR.glob("*.pdf"))
    source_by_name = {path.name.casefold(): path for path in sources}
    doc_ids = [
        cache[path.name]["doc_id"]
        for path in sources
        if cache.get(path.name, {}).get("doc_id")
    ]
    if not doc_ids:
        return []

    try:
        payload = _chat_search(doc_ids, query.strip(), headers)
    except (requests.RequestException, RuntimeError, TimeoutError) as error:
        print(f"PageIndex query failed: {error}")
        return []

    choices = payload.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    answer = message.get("content") if isinstance(message, dict) else ""
    if not isinstance(answer, str) or not answer.strip():
        return []

    citation_items = message.get("citations") or payload.get("citations") or []
    citations_by_tag = {}
    if isinstance(citation_items, list):
        for citation in citation_items:
            if not isinstance(citation, dict):
                continue
            document = citation.get("doc") or citation.get("document") or citation.get("source")
            page = citation.get("page") or citation.get("page_index")
            block = citation.get("block") or citation.get("block_id")
            if document and page:
                try:
                    citations_by_tag[(Path(str(document)).name.casefold(), int(page), str(block or ""))] = citation
                except (TypeError, ValueError):
                    continue

    citation_tags = re.findall(r"<doc=([^;>]+);page=(\d+)(?:;block=([^>]+))?>", answer)
    results = []
    seen = set()
    for document, page_text, block_text in citation_tags:
        source = source_by_name.get(Path(document).name.casefold())
        if source is None:
            continue
        page_index = int(page_text)
        block_id = block_text or ""
        key = (source.name.casefold(), page_index, block_id)
        if key in seen:
            continue
        seen.add(key)
        citation = citations_by_tag.get(key, {})
        doc_id = cache[source.name]["doc_id"]
        evidence = citation.get("text") or citation.get("markdown")
        if not evidence and block_id:
            try:
                block_response = requests.get(
                    f"{API_BASE_URL}/doc/{doc_id}/block/{quote(block_id, safe='')}/",
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
                block_response.raise_for_status()
                evidence = block_response.json().get("text")
            except (requests.RequestException, ValueError):
                evidence = None
        if not evidence:
            continue
        result_id = f"pageindex:{doc_id}:page-{page_index}"
        if block_id:
            result_id += f":{block_id}"
        results.append(
            {
                "id": result_id,
                "content": str(evidence).strip(),
                "score": 1.0 / (len(results) + 1),
                "metadata": {
                    "source": source.name,
                    "title": source.stem,
                    "doc_type": "legal",
                    "url": None,
                    "chunk_index": page_index - 1,
                },
                "retrieval_method": "pageindex",
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    upload_documents()
