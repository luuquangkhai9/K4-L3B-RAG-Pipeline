"""
Task 8 — PageIndex vectorless fallback.

Chủ đề nhóm: IELTS Writing — band descriptors, tiêu chí chấm điểm, bài mẫu.

PageIndex là dịch vụ ngoài (api.pageindex.ai). SDK 0.2.8 chỉ nhận **file PDF**,
nên tài liệu Markdown của Task 3 được chuyển sang PDF tạm trong pageindex_pdfs/
trước khi upload. Tài liệu legal đã là PDF gốc nên upload trực tiếp.

Luồng: submit_document -> doc_id -> submit_query -> retrieval_id -> get_retrieval.
Cần PAGEINDEX_API_KEY trong .env; thiếu key thì báo lỗi để Task 9 rơi về hybrid.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = ROOT / "data" / "standardized"
LEGAL_DIR = ROOT / "data" / "landing" / "legal"

# Hai đường dẫn này đã có sẵn trong .gitignore.
DOC_IDS_PATH = ROOT / "pageindex_doc_ids.json"
PDF_DIR = ROOT / "pageindex_pdfs"

# PageIndex xử lý bất đồng bộ; chờ tối đa 90s cho mỗi retrieval.
POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 90

# fpdf2 không có font Unicode mặc định; cần TTF hỗ trợ tiếng Việt.
VIETNAMESE_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\times.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def _client():
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "PAGEINDEX_API_KEY chưa được cấu hình trong .env — "
            "Task 9 sẽ dùng hybrid results thay vì fallback."
        )
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_doc_ids() -> dict[str, str]:
    if DOC_IDS_PATH.exists():
        try:
            return json.loads(DOC_IDS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_doc_ids(mapping: dict[str, str]) -> None:
    DOC_IDS_PATH.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _find_font() -> Path | None:
    for path in VIETNAMESE_FONT_CANDIDATES:
        if path.exists():
            return path
    return None


def _markdown_to_pdf(markdown_text: str, output_path: Path) -> Path:
    """Chuyển Markdown sang PDF để PageIndex đọc được."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = _find_font()
    if font_path is not None:
        pdf.add_font("body", "", str(font_path))
        pdf.set_font("body", size=11)
    else:
        # Không có TTF thì mất dấu tiếng Việt, nhưng vẫn upload được.
        pdf.set_font("helvetica", size=11)

    for line in markdown_text.splitlines():
        # latin-1 chỉ là đường lui khi thiếu font Unicode.
        text = line if font_path is not None else line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    return output_path


def _prepare_pdf(source: Path, doc_type: str) -> Path:
    """Legal đã là PDF; news/legal Markdown được convert sang PDF tạm."""
    if source.suffix.lower() == ".pdf":
        return source
    target = PDF_DIR / f"{doc_type}__{source.stem}.pdf"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    return _markdown_to_pdf(source.read_text(encoding="utf-8"), target)


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    client = _client()
    doc_ids = _load_doc_ids()

    sources: list[tuple[str, Path, str]] = []
    # Legal đã có PDF gốc -> upload trực tiếp, không convert lại từ Markdown
    # (tránh đẩy cùng một nội dung lên PageIndex hai lần).
    for path in sorted(LEGAL_DIR.glob("*.pdf")):
        sources.append((f"legal/{path.name}", path, "legal"))
    for path in sorted((STANDARDIZED_DIR / "news").glob("*.md")):
        key = path.relative_to(STANDARDIZED_DIR).as_posix()
        sources.append((key, path, "news"))

    for key, path, doc_type in sources:
        cached = doc_ids.get(key)
        if cached:
            print(f"Skip (đã upload): {key} -> {cached}")
            continue
        try:
            pdf_path = _prepare_pdf(path, doc_type)
            response = client.submit_document(str(pdf_path))
        except Exception as error:
            print(f"Failed: {key} — {type(error).__name__}: {error}")
            continue

        doc_id = response.get("doc_id")
        if not doc_id:
            print(f"Failed: {key} — response thiếu doc_id: {response}")
            continue
        doc_ids[key] = doc_id
        _save_doc_ids(doc_ids)
        print(f"Uploaded: {key} -> {doc_id}")

    _save_doc_ids(doc_ids)
    print(f"\nĐã có {len(doc_ids)} document ID trong {DOC_IDS_PATH.name}")


def _nodes_from_retrieval(payload: dict) -> list[dict]:
    """Lấy danh sách node từ response retrieval (shape có thể đổi theo phiên bản)."""
    for key in ("nodes", "results", "retrieved_nodes", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _node_text(node: dict) -> str:
    for key in ("text", "content", "markdown", "summary"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    client = _client()
    doc_ids = _load_doc_ids()
    if not doc_ids:
        raise RuntimeError("Chưa upload tài liệu: chạy `python -m src.task8_pageindex_vectorless` trước.")

    collected: list[dict] = []
    for source_key, doc_id in doc_ids.items():
        if len(collected) >= top_k:
            break
        try:
            submitted = client.submit_query(doc_id, query)
            retrieval_id = submitted.get("retrieval_id")
            if not retrieval_id:
                continue

            deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
            payload: dict = {}
            while time.monotonic() < deadline:
                payload = client.get_retrieval(retrieval_id)
                if _nodes_from_retrieval(payload) or payload.get("status") in {
                    "completed",
                    "failed",
                    "ready",
                }:
                    break
                time.sleep(POLL_INTERVAL_SECONDS)
        except Exception as error:
            print(f"  PageIndex lỗi ở {source_key}: {type(error).__name__}: {error}")
            continue

        for node_index, node in enumerate(_nodes_from_retrieval(payload)):
            text = _node_text(node)
            if not text:
                continue
            rank = len(collected)
            collected.append(
                {
                    # node_id có thể thiếu hoặc trùng giữa các tài liệu, nên
                    # ghép thêm chỉ số để id luôn duy nhất.
                    "id": f"pageindex::{doc_id}::{node.get('node_id') or 'node'}-{node_index}",
                    "content": text,
                    "score": 1.0 / (1 + rank),
                    "metadata": {
                        "source": source_key,
                        "title": node.get("title") or source_key,
                        "doc_type": "legal" if source_key.startswith("legal/") else "news",
                        "url": "",
                        "chunk_index": int(node.get("page_index") or 0),
                    },
                    "retrieval_method": "pageindex",
                }
            )
            if len(collected) >= top_k:
                break

    # Contract yêu cầu score giảm dần; score ở trên đã giảm theo thứ tự thu thập.
    return collected[:top_k]


if __name__ == "__main__":
    upload_documents()