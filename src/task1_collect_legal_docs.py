"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

_IELTS = "https://ielts.org/cdn/ielts-guides/"

# URL = None: file đã commit sẵn trong repo, chỉ giữ lại (không tải).
SOURCES = {
    "ielts-writing-band-descriptors.pdf": _IELTS + "ielts-writing-band-descriptors.pdf",
    "ielts-writing-key-assessment-criteria.pdf": _IELTS + "ielts-writing-key-assessment-criteria.pdf",
    "ielts-academic-writing-example-responses-to-parts-1-and-2-with-band-scores-and-examiner-comments.pdf": None,
    "ielts-academic-writing-access-arrangement-modified-large-print-question-paper.pdf": None,
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    import requests

    for filename, url in SOURCES.items():
        target = DATA_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"Skip (exists): {filename}")
            continue
        if not url:
            print(f"Missing (no URL): {filename}")
            continue
        try:
            response = requests.get(
                url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError("Response is not a PDF")
            target.write_bytes(response.content)
            print(f"Saved: {target}")
        except Exception as error:
            print(f"Failed: {filename} — {error}")


if __name__ == "__main__":
    setup_directory()
    download_documents()
