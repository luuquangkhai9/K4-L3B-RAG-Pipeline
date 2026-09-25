"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Chủ đề nhóm: IELTS Writing — band descriptors, tiêu chí chấm điểm, bài viết mẫu.

Nguồn: tài liệu chính thức của ielts.org (IELTS Partners: British Council,
IDP IELTS, Cambridge University Press & Assessment).

    1. Writing Band Descriptors (cập nhật 5/2023) — thang điểm 4 tiêu chí
       cho Task 1 và Task 2, cả Academic và General Training.
    2. Writing Key Assessment Criteria — giải thích 4 tiêu chí chấm.
    3. Sample Candidate Writing Responses kèm band score và nhận xét giám khảo.
    4. Đề bài Academic Writing bản Modified Large Print (access arrangement).

Chạy lại an toàn: file đã tải đủ lớn sẽ được bỏ qua, không tạo dữ liệu trùng.
"""

from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

_IELTS_CDN = "https://ielts.org/cdn"

SOURCES: dict[str, str] = {
    "ielts-writing-band-descriptors.pdf": (
        f"{_IELTS_CDN}/Guides/ielts-writing-band-descriptors.pdf"
    ),
    "ielts-writing-key-assessment-criteria.pdf": (
        f"{_IELTS_CDN}/ielts-guides/ielts-writing-key-assessment-criteria.pdf"
    ),
    "ielts-academic-writing-example-responses-to-parts-1-and-2-with-band-scores"
    "-and-examiner-comments.pdf": (
        f"{_IELTS_CDN}/computer-delivered-sample-tests-academic-writing/"
        "ielts-academic-writing-example-responses-to-parts-1-and-2-with-band-scores"
        "-and-examiner-comments.pdf"
    ),
    "ielts-academic-writing-access-arrangement-modified-large-print"
    "-question-paper.pdf": (
        f"{_IELTS_CDN}/ielts-access-arrangements-sample-tests/ielts-modified-large-print/"
        "ielts-academic-writing-access-arrangement-modified-large-print"
        "-question-paper.pdf"
    ),
}

# Một số site trả trang HTML lỗi cho User-Agent mặc định của requests.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# test_acceptance yêu cầu mỗi file > 1024 byte; dùng ngưỡng lớn hơn để phát hiện
# file cụt hoặc trang HTML lỗi bị lưu nhầm đuôi .pdf.
MIN_PDF_BYTES = 10_240


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents(force: bool = False) -> None:
    """Tải các PDF chính sách từ nguồn công khai.

    Bỏ qua file đã có nếu kích thước hợp lệ và ``force`` không bật, nhờ đó
    chạy lại pipeline không sinh dữ liệu trùng.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    for filename, url in SOURCES.items():
        target = DATA_DIR / filename
        if not force and target.exists() and target.stat().st_size >= MIN_PDF_BYTES:
            print(f"Skip (exists): {filename} — {target.stat().st_size:,} bytes")
            continue

        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
        except requests.RequestException as error:
            failed.append(filename)
            print(f"Failed: {filename} — {error}")
            continue

        content = response.content
        if not content.startswith(b"%PDF"):
            failed.append(filename)
            print(
                f"Failed: {filename} — không phải PDF "
                f"(nhận {len(content):,} byte, có thể là trang HTML chặn tải)"
            )
            continue
        if len(content) < MIN_PDF_BYTES:
            failed.append(filename)
            print(f"Failed: {filename} — file quá nhỏ ({len(content):,} bytes)")
            continue

        target.write_bytes(content)
        print(f"Saved: {filename} — {len(content):,} bytes")

    if failed:
        raise SystemExit(f"Tải thất bại {len(failed)} file: {', '.join(failed)}")
    print(f"Done: {len(SOURCES)} tài liệu trong {DATA_DIR}")


if __name__ == "__main__":
    setup_directory()
    download_documents()