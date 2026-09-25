"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Chủ đề nhóm: IELTS Writing — band descriptors, tiêu chí chấm điểm, bài mẫu.

Quy ước:
    - PDF/DOCX trong data/landing/legal/ -> data/standardized/legal/<stem>.md
    - JSON trong data/landing/news/      -> data/standardized/news/<stem>.md

Mỗi file Markdown mở đầu bằng khối metadata để Task 4 đọc lại được nguồn,
tiêu đề và URL gốc; phần thân giữ nguyên nội dung đã convert.

Chạy lại an toàn: bỏ qua file Markdown đã mới hơn hoặc bằng file nguồn.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from .task1_collect_legal_docs import SOURCES as LEGAL_SOURCES


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

LEGAL_SUFFIXES = {".pdf", ".doc", ".docx"}

# Khối metadata ở đầu mỗi file Markdown, dạng "**Key:** value".
META_PATTERN = re.compile(r"^\*\*(?P<key>[^:*]+):\*\*\s*(?P<value>.*)$", re.MULTILINE)

DOC_TYPE_LABEL = {"legal": "legal", "news": "news"}


def _header(title: str, doc_type: str, source: str, url: str) -> str:
    return (
        f"# {title}\n\n"
        f"**Doc type:** {doc_type}\n"
        f"**Source:** {source}\n"
        f"**URL:** {url}\n"
        f"**Retrieved:** {datetime.now().date().isoformat()}\n\n"
        f"---\n\n"
    )


def parse_metadata(text: str) -> dict[str, str]:
    """Đọc khối metadata ở đầu file Markdown do Task 3 tạo."""
    return {
        match.group("key").strip().lower(): match.group("value").strip()
        for match in META_PATTERN.finditer(text[:1024])
    }


def _title_of(path: Path) -> str:
    """Lấy dòng tiêu đề H1 đầu tiên, fallback về tên file."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _is_fresh(source: Path, output: Path) -> bool:
    return output.exists() and output.stat().st_mtime >= source.stat().st_mtime


def _clean_cell(cell: object) -> str:
    """Gộp khoảng trắng và xuống dòng bên trong một ô của bảng PDF."""
    return " ".join(str(cell or "").split())


# Bảng band descriptors có 5 cột: band + 4 tiêu chí chấm.
BAND_TABLE_MIN_COLUMNS = 5


def extract_band_table_sections(path: Path) -> str:
    """Trích bảng tiêu chí chấm thành mục theo từng Band.

    MarkItDown làm phẳng bảng thành text trộn lẫn các cột, khiến số band dính
    vào mô tả của cột khác (ví dụ "9 Any lapses in coherence... sophisticated
    control of lexical features"). pdfplumber giữ đúng cấu trúc ô, nên mỗi band
    trở thành một mục liền mạch và chunk theo band cho retrieval chính xác hơn.

    Trả về chuỗi rỗng nếu file không chứa bảng dạng này.
    """
    import pdfplumber

    # task_label -> criteria -> [(band, values)]
    groups: dict[str, tuple[list[str], list[tuple[str, list[str]]]]] = {}

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if not table or len(table) < 2 or len(table[0]) < BAND_TABLE_MIN_COLUMNS:
                    continue
                criteria = [_clean_cell(cell) for cell in table[0][1:]]
                if not criteria[0]:
                    continue

                task_label = (
                    "Writing Task 1"
                    if criteria[0].lower().startswith("task achievement")
                    else "Writing Task 2"
                )
                rows = groups.setdefault(task_label, (criteria, []))[1]

                for row in table[1:]:
                    band = _clean_cell(row[0])
                    if not band.isdigit():
                        continue
                    values = [_clean_cell(cell) for cell in row[1:]]
                    if any(values):
                        rows.append((band, values))

    if not groups:
        return ""

    parts: list[str] = []
    for task_label in sorted(groups):
        criteria, rows = groups[task_label]
        parts.append(f"## {task_label} — Band Descriptors\n")
        for band, values in sorted(rows, key=lambda item: -int(item[0])):
            parts.append(f"### Band {band}\n")
            for name, value in zip(criteria, values):
                if value:
                    parts.append(f"- **{name}:** {value}")
            parts.append("")
    return "\n".join(parts).strip()


def convert_legal_docs() -> int:
    """Convert PDF/DOCX vào standardized/legal."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()
    converted = 0

    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in LEGAL_SUFFIXES or path.name.startswith("."):
            continue
        output = output_dir / f"{path.stem}.md"
        if _is_fresh(path, output):
            print(f"Skip (up to date): {output.name}")
            continue

        # Tài liệu dạng bảng (band descriptors) phải trích bằng pdfplumber,
        # nếu không sẽ bị làm phẳng sai thứ tự cột.
        body = ""
        if path.suffix.lower() == ".pdf":
            try:
                body = extract_band_table_sections(path)
            except Exception as error:
                print(f"  Bỏ qua trích bảng ({type(error).__name__}: {error})")

        if not body:
            try:
                result = converter.convert(str(path))
                body = (result.text_content or "").strip()
            except Exception as error:
                print(f"Failed: {path.name} — {error}")
                continue

        if not body:
            print(f"Failed: {path.name} — không trích được nội dung")
            continue

        output.write_text(
            _header(
                title=path.stem.replace("-", " ").title(),
                doc_type=DOC_TYPE_LABEL["legal"],
                source=path.name,
                url=LEGAL_SOURCES.get(path.name, ""),
            )
            + body
            + "\n",
            encoding="utf-8",
        )
        converted += 1
        print(f"Saved: {output.name} — {len(body):,} ký tự")

    return converted


def convert_news_articles() -> int:
    """Convert JSON vào standardized/news."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    for path in sorted(news_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        output = output_dir / f"{path.stem}.md"
        if _is_fresh(path, output):
            print(f"Skip (up to date): {output.name}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            print(f"Failed: {path.name} — JSON lỗi: {error}")
            continue

        body = str(data.get("content_markdown", "")).strip()
        if not body:
            print(f"Failed: {path.name} — content_markdown rỗng")
            continue

        url = str(data.get("url", "")).strip()
        output.write_text(
            _header(
                title=str(data.get("title", "")).strip() or path.stem,
                doc_type=DOC_TYPE_LABEL["news"],
                # Schema landing/news chỉ có url, nên suy domain ra để làm nhãn nguồn.
                source=urlparse(url).netloc or path.stem,
                url=url,
            )
            + body
            + "\n",
            encoding="utf-8",
        )
        converted += 1
        print(f"Saved: {output.name} — {len(body):,} ký tự")

    return converted


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    legal = convert_legal_docs()
    news = convert_news_articles()
    print(f"\nSaved Markdown to: {OUTPUT_DIR} ({legal} legal, {news} news)")


if __name__ == "__main__":
    convert_all()