"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_EXTENSIONS = {".pdf", ".doc", ".docx"}


def _write_markdown(path: Path, content: str) -> None:
    """Write a complete UTF-8 Markdown file, replacing the previous run."""
    text = content.strip()
    if not text:
        raise ValueError("Refusing to write empty Markdown")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _convert_without_markitdown(source: Path) -> str:
    """Extract PDF/DOCX text with lightweight libraries when MarkItDown is unavailable."""
    if source.suffix.lower() == ".pdf":
        import pdfplumber

        with pdfplumber.open(source) as pdf:
            pages = [
                f"## Page {index}\n\n{text.strip()}"
                for index, page in enumerate(pdf.pages, start=1)
                if (text := page.extract_text()) and text.strip()
            ]
        return "\n\n".join(pages)

    if source.suffix.lower() == ".docx":
        with zipfile.ZipFile(source) as archive:
            document_xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(document_xml)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for paragraph in root.findall(".//w:p", namespace):
            text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace))
            if text.strip():
                paragraphs.append(text.strip())
        return "\n\n".join(paragraphs)

    raise RuntimeError(
        f"Cannot convert {source.suffix} without MarkItDown; install a working MarkItDown runtime"
    )


def convert_legal_docs() -> None:
    """Convert all supported legal source documents to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    sources = sorted(
        path for path in legal_dir.iterdir()
        if path.is_file() and path.suffix.lower() in LEGAL_EXTENSIONS
    )
    if not sources:
        raise FileNotFoundError(f"No PDF/DOC/DOCX files found in {legal_dir}")

    output_dir = OUTPUT_DIR / "legal"
    try:
        from markitdown import MarkItDown

        converter = MarkItDown()
    except Exception as error:
        converter = None
        print(f"MarkItDown unavailable ({error}); using local PDF/DOCX extraction")
    errors = []
    for source in sources:
        try:
            if converter is not None:
                try:
                    content = converter.convert(str(source)).text_content
                except Exception:
                    content = _convert_without_markitdown(source)
            else:
                content = _convert_without_markitdown(source)
            _write_markdown(output_dir / f"{source.stem}.md", content)
            print(f"Converted legal: {source.name}")
        except Exception as error:
            errors.append(f"{source.name}: {error}")
    if errors:
        raise RuntimeError("Legal conversion failed:\n- " + "\n- ".join(errors))


def convert_news_articles() -> None:
    """Convert crawled article JSON files while retaining their provenance."""
    news_dir = LANDING_DIR / "news"
    sources = sorted(news_dir.glob("*.json"))
    if not sources:
        raise FileNotFoundError(f"No article JSON files found in {news_dir}")

    output_dir = OUTPUT_DIR / "news"
    required_fields = {"url", "title", "date_crawled", "content_markdown"}
    errors = []
    for source in sources:
        try:
            article = json.loads(source.read_text(encoding="utf-8"))
            missing = required_fields - article.keys()
            if missing:
                raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")
            values = {key: str(article[key]).strip() for key in required_fields}
            empty = [key for key, value in values.items() if not value]
            if empty:
                raise ValueError(f"Empty fields: {', '.join(sorted(empty))}")

            markdown = (
                f"# {values['title']}\n\n"
                f"**Source:** {values['url']}\n\n"
                f"**Crawled:** {values['date_crawled']}\n\n"
                "---\n\n"
                f"{values['content_markdown']}"
            )
            _write_markdown(output_dir / f"{source.stem}.md", markdown)
            print(f"Converted news: {source.name}")
        except Exception as error:
            errors.append(f"{source.name}: {error}")
    if errors:
        raise RuntimeError("News conversion failed:\n- " + "\n- ".join(errors))


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
