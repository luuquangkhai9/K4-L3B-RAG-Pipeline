"""Task 3: normalize IELTS Writing PDFs and articles to Markdown."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEGAL_DIR = ROOT / "data" / "landing" / "legal"
NEWS_DIR = ROOT / "data" / "landing" / "news"
STANDARDIZED_DIR = ROOT / "data" / "standardized"


def clean_text(text: str) -> str:
    """Join words broken by line wraps and normalize whitespace."""
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _write_markdown(path: Path, content: str) -> None:
    text = content.strip()
    if not text:
        raise ValueError(f"Refusing to write empty Markdown: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def _pdf_pages(path: Path) -> list[str]:
    """Use pypdf when available and pdfplumber as a fallback."""
    try:
        from pypdf import PdfReader
    except ImportError:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    else:
        pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
    return [f"## Page {number}\n\n{clean_text(text)}" for number, text in enumerate(pages, 1) if text.strip()]


def convert_legal_docs() -> None:
    sources = sorted(LEGAL_DIR.glob("*.pdf"))
    if not sources:
        raise FileNotFoundError(f"No legal PDFs in {LEGAL_DIR}")
    for source in sources:
        pages = _pdf_pages(source)
        if not pages:
            raise ValueError(f"No extractable text in {source.name}")
        title = source.stem.replace("-", " ").title()
        content = f"# {title}\n\n" + "\n\n".join(pages)
        _write_markdown(STANDARDIZED_DIR / "legal" / f"{source.stem}.md", content)
        print(f"Converted legal: {source.name}")


def convert_news() -> None:
    sources = sorted(NEWS_DIR.glob("*.json"))
    if not sources:
        raise FileNotFoundError(f"No news JSON in {NEWS_DIR}")
    for source in sources:
        article = json.loads(source.read_text(encoding="utf-8"))
        for key in ("url", "title", "date_crawled", "content_markdown"):
            if not str(article.get(key) or "").strip():
                raise ValueError(f"{source.name}: missing {key}")
        body = clean_text(article["content_markdown"])
        content = (
            f"# {article['title']}\n\n"
            f"**Source:** {article['url']}\n\n"
            f"**Crawled:** {article['date_crawled']}\n\n"
            f"---\n\n{body}"
        )
        _write_markdown(STANDARDIZED_DIR / "news" / f"{source.stem}.md", content)
        print(f"Converted news: {source.name}")


def convert_news_articles() -> None:
    """Compatibility alias for callers from the other branches."""
    convert_news()


def convert_all() -> None:
    convert_legal_docs()
    convert_news()


if __name__ == "__main__":
    convert_all()
