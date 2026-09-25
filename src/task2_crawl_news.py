"""
Task 2 — Crawl bài viết/thông báo về IELTS Writing.

Chủ đề nhóm: IELTS Writing — band descriptors, tiêu chí chấm điểm, bài viết mẫu.
Các bài viết bổ sung ngữ cảnh thực tế cho bộ tài liệu chính sách ở Task 1
(cập nhật cách thức thi, hướng dẫn viết Task 1/2, mẹo đạt điểm cao).

Cài browser trước khi chạy:
    python -m playwright install chromium

Lưu ý về chất lượng dữ liệu: crawl toàn trang sẽ kéo theo menu, banner quảng cáo
và footer — với ielts.org phần nhiễu chiếm ~80% nội dung. Vì vậy mặc định chỉ
lấy phần tử <main>; trang nào không có <main> thì crawl lại toàn trang.

Chạy lại an toàn: bài đã crawl thành công sẽ được bỏ qua.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    # Cập nhật cách thức tổ chức thi IELTS (ielts.org — nguồn chính thức).
    "https://ielts.org/news-and-insights/updates-to-ielts-test-delivery",
    # Hướng dẫn và mẹo viết từ IDP Vietnam.
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-ielts-writing-skills",
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-best-ielts-writing-tips-for-high-scoring-essays",
    "https://ielts.idp.com/prepare/article-5-tips-to-maximise-your-ielts-writing-score",
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-ielts-writing-task-1-2-how-to-write-clearly",
]

MIN_CONTENT_CHARS = 500

# Chỉ lấy thân bài, bỏ menu/header/footer.
MAIN_CONTENT_SELECTOR = "main"

# Một số toà soạn trả trang chặn cho User-Agent mặc định của requests.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _markdown_to_text(result) -> str:
    """Lấy Markdown thô từ kết quả crawl4ai (kiểu dữ liệu đổi theo phiên bản)."""
    markdown = getattr(result, "markdown", None)
    if markdown is None:
        return ""
    # crawl4ai >= 0.6 trả MarkdownGenerationResult thay vì str.
    raw = getattr(markdown, "raw_markdown", None)
    if isinstance(raw, str):
        return raw
    return markdown if isinstance(markdown, str) else str(markdown)


def _title_from(result, url: str) -> str:
    metadata = getattr(result, "metadata", None) or {}
    title = metadata.get("title") or ""
    if not str(title).strip():
        # Một số trang không set <title> trong metadata, thử lấy từ <head>.
        head = getattr(result, "html", "") or ""
        if "<title>" in head.lower():
            start = head.lower().index("<title>") + len("<title>")
            end = head.lower().find("</title>", start)
            if end > start:
                title = head[start:end]
    return str(title).strip()


async def _crawl_with_crawl4ai(url: str) -> dict:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

    async with AsyncWebCrawler() as crawler:
        # Ưu tiên chỉ lấy <main> để loại menu/banner/footer.
        result = await crawler.arun(
            url=url, config=CrawlerRunConfig(css_selector=MAIN_CONTENT_SELECTOR)
        )
        content = _markdown_to_text(result).strip()

        if len(content) < MIN_CONTENT_CHARS:
            # Trang không có <main> (hoặc <main> rỗng) -> lấy lại toàn trang.
            result = await crawler.arun(url=url)
            content = _markdown_to_text(result).strip()

    if not getattr(result, "success", True):
        raise RuntimeError(f"crawl4ai báo lỗi: {getattr(result, 'error_message', 'unknown')}")

    return {
        "url": url,
        "title": _title_from(result, url) or urlparse(url).netloc,
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "content_markdown": content,
    }


# Các vùng HTML thường chứa thân bài trên báo điện tử Việt Nam.
ARTICLE_SELECTORS = (
    "main",
    "article",
    "[itemprop='articleBody']",
    ".article-body",
    ".detail-content",
    ".content-detail",
    ".detail__content",
    ".post-content",
    ".maincontent",
    "#main-detail-body",
)


def _extract_article_html(html: str) -> str:
    """Lấy phần thân bài, bỏ menu/quảng cáo để Markdown sạch hơn."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    best = None
    for selector in ARTICLE_SELECTORS:
        node = soup.select_one(selector)
        if node is not None and (best is None or len(node.get_text(strip=True)) > len(best.get_text(strip=True))):
            best = node
    if best is None:
        best = soup.body or soup
    return str(best)


def crawl_article_http(url: str) -> dict:
    """Dự phòng khi không chạy được browser: tải HTML rồi tự chuyển Markdown."""
    import requests
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    return {
        "url": url,
        "title": title or urlparse(url).netloc,
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "content_markdown": markdownify(
            _extract_article_html(response.text), heading_style="ATX"
        ).strip(),
    }


async def crawl_article(url: str) -> dict:
    """Crawl một bài viết và trả về dict theo schema landing/news.

    Ưu tiên crawl4ai; nếu browser chưa cài hoặc crawl lỗi thì dùng HTTP thuần
    để pipeline không bị chặn bởi bước cài Playwright.
    """
    try:
        article = await _crawl_with_crawl4ai(url)
        if len(article["content_markdown"]) >= MIN_CONTENT_CHARS:
            return article
        print("  crawl4ai trả nội dung ngắn, thử lại bằng HTTP.")
    except Exception as error:
        print(f"  crawl4ai không dùng được ({type(error).__name__}: {error}). Dùng HTTP.")

    return crawl_article_http(url)


def _is_crawled(path: Path) -> bool:
    """Kiểm tra file đã crawl hợp lệ để chạy lại không tốn thời gian."""
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        bool(str(item.get("title", "")).strip())
        and len(str(item.get("content_markdown", ""))) >= MIN_CONTENT_CHARS
    )


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    for index, url in enumerate(ARTICLE_URLS, 1):
        output = DATA_DIR / f"article_{index:02d}.json"
        if _is_crawled(output):
            print(f"Skip (exists): {output.name}")
            continue
        try:
            article = await crawl_article(url)
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output.name} — {len(article['content_markdown']):,} ký tự")
        except Exception as error:
            failed.append(url)
            print(f"Failed: {url} — {error}")

    if failed:
        print(f"\n{len(failed)}/{len(ARTICLE_URLS)} bài crawl thất bại.")
    else:
        print(f"\nDone: {len(ARTICLE_URLS)} bài trong {DATA_DIR}")


if __name__ == "__main__":
    asyncio.run(crawl_all())