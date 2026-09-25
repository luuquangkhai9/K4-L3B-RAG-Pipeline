"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Dùng requests + BeautifulSoup + MarkItDown (không cần browser). Nếu trang
cần render JS, đổi sang Crawl4AI (python -m playwright install chromium).
"""

import asyncio
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://ielts.org/news-and-insights/updates-to-ielts-test-delivery",
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-ielts-writing-skills",
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-best-ielts-writing-tips-for-high-scoring-essays",
    "https://ielts.idp.com/prepare/article-5-tips-to-maximise-your-ielts-writing-score",
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-ielts-writing-task-1-2-how-to-write-clearly",
]


async def crawl_article(url: str) -> dict:
    """Tải HTML bằng requests, lấy phần nội dung chính rồi đổi sang Markdown."""
    import io
    from datetime import datetime

    import requests
    from bs4 import BeautifulSoup
    from markitdown import MarkItDown

    response = await asyncio.to_thread(
        requests.get, url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else "Unknown"
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe"]):
        tag.decompose()
    candidates = soup.find_all(["article", "main"]) or [soup.body or soup]
    main = max(candidates, key=lambda tag: len(tag.get_text(strip=True)))
    if len(main.get_text(strip=True)) < 1500 and soup.body:
        main = soup.body
    markdown = MarkItDown().convert_stream(
        io.BytesIO(str(main).encode("utf-8")), file_extension=".html"
    ).text_content.strip()
    if not markdown:
        raise RuntimeError("Empty content")
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": markdown,
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
