"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    # TODO: Thêm ít nhất 5 public URL.
    "https://ielts.org/news-and-insights/updates-to-ielts-test-delivery",
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-ielts-writing-skills",
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-best-ielts-writing-tips-for-high-scoring-essays",
    "https://ielts.idp.com/prepare/article-5-tips-to-maximise-your-ielts-writing-score",
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-ielts-writing-task-1-2-how-to-write-clearly"

]


from playwright.async_api import async_playwright
import datetime

async def crawl_article(url: str) -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Tạo context giả lập trình duyệt thật
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        try:
            # Chờ trang tải xong DOM
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            
            # Lấy tiêu đề bài viết
            title = await page.title()
            
            # Trích xuất nội dung bài viết chính
            content = await page.content()
            
            await browser.close()
            
            return {
                "url": url,
                "title": title,
                "html": content,
                "crawled_at": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            await browser.close()
            raise RuntimeError(f"Lỗi khi crawl {url}: {e}")

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
