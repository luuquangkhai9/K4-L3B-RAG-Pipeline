import asyncio
import os
import json
import datetime
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import html2text

ARTICLE_URLS = [
    "https://ielts.org/news-and-insights/updates-to-ielts-test-delivery",
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-ielts-writing-skills",
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-best-ielts-writing-tips-for-high-scoring-essays",
    "https://ielts.idp.com/prepare/article-5-tips-to-maximise-your-ielts-writing-score",
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-ielts-writing-task-1-2-how-to-write-clearly"
]

OUTPUT_DIR = Path("data/landing/news")

async def crawl_and_format():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        for idx, url in enumerate(ARTICLE_URLS, 1):
            print(f"Crawling ({idx}/5): {url}")
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                title = await page.title()
                html_content = await page.content()
                
                # Parse HTML và chuyển sang Markdown sạch
                soup = BeautifulSoup(html_content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.extract()
                
                main_content = soup.find("article") or soup.find("main") or soup
                markdown_text = h.handle(str(main_content)).strip()

                # Cấu trúc JSON đúng chuẩn yêu cầu của bạn
                data = {
                    "url": url,
                    "title": title,
                    "date_crawled": datetime.datetime.now().isoformat(),
                    "content_markdown": markdown_text
                }

                out_path = OUTPUT_DIR / f"article_{idx:02d}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f" Saved & Formatted: {out_path.name}")

            except Exception as e:
                print(f" Lỗi crawl {url}: {e}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(crawl_and_format())