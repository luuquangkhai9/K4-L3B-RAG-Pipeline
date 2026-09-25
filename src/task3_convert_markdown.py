import os
import json
import re
import glob
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup
import html2text

LEGAL_DIR = Path("data/landing/legal")
NEWS_DIR = Path("data/landing/news")
STANDARDIZED_DIR = Path("data/standardized")

def clean_text(text: str) -> str:
    """Loại bỏ footer lặp lại, sửa ngắt dòng và chuẩn hóa khoảng trắng."""
    text = re.sub(r'IELTS is jointly owned by.*?(Cambridge Assessment English|\.org)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    # Ghép các từ bị ngắt dòng do ngắt trang PDF (ví dụ: asses-\nsment -> assessment)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Thu gọn khoảng trắng thừa và dòng trống
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def convert_legal_docs():
    """Đọc các file PDF trong data/landing/legal và lưu sang data/standardized/*.md"""
    STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(LEGAL_DIR.glob("*.pdf"))
    
    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(f"## Page {i + 1}\n\n{clean_text(page_text)}")
        
        full_content = f"# {pdf_path.stem.replace('-', ' ').title()}\n\n" + "\n\n".join(extracted_pages)
        
        out_file = STANDARDIZED_DIR / f"{pdf_path.stem}.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_content)
        print(f"Converted Legal: {out_file.name}")

def convert_news():
    """Đọc các file JSON trong data/landing/news và chuyển HTML sang data/standardized/*.md"""
    STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
    json_files = list(NEWS_DIR.glob("*.json"))
    
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Hỗ trợ linh hoạt các key html, raw_html hoặc content
        html_content = data.get("html") or data.get("raw_html") or data.get("content", "")
        title = data.get("title", json_file.stem)
        url = data.get("url", "")
        
        soup = BeautifulSoup(html_content, "html.parser")
        # Loại bỏ các thành phần rác như thanh điều hướng, script, style
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.extract()

        # Ưu tiên lấy phần thẻ bài viết chính nếu có
        main_content = soup.find("article") or soup.find("main") or soup
        md_body = h.handle(str(main_content))
        md_body = clean_text(md_body)

        full_content = f"# {title}\n\nSource: {url}\n\n{md_body}"
        
        out_file = STANDARDIZED_DIR / f"{json_file.stem}.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_content)
        print(f"Converted News: {out_file.name}")

def convert_all():
    convert_legal_docs()
    convert_news()

if __name__ == "__main__":
    convert_all()