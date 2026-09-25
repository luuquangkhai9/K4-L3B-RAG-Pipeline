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

import base64
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


OCR_PROMPT = (
    "Chép lại chính xác toàn bộ văn bản trong ảnh này thành Markdown, giữ nguyên ngôn ngữ gốc. "
    "Tiêu đề ở đầu trang (vd. tên phần, 'Writing Task 2 Band Descriptors') phải giữ thành heading Markdown '#'. "
    "Bảng nhiều cột phải chép thành bảng Markdown, mỗi ô đúng cột của nó. "
    "Không thêm lời giải thích."
)

# PDF có bảng nhiều cột: text layer bị trộn cột nên chép lại bằng vision LLM.
VISION_PDFS = {"ielts-writing-band-descriptors.pdf"}

# Nguồn gốc của tài liệu legal, ghi vào header để trích dẫn.
LEGAL_SOURCES = {
    "ielts-writing-band-descriptors.pdf": "https://ielts.org/cdn/ielts-guides/ielts-writing-band-descriptors.pdf",
    "ielts-writing-key-assessment-criteria.pdf": "https://ielts.org/cdn/ielts-guides/ielts-writing-key-assessment-criteria.pdf",
}

# Khối cuối trang IDP (tác giả, chia sẻ, bài liên quan) không phải nội dung bài.
_NEWS_FOOTER = re.compile(r"^#{1,6} (WRITTEN BY|Chia sẻ bài viết|Share this article)\b.*", re.M | re.S)
_IMAGE_LINE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$", re.M)
_CONTENT_TAGS = re.compile(r"^#{1,6} Content tags\s*\n+[^\n]*\n", re.M)


def structure_headings(text: str) -> str:
    """Đổi các dòng cấu trúc của tài liệu bài mẫu thành heading Markdown.

    PDF chỉ có dòng trơn "Sample Academic Writing Part 2" / "Candidate Response 1" /
    "Examiner comment" + "Band 7.5". Khi chunk cắt giữa "Band 7.5" và lời nhận xét,
    chunk nhận xét mất nhãn band. Heading đủ ngữ cảnh giúp splitter cắt đúng chỗ.
    """
    lines = text.splitlines()
    output: list[str] = []
    part = response = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.fullmatch(r"Sample Academic Writing Part \d", line):
            part = line
        elif re.fullmatch(r"Candidate Response \d", line):
            response = line
            output.append(f"## {part}, {response}".strip(", "))
            i += 1
            continue
        elif line == "Examiner comment":
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            band = lines[j].strip() if j < len(lines) and re.fullmatch(r"Band [\d.]+", lines[j].strip()) else ""
            label = f"{part}, {response}".strip(", ")
            output.append(f"### Examiner comment — {label} — {band}".rstrip(" —"))
            i = j + 1 if band else i + 1
            continue
        if line != part:
            output.append(lines[i])
        i += 1
    return "\n".join(output)


def clean_news(body: str) -> str:
    """Bỏ footer, ảnh và tag điều hướng khỏi bài crawl."""
    body = _NEWS_FOOTER.sub("", body)
    body = _CONTENT_TAGS.sub("", body)
    body = _IMAGE_LINE.sub("", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def ocr_pdf(path: Path) -> str:
    """OCR PDF scan bằng vision LLM qua API (openai hoặc gemini), không cần model local."""
    import io

    import pypdfium2

    provider = os.getenv("OCR_PROVIDER") or (
        "openai" if os.getenv("OPENAI_API_KEY") else "gemini"
    )
    pages = []
    for page in pypdfium2.PdfDocument(str(path)):
        buffer = io.BytesIO()
        page.render(scale=2).to_pil().convert("RGB").save(buffer, "JPEG", quality=85)
        pages.append(buffer.getvalue())

    texts = []
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(max_retries=8)  # ảnh tốn nhiều token, dễ chạm rate limit TPM
        # gpt-4o-mini lệch cột trên bảng band descriptors dày chữ; gpt-4o chép đúng.
        model = os.getenv("OCR_MODEL", "gpt-4o")
        for image in pages:
            data_url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}],
            )
            texts.append(response.choices[0].message.content or "")
    else:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model = os.getenv("OCR_MODEL", "gemini-2.5-flash")
        for image in pages:
            response = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=image, mime_type="image/jpeg"), OCR_PROMPT],
            )
            texts.append(response.text or "")
    # LLM hay bọc output trong ```markdown ... ```; bỏ các dòng fence đó.
    texts = [re.sub(r"^```(?:markdown)?\s*$", "", t, flags=re.M) for t in texts]
    # Bỏ dấu cách đệm cho thẳng cột trong bảng (tốn chỗ trong chunk).
    texts = [re.sub(r" {2,}\|", " |", re.sub(r"-{4,}", "---", t)) for t in texts]
    return "\n\n".join(t.strip() for t in texts)


def convert_legal_docs() -> None:
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()
    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        target = output_dir / f"{path.stem}.md"
        text = ""
        if path.name in VISION_PDFS:
            if target.exists() and target.stat().st_size > 0:
                print(f"Skip (vision cached): {path.name}")
                continue
            print(f"Table layout, vision OCR: {path.name}")
            try:
                text = ocr_pdf(path).strip()
            except Exception as error:
                print(f"OCR failed: {path.name} — {error}")
        if not text:
            try:
                text = converter.convert(str(path)).text_content.strip()
            except Exception as error:
                print(f"Failed: {path.name} — {error}")
                continue
        if not text and path.suffix.lower() == ".pdf":
            print(f"No text layer, OCR: {path.name}")
            try:
                text = ocr_pdf(path).strip()
            except Exception as error:
                print(f"OCR failed: {path.name} — {error}")
        # Bỏ bảng rỗng (khung kẻ dòng của trang giấy làm bài), dòng chỉ có số
        # trang, rồi gộp dòng trống.
        text = re.sub(r"^\|[ |]*\|[ \t]*\n\|[ |:-]*\|[ \t]*$", "", text, flags=re.M)
        text = re.sub(r"^\|[ |]*\|[ \t]*$", "", text, flags=re.M)
        text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.M)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        # pdfminer không map được glyph en-dash trong khoảng số ("0–9" thành "0?9").
        text = re.sub(r"(?<=\d)\?(?=\d)", "–", text)
        text = structure_headings(text)
        if not text:
            print(f"Skip (empty): {path.name}")
            continue
        title = path.stem.replace("-", " ").title().replace("Ielts", "IELTS")
        header = f"# {title}\n\n**File:** {path.name}\n\n"
        if path.name in LEGAL_SOURCES:
            header += f"**Source:** {LEGAL_SOURCES[path.name]}\n\n"
        target.write_text(header + "---\n\n" + text + "\n", encoding="utf-8")
        print(f"Converted: {path.name}")


def convert_news_articles() -> None:
    import json

    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        body = clean_news(data.get("content_markdown", ""))
        if not body:
            print(f"Skip (empty): {path.name}")
            continue
        header = (
            f"# {data['title']}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n---\n\n"
        )
        (output_dir / f"{path.stem}.md").write_text(
            header + body + "\n", encoding="utf-8"
        )
        print(f"Converted: {path.name}")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
