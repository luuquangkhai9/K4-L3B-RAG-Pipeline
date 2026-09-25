# Báo cáo đóng góp cá nhân — Phạm Hương Giang

## Thông tin

- Họ và tên: Phạm Hương Giang
- Mã học viên: 2A202602359
- Nhóm: K4-L3B
- Repository/nhánh: `K4-L3B-RAG-Pipeline` / `giangpahm`
- Vai trò được phân công: dữ liệu

## Phần việc đã thực hiện

| Hạng mục | Đóng góp có thể đối chiếu | File/commit | Trạng thái |
| --- | --- | --- | --- |
| Bài viết IELTS Writing | Tạo năm bản ghi `article_01.json`–`article_05.json` gồm URL, tiêu đề, thời điểm thu thập và nội dung Markdown. | `data/landing/news/`, `src/task2_crawl_news.py`; `a4337ce`, `2593b29` | Hoàn thành |
| Crawler và làm sạch | Dùng Playwright mở trang, BeautifulSoup bỏ các thành phần điều hướng/script, `html2text` tạo văn bản Markdown; lưu JSON bằng UTF-8. | `src/task2_crawl_news.py`; `2593b29` | Hoàn thành |
| Dữ liệu chuẩn hóa | Đưa bốn tài liệu PDF IELTS Writing và năm bài viết sang Markdown để nhóm dùng cho chunking và search; phần PDF có nhãn trang. | `data/standardized/`, `src/task3_convert_markdown.py`; `a4337ce` | Hoàn thành đối với dữ liệu đã nộp |

## Quyết định kỹ thuật quan trọng

1. **Giữ dữ liệu gốc và dữ liệu đã chuẩn hóa ở hai thư mục.** JSON/PDF trong `data/landing/` còn Markdown trong `data/standardized/`; nhờ vậy có thể đối chiếu nguồn khi nội dung trích xuất sai. Đổi lại cần bảo đảm script chuyển đổi đọc đúng schema của JSON.
2. **Giữ URL và thời điểm thu thập trong JSON.** Các trường này giúp truy vết bài viết IELTS/IDP và phân biệt nội dung trang theo thời điểm; dữ liệu web có thể thay đổi khi chạy crawler lần sau.

## Kiểm tra và kết quả

- Commit `a4337ce` có năm JSON bài viết và chín Markdown chuẩn hóa (bốn legal, năm news); nhánh cũng chứa bốn PDF nguồn trong `data/landing/legal/`.
- `src/task2_crawl_news.py` có danh sách năm URL IELTS/IDP và ghi các trường `url`, `title`, `date_crawled`, `content_markdown`.
- Đã phát hiện một điểm cần sửa trước khi tái tạo dữ liệu: `convert_news()` trên nhánh hiện đọc các khóa `html`, `raw_html` hoặc `content`, trong khi crawler ghi `content_markdown`. Các Markdown đã nộp là hiện vật có trong commit, nhưng chạy lại Task 3 bằng đúng JSON hiện tại có thể làm rỗng nội dung news.

## Hạn chế và hướng tiếp theo

- `src/task1_collect_legal_docs.py` trên nhánh vẫn là stub; bốn PDF được lưu trong repo nhưng module này chưa tự tải lại chúng.
- Tôi sẽ ưu tiên sửa Task 3 đọc trực tiếp `content_markdown`, sau đó chạy lại và so độ dài/nội dung của năm Markdown với JSON gốc.

## Xác nhận đóng góp

Nội dung trên phản ánh phần dữ liệu được giao và giới hạn tái tạo đã kiểm tra trên nhánh `giangpahm`.

- Ngày: 25/09/2026
- Thành viên: Phạm Hương Giang
