# Tích hợp các nhánh vào main

## Cách chọn mã nguồn khi các nhánh cùng sửa một module

| Phần việc | Nguồn được dùng | Ghi chú |
| --- | --- | --- |
| Dữ liệu bài viết và crawler | `giangpahm` | Giữ HTML thô của Giang; bổ sung `date_crawled` và `content_markdown` từ dữ liệu đã chuẩn hóa cùng URL để đúng schema của Task 2/3. |
| Chuyển Markdown | `giangpahm` và bản sửa tích hợp | Giữ bước làm sạch của Giang; sửa đường dẫn ra `legal/` và `news/`, đọc `content_markdown`, có dự phòng PDF khi thiếu `pypdf`. |
| Tải PDF | `truong` | Task 1 ở nhánh Giang còn là stub; dùng module tải PDF đã triển khai ở nhánh Trường. |
| Dense, BM25, RRF | `truong` | Task 5–7. |
| Search mở rộng và UI | `nguyenha` | Task 9 dịch câu hỏi, trace truy xuất; Task 10 tương thích UI; `app.py` và `ui_evaluation.py`. |
| PageIndex fallback | `khailq` | Task 8; Task 9 của Hạ gọi fallback này. |
| Đánh giá chính | `khailq` | Bộ 15 câu và báo cáo chính tại `group_project/evaluation/RESULT.md`. |

## Lưu kết quả riêng của mỗi nhánh

- Khải: `golden_dataset_khailq.json`, `evaluation_results.json` và `RESULT.md`.
- Hạ: `golden_dataset_nguyenha.json`, `results.json`, `metric_selftest.json`; tab A/B đọc `results.json`.
- Trường: `golden_dataset_truong.json`, `eval_runs.json`, `RESULT-truong.md`.

Các bộ điểm được sinh trên corpus, mô hình và cách chấm của từng nhánh, nên không so trực tiếp số điểm của ba lượt chạy. `golden_dataset.json` giữ bộ 15 câu dùng cho kiểm tra acceptance của sản phẩm tích hợp.

Các Markdown chuẩn hóa hiện có được giữ để bảo toàn bản dữ liệu đã dùng trong lượt đánh giá của Khải. Task 3 tích hợp dùng cách làm sạch của Giang và đã được kiểm tra tạo đủ 4 legal + 5 news; khi làm mới dữ liệu, chạy lại Task 3 rồi Task 4 để cập nhật vector index.

## Chạy lại

```powershell
python -m pip install -e ".[dev]"
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown
python -m src.task4_chunking_indexing
python -m pytest -q
streamlit run app.py
```

`task2_crawl_news.py` truy cập web thật, nên chỉ cần chạy lại khi muốn làm mới bài viết. Sau khi thay đổi Markdown, cần chạy lại Task 4 để ChromaDB khớp nội dung.
