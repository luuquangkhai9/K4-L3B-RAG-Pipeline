# Báo cáo đóng góp cá nhân — Lê Thanh Trường

## Thông tin

- Họ và tên: Lê Thanh Trường
- Mã học viên: 2A202602492
- Nhóm: K4 — L3B
- Kho lưu trữ/nhánh: `K4-L3B-RAG-Pipeline` / `truong`
- Vai trò được phân công: search (dense, BM25 và RRF)

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| **Task 5** — Tìm kiếm ngữ nghĩa | Embed câu hỏi bằng đúng `embed_texts()` của Task 4; truy vấn ChromaDB; đổi cosine distance thành similarity; trả `SearchResult` đã sắp xếp giảm dần và không vượt `top_k` | `src/task5_semantic_search.py`, commit `5a0708a` | Hoàn thành |
| **Task 6** — Tìm kiếm từ khóa (BM25) | Tách token cho tiếng Việt; dựng BM25 trên cùng tập chunk với Task 5; lọc theo token trùng với câu hỏi thay vì theo score | `src/task6_lexical_search.py`, commit `5a0708a` | Hoàn thành |
| **Task 7** — Hợp nhất RRF | Gộp nhiều bảng xếp hạng theo `sum(1/(k+rank))`, khử trùng theo `id`, đánh dấu `retrieval_method="hybrid"` | `src/task7_reranking.py`, commit `5a0708a` | Hoàn thành |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Ở Task 6, lọc kết quả theo **token trùng với query** thay vì lọc theo `score > 0` như gợi ý trong stub.
   **Lý do/evidence:** `BM25Okapi` trả IDF = 0 khi một term xuất hiện ở đúng một nửa số tài liệu — với corpus 2 tài liệu, `log((2-1+0.5)/(1+0.5)) = log(1) = 0`, nên mọi score đều bằng 0 và bộ lọc `score <= 0` xóa sạch kết quả. Đo được: `scores = [0. 0.]` và `idf = {'tuition': 0.0, 'fee': 0.0, ...}` trong khi tài liệu chứa đúng hai từ khoá của query vẫn bị loại. Sau khi chuyển sang lọc theo token trùng, tài liệu chứa "tuition fee" được trả về đúng.
   **Đánh đổi:** Phải tokenize corpus một lần nữa ở bước lọc (đã gộp chung một lần tokenize để không lặp), và kết quả phụ thuộc chất lượng tokenizer thay vì để BM25 tự quyết.

2. **Quyết định:** Task 6 tái tạo corpus bằng `chunk_documents(load_documents())` của Task 4 thay vì lưu một bản corpus riêng.
   **Lý do/evidence:** BM25 và dense search bắt buộc phải chấm trên cùng tập chunk, nếu không `id` sẽ lệch và bước RRF ở Task 7 khử trùng theo `id` sẽ sai. Tái tạo lại là hàm thuần tuý (không gọi embedding) nên không tốn thêm chi phí API mà vẫn đảm bảo `id` và nội dung khớp đúng với những gì đã index vào ChromaDB.
   **Đánh đổi:** Mỗi lần gọi `lexical_search` phải dựng lại BM25 index; chấp nhận được với corpus 330 chunk nhưng sẽ thành điểm nghẽn nếu corpus lớn hơn nhiều.

## Kiểm thử và kết quả

- `pytest tests/test_contracts.py` — **15/15 đạt** trong lượt kiểm tra ghi ở báo cáo gốc, gồm 3 phép kiểm tra đúng phần tôi làm:
  - `test_semantic_search_uses_shared_embedding_and_contract`: xác nhận Task 5 dùng chung `embed_texts()` của Task 4 và trả đúng schema `dense`, có sort và không vượt `top_k`.
  - `test_lexical_search_returns_bm25_contract`: trả đúng schema `bm25`, không trùng `id`; tài liệu chứa từ khoá xếp đầu.
  - `test_rrf_uses_rank_deduplicates_and_marks_hybrid`: `chunk-1` xuất hiện ở cả hai bảng xếp hạng nên được gộp làm một với `score = 1/62 + 1/61` và đứng đầu — đúng công thức RRF, chứng minh không cộng trực tiếp cosine với BM25.
- **Đối chiếu thang đo:** đo trên 24 câu hỏi trong domain, cosine của dense nằm trong khoảng **0.470 – 0.752**; 7 câu ngoài domain chỉ đạt **0.121 – 0.274**. RRF score của kết quả hybrid chỉ khoảng **0.016 – 0.031** vì phản ánh thứ hạng chứ không phải độ tương đồng — đây là lý do Task 9 phải dùng cosine gốc của dense để quyết định fallback, không dùng RRF score.
- **Lỗi đã phát hiện và xử lý:** chính là lỗi IDF = 0 ở quyết định 1; phát hiện bằng cách in trực tiếp `bm25.idf` và `get_scores()` trên corpus 2 tài liệu.

## Điều còn hạn chế

- **Hạn chế cụ thể của phần tôi làm:** tokenizer của Task 6 chỉ tách theo từ và giữ nguyên hình thái, không stemming/lemmatization — nên "write" và "writing", hay "coherence" và "coherent", là hai token khác nhau và BM25 sẽ bỏ sót. Với corpus song ngữ Anh–Việt như của nhóm thì hạn chế này ảnh hưởng rõ nhất tới các câu hỏi tiếng Anh dùng biến thể từ.
- **Nếu có thêm thời gian:** thêm bước chuẩn hoá hình thái trước khi tokenize (ví dụ bỏ hậu tố tiếng Anh phổ biến) và đo lại context recall trên golden dataset để xem có cải thiện thật không.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/09/2026
- Tên thành viên: Lê Thanh Trường
