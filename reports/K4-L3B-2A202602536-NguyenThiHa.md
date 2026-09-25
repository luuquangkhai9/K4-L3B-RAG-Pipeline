# Báo cáo đóng góp cá nhân — Nguyễn Thị Hạ

## Thông tin

- Họ và tên: Nguyễn Thị Hạ
- Mã học viên: 2A202602536
- Nhóm: K4-L3B
- Repository/nhánh: `K4-L3B-RAG-Pipeline` / `nguyenha`
- Vai trò được phân công: giao diện người dùng (UI) và search

## Phần việc đã thực hiện

| Hạng mục | Đóng góp có thể đối chiếu | File/commit | Trạng thái |
| --- | --- | --- | --- |
| Giao diện hỏi đáp | Xây dựng màn hình chat IELTS Writing, câu hỏi gợi ý, bộ chọn `top_k` và Dense/Hybrid, lịch sử hội thoại, khu vực xem nguồn và điểm truy xuất. | `app.py`; `9308984`, `62b8fda` | Hoàn thành |
| Hiển thị nguồn và trạng thái | Gắn citation `[n]` với nguồn trong câu trả lời; hiển thị tên tài liệu, loại nguồn, phương pháp truy xuất, score, liên kết và nội dung chunk; thông báo khi hệ thống từ chối hoặc PageIndex không dùng được. | `app.py`; `9308984` | Hoàn thành |
| Search | Phát triển truy vấn dense và BM25, hợp nhất bằng RRF; mở rộng câu hỏi tiếng Việt bằng bản dịch tiếng Anh và đưa trace truy xuất vào pipeline để giao diện giải thích nguồn, score và fallback. | `src/task5_semantic_search.py`, `src/task6_lexical_search.py`, `src/task7_reranking.py`, `src/task9_retrieval_pipeline.py`; `9308984`, `62b8fda` | Hoàn thành |
| Giao diện đánh giá | Tạo tab A/B đọc kết quả từ `group_project/evaluation/results.json`, trình bày chỉ số, biểu đồ bốn metric và các trường hợp cần xem lại. | `ui_evaluation.py`, `app.py`; `62b8fda` | Hoàn thành |

## Quyết định kỹ thuật quan trọng

1. **Tách phần chat và đánh giá thành hai tab.** Người dùng hỏi đáp trong `app.py`; tab đánh giá gọi `render_evaluation()` từ `ui_evaluation.py` để đọc kết quả đã lưu. Cách này giúp giao diện phản ánh đúng một lượt đánh giá và tránh chạy lại các yêu cầu API khi mở trang; dữ liệu hiển thị phụ thuộc vào file kết quả hiện có.
2. **Cho xem nội dung chunk sau mỗi citation.** Mỗi nguồn có tên, score, URL nếu có và phần nội dung có thể mở rộng. Người xem demo có thể kiểm tra câu trả lời với chứng cứ; giao diện cần thêm diện tích cho bảng chi tiết nguồn.

## Kiểm tra và kết quả

- Đối chiếu luồng trong `app.py`: câu hỏi được đưa tới `generate_with_trace()`, câu trả lời và trace được lưu vào `st.session_state`, sau đó `render_answer()` và `render_details()` hiển thị lại nguồn và trạng thái.
- `retrieve_with_trace()` trên nhánh dùng cả dense và BM25, gộp các danh sách bằng RRF một lần; khi câu hỏi tiếng Việt được dịch, trace ghi bản dịch và cosine cao nhất để hiển thị trong UI. Fallback quyết định theo cosine gốc của dense.
- `group_project/evaluation/results.json` trên nhánh có 18 câu golden và ba cấu hình A/B/C. Tab đánh giá đọc trực tiếp file này; điểm trung bình lưu trong file là 0,8006 cho A và 0,8204 cho B. Đây là số liệu của lượt đánh giá đã lưu, không phải phép đo riêng của UI.
- Giao diện có nhánh xử lý ngoại lệ để hiển thị thông báo an toàn khi pipeline lỗi, thay vì làm mất phiên chat.

## Hạn chế và hướng tiếp theo

- UI phụ thuộc vào ChromaDB, mô hình và file kết quả đánh giá có sẵn trong môi trường chạy; chưa có dữ liệu thì phần thống kê hoặc biểu đồ không thể hiện đủ.
- Nếu tiếp tục phát triển, tôi sẽ kiểm tra giao diện trên màn hình nhỏ và thêm liên kết trực tiếp đến trang hoặc số trang của tài liệu PDF trong mỗi citation.

## Xác nhận đóng góp

Nội dung trên mô tả phần giao diện và search được giao, có thể đối chiếu bằng file và commit trên nhánh `nguyenha`.

- Ngày: 25/09/2026
- Thành viên: Nguyễn Thị Hạ
