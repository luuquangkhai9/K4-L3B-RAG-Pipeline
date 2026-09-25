# Báo cáo đóng góp cá nhân — Lưu Quang Khải

## Thông tin

- Họ và tên: Lưu Quang Khải
- Mã học viên: 2A202602599
- Nhóm: K4-L3B
- Repository/nhánh: `K4-L3B-RAG-Pipeline` / `khailq`
- Vai trò được phân công: search, fallback và evaluation

## Phần việc đã thực hiện

| Hạng mục | Đóng góp có thể đối chiếu | File/commit | Trạng thái |
| --- | --- | --- | --- |
| Search và hợp nhất kết quả | Hoàn thiện truy vấn dense trên ChromaDB, BM25 trên cùng tập chunk và hợp nhất bằng RRF theo ID; trả kết quả theo schema chung. | `src/task5_semantic_search.py`, `src/task6_lexical_search.py`, `src/task7_reranking.py`; `0b43ec3` | Hoàn thành |
| Fallback và retrieval pipeline | Tích hợp PageIndex cho PDF, lưu document ID để tái sử dụng; Task 9 quyết định fallback theo cosine score gốc của dense và trả kết quả local khi dịch vụ ngoài lỗi. | `src/task8_pageindex_vectorless.py`, `src/task9_retrieval_pipeline.py`; `0b43ec3` | Hoàn thành; phụ thuộc PageIndex khi chạy thật |
| Đánh giá A/B | Tạo 15 câu golden; chạy 30 lượt trả lời trên dense-only và hybrid + RRF với cùng generator, evaluator và `top_k`; lưu từng lượt chấm và báo cáo bốn metric. | `src/evaluate_pipeline.py`, `group_project/evaluation/golden_dataset.json`, `evaluation_results.json`, `RESULT.md`; `0b43ec3` | Hoàn thành |

## Quyết định kỹ thuật quan trọng

1. **Dùng cosine của dense để kích hoạt fallback.** RRF score là điểm hợp nhất thứ hạng, không cùng thang đo với cosine. `retrieve()` chỉ gọi PageIndex khi cosine tốt nhất dưới ngưỡng; nếu PageIndex lỗi, pipeline giữ kết quả local. Cần hiệu chỉnh ngưỡng trên câu trong và ngoài chủ đề trước khi dùng ổn định.
2. **Tắt fallback trong phép so sánh A/B.** Hai cấu hình chỉ khác chiến lược retrieval, cùng 15 câu, prompt, mô hình và `top_k=5`; ngưỡng `-1.0` ngăn PageIndex làm nhiễu kết quả. Phép so sánh này không đo hiệu quả của PageIndex.

## Kiểm tra và kết quả

- `python -m pytest tests/test_contracts.py tests/test_acceptance.py -q`: 20 kiểm tra đạt; `python -m compileall -q app.py src` đạt trong lần kiểm tra đã ghi ở nhánh.
- Báo cáo A/B: điểm trung bình bốn metric của hybrid + RRF là **0,767**, dense-only là **0,742**. Context recall và context precision cùng tăng từ **0,867** lên **0,900**; faithfulness giữ ở **0,633**.
- Mỗi cấu hình có **5/15** câu bị từ chối an toàn. Lượt đánh giá đầu phát hiện model ghi sai tiền tố citation; prompt được sửa để ID nguồn khớp rồi đánh giá lại toàn bộ.
- Giao diện Streamlit khởi động và endpoint kiểm tra sức khỏe trả HTTP 200 trong lượt demo cục bộ; chi tiết ở `docs/DEMO.md`.

## Hạn chế và hướng tiếp theo

- Bốn metric do một lượt LLM-as-judge chấm cho mỗi câu trả lời, nên cần kiểm tra mẫu thủ công trước khi diễn giải chênh lệch nhỏ 0,025 là cải thiện chắc chắn.
- Ngưỡng fallback mặc định 0,30 chưa được hiệu chỉnh trên bộ câu hỏi ngoài chủ đề có nhãn. Tôi sẽ đo tỉ lệ kích hoạt đúng/sai và kiểm tra PageIndex riêng, rồi mới chốt ngưỡng.

## Xác nhận đóng góp

Nội dung trên có thể đối chiếu với code, commit và file kết quả đánh giá của nhánh `khailq`.

- Ngày: 25/09/2026
- Thành viên: Lưu Quang Khải
