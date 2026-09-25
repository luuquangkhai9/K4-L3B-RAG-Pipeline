# RAG evaluation results

Đề tài: **IELTS Writing** — band descriptors, tiêu chí chấm điểm, bài viết mẫu.
Corpus: 9 tài liệu (4 tài liệu chính thức từ ielts.org + 5 bài viết), 330 chunk.

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 |
| Framework and version              | ragas 0.4.3 |
| Evaluator model                    | qwen3.8-flash (`enable_thinking=False`) |
| Generator model                    | qwen3.8-flash |
| Embedding model                    | text-embedding-v4, 1024 chiều |
| Corpus version/commit              | `5a0708a` |
| Golden dataset size                | 24 câu |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.40 — đo bằng `python -m src.calibrate_threshold`: 24 câu in-domain có cosine thấp nhất 0.470, 7 câu out-domain cao nhất 0.274, hai phân bố tách hoàn toàn |

Ghi chú về evaluator: `qwen3.8-flash` sinh khoảng 200 reasoning token cho mỗi lượt
chấm, đủ để 192 lượt chấm kéo dài hàng giờ, nên evaluator được gọi với
`enable_thinking=False` (đo được: 6.0s → 0.6s mỗi lượt). Generator giữ nguyên
reasoning vì đó là hành vi thật của sản phẩm.

## Configurations

- **Config A — dense-only:** `retrieve(query, top_k=5, use_reranking=False)`, chỉ dùng
  kết quả `semantic_search` trên ChromaDB.
- **Config B — hybrid + RRF:** `retrieve(query, top_k=5, use_reranking=True)`, gộp
  `semantic_search` và `lexical_search` bằng RRF.

Hai config dùng cùng golden dataset, generator, evaluator, prompt, `top_k` và
threshold; chỉ khác retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |   0.9691 |   0.9809 |   +0.0118 |
| Answer relevance  |   0.7266 |   0.7273 |   +0.0007 |
| Context recall    |   0.7708 |   0.7569 |   -0.0139 |
| Context precision |   0.7618 |   0.4509 |   -0.3109 |
| **Average**       |   0.8071 |   0.7290 |   -0.0781 |

Cả 48 lượt chấm (24 câu × 2 config) đều cho điểm hợp lệ, không có giá trị NaN.

## A/B comparison

- **Cấu hình tốt hơn: A (dense-only), 0.8071 so với 0.7290.**
- **Evidence:** toàn bộ mức giảm nằm ở `context_precision` (−0.3109); ba metric còn lại
  gần như không đổi. Nguyên nhân là RRF **đẩy tài liệu chính thức ra khỏi top-5** để
  nhường chỗ cho bài viết:

  | Config | Chunk `legal` trong top-5 | Chunk `news` trong top-5 |
  |---|---:|---:|
  | A — dense-only | 3.2 / 5 | 1.8 / 5 |
  | B — hybrid + RRF | 1.9 / 5 | 3.1 / 5 |

  Ví dụ rõ nhất ở câu *"Bốn tiêu chí chấm điểm của Task 2 là gì?"*: số chunk `news`
  trong top-5 tăng từ 0 lên 3, precision rơi từ 1.00 xuống 0.33.

  **Cơ chế:** RRF xếp hạng theo *số bảng xếp hạng mà tài liệu xuất hiện*, không theo
  mức độ mạnh của từng bảng. Chunk quy chế ngắn và đặc thù nên đứng đầu ở dense nhưng
  không lọt top-10 của BM25 (khác từ vựng) → chỉ được một phiếu. Bài viết dài chia sẻ
  nhiều token chung với query nên lọt cả hai bảng → được hai phiếu → thắng. Đây là
  điểm yếu thiên vị độ dài tài liệu đã biết của RRF.

- **Trade-off về latency/cost:** Config B tốn thêm một lượt `lexical_search` (dựng BM25
  trên 330 chunk) nhưng không gọi thêm API embedding; chi phí gần như tương đương
  Config A. Đổi lại chất lượng giảm, nên trong cấu hình hiện tại B không đáng dùng.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | Bài viết Task 2 dưới 250 từ bị đánh giá thế nào trong tài liệu bài mẫu? | B | 1.00 | 0.00 | 0.00 | 0.00 | retrieval | Chunk đáp án `legal/…example-responses…::chunk-26` tồn tại trong index nhưng không lọt top-10 của cả dense lẫn BM25 |
|   2 | Ba lỗi phổ biến khi viết IELTS Writing được bài viết của IDP nêu ra là gì? | B | 1.00 | 0.00 | 0.00 | 0.00 | retrieval | Đáp án nằm rải ở `news/article_05.md::chunk-47` và `::chunk-49`; BM25 khớp rời rạc nên không chunk nào đủ điểm để vào top-5 |
|   3 | Bài viết bị trừ điểm trong những trường hợp nào? | B | 1.00 | 0.54 | 0.00 | 0.00 | retrieval | Chunk đúng `legal/…key-assessment-criteria.md::chunk-2` chỉ xuất hiện ở một bảng xếp hạng nên bị RRF đẩy xuống dưới top-5 |

Đã kiểm chứng nguyên nhân bằng cách truy vết ngược: cả ba đoạn văn bản đáp án đều có
mặt trong `data/standardized/` và trong chunk tương ứng, nên đây là **lỗi xếp hạng khi
truy xuất**, không phải mất dữ liệu ở khâu chuẩn hoá hay chunking.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Bỏ RRF hoặc chỉ fuse khi cosine của dense thấp hơn threshold, thay vì luôn fuse | Precision rơi 0.31 khi bật RRF; chunk legal trong top-5 giảm 3.2 → 1.9 | Kéo precision về mức của Config A, giữ nguyên các metric khác | Chạy lại `python -m src.evaluation`, so cột Context precision giữa A và B |
|        2 | Tăng `top_k` của hai nhánh trước khi fuse (ví dụ 20 thay vì 10) để chunk đúng có cơ hội lọt vào bảng xếp hạng | Ở cả 3 câu kém nhất, chunk đáp án không lọt top-10 của dense lẫn BM25 | Tăng context recall; đổi lại có thể giảm precision nếu không lọc lại | Đo lại recall trên cùng 3 câu, đồng thời theo dõi precision toàn tập |
|        3 | Chunk lại `news/article_05.md` để ba lỗi phổ biến nằm trong cùng một chunk thay vì bị cắt sang hai chunk 47 và 49 | Câu "Ba lỗi phổ biến" có recall 0 vì BM25 khớp rời rạc giữa hai chunk | Câu này trở nên trả lời được; không ảnh hưởng các câu khác | Chạy lại riêng câu đó, kiểm tra `context_recall` > 0 |
|        4 | Thêm bước chuẩn hoá hình thái trước khi tokenize BM25 | Golden dataset có các câu dùng biến thể từ tiếng Anh (`write`/`writing`) | Tăng recall cho nhóm câu hỏi tiếng Anh | Đo lại recall của các câu hỏi tiếng Anh trong golden dataset |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| PageIndex vectorless fallback | hybrid + RRF | Không đo được | +0 API | Chưa thực hiện: nhóm không có `PAGEINDEX_API_KEY`, và `.env.example` xếp key này vào nhóm "Optional services". Hàm `pageindex_search()` đã có đủ theo interface bắt buộc và được gọi đúng lúc (cosine < 0.40); khi thiếu key, pipeline ghi log rồi trả kết quả hybrid thay vì crash — đúng yêu cầu tại `docs/MODULE_CONTRACTS.md`. Đây là hạn chế đã biết, không phải lỗi. |