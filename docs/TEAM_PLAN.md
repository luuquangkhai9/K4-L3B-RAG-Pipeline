# Kế hoạch nhóm

## Đề tài

**Dịch vụ sinh viên Đại học Bách khoa Hà Nội** (thuộc nhóm "Dịch vụ đại học"): học phí, học bổng, quy chế đào tạo.

Chatbot RAG trả lời câu hỏi về quy chế đào tạo, học phí 2025-2026, học bổng KKHT và học bổng theo Nghị định 179/2026/NĐ-CP, có trích dẫn nguồn.

## Nguồn dữ liệu

| Loại | Nguồn | Tên file/vị trí |
|---|---|---|
| Legal | Quy chế đào tạo ĐHBK 2025 | `quy_che_dao_tao_dhbk_2025.pdf` |
| Legal | Quyết định học phí 2025-2026 | `quyet_dinh_hoc_phi_2025_2026.pdf` |
| Legal | Quy định xét cấp học bổng KKHT | `quy_dinh_xet_hoc_bong_khkt.pdf` |
| Legal | QĐ 1826/QĐ-BGDĐT danh mục ngành học bổng (NĐ 179 là PDF scan nên chưa dùng) | `quyet_dinh_1826_danh_muc_nganh_hoc_bong.pdf` |
| News | 7 URL trong `src/task2_crawl_news.py` | `data/landing/news/article_XX.json` |

Lưu ý mâu thuẫn số liệu: VnExpress ghi ~9.700 chỉ tiêu 2026, Vietbao ghi 9.880. Dùng làm một câu golden dataset kiểm tra citation, hoặc bỏ một bài.
File học bổng KKHT là bản 2020; nên thay bằng bản áp dụng từ HK1 2022-2023 tại trang quy chế của trường.

## Phân vai (điền tên thành viên)

| Vai | Thành viên | Phụ trách | Task |
|---|---|---|---|
| Data | | Thu thập legal/news, chuẩn hoá Markdown | 1, 2, 3 |
| Indexing | | Chunking, embedding API, ChromaDB | 4 |
| Retrieval | | Dense, BM25, RRF | 5, 6, 7 |
| Pipeline & fallback | | PageIndex fallback, threshold, pipeline | 8, 9 |
| Generation & UI | | Citation, Streamlit | 10, `app.py` |
| Evaluation | | Golden dataset 15+ câu, 4 metric, A/B, `RESULT.md` | — |

Mỗi thành viên ghi lại commit mình phụ trách để viết báo cáo cá nhân (`reports/INDIVIDUAL_REPORT.md`).

## Quyết định kỹ thuật

- Embedding qua API (không tải model local): `EMBEDDING_PROVIDER=openai`, `text-embedding-3-small`; có thể đổi sang `gemini` trong `.env`.
- Chunking: `RecursiveCharacterTextSplitter`, size 500, overlap 50.
- Vector store: ChromaDB persistent, cosine distance, ID ổn định `<đường dẫn>::chunk-<n>`.
