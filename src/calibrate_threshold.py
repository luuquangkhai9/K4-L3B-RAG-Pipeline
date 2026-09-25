"""
Hiệu chỉnh SCORE_THRESHOLD cho bước fallback của Task 9.

Task 9 quyết định gọi PageIndex fallback khi cosine score gốc của dense
retrieval thấp hơn threshold. Không có một con số đúng cho mọi corpus, nên
threshold phải được chọn từ phân bố điểm thật:

    - Query in-domain  : câu hỏi lấy từ golden dataset (phải trả lời được)
    - Query out-domain : chủ đề ngoài corpus (phải kích hoạt fallback)

Threshold tốt nằm giữa hai phân bố: cao hơn mức điểm của query ngoài domain
và thấp hơn mức điểm của query trong domain.

Chạy: python -m src.calibrate_threshold
"""

import json
import statistics
from pathlib import Path

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search


load_dotenv()

ROOT = Path(__file__).parent.parent
GOLDEN_DATASET = ROOT / "group_project" / "evaluation" / "golden_dataset.json"

# Chủ đề hoàn toàn ngoài corpus IELTS Writing.
OUT_OF_DOMAIN_QUERIES = [
    "Cách nấu phở bò tại nhà",
    "Giá bitcoin hôm nay là bao nhiêu?",
    "Lịch thi đấu bóng đá ngoại hạng Anh",
    "Hướng dẫn sửa xe máy bị hỏng bugi",
    "Danh sách phim chiếu rạp mùa hè 2026",
    "Công thức làm bánh mì bơ tỏi",
    "Chính sách thuế thu nhập cá nhân mới nhất",
]


def best_dense_score(query: str) -> float:
    """Cosine score cao nhất mà dense retrieval trả về cho query."""
    results = semantic_search(query, top_k=5)
    return results[0]["score"] if results else 0.0


def main() -> None:
    queries = [item["question"] for item in json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))]

    print(f"=== Query IN-DOMAIN ({len(queries)} câu từ golden dataset) ===")
    in_scores = []
    for query in queries:
        score = best_dense_score(query)
        in_scores.append(score)
        print(f"  {score:.4f}  {query[:66]}")

    print(f"\n=== Query OUT-DOMAIN ({len(OUT_OF_DOMAIN_QUERIES)} câu) ===")
    out_scores = []
    for query in OUT_OF_DOMAIN_QUERIES:
        score = best_dense_score(query)
        out_scores.append(score)
        print(f"  {score:.4f}  {query[:66]}")

    print("\n=== Phân bố ===")
    print(
        f"  in-domain : min={min(in_scores):.4f}  p10={sorted(in_scores)[len(in_scores)//10]:.4f}  "
        f"median={statistics.median(in_scores):.4f}  max={max(in_scores):.4f}"
    )
    print(
        f"  out-domain: min={min(out_scores):.4f}  median={statistics.median(out_scores):.4f}  "
        f"max={max(out_scores):.4f}"
    )

    # Threshold đề xuất: nằm giữa điểm cao nhất của out-domain và
    # phân vị 10% của in-domain (bỏ qua vài câu in-domain điểm thấp bất thường).
    low = max(out_scores)
    high = sorted(in_scores)[len(in_scores) // 10]
    if low < high:
        print(f"\n  Dải tách được: ({low:.4f}, {high:.4f}]")
        print(f"  => SCORE_THRESHOLD đề xuất: {(low + high) / 2:.2f}")
        print(f"     (out-domain tối đa {low:.4f} < threshold <= in-domain p10 {high:.4f})")
    else:
        print(
            f"\n  CẢNH BÁO: hai phân bố chồng lấn "
            f"(out-domain max {low:.4f} >= in-domain p10 {high:.4f})."
        )
        print("  Không có threshold tách hoàn hảo; cần xem lại corpus hoặc embedding model.")


if __name__ == "__main__":
    main()