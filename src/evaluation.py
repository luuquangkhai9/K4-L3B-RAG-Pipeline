"""
Đánh giá RAG — 4 metric và so sánh A/B.

Config A (dense-only) và Config B (hybrid + RRF) dùng CÙNG golden dataset,
generator, evaluator, prompt và top_k; chỉ khác retrieval strategy.

Bốn metric (ragas):
    - faithfulness      : câu trả lời có bám vào context không
    - answer_relevancy  : câu trả lời có đúng trọng tâm câu hỏi không
    - context_recall    : context lấy về có đủ thông tin để trả lời không
    - context_precision : trong context lấy về, bao nhiêu phần thực sự liên quan

Kết quả từng câu được lưu vào group_project/evaluation/eval_runs.json để
không phải chạy lại từ đầu khi chỉ muốn xem báo cáo.

Chạy: python -m src.evaluation
"""

import json
import os
import statistics
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

ROOT = Path(__file__).parent.parent
GOLDEN_DATASET = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
RUNS_PATH = ROOT / "group_project" / "evaluation" / "eval_runs.json"

TOP_K = 5

LLM_MODEL = os.getenv("LLM_MODEL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

CONFIGS = {
    "A_dense_only": {"label": "A — dense-only", "use_reranking": False},
    "B_hybrid_rrf": {"label": "B — hybrid + RRF", "use_reranking": True},
}


def _openai_client(*, disable_thinking: bool = False):
    """Client trỏ vào gateway OpenAI-compatible.

    disable_thinking=True dành cho EVALUATOR: qwen3.8-flash sinh ~200 reasoning
    token cho mỗi lượt chấm, đủ để 192 lượt chấm kéo dài hàng giờ. Đo được
    enable_thinking=False giảm mỗi lượt từ 6.0s xuống 0.6s. Generator (Task 10)
    vẫn giữ reasoning vì đó là hành vi thật của sản phẩm.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    if not disable_thinking:
        return client

    original_create = client.chat.completions.create

    def create(*args, **kwargs):
        body = dict(kwargs.get("extra_body") or {})
        body.setdefault("enable_thinking", False)
        kwargs["extra_body"] = body
        return original_create(*args, **kwargs)

    client.chat.completions.create = create
    return client


def _ragas_llm_and_embeddings():
    """Bọc gateway OpenAI-compatible thành LLM/embedding cho ragas.

    Lưu ý: embedding_factory của ragas 0.4.3 trả về lớp chỉ có embed_text,
    thiếu embed_query mà answer_relevancy cần, nên phải đi qua
    LangchainEmbeddingsWrapper. check_embedding_ctx_length=False để gửi chuỗi
    thô thay vì mảng token — gateway không nhận dạng input đó.
    """
    from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import llm_factory

    client = _openai_client(disable_thinking=True)
    llm = llm_factory(LLM_MODEL, provider="openai", client=client)

    embeddings = LangchainEmbeddingsWrapper(
        LangchainOpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            check_embedding_ctx_length=False,
        )
    )
    return llm, embeddings


def _answer_from_chunks(query: str, chunks: list[dict]) -> str:
    """Sinh câu trả lời từ chunks — dùng đúng logic của Task 10."""
    from .task10_generation import SAFE_REFUSAL, SYSTEM_PROMPT, call_llm, format_context, reorder_for_llm

    if not chunks:
        return SAFE_REFUSAL
    context = format_context(reorder_for_llm(chunks))
    try:
        return call_llm(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {query}") or SAFE_REFUSAL
    except Exception as error:
        print(f"    LLM lỗi: {type(error).__name__}: {error}")
        return SAFE_REFUSAL


def collect_rows(force: bool = False) -> list[dict]:
    """Chạy retrieval + generation cho cả hai config trên golden dataset.

    Nếu eval_runs.json đã có đủ dòng thì dùng lại, để chỉ phải chạy lại bước
    chấm điểm khi phần sinh câu trả lời đã xong.
    """
    from .task9_retrieval_pipeline import retrieve

    golden = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))
    expected = len(golden) * len(CONFIGS)

    if not force and RUNS_PATH.exists():
        cached = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
        rows = cached if isinstance(cached, list) else cached.get("rows", [])
        if len(rows) == expected:
            print(f"Dùng lại {len(rows)} dòng đã sinh trong {RUNS_PATH.name}")
            return rows
        print(f"{RUNS_PATH.name} có {len(rows)}/{expected} dòng — sinh lại từ đầu.")

    rows: list[dict] = []

    for config_key, config in CONFIGS.items():
        print(f"\n=== Config {config['label']} ===")
        for index, item in enumerate(golden, 1):
            chunks = retrieve(item["question"], top_k=TOP_K, use_reranking=config["use_reranking"])
            answer = _answer_from_chunks(item["question"], chunks)
            rows.append(
                {
                    "config": config_key,
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "expected_context": item["expected_context"],
                    "answer": answer,
                    "contexts": [chunk["content"] for chunk in chunks],
                    "sources": [chunk["id"] for chunk in chunks],
                    "retrieval_method": chunks[0]["retrieval_method"] if chunks else "none",
                    "difficulty": item.get("difficulty", ""),
                    "source": item.get("source", ""),
                }
            )
            print(f"  [{index:2}/{len(golden)}] {item['question'][:58]}")

    RUNS_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu {len(rows)} dòng vào {RUNS_PATH.name}")
    return rows


def score_rows(rows: list[dict]) -> list[dict]:
    """Chấm 4 metric bằng ragas cho từng config."""
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    llm, embeddings = _ragas_llm_and_embeddings()

    # Gateway api.ai-box.vn giới hạn ~1 request đồng thời: đo được 8 request song
    # song mất 22.1s trong khi 3 request tuần tự chỉ mất 4.5s. Để max_workers mặc
    # định (16) thì request xếp hàng, vượt timeout rồi retry nên càng chậm.
    run_config = RunConfig(max_workers=2, timeout=600, max_retries=2)
    scored: list[dict] = []

    for config_key, config in CONFIGS.items():
        subset = [row for row in rows if row["config"] == config_key]
        samples = [
            SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
                reference=row["expected_answer"],
            )
            for row in subset
        ]
        print(f"\n=== Chấm điểm config {config['label']} ({len(samples)} câu) ===")
        result = evaluate(
            EvaluationDataset(samples=samples),
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            run_config=run_config,
        )
        frame = result.to_pandas()
        for row, (_, scored_row) in zip(subset, frame.iterrows()):
            scored.append(
                {
                    **{key: row[key] for key in ("config", "question", "answer", "difficulty", "source")},
                    **{name: (None if scored_row.get(name) is None else float(scored_row[name])) for name in METRIC_NAMES},
                }
            )
            print(f"  {row['question'][:52]}")

    RUNS_PATH.write_text(json.dumps({"rows": rows, "scores": scored}, ensure_ascii=False, indent=2), encoding="utf-8")
    return scored


def report(scored: list[dict]) -> None:
    """In bảng tổng hợp và danh sách câu kém nhất."""
    print("\n" + "=" * 68)
    print("TỔNG HỢP")
    print("=" * 68)
    summary: dict[str, dict[str, float]] = {}
    for config_key, config in CONFIGS.items():
        subset = [row for row in scored if row["config"] == config_key]
        averages = {}
        for name in METRIC_NAMES:
            values = [row[name] for row in subset if row[name] is not None]
            averages[name] = statistics.mean(values) if values else float("nan")
        averages["average"] = statistics.mean(averages.values())
        summary[config_key] = averages
        print(f"\n{config['label']}")
        for name in METRIC_NAMES + ["average"]:
            print(f"  {name:20} {averages[name]:.4f}")

    print("\nSo sánh B − A:")
    for name in METRIC_NAMES + ["average"]:
        delta = summary["B_hybrid_rrf"][name] - summary["A_dense_only"][name]
        flag = "cải thiện" if delta > 0.005 else ("giảm" if delta < -0.005 else "tương đương")
        print(f"  {name:20} {delta:+.4f}  ({flag})")

    print("\nCâu kém nhất (config B):")
    worst = sorted(
        (row for row in scored if row["config"] == "B_hybrid_rrf" and row["faithfulness"] is not None),
        key=lambda row: (row["faithfulness"], row["context_recall"] or 0),
    )[:5]
    for row in worst:
        print(f"  F={row['faithfulness']:.2f} R={row['context_recall']:.2f}  {row['question'][:56]}")


def main() -> None:
    rows = collect_rows()
    scored = score_rows(rows)
    report(scored)


if __name__ == "__main__":
    main()