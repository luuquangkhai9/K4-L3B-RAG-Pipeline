"""
Evaluation — golden dataset, 4 metrics và A/B comparison.

Metric theo định nghĩa RAGAS, chấm bằng LLM-as-judge (JSON, temperature 0):
    - faithfulness: tỉ lệ claim trong answer được context hỗ trợ.
    - answer_relevance: cosine giữa câu hỏi gốc và các câu hỏi sinh ngược từ answer.
    - context_recall: tỉ lệ statement của expected_answer tìm thấy trong context.
    - context_precision: average precision của các chunk liên quan theo thứ hạng.

Config (cùng generator, prompt, top_k; chỉ đổi retrieval):
    A — dense-only, B — hybrid + RRF, C — hybrid + RRF nhưng tắt query translation.

Chạy: python -m src.evaluation   (ghi group_project/evaluation/results.json)
"""

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from . import task9_retrieval_pipeline as pipeline
from .task4_chunking_indexing import EMBEDDING_MODEL, embed_texts
from .task10_generation import LLM_MODEL, LLM_PROVIDER, REFUSAL, generate_with_trace


load_dotenv()

EVAL_DIR = Path(__file__).parent.parent / "group_project" / "evaluation"
GOLDEN_PATH = EVAL_DIR / "golden_dataset_nguyenha.json"
RESULTS_PATH = EVAL_DIR / "results.json"

EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o")
TOP_K = 5
WORKERS = 3
METRICS = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]

CONFIGS = {
    "A": {"name": "Dense only", "use_reranking": False, "translation": True},
    "B": {"name": "Hybrid + RRF", "use_reranking": True, "translation": True},
    "C": {"name": "Hybrid + RRF, không dịch query", "use_reranking": True, "translation": False},
}

# Câu ngoài phạm vi: đo tỉ lệ safe refusal (không nằm trong 4 metric).
OUT_OF_DOMAIN = [
    "Giá vàng hôm nay bao nhiêu?",
    "What is the TOEFL iBT speaking section format?",
    "Học phí Đại học Bách khoa Hà Nội năm 2025 là bao nhiêu?",
    "Who won the 2022 FIFA World Cup?",
    "Cách đăng ký học phần tại trường đại học",
]


def judge(instruction: str, payload: dict) -> dict:
    """Gọi evaluator LLM, trả JSON."""
    from openai import OpenAI

    response = OpenAI(max_retries=8).chat.completions.create(
        model=EVAL_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": instruction + "\nRespond with JSON only."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    return json.loads(response.choices[0].message.content or "{}")


def _contexts(chunks: list[dict]) -> list[str]:
    return [c["content"] for c in chunks]


def faithfulness(answer: str, contexts: list[str]) -> float:
    result = judge(
        "Break the answer into atomic factual claims (ignore citation markers like [1]). "
        "For each claim decide whether it is directly supported by the contexts. "
        'Return {"claims": [{"claim": str, "supported": bool}]}.',
        {"answer": answer, "contexts": contexts},
    )
    claims = result.get("claims") or []
    return sum(bool(c.get("supported")) for c in claims) / len(claims) if claims else 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def answer_relevance(question: str, answer: str) -> float:
    result = judge(
        "Write 3 different questions that the given answer would directly answer, "
        'in the same language as the answer. Return {"questions": [str, str, str]}.',
        {"answer": answer},
    )
    generated = [q for q in result.get("questions") or [] if q.strip()]
    if not generated:
        return 0.0
    vectors = embed_texts([question, *generated])
    return sum(_cosine(vectors[0], v) for v in vectors[1:]) / len(generated)


def context_recall(expected_answer: str, contexts: list[str]) -> float:
    result = judge(
        "Split the reference answer into its individual statements. For each statement decide "
        "whether it can be attributed to the contexts (the contexts may be in another language). "
        'Return {"statements": [{"statement": str, "attributed": bool}]}.',
        {"reference_answer": expected_answer, "contexts": contexts},
    )
    statements = result.get("statements") or []
    return sum(bool(s.get("attributed")) for s in statements) / len(statements) if statements else 0.0


def context_precision(question: str, expected_answer: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    result = judge(
        "For each context (in order) decide whether it is useful for arriving at the reference answer "
        "to the question (contexts may be in another language). "
        'Return {"verdicts": [bool, ...]} with exactly one verdict per context.',
        {"question": question, "reference_answer": expected_answer,
         "contexts": [{"rank": i, "text": c} for i, c in enumerate(contexts, 1)]},
    )
    verdicts = [bool(v) for v in (result.get("verdicts") or [])][: len(contexts)]
    hits, total = 0, 0.0
    for rank, relevant in enumerate(verdicts, 1):
        if relevant:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


def evaluate_case(case: dict, use_reranking: bool) -> dict:
    started = time.perf_counter()
    result, trace = generate_with_trace(case["question"], top_k=TOP_K, use_reranking=use_reranking)
    latency = time.perf_counter() - started
    retrieved = trace.get("retrieved") or []
    contexts = _contexts(retrieved)
    refused = result["retrieval_source"] == "none"
    scores = {
        # Từ chối thì không có claim sai (faithful) nhưng không trả lời câu hỏi.
        "faithfulness": 1.0 if refused else faithfulness(result["answer"], contexts),
        "answer_relevance": 0.0 if refused else answer_relevance(case["question"], result["answer"]),
        "context_recall": context_recall(case["expected_answer"], contexts),
        "context_precision": context_precision(case["question"], case["expected_answer"], contexts),
    }
    return {
        "question": case["question"],
        "language": case.get("language"),
        "expected_source": case.get("source"),
        "answer": result["answer"],
        "refused": refused,
        "refusal_reason": trace.get("refusal_reason"),
        "retrieved": [r["id"] for r in retrieved],
        "expected_source_hit": any(case.get("source", "") in r["id"] for r in retrieved),
        "best_dense_score": trace.get("best_dense_score"),
        "latency": latency,
        **{k: round(v, 4) for k, v in scores.items()},
    }


def _summary(rows: list[dict]) -> dict:
    summary = {m: round(sum(r[m] for r in rows) / len(rows), 4) for m in METRICS}
    summary["average"] = round(sum(summary[m] for m in METRICS) / len(METRICS), 4)
    summary["answered_rate"] = round(sum(not r["refused"] for r in rows) / len(rows), 4)
    summary["source_hit_rate"] = round(sum(r["expected_source_hit"] for r in rows) / len(rows), 4)
    summary["avg_latency"] = round(sum(r["latency"] for r in rows) / len(rows), 2)
    return summary


def run() -> dict:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    output = {
        "run_info": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "framework": "Custom LLM-as-judge (RAGAS metric definitions)",
            "evaluator_model": f"openai / {EVAL_MODEL}",
            "generator_model": f"{LLM_PROVIDER} / {LLM_MODEL}",
            "embedding_model": EMBEDDING_MODEL,
            "golden_size": len(golden),
            "top_k": TOP_K,
            "score_threshold": pipeline.SCORE_THRESHOLD,
        },
        "configs": {},
    }
    original_translation = pipeline.QUERY_TRANSLATION
    try:
        for key, config in CONFIGS.items():
            # Cờ module-level: chạy từng config tuần tự, song song trong config.
            pipeline.QUERY_TRANSLATION = config["translation"]
            print(f"== Config {key}: {config['name']}")
            with ThreadPoolExecutor(WORKERS) as pool:
                rows = list(pool.map(lambda c: evaluate_case(c, config["use_reranking"]), golden))
                ood = list(pool.map(
                    lambda q: generate_with_trace(q, top_k=TOP_K, use_reranking=config["use_reranking"])[0],
                    OUT_OF_DOMAIN,
                ))
            summary = _summary(rows)
            summary["ood_refusal_rate"] = round(
                sum(r["retrieval_source"] == "none" for r in ood) / len(ood), 4
            )
            output["configs"][key] = {**config, "summary": summary, "rows": rows,
                                      "ood": [{"question": q, "refused": r["answer"] == REFUSAL}
                                              for q, r in zip(OUT_OF_DOMAIN, ood)]}
            print("  ", summary)
    finally:
        pipeline.QUERY_TRANSLATION = original_translation

    RESULTS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {RESULTS_PATH}")
    return output


def selftest() -> list[dict]:
    """Kiểm tra metric với ca biết trước kết quả (tốt phải cao, xấu phải thấp)."""
    question = "What is the minimum word count for Task 2?"
    reference = "Task 2 requires at least 250 words."
    good_ctx = ["Write at least 250 words for Task 2. You should spend about 40 minutes on this task."]
    bad_ctx = ["Reading regularly helps you build vocabulary.", "Mind mapping helps brainstorm ideas."]
    cases = [
        ("faithfulness", "grounded answer", faithfulness("Task 2 requires at least 250 words [1].", good_ctx), ">=", 0.9),
        ("faithfulness", "hallucinated answer", faithfulness(
            "Task 2 requires at least 400 words and must be handwritten in pen [1].", good_ctx), "<=", 0.3),
        ("faithfulness", "half hallucinated", faithfulness(
            "Task 2 requires at least 250 words. Candidates get a dictionary during the test.", good_ctx), "<=", 0.6),
        ("answer_relevance", "on-topic answer", answer_relevance(question, "Task 2 requires at least 250 words."), ">=", 0.7),
        ("answer_relevance", "off-topic answer", answer_relevance(question, "Reading novels improves your vocabulary."), "<=", 0.5),
        ("context_recall", "context has answer", context_recall(reference, good_ctx), ">=", 0.9),
        ("context_recall", "context lacks answer", context_recall(reference, bad_ctx), "<=", 0.1),
        ("context_precision", "relevant ranked first", context_precision(question, reference, good_ctx + bad_ctx), ">=", 0.9),
        ("context_precision", "relevant ranked last", context_precision(question, reference, bad_ctx + good_ctx), "<=", 0.5),
        ("context_precision", "no relevant context", context_precision(question, reference, bad_ctx), "<=", 0.1),
    ]
    rows = []
    for metric, case, score, op, bound in cases:
        passed = score >= bound if op == ">=" else score <= bound
        rows.append({"metric": metric, "case": case, "score": round(score, 3),
                     "expect": f"{op} {bound}", "passed": passed})
        print(f"{'PASS' if passed else 'FAIL'}  {metric:18} {case:24} score={score:.3f} (expect {op} {bound})")
    return rows


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        results = selftest()
        (EVAL_DIR / "metric_selftest.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        run()
