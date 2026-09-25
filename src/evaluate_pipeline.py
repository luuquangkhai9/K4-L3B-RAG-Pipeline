"""Run the golden-set A/B evaluation (dense-only vs hybrid+RRF)."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
OUTPUT_PATH = ROOT / "group_project" / "evaluation" / "evaluation_results.json"
METRICS = ("faithfulness", "answer_relevance", "context_recall", "context_precision")


def _judge(client, question: str, answer: str, retrieved: list[str], expected: list[str]) -> dict:
    prompt = {
        "question": question,
        "answer": answer,
        "retrieved_contexts": retrieved,
        "reference_contexts": expected,
        "instructions": (
            "Score each metric from 0 to 1. Faithfulness: fraction of factual claims supported by retrieved contexts. "
            "Answer relevance: directly answers the question. Context recall: retrieved contexts cover facts in reference contexts. "
            "Context precision: retrieved contexts are relevant to the question/reference. Treat unsupported or empty answers as 0. "
            "Return only JSON with numeric fields faithfulness, answer_relevance, context_recall, context_precision."
        ),
    }
    response = client.chat.completions.create(
        model=os.getenv("EVALUATOR_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a strict, consistent RAG evaluation judge. Return valid JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content or "{}"
    score = json.loads(raw)
    return {metric: max(0.0, min(1.0, float(score[metric]))) for metric in METRICS}


def main() -> None:
    load_dotenv(ROOT / ".env")
    from openai import OpenAI
    import src.task10_generation as generation
    from src.task9_retrieval_pipeline import retrieve as base_retrieve

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to evaluate the pipeline")
    client = OpenAI(api_key=api_key)
    dataset = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    results = {"metadata": {}, "records": []}
    if OUTPUT_PATH.exists():
        try:
            results = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    finished = {(row["case_id"], row["configuration"]) for row in results.get("records", [])}
    for config in ("dense-only", "hybrid+RRF"):
        for case_id, case in enumerate(dataset, start=1):
            if (case_id, config) in finished:
                continue
            captured = []

            def configured_retrieve(query, top_k=5):
                found = base_retrieve(
                    query, top_k=top_k, score_threshold=-1.0,
                    use_reranking=(config == "hybrid+RRF"),
                )
                captured.extend(found)
                return found

            generation.retrieve = configured_retrieve
            started = time.perf_counter()
            generated = generation.generate_with_citation(case["question"], top_k=5)
            generation_seconds = time.perf_counter() - started
            contexts = [chunk["content"] for chunk in captured]
            judge_started = time.perf_counter()
            scores = _judge(
                client, case["question"], generated["answer"], contexts,
                case["expected_context"] if isinstance(case["expected_context"], list) else [case["expected_context"]],
            )
            judge_seconds = time.perf_counter() - judge_started
            results.setdefault("records", []).append({
                "case_id": case_id,
                "question": case["question"],
                "configuration": config,
                "answer": generated["answer"],
                "retrieval_source": generated["retrieval_source"],
                "retrieved_ids": [chunk["id"] for chunk in captured],
                "retrieved_contexts": contexts,
                "scores": scores,
                "generation_seconds": generation_seconds,
                "judge_seconds": judge_seconds,
            })
            results["metadata"] = {
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "evaluator": os.getenv("EVALUATOR_MODEL", "gpt-4o-mini"),
                "generator": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                "embedding": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                "top_k": 5,
                "fallback_disabled_for_ab": True,
                "judge_method": "one structured LLM-as-judge pass per answer; 0-1 scale",
            }
            OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{config}: {case_id}/{len(dataset)} complete")

    generation.retrieve = base_retrieve
    print(f"Saved {len(results.get('records', []))} evaluations to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
