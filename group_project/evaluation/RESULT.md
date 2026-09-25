# RAG evaluation results

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-25 |
| Framework and version | Custom OpenAI structured LLM-as-judge; 1 judge call per answer |
| Evaluator model | `gpt-4o-mini` |
| Generator model | `gpt4o` (configured alias normalized to provider model ID) |
| Embedding model | `text-embedding-3-small` |
| Corpus version/commit | Branch `khailq`; working tree corpus, 4 legal PDFs + 5 news pages |
| Golden dataset size | 15 questions; 30 scored answers |
| `top_k` | 5 |
| Fallback threshold and calibration | Disabled in A/B with threshold `-1.0` to isolate the retrieval strategy. Production default remains cosine similarity 0.30; this threshold is provisional and needs in-domain/out-of-domain calibration. |

## Configurations

- **Config A ? dense-only:** Chroma cosine retrieval, top 5; no BM25 or RRF.
- **Config B ? hybrid + RRF:** dense and BM25 each retrieve top 10, fused once with RRF into top 5.

Both configurations used the same golden set, generator, evaluator, prompt, corpus and `top_k`. PageIndex fallback was disabled during A/B to isolate the retrieval strategy. Metrics use a single structured LLM judge response per generated answer on a 0?1 scale; results are directional and should be interpreted with manual review.

## Overall scores

| Metric | Config A | Config B | Delta B?A |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.633 | 0.633 | 0.000 |
| Answer relevance | 0.600 | 0.633 | 0.033 |
| Context recall | 0.867 | 0.900 | 0.033 |
| Context precision | 0.867 | 0.900 | 0.033 |
| **Average** | **0.742** | **0.767** | **0.025** |

## A/B comparison

- **Better configuration:** Hybrid + RRF, by 0.025 average metric points.
- **Evidence:** Context recall rose from 0.867 to 0.900; context precision rose from 0.867 to 0.900. Answer relevance changed by 0.033; faithfulness was unchanged. Safe refusals: 5/15 vs 5/15.
- **Latency/cost trade-off:** Mean generation time was 3.11s (dense) vs 2.43s (hybrid) on this run, excluding judge time. Hybrid adds BM25/RRF computation; observed timing is noisy and does not measure embedding/API cost precisely.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | What is expected of a Band 7 Task 2 response? | dense-only | 0.000 | 0.000 | 0.000 | 0.000 | retrieval/generation | Relevant band descriptor did not land consistently in top 5; some runs returned a safe refusal or judge scored response as unsupported. |
| 2 | What is the Band 7 expectation for Task 1 coherence? | dense-only | 0.000 | 0.000 | 0.000 | 0.000 | retrieval/generation | Relevant band descriptor did not land consistently in top 5; some runs returned a safe refusal or judge scored response as unsupported. |
| 3 | What is the Band 7 expectation for Task 1 coherence? | hybrid+RRF | 0.000 | 0.000 | 0.000 | 0.000 | retrieval/generation | Relevant band descriptor did not land consistently in top 5; some runs returned a safe refusal or judge scored response as unsupported. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| ---: | --- | --- | --- | --- |
| 1 | Improve heading-aware chunking for band descriptors and include section/band labels in each chunk. | Band 7 questions were among the lowest scoring cases across both configurations. | Better exact retrieval of criterion and band text. | Re-run these cases and compare all four scores. |
| 2 | Review safe-refusal and citation rate using a hand-audited sample. | Some retrieved cases still produced safe refusals; strict ID citation validation rejects ungrounded output by design. | Fewer unnecessary refusals while keeping citations traceable. | Count supported answers with valid source IDs; manually inspect 10 outputs. |
| 3 | Calibrate fallback threshold on a separate in-domain/out-of-domain set and separately test PageIndex. | A/B disabled fallback to isolate retrieval; default threshold 0.30 has not been calibrated in this run. | More reliable fallback choices without confusing fusion and cosine score scales. | Report false fallback and missed fallback rates on labeled queries. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| --- | --- | ---: | ---: | --- |
| Dense-only vs hybrid + RRF | Dense-only | Average +{fmt(delta_avg)} for hybrid | Generation mean {lat['dense-only']:.2f}s vs {lat['hybrid+RRF']:.2f}s; excludes indexing and judge cost | Hybrid + RRF performed better on average, led by context retrieval metrics. |
