# Individual contribution report — draft

> Draft based on the artifacts completed in the `khailq` working tree. Fill in personal identity/team details and confirm authorship before submitting.

## Information

- Full name: To be filled in
- Student ID: To be filled in
- Team: To be filled in
- Repository/branch: `K4-L3B-RAG-Pipeline` / `khailq`

## Work represented by this draft

| Deliverable | Evidence in repository | Status |
| --- | --- | --- |
| Data standardization and indexing | `src/task3_convert_markdown.py`, `src/task4_chunking_indexing.py`, `data/standardized/`, `chroma_db/` | Complete in working tree |
| Search and fallback | `src/task5_semantic_search.py` through `src/task9_retrieval_pipeline.py` | Complete; PageIndex is an external dependency |
| Answer generation and chat UI | `src/task10_generation.py`, `app.py` | Complete |
| Evaluation | `group_project/evaluation/golden_dataset.json`, `evaluation_results.json`, `RESULT.md` | 15 questions, 30 scored answers |

## Technical decisions and evidence

1. **Use cosine similarity from dense retrieval for fallback.** RRF scores rank results on a different scale; PageIndex fallback is disabled in the A/B run so the retrieval strategy comparison stays isolated.
2. **Validate citations against retrieved chunk IDs.** The generation result only includes cited chunks; when the model/provider fails or produces no valid citation, the pipeline returns a safe refusal.

## Checks and outcomes

- `python -m pytest tests/test_contracts.py tests/test_acceptance.py -q`: 20 passed.
- `python -m compileall -q app.py src`: passed.
- A/B evaluation: hybrid + RRF mean score 0.767 vs dense-only 0.742 on the four reported metrics; see `group_project/evaluation/RESULT.md` for per-metric scores and limitations.
- A citation format mismatch was found during the first evaluation run, corrected in the generation prompt, and the A/B run was repeated.

## Limitations and follow-up

- The four metrics use a single structured LLM-as-judge call per answer; the scores are directional and should be spot-checked by a human.
- Five cases in each configuration returned safe refusals. Chunking around band headings and citation/refusal behavior need further review.
- The default fallback threshold of 0.30 has not yet been calibrated against a labeled out-of-domain query set.

## Confirmation

Complete name, student ID and team above, then verify that this draft accurately describes your own contribution before signing.

- Date: To be filled in
- Member name: To be filled in
