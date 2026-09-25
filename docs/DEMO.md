# IELTS Writing RAG demo

## Start the app

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Confirm that `.env` has the provider keys used by the app and keep it out of version control.

## Suggested 3–5 minute walkthrough

1. Explain the corpus: four IELTS Writing PDFs and five collected IELTS pages, standardized as Markdown and indexed in ChromaDB.
2. Ask: **“What four criteria are used to assess IELTS Writing?”** Show the answer and expand its source list to verify the cited chunk.
3. Ask: **“What is expected of a Band 7 Task 2 response?”** Discuss the retrieved band descriptor and the importance of citing the source.
4. Ask an unrelated question, such as **“Who won the latest World Cup?”** The assistant should refuse to verify it from this corpus. If dense confidence is low, PageIndex may be used for legal PDFs.
5. Show `group_project/evaluation/RESULT.md`: 15 golden questions, dense-only vs hybrid + RRF, four metrics, failure cases and follow-up actions.

## Reproduce local checks and evaluation

```powershell
python -m pytest tests/test_contracts.py tests/test_acceptance.py -q
python -m src.evaluate_pipeline
```

Evaluation calls the configured generator and OpenAI evaluator and writes per-answer evidence to `group_project/evaluation/evaluation_results.json`.
