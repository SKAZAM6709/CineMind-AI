# Planned Architecture

```text
Streamlit frontend -> FastAPI backend (later)
                           |
                 Recommendation orchestrator
                   /                   \
      Semantic search (embeddings)   User-preference signals
                   \                   /
                  Candidate reranking
                           |
                Grounded LLM explanation
                           |
                      UI response
```

Stage 1 currently implements only data validation and a title-search frontend. The planned blocks are placeholders, not working integrations.
