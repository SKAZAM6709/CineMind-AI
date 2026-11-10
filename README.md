# RAG-Based Intelligent Movie Recommendation System

An **industry-oriented AI engineering portfolio project** using MovieLens, semantic movie retrieval, personalization, and later a grounded LLM response generator.

## Current project status

**Stage 1 (scaffold + uploaded data) only.** Real embedding creation, vector search, model integration, recommendation quality evaluation, and API deployment are planned, not implemented yet.

## Getting started (Windows PowerShell)

1. Open this folder in VS Code.
2. Run `py -m venv .venv` (or `python -m venv .venv`).
3. Activate using `.\.venv\Scripts\Activate.ps1` or run commands with `.\.venv\Scripts\python.exe` directly.
4. Run `python -m pip install -r requirements.txt`.
5. Verify the files: `python scripts/validate_data.py`.
6. Launch the starter UI: `python -m streamlit run apps/frontend/app.py`.

For macOS/Linux, activate using `source .venv/bin/activate`.

## Data

The provided MovieLens Latest Small original files are in `data/raw/movielens/`:
- `movies.csv`: `movieId,title,genres`
- `ratings.csv`: `userId,movieId,rating,timestamp`
- `tags.csv`: `userId,movieId,tag,timestamp`
- `links.csv`: `movieId,imdbId,tmdbId`
- `README.txt`: official dataset documentation and license

The current dataset has 9,742 movies, 100,836 ratings, 3,683 tags, and 610 distinct users. Eight `tmdbId` entries are blank; this is expected and should be handled during enrichment. There are no movie plot descriptions yet; a future TMDb enrichment step can populate `data/external/tmdb/` subject to TMDb terms and API limits.

## Directory map

- `apps/frontend/`: Streamlit UI
- `apps/api/`: future FastAPI web backend
- `src/movie_recommender/data/`: cleaning, joining, and enrichment logic
- `src/movie_recommender/retrieval/`: embeddings, vector indexes, and search
- `src/movie_recommender/recommendation/`: collaborative filtering, personalization, reranking
- `src/movie_recommender/llm/`: prompting, generation, and grounding
- `src/movie_recommender/evaluation/`: metrics and evaluation pipeline
- `scripts/`: reproducible command-line utilities
- `tests/`: automated tests
- `data/raw/`: original dataset (do not edit)
- `data/external/`: optional metadata from third-party providers
- `data/interim/`: intermediate data transformations
- `data/processed/`: cleaned datasets and features
- `data/vector_store/`: local FAISS/other vector indexes
- `configs/`: non-secret settings
- `models/`: locally saved models (not tracked)
- `docs/`, `notebooks/`: engineering notes and exploration

## Planned stages

1. Data inspection, integrity checks, and preprocessing
2. Metadata enhancement and semantic text representation
3. Embeddings and vector search
4. Personalized candidate reranking
5. LLM-driven grounded responses
6. API, UI, integration tests, Docker/deployment

## Licensing

MovieLens data is governed by `data/raw/movielens/README.txt`. The dataset may not be used for commercial/revenue-bearing purposes without the specified GroupLens permission. Third-party metadata (if added) has its own terms. Please review licenses before redistributing or deploying a commercial service.
