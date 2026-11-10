
"""
CineMind AI - RAG Evaluation

Evaluates:
1. Retrieval latency
2. LLM generation latency
3. RAG execution success
4. Citation validity
5. Human relevance review preparation

Outputs:
reports/evaluation/rag_results.csv
reports/evaluation/retrieval_review.csv

Note:
Citation validity does not prove factual groundedness.
"""

from pathlib import Path
import re
import time

import pandas as pd

from src.movie_recommender.retrieval.semantic_search import (
    SemanticMovieSearch,
)

from src.movie_recommender.llm.rag_assistant import (
    RAGMovieAssistant,
)


ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = ROOT / "reports" / "evaluation"

TOP_K = 5


# ==========================================
# TEST QUERIES
# ==========================================

TEST_QUERIES = [
    {
        "id": "Q01",
        "query": (
            "Recommend a magical family adventure "
            "with fantasy and mystery."
        ),
    },
    {
        "id": "Q02",
        "query": (
            "I want a romantic drama about "
            "relationships and emotional conflict."
        ),
    },
    {
        "id": "Q03",
        "query": (
            "Find a crime thriller involving "
            "danger, criminals and suspense."
        ),
    },
    {
        "id": "Q04",
        "query": (
            "Recommend a funny movie about "
            "friendship and everyday life."
        ),
    },
]


# ==========================================
# HELPERS
# ==========================================

def extract_references(answer):
    """
    Extract references in the format [1], [2], etc.
    """
    return sorted({
        int(value)
        for value in re.findall(
            r"\[(\d+)\]",
            str(answer),
        )
    })


def clean(value):
    if pd.isna(value):
        return ""
    return str(value)


# ==========================================
# EVALUATION PIPELINE
# ==========================================

def run_evaluation():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n===== CINEMIND AI RAG EVALUATION =====")

    print("\nLoading models...")

    search_engine = SemanticMovieSearch()

    rag_engine = RAGMovieAssistant(
        model="gemma3:1b"
    )

    evaluation_records = []
    review_records = []

    for test in TEST_QUERIES:

        query_id = test["id"]
        query = test["query"]

        print(f"\nRunning {query_id}: {query}")

        # ----------------------------------
        # RETRIEVAL
        # ----------------------------------

        start = time.perf_counter()

        results = search_engine.search(
            query,
            top_k=TOP_K,
        )

        retrieval_seconds = (
            time.perf_counter() - start
        )

        # ----------------------------------
        # GENERATION
        # ----------------------------------

        start = time.perf_counter()

        answer, success, error = (
            rag_engine.generate(
                query,
                results,
                history=[],
            )
        )

        generation_seconds = (
            time.perf_counter() - start
        )

        # ----------------------------------
        # CITATION CHECK
        # ----------------------------------

        references = extract_references(answer)

        valid_range = set(
            range(1, len(results) + 1)
        )

        invalid = [
            reference
            for reference in references
            if reference not in valid_range
        ]

        citation_present = len(references) > 0

        citation_valid = (
            citation_present
            and len(invalid) == 0
        )

        # ----------------------------------
        # SAVE RAG MEASUREMENTS
        # ----------------------------------

        evaluation_records.append({
            "query_id": query_id,
            "query": query,
            "retrieved_count": len(results),
            "retrieval_seconds": round(
                retrieval_seconds, 3
            ),
            "generation_seconds": round(
                generation_seconds, 3
            ),
            "llm_success": bool(success),
            "citation_present": citation_present,
            "citation_valid": citation_valid,
            "references": ",".join(
                str(value)
                for value in references
            ),
            "answer": answer,
            "error": error or "",
        })

        # ----------------------------------
        # BUILD HUMAN REVIEW DATA
        # ----------------------------------

        for index, (_, movie) in enumerate(
            results.iterrows(),
            start=1,
        ):

            review_records.append({
                "query_id": query_id,
                "query": query,
                "rank": index,
                "movieId": int(movie["movieId"]),
                "title": clean(
                    movie.get("title")
                ),
                "genres": clean(
                    movie.get("genres")
                ),
                "overview": clean(
                    movie.get("overview")
                ),
                "semantic_similarity": float(
                    movie.get(
                        "semantic_similarity", 0
                    )
                ),
                # Fill in manually with 0 or 1.
                "relevance_label": "",
            })

        print(
            "Retrieval:",
            round(retrieval_seconds, 3),
            "seconds"
        )

        print(
            "Generation:",
            round(generation_seconds, 3),
            "seconds"
        )

        print("LLM success:", success)
        print("Citation valid:", citation_valid)

    # ======================================
    # SAVE REPORTS
    # ======================================

    results_df = pd.DataFrame(
        evaluation_records
    )

    review_df = pd.DataFrame(
        review_records
    )

    results_path = (
        OUTPUT_DIR / "rag_results.csv"
    )

    review_path = (
        OUTPUT_DIR / "retrieval_review.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig"
    )

    review_df.to_csv(
        review_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================
    # SUMMARY
    # ======================================

    total = len(results_df)

    success_rate = (
        results_df["llm_success"].mean() * 100
    )

    citation_rate = (
        results_df["citation_valid"].mean() * 100
    )

    average_retrieval = (
        results_df["retrieval_seconds"].mean()
    )

    average_generation = (
        results_df["generation_seconds"].mean()
    )

    print("\n===== EVALUATION SUMMARY =====")

    print("Queries evaluated:", total)

    print(
        f"LLM success rate: "
        f"{success_rate:.1f}%"
    )

    print(
        f"Valid citation rate: "
        f"{citation_rate:.1f}%"
    )

    print(
        f"Average retrieval time: "
        f"{average_retrieval:.3f} seconds"
    )

    print(
        f"Average generation time: "
        f"{average_generation:.3f} seconds"
    )

    print("\nReports saved:")
    print(results_path)
    print(review_path)

    print(
        "\nNote: Human relevance and factual "
        "grounding review are still required."
    )


if __name__ == "__main__":
    run_evaluation()
