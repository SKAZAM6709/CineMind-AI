
"""
CineMind AI - RAG Integration Test

Tests:
1. Semantic movie retrieval
2. Ollama LLM generation
3. Grounded movie recommendations
"""

from src.movie_recommender.retrieval.semantic_search import (
    SemanticMovieSearch,
)

from src.movie_recommender.llm.rag_assistant import (
    RAGMovieAssistant,
)


def main():

    query = (
        "Recommend a magical family adventure "
        "with fantasy, friendship and mystery."
    )

    print("\n===== CINEMIND AI RAG TEST =====")

    # Step 1: Load semantic retrieval
    print("\nLoading semantic search...")

    search_engine = SemanticMovieSearch()

    # Step 2: Retrieve movie information
    print("\nSearching for matching movies...")

    movies = search_engine.search(
        query,
        top_k=5
    )

    print("\nRetrieved Movies:")

    for index, row in movies.iterrows():
        print(
            f"[{index + 1}] {row['title']}"
        )

    # Step 3: Generate answer with Ollama
    print("\nGenerating RAG response...")

    assistant = RAGMovieAssistant(
        model="gemma3:1b"
    )

    answer, llm_used, error = assistant.generate(
        query,
        movies
    )

    print("\n===== AI GENERATED ANSWER =====\n")

    print(answer)

    print("\nLLM generation successful:", llm_used)

    if error:
        print("\nError details:", error)


if __name__ == "__main__":
    main()
