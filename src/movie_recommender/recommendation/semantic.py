
"""
CineMind AI - Semantic Movie Recommendation

Model: sentence-transformers/all-MiniLM-L6-v2

Uses:
- Movie descriptions
- Genres
- Taglines
- Sentence embeddings
- Cosine similarity
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT / "data" / "external" / "tmdb"
    / "tmdb_movies.csv"
)

EMBEDDINGS_PATH = (
    ROOT / "models" / "semantic_embeddings.npz"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_movies():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Movie catalog not found: {DATA_PATH}"
        )

    movies = pd.read_csv(DATA_PATH)

    required = [
        "movieId",
        "title",
        "overview",
        "genres",
        "tagline"
    ]

    missing = [
        col for col in required
        if col not in movies.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    movies = movies.dropna(
        subset=["movieId"]
    ).copy()

    movies["movieId"] = (
        movies["movieId"].astype(int)
    )

    movies = movies.drop_duplicates(
        subset=["movieId"],
        keep="last"
    )

    for col in [
        "title",
        "overview",
        "genres",
        "tagline"
    ]:
        movies[col] = (
            movies[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    movies = movies[
        movies["overview"].str.len() > 0
    ].reset_index(drop=True)

    movies["semantic_text"] = (
        "Movie: " + movies["title"]
        + ". Genres: "
        + movies["genres"].str.replace(
            "|", ", ", regex=False
        )
        + ". Plot: "
        + movies["overview"]
        + ". Tagline: "
        + movies["tagline"]
    )

    return movies


def build_semantic_embeddings():

    print("\nBuilding CineMind AI Semantic Embeddings...")

    movies = load_movies()

    if movies.empty:
        raise ValueError(
            "No movie descriptions available."
        )

    print("Loading Sentence Transformer model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")

    embeddings = model.encode(
        movies["semantic_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings, dtype=np.float32
    )

    EMBEDDINGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        movie_ids=movies["movieId"].to_numpy(),
        model_name=np.array(MODEL_NAME),
    )

    print("\nSemantic embeddings created!")
    print("Movies processed:", len(movies))
    print("Embedding shape:", embeddings.shape)
    print("Saved to:", EMBEDDINGS_PATH)


class SemanticRecommender:

    def __init__(self):

        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                "Semantic embeddings not found. "
                "Generate them first."
            )

        self.movies = load_movies()

        with np.load(
            EMBEDDINGS_PATH,
            allow_pickle=False
        ) as saved:

            self.embeddings = (
                saved["embeddings"].copy()
            )

            self.movie_ids = (
                saved["movie_ids"].astype(int).tolist()
            )

            saved_model_name = str(
                saved["model_name"].item()
            )

        if saved_model_name != MODEL_NAME:
            raise ValueError(
                "Embedding model mismatch."
            )

        current_ids = (
            self.movies["movieId"].tolist()
        )

        if current_ids != self.movie_ids:
            raise ValueError(
                "Catalog has changed. "
                "Regenerate semantic embeddings."
            )

        if (
            len(self.embeddings) != len(self.movies)
            or self.embeddings.ndim != 2
        ):
            raise ValueError(
                "Invalid semantic embedding matrix."
            )

        self.id_to_index = {
            movie_id: index
            for index, movie_id
            in enumerate(self.movie_ids)
        }

    def recommend(self, movie_id, top_n=10):

        movie_id = int(movie_id)

        if movie_id not in self.id_to_index:
            raise ValueError(
                f"Movie ID {movie_id} not found."
            )

        if top_n <= 0:
            return pd.DataFrame()

        index = self.id_to_index[movie_id]

        # Normalized vectors: dot product equals cosine similarity
        similarity = (
            self.embeddings
            @ self.embeddings[index]
        )

        similarity[index] = -np.inf

        ranked = np.argsort(
            -similarity,
            kind="stable"
        )

        ranked = [
            int(i)
            for i in ranked
            if similarity[i] > 0
        ][:top_n]

        results = self.movies.iloc[
            ranked
        ].copy()

        results["semantic_similarity"] = [
            round(
                float(similarity[i]) * 100,
                2
            )
            for i in ranked
        ]

        return results[
            [
                "movieId",
                "title",
                "genres",
                "semantic_similarity"
            ]
        ].reset_index(drop=True)


def main():

    build_semantic_embeddings()

    recommender = SemanticRecommender()

    selected_movie = recommender.movies.iloc[0]

    print(
        "\nSelected movie:",
        selected_movie["title"]
    )

    recommendations = recommender.recommend(
        movie_id=int(selected_movie["movieId"]),
        top_n=5
    )

    print("\nSemantic Recommendations:\n")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
