
"""
CineMind AI - Natural Language Semantic Movie Search

Uses the existing trained embeddings:
models/semantic_embeddings.npz

No model retraining is required.
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


class SemanticMovieSearch:

    def __init__(self):

        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Movie catalog not found: {DATA_PATH}"
            )

        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                "Semantic embeddings not found. "
                "Build the semantic model first."
            )

        # Load the same catalog used during training.
        movies = pd.read_csv(DATA_PATH)

        required = [
            "movieId",
            "title",
            "overview",
            "genres",
            "tagline",
        ]

        missing = [
            col for col in required
            if col not in movies.columns
        ]

        if missing:
            raise ValueError(
                f"Missing movie columns: {missing}"
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

        for column in [
            "title", "overview", "genres", "tagline"
        ]:
            movies[column] = (
                movies[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        # Match the exact row filtering used when
        # semantic_embeddings.npz was generated.
        movies = movies[
            movies["overview"].str.len() > 0
        ].reset_index(drop=True)

        with np.load(
            EMBEDDINGS_PATH,
            allow_pickle=False
        ) as saved:

            self.embeddings = (
                saved["embeddings"]
                .astype(np.float32)
                .copy()
            )

            self.movie_ids = (
                saved["movie_ids"]
                .astype(int)
                .tolist()
            )

            self.model_name = str(
                saved["model_name"].item()
            )

        current_ids = movies["movieId"].tolist()

        if self.movie_ids != current_ids:
            raise ValueError(
                "Movie catalog and embeddings do not "
                "match. Rebuild semantic embeddings."
            )

        if (
            self.embeddings.ndim != 2
            or self.embeddings.shape[0] != len(movies)
        ):
            raise ValueError(
                "Invalid semantic embedding matrix."
            )

        self.movies = movies

        # Load the exact same pretrained transformer.
        self.model = SentenceTransformer(
            self.model_name
        )

        if (
            self.model.get_embedding_dimension()
            != self.embeddings.shape[1]
        ):
            raise ValueError(
                "Embedding dimensions do not match."
            )

        # Already normalized by the training script.
        # Normalize again defensively.
        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True,
        )

        norms = np.maximum(norms, 1e-12)

        self.embeddings = self.embeddings / norms

        print(
            "Semantic Movie Search ready:",
            len(self.movies),
            "movies"
        )

    def search(self, query, top_k=8):

        query = str(query).strip()
        top_k = int(top_k)

        if not query:
            raise ValueError(
                "Please enter a movie description."
            )

        if top_k <= 0:
            return pd.DataFrame()

        # Convert the user's natural-language
        # description to a normalized embedding.
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)

        # Cosine similarity between the query
        # and every movie embedding.
        similarities = (
            self.embeddings @ query_vector
        )

        ranked = np.argsort(
            -similarities,
            kind="stable",
        )

        ranked = [
            int(i)
            for i in ranked
            if np.isfinite(similarities[i])
            and similarities[i] > 0
        ][:top_k]

        results = self.movies.iloc[
            ranked
        ].copy()

        results["semantic_similarity"] = [
            round(
                float(similarities[i]) * 100,
                2
            )
            for i in ranked
        ]

        return results.reset_index(drop=True)


def main():

    engine = SemanticMovieSearch()

    query = (
        "A magical family adventure involving "
        "fantasy and unexpected dangers"
    )

    print("\nUser request:", query)

    results = engine.search(
        query,
        top_k=5,
    )

    print("\nSearch results:\n")

    print(
        results[
            [
                "title",
                "semantic_similarity",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
