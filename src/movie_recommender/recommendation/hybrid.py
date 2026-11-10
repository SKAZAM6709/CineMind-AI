
"""
CineMind AI - Hybrid Movie Recommendation Engine

Combines:
- Content-Based TF-IDF (30%)
- NLP TF-IDF (20%)
- Semantic Embeddings (35%)
- TMDb Popularity (10%)
- Bayesian Weighted Rating (5%)

Supports:
- Single-movie recommendations
- Multi-movie preference recommendations
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[3]

CONTENT_MODEL = ROOT / "models" / "content_tfidf.joblib"
NLP_MODEL = ROOT / "models" / "nlp_tfidf.joblib"
SEMANTIC_MODEL = ROOT / "models" / "semantic_embeddings.npz"

MOVIELENS_PATH = ROOT / "data" / "processed" / "movie_catalog.csv"
TMDB_PATH = ROOT / "data" / "external" / "tmdb" / "tmdb_movies.csv"


# ==========================================
# MODEL WEIGHTS
# ==========================================

WEIGHTS = {
    "content": 0.30,
    "nlp": 0.20,
    "semantic": 0.35,
    "popularity": 0.10,
    "rating": 0.05,
}


class HybridRecommender:

    def __init__(self):

        print("Loading CineMind AI Hybrid Engine...")

        for path in [
            CONTENT_MODEL,
            NLP_MODEL,
            SEMANTIC_MODEL,
            MOVIELENS_PATH,
            TMDB_PATH,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {path}"
                )

        # Load trained recommendation models
        self.content_model = joblib.load(CONTENT_MODEL)
        self.nlp_model = joblib.load(NLP_MODEL)

        with np.load(SEMANTIC_MODEL, allow_pickle=False) as saved:
            self.semantic_matrix = (
                saved["embeddings"].astype(np.float32).copy()
            )

            semantic_ids = (
                saved["movie_ids"].astype(int).tolist()
            )

        self.content_matrix = self.content_model["tfidf_matrix"]
        self.nlp_matrix = self.nlp_model["matrix"]

        # Build movie ID mappings
        self.content_index = {
            int(mid): i
            for i, mid in enumerate(self.content_model["movie_ids"])
        }

        self.nlp_index = {
            int(mid): i
            for i, mid in enumerate(self.nlp_model["movie_ids"])
        }

        self.semantic_index = {
            mid: i for i, mid in enumerate(semantic_ids)
        }

        # Validate model dimensions
        if self.content_matrix.shape[0] != len(self.content_index):
            raise ValueError("Invalid content model.")

        if self.nlp_matrix.shape[0] != len(self.nlp_index):
            raise ValueError("Invalid NLP model.")

        if self.semantic_matrix.shape[0] != len(self.semantic_index):
            raise ValueError("Invalid semantic model.")

        # Load movie metadata
        movielens = pd.read_csv(MOVIELENS_PATH)
        tmdb = pd.read_csv(TMDB_PATH)

        for df in [movielens, tmdb]:
            df["movieId"] = pd.to_numeric(
                df["movieId"], errors="coerce"
            )

            df.dropna(subset=["movieId"], inplace=True)
            df["movieId"] = df["movieId"].astype(int)

            df.drop_duplicates(
                "movieId", keep="last", inplace=True
            )

        movielens = movielens[
            ["movieId", "rating_count", "average_rating", "tags"]
        ]

        self.movies = tmdb.merge(
            movielens,
            on="movieId",
            how="inner",
            validate="one_to_one",
        )

        # Keep movies available in all three models
        available_ids = (
            set(self.content_index)
            & set(self.nlp_index)
            & set(self.semantic_index)
        )

        self.movies = self.movies[
            self.movies["movieId"].isin(available_ids)
        ].copy()

        self.movies = (
            self.movies.sort_values("movieId")
            .reset_index(drop=True)
        )

        if len(self.movies) < 2:
            raise ValueError("Not enough common movies across models.")

        # Clean metadata
        self.movies["vote_average"] = pd.to_numeric(
            self.movies["vote_average"],
            errors="coerce",
        ).fillna(0).clip(0, 10)

        self.movies["vote_count"] = pd.to_numeric(
            self.movies["vote_count"],
            errors="coerce",
        ).fillna(0).clip(lower=0)

        self.movies["popularity"] = pd.to_numeric(
            self.movies["popularity"],
            errors="coerce",
        ).fillna(0).clip(lower=0)

        self.movies["title"] = self.movies["title"].fillna("")

        # Popularity normalization
        log_popularity = np.log1p(
            self.movies["popularity"].to_numpy()
        )

        max_popularity = log_popularity.max()

        if max_popularity > 0:
            self.movies["popularity_score"] = (
                log_popularity / max_popularity
            )
        else:
            self.movies["popularity_score"] = 0.0

        # Bayesian weighted rating
        valid = self.movies[
            self.movies["vote_count"] > 0
        ]

        global_mean = (
            float(valid["vote_average"].mean())
            if not valid.empty else 0.0
        )

        minimum_votes = max(
            float(self.movies["vote_count"].quantile(0.60)),
            1.0,
        )

        v = self.movies["vote_count"].to_numpy()
        R = self.movies["vote_average"].to_numpy()

        weighted_rating = (
            (v / (v + minimum_votes)) * R
            + (minimum_votes / (v + minimum_votes)) * global_mean
        )

        self.movies["rating_score"] = np.clip(
            weighted_rating / 10.0, 0.0, 1.0
        )

        # Store movie IDs
        self.movie_ids = (
            self.movies["movieId"].astype(int).tolist()
        )

        # Align model rows with candidate catalog
        self.content_rows = [
            self.content_index[mid] for mid in self.movie_ids
        ]

        self.nlp_rows = [
            self.nlp_index[mid] for mid in self.movie_ids
        ]

        self.semantic_rows = [
            self.semantic_index[mid] for mid in self.movie_ids
        ]

        self.catalog_index = {
            mid: i for i, mid in enumerate(self.movie_ids)
        }

        print(
            f"Hybrid engine ready: {len(self.movies)} common movies."
        )

    # ==========================================
    # MULTI-MOVIE PREFERENCE RECOMMENDATIONS
    # ==========================================

    def recommend_for_preferences(self, movie_ids, top_n=10):

        movie_ids = list(dict.fromkeys(
            int(mid) for mid in movie_ids
        ))

        top_n = int(top_n)

        if not movie_ids:
            raise ValueError("Select at least one movie.")

        if top_n <= 0:
            return self.movies.iloc[0:0].copy()

        invalid_ids = [
            mid for mid in movie_ids
            if mid not in self.catalog_index
        ]

        if invalid_ids:
            raise ValueError(
                f"Movies unavailable in hybrid catalog: {invalid_ids}"
            )

        # Find selected movie rows in each model
        content_sources = [
            self.content_index[mid] for mid in movie_ids
        ]

        nlp_sources = [
            self.nlp_index[mid] for mid in movie_ids
        ]

        semantic_sources = [
            self.semantic_index[mid] for mid in movie_ids
        ]

        # Average content similarity across selected movies
        content_scores = (
            self.content_matrix[content_sources]
            @ self.content_matrix[self.content_rows].T
        ).toarray().mean(axis=0)

        # Average NLP similarity
        nlp_scores = (
            self.nlp_matrix[nlp_sources]
            @ self.nlp_matrix[self.nlp_rows].T
        ).toarray().mean(axis=0)

        # Average semantic similarity
        semantic_scores = (
            self.semantic_matrix[self.semantic_rows]
            @ self.semantic_matrix[semantic_sources].T
        ).mean(axis=1)

        # Keep similarity scores between 0 and 1
        content_scores = np.clip(content_scores, 0, 1)
        nlp_scores = np.clip(nlp_scores, 0, 1)
        semantic_scores = np.clip(semantic_scores, 0, 1)

        popularity_scores = (
            self.movies["popularity_score"].to_numpy()
        )

        rating_scores = (
            self.movies["rating_score"].to_numpy()
        )

        # Calculate hybrid recommendation score
        hybrid_scores = (
            WEIGHTS["content"] * content_scores
            + WEIGHTS["nlp"] * nlp_scores
            + WEIGHTS["semantic"] * semantic_scores
            + WEIGHTS["popularity"] * popularity_scores
            + WEIGHTS["rating"] * rating_scores
        )

        # Exclude every selected movie
        for mid in movie_ids:
            position = self.catalog_index[mid]
            hybrid_scores[position] = -np.inf

        # Rank recommendations
        ranked = np.argsort(
            -hybrid_scores,
            kind="stable",
        )

        ranked = ranked[:top_n]

        results = self.movies.iloc[ranked].copy()

        # Individual recommendation contributions
        results["content_similarity"] = (
            content_scores[ranked] * 100
        ).round(2)

        results["nlp_similarity"] = (
            nlp_scores[ranked] * 100
        ).round(2)

        results["semantic_similarity"] = (
            semantic_scores[ranked] * 100
        ).round(2)

        results["hybrid_score"] = (
            hybrid_scores[ranked] * 100
        ).round(2)

        columns = [
            "movieId",
            "title",
            "genres",
            "poster_url",
            "overview",
            "vote_average",
            "content_similarity",
            "nlp_similarity",
            "semantic_similarity",
            "hybrid_score",
        ]

        return results[columns].reset_index(drop=True)

    # ==========================================
    # SINGLE-MOVIE RECOMMENDATIONS
    # ==========================================

    def recommend(self, movie_id, top_n=10):

        return self.recommend_for_preferences(
            movie_ids=[movie_id],
            top_n=top_n,
        )


# ==========================================
# MODEL TEST
# ==========================================

def main():

    engine = HybridRecommender()

    print("\nTesting Multi-Movie Preferences\n")

    selected_movies = []

    for search_title in [
        "Toy Story",
        "Jumanji",
        "Babe",
    ]:

        matches = engine.movies[
            engine.movies["title"].str.contains(
                search_title,
                case=False,
                na=False,
            )
        ]

        if not matches.empty:
            selected_movies.append(
                int(matches.iloc[0]["movieId"])
            )

    if not selected_movies:
        raise ValueError("No test movies found.")

    print("Selected Movie IDs:", selected_movies)

    recommendations = engine.recommend_for_preferences(
        movie_ids=selected_movies,
        top_n=10,
    )

    print("\nPersonalized Hybrid Recommendations:\n")

    print(
        recommendations[
            [
                "title",
                "content_similarity",
                "nlp_similarity",
                "semantic_similarity",
                "hybrid_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
