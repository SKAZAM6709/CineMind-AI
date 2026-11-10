
"""
CineMind AI - NLP-Based Movie Recommender

Uses:
- Movie overviews
- Movie taglines
- TF-IDF vectorization
- Cosine similarity

Input:
data/external/tmdb/tmdb_movies.csv

Output:
models/nlp_tfidf.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT / "data" / "external"
    / "tmdb" / "tmdb_movies.csv"
)

MODEL_PATH = ROOT / "models" / "nlp_tfidf.joblib"


# ==========================================
# DATA PREPARATION
# ==========================================

def load_nlp_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"TMDb catalog not found: {DATA_PATH}"
        )

    movies = pd.read_csv(DATA_PATH)

    required = [
        "movieId",
        "title",
        "overview",
        "tagline",
    ]

    for column in required:
        if column not in movies.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    movies = movies.dropna(subset=["movieId"]).copy()

    movies["movieId"] = movies["movieId"].astype(int)

    movies = movies.drop_duplicates(
        subset=["movieId"],
        keep="last"
    )

    for column in ["title", "overview", "tagline"]:
        movies[column] = (
            movies[column].fillna("").astype(str)
        )

    # Use descriptions and taglines as NLP features
    movies["nlp_text"] = (
        movies["overview"].str.strip()
        + " "
        + movies["tagline"].str.strip()
    ).str.strip()

    # Only train on movies with usable descriptions
    movies = movies[
        movies["nlp_text"].str.len() > 0
    ].reset_index(drop=True)

    if movies.empty:
        raise ValueError(
            "No movie descriptions available for NLP."
        )

    return movies


# ==========================================
# TRAIN NLP MODEL
# ==========================================

def build_nlp_model():

    print("\nBuilding CineMind AI NLP Model...")

    movies = load_nlp_data()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )

    tfidf_matrix = vectorizer.fit_transform(
        movies["nlp_text"]
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Store movie IDs in the exact order of matrix rows
    model = {
        "vectorizer": vectorizer,
        "matrix": tfidf_matrix,
        "movie_ids": movies["movieId"].tolist(),
    }

    joblib.dump(model, MODEL_PATH)

    print("\nNLP Model successfully created!")
    print("Movies processed:", len(movies))
    print("Features:", tfidf_matrix.shape[1])
    print("Matrix shape:", tfidf_matrix.shape)
    print("Saved model:", MODEL_PATH)


# ==========================================
# RECOMMENDATION CLASS
# ==========================================

class NLPRecommender:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "NLP model not found. Train it first."
            )

        self.movies = load_nlp_data()

        self.model = joblib.load(MODEL_PATH)

        self.matrix = self.model["matrix"]
        self.movie_ids = self.model["movie_ids"]

        # Protect against stale or mismatched models
        if self.movie_ids != self.movies["movieId"].tolist():
            raise ValueError(
                "NLP model and catalog do not match. "
                "Please rebuild the NLP model."
            )

        self.id_to_index = {
            movie_id: index
            for index, movie_id in enumerate(self.movie_ids)
        }

    def recommend(self, movie_id, top_n=10):

        if movie_id not in self.id_to_index:
            raise ValueError(
                "Movie not found in the NLP catalog."
            )

        selected_index = self.id_to_index[movie_id]

        # Sparse cosine similarity:
        # TF-IDF vectors are L2-normalized
        similarities = (
            self.matrix[selected_index]
            @ self.matrix.T
        ).toarray().ravel()

        similarities[selected_index] = -1

        ranked = np.argsort(
            -similarities,
            kind="stable"
        )

        results = []

        for index in ranked:
            if similarities[index] <= 0:
                break

            results.append(index)

            if len(results) >= top_n:
                break

        output = self.movies.iloc[results].copy()

        output["nlp_similarity"] = [
            round(float(similarities[i]) * 100, 2)
            for i in results
        ]

        return output[
            [
                "movieId",
                "title",
                "overview",
                "nlp_similarity",
            ]
        ].reset_index(drop=True)


# ==========================================
# TEST NLP RECOMMENDATIONS
# ==========================================

def main():

    build_nlp_model()

    recommender = NLPRecommender()

    # Use the first movie from the available catalog
    selected = recommender.movies.iloc[0]

    print("\nSelected Movie:", selected["title"])

    recommendations = recommender.recommend(
        movie_id=int(selected["movieId"]),
        top_n=5
    )

    print("\nRecommended Movies:\n")

    print(
        recommendations[
            ["title", "nlp_similarity"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
