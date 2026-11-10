
"""
Content-Based Movie Recommendation Engine
CineMind AI

Techniques:
1. TF-IDF Vectorization
2. Cosine Similarity
3. Movie genres and tags
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

CATALOG_PATH = (
    ROOT / "data" / "processed" / "movie_catalog.csv"
)

MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "content_tfidf.joblib"


# --------------------------------------------------
# 2. PREPARE MOVIE TEXT
# --------------------------------------------------

def prepare_movie_text(row):

    title = str(row.get("title", ""))
    genres = str(row.get("genres", ""))
    tags = str(row.get("tags", ""))

    # Remove missing values
    if genres == "nan":
        genres = ""

    if tags == "nan":
        tags = ""

    # Replace separators
    genres = genres.replace("|", " ")
    tags = tags.replace("|", " ")

    # Combine movie information
    text = f"{title} {genres} {tags}"

    return text.strip()


# --------------------------------------------------
# 3. BUILD CONTENT-BASED MODEL
# --------------------------------------------------

def build_content_model():

    print("\nBuilding CineMind AI Content Model...")

    # Load movie catalog
    catalog = pd.read_csv(CATALOG_PATH)

    # Check required columns
    required = ["movieId", "title", "genres", "tags"]

    for column in required:
        if column not in catalog.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    # Generate text features
    catalog["content_text"] = catalog.apply(
        prepare_movie_text,
        axis=1
    )

    # Initialize TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=100000
    )

    # Convert text into sparse numerical vectors
    tfidf_matrix = vectorizer.fit_transform(
        catalog["content_text"]
    )

    # Save trained model
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "movie_ids": catalog["movieId"].tolist()
        },
        MODEL_PATH
    )

    print("\nContent model successfully created!")

    print("Movies processed:", len(catalog))

    print(
        "TF-IDF features:",
        len(vectorizer.get_feature_names_out())
    )

    print("Matrix shape:", tfidf_matrix.shape)

    print("Saved model:", MODEL_PATH)


# --------------------------------------------------
# 4. CONTENT-BASED RECOMMENDER
# --------------------------------------------------

class ContentBasedRecommender:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Build the model first."
            )

        self.catalog = pd.read_csv(CATALOG_PATH)

        self.model = joblib.load(MODEL_PATH)

        self.vectorizer = self.model["vectorizer"]

        self.tfidf_matrix = self.model["tfidf_matrix"]

        self.movie_ids = self.model["movie_ids"]

        # Ensure model and catalog use the same movie order
        if self.movie_ids != self.catalog["movieId"].tolist():
            raise ValueError(
                "Model and catalog are inconsistent. "
                "Rebuild the model."
            )

        self.id_to_index = {
            movie_id: index
            for index, movie_id in enumerate(self.movie_ids)
        }

    # --------------------------------------------------
    # 5. GENERATE RECOMMENDATIONS
    # --------------------------------------------------

    def recommend(
        self,
        movie_id=None,
        title=None,
        top_n=10
    ):

        if movie_id is None and title is None:
            raise ValueError(
                "Provide a movie ID or movie title."
            )

        # Find selected movie
        if movie_id is not None:

            if movie_id not in self.id_to_index:
                raise ValueError(
                    f"Movie ID {movie_id} not found."
                )

            selected_index = self.id_to_index[movie_id]

        else:

            matches = self.catalog[
                self.catalog["title"].str.lower()
                == title.lower()
            ]

            if matches.empty:
                raise ValueError(
                    f"Movie not found: {title}"
                )

            selected_id = matches.iloc[0]["movieId"]

            selected_index = self.id_to_index[selected_id]

        # Calculate cosine similarity
        similarity_scores = (
            self.tfidf_matrix[selected_index]
            @ self.tfidf_matrix.T
        ).toarray().flatten()

        # Exclude selected movie
        similarity_scores[selected_index] = -1

        # Get highest scoring movies
        ranked_indices = np.argsort(
            -similarity_scores,
            kind="stable"
        )

        results = []

        for index in ranked_indices:

            score = similarity_scores[index]

            if score <= 0:
                break

            results.append(index)

            if len(results) >= top_n:
                break

        # Return recommendation results
        recommendations = self.catalog.iloc[
            results
        ].copy()

        recommendations["similarity_score"] = [
            similarity_scores[index]
            for index in results
        ]

        recommendations["similarity_percentage"] = (
            recommendations["similarity_score"] * 100
        ).round(2)

        return recommendations[
            [
                "movieId",
                "title",
                "genres",
                "average_rating",
                "rating_count",
                "similarity_percentage"
            ]
        ].reset_index(drop=True)


# --------------------------------------------------
# 6. TEST MODEL
# --------------------------------------------------

def main():

    build_content_model()

    recommender = ContentBasedRecommender()

    print("\nTesting recommendations for Toy Story...\n")

    recommendations = recommender.recommend(
        title="Toy Story (1995)",
        top_n=10
    )

    print(
        recommendations.to_string(index=False)
    )


if __name__ == "__main__":
    main()
