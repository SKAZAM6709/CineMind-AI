
"""
CineMind AI - Step 5.14B
Safe Expanded Model Rebuild

Creates staged NLP and semantic artifacts.
Validates hybrid recommendation calculations.

Does NOT overwrite live models or data.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = (
    ROOT / "data/processed/movie_catalog.csv"
)

EXPANDED_PATH = (
    ROOT / "data/external/tmdb/"
    "tmdb_movies_expanded.csv"
)

LIVE_CONTENT = (
    ROOT / "models/content_tfidf.joblib"
)

LIVE_SEMANTIC = (
    ROOT / "models/semantic_embeddings.npz"
)

STAGING = ROOT / "models/staging_500"

REPORT_PATH = (
    ROOT / "reports/evaluation/"
    "expanded_hybrid_test.csv"
)

WEIGHTS = {
    "content": 0.30,
    "nlp": 0.20,
    "semantic": 0.35,
    "popularity": 0.10,
    "rating": 0.05,
}


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_joblib_safely(data, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        suffix=".joblib",
        dir=path.parent,
        delete=False,
    ) as temp:
        temporary = Path(temp.name)

    try:
        joblib.dump(data, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_npz_safely(path, **arrays):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        suffix=".npz",
        dir=path.parent,
        delete=False,
    ) as temp:
        temporary = Path(temp.name)

    try:
        np.savez_compressed(
            temporary,
            **arrays,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_movies():
    if not EXPANDED_PATH.exists():
        raise FileNotFoundError(
            f"Expanded catalog missing: {EXPANDED_PATH}"
        )

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"MovieLens catalog missing: {CATALOG_PATH}"
        )

    tmdb = pd.read_csv(
        EXPANDED_PATH,
        encoding="utf-8-sig",
    )

    catalog = pd.read_csv(
        CATALOG_PATH,
    )

    required = [
        "movieId",
        "title",
        "overview",
        "genres",
    ]

    missing = [
        column for column in required
        if column not in tmdb.columns
    ]

    if missing:
        raise ValueError(
            f"Expanded TMDb columns missing: {missing}"
        )

    if "tagline" not in tmdb.columns:
        tmdb["tagline"] = ""

    if tmdb["movieId"].isna().any():
        raise ValueError(
            "Expanded catalog contains null movie IDs."
        )

    tmdb["movieId"] = pd.to_numeric(
        tmdb["movieId"],
        errors="raise",
    ).astype(int)

    if tmdb["movieId"].duplicated().any():
        raise ValueError(
            "Duplicate movie IDs in expanded catalog."
        )

    for column in [
        "title", "overview", "genres", "tagline"
    ]:
        tmdb[column] = (
            tmdb[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # All records must be usable for NLP and semantic
    # retrieval. Do not silently reorder/filter rows.
    invalid = (
        tmdb["overview"].str.len() < 30
    ) | (
        tmdb["title"].str.len() == 0
    )

    if invalid.any():
        raise ValueError(
            f"{invalid.sum()} movies have missing or "
            "very short descriptions/titles."
        )

    if len(tmdb) <= 150:
        raise ValueError(
            "Expanded dataset must contain "
            "more than 150 movies."
        )

    if not {
        "movieId", "rating_count", "average_rating"
    }.issubset(catalog.columns):
        raise ValueError(
            "MovieLens catalog lacks rating columns."
        )

    catalog["movieId"] = pd.to_numeric(
        catalog["movieId"],
        errors="raise",
    ).astype(int)

    catalog = catalog.drop_duplicates(
        subset="movieId",
    )

    catalog = catalog[
        [
            "movieId",
            "rating_count",
            "average_rating",
        ]
    ]

    movies = tmdb.merge(
        catalog,
        on="movieId",
        how="left",
        validate="one_to_one",
    )

    movies["rating_count"] = pd.to_numeric(
        movies["rating_count"],
        errors="coerce",
    ).fillna(0)

    movies["average_rating"] = pd.to_numeric(
        movies["average_rating"],
        errors="coerce",
    ).fillna(0)

    return movies


def build_nlp(movies):
    print("\n[1/4] Building NLP TF-IDF...")

    texts = (
        movies["title"]
        + " "
        + movies["genres"]
        + " "
        + movies["overview"]
        + " "
        + movies["tagline"]
    ).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=40000,
        min_df=1,
        dtype=np.float32,
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    if matrix.shape[0] != len(movies):
        raise AssertionError(
            "NLP matrix row count mismatch."
        )

    artifact = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "movie_ids": movies[
            "movieId"
        ].tolist(),
    }

    path = STAGING / "nlp_tfidf.joblib"

    save_joblib_safely(
        artifact,
        path,
    )

    print(
        "NLP matrix:",
        matrix.shape,
    )

    return artifact


def get_semantic_model_name():
    if not LIVE_SEMANTIC.exists():
        raise FileNotFoundError(
            "Original semantic model metadata missing."
        )

    with np.load(
        LIVE_SEMANTIC,
        allow_pickle=False,
    ) as saved:
        name = str(
            saved["model_name"].item()
        )

    return name


def build_semantic(movies, batch_size):
    print("\n[2/4] Building semantic embeddings...")

    model_name = get_semantic_model_name()

    print("Model:", model_name)

    encoder = SentenceTransformer(
        model_name
    )

    texts = (
        movies["title"]
        + ". Genres: "
        + movies["genres"]
        + ". Overview: "
        + movies["overview"]
        + ". Tagline: "
        + movies["tagline"]
    ).tolist()

    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    if embeddings.shape[0] != len(movies):
        raise AssertionError(
            "Semantic embedding row count mismatch."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Semantic embeddings contain "
            "non-finite values."
        )

    path = (
        STAGING / "semantic_embeddings.npz"
    )

    save_npz_safely(
        path,
        embeddings=embeddings,
        movie_ids=movies[
            "movieId"
        ].to_numpy(dtype=np.int64),
        model_name=np.array(
            model_name
        ),
    )

    print(
        "Semantic matrix:",
        embeddings.shape,
    )

    return embeddings, model_name


def load_content(movies):
    print("\n[3/4] Validating content model...")

    if not LIVE_CONTENT.exists():
        raise FileNotFoundError(
            LIVE_CONTENT
        )

    content = joblib.load(
        LIVE_CONTENT
    )

    required = [
        "vectorizer",
        "tfidf_matrix",
        "movie_ids",
    ]

    for key in required:
        if key not in content:
            raise ValueError(
                f"Missing content model key: {key}"
            )

    matrix = content["tfidf_matrix"]

    if not issparse(matrix):
        raise ValueError(
            "Expected sparse content TF-IDF matrix."
        )

    content_ids = [
        int(mid)
        for mid in content["movie_ids"]
    ]

    if len(content_ids) != matrix.shape[0]:
        raise ValueError(
            "Content model IDs and rows differ."
        )

    index = {
        mid: i
        for i, mid in enumerate(content_ids)
    }

    if len(index) != len(content_ids):
        raise ValueError(
            "Duplicate content model movie IDs."
        )

    missing_ids = [
        int(mid)
        for mid in movies["movieId"]
        if int(mid) not in index
    ]

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} expanded movies "
            "are missing from the content model. "
            f"Examples: {missing_ids[:10]}"
        )

    aligned_rows = [
        index[int(mid)]
        for mid in movies["movieId"]
    ]

    aligned = matrix[
        aligned_rows
    ]

    print(
        "Aligned content matrix:",
        aligned.shape,
    )

    return aligned


def normalize_positive(values):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    values = np.maximum(
        values, 0
    )

    maximum = float(
        np.max(values)
    )

    if maximum <= 0:
        return np.zeros_like(values)

    return values / maximum


def calculate_prior_scores(movies):
    popularity = pd.to_numeric(
        movies.get(
            "popularity",
            pd.Series(
                np.zeros(len(movies))
            ),
        ),
        errors="coerce",
    ).fillna(0).to_numpy()

    popularity = normalize_positive(
        np.log1p(
            np.maximum(popularity, 0)
        )
    )

    counts = movies[
        "rating_count"
    ].to_numpy(dtype=np.float32)

    averages = movies[
        "average_rating"
    ].to_numpy(dtype=np.float32)

    observed = counts > 0

    if observed.any():
        global_mean = float(
            np.average(
                averages[observed],
                weights=counts[observed],
            )
        )
    else:
        global_mean = 0.0

    # Bayesian shrinkage with an explicit prior.
    prior_count = 50.0

    bayesian_rating = (
        (
            counts * averages
            + prior_count * global_mean
        )
        / (counts + prior_count)
    )

    # MovieLens ratings are on a 0-5 scale.
    rating_score = np.clip(
        bayesian_rating / 5.0,
        0.0,
        1.0,
    )

    return popularity, rating_score


def test_hybrid(
    movies,
    content_matrix,
    nlp_artifact,
    semantic_embeddings,
):
    print("\n[4/4] Hybrid smoke test...")

    ids = movies[
        "movieId"
    ].astype(int).tolist()

    nlp_ids = [
        int(mid)
        for mid in nlp_artifact["movie_ids"]
    ]

    if ids != nlp_ids:
        raise ValueError(
            "NLP movie IDs are misaligned."
        )

    nlp_matrix = nlp_artifact["matrix"]

    if (
        len(ids) != content_matrix.shape[0]
        or len(ids) != nlp_matrix.shape[0]
        or len(ids) != semantic_embeddings.shape[0]
    ):
        raise ValueError(
            "Hybrid component row counts differ."
        )

    id_to_index = {
        mid: index
        for index, mid in enumerate(ids)
    }

    # Prefer Toy Story if it exists.
    seed_id = (
        1 if 1 in id_to_index else ids[0]
    )

    seed = id_to_index[seed_id]

    content_scores = (
        content_matrix
        @ content_matrix[seed].T
    ).toarray().ravel()

    nlp_scores = (
        nlp_matrix
        @ nlp_matrix[seed].T
    ).toarray().ravel()

    semantic_scores = (
        semantic_embeddings
        @ semantic_embeddings[seed]
    )

    popularity, rating = (
        calculate_prior_scores(movies)
    )

    final = (
        WEIGHTS["content"]
        * np.clip(content_scores, 0, 1)
        + WEIGHTS["nlp"]
        * np.clip(nlp_scores, 0, 1)
        + WEIGHTS["semantic"]
        * np.clip(semantic_scores, 0, 1)
        + WEIGHTS["popularity"]
        * popularity
        + WEIGHTS["rating"]
        * rating
    )

    final = np.asarray(final)

    if not np.isfinite(final).all():
        raise ValueError(
            "Non-finite hybrid scores."
        )

    # Never recommend the input movie.
    final[seed] = -np.inf

    ranked = np.argsort(
        -final,
        kind="stable",
    )[:10]

    report = movies.iloc[
        ranked
    ][
        ["movieId", "title", "genres"]
    ].copy()

    report["hybrid_score"] = [
        round(
            float(final[i]) * 100,
            2,
        )
        for i in ranked
    ]

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Source movie:",
        movies.iloc[seed]["title"],
    )

    print("\nTop 10 recommendations:")

    print(
        report.to_string(index=False)
    )

    return seed_id


def validate_saved_artifacts(movies):
    """Read files back and verify row alignment."""

    ids = movies["movieId"].astype(
        int
    ).tolist()

    saved_nlp = joblib.load(
        STAGING / "nlp_tfidf.joblib"
    )

    with np.load(
        STAGING / "semantic_embeddings.npz",
        allow_pickle=False,
    ) as semantic:

        semantic_ids = semantic[
            "movie_ids"
        ].astype(int).tolist()

        shape = semantic[
            "embeddings"
        ].shape

    if ids != [
        int(x)
        for x in saved_nlp["movie_ids"]
    ]:
        raise AssertionError(
            "Saved NLP IDs do not match."
        )

    if ids != semantic_ids:
        raise AssertionError(
            "Saved semantic IDs do not match."
        )

    if shape[0] != len(ids):
        raise AssertionError(
            "Saved semantic row count mismatch."
        )

    print(
        "\nSaved artifact verification: PASSED"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error(
            "--batch-size must be positive."
        )

    print(
        "\n===== CINEMIND AI MODEL REBUILD ====="
    )

    STAGING.mkdir(
        parents=True,
        exist_ok=True,
    )

    movies = load_movies()

    print(
        "Expanded movies:",
        len(movies),
    )

    nlp_artifact = build_nlp(
        movies
    )

    embeddings, model_name = build_semantic(
        movies,
        args.batch_size,
    )

    content = load_content(
        movies
    )

    seed_id = test_hybrid(
        movies,
        content,
        nlp_artifact,
        embeddings,
    )

    validate_saved_artifacts(
        movies
    )

    manifest = {
        "status": "staging_validated",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "movie_count": len(movies),
        "movie_ids": movies[
            "movieId"
        ].astype(int).tolist(),
        "catalog_sha256": file_sha256(
            EXPANDED_PATH
        ),
        "semantic_model": model_name,
        "embedding_dimensions": int(
            embeddings.shape[1]
        ),
        "weights": WEIGHTS,
        "smoke_test_seed_movie_id": seed_id,
        "note": (
            "Staged artifacts only. "
            "Live models are unchanged."
        ),
    }

    manifest_path = (
        STAGING / "build_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n===== BUILD COMPLETED ====="
    )

    print(
        "Movies indexed:",
        len(movies),
    )

    print(
        "Semantic dimensions:",
        embeddings.shape[1],
    )

    print(
        "Staged artifacts:",
        STAGING,
    )

    print(
        "Hybrid test:",
        REPORT_PATH,
    )

    print(
        "Live model files were not modified."
    )


if __name__ == "__main__":
    main()
