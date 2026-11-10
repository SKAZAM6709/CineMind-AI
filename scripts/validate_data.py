"""Inspect and validate the uploaded MovieLens data. Usage: python scripts/validate_data.py"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "movielens"
EXPECTED = {
    "movies": {"movieId", "title", "genres"},
    "ratings": {"userId", "movieId", "rating", "timestamp"},
    "tags": {"userId", "movieId", "tag", "timestamp"},
    "links": {"movieId", "imdbId", "tmdbId"},
}


def main():
    frames = {}
    for name, columns in EXPECTED.items():
        path = DATA / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        df = pd.read_csv(path)
        if set(df.columns) != columns:
            raise ValueError(f"Unexpected columns in {name}: {list(df.columns)}")
        frames[name] = df
        print(f"{name:8} rows={len(df):>7,} missing_cells={int(df.isna().sum().sum()):>4} duplicate_rows={int(df.duplicated().sum()):>4}")

    movies, ratings, tags, links = (frames[name] for name in ("movies", "ratings", "tags", "links"))
    movie_ids = set(movies["movieId"])
    assert movies["movieId"].is_unique, "Movie IDs must be unique"
    assert links["movieId"].is_unique, "Linked movie IDs must be unique"
    assert set(links["movieId"]) == movie_ids, "Links and movies have different movie IDs"
    assert set(ratings["movieId"]).issubset(movie_ids), "Ratings reference missing movies"
    assert set(tags["movieId"]).issubset(movie_ids), "Tags reference missing movies"
    assert ratings["rating"].between(0.5, 5).all(), "Ratings outside 0.5–5 scale"
    print(f"Unique users: {ratings['userId'].nunique():,}")
    print(f"Missing TMDb IDs: {links['tmdbId'].isna().sum():,}")
    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
