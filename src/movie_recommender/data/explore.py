"""Reusable, transparent exploratory statistics for MovieLens."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "data" / "processed" / "movie_catalog.csv"
RATINGS = ROOT / "data" / "raw" / "movielens" / "ratings.csv"
REPORTS = ROOT / "reports" / "eda"


def analyze(catalog: pd.DataFrame, ratings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return analysis tables without modifying inputs or using future target data."""
    required_catalog = {"movieId", "title", "genres", "rating_count", "average_rating"}
    required_ratings = {"userId", "movieId", "rating"}
    if not required_catalog.issubset(catalog.columns):
        raise ValueError(f"Catalog missing columns: {sorted(required_catalog - set(catalog.columns))}")
    if not required_ratings.issubset(ratings.columns):
        raise ValueError(f"Ratings missing columns: {sorted(required_ratings - set(ratings.columns))}")

    genre_counts = (
        catalog.assign(genre=catalog["genres"].fillna("").str.split("|"))
        .explode("genre")
        .query("genre != ''")
        .groupby("genre").size().reset_index(name="movie_count")
        .sort_values(["movie_count", "genre"], ascending=[False, True])
        .reset_index(drop=True)
    )
    rating_distribution = (ratings.groupby("rating").size()
                           .reset_index(name="rating_count").sort_values("rating"))
    user_activity = (ratings.groupby("userId").size()
                     .reset_index(name="ratings_given").sort_values("userId"))
    # Avoid ranking films with only one high rating as best overall.
    popular = catalog.loc[catalog["rating_count"] > 0,
                          ["movieId", "title", "rating_count", "average_rating"]]
    popular = popular.sort_values(["rating_count", "average_rating", "movieId"],
                                  ascending=[False, False, True]).head(20)
    top_rated = catalog.loc[catalog["rating_count"] >= 20,
                            ["movieId", "title", "rating_count", "average_rating"]]
    top_rated = top_rated.sort_values(["average_rating", "rating_count", "movieId"],
                                      ascending=[False, False, True]).head(20)
    return {"genre_counts": genre_counts, "rating_distribution": rating_distribution,
            "user_activity": user_activity, "most_rated_movies": popular,
            "top_rated_min20": top_rated}


def main() -> None:
    if not CATALOG.exists():
        raise SystemExit("Missing processed catalog. Run python scripts/prepare_data.py first.")
    catalog, ratings = pd.read_csv(CATALOG), pd.read_csv(RATINGS)
    results = analyze(catalog, ratings)
    REPORTS.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(REPORTS / f"{name}.csv", index=False)
    print("EDA reports saved to:", REPORTS)
    print(f"Movies: {len(catalog):,} | Ratings: {len(ratings):,} | Users: {ratings.userId.nunique():,}")
    print("\nTop 5 genres (a movie may appear in multiple genres):")
    print(results["genre_counts"].head().to_string(index=False))
    print("\nTop 5 most-rated movies:")
    print(results["most_rated_movies"].head().to_string(index=False))
    print("\nCreated 5 CSV analysis reports.")

if __name__ == "__main__":
    main()
