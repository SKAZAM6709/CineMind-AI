"""EDA behavior and basic invariants."""
import pandas as pd
from src.movie_recommender.data.explore import analyze


def test_genre_counts_and_minimum_rating_threshold():
    catalog = pd.DataFrame({
        "movieId": [1, 2, 3], "title": ["A", "B", "C"],
        "genres": ["Comedy|Drama", "Comedy", ""],
        "rating_count": [1, 20, 0], "average_rating": [5.0, 4.0, float("nan")]
    })
    ratings = pd.DataFrame({"userId": [1, 2, 3], "movieId": [1, 2, 2],
                            "rating": [5.0, 4.0, 4.0]})
    result = analyze(catalog, ratings)
    counts = dict(zip(result["genre_counts"].genre, result["genre_counts"].movie_count))
    assert counts == {"Comedy": 2, "Drama": 1}
    assert result["top_rated_min20"].movieId.tolist() == [2]
    assert result["rating_distribution"].rating_count.sum() == len(ratings)
