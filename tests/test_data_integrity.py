"""Lightweight tests against the provided movie metadata."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MOVIES = ROOT / "data" / "raw" / "movielens" / "movies.csv"


def test_movies_have_unique_ids():
    movies = pd.read_csv(MOVIES)
    assert movies["movieId"].is_unique


def test_movie_titles_are_present():
    movies = pd.read_csv(MOVIES)
    assert movies["title"].notna().all()
