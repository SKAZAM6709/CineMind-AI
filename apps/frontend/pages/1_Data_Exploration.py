"""EDA dashboard. Run: python -m streamlit run apps/frontend/app.py"""
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.movie_recommender.data.explore import analyze

st.set_page_config(page_title="MovieLens Data Exploration", page_icon="📊", layout="wide")
st.title("📊 MovieLens — Exploratory Data Analysis")
st.caption("Descriptive analytics; this is not yet a recommender or an LLM system.")

@st.cache_data
def load_data():
    return (pd.read_csv(ROOT / "data" / "processed" / "movie_catalog.csv"),
            pd.read_csv(ROOT / "data" / "raw" / "movielens" / "ratings.csv"))

try:
    catalog, ratings = load_data()
except FileNotFoundError:
    st.error("Catalog not found. Run: python scripts/prepare_data.py")
    st.stop()

results = analyze(catalog, ratings)
a, b, c, d = st.columns(4)
a.metric("Movies", f"{len(catalog):,}")
b.metric("Ratings", f"{len(ratings):,}")
c.metric("Users", f"{ratings.userId.nunique():,}")
d.metric("Mean rating", f"{ratings.rating.mean():.2f} / 5")

st.subheader("Movies by genre")
st.caption("A movie can have multiple genres; category counts therefore overlap.")
st.bar_chart(results["genre_counts"].set_index("genre")["movie_count"], width="stretch")

st.subheader("Rating distribution")
st.bar_chart(results["rating_distribution"].set_index("rating")["rating_count"], width="stretch")

st.subheader("Most-rated movies")
st.dataframe(results["most_rated_movies"], width="stretch", hide_index=True)

st.subheader("Top-rated movies with at least 20 ratings")
st.caption("The minimum-rating filter reduces the risk of highlighting a movie with only one 5-star review.")
st.dataframe(results["top_rated_min20"], width="stretch", hide_index=True)

st.subheader("User rating activity")
st.dataframe(results["user_activity"].describe().round(1), width="stretch")
st.info("Next, we will enrich movie descriptions to improve text retrieval. No popularity or relevance improvement is claimed yet.")
