
"""Standalone visual prototype. Existing app.py and EDA pages are unchanged."""
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets" / "css"
CATALOG = ROOT / "data" / "processed" / "movie_catalog.csv"

st.set_page_config(page_title="LUMIÈRE | Movie Discovery", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")
for filename in ("main.css", "animations.css"):
    st.markdown(f"<style>{(ASSETS / filename).read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

if not CATALOG.exists():
    st.error("Movie catalog not found. Run: python scripts/prepare_data.py")
    st.stop()

movies = load_catalog(str(CATALOG))
st.markdown("""
<div class="cine-nav">
  <div class="cine-logo">LUMI<span>ÈRE</span> <span style="font-size:.7rem;letter-spacing:.12em;color:#ac9bbb;">AI</span></div>
  <div class="cine-links"><a href="#discover">Discover</a><a href="#movie-collection">Movies</a></div>
  <div class="cine-pill">✦ Personalized movie discovery</div>
</div>
<div class="cine-hero">
  <div class="cine-hero-content">
    <div class="cine-kicker">The next chapter of movie discovery</div>
    <h1>Stories for <em>every mood.</em></h1>
    <div class="cine-sub">Explore cinema with a new perspective. Browse titles now; conversational AI discovery is coming in a later development phase.</div>
    <span class="cine-chip">✦ Discover</span><span class="cine-chip">◈ Explore genres</span><span class="cine-chip">⌁ Curated by data</span>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown('<div id="discover"></div><div class="cine-section">Discover your next film</div><div class="cine-section-desc">Browse real titles from your MovieLens catalog. Search is keyword-based for now.</div>', unsafe_allow_html=True)
search_col, genre_col = st.columns([3, 1])
with search_col:
    search = st.text_input("Search movie titles", placeholder="Try: Interstellar, Toy Story, Matrix…")
with genre_col:
    genre = st.selectbox("Filter by genre", ["All genres", "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller"])

filtered = movies.copy()
if search.strip():
    filtered = filtered[filtered["title"].str.contains(search.strip(), case=False, na=False, regex=False)]
if genre != "All genres":
    filtered = filtered[filtered["genres"].fillna("").str.split("|").apply(lambda values: genre in values)]

# An exploratory showcase, not an AI ranking. Use rating count for reproducibility.
filtered = filtered.sort_values(["rating_count", "movieId"], ascending=[False, True]).head(10)
colors = ["#8b56bd", "#5173b7", "#ae5572", "#3b9b8c", "#ad7b42"]
cards = []
for i, movie in enumerate(filtered.itertuples(index=False)):
    name = escape(str(movie.title))
    initial = escape(str(movie.title).strip()[:1].upper())
    label = escape(str(movie.genres).replace("|", " · ").split(" · ")[0])
    rating = "Not rated" if pd.isna(movie.average_rating) else f"★ {float(movie.average_rating):.1f}/5"
    cards.append(f"""<article class="cine-card"><div class="cine-cover" style="--card-accent:{colors[i%len(colors)]}"><span class="cine-initial">{initial}</span></div><div class="cine-card-content"><div class="cine-movie-title">{name}</div><div class="cine-movie-meta">{label} · {rating}</div></div></article>""")
st.markdown('<div id="movie-collection"></div><div class="cine-section">Movie collection</div><div class="cine-section-desc">Most-rated titles matching your filters · ' + f'{len(filtered)} shown</div>', unsafe_allow_html=True)
if cards:
    st.markdown('<div class="cine-cards">' + ''.join(cards) + '</div>', unsafe_allow_html=True)
else:
    st.info("No matching movies. Try another title or genre.")
st.markdown('<div class="cine-disclaimer">Visual prototype · No TMDb posters, AI-generated recommendations, or user profiles connected yet. Demo brand: LUMIÈRE AI.</div>', unsafe_allow_html=True)
