
from pathlib import Path
import sys
import html

import pandas as pd
import streamlit as st


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.frontend.components.movie_card import show_movie_details


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "tmdb"
    / "tmdb_movies_sample.csv"
)

BRAND_NAME = "CineMind AI"


st.set_page_config(
    page_title=f"{BRAND_NAME} | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================
# CINEMATIC STYLES
# ==========================================

st.markdown("""
<style>
:root {
    --bg: #080811;
    --surface: #151521;
    --border: #302d42;
    --purple: #a855f7;
    --text: #ffffff;
    --muted: #b3afc5;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #ffffff !important;
}

/* BRAND */

.brand-title {
    font-size: 32px;
    font-weight: 850;
    letter-spacing: -1px;
    color: white;
}

.brand-title span {
    color: #c084fc;
}

.brand-subtitle {
    color: #a8a7bb;
    font-size: 12px;
    letter-spacing: 2px;
    margin: 5px 0 28px;
}

/* FEATURED HERO */

.hero {
    min-height: 390px;
    background-size: cover;
    background-position: center;
    border-radius: 24px;
    padding: 60px 45px;
    display: flex;
    align-items: center;
    overflow: hidden;
    animation: fadeUp .7s ease both;
}

.hero-content {
    max-width: 560px;
}

.hero-label {
    color: #c084fc;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 800;
}

.hero-title {
    color: white;
    font-size: clamp(32px, 5vw, 56px);
    font-weight: 850;
    line-height: 1.15;
    margin: 18px 0;
}

.hero-description {
    color: #e2dfea;
    line-height: 1.8;
    font-size: 15px;
}

/* MOVIE CARDS */

.movie-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 15px;
    overflow: hidden;
    margin: 8px 0 10px;
    transition:
        transform .3s ease,
        box-shadow .3s ease,
        border-color .3s ease;
}

.movie-card:hover {
    transform: translateY(-7px);
    border-color: var(--purple);
    box-shadow: 0 14px 35px rgba(168,85,247,.2);
}

.movie-poster {
    display: block;
    width: 100%;
    aspect-ratio: 2 / 3;
    object-fit: cover;
}

.poster-placeholder {
    aspect-ratio: 2 / 3;
    display: grid;
    place-items: center;
    color: #a8a7bb;
    background: #232235;
}

.movie-info {
    padding: 14px;
}

.movie-title {
    font-weight: 700;
    color: white;
    font-size: 14px;
    min-height: 42px;
}

.movie-rating {
    color: #fbbf24;
    font-size: 13px;
    padding-top: 8px;
}

/* ACTION BUTTONS */

div.stButton > button {
    background: linear-gradient(
        100deg, #9333ea, #6d28d9
    );
    color: #ffffff;
    border: none;
    border-radius: 10px;
    transition:
        transform .2s ease,
        box-shadow .2s ease;
}

div.stButton > button:hover {
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 7px 20px rgba(168,85,247,.3);
}

/* ==========================================
   DARK MOVIE DETAILS DIALOG
   ========================================== */

div[role="dialog"][aria-modal="true"] {
    background: #11101d !important;
    color: #f8f7ff !important;
    border: 1px solid #403454 !important;
    border-radius: 22px !important;
    box-shadow:
        0 28px 90px rgba(0,0,0,.75),
        0 0 35px rgba(168,85,247,.12);
    max-height: 90vh;
    overflow-y: auto;
    animation: dialogEnter .28s ease-out;
}

div[role="dialog"][aria-modal="true"] * {
    color: inherit;
}

div[role="dialog"][aria-modal="true"] h1,
div[role="dialog"][aria-modal="true"] h2,
div[role="dialog"][aria-modal="true"] h3 {
    color: #ffffff !important;
}

div[role="dialog"][aria-modal="true"] img {
    border-radius: 12px;
}

.detail-title {
    font-size: clamp(25px, 3vw, 36px);
    font-weight: 850;
    color: white;
    letter-spacing: -.7px;
    line-height: 1.2;
}

.detail-subtitle {
    color: #b790f8 !important;
    letter-spacing: 2px;
    font-size: 10px;
    font-weight: 750;
    margin-top: 6px;
    margin-bottom: 22px;
}

.detail-rating {
    display: inline-block;
    color: #fbbf24 !important;
    background: rgba(251,191,36,.1);
    border: 1px solid rgba(251,191,36,.2);
    border-radius: 30px;
    padding: 7px 13px;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 20px;
}

.detail-meta {
    display: flex;
    flex-direction: column;
    gap: 5px;
    border-bottom: 1px solid #302d42;
    padding: 11px 0;
}

.detail-meta span {
    color: #aaa5bd !important;
    font-size: 12px;
}

.detail-meta strong {
    color: #f7f5ff !important;
    font-size: 14px;
    font-weight: 650;
}

.detail-section-title {
    font-size: 19px;
    color: #ffffff !important;
    font-weight: 800;
    margin-top: 22px;
    margin-bottom: 9px;
}

/* ANIMATION */

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes dialogEnter {
    from {
        opacity: 0;
        transform: translateY(10px) scale(.985);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

/* MOBILE */

@media (max-width: 768px) {
    .hero {
        min-height: 310px;
        padding: 30px 22px;
    }

    .hero-title {
        font-size: 32px;
    }

    .brand-title {
        font-size: 26px;
    }
}

@media (prefers-reduced-motion: reduce) {
    .hero,
    .movie-card,
    div.stButton > button,
    div[role="dialog"][aria-modal="true"] {
        animation: none;
        transition: none;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MOVIES
# ==========================================

@st.cache_data
def load_movies():
    return pd.read_csv(DATA_PATH).fillna("")


if not DATA_PATH.exists():
    st.error("TMDb metadata file not found.")
    st.info("Run: python -m scripts.fetch_tmdb_metadata")
    st.stop()

movies = load_movies()


# ==========================================
# HEADER
# ==========================================

st.markdown(
    f'<div class="brand-title">🎬 {html.escape(BRAND_NAME)}</div>'
    '<div class="brand-subtitle">'
    'DISCOVER STORIES BEYOND THE ORDINARY'
    '</div>',
    unsafe_allow_html=True,
)


# ==========================================
# FEATURED MOVIE
# ==========================================

featured_movies = movies[
    movies["backdrop_url"]
    .astype(str)
    .str.startswith("https://")
]

if not featured_movies.empty:
    featured = featured_movies.iloc[0]

    title = html.escape(str(featured["title"]))
    overview = html.escape(str(featured["overview"]))
    backdrop = html.escape(
        str(featured["backdrop_url"]),
        quote=True
    )

    hero_html = f"""
<div class="hero" style="background-image:
linear-gradient(90deg,rgba(8,8,17,.98) 0%,rgba(8,8,17,.8) 46%,rgba(8,8,17,.15) 100%),
url('{backdrop}');">
    <div class="hero-content">
        <div class="hero-label">FEATURED MOVIE</div>
        <div class="hero-title">{title}</div>
        <p class="hero-description">{overview[:280]}</p>
    </div>
</div>
"""

    st.markdown(
        hero_html.strip(),
        unsafe_allow_html=True,
    )

    if st.button(
        "🎬 View Featured Movie Details",
        key="featured_details",
    ):
        show_movie_details(featured)


# ==========================================
# MOVIE SEARCH
# ==========================================

st.markdown("## Explore Movies")

search = st.text_input(
    "Search movies",
    placeholder="Search by movie title...",
)

filtered = movies.copy()

if search.strip():
    filtered = filtered[
        filtered["title"]
        .astype(str)
        .str.contains(
            search.strip(),
            case=False,
            na=False,
            regex=False
        )
    ]

st.caption(f"Showing {len(filtered)} movies")


# ==========================================
# MOVIE CARDS
# ==========================================

columns = st.columns(5)

for index, (_, movie) in enumerate(filtered.iterrows()):
    with columns[index % 5]:

        title = html.escape(str(movie["title"]))
        poster = str(movie["poster_url"])

        rating = pd.to_numeric(
            movie["vote_average"],
            errors="coerce"
        )

        rating_text = (
            f"{rating:.1f}/10"
            if pd.notna(rating)
            else "Not rated"
        )

        if poster.startswith("https://"):
            safe_url = html.escape(poster, quote=True)
            poster_html = (
                f'<img class="movie-poster" '
                f'src="{safe_url}" '
                f'alt="{title}" loading="lazy">'
            )
        else:
            poster_html = (
                '<div class="poster-placeholder">'
                'No poster available'
                '</div>'
            )

        card_html = f"""
<div class="movie-card">
    {poster_html}
    <div class="movie-info">
        <div class="movie-title">{title}</div>
        <div class="movie-rating">★ {rating_text}</div>
    </div>
</div>
"""

        st.markdown(
            card_html.strip(),
            unsafe_allow_html=True,
        )

        if st.button(
            "View Details",
            key=f"details_{int(movie['movieId'])}",
            width="stretch",
        ):
            show_movie_details(movie)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "This product uses the TMDB API but is not "
    "endorsed or certified by TMDB."
)
