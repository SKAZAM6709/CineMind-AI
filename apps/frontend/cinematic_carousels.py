
from pathlib import Path
import html
import sys

import pandas as pd
import streamlit as st


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.frontend.components.movie_card import show_movie_details


BRAND_NAME = "CineMind AI"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "tmdb"
    / "tmdb_movies.csv"
)


# ==========================================
# CINEMATIC WEBSITE STYLES
# ==========================================

st.markdown("""
<style>

/* GLOBAL THEME */

:root {
    --background: #080811;
    --surface: #171523;
    --border: #302b40;
    --purple: #a855f7;
    --purple-light: #c084fc;
    --muted: #aaa5bd;
}

.stApp {
    background: var(--background);
    color: white;
}

.block-container {
    max-width: 1500px;
    padding-top: 4rem !important;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: white !important;
}


/* ==========================================
   MODERN CINEMIND HEADER
   ========================================== */

.cinemind-header {
    display: flex;
    align-items: center;
    gap: 18px;
    min-height: 100px;
    padding: 14px 0 28px;
    margin-top: 0;
    overflow: visible;
}

.cinemind-icon {
    width: 64px;
    height: 64px;
    border-radius: 17px;
    flex-shrink: 0;

    display: flex;
    align-items: center;
    justify-content: center;

    background: linear-gradient(
        135deg,
        #c084fc,
        #6d28d9
    );

    box-shadow:
        0 8px 30px rgba(168, 85, 247, .28);

    transition:
        transform .3s ease,
        box-shadow .3s ease;
}

.cinemind-icon:hover {
    transform: rotate(-5deg) scale(1.05);
    box-shadow:
        0 12px 40px rgba(168, 85, 247, .4);
}

.cinemind-icon svg {
    width: 38px;
    height: 38px;
}

.cinemind-name {
    color: #ffffff;
    font-size: clamp(27px, 3vw, 38px);
    font-weight: 850;
    letter-spacing: -1.2px;
    line-height: 1.2;
    white-space: normal;
}

.cinemind-name span {
    color: #c084fc;
}

.cinemind-tagline {
    color: #aaa5bd;
    font-size: 11px;
    letter-spacing: 2px;
    margin-top: 7px;
}


/* ==========================================
   FEATURED MOVIE HERO
   ========================================== */

.hero {
    min-height: 420px;

    background-size: cover;
    background-position: center;

    border: 1px solid #363047;
    border-radius: 24px;

    display: flex;
    align-items: center;

    padding: 55px 48px;
    margin-top: 15px;

    overflow: hidden;

    animation: fadeUp .65s ease both;
}

.hero-content {
    max-width: 570px;
}

.hero-label {
    color: #c084fc;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-title {
    color: #ffffff;
    font-size: clamp(36px, 5vw, 60px);
    font-weight: 850;
    line-height: 1.12;
    margin: 18px 0;
}

.hero-overview {
    color: #e5e1ec;
    font-size: 15px;
    line-height: 1.8;
}


/* ==========================================
   MOVIE COLLECTIONS
   ========================================== */

.section-heading {
    color: white;
    font-size: 25px;
    font-weight: 800;
    margin: 32px 0 4px;
}

.section-subtitle {
    color: #aaa5bd;
    font-size: 13px;
    margin-bottom: 15px;
}


/* ==========================================
   MOVIE CARDS
   ========================================== */

.cine-card {
    background: #171523;

    border: 1px solid #302b40;
    border-radius: 15px;

    overflow: hidden;

    transition:
        transform .3s ease,
        border-color .3s ease,
        box-shadow .3s ease;
}

.cine-card:hover {
    transform: translateY(-6px);
    border-color: #b96aff;

    box-shadow:
        0 15px 35px rgba(168, 85, 247, .23);
}

.cine-poster-wrap {
    overflow: hidden;
}

.cine-poster {
    display: block;
    width: 100%;
    aspect-ratio: 2 / 3;
    object-fit: cover;

    transition: transform .4s ease;
}

.cine-card:hover .cine-poster {
    transform: scale(1.05);
}

.cine-placeholder {
    aspect-ratio: 2 / 3;

    display: grid;
    place-items: center;

    background: #302744;
    color: #d6c9e9;
}

.cine-info {
    padding: 12px;
}

.cine-title {
    color: white;
    font-size: 13px;
    font-weight: 700;

    height: 38px;
    overflow: hidden;
}

.cine-rating {
    color: #fbbf24;
    font-size: 12px;
    margin-top: 7px;
}


/* ==========================================
   ACTION BUTTONS
   ========================================== */

div.stButton > button {
    background: linear-gradient(
        100deg,
        #9333ea,
        #6d28d9
    );

    color: white;
    border: none;
    border-radius: 10px;

    transition:
        transform .2s ease,
        box-shadow .2s ease;
}

div.stButton > button:hover {
    background: #a855f7;
    color: white;

    transform: translateY(-2px);

    box-shadow:
        0 7px 20px rgba(168, 85, 247, .3);
}


/* ==========================================
   ANIMATIONS
   ========================================== */

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


/* ==========================================
   RESPONSIVE DESIGN
   ========================================== */

@media (max-width: 768px) {

    .block-container {
        padding-top: 3rem !important;
    }

    .cinemind-header {
        gap: 12px;
        min-height: 82px;
    }

.cinemind-icon {
    width: 64px;
    height: 64px;
    flex-shrink: 0;
    border-radius: 17px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    box-shadow: 0 8px 30px rgba(168, 85, 247, 0.2);
}

    .cinemind-icon svg {
        width: 30px;
        height: 30px;
    }

    .cinemind-name {
        font-size: 27px;
    }

    .cinemind-tagline {
        font-size: 9px;
        letter-spacing: 1px;
    }

    .hero {
        min-height: 320px;
        padding: 30px 22px;
    }

    .hero-title {
        font-size: 35px;
    }
}

@media (prefers-reduced-motion: reduce) {

    .hero,
    .cine-card,
    .cine-poster,
    .cinemind-icon,
    div.stButton > button {
        animation: none;
        transition: none;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TMDB MOVIE DATA
# ==========================================

@st.cache_data
def load_movies():
    data = pd.read_csv(DATA_PATH)

    text_columns = [
        "title",
        "overview",
        "genres",
        "poster_url",
        "backdrop_url",
    ]

    for column in text_columns:
        data[column] = data[column].fillna("")

    numeric_columns = [
        "vote_average",
        "vote_count",
        "popularity",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        ).fillna(0)

    return data


if not DATA_PATH.exists():
    st.error("TMDb metadata file not found.")
    st.info("Run: python -m scripts.fetch_tmdb_metadata")
    st.stop()

movies = load_movies()


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def safe(value):
    return html.escape(str(value), quote=True)


def render_movie_card(movie, unique_key):
    title = safe(movie["title"])
    poster = str(movie["poster_url"])

    rating = float(movie["vote_average"])

    if poster.startswith("https://"):
        poster_html = (
            f'<img class="cine-poster" '
            f'src="{safe(poster)}" '
            f'alt="{title}" loading="lazy">'
        )
    else:
        poster_html = (
            '<div class="cine-placeholder">'
            'No poster available'
            '</div>'
        )

    # Keep HTML on one logical line to avoid
    # Markdown interpreting indentation as code.
    card_html = (
        '<div class="cine-card">'
        '<div class="cine-poster-wrap">'
        + poster_html
        + '</div>'
        '<div class="cine-info">'
        f'<div class="cine-title">{title}</div>'
        f'<div class="cine-rating">★ {rating:.1f}/10</div>'
        '</div>'
        '</div>'
    )

    with st.container(width=205):
        st.markdown(
            card_html,
            unsafe_allow_html=True,
        )

        if st.button(
            "View Details",
            key=unique_key,
            width="stretch",
        ):
            show_movie_details(movie)


def render_carousel(
    heading,
    subtitle,
    dataframe,
    key,
):
    if dataframe.empty:
        return

    st.markdown(
        (
            f'<div class="section-heading">'
            f'{safe(heading)}</div>'
            f'<div class="section-subtitle">'
            f'{safe(subtitle)}</div>'
        ),
        unsafe_allow_html=True,
    )

    with st.container(
        horizontal=True,
        wrap=False,
        gap="medium",
    ):
        for _, movie in dataframe.head(15).iterrows():
            render_movie_card(
                movie,
                f"{key}_{int(movie['movieId'])}",
            )



# ==========================================
# MODERN CINEMIND HEADER — FIXED
# ==========================================

import base64

LOGO_PATH = (
    PROJECT_ROOT
    / "apps"
    / "frontend"
    / "assets"
    / "images"
    / "cinemind-logo.svg"
)

if LOGO_PATH.exists():
    logo_data = base64.b64encode(
        LOGO_PATH.read_bytes()
    ).decode("utf-8")

    logo_html = (
        f'<img src="data:image/svg+xml;base64,{logo_data}" '
        'alt="CineMind AI Logo" '
        'style="width:64px;height:64px;'
        'border-radius:17px;display:block;">'
    )
else:
    logo_html = "🎬"

header_html = (
    '<div class="cinemind-header">'
    f'<div class="cinemind-icon">{logo_html}</div>'
    '<div>'
    '<div class="cinemind-name">'
    'Cine<span>Mind</span> AI'
    '</div>'
    '<div class="cinemind-tagline">'
    'DISCOVER BEYOND THE ORDINARY'
    '</div>'
    '</div>'
    '</div>'
)

st.markdown(
    header_html,
    unsafe_allow_html=True
)



# ==========================================
# FEATURED MOVIE SELECTOR
# ==========================================

st.markdown("### Featured Movie")

featured_options = movies[
    movies["backdrop_url"].str.startswith("https://")
].copy()

if not featured_options.empty:

    featured_options = featured_options.sort_values(
        "popularity",
        ascending=False,
    ).head(15)

    featured_id = st.selectbox(
        "Choose a featured movie",
        options=featured_options["movieId"].tolist(),
        format_func=lambda movie_id: (
            featured_options.loc[
                featured_options["movieId"] == movie_id,
                "title",
            ].iloc[0]
        ),
    )

    featured = featured_options[
        featured_options["movieId"] == featured_id
    ].iloc[0]

    backdrop = safe(featured["backdrop_url"])
    title = safe(featured["title"])
    overview = safe(featured["overview"])

    hero_html = (
        '<div class="hero" style="background-image:'
        'linear-gradient(90deg,'
        'rgba(8,8,17,.98) 0%,'
        'rgba(8,8,17,.80) 48%,'
        'rgba(8,8,17,.18) 100%),'
        f"url('{backdrop}');\">"
        '<div class="hero-content">'
        '<div class="hero-label">FEATURED MOVIE</div>'
        f'<div class="hero-title">{title}</div>'
        f'<p class="hero-overview">{overview[:300]}</p>'
        '</div>'
        '</div>'
    )

    st.markdown(
        hero_html,
        unsafe_allow_html=True,
    )

    if st.button(
        "🎬 Explore Featured Movie",
        key="featured_details",
    ):
        show_movie_details(featured)


# ==========================================
# MOVIE SEARCH
# ==========================================

st.markdown("## Discover Movies")

search_query = st.text_input(
    "Search movie titles",
    placeholder="Search for a movie...",
)

if search_query.strip():

    search_results = movies[
        movies["title"].str.contains(
            search_query.strip(),
            case=False,
            na=False,
            regex=False,
        )
    ]

    render_carousel(
        "Search Results",
        f"{len(search_results)} matching movies",
        search_results,
        "search",
    )

    if search_results.empty:
        st.info("No matching movies found.")


# ==========================================
# POPULAR MOVIES
# ==========================================

popular_movies = movies.sort_values(
    "popularity",
    ascending=False,
)

render_carousel(
    "Popular Movies",
    "Discover popular movies from our TMDb collection",
    popular_movies,
    "popular",
)


# ==========================================
# TOP RATED MOVIES
# ==========================================

top_rated = movies[
    movies["vote_count"] >= 20
].sort_values(
    ["vote_average", "vote_count"],
    ascending=False,
)

render_carousel(
    "Top Rated Movies",
    "Highly rated films with at least 20 TMDb votes",
    top_rated,
    "toprated",
)


# ==========================================
# GENRE DISCOVERY
# ==========================================

st.markdown("## Explore by Genre")

available_genres = sorted({
    genre.strip()
    for genre_string in movies["genres"]
    for genre in genre_string.split("|")
    if genre.strip()
})

if available_genres:

    selected_genre = st.selectbox(
        "Choose a movie genre",
        available_genres,
    )

    genre_movies = movies[
        movies["genres"].apply(
            lambda value: selected_genre in [
                genre.strip()
                for genre in value.split("|")
            ]
        )
    ].sort_values(
        "popularity",
        ascending=False,
    )

    render_carousel(
        f"{selected_genre} Movies",
        f"{len(genre_movies)} movies in this genre",
        genre_movies,
        "genre",
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "This product uses the TMDB API but is not "
    "endorsed or certified by TMDB."
)

st.caption(
    f"Movie catalog: {len(movies):,} TMDb-enriched movies. "
    "AI-powered personalized recommendations are coming soon."
)