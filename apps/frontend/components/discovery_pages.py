
"""
CineMind AI - Movie Discovery Pages

Shared pages:
- Trending Movies
- Top Rated Movies
- Explore by Genre

Features:
- Equal-height movie cards
- Aligned titles, genres, ratings and buttons
- Cinematic poster hover animations
- Movie Details popup
- Watchlist integration
- Genre filtering
"""

from pathlib import Path
import sys
import html

import pandas as pd
import streamlit as st


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from apps.frontend.components.movie_card import (
    show_movie_details,
)

from apps.frontend.components.watchlist_manager import (
    initialize_watchlist,
    is_saved,
    add_to_watchlist,
    remove_from_watchlist,
)


DATA_PATH = (
    ROOT
    / "data"
    / "external"
    / "tmdb"
    / "tmdb_movies.csv"
)


# ==========================================
# CINEMATIC CSS
# ==========================================

DISCOVERY_CSS = """
<style>

/* PAGE HEADER */

.dc-header {
    font-size: clamp(30px, 4vw, 44px);
    font-weight: 850;
    letter-spacing: -1px;
    color: #ffffff;
    margin-bottom: 8px;
}

.dc-header span {
    color: #c084fc;
}

.dc-subtitle {
    font-size: 15px;
    color: #aaa5bd;
    margin-bottom: 24px;
}


/* MOVIE CARD */

.dc-card {
    width: 100%;
    background: #171421;
    border: 1px solid #342743;
    border-radius: 15px;
    overflow: hidden;

    margin: 0 0 8px 0;

    transition:
        transform 0.3s ease,
        border-color 0.3s ease,
        box-shadow 0.3s ease;
}

.dc-card:hover {
    transform: translateY(-5px);
    border-color: #a855f7;
    box-shadow:
        0 12px 28px rgba(168, 85, 247, 0.20);
}


/* FIXED POSTER DIMENSIONS */

.dc-poster {
    width: 100%;
    aspect-ratio: 2 / 3;
    position: relative;
    overflow: hidden;
    background: #252033;
}

.dc-poster img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;

    transition: transform 0.4s ease;
}

.dc-card:hover .dc-poster img {
    transform: scale(1.06);
}


/* POSTER HOVER */

.dc-poster-overlay {
    position: absolute;
    inset: 0;

    background: linear-gradient(
        to top,
        rgba(10, 8, 20, 0.65),
        transparent 70%
    );

    opacity: 0;
    transition: opacity 0.3s ease;
}

.dc-card:hover .dc-poster-overlay {
    opacity: 1;
}


/* POSTER FALLBACK */

.dc-no-poster {
    width: 100%;
    height: 100%;

    display: flex;
    align-items: center;
    justify-content: center;

    color: #aaa5bd;
    font-size: 13px;
    text-align: center;
}


/* FIXED INFORMATION AREA */

.dc-body {
    height: 118px;
    min-height: 118px;

    box-sizing: border-box;
    padding: 12px;

    display: flex;
    flex-direction: column;
}


/* TITLE - ALWAYS RESERVE TWO LINES */

.dc-title {
    color: #ffffff;
    font-size: 15px;
    font-weight: 750;
    line-height: 22px;

    height: 44px;
    min-height: 44px;
    max-height: 44px;

    overflow: hidden;

    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    line-clamp: 2;

    overflow-wrap: anywhere;
}


/* GENRE - SAME POSITION FOR EVERY MOVIE */

.dc-genres {
    color: #aaa5bd;
    font-size: 12px;
    line-height: 18px;

    margin-top: 5px;

    height: 18px;
    min-height: 18px;

    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}


/* RATING - SAME POSITION */

.dc-rating {
    color: #fbbf24;
    font-size: 14px;
    font-weight: 650;
    line-height: 20px;

    margin-top: 7px;

    height: 20px;
    min-height: 20px;
}


/* CARD BUTTONS */

.dc-actions {
    margin-top: 0;
}


/* RESPONSIVE */

@media (max-width: 650px) {

    .dc-title {
        font-size: 13px;
        line-height: 20px;

        height: 40px;
        min-height: 40px;
    }

    .dc-body {
        height: 114px;
        min-height: 114px;
        padding: 10px;
    }

    .dc-genres {
        font-size: 11px;
    }

    .dc-rating {
        font-size: 12px;
    }
}


/* ACCESSIBILITY */

@media (prefers-reduced-motion: reduce) {

    .dc-card,
    .dc-poster img,
    .dc-poster-overlay {
        transition: none;
    }

    .dc-card:hover {
        transform: none;
    }

    .dc-card:hover img {
        transform: none;
    }
}

</style>
"""


# ==========================================
# LOAD MOVIES
# ==========================================

@st.cache_data(show_spinner=False)
def load_movies(path_text, modified_time):

    movies = pd.read_csv(
        path_text,
        encoding="utf-8-sig"
    )

    movies = movies.drop_duplicates(
        subset="movieId"
    ).copy()

    text_columns = [
        "title",
        "genres",
        "poster_url",
        "overview",
    ]

    for column in text_columns:

        if column not in movies.columns:
            movies[column] = ""

        movies[column] = (
            movies[column]
            .fillna("")
            .astype(str)
        )

    numeric_columns = [
        "popularity",
        "vote_average",
        "vote_count",
    ]

    for column in numeric_columns:

        if column not in movies.columns:
            movies[column] = 0

        movies[column] = pd.to_numeric(
            movies[column],
            errors="coerce"
        ).fillna(0)

    return movies


# ==========================================
# MOVIE CARD COMPONENT
# ==========================================

def movie_card(movie, prefix):

    movie_id = int(movie["movieId"])

    title = html.escape(
        str(movie.get("title", "")),
        quote=True
    )

    poster = str(
        movie.get("poster_url", "")
    )

    genres = html.escape(
        str(movie.get("genres", "")),
        quote=True
    ).replace("|", " • ")

    try:
        rating = float(
            movie.get("vote_average", 0)
        )
    except (TypeError, ValueError):
        rating = 0.0

    # --------------------------------------
    # POSTER
    # --------------------------------------

    if poster.startswith("https://"):

        safe_poster = html.escape(
            poster,
            quote=True
        )

        poster_html = (
            f'<img src="{safe_poster}" '
            f'alt="{title}" loading="lazy">'
        )

    else:

        poster_html = (
            '<div class="dc-no-poster">'
            'Poster Unavailable'
            '</div>'
        )

    # --------------------------------------
    # CARD HTML
    # --------------------------------------

    card_html = (
        '<div class="dc-card">'
        '<div class="dc-poster">'
        f'{poster_html}'
        '<div class="dc-poster-overlay"></div>'
        '</div>'

        '<div class="dc-body">'

        f'<div class="dc-title" title="{title}">'
        f'{title}'
        '</div>'

        f'<div class="dc-genres" title="{genres}">'
        f'{genres}'
        '</div>'

        '<div class="dc-rating">'
        f'★ {rating:.1f}/10'
        '</div>'

        '</div>'
        '</div>'
    )

    # Keep HTML as one continuous string.
    st.markdown(
        card_html,
        unsafe_allow_html=True
    )

    # --------------------------------------
    # VIEW DETAILS
    # --------------------------------------

    if st.button(
        "View Details",
        key=f"{prefix}_details_{movie_id}",
        width="stretch",
    ):

        show_movie_details(movie)

    # --------------------------------------
    # WATCHLIST
    # --------------------------------------

    if is_saved(movie_id):

        if st.button(
            "✓ Saved — Remove",
            key=f"{prefix}_remove_{movie_id}",
            width="stretch",
        ):

            remove_from_watchlist(
                movie_id
            )

            st.rerun()

    else:

        if st.button(
            "♡ Add to Watchlist",
            key=f"{prefix}_save_{movie_id}",
            width="stretch",
        ):

            add_to_watchlist(
                movie_id
            )

            st.rerun()


# ==========================================
# MAIN DISCOVERY PAGE
# ==========================================

def render_discovery_page(mode):

    initialize_watchlist()

    st.markdown(
        DISCOVERY_CSS,
        unsafe_allow_html=True
    )

    if not DATA_PATH.exists():

        st.error(
            f"Movie catalog not found: {DATA_PATH}"
        )

        return

    movies = load_movies(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime_ns,
    )

    # --------------------------------------
    # PAGE CONFIGURATION
    # --------------------------------------

    page_config = {

        "trending": (
            "Trending Movies",
            "Discover popular movies from the "
            "CineMind AI catalog.",
        ),

        "top_rated": (
            "Top Rated Movies",
            "Explore the highest-rated movies "
            "in our collection.",
        ),

        "genres": (
            "Explore by Genre",
            "Find movies based on your "
            "favorite genres.",
        ),
    }

    if mode not in page_config:

        st.error(
            "Unknown discovery page."
        )

        return

    title, description = (
        page_config[mode]
    )

    st.markdown(
        '<div class="dc-header">'
        f'{html.escape(title)}'
        '</div>'
        '<div class="dc-subtitle">'
        f'{html.escape(description)}'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------
    # GENRE FILTER
    # --------------------------------------

    all_genres = sorted({

        genre.strip()

        for value in movies["genres"]

        for genre in value.split("|")

        if genre.strip()

    })

    col1, col2 = st.columns(
        [2, 1],
        gap="medium"
    )

    with col1:

        selected_genre = st.selectbox(
            "Select Genre",
            ["All Genres"] + all_genres,
            key=f"{mode}_genre",
        )

    with col2:

        result_count = st.selectbox(
            "Number of Movies",
            [8, 12, 20, 40],
            index=2,
            key=f"{mode}_count",
        )

    # --------------------------------------
    # APPLY GENRE FILTER
    # --------------------------------------

    if selected_genre != "All Genres":

        movies = movies[
            movies["genres"].apply(
                lambda value:
                selected_genre in [
                    item.strip()
                    for item in value.split("|")
                ]
            )
        ].copy()

    # --------------------------------------
    # SORT MOVIES
    # --------------------------------------

    if mode == "trending":

        movies = movies.sort_values(
            by=[
                "popularity",
                "vote_count",
            ],
            ascending=False,
        )

        st.caption(
            "Ranked by stored TMDb popularity. "
            "Not real-time daily trends."
        )

    elif mode == "top_rated":

        movies = movies[
            movies["vote_count"] >= 100
        ].copy()

        movies = movies.sort_values(
            by=[
                "vote_average",
                "vote_count",
            ],
            ascending=False,
        )

    elif mode == "genres":

        movies = movies.sort_values(
            by=[
                "popularity",
                "vote_average",
            ],
            ascending=False,
        )

    # --------------------------------------
    # LIMIT RESULTS
    # --------------------------------------

    movies = movies.head(
        result_count
    )

    st.caption(
        f"Showing {len(movies)} movies"
    )

    if movies.empty:

        st.warning(
            "No movies match the selected filters."
        )

        return

    # --------------------------------------
    # DISPLAY MOVIE GRID
    # --------------------------------------

    records = movies.to_dict(
        "records"
    )

    movies_per_row = 4

    for start in range(
        0,
        len(records),
        movies_per_row,
    ):

        row_movies = records[
            start:start + movies_per_row
        ]

        columns = st.columns(
            movies_per_row,
            gap="medium",
        )

        for index, movie in enumerate(
            row_movies
        ):

            with columns[index]:

                movie_card(
                    movie,
                    mode,
                )

        # Space between rows.
        st.write("")


# ==========================================
# END OF FILE
# ==========================================
