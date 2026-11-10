from pathlib import Path
import sys
import html

import pandas as pd
import streamlit as st

# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.frontend.components.movie_card import show_movie_details

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "tmdb"
    / "tmdb_movies.csv"
)

# ==========================================
# CUSTOM PAGE STYLING
# ==========================================

st.markdown("""
<style>
.explore-heading {
    font-size: clamp(32px, 4vw, 48px);
    font-weight: 850;
    letter-spacing: -1px;
    margin-bottom: 8px;
    color: #ffffff;
}

.explore-heading span {
    color: #c084fc;
}

.explore-description {
    color: #aaa5bd;
    font-size: 15px;
    margin-bottom: 25px;
}

.result-title {
    font-size: 22px;
    font-weight: 800;
    margin: 25px 0 8px;
    color: #ffffff;
}

.movie-card-title {
    color: #ffffff;
    font-size: 16px;
    font-weight: 800;
    line-height: 1.45;
    margin-top: 10px;
    margin-bottom: 6px;

    min-height: 72px;
    max-height: 72px;
    overflow: hidden;

    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.movie-card-meta {
    color: #cbb7ff;
    font-size: 14px;
    margin-bottom: 12px;
    min-height: 24px;
}

.poster-placeholder {
    width: 100%;
    aspect-ratio: 2 / 3;
    display: grid;
    place-items: center;
    background: #171523;
    border: 1px solid #302b40;
    border-radius: 14px;
    color: #aaa5bd;
    font-size: 14px;
    text-align: center;
    padding: 12px;
}

div.stButton > button {
    border-radius: 12px;
    border: 1px solid #7c3aed;
    background: transparent;
    color: #ffffff;
    transition:
        transform .2s ease,
        border-color .2s ease,
        box-shadow .2s ease;
}

div.stButton > button:hover {
    border-color: #c084fc;
    color: #ffffff;
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(168, 85, 247, 0.18);
}

@media (max-width: 768px) {
    .movie-card-title {
        min-height: 64px;
        max-height: 64px;
        font-size: 15px;
    }
}

/* Premium interactive movie cards */

.explore-hover-card {
    border: 1px solid #302b40;
    border-radius: 16px;
    background: #171523;
    overflow: hidden;
    transition:
        transform .3s ease,
        box-shadow .3s ease,
        border-color .3s ease;
}

.explore-hover-card:hover {
    transform: translateY(-7px);
    border-color: #c084fc;
    box-shadow: 0 15px 35px rgba(168,85,247,.28);
}

.explore-poster-wrap {
    position: relative;
    aspect-ratio: 2 / 3;
    overflow: hidden;
    background: #292438;
}

.explore-poster-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform .45s ease;
}

.explore-hover-card:hover img {
    transform: scale(1.08);
}

.explore-hover-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 16px;
    background: linear-gradient(
        to top,
        rgba(8,8,17,.98),
        rgba(8,8,17,.12) 70%
    );
    opacity: 0;
    transition: opacity .3s ease;
}

.explore-hover-card:hover .explore-hover-overlay {
    opacity: 1;
}

.explore-hover-overlay strong {
    color: #fff;
    font-size: 15px;
}

.explore-hover-overlay span {
    color: #d1c6e8;
    font-size: 12px;
    margin-top: 6px;
}

.explore-card-content {
    padding: 12px;
}

.explore-card-name {
    color: white;
    font-size: 14px;
    font-weight: 750;
    line-height: 1.45;
    height: 42px;
    overflow: hidden;
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
}

.explore-card-rating {
    color: #fbbf24;
    font-size: 12px;
    padding-top: 8px;
}

@media (prefers-reduced-motion: reduce) {
    .explore-hover-card,
    .explore-poster-wrap img,
    .explore-hover-overlay {
        transition: none;
        transform: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MOVIE DATA
# ==========================================

@st.cache_data
def load_movies():
    data = pd.read_csv(DATA_PATH)

    text_columns = [
        "title",
        "genres",
        "overview",
        "poster_url",
        "release_date",
    ]

    for column in text_columns:
        data[column] = data[column].fillna("").astype(str)

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

    data["year"] = pd.to_datetime(
        data["release_date"],
        errors="coerce",
    ).dt.year

    return data

if not DATA_PATH.exists():
    st.error("Movie metadata file was not found.")
    st.info("Run: python -m scripts.fetch_tmdb_full")
    st.stop()

movies = load_movies()

# ==========================================
# PAGE HEADER
# ==========================================

st.markdown(
    '<div class="explore-heading">Explore <span>Movies</span></div>'
    '<div class="explore-description">'
    'Discover films with advanced search, genres, ratings, and release-year filters.'
    '</div>',
    unsafe_allow_html=True,
)

# ==========================================
# SEARCH
# ==========================================

search = st.text_input(
    "Search movie titles",
    placeholder="Search for Toy Story, Jumanji, Heat...",
    key="explore_search",
)

# ==========================================
# FILTER OPTIONS
# ==========================================

all_genres = sorted({
    genre.strip()
    for value in movies["genres"]
    for genre in value.split("|")
    if genre.strip()
})

valid_years = movies["year"].dropna()

if valid_years.empty:
    st.error("No valid release years found.")
    st.stop()

min_year = int(valid_years.min())
max_year = int(valid_years.max())

st.markdown("### Advanced Filters")

col1, col2, col3 = st.columns(3)

with col1:
    selected_genre = st.selectbox(
        "Movie Genre",
        ["All Genres"] + all_genres,
        key="explore_genre",
    )

with col2:
    minimum_rating = st.slider(
        "Minimum TMDb Rating",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        key="explore_rating",
    )

with col3:
    year_range = st.slider(
        "Release Year",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        key="explore_year",
    )

# ==========================================
# SORTING
# ==========================================

sort_option = st.selectbox(
    "Sort Movies By",
    [
        "Most Popular",
        "Highest Rated",
        "Newest First",
        "Oldest First",
        "Title A-Z",
    ],
    key="explore_sort",
)

# ==========================================
# APPLY FILTERS
# ==========================================

filtered = movies.copy()

if search.strip():
    filtered = filtered[
        filtered["title"].str.contains(
            search.strip(),
            case=False,
            na=False,
            regex=False,
        )
    ]

if selected_genre != "All Genres":
    filtered = filtered[
        filtered["genres"].apply(
            lambda value: selected_genre in [
                genre.strip()
                for genre in value.split("|")
            ]
        )
    ]

filtered = filtered[
    filtered["vote_average"] >= minimum_rating
]

filtered = filtered[
    filtered["year"].between(
        year_range[0],
        year_range[1],
    )
]

# ==========================================
# APPLY SORTING
# ==========================================

if sort_option == "Most Popular":
    filtered = filtered.sort_values(
        "popularity",
        ascending=False,
    )

elif sort_option == "Highest Rated":
    filtered = filtered.sort_values(
        ["vote_average", "vote_count"],
        ascending=False,
    )

elif sort_option == "Newest First":
    filtered = filtered.sort_values(
        "year",
        ascending=False,
    )

elif sort_option == "Oldest First":
    filtered = filtered.sort_values(
        "year",
        ascending=True,
    )

elif sort_option == "Title A-Z":
    filtered = filtered.sort_values("title")

# ==========================================
# SEARCH RESULTS
# ==========================================

st.divider()

st.markdown(
    '<div class="result-title">Search Results</div>',
    unsafe_allow_html=True,
)

st.caption(
    f"Found {len(filtered)} movies from {len(movies)} available movies."
)

if filtered.empty:
    st.warning(
        "No movies match your filters. Try changing the search or rating."
    )
    st.stop()


# ==========================================
# ANIMATED MOVIE GRID
# ==========================================

# Group movies into rows for consistent alignment
movies_per_row = 4
movie_records = filtered.to_dict("records")

for row_start in range(0, len(movie_records), movies_per_row):

    row_movies = movie_records[
        row_start:row_start + movies_per_row
    ]

    columns = st.columns(movies_per_row, gap="medium")

    for column_index, movie in enumerate(row_movies):

        with columns[column_index]:

            title = html.escape(str(movie["title"]))
            poster = str(movie.get("poster_url", ""))

            rating = float(movie.get("vote_average", 0))
            year_value = movie.get("year")

            year = (
                str(int(year_value))
                if pd.notna(year_value)
                else "N/A"
            )

            genres = html.escape(
                str(movie.get("genres", "")).replace("|", " • ")
            )

            if poster.startswith("https://"):
                poster_html = (
                    f'<img src="{html.escape(poster, quote=True)}" '
                    f'alt="{title}" loading="lazy">'
                )
            else:
                poster_html = (
                    '<div style="height:100%;display:grid;'
                    'place-items:center;color:#aaa5bd;">'
                    'Poster unavailable</div>'
                )

            card_html = (
                '<div class="explore-hover-card">'
                '<div class="explore-poster-wrap">'
                f'{poster_html}'
                '<div class="explore-hover-overlay">'
                f'<strong>{title}</strong>'
                f'<span>{genres}</span>'
                '</div>'
                '</div>'
                '<div class="explore-card-content">'
                f'<div class="explore-card-name" title="{title}">'
                f'{title}</div>'
                '<div class="explore-card-rating">'
                f'★ {rating:.1f}/10 · {year}'
                '</div>'
                '</div>'
                '</div>'
            )

            st.markdown(
                card_html,
                unsafe_allow_html=True
            )

            if st.button(
                "View Details",
                key=f"explore_{int(movie['movieId'])}",
                width="stretch"
            ):
                show_movie_details(movie)

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Movie metadata provided by TMDb. "
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)