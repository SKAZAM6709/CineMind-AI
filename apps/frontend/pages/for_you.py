
"""
CineMind AI - Multi-Movie Recommendation Page

Features:
- Select multiple favorite movies
- Real hybrid recommendation engine
- Watchlist integration
- Interactive movie cards
- Recommendation explanations
- Movie details popup
"""

from pathlib import Path
import sys
import html

import numpy as np
import pandas as pd
import streamlit as st


# ==========================================
# PROJECT IMPORTS
# ==========================================

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.movie_recommender.recommendation.hybrid import (
    HybridRecommender,
)

from apps.frontend.components.movie_card import (
    show_movie_details,
)

from apps.frontend.components.watchlist_manager import (
    initialize_watchlist,
    get_watchlist,
    is_saved,
    add_to_watchlist,
    remove_from_watchlist,
)


initialize_watchlist()


# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
.fy-heading {
    font-size: clamp(34px, 4vw, 50px);
    font-weight: 850;
    letter-spacing: -1.5px;
    color: #ffffff;
    margin-bottom: 8px;
}

.fy-heading span {
    color: #c084fc;
}

.fy-subtitle {
    color: #aaa5bd;
    font-size: 15px;
    margin-bottom: 26px;
}

.fy-section-title {
    font-size: 25px;
    font-weight: 800;
    color: #ffffff;
    margin: 22px 0 10px;
}

.fy-selected-box {
    border-radius: 16px;
    border: 1px solid #49335f;
    background: linear-gradient(
        120deg, #231638, #12101f
    );
    padding: 19px;
    margin-bottom: 15px;
}

.fy-selected-heading {
    color: #c084fc;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 7px;
}

.fy-selected-text {
    color: white;
    font-size: 16px;
    font-weight: 650;
    line-height: 1.8;
}

.fy-card {
    background: #171421;
    border: 1px solid #342743;
    border-radius: 16px;
    overflow: hidden;
    transition:
        transform .3s ease,
        box-shadow .3s ease,
        border-color .3s ease;
}

.fy-card:hover {
    transform: translateY(-7px);
    border-color: #b56cff;
    box-shadow: 0 15px 35px rgba(168,85,247,.25);
}

.fy-poster {
    position: relative;
    aspect-ratio: 2 / 3;
    overflow: hidden;
    background: #272133;
}

.fy-poster img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform .4s ease;
}

.fy-card:hover .fy-poster img {
    transform: scale(1.07);
}

.fy-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        to top,
        rgba(8,8,17,.97),
        rgba(8,8,17,.05) 75%
    );
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 15px;
    opacity: 0;
    transition: opacity .3s ease;
}

.fy-card:hover .fy-overlay {
    opacity: 1;
}

.fy-overlay-title {
    font-size: 15px;
    font-weight: 750;
    color: white;
    line-height: 1.4;
}

.fy-overlay-genres {
    font-size: 12px;
    color: #d1c2e8;
    margin-top: 6px;
}

.fy-card-body {
    padding: 12px;
}

.fy-card-title {
    color: #ffffff;
    font-size: 14px;
    font-weight: 800;
    line-height: 1.45;
    height: 42px;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.fy-card-rating {
    color: #fbbf24;
    font-size: 13px;
    padding-top: 8px;
}

.fy-card-score {
    color: #c084fc;
    font-size: 13px;
    font-weight: 750;
    padding-top: 6px;
}

.fy-placeholder {
    height: 100%;
    display: grid;
    place-items: center;
    color: #aaa5bd;
    font-size: 13px;
}

@media (prefers-reduced-motion: reduce) {
    .fy-card,
    .fy-poster img,
    .fy-overlay {
        transition: none;
    }

    .fy-card:hover,
    .fy-card:hover img {
        transform: none;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD HYBRID ENGINE
# ==========================================

@st.cache_resource(
    show_spinner="Loading CineMind AI recommendation models..."
)
def load_engine():
    return HybridRecommender()


try:
    engine = load_engine()

except (FileNotFoundError, ValueError, KeyError) as error:
    st.error("Unable to load hybrid recommendation models.")
    st.code(str(error))
    st.stop()


movies = engine.movies.copy()


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def safe(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (ValueError, TypeError):
        pass

    return html.escape(str(value), quote=True)


def number(value, default=0.0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def get_genres(value):
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    return [
        part.strip()
        for part in str(value).split("|")
        if part.strip()
    ]


def explain_recommendation(movie, preferences):

    liked_genres = set()

    for preference in preferences:
        liked_genres.update(
            get_genres(preference.get("genres"))
        )

    recommendation_genres = set(
        get_genres(movie.get("genres"))
    )

    common = sorted(
        liked_genres & recommendation_genres
    )

    content = number(
        movie.get("content_similarity")
    )

    nlp = number(
        movie.get("nlp_similarity")
    )

    semantic = number(
        movie.get("semantic_similarity")
    )

    contributions = {
        "content": 0.30 * content,
        "NLP": 0.20 * nlp,
        "semantic": 0.35 * semantic,
    }

    strongest = max(
        contributions,
        key=contributions.get
    )

    explanation = []

    if common:
        explanation.append(
            "Shared genres with your selected movies: "
            + ", ".join(common[:4]) + "."
        )

    explanation.append(
        f"The strongest text-based contribution "
        f"comes from {strongest} similarity."
    )

    return " ".join(explanation)


# ==========================================
# PAGE HEADER
# ==========================================

st.markdown(
    '<div class="fy-heading">'
    'Made <span>For You</span>'
    '</div>'
    '<div class="fy-subtitle">'
    'Select movies you enjoy and let CineMind AI '
    'discover similar films using five ranking signals.'
    '</div>',
    unsafe_allow_html=True,
)


# ==========================================
# MOVIE PREFERENCES
# ==========================================

st.markdown("### Build Your Movie Preference Profile")

movie_lookup = {
    int(row["movieId"]): str(row["title"])
    for _, row in movies.iterrows()
}

movie_ids = list(movie_lookup.keys())

default_ids = [
    mid for mid in [1, 2, 34]
    if mid in movie_lookup
]

if not default_ids:
    default_ids = movie_ids[:1]

selected_ids = st.multiselect(
    "Select movies you like",
    options=movie_ids,
    default=default_ids,
    format_func=lambda mid: movie_lookup[mid],
    max_selections=10,
    key="fy_multi_preferences",
    help="Choose up to 10 movies. The engine combines "
         "their similarity signals.",
)

col1, col2 = st.columns([2, 1])

with col1:
    st.caption(
        f"{len(selected_ids)} movies selected."
    )

with col2:
    number_of_results = st.selectbox(
        "Recommendations",
        options=[4, 8, 12, 16, 20],
        index=1,
        key="fy_multi_count",
    )

if not selected_ids:
    st.info(
        "Choose at least one movie to generate recommendations."
    )
    st.stop()


# ==========================================
# DISPLAY SELECTED PREFERENCES
# ==========================================

selected_movies = movies[
    movies["movieId"].isin(selected_ids)
].copy()

selected_records = selected_movies.to_dict("records")

selected_names = [
    safe(movie_lookup[mid])
    for mid in selected_ids
]

st.markdown(
    '<div class="fy-selected-box">'
    '<div class="fy-selected-heading">'
    'YOUR MOVIE PREFERENCES'
    '</div>'
    '<div class="fy-selected-text">'
    + " • ".join(selected_names)
    + '</div></div>',
    unsafe_allow_html=True,
)

with st.expander("View my selected movies"):
    preview_columns = st.columns(
        min(4, len(selected_ids))
    )

    for index, movie_id in enumerate(selected_ids):
        movie = selected_movies[
            selected_movies["movieId"] == movie_id
        ].iloc[0]

        with preview_columns[
            index % len(preview_columns)
        ]:
            poster = str(
                movie.get("poster_url", "")
            )

            if poster.startswith("https://"):
                st.image(
                    poster,
                    width="stretch",
                )
            else:
                st.info("Poster unavailable")

            st.caption(str(movie["title"]))


# ==========================================
# GENERATE HYBRID RECOMMENDATIONS
# ==========================================

try:
    # Request enough results to filter out saved movies.
    recommendations = engine.recommend_for_preferences(
        movie_ids=selected_ids,
        top_n=len(movies),
    )

except ValueError as error:
    st.error(str(error))
    st.stop()


saved_ids = set(get_watchlist())

# Do not recommend movies already saved.
recommendations = recommendations[
    ~recommendations["movieId"].isin(saved_ids)
].copy()

recommendations = recommendations.head(
    number_of_results
).reset_index(drop=True)


# ==========================================
# RECOMMENDATION RESULTS
# ==========================================

st.divider()

st.markdown(
    '<div class="fy-section-title">'
    'Your Personalized Recommendations'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    f"Recommended using {len(selected_ids)} selected "
    f"movie(s) from {len(movies)} available movies."
)

if saved_ids:
    st.caption(
        "Movies already in your watchlist "
        "are excluded from these results."
    )

if recommendations.empty:
    st.warning(
        "No recommendations remain. Try fewer "
        "saved movies or different preferences."
    )
    st.stop()


# ==========================================
# CINEMATIC MOVIE CARDS
# ==========================================

records = recommendations.to_dict("records")

movies_per_row = 4

for row_start in range(
    0, len(records), movies_per_row
):

    row_movies = records[
        row_start:row_start + movies_per_row
    ]

    columns = st.columns(
        movies_per_row,
        gap="medium",
    )

    for column_index, movie in enumerate(row_movies):

        with columns[column_index]:

            movie_id = int(movie["movieId"])

            title = safe(movie.get("title"))
            genres = safe(
                movie.get("genres", "")
            ).replace("|", " • ")

            poster = str(
                movie.get("poster_url", "")
            )

            rating = number(
                movie.get("vote_average")
            )

            score = number(
                movie.get("hybrid_score")
            )

            if poster.startswith("https://"):
                poster_html = (
                    f'<img src="{safe(poster)}" '
                    f'alt="{title}" loading="lazy">'
                )
            else:
                poster_html = (
                    '<div class="fy-placeholder">'
                    'Poster unavailable'
                    '</div>'
                )

            card_html = (
                '<div class="fy-card">'
                '<div class="fy-poster">'
                f'{poster_html}'
                '<div class="fy-overlay">'
                f'<div class="fy-overlay-title">{title}</div>'
                f'<div class="fy-overlay-genres">'
                f'{genres}</div>'
                '</div>'
                '</div>'
                '<div class="fy-card-body">'
                f'<div class="fy-card-title" title="{title}">'
                f'{title}</div>'
                f'<div class="fy-card-rating">'
                f'★ {rating:.1f}/10'
                '</div>'
                f'<div class="fy-card-score">'
                f'AI Match Score: {score:.2f}'
                '</div>'
                '</div>'
                '</div>'
            )

            st.markdown(
                card_html,
                unsafe_allow_html=True,
            )

            # BUTTON 1: MOVIE DETAILS
            if st.button(
                "View Details",
                key=f"fy_details_{movie_id}",
                width="stretch",
            ):

                full_movie = movies[
                    movies["movieId"] == movie_id
                ].iloc[0]

                show_movie_details(full_movie)

            # BUTTON 2: WATCHLIST
            if is_saved(movie_id):

                if st.button(
                    "✓ Saved — Remove",
                    key=f"fy_remove_{movie_id}",
                    width="stretch",
                ):
                    remove_from_watchlist(movie_id)
                    st.rerun()

            else:

                if st.button(
                    "♡ Add to Watchlist",
                    key=f"fy_save_{movie_id}",
                    width="stretch",
                ):
                    add_to_watchlist(movie_id)
                    st.rerun()

            # BUTTON 3: RECOMMENDATION EXPLANATION
            with st.expander("Why this movie?"):

                st.write(
                    explain_recommendation(
                        movie,
                        selected_records,
                    )
                )

                for label, field in [
                    (
                        "Content Similarity",
                        "content_similarity",
                    ),
                    (
                        "NLP Similarity",
                        "nlp_similarity",
                    ),
                    (
                        "Semantic Similarity",
                        "semantic_similarity",
                    ),
                ]:

                    value = number(
                        movie.get(field)
                    )

                    st.caption(
                        f"{label}: {value:.2f}%"
                    )

                    st.progress(
                        min(
                            max(value / 100.0, 0.0),
                            1.0,
                        )
                    )

                st.caption(
                    f"Final hybrid score: {score:.2f}/100"
                )

                st.caption(
                    "This score also includes "
                    "popularity and Bayesian rating."
                )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "CineMind AI • Multi-Movie Hybrid Recommendation"
)

st.caption(
    "AI Match Scores are modeled relevance scores, "
    "not predicted probabilities of enjoyment."
)

st.caption(
    "Movie metadata provided by TMDb. "
    "This product uses the TMDB API but is not "
    "endorsed or certified by TMDB."
)
