
"""
CineMind AI - Recommendation Insights

Explain hybrid recommendations using actual
outputs from the trained HybridRecommender.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# ==========================================
# PROJECT PATH
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


# ==========================================
# STYLE
# ==========================================

st.markdown("""
<style>

.insight-heading {
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 850;
    letter-spacing: -1px;
    color: white;
    margin-bottom: 8px;
}

.insight-heading span {
    color: #c084fc;
}

.insight-subtitle {
    color: #aaa5bd;
    font-size: 15px;
    margin-bottom: 26px;
}

.insight-note {
    padding: 16px;
    border-radius: 14px;
    background: #1d172b;
    border: 1px solid #49335f;
    color: #ded3ed;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource(
    show_spinner="Loading hybrid recommender..."
)
def load_engine():
    return HybridRecommender()


try:
    engine = load_engine()

except Exception as error:
    st.error(
        "The hybrid recommendation model "
        "could not be loaded."
    )
    st.exception(error)
    st.stop()


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="insight-heading">'
    'Recommendation <span>Insights</span>'
    '</div>'
    '<div class="insight-subtitle">'
    'Explore how CineMind AI combines multiple '
    'recommendation techniques.'
    '</div>',
    unsafe_allow_html=True,
)


# ==========================================
# MODEL INFORMATION
# ==========================================

weights = {
    "Content-based TF-IDF": 0.30,
    "NLP TF-IDF": 0.20,
    "Semantic Similarity": 0.35,
    "TMDb Popularity": 0.10,
    "Bayesian Rating": 0.05,
}

st.markdown("### Hybrid Recommendation Architecture")

weight_df = pd.DataFrame({
    "Component": list(weights.keys()),
    "Weight": [
        value * 100
        for value in weights.values()
    ],
})

st.bar_chart(
    weight_df.set_index("Component"),
    horizontal=True,
    x_label="Weight (%)",
)

st.caption(
    "The five model weights sum to 100%. "
    "Semantic similarity has the highest "
    "weight at 35%."
)





# ==========================================
# MOVIE SELECTION - SINGLE DROPDOWN
# ==========================================

st.markdown("### Explore Recommendations")

movies = engine.movies.copy()

movies["movieId"] = pd.to_numeric(
    movies["movieId"],
    errors="coerce",
)

movies = movies.dropna(
    subset=["movieId"]
).drop_duplicates(
    subset=["movieId"]
)

movies["movieId"] = movies["movieId"].astype(int)

movies["title"] = (
    movies["title"]
    .fillna("Untitled")
    .astype(str)
)

movies = movies.sort_values("title")

movie_lookup = dict(
    zip(
        movies["movieId"],
        movies["title"],
    )
)

movie_ids = list(movie_lookup.keys())

if not movie_ids:
    st.warning("No movies are available.")
    st.stop()

default_id = (
    1 if 1 in movie_lookup
    else movie_ids[0]
)

default_index = movie_ids.index(default_id)

# ONLY ONE MOVIE DROPDOWN
selected_id = st.selectbox(
    "Select a movie",
    options=movie_ids,
    index=default_index,
    format_func=lambda mid: movie_lookup[mid],
    key="insights_selected_movie",
)

st.caption(
    f"Movies available to the hybrid model: {len(movies)}"
)

# NUMBER OF RECOMMENDATIONS
result_count = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=15,
    value=5,
    step=1,
    key="insights_result_count",
)

# ==========================================
# GET RECOMMENDATIONS
# ==========================================

try:
    recommendations = engine.recommend(
        movie_id=selected_id,
        top_n=result_count,
    )

except Exception as error:
    st.error(
        "Unable to generate recommendations."
    )
    st.exception(error)
    st.stop()


if recommendations is None or recommendations.empty:
    st.info(
        "No recommendations were returned."
    )
    st.stop()


recommendations = recommendations.copy()

for column in [
    "content_similarity",
    "nlp_similarity",
    "semantic_similarity",
    "hybrid_score",
]:
    if column not in recommendations.columns:
        recommendations[column] = 0.0

    recommendations[column] = (
        pd.to_numeric(
            recommendations[column],
            errors="coerce",
        ).fillna(0.0)
    )


# ==========================================
# RESULT SUMMARY
# ==========================================

st.markdown("### Recommendation Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Selected Movie",
        movie_lookup[selected_id],
    )

with col2:
    st.metric(
        "Recommendations",
        len(recommendations),
    )

with col3:
    st.metric(
        "Highest Hybrid Score",
        f"{recommendations['hybrid_score'].max():.2f}%",
    )


display_columns = [
    "title",
    "hybrid_score",
    "content_similarity",
    "nlp_similarity",
    "semantic_similarity",
]

score_table = recommendations[
    display_columns
].rename(
    columns={
        "title": "Movie",
        "hybrid_score": "Hybrid Score (%)",
        "content_similarity": "Content",
        "nlp_similarity": "NLP",
        "semantic_similarity": "Semantic",
    }
)

st.dataframe(
    score_table,
    hide_index=True,
    width="stretch",
)


# ==========================================
# VISUALIZE HYBRID SCORES
# ==========================================

st.markdown("### Recommendation Ranking")

ranking_df = recommendations[
    ["title", "hybrid_score"]
].copy()

ranking_df = ranking_df.set_index(
    "title"
).rename(
    columns={
        "hybrid_score": "Hybrid Score (%)"
    }
)

st.bar_chart(
    ranking_df,
    horizontal=True,
    x_label="Hybrid Relevance Score (%)",
)


# ==========================================
# EXPLAIN INDIVIDUAL MOVIE
# ==========================================

st.markdown("### Why Was This Movie Recommended?")

result_records = recommendations.to_dict(
    "records"
)

result_ids = [
    int(movie["movieId"])
    for movie in result_records
]

result_lookup = {
    int(movie["movieId"]): movie
    for movie in result_records
}

explanation_id = st.selectbox(
    "Choose a recommended movie",
    options=result_ids,
    format_func=lambda mid: str(
        result_lookup[mid]["title"]
    ),
    key="insights_explanation_movie",
)

selected_movie = result_lookup[
    explanation_id
]


# ==========================================
# COMPONENT SCORES
# ==========================================

content = float(
    selected_movie["content_similarity"]
)

nlp = float(
    selected_movie["nlp_similarity"]
)

semantic = float(
    selected_movie["semantic_similarity"]
)

hybrid = float(
    selected_movie["hybrid_score"]
)


score_columns = st.columns(4)

with score_columns[0]:
    st.metric(
        "Content Similarity",
        f"{content:.2f}%",
    )

with score_columns[1]:
    st.metric(
        "NLP Similarity",
        f"{nlp:.2f}%",
    )

with score_columns[2]:
    st.metric(
        "Semantic Similarity",
        f"{semantic:.2f}%",
    )

with score_columns[3]:
    st.metric(
        "Hybrid Score",
        f"{hybrid:.2f}%",
    )


similarity_df = pd.DataFrame({
    "Component": [
        "Content TF-IDF",
        "NLP TF-IDF",
        "Semantic Embedding",
    ],
    "Similarity (%)": [
        content,
        nlp,
        semantic,
    ],
})

st.bar_chart(
    similarity_df.set_index("Component"),
    horizontal=True,
    x_label="Similarity (%)",
)


# ==========================================
# PLAIN-LANGUAGE EXPLANATION
# ==========================================

components = {
    "content-based similarity": content,
    "NLP description similarity": nlp,
    "semantic meaning similarity": semantic,
}

strongest_component = max(
    components,
    key=components.get,
)

st.markdown(
    f"""
**{selected_movie['title']}** received a hybrid
relevance score of **{hybrid:.2f}%**.

Among the three text-based similarity signals,
its strongest score is
**{strongest_component}**.

The hybrid recommender combines these signals
with TMDb popularity and Bayesian rating
components to determine the final ranking.
"""
)

st.info(
    "The hybrid score is a weighted ranking score, "
    "not the probability that a user will like "
    "this movie. The similarity values shown "
    "above are model outputs."
)


# ==========================================
# MOVIE DETAILS
# ==========================================

if st.button(
    "View Recommended Movie Details",
    width="stretch",
):
    show_movie_details(
        selected_movie
    )


# ==========================================
# FINAL EXPLANATION
# ==========================================

st.markdown("### How the Hybrid System Works")

st.markdown(
    """
**1. Content-based filtering:** Compares
movie titles, genres and tags.

**2. NLP filtering:** Measures similarity
between movie descriptions using TF-IDF.

**3. Semantic filtering:** Uses Sentence
Transformers to compare the meanings of
movie descriptions.

**4. Popularity:** Includes normalized
TMDb popularity information.

**5. Bayesian rating:** Incorporates movie
ratings while accounting for rating counts.

The weighted combination produces the
final recommendation ranking.
"""
)

st.divider()

st.caption(
    "CineMind AI • Explainable Hybrid Recommendations"
)
