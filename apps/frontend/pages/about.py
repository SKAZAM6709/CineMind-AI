
"""
CineMind AI - About Page

Project overview, architecture, technology stack,
features, datasets, evaluation and limitations.
"""

from pathlib import Path
import sys
import html
import json

import numpy as np
import pandas as pd
import streamlit as st


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMDB_PATH = (
    ROOT / "data" / "external" / "tmdb"
    / "tmdb_movies.csv"
)

CATALOG_PATH = (
    ROOT / "data" / "processed"
    / "movie_catalog.csv"
)

SEMANTIC_PATH = (
    ROOT / "models"
    / "semantic_embeddings.npz"
)

MANIFEST_PATH = (
    ROOT / "models" / "staging_500"
    / "build_manifest.json"
)


# ==========================================
# PAGE STYLING
# ==========================================

st.markdown("""
<style>

.about-title {
    font-size: clamp(36px, 5vw, 54px);
    font-weight: 850;
    letter-spacing: -1.5px;
    color: #ffffff;
    margin: 0 0 8px;
}

.about-title span {
    color: #c084fc;
}

.about-subtitle {
    font-size: 16px;
    color: #b6adc9;
    margin-bottom: 24px;
}

.about-hero {
    background: linear-gradient(
        120deg, #251632, #15111f
    );
    border: 1px solid #503464;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 22px;
    line-height: 1.8;
    color: #e9e0f1;
}

.about-hero strong {
    color: #d8b4fe;
}

.about-section {
    font-size: 23px;
    font-weight: 750;
    color: #f6f0ff;
    margin: 28px 0 12px;
}

.about-feature {
    background: #1b1628;
    border: 1px solid #392b4a;
    border-radius: 14px;
    padding: 18px;
    min-height: 150px;
    margin-bottom: 12px;
}

.about-feature-title {
    font-weight: 750;
    color: #e9d5ff;
    margin-bottom: 8px;
    font-size: 16px;
}

.about-feature-text {
    color: #b5adc7;
    font-size: 13px;
    line-height: 1.65;
}

.about-step {
    text-align: center;
    background: #211a30;
    border: 1px solid #463457;
    padding: 15px;
    border-radius: 12px;
    color: #f4edff;
    font-weight: 650;
}

.about-note {
    color: #b5adc7;
    font-size: 13px;
    line-height: 1.7;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD PROJECT STATISTICS
# ==========================================

@st.cache_data(show_spinner=False)
def load_statistics(
    tmdb_mtime,
    catalog_mtime,
    semantic_mtime,
):

    stats = {
        "movie_count": None,
        "movielens_count": None,
        "embedding_dimensions": None,
        "semantic_movies": None,
    }

    if TMDB_PATH.exists():
        movies = pd.read_csv(
            TMDB_PATH,
            encoding="utf-8-sig",
        )
        stats["movie_count"] = (
            movies["movieId"].nunique()
        )

    if CATALOG_PATH.exists():
        catalog = pd.read_csv(
            CATALOG_PATH,
        )
        stats["movielens_count"] = (
            catalog["movieId"].nunique()
        )

    if SEMANTIC_PATH.exists():
        with np.load(
            SEMANTIC_PATH,
            allow_pickle=False,
        ) as artifact:

            shape = artifact[
                "embeddings"
            ].shape

            stats["semantic_movies"] = (
                shape[0]
            )

            stats["embedding_dimensions"] = (
                shape[1]
            )

    return stats


def file_mtime(path):
    if path.exists():
        return path.stat().st_mtime_ns
    return 0


try:
    stats = load_statistics(
        file_mtime(TMDB_PATH),
        file_mtime(CATALOG_PATH),
        file_mtime(SEMANTIC_PATH),
    )

except (
    OSError,
    ValueError,
    KeyError,
) as error:

    st.warning(
        "Some project statistics could not "
        "be loaded."
    )

    st.caption(str(error))

    stats = {
        "movie_count": None,
        "movielens_count": None,
        "semantic_movies": None,
        "embedding_dimensions": None,
    }


def format_count(value):
    if value is None:
        return "N/A"
    return f"{value:,}"


def feature_card(title, description):
    return (
        '<div class="about-feature">'
        f'<div class="about-feature-title">'
        f'{html.escape(title)}'
        '</div>'
        f'<div class="about-feature-text">'
        f'{html.escape(description)}'
        '</div>'
        '</div>'
    )


# ==========================================
# HERO SECTION
# ==========================================

st.markdown(
    '<div class="about-title">'
    'About <span>CineMind AI</span>'
    '</div>'
    '<div class="about-subtitle">'
    'Intelligent Movie Discovery and Recommendation'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="about-hero">'
    '<strong>CineMind AI</strong> is an applied '
    'Artificial Intelligence and Machine Learning '
    'project that helps users discover movies '
    'through personalized recommendations and '
    'natural-language interactions. '
    'It combines content-based filtering, '
    'NLP, semantic embeddings, popularity, '
    'and rating signals in a hybrid '
    'recommendation system. '
    'Its conversational RAG assistant retrieves '
    'relevant movie information and uses a '
    'local language model to generate explanations.'
    '</div>',
    unsafe_allow_html=True,
)


# ==========================================
# PROJECT METRICS
# ==========================================

st.markdown("### Project at a Glance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Enriched Movies",
        format_count(stats["movie_count"]),
    )

with col2:
    st.metric(
        "MovieLens Catalog",
        format_count(stats["movielens_count"]),
    )

with col3:
    st.metric(
        "Semantic Vectors",
        format_count(stats["semantic_movies"]),
    )

with col4:
    st.metric(
        "Embedding Dimensions",
        format_count(
            stats["embedding_dimensions"]
        ),
    )


# ==========================================
# CORE FEATURES
# ==========================================

st.markdown(
    '<div class="about-section">'
    'Core Features'
    '</div>',
    unsafe_allow_html=True,
)

features = [
    (
        "Movie Discovery",
        "Browse movies through Home, Explore, "
        "Trending, Top Rated and Genres.",
    ),
    (
        "Hybrid Recommendations",
        "Combine content similarity, NLP, semantic "
        "similarity, popularity and Bayesian ratings.",
    ),
    (
        "Personalized For You",
        "Choose multiple movies you enjoy and receive "
        "recommendations based on combined preferences.",
    ),
    (
        "AI Movie Assistant",
        "Describe the type of movie you want "
        "and ask follow-up questions using "
        "conversational RAG.",
    ),
    (
        "My Watchlist",
        "Save and manage favorite movies "
        "within the current application session.",
    ),
    (
        "Model Explainability",
        "Explore hybrid relevance scores, "
        "component similarities and "
        "recommendation rankings.",
    ),
    (
        "Model Evaluation",
        "Review Precision@5, runtime measurements "
        "and basic RAG citation validation.",
    ),
    (
        "Cinematic Interface",
        "Use a dark-themed Streamlit interface "
        "with movie posters and interactive cards.",
    ),
]

for start in range(0, len(features), 2):

    columns = st.columns(2)

    for index, feature in enumerate(
        features[start:start + 2]
    ):

        with columns[index]:
            st.markdown(
                feature_card(*feature),
                unsafe_allow_html=True,
            )


# ==========================================
# SYSTEM ARCHITECTURE
# ==========================================

st.markdown(
    '<div class="about-section">'
    'System Architecture'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="about-step">'
    'MovieLens + TMDb Data'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='text-align:center;color:#c084fc;"
    "font-size:24px;margin:10px'>↓</div>",
    unsafe_allow_html=True,
)

left, right = st.columns(2)

with left:
    st.markdown(
        '<div class="about-step">'
        'Hybrid Recommendation Engine'
        '</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        '<div class="about-step">'
        'Semantic Retrieval + Ollama RAG'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    "<div style='text-align:center;color:#c084fc;"
    "font-size:24px;margin:10px'>↓</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="about-step">'
    'Streamlit User Interface'
    '</div>',
    unsafe_allow_html=True,
)


# ==========================================
# TECHNOLOGY STACK
# ==========================================

st.markdown(
    '<div class="about-section">'
    'Technology Stack'
    '</div>',
    unsafe_allow_html=True,
)

tech_stack = pd.DataFrame([
    ["Programming", "Python"],
    ["Frontend", "Streamlit"],
    ["Data Processing", "Pandas, NumPy"],
    ["Machine Learning", "Scikit-learn"],
    ["NLP", "TF-IDF"],
    [
        "Semantic Embeddings",
        "Sentence Transformers (all-MiniLM-L6-v2)",
    ],
    [
        "Local LLM",
        "Ollama (Gemma 3 1B)",
    ],
    [
        "Recommendation",
        "Weighted Hybrid Filtering",
    ],
    [
        "Data Sources",
        "MovieLens, TMDb",
    ],
])

tech_stack.columns = [
    "Category",
    "Technology",
]

st.dataframe(
    tech_stack,
    hide_index=True,
    width="stretch",
)


# ==========================================
# HYBRID RECOMMENDATION SYSTEM
# ==========================================

st.markdown(
    '<div class="about-section">'
    'Hybrid Recommendation Methodology'
    '</div>',
    unsafe_allow_html=True,
)

weights = pd.DataFrame({
    "Component": [
        "Content-Based TF-IDF",
        "NLP TF-IDF",
        "Semantic Embeddings",
        "Popularity",
        "Bayesian Rating",
    ],
    "Weight (%)": [
        30, 20, 35, 10, 5,
    ],
})

st.bar_chart(
    weights.set_index("Component"),
    horizontal=True,
    x_label="Weight (%)",
)

st.caption(
    "These values are configured model weights, "
    "not independently measured feature importance."
)


# ==========================================
# RAG OVERVIEW
# ==========================================

st.markdown(
    '<div class="about-section">'
    'Retrieval-Augmented Generation'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
CineMind AI's conversational assistant follows
a retrieval-augmented generation workflow:

1. **Query Encoding:** The user's movie request is
   converted into a semantic embedding.
2. **Semantic Retrieval:** The request is compared
   with the precomputed movie embeddings.
3. **Evidence Preparation:** Retrieved movie titles,
   genres and descriptions are prepared as context.
4. **Answer Generation:** A locally running Gemma 3
   model generates a response using that context.
5. **Conversation Context:** Recent interactions
   help interpret follow-up questions.

The system uses exact cosine similarity search over
NumPy embeddings rather than a dedicated vector
database.
"""
)


# ==========================================
# DATASETS AND EVALUATION
# ==========================================

st.markdown(
    '<div class="about-section">'
    'Datasets and Evaluation'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
**MovieLens:** Supplies movie identifiers,
genres, ratings and user-provided tags.

**TMDb:** Supplies enriched titles, descriptions,
posters, popularity information and other metadata.

**Evaluation:** Precision@5 is calculated using
reviewed relevance labels. RAG tests record
retrieval and generation latency, execution
success and basic citation validity.

Evaluation results can be viewed on the
**Model Performance** page.
"""
)


# ==========================================
# PROJECT LIMITATIONS
# ==========================================

st.markdown(
    '<div class="about-section">'
    'Current Limitations'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
- The AI assistant searches only movies represented
  in the active semantic index.
- Locally generated LLM responses can be slow and
  may contain unsupported statements.
- Citation-number validation does not guarantee
  factual correctness.
- The Watchlist is stored in Streamlit session state,
  not in a persistent user database.
- Stored TMDb popularity is not a live trending feed.
- Evaluation results are based on a small query set;
  broader testing is required for research claims.
"""
)


# ==========================================
# CREDITS
# ==========================================

st.markdown(
    '<div class="about-section">'
    'Data Attribution and Project Credits'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Movie metadata and images are provided by TMDb. "
    "This product uses the TMDB API but is not "
    "endorsed or certified by TMDB."
)

st.markdown(
    "MovieLens dataset: "
    "[GroupLens Research](https://grouplens.org/datasets/movielens/)"
)

st.markdown(
    "Movie metadata: "
    "[The Movie Database](https://www.themoviedb.org/)"
)

st.divider()

st.caption(
    "CineMind AI • Applied Machine Learning, "
    "Recommendation Systems and Conversational RAG"
)
