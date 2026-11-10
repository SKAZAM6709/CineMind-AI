
"""
CineMind AI - Conversational RAG Frontend

Features:
- Conversation history
- Context-aware movie retrieval
- Ollama-generated answers
- Numbered movie evidence
- Animated movie cards
- Movie details and watchlist
- Clear conversation option
"""

from pathlib import Path
import sys
import html
import re

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.movie_recommender.retrieval.semantic_search import (
    SemanticMovieSearch,
)

from src.movie_recommender.llm.rag_assistant import (
    RAGMovieAssistant,
)

from apps.frontend.components.movie_card import (
    show_movie_details,
)

from apps.frontend.components.watchlist_manager import (
    initialize_watchlist,
    is_saved,
    add_to_watchlist,
    remove_from_watchlist,
)


# ==========================================
# INITIALIZE SESSION
# ==========================================

initialize_watchlist()

if "cinemind_chat_history" not in st.session_state:
    st.session_state["cinemind_chat_history"] = []


# ==========================================
# CUSTOM PAGE STYLING
# ==========================================

st.markdown("""
<style>

.ai-heading {
    font-size: clamp(32px, 4vw, 48px);
    font-weight: 850;
    color: #ffffff;
    letter-spacing: -1.5px;
    margin-bottom: 8px;
}

.ai-heading span {
    color: #c084fc;
}

.ai-description {
    color: #aaa5bd;
    font-size: 15px;
    margin-bottom: 25px;
}

.ai-card {
    background: #171421;
    border: 1px solid #342743;
    border-radius: 16px;
    overflow: hidden;
    transition:
        transform .3s ease,
        box-shadow .3s ease,
        border-color .3s ease;
}

.ai-card:hover {
    transform: translateY(-7px);
    border-color: #c084fc;
    box-shadow: 0 14px 30px rgba(168,85,247,.25);
}

.ai-poster {
    position: relative;
    aspect-ratio: 2 / 3;
    overflow: hidden;
    background: #282134;
}

.ai-poster img {
    width: 100%;
    height: 100%;
    display: block;
    object-fit: cover;
    transition: transform .4s ease;
}

.ai-card:hover .ai-poster img {
    transform: scale(1.07);
}

.ai-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        to top,
        rgba(8,8,17,.96),
        transparent 75%
    );
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 12px;
    opacity: 0;
    transition: opacity .3s ease;
}

.ai-card:hover .ai-overlay {
    opacity: 1;
}

.ai-overlay-title {
    color: #ffffff;
    font-size: 14px;
    font-weight: 750;
    line-height: 1.4;
}

.ai-card-body {
    padding: 12px;
}

.ai-title {
    color: #ffffff;
    font-size: 14px;
    font-weight: 800;
    line-height: 1.45;
    height: 42px;
    overflow: hidden;
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.ai-rating {
    color: #fbbf24;
    font-size: 13px;
    margin-top: 8px;
}

.ai-score {
    color: #c084fc;
    font-size: 12px;
    font-weight: 750;
    margin-top: 7px;
}

.ai-empty {
    display: grid;
    place-items: center;
    height: 100%;
    color: #aaa5bd;
    font-size: 12px;
}

@media (prefers-reduced-motion: reduce) {
    .ai-card,
    .ai-poster img,
    .ai-overlay {
        transition: none;
    }

    .ai-card:hover,
    .ai-card:hover img {
        transform: none;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD AI ENGINES
# ==========================================

@st.cache_resource(
    show_spinner="Loading CineMind AI..."
)
def load_engines():

    search_engine = SemanticMovieSearch()

    rag_engine = RAGMovieAssistant(
        model="gemma3:1b"
    )

    return search_engine, rag_engine


try:
    search_engine, rag_engine = load_engines()

except (
    FileNotFoundError,
    ValueError,
    KeyError,
    OSError,
) as error:

    st.error(
        "CineMind AI could not load the RAG system."
    )
    st.code(str(error))
    st.stop()


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def safe(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return html.escape(
        str(value),
        quote=True
    )


def number(value, default=0.0):

    try:
        value = float(value)

        if pd.notna(value):
            return value

    except (TypeError, ValueError):
        pass

    return default


def is_followup(question):
    """Identify common references to earlier results."""

    pattern = (
        r"\b(those|these|them|earlier|previous|"
        r"above|which one|of the ones|among them|"
        r"first one|second one|third one|"
        r"one of them|that movie)\b"
    )

    return bool(
        re.search(
            pattern,
            question.lower()
        )
    )


def retrieve_movies(question, history, top_k=5):

    # If the question refers explicitly to earlier
    # results, reuse those records.
    if history and is_followup(question):

        previous_results = history[-1].get(
            "results", []
        )

        if previous_results:

            return pd.DataFrame(
                previous_results
            ).head(top_k).reset_index(drop=True)

    # Otherwise search normally using the user's
    # new natural-language request.
    return search_engine.search(
        question,
        top_k=top_k,
    )


def movie_card(movie, turn_index):

    movie_id = int(movie["movieId"])

    title = safe(
        movie.get("title", "")
    )

    poster = str(
        movie.get("poster_url", "")
    )

    genres = safe(
        movie.get("genres", "")
    ).replace("|", " • ")

    rating = number(
        movie.get("vote_average")
    )

    score = number(
        movie.get("semantic_similarity")
    )

    if poster.startswith("https://"):

        poster_html = (
            f'<img src="{safe(poster)}" '
            f'alt="{title}" loading="lazy">'
        )

    else:

        poster_html = (
            '<div class="ai-empty">'
            'Poster unavailable'
            '</div>'
        )

    card_html = (
        '<div class="ai-card">'
        '<div class="ai-poster">'
        f'{poster_html}'
        '<div class="ai-overlay">'
        f'<div class="ai-overlay-title">{title}</div>'
        f'<div style="color:#d1c2e8;font-size:12px;">'
        f'{genres}</div>'
        '</div>'
        '</div>'
        '<div class="ai-card-body">'
        f'<div class="ai-title" title="{title}">'
        f'{title}</div>'
        f'<div class="ai-rating">'
        f'★ {rating:.1f}/10'
        '</div>'
        f'<div class="ai-score">'
        f'Semantic Match: {score:.2f}%'
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
        key=f"ai_details_{turn_index}_{movie_id}",
        width="stretch",
    ):
        show_movie_details(movie)

    if is_saved(movie_id):

        if st.button(
            "✓ Saved — Remove",
            key=f"ai_remove_{turn_index}_{movie_id}",
            width="stretch",
        ):
            remove_from_watchlist(movie_id)
            st.rerun()

    else:

        if st.button(
            "♡ Add to Watchlist",
            key=f"ai_save_{turn_index}_{movie_id}",
            width="stretch",
        ):
            add_to_watchlist(movie_id)
            st.rerun()


def display_sources(results):

    for index, (_, movie) in enumerate(
        results.iterrows(),
        start=1,
    ):

        st.markdown(
            f"**[{index}] {safe(movie.get('title'))}**",
            unsafe_allow_html=True,
        )

        st.caption(
            str(movie.get("genres", ""))
        )

        overview = movie.get(
            "overview", ""
        )

        if pd.notna(overview):
            text = str(overview).strip()
            if text:
                st.write(text[:550])


# ==========================================
# PAGE HEADER
# ==========================================

st.markdown(
    '<div class="ai-heading">'
    'AI Movie <span>Assistant</span>'
    '</div>'
    '<div class="ai-description">'
    'Discover movies through conversational '
    'search powered by CineMind AI RAG.'
    '</div>',
    unsafe_allow_html=True,
)


# ==========================================
# HEADER CONTROLS
# ==========================================

col1, col2 = st.columns(
    [3, 1],
    vertical_alignment="bottom",
)

with col1:

    st.caption(
        f"Searchable movies: "
        f"{len(search_engine.movies)}"
    )

with col2:

    if st.button(
        "Clear Chat",
        width="stretch",
    ):

        st.session_state[
            "cinemind_chat_history"
        ] = []

        st.rerun()


# ==========================================
# EXAMPLE QUESTIONS
# ==========================================

if not st.session_state[
    "cinemind_chat_history"
]:

    st.info(
        "Ask for a movie recommendation. "
        "Then ask follow-up questions about "
        "the recommended movies."
    )

    st.markdown(
        "Example: **Recommend a magical family "
        "adventure with fantasy and mystery.**"
    )


# ==========================================
# PROCESS NEW USER MESSAGE
# ==========================================

question = st.chat_input(
    "Ask CineMind AI about movies...",
    key="cinemind_conversation_input",
    max_chars=500,
)

if question and question.strip():

    question = question.strip()

    history = st.session_state[
        "cinemind_chat_history"
    ]

    with st.spinner(
        "Retrieving evidence and generating an answer..."
    ):

        try:
            results = retrieve_movies(
                question,
                history,
                top_k=5,
            )

            answer, llm_used, error = (
                rag_engine.generate(
                    question,
                    results,
                    history=history,
                )
            )

        except (
            ValueError,
            RuntimeError,
        ) as error:

            st.error(
                f"Unable to process the request: {error}"
            )
            st.stop()

    # Store the evidence with its answer so
    # older conversations remain consistent.
    history.append({
        "query": question,
        "answer": answer,
        "llm_used": llm_used,
        "error": error,
        "results": results.to_dict("records"),
    })

    # Keep a bounded in-session conversation.
    if len(history) > 12:
        history.pop(0)

    st.rerun()


# ==========================================
# DISPLAY CONVERSATION HISTORY
# ==========================================

history = st.session_state[
    "cinemind_chat_history"
]

for turn_index, turn in enumerate(history):

    with st.chat_message("user"):
        st.write(turn["query"])

    with st.chat_message("assistant"):

        st.markdown(
            turn["answer"]
        )

        if not turn.get("llm_used", False):

            st.caption(
                "Showing a retrieval-based fallback "
                "because LLM generation or citation "
                "validation did not complete."
            )

            with st.expander("Technical details"):
                st.code(
                    str(turn.get("error", ""))
                )

        turn_results = pd.DataFrame(
            turn.get("results", [])
        )

        if not turn_results.empty:

            with st.expander(
                "View retrieved movie evidence"
            ):
                display_sources(
                    turn_results
                )


# ==========================================
# SHOW LATEST MOVIE RESULTS
# ==========================================

if history:

    latest_index = len(history) - 1

    latest_results = pd.DataFrame(
        history[-1].get("results", [])
    )

    if not latest_results.empty:

        st.markdown(
            "### Movies from the Latest Search"
        )

        st.caption(
            "These movie cards correspond to "
            "the latest assistant response."
        )

        records = latest_results.to_dict("records")

        for start in range(
            0,
            len(records),
            4,
        ):

            row_movies = records[
                start:start + 4
            ]

            columns = st.columns(
                4,
                gap="medium",
            )

            for index, movie in enumerate(row_movies):

                with columns[index]:
                    movie_card(
                        movie,
                        latest_index,
                    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "CineMind AI • Conversational RAG Assistant"
)

st.caption(
    "Conversation history is stored in the "
    "current Streamlit session. "
    "Movie facts are based on retrieved "
    "TMDb catalog records."
)

st.caption(
    "This is a prototype. Generated answers "
    "may still contain factual errors. "
    "Movie metadata provided by TMDb."
)
