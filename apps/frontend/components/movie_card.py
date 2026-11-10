
import html

import pandas as pd
import streamlit as st


def get_text(value, default="Not available"):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    result = str(value).strip()
    return result if result else default


@st.dialog("Movie Details", width="large")
def show_movie_details(movie):

    title = get_text(movie.get("title"), "Unknown Movie")
    overview = get_text(
        movie.get("overview"),
        "No description available."
    )
    poster = get_text(movie.get("poster_url"), "")
    genres = get_text(movie.get("genres"))

    rating = pd.to_numeric(
        movie.get("vote_average"),
        errors="coerce"
    )
    rating_text = (
        f"{rating:.1f}/10"
        if pd.notna(rating)
        else "N/A"
    )

    runtime = pd.to_numeric(
        movie.get("runtime"),
        errors="coerce"
    )
    runtime_text = (
        f"{int(runtime)} min"
        if pd.notna(runtime) and runtime > 0
        else "N/A"
    )

    release_date = get_text(movie.get("release_date"))
    year = release_date[:4] if release_date != "Not available" else "N/A"

    st.markdown("""
    <style>
    div[role="dialog"][aria-modal="true"] {
        width: min(760px, calc(100vw - 24px)) !important;
        max-width: 760px !important;
        background: #11101d !important;
        border: 1px solid #393049 !important;
        border-radius: 20px !important;
        box-shadow: 0 25px 80px rgba(0,0,0,.65) !important;
    }

    div[role="dialog"][aria-modal="true"]
    [data-testid="stDialogContent"] {
        max-height: 80vh;
        overflow-y: auto;
    }

    .cm-detail-title {
        color: #ffffff;
        font-size: 27px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 12px;
    }

    .cm-detail-rating {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 22px;
        color: #fbbf24;
        background: #302618;
        font-weight: 750;
        margin-bottom: 14px;
    }

    .cm-detail-label {
        color: #aca4bf;
        font-size: 12px;
        margin-top: 10px;
    }

    .cm-detail-value {
        color: #ffffff;
        font-size: 14px;
        font-weight: 650;
        margin-top: 3px;
    }

    .cm-detail-overview {
        color: #ded9e8;
        font-size: 14px;
        line-height: 1.75;
        margin-top: 8px;
    }

    .cm-section-heading {
        color: #c084fc;
        font-weight: 750;
        font-size: 16px;
        margin-top: 15px;
    }

    @media (prefers-reduced-motion: reduce) {
        div[role="dialog"][aria-modal="true"] {
            animation: none !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    poster_col, info_col = st.columns(
        [1, 1.7],
        gap="medium",
        vertical_alignment="top"
    )

    with poster_col:
        if poster.startswith("https://"):
            st.image(poster, width=190)
        else:
            st.info("Poster unavailable")

    with info_col:
        st.markdown(
            f'<div class="cm-detail-title">'
            f'{html.escape(title)}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="cm-detail-rating">'
            f'★ {rating_text}</div>',
            unsafe_allow_html=True
        )

        details = [
            ("Release Year", year),
            ("Runtime", runtime_text),
            ("Genres", genres),
        ]

        for label, value in details:
            st.markdown(
                f'<div class="cm-detail-label">'
                f'{html.escape(label)}</div>'
                f'<div class="cm-detail-value">'
                f'{html.escape(str(value))}</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="cm-section-heading">Story Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="cm-detail-overview">'
        f'{html.escape(overview)}</div>',
        unsafe_allow_html=True
    )

    st.caption("Movie information provided by TMDb.")
