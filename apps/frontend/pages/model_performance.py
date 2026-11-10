
"""
CineMind AI - Model Performance Dashboard

Displays measured RAG and retrieval evaluation
results from the project's existing CSV files.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EVAL_DIR = ROOT / "reports" / "evaluation"

REVIEW_FILES = [
    EVAL_DIR / "retrieval_review_utf8.csv",
    EVAL_DIR / "retrieval_review.csv",
]

RAG_FILE = EVAL_DIR / "rag_results.csv"


# ==========================================
# STYLE
# ==========================================

st.markdown("""
<style>

.mp-heading {
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 850;
    letter-spacing: -1.4px;
    color: #ffffff;
    margin-bottom: 8px;
}

.mp-heading span {
    color: #c084fc;
}

.mp-subtitle {
    font-size: 15px;
    color: #ada5c2;
    margin-bottom: 28px;
}

.mp-section {
    margin-top: 26px;
    margin-bottom: 10px;
    font-size: 22px;
    font-weight: 750;
    color: #f6f0ff;
}

.mp-note {
    border: 1px solid #4c345d;
    background: #21172e;
    border-radius: 14px;
    padding: 16px;
    color: #e4d8ed;
    line-height: 1.7;
    margin: 16px 0;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# DATA LOADING
# ==========================================

def read_csv_safely(path):
    """Read UTF-8 or common Windows-encoded CSV."""

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp1252",
    ]:

        try:
            return pd.read_csv(
                path,
                encoding=encoding,
            )

        except UnicodeDecodeError:
            continue

    raise UnicodeError(
        f"Unable to decode CSV: {path}"
    )


@st.cache_data(show_spinner=False)
def load_csv(path_text, modified_time):

    path = Path(path_text)

    return read_csv_safely(path)


def load_optional_csv(paths):

    errors = []

    for path in paths:

        if not path.exists():
            continue

        try:
            frame = load_csv(
                str(path),
                path.stat().st_mtime_ns,
            )

            return frame, path, None

        except (
            OSError,
            ValueError,
            UnicodeError,
            pd.errors.ParserError,
        ) as error:

            errors.append(
                f"{path.name}: {error}"
            )

    return (
        None,
        None,
        "; ".join(errors) if errors else None,
    )


def validate_columns(df, required):

    return [
        col for col in required
        if col not in df.columns
    ]


def numeric_column(df, column):

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def boolean_column(df, column):

    return (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )


def calculate_precision(df):

    required = [
        "query_id",
        "rank",
        "relevance_label",
    ]

    missing = validate_columns(
        df, required
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = df.copy()

    data["rank"] = numeric_column(
        data, "rank"
    )

    data["relevance_label"] = numeric_column(
        data, "relevance_label"
    )

    data = data[
        data["rank"].between(1, 5)
    ]

    records = []

    for query_id, group in data.groupby(
        "query_id"
    ):

        group = group.sort_values("rank")

        # Require exactly one label per rank.
        if (
            len(group) != 5
            or set(group["rank"]) != set(
                range(1, 6)
            )
        ):
            continue

        labels = group["relevance_label"]

        # Only valid, completed binary labels
        # contribute to Precision@5.
        if (
            labels.isna().any()
            or not labels.isin([0, 1]).all()
        ):
            continue

        relevant = int(labels.sum())

        records.append({
            "query_id": query_id,
            "relevant": relevant,
            "retrieved": 5,
            "precision_at_5": relevant / 5,
        })

    return pd.DataFrame(
        records,
        columns=[
            "query_id",
            "relevant",
            "retrieved",
            "precision_at_5",
        ],
    )


# ==========================================
# PAGE HEADER
# ==========================================

st.markdown(
    '<div class="mp-heading">'
    'Model <span>Performance</span>'
    '</div>'
    '<div class="mp-subtitle">'
    'Real evaluation metrics for CineMind AI '
    'retrieval and conversational RAG.'
    '</div>',
    unsafe_allow_html=True,
)


# ==========================================
# LOAD EVALUATION FILES
# ==========================================

review_df, review_path, review_error = (
    load_optional_csv(REVIEW_FILES)
)

rag_df, rag_path, rag_error = (
    load_optional_csv([RAG_FILE])
)


if review_error:
    st.warning(review_error)

if rag_error:
    st.warning(rag_error)


# ==========================================
# RETRIEVAL QUALITY
# ==========================================

st.markdown(
    '<div class="mp-section">'
    'Retrieval Quality'
    '</div>',
    unsafe_allow_html=True,
)

if review_df is None:

    st.warning(
        "No retrieval evaluation CSV found."
    )

    st.code(
        "reports/evaluation/retrieval_review_utf8.csv"
    )

else:

    try:
        precision_df = calculate_precision(
            review_df
        )

    except ValueError as error:
        st.error(str(error))
        precision_df = pd.DataFrame()

    if precision_df.empty:

        st.info(
            "No fully labeled top-five query groups "
            "were found. Complete relevance_label "
            "with 0 or 1 for each rank."
        )

    else:

        mean_precision = (
            precision_df["precision_at_5"].mean()
        )

        relevant_total = int(
            precision_df["relevant"].sum()
        )

        retrieved_total = int(
            precision_df["retrieved"].sum()
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Mean Precision@5",
                f"{mean_precision:.1%}",
            )

        with col2:
            st.metric(
                "Relevant Movies",
                f"{relevant_total}/{retrieved_total}",
            )

        with col3:
            st.metric(
                "Evaluated Queries",
                len(precision_df),
            )

        st.markdown(
            "#### Precision@5 by Query"
        )

        plot_df = (
            precision_df[
                ["query_id", "precision_at_5"]
            ]
            .set_index("query_id")
            .rename(
                columns={
                    "precision_at_5":
                    "Precision@5"
                }
            )
            * 100
        )

        st.bar_chart(
            plot_df,
            y_label="Precision@5 (%)",
        )

        display_df = precision_df.copy()

        display_df["precision_at_5"] = (
            display_df["precision_at_5"]
            .map(lambda value: f"{value:.0%}")
        )

        display_df = display_df.rename(
            columns={
                "query_id": "Query",
                "relevant": "Relevant",
                "retrieved": "Retrieved",
                "precision_at_5": "Precision@5",
            }
        )

        st.dataframe(
            display_df,
            hide_index=True,
            width="stretch",
        )

    st.caption(
        f"Source: {review_path.name}"
    )

    st.info(
        "Precision@5 measures relevance among "
        "the top five retrieved movies. "
        "It is not classification accuracy, "
        "answer faithfulness, or Recall@5."
    )

    with st.expander(
        "Inspect labeled movie records"
    ):

        st.dataframe(
            review_df,
            hide_index=True,
            width="stretch",
        )


# ==========================================
# RAG EVALUATION
# ==========================================

st.markdown(
    '<div class="mp-section">'
    'RAG System Evaluation'
    '</div>',
    unsafe_allow_html=True,
)

if rag_df is None:

    st.warning(
        "No RAG evaluation data found."
    )

    st.code(
        "python -m "
        "src.movie_recommender.evaluation."
        "rag_evaluation"
    )

else:

    required_rag = [
        "query_id",
        "retrieval_seconds",
        "generation_seconds",
        "llm_success",
        "citation_valid",
    ]

    missing = validate_columns(
        rag_df,
        required_rag,
    )

    if missing:

        st.error(
            f"Missing RAG evaluation columns: "
            f"{missing}"
        )

    elif rag_df.empty:

        st.info(
            "The RAG evaluation file is empty."
        )

    else:

        data = rag_df.copy()

        data["retrieval_seconds"] = (
            numeric_column(
                data,
                "retrieval_seconds",
            )
        )

        data["generation_seconds"] = (
            numeric_column(
                data,
                "generation_seconds",
            )
        )

        data["llm_success"] = (
            boolean_column(
                data,
                "llm_success",
            )
        )

        data["citation_valid"] = (
            boolean_column(
                data,
                "citation_valid",
            )
        )

        success_rate = (
            data["llm_success"].mean()
        )

        citation_rate = (
            data["citation_valid"].mean()
        )

        avg_retrieval = (
            data["retrieval_seconds"].mean()
        )

        avg_generation = (
            data["generation_seconds"].mean()
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "LLM Success Rate",
                f"{success_rate:.1%}",
            )

        with col2:
            st.metric(
                "Citation Validation Rate",
                f"{citation_rate:.1%}",
            )

        col3, col4 = st.columns(2)

        with col3:
            st.metric(
                "Avg. Retrieval Time",
                f"{avg_retrieval:.3f} s",
            )

        with col4:
            st.metric(
                "Avg. Generation Time",
                f"{avg_generation:.3f} s",
            )

        st.markdown(
            "#### Execution Time by Query"
        )

        time_df = data[
            [
                "query_id",
                "retrieval_seconds",
                "generation_seconds",
            ]
        ].set_index("query_id")

        time_df = time_df.rename(
            columns={
                "retrieval_seconds":
                "Retrieval (seconds)",
                "generation_seconds":
                "Generation (seconds)",
            }
        )

        st.bar_chart(
            time_df,
            y_label="Seconds",
        )

        st.markdown(
            "#### Detailed RAG Evaluation"
        )

        st.dataframe(
            data[
                [
                    "query_id",
                    "retrieval_seconds",
                    "generation_seconds",
                    "llm_success",
                    "citation_valid",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

        st.caption(
            f"Source: {rag_path.name}"
        )

        st.info(
            "Citation validation checks the "
            "presence and range of source numbers. "
            "It does not establish whether every "
            "generated factual claim is supported."
        )

        with st.expander(
            "View generated answers and errors"
        ):

            for _, record in rag_df.iterrows():

                st.markdown(
                    f"**{record.get('query_id', 'Query')}**"
                )

                st.write(
                    str(record.get("query", ""))
                )

                st.markdown(
                    str(record.get("answer", ""))
                )

                error = record.get("error", "")

                if pd.notna(error) and str(error).strip():
                    st.warning(str(error))

                st.divider()


# ==========================================
# PROJECT INFORMATION
# ==========================================

st.markdown(
    '<div class="mp-section">'
    'Evaluation Methodology'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="mp-note">'
    '<b>Retrieval engine:</b> '
    'Sentence Transformers semantic search'
    '<br>'
    '<b>Generation engine:</b> '
    'Ollama Gemma 3 1B'
    '<br>'
    '<b>Evaluation:</b> '
    'Top-five relevance labels, '
    'runtime measurements and '
    'basic citation checks'
    '</div>',
    unsafe_allow_html=True,
)

st.warning(
    "This is an exploratory evaluation on "
    "a small test set. Relevance labels "
    "should be independently reviewed before "
    "using the results as final research evidence."
)

st.caption(
    "CineMind AI • Model Performance Dashboard"
)
