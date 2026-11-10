
from pathlib import Path

import streamlit as st


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

FRONTEND_DIR = Path(__file__).resolve().parent
PAGES_DIR = FRONTEND_DIR / "pages"

LOGO_PATH = (
    FRONTEND_DIR
    / "assets"
    / "images"
    / "cinemind-logo.svg"
)

BRAND_NAME = "CineMind AI"


st.set_page_config(
    page_title=f"{BRAND_NAME} | Discover Movies",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# MAIN APPLICATION STYLES
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #080811;
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background: #12111e;
    border-right: 1px solid #28243b;
}

[data-testid="stSidebarNav"] {
    padding-top: 1rem;
}

[data-testid="stSidebarNav"] a {
    border-radius: 10px;
    transition: background .2s ease;
}

[data-testid="stSidebarNav"] a:hover {
    background: rgba(168, 85, 247, .12);
}

/* Keep content below Streamlit's toolbar */
.block-container {
    padding-top: 3.5rem;
    padding-bottom: 3rem;
}

@media (prefers-reduced-motion: reduce) {
    [data-testid="stSidebarNav"] a {
        transition: none;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# PAGE DEFINITIONS
# ==========================================

home = st.Page(
    str(FRONTEND_DIR / "cinematic_carousels.py"),
    title="Home",
    icon=":material/home:",
    default=True,
)

explore = st.Page(
    str(PAGES_DIR / "explore.py"),
    title="Explore",
    icon=":material/explore:",
)

trending = st.Page(
    str(PAGES_DIR / "trending.py"),
    title="Trending",
    icon=":material/trending_up:",
)

top_rated = st.Page(
    str(PAGES_DIR / "top_rated.py"),
    title="Top Rated",
    icon=":material/star:",
)

genres = st.Page(
    str(PAGES_DIR / "genres.py"),
    title="Genres",
    icon=":material/category:",
)

for_you = st.Page(
    str(PAGES_DIR / "for_you.py"),
    title="For You",
    icon=":material/auto_awesome:",
)

ai_assistant = st.Page(
    str(PAGES_DIR / "ai_assistant.py"),
    title="AI Movie Assistant",
    icon=":material/smart_toy:",
)

watchlist = st.Page(
    str(PAGES_DIR / "watchlist.py"),
    title="My Watchlist",
    icon=":material/bookmark:",
)

insights = st.Page(
    str(PAGES_DIR / "insights.py"),
    title="Recommendation Insights",
    icon=":material/analytics:",
)

model_performance = st.Page(
    str(PAGES_DIR / "model_performance.py"),
    title="Model Performance",
    icon=":material/monitoring:",
)

about = st.Page(
    str(PAGES_DIR / "about.py"),
    title="About",
    icon=":material/info:",
)


# ==========================================
# NAVIGATION
# ==========================================

navigation = st.navigation(
    {
        "MOVIE DISCOVERY": [
            home,
            explore,
            trending,
            top_rated,
            genres,
        ],
        "PERSONALIZED EXPERIENCE": [
            for_you,
            ai_assistant,
            watchlist,
        ],
        "ANALYTICS": [
            insights,
            model_performance,
        ],
        "INFORMATION": [
            about,
        ],
    },
    position="sidebar",
)


# ==========================================
# SIDEBAR BRANDING
# ==========================================

if LOGO_PATH.exists():
    st.logo(
        str(LOGO_PATH),
        size="large",
        icon_image=str(LOGO_PATH),
    )

with st.sidebar:
    st.caption("YOUR INTELLIGENT MOVIE COMPANION")


# ==========================================
# RUN SELECTED PAGE
# ==========================================

navigation.run()
