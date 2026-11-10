
"""
CineMind AI - Watchlist Manager

Session-based watchlist storage.
"""

import streamlit as st


WATCHLIST_KEY = "cinemind_watchlist"


def initialize_watchlist():
    if WATCHLIST_KEY not in st.session_state:
        st.session_state[WATCHLIST_KEY] = []


def get_watchlist():
    initialize_watchlist()
    return list(st.session_state[WATCHLIST_KEY])


def is_saved(movie_id):
    return int(movie_id) in get_watchlist()


def add_to_watchlist(movie_id):
    initialize_watchlist()

    movie_id = int(movie_id)

    if movie_id not in st.session_state[WATCHLIST_KEY]:
        st.session_state[WATCHLIST_KEY].append(movie_id)
        return True

    return False


def remove_from_watchlist(movie_id):
    initialize_watchlist()

    movie_id = int(movie_id)

    if movie_id in st.session_state[WATCHLIST_KEY]:
        st.session_state[WATCHLIST_KEY].remove(movie_id)
        return True

    return False


def clear_watchlist():
    st.session_state[WATCHLIST_KEY] = []
