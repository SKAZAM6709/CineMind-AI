
"""
CineMind AI - Expand TMDb Movie Catalog

Build a staged 500-movie metadata catalog while
preserving the existing live 150-movie dataset.

Input:
    data/processed/movie_catalog.csv
    data/external/tmdb/tmdb_movies.csv

Output:
    data/external/tmdb/tmdb_movies_expanded.csv

Features:
- Resume incomplete runs
- Skip duplicate movieIds
- Fetch official TMDb movie details
- Require a usable overview
- Retry temporary API failures
- Save progress every 20 movies
- Never overwrite the live catalog
"""

import argparse
import os
import time

from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = (
    ROOT / "data" / "processed" / "movie_catalog.csv"
)

LIVE_PATH = (
    ROOT / "data" / "external" / "tmdb"
    / "tmdb_movies.csv"
)

STAGED_PATH = (
    ROOT / "data" / "external" / "tmdb"
    / "tmdb_movies_expanded.csv"
)

API_BASE = "https://api.themoviedb.org/3"

IMAGE_BASE = "https://image.tmdb.org/t/p"

COLUMNS = [
    "movieId",
    "tmdbId",
    "title",
    "original_title",
    "overview",
    "tagline",
    "release_date",
    "runtime",
    "status",
    "original_language",
    "vote_average",
    "vote_count",
    "popularity",
    "poster_path",
    "backdrop_path",
    "poster_url",
    "backdrop_url",
    "genres",
]


def load_token():

    load_dotenv(ROOT / ".env")

    token = os.getenv(
        "TMDB_READ_ACCESS_TOKEN", ""
    ).strip()

    if not token:
        raise RuntimeError(
            "TMDB_READ_ACCESS_TOKEN was not found "
            "in the project .env file."
        )

    return token


def load_inputs():

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            CATALOG_PATH
        )

    if not LIVE_PATH.exists():
        raise FileNotFoundError(
            LIVE_PATH
        )

    catalog = pd.read_csv(
        CATALOG_PATH
    )

    source = (
        STAGED_PATH
        if STAGED_PATH.exists()
        else LIVE_PATH
    )

    existing = pd.read_csv(source)

    required_catalog = [
        "movieId",
        "tmdbId",
        "rating_count",
    ]

    for column in required_catalog:
        if column not in catalog.columns:
            raise ValueError(
                f"Missing catalog column: {column}"
            )

    for column in COLUMNS:
        if column not in existing.columns:
            existing[column] = ""

    catalog["movieId"] = pd.to_numeric(
        catalog["movieId"],
        errors="coerce"
    )

    catalog["tmdbId"] = pd.to_numeric(
        catalog["tmdbId"],
        errors="coerce"
    )

    catalog["rating_count"] = pd.to_numeric(
        catalog["rating_count"],
        errors="coerce"
    ).fillna(0)

    catalog = catalog.dropna(
        subset=["movieId", "tmdbId"]
    )

    catalog = catalog[
        (catalog["movieId"] > 0)
        & (catalog["tmdbId"] > 0)
    ]

    catalog["movieId"] = (
        catalog["movieId"].astype(int)
    )

    catalog["tmdbId"] = (
        catalog["tmdbId"].astype(int)
    )

    catalog = catalog.drop_duplicates(
        subset=["movieId"]
    )

    # Prefer MovieLens movies with many ratings.
    catalog = catalog.sort_values(
        "rating_count",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    existing["movieId"] = pd.to_numeric(
        existing["movieId"],
        errors="coerce"
    )

    existing = existing.dropna(
        subset=["movieId"]
    )

    existing["movieId"] = (
        existing["movieId"].astype(int)
    )

    existing = existing.drop_duplicates(
        subset=["movieId"],
        keep="first"
    )

    return (
        catalog,
        existing[COLUMNS].copy(),
        source,
    )


def fetch_movie(session, tmdb_id):

    url = (
        f"{API_BASE}/movie/{tmdb_id}"
    )

    for attempt in range(4):

        try:
            response = session.get(
                url,
                params={"language": "en-US"},
                timeout=25,
            )

        except requests.RequestException:
            if attempt == 3:
                return None

            time.sleep(
                min(2 ** attempt, 8)
            )
            continue

        if response.status_code == 404:
            return None

        if response.status_code in (
            401, 403
        ):
            raise RuntimeError(
                "TMDb authentication failed "
                f"(HTTP {response.status_code}). "
                "Check your .env access token."
            )

        if (
            response.status_code == 429
            or response.status_code >= 500
        ):
            if attempt == 3:
                return None

            retry_after = (
                response.headers.get(
                    "Retry-After", ""
                )
            )

            try:
                delay = min(
                    max(float(retry_after), 1),
                    60
                )
            except ValueError:
                delay = min(
                    2 ** (attempt + 1), 30
                )

            print(
                "Temporary TMDb error",
                response.status_code,
                "- retrying after",
                delay,
                "seconds",
            )

            time.sleep(delay)
            continue

        response.raise_for_status()

        return response.json()

    return None


def image_url(path, size):

    if not isinstance(path, str):
        return ""

    if not path.startswith("/"):
        return ""

    return (
        f"{IMAGE_BASE}/{size}{path}"
    )


def build_movie_record(
    movie_id,
    tmdb_id,
    data,
):

    overview = str(
        data.get("overview") or ""
    ).strip()

    # Only index movies with useful descriptions.
    if len(overview) < 30:
        return None

    title = str(
        data.get("title") or ""
    ).strip()

    if not title:
        return None

    poster_path = (
        data.get("poster_path") or ""
    )

    backdrop_path = (
        data.get("backdrop_path") or ""
    )

    genres = " | ".join(
        genre.get("name", "")
        for genre in data.get("genres", [])
        if genre.get("name")
    )

    return {
        "movieId": int(movie_id),
        "tmdbId": int(tmdb_id),
        "title": title,
        "original_title": (
            data.get("original_title") or ""
        ),
        "overview": overview,
        "tagline": (
            data.get("tagline") or ""
        ),
        "release_date": (
            data.get("release_date") or ""
        ),
        "runtime": (
            data.get("runtime") or 0
        ),
        "status": (
            data.get("status") or ""
        ),
        "original_language": (
            data.get("original_language") or ""
        ),
        "vote_average": (
            data.get("vote_average") or 0
        ),
        "vote_count": (
            data.get("vote_count") or 0
        ),
        "popularity": (
            data.get("popularity") or 0
        ),
        "poster_path": poster_path,
        "backdrop_path": backdrop_path,
        "poster_url": image_url(
            poster_path, "w500"
        ),
        "backdrop_url": image_url(
            backdrop_path, "w1280"
        ),
        "genres": genres,
    }


def save_progress(rows):

    frame = pd.DataFrame(
        rows,
        columns=COLUMNS,
    )

    frame = frame.drop_duplicates(
        subset=["movieId"],
        keep="first"
    )

    STAGED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = STAGED_PATH.with_suffix(
        ".tmp.csv"
    )

    frame.to_csv(
        temp_path,
        index=False,
        encoding="utf-8-sig",
    )

    temp_path.replace(
        STAGED_PATH
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--max-new",
        type=int,
        default=350,
    )

    args = parser.parse_args()

    if args.target <= 0 or args.max_new <= 0:
        parser.error(
            "Both --target and --max-new "
            "must be positive."
        )

    token = load_token()

    catalog, existing, source = (
        load_inputs()
    )

    rows = existing.to_dict("records")

    saved_ids = set(
        existing["movieId"].tolist()
    )

    print(
        "\n===== CINEMIND AI CATALOG EXPANSION ====="
    )

    print(
        "Starting dataset:",
        source.name,
    )

    print(
        "Existing movies:",
        len(saved_ids),
    )

    print(
        "Target movies:",
        args.target,
    )

    print(
        "Maximum new movies this run:",
        args.max_new,
    )

    if len(saved_ids) >= args.target:

        print(
            "Target already reached. "
            "No new API requests needed."
        )

        return

    session = requests.Session()

    session.headers.update({
        "Authorization": (
            f"Bearer {token}"
        ),
        "accept": "application/json",
    })

    new_count = 0
    attempted = 0

    try:

        for _, movie in catalog.iterrows():

            if len(saved_ids) >= args.target:
                break

            if new_count >= args.max_new:
                break

            movie_id = int(
                movie["movieId"]
            )

            tmdb_id = int(
                movie["tmdbId"]
            )

            if movie_id in saved_ids:
                continue

            attempted += 1

            data = fetch_movie(
                session,
                tmdb_id,
            )

            if data is None:
                continue

            record = build_movie_record(
                movie_id,
                tmdb_id,
                data,
            )

            if record is None:
                continue

            rows.append(record)

            saved_ids.add(movie_id)

            new_count += 1

            print(
                f"[{len(saved_ids)}/{args.target}] "
                f"{record['title']}"
            )

            if new_count % 20 == 0:
                save_progress(rows)
                print("Progress saved.")

            # Keep requests reasonably spaced.
            time.sleep(0.25)

    finally:

        session.close()

        if new_count > 0:
            save_progress(rows)

    print(
        "\n===== EXPANSION SUMMARY ====="
    )

    print("Total movies:", len(saved_ids))

    print("New movies added:", new_count)

    print("API candidates attempted:", attempted)

    print(
        "Staged dataset:",
        STAGED_PATH
    )

    print(
        "Live dataset remains unchanged:",
        LIVE_PATH
    )


if __name__ == "__main__":
    main()
