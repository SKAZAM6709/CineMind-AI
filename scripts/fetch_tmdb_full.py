
import time
from pathlib import Path

import pandas as pd
import requests

from src.movie_recommender.data.tmdb_client import TMDbClient


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

LINKS_PATH = (
    ROOT / "data" / "raw" / "movielens" / "links.csv"
)

SAMPLE_PATH = (
    ROOT / "data" / "external" / "tmdb"
    / "tmdb_movies_sample.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "external" / "tmdb"
    / "tmdb_movies.csv"
)

# Number of new movies to attempt per run
BATCH_SIZE = 100

# Save progress periodically
SAVE_EVERY = 10

REQUEST_DELAY = 0.35

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


# ==========================================
# LOAD EXISTING PROGRESS
# ==========================================

def load_existing():

    if OUTPUT_PATH.exists():
        print("Resuming existing TMDb catalog.")
        data = pd.read_csv(OUTPUT_PATH)

    elif SAMPLE_PATH.exists():
        print("Starting from the existing sample.")
        data = pd.read_csv(SAMPLE_PATH)

    else:
        print("Starting a new catalog.")
        data = pd.DataFrame(columns=COLUMNS)

    for column in COLUMNS:
        if column not in data.columns:
            data[column] = None

    data = data[COLUMNS]

    if not data.empty:
        data = data.drop_duplicates(
            subset=["movieId"],
            keep="last"
        )

    return data


def save_progress(records):

    output = pd.DataFrame(records, columns=COLUMNS)

    output = output.drop_duplicates(
        subset=["movieId"],
        keep="last"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save through a temporary file
    temporary = OUTPUT_PATH.with_suffix(".tmp")

    output.to_csv(
        temporary,
        index=False,
        encoding="utf-8"
    )

    temporary.replace(OUTPUT_PATH)


# ==========================================
# FORMAT TMDB MOVIE DETAILS
# ==========================================

def prepare_record(movie_id, tmdb_id, details, client):

    return {
        "movieId": movie_id,
        "tmdbId": tmdb_id,
        "title": details.get("title"),
        "original_title": details.get("original_title"),
        "overview": details.get("overview"),
        "tagline": details.get("tagline"),
        "release_date": details.get("release_date"),
        "runtime": details.get("runtime"),
        "status": details.get("status"),
        "original_language": details.get(
            "original_language"
        ),
        "vote_average": details.get("vote_average"),
        "vote_count": details.get("vote_count"),
        "popularity": details.get("popularity"),
        "poster_path": details.get("poster_path"),
        "backdrop_path": details.get("backdrop_path"),
        "poster_url": client.build_image_url(
            details.get("poster_path"), "w500"
        ),
        "backdrop_url": client.build_image_url(
            details.get("backdrop_path"), "w780"
        ),
        "genres": " | ".join(
            genre.get("name", "")
            for genre in details.get("genres", [])
        ),
    }


# ==========================================
# FETCH DETAILS WITH RETRIES
# ==========================================

def fetch_with_retry(client, tmdb_id):

    for attempt in range(3):

        try:
            return client.get_movie_details(tmdb_id)

        except requests.HTTPError as error:

            status = (
                error.response.status_code
                if error.response is not None
                else None
            )

            if status in (401, 403):
                raise RuntimeError(
                    "TMDb authentication or permission error. "
                    "Check your API credentials."
                ) from error

            if status == 429 or (
                status is not None and status >= 500
            ):
                delay = 3 * (attempt + 1)

                if error.response is not None:
                    retry_after = error.response.headers.get(
                        "Retry-After"
                    )
                    if retry_after:
                        try:
                            delay = max(
                                delay, float(retry_after)
                            )
                        except ValueError:
                            pass

                print(
                    f"Temporary HTTP {status}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
                continue

            raise

        except requests.RequestException:
            time.sleep(3 * (attempt + 1))

    raise RuntimeError(
        f"Request failed after retries: {tmdb_id}"
    )


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():

    links = pd.read_csv(LINKS_PATH)
    links = links.dropna(subset=["tmdbId"]).copy()

    links["tmdbId"] = links["tmdbId"].astype(int)
    links["movieId"] = links["movieId"].astype(int)

    existing = load_existing()

    records = existing.to_dict("records")

    completed_ids = set(
        existing["movieId"].astype(int).tolist()
    )

    remaining = links[
        ~links["movieId"].isin(completed_ids)
    ]

    client = TMDbClient()

    successful = 0
    attempted = 0
    not_found = 0
    failed = 0

    print(f"Movies already saved: {len(records)}")
    print(f"Remaining movies: {len(remaining)}")
    print(f"Batch size: {BATCH_SIZE}")

    for _, row in remaining.head(BATCH_SIZE).iterrows():

        movie_id = int(row["movieId"])
        tmdb_id = int(row["tmdbId"])

        attempted += 1

        try:
            details = fetch_with_retry(
                client, tmdb_id
            )

            if details is None:
                not_found += 1
                print(f"Not found: {tmdb_id}")
                continue

            record = prepare_record(
                movie_id,
                tmdb_id,
                details,
                client
            )

            records.append(record)
            successful += 1

            print(
                f"[{attempted}/{BATCH_SIZE}] "
                f"Fetched: {record['title']}"
            )

        except RuntimeError as error:
            print(f"Stopped: {error}")
            break

        except requests.RequestException as error:
            failed += 1
            print(f"Request failed for {tmdb_id}: {error}")

        finally:
            if attempted % SAVE_EVERY == 0:
                save_progress(records)
                print("Progress saved.")

            time.sleep(REQUEST_DELAY)

    save_progress(records)

    print("\n===== BATCH SUMMARY =====")
    print(f"Attempted: {attempted}")
    print(f"Successful new records: {successful}")
    print(f"Not found: {not_found}")
    print(f"Failed: {failed}")
    print(f"Total saved records: {len(records)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
