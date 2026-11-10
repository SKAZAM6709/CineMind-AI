import time
from pathlib import Path

import pandas as pd

from src.movie_recommender.data.tmdb_client import TMDbClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINKS_PATH = PROJECT_ROOT / "data" / "raw" / "movielens" / "links.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "external" / "tmdb" / "tmdb_movies_sample.csv"


def main(limit: int = 50) -> None:
    links = pd.read_csv(LINKS_PATH)

    # Keep rows that have a valid TMDb ID
    links = links[links["tmdbId"].notna()].copy()
    links["tmdbId"] = links["tmdbId"].astype(int)

    sample = links.head(limit)

    client = TMDbClient()
    rows = []

    for index, row in sample.iterrows():
        movie_id = int(row["movieId"])
        tmdb_id = int(row["tmdbId"])

        try:
            details = client.get_movie_details(tmdb_id)

            if not details:
                print(f"Skipped movieId={movie_id}, tmdbId={tmdb_id} (not found)")
                continue

            rows.append(
                {
                    "movieId": movie_id,
                    "tmdbId": tmdb_id,
                    "title": details.get("title"),
                    "original_title": details.get("original_title"),
                    "overview": details.get("overview"),
                    "tagline": details.get("tagline"),
                    "release_date": details.get("release_date"),
                    "runtime": details.get("runtime"),
                    "status": details.get("status"),
                    "original_language": details.get("original_language"),
                    "vote_average": details.get("vote_average"),
                    "vote_count": details.get("vote_count"),
                    "popularity": details.get("popularity"),
                    "poster_path": details.get("poster_path"),
                    "backdrop_path": details.get("backdrop_path"),
                    "poster_url": client.build_image_url(details.get("poster_path"), size="w500"),
                    "backdrop_url": client.build_image_url(details.get("backdrop_path"), size="w780"),
                    "genres": " | ".join(
                        genre["name"] for genre in details.get("genres", [])
                    ),
                }
            )

            print(f"Fetched: movieId={movie_id}, tmdbId={tmdb_id}, title={details.get('title')}")
            time.sleep(0.25)

        except Exception as error:
            print(f"Error for movieId={movie_id}, tmdbId={tmdb_id}: {error}")

    output_df = pd.DataFrame(rows)
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\nTMDb metadata fetch completed.")
    print(f"Rows saved: {len(output_df)}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main(limit=50)