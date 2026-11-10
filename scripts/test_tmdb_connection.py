
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Read our TMDb access token
token = os.getenv("TMDB_READ_ACCESS_TOKEN")

if not token:
    raise ValueError("TMDb token not found in .env file")

# TMDb movie details API
url = "https://api.themoviedb.org/3/movie/550"

headers = {
    "Authorization": f"Bearer {token}",
    "accept": "application/json"
}

try:
    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()
    movie = response.json()

    print("TMDb API connected successfully!")
    print("Movie:", movie["title"])
    print("Release Date:", movie["release_date"])
    print("Rating:", movie["vote_average"])
    print("Poster:", movie["poster_path"])
    print("Backdrop:", movie["backdrop_path"])

except requests.exceptions.RequestException as error:
    print("TMDb API connection failed.")
    print("Error:", error)
