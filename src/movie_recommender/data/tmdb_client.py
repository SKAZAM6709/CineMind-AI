import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


load_dotenv()


class TMDbClient:
    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self) -> None:
        token = os.getenv("TMDB_READ_ACCESS_TOKEN")
        if not token:
            raise ValueError("TMDB_READ_ACCESS_TOKEN not found in environment.")

        self.headers = {
            "Authorization": f"Bearer {token}",
            "accept": "application/json",
        }

    def get_movie_details(self, tmdb_id: int) -> Optional[Dict[str, Any]]:
        url = f"{self.BASE_URL}/movie/{tmdb_id}"
        response = requests.get(url, headers=self.headers, timeout=20)

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return response.json()

    @staticmethod
    def build_image_url(path: Optional[str], size: str = "w500") -> Optional[str]:
        if not path:
            return None
        return f"https://image.tmdb.org/t/p/{size}{path}"