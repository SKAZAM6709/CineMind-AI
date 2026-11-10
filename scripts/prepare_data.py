"""Run from project root: python scripts/prepare_data.py"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.movie_recommender.data.prepare_catalog import main

if __name__ == '__main__':
    main()
