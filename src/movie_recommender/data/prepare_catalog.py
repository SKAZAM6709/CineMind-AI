"""Prepare a movie-level catalog from the MovieLens Latest Small dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'data' / 'raw' / 'movielens'
OUT = ROOT / 'data' / 'processed' / 'movie_catalog.csv'


def prepare_catalog() -> pd.DataFrame:
    movies = pd.read_csv(RAW / 'movies.csv')
    ratings = pd.read_csv(RAW / 'ratings.csv')
    tags = pd.read_csv(RAW / 'tags.csv')
    links = pd.read_csv(RAW / 'links.csv')

    # Movie-level ratings; do not confuse an unrated movie with a zero-star movie.
    stats = (ratings.groupby('movieId')['rating']
             .agg(rating_count='count', average_rating='mean')
             .reset_index())

    # Combine distinct user-generated tags for each movie.
    tags['tag'] = tags['tag'].astype(str).str.strip()
    tags = tags[tags['tag'].ne('')]
    distinct = tags.drop_duplicates(['movieId', 'tag'])
    tag_stats = (distinct.groupby('movieId')['tag']
                 .agg(lambda values: ' | '.join(values))
                 .reset_index(name='tags'))

    catalog = (movies.merge(links, on='movieId', how='left', validate='one_to_one')
              .merge(stats, on='movieId', how='left', validate='one_to_one')
              .merge(tag_stats, on='movieId', how='left', validate='one_to_one'))

    catalog['rating_count'] = catalog['rating_count'].fillna(0).astype(int)
    catalog['average_rating'] = catalog['average_rating'].round(2)
    catalog['tags'] = catalog['tags'].fillna('')
    catalog['genres'] = catalog['genres'].replace('(no genres listed)', '')
    # Text intended for later embeddings. This is only title/genre/tag text,
    # not a plot description or a finished RAG document.
    catalog['search_text'] = (
        'Title: ' + catalog['title'] + '. Genres: ' +
        catalog['genres'].str.replace('|', ', ', regex=False) +
        '. Tags: ' + catalog['tags'].str.replace('|', ', ', regex=False)
    )
    assert len(catalog) == len(movies) and catalog['movieId'].is_unique
    return catalog


def main() -> None:
    catalog = prepare_catalog()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(OUT, index=False, encoding='utf-8')
    print(f'Catalog created: {OUT}')
    print(f'Movies: {len(catalog):,}')
    print(f'Movies with ratings: {(catalog.rating_count > 0).sum():,}')
    print(f'Movies with tags: {(catalog.tags != "").sum():,}')
    print(f'Missing TMDb IDs: {catalog.tmdbId.isna().sum():,}')
    print('\nSample:')
    print(catalog[['movieId', 'title', 'rating_count', 'average_rating', 'tags']].head(5).to_string(index=False))


if __name__ == '__main__':
    main()
