from src.movie_recommender.data.prepare_catalog import prepare_catalog


def test_catalog_is_complete_and_unique():
    catalog = prepare_catalog()
    assert len(catalog) == 9742
    assert catalog.movieId.is_unique
    assert (catalog.rating_count >= 0).all()
    assert catalog.search_text.notna().all()
