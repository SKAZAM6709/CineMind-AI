
"""
CineMind AI - Safe Expanded Catalog Activation

Commands:
    python -m scripts.activate_expanded_catalog --check
    python -m scripts.activate_expanded_catalog --activate
    python -m scripts.activate_expanded_catalog --restore

Stop Streamlit before activation or restoration.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LIVE_DATA = ROOT / "data/external/tmdb/tmdb_movies.csv"
STAGED_DATA = ROOT / "data/external/tmdb/tmdb_movies_expanded.csv"

LIVE_NLP = ROOT / "models/nlp_tfidf.joblib"
STAGED_NLP = ROOT / "models/staging_500/nlp_tfidf.joblib"

LIVE_SEM = ROOT / "models/semantic_embeddings.npz"
STAGED_SEM = ROOT / "models/staging_500/semantic_embeddings.npz"

CONTENT = ROOT / "models/content_tfidf.joblib"
MANIFEST = ROOT / "models/staging_500/build_manifest.json"

BACKUP = ROOT / "models/backups/pre_expansion_150"

BACKUP_DATA = BACKUP / "tmdb_movies.csv"
BACKUP_NLP = BACKUP / "nlp_tfidf.joblib"
BACKUP_SEM = BACKUP / "semantic_embeddings.npz"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def movie_ids(df):
    values = pd.to_numeric(
        df["movieId"],
        errors="raise",
    )

    if values.isna().any():
        raise ValueError("Missing movie IDs.")

    ids = values.astype(int).tolist()

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate movie IDs.")

    return ids


def validate_staging():

    paths = [
        STAGED_DATA,
        STAGED_NLP,
        STAGED_SEM,
        CONTENT,
        MANIFEST,
    ]

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    if manifest.get("status") != "staging_validated":
        raise ValueError(
            "Staging manifest is not validated."
        )

    if sha256(STAGED_DATA) != manifest["catalog_sha256"]:
        raise ValueError(
            "Expanded catalog changed after model training. "
            "Rebuild the staged models first."
        )

    catalog = pd.read_csv(
        STAGED_DATA,
        encoding="utf-8-sig",
    )

    ids = movie_ids(catalog)

    if len(ids) != manifest["movie_count"]:
        raise ValueError("Catalog row count mismatch.")

    if ids != manifest["movie_ids"]:
        raise ValueError("Catalog order differs from manifest.")

    if not {"title", "overview"}.issubset(catalog.columns):
        raise ValueError("Required metadata is missing.")

    if (
        catalog["title"].fillna("").str.strip().eq("").any()
        or catalog["overview"].fillna("").str.strip().eq("").any()
    ):
        raise ValueError("Missing titles or descriptions.")

    nlp = joblib.load(STAGED_NLP)

    nlp_ids = [
        int(x) for x in nlp["movie_ids"]
    ]

    if nlp_ids != ids:
        raise ValueError("NLP IDs do not match catalog.")

    if nlp["matrix"].shape[0] != len(ids):
        raise ValueError("NLP row count mismatch.")

    with np.load(
        STAGED_SEM,
        allow_pickle=False,
    ) as data:

        sem_ids = data["movie_ids"].astype(int).tolist()
        embeddings = data["embeddings"]

        if sem_ids != ids:
            raise ValueError(
                "Semantic IDs do not match catalog."
            )

        if embeddings.shape != (
            len(ids),
            manifest["embedding_dimensions"],
        ):
            raise ValueError(
                "Semantic embedding shape mismatch."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Non-finite semantic embeddings."
            )

    content = joblib.load(CONTENT)

    content_ids = {
        int(x) for x in content["movie_ids"]
    }

    if not set(ids).issubset(content_ids):
        raise ValueError(
            "Some movies are missing from content TF-IDF."
        )

    print("Staging validation: PASSED")
    print("Movies:", len(ids))
    print("NLP matrix:", nlp["matrix"].shape)
    print(
        "Semantic matrix:",
        (len(ids), manifest["embedding_dimensions"]),
    )


def replace_file(source, destination):

    temporary = destination.with_name(
        destination.name + ".switch_tmp"
    )

    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_backup():

    pairs = [
        (LIVE_DATA, BACKUP_DATA),
        (LIVE_NLP, BACKUP_NLP),
        (LIVE_SEM, BACKUP_SEM),
    ]

    existing = [
        destination.exists()
        for _, destination in pairs
    ]

    if all(existing):
        print("Existing backup detected.")
        return

    if any(existing):
        raise RuntimeError(
            "Incomplete backup found. "
            "Inspect the backup folder before continuing."
        )

    BACKUP.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source, _ in pairs:
        if not source.exists():
            raise FileNotFoundError(source)

    for source, destination in pairs:
        shutil.copy2(source, destination)

    print("Original live system backed up.")


def restore():

    pairs = [
        (BACKUP_DATA, LIVE_DATA),
        (BACKUP_NLP, LIVE_NLP),
        (BACKUP_SEM, LIVE_SEM),
    ]

    for source, _ in pairs:
        if not source.exists():
            raise FileNotFoundError(
                f"Missing backup: {source}"
            )

    for source, destination in pairs:
        replace_file(source, destination)

    print("\nOriginal system restored.")
    print("Restart Streamlit to clear cached models.")


def activate():

    validate_staging()
    ensure_backup()

    pairs = [
        (STAGED_DATA, LIVE_DATA),
        (STAGED_NLP, LIVE_NLP),
        (STAGED_SEM, LIVE_SEM),
    ]

    try:
        for source, destination in pairs:
            replace_file(source, destination)

        # Confirm the active dataset matches the staging one.
        if sha256(LIVE_DATA) != sha256(STAGED_DATA):
            raise ValueError(
                "Active dataset verification failed."
            )

        if sha256(LIVE_NLP) != sha256(STAGED_NLP):
            raise ValueError(
                "Active NLP verification failed."
            )

        if sha256(LIVE_SEM) != sha256(STAGED_SEM):
            raise ValueError(
                "Active semantic verification failed."
            )

    except Exception:
        print("Activation failed. Attempting rollback.")
        restore()
        raise

    print("\nExpanded catalog activated successfully.")
    print("Active movies:", len(pd.read_csv(LIVE_DATA)))
    print("Original backup:", BACKUP)
    print("Restart Streamlit before testing.")


def main():

    parser = argparse.ArgumentParser()

    action = parser.add_mutually_exclusive_group(
        required=True
    )

    action.add_argument(
        "--check",
        action="store_true",
    )

    action.add_argument(
        "--activate",
        action="store_true",
    )

    action.add_argument(
        "--restore",
        action="store_true",
    )

    args = parser.parse_args()

    if args.check:
        validate_staging()
    elif args.activate:
        activate()
    elif args.restore:
        restore()


if __name__ == "__main__":
    main()
