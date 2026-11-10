
"""
CineMind AI - RAG Movie Assistant

Generates movie recommendations using
retrieved movie evidence and Ollama Gemma 3.

Features:
- Grounded movie recommendations
- Conversation history
- Source citation validation
- Official movie title normalization
- Maximum 3 recommendations
- Ollama model warm loading
- Safe fallback recommendations
"""

import os
import re
import requests
import pandas as pd


# ==========================================
# CONFIGURATION
# ==========================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate",
)

DEFAULT_MODEL = "gemma3:1b"

MAX_RECOMMENDATIONS = 3
MAX_HISTORY_TURNS = 2
MAX_DESCRIPTION_LENGTH = 300


# ==========================================
# TEXT HELPERS
# ==========================================

def clean_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def normalize_title(title):
    """
    Remove Markdown formatting and numbering
    when comparing generated titles.
    """

    title = re.sub(
        r"\[\d+\]",
        "",
        str(title),
    )

    title = re.sub(
        r"^\s*\d+[.)]\s*",
        "",
        title,
    )

    title = re.sub(
        r"[*_`]",
        "",
        title,
    )

    return clean_text(title).casefold()


# ==========================================
# PREPARE RETRIEVED MOVIES
# ==========================================

def prepare_movies(results):

    if results is None:
        return []

    if isinstance(results, pd.DataFrame):
        records = results.to_dict("records")

    elif isinstance(results, dict):
        records = [results]

    else:
        records = list(results)

    movies = []

    for item in records:

        if not isinstance(item, dict):
            continue

        title = clean_text(
            item.get("title", "")
        )

        if not title:
            continue

        genres = clean_text(
            item.get("genres", "")
        )

        overview = clean_text(
            item.get("overview", "")
        )

        release_date = clean_text(
            item.get("release_date", "")
        )

        if len(overview) > MAX_DESCRIPTION_LENGTH:
            overview = (
                overview[:MAX_DESCRIPTION_LENGTH]
                .rsplit(" ", 1)[0]
                + "..."
            )

        movies.append({
            "title": title,
            "genres": genres,
            "overview": overview,
            "release_date": release_date,
            "movieId": item.get("movieId"),
        })

    return movies


# ==========================================
# BUILD MOVIE EVIDENCE
# ==========================================

def build_evidence(movies):

    evidence = []

    for number, movie in enumerate(
        movies,
        start=1,
    ):

        evidence.append(
            f"[{number}]\n"
            f"Title: {movie['title']}\n"
            f"Genres: {movie['genres']}\n"
            f"Release Date: {movie['release_date']}\n"
            f"Overview: {movie['overview']}"
        )

    return "\n\n".join(evidence)


# ==========================================
# BUILD CONVERSATION HISTORY
# ==========================================

def build_history(history=None, limit=2):

    if not history:
        return ""

    lines = []

    for turn in history[-limit:]:

        if not isinstance(turn, dict):
            continue

        question = clean_text(
            turn.get(
                "query",
                turn.get("question", ""),
            )
        )

        answer = clean_text(
            turn.get("answer", "")
        )

        if question:
            lines.append(
                f"User: {question[:300]}"
            )

        if answer:
            lines.append(
                f"Assistant: {answer[:500]}"
            )

    return "\n".join(lines)


# ==========================================
# BUILD RAG PROMPT
# ==========================================

def build_prompt(query, movies, history=None):

    evidence = build_evidence(movies)

    conversation = build_history(
        history,
        limit=MAX_HISTORY_TURNS,
    )

    count = min(
        MAX_RECOMMENDATIONS,
        len(movies),
    )

    return f"""
You are CineMind AI, a helpful movie recommendation assistant.

Recommend relevant movies using only the
retrieved evidence.

USER REQUEST:
{query}

RECENT CONVERSATION:
{conversation if conversation else "None"}

RETRIEVED MOVIE EVIDENCE:
{evidence}

STRICT INSTRUCTIONS:

1. Recommend at most {count} movies.

2. Use only movies listed in the retrieved evidence.

3. Every recommendation must include its
   correct source citation in square brackets.

4. Write the FULL official movie title
   exactly as it appears in the evidence.

5. Place the citation immediately after
   each movie title.

6. Do not renumber the original sources.

7. Give only one short explanation per movie.

8. Do not invent movies, characters,
   descriptions, genres or release dates.

9. Keep the response concise.

10. Do not recommend all retrieved movies.

RESPONSE FORMAT:

**Exact Movie Title** [source number] — Short explanation.

IMPORTANT:
Use actual movie titles and source numbers
from the retrieved evidence, not examples.

Generate the answer now.
""".strip()


# ==========================================
# VALIDATE CITATIONS
# ==========================================

def validate_citations(answer, movies):
    """
    Check that each recommendation has a
    valid source citation and official title.

    This is a structural validation, not a
    full factual verification of the text.
    """

    if not answer or not movies:
        return False

    valid_sources = {
        number: movie["title"]
        for number, movie in enumerate(
            movies,
            start=1,
        )
    }

    recommendation_lines = [
        line.strip()
        for line in answer.splitlines()
        if re.search(r"\[\d+\]", line)
    ]

    if not recommendation_lines:
        return False

    if (
        len(recommendation_lines)
        > MAX_RECOMMENDATIONS
    ):
        return False

    for line in recommendation_lines:

        citations = re.findall(
            r"\[(\d+)\]",
            line,
        )

        if len(citations) != 1:
            return False

        source_number = int(
            citations[0]
        )

        if source_number not in valid_sources:
            return False

        official_title = valid_sources[
            source_number
        ]

        # Check that the correct official title
        # appears before the citation.
        title_part = line.split(
            f"[{source_number}]",
            1,
        )[0]

        if normalize_title(
            title_part
        ) != normalize_title(
            official_title
        ):
            return False

    return True


# ==========================================
# NORMALIZE OLLAMA ANSWER
# ==========================================

def normalize_llm_answer(answer, movies):
    """
    Extract up to three valid recommendations.

    Accept exact titles and recognizable
    shortened prefixes, then restore the
    official retrieved titles.

    Skip invalid citations and unrelated
    movie titles.
    """

    if not answer or not movies:
        return ""

    valid_sources = {
        number: movie["title"]
        for number, movie in enumerate(
            movies,
            start=1,
        )
    }

    normalized_lines = []
    used_sources = set()

    for line in answer.splitlines():

        line = line.strip()

        if not line:
            continue

        # Source citation.
        citation_match = re.search(
            r"\[(\d+)\]",
            line,
        )

        if not citation_match:
            continue

        source_number = int(
            citation_match.group(1)
        )

        if source_number not in valid_sources:
            continue

        if source_number in used_sources:
            continue

        # Accept common Markdown explanations.
        if "—" in line:

            title_part, explanation = (
                line.split("—", 1)
            )

        elif " – " in line:

            title_part, explanation = (
                line.split(" – ", 1)
            )

        elif " - " in line:

            title_part, explanation = (
                line.split(" - ", 1)
            )

        else:
            continue

        # Ensure citation is beside the title.
        if not re.search(
            r"\[\d+\]",
            title_part,
        ):
            continue

        official_title = valid_sources[
            source_number
        ]

        generated_title = normalize_title(
            title_part
        )

        expected_title = normalize_title(
            official_title
        )

        # Allow an exact title or a meaningful
        # shortened prefix of the official title.
        title_matches = (
            generated_title == expected_title
            or (
                len(generated_title) >= 10
                and expected_title.startswith(
                    generated_title
                )
            )
        )

        if not title_matches:
            continue

        explanation = explanation.strip()

        if not explanation:
            continue

        # Do not keep explanations that mention
        # unrelated retrieved titles.
        # The source reference must correspond
        # to the movie written in the heading.

        normalized_lines.append(
            f"**{official_title}** "
            f"[{source_number}] — {explanation}"
        )

        used_sources.add(source_number)

        if (
            len(normalized_lines)
            >= MAX_RECOMMENDATIONS
        ):
            break

    return "\n\n".join(
        normalized_lines
    )


# ==========================================
# FALLBACK ANSWER
# ==========================================

def fallback_answer(movies):

    if not movies:
        return (
            "I couldn't find matching movies "
            "in the current catalog."
        )

    lines = [
        "Here are the closest movies "
        "from the available catalog:"
    ]

    for number, movie in enumerate(
        movies[:MAX_RECOMMENDATIONS],
        start=1,
    ):

        lines.append(
            f"\n**{movie['title']}** "
            f"[{number}] — {movie['genres']}"
        )

    return "\n".join(lines)


# ==========================================
# MAIN RAG ASSISTANT
# ==========================================

class RAGMovieAssistant:

    def __init__(
        self,
        model=DEFAULT_MODEL,
        ollama_url=OLLAMA_URL,
    ):

        self.model = model
        self.ollama_url = ollama_url

    # --------------------------------------
    # GENERATE RESPONSE
    # --------------------------------------

    def generate(
        self,
        query,
        results,
        history=None,
    ):

        movies = prepare_movies(results)

        if not movies:

            return (
                "I couldn't find matching movies "
                "in the current catalog.",
                False,
                "No retrieved movies available.",
            )

        prompt = build_prompt(
            query=query,
            movies=movies,
            history=history,
        )

        try:

            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 450,
                    },
                },
                timeout=(5, 180),
            )

            response.raise_for_status()

            data = response.json()

            raw_answer = data.get(
                "response",
                "",
            ).strip()

            print(
                "\n===== RAW OLLAMA ANSWER ====="
            )

            print(raw_answer)

            print(
                "============================\n"
            )

            if not raw_answer:

                return (
                    fallback_answer(movies),
                    False,
                    "Ollama returned an empty answer.",
                )

            # Normalize citations and titles.
            final_answer = normalize_llm_answer(
                raw_answer,
                movies,
            )

            # Validate the normalized answer.
            is_valid = validate_citations(
                final_answer,
                movies,
            )

            if is_valid:

                return (
                    final_answer,
                    True,
                    None,
                )

            # Safe fallback if LLM response
            # does not satisfy source rules.
            return (
                fallback_answer(movies),
                False,
                "Generated answer has missing "
                "or invalid source references.",
            )

        except requests.exceptions.ConnectionError:

            return (
                fallback_answer(movies),
                False,
                "Cannot connect to Ollama. "
                "Make sure Ollama is running.",
            )

        except requests.exceptions.Timeout:

            return (
                fallback_answer(movies),
                False,
                "Ollama generation timed out.",
            )

        except requests.exceptions.RequestException as error:

            return (
                fallback_answer(movies),
                False,
                f"Ollama request error: {error}",
            )

        except (
            ValueError,
            KeyError,
            TypeError,
        ) as error:

            return (
                fallback_answer(movies),
                False,
                f"Invalid Ollama response: {error}",
            )
