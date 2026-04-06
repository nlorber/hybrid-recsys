"""LLM-based re-ranking with prompt generation and graceful fallback."""

import ast
import logging

from hybrid_recsys.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATES: dict[str, str] = {
    "fr": (
        "Voici une liste de programmes disponibles en podcast, "
        "ainsi que leurs descriptions.\n"
        "Sélectionne exactement {size} programme(s) dans cette liste "
        "pouvant correspondre à la requête suivante d'un utilisateur: {query}\n\n"
        "Ta réponse devra absolument et exclusivement prendre la forme "
        "d'une liste de strings comportant les `program_id` des programmes "
        "sélectionnés, classés par pertinence décroissante.\n"
        "Par exemple : ['program_id1', 'program_id2', 'program_id3'].\n\n"
        "Voici les données contextuelles disponibles :\n{context}"
    ),
    "en": (
        "Here is a list of available podcast programs, "
        "along with their descriptions.\n"
        "Select exactly {size} program(s) from this list that may match "
        "the following user request: {query}\n\n"
        "Your response must strictly and exclusively take the form of "
        "a list of strings containing the `program_id` of the selected programs, "
        "sorted by decreasing relevance.\n"
        "For example: ['program_id1', 'program_id2', 'program_id3'].\n\n"
        "Here are the available contextual data:\n{context}"
    ),
    "de": (
        "Hier ist eine Liste der verfügbaren Podcast-Programme "
        "zusammen mit ihren Beschreibungen.\n"
        "Wählen Sie genau {size} Programm(e) aus dieser Liste, "
        "die zur folgenden Benutzeranfrage passen könnten: {query}\n\n"
        "Ihre Antwort muss strikt und ausschließlich in Form "
        "einer Liste von Zeichenfolgen erfolgen, die die `program_id` "
        "der ausgewählten Programme enthalten, "
        "sortiert nach abnehmender Relevanz.\n"
        "Zum Beispiel: ['program_id1', 'program_id2', 'program_id3'].\n\n"
        "Hier sind die verfügbaren Kontextdaten:\n{context}"
    ),
}


def build_rerank_prompt(
    query: str,
    candidates: list[dict[str, str]],
    size: int,
    lang: str,
) -> str:
    """Build a re-ranking prompt with program descriptions.

    Args:
        query: User's search query.
        candidates: List of dicts with 'program_id' and 'description'.
        size: Number of programs to select.
        lang: Language code for prompt template.

    Returns:
        Formatted prompt string.
    """
    context_lines = []
    for c in candidates:
        context_lines.append(f"program_id : {c['program_id']}")
        context_lines.append(f"Description : {c['description']}")
        context_lines.append("----------")
    context = "\n".join(context_lines)

    template = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES["en"])
    return template.format(query=query, size=size, context=context)


def parse_rerank_response(response: str) -> list[str]:
    """Parse an LLM response as a Python list of program IDs.

    Args:
        response: Raw LLM response text.

    Returns:
        List of program_id strings, or empty list on parse failure.
    """
    try:
        result = ast.literal_eval(response)
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            return result
    except (SyntaxError, ValueError):
        logger.warning("Failed to parse LLM rerank response: %s", response[:200])
    return []


def rerank_programs(
    llm: LLMProvider,
    query: str,
    rrf_ranking: list[str],
    descriptions: dict[str, str],
    size: int,
    lang: str,
) -> list[str]:
    """Re-rank programs via LLM with fallback to RRF ranking.

    Only invokes the LLM if RRF produced more candidates than requested.
    On LLM failure, falls back to RRF order.

    Args:
        llm: LLM provider for re-ranking.
        query: User's search query.
        rrf_ranking: Program IDs ranked by RRF.
        descriptions: Mapping of program_id to description.
        size: Number of results to return.
        lang: Language code.

    Returns:
        List of program_ids of length <= size.
    """
    if len(rrf_ranking) <= size:
        logger.info(
            "RRF produced %d results <= %d, skipping LLM", len(rrf_ranking), size
        )
        return rrf_ranking

    candidates = [
        {"program_id": pid, "description": descriptions.get(pid, "")}
        for pid in rrf_ranking
    ]

    try:
        result = llm.rerank(query, candidates, size, lang)
    except Exception:
        logger.exception("LLM reranking failed, falling back to RRF")
        return rrf_ranking[:size]

    # Pad if LLM underselected
    if len(result) < size:
        logger.warning("LLM returned %d < %d, padding from RRF", len(result), size)
        seen = set(result)
        for pid in rrf_ranking:
            if pid not in seen:
                result.append(pid)
                seen.add(pid)
            if len(result) >= size:
                break

    return result[:size]
