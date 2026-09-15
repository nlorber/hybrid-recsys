"""LLM-based re-ranking with prompt generation and graceful fallback."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

from hybrid_recsys.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Output schema the LLM providers enforce via structured output, read by parse_rerank_response.
RERANK_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"program_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["program_ids"],
    "additionalProperties": False,
}

PROMPT_TEMPLATES: dict[str, str] = {
    "fr": (
        "Voici une liste de programmes disponibles en podcast, "
        "ainsi que leurs descriptions.\n"
        "Sélectionne exactement {size} programme(s) dans cette liste "
        "pouvant correspondre à la requête suivante d'un utilisateur: {query}\n\n"
        "Renvoie les `program_id` des programmes sélectionnés dans le champ "
        "`program_ids`, classés par pertinence décroissante.\n\n"
        "Voici les données contextuelles disponibles :\n{context}"
    ),
    "en": (
        "Here is a list of available podcast programs, "
        "along with their descriptions.\n"
        "Select exactly {size} program(s) from this list that may match "
        "the following user request: {query}\n\n"
        "Return the `program_id` of the selected programs in the `program_ids` field, "
        "sorted by decreasing relevance.\n\n"
        "Here are the available contextual data:\n{context}"
    ),
    "de": (
        "Hier ist eine Liste der verfügbaren Podcast-Programme "
        "zusammen mit ihren Beschreibungen.\n"
        "Wählen Sie genau {size} Programm(e) aus dieser Liste, "
        "die zur folgenden Benutzeranfrage passen könnten: {query}\n\n"
        "Geben Sie die `program_id` der ausgewählten Programme im Feld "
        "`program_ids` zurück, sortiert nach abnehmender Relevanz.\n\n"
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
    """Parse an LLM response matching ``RERANK_RESPONSE_SCHEMA``.

    Args:
        response: Raw LLM response text, a JSON object with a ``program_ids`` array.

    Returns:
        List of program_id strings, or empty list if the response does not match the schema.
    """
    try:
        program_ids = json.loads(response).get("program_ids")
    except (ValueError, AttributeError):
        program_ids = None
    if isinstance(program_ids, list) and all(isinstance(x, str) for x in program_ids):
        return program_ids
    logger.warning("Failed to parse LLM rerank response: %s", response[:200])
    return []


def rerank_programs(
    llm: LLMProvider,
    query: str,
    rrf_ranking: list[str],
    descriptions: dict[str, str],
    size: int,
    lang: str,
    timeout: float = 5.0,
) -> list[str]:
    """Re-rank programs via LLM with fallback to RRF ranking.

    Only invokes the LLM if RRF produced more candidates than requested.
    On LLM failure or timeout, falls back to RRF order.

    Args:
        llm: LLM provider for re-ranking.
        query: User's search query.
        rrf_ranking: Program IDs ranked by RRF.
        descriptions: Mapping of program_id to description.
        size: Number of results to return.
        lang: Language code.
        timeout: Maximum seconds to wait for the LLM response (RECSYS_LLM_RERANK_TIMEOUT).

    Returns:
        List of program_ids of length <= size.
    """
    if len(rrf_ranking) <= size:
        logger.info("RRF produced %d results <= %d, skipping LLM", len(rrf_ranking), size)
        return rrf_ranking

    candidates = [
        {"program_id": pid, "description": descriptions.get(pid, "")} for pid in rrf_ranking
    ]

    # Not a context manager: its exit joins the worker, so a hung provider call would hold
    # the request past the timeout. The abandoned worker finishes in the background.
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        result = executor.submit(llm.rerank, query, candidates, size, lang).result(timeout=timeout)
    except FuturesTimeoutError:
        logger.warning("LLM reranking timed out after %.1fs, falling back to RRF", timeout)
        return rrf_ranking[:size]
    except Exception:
        # Re-ranking is a best-effort optimization over the RRF ranking: any provider
        # failure (network/API errors, bad responses, etc.) must degrade gracefully
        # rather than fail the request.
        logger.exception("LLM reranking failed, falling back to RRF")
        return rrf_ranking[:size]
    finally:
        executor.shutdown(wait=False)

    # Drop IDs the LLM may have hallucinated outside the candidate set
    candidate_set = set(rrf_ranking)
    hallucinated = [pid for pid in result if pid not in candidate_set]
    if hallucinated:
        logger.warning(
            "LLM returned %d ID(s) not in candidates: %r",
            len(hallucinated),
            hallucinated[:5],
        )
    result = [pid for pid in result if pid in candidate_set]

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
