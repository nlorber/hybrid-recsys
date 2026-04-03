"""Main recommendation pipeline: orchestrates dual retrieval, fusion, and re-ranking."""

import logging

from hybrid_recsys.config import Settings
from hybrid_recsys.indexing.store import IndexStore, LanguageIndex
from hybrid_recsys.indexing.tfidf import TfidfPipeline
from hybrid_recsys.models import RecoRequest, RecoResponse
from hybrid_recsys.providers.embeddings.base import EmbeddingProvider
from hybrid_recsys.providers.llm.base import LLMProvider
from hybrid_recsys.providers.nlp.spacy import SpacyNLP
from hybrid_recsys.retrieval.ann_search import query_annoy_index
from hybrid_recsys.retrieval.fusion import reciprocal_rank_fusion
from hybrid_recsys.retrieval.reranker import rerank_programs
from hybrid_recsys.retrieval.scorer import duration_score

logger = logging.getLogger(__name__)


class RecommendationPipeline:
    """Full recommendation pipeline: query -> ranked programs + media.

    Loads pre-built per-language indexes, then for each request:
    1. Embed query + TF-IDF vectorize query
    2. ANN search both indexes
    3. RRF fuse program rankings
    4. LLM rerank (if needed)
    5. Select earliest episode per ranked program
    6. Score media by duration
    7. RRF fuse media rankings
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        settings: Settings,
    ) -> None:
        self._embedder = embedding_provider
        self._llm = llm_provider
        self._settings = settings
        self._nlp = SpacyNLP()
        self._tfidf = TfidfPipeline(self._nlp)
        self._store = IndexStore()
        self._index_cache: dict[str, LanguageIndex] = {}

    def _load_index(self, lang: str) -> LanguageIndex:
        """Load and cache a language index."""
        if lang not in self._index_cache:
            self._index_cache[lang] = self._store.load(lang, self._settings.index_dir)
        return self._index_cache[lang]

    def recommend(self, request: RecoRequest) -> RecoResponse:
        """Run the full recommendation pipeline.

        Args:
            request: Recommendation request with query, language, size,
                optional duration.

        Returns:
            RecoResponse with ranked program and media IDs.
        """
        index = self._load_index(request.lang)

        # Step 1: Embed and vectorize query
        query_embedding = self._embedder.embed(request.query)
        query_tfidf = self._tfidf.transform_query(
            request.query, index.tfidf_vectorizer, request.lang
        )

        # Step 2: ANN search both indexes
        k = min(self._settings.ann_query_k, len(index.program_ids))
        emb_indices = query_annoy_index(index.ann_embedding, query_embedding, k)
        tfidf_indices = query_annoy_index(index.ann_tfidf, query_tfidf, k)

        emb_programs = [index.program_ids[i] for i in emb_indices]
        tfidf_programs = [index.program_ids[i] for i in tfidf_indices]

        # Step 3: RRF fuse program rankings
        program_rrf = reciprocal_rank_fusion(
            ranked_lists=[emb_programs, tfidf_programs],
            weights=self._settings.rrf_program_weights,
            k=self._settings.rrf_program_k,
        )

        # Step 4: LLM rerank
        programs = rerank_programs(
            llm=self._llm,
            query=request.query,
            rrf_ranking=program_rrf,
            descriptions=index.program_descriptions,
            size=request.size,
            lang=request.lang,
        )

        # Step 5-7: Media ranking
        requested_duration = request.duration or self._settings.default_duration
        medias = self._rank_media(
            index=index,
            emb_programs=emb_programs,
            tfidf_programs=tfidf_programs,
            requested_duration=requested_duration,
            size=request.size,
        )

        return RecoResponse(programs=programs, medias=medias)

    def _rank_media(
        self,
        index: LanguageIndex,
        emb_programs: list[str],
        tfidf_programs: list[str],
        requested_duration: int,
        size: int,
    ) -> list[str]:
        """Rank media items using embedding, TF-IDF, and duration signals.

        For each program, selects the earliest episode. Builds three ranked lists
        and fuses them via RRF.
        """
        # Get earliest episode per program
        earliest: dict[str, dict] = {}
        all_program_ids = set(emb_programs) | set(tfidf_programs)
        for pid in all_program_ids:
            media_list = index.media_data.get(pid, [])
            if not media_list:
                continue
            first_ep = min(media_list, key=lambda m: m["episode"])
            earliest[pid] = first_ep

        # Media from embedding-ranked programs (ordered by program rank)
        emb_media = [
            earliest[pid]["media_id"] for pid in emb_programs if pid in earliest
        ][: size * 3]

        # Media from TF-IDF-ranked programs (ordered by program rank)
        tfidf_media = [
            earliest[pid]["media_id"] for pid in tfidf_programs if pid in earliest
        ][: size * 3]

        # Media scored by duration proximity
        duration_scored = []
        for pid in all_program_ids:
            if pid not in earliest:
                continue
            ep = earliest[pid]
            delta = requested_duration - ep["duration"]
            score = duration_score(delta, penalty=self._settings.duration_penalty)
            duration_scored.append((ep["media_id"], score))
        duration_scored.sort(key=lambda x: x[1], reverse=True)
        duration_media = [mid for mid, _ in duration_scored][: size * 3]

        # RRF fuse media lists
        return reciprocal_rank_fusion(
            ranked_lists=[emb_media, tfidf_media, duration_media],
            weights=self._settings.rrf_media_weights,
            k=self._settings.rrf_media_k,
            output_size=size,
        )
