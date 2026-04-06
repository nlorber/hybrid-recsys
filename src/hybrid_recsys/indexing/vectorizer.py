"""Orchestrates offline index building: embed, vectorize, build ANN indexes, save."""

import logging
from typing import Any

from hybrid_recsys.config import Settings
from hybrid_recsys.indexing.store import IndexStore, LanguageIndex
from hybrid_recsys.indexing.tfidf import TfidfPipeline
from hybrid_recsys.models import CatalogItem
from hybrid_recsys.providers.embeddings.base import EmbeddingProvider
from hybrid_recsys.providers.nlp.spacy import SpacyNLP
from hybrid_recsys.retrieval.ann_search import build_annoy_index

logger = logging.getLogger(__name__)


class Vectorizer:
    """Builds per-language indexes from a catalog.

    For each language: generates embeddings, fits TF-IDF, builds Annoy indexes,
    and saves all artifacts to disk.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        nlp: SpacyNLP,
        settings: Settings,
    ) -> None:
        self._embedder = embedding_provider
        self._tfidf = TfidfPipeline(nlp)
        self._settings = settings
        self._store = IndexStore()

    def build(self, catalog: list[CatalogItem]) -> None:
        """Build and save indexes for all languages in the catalog.

        Args:
            catalog: Full catalog of programs with media.
        """
        by_lang: dict[str, list[CatalogItem]] = {}
        for item in catalog:
            by_lang.setdefault(item.lang, []).append(item)

        for lang, items in by_lang.items():
            logger.info("Building index for '%s' (%d programs)", lang, len(items))
            self._build_language(lang, items)
            logger.info("Index for '%s' saved", lang)

    def _build_language(self, lang: str, items: list[CatalogItem]) -> None:
        """Build and save index for a single language."""
        program_ids = [item.program_id for item in items]
        descriptions = [item.description for item in items]

        # Generate embeddings
        logger.info("Generating embeddings for %d programs", len(items))
        embedding_vectors = self._embedder.embed_batch(descriptions)

        # Fit TF-IDF
        logger.info("Fitting TF-IDF for %d programs", len(items))
        tfidf_vectorizer, tfidf_vectors = self._tfidf.fit_transform(
            program_ids, descriptions, lang
        )
        tfidf_vector_list = [tfidf_vectors[pid] for pid in program_ids]

        # Build Annoy indexes
        ann_embedding = build_annoy_index(
            embedding_vectors,
            metric=self._settings.ann_metric,
            n_trees=self._settings.ann_n_trees,
        )
        ann_tfidf = build_annoy_index(
            tfidf_vector_list,
            metric=self._settings.ann_metric,
            n_trees=self._settings.ann_n_trees,
        )

        # Prepare metadata
        program_descriptions = dict(zip(program_ids, descriptions, strict=True))
        media_data: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            media_data[item.program_id] = [m.model_dump() for m in item.media]

        # Save
        index = LanguageIndex(
            program_ids=program_ids,
            program_descriptions=program_descriptions,
            media_data=media_data,
            ann_embedding=ann_embedding,
            ann_tfidf=ann_tfidf,
            tfidf_vectorizer=tfidf_vectorizer,
            embedding_dim=len(embedding_vectors[0]),
            tfidf_dim=len(tfidf_vector_list[0]),
        )
        self._store.save(lang, index, self._settings.index_dir)
