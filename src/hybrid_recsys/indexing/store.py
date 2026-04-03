"""Persistence for per-language index artifacts."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class LanguageIndex:
    """All artifacts needed to serve recommendations for one language."""

    program_ids: list[str]
    program_descriptions: dict[str, str]
    media_data: dict[str, list[dict[str, Any]]]
    ann_embedding: AnnoyIndex
    ann_tfidf: AnnoyIndex
    tfidf_vectorizer: TfidfVectorizer
    embedding_dim: int
    tfidf_dim: int


class IndexStore:
    """Save and load per-language index artifacts to/from disk."""

    def save(self, lang: str, index: LanguageIndex, base_dir: Path) -> None:
        """Save all index artifacts for a language.

        Args:
            lang: Language code (fr/en/de).
            index: Language index artifacts to save.
            base_dir: Base directory for index storage (e.g., data/index).
        """
        lang_dir = base_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        index.ann_embedding.save(str(lang_dir / "ann_embedding.ann"))
        index.ann_tfidf.save(str(lang_dir / "ann_tfidf.ann"))

        with open(lang_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(index.tfidf_vectorizer, f)

        metadata = {
            "program_ids": index.program_ids,
            "program_descriptions": index.program_descriptions,
            "media_data": index.media_data,
            "embedding_dim": index.embedding_dim,
            "tfidf_dim": index.tfidf_dim,
        }
        with open(lang_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load(self, lang: str, base_dir: Path) -> LanguageIndex:
        """Load all index artifacts for a language.

        Args:
            lang: Language code (fr/en/de).
            base_dir: Base directory for index storage.

        Returns:
            LanguageIndex with all artifacts loaded.

        Raises:
            FileNotFoundError: If index directory doesn't exist.
        """
        lang_dir = base_dir / lang
        if not lang_dir.exists():
            msg = f"No index found for language '{lang}' at {lang_dir}"
            raise FileNotFoundError(msg)

        with open(lang_dir / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)

        ann_embedding = AnnoyIndex(metadata["embedding_dim"], "angular")
        ann_embedding.load(str(lang_dir / "ann_embedding.ann"))

        ann_tfidf = AnnoyIndex(metadata["tfidf_dim"], "angular")
        ann_tfidf.load(str(lang_dir / "ann_tfidf.ann"))

        with open(lang_dir / "tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)  # noqa: S301

        return LanguageIndex(
            program_ids=metadata["program_ids"],
            program_descriptions=metadata["program_descriptions"],
            media_data=metadata["media_data"],
            ann_embedding=ann_embedding,
            ann_tfidf=ann_tfidf,
            tfidf_vectorizer=tfidf_vectorizer,
            embedding_dim=metadata["embedding_dim"],
            tfidf_dim=metadata["tfidf_dim"],
        )
