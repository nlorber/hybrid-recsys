"""Tests for Settings configuration parsing."""

import pytest

from hybrid_recsys.config import Settings


class TestSettings:
    def test_default_values(self) -> None:
        settings = Settings()
        assert settings.embedding_provider == "sentence-transformers"
        assert settings.llm_provider == "mock"
        assert settings.default_duration == 600

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RECSYS_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("RECSYS_DEFAULT_DURATION", "1200")
        settings = Settings()
        assert settings.embedding_provider == "openai"
        assert settings.default_duration == 1200

    def test_index_dir_property(self) -> None:
        settings = Settings()
        assert settings.index_dir == settings.data_dir / "index"

    def test_catalog_path_property(self) -> None:
        settings = Settings()
        assert settings.catalog_path == settings.data_dir / "catalog.json"
