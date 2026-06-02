"""Tests for Settings configuration parsing."""

import pytest

from hybrid_recsys.config import Settings


class TestSettings:
    def test_default_values(self) -> None:
        # _env_file=None isolates the test from any local .env so it asserts code defaults.
        settings = Settings(_env_file=None)
        assert settings.embedding_provider == "sentence-transformers"
        assert settings.llm_provider == "mock"
        assert settings.default_duration == 600

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RECSYS_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("RECSYS_DEFAULT_DURATION", "1200")
        settings = Settings()
        assert settings.embedding_provider == "openai"
        assert settings.default_duration == 1200

    def test_blank_optional_env_vars_coerced_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blank env vars (e.g. RECSYS_LLM_BASE_URL=) must become None, not '',
        so they don't break provider SDK clients."""
        monkeypatch.setenv("RECSYS_LLM_BASE_URL", "")
        monkeypatch.setenv("RECSYS_LLM_API_KEY", "")
        monkeypatch.setenv("RECSYS_EMBEDDING_BASE_URL", "  ")
        settings = Settings(_env_file=None)
        assert settings.llm_base_url is None
        assert settings.llm_api_key is None
        assert settings.embedding_base_url is None

    def test_index_dir_property(self) -> None:
        settings = Settings()
        assert settings.index_dir == settings.data_dir / "index"

    def test_catalog_path_property(self) -> None:
        settings = Settings()
        assert settings.catalog_path == settings.data_dir / "catalog.json"
