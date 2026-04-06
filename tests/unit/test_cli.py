"""Smoke tests for CLI commands."""

import pytest
from typer.testing import CliRunner

from hybrid_recsys.cli import app

runner = CliRunner()


class TestCLI:
    def test_app_with_no_command_exits_cleanly(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0

    def test_index_without_catalog_exits_with_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Point data_dir to an empty temp directory so catalog.json is absent
        monkeypatch.setenv("RECSYS_DATA_DIR", str(tmp_path))
        result = runner.invoke(app, ["index"])
        assert result.exit_code == 1
        assert "Catalog not found" in result.output

    def test_demo_help(self) -> None:
        result = runner.invoke(app, ["demo", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower()

    def test_serve_help(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0

    def test_index_help(self) -> None:
        result = runner.invoke(app, ["index", "--help"])
        assert result.exit_code == 0
