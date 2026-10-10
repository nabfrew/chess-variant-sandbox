"""The database guard: URL set and driver missing fails the run; URL unset skips quietly."""

from __future__ import annotations

import builtins
import sys
from typing import Any

import pytest
import site_store


def _without_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "psycopg", raising=False)
    real = builtins.__import__

    def fake(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "psycopg":
            raise ImportError("simulated")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)


def test_url_set_and_driver_missing_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(site_store.URL_ENV, "postgresql://nowhere/x")
    _without_driver(monkeypatch)
    with pytest.raises(pytest.UsageError, match=r"uv sync --extra dev --extra site"):
        site_store.check_driver()


def test_url_unset_is_quiet_even_without_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(site_store.URL_ENV, raising=False)
    _without_driver(monkeypatch)
    site_store.check_driver()


def test_url_set_and_driver_present_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(site_store.URL_ENV, "postgresql://nowhere/x")
    fake = type(sys)("psycopg")
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    site_store.check_driver()
