"""`refresh --no-slow-tier` reaches the `Refresher`; its absence keeps the default (Q828)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import cli, refresher


class _Machine:
    def __init__(self, **_: Any) -> None: ...
    def exists(self) -> bool:
        return True


def _built(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    class _Loop:
        def __init__(self, machine: Any, ledger: Any, **kwargs: Any) -> None:
            seen.update(kwargs)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(refresher, "GitMachine", _Machine)
    monkeypatch.setattr(refresher, "Refresher", _Loop)
    monkeypatch.setattr(refresher, "main_checkout_of", lambda _: tmp_path)
    return seen


@pytest.mark.parametrize(("extra", "expected"), [([], None), (["--no-slow-tier"], False)])
def test_slow_tier_switch_reaches_refresher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: list[str], expected: bool | None
) -> None:
    seen = _built(monkeypatch, tmp_path)
    runs = tmp_path / "runs"
    args = ["refresh", "--library", "--watch", "--runs", str(runs),
            "--worktree", str(tmp_path / "wt"), *extra]
    assert cli.main(args) == 0
    assert seen["slow_tier"] is expected
