"""`tests/corpus_pins.py`, the command that writes a new game's freeze rows (Q1154).

It may add a row for a game no freeze has and drop a generated row whose game is gone. It may
never change a row that exists: a moved digest is a failing freeze to argue, not a snapshot
to refresh (`docs/reference/corpus-pins.md` §2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import corpus_pins
import pytest
import test_hex_readers
import test_hex_resolver


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "corpus_pins.json"
    monkeypatch.setattr(corpus_pins, "SNAPSHOT", target)
    return target


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> dict[str, dict[str, list[str]]]:
    monkeypatch.setattr(sys, "argv", ["corpus_pins.py", *args])
    return corpus_pins.sync()


def test_a_new_game_gets_its_row_and_a_gone_one_loses_it(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed = _committed()
    # Pretend one hand-frozen game is new: take it out of the hand table for this test.
    path = test_hex_readers.LIBRARY[0]
    key = test_hex_readers._key(path)
    hand = dict(test_hex_readers._DESCRIPTIONS_BEFORE_THE_HEXAGON)
    expected = hand.pop(key)
    monkeypatch.setattr(test_hex_readers, "_DESCRIPTIONS_BEFORE_THE_HEXAGON", hand)
    stale = {**committed["descriptions"], "archive/no-such-game": "0" * 16}
    snapshot.write_text(
        json.dumps({"descriptions": stale, "features": committed["features"]}), encoding="utf-8"
    )

    assert _run(monkeypatch, "--check")["descriptions"] == {
        "add": [key], "drop": ["archive/no-such-game"]
    }
    assert json.loads(snapshot.read_text(encoding="utf-8"))["descriptions"] == stale

    _run(monkeypatch)
    written = json.loads(snapshot.read_text(encoding="utf-8"))["descriptions"]
    assert written[key] == expected, "the generated row is the digest the freeze would hold"
    assert "archive/no-such-game" not in written


def test_an_existing_row_is_never_rewritten(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed = _committed()
    path = test_hex_resolver.LIBRARY[0]
    key = test_hex_resolver._key(path)
    hand = dict(test_hex_resolver._FEATURES_BEFORE_THE_RESOLVER)
    hand.pop(key)
    monkeypatch.setattr(test_hex_resolver, "_FEATURES_BEFORE_THE_RESOLVER", hand)
    wrong = {**committed["features"], key: "f" * 16}
    snapshot.write_text(
        json.dumps({"descriptions": committed["descriptions"], "features": wrong}),
        encoding="utf-8",
    )
    assert _run(monkeypatch)["features"] == {"add": [], "drop": []}
    assert corpus_pins.load()["features"][key] == "f" * 16


def _committed() -> dict[str, dict[str, str]]:
    committed = json.loads(
        (Path(__file__).resolve().parent / "corpus_pins.json").read_text(encoding="utf-8")
    )
    return {name: dict(committed[name]) for name in corpus_pins.FREEZES}
