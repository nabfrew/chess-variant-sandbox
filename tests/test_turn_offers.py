"""The turn choice asks the search (R-L3's B4, `docs/reference/refusal-offers.md` §6.4 and §9).

`designing.take_turns` used to mend a refused turn choice with `_remedy`, regular expressions
over the validator's sentences for one rule and six partners. Now the changes are the nearest
draft the validator allows that still states the way's rule (`instead.nearest`), and leaving
rounds clears the piece keys only rounds read, by key. These hold B4's row: no sentence is read
on the way, chess in rounds is the pinned offer, and every committed ruleset's turn choices give
a valid draft wherever they gave one before.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.spec import instead, validate_spec
from chess_variant_sandbox.web import designing

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"
CHESS = json.loads((RULESETS / "chess.json").read_text(encoding="utf-8"))

# Turn choices on committed rulesets that no draft near them allows, each for a reason no
# leaving-out mends (`_remedy` refused every one of them too). Pinned so that a lift, or a
# search that newly finds a way, is seen: the list may only shrink.
REFUSED = {
    ("dohyo", "rounds"),
    ("hunger", "rounds"),
    ("hunger", "several"),
    ("leapfrog", "rounds"),
    ("tessabi", "rounds"),
}
# The dearest searches, each over half a second alone (the shogis in rounds: three levels of
# a 2.5 ms validator); measured 2026-11-20, `refusal-offers.md` §6.4's *As built*.
SLOW = {"cannon-shogi", "shogi", "minishogi"}


def _problems(spec: dict[str, Any]) -> list[str]:
    return validate_spec(designing.settled(spec))


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    designing._turn_offer_kept.cache_clear()
    yield
    designing._turn_offer_kept.cache_clear()


def test_no_sentence_is_read_on_the_way() -> None:
    """No regular expression in `take_turns`' path, and `_remedy` is gone. `plain_problems`
    words a refusal as the check does, and decides nothing."""
    assert not hasattr(designing, "_remedy")
    tree = ast.parse(inspect.getsource(designing))
    path = {"take_turns", "_round_cleanups", "_turn_offer_kept"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in path:
            found.add(node.name)
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert "re" not in names, node.name
            assert "_spec_plain_problem" not in names, node.name
    assert found == path


def test_chess_in_rounds_wins_by_capture_and_drops_en_passant() -> None:
    answer = designing.take_turns(CHESS, "rounds")
    made = answer["draft"]
    assert answer["errors"] == []
    assert made["rules"]["win"] == "royal_capture"
    assert "en_passant" not in made["rules"]
    assert isinstance(made["rules"]["simultaneous"], dict)
    assert _problems(made) == []


def test_the_answer_does_not_depend_on_a_sentences_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """§2 rule 3: with every sentence replaced by a token of its identity, the same draft."""
    plain = designing.take_turns(CHESS, "rounds")
    designing._turn_offer_kept.cache_clear()
    real = validate_spec

    def tokens(spec: Any) -> list[str]:
        return [hashlib.sha256(line.encode()).hexdigest()[:12] for line in real(spec)]

    monkeypatch.setattr(designing, "validate_spec", tokens)
    monkeypatch.setattr(instead, "validate_spec", tokens)
    hashed = designing.take_turns(CHESS, "rounds")
    assert hashed["draft"] == plain["draft"]
    assert hashed["changes"] == plain["changes"]


@pytest.mark.parametrize("key", ["speed", "spy", "beats"])
def test_what_leaving_rounds_clears_is_refused_outside_rounds(key: str) -> None:
    """The premise of `_round_cleanups`: each key it clears is refused without rounds."""
    value = {"speed": 1, "spy": True, "beats": ["P"]}[key]
    rounds = designing.take_turns(CHESS, "rounds")["draft"]
    rounds["pieces"][1][key] = value
    alone = copy.deepcopy(rounds)
    del alone["rules"]["simultaneous"]
    assert _problems(alone), key
    back = designing.take_turns(rounds, "alternate")
    if back["draft"] is not None:
        assert key not in back["draft"]["pieces"][1]
        assert _problems(back["draft"]) == []


def test_an_answer_asked_twice_is_kept() -> None:
    first = designing.take_turns(CHESS, "rounds")
    first["draft"]["rules"]["win"] = "changed by the caller"
    again = designing.take_turns(CHESS, "rounds")
    assert again["draft"]["rules"]["win"] == "royal_capture"
    assert designing._turn_offer_kept.cache_info().hits == 1


def test_a_draft_already_refused_is_not_searched() -> None:
    """The search would mend what the designer is still writing: refused, as before."""
    broken = copy.deepcopy(CHESS)
    broken["pieces"][1]["speed"] = 1  # refused outside rounds
    assert _problems(broken)
    answer = designing.take_turns(broken, "several")
    assert answer["draft"] is not None  # a long turn adds no problem of its own
    sweeping = copy.deepcopy(CHESS)
    sweeping["pieces"][2]["moves"][0]["kind"] = "sweep"
    sweeping["pieces"][1]["speed"] = 1
    stuck = designing.take_turns(sweeping, "rounds")
    assert stuck["draft"] is None and stuck["errors"]


def _rulesets() -> list[Any]:
    out = []
    for path in sorted(RULESETS.glob("*.json")):
        marks = [pytest.mark.slow] if path.stem in SLOW else []
        out.append(pytest.param(path, id=path.stem, marks=marks))
    return out


@pytest.mark.parametrize("path", _rulesets())
def test_every_turn_choice_of_a_committed_ruleset_gives_a_valid_draft(path: Path) -> None:
    """B4's row: every turn choice that gave a draft before still gives a valid one. Before
    B4, 139 of the 153 did; now 148, the five of `REFUSED` aside."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    for way in designing.TURN_WAYS:
        answer = designing.take_turns(spec, way)
        if (path.stem, way) in REFUSED:
            assert answer["draft"] is None and answer["errors"], (path.stem, way)
            continue
        assert answer["errors"] == [], (path.stem, way, answer["errors"])
        made = answer["draft"]
        assert _problems(made) == [], (path.stem, way)
        rules = made.get("rules") or {}
        assert ("simultaneous" in rules) == (way == "rounds"), (path.stem, way)
        assert ("turn" in rules) == (way == "several"), (path.stem, way)
