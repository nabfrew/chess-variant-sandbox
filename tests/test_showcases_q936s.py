"""The showcase of ``"per_type": "own"`` (Q936s, `docs/reference/extinction-any-kind.md` §5).

`archive/riders-and-leapers` gives two armies of different kinds, riders (rook, bishop, guard)
against leapers (champion, knight, elephant), and loses a side when any one of its own kinds
is gone. Held here, every answer the core's through `Engine`:

1. The compiler resolves each side's own kinds from the opening board, and nothing else.
2. A real line (a game the roster played in design, 6400 against 25600 nodes, seed 3) ends at
   ply 28 when an elephant takes the riders' last bishop: `"own"` ends it there, with the
   riders still holding both rooks and both guards, and the same file under the summed rule
   (`per_type` false, the default) plays on from the same position.
3. `per_type: true`, the one shared list, cannot say this game: the validator refuses it,
   since each side starts with none of the other's kinds.

The reverse (the summed rule ending a game `"own"` would not) cannot happen here: a side whose
own kinds are all gone has run out of each of them, and no capture here removes the mover's
own piece, so `"own"` always ends the game first.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "archive" / "riders-and-leapers.json"

LINE = (
    "Pf2-f3", "Ca6-c4", "Pd2-d3", "Cc4-a6", "Pc2-c3", "Pb5-b4", "Pa2-a3", "Pb4xc3",
    "Pb2xc3", "Ec6-b5", "Pa3-a4", "Nb6xa4", "Pc3-c4", "Pd5xc4", "Pd3xc4", "Ca6xc4",
    "Bd1-b3", "Cc4-c3", "Bb3xe6", "Na4-b6", "Be6-a2", "Eb5-c4", "Bc1-d2", "Ed6-b4",
    "Bd2xc3", "Eb4xc3", "Ge1-d2", "Ec4xa2",
)
"""Seed 3 of `play` on the design's draft (`30c055a056ec`, minimax 6400 against 25600 nodes)."""


def _spec(per_type: Any = "own") -> dict[str, Any]:
    spec = copy.deepcopy(load_spec(SHOWCASE))
    spec["rules"]["extinction"]["per_type"] = per_type
    return spec


def _kinds(engine: Engine, state: GameState) -> list[dict[str, int]]:
    """Each player's officers on the core's board, by id."""
    ids = engine.spec.piece_ids
    found: list[dict[str, int]] = [{}, {}]
    for value in state.board:
        if value:
            player, kind = divmod(value - 1, len(ids))
            found[player][ids[kind]] = found[player].get(ids[kind], 0) + 1
    return found


def _play(engine: Engine, texts: tuple[str, ...]) -> GameState:
    state = engine.initial_state()
    for text in texts:
        assert engine.terminal(state) is None, text
        state = engine.apply(state, engine.find_move(state, text))
    return state


def test_the_showcase_is_the_measured_game_and_says_own() -> None:
    spec = load_spec(SHOWCASE)
    assert validate_spec(spec) == []
    assert spec_hash(spec) == "26636602fae0"
    assert spec["rules"]["extinction"]["per_type"] == "own"


def test_each_side_counts_only_its_own_three_kinds() -> None:
    engine = Engine(load_spec(SHOWCASE))
    rule = engine.spec.extinction
    assert rule is not None and rule.per_type and rule.own
    ids = engine.spec.piece_ids
    own = [{ids[t] for t in side} for side in rule.side_types]
    assert own == [{"R", "B", "G"}, {"C", "N", "E"}]


def test_own_ends_the_line_where_the_summed_rule_plays_on() -> None:
    own = Engine(_spec("own"))
    summed = Engine(_spec(False))
    before = _play(own, LINE[:-1])
    assert _kinds(own, before)[0]["B"] == 1, "the riders are on their last bishop"

    end = _play(own, LINE)
    left = _kinds(own, end)
    assert "B" not in left[0]
    assert left[0]["R"] == 2 and left[0]["G"] == 2, "every other own kind still stands"
    outcome = own.terminal(end)
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (1, "extinction")

    same = _play(summed, LINE)
    assert same.board == end.board, "the rule moves no piece, only the ending"
    assert summed.terminal(same) is None
    assert summed.legal_moves(same), "the riders have moves and play on"


def test_the_shared_list_cannot_say_this_game() -> None:
    errors = validate_spec(_spec(True))
    assert any("per_type" in error and "before the first move" in error for error in errors), errors
