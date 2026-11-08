"""H3 (Q1126, `docs/plan-hunger.md` §4 and §R3): the pass, ``rules.pass``.

The pass is the null move filtered as every move is (D1 as Q1130 re-cut it): offered once,
after every filter, where the side to move is not in check in a game with check, and always in
a game without. It changes nothing on the board, pays nothing (D4), closes the en passant
window, counts toward the no-progress clock, keeps the repetition window open and counts a
player's rests down. Held here on constructed positions (stalemate gone, mate kept, a royal
capture game passing under attack), by a perft against a walk of fresh applies with the key
checked at every ply, against Fairy-Stockfish's own ``pass = true`` where ``pyffish`` is
installed, and by name for every refusal.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_PASS, encode_move, move_type
from chess_variant_sandbox.engine.state import NO_SQUARE
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.describe import describe_spec as describe
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_one_hundred_sixteen_uses as uses

ROOT = Path(__file__).resolve().parents[1]
PASS = encode_move(0, 0, MOVE_PASS, 0)


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _passing(path: str = "rulesets/chess.json") -> dict[str, Any]:
    spec = _load(path)
    spec["rules"]["pass"] = True
    return spec


def _position(pieces: dict[str, tuple[str, int]], **rules: Any) -> dict[str, Any]:
    """Chess's army on a set position (square: (piece, player)), white to move, with the pass,
    no castling, en passant or promotion (nothing here needs them)."""
    spec = _load("rulesets/chess.json")
    spec["setup"] = {
        "mirror": False,
        "placements": [
            {"piece": piece, "player": player, "squares": [square]}
            for square, (piece, player) in pieces.items()
        ],
    }
    for key in ("castling", "en_passant", "insufficient_material", "promotion"):
        spec["rules"].pop(key, None)
    spec["rules"]["pass"] = True
    spec["rules"].update(rules)
    return spec


def _passes(engine: Engine, state: Any) -> list[int]:
    return [m for m in engine.legal_moves(state) if move_type(m) == MOVE_PASS]


def test_the_pass_is_offered_once_last_and_written_pass() -> None:
    engine = Engine(_passing())
    state = engine.initial_state()
    moves = engine.legal_moves(state)
    assert len(moves) == 21 and moves[-1] == PASS
    assert engine.describe_move(state, PASS) == "pass"


def test_two_passes_repeat_the_position_and_tick_the_clock() -> None:
    engine = Engine(_passing())
    start = engine.initial_state()
    once = engine.apply(start, PASS)
    assert once.board == start.board and once.side_to_move == 1
    assert once.halfmove == 1 and once.key == engine._compute_hash(once)
    twice = engine.apply(once, PASS)
    assert twice.key == start.key and twice.halfmove == 2
    state = twice
    for _ in range(2):
        state = engine.apply(engine.apply(state, PASS), PASS)
    outcome = engine.terminal(state)
    assert outcome is not None and outcome.reason == "repetition"


def test_a_pass_closes_the_en_passant_window() -> None:
    engine = Engine(_passing())
    state = engine.initial_state()
    double = next(m for m in engine.legal_moves(state) if engine.describe_move(state, m) == "Pe2-e4")
    after = engine.apply(state, double)
    assert after.ep_target != NO_SQUARE
    passed = engine.apply(after, PASS)
    assert passed.ep_target == NO_SQUARE and passed.key == engine._compute_hash(passed)


def test_stalemate_is_gone_and_mate_stays() -> None:
    """White's king on a1, the black queen on b3: no move but the pass, so no stalemate. With the
    queen on b2 guarded by the king, white is in check with no evasion: mate, no pass."""
    stale = Engine(_position({"a1": ("K", 0), "b3": ("Q", 1), "h8": ("K", 1)}))
    state = stale.initial_state()
    assert stale.legal_moves(state) == [PASS]
    assert stale.terminal(state) is None
    mate = Engine(_position({"a1": ("K", 0), "b2": ("Q", 1), "c3": ("K", 1)}))
    state = mate.initial_state()
    assert mate.legal_moves(state) == []
    outcome = mate.terminal(state)
    assert outcome is not None and outcome.reason == "checkmate" and outcome.winner == 1


def test_in_check_there_is_no_pass_where_the_game_has_check() -> None:
    engine = Engine(_position({"a1": ("K", 0), "a8": ("R", 1), "h8": ("K", 1), "h1": ("R", 0)}))
    state = engine.initial_state()
    assert engine.in_check(state, 0) and _passes(engine, state) == []


def test_without_check_a_pass_is_always_offered() -> None:
    """Royal capture: the null move is filtered by nothing, so an attacked royal may pass."""
    spec = _position(
        {"a1": ("K", 0), "a8": ("R", 1), "h8": ("K", 1), "h1": ("R", 0)}, win="royal_capture",
    )
    assert validate_spec(spec) == []
    engine = Engine(spec)
    assert _passes(engine, engine.initial_state()) == [PASS]


def test_perft_unmakes_a_pass() -> None:
    """Perft over make/unmake against a walk of fresh applies, at random positions of chess with
    the pass, and the incremental key at every ply."""
    engine = Engine(_passing())
    rng = random.Random(116)
    state = engine.initial_state()
    for _ in range(30):
        moves = engine.legal_moves(state)
        fresh = sum(len(engine.legal_moves(engine.apply(state, m))) for m in moves)
        assert engine.perft(state, 2) == fresh
        state = engine.apply(state, rng.choice(moves))
        assert state.key == engine._compute_hash(state)
        if engine.terminal(state) is not None:
            break


def test_a_pass_pays_nothing_and_an_empty_purse_musters() -> None:
    """Beside H1's upkeep (chess with Hunger's charges): the pass costs 0, and a purse that can pay
    for nothing passes instead of being stalemated (`test_upkeep`'s stalemate, lifted)."""
    spec = _load("tests/_purse_upkeep.json")
    spec["rules"]["pass"] = True
    spec["rules"]["purse"]["start"] = 0
    assert validate_spec(spec) == []
    engine = Engine(spec)
    state = engine.initial_state()
    assert engine.legal_moves(state) == [PASS]
    assert engine.move_cost(state, PASS) == 0
    after = engine.apply(state, PASS)
    assert after.purse == state.purse


def test_a_pass_is_a_turn_for_the_rest() -> None:
    """The weary queen rests a turn after moving; the player's pass counts the rest down as any
    other move of theirs does."""
    engine = Engine(_passing("archive/chess-with-a-weary-queen.json"))
    state = engine.initial_state()
    for text in ("Pe2-e4", "Pe7-e5", "Qd1-h5", "Nb8-c6"):
        state = engine.apply(state, next(
            m for m in engine.legal_moves(state) if engine.describe_move(state, m) == text
        ))
    assert state.rest[0]
    passed = engine.apply(state, PASS)
    moved = engine.apply(state, next(
        m for m in engine.legal_moves(state) if engine.describe_move(state, m) == "Ng1-f3"
    ))
    assert passed.rest == moved.rest
    assert passed.key == engine._compute_hash(passed)


def test_the_off_value_and_the_tables() -> None:
    chess = _load("rulesets/chess.json")
    off = copy.deepcopy(chess)
    off["rules"]["pass"] = False
    assert validate_spec(off) == [] and spec_hash(off) == spec_hash(chess)
    assert build_tables(compile_spec(chess))["passes"] is False
    assert build_tables(compile_spec(_passing()))["passes"] is True
    assert spec_hash(_passing()) != spec_hash(chess)


def test_the_rule_must_be_a_boolean_and_kills_stalemate() -> None:
    bad = _passing()
    bad["rules"]["pass"] = "yes"
    assert any("rules.pass must be a boolean" in e for e in validate_spec(bad))
    dead = _passing()
    dead["rules"]["stalemate"] = "loss"
    assert any("beside rules.pass" in e for e in validate_spec(dead))
    drawn = _passing()
    drawn["rules"]["stalemate"] = "draw"
    assert validate_spec(drawn) == []


def _set(key: str, value: Any) -> Any:
    def edit(spec: dict[str, Any]) -> None:
        spec["rules"][key] = value
    return edit


def _placement(spec: dict[str, Any]) -> None:
    spec["rules"]["drops"] = {"from_captures": True, "placement": True, "stock": {"N": 1}}


def _push(spec: dict[str, Any]) -> None:
    spec["pieces"][3]["moves"][0]["push"] = True


REFUSED = {
    "pass.P1": _set("simultaneous", {}),
    "pass.P2": _set("duck", {"piece": "D"}),
    "pass.P3": _set("turn", {"moves": 2}),
    "pass.P4": _set("must_capture", True),
    "pass.P5": _placement,
    "pass.P6": _set("npcs", [{"piece": "N", "route": ["d4", "d5"]}]),
    "pass.P7": _set("portals", [{"at": "d4", "to": "e5"}]),
    "pass.P8": _push,
}


@pytest.mark.parametrize("rid", sorted(REFUSED))
def test_each_refusal_by_name(rid: str) -> None:
    spec = _passing()
    REFUSED[rid](spec)
    assert rid in refused_ids(validate_spec(spec))


def test_the_names_the_words() -> None:
    assert uses(_passing()) == {"rules.pass"}
    assert uses(_load("rulesets/chess.json")) == set()
    words = describe(_passing())
    assert "may pass" in words and "no position is stalemate" in words
    assert "A player with no legal move" not in words


def test_fairy_writes_pass_beside_its_king_only() -> None:
    variant = translate(_passing())
    assert "pass = true" in variant.config
    capture = _passing()
    capture["rules"]["win"] = "royal_capture"
    capture["rules"].pop("insufficient_material", None)
    with pytest.raises(Untranslatable, match="rules.pass"):
        translate(capture)


def test_fairy_stockfish_agrees_on_the_pass() -> None:
    """Their `pass = true` and ours offer the same moves over random chess games with the pass,
    the pass spelled as the king standing still (`position.uci_of`)."""
    pytest.importorskip("pyffish")
    from chess_variant_sandbox.fairy.verify import PyffishBackend, verify

    result = verify(_passing(), PyffishBackend(), "", depth=2, games=2, plies=60)
    assert result.ok, result.summary()
