"""H2 (Q1127, `docs/plan-hunger.md` §3 and §R2): income, ``rules.income``.

At the start of each of a player's turns, the first mover's first turn included, the bank pays
them ``per_piece`` coins for each of their pieces standing on the region (written for the first
player and reflected; by default their first rank), into the purse, capped at 99. Held here by a
declared oracle (`_earned`, which reads the spec's square names and the board, never the
tables) over random walks of three games, by a perft over make/unmake against fresh applies
(the income is lossy at the cap, so `unmake` reads it back off a log), with the key checked at
every ply, beside the pass, the duck and a long turn, and every refusal by name.
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
from chess_variant_sandbox.engine.moves import MOVE_PASS, encode_move
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.describe import describe_spec as describe
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import PURSE_MAX, parse_square, validate_spec
from tests.feature_inventory import _schema_one_hundred_seventeen_uses as uses

ROOT = Path(__file__).resolve().parents[1]
PASS = encode_move(0, 0, MOVE_PASS, 0)
CENTRE = ["d4", "d5", "e4", "e5"]


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _income(income: Any = True, path: str = "tests/_purse_income.json") -> dict[str, Any]:
    spec = _load(path)
    spec["rules"]["income"] = income
    return spec


def _no_bounty(spec: dict[str, Any]) -> dict[str, Any]:
    """The same game with no booty, so the only coins a player is paid are their income."""
    for piece in spec["pieces"]:
        piece.pop("bounty", None)
    return spec


# ------------------------------------------------------------------------------- the oracle


def _region(spec: dict[str, Any], player: int) -> set[int]:
    """Where ``player``'s pieces earn, read off the spec's own words: the region's square names
    (or the first rank), turned round for the second player of a mirrored setup. Never the
    compiled tables."""
    width, height = spec["board"]["width"], spec["board"]["height"]
    income = spec["rules"]["income"]
    names = income.get("region") if isinstance(income, dict) else None
    if names is None:
        rank = 0 if player == 0 else height - 1
        return {rank * width + file for file in range(width)}
    out = set()
    for name in names:
        coords = parse_square(name, width, height)
        assert coords is not None
        file, rank = coords
        if player == 1:
            rank = height - 1 - rank
        out.add(rank * width + file)
    return out


def _earned(engine: Engine, spec: dict[str, Any], board: bytes, player: int) -> int:
    income = spec["rules"]["income"]
    per_piece = income.get("per_piece", 1) if isinstance(income, dict) else 1
    standing = sum(
        1 for square in _region(spec, player)
        if board[square] and engine.spec.owner_of(board[square]) == player
    )
    return per_piece * standing


def _walk(spec: dict[str, Any], seed: int, plies: int = 120) -> int:
    """Random play; at each ply the player about to move holds what they held plus their
    income on the board the ply left (capped), and the mover what they held less the move's
    cost. Returns how many plies paid anything, so a caller can say the walk exercised it."""
    engine = Engine(spec)
    rng = random.Random(seed)
    state = engine.initial_state()
    start = spec["rules"]["purse"]["start"]
    assert state.purse == (min(PURSE_MAX, start + _earned(engine, spec, state.board, 0)), start)
    paid = 0
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        moves = engine.legal_moves(state)
        move = rng.choice(moves)
        mover = state.side_to_move
        after = engine.apply(state, move)
        nxt = 1 - mover
        assert after.side_to_move == nxt
        earned = _earned(engine, spec, after.board, nxt)
        assert after.purse[nxt] == min(PURSE_MAX, state.purse[nxt] + earned)
        assert after.purse[mover] == state.purse[mover] - engine.move_cost(state, move)
        assert after.key == engine._compute_hash(after)
        paid += earned > 0 and after.purse[nxt] > state.purse[nxt]
        state = after
    return paid


@pytest.mark.parametrize(
    ("income", "seed"),
    [
        (True, 1),
        ({"per_piece": 3}, 2),
        ({"per_piece": 2, "region": CENTRE}, 3),
        ({"region": [f"{f}{r}" for f in "abcdefgh" for r in "5678"]}, 4),
    ],
    ids=["first-rank", "three-a-piece", "the-centre", "plunder"],
)
def test_the_oracle_agrees_on_random_walks(income: Any, seed: int) -> None:
    spec = _no_bounty(_income(income))
    # A full purse, so random play wanders far enough from home to stand on any ground.
    spec["rules"]["purse"]["start"] = 90
    assert validate_spec(spec) == []
    assert _walk(spec, seed, plies=200) > 0


def test_the_first_mover_starts_with_their_first_turns_income() -> None:
    """D6: White's first turn included, so the opening position holds 10 + 8 for White and 10
    for Black, and its key is the computed one."""
    engine = Engine(_income())
    state = engine.initial_state()
    assert state.purse == (18, 10)
    assert state.key == engine._compute_hash(state)


def test_perft_unmakes_income() -> None:
    """Perft over make/unmake against fresh applies, from positions where the purse binds (an
    empty start, so what a move may cost is what income paid) and at the cap."""
    for start in (0, PURSE_MAX):
        spec = _income()
        spec["rules"]["purse"]["start"] = start
        engine = Engine(spec)
        rng = random.Random(117 + start)
        state = engine.initial_state()
        for _ in range(16):
            moves = engine.legal_moves(state)
            fresh = sum(len(engine.legal_moves(engine.apply(state, m))) for m in moves)
            assert engine.perft(state, 2) == fresh
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state)
            if engine.terminal(state) is not None:
                break


def test_income_lands_after_a_pass_and_the_cap_holds() -> None:
    """A pass ends the turn, so the next player's income lands after it; two full purses passing
    back and forth repeat the position (D7's end for a mutual muster)."""
    spec = _no_bounty(_income())
    spec["rules"]["purse"]["start"] = 95
    engine = Engine(spec)
    state = engine.initial_state()
    assert state.purse == (PURSE_MAX, 95)
    state = engine.apply(state, PASS)
    assert state.purse == (PURSE_MAX, PURSE_MAX)
    state = engine.apply(state, PASS)
    assert state.purse == (PURSE_MAX, PURSE_MAX)
    again = engine.apply(engine.apply(state, PASS), PASS)
    assert again.key == state.key


def test_a_long_turn_pays_once_a_turn() -> None:
    """Under ``rules.turn`` income lands where the side flips: once a turn, at its first move."""
    spec = _load("rulesets/chess.json")
    spec["rules"]["turn"] = {"moves": 2}
    for key in ("castling", "en_passant"):
        spec["rules"].pop(key, None)
    spec["rules"]["purse"] = {"start": 0, "market": ["N"]}
    spec["rules"]["income"] = True
    for piece in spec["pieces"]:
        if piece["id"] == "N":
            piece["price"] = 30
    assert validate_spec(spec) == []
    engine = Engine(spec)
    state = engine.initial_state()
    assert state.purse == (8, 0)
    rng = random.Random(5)
    purses = [state.purse]
    for _ in range(8):
        state = engine.apply(state, rng.choice(engine.legal_moves(state)))
        purses.append(state.purse)
    # Black's purse moves on the second ply (the flip), White's on the fourth, never mid-turn.
    assert purses[1][1] == 0 and purses[2][1] > 0
    assert purses[3][0] == purses[2][0] and purses[4][0] > purses[3][0]


def test_the_duck_half_ends_the_turn_and_pays() -> None:
    """Under the duck a turn is two plies, the piece's and the duck's (`make_duck`, which flips
    the side itself): the income lands after the duck half, once a turn, and is taken back with
    it (perft over make/unmake against fresh applies)."""
    spec = _load("tests/_duck_chess.json")
    spec["rules"]["purse"] = {"start": 0, "market": ["N"]}
    spec["rules"]["income"] = True
    for piece in spec["pieces"]:
        if piece["id"] == "N":
            piece["price"] = 20
    assert validate_spec(spec) == []
    engine = Engine(spec)
    rng = random.Random(9)
    state = engine.initial_state()
    assert state.purse == (8, 0)
    for _ in range(16):
        moves = engine.legal_moves(state)
        fresh = sum(len(engine.legal_moves(engine.apply(state, m))) for m in moves)
        assert engine.perft(state, 2) == fresh
        after = engine.apply(state, rng.choice(moves))
        if after.side_to_move == state.side_to_move:
            assert after.purse == state.purse  # the piece half: no income yet
        else:
            nxt = after.side_to_move
            assert after.purse[nxt] > state.purse[nxt]  # eight pieces still at home
        state = after
        assert state.key == engine._compute_hash(state)


# ------------------------------------------------------------------------- the spelling


def test_the_spellings_are_one_game() -> None:
    short = _income(True)
    hashes = {
        spec_hash(_income(spelling)) for spelling in (
            True, {}, {"per_piece": 1},
            {"region": ["h1", "g1", "f1", "e1", "d1", "c1", "b1", "a1"]},
            {"per_piece": 1, "region": ["a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1"]},
        )
    }
    assert hashes == {spec_hash(short)}
    assert spec_hash(_income({"region": CENTRE})) == spec_hash(_income({"region": CENTRE[::-1]}))
    assert spec_hash(_income({"per_piece": 2})) != spec_hash(short)
    off = _income(False)
    off["rules"].pop("pass")
    without = _income(True)
    without["rules"].pop("income")
    without["rules"].pop("pass")
    assert spec_hash(off) == spec_hash(without)


def test_the_tables_are_empty_without_it() -> None:
    chess = _load("rulesets/chess.json")
    tables = build_tables(compile_spec(chess))
    assert tables["income_per_piece"] == 0 and tables["income_squares"] == []
    tables = build_tables(compile_spec(_income({"per_piece": 2, "region": CENTRE})))
    assert tables["income_per_piece"] == 2
    n = 64
    assert sorted(tables["income_squares"]) == sorted([27, 28, 35, 36] + [n + s for s in (27, 28, 35, 36)])


def test_the_shape() -> None:
    for bad, said in (
        ("yes", "rules.income must be true, false or"),
        ({"per_piece": 0}, "rules.income.per_piece must be"),
        ({"per_piece": 6}, "rules.income.per_piece must be"),
        ({"region": []}, "rules.income.region must be a non-empty list"),
        ({"region": ["z9"]}, "not a square on this board"),
        ({"region": ["a1", "a1"]}, "names a square twice"),
        ({"rate": 1}, "rules.income"),
    ):
        assert any(said in error for error in validate_spec(_income(bad))), bad


def _set(key: str, value: Any) -> Any:
    def edit(spec: dict[str, Any]) -> None:
        spec["rules"][key] = value
    return edit


REFUSED = {
    "income.I1": _set("scoring", {"count": "points"}),
    "income.I2": _set("simultaneous", {}),
}


@pytest.mark.parametrize("rid", sorted(REFUSED))
def test_each_refusal_by_name(rid: str) -> None:
    spec = _income()
    REFUSED[rid](spec)
    assert rid in refused_ids(validate_spec(spec))


def test_income_needs_a_purse_and_ground() -> None:
    spec = _load("rulesets/chess.json")
    spec["rules"]["income"] = True
    assert any("rules.income without rules.purse" in e for e in validate_spec(spec))
    walled = _income({"region": ["d4"]})
    walled["rules"]["obstacles"] = {"squares": ["d4"]}
    assert any("has no square a piece could stand on" in e for e in validate_spec(walled))


def test_the_names_the_words_and_fairy() -> None:
    assert uses(_income()) == {"rules.income"}
    assert uses(_income(False)) == set()
    assert uses(_income({"per_piece": 2, "region": CENTRE})) == {
        "rules.income", "rules.income.per_piece", "rules.income.region",
    }
    assert uses(_income({"region": ["a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1"]})) == {
        "rules.income",
    }
    words = describe(_income())
    assert "the bank pays them 1 coin for each piece of theirs" in words
    assert "their own first rank" in words
    assert "d4" in describe(_income({"per_piece": 2, "region": CENTRE}))
    with pytest.raises(Untranslatable, match="rules.income"):
        translate(copy.deepcopy(_income()))
