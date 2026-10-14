"""Q844b: Sluice, Crease and pawnproof wormhole, the showcases of schemas 21 and 24's portals.

`docs/reference/portal-showcase-ideas.md` (*Q844b: the games, and what they predict*)
pre-registered these clauses before this file was written; the class names follow it. Sluice
is the game of schema 21's maps (`ends`, `to`, `enter`): a funnel a side. Crease is the game of
schema 24's folds (`kind: "fold"`): a landing on a fold stays or goes through. Pawnproof
wormhole is `wormhole` with one filter (`carries`), kept as a variant because the contrast is
its point.

Every test drives the core through `Engine` and asserts what the core did. The only choice of
a move in Python is the seeded random walk, as Q772a's and Q876's tests do. Squares are named,
never computed from an index. `tally` is also how the measured clauses are read from the
persisted games (`runs/<hash>-<seed>/games.jsonl`): it replays each move list and classifies
moves the core already made.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.engine.moves import (
    MOVE_FOLD,
    MOVE_PORTAL,
    move_fold,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec, spec_hash

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(6)  # seeds 0 to 5, as pre-registered
PLIES = 300
LIBRARY_SEEDS = (0, 1000, 2000, 3000)
STRONG = ("nodes=1600", "nodes=6400")
# Taken when this file was written: regression pins, not predictions.
LADDERS: dict[str, dict[int, int]] = {
    "sluice": {1: 24, 2: 533, 3: 12360},
    "crease": {1: 58, 2: 3324, 3: 197593},
    "pawnproof-wormhole": {1: 5, 2: 27, 3: 179},
}


@cache
def _spec(name: str) -> dict[str, Any]:
    return load_spec(ROOT / "archive" / f"{name}.json")


@cache
def _engine(name: str) -> Engine:
    return Engine(copy.deepcopy(_spec(name)))


def _random_game(engine: Engine, seed: int) -> list[int]:
    rng = random.Random(seed)
    state = engine.initial_state()
    moves: list[int] = []
    for _ in range(PLIES):
        if engine.terminal(state) is not None:
            break
        legal = engine.legal_moves(state)
        if not legal:
            break
        move = rng.choice(legal)
        moves.append(move)
        state = engine.apply(state, move)
    return moves


@cache
def _walks(name: str) -> tuple[tuple[int, ...], ...]:
    engine = _engine(name)
    return tuple(tuple(_random_game(engine, seed)) for seed in SEEDS)


def _replay(engine: Engine, moves: Iterable[int]) -> Iterator[tuple[GameState, int]]:
    state = engine.initial_state()
    for move in moves:
        yield state, move
        state = engine.apply(state, move)


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def _id(engine: Engine, value: int) -> str:
    return engine.spec.piece_ids[engine.spec.type_of(value)]


def _ends(engine: Engine, state: GameState) -> dict[str, int]:
    """Each momentum end on the board, by its name (`1a`, `2c`, ...)."""
    return {
        _id(engine, value): square
        for square, value in enumerate(state.board)
        if value and engine.spec.owner_of(value) == WILD
    }


def _entered(engine: Engine, state: GameState, move: int) -> str:
    entry = engine.portal_entry(state, move)
    assert entry is not None
    return _id(engine, state.board[entry])


def _openers(name: str) -> set[str]:
    engine = _engine(name)
    return set(engine.move_texts(engine.initial_state()))


def _rung(agent: str) -> str:
    return next((part for part in agent.split(",") if part.startswith("nodes=")), agent)


def tally(engine: Engine, games: Iterable[dict[str, Any]]) -> Counter[str]:
    """Count what the portals did in ``games`` (each a `games.jsonl` row, or a dict with at
    least ``moves``): per-game flags as ``games_<flag>``, per-move counts by name.

    A *transit* is a `MOVE_PORTAL`, counted by the end gone in by and the piece. A *gate
    move* (Sluice) is a move of a momentum end. A *fold move* is a `MOVE_FOLD`, *onto a
    piece* when its target holds one. A *go-through* is an ordinary move tagged with the fold
    it went in by; a *stay* is an untagged move ending on an empty fold.
    """
    out: Counter[str] = Counter()
    for game in games:
        out["games"] += 1
        moves = game["moves"]
        out["plies"] += len(moves)
        if "reason" in game:
            out[f"reason_{game['reason']}"] += 1
        if "agents" in game:
            rungs = tuple(_rung(a) for a in game["agents"])
            if set(rungs) <= set(STRONG):
                out["top_games"] += 1
                if game["winner"] is not None:
                    out["top_decided"] += 1
                    out["top_first_seat_won"] += game["winner"] == 0
            if sorted(rungs) == sorted(STRONG):
                out["strong_games"] += 1
                out["strong_decided"] += game["winner"] is not None
        flags: set[str] = set()
        for state, move in _replay(engine, moves):
            player = state.side_to_move
            kind = move_type(move)
            origin, target = move_from(move), move_to(move)
            mover = state.board[origin]
            if kind == MOVE_FOLD:
                out["fold_moves"] += 1
                flags |= {"fold_move", f"fold_move_by_{player}"}
                if state.board[target]:
                    out["fold_onto_piece"] += 1
                    flags.add("fold_onto_piece")
            elif mover and engine.spec.owner_of(mover) == WILD:
                end = _id(engine, mover)
                out["gate_moves"] += 1
                out[f"gate_moves_{end}"] += 1
                flags |= {"gate_move", f"gate_move_{end}"}
                if state.board[target]:
                    out["gate_onto_piece"] += 1
            elif kind == MOVE_PORTAL:
                end = _entered(engine, state, move)
                piece = _id(engine, mover)
                out["transits"] += 1
                out[f"enter_{end}_by_{player}"] += 1
                out[f"transit_{piece}"] += 1
                flags |= {"transit", f"enter_{end}", f"transit_by_{player}"}
                if state.board[target]:
                    out["transit_captures"] += 1
            elif move_fold(move):
                out["through"] += 1
                flags |= {"through", f"through_by_{player}"}
                if state.board[target]:
                    out["through_captures"] += 1
                    flags.add("through_capture")
            elif state.folds and target in state.folds and not state.board[target]:
                out["stays"] += 1
                flags |= {"stay", f"stay_by_{player}"}
        for name in ("transit", "through", "stay", "fold_move"):
            if {f"{name}_by_0", f"{name}_by_1"} <= flags:
                flags.add(f"{name}_by_both")
        for flag in flags:
            out[f"games_{flag}"] += 1
    return out


def read_runs(name: str, runs: Path) -> Counter[str]:
    """`tally` over the library seeds' persisted games of ``name`` under ``runs``."""
    digest = spec_hash(_spec(name))[:12]
    engine = _engine(name)
    rows = (
        json.loads(line)
        for seed in LIBRARY_SEEDS
        for line in (runs / f"{digest}-{seed}" / "games.jsonl").open(encoding="utf-8")
    )
    return tally(engine, rows)


def _random_tally(name: str) -> Counter[str]:
    return tally(_engine(name), ({"moves": list(moves)} for moves in _walks(name)))


# ------------------------------------------------------------------------------ Sluice


class TestSluice:
    """Clause 1: two doors a side lead to one gate, only the owner goes in, and the side a
    gate threatens moves it."""

    DOORS = {0: {"1a", "1b"}, 1: {"2a", "2b"}}
    GATES = {"1c", "2c"}

    def test_the_doors_lead_beyond_the_gate(self) -> None:
        assert {"Pa2-a3>e6", "Ph2-h3>e6", "Nb1-a3>d6", "Ng1-h3>f6"} <= _openers("sluice")

    def test_white_moves_blacks_gate_and_nothing_else_of_the_board(self) -> None:
        engine = _engine("sluice")
        state = engine.initial_state()
        ends = set(_ends(engine, state).values())
        moved = [m for m in engine.legal_moves(state) if move_from(m) in ends]
        assert {_id(engine, state.board[move_from(m)]) for m in moved} == {"2c"}
        assert len(moved) == 7

    def test_a_gate_may_go_back_on_its_movers_next_turn(self) -> None:
        # No bar: one player moves each gate, so a bar on the next ply could never bind.
        engine = _engine("sluice")
        state = engine.initial_state()
        for text in ("Pe2-e3", "1c:e5-d5", "Pd2-d3"):
            state = engine.apply(state, engine.find_move(state, text))
        assert {"1c:d5-e5", "1c:d5-c5", "1c:d5-d4"} <= set(engine.move_texts(state))

    def test_in_play_each_side_uses_its_own_doors_and_moves_the_other_gate(self) -> None:
        engine = _engine("sluice")
        transit_by: set[int] = set()
        gate_by: set[int] = set()
        for moves in _walks("sluice"):
            for state, move in _replay(engine, moves):
                ends = _ends(engine, state)
                player = state.side_to_move
                foreign = {ends[d] for d in self.DOORS[1 - player]} | {
                    ends[g] for g in self.GATES
                }
                origin, target = move_from(move), move_to(move)
                if origin in ends.values():
                    gate = _id(engine, state.board[origin])
                    assert gate == ("2c" if player == 0 else "1c")
                    assert not state.board[target]
                    gate_by.add(player)
                    continue
                assert target not in foreign, engine.describe_move(state, move)
                if move_type(move) == MOVE_PORTAL:
                    assert _entered(engine, state, move) in self.DOORS[player]
                    transit_by.add(player)
        assert transit_by == gate_by == {0, 1}


# ------------------------------------------------------------------------------ Crease


def _crease(placement: dict[str, tuple[str, int]], side: int = 0) -> tuple[Engine, GameState]:
    engine = _engine("crease")
    kings = {"f1": ("K", 0), "f8": ("K", 1)}
    return engine, engine.make_state({**kings, **placement}, side_to_move=side)


def _rook_texts(engine: Engine, state: GameState) -> set[str]:
    return {t for t in engine.move_texts(state) if t.startswith("Rb1-b4") or t == "Rb1xb4"}


class TestCrease:
    """Clause 2: a landing on an empty fold stays or goes through, capturing at the far end;
    a piece in either fold changes that; folds are moved by both, onto pieces too."""

    def test_the_opening_offers_stay_and_go_through(self) -> None:
        openers = _openers("crease")
        assert {"Pb2-b4", "Pb2-b4~i5", "Pi2-i4", "Pi2-i4~b5"} <= openers
        for end in ("1o:b4", "1b:i5", "2o:i4", "2b:b5"):
            assert sum(t.startswith(end + "-") for t in openers) == 7

    def test_a_rook_stays_or_goes_through_to_an_empty_far_end(self) -> None:
        engine, state = _crease({"b1": ("R", 0)})
        assert _rook_texts(engine, state) == {"Rb1-b4", "Rb1-b4~i5"}

    def test_going_through_takes_what_stands_on_the_far_end(self) -> None:
        engine, state = _crease({"b1": ("R", 0), "i5": ("N", 1)})
        assert _rook_texts(engine, state) == {"Rb1-b4", "Rb1-b4~xi5"}

    def test_no_way_through_onto_ones_own_piece(self) -> None:
        engine, state = _crease({"b1": ("R", 0), "i5": ("N", 0)})
        assert _rook_texts(engine, state) == {"Rb1-b4"}

    def test_a_piece_in_the_fold_shuts_it(self) -> None:
        engine, state = _crease({"b1": ("R", 0), "b4": ("N", 1)})
        assert _rook_texts(engine, state) == {"Rb1xb4"}
        engine, state = _crease({"b1": ("R", 0), "b4": ("N", 0)})
        assert _rook_texts(engine, state) == set()

    def test_a_king_on_the_far_end_is_in_check_through_the_fold(self) -> None:
        engine = _engine("crease")
        open_door = {"f1": ("K", 0), "i5": ("K", 1), "b1": ("R", 0)}
        assert engine.in_check(engine.make_state(open_door, side_to_move=1), 1)
        shut = engine.make_state({**open_door, "b4": ("P", 1)}, side_to_move=1)
        assert not engine.in_check(shut, 1)

    def test_in_play_both_go_through_stay_and_move_folds_onto_pieces(self) -> None:
        counts = _random_tally("crease")
        for flag in ("through_by_0", "through_by_1", "fold_move_by_0", "fold_move_by_1"):
            assert counts[f"games_{flag}"] >= 1, flag
        assert counts["stays"] >= 1 and counts["fold_onto_piece"] >= 1


# ------------------------------------------------------------------------------ pawnproof wormhole


class TestPawnproofWormhole:
    """Clause 3: every piece but the pawn goes through; the filter is the one difference."""

    def test_no_pawn_ever_moves_onto_an_end(self) -> None:
        engine = _engine("pawnproof-wormhole")
        ends = {_sq(engine, "b3"), _sq(engine, "d3")}
        openers = {t.split(">")[0] for t in _openers("pawnproof-wormhole")}
        assert not {"Pb2-b3", "Pd2-d3"} & openers
        for moves in _walks("pawnproof-wormhole"):
            for state, move in _replay(engine, moves):
                if _id(engine, state.board[move_from(move)]) == "P":
                    assert move_to(move) not in ends

    def test_a_knight_goes_through_both_ends_and_a_pawn_has_no_way_in(self) -> None:
        engine = _engine("pawnproof-wormhole")
        state = engine.make_state({"b3": ("1o", WILD), "d3": ("1b", WILD), "e1": ("K", 0),
                                   "e5": ("K", 1), "c1": ("N", 0), "b2": ("P", 0)})
        moves = engine.legal_moves(state)
        knight = {_entered(engine, state, m) for m in moves
                  if move_type(m) == MOVE_PORTAL and move_from(m) == _sq(engine, "c1")}
        assert knight == {"1o", "1b"}
        assert not [m for m in moves if move_from(m) == _sq(engine, "b2")
                    and move_to(m) == _sq(engine, "b3")]

    def test_without_the_filter_it_is_wormhole(self) -> None:
        spec = copy.deepcopy(_spec("pawnproof-wormhole"))
        del spec["rules"]["portals"][0]["carries"]
        spec["schema_version"] = _spec("wormhole")["schema_version"]
        assert spec_hash(spec) == spec_hash(_spec("wormhole"))


# ------------------------------------------------------------------------------ the ladders


class TestTheLadders:
    """Clause 4: taken, not predicted."""

    @pytest.mark.parametrize(
        ("name", "depth"), [(n, d) for n, ladder in LADDERS.items() for d in sorted(ladder)]
    )
    def test_the_ladder(self, name: str, depth: int) -> None:
        engine = _engine(name)
        assert engine.perft(engine.initial_state(), depth) == LADDERS[name][depth]
