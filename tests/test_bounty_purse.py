"""TC4 (`docs/plan-turncoat.md` §5.1 to §5.3): bounties on any piece type, and the purse's state.

`pieces[].bounty` pays the player whose move takes a piece of that type (any piece but their
own: an enemy's, a neutral) into the `points` tally, which `scoring.count: "points"` compares
and which no longer needs wild pieces. The purse (`Game::purse`) is coins each player holds:
raised by the same payments, capped at `PURSE_MAX`, in the Zobrist key, crossing as slot 11's
twelfth element. TC4 built it with no spec key and a test hook; TC5's `rules.purse` replaced
the hook, so the purse here is a market selling the scout (`S`) for `_PRICE` coins
(`_purse_spec`), whose buys `tests/test_purse_market.py` covers.

Hand-built specs, not chess: a 6x6 board, two kings, a rook and guards. The oracle for what a
move pays is this file's `_paid`: the material of the types with a bounty that the mover's
opponents lost on the move, read off two boards (a declared test oracle, rule 3).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_BUY, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.schema import PURSE_MAX, validate_spec
from chess_variant_sandbox.tournament.play import play_game

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(**rules: Any) -> dict[str, Any]:
    """6x6: kings a1/a6, a rook (bounty 4) f1/f6, guards (bounty 1) b1-d1/b6-d6, a clock of
    60 plies counted in points."""
    return {
        "schema_version": 31,
        "name": "the hunt",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK], "bounty": 4},
            {"id": "G", "value": 2, "moves": [KING], "bounty": 1},
            {"id": "S", "value": 1, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["f1"]},
            {"piece": "G", "squares": ["b1", "c1", "d1"]},
            {"piece": "S", "squares": ["e1"]},
        ]},
        "rules": {
            "win": "checkmate", "stalemate": "draw", "max_plies": 60,
            "scoring": {"count": "points"}, **rules,
        },
    }


# What the market charges for a scout: dear, so a walk near the cap buys now and then.
_PRICE = 40


def _purse_spec(start: int, **rules: Any) -> dict[str, Any]:
    """`_spec` with a purse of ``start`` coins each and a market selling the scout."""
    spec = _spec(**rules)
    spec["pieces"][3]["price"] = _PRICE
    spec["rules"]["purse"] = {"start": start, "market": ["S"]}
    return spec


def _without_bounties(spec: dict[str, Any]) -> dict[str, Any]:
    plain = copy.deepcopy(spec)
    for piece in plain["pieces"]:
        piece.pop("bounty", None)
    return plain


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0,
              purse: tuple[int, int] | None = None) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement), purse=purse)


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    for move in engine.legal_moves(state):
        if (move & 0xFF, (move >> 8) & 0xFF) == (_sq(origin), _sq(target)):
            return move
    raise AssertionError(f"no move {origin}-{target}")


def _paid(engine: Engine, before: GameState, after: GameState) -> int:
    """The oracle: the bounties of the pieces the mover's opponents lost on this move."""
    spec = engine.spec
    mover = before.side_to_move

    def lost(board: bytes) -> list[int]:
        return [
            spec.bounty[spec.type_of(value)] for value in board
            if value and spec.owner_of(value) != mover
        ]

    return sum(lost(before.board)) - sum(lost(after.board))


# --------------------------------------------------------------------------- the spec


class TestTheKey:
    def test_a_bounty_game_is_valid_and_its_off_value_is_the_game_without(self) -> None:
        assert validate_spec(_spec()) == []
        zero = _without_bounties(_spec())
        zero["rules"].pop("scoring")
        written = copy.deepcopy(zero)
        written["pieces"][1]["bounty"] = 0
        assert validate_spec(written) == []
        assert spec_hash(written) == spec_hash(zero)

    @pytest.mark.parametrize("bad", [21, -1, 1.5, True, "3"])
    def test_a_bounty_is_a_whole_number_to_twenty(self, bad: Any) -> None:
        spec = _spec()
        spec["pieces"][1]["bounty"] = bad
        assert any("pieces[1].bounty" in error for error in validate_spec(spec))

    def test_a_royal_carries_none(self) -> None:
        spec = _spec()
        spec["pieces"][0]["bounty"] = 3
        assert any("on a royal" in error for error in validate_spec(spec))

    def test_a_wild_type_carries_its_routes_bounty_not_a_types(self) -> None:
        spec = _spec()
        spec["pieces"].append({"id": "W", "value": 2, "moves": [], "bounty": 2})
        spec["rules"]["npcs"] = [{"piece": "W", "route": ["c3", "d3"]}]
        assert any("on a wild type" in error for error in validate_spec(spec))

    def test_the_duck_carries_none(self) -> None:
        spec = _spec(win="royal_capture")
        spec["pieces"].append({"id": "D", "value": 0, "moves": [], "bounty": 2})
        spec["rules"]["duck"] = {"piece": "D"}
        assert any("on the duck" in error for error in validate_spec(spec))

    def test_a_bounty_needs_a_points_count(self) -> None:
        spec = _spec()
        spec["rules"].pop("scoring")
        assert any("needs rules.scoring.count" in error for error in validate_spec(spec))

    def test_a_points_count_needs_a_bounty_and_no_longer_a_wild_piece(self) -> None:
        assert validate_spec(_spec()) == []
        errors = validate_spec(_without_bounties(_spec()))
        assert any('"points" needs a bounty above zero' in error for error in errors)

    def test_the_description_says_who_is_paid(self) -> None:
        text = describe_spec(_spec())
        assert "pays a bounty of 4" in text
        assert "R 4" in text and "G 1" in text
        assert "taking pieces with a bounty" in text


# --------------------------------------------------------------------------- the tables


class TestTheTables:
    def test_a_game_without_bounties_crosses_none(self) -> None:
        spec = _without_bounties(_spec())
        spec["rules"].pop("scoring")
        compiled = compile_spec(spec)
        assert compiled.bounty == () and compiled.purse is None
        tables = canonical_tables(compiled)
        assert not {"piece_bounty", "purse_start", "zobrist_purse"} & set(tables)

    def test_each_type_crosses_its_bounty(self) -> None:
        tables = build_tables(compile_spec(_spec()))
        assert tables["piece_bounty"] == [0, 4, 1, 0]
        assert tables["purse_start"] == -1 and tables["zobrist_purse"] == []

    def test_fairy_stockfish_refuses_a_bounty_by_name(self) -> None:
        with pytest.raises(Untranslatable, match="pieces.bounty"):
            translate(_spec())


# --------------------------------------------------------------------------- paying


class TestPaying:
    def test_taking_an_enemy_piece_pays_its_bounty(self) -> None:
        engine = Engine(_spec())
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0),
                                   "f5": ("R", 1), "c4": ("G", 1)})
        after = engine.apply(state, _move(engine, state, "f1", "f5"))
        assert after.points == (4.0, 0.0)
        after = engine.apply(after, _move(engine, after, "a6", "b6"))
        assert after.points == (4.0, 0.0)

    def test_a_type_without_a_bounty_pays_nothing(self) -> None:
        engine = Engine(_spec())
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0),
                                   "f5": ("S", 1)})
        assert engine.apply(state, _move(engine, state, "f1", "f5")).points == (0.0, 0.0)

    def test_the_second_player_is_paid_into_their_own_tally(self) -> None:
        engine = Engine(_spec())
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f2": ("G", 0),
                                   "f6": ("R", 1)}, side=1)
        assert engine.apply(state, _move(engine, state, "f6", "f2")).points == (0.0, 1.0)

    def test_taking_a_piece_nobody_owns_pays_its_taker(self) -> None:
        spec = _spec()
        spec["pieces"].append({"id": "C", "value": 3, "moves": [KING], "bounty": 2})
        spec["setup"]["placements"].append(
            {"piece": "C", "player": "nobody", "squares": ["c3"]}
        )
        assert validate_spec(spec) == [], validate_spec(spec)
        engine = Engine(spec)
        state = engine.initial_state()
        state = engine.apply(state, _move(engine, state, "c1", "c2"))
        state = engine.apply(state, _move(engine, state, "c6", "c5"))
        state = engine.apply(state, _move(engine, state, "c2", "c3"))
        assert state.points == (2.0, 0.0)

    def test_a_game_decided_by_bounty_points(self) -> None:
        engine = Engine(_spec(max_plies=2))
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0),
                                   "f5": ("G", 1), "d6": ("S", 1)})
        state = engine.apply(state, _move(engine, state, "f1", "f5"))
        state = engine.apply(state, _move(engine, state, "d6", "d5"))
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.winner == 0

    def test_a_search_puts_every_payment_back(self) -> None:
        """`unmake` refunds what `make` paid: a game the core plays with its searching players
        (make and unmake on one position, thousands of times a move) ends where a replay of
        its moves through `apply` ends, and is decided by the same points."""
        engine = Engine(_spec(max_plies=24))
        for seed in range(3):
            record = play_game(engine, ("minimax:nodes=300", "minimax:nodes=300"), seed)
            state = engine.initial_state()
            for move in record.moves:
                state = engine.apply(state, move)
            outcome = engine.terminal(state)
            assert outcome is not None
            assert (outcome.winner, outcome.reason) == (record.winner, record.reason)


# --------------------------------------------------------------------------- the purse


class TestThePurse:
    def test_the_validator_refuses_what_the_purse_is_refused_beside(self) -> None:
        """TC4's hook refused these; TC5's `rules.purse` is refused by the validator."""
        refusals = {
            "simultaneous": _purse_spec(3, simultaneous={}, win="royal_capture"),
            "explode_on_capture": _purse_spec(3, explode_on_capture={}),
            "a turn longer than one move": _purse_spec(3, turn={"moves": 2}),
        }
        for word, spec in refusals.items():
            assert any(word in error for error in validate_spec(spec)), (word, validate_spec(spec))
        for start in (-1, PURSE_MAX + 1, True):
            assert any("rules.purse.start" in error for error in validate_spec(_purse_spec(start)))

    def test_a_purse_holds_whole_coins_only(self) -> None:
        spec = _purse_spec(0)
        spec["pieces"].append({"id": "W", "value": 2, "moves": []})
        spec["rules"]["npcs"] = [{"piece": "W", "route": ["c3", "d3"], "bounty": 1.5}]
        assert any("whole coins" in error for error in validate_spec(spec))

    def test_both_players_start_with_the_amount_and_it_is_in_the_key(self) -> None:
        plain = Engine(_spec())
        rich = Engine(_purse_spec(7))
        assert plain.initial_state().purse == (0, 0)
        assert rich.initial_state().purse == (7, 7)
        assert len(compile_spec(_purse_spec(7)).purse.zobrist) == 2 * (PURSE_MAX + 1)
        placement = {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0)}
        keys = {_position(rich, placement, purse=purse).key
                for purse in ((0, 0), (1, 0), (0, 1), (5, 5), (99, 0))}
        assert len(keys) == 5
        assert _position(plain, placement, purse=(5, 5)).purse == (0, 0)

    def test_a_bounty_fills_the_purse_and_the_cap_loses_the_rest(self) -> None:
        engine = Engine(_purse_spec(0))
        placement = {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0), "f5": ("R", 1)}
        state = _position(engine, placement, purse=(10, 3))
        after = engine.apply(state, _move(engine, state, "f1", "f5"))
        assert after.purse == (14, 3) and after.points == (4.0, 0.0)
        assert after.key == engine._compute_hash(after)
        state = _position(engine, placement, purse=(PURSE_MAX - 1, 3))
        after = engine.apply(state, _move(engine, state, "f1", "f5"))
        assert after.purse == (PURSE_MAX, 3) and after.points == (4.0, 0.0)
        assert after.key == engine._compute_hash(after)

    def test_the_purse_round_trips_over_twenty_walks(self) -> None:
        """Twenty random walks under a purse: after every ply the purse is the oracle's (paid
        bounties in, capped, a buy's price out), the incremental key is the full scan's, and
        the state crosses back unchanged."""
        engine = Engine(_purse_spec(90, max_plies=80))
        paid_somewhere = False
        for walk in range(20):
            rng = random.Random(walk)
            state = engine.initial_state()
            for _ in range(80):
                moves = engine.legal_moves(state)
                if engine.terminal(state, moves) is not None:
                    break
                move = rng.choice(moves)
                after = engine.apply(state, move)
                paid = _paid(engine, state, after)
                mover = state.side_to_move
                expected = list(state.purse)
                expected[mover] = min(PURSE_MAX, expected[mover] + paid)
                if move_type(move) == MOVE_BUY:
                    expected[mover] -= _PRICE
                assert after.purse == tuple(expected)
                assert after.points[mover] == state.points[mover] + paid
                assert after.key == engine._compute_hash(after)
                paid_somewhere |= paid > 0
                state = after
            assert dataclasses.replace(state) == state
        assert paid_somewhere

    def test_unmake_puts_the_purse_back(self) -> None:
        """`perft` makes and unmakes on one position, reading the purse's log; a walk over
        `apply` builds every position fresh. Near the cap, where a read-back would be wrong."""
        engine = Engine(_purse_spec(0))
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0),
                                   "f5": ("R", 1), "c4": ("G", 1), "c2": ("G", 0)},
                          purse=(PURSE_MAX - 2, PURSE_MAX - 1))

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            if engine.terminal(position) is not None:
                return 0
            return sum(count(engine.apply(position, move), depth - 1)
                       for move in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_a_searching_game_under_a_purse_replays(self) -> None:
        engine = Engine(_purse_spec(97, max_plies=24))
        record = play_game(engine, ("minimax:nodes=300", "minimax:nodes=300"), 1)
        state = engine.initial_state()
        for move in record.moves:
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state)
        outcome = engine.terminal(state)
        assert outcome is not None and (outcome.winner, outcome.reason) == (
            record.winner, record.reason
        )

    def test_fairy_stockfish_refuses_the_purse(self) -> None:
        with pytest.raises(Untranslatable, match="purse"):
            translate(_purse_spec(3))


# --------------------------------------------------------------------------- inert


# The four wild-bounty games' tallies and keys on five random walks and two searching games,
# hashed; computed on main (1db928a1) before TC4 and unchanged by it (`docs/plan-turncoat.md`
# §9's TC4 acceptance: their tallies byte-equal).
_WILD_TALLIES = {
    "wolves-bounty": "f655a89d9030b954",
    "the-shepherds-duck": "7fd205ad36d5603a",
    "kraken-ford": "d0c14ca3058b6119",
    "the-hydra": "8d381ecf02c3a486",
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(_WILD_TALLIES))
def test_the_wild_bounty_games_play_as_they_did(name: str) -> None:
    spec = json.loads((ROOT / "archive" / f"{name}.json").read_text(encoding="utf-8"))
    engine = Engine(spec)
    digest = hashlib.sha256()
    for seed in range(5):
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(120):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            state = engine.apply(state, rng.choice(moves))
            digest.update(repr((state.points, state.key)).encode())
        digest.update(repr(engine.terminal(state)).encode())
    for seed in range(2):
        record = play_game(engine, ("greedy", "minimax:nodes=200"), seed)
        digest.update(repr((record.moves, record.winner, record.reason, record.material)).encode())
    assert digest.hexdigest()[:16] == _WILD_TALLIES[name]
