"""SR8j (`docs/plan-schema-next.md` §2.11 O15, *the mover's reading*, ruled 2026-10-28): a held
piece promotes.

A held piece entering its **holder's** promotion zone promotes as the holder's own piece of the
type would, to a target the holder's promotion offers and the holder chooses (`only_lost`
counted for the holder), never a royal or a type no player owns (a wild type, the duck), and
stays nobody's under the target type's ``control``: it stands, or is held by the count, or is
shared. It enters play at its type's initial facing in nobody's frame. The core offers it
(`foreign.rs::push_held_promotions`) and writes it (`Tables::promoted_value`);
`tests/neutral_oracle.py` reads the same moves off the holder's twin and is held to the core here,
list by list and by perft.

Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
import tomllib

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_NORMAL,
    MOVE_PROMOTION,
    move_foreign,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.spec import refusals
from chess_variant_sandbox.spec.schema import held_promotion_targets, nobodys_ids, validate_spec
from tests.neutral_oracle import NeutralOracle
from tests.test_held_in_holders_frame import (
    KING,
    PAWN,
    _engine,
    _oracle_perft,
    _position,
    _spec,
    _sq,
    _walk,
)

HERE = Path(__file__).parent


def _promoting(**promotion: Any) -> dict[str, Any]:
    """The held pawn ``H`` promoting on its holder's last rank, to the rook, the guard or the
    king (a royal, never offered to a held piece); ``promotion`` adds to the entry."""
    spec = _spec({"moves": copy.deepcopy(PAWN)})
    spec["rules"]["promotion"] = [
        {"piece": "H", "rank": "last", "to": ["R", "G", "K"], **promotion},
    ]
    return spec


def _into_a_held_type() -> dict[str, Any]:
    """The held pawn promotes into ``M``, a type held by the count beside it, or the rook."""
    spec = _promoting()
    spec["pieces"].append({"id": "M", "value": 4, "control": "majority", "moves": [copy.deepcopy(KING)]})
    spec["rules"]["promotion"][0]["to"] = ["M", "R"]
    return spec


def _into_a_shared_type() -> dict[str, Any]:
    spec = _into_a_held_type()
    spec["pieces"][-1]["control"] = "shared"
    return spec


def _into_an_arrow() -> dict[str, Any]:
    """The held pawn promotes into ``A``, a type that faces (initially forward) and turns."""
    spec = _promoting()
    spec["pieces"].append({
        "id": "A", "value": 3, "moves": [{"kind": "leap", "vectors": [[0, 1]]}],
        "facing": {"axes": 4, "initial": "forward"},
    })
    spec["rules"]["promotion"][0]["to"] = ["A"]
    return spec


def _shared_pawn() -> dict[str, Any]:
    spec = _promoting()
    spec["pieces"][3]["control"] = "shared"
    return spec


def _optional() -> dict[str, Any]:
    return _promoting(mandatory=False)


def _only_lost() -> dict[str, Any]:
    return _promoting(only_lost=True)


ALL_SPECS = {
    "mandatory": _promoting, "optional": _optional, "only lost": _only_lost,
    "into a held type": _into_a_held_type, "into a shared type": _into_a_shared_type,
    "into an arrow": _into_an_arrow, "shared pawn": _shared_pawn,
}


def _held_from(engine: Engine, state, square: str) -> list[int]:
    return sorted(
        m for m in engine.legal_moves(state) if move_foreign(m) and move_from(m) == _sq(square)
    )


def _promotions(engine: Engine, state, square: str) -> dict[str, int]:
    """The held piece on ``square``'s promotions, by the id of what it becomes."""
    return {
        engine.spec.piece_ids[move_promotion(m)]: m
        for m in _held_from(engine, state, square) if move_type(m) == MOVE_PROMOTION
    }


# The held pawn one step from its holder's last rank: c5 for the first player, c2 for the second.
NEAR = {0: "c5", 1: "c2"}
LAST = {0: "c6", 1: "c1"}


def _near(engine: Engine, holder: int, **extra: tuple[str, int]):
    near = NEAR[holder]
    guard = "b5" if holder == 0 else "b2"
    placement = {"a1": ("K", 0), "f6": ("K", 1), guard: ("G", holder), near: ("H", 2), **extra}
    return _position(engine, placement, side=holder)


class TestTheRefusalGoes:
    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_each_spec_validates(self, name):
        assert validate_spec(ALL_SPECS[name]()) == []

    def test_the_row_is_lifted(self):
        table = tomllib.loads(
            (Path(refusals.__file__).parent / "refusals.toml").read_text(encoding="utf-8")
        )
        assert "O15" not in {row["id"] for row in table["refusal"]}
        assert "O15" in {row["id"] for row in table["lifted"] if row.get("by") == "SR8j"}

    def test_a_held_sweep_is_lifted_too(self):
        """SR8k (O16): a held sweeper promotes as any held piece (`tests/test_held_sweep.py`)."""
        spec = _promoting()
        spec["pieces"][3]["moves"] = [{"kind": "sweep", "vectors": [[0, 1]], "mode": "both"}]
        assert validate_spec(spec) == []


class TestWhatItMayBecome:
    def test_never_a_royal_or_a_type_no_player_owns(self):
        spec = _promoting()
        spec["rules"]["promotion"][0]["to"] = ["R", "G", "K", "W"]
        spec["pieces"].append({"id": "W", "value": 1, "moves": [copy.deepcopy(KING)]})
        spec["rules"]["npcs"] = [{"piece": "W", "route": ["a3", "b3"]}]
        assert held_promotion_targets(spec, "H") == {"R", "G"}

    def test_what_it_becomes_is_nobodys(self):
        assert nobodys_ids(_promoting()) == {"H", "R", "G"}

    def test_an_unpromoting_game_crosses_no_table(self):
        spec = _spec({"moves": copy.deepcopy(PAWN)})
        assert build_tables(compile_spec(spec))["neutral_held_promotion"] == []

    def test_the_table_names_the_offered_targets(self):
        table = build_tables(compile_spec(_promoting()))["neutral_held_promotion"]
        # K, R, G, H for each holder: the rook and the guard at facing 0 (plus one).
        assert table == [0, 1, 1, 0, 0, 1, 1, 0]


class TestThePromotion:
    @pytest.mark.parametrize("holder", [0, 1])
    def test_in_its_holders_zone_to_the_holders_targets(self, holder):
        engine = _engine(_promoting())
        state = _near(engine, holder)
        assert engine.controller_of(state, _sq(NEAR[holder])) == holder
        promotions = _promotions(engine, state, NEAR[holder])
        assert set(promotions) == {"R", "G"}
        assert {move_to(m) for m in promotions.values()} == {_sq(LAST[holder])}
        # Mandatory: no plain step onto the last rank.
        plain = [m for m in _held_from(engine, state, NEAR[holder]) if move_type(m) == MOVE_NORMAL]
        assert _sq(LAST[holder]) not in {move_to(m) for m in plain}

    @pytest.mark.parametrize("holder", [0, 1])
    def test_it_stays_nobodys_as_the_new_type(self, holder):
        engine = _engine(_promoting())
        oracle = NeutralOracle(_promoting(), engine)
        state = _near(engine, holder)
        rook = _promotions(engine, state, NEAR[holder])["R"]
        after = engine.apply(state, rook)
        value = after.board[_sq(LAST[holder])]
        assert value == oracle.promoted_byte(holder, engine.spec.piece_ids.index("R"))
        assert engine.spec.owner_of(value) == 2
        assert engine._compute_hash(after) == after.key
        # A rook nobody owns stands: nobody moves it, the holder included.
        again = _position(engine, {
            "a1": ("K", 0), "f6": ("K", 1), "b5": ("G", 0), "b6": ("G", 0), LAST[holder]: ("R", 2),
        }, side=0)
        assert not _held_from(engine, again, LAST[holder])

    def test_optional_adds_the_plain_step(self):
        engine = _engine(_optional())
        state = _near(engine, 0)
        moves = _held_from(engine, state, NEAR[0])
        assert {move_type(m) for m in moves if move_to(m) == _sq(LAST[0])} == {
            MOVE_NORMAL, MOVE_PROMOTION,
        }

    def test_only_lost_counts_the_holders_pieces(self):
        engine = _engine(_only_lost())
        # The first player started with one rook and one guard and has both: neither target
        # is short, so no promotion, and under a mandatory promotion no step onto the last
        # rank either.
        full = _near(engine, 0, f1=("R", 0))
        assert _promotions(engine, full, NEAR[0]) == {}
        # Without its rook, the rook is offered; the second player's rook does not count.
        short = _near(engine, 0, f5=("R", 1))
        assert set(_promotions(engine, short, NEAR[0])) == {"R"}

    def test_into_a_held_type_it_is_held_by_the_count_after(self):
        engine = _engine(_into_a_held_type())
        state = _near(engine, 0)
        after = engine.apply(state, _promotions(engine, state, NEAR[0])["M"])
        square = _sq(LAST[0])
        assert engine.spec.owner_of(after.board[square]) == 2
        # The first player's guard on b5 is next to c6: the M is theirs to move next time.
        assert engine.controller_of(after, square) == 0

    def test_into_a_shared_type_either_player_moves_it(self):
        engine = _engine(_into_a_shared_type())
        state = _near(engine, 0)
        after = engine.apply(state, _promotions(engine, state, NEAR[0])["M"])
        assert any(
            move_foreign(m) and move_from(m) == _sq(LAST[0]) for m in engine.legal_moves(after)
        )

    @pytest.mark.parametrize("holder", [0, 1])
    def test_it_enters_play_at_its_types_initial_facing_in_nobodys_frame(self, holder):
        """Forward in nobody's frame (the first player's), whoever held it, as a nobody's
        placement that states no facing stands."""
        engine = _engine(_into_an_arrow())
        state = _near(engine, holder)
        after = engine.apply(state, _promotions(engine, state, NEAR[holder])["A"])
        value = after.board[_sq(LAST[holder])]
        assert engine.spec.owner_of(value) == 2 and engine.spec.facing_of(value) == 0

    def test_a_capture_promotes_too(self):
        engine = _engine(_promoting())
        state = _near(engine, 0, d6=("R", 1), b6=("G", 0))
        promotions = [
            m for m in _held_from(engine, state, NEAR[0])
            if move_type(m) == MOVE_PROMOTION and move_to(m) == _sq("d6")
        ]
        assert len(promotions) == 2


# ------------------------------------------------------------------ the oracle


def _walk_spec(make) -> dict[str, Any]:
    """Each player holds a nobody's pawn one step from its last rank: c5 by the first player's
    guard on b4, its image c2 by the second's on b3."""
    spec = make()
    nobodys = spec["setup"]["placements"][3]
    spec["setup"]["placements"] = [
        {"piece": "K", "squares": ["a1"]},
        {"piece": "R", "squares": ["f1"]},
        {"piece": "G", "squares": ["a2", "b4", "f2"]},
        {**nobodys, "squares": ["c5"]},
    ]
    return spec


class TestTheOracle:
    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_walks_agree(self, name):
        spec = _walk_spec(ALL_SPECS[name])
        assert sum(_walk(spec, seed, 40) for seed in range(2)) > 0

    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_a_perft_agrees(self, name):
        spec = _walk_spec(ALL_SPECS[name])
        engine = _engine(spec)
        oracle = NeutralOracle(spec, engine)
        state = engine.initial_state()
        held = [m for m in oracle.legal_moves(state) if move_foreign(m)]
        # Under `only_lost` nothing is short at the opening, so the held pawns cannot move.
        assert bool(held) is (name != "only lost")
        assert engine.perft(state, 3) == _oracle_perft(engine, oracle, state, 3)

    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_every_held_promotion_writes_the_oracles_byte(self, name):
        spec = _walk_spec(ALL_SPECS[name])
        engine = _engine(spec)
        oracle = NeutralOracle(spec, engine)
        rng = random.Random(7)
        state = engine.initial_state()
        seen = 0
        for _ in range(60):
            if engine.terminal(state) is not None:
                break
            moves = sorted(engine.legal_moves(state))
            for m in moves:
                if move_foreign(m) and move_type(m) == MOVE_PROMOTION:
                    after = engine.apply(state, m)
                    want = oracle.promoted_byte(state.side_to_move, move_promotion(m))
                    assert after.board[move_to(m)] == want
                    assert engine._compute_hash(after) == after.key
                    seen += 1
            state = engine.apply(state, rng.choice(moves))
        assert seen > 0 or name == "only lost"

    def test_in_rounds_a_held_promotion_stays_nobodys(self):
        """SR7n's held condottieri in rounds, promoting in the middle two ranks: every round
        against the declared round oracle, and each held promotion played leaves a piece
        nobody owns of the type it named."""
        from tests.simultaneous_oracle import resolve_round

        spec = json.loads((HERE / "_simultaneous_held.json").read_text(encoding="utf-8"))
        spec["rules"]["promotion"] = {"piece": "C", "ranks": [2, 3], "to": ["R", "G", "K"]}
        engine = _engine(spec)
        promoted = 0
        for seed in range(6):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(30):
                if engine.terminal(state) is not None:
                    break
                lists = [engine.legal_moves_for(state, player) for player in (0, 1)]
                if not all(lists):
                    break
                held = [[m for m in moves if move_foreign(m)] for moves in lists]
                pair = [rng.choice(h) if h and rng.random() < 0.7 else rng.choice(m)
                        for h, m in zip(held, lists, strict=True)]
                for m in pair:
                    if move_foreign(m) and move_type(m) == MOVE_PROMOTION:
                        assert engine.spec.piece_ids[move_promotion(m)] in ("R", "G")
                after, _, _ = engine.apply_round_with_phase(state, *pair)
                assert after == resolve_round(engine, state, *pair)
                assert after.key == engine._compute_hash(after)
                for m in pair:
                    if move_foreign(m) and move_type(m) == MOVE_PROMOTION:
                        value = after.board[move_to(m)]
                        if value and engine.spec.type_of(value) == move_promotion(m):
                            assert engine.spec.owner_of(value) == 2
                            promoted += 1
                state = after
        assert promoted > 0

    @pytest.mark.slow
    @pytest.mark.parametrize("name", sorted(ALL_SPECS))
    def test_long_walks_agree(self, name):
        spec = _walk_spec(ALL_SPECS[name])
        assert sum(_walk(spec, seed, 100) for seed in range(10)) > 10


# ------------------------------------------------------------------ beside portals


def _through_portals(moves: list[int], origin: int) -> set[int]:
    """The moves from ``origin`` through a portal: transits (promoting or not) and fold
    go-throughs (plain or promoting), less the foreign bit."""
    from chess_variant_sandbox.engine.moves import MOVE_FOREIGN, MOVE_PORTAL, move_fold

    return {
        m & ~MOVE_FOREIGN for m in moves
        if move_from(m) == origin and (
            move_type(m) == MOVE_PORTAL
            or (move_type(m) in (MOVE_NORMAL, MOVE_PROMOTION) and move_fold(m))
        )
    }


class TestBesidePortals:
    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("game", ["momentum", "fold", "both"])
    def test_a_transit_into_the_holders_zone_promotes_as_the_twins(self, side, game):
        """SR3g's crowd at the portals, the condottiere promoting in its holder's far half: a
        held piece's transits and go-throughs are its twin's (the holder's own condottiere),
        promotions included, but never into the king."""
        from chess_variant_sandbox.engine.compile import WILD
        from tests.test_nobody_beside_portals import ENDS, GAMES, _crowd, _name

        def promoting(majority: bool) -> dict[str, Any]:
            spec = _crowd(GAMES[game], majority=majority)
            spec["rules"]["promotion"] = {"piece": "C", "ranks": [2, 3, 4, 5], "to": ["R", "K"]}
            return spec

        engine, twin = _engine(promoting(True)), _engine(promoting(False))
        holders = ("b2", "c2") if side == 0 else ("b3", "d2")
        placement = {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0), "f6": ("R", 1),
                     "c3": ("C", WILD), "c4": ("S", WILD)}
        if game != "fold":
            placement |= ENDS
        for square in holders:
            placement[square] = ("R", side)
        state = engine.make_state(placement, side_to_move=side, moved=list(placement))
        mirror_placement = dict(placement)
        mirror_placement["c3"] = ("C", side)
        mirror = twin.make_state(mirror_placement, side_to_move=side, moved=list(placement))
        king = engine.spec.piece_ids.index("K")
        expected = {
            m for m in _through_portals(twin.pseudo_legal_moves(mirror), _sq("c3"))
            if not (move_type(m) == MOVE_PROMOTION and move_promotion(m) == king)
            and not (move_type(m) != MOVE_PROMOTION and m >> 20 & 0x1F == king + 1)
            and (not state.board[move_to(m)] or (
                engine.spec.owner_of(state.board[move_to(m)]) == 1 - side
                and engine.spec.type_of(state.board[move_to(m)]) not in engine.spec.royal_types
            ))
        }
        held = [m for m in engine.pseudo_legal_moves(state) if move_foreign(m)]
        got = _through_portals(held, _sq("c3"))
        assert got == expected, (_name(_sq("c3")), sorted(got ^ expected))
        promoting_words = [
            m for m in got
            if move_type(m) == MOVE_PROMOTION or (move_type(m) != MOVE_NORMAL and m >> 20 & 0x1F)
        ]
        # The second holder's exits (d6, the far fold end) lie outside its zone: no promotion.
        assert bool(promoting_words) is (side == 0)
        for m in held:
            if m & ~(1 << 30) in promoting_words:
                after = engine.apply(state, m)
                assert engine._compute_hash(after) == after.key
                assert engine.spec.owner_of(after.board[move_to(m)]) == WILD


class TestTheLoopReachesIt:
    """The second of the two edits by which the loop reaches the pair (no stored game is one
    edit from it: `tests/test_operators.py::_CANNOT_ENTER`): a promoting type standing for
    nobody, handed to the majority by `gather-a-crowd`."""

    def test_gather_a_crowd_hands_a_standing_promoting_type_to_the_majority(self):
        from chess_variant_sandbox.loop.operators import OPERATORS
        from tests.feature_inventory import _schema_one_hundred_twenty_seven_uses as uses

        apply = next(op for op in OPERATORS if op.name == "gather-a-crowd")
        crowd = HERE.parent / "archive" / "the-crowd.json"
        parent = json.loads(crowd.read_text(encoding="utf-8"))
        condottiere = next(p for p in parent["pieces"] if p["id"] == "C")
        del condottiere["control"], condottiere["takes_royal"]
        parent["rules"]["promotion"] = [
            parent["rules"]["promotion"], {"piece": "C", "ranks": [6, 7], "to": ["R"]},
        ]
        del parent["rules"]["en_passant"]  # defined against a single promoting piece
        assert validate_spec(parent) == [], validate_spec(parent)
        assert not uses(parent)
        for seed in range(30):
            child = copy.deepcopy(parent)
            if apply.apply(child, random.Random(seed)) is None:
                continue
            assert validate_spec(child) == [], validate_spec(child)
            if uses(child):
                return
        pytest.fail("gather-a-crowd never entered the pair in 30 seeds")
