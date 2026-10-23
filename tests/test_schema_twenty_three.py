"""Schema 23 (Q850): several wild pieces on one walk, ``rules.npcs[].also_at``.

`docs/reference/designer-case-study-wolves.md` §2.2-2.3. An entry's ``also_at`` lists squares
of its own route, other than the first, where a further wild piece of the same kind stands at
ply zero. The three per-square tables are the route's, so the compiler changes only
``starts``; the core steps the pieces of one route **front first** (``npc_step_order`` in
``rust/src/game.rs``), so a queue moves as one when its front moves and waits as one when its
front waits, and a route full from end to end waits whole.

Four parts: the validator (one test per rule), the compiler (only ``starts`` moves), the core
(one hand-built position per stepping case, the legality filter and the coming-phase arrows),
and `_phase_oracle`, a declared test oracle stating the phase recursively -- a wild piece
steps if its next square is empty, holds a piece its type may take, or holds a wild piece
that steps -- held against the core over random walks of the showcase.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
from test_schema_nine import FROZEN

from chess_variant_sandbox import fuzz
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.coming import coming_phases
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_rules
from chess_variant_sandbox.spec.distance import distance
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION
from tests.feature_inventory import _schema_twenty_three_uses

ROOT = Path(__file__).resolve().parents[1]
PATROL = ROOT / "archive" / "wolves-patrol.json"
CIRCUIT = [
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "h5", "g5", "f5", "e5", "d5", "c5", "b5", "a5",
]


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _name(square: int, width: int = 8) -> str:
    return chr(ord("a") + square % width) + str(square // width + 1)


def _patrol() -> dict[str, Any]:
    return load_spec(PATROL)


def _with_npcs(*npcs: dict[str, Any], **piece: Any) -> dict[str, Any]:
    """The showcase with its ``rules.npcs`` replaced, and ``piece`` merged into the wolf."""
    spec = copy.deepcopy(_patrol())
    spec["rules"]["npcs"] = list(npcs)
    wolf = next(p for p in spec["pieces"] if p["id"] == "W")
    wolf.update(piece)
    return spec


def _wild_squares(engine: Engine, state: Any) -> list[str]:
    return sorted(
        _name(square)
        for square, value in enumerate(state.board)
        if value and engine.spec.owner_of(value) == WILD
    )


def _move(engine: Engine, state: Any, origin: str, target: str) -> int:
    found = [
        m for m in engine.legal_moves(state)
        if _name(m & 0xFF) == origin and _name((m >> 8) & 0xFF) == target
    ]
    assert len(found) == 1, f"{origin}-{target}: {len(found)} moves"
    return found[0]


# -- the version and the collector --------------------------------------------------------


class TestTheVersionIsBumped:
    def test_the_schema_version_is_twenty_three_or_later(self) -> None:
        assert SCHEMA_VERSION >= 23

    def test_without_also_at_the_collector_is_empty(self) -> None:
        spec = _with_npcs({"piece": "W", "route": CIRCUIT})
        assert _schema_twenty_three_uses(spec) == set()

    def test_the_collector_names_also_at(self) -> None:
        assert _schema_twenty_three_uses(_patrol()) == {"rules.npcs.also_at"}


# -- the validator, one test per rule -----------------------------------------------------


def _problems(**entry: Any) -> list[str]:
    return validate_spec(_with_npcs({"piece": "W", "route": CIRCUIT, **entry}))


class TestTheValidator:
    def test_the_case_study_spelling_is_valid(self) -> None:
        assert _problems(also_at=["h5"]) == []

    def test_several_starts_in_route_order_are_valid(self) -> None:
        assert _problems(also_at=["e4", "h5", "d5"]) == []

    def test_also_at_must_be_a_non_empty_list(self) -> None:
        assert any("also_at must be a non-empty array" in p for p in _problems(also_at=[]))
        assert any("also_at must be a non-empty array" in p for p in _problems(also_at="h5"))

    def test_a_square_off_the_route_is_refused(self) -> None:
        assert any("not on its route" in p for p in _problems(also_at=["e6"]))

    def test_a_square_off_the_board_is_refused(self) -> None:
        assert any("not on its route" in p for p in _problems(also_at=["z9"]))

    def test_a_non_square_is_refused(self) -> None:
        assert any("is not a square" in p for p in _problems(also_at=[3]))

    def test_the_first_square_is_refused(self) -> None:
        assert any("the route's first square" in p for p in _problems(also_at=["a4"]))

    def test_a_square_named_twice_is_refused(self) -> None:
        assert any("twice" in p for p in _problems(also_at=["h5", "h5"]))

    def test_squares_out_of_route_order_are_refused(self) -> None:
        problems = _problems(also_at=["d5", "h5"])
        assert any("in route order (h5, d5)" in p for p in problems), problems

    def test_a_start_under_the_opening_array_is_refused(self) -> None:
        spec = _with_npcs({"piece": "W", "route": ["a4", "a3", "a2"], "also_at": ["a2"]})
        problems = validate_spec(spec)
        assert any("also_at a2 starts on a2" in p for p in problems), problems

    def test_a_mirrored_start_under_the_opening_array_is_refused(self) -> None:
        # The chess mirror of a4 b4 b2 is a5 b5 b7, and b7 holds Black's pawn.
        spec = _with_npcs(
            {"piece": "W", "route": ["a4", "b4", "b2"], "also_at": ["b2"], "mirror": True}
        )
        problems = validate_spec(spec)
        assert any("also_at b2 (mirrored) starts on b7" in p for p in problems), problems

    def test_two_entries_sharing_squares_are_still_refused_and_pointed_at_also_at(self) -> None:
        rotated = CIRCUIT[8:] + CIRCUIT[:8]
        problems = validate_spec(_with_npcs(
            {"piece": "W", "route": CIRCUIT}, {"piece": "W", "route": rotated}
        ))
        assert len(problems) == 16
        assert all("also_at" in p for p in problems)

    def test_also_at_beside_a_mirror_is_valid(self) -> None:
        assert validate_spec(_with_npcs(
            {"piece": "W", "route": ["a4", "b5", "c4", "b3", "a3"], "also_at": ["c4"],
             "mirror": True},
        )) == []


# -- the compiler: only `starts` moves -----------------------------------------------------


class TestTheCompiler:
    def test_the_starts_are_a4_then_h5(self) -> None:
        npcs = compile_spec(_patrol()).npcs
        assert npcs is not None
        wolf = compile_spec(_patrol()).piece_ids.index("W")
        assert npcs.starts == ((_square("a4"), wolf), (_square("h5"), wolf))

    def test_the_route_tables_are_the_one_wolf_circuits(self) -> None:
        one = compile_spec(_with_npcs({"piece": "W", "route": CIRCUIT})).npcs
        two = compile_spec(_patrol()).npcs
        assert one is not None and two is not None
        assert (one.next, one.bounty, one.piece) == (two.next, two.bounty, two.piece)

    def test_the_tables_differ_only_in_the_opening_board(self) -> None:
        one = canonical_tables(compile_spec(_with_npcs({"piece": "W", "route": CIRCUIT})))
        two = canonical_tables(compile_spec(_patrol()))
        # The Zobrist draws are seeded by the mechanics hash, which `also_at` moves.
        differ = {key for key in one if one[key] != two[key] and not key.startswith("zobrist")}
        assert differ == {"initial_board"}

    def test_a_mirrored_entrys_starts_follow_its_originals(self) -> None:
        spec = _with_npcs(
            {"piece": "W", "route": ["a4", "b5", "c4", "b3", "a3"], "also_at": ["c4"],
             "mirror": True},
        )
        npcs = compile_spec(spec).npcs
        assert npcs is not None
        assert [_name(square) for square, _ in npcs.starts] == ["a4", "c4", "a5", "c5"]

    def test_the_opening_position_holds_both_wolves(self) -> None:
        engine = Engine(_patrol())
        assert _wild_squares(engine, engine.initial_state()) == ["a4", "h5"]


# -- the core: one hand-built position per stepping case -----------------------------------

KINGS = {"a1": ("K", 0), "h8": ("K", 1)}


def _line(route: list[str], **piece: Any) -> Engine:
    """A wolf route of the caller's, the wolf merged with ``piece``."""
    return Engine(_with_npcs({"piece": "W", "route": route, "also_at": route[1:2]}, **piece))


def _after_a_quiet_move(engine: Engine, placement: dict[str, tuple[str, int]]) -> list[str]:
    state = engine.make_state({**KINGS, **placement}, moved=tuple(placement))
    return _wild_squares(engine, engine.apply(state, _move(engine, state, "a1", "b1")))


class TestTheCoreStepsFrontFirst:
    def test_a_queue_moves_as_one_when_the_front_has_the_higher_index(self) -> None:
        """Ascending order alone would make the back piece (c3) wait for the front (c4):
        this is the case the bump changed."""
        engine = _line(["c3", "c4", "c5", "c6"])
        placed = {"c3": ("W", WILD), "c4": ("W", WILD)}
        assert _after_a_quiet_move(engine, placed) == ["c4", "c5"]

    def test_a_queue_moves_as_one_when_the_front_has_the_lower_index(self) -> None:
        engine = _line(["c6", "c5", "c4", "c3"])
        placed = {"c6": ("W", WILD), "c5": ("W", WILD)}
        assert _after_a_quiet_move(engine, placed) == ["c4", "c5"]

    def test_a_queue_waits_as_one_when_the_front_waits(self) -> None:
        engine = _line(["c3", "c4", "c5", "c6"], captures=[])
        placed = {"c3": ("W", WILD), "c4": ("W", WILD), "c5": ("R", 0)}
        assert _after_a_quiet_move(engine, placed) == ["c3", "c4"]

    def test_a_queue_follows_a_front_that_eats(self) -> None:
        engine = _line(["c3", "c4", "c5", "c6"])
        state = engine.make_state(
            {**KINGS, "c3": ("W", WILD), "c4": ("W", WILD), "c5": ("R", 1)},
            moved=("c3", "c4", "c5"),
        )
        after, steps = engine.apply_with_phase(state, _move(engine, state, "a1", "b1"))
        assert _wild_squares(engine, after) == ["c4", "c5"]
        assert [(_name(s.origin), _name(s.target), bool(s.eaten)) for s in steps] == [
            ("c4", "c5", True), ("c3", "c4", False),
        ]

    def test_a_gap_closes_and_then_the_two_walk_in_step(self) -> None:
        engine = _line(["c3", "c4", "c5", "c6", "d6", "d5"], captures=[])
        state = engine.make_state(
            {**KINGS, "c3": ("W", WILD), "c5": ("W", WILD), "c6": ("R", 0)},
            moved=("c3", "c5", "c6"),
        )
        state = engine.apply(state, _move(engine, state, "a1", "b1"))
        assert _wild_squares(engine, state) == ["c4", "c5"]  # front waits, back closes up
        state = engine.apply(state, _move(engine, state, "h8", "g8"))
        assert _wild_squares(engine, state) == ["c4", "c5"]  # both wait
        state = engine.apply(state, _move(engine, state, "c6", "a6"))
        assert _wild_squares(engine, state) == ["c5", "c6"]  # both walk

    def test_a_full_route_waits_whole(self) -> None:
        engine = Engine(_with_npcs({"piece": "W", "route": ["c3", "c4", "c5"],
                                    "also_at": ["c4", "c5"]}))
        state = engine.initial_state()
        assert _wild_squares(engine, state) == ["c3", "c4", "c5"]
        for _ in range(3):
            state = engine.apply(state, engine.legal_moves(state)[0])
            assert _wild_squares(engine, state) == ["c3", "c4", "c5"]

    def test_the_two_wolves_of_the_showcase_walk_the_circuit_half_a_lap_apart(self) -> None:
        engine = Engine(_patrol())
        state = engine.make_state(
            {**KINGS, "a4": ("W", WILD), "h5": ("W", WILD)}, moved=("a4", "h5")
        )
        for step in range(1, 17):
            side = state.side_to_move
            king = next(
                square for square, value in enumerate(state.board)
                if value and engine.spec.owner_of(value) == side
            )
            target = next(m for m in engine.legal_moves(state) if (m & 0xFF) == king)
            state = engine.apply(state, target)
            assert _wild_squares(engine, state) == sorted(
                [CIRCUIT[step % 16], CIRCUIT[(step + 8) % 16]]
            )


class TestTheFilterAndTheArrowsUseTheSameOrder:
    def _discovered(self) -> tuple[Engine, Any]:
        """White king on h3, a Black rook on a3, wolves on c3 (back) and c4 (front) of
        c3 c4 c5 c6. After White's move the queue walks, c3 empties, and the rook sees the
        king along the third rank -- unless White's move blocked it."""
        engine = _line(["c3", "c4", "c5", "c6"])
        state = engine.make_state(
            {"h3": ("K", 0), "h8": ("K", 1), "a3": ("R", 1), "e2": ("P", 0),
             "c3": ("W", WILD), "c4": ("W", WILD)},
            moved=("h3", "h8", "a3", "c3", "c4"),
        )
        return engine, state

    def test_a_move_that_leaves_the_king_to_the_rook_is_illegal(self) -> None:
        engine, state = self._discovered()
        legal = {
            (_name(m & 0xFF), _name((m >> 8) & 0xFF)) for m in engine.legal_moves(state)
        }
        assert ("e2", "e3") in legal
        assert ("e2", "e4") not in legal
        assert ("h3", "g3") not in legal
        assert ("h3", "h2") in legal

    def test_the_coming_phase_steps_the_front_first(self) -> None:
        engine, state = self._discovered()
        first, second = coming_phases(engine, state)
        assert [(_name(s.origin), _name(s.target)) for s in first] == [
            ("c4", "c5"), ("c3", "c4"),
        ]
        assert [(_name(s.origin), _name(s.target)) for s in second] == [
            ("c5", "c6"), ("c4", "c5"),
        ]


# -- the oracle --------------------------------------------------------------------------


def _phase_oracle(engine: Engine, board: bytes) -> list[str]:
    """Where the wild pieces stand after one phase on ``board``, stated recursively.

    A wild piece steps when its next square is empty, holds a player's piece its type may
    take, or holds a wild piece that itself steps; a chain that comes back to where it
    started (a full route) does not step. Independent of the core's passes: this asks the
    question per piece, the core answers it per pass.
    """
    spec = engine.spec
    npcs = spec.npcs
    assert npcs is not None
    wild = {
        square for square, value in enumerate(board)
        if value and spec.owner_of(value) == WILD
    }

    def steps(square: int, seen: frozenset[int]) -> bool:
        target = npcs.next[square]
        if target in (-1, square) or target in seen:
            return False
        if board[target] == 0:
            return True
        if target in wild:
            return steps(target, seen | {square})
        mover, victim = spec.type_of(board[square]), spec.type_of(board[target])
        return bool(spec.capture_allowed[mover][victim])

    return sorted(
        _name(npcs.next[square] if steps(square, frozenset()) else square, spec.width)
        for square in wild
    )


def _before_phase(engine: Engine, after: Any, steps: tuple[Any, ...]) -> bytes:
    board = bytearray(after.board)
    for step in reversed(steps):
        board[step.origin] = board[step.target]
        board[step.target] = step.eaten
    return bytes(board)


def _shy_patrol() -> dict[str, Any]:
    """The showcase with wolves that eat only pawns: a wolf waits before any other piece,
    so the one behind closes up and queues form. In the showcase itself a wolf eats
    anything but a wolf, so the two stay half a lap apart until one is taken."""
    return _with_npcs({"piece": "W", "route": CIRCUIT, "also_at": ["h5"]}, captures=["P"])


def _walk(seed: int, games: int, plies: int, spec: dict[str, Any] | None = None) -> int:
    engine = Engine(spec or _patrol())
    rng = random.Random(seed)
    queued = 0
    for _ in range(games):
        state = engine.initial_state()
        for _ in range(plies):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None or not moves:
                break
            after, steps = engine.apply_with_phase(state, rng.choice(moves))
            before = _before_phase(engine, after, steps)
            assert _wild_squares(engine, after) == _phase_oracle(engine, before)
            # Two wolves a step apart: the case the front-first order decides.
            wolves = [s for s, v in enumerate(before) if v and engine.spec.owner_of(v) == WILD]
            queued += any(engine.spec.npcs.next[a] == b for a in wolves for b in wolves)
            assert after.key == engine._compute_hash(after)
            state = after
    return queued


class TestTheOracleAgrees:
    def test_random_walks_of_the_showcase(self) -> None:
        _walk(23, 4, 80)

    def test_random_walks_where_queues_form(self) -> None:
        assert _walk(32, 4, 80, _shy_patrol()) > 0

    def test_hand_built_queues(self) -> None:
        for route, placed, wall in [
            (["c3", "c4", "c5", "c6"], {"c3": ("W", WILD), "c4": ("W", WILD)}, False),
            (["c6", "c5", "c4", "c3"], {"c6": ("W", WILD), "c5": ("W", WILD)}, False),
            (["c3", "c4", "c5", "c6"],
             {"c3": ("W", WILD), "c4": ("W", WILD), "c5": ("R", 0)}, True),
        ]:
            engine = _line(route, **({"captures": []} if wall else {}))
            state = engine.make_state({**KINGS, **placed}, moved=tuple(placed))
            after, steps = engine.apply_with_phase(state, _move(engine, state, "a1", "b1"))
            assert _wild_squares(engine, after) == _phase_oracle(
                engine, _before_phase(engine, after, steps)
            )

    @pytest.mark.slow
    def test_long_random_walks(self) -> None:
        _walk(2323, 12, 200)
        assert _walk(3232, 12, 200, _shy_patrol()) > 0

    def test_the_fuzz_oracles_hold_on_the_showcase(self) -> None:
        for seed in range(3):
            fuzz.walk(_patrol(), seed, plies=120)


# -- the other surfaces ----------------------------------------------------------------------


class TestTheOtherSurfaces:
    def test_describe_names_both_starts(self) -> None:
        text = " ".join(describe_rules(_patrol()))
        assert "starts on a4" in text and "h5" in text
        assert "behind" in text

    def test_distance_sees_the_second_wolf(self) -> None:
        one = _with_npcs({"piece": "W", "route": CIRCUIT})
        assert distance(one, _patrol()) > 0

    def test_fairy_refuses_it_whole(self) -> None:
        with pytest.raises(Untranslatable):
            translate(_patrol())

    def test_the_showcase_file_is_the_case_studys_spelling(self) -> None:
        npcs = json.loads(PATROL.read_text(encoding="utf-8"))["rules"]["npcs"]
        assert npcs == [{"piece": "W", "route": CIRCUIT, "also_at": ["h5"]}]


# -- the inertness proof ---------------------------------------------------------------------


class TestTheBumpMovedNothing:
    def test_every_hash_is_the_one_frozen(self) -> None:
        for path, (digest, _) in FROZEN.items():
            assert spec_hash(load_spec(ROOT / path)) == digest, path

    def test_only_the_showcase_uses_schema_twenty_three(self) -> None:
        assert [
            path for path in FROZEN if _schema_twenty_three_uses(load_spec(ROOT / path))
        ] == [
            "archive/wolves-patrol.json",
            "archive/pitchfork.json",  # Q876: its pairs
            "archive/the-golem-well.json",  # Q994p: two golems on one ring (also_at)
        ]


# Frozen from the schema 23 core, 2026-10-11 (no earlier core can play the game). Rungs
# over `FAST_BUDGET_NODES` are slow, as in `test_variants.py`.
FAST_BUDGET_NODES = 150_000
PATROL_PERFT = {1: 19, 2: 361, 3: 7_564, 4: 159_876}


class TestThePerftIsFrozen:
    @pytest.mark.parametrize(("depth", "expected"), [
        pytest.param(depth, expected, marks=[pytest.mark.slow] if expected > FAST_BUDGET_NODES else [])
        for depth, expected in sorted(PATROL_PERFT.items())
    ])
    def test_perft(self, depth: int, expected: int) -> None:
        engine = Engine(_patrol())
        assert engine.perft(engine.initial_state(), depth) == expected
