"""TC2 (`docs/plan-turncoat.md` §6): a piece that changes sides on the board keeps its heading.

Its colour changes; its arrow does not. Facing indices are relative to the owner's forward,
so keeping the heading is a new index, `compile._coat_facing`'s, read by the core's
`Tables::rehome_value` for Andernach's switch and a contact flip. Before TC2 a switching
capturer took its type's initial facing for its new owner; every test here pins a position
where the two readings differ, so each one says which reading the core plays.

Hand-built specs, not chess: a 5x5 board, two kings, a one-vector lancer that faces, and a
symmetric guard that does not.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import _coat_facing, compile_spec
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.schema import validate_spec
from tests.simultaneous_oracle import resolve_round as oracle_round

FORWARD, RIGHT, BACKWARD, LEFT = "forward", "right", "backward", "left"


def _square(name: str) -> int:
    return (int(name[1:]) - 1) * 5 + ord(name[0]) - ord("a")


def _spec(
    *,
    axes: int = 4,
    orientation: str = "mirror",
    facing: str = FORWARD,
    victim: str = "c3",
    turns: bool = True,
    extra: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """White's lancer on c2 at `facing`, a black guard on `victim`, Andernach for both types."""
    lancer_facing: dict[str, Any] = {"axes": axes}
    if not turns:
        lancer_facing["turns"] = False
    spec: dict[str, Any] = {
        "schema_version": 31,
        "name": "turncoat lancers",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {
                "id": "K",
                "royal": True,
                "value": 0,
                "moves": [
                    {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
                ],
            },
            {
                "id": "L",
                "value": 2,
                "facing": lancer_facing,
                "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 2}],
            },
            {
                "id": "G",
                "value": 1,
                "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "dihedral8"}],
            },
        ],
        "setup": {
            "mirror": False,
            "orientation": orientation,
            "placements": [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "L", "player": 0, "squares": ["c2"], "facing": facing},
                {"piece": "G", "player": 1, "squares": [victim]},
                *(extra or []),
            ],
        },
        "rules": {
            "win": "checkmate",
            "max_plies": 80,
            "strike": {"effect": "capture", "switches_sides": ["L", "G"], "immune": ["K"]},
        },
    }
    errors = validate_spec(spec)
    assert errors == [], errors
    return spec


def _take(engine: Engine, origin: str, target: str) -> Any:
    state = engine.initial_state()
    wanted = (_square(origin), _square(target))
    moves = [m for m in engine.legal_moves(state) if (move_from(m), move_to(m)) == wanted]
    assert len(moves) == 1, (origin, target, engine.move_texts(state))
    after = engine.apply(state, moves[0])
    assert after.key == engine._compute_hash(after), "hash drifted"
    return after


def _heading(engine: Engine, value: int) -> int:
    spec = engine.spec
    return spec.facing_headings()[spec.owner_of(value)][spec.facing_of(value)]


class TestTheTable:
    """`_coat_facing` read off the frames' own vectors, as §6 states it."""

    @pytest.mark.parametrize(
        ("axes", "orientation", "white", "black"),
        [
            # mirror: forward becomes backward, right stays right.
            (4, "mirror", (2, 1, 0, 3), (2, 1, 0, 3)),
            # rotate: forward becomes backward, right becomes left.
            (4, "rotate", (2, 3, 0, 1), (2, 3, 0, 1)),
            (2, "mirror", (1, 0), (1, 0)),
            (2, "rotate", (1, 0), (1, 0)),
        ],
    )
    def test_each_facing_maps_to_the_index_with_the_same_heading(
        self, axes: int, orientation: str, white: tuple[int, ...], black: tuple[int, ...]
    ) -> None:
        spec = compile_spec(_spec(axes=axes, orientation=orientation))
        assert spec.coat_facing == (white, black)
        headings = spec.facing_headings()
        for owner in (0, 1):
            for facing, kept in enumerate(spec.coat_facing[owner]):
                assert headings[1 - owner][kept] == headings[owner][facing]

    def test_a_heading_with_no_match_is_a_compiler_bug(self) -> None:
        with pytest.raises(AssertionError):
            _coat_facing(((0, 90), (0, 270)))

    def test_it_crosses_only_where_a_facing_game_switches_sides(self) -> None:
        switching = _spec()
        assert build_tables(compile_spec(switching))["coat_facing"] == [2, 1, 0, 3, 2, 1, 0, 3]
        plain = copy.deepcopy(switching)
        del plain["rules"]["strike"]
        assert build_tables(compile_spec(plain))["coat_facing"] == []


class TestAndernachKeepsTheHeading:
    """The capturer changes colour where it took, pointing the way it pointed."""

    @pytest.mark.parametrize(
        ("axes", "orientation", "facing", "victim", "kept"),
        [
            # Pointing up the board: black's backward, under either reading of the setup.
            (4, "mirror", FORWARD, "c3", 2),
            (4, "rotate", FORWARD, "c3", 2),
            (2, "mirror", FORWARD, "c3", 1),
            (2, "rotate", FORWARD, "c3", 1),
            # Pointing right: black's right under mirror, black's left under rotate.
            (4, "mirror", RIGHT, "d2", 1),
            (4, "rotate", RIGHT, "d2", 3),
            # Pointing down at a piece behind it: black's forward.
            (4, "mirror", BACKWARD, "c1", 0),
            (2, "rotate", BACKWARD, "c1", 0),
        ],
    )
    def test_the_switched_lancer_points_where_it_pointed(
        self, axes: int, orientation: str, facing: str, victim: str, kept: int
    ) -> None:
        engine = Engine(_spec(axes=axes, orientation=orientation, facing=facing, victim=victim))
        before = engine.initial_state().board[_square("c2")]
        after = _take(engine, "c2", victim)
        value = after.board[_square(victim)]
        spec = engine.spec
        assert (spec.owner_of(value), spec.piece_ids[spec.type_of(value)]) == (1, "L")
        assert spec.facing_of(value) == kept
        assert _heading(engine, value) == _heading(engine, before)

    def test_the_old_reading_would_have_turned_it(self) -> None:
        """Both ways: the pre-TC2 answer (black's initial facing, forward) points down, and the
        core no longer plays it."""
        engine = Engine(_spec())
        after = _take(engine, "c2", "c3")
        value = after.board[_square("c3")]
        spec = engine.spec
        assert value != spec.encode_piece(1, spec.piece_ids.index("L"), 0)
        assert value == spec.encode_piece(1, spec.piece_ids.index("L"), 2)

    def test_a_fixed_facing_keeps_its_heading_too(self) -> None:
        """`turns: false`: the defector cannot turn round, so it points into its new camp."""
        engine = Engine(
            _spec(
                turns=False,
                extra=[{"piece": "L", "player": 1, "squares": ["a5"], "facing": RIGHT}],
            )
        )
        before = engine.initial_state().board[_square("c2")]
        after = _take(engine, "c2", "c3")
        value = after.board[_square("c3")]
        assert engine.spec.facing_of(value) == 2
        assert _heading(engine, value) == _heading(engine, before)

    def test_a_type_that_does_not_face_enters_at_its_one_orientation(self) -> None:
        """The guard has one orientation: nothing to keep, so it stands at facing 0."""
        spec = _spec(
            victim="d4",
            extra=[{"piece": "G", "player": 0, "squares": ["c4"]}],
        )
        engine = Engine(spec)
        after = _take(engine, "c4", "d4")
        value = after.board[_square("d4")]
        assert engine.spec.owner_of(value) == 1
        assert engine.spec.piece_ids[engine.spec.type_of(value)] == "G"
        assert engine.spec.facing_of(value) == 0


class TestARoundKeepsTheHeading:
    """The same rule in a round (`rounds.rs`), held against the declared oracle."""

    @pytest.mark.parametrize("orientation", ["mirror", "rotate"])
    def test_a_switch_in_a_round(self, orientation: str) -> None:
        """`strike_after`; a round refuses `stay`, so `contact_in_place` never switches."""
        spec = _spec(orientation=orientation, facing=RIGHT, victim="d2")
        spec["rules"]["win"] = "royal_capture"
        spec["rules"]["simultaneous"] = {}
        assert validate_spec(spec) == [], validate_spec(spec)
        engine = Engine(spec)
        state = engine.initial_state()
        wanted = (_square("c2"), _square("d2"))
        mine = [m for m in engine.legal_moves_for(state, 0) if (move_from(m), move_to(m)) == wanted]
        assert len(mine) == 1
        theirs = [m for m in engine.legal_moves_for(state, 1) if move_from(m) == _square("e5")][0]
        after = engine.apply_round(state, mine[0], theirs)
        assert after.board == oracle_round(engine, state, mine[0], theirs).board
        value = after.board[_square("d2")]
        assert engine.spec.owner_of(value) == 1
        assert engine.spec.facing_of(value) == (1 if orientation == "mirror" else 3)
        assert _heading(engine, value) == 90


class TestTheWords:
    def test_describe_says_it_only_where_a_switcher_faces(self) -> None:
        assert "keeps pointing the way it pointed" in describe_spec(_spec())
        plain = _spec()
        plain["rules"]["strike"]["switches_sides"] = ["G"]
        assert "keeps pointing" not in describe_spec(plain)
