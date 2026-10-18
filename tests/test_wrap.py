"""Schema 5: a board whose opposite edges are the same edge.

There is no external implementation to check a wrapped board against -- no engine in the
reference set plays a cylinder, and none at all plays a *one-way* one -- so the frozen
perft ladders here are this project's own, exactly as the RPS ladders are. What keeps
them from being a tautology is the rest of this file, and one test in particular.

**A cyclic shift of every piece along a joined axis is an isomorphism.** Slide the whole
position one file to the left on a cylinder and it is the same position; perft must not
move. That holds only if the geometry is right, it holds at every depth, it needs no
second implementation, and an off-by-one in the wrap -- or a seam that is joined in one
direction and not the other -- breaks it immediately. It is the closest thing available
to an independent check, and it is why the literals below are pinned rather than
trusted.

Two smaller anchors are hand-verifiable and are checked as such: cylinder chess opens
with the same twenty moves as chess, and its perft(2) is 392 rather than 400 because
four of White's first moves open a *wrapped* line that pins a Black pawn against its
king -- the f1 bishop's diagonal f1-g2-h3-a4-b5-c6-d7 and the d1 queen's c2-b3-a4-h5-g6-f7.

Six boards are pinned here, and the file shift covers four of them -- every board whose
files are joined in *both* directions. The other two are not gaps but the reason the shift
has a precondition, and it is asserted rather than assumed: `onewaycylinder` and
`onewaytorus` join their files one way per player, which still leaves each player an
*edge* in the other direction. An edge sits at an absolute file, so the army's position
along the board matters and the shift is asserted to **change** its perft, exactly as it
changes a board with no seam at all.

No chess-descended board is rank-shiftable whatever its seam: a pawn has an absolute
forward direction and promotion names a rank. That axis is checked on a synthetic probe
with neither -- `_toroidal_probe`, rooks and kings only -- which is what that probe is for.

Ten boards were pinned here until the library audit (Q868, 2026-10-12,
`docs/archive/plan-library-audit.md`) reduced the wrap series to a canonical set and retired
`tunnel-vertical`, `onewaytunnel`, `torus-extended` and `sardines-torus`; their ladders are
kept in the comment above `WRAPPED`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_CASTLE, move_from, move_to, move_type
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.schema import validate_spec

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"

# The five boards schema 5 shipped with, and the four added after them. `tunnel` was
# renamed `tunnel-vertical` when `tunnel-horizontal` arrived beside it: only the display
# name moved, the mechanics hash is still `0268bb0926bf`, and every number
# `reference/wrap-boards.md` reports under the old name describes this file.
# `tunnel-horizontal` (`195564f716ec`, 7x8, the file seam one rook down) was retired by
# Q434 on 2026-09-25: its promotion rank lay past the board. Its ladder was
# {1: 18, 2: 324, 3: 7_245, 4: 158_713}. `torus-less-extended` and `crazyhouse-torus`
# keep an odd-width file seam under the shift below.
# Q868, 2026-10-12, retired four more, with their ladders: `tunnel-vertical`
# (`0268bb0926bf`, {1: 64, 2: 3_593, 3: 206_336}), `onewaytunnel` (`da7647bf02e1`,
# {1: 47, 2: 2_209, 3: 106_034}), `torus-extended` (`fdadd9839d34`, {1: 45, 2: 2_013,
# 3: 98_194, 4: 4_760_019}) and `sardines-torus` (`f5585062433b`, {1: 63, 2: 2_985,
# 3: 170_616, 4: 8_030_690}).
WRAPPED = (
    "cylinder",
    "onewaycylinder",
    "torus",
    "onewaytorus",
    "torus-less-extended",
    "crazyhouse-torus",
)

# This project's own numbers, frozen as regression pins. Depth 1 and 2 on `cylinder` are
# hand-derived in the module docstring; everything else is pinned by the shift
# isomorphism below rather than by an external engine -- which covers four of the six,
# every board whose files are joined in both directions. See the module docstring for
# what holds the other three up.
OPENING_PERFT: dict[str, dict[int, int]] = {
    "cylinder": {1: 20, 2: 392, 3: 9_162, 4: 211_036},
    "onewaycylinder": {1: 20, 2: 396, 3: 9_051, 4: 204_095},
    "torus": {1: 72, 2: 4_586, 3: 296_604},
    # **The first four depths are `onewaycylinder`'s exactly**, and the fifth is where
    # this board stops being that one. Its rank seam is one-way -- the first player may
    # cross the top edge and the second the bottom -- and from the opening array nothing
    # reaches a far rank inside four plies, so for four plies the join contributes
    # nothing at all. Depth 5 is 5,220,202 against the cylinder's 5,218,605, a difference
    # of 1,597, and it is carried here for that reason: a ladder that stopped at 4 would
    # pin a board indistinguishable from one already in this table.
    "onewaytorus": {1: 20, 2: 396, 3: 9_051, 4: 204_095, 5: 5_220_202},
    "torus-less-extended": {1: 36, 2: 1_296, 3: 52_563, 4: 2_112_956},
    "crazyhouse-torus": {1: 36, 2: 1_296, 3: 52_563, 4: 2_112_956},
}


def _load(name: str) -> dict[str, Any]:
    return load_spec(RULESETS / f"{name}.json")


def _files_join_both_ways(name: str) -> bool:
    """Whether both file edges are crossable in both directions by both players.

    That is the condition for the shift below to be an isomorphism, and it is read off
    the spec rather than listed, so a wrapped board added later is covered by the test
    that keeps its perft honest instead of only by the pin above.
    """
    wrap = (_load(name).get("rules") or {}).get("wrap") or {}
    stated = {wrap.get("files")} | set((wrap.get("files_by_player") or {}).values())
    return stated - {None} == {"both"}


CYLINDRICAL = tuple(name for name in WRAPPED if _files_join_both_ways(name))
# Every (board, offset) of the shift below, one test each (Q749): a whole board's seven
# shifts, a compile and a perft apiece, were 2.2-2.3 s in the gate.
SHIFTS = tuple(
    pytest.param(name, by, id=f"{name}-{by}")
    for name in CYLINDRICAL
    for by in range(1, _load(name)["board"]["width"])
)


def _piece(spec: dict[str, Any], piece_id: str) -> int:
    """The compiled index of a piece, so a test reads by name rather than by position."""
    return [p["id"] for p in spec["pieces"]].index(piece_id)


def _shift_files(spec: dict[str, Any], by: int) -> dict[str, Any]:
    """Slide every placement cyclically along the file axis.

    Done to the *spec* rather than to a position, so the whole test uses public API and
    the mirrored second army comes out shifted the same way for free.
    """
    width = spec["board"]["width"]
    moved = copy.deepcopy(spec)
    for placement in moved["setup"]["placements"]:
        placement["squares"] = [
            chr(ord("a") + (ord(square[0]) - ord("a") + by) % width) + square[1:]
            for square in placement["squares"]
        ]
    return moved


def _shift_ranks(spec: dict[str, Any], by: int) -> dict[str, Any]:
    height = spec["board"]["height"]
    moved = copy.deepcopy(spec)
    for placement in moved["setup"]["placements"]:
        placement["squares"] = [
            square[0] + str((int(square[1:]) - 1 + by) % height + 1)
            for square in placement["squares"]
        ]
    return moved


# --------------------------------------------------------------------------- geometry


class TestTheShiftIsomorphism:
    """The one check here that is not a frozen number.

    A joined axis has no distinguished position along it, so sliding the whole board
    along it is a relabelling. Perft is invariant under a relabelling; it is not
    invariant under a wrong seam.
    """

    @staticmethod
    def _without_castling(name: str) -> dict[str, Any]:
        # Castling is the one move stated as a span of one rank rather than as a piece
        # following a line, so it does not wrap -- which means a shifted king and rook
        # castle differently and the shift is not an isomorphism for it. Removed rather
        # than worked around, because that asymmetry is itself asserted below.
        spec = _load(name)
        # `pop` rather than `del`: a wrapped board need not declare castling at all,
        # and a ruleset that does not is not a ruleset this check should skip.
        spec["rules"].pop("castling", None)
        return spec

    @pytest.mark.parametrize(("name", "by"), SHIFTS)
    def test_a_file_shift_does_not_change_perft(self, name: str, by: int) -> None:
        """Every board whose files join both ways, at every offset along the seam.

        This is what the frozen ladders rest on. A pin derived from the engine and
        checked against the engine says nothing; a pin that also has to survive being
        slid along an axis with no distinguished position says the geometry is right.
        """
        spec = self._without_castling(name)
        base = Engine(spec)
        expected = base.perft(base.initial_state(), 3)
        shifted = Engine(_shift_files(spec, by))
        assert expected == shifted.perft(
            shifted.initial_state(), 3
        ), f"shifting {name} {by} files changed the move tree"

    @pytest.mark.parametrize("by", range(1, 8))
    def test_a_file_shift_is_not_an_isomorphism_with_edges(self, by: int) -> None:
        """The control: the same shift on a board with edges *does* change the game.

        Without this the test above would pass on an engine that ignored `wrap`
        entirely, because chess's own back rank is nearly file-symmetric.
        """
        spec = self._without_castling("cylinder")
        del spec["rules"]["wrap"]
        spec["schema_version"] = 1
        base = Engine(spec)
        shifted = Engine(_shift_files(spec, by))
        assert base.perft(base.initial_state(), 3) != shifted.perft(
            shifted.initial_state(), 3
        )

    @pytest.mark.parametrize("by", range(1, 8))
    def test_a_one_way_seam_is_not_an_isomorphism_either(self, by: int) -> None:
        """The sharper control, and one of the two reasons `CYLINDRICAL` is six of nine.

        A one-way seam still leaves each player an *edge*: White may cross westward, so
        going east it stops at the h-file and no further. An edge sits at an absolute
        file, so how far a piece reaches eastward depends on which file it stands on and
        sliding the army along the board is not a relabelling. The two-way boards above
        would pass this test wrongly if a permission were ever read as a topology.
        """
        spec = self._without_castling("onewaycylinder")
        base = Engine(spec)
        shifted = Engine(_shift_files(spec, by))
        assert base.perft(base.initial_state(), 3) != shifted.perft(
            shifted.initial_state(), 3
        )

    @staticmethod
    def _toroidal_probe() -> dict[str, Any]:
        """Rooks and kings on a torus: no pawn, no promotion, no distinguished rank.

        Chess cannot test a *rank* shift -- pawns have an absolute forward direction and
        promotion names a rank -- so the probe drops everything that reads a rank. What
        is left is a position whose only structure is where the pieces stand.
        """
        return {
            "schema_version": 5,
            "name": "toroidal probe",
            "description": "rooks and kings on a board with no edges",
            "board": {"width": 5, "height": 5},
            "pieces": [
                {
                    "id": "R", "name": "rook", "value": 5,
                    "moves": [
                        {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
                    ],
                },
                {
                    "id": "K", "name": "king", "value": 0, "royal": True,
                    "moves": [
                        {"kind": "leap", "vectors": [[1, 0], [1, 1]],
                         "symmetry": "rotate4"}
                    ],
                },
            ],
            "setup": {
                "mirror": False,
                "placements": [
                    {"piece": "R", "player": 0, "squares": ["a1", "b1"]},
                    {"piece": "K", "player": 0, "squares": ["c1"]},
                    {"piece": "R", "player": 1, "squares": ["d4", "e4"]},
                    {"piece": "K", "player": 1, "squares": ["a4"]},
                ],
            },
            "rules": {"win": "royal_capture",
                      "wrap": {"files": "both", "ranks": "both"}},
        }

    @pytest.mark.parametrize("axis", ("files", "ranks"))
    @pytest.mark.parametrize("by", (1, 2, 3, 4))
    def test_both_axes_shift_freely_on_a_torus(self, axis: str, by: int) -> None:
        spec = self._toroidal_probe()
        shift = _shift_files if axis == "files" else _shift_ranks
        base = Engine(spec)
        moved = Engine(shift(spec, by))
        assert base.perft(base.initial_state(), 3) == moved.perft(
            moved.initial_state(), 3
        ), f"shifting the torus probe {by} along {axis} changed the move tree"


class TestTheRayStopsComingRound:
    """A ray ends when it reaches a square it has already been on, the origin included."""

    @staticmethod
    def _lone_rook(width: int, wrap: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 5,
            "name": "lone rook", "description": "one rook and a king each",
            "board": {"width": width, "height": 3},
            "pieces": [
                {"id": "R", "name": "rook", "value": 5,
                 "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
                {"id": "K", "name": "king", "value": 0, "royal": True,
                 "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
            ],
            "setup": {"mirror": False, "placements": [
                {"piece": "R", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 0, "squares": ["b1"]},
                {"piece": "K", "player": 1, "squares": [f"{chr(ord('a') + width - 1)}3"]},
            ]},
            "rules": {"win": "royal_capture", "wrap": wrap},
        }

    def test_a_ray_is_seven_squares_long_on_an_eight_wide_cylinder(self) -> None:
        """The rule stated directly: round once, stopping short of the origin.

        Seven and not eight, so the rook does not attack its own square; seven and not
        unbounded, which is what a ray with no edge to stop at would otherwise be.
        """
        compiled = compile_spec(self._lone_rook(8, {"files": "both"}))
        along_the_rank = [
            ray for ray, _, _ in compiled.slides[0][0][0] if all(sq < 8 for sq in ray)
        ]
        assert sorted(len(ray) for ray in along_the_rank) == [7, 7]

    def test_a_wrapped_ray_emits_each_move_once(self) -> None:
        """Two rays reach the same squares from opposite directions; one move each.

        The rook stands on a1 with its own king on b1, so eastward it is blocked at
        once. Westward it wraps -- h1, g1, f1, e1, d1, c1 -- and stops on the king
        rather than passing it. Two more moves go up the a-file.
        """
        engine = Engine(self._lone_rook(8, {"files": "both"}))
        state = engine.initial_state()
        moves = engine.legal_moves(state)
        assert len(moves) == len(set(moves)), "a wrapped ray emitted a move twice"
        rook_targets = sorted(move_to(m) for m in moves if move_from(m) == 0)
        assert rook_targets == [2, 3, 4, 5, 6, 7, 8, 16], rook_targets

    def test_the_ray_never_revisits_a_square(self) -> None:
        for wrap in ({"files": "both"}, {"files": "both", "ranks": "both"}):
            compiled = compile_spec(self._lone_rook(8, wrap))
            for player in (0, 1):
                for square in range(compiled.n_squares):
                    for ray, _, _ in compiled.slides[player][0][square]:
                        assert len(set(ray)) == len(ray), "a ray visited a square twice"
                        assert square not in ray, "a ray came back to its own origin"

    def test_a_leap_that_comes_all_the_way_round_is_not_a_move(self) -> None:
        """A step whose offset is the board's own width lands on its origin."""
        spec = self._lone_rook(4, {"files": "both"})
        spec["pieces"][0]["moves"] = [
            {"kind": "leap", "vectors": [[4, 0]], "symmetry": "none"}
        ]
        compiled = compile_spec(spec)
        assert compiled.leaps[0][0][0] == ()


class TestDirection:
    """A permission belongs to a player and to an edge, and both halves are load-bearing."""

    def test_one_player_crosses_and_the_other_does_not(self) -> None:
        spec = _load("onewaycylinder")
        rook = _piece(spec, "R")
        compiled = compile_spec(spec)
        # The rook's westward ray from a1 exists for the first player only.
        first = compiled.slides[0][rook][0]
        second = compiled.slides[1][rook][0]
        assert any(ray[0] == 7 for ray, _, _ in first), "White cannot leave by the a-file"
        assert not any(ray[0] == 7 for ray, _, _ in second), "Black crossed the wrong edge"

    def test_the_second_player_crosses_the_other_edge(self) -> None:
        spec = _load("onewaycylinder")
        rook = _piece(spec, "R")
        compiled = compile_spec(spec)
        # From h1 (square 7) the eastward ray wraps to a1 for the second player only.
        assert any(ray[0] == 0 for ray, _, _ in compiled.slides[1][rook][7])
        assert not any(ray[0] == 0 for ray, _, _ in compiled.slides[0][rook][7])

    def test_an_edge_named_once_is_not_the_other_edge(self) -> None:
        spec = copy.deepcopy(_load("cylinder"))
        spec["rules"]["wrap"] = {"files": "left"}
        rook = _piece(spec, "R")
        compiled = compile_spec(spec)
        assert any(ray[0] == 7 for ray, _, _ in compiled.slides[0][rook][0])
        assert not any(ray[0] == 0 for ray, _, _ in compiled.slides[0][rook][7])


class TestWhatDoesNotWrap:
    def test_castling_stays_on_its_rank(self) -> None:
        """Castling is a span of one rank and does not use the join.

        Said in the prose because a player told the board is a cylinder will assume the
        opposite, and asserted here because nothing else would notice if it changed: a
        wrapping castle would put the king on a file it never crossed, and every perft
        would simply count one more move.
        """
        engine = Engine(_load("cylinder"))
        # A bare king and its two rooks: both ordinary castles are available, and a
        # third "round the back" one would be too if castling followed the board.
        state = engine.make_state({
            "e1": ("K", 0), "a1": ("R", 0), "h1": ("R", 0),
            "e8": ("K", 1), "a8": ("R", 1), "h8": ("R", 1),
        })
        destinations = sorted(
            move_to(m) for m in engine.legal_moves(state)
            if move_from(m) == 4 and move_type(m) == MOVE_CASTLE
        )
        assert destinations == [2, 6], destinations

    def test_royal_facing_sees_the_line_that_runs_the_other_way_round(self) -> None:
        """What the refusal cost, and what replaced it.

        Schema 5 refused `royal_facing` beside joined ranks because `royals_face` walked
        one file from the lower royal to the higher and could not see a line running the
        other way. It is a compiled table now, and the table holds **two** lines between
        two squares on a joined file -- one each way round -- so the rule is satisfied by
        whichever is open.

        Generals on e1 and e9 of a ten-rank board: the short way is over e10 alone, and
        the long way is e2 through e8. Three positions, because only all three together
        say the table is right rather than merely non-empty.
        """
        spec = _load("xiangqi")
        spec["schema_version"] = 5
        spec["rules"]["wrap"] = {"ranks": "both"}
        engine = Engine(spec)

        def facing(blockers: dict[str, tuple[str, int]]) -> bool:
            """Whether a bystander is frozen, which is what facing does to a position.

            Asked of a soldier away on the a-file rather than of the king, because the
            king can always step off the file and so always has a move. A piece that
            cannot legally move while sitting nowhere near the action is the rule's own
            signature.
            """
            state = engine.make_state(
                {"e1": ("K", 0), "e9": ("K", 1), "a5": ("S", 1), **blockers},
                side_to_move=1,
            )
            bystander = 4 * 9  # a5
            return not [
                m for m in engine.legal_moves(state) if move_from(m) == bystander
            ]

        # Horses, because they are the one blocker that neither attacks the king it
        # stands beside nor is confined to a region: a soldier on e8 would put the king
        # on e9 in check and the test would be measuring that instead.
        long_way = {f"e{rank}": ("H", 0) for rank in range(2, 9)}
        # Nothing in between either way: the seam is the shorter line and it is open.
        assert facing({})
        # The short way blocked, the long way open: still facing, and this is the case
        # the arithmetic could not see at all.
        assert facing({"e10": ("H", 0)})
        # Both blocked: not facing. Without this the test would pass on a table that
        # said every pair of squares on a file faces.
        assert not facing({"e10": ("H", 0), **long_way})

    def test_royal_facing_is_unmoved_by_a_join_it_cannot_read(self) -> None:
        """The rule reads one axis, so joining the other must change nothing."""
        spec = _load("xiangqi")
        spec["schema_version"] = 5
        spec["rules"]["wrap"] = {"files": "both"}
        engine = Engine(spec)
        plain = Engine(_load("xiangqi"))
        state = {"e1": ("K", 0), "e9": ("K", 1), "e5": ("S", 0)}
        for side in (0, 1):
            assert (
                sorted(engine.legal_moves(engine.make_state(state, side_to_move=side)))
                == sorted(plain.legal_moves(plain.make_state(state, side_to_move=side)))
            )


class TestEnPassantIsATable:
    """The square a pawn stepped over is read off the line it moved along.

    It used to be halved out of the move's delta, which is right on a board with edges
    and silently wrong on one without -- and no perft would have moved either way.
    """

    @pytest.mark.parametrize(
        "name",
        sorted(
            path.stem for path in RULESETS.glob("*.json")
            if (load_spec(path).get("rules") or {}).get("en_passant")
            and not (load_spec(path).get("rules") or {}).get("wrap")
        ),
    )
    def test_the_table_reproduces_the_arithmetic_where_the_board_has_edges(
        self, name: str
    ) -> None:
        compiled = compile_spec(_load(name))
        width = compiled.width
        for player in (0, 1):
            for origin, entries in enumerate(compiled.ep_passes[player]):
                for target, passed in entries:
                    delta_file = target % width - origin % width
                    delta_rank = target // width - origin // width
                    assert max(abs(delta_file), abs(delta_rank)) == 2
                    assert (
                        origin + delta_rank // 2 * width + delta_file // 2 == passed
                    ), f"{name}: the crossed square moved for {origin}->{target}"

    def test_a_wrapped_double_step_names_the_square_it_actually_crossed(self) -> None:
        """The case the old arithmetic got wrong, constructed deliberately.

        A pawn double-stepping across the seam has a raw delta that is neither its
        direction nor its distance, so halving it names a square on the wrong side of
        the board.
        """
        spec = {
            "schema_version": 5,
            "name": "seam pawn", "description": "a pawn that steps over the join",
            "board": {"width": 4, "height": 4},
            "pieces": [
                {"id": "P", "name": "pawn", "value": 1, "moves": [
                    {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
                    {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2,
                     "mode": "move", "condition": {"from_rank": 2}},
                    {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral",
                     "mode": "capture"},
                ]},
                {"id": "K", "name": "king", "value": 0, "royal": True, "moves": [
                    {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
                ]},
            ],
            "setup": {"mirror": False, "placements": [
                {"piece": "P", "player": 0, "squares": ["a3"]},
                {"piece": "K", "player": 0, "squares": ["d1"]},
                {"piece": "K", "player": 1, "squares": ["d4"]},
            ]},
            "rules": {
                "win": "royal_capture",
                "promotion": {"piece": "P", "rank": 1, "to": ["K"]},
                "en_passant": True,
                "wrap": {"ranks": "both"},
            },
        }
        compiled = compile_spec(spec)
        # a3 is square 8 on a 4-wide board: rank index 2, which is the double step's rank.
        # Two steps north from rank 2 goes 3, then over the join to 0.
        passes = dict(compiled.ep_passes[0][8])
        assert passes == {0: 12}, passes
        # The old arithmetic would have halved a delta of -2 and named rank 1.
        assert passes[0] != 4


class TestTheBlastFollowsTheBoard:
    def test_an_explosion_reaches_across_the_join(self) -> None:
        """A blast reads the board's shape rather than a player's permission.

        The two file edges are one seam whoever may cross it, so a capture beside the
        join takes what is on the other side -- including for a player who may not move
        through it.
        """
        spec = _load("atomic")
        spec["schema_version"] = 5
        spec["rules"]["wrap"] = {"files_by_player": {"0": "left", "1": "none"}}
        compiled = compile_spec(spec)
        assert compiled.explosion is not None
        # a1 is square 0; its blast should include h1 (7), h2 (15) and b2 (9).
        assert {7, 15} <= set(compiled.explosion.blast[0]), compiled.explosion.blast[0]
        assert len(set(compiled.explosion.blast[0])) == len(compiled.explosion.blast[0])


# --------------------------------------------------------------------------- corpus


class TestTheCorpus:
    @pytest.mark.parametrize("name", WRAPPED)
    def test_it_validates(self, name: str) -> None:
        assert validate_spec(_load(name)) == []

    @pytest.mark.parametrize("name", WRAPPED)
    def test_the_perft_ladder_is_frozen(self, name: str) -> None:
        engine = Engine(_load(name))
        state = engine.initial_state()
        for depth, nodes in OPENING_PERFT[name].items():
            assert engine.perft(state, depth) == nodes, f"{name} perft({depth})"

    def test_cylinder_chess_opens_like_chess(self) -> None:
        """Twenty moves, by hand: every wrapped line is blocked by the mover's own army."""
        engine = Engine(_load("cylinder"))
        assert len(engine.legal_moves(engine.initial_state())) == 20

    def test_drops_add_no_move_until_something_has_been_captured(self) -> None:
        """`crazyhouse-torus` is `torus-less-extended` plus a hand, and for four plies that is
        a difference in what a position is *worth* rather than in what may be played.

        The same statement the schema-4 rulesets make about territory, and the same value:
        a hand that generated moves while empty, or a drop offered before a capture, moves
        a number here. The divergence is pinned too, at the first ply that can follow a
        capture, so this is not a claim that the rule does nothing.
        """
        plain = Engine(_load("torus-less-extended"))
        with_hand = Engine(_load("crazyhouse-torus"))
        for depth in (1, 2, 3, 4):
            assert plain.perft(plain.initial_state(), depth) == with_hand.perft(
                with_hand.initial_state(), depth
            ), f"the hand changed the move tree at depth {depth}"

    @pytest.mark.slow
    def test_the_hand_is_live_at_the_first_ply_that_can_follow_a_capture(self) -> None:
        plain = Engine(_load("torus-less-extended"))
        with_hand = Engine(_load("crazyhouse-torus"))
        assert plain.perft(plain.initial_state(), 5) == 94_701_949
        # Q869: 95_547_475 while the hand held the rear pawn as a type of its own; since
        # the respelling a captured rear pawn comes back facing forward, onto the front
        # pawns' ranks. The plain game's number did not move: it is the same game.
        assert with_hand.perft(with_hand.initial_state(), 5) == 95_545_239

    def test_an_eight_by_eight_torus_is_over_before_it_starts(self) -> None:
        """Why `torus.json` is played on the tall board, asserted rather than assumed.

        Joining the ranks of an ordinary chess setup puts the two kings a single step
        apart through the seam, so **both** sides are in check in the opening position --
        which no legal move can produce, and which the illegal-position guard in
        `terminal_prefix` therefore steps aside for by name. It is White to move, and
        White's every legal move is a capture of the black king: the game is decided at
        ply one, with four ways to say so.

        **Q133 flipped which way.** This used to assert `legal_moves == []` and a
        checkmate against White: the filter refused every move for leaving White's king
        attacked, including the four that take the attacker off the board. A move that
        removes the opponent's last royal wins in `terminal_prefix` before anyone moves
        again, so what it leaves the mover's own king exposed to cannot matter, and
        `explode_on_capture` is no longer what decides that
        (`docs/reference/derived-royal-loss.md`). The claim this test is for is
        unchanged and is the reason `torus.json` uses a tall board.
        """
        spec = _load("chess")
        spec["schema_version"] = 5
        spec["rules"]["wrap"] = {"files": "both", "ranks": "both"}
        engine = Engine(spec)
        state = engine.initial_state()
        assert engine.in_check(state, 0) and engine.in_check(state, 1)

        moves = engine.legal_moves(state)
        assert len(moves) == 4, "the queen, king, bishop and knight all reach e8"
        for move in moves:
            outcome = engine.terminal_prefix(engine.apply(state, move))
            assert outcome is not None, "every legal move here takes the black king"
            assert (outcome.winner, outcome.reason) == (0, "royal_destroyed")


class TestInertness:
    """Schema 5 changes nothing for a board with edges."""

    @pytest.mark.parametrize(
        "path", sorted(RULESETS.glob("*.json")), ids=lambda p: p.stem
    )
    def test_only_the_wrapped_rulesets_carry_a_seam(self, path: Path) -> None:
        spec = load_spec(path)
        compiled = compile_spec(spec)
        assert (compiled.wrap is not None) == (path.stem in WRAPPED)

    def test_the_committed_hashes_have_not_moved(self) -> None:
        """Chess in particular: four schema bumps and the same twelve characters."""
        assert spec_hash(_load("chess")) == "4c1922ffae26"

    @pytest.mark.parametrize("name", WRAPPED)
    def test_the_seam_is_part_of_what_a_ruleset_is(self, name: str) -> None:
        """`rules` is inside the mechanics hash, so a board that wraps is a different game.

        The direction that matters: two specs identical but for the seam must not
        collide, or the design loop would skip the tournament for the second of them.
        """
        with_seam = _load(name)
        without = copy.deepcopy(with_seam)
        del without["rules"]["wrap"]
        assert spec_hash(with_seam) != spec_hash(without)


class TestTheOtherSurfaces:
    @pytest.mark.parametrize("name", WRAPPED)
    def test_the_prose_states_the_seam(self, name: str) -> None:
        from chess_variant_sandbox.spec.describe import describe_rules

        lines = describe_rules(_load(name))
        assert any("neighbours" in line for line in lines), lines

    @pytest.mark.parametrize("name", WRAPPED)
    def test_rule_mass_counts_it(self, name: str) -> None:
        from chess_variant_sandbox.spec.complexity import measure_complexity

        spec = _load(name)
        without = copy.deepcopy(spec)
        del without["rules"]["wrap"]
        counted, bare = measure_complexity(spec), measure_complexity(without)
        assert counted.special_rules - bare.special_rules == 1.5
        # Rule mass is what the index reads: a seam is 1.5 special rules at 6.0 apiece.
        assert counted.rule_mass - bare.rule_mass == 9.0

    def test_a_board_with_edges_spelled_out_is_a_board_with_edges(self) -> None:
        """Every spelling of "no seam" is the game that was already there."""
        from chess_variant_sandbox.spec.complexity import measure_complexity

        base = _load("chess")
        for stated in (
            {"files": "none"},
            {"ranks": "none"},
            {"files_by_player": {"0": "none", "1": "none"}},
        ):
            spec = copy.deepcopy(base)
            spec["schema_version"] = 5
            spec["rules"]["wrap"] = stated
            assert validate_spec(spec) == []
            assert compile_spec(spec).wrap is None
            assert (
                measure_complexity(spec).special_rules
                == measure_complexity(base).special_rules
            )

    def test_the_generated_schema_can_express_it(self) -> None:
        """A rule the model cannot propose does not exist as far as the loop is concerned."""
        from chess_variant_sandbox.spec.jsonschema import rule_spec_schema

        wrap = rule_spec_schema()["properties"]["rules"]["properties"]["wrap"]
        assert set(wrap["properties"]) == {
            "files", "ranks", "files_by_player", "ranks_by_player",
            # Schema 16: a hexagon's three pairs of sides.
            "n_s", "ne_sw", "nw_se", "n_s_by_player", "ne_sw_by_player", "nw_se_by_player",
        }

    def test_an_operator_can_reach_it(self) -> None:
        import random

        from chess_variant_sandbox.loop.operators import OPERATORS, mutate

        joiner = tuple(op for op in OPERATORS if op.name == "join-the-edges")
        assert joiner, "the mechanical operator went missing"
        child = mutate(copy.deepcopy(_load("chess")), random.Random(7), operators=joiner)
        assert child is not None
        assert (child.spec["rules"].get("wrap") or {}) != {}
        assert child.spec["schema_version"] == 5


class TestValidation:
    @pytest.mark.parametrize(
        "wrap,message",
        [
            ({"files": "top"}, "rules.wrap.files"),
            ({"ranks": "left"}, "rules.wrap.ranks"),
            ({"files": "both", "files_by_player": {"0": "left", "1": "left"}}, "one or the other"),
            ({"files_by_player": {"0": "left"}}, "keyed by '0' and '1'"),
            ({"files_by_player": {"0": "left", "1": "sideways"}}, "must be one of"),
            ({"edges": "both"}, "unknown key"),
        ],
    )
    def test_it_rejects_what_it_cannot_keep(
        self, wrap: dict[str, Any], message: str
    ) -> None:
        spec = _load("chess")
        spec["schema_version"] = 5
        spec["rules"]["wrap"] = wrap
        errors = validate_spec(spec)
        assert any(message in error for error in errors), errors

    def test_a_spec_declaring_four_may_not_use_it(self) -> None:
        spec = _load("chess")
        spec["schema_version"] = 4
        spec["rules"]["wrap"] = {"files": "both"}
        assert any("schema 5" in error for error in validate_spec(spec))
