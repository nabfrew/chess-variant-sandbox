"""Schema 8: a piece that has a direction, and may spend a move turning.

A facing fails in the way a strike does rather than in the way an obstacle does: it is
*mostly* invisible to a perft. A rotation is a move, so the count moves and a perft sees
that much -- but every other way this rule can be wrong changes no count at all. A piece
reading the wrong frame still generates a legal-looking list. A promotion writing a facing
its type cannot be in still leaves a playable position. A rotation the check filter plays
as a no-op still produces moves. So most of what is asserted below is the *content* of a
move list, the incremental key against its slow definition, and the placement of the
board byte.

**Three predictions the plan made about this rule were wrong**, and each is pinned here
rather than merely noted:

* the encoding it specified interleaves the two armies' values, which breaks every
  ownership range test in the core silently -- `test_one_army_s_values_are_contiguous`;
* it expected `rust_supports` to refuse `facing` beside `drops` and beside
  `explode_on_capture`, and neither is refused, because *a piece that enters play takes
  its type's initial facing* answers both -- `TestEnteringPlay`;
* it expected the fuzzer to owe no oracle because facing names no squares. It names
  states, and `chess_variant_sandbox/fuzz.py` has the oracle.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_ROTATE, decode_move
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.schema import SpecError, validate_spec


def _lancer_spec(**overrides: Any) -> dict[str, Any]:
    """A 5x5 board with one directional slider each, and kings that cannot turn.

    Deliberately not chess. The piece is a two-square slider with a single forward
    vector, which is the smallest thing that is *not* symmetric under a quarter turn --
    a rook would be refused, and that refusal is its own test.
    """
    spec: dict[str, Any] = {
        "schema_version": 8,
        "name": "lancers",
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
                "facing": {"axes": 4},
                "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 2}],
            },
        ],
        "setup": {
            "mirror": True,
            "placements": [
                {"piece": "K", "squares": ["c1"]},
                {"piece": "L", "squares": ["b2", "d2"]},
            ],
        },
        "rules": {"win": "checkmate", "max_plies": 80},
    }
    spec.update(overrides)
    return spec


@pytest.fixture
def lancers() -> Engine:
    return Engine(_lancer_spec())


def _rotations(engine: Engine, state: Any) -> list[int]:
    return [m for m in engine.legal_moves(state) if decode_move(m)[2] == MOVE_ROTATE]


class TestTheEncoding:
    """The board byte, and the one property the plan's spelling would have broken."""

    def test_one_army_s_values_are_contiguous(self, lancers: Engine) -> None:
        """Player-major, which is what keeps every ownership test a range comparison.

        Eleven places in the core ask "is this piece mine?" as `low <= value < high`.
        The plan specified `facing * (2 * n_types) + player * n_types + type + 1`, which
        interleaves the two armies -- and breaks all eleven *silently*, because a piece
        the generator thinks belongs to the other side is one it never generates, and no
        perft can count a move that was never made.
        """
        spec = lancers.spec
        first = {
            spec.encode_piece(0, piece_type, facing)
            for piece_type in range(spec.n_types)
            for facing in range(spec.n_facings)
        }
        second = {
            spec.encode_piece(1, piece_type, facing)
            for piece_type in range(spec.n_types)
            for facing in range(spec.n_facings)
        }
        assert max(first) < min(second)
        assert len(first) == len(second) == spec.side_stride

    def test_a_value_decodes_to_its_owner_type_and_facing(self, lancers: Engine) -> None:
        spec = lancers.spec
        for player in (0, 1):
            for piece_type in range(spec.n_types):
                for facing in range(spec.n_facings):
                    value = spec.encode_piece(player, piece_type, facing)
                    assert spec.owner_of(value) == player
                    assert spec.type_of(value) == piece_type
                    assert spec.facing_of(value) == facing

    def test_the_frame_count_follows_the_facing_count(self, lancers: Engine) -> None:
        spec = lancers.spec
        assert (spec.n_facings, spec.n_frames) == (4, 8)
        for table in (spec.leaps, spec.slides, spec.hops,
                      spec.dedup_squares, spec.immobile_squares):
            assert len(table) == spec.n_frames

    def test_a_ruleset_with_no_facing_is_the_shape_it_always_was(
        self, chess_spec: dict
    ) -> None:
        from chess_variant_sandbox.engine.compile import compile_spec

        spec = compile_spec(chess_spec)
        assert (spec.n_facings, spec.n_frames) == (1, 2)
        assert spec.side_stride == spec.n_types
        assert spec.encode_piece(1, 3) == 1 * spec.n_types + 3 + 1


class TestTurning:
    def test_a_facing_piece_offers_one_move_per_other_orientation(
        self, lancers: Engine
    ) -> None:
        state = lancers.initial_state()
        # Two lancers, four axes each, so three turns apiece.
        assert len(_rotations(lancers, state)) == 6

    def test_a_turn_changes_which_way_the_piece_moves(self, lancers: Engine) -> None:
        """The whole feature, as a move list rather than as a board byte."""
        spec = lancers.spec
        state = lancers.initial_state()
        b2 = spec.square_index(1, 1)
        before = {
            decode_move(m)[1]
            for m in lancers.legal_moves(state)
            if decode_move(m)[0] == b2 and decode_move(m)[2] != MOVE_ROTATE
        }
        assert before == {spec.square_index(1, 2), spec.square_index(1, 3)}  # b3, b4

        turn = lancers.find_move(state, "Lb2^right")
        after = lancers.apply(state, turn)
        # A reply, so it is the first player's move again.
        after = lancers.apply(after, lancers.legal_moves(after)[0])
        east = {
            decode_move(m)[1]
            for m in lancers.legal_moves(after)
            if decode_move(m)[0] == b2 and decode_move(m)[2] != MOVE_ROTATE
        }
        # c2 only: d2 holds its own lancer, and a slide stops at the first occupied
        # square whether or not it may take what stands there.
        assert east == {spec.square_index(2, 1)}

    def test_a_turn_is_written_as_a_turn(self, lancers: Engine) -> None:
        """`Lb2-b2` would say the piece moved to the square it is already on."""
        texts = lancers.move_texts(lancers.initial_state())
        assert "Lb2^right" in texts
        assert "Lb2^backward" in texts
        assert not any(text.startswith("Lb2-b2") for text in texts)

    def test_the_second_player_turns_the_same_way_relative_to_itself(
        self, lancers: Engine
    ) -> None:
        """A facing is relative, exactly as a move vector is.

        Both armies start facing their own forward, so both start in facing 0 -- which is
        the property that makes a mirrored setup mirror the orientations rather than
        pointing one army at its own back rank.
        """
        spec = lancers.spec
        board = lancers.initial_state().board
        for square, value in enumerate(board):
            if value and spec.type_of(value) == spec.piece_ids.index("L"):
                assert spec.facing_of(value) == 0, square


class TestTheKeyAndTheClock:
    def test_two_facings_of_one_piece_are_two_positions(self, lancers: Engine) -> None:
        """The Zobrist term is free only because the piece *value* changed."""
        state = lancers.initial_state()
        turned = lancers.apply(state, lancers.find_move(state, "Lb2^right"))
        assert turned.key != state.key
        assert turned.key == lancers._compute_hash(turned)

    def test_the_incremental_key_survives_a_whole_game_of_turning(
        self, lancers: Engine
    ) -> None:
        rng = random.Random(8)
        for _ in range(20):
            state = lancers.initial_state()
            for _ in range(40):
                moves = lancers.legal_moves(state)
                if lancers.terminal(state, moves) is not None:
                    break
                state = lancers.apply(state, rng.choice(moves))
                assert state.key == lancers._compute_hash(state)

    def test_a_turn_does_not_reset_the_no_progress_clock(self, lancers: Engine) -> None:
        """A rotation is reversible, and the clock is what says a game is going nowhere."""
        state = lancers.initial_state()
        turned = lancers.apply(state, lancers.find_move(state, "Lb2^right"))
        assert turned.halfmove == state.halfmove + 1

    def test_turning_back_and_forth_is_caught_by_repetition(self, lancers: Engine) -> None:
        """The failure this pins is a *cleared window*, not a missing key term.

        `irreversible` swept a rotation in through `kind != MOVE_NORMAL`, which would
        have reset the repetition window at every turn -- so a game of turning back and
        forth would never have repeated, whatever the key said.
        """
        state = lancers.initial_state()
        cycle = ["Lb2^right", "Kc5-b5", "Lb2^forward", "Kb5-c5"]
        for _ in range(3):
            for text in cycle:
                outcome = lancers.terminal(state)
                if outcome is not None:
                    assert outcome.reason == "repetition"
                    return
                state = lancers.apply(state, lancers.find_move(state, text))
        pytest.fail("turning back and forth never repeated")


class TestTheFiltersSeeATurn:
    """Three filters play a move, ask a question and put it back -- and all three write
    `origin = 0` then `target = moving`, which cancels when the two squares are one."""

    def test_a_pinned_piece_may_still_turn(self) -> None:
        """A rotation cannot expose your own royal, and the reason is worth stating.

        The piece does not leave its square, so it still blocks every line it blocked --
        which means the check filter's answer is the same whether it plays the rotation
        or plays nothing. That makes the filter *accidentally* right about this case, and
        the case below is the one where it is not.
        """
        spec = _lancer_spec()
        spec["pieces"].append(
            {
                "id": "R",
                "value": 5,
                "moves": [
                    {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
                ],
            }
        )
        spec["setup"] = {
            "mirror": False,
            "placements": [
                {"piece": "K", "player": 0, "squares": ["c1"]},
                {"piece": "L", "player": 0, "squares": ["c3"]},
                {"piece": "K", "player": 1, "squares": ["a5"]},
                {"piece": "R", "player": 1, "squares": ["c5"]},
            ],
        }
        engine = Engine(spec)
        state = engine.initial_state()
        texts = engine.move_texts(state)
        assert "Lc3^right" in texts  # turns, and goes on blocking the file
        assert "Lc3-d3" not in texts  # stepping off the file does not

    def test_a_turn_does_not_strike_the_piece_that_turns(self) -> None:
        """Where the filter is *not* accidentally right, and no perft could see it.

        A rotation's target square is its origin, so the square it "takes" holds the
        piece doing the turning. Under `rules.strike` the ordinary path would read that
        as a capture and transform the turner into whatever the chain says -- in `make`,
        and again in each of the two filters that mirror `make`'s strike, so the move
        list and the position would disagree about what happened.
        """
        spec = _lancer_spec()
        spec["pieces"].append(
            {
                "id": "P",
                "value": 1,
                "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1}],
            }
        )
        spec["rules"] = {
            "win": "checkmate",
            "max_plies": 80,
            "strike": {
                "effect": "demote",
                "position": "stay",
                "immune": ["K"],
                "chain": [{"from": "L", "to": "P"}],
            },
        }
        engine = Engine(spec)
        state = engine.initial_state()
        b2 = engine.spec.square_index(1, 1)
        before = state.board[b2]
        after = engine.apply(state, engine.find_move(state, "Lb2^right"))
        assert engine.spec.type_of(after.board[b2]) == engine.spec.type_of(before)
        assert engine.spec.facing_of(after.board[b2]) == 1

    def test_a_turn_that_gives_check_is_illegal_under_forbid_check(self) -> None:
        spec = _lancer_spec()
        spec["rules"] = {"win": "checkmate", "forbid_check": True, "max_plies": 80}
        spec["setup"] = {
            "mirror": False,
            "placements": [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "L", "player": 0, "squares": ["c3"], "facing": "forward"},
                {"piece": "K", "player": 1, "squares": ["e3"]},
            ],
        }
        engine = Engine(spec)
        state = engine.initial_state()
        texts = engine.move_texts(state)
        # Facing right, the c3 lancer would slide onto d3 and e3 -- so turning to face
        # right gives check, and `forbid_check` forbids it. Facing left does not.
        assert "Lc3^right" not in texts
        assert "Lc3^left" in texts

    def test_a_turn_is_not_a_capture_under_must_capture(self) -> None:
        """A rotation's target square holds the piece doing the turning."""
        spec = _lancer_spec()
        spec["rules"] = {"win": "checkmate", "must_capture": True, "max_plies": 80}
        spec["setup"] = {
            "mirror": False,
            "placements": [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "L", "player": 0, "squares": ["c1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "L", "player": 1, "squares": ["c2"]},
            ],
        }
        engine = Engine(spec)
        state = engine.initial_state()
        moves = engine.legal_moves(state)
        # A capture is available -- Lc1 takes on c2 -- so it is the only move, and none
        # of the three turns survives the filter.
        assert not [m for m in moves if decode_move(m)[2] == MOVE_ROTATE]
        assert engine.move_texts(state) == ["Lc1xc2"]


class TestEnteringPlay:
    """One rule, three callers: a piece that enters play takes its type's initial facing.

    The plan expected `rust_supports` to refuse `facing` beside `drops` and beside
    `explode_on_capture` rather than answer them. This is the answer.
    """

    def test_a_promoted_piece_takes_its_new_type_s_initial_facing(self) -> None:
        spec = _lancer_spec()
        spec["pieces"][1]["facing"] = {"axes": 4, "initial": "backward"}
        spec["pieces"].append(
            {
                "id": "P",
                "value": 1,
                "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1}],
            }
        )
        spec["setup"] = {
            "mirror": False,
            "placements": [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "P", "player": 0, "squares": ["c4"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
            ],
        }
        spec["rules"] = {
            "win": "checkmate",
            "max_plies": 80,
            "promotion": {"piece": "P", "rank": "last", "to": ["L"]},
        }
        engine = Engine(spec)
        state = engine.initial_state()
        after = engine.apply(state, engine.find_move(state, "Pc4-c5=L"))
        value = after.board[engine.spec.square_index(2, 4)]
        assert engine.spec.type_of(value) == engine.spec.piece_ids.index("L")
        # "backward" is two quarter turns, and the spec has four facings.
        assert engine.spec.facing_of(value) == 2

    def test_a_dropped_piece_takes_its_type_s_initial_facing(self) -> None:
        spec = _lancer_spec()
        spec["pieces"][1]["facing"] = {"axes": 4, "initial": "right"}
        spec["rules"] = {
            "win": "checkmate",
            "max_plies": 80,
            "drops": {"from_captures": True},
        }
        # Both lancers start facing *right*, so the capture on the rank is the one
        # available: which is itself the rule under test, one move earlier.
        spec["setup"] = {
            "mirror": False,
            "placements": [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "L", "player": 0, "squares": ["c1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "L", "player": 1, "squares": ["d1"]},
            ],
        }
        engine = Engine(spec)
        state = engine.initial_state()
        state = engine.apply(state, engine.find_move(state, "Lc1xd1"))
        state = engine.apply(state, engine.legal_moves(state)[0])
        drop = next(
            m for m in engine.legal_moves(state)
            if engine.describe_move(state, m).startswith("L@")
        )
        after = engine.apply(state, drop)
        value = after.board[decode_move(drop)[1]]
        assert engine.spec.facing_of(value) == 1  # "right"


class TestWhatTheValidatorRefuses:
    def test_a_royal_may_not_face(self) -> None:
        spec = _lancer_spec()
        spec["pieces"][0]["facing"] = {"axes": 4}
        errors = validate_spec(spec)
        assert any("royal" in error for error in errors), errors

    def test_a_piece_whose_turn_changes_nothing_may_not_face(self) -> None:
        """A rotation that changes no move is a *pass*, and a pass is a different game."""
        spec = _lancer_spec()
        # A rook is its own quarter turn.
        spec["pieces"][1]["moves"] = [
            {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
        ]
        assert validate_spec(spec) == []  # the validator cannot see it; the compiler can
        with pytest.raises(SpecError, match="does nothing"):
            Engine(spec)

    def test_a_half_turn_piece_may_not_start_pointing_sideways(self) -> None:
        spec = _lancer_spec()
        spec["pieces"][1]["facing"] = {"axes": 2, "initial": "right"}
        errors = validate_spec(spec)
        assert any("orientations" in error for error in errors), errors

    def test_a_placement_may_not_name_a_facing_the_piece_cannot_be_in(self) -> None:
        spec = _lancer_spec()
        spec["pieces"][1]["facing"] = {"axes": 2}
        spec["setup"]["placements"][1]["facing"] = "left"
        errors = validate_spec(spec)
        assert any("orientations" in error for error in errors), errors

    def test_a_placement_facing_needs_a_piece_that_has_one(self) -> None:
        spec = _lancer_spec()
        spec["setup"]["placements"][0]["facing"] = "right"  # the king
        errors = validate_spec(spec)
        assert any("no facing" in error for error in errors), errors

    def test_a_facing_declares_schema_eight(self) -> None:
        spec = _lancer_spec(schema_version=7)
        errors = validate_spec(spec)
        assert any("schema 8" in error for error in errors), errors


class TestPlacementFacing:
    def test_a_placement_decides_which_way_those_pieces_start(self) -> None:
        spec = _lancer_spec()
        spec["setup"] = {
            "mirror": True,
            "placements": [
                {"piece": "K", "squares": ["c1"]},
                {"piece": "L", "squares": ["b2"], "facing": "right"},
                {"piece": "L", "squares": ["d2"], "facing": "left"},
            ],
        }
        engine = Engine(spec)
        board = engine.initial_state().board
        assert engine.spec.facing_of(board[engine.spec.square_index(1, 1)]) == 1
        assert engine.spec.facing_of(board[engine.spec.square_index(3, 1)]) == 3

    def test_a_mirrored_placement_gives_both_armies_the_same_orientation(self) -> None:
        """Relative, like a move vector: `right` is each player's own right."""
        spec = _lancer_spec()
        spec["setup"] = {
            "mirror": True,
            "placements": [
                {"piece": "K", "squares": ["c1"]},
                {"piece": "L", "squares": ["b2"], "facing": "right"},
                {"piece": "L", "squares": ["d2"], "facing": "right"},
            ],
        }
        engine = Engine(spec)
        spec_c = engine.spec
        board = engine.initial_state().board
        for square, value in enumerate(board):
            if value and spec_c.type_of(value) == spec_c.piece_ids.index("L"):
                assert spec_c.facing_of(value) == 1, square

    def test_the_prose_says_which_way_they_start(self) -> None:
        from chess_variant_sandbox.spec.describe import describe_spec

        spec = _lancer_spec()
        with_facing = copy.deepcopy(spec)
        with_facing["setup"]["placements"][1]["facing"] = "backward"
        assert describe_spec(with_facing) != describe_spec(spec)


class TestTheBoardCanBeRead:
    def test_the_two_armies_turn_opposite_ways_on_screen(self, lancers: Engine) -> None:
        """The heading is not `facing * 90`, and that is the point of the table.

        Under `mirror` the second player's vectors are reflected, and a reflection
        reverses handedness -- so one facing index turns clockwise for the first army and
        anticlockwise for the second. A renderer that assumed `facing * 90` would draw
        one of them pointing the wrong way, and nothing else in the project would notice.
        """
        first, second = lancers.spec.facing_headings()
        assert first == (0, 90, 180, 270)
        assert second == (180, 90, 0, 270)

    def test_a_board_with_no_facing_keeps_its_two_headings(self, chess_spec: dict) -> None:
        from chess_variant_sandbox.engine.compile import compile_spec

        assert compile_spec(chess_spec).facing_headings() == ((0,), (180,))

    def test_an_icon_pair_is_not_turned_round_for_the_second_player(self) -> None:
        """A piece the board turns to *point* cannot also be turned to show its side."""
        from chess_variant_sandbox.viz.pieces import icons_for

        drawing = ("data:image/svg+xml;base64,AAA", "data:image/svg+xml;base64,BBB")
        pair = icons_for(["lancer"], ids=["L"], overrides={"L": list(drawing)})[0]
        assert pair[0] is not None and pair[1] is not None
        assert pair[0]["href"] != pair[1]["href"]
        assert pair[0]["rotate"] is False and pair[1]["rotate"] is False

    def test_one_drawing_is_still_turned_round(self) -> None:
        from chess_variant_sandbox.viz.pieces import icons_for

        pair = icons_for(
            ["lancer"], ids=["L"], overrides={"L": "data:image/svg+xml;base64,AAA"}
        )[0]
        assert pair[0] is not None and pair[1] is not None
        assert pair[0]["rotate"] is False and pair[1]["rotate"] is True


class TestTheShowcase:
    """`weathervane` is `sardines` with directional pawns and nothing else changed.

    Since Q180 (2026-09-11) all four were cells of one family, built by one generator from
    one placement procedure, so the equalities below hold by construction rather than by
    a test noticing afterwards. The two wrap cells, `sardines-torus` and
    `weathervane-torus`, left the library on 2026-10-12 (Q868, `docs/plan-library-audit.md`):
    the wrap changed nothing a reader could see on a full board, so the family is now
    `sardines` and its facing cell. The positions differ between cells -- each cell picked
    its own seed on the screen -- and the equalities are over board, rules and pieces,
    which is where a feature lives. `docs/reference/sardines-family.md` is the campaign.
    """

    FAMILY = {
        "archive/weathervane.json": "facing",
    }

    def test_it_validates_and_compiles(self) -> None:
        from chess_variant_sandbox.spec import load_spec

        spec = load_spec("archive/weathervane.json")
        assert validate_spec(spec) == []
        engine = Engine(spec)
        assert engine.spec.n_facings == 4
        assert engine.spec.hash == "1442513d38d7"

    def test_only_the_pawn_differs_from_sardines(self) -> None:
        """The control is exact, which is what makes the measurement readable."""
        from chess_variant_sandbox.spec import load_spec

        weathervane = load_spec("archive/weathervane.json")
        sardines = load_spec("rulesets/sardines.json")
        assert weathervane["board"] == sardines["board"]
        assert weathervane["rules"] == sardines["rules"]
        stripped = [
            {k: v for k, v in piece.items() if k != "facing"}
            for piece in weathervane["pieces"]
        ]
        assert stripped == sardines["pieces"]
        assert [p["id"] for p in weathervane["pieces"] if "facing" in p] == ["P"]

    @pytest.mark.parametrize("path", sorted(FAMILY))
    def test_each_cell_differs_from_the_plain_one_by_its_feature_alone(
        self, path: str
    ) -> None:
        """The same control over every cell: its feature, and nothing else.

        A cell that gained a rule elsewhere would still pass the pairwise check above
        while breaking the family, so each cell is read against `sardines` directly.
        """
        from chess_variant_sandbox.spec import load_spec

        cell = load_spec(path)
        sardines = load_spec("rulesets/sardines.json")
        feature = self.FAMILY[path]
        wraps = feature in ("wrap", "both")
        faces = feature in ("facing", "both")

        assert cell["board"] == sardines["board"]
        rules = {k: v for k, v in cell["rules"].items() if k != "wrap"}
        assert rules == sardines["rules"]
        assert ("wrap" in cell["rules"]) is wraps
        if wraps:
            assert cell["rules"]["wrap"] == {"files": "both", "ranks": "both"}

        stripped = [
            {k: v for k, v in piece.items() if k != "facing"} for piece in cell["pieces"]
        ]
        assert stripped == sardines["pieces"]
        assert [p["id"] for p in cell["pieces"] if "facing" in p] == (["P"] if faces else [])

    def test_most_of_its_opening_moves_are_turns(self) -> None:
        """The branching cost, as the number the plan pre-registered against."""
        from chess_variant_sandbox.spec import load_spec

        engine = Engine(load_spec("archive/weathervane.json"))
        moves = engine.legal_moves(engine.initial_state())
        turns = [m for m in moves if decode_move(m)[2] == MOVE_ROTATE]
        assert (len(moves), len(turns)) == (113, 72)

    @pytest.mark.slow
    def test_a_whole_game_keeps_its_key_and_its_orientations(self) -> None:
        """The two things no perft can see, walked over a real game.

        `every_piece_is_in_an_orientation_it_may_be_in` is the fuzzer's oracle; this is
        the same check on the one ruleset that actually ships with a facing.
        """
        from chess_variant_sandbox.spec import load_spec

        engine = Engine(load_spec("archive/weathervane.json"))
        spec = engine.spec
        rng = random.Random(11)
        state = engine.initial_state()
        for _ in range(120):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state)
            for value in state.board:
                if value:
                    step = spec.facing_step(spec.type_of(value))
                    assert spec.facing_of(value) % step == 0


# ------------------------------------------------ Q922: the byte bound, from both sides


def _board_value_errors(pieces: list, owners: int = 2, extra_types: int = 0) -> list[str]:
    from chess_variant_sandbox.spec.schema import _check_facing_coherence

    errors: list[str] = []
    _check_facing_coherence(pieces, errors, owners=owners, extra_types=extra_types)
    return errors


def _facing(axes: object) -> dict:
    return {"id": "X", "facing": {"axes": axes}}


class TestTheByteBoundSitsAtTwoFiftySix:
    """`owners * facings * types` must stay under 256: 255 is the last that fits."""

    @pytest.mark.parametrize(
        ("pieces", "owners", "extra", "refused"),
        [
            ([{"id": "X"}] * 127, 2, 0, False),  # 254
            ([{"id": "X"}] * 128, 2, 0, True),   # 256: the boundary itself
            ([{"id": "X"}] * 85, 3, 0, False),   # 255
            ([{"id": "X"}] * 86, 3, 0, True),    # 258
            ([{"id": "X"}] * 120, 2, 8, True),   # extra types count: 256
            ([{"id": "X"}] * 120, 2, 7, False),  # 254
            ([_facing(4)] + [{"id": "X"}] * 23, 2, 8, True),   # 4 facings, 32 types: 256
            ([_facing(4)] + [{"id": "X"}] * 23, 2, 7, False),  # 31 types: 248
            ([_facing(2)] + [{"id": "X"}] * 23, 2, 40, True),  # 2 facings, 64 types: 256
            ([_facing(2)] + [{"id": "X"}] * 23, 2, 39, False),  # 63 types: 252
        ],
        ids=range(10),
    )
    def test_the_bound(self, pieces: list, owners: int, extra: int, refused: bool) -> None:
        highest = owners * max(
            (p["facing"]["axes"] for p in pieces if "facing" in p), default=1
        ) * (len(pieces) + extra)
        assert (highest >= 256) is refused  # the table's own arithmetic
        assert bool(_board_value_errors(pieces, owners, extra)) is refused

    def test_a_facing_count_that_is_not_a_whole_number_is_not_read(self) -> None:
        """A bool or a None where the axes go is the facing check's to refuse, not a count."""
        for axes in (None, "4", 2.5, False):
            assert _board_value_errors([_facing(axes)] * 127) == []
            assert _board_value_errors([_facing(axes)] * 128) != []

    def test_the_message_names_the_owners_and_the_arithmetic(self) -> None:
        (two,) = _board_value_errors([{"id": "X"}] * 128)
        assert two == (
            "128 piece types at 1 facings for two players need 257 board values, "
            "and a square is one byte; reduce the piece count or the facings"
        )
        (three,) = _board_value_errors([{"id": "X"}] * 90, owners=3, extra_types=2)
        assert "92 piece types at 1 facings for two players and a wild third owner need 277" in three


class TestAWildOwnerOrAPortalIsAThirdOwner:
    """The caller decides the owners: `validate_spec` passes three when anything is wild."""

    @staticmethod
    def _errors(n: int, rules: dict) -> list[str]:
        from chess_variant_sandbox.spec import validate_spec

        spec = load_spec(Path(__file__).resolve().parents[1] / "rulesets" / "chess.json")
        spec["pieces"] = [
            {"id": f"{chr(65 + i // 10)}{i % 10}", "moves": [{"kind": "leap", "vectors": [[1, 0]]}],
             "facing": {"axes": 4}}
            for i in range(n)
        ]
        spec["setup"] = {"placements": []}
        spec["rules"] = rules
        return [e for e in validate_spec(spec) if "board values" in e]

    @pytest.mark.parametrize(
        ("count", "rules", "refused"),
        [
            (24, {}, False),                  # 2 * 4 * 24 = 192
            (31, {}, False),                  # 248: past the piece-count limit, still fits
            (32, {}, True),                   # 256 (the piece count is refused apart)
            (21, {"npcs": [{}]}, False),      # 3 * 4 * 21 = 252
            (22, {"npcs": [{}]}, True),       # 264
            (20, {"portals": [{}]}, True),    # two internal types a pair: 3 * 4 * 22 = 264
            (19, {"portals": [{}]}, False),   # 3 * 4 * 21 = 252
            (22, {"portals": []}, False),     # an empty list is no portals
        ],
        ids=str,
    )
    def test_the_owners_follow_the_rules(self, count: int, rules: dict, refused: bool) -> None:
        assert bool(self._errors(count, rules)) is refused
