"""Engine mutation report, F-engine-3: defaults left out and rule variants.

The `compile.py` (and a few `bridge.py`) mutants of `docs/reference/mutation/engine.md`
whose killing test belongs beside `test_schema_two.py`, `test_schema_six.py`,
`test_schema_thirteen.py`, `test_schema_fifteen.py`, `test_schema_seventeen.py`,
`test_schema_twenty_six.py`, `test_goal_terms.py`, `test_seal.py` and the smaller schema files:
a key a spec leaves out, a rule written in a second shape, a goal race of a new shape, a hop
or wall beside a sweep. Each test names the input the mutant could not survive. Every spec is
a small purpose-made dictionary or a committed ruleset with one key changed; none asserts a
number that only chess has.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import _first_disagreement, _where, rust_supports
from chess_variant_sandbox.engine.compile import UNREACHABLE, compile_spec
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.schema import MAX_LANDING_CANDIDATES, SpecError

ROOT = Path(__file__).resolve().parents[1]
KING = {"id": "K", "name": "king", "value": 0, "royal": True,
        "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]}


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _probe(
    pieces: list[dict[str, Any]],
    rules: dict[str, Any] | None = None,
    placements: list[dict[str, Any]] | None = None,
    width: int = 5,
    height: int = 5,
    setup: dict[str, Any] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Two kings and ``pieces`` on a bare board; ``placements`` replaces the kings' squares."""
    spec = {
        "schema_version": 26,
        "name": "probe",
        "board": {"width": width, "height": height},
        "pieces": [KING, *pieces],
        "setup": {"mirror": False, **(setup or {}), "placements": placements or [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": [f"{chr(96 + width)}{height}"]},
        ]},
        "rules": {"win": "royal_capture", **(rules or {})},
    }
    if validate:
        assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _texts(engine: Engine, state: Any, piece: str) -> list[str]:
    return sorted(t for t in engine.move_texts(state) if t.startswith(piece))


def _targets(engine: Engine, state: Any, piece: str, origin: str) -> set[str]:
    prefix = f"{piece}{origin}"
    return {t[len(prefix) + 1:] for t in engine.move_texts(state) if t.startswith(prefix)}


# ----------------------------------------------------------------------- facing and turning


class TestFacingAndTurning:
    FACER = {"id": "F", "name": "facer", "value": 1, "facing": {"axes": 4},
             "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 2}]}

    def test_a_piece_without_a_facing_never_turns_and_a_facing_one_does(self) -> None:
        chess = compile_spec(_load("rulesets/chess.json"))
        assert [chess.turns(t) for t in range(chess.n_types)] == [False] * chess.n_types
        facing = compile_spec(_probe([self.FACER]))
        assert [facing.turns(t) for t in range(facing.n_types)] == [False, True]

    def test_the_headings_of_a_rotated_army_are_the_first_players_turned_half_way(self) -> None:
        chess = _load("rulesets/chess.json")
        chess["schema_version"] = 2
        chess["setup"]["orientation"] = "rotate"
        assert compile_spec(chess).facing_headings() == ((0,), (180,))
        spec = _probe([self.FACER], setup={"orientation": "rotate"})
        first, second = compile_spec(spec).facing_headings()
        assert first == (0, 90, 180, 270)
        assert second == (180, 270, 0, 90)

    def test_the_headings_of_a_mirrored_army_keep_the_sideways_ones(self) -> None:
        spec = _probe([self.FACER], setup={"orientation": "mirror"})
        assert compile_spec(spec).facing_headings()[1] == (180, 90, 0, 270)

    def test_a_turn_that_is_a_pass_for_the_first_player_alone_is_still_refused(self) -> None:
        # A 2x4 board walled at b2-b4: the leap (1, 2) goes nowhere from any square, in
        # either direction, for the first player. The second player's frame reflects the
        # leap, so a3 and b1 reach each other, and reading that frame would let it pass.
        piece = {"id": "F", "name": "facer", "value": 1, "facing": {"axes": 2},
                 "moves": [{"kind": "leap", "vectors": [[1, 2]]}]}
        spec = _probe(
            [piece],
            rules={"obstacles": {"squares": ["b2", "b3", "b4"]}},
            placements=[
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["a4"]},
                {"piece": "F", "player": 0, "squares": ["a2"]},
            ],
            width=2, height=4, validate=False,
        )
        with pytest.raises(SpecError, match="facing"):
            compile_spec(spec)


# ----------------------------------------------------------------------- walls


class TestWalls:
    WAZIR = {"id": "W", "name": "wazir", "value": 1,
             "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]}
    HORSE = {"id": "H", "name": "horse", "value": 1,
             "moves": [{"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8"}]}

    @staticmethod
    def _horse_game(walls: list[str], horse: dict[str, Any]) -> tuple[Engine, Any]:
        spec = _probe(
            [horse],
            rules={"obstacles": {"squares": walls}},
            placements=[
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "H", "player": 0, "squares": ["c3"]},
            ],
        )
        engine = Engine(spec)
        return engine, engine.make_state(
            {"a1": ("K", 0), "e5": ("K", 1), "c3": ("H", 0)}
        )

    ALL = {"a2", "a4", "b1", "b5", "d1", "d5", "e2", "e4"}
    LEG_OF = {"a2": "b3", "a4": "b3", "b1": "c2", "d1": "c2",
              "b5": "c4", "d5": "c4", "e2": "d3", "e4": "d3"}

    @pytest.mark.parametrize("target", sorted(ALL))
    def test_a_wall_on_one_landing_takes_that_jump_and_no_other(self, target: str) -> None:
        engine, state = self._horse_game([target], self.HORSE)
        assert _targets(engine, state, "H", "c3") == self.ALL - {target}

    @pytest.mark.parametrize("leg", ["b3", "c2", "c4", "d3"])
    def test_a_wall_on_a_leg_takes_the_two_jumps_over_it_and_no_other(self, leg: str) -> None:
        engine, state = self._horse_game([leg], self.HORSE)
        blocked = {t for t, wall in self.LEG_OF.items() if wall == leg}
        assert len(blocked) == 2
        assert _targets(engine, state, "H", "c3") == self.ALL - blocked

    @pytest.mark.parametrize("direction", [(1, 1), (-1, 1), (1, -1), (-1, -1)])
    def test_a_long_diagonal_lame_leap_is_stopped_at_its_first_square_not_its_second(
        self, direction: tuple[int, int]
    ) -> None:
        # The leg of a (3, 3) leap is (1, 1): halved toward the mover, in every direction.
        leaper = {"id": "L", "name": "leaper", "value": 1,
                  "moves": [{"kind": "lame", "vectors": [[3, 3]], "symmetry": "rotate4"}]}
        dx, dy = direction

        def square(file: int, rank: int) -> str:
            return f"{chr(ord('a') + file)}{rank + 1}"

        for wall, stops in ((square(3 + dx, 3 + dy), True), (square(3 + 2 * dx, 3 + 2 * dy), False)):
            spec = _probe(
                [leaper],
                rules={"obstacles": {"squares": [wall]}},
                placements=[
                    {"piece": "K", "player": 0, "squares": ["d1"]},
                    {"piece": "K", "player": 1, "squares": ["d7"]},
                    {"piece": "L", "player": 0, "squares": ["d4"]},
                ],
                width=7, height=7,
            )
            engine = Engine(spec)
            state = engine.make_state({"d1": ("K", 0), "d7": ("K", 1), "d4": ("L", 0)})
            target = square(3 + 3 * dx, 3 + 3 * dy)
            assert (target in _targets(engine, state, "L", "d4")) is (not stops), (wall, target)

    def test_a_one_square_lame_leap_of_a_stored_spec_gives_the_piece_no_such_move(self) -> None:
        # Admission refuses it (Q884); a stored spec still compiles, and the step is no move.
        piece = {"id": "H", "name": "horse", "value": 1,
                 "moves": [{"kind": "lame", "vectors": [[1, 2], [1, 0]],
                            "symmetry": "dihedral8"}]}
        spec = _probe([piece], placements=[
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["e5"]},
            {"piece": "H", "player": 0, "squares": ["c3"]},
        ], validate=False)
        engine = Engine(spec)
        state = engine.make_state({"a1": ("K", 0), "e5": ("K", 1), "c3": ("H", 0)})
        assert _targets(engine, state, "H", "c3") == TestWalls.ALL

    def test_a_player_the_walls_shut_in_is_named_whichever_player_it_is(self) -> None:
        # Unmirrored walls round the second player's only piece, in a corner.
        spec = _probe(
            [],
            rules={"obstacles": {"squares": ["b3", "b2", "c2"]}},
            placements=[
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["c3"]},
            ],
            width=3, height=3, validate=False,
        )
        with pytest.raises(SpecError, match="walls player 1 in"):
            compile_spec(spec)
        shut_first = copy.deepcopy(spec)
        shut_first["setup"]["placements"] = [
            {"piece": "K", "player": 0, "squares": ["c3"]},
            {"piece": "K", "player": 1, "squares": ["a1"]},
        ]
        with pytest.raises(SpecError, match="walls player 0 in"):
            compile_spec(shut_first)


# ----------------------------------------------------------------------- sweeps and hops


class TestSweepsAndHops:
    SWEEPER = {"id": "S", "name": "sweeper", "value": 1,
               "moves": [{"kind": "sweep", "vectors": [[1, 0]], "range": 4, "mode": "move"}]}
    CANNON = {"id": "C", "name": "cannon", "value": 1,
              "moves": [{"kind": "hop", "vectors": [[1, 0]], "mode": "capture"}]}
    BLOCK = {"id": "B", "name": "block", "value": 1,
             "moves": [{"kind": "leap", "vectors": [[0, 1]]}]}

    def test_the_sweep_path_is_empty_where_no_piece_stands(self) -> None:
        # The sweeper is the last type declared, which is the type an empty square's byte
        # (zero) decodes to: asked of an empty square it must still answer "no sweep".
        compiled = compile_spec(_probe([self.SWEEPER], placements=[
            {"piece": "K", "player": 0, "squares": ["a5"]},
            {"piece": "K", "player": 1, "squares": ["e5"]},
            {"piece": "S", "player": 0, "squares": ["a1"]},
        ]))
        board = bytes(compiled.initial_board)
        a1, c1, e1 = 0, 2, 4
        assert compiled.sweep_path(board, a1, e1) == (1, 2, 3)
        assert compiled.sweep_path(board, c1, e1) == ()
        assert compiled.sweep_path(board, 1, e1) == ()

    def test_a_hop_still_needs_its_screen_in_a_game_that_also_sweeps(self) -> None:
        spec = _probe(
            [self.SWEEPER, self.CANNON, self.BLOCK],
            placements=[
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e3"]},
                {"piece": "S", "player": 0, "squares": ["a5"]},
                {"piece": "C", "player": 0, "squares": ["a3"]},
            ],
        )
        engine = Engine(spec)
        assert engine.spec.has_sweep and engine.spec.has_hop_attacks
        bare = {"a1": ("K", 0), "e3": ("K", 1), "a3": ("C", 0), "a5": ("S", 0)}
        assert not engine.in_check(engine.make_state(bare, side_to_move=1), 1)
        screened = engine.make_state({**bare, "c3": ("B", 0)}, side_to_move=1)
        assert engine.in_check(screened, 1)

    def test_a_hop_with_a_near_screen_is_exact_in_a_game_that_also_sweeps(self) -> None:
        grasshopper = {"id": "G", "name": "grasshopper", "value": 1,
                       "moves": [{"kind": "hop", "vectors": [[1, 0]], "mode": "capture",
                                  "screen_range": 1}]}
        spec = _probe(
            [self.SWEEPER, grasshopper, self.BLOCK],
            placements=[
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e3"]},
                {"piece": "S", "player": 0, "squares": ["a5"]},
                {"piece": "G", "player": 0, "squares": ["a3"]},
            ],
        )
        engine = Engine(spec)
        base = {"a1": ("K", 0), "e3": ("K", 1), "a3": ("G", 0), "a5": ("S", 0)}
        far = engine.make_state({**base, "c3": ("B", 0)}, side_to_move=1)
        near = engine.make_state({**base, "d3": ("B", 0)}, side_to_move=1)
        assert not engine.in_check(far, 1)
        assert engine.in_check(near, 1)

    def test_a_move_only_sweep_checks_a_royal_it_could_pass_but_not_one_at_its_end(
        self,
    ) -> None:
        def game(king: str) -> bool:
            spec = _probe([self.SWEEPER], placements=[
                {"piece": "K", "player": 0, "squares": ["a5"]},
                {"piece": "K", "player": 1, "squares": [king]},
                {"piece": "S", "player": 0, "squares": ["a1"]},
            ])
            engine = Engine(spec)
            state = engine.make_state(
                {"a5": ("K", 0), king: ("K", 1), "a1": ("S", 0)}, side_to_move=1
            )
            return engine.in_check(state, 1)

        assert game("d1") is True   # a landing beyond it, on e1
        compiled = compile_spec(_probe([self.SWEEPER], placements=[
            {"piece": "K", "player": 0, "squares": ["a5"]},
            {"piece": "K", "player": 1, "squares": ["e5"]},
            {"piece": "S", "player": 0, "squares": ["a1"]},
        ]))
        # The compiled table says the same: the last square is no source at all, and a
        # square with a landing past it is one whose bits are "beyond" alone.
        sources = compiled.attack_sources[0]
        assert [s for s in sources[4] if s[0] == 0] == []
        assert [s[7] for s in sources[3] if s[0] == 0] == [2]
        assert game("e1") is False  # the last square of the ray: nothing beyond, none to take

    def test_a_wild_piece_whose_next_step_is_a_one_attacks_it(self) -> None:
        wild = {"id": "W", "name": "boar", "value": 1,
                "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]}
        spec = _probe(
            [wild],
            rules={"npcs": [{"piece": "W", "route": ["b1", "a1"], "mirror": False}]},
            placements=[
                {"piece": "K", "player": 0, "squares": ["c3"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
            ],
        )
        compiled = compile_spec(spec)
        a1 = 0
        wild_block = compiled.attack_sources[2]
        assert [source[0] for source in wild_block[a1]] == [1]  # the boar on b1 steps to a1
        assert [source[0] for source in wild_block[1]] == [0]   # and from a1 back to b1


# ----------------------------------------------------------------------- the boundary's words


class TestTheBoundarySaysWhereItDisagrees:
    def test_a_path_is_named_innermost_field_first(self) -> None:
        assert _where((11,)) == "slot 11"
        assert _where((11, 2)) == "field 2 of slot 11"
        assert _where((3, 4, 5)) == "field 5 of field 4 of slot 3"

    def test_a_group_on_one_side_and_a_value_on_the_other_is_named(self) -> None:
        mine = (1, ((1, (None,)),))
        theirs = (1, (None,))
        assert _first_disagreement(mine, theirs) == (
            "slot 0 is a group on one side of the boundary and a value on the other"
        )
        assert _first_disagreement(theirs, mine) == (
            "slot 0 is a group on one side of the boundary and a value on the other"
        )
        assert _first_disagreement(theirs, theirs) is None

    def test_a_field_count_that_differs_is_named_with_its_place(self) -> None:
        mine = (2, ((2, (None, None)), None))
        theirs = (2, ((3, (None, None, None)), None))
        assert _first_disagreement(mine, theirs) == (
            "slot 0 carries 3 fields where this package writes 2"
        )
        deeper = (1, ((2, (None, None)),))
        assert _first_disagreement(deeper, (1, ((3, (None,) * 3),))) == (
            "slot 0 carries 3 fields where this package writes 2"
        )
        nested = (1, ((1, ((2, (None, None)),)),))
        assert _first_disagreement(nested, (1, ((1, ((3, (None,) * 3),)),))) == (
            "field 0 of slot 0 carries 3 fields where this package writes 2"
        )


class TestALandingAtTheBound:
    @staticmethod
    def _throw() -> dict[str, Any]:
        return _load("archive/hierarchy-rise-throw.json")

    def test_exactly_the_most_candidates_one_square_may_have_is_still_supported(self) -> None:
        compiled = compile_spec(self._throw())
        strike = compiled.strike
        assert strike is not None
        for count, supported in ((MAX_LANDING_CANDIDATES, True), (MAX_LANDING_CANDIDATES + 1, False)):
            wide = tuple(range(count))
            landing = tuple(
                tuple(wide if square == 0 else row for square, row in enumerate(per))
                for per in strike.landing
            )
            widened = dataclasses.replace(
                compiled, strike=dataclasses.replace(strike, landing=landing)
            )
            assert (rust_supports(widened) is None) is supported, count

    def test_a_far_landing_is_not_stopped_by_a_wall_between_unless_the_spec_says_so(self) -> None:
        spec = self._throw()
        spec["rules"]["strike"]["landing"] = {"vectors": [[0, 2]]}
        spec["rules"]["obstacles"] = {"squares": ["e5"]}
        strike = compile_spec(spec).strike
        assert strike is not None
        e4, e6 = 3 * 8 + 4, 5 * 8 + 4
        assert strike.landing[0][e4] == (e6,)
        assert strike.landing[1][5 * 8 + 4] == (e4,)

    def test_a_wrapped_board_lands_the_second_player_across_the_seam_too(self) -> None:
        spec = self._throw()
        spec["rules"]["strike"]["landing"] = {"vectors": [[1, 0]]}
        spec["rules"]["wrap"] = {"files": "both"}
        spec["schema_version"] = max(spec["schema_version"], 12)
        strike = compile_spec(spec).strike
        assert strike is not None
        h4, a4 = 3 * 8 + 7, 3 * 8
        assert strike.landing[0][h4] == (a4,)
        assert strike.landing[1][h4] == (a4,)


# ----------------------------------------------------------------------- one-way pieces


def _gardner_with_ring(enter: list[int] | None) -> dict[str, Any]:
    """Gardner with a ring of three portal ends on the empty third rank; ``enter`` (when
    given) says whose pieces may go in at every end."""
    spec = _load("rulesets/gardner.json")
    spec["schema_version"] = 26
    # The pawn declared after the other pieces, so its type index is above zero.
    spec["pieces"].sort(key=lambda piece: piece["id"] == "P")
    spec["pieces"].append({"id": "O", "name": "portal", "value": 0, "moves": []})
    ends = [{"at": "a3", "to": 1}, {"at": "c3", "to": 2}, {"at": "e3", "to": 0}]
    if enter is not None:
        ends = [{**end, "enter": enter} for end in ends]
    spec["rules"]["portals"] = [{"piece": "O", "ends": ends}]
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


class TestWhichPiecesAreOneWayBesidePortals:
    def test_a_pawn_a_door_admits_is_not_one_way_whoever_may_go_in(self) -> None:
        for enter in (None, [0], [1]):
            compiled = compile_spec(_gardner_with_ring(enter))
            pawn = compiled.piece_ids.index("P")
            assert pawn > 0  # a type above the first: the mask is read bit by bit
            assert pawn not in compiled.one_way_types, enter

    def test_a_pawn_no_door_admits_stays_one_way(self) -> None:
        spec = _gardner_with_ring(None)
        spec["rules"]["portals"][0]["ends"] = [
            {**end, "carries": ["N"]} for end in spec["rules"]["portals"][0]["ends"]
        ]
        assert validate_spec(spec) == [], validate_spec(spec)
        compiled = compile_spec(spec)
        assert compiled.piece_ids.index("P") in compiled.one_way_types
        assert compiled.piece_ids.index("N") not in compiled.one_way_types


class TestAFixedFacingWhoseCycleIsInASidewaysFacing:
    def test_a_stepper_fixed_sideways_on_a_wrapped_rank_is_not_one_way(self) -> None:
        piece = {"id": "F", "name": "marcher", "value": 1,
                 "facing": {"axes": 4, "turns": False},
                 "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]}
        spec = _probe(
            [piece],
            # Only the first player's files wrap: the second player's graph is acyclic, so
            # the cycle is the first player's alone, and it lies in a later facing.
            rules={"wrap": {"files_by_player": {"0": "both", "1": "none"}}},
            placements=[
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["d3"]},
                {"piece": "F", "player": 0, "squares": ["b1"], "facing": "forward"},
                {"piece": "F", "player": 0, "squares": ["c1"], "facing": "right"},
            ],
            width=4, height=3,
        )
        compiled = compile_spec(spec)
        assert compiled.piece_ids.index("F") not in compiled.one_way_types
        # With no seam no facing circles a rank, and the same stepper only ever climbs.
        del spec["rules"]["wrap"]
        compiled = compile_spec(spec)
        assert compiled.piece_ids.index("F") in compiled.one_way_types


# ----------------------------------------------------------------------- simultaneous games


def _round_game(**simultaneous: Any) -> dict[str, Any]:
    spec = _load("tests/_simultaneous_gardner.json")
    spec["rules"]["simultaneous"] = simultaneous
    return spec


class TestTheSimultaneousRecord:
    def test_the_three_ties_are_three_numbers(self) -> None:
        from chess_variant_sandbox.engine.compile import (
            TIE_BOTH,
            TIE_IMPOSSIBLE,
            TIE_INITIATIVE,
        )

        assert (TIE_BOTH, TIE_INITIATIVE, TIE_IMPOSSIBLE) == (0, 1, 2)

    def test_an_impossible_tie_is_compiled_as_the_third_word(self) -> None:
        circle = _circle()
        assert validate_spec(circle) == [], validate_spec(circle)
        rule = compile_spec(circle).simultaneous
        assert rule is not None and rule.tie == 2
        assert compile_spec(_round_game(tie="initiative")).simultaneous.tie == 1  # type: ignore[union-attr]
        assert compile_spec(_round_game()).simultaneous.tie == 0  # type: ignore[union-attr]

    def test_a_bare_plain_capture_strike_beside_rounds_is_no_collision_strike(self) -> None:
        spec = _round_game()
        spec["rules"]["strike"] = {"effect": "capture"}
        assert validate_spec(spec) == [], validate_spec(spec)
        rule = compile_spec(spec).simultaneous
        assert rule is not None and rule.collision_strikes is False
        spec["rules"]["strike"] = {"effect": "promote", "immune": ["K"], "position": "swap",
                                   "chain": [{"from": "P", "to": "Q"}]}
        assert validate_spec(spec) == [], validate_spec(spec)
        live = compile_spec(spec).simultaneous
        assert live is not None and live.collision_strikes is True

    def test_a_game_without_folds_draws_no_fold_keys_before_the_standoff_ones(self) -> None:
        from chess_variant_sandbox.engine.compile import _build_zobrist

        for tie in ("initiative", None):
            compiled = compile_spec(_round_game(**({"tie": tie} if tie else {})))
            keys = _build_zobrist(
                compiled.hash, compiled.n_squares, compiled.n_types, compiled.width,
                compiled.n_facings, compiled.n_owners,
                simultaneous=True, initiative=tie == "initiative",
            )
            rule = compiled.simultaneous
            assert rule is not None and keys[7] is not None
            assert keys[7] == (rule.zobrist_initiative, rule.zobrist_squares, rule.zobrist_states)
            # Schema 29's resting pieces (Q983a) are drawn last of all, after the standoff's,
            # and only where a type rests: none here.
            assert keys[8] == ()
            assert compiled.zobrist_rest == ()

    def test_the_standoff_keys_are_as_many_as_they_are_read_and_sixty_four_bits_wide(
        self,
    ) -> None:
        widest = {"initiative": 0, "squares": 0, "states": 0}
        for width, height in [(w, h) for w in range(4, 17) for h in (4, 5)]:
            for tie in ("initiative", None):
                spec = _round_game(**({"tie": tie} if tie else {}))
                spec["board"]["width"] = width
                spec["board"]["height"] = height
                spec["setup"] = {"mirror": False, "placements": [
                    {"piece": "K", "player": 0, "squares": ["a1"]},
                    {"piece": "K", "player": 1, "squares": [f"{chr(96 + width)}{height}"]},
                ]}
                spec["rules"].pop("promotion", None)
                spec["pieces"] = [p for p in spec["pieces"] if p["id"] == "K"]
                compiled = compile_spec(spec)
                rule = compiled.simultaneous
                assert rule is not None
                assert len(rule.zobrist_squares) == 2 * compiled.n_squares
                assert len(rule.zobrist_states) == rule.void_run
                keys = [rule.zobrist_initiative, *rule.zobrist_squares, *rule.zobrist_states]
                assert all(0 <= key < 2**64 for key in keys)
                if tie is None:
                    assert rule.zobrist_initiative == 0
                widest["initiative"] = max(widest["initiative"], rule.zobrist_initiative.bit_length())
                widest["squares"] = max(widest["squares"], *(k.bit_length() for k in rule.zobrist_squares))
                widest["states"] = max(widest["states"], *(k.bit_length() for k in rule.zobrist_states))
        # Each group is a full 64-bit draw; over twenty-six boards one of its keys has the top bit.
        assert widest == {"initiative": 64, "squares": 64, "states": 64}


def _circle() -> dict[str, Any]:
    """Rock, paper, scissors for each player, every piece unique: no tie is possible."""
    steps = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
    pieces = [
        {"id": "R", "moves": steps, "captures": ["s", "r2"], "royal": True},
        {"id": "S", "moves": steps, "captures": ["p", "s2"]},
        {"id": "P", "moves": steps, "captures": ["r", "p2"]},
        {"id": "r", "moves": steps, "captures": ["S", "R"], "royal": True},
        {"id": "s", "moves": steps, "captures": ["P", "S"]},
        {"id": "p", "moves": steps, "captures": ["R", "P"]},
        {"id": "r2", "moves": steps, "captures": ["S"]},
        {"id": "s2", "moves": steps, "captures": ["P"]},
        {"id": "p2", "moves": steps, "captures": ["R"]},
    ]
    return {
        "schema_version": 26,
        "name": "circle",
        "board": {"width": 4, "height": 4},
        "pieces": pieces,
        "setup": {"mirror": False, "placements": [
            {"piece": "R", "player": 0, "squares": ["a1"]},
            {"piece": "S", "player": 0, "squares": ["b1"]},
            {"piece": "P", "player": 0, "squares": ["c1"]},
            {"piece": "r", "player": 1, "squares": ["d4"]},
            {"piece": "s", "player": 1, "squares": ["c4"]},
            {"piece": "p", "player": 1, "squares": ["b4"]},
        ]},
        "rules": {"win": "royal_capture", "simultaneous": {"tie": "impossible"}},
    }


# ----------------------------------------------------------------------- defaults left out


def _chess(**rules: Any) -> dict[str, Any]:
    spec = _load("rulesets/chess.json")
    spec["rules"].update(rules)
    return spec


def _move_between(engine: Engine, state: Any, origin: str, target: str) -> int:
    from chess_variant_sandbox.engine.moves import move_from, move_to

    def square(name: str) -> int:
        return (int(name[1:]) - 1) * engine.spec.width + ord(name[0]) - ord("a")

    found = [m for m in engine.legal_moves(state)
             if (move_from(m), move_to(m)) == (square(origin), square(target))]
    assert len(found) == 1, (origin, target, found)
    return found[0]


class TestADefaultALeftOutKeepsItsMeaning:
    def test_a_wild_piece_with_no_value_and_no_bounty_is_worth_one(self) -> None:
        wild = {"id": "W", "name": "boar",
                "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]}
        spec = _probe(
            [wild],
            rules={"npcs": [{"piece": "W", "route": ["b2", "b3", "c3", "c2"],
                             "mirror": False}]},
            placements=[
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
            ],
        )
        npcs = compile_spec(spec).npcs
        assert npcs is not None
        route = [(1 * 5 + 1), (2 * 5 + 1), (2 * 5 + 2), (1 * 5 + 2)]
        assert [npcs.bounty[square] for square in route] == [1.0, 1.0, 1.0, 1.0]

    def test_drops_that_leave_out_from_captures_still_hand_over_what_was_taken(self) -> None:
        spec = _load("rulesets/crazyhouse.json")
        del spec["rules"]["drops"]["from_captures"]
        compiled = compile_spec(spec)
        assert compiled.drops is not None and compiled.drops.from_captures is True
        engine = Engine(spec)
        state = engine.initial_state()
        for origin, target in (("e2", "e4"), ("d7", "d5")):
            state = engine.apply(state, _move_between(engine, state, origin, target))
        after = engine.apply(state, _move_between(engine, state, "e4", "d5"))
        assert any(after.hands[0]) and not any(after.hands[1])

    def test_an_extinction_that_leaves_out_min_count_ends_on_the_last_piece_taken(self) -> None:
        spec = _load("rulesets/chess.json")
        spec["schema_version"] = 3
        spec["rules"]["extinction"] = {"pieces": ["Q"]}
        spec["rules"]["win"] = "royal_capture"
        assert validate_spec(spec) == [], validate_spec(spec)
        compiled = compile_spec(spec)
        assert compiled.extinction is not None and compiled.extinction.min_count == 1
        engine = Engine(spec)
        state = engine.make_state({
            "a1": ("K", 0), "h1": ("R", 0), "a2": ("Q", 0), "a8": ("K", 1),
            "d8": ("Q", 1), "h8": ("Q", 1),
        })
        first = engine.apply(state, _move_between(engine, state, "h1", "h8"))
        assert engine.terminal(first) is None  # one queen is gone, one is left
        second_state = engine.make_state({
            "a1": ("K", 0), "d1": ("R", 0), "a2": ("Q", 0), "a8": ("K", 1), "d8": ("Q", 1),
        })
        last = engine.apply(second_state, _move_between(engine, second_state, "d1", "d8"))
        outcome = engine.terminal(last)
        assert outcome is not None and outcome.winner == 0

    def test_a_link_whose_mover_is_the_second_type_alone_needs_the_pair_of_tables(self) -> None:
        chess = _load("rulesets/chess.json")
        second = compile_spec(chess).piece_ids[1]
        spec = copy.deepcopy(chess)
        spec["schema_version"] = 10
        spec["rules"]["contacts"] = {
            "position": "stay", "immune": ["K"],
            "links": [{"mover": second, "onto": "P", "onto_to": "R"}],
        }
        assert validate_spec(spec) == [], validate_spec(spec)
        strike = compile_spec(spec).strike
        assert strike is not None and strike.compact_to is None


# ----------------------------------------------------------------------- a hexagon's seams


def _hexagon(**rules: Any) -> dict[str, Any]:
    """A hexagon of radius two (19 cells) and two kings, schema 18 for its hex symmetry."""
    spec = {
        "schema_version": 18,
        "name": "tiny hexagon",
        "board": {"width": 5, "height": 5, "lattice": "hex", "shape": "hexagon"},
        "pieces": [{"id": "K", "name": "king", "value": 0, "royal": True,
                    "moves": [{"kind": "leap", "vectors": [[0, 1], [1, 0]],
                               "symmetry": "rotate6"}]}],
        "setup": {"mirror": False, "orientation": "rotate", "placements": [
            {"piece": "K", "player": 0, "squares": ["c1"]},
            {"piece": "K", "player": 1, "squares": ["c5"]},
        ]},
        "rules": {"win": "royal_capture", **rules},
    }
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


class TestAHexagonsSeamsAreClosedUnlessTheSpecJoinsThem:
    ALL_JOINED = {"n_s": "both", "ne_sw": "both", "nw_se": "both"}

    def test_a_royal_line_on_an_unwrapped_hexagon_stops_at_the_edge(self) -> None:
        compiled = compile_spec(_hexagon(royal_facing="forbidden"))
        real = [s for s in range(compiled.n_squares) if s not in compiled.obstacles]
        column = {s: [t for t in real if t % compiled.width == s % compiled.width] for s in real}
        for square in real:
            targets = [target for target, _ in compiled.facing_lines[square]]
            assert sorted(targets) == sorted(t for t in column[square] if t != square), square

    def test_a_royal_line_through_every_joined_seam_holds_the_whole_circuit(self) -> None:
        compiled = compile_spec(_hexagon(royal_facing="forbidden", wrap=self.ALL_JOINED))
        real = [s for s in range(compiled.n_squares) if s not in compiled.obstacles]
        assert compiled.height < len(real) - 1
        for square in real:
            assert len(compiled.facing_lines[square]) == 2 * (len(real) - 1), square

    def test_a_blast_on_an_unwrapped_hexagon_takes_only_cells_on_the_board(self) -> None:
        compiled = compile_spec(_hexagon(explode_on_capture={"radius": 1}))
        assert compiled.explosion is not None
        real = {s for s in range(compiled.n_squares) if s not in compiled.obstacles}
        width = compiled.width
        for square in real:
            blast = set(compiled.explosion.blast[square])
            assert blast <= real, square
            # No cell is reached round the back: each is a neighbour on the rhombus.
            assert all(
                abs(t % width - square % width) <= 1 and abs(t // width - square // width) <= 1
                for t in blast
            ), square


class TestOneSquaresTables:
    def test_a_move_written_as_a_leap_and_as_a_slide_keeps_only_the_leap(self) -> None:
        from chess_variant_sandbox.engine.compile import _compile_square

        piece = {"id": "W", "moves": [
            {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"},
            {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "range": 1},
        ]}
        leaps, slides, hops, sweeps = _compile_square(piece, 0, 12, 5, 5)
        assert sorted(t for squares, _, _ in leaps for t in squares) == [7, 11, 13, 17]
        assert slides == () and hops == () and sweeps == ()

    def test_a_wall_in_a_corner_of_a_royal_line_leaves_every_other_squares_line(self) -> None:
        spec = _probe([], rules={"royal_facing": "forbidden",
                                 "obstacles": {"squares": ["a2"]}})
        compiled = compile_spec(spec)
        assert len(compiled.facing_lines) == compiled.n_squares
        wall = 1 * 5 + 0
        assert compiled.facing_lines[wall] == ()
        for square in range(wall + 1, compiled.n_squares):
            assert compiled.facing_lines[square] != (), square


# ----------------------------------------------------------------------- promotion on leaving


def _leaving_game(
    piece: dict[str, Any],
    rules: dict[str, Any],
    placements: list[dict[str, Any]],
    width: int = 3,
    height: int = 3,
) -> dict[str, Any]:
    """Kings, ``piece`` and a queen-like target; ``piece`` promotes on the last rank and, since
    it says ``on_leaving``, on a move that starts there."""
    target = {"id": "Q", "name": "target", "value": 1,
              "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]}
    spec = _probe(
        [piece, target],
        rules={"promotion": {"piece": piece["id"], "rank": "last", "to": ["Q"],
                             "on_leaving": True}, **rules},
        placements=placements,
        width=width, height=height,
    )
    return spec


def _leaves(spec: dict[str, Any]) -> tuple[frozenset[str], bool]:
    from chess_variant_sandbox.engine.compile import pieces_that_can_leave_their_zone

    kept = compile_spec(spec).promotions[0].on_leaving
    return pieces_that_can_leave_their_zone(spec), kept


class TestPromotionOnLeaving:
    PAWN = {"id": "P", "name": "pawn", "value": 1,
            "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]}

    @pytest.mark.parametrize("ranks", [{"0": "top", "1": "none"}, {"0": "none", "1": "bottom"}])
    def test_a_seam_only_one_player_may_cross_lets_the_pawn_leave_its_zone(
        self, ranks: dict[str, str]
    ) -> None:
        spec = _leaving_game(
            self.PAWN,
            {"wrap": {"ranks_by_player": ranks}},
            [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["c3"]},
                {"piece": "P", "player": 0, "squares": ["b1"]},
                {"piece": "P", "player": 1, "squares": ["b3"]},
            ],
        )
        assert _leaves(spec) == (frozenset({"P"}), True)

    def test_a_pawn_with_no_seam_cannot_leave_and_the_flag_is_dropped(self) -> None:
        spec = _leaving_game(self.PAWN, {}, [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["c3"]},
            {"piece": "P", "player": 0, "squares": ["b1"]},
            {"piece": "P", "player": 1, "squares": ["b3"]},
        ])
        assert _leaves(spec) == (frozenset(), False)

    def test_a_leap_over_a_wall_is_not_stopped_unless_the_spec_says_leaps_are(self) -> None:
        leaper = {"id": "M", "name": "backstepper", "value": 1,
                  "moves": [{"kind": "leap", "vectors": [[0, -2]], "mode": "move"}]}
        walls = [f"{file}{rank}" for file in "abcde" for rank in (2, 4)]
        spec = _leaving_game(
            leaper,
            {"obstacles": {"squares": walls}},
            [
                {"piece": "K", "player": 0, "squares": ["a3"]},
                {"piece": "K", "player": 1, "squares": ["e3"]},
                {"piece": "M", "player": 0, "squares": ["c5"]},
                {"piece": "M", "player": 1, "squares": ["c1"]},
            ],
            width=5, height=5,
        )
        assert _leaves(spec) == (frozenset({"M"}), True)
        stopped = copy.deepcopy(spec)
        stopped["rules"]["obstacles"]["blocks_leaps"] = True
        assert _leaves(stopped) == (frozenset(), False)

    def test_a_facing_piece_leaves_its_zone_in_the_facing_that_walks_back(self) -> None:
        marcher = {"id": "F", "name": "marcher", "value": 1, "facing": {"axes": 4},
                   "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]}
        spec = _leaving_game(marcher, {}, [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["c3"]},
            {"piece": "F", "player": 0, "squares": ["b1"]},
            {"piece": "F", "player": 1, "squares": ["b3"]},
        ])
        assert _leaves(spec) == (frozenset({"F"}), True)


# ----------------------------------------------------------------------- what a capture leaves behind


class TestACaptureThatLeavesPiecesBehind:
    PAWN = {"id": "P", "name": "pawn", "value": 1,
            "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]}
    ROOK = {"id": "R", "name": "rook", "value": 1,
            "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]}

    @classmethod
    def _circe(cls, homes: list[str], hex_lattice: bool = False) -> Any:
        """Compiled circe on a 5x5 board whose first player has a rook on each of ``homes``."""
        rules = {"strike": {"effect": "capture", "position": "home", "immune": ["K"]}}
        placements = [
            {"piece": "K", "player": 0, "squares": ["b2"]},
            {"piece": "K", "player": 1, "squares": ["d4"]},
            {"piece": "R", "player": 0, "squares": homes},
        ]
        if hex_lattice:
            piece = {**cls.ROOK, "moves": [{"kind": "slide", "vectors": [[1, 0], [0, 1]]}]}
            king = {**KING, "moves": [{"kind": "leap", "vectors": [[1, 0], [0, 1]]}]}
            spec = {
                "schema_version": 18, "name": "hex circe",
                "board": {"width": 5, "height": 5, "lattice": "hex"},
                "pieces": [king, piece],
                "setup": {"mirror": False, "orientation": "rotate", "placements": placements},
                "rules": {"win": "royal_capture", **rules},
            }
            assert validate_spec(spec) == [], validate_spec(spec)
            return compile_spec(spec)
        return compile_spec(_probe([cls.ROOK], rules=rules, placements=placements))

    @staticmethod
    def _home(compiled: Any, taken: str) -> str:
        n = compiled.n_squares
        rook = compiled.piece_ids.index("R")
        square = (int(taken[1:]) - 1) * compiled.width + ord(taken[0]) - ord("a")
        home = compiled.strike.rebirth[(0 * compiled.n_types + rook) * n + square]
        return f"{chr(ord('a') + home % compiled.width)}{home // compiled.width + 1}"

    def test_a_type_one_side_never_set_up_has_no_home_to_go_back_to(self) -> None:
        compiled = self._circe(["a2"])
        n, rook = compiled.n_squares, compiled.piece_ids.index("R")
        second_players_rooks = compiled.strike.rebirth[
            (compiled.n_types + rook) * n:(compiled.n_types + rook + 1) * n
        ]
        assert second_players_rooks == (-1,) * n
        assert all(home >= 0 for home in compiled.strike.rebirth[rook * n:(rook + 1) * n])

    @pytest.mark.parametrize(("taken", "home"), [("b3", "a1"), ("b2", "a1"), ("b4", "a5"),
                                                  ("c3", "a1"), ("c4", "a5")])
    def test_two_homes_on_one_file_the_nearer_rank_wins_and_a_tie_the_lower_square(
        self, taken: str, home: str
    ) -> None:
        assert self._home(self._circe(["a1", "a5"]), taken) == home

    @pytest.mark.parametrize(("taken", "home"), [("b3", "a1"), ("d3", "e1"), ("b4", "a1"),
                                                  ("d2", "e1")])
    def test_two_homes_on_one_rank_the_nearer_file_wins(self, taken: str, home: str) -> None:
        assert self._home(self._circe(["a1", "e1"]), taken) == home

    def test_on_a_hex_board_the_colour_is_not_asked_and_a_rank_apart_is_a_whole_rank(self) -> None:
        # d3 is a file from c1 and from e4, and two ranks from c1 but one from e4.
        assert self._home(self._circe(["c1", "e4"], hex_lattice=True), "d3") == "e4"

    def test_a_strike_that_spares_a_type_early_still_removes_the_types_declared_after(self) -> None:
        spec = _probe(
            [self.PAWN, self.ROOK],
            rules={"strike": {"effect": "capture", "position": "stay", "immune": ["K", "P"]}},
            placements=[
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "R", "player": 0, "squares": ["a2"]},
            ],
        )
        compiled = compile_spec(spec)
        assert compiled.piece_ids == ("K", "P", "R")
        assert compiled.strike is not None and compiled.strike.compact_to == (-1, -1, -2)


# ----------------------------------------------------------------------- races to a goal


def _race(
    racer: dict[str, Any],
    squares: list[str],
    piece: str | None = "R",
    width: int = 5,
    height: int = 5,
    kings: tuple[str, str] = ("a1", "e5"),
    still_kings: bool = False,
    **reach: Any,
) -> Any:
    """Compiled spec of a bare board, two kings and a ``racer`` each, racing to ``squares``.

    ``still_kings`` gives the kings capture-only steps, so no king ever moves into a goal."""
    rule: dict[str, Any] = {"squares": squares, "mirror": True, "result": "win", **reach}
    if piece:
        rule["piece"] = piece
    spec = _probe(
        [racer],
        rules={"reach": rule},
        placements=[
            {"piece": "K", "player": 0, "squares": [kings[0]]},
            {"piece": "K", "player": 1, "squares": [kings[1]]},
            {"piece": "R", "player": 0, "squares": ["c1"]},
            {"piece": "R", "player": 1, "squares": [f"c{height}"]},
        ],
        width=width, height=height,
    )
    if still_kings:
        spec["pieces"][0] = {**KING, "moves": [{"kind": "leap", "vectors": [[1, 0]],
                                                 "symmetry": "rotate4", "mode": "capture"}]}
    return compile_spec(spec)


STEPPER = {"id": "R", "name": "racer", "value": 1,
           "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4",
                      "mode": "move"}]}
FORWARD = {"id": "R", "name": "racer", "value": 1,
           "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]}


class TestTheGoalRaceTables:
    def test_a_named_piece_racing_several_goals_gets_a_table_per_goal(self) -> None:
        compiled = _race(STEPPER, ["a5", "b5", "c5"], piece="R")
        each = compiled.goal_distance_each
        assert [len(per_goal) for per_goal in each] == [3, 3]

    def test_a_royal_racer_gets_no_per_goal_table(self) -> None:
        compiled = _race(STEPPER, ["a5", "b5"], piece="K")
        assert compiled.goal_distance_each == ()

    def test_exactly_two_goals_are_a_race_with_a_table_each_and_one_goal_is_not(self) -> None:
        assert [len(p) for p in _race(STEPPER, ["a5", "b5"], piece=None).goal_distance_each] == [2, 2]
        assert _race(STEPPER, ["a5"], piece=None).goal_distance_each == ()

    def test_a_forward_only_racer_is_zero_moves_from_each_goal_it_arrives_on(self) -> None:
        compiled = _race(FORWARD, ["a5", "b5"], piece=None)
        racer = compiled.piece_ids.index("R")
        for player, goals in ((0, (20, 21)), (1, (0, 1))):  # the second side's are mirrored
            for goal, table in zip(goals, compiled.goal_distance_each[player], strict=True):
                assert table[racer][goal] == 0
        # ... and on leaving, a goal it can never step off is no goal at all.
        leaving = _race(FORWARD, ["a5", "b5"], piece=None, on="leave")
        assert all(
            distance == UNREACHABLE
            for table in leaving.goal_distance_each[0]
            for distance in table[racer]
        )

    def test_the_diameter_is_never_below_one_even_when_every_distance_is_one_or_nothing(
        self,
    ) -> None:
        king = {"id": "R", "name": "racer", "value": 1,
                "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]],
                           "symmetry": "dihedral8", "mode": "move"}]}
        compiled = _race(king, ["b2"], piece=None, width=3, height=3, kings=("a1", "a3"),
                         still_kings=True)
        racer = compiled.piece_ids.index("R")
        finite = {d for d in compiled.goal_distance[0][racer] if d != UNREACHABLE}
        assert finite == {0, 1}
        assert compiled.goal_diameter == 1

    def test_a_race_nobody_can_run_still_has_a_diameter_to_divide_by(self) -> None:
        hopper = {"id": "R", "name": "racer", "value": 1,
                  "moves": [{"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4"}]}
        compiled = _race(hopper, ["c3"], piece=None, still_kings=True)
        racer = compiled.piece_ids.index("R")
        finite = {d for d in compiled.goal_distance[0][racer] if d != UNREACHABLE}
        assert finite == {0}
        assert compiled.goal_diameter == 1

    def test_a_racer_that_turns_is_measured_as_an_orthogonal_stepper(self) -> None:
        turner = {**FORWARD, "facing": {"axes": 4}}
        turning, plain = _race(turner, ["c3"], piece=None), _race(STEPPER, ["c3"], piece=None)
        for player in (0, 1):
            assert (turning.goal_distance[player][turning.piece_ids.index("R")]
                    == plain.goal_distance[player][plain.piece_ids.index("R")])

    def test_a_racer_that_only_sweeps_still_has_a_distance(self) -> None:
        sweeper = {"id": "R", "name": "racer", "value": 1,
                   "moves": [{"kind": "sweep", "vectors": [[0, 1]], "mode": "move"}]}
        compiled = _race(sweeper, ["c5"], piece=None)
        racer = compiled.piece_ids.index("R")
        assert compiled.goal_distance[0][racer][2] == 1  # c1 -> c5 in one sweep
        assert compiled.goal_distance[0][racer][2 + 5 * 3] == 1
