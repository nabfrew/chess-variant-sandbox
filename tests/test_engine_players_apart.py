"""Engine mutation report, F-engine-2: the two players kept apart.

The mutants of `docs/reference/mutation/engine.md` that survived because every spec the suite
compiles gives both players the same table: a mirrored army, a symmetric wrap, equal counts.
A mutant that reads player 1's column for player 0, or merges the two, then answers the same.
Each test below builds one asymmetric spec per function, so the two columns differ and the
swap shows. Every spec is a small purpose-made dictionary; none asserts a number that only
chess has.
"""

from __future__ import annotations

from typing import Any

import pytest
from portal_oracle import PortalOracle, Position

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import (
    PORTAL_DIRECTIONS,
    WILD,
    TurnRule,
    _compile_turn,
    compile_spec,
)
from chess_variant_sandbox.engine.moves import (
    MOVE_PORTAL,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.schema import royalty_can_multiply

KING = {"id": "K", "name": "king", "value": 0, "royal": True,
        "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]}
PAWN = {"id": "P", "name": "pawn", "value": 1,
        "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"}]}
SLIDER = [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]


def _piece(piece_id: str, value: int = 3) -> dict[str, Any]:
    return {"id": piece_id, "name": piece_id.lower(), "value": value, "moves": SLIDER}


def _spec(
    pieces: list[dict[str, Any]],
    placements: list[dict[str, Any]],
    rules: dict[str, Any] | None = None,
    width: int = 5,
    height: int = 5,
) -> dict[str, Any]:
    spec = {
        "schema_version": 26,
        "name": "apart",
        "board": {"width": width, "height": height},
        "pieces": pieces,
        "setup": {"mirror": False, "placements": placements},
        "rules": {"win": "royal_capture", **(rules or {})},
    }
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _from(engine: Engine, state: Any, square: str) -> list[str]:
    return sorted(t for t in engine.move_texts(state) if t[1:].startswith(square))


# ------------------------------------------------------------------ promotion targets


class TestPromotionTargetsAreEachPlayersOwn:
    """`bridge.build_tables` fills `promo_targets_flat` per player, in the player's own turn."""

    def test_each_player_is_offered_the_targets_written_for_that_player(self) -> None:
        spec = _spec(
            [KING, PAWN, _piece("Q"), _piece("R"), _piece("N")],
            [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "P", "player": 0, "squares": ["c4"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "P", "player": 1, "squares": ["c2"]},
            ],
            {"promotion": {"piece": "P", "rank": "last",
                           "to_by_player": {"0": ["Q", "R"], "1": ["R", "N"]}}},
        )
        engine = Engine(spec)
        white = engine.make_state(
            {"a1": ("K", 0), "c4": ("P", 0), "e5": ("K", 1), "c2": ("P", 1)}, 0)
        black = engine.make_state(
            {"a1": ("K", 0), "c4": ("P", 0), "e5": ("K", 1), "c2": ("P", 1)}, 1)
        assert _from(engine, white, "c4") == ["Pc4-c5=Q", "Pc4-c5=R"]
        assert _from(engine, black, "c2") == ["Pc2-c1=N", "Pc2-c1=R"]


# ------------------------------------------------------------------ only_lost counts


def _only_lost(white: list[str], black: list[str]) -> dict[str, Any]:
    """Pawns c4 and c2 one step from their zones, offering Q only to a side short of its own."""
    return _spec(
        [_piece("Q", 9), KING, PAWN],
        [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["e5"]},
            {"piece": "P", "player": 0, "squares": ["c4"]},
            {"piece": "P", "player": 1, "squares": ["c2"]},
            {"piece": "Q", "player": 0, "squares": white},
            {"piece": "Q", "player": 1, "squares": black},
        ],
        {"promotion": {"piece": "P", "rank": "last", "to": ["Q"], "only_lost": True}},
    )


def _queens(white: list[str], black: list[str]) -> dict[str, tuple[str, int]]:
    held = {"a1": ("K", 0), "e5": ("K", 1), "c4": ("P", 0), "c2": ("P", 1)}
    held.update({square: ("Q", 0) for square in white})
    held.update({square: ("Q", 1) for square in black})
    return held


class TestOnlyLostCountsEachSidesOwnArmy:
    def test_the_compiler_counts_each_player_apart(self) -> None:
        compiled = compile_spec(_only_lost(["b1", "d1"], ["d5"]))
        (rule,) = compiled.promotions
        counted = {
            player: dict(zip(compiled.piece_ids, rule.opening[player], strict=True))
            for player in (0, 1)
        }
        assert counted[0] == {"K": 1, "P": 1, "Q": 2}
        assert counted[1] == {"K": 1, "P": 1, "Q": 1}

    def test_a_side_is_short_by_its_own_count_and_not_the_other_sides(self) -> None:
        # White began with two queens, Black with one. Each holds one now: short for White,
        # whole for Black. A mutant that reads the other player's count (or one count for
        # both) flips one of the two answers.
        engine = Engine(_only_lost(["b1", "d1"], ["d5"]))
        held = _queens(["b1"], ["d5"])
        assert _from(engine, engine.make_state(held, 0), "c4") == ["Pc4-c5=Q"]
        assert _from(engine, engine.make_state(held, 1), "c2") == []

    def test_the_other_way_round_gives_the_other_answer(self) -> None:
        # White began with one queen, Black with two; Black has lost one, White none.
        engine = Engine(_only_lost(["b1"], ["d5", "e4"]))
        held = _queens(["b1"], ["d5"])
        assert _from(engine, engine.make_state(held, 0), "c4") == []
        assert _from(engine, engine.make_state(held, 1), "c2") == ["Pc2-c1=Q"]

    def test_a_third_owner_is_not_counted_as_either_player(self) -> None:
        # A wild piece is nobody's: one beside the pieces must compile, and leave both
        # counts as the players' own. (Owner 2 indexes past the two count rows.)
        spec = _only_lost(["b1"], ["d5"])
        spec["pieces"].append(
            {"id": "W", "name": "wild", "value": 1, "moves": [
                {"kind": "leap", "vectors": [[1, 0]], "mode": "both"}]})
        spec["rules"]["npcs"] = [{"mirror": False, "piece": "W", "route": ["e3", "e2", "d2"]}]
        assert validate_spec(spec) == [], validate_spec(spec)
        compiled = compile_spec(spec)
        (rule,) = compiled.promotions
        wild = compiled.piece_ids.index("W")
        assert (rule.opening[0][wild], rule.opening[1][wild]) == (0, 0)
        queen = compiled.piece_ids.index("Q")
        assert (rule.opening[0][queen], rule.opening[1][queen]) == (1, 1)


# ------------------------------------------------------------------ two royals of one side


def _kings(white: list[str], black: list[str], king_first: bool) -> dict[str, Any]:
    pieces = [KING, _piece("Q", 9)] if king_first else [_piece("Q", 9), KING]
    return _spec(
        pieces,
        [
            {"piece": "K", "player": 0, "squares": white},
            {"piece": "K", "player": 1, "squares": black},
        ],
        {"win": "royal_capture"},
    )


class TestTwoRoyalsOfOneTypeOnOneSide:
    """`_royalty_can_multiply` reads the board once per player; the sides differ here."""

    @pytest.mark.parametrize("king_first", [True, False], ids=["king-first", "king-second"])
    @pytest.mark.parametrize(
        ("white", "black", "expected"),
        [
            (["a1", "b1"], ["e5"], True),
            (["a1"], ["e5", "d5"], True),
            (["a1", "b1"], ["e5", "d5"], True),
            (["a1"], ["e5"], False),
        ],
        ids=["white-has-two", "black-has-two", "both-have-two", "one-each"],
    )
    def test_the_answer_follows_the_side_that_holds_two(
        self, white: list[str], black: list[str], expected: bool, king_first: bool
    ) -> None:
        spec = _kings(white, black, king_first)
        assert compile_spec(spec).multi_royal is expected
        assert royalty_can_multiply(spec) is expected

    def test_two_royal_types_are_two_royals_whatever_the_board_holds(self) -> None:
        second = {**KING, "id": "J", "name": "jarl"}
        spec = _spec(
            [KING, second, _piece("Q", 9)],
            [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "J", "player": 1, "squares": ["e5"]},
            ],
            {"win": "royal_capture"},
        )
        assert compile_spec(spec).multi_royal is True
        assert royalty_can_multiply(spec) is True


# ------------------------------------------------------------------ the default rook file


class TestACastlingThatNamesOnlyTheKingsFiles:
    """`_compile_castling`: the rook lands on the square the king crossed, on either wing."""

    def test_the_rook_lands_beside_the_king_on_the_wing_it_castled_to(self) -> None:
        spec = _spec(
            [KING, _piece("R", 5)],
            [
                {"piece": "K", "player": 0, "squares": ["e1"]},
                {"piece": "R", "player": 0, "squares": ["a1", "h1"]},
                {"piece": "K", "player": 1, "squares": ["e8"]},
            ],
            {"castling": {"king": "K", "rook": "R",
                          "king_file": {"queenside": "b", "kingside": "g"}}},
            width=8, height=8,
        )
        compiled = compile_spec(spec)
        assert compiled.castling is not None
        assert compiled.castling.king_files == (1, 6)
        assert compiled.castling.rook_files == (2, 5)
        engine = Engine(spec)
        state = engine.make_state({"e1": ("K", 0), "a1": ("R", 0), "h1": ("R", 0),
                                   "e8": ("K", 1)}, 0)
        castles = [engine.apply(state, m) for m in engine.legal_moves(state)
                   if engine.describe_move(state, m).endswith("(castle)")]
        assert len(castles) == 2
        squares = sorted(
            sorted(f"{chr(97 + s % 8)}{s // 8 + 1}" for s, v in enumerate(after.board)
                   if v and engine.spec.owner_of(v) == 0)
            for after in castles
        )
        assert squares == [["a1", "f1", "g1"], ["b1", "c1", "h1"]]


# ------------------------------------------------------------------ a turn of one move


class TestATurnOfOneMoveBesideARamp:
    """`_compile_turn` drops a turn only when it is one move long *and* no ramp lengthens it."""

    @pytest.mark.parametrize(
        ("turn", "expected"),
        [
            ({"moves": 1}, None),
            ({"moves": 1, "ramp": [1]}, None),
            ({"moves": 1, "ramp": [1, 1]}, None),
            ({"moves": 1, "ramp": [2]}, TurnRule(moves=1, lengths=(2,))),
            ({"moves": 1, "ramp": [1, 2]}, TurnRule(moves=1, lengths=(1, 2))),
            ({"moves": 2, "ramp": [1]}, TurnRule(moves=2, lengths=(1,))),
            ({"moves": 2}, TurnRule(moves=2, lengths=())),
            ({"moves": 0, "ramp": [1]}, TurnRule(moves=0, lengths=(1,))),
        ],
    )
    def test_the_turn_is_kept_exactly_when_it_is_not_the_ordinary_one(
        self, turn: dict[str, Any], expected: TurnRule | None
    ) -> None:
        assert _compile_turn(turn) == expected


# ------------------------------------------------------------------ a hop with a bounded screen


def _grasshopper(screen_range: int) -> Engine:
    hopper = {"id": "G", "name": "hopper", "value": 3, "moves": [
        {"kind": "hop", "vectors": [[0, 1]], "symmetry": "rotate4",
         "screen_range": screen_range, "mode": "both"}]}
    return Engine(_spec(
        [KING, hopper, _piece("Q", 9)],
        [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["a5"]},
            {"piece": "G", "player": 1, "squares": ["e5"]},
        ],
    ))


class TestAHopWhoseScreenMustStandNextToTheLanding:
    """`_build_attack_sources`: the squares before the screen's reach must be empty."""

    @pytest.mark.parametrize(
        ("screen_range", "screen", "checked"),
        [
            (1, "e2", True),   # the screen stands right before the landing
            (1, None, False),  # no screen, no hop
            (1, "e3", False),  # a screen further back than one square is out of reach
            (2, "e3", True),   # two squares of reach: e3 and e2 may hold it
            (2, "e2", True),
            (2, "e4", False),  # e4 is outside the last two squares
            (3, "e4", True),
        ],
    )
    def test_the_king_is_attacked_exactly_when_the_screen_is_in_reach(
        self, screen_range: int, screen: str | None, checked: bool
    ) -> None:
        engine = _grasshopper(screen_range)
        board = {"e1": ("K", 0), "a5": ("K", 1), "e5": ("G", 1)}
        if screen is not None:
            board[screen] = ("Q", 0)
        state = engine.make_state(board, 0)
        assert engine.in_check(state, 0) is checked


# ------------------------------------------------------------------ portals


def _portal_spec(
    groups: list[dict[str, Any]],
    wrap: dict[str, Any] | None = None,
    pieces: list[dict[str, Any]] | None = None,
    width: int = 5,
    height: int = 5,
) -> dict[str, Any]:
    """Two kings and a rook-like ``R`` beside portals nobody moves."""
    declared = pieces if pieces is not None else [KING, _piece("R", 5)]
    spec = _spec(
        declared,
        [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["e5"]},
        ],
        {"portals": groups, **({"wrap": wrap} if wrap else {})},
        width,
        height,
    )
    return spec


def _position(engine: Engine, state: Any) -> Position:
    spec = engine.spec
    return {
        square: (spec.piece_ids[spec.type_of(value)],
                 "portal" if spec.owner_of(value) == WILD else spec.owner_of(value))
        for square, value in enumerate(state.board)
        if value
    }


def _core_transits(engine: Engine, state: Any) -> set[tuple[int, int, int, str | None]]:
    spec = engine.spec
    found = set()
    for move in engine.pseudo_legal_moves(state):
        if move_type(move) != MOVE_PORTAL:
            continue
        promoted = move_promotion(move)
        entry = engine.portal_entry(state, move)
        assert entry is not None
        found.add((move_from(move), entry, move_to(move),
                   spec.piece_ids[promoted - 1] if promoted else None))
    return found


class TestAPortalEndIsNamedForItsPlace:
    """`_portal_end_pieces`: ``portal <group number> <end's key or letter>``, after the
    declared pieces and the portals' movement type (`schema.portal_movement_types`)."""

    def test_a_pair_names_its_ends_by_colour_and_a_map_by_letter(self) -> None:
        spec = _portal_spec([
            {"orange": "b3", "blue": "d3"},
            {"ends": [{"at": "a4", "to": 1}, {"at": "c4", "to": 2},
                                    {"at": "e4", "to": 0}]},
        ])
        assert compile_spec(spec).piece_names[3:] == (
            "portal 1 orange", "portal 1 blue", "portal 2 a", "portal 2 b", "portal 2 c",
        )


class TestEachPlayersExitFromAPortalIsThatPlayersOwnWrap:
    """`_portal_exits`: a transit leaves under the mover's wrap permission."""

    WRAP = {"files_by_player": {"0": "both", "1": "none"}}

    def _spec(self) -> dict[str, Any]:
        return _portal_spec([{"orange": "a3", "blue": "c3"}], self.WRAP)

    def test_the_table_wraps_for_the_first_player_and_stops_the_second_at_the_edge(self) -> None:
        portals = compile_spec(self._spec()).portals
        assert portals is not None
        east, west = PORTAL_DIRECTIONS.index((1, 0)), PORTAL_DIRECTIONS.index((-1, 0))
        e3, a3 = 2 * 5 + 4, 2 * 5
        assert portals.exits[0][e3][east] == a3
        assert portals.exits[0][a3][west] == e3
        assert portals.exits[1][e3][east] == -1
        assert portals.exits[1][a3][west] == -1
        assert portals.exits[1][a3][east] == a3 + 1  # inside the board both agree
        assert portals.exits[0][a3][east] == a3 + 1

    @pytest.mark.parametrize("player", [0, 1])
    def test_the_core_and_the_oracle_agree_for_each_side(self, player: int) -> None:
        spec = self._spec()
        engine = Engine(spec)
        oracle = PortalOracle(spec)
        placement = {"a1": ("K", 0), "e5": ("K", 1), "d3": ("R", player),
                     "a3": ("1o", WILD), "c3": ("1b", WILD)}
        state = engine.make_state(placement, side_to_move=player)
        unmoved = {sq for sq in range(25) if state.unmoved >> sq & 1}
        expected = oracle.transits(_position(engine, state), player, unmoved)
        assert _core_transits(engine, state) == expected
        # The rook goes west into c3 and comes out of a3 one step on: onto e3 across the
        # seam for the first player, nowhere for the second.
        assert bool(expected) is (player == 0)


class TestALameLeapIsNotAMoveThroughItsLegEnd:
    """`_compile_square_full`: the leg of a lame leap carries no momentum, so it enters nothing."""

    def test_a_portal_end_standing_on_the_leg_is_not_entered(self) -> None:
        lame = {"id": "N", "name": "lame", "value": 3, "moves": [
            {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8"}]}
        spec = _portal_spec(
            [{"orange": "b2", "blue": "c3"}], pieces=[KING, lame])
        engine = Engine(spec)
        oracle = PortalOracle(spec)
        placement = {"a1": ("K", 0), "e4": ("K", 1), "b1": ("N", 0),
                     "b2": ("1o", WILD), "c3": ("1b", WILD)}
        state = engine.make_state(placement, side_to_move=0)
        unmoved = {sq for sq in range(25) if state.unmoved >> sq & 1}
        assert _core_transits(engine, state) == oracle.transits(
            _position(engine, state), 0, unmoved) == set()
        # The leg is held, so both leaps over it are blocked as well.
        texts = engine.move_texts(state)
        assert "Nb1-a3" not in texts and "Nb1-c3" not in texts and "Nb1-d2" in texts
