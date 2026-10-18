"""Schema 26 (Q918a): simultaneous moves, the smallest playable slice.

`docs/reference/rounds.md`: the keys and every refusal of §4 and §5.2; the compiler's
tables; one hand-built round per row of §2.2 and per step of §4; a swap's three outcomes
(§2.4); the void bar (§2.7); the endings (§2.6); joint perft frozen; determinism of whole
games. The core's rounds are compared with the declared oracle on every hand-built round
here as well as on the random walks of `tests/test_simultaneous_oracle.py`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from simultaneous_oracle import resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import VOID_RUN, compile_spec
from chess_variant_sandbox.engine.moves import MOVE_DROP, MOVE_PORTAL, MOVE_PROMOTION, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, validate_spec
from chess_variant_sandbox.tournament.play import play_game

HERE = Path(__file__).parent
ROOT = HERE.parent


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"_simultaneous_{name}.json").read_text(encoding="utf-8"))


GARDNER = load("gardner")


def with_rules(spec: dict[str, Any], **rules: Any) -> dict[str, Any]:
    changed = copy.deepcopy(spec)
    changed["rules"].update(rules)
    return changed


def with_piece(spec: dict[str, Any], piece_id: str, **keys: Any) -> dict[str, Any]:
    changed = copy.deepcopy(spec)
    for piece in changed["pieces"]:
        if piece["id"] == piece_id:
            piece.update(keys)
    return changed


# A small game for the void bar: pawns whose diagonal is capture-only, and a wazir.
VOID = {
    "schema_version": 26,
    "name": "void standoff",
    "board": {"width": 5, "height": 5},
    "pieces": [
        {"id": "P", "moves": [
            {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
            {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
        ]},
        {"id": "W", "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
    ],
    "setup": {"placements": [{"piece": "P", "squares": ["b2"]}, {"piece": "W", "squares": ["e1"]}]},
    "rules": {"win": "annihilation", "simultaneous": {}, "max_plies": 400},
}


def square(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def find(
    engine: Engine, state: GameState, player: int, origin: str | None, target: str,
    kind: int | None = None, promotion: int | None = None,
) -> int:
    """The legal move of `player` from `origin` (None: a drop) to `target`."""
    want_to = square(engine, target)
    for move in engine.legal_moves_for(state, player):
        o, t, k, p = decode_move(move)
        if t != want_to or (kind is not None and k != kind):
            continue
        if promotion is not None and p != promotion:
            continue
        if origin is None and k == MOVE_DROP:
            return move
        if origin is not None and k != MOVE_DROP and o == square(engine, origin):
            return move
    raise AssertionError(f"player {player} has no move {origin}->{target}")


def place(engine: Engine, pieces: dict[str, tuple[str, int]], **changes: Any) -> GameState:
    state = engine.make_state(pieces)
    # A hand-built position holds no hands; a game with drops starts with empty ones.
    opening = engine.initial_state()
    empty = tuple(tuple(0 for _ in hand) for hand in opening.hands)
    changes = {"hands": empty, **changes}
    portals = engine.spec.portals
    if portals is not None and "board" not in changes:
        # The rule places the portal ends; a hand-built board keeps them where they start.
        board = bytearray(state.board)
        for at, value in enumerate(opening.board):
            if value and engine.spec.type_of(value) in portals.end_types and not board[at]:
                board[at] = value
        changes["board"] = bytes(board)
    if changes:
        state = state.with_(**changes)
        key = engine._compute_hash(state)
        state = state.with_(key=key, repetition=(key,))
    return state


def round_of(engine: Engine, state: GameState, first: int, second: int):
    """Play one round through the core, check it against the oracle, return it."""
    after, phase, events = engine.apply_round_with_phase(state, first, second)
    assert after == resolve_round(engine, state, first, second)
    assert after.key == engine._compute_hash(after)
    return after, phase, events


def at(engine: Engine, state: GameState, name: str) -> tuple[str, int] | None:
    value = state.board[square(engine, name)]
    if value == 0:
        return None
    spec = engine.spec
    return spec.piece_ids[spec.type_of(value)], spec.owner_of(value)


# --------------------------------------------------------------------- the keys


class TestTheKeys:
    def test_the_version_is_twenty_six(self) -> None:
        assert SCHEMA_VERSION >= 26
        assert VOID_RUN == 2

    def test_the_test_specs_validate(self) -> None:
        for path in sorted(HERE.glob("_simultaneous_*.json")):
            assert validate_spec(json.loads(path.read_text(encoding="utf-8"))) == [], path
        assert validate_spec(VOID) == []

    def test_an_older_version_may_not_use_them(self) -> None:
        spec = with_piece(GARDNER, "P", beats=["P"])
        spec["schema_version"] = 25
        errors = validate_spec(spec)
        assert any(
            "pieces[].beats, rules.simultaneous" in error and "schema 26" in error
            for error in errors
        ), errors

    @pytest.mark.parametrize(
        ("tie", "words"),
        [("both", "leave it out"), ("first", "must be one of")],
    )
    def test_a_tie_spelled_otherwise_is_refused(self, tie: str, words: str) -> None:
        errors = validate_spec(with_rules(GARDNER, simultaneous={"tie": tie}))
        assert any(words in error for error in errors), errors

    def test_an_unknown_key_is_refused(self) -> None:
        assert validate_spec(with_rules(GARDNER, simultaneous={"ties": "initiative"}))

    def test_beats_without_the_rule_is_refused(self) -> None:
        spec = with_piece(GARDNER, "P", beats=["P"])
        del spec["rules"]["simultaneous"]
        spec["rules"]["win"] = "checkmate"
        errors = validate_spec(spec)
        assert any("only matters in a simultaneous game" in error for error in errors), errors

    def test_beats_equal_to_its_default_is_refused(self) -> None:
        everything = [piece["id"] for piece in GARDNER["pieces"]]
        errors = validate_spec(with_piece(GARDNER, "P", beats=everything))
        assert any("leave it out" in error for error in errors), errors
        narrow = with_piece(GARDNER, "P", captures=["P"], beats=["P"])
        assert any("leave it out" in error for error in validate_spec(narrow))
        assert validate_spec(with_piece(GARDNER, "P", captures=["P"], beats=[])) == []

    def test_beats_names_declared_pieces(self) -> None:
        errors = validate_spec(with_piece(GARDNER, "P", beats=["Z"]))
        assert any("beats entry 'Z' is not a declared piece" in error for error in errors), errors

    def test_beats_on_a_portal_is_refused(self) -> None:
        errors = validate_spec(with_piece(load("portals"), "O", beats=[]))
        assert any("a portal never collides" in error for error in errors), errors

    def test_impossible_is_refused_where_a_type_can_tie(self) -> None:
        errors = validate_spec(with_rules(GARDNER, simultaneous={"tie": "impossible"}))
        assert any('"impossible", but' in error for error in errors), errors

    def test_impossible_is_proved_for_a_circle_of_disjoint_armies(self) -> None:
        spec = circle()
        assert validate_spec(spec) == []
        initiative = copy.deepcopy(spec)
        initiative["rules"]["simultaneous"] = {"tie": "initiative"}
        assert any("can never decide anything" in error for error in validate_spec(initiative))

    def test_initiative_is_accepted_where_a_tie_can_happen(self) -> None:
        assert validate_spec(with_rules(GARDNER, simultaneous={"tie": "initiative"})) == []


def circle() -> dict[str, Any]:
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


REFUSED = [
    ("rules.win 'checkmate' (the default)", lambda s: s["rules"].pop("win")),
    ("rules.checkmate", lambda s: s["rules"].update(checkmate="loss")),
    ("rules.forbid_check", lambda s: s["rules"].update(forbid_check=True)),
    ("rules.check_limit", lambda s: s["rules"].update(check_limit=3)),
    ("rules.royal_facing 'forbidden'", lambda s: s["rules"].update(royal_facing="forbidden")),
    ("rules.castling", lambda s: s["rules"].update(castling={"king": "K", "rook": "R"})),
    ("rules.en_passant", lambda s: s["rules"].update(en_passant=True)),
    ("a turn longer than one move", lambda s: s["rules"].update(turn={"moves": 2})),
    ("a sweep", lambda s: s["pieces"][3]["moves"].append(
        {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"})),
    ("rules.strike.position 'choose'", lambda s: s["rules"].update(strike={
        "effect": "demote", "chain": [{"from": "Q", "to": "R"}], "position": "choose",
        "landing": {"vectors": [[1, 0]], "symmetry": "rotate4"}, "immune": ["K"]})),
    ("rules.strike.position 'stay'", lambda s: s["rules"].update(strike={
        "position": "stay", "immune": ["K"]})),
    ("rules.contacts.position 'choose'", lambda s: s["rules"].update(contacts={
        "links": [{"mover": "N", "onto": "Q", "onto_to": "R"}], "position": "choose",
        "landing": {"vectors": [[1, 0]], "symmetry": "rotate4"}, "immune": ["K"]})),
    ("an own rules.contacts link with position 'swap'", lambda s: s["rules"].update(contacts={
        "links": [{"mover": "N", "onto": "B", "onto_to": "R", "own": True}],
        "position": "swap", "immune": ["K"]})),
]


@pytest.mark.parametrize(("words", "change"), REFUSED, ids=[words for words, _ in REFUSED])
def test_each_combination_section_four_refuses_is_refused(words: str, change: Any) -> None:
    spec = copy.deepcopy(GARDNER)
    change(spec)
    errors = validate_spec(spec)
    assert any(f"rules.simultaneous beside {words}" in error for error in errors), errors


def test_a_portal_a_player_moves_is_refused() -> None:
    spec = load("portals")
    spec["rules"]["portals"][0]["moved_by"] = {"orange": [0]}
    errors = validate_spec(spec)
    assert any("a portal a player moves" in error for error in errors), errors


def test_drop_mate_and_a_royal_in_a_placement_stock_are_refused() -> None:
    spec = load("shogi")
    spec["rules"]["drops"]["no_mate"] = ["C"]
    assert any("rules.drops.no_mate" in error for error in validate_spec(spec))
    stocked = copy.deepcopy(GARDNER)
    stocked["setup"]["placements"] = [p for p in stocked["setup"]["placements"] if p["piece"] != "K"]
    stocked["rules"]["drops"] = {"stock": {"K": 1}, "placement": True, "from_captures": False}
    errors = validate_spec(stocked)
    assert any("a royal in a placement stock" in error for error in errors), errors


# ------------------------------------------------------------------ the compiler


class TestTheCompiler:
    def test_an_alternating_game_is_untouched(self) -> None:
        chess = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
        compiled = compile_spec(chess)
        assert compiled.hash == "4c1922ffae26"
        assert compiled.simultaneous is None
        tables = canonical_tables(compiled)
        assert not {"sim_rule", "sim_collision", "zobrist_sim"} & set(tables)

    def test_the_rule_resolves_into_tables(self) -> None:
        compiled = compile_spec(with_piece(GARDNER, "P", beats=[]))
        sim = compiled.simultaneous
        assert sim is not None
        assert (sim.tie, sim.initiative_start, sim.void_run, sim.collision_strikes) == (0, 1, 2, False)
        n = compiled.n_types
        pawn, knight = compiled.piece_ids.index("P"), compiled.piece_ids.index("N")
        assert not any(sim.collision_allowed[pawn * n : (pawn + 1) * n])
        assert sim.collision_allowed[knight * n + pawn]
        assert sim.zobrist_initiative == 0
        assert len(sim.zobrist_squares) == 2 * compiled.n_squares
        tables = build_tables(compiled)
        assert tables["sim_rule"] == [0, 1, 2, 0]
        assert compile_spec(load("swap")).simultaneous.collision_strikes
        assert compile_spec(load("merge")).simultaneous.zobrist_initiative != 0

    def test_the_draws_come_after_every_other_key(self) -> None:
        """The same game alternating and simultaneous hash apart, but the simultaneous
        spec's every other table is what its own alternating twin would draw from the same
        seed: the round's keys are drawn last (§5.3)."""
        compiled = compile_spec(GARDNER)
        assert compiled.simultaneous is not None
        assert len(compiled.zobrist_pieces) == len(compile_spec(rulesets("gardner")).zobrist_pieces)


def rulesets(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "rulesets" / f"{name}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------- one round per row


@pytest.fixture(scope="module")
def gardner() -> Engine:
    return Engine(GARDNER)


KINGS = {"e1": ("K", 0), "e5": ("K", 1)}


class TestEachRelationOfTwoMoves:
    """§2.2's table, one row each, and §2.4's swap in its three outcomes."""

    def test_the_same_destination_both_beating_is_both_eliminated(self, gardner: Engine) -> None:
        state = place(gardner, {**KINGS, "a1": ("R", 0), "c3": ("R", 1)})
        first = find(gardner, state, 0, "a1", "a3")
        second = find(gardner, state, 1, "c3", "a3")
        after, _, events = round_of(gardner, state, first, second)
        assert at(gardner, after, "a3") is None and at(gardner, after, "a1") is None
        assert at(gardner, after, "c3") is None
        assert events == ((0, square(gardner, "a3"), -1),)
        assert after.halfmove == 0

    def test_the_piece_that_beats_the_other_stands(self) -> None:
        engine = Engine(with_piece(GARDNER, "P", beats=[]))
        state = place(engine, {**KINGS, "b2": ("P", 0), "d4": ("N", 1)})
        first = find(engine, state, 0, "b2", "b3")
        second = find(engine, state, 1, "d4", "b3")
        after, _, events = round_of(engine, state, first, second)
        assert at(engine, after, "b3") == ("N", 1)
        assert events == ((0, square(engine, "b3"), 1),)

    def test_neither_beating_is_both_eliminated_and_nobody_captured(self) -> None:
        engine = Engine(with_piece(load("shogi"), "C", beats=[]))
        state = place(engine, {"b1": ("L", 0), "b4": ("L", 1), "b2": ("C", 0), "a4": ("G", 1)})
        state = place(engine, {"b1": ("L", 0), "b4": ("L", 1), "a2": ("C", 0), "a4": ("C", 1)})
        first = find(engine, state, 0, "a2", "a3")
        second = find(engine, state, 1, "a4", "a3")
        after, _, events = round_of(engine, state, first, second)
        assert at(engine, after, "a3") is None
        assert after.hands == state.hands
        assert events == ((0, square(engine, "a3"), -1),)

    def test_both_beating_under_drops_gives_each_side_the_other_piece(self) -> None:
        engine = Engine(load("shogi"))
        state = place(engine, {"b1": ("L", 0), "b4": ("L", 1), "a2": ("C", 0), "a4": ("C", 1)})
        first = find(engine, state, 0, "a2", "a3")
        second = find(engine, state, 1, "a4", "a3")
        after, _, _ = round_of(engine, state, first, second)
        chick = engine.spec.piece_ids.index("C")
        assert after.hands[0][chick] == 1 and after.hands[1][chick] == 1

    def test_the_initiative_decides_a_tie_and_passes(self) -> None:
        engine = Engine(load("merge"))
        state = place(engine, {**KINGS, "a1": ("R", 0), "c3": ("R", 1)})
        assert state.initiative == 1
        first = find(engine, state, 0, "a1", "a3")
        second = find(engine, state, 1, "c3", "a3")
        after, _, events = round_of(engine, state, first, second)
        assert at(engine, after, "a3") == ("R", 1)
        assert after.initiative == 0
        assert (2, square(engine, "a3"), 1) in events
        assert after.repetition == (after.key,)

    def test_a_swap_of_two_landing_moves_trades_places(self, gardner: Engine) -> None:
        state = place(gardner, {**KINGS, "a1": ("R", 0), "a4": ("R", 1)})
        first = find(gardner, state, 0, "a1", "a4")
        second = find(gardner, state, 1, "a4", "a1")
        after, _, events = round_of(gardner, state, first, second)
        assert at(gardner, after, "a4") == ("R", 0) and at(gardner, after, "a1") == ("R", 1)
        assert events == ()

    def test_a_swap_with_one_capture_only_move_takes_the_void_piece(self, gardner: Engine) -> None:
        state = place(gardner, {**KINGS, "b2": ("P", 0), "c3": ("B", 1)})
        first = find(gardner, state, 0, "b2", "c3")
        second = find(gardner, state, 1, "c3", "b2")
        after, _, events = round_of(gardner, state, first, second)
        assert at(gardner, after, "b2") == ("B", 1) and at(gardner, after, "c3") is None
        assert events == ((1, square(gardner, "b2"), 0),)
        assert after.void_run == (-1, -1)

    def test_a_swap_of_two_capture_only_moves_is_void_both_ways(self, gardner: Engine) -> None:
        state = place(gardner, {**KINGS, "b2": ("P", 0), "c3": ("P", 1)})
        first = find(gardner, state, 0, "b2", "c3")
        second = find(gardner, state, 1, "c3", "b2")
        after, _, events = round_of(gardner, state, first, second)
        assert after.board == state.board
        assert after.void_run == (square(gardner, "b2"), square(gardner, "c3"))
        assert len(events) == 2 and all(event[0] == 1 for event in events)

    def test_a_chase_lands_on_the_square_its_victim_left(self, gardner: Engine) -> None:
        state = place(gardner, {**KINGS, "b1": ("N", 0), "c3": ("B", 1)})
        first = find(gardner, state, 0, "b1", "c3")
        second = find(gardner, state, 1, "c3", "a5")
        after, _, _ = round_of(gardner, state, first, second)
        assert at(gardner, after, "c3") == ("N", 0) and at(gardner, after, "a5") == ("B", 1)
        assert after.halfmove == state.halfmove + 2

    def test_a_capture_of_a_standing_piece_is_ordinary(self, gardner: Engine) -> None:
        state = place(gardner, {**KINGS, "b1": ("N", 0), "c3": ("B", 1), "a5": ("P", 1)})
        first = find(gardner, state, 0, "b1", "c3")
        second = find(gardner, state, 1, "a5", "a4")
        after, _, _ = round_of(gardner, state, first, second)
        assert at(gardner, after, "c3") == ("N", 0) and after.halfmove == 0

    def test_crossing_paths_is_no_collision(self, gardner: Engine) -> None:
        state = place(gardner, {"e1": ("K", 0), "e4": ("K", 1), "a1": ("R", 0), "a5": ("R", 1)})
        first = find(gardner, state, 0, "a1", "a4")
        second = find(gardner, state, 1, "a5", "a2")
        after, _, events = round_of(gardner, state, first, second)
        assert at(gardner, after, "a4") == ("R", 0) and at(gardner, after, "a2") == ("R", 1)
        assert events == ()

    def test_independent_moves_are_both_made(self, gardner: Engine) -> None:
        state = gardner.initial_state()
        first = find(gardner, state, 0, "a2", "a3")
        second = find(gardner, state, 1, "e4", "e3")
        after, _, _ = round_of(gardner, state, first, second)
        assert at(gardner, after, "a3") == ("P", 0) and at(gardner, after, "e3") == ("P", 1)
        assert after.ply == 2


class TestTheOrderOfARound:
    """§4's steps, one hand-built round each."""

    def test_the_promoted_piece_is_the_one_that_meets(self) -> None:
        engine = Engine(with_piece(GARDNER, "N", beats=["P"]))
        state = place(engine, {"a1": ("K", 0), "e1": ("K", 1), "d4": ("P", 0), "c3": ("N", 1)})
        queen = engine.spec.piece_ids.index("Q")
        first = find(engine, state, 0, "d4", "d5", MOVE_PROMOTION, queen)
        second = find(engine, state, 1, "c3", "d5")
        after, _, events = round_of(engine, state, first, second)
        assert at(engine, after, "d5") == ("Q", 0)
        assert events == ((0, square(engine, "d5"), 0),)

    def test_a_merge_comes_before_the_meeting(self) -> None:
        engine = Engine(load("merge"))
        state = place(engine, {**KINGS, "c2": ("P", 0), "d2": ("P", 0), "d4": ("R", 1)})
        first = find(engine, state, 0, "c2", "d2")
        second = find(engine, state, 1, "d4", "d2")
        after, _, events = round_of(engine, state, first, second)
        # The merged rook meets the rook: a tie, decided by the initiative (player 1).
        assert at(engine, after, "d2") == ("R", 1) and at(engine, after, "c2") is None
        assert (2, square(engine, "d2"), 1) in events

    def test_every_capture_is_decided_before_any_explosion(self) -> None:
        engine = Engine(load("atomic"))
        state = place(engine, {
            "a1": ("K", 0), "e5": ("K", 1), "b1": ("N", 0), "c3": ("B", 1), "d3": ("N", 1),
            "e1": ("R", 0),
        })
        first = find(engine, state, 0, "b1", "c3")
        second = find(engine, state, 1, "d3", "e1")
        after, _, _ = round_of(engine, state, first, second)
        # Both captured; both blasts went off; the knight that took e1 had left d3 first.
        assert at(engine, after, "c3") is None and at(engine, after, "e1") is None
        assert at(engine, after, "d3") is None

    def test_a_captor_destroyed_by_a_blast_has_still_captured(self) -> None:
        engine = Engine(load("atomic"))
        state = place(engine, {
            "a1": ("K", 0), "e5": ("K", 1), "a3": ("R", 0), "c3": ("B", 1), "b4": ("N", 1),
            "c2": ("N", 0),
        })
        first = find(engine, state, 0, "a3", "c3")
        second = find(engine, state, 1, "b4", "c2")
        after, _, _ = round_of(engine, state, first, second)
        # The black knight took c2 and stands in c3's blast; its capture still counted,
        # so the white knight is gone and c2's own blast went off as well.
        for name in ("a3", "b4", "c2", "c3"):
            assert at(engine, after, name) is None
        assert after.halfmove == 0

    def test_a_reborn_piece_needs_its_home_empty_after_the_meetings(self) -> None:
        engine = Engine(load("circe"))
        base = {"a1": ("K", 0), "b5": ("K", 1), "b1": ("N", 0), "c3": ("R", 1), "d5": ("P", 1)}
        state = place(engine, base)
        first = find(engine, state, 0, "b1", "c3")
        second = find(engine, state, 1, "d5", "d4")
        after, _, _ = round_of(engine, state, first, second)
        assert at(engine, after, "a5") == ("R", 1)
        second = find(engine, state, 1, "b5", "a5")
        after, _, _ = round_of(engine, state, first, second)
        assert at(engine, after, "a5") == ("K", 1)
        assert sum(1 for value in after.board if value) == 4
        assert ("R", 1) not in [at(engine, after, f"{f}{r}") for f in "abcde" for r in range(1, 6)]

    def test_a_struck_piece_goes_to_its_captors_origin_only_if_empty(self) -> None:
        engine = Engine(load("swap"))
        base = {"a1": ("K", 0), "e5": ("K", 1), "b1": ("N", 0), "c3": ("Q", 1), "d3": ("B", 1)}
        state = place(engine, base)
        first = find(engine, state, 0, "b1", "c3")
        second = find(engine, state, 1, "d3", "e4")
        after, _, _ = round_of(engine, state, first, second)
        assert at(engine, after, "b1") == ("R", 1) and at(engine, after, "c3") == ("N", 0)
        second = find(engine, state, 1, "d3", "b1")
        after, _, _ = round_of(engine, state, first, second)
        # The bishop's chase lands on b1 first, so the struck queen is removed.
        assert at(engine, after, "b1") == ("B", 1) and at(engine, after, "c3") == ("N", 0)
        assert ("R", 1) not in [at(engine, after, s) for s in ("a1", "b1", "c1", "d1", "e1")]

    def test_claims_follow_the_meetings_and_the_wild_piece_steps_last(self) -> None:
        engine = Engine(load("wild"))
        state = place(engine, {
            **KINGS, "a1": ("R", 0), "c4": ("R", 1), "b2": ("P", 0), "b4": ("P", 1),
            "a3": ("N", 2),
        })
        first = find(engine, state, 0, "a1", "a2")
        second = find(engine, state, 1, "c4", "c2")
        after, phase, _ = round_of(engine, state, first, second)
        assert after.colour[square(engine, "a2")] == 1 and after.colour[square(engine, "c2")] == 2
        assert [(step.origin, step.target) for step in phase] == [
            (square(engine, "a3"), square(engine, "b3"))
        ]
        tied = place(engine, {**KINGS, "a1": ("R", 0), "c3": ("R", 1), "e3": ("N", 2)})
        first = find(engine, tied, 0, "a1", "a3")
        second = find(engine, tied, 1, "c3", "a3")
        after, _, _ = round_of(engine, tied, first, second)
        assert after.colour[square(engine, "a3")] == 0

    def test_a_transit_meets_at_its_exit(self) -> None:
        engine = Engine(load("portals"))
        state = place(engine, {"a1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0), "c5": ("Q", 1)})
        transits = [
            move for move in engine.legal_moves_for(state, 0)
            if decode_move(move)[2] == MOVE_PORTAL and decode_move(move)[0] == square(engine, "b1")
        ]
        assert transits
        exit_square = decode_move(transits[0])[1]
        name = f"{'abcde'[exit_square % 5]}{exit_square // 5 + 1}"
        second = find(engine, state, 1, "c5", name)
        after, _, events = round_of(engine, state, transits[0], second)
        assert (0, exit_square, -1) in events


# ------------------------------------------------------------------ the void bar


@pytest.fixture(scope="module")
def void_engine() -> Engine:
    return Engine(VOID)


def standoff(engine: Engine, **extra: tuple[str, int]) -> GameState:
    return place(engine, {"b2": ("P", 0), "c3": ("P", 1), "e1": ("W", 0), "a5": ("W", 1), **extra})


def void_round(engine: Engine, state: GameState) -> GameState:
    first = find(engine, state, 0, "b2", "c3")
    second = find(engine, state, 1, "c3", "b2")
    return round_of(engine, state, first, second)[0]


def wazirs(engine: Engine, state: GameState, first: tuple[str, str], second: tuple[str, str]) -> GameState:
    return round_of(
        engine, state, find(engine, state, 0, *first), find(engine, state, 1, *second)
    )[0]


def captures(engine: Engine, state: GameState, player: int) -> set[tuple[int, int]]:
    return {decode_move(m)[:2] for m in engine.legal_moves_for(state, player)}


class TestTheVoidBar:
    """§2.7, each clause of §7's Q918a box a test."""

    def test_two_void_rounds_in_a_row_bar_the_pair_for_both(self, void_engine: Engine) -> None:
        engine = void_engine
        b2, c3 = square(engine, "b2"), square(engine, "c3")
        state = void_round(engine, standoff(engine))
        assert state.void_run == (b2, c3) and state.bars == ()
        assert (b2, c3) in captures(engine, state, 0)
        state = void_round(engine, state)
        assert state.bars == ((b2, c3),) and state.void_run == (-1, -1)
        assert (b2, c3) not in captures(engine, state, 0)
        assert (c3, b2) not in captures(engine, state, 1)
        assert (c3, square(engine, "c2")) in captures(engine, state, 1)

    def test_another_piece_may_still_take_on_a_barred_square(self, void_engine: Engine) -> None:
        engine = void_engine
        state = standoff(engine, d2=("P", 0))
        state = void_round(engine, void_round(engine, state))
        assert state.bars
        assert (square(engine, "d2"), square(engine, "c3")) in captures(engine, state, 0)

    @pytest.mark.parametrize(
        ("extra", "black"),
        [({}, ("a5", "a4")), ({"d4": ("W", 1)}, ("d4", "d3"))],
        ids=["far", "near"],
    )
    def test_a_different_round_between_resets_the_run(
        self, void_engine: Engine, extra: dict[str, tuple[str, int]], black: tuple[str, str]
    ) -> None:
        engine = void_engine
        state = void_round(engine, standoff(engine, **extra))
        state = wazirs(engine, state, ("e1", "e2"), black)
        assert state.void_run == (-1, -1)
        state = void_round(engine, state)
        assert state.void_run == (square(engine, "b2"), square(engine, "c3"))
        assert state.bars == ()

    def test_a_void_round_of_another_pair_starts_a_new_run(self, void_engine: Engine) -> None:
        engine = void_engine
        state = standoff(engine, d2=("P", 0), e3=("P", 1))
        state = void_round(engine, state)
        first = find(engine, state, 0, "d2", "e3")
        second = find(engine, state, 1, "e3", "d2")
        state = round_of(engine, state, first, second)[0]
        assert state.void_run == (square(engine, "d2"), square(engine, "e3"))
        assert state.bars == ()

    def test_the_bar_lasts_until_either_square_is_touched(self, void_engine: Engine) -> None:
        engine = void_engine
        state = standoff(engine, d1=("W", 0), d5=("W", 1))
        state = void_round(engine, void_round(engine, state))
        bar = state.bars
        assert bar
        for _ in range(2):
            state = wazirs(engine, state, ("e1", "e2"), ("a5", "a4"))
            state = wazirs(engine, state, ("e2", "e1"), ("a4", "a5"))
            assert state.bars == bar
        # Leaving c3 and coming back: the square was touched, so the bar is gone.
        first = find(engine, state, 0, "e1", "e2")
        second = find(engine, state, 1, "c3", "c2")
        state = round_of(engine, state, first, second)[0]
        assert state.bars == ()

    def test_a_wild_piece_eating_one_leaves_neither_run_nor_bar(self) -> None:
        spec = copy.deepcopy(VOID)
        spec["rules"]["npcs"] = [{"piece": "W", "route": ["a3", "b3", "c3", "d3", "e3"]}]
        engine = Engine(spec)
        state = standoff(engine, b3=("W", 2))
        after = void_round(engine, state)
        assert at(engine, after, "c3") == ("W", 2)
        assert after.void_run == (-1, -1) and after.bars == ()

    def test_a_barred_only_move_is_no_move(self) -> None:
        spec = copy.deepcopy(VOID)
        spec["rules"]["stalemate"] = "loss"
        engine = Engine(spec)
        state = place(engine, {"b2": ("P", 0), "c3": ("P", 1), "b3": ("P", 1)})
        assert engine.legal_moves_for(state, 0) == [find(engine, state, 0, "b2", "c3")]
        state = void_round(engine, void_round(engine, state))
        assert engine.legal_moves_for(state, 0) == []
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.winner == 1 and outcome.reason == "stalemate_loss"

    def test_two_void_rounds_are_no_repetition_but_void_away_and_back_is(
        self, void_engine: Engine
    ) -> None:
        engine = void_engine
        state = void_round(engine, void_round(engine, standoff(engine)))
        assert engine.terminal(state) is None
        state = standoff(engine)
        for _ in range(3):
            state = void_round(engine, state)
            if engine.terminal(state) is not None:
                break
            state = wazirs(engine, state, ("e1", "e2"), ("a5", "a4"))
            state = wazirs(engine, state, ("e2", "e1"), ("a4", "a5"))
            if engine.terminal(state) is not None:
                break
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.reason == "repetition"

    def test_the_run_and_a_bar_are_in_the_key(self, void_engine: Engine) -> None:
        engine = void_engine
        state = standoff(engine)
        pair = (square(engine, "b2"), square(engine, "c3"))
        keys = {
            engine._compute_hash(state),
            engine._compute_hash(state.with_(void_run=pair)),
            engine._compute_hash(state.with_(bars=(pair,))),
            engine._compute_hash(state.with_(bars=(pair, (pair[0], 0)))),
            engine._compute_hash(state.with_(bars=((0, 1), (2, 3)))),
            engine._compute_hash(state.with_(bars=((0, 3), (2, 1)))),
        }
        assert len(keys) == 6


# --------------------------------------------------------------------- endings


class TestTheEndings:
    def test_both_royals_lost_in_one_round_is_a_draw(self, gardner: Engine) -> None:
        state = place(gardner, {"a1": ("K", 0), "c3": ("K", 1)})
        first = find(gardner, state, 0, "a1", "b2")
        second = find(gardner, state, 1, "c3", "b2")
        after, _, _ = round_of(gardner, state, first, second)
        outcome = gardner.terminal(after)
        assert outcome is not None and outcome.winner is None
        assert outcome.reason == "royal_captured"

    def test_a_royal_taken_standing_loses(self, gardner: Engine) -> None:
        state = place(gardner, {"a1": ("K", 0), "c3": ("K", 1), "b1": ("N", 0)})
        first = find(gardner, state, 0, "b1", "c3")
        second = find(gardner, state, 1, "c3", "c4")
        after, _, _ = round_of(gardner, state, first, second)
        assert gardner.terminal(after) is None
        second = find(gardner, state, 1, "c3", "d3")
        after, _, _ = round_of(gardner, state, first, second)
        assert gardner.terminal(after) is None
        state = place(gardner, {"a1": ("K", 0), "c3": ("K", 1), "b1": ("N", 0), "e5": ("P", 1)})
        first = find(gardner, state, 0, "b1", "c3")
        second = find(gardner, state, 1, "e5", "e4")
        after, _, _ = round_of(gardner, state, first, second)
        outcome = gardner.terminal(after)
        assert outcome is not None and outcome.winner == 0

    def test_a_round_is_two_plies_on_the_clock(self) -> None:
        engine = Engine(with_rules(GARDNER, max_plies=4))
        state = engine.initial_state()
        for _ in range(2):
            state = round_of(
                engine, state, engine.legal_moves_for(state, 0)[0], engine.legal_moves_for(state, 1)[0]
            )[0]
        assert state.ply == 4
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.reason == "ply_limit"


# ------------------------------------------------------------- perft and games


JOINT_PERFT = {
    # Frozen after the oracle agreed on the rounds below them (§7's Q918a box).
    "gardner": (49, 4393),
    "shogi": (16, 732),
}


@pytest.mark.parametrize("name", sorted(JOINT_PERFT))
def test_joint_perft_is_frozen(name: str) -> None:
    engine = Engine(load(name))
    state = engine.initial_state()
    assert (engine.perft(state, 1), engine.perft(state, 2)) == JOINT_PERFT[name]
    first, second = engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1)
    assert engine.perft(state, 1) == len(first) * len(second)


@pytest.mark.parametrize("name", ["gardner", "shogi", "merge"])
def test_a_game_is_a_function_of_its_seed(name: str) -> None:
    engine = Engine(load(name))
    one = play_game(engine, ("random", "random"), 7)
    two = play_game(engine, ("random", "random"), 7)
    assert one.moves == two.moves and one.reason == two.reason
    assert one.plies == len(one.moves) and one.plies % 2 == 0
    assert len(one.branching) == one.plies
    state = engine.initial_state()
    for index in range(0, len(one.moves), 2):
        assert one.moves[index] in engine.legal_moves_for(state, 0)
        assert one.moves[index + 1] in engine.legal_moves_for(state, 1)
        state = engine.apply_round(state, one.moves[index], one.moves[index + 1])
    outcome = engine.terminal(state)
    assert outcome is not None and outcome.reason == one.reason and outcome.winner == one.winner


def test_the_alternating_searches_are_refused_on_rounds() -> None:
    """Q918b: random, greedy and matrix play rounds; minimax and mcts are refused (§6.2)."""
    engine = Engine(GARDNER)
    for searcher in ("minimax:nodes=100", "mcts:nodes=50"):
        with pytest.raises(ValueError, match="minimax and mcts search one mover at a time"):
            play_game(engine, ("random", searcher), 0)


# --------------------------------------------------------- the alternating twin


def twin(spec: dict[str, Any]) -> dict[str, Any]:
    """The same file without `simultaneous` and `beats` (§6.5): an alternating game."""
    alternating = copy.deepcopy(spec)
    del alternating["rules"]["simultaneous"]
    for piece in alternating["pieces"]:
        piece.pop("beats", None)
    return alternating


@pytest.mark.parametrize("name", ["gardner", "shogi", "portals", "wild"])
def test_a_rounds_moves_are_the_alternating_twins(name: str) -> None:
    """Each player's moves in a round are exactly the twin's for that player to move."""
    import random

    engine, other = Engine(load(name)), Engine(twin(load(name)))
    rng = random.Random(5)
    state = engine.initial_state()
    for _ in range(30):
        if engine.terminal(state) is not None:
            break
        lists = []
        for player in (0, 1):
            ours = engine.legal_moves_for(state, player)
            alternating = state.with_(side_to_move=player, initiative=-1, void_run=(-1, -1), bars=())
            if not state.bars:
                assert ours == other.legal_moves(alternating)
            lists.append(ours)
        state = engine.apply_round(state, rng.choice(lists[0]), rng.choice(lists[1]))


def test_the_twin_plays_the_same_moves_as_fairy_stockfish() -> None:
    pytest.importorskip("pyffish")
    from chess_variant_sandbox.fairy.verify import PyffishBackend, verify

    refused = verify(GARDNER, PyffishBackend(), "", depth=1, games=0, plies=0)
    assert not refused.ok and any("rules.simultaneous" in reason for reason in refused.reasons)
    result = verify(twin(GARDNER), PyffishBackend(), "", depth=2, games=2, plies=40)
    assert result.ok, result.summary()
