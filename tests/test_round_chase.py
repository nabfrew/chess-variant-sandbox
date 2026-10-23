"""Schema 27 (Q942a): the three switches against the royal chase.

`docs/archive/plan-round-chase.md` §3 R1 (the rest), R2 (`swap: "collision"`) and X1 (`both:
"score"`), and the Q942a row of its §4.2. Schema 29 (Q983a, `docs/plan-rest.md` §4.4) retired
R1's spelling, `rules.simultaneous.rest`, for `"rest": 1` on every piece, which is the same rest
in rounds; the tests of R1 below play that spelling, and the old one is refused: the keys and their refusals, one test
per rule on the positions of §2 and §3, the rest's legal-move filter and the rounds against
the declared oracle (`tests/simultaneous_oracle.py`), the key against `_compute_hash`, and
the joint perft of a `rest` spec frozen after the oracle agreed with it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from simultaneous_oracle import legal_for, resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.canonical import load_spec
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_twenty_seven_uses

HERE = Path(__file__).parent
ROOT = HERE.parent

EVENT_VOID, EVENT_SWAP = 1, 4


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"_simultaneous_{name}.json").read_text(encoding="utf-8"))


GARDNER = load("gardner")


def gardner(*, resting: bool = False, **simultaneous: Any) -> dict[str, Any]:
    """Gardner in rounds at schema 27 with these switches, and material scoring under
    `both`; ``resting``, R1, at schema 29: `"rest": 1` on every piece."""
    spec = copy.deepcopy(GARDNER)
    spec["schema_version"] = 29 if resting else 27
    spec["rules"]["simultaneous"] = dict(simultaneous)
    if simultaneous.get("both") == "score":
        spec["rules"]["scoring"] = {"count": "material"}
    if resting:
        for piece in spec["pieces"]:
            piece["rest"] = 1
    return spec


def rests(engine: Engine, first: str | None, second: str | None) -> tuple[Any, Any]:
    """The position's rest when each player's moved piece rests one round on that square."""
    return tuple(
        ((square(engine, name), 1),) if name else () for name in (first, second)
    )


def square(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def find(engine: Engine, state: GameState, player: int, origin: str, target: str) -> int:
    for move in engine.legal_moves_for(state, player):
        o, t, _, _ = decode_move(move)
        if o == square(engine, origin) and t == square(engine, target):
            return move
    raise AssertionError(f"player {player} has no move {origin}->{target}")


def place(engine: Engine, pieces: dict[str, tuple[str, int]]) -> GameState:
    state = engine.make_state(pieces)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def round_of(engine: Engine, state: GameState, first: int, second: int):
    """One round through the core, held to the oracle and the slow key."""
    after, _, events = engine.apply_round_with_phase(state, first, second)
    assert after == resolve_round(engine, state, first, second)
    assert after.key == engine._compute_hash(after)
    for player in (0, 1):
        assert engine.legal_moves_for(after, player) == legal_for(engine, after, player)
    return after, events


# The lock of §2 (game 217 of runs/e043fce6968e-1000, round 37). White is player 0.
LOCK = {
    "e1": ("K", 0), "b5": ("Q", 0), "c1": ("B", 0), "d2": ("P", 0), "e3": ("P", 0),
    "b4": ("K", 1), "a1": ("Q", 1), "b3": ("R", 1), "d3": ("P", 1), "e4": ("P", 1),
}


# ------------------------------------------------------------------ the keys


def test_the_collector_names_each_switch_unless_it_is_off() -> None:
    for key, value in (("swap", "collision"), ("both", "score")):
        spec = gardner(**{key: value})
        assert f"rules.simultaneous.{key}" in _schema_twenty_seven_uses(spec)
    assert not {
        name for name in _schema_twenty_seven_uses(GARDNER) if "simultaneous" in name
    }


@pytest.mark.parametrize(
    ("simultaneous", "words"),
    [
        ({"rest": True}, "rest"),
        ({"rest": False}, "rest"),
        ({"swap": "always"}, 'swap must be "collision" or "chase"'),
        ({"both": "win"}, 'both must be "score" or "draw"'),
    ],
)
def test_unknown_values_and_the_retired_rest_are_refused(
    simultaneous: dict[str, Any], words: str
) -> None:
    assert any(words in error for error in validate_spec(gardner(**simultaneous)))


@pytest.mark.parametrize("simultaneous", [{"swap": "chase"}, {"both": "draw"}, {"tie": "both"}])
def test_an_off_value_is_the_game_without_it(simultaneous: dict[str, Any]) -> None:
    """Q1008c: the off value written out is accepted and names the game without it."""
    from chess_variant_sandbox.spec import spec_hash
    from chess_variant_sandbox.spec.spelling import current_spelling

    written = gardner(**simultaneous)
    assert validate_spec(written) == []
    assert spec_hash(written) == spec_hash(gardner())
    assert current_spelling(written)["rules"]["simultaneous"] == {}


def test_both_without_scoring_is_refused() -> None:
    spec = gardner(both="score")
    del spec["rules"]["scoring"]
    assert any("and there is none" in error for error in validate_spec(spec))


def test_a_swap_rule_where_no_swap_can_happen_is_refused() -> None:
    """The watering hole: no animal of one side takes one of the other that takes it back."""
    spec = load_spec(ROOT / "archive" / "watering-hole.json")
    spec["schema_version"] = 27
    spec["rules"]["simultaneous"]["swap"] = "collision"
    assert any("can never fire" in error for error in validate_spec(spec))


def test_each_switch_compiles_into_the_rule_and_crosses_only_when_on() -> None:
    plain = compile_spec(GARDNER)
    assert "sim_chase" not in canonical_tables(plain)
    assert "zobrist_rest" not in canonical_tables(plain)
    for key, value, index in (("swap", "collision", 1), ("both", "score", 2)):
        compiled = compile_spec(gardner(**{key: value}))
        assert compiled.simultaneous is not None
        tables = build_tables(compiled)
        # The first slot was the round's rest, always 0 since schema 29.
        assert tables["sim_chase"] == [int(i == index) for i in range(3)]
        assert tables["zobrist_rest"] == [] and tables["piece_rest"] == []
    # The rest is a piece property now: `"rest": 1` on every piece crosses as `piece_rest`,
    # with one key per (player, square), schema 27's layout, and no chase switch.
    compiled = compile_spec(gardner(resting=True))
    tables = canonical_tables(compiled)
    assert "sim_chase" not in tables
    assert tables["piece_rest"] == [1] * compiled.n_types
    assert len(tables["zobrist_rest"]) == 2 * compiled.n_squares


# ------------------------------------------------------------ R1, the rest


def test_under_rest_the_two_pieces_of_the_lock_may_not_move_the_next_round() -> None:
    engine = Engine(gardner(resting=True))
    state = place(engine, LOCK)
    queen = find(engine, state, 0, "b5", "b4")
    king = find(engine, state, 1, "b4", "b5")
    after, _ = round_of(engine, state, queen, king)
    assert after.rest == rests(engine, "b4", "b5")
    white = engine.legal_moves_for(after, 0)
    black = engine.legal_moves_for(after, 1)
    assert white and black
    assert all(decode_move(m)[0] != square(engine, "b4") for m in white)
    assert all(decode_move(m)[0] != square(engine, "b5") for m in black)
    # Without the rule the same two pieces are offered their moves.
    plain = Engine(gardner())
    loose = plain.apply_round(place(plain, LOCK), queen, king)
    assert any(decode_move(m)[0] == square(plain, "b4") for m in plain.legal_moves_for(loose, 0))


def test_the_lift_offers_a_lone_piece_its_moves() -> None:
    engine = Engine(gardner(resting=True))
    state = place(engine, {"c1": ("K", 0), "c5": ("K", 1)})
    after, _ = round_of(
        engine, state, find(engine, state, 0, "c1", "c2"), find(engine, state, 1, "c5", "c4")
    )
    assert after.rest == rests(engine, "c2", "c4")
    for player, at in ((0, "c2"), (1, "c4")):
        moves = engine.legal_moves_for(after, player)
        assert moves and all(decode_move(m)[0] == square(engine, at) for m in moves)
        assert moves == engine.legal_moves_for(after.with_(rest=((), ())), player)


def test_a_rested_royal_attacked_by_a_standing_piece_is_taken() -> None:
    engine = Engine(gardner(resting=True))
    state = place(engine, {"e1": ("K", 0), "d1": ("R", 0), "e5": ("K", 1), "a4": ("P", 1)})
    # Round 1: White's king steps aside, Black's king walks into the rook's file.
    state, _ = round_of(
        engine, state, find(engine, state, 0, "e1", "e2"), find(engine, state, 1, "e5", "d5")
    )
    assert engine.terminal(state) is None
    black = engine.legal_moves_for(state, 1)
    assert black and all(decode_move(m)[0] != square(engine, "d5") for m in black)
    # Round 2: the rook, which stood, strikes; the king cannot run.
    state, _ = round_of(engine, state, find(engine, state, 0, "d1", "d5"), black[0])
    outcome = engine.terminal(state)
    assert outcome is not None and outcome.winner == 0 and outcome.reason == "royal_captured"


# ---------------------------------------------------------- R2, the swap meets


def test_under_swap_the_king_and_queen_of_the_lock_meet_and_white_wins() -> None:
    engine = Engine(gardner(swap="collision"))
    state = place(engine, LOCK)
    after, events = round_of(
        engine, state, find(engine, state, 0, "b5", "b4"), find(engine, state, 1, "b4", "b5")
    )
    assert after.board[square(engine, "b4")] == 0 and after.board[square(engine, "b5")] == 0
    assert (EVENT_SWAP, square(engine, "b4"), -1) in events
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0 and outcome.reason == "royal_captured"
    # Without the switch the two pass each other and both stand.
    plain = Engine(gardner())
    passed = plain.apply_round(place(plain, LOCK), find(engine, state, 0, "b5", "b4"),
                               find(engine, state, 1, "b4", "b5"))
    assert passed.board[square(plain, "b4")] and passed.board[square(plain, "b5")]
    assert plain.terminal(passed) is None


@pytest.mark.parametrize("scored", [False, True])
def test_two_bare_kings_striking_each_other_draw_at_once(scored: bool) -> None:
    switches: dict[str, Any] = {"swap": "collision"}
    if scored:
        switches["both"] = "score"
    engine = Engine(gardner(**switches))
    state = place(engine, {"e1": ("K", 0), "d2": ("K", 1)})
    after, _ = round_of(
        engine, state, find(engine, state, 0, "e1", "d2"), find(engine, state, 1, "d2", "e1")
    )
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner is None and outcome.reason == "royal_captured"


def test_a_pawns_diagonal_against_a_bishop_meets_by_beats_and_is_not_void() -> None:
    spec = gardner(swap="collision")
    for piece in spec["pieces"]:
        if piece["id"] == "P":
            piece["beats"] = ["B"]
        if piece["id"] == "B":
            piece["beats"] = []
    engine = Engine(spec)
    pieces = {"a1": ("K", 0), "c2": ("P", 0), "a5": ("K", 1), "d3": ("B", 1)}
    state = place(engine, pieces)
    pawn = find(engine, state, 0, "c2", "d3")
    bishop = find(engine, state, 1, "d3", "c2")
    after, events = round_of(engine, state, pawn, bishop)
    assert not [event for event in events if event[0] == EVENT_VOID]
    assert (EVENT_SWAP, square(engine, "d3"), 0) in events
    assert engine.spec.type_of(after.board[square(engine, "d3")]) == engine.spec.piece_ids.index("P")
    assert after.board[square(engine, "c2")] == 0
    # Without the switch the pawn's capture-only move is void and the bishop takes it.
    del spec["rules"]["simultaneous"]["swap"]
    spec["schema_version"] = 26
    plain = Engine(spec)
    _, _, plain_events = plain.apply_round_with_phase(place(plain, pieces), pawn, bishop)
    assert [event for event in plain_events if event[0] == EVENT_VOID]


# --------------------------------------------------------- X1, both by score


def double_capture(extra: dict[str, tuple[str, int]]) -> tuple[Engine, GameState]:
    engine = Engine(gardner(both="score"))
    pieces = {"a1": ("K", 0), "e1": ("R", 0), "e5": ("K", 1), "a5": ("R", 1), **extra}
    state = place(engine, pieces)
    after, _ = round_of(
        engine, state, find(engine, state, 0, "e1", "e5"), find(engine, state, 1, "a5", "a1")
    )
    return engine, after


def test_two_royals_taken_in_one_round_go_to_the_side_ahead() -> None:
    engine, after = double_capture({"c3": ("N", 0)})
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0 and outcome.reason == "royal_captured"
    engine, after = double_capture({"c3": ("N", 1)})
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 1 and outcome.reason == "royal_captured"


def test_two_royals_taken_in_one_round_with_equal_material_draw() -> None:
    engine, after = double_capture({})
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner is None and outcome.reason == "royal_captured"


def test_without_the_switch_a_double_capture_with_unequal_material_draws() -> None:
    spec = gardner()
    spec["rules"]["scoring"] = {"count": "material"}
    engine = Engine(spec)
    state = place(engine, {"a1": ("K", 0), "e1": ("R", 0), "c3": ("N", 0), "e5": ("K", 1),
                           "a5": ("R", 1)})
    after = engine.apply_round(
        state, find(engine, state, 0, "e1", "e5"), find(engine, state, 1, "a5", "a1")
    )
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner is None


# ------------------------------------------------------------ perft, frozen


def oracle_perft(engine: Engine, state: GameState, depth: int) -> int:
    """Joint perft from the oracle's rest filter and resolution alone."""
    first, second = legal_for(engine, state, 0), legal_for(engine, state, 1)
    if depth == 1:
        return len(first) * len(second)
    return sum(
        oracle_perft(engine, resolve_round(engine, state, a, b), depth - 1)
        for a in first for b in second
    )


JOINT_PERFT = {
    # Frozen after `oracle_perft` agreed (Q942a). Gardner in rounds without the rest is
    # (49, 4393), `tests/test_schema_twenty_six.py`.
    "resting": (49, 3031),
}


def test_the_joint_perft_of_a_resting_game_agrees_with_the_oracle_and_is_frozen() -> None:
    engine = Engine(load("resting"))
    state = engine.initial_state()
    counted = (engine.perft(state, 1), engine.perft(state, 2))
    assert counted == (oracle_perft(engine, state, 1), oracle_perft(engine, state, 2))
    assert counted == JOINT_PERFT["resting"]


def test_the_page_names_a_swap_that_met_and_does_not_call_it_a_pass() -> None:
    """The page's words for a round read the core's `EVENT_SWAP` (Q942b draws it)."""
    from chess_variant_sandbox.web.play import round_reveal

    engine = Engine(gardner(swap="collision"))
    state = place(engine, LOCK)
    moves = (find(engine, state, 0, "b5", "b4"), find(engine, state, 1, "b4", "b5"))
    after, _, events = engine.apply_round_with_phase(state, *moves)
    reveal = round_reveal(engine, state, after, moves, events, ("White", "Black"))
    assert [event["kind"] for event in reveal["events"]] == ["swap"]
    assert reveal["results"] == ["both lost", "both lost"]
    assert not any("each left the square the other took" in words for words in reveal["words"])
    assert any("met" in words and "both were lost" in words for words in reveal["words"])
