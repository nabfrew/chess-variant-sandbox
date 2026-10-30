"""F10c (Q752c): the last of `rust/src/game.rs`'s unreached behaviour lines, with
`rounds.rs`', `rest.rs`' and `tables.rs`' few, held to an oracle or a hand count.

`docs/plan-tests.md` F10c lists them from Q752b's map (`runs/q752b/after-classified.txt`).
Each test builds the one game and position that walks the line, and asserts what the line
decides, so a mutant there fails it: a territory colour sized at load; a wild piece eating one
of two royals; a sweep's attack read past a square it may not land on; castling's corners (a
rook landing off the board, a crossed square attacked under file castling, a king step past
the edge, a castling word with no rook); circe beside Andernach with the home taken; a capture
into an unlimited hand slot; attacks through a portal whose far end is missing, and through a
portal or a fold by a piece that may not take the victim or may not go in; a round's circe
victim with no home; a resting entry on an empty square; and a sweep word no sweep makes
(refused by `apply` since CS6, so `sweep_path`'s two empty answers are no longer reached).
Where the core answers an attack, `tests/portal_oracle.py`, `tests/fold_oracle.py` or
`tests/simultaneous_oracle.py` answers it too.

**Not tested, because no test can reach them while the code is right** (recorded in F10c):

* `game.rs` 968: `attacks_square` on an empty square under a capture matrix. Every caller
  (`threatened`, `attacks_any`, the sight report) asks of a royal's square or of an occupied
  goal square; castling's empty transit squares go through `threatened_as` with the king's type.
* `game.rs` 1077: a sweep source with neither `SWEEP_LANDS` nor `SWEEP_BEYOND`. The compiler
  files a sweep source only when one bit is set (`compile._build_attack_sources`), and a source
  with `SWEEP_LANDS` has returned on the line before.
* `game.rs` 1373: the check memo read while `scratch` is up. `scratch` is raised only by
  `leaves_royal_attacked`, `restrict`'s `forbid_check` filter and the drop-mate test's
  placement, and each asks `threatened`/`attacks_any`, never `in_check`; the guard is the
  memo's rule, kept by every caller.
* `game.rs` 1752, 1761: a strike arm with `becomes < 0` and an unchanged attacker. That is
  `becomes == -1` (the early `return None` above the match took it) or `REMOVED_IN_PLACE`,
  which `tables.rs` refuses outside `stay`; `advance` with a survivor is refused by the
  validator (*has nowhere to put a struck piece that survives*).
* `tables.rs` 1469: `facing_from` past the table. `royals_face` is asked only under
  `royal_facing: "forbidden"`, and then `compile._facing_lines` has a row for every square.
"""

from __future__ import annotations

import copy
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from fold_oracle import FoldOracle
from portal_oracle import PortalOracle
from simultaneous_oracle import resolve_round
from test_backends import _assert_same_played
from test_round_corners import _word
from test_schema_nineteen import _position, _unmoved, wormhole
from test_schema_twenty_four import _folds, _gardner_folds

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.engine.moves import MOVE_CASTLE, MOVE_SWEEP, encode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import validate_spec

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, folder: str = "rulesets") -> dict[str, Any]:
    return json.loads((ROOT / folder / f"{name}.json").read_text(encoding="utf-8"))


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def _on(engine: Engine, state: GameState) -> dict[str, tuple[str, int]]:
    """The board by square name: ``{"d5": ("N", 1), ...}``."""
    spec = engine.spec
    names = {_sq(engine, f"{chr(97 + f)}{r + 1}"): f"{chr(97 + f)}{r + 1}"
             for f in range(spec.width) for r in range(spec.height)}
    return {
        names[square]: (spec.piece_ids[spec.type_of(value)], spec.owner_of(value))
        for square, value in enumerate(state.board) if value
    }


def _between(engine: Engine, state: GameState, origin: str, target: str) -> int:
    """The one legal move from ``origin`` to ``target``."""
    wanted = (_sq(engine, origin), _sq(engine, target))
    found = [m for m in engine.legal_moves(state) if (m & 0xFF, (m >> 8) & 0xFF) == wanted]
    assert len(found) == 1, (origin, target, found)
    return found[0]


def _castles(engine: Engine, state: GameState) -> list[str]:
    return sorted(text for text in engine.move_texts(state) if "castle" in text)


# -- a territory colour sized at load (game.rs 686) ----------------------------------------


@cache
def _territory() -> Engine:
    spec = _load("gardner")
    spec["rules"]["territory"] = {"mode": "repaint"}
    return _engine(spec)


def test_mcts_in_a_territory_game_is_the_oracles_move_for_move() -> None:
    """`Game::restore` (line 686) is MCTS's alone: every node it expands restores its
    parent's snapshot, colouring and all. Q752b's Gardner endgame (a king, a queen and a rook
    against a lone king) with territory repainted, played by MCTS against random to the end
    in the core and through the Python oracle (`agents/mcts.py`), identical move for move."""
    spec = _load("gardner")
    spec["setup"] = {"mirror": False, "placements": [
        {"piece": "K", "squares": ["e1"], "player": 0},
        {"piece": "Q", "squares": ["d1"], "player": 0},
        {"piece": "R", "squares": ["a1"], "player": 0},
        {"piece": "K", "squares": ["c5"], "player": 1},
    ]}
    del spec["rules"]["promotion"]
    spec["rules"].update(max_plies=40, territory={"mode": "repaint"})
    _assert_same_played(_engine(spec), ("mcts:nodes=24,rollout_depth=8", "random"), seed=3)


def test_a_territory_game_given_no_colouring_sizes_one() -> None:
    """A territory game's position handed over with no colouring at all (``colour=b""``)
    plays as if nobody owned a square: `from_snapshot` sizes the colouring to the board, and
    the knight's move paints c3 as it does from the sized position."""
    engine = _territory()
    sized = engine.initial_state()
    assert len(sized.colour) == 25 and not any(sized.colour)
    bare = sized.with_(colour=b"")
    move = engine.find_move(sized, "Nb1-c3")
    after = engine.apply(bare, move)
    assert after.colour == engine.apply(sized, move).colour
    assert len(after.colour) == 25
    assert after.colour[_sq(engine, "c3")] == 1 and sum(after.colour) == 1


# -- a wild piece and several royals (game.rs 863) -----------------------------------------


@cache
def _boar_and_two_kings() -> Engine:
    """Chess under checkmate with a second white king placed (so a side may hold several
    royals) and one wild boar walking c3 -> c2 -> c3."""
    spec = _load("chess")
    spec["rules"].pop("castling")
    spec["pieces"].append({"id": "W", "name": "boar", "value": 3,
                           "moves": [{"kind": "leap", "vectors": [[1, 2]],
                                      "symmetry": "dihedral8"}]})
    spec["rules"]["npcs"] = [{"piece": "W", "route": ["c3", "c2"], "mirror": False}]
    spec["setup"]["placements"].append({"piece": "K", "squares": ["d3"]})
    return _engine(spec)


def test_a_king_may_not_step_where_the_boar_eats_it_though_another_king_stands() -> None:
    """Counted by hand: the boar on c3 steps to c2 after White's move, so the king on b1
    stepping to c2 is eaten in its own move's phase. White keeps a second king on f1, so the
    move leaves a royal standing and unattacked; it is refused because it *loses* one of the
    two (`royal_count` before the move, two, against one after). Every other king step
    stands."""
    engine = _boar_and_two_kings()
    assert engine.spec.multi_royal
    state = engine.make_state(
        {"b1": ("K", 0), "f1": ("K", 0), "h8": ("K", 1), "a8": ("K", 1), "c3": ("W", WILD)},
        moved=("c3",),
    )
    kings = sorted(text for text in engine.move_texts(state) if text.startswith("K"))
    assert kings == ["Kb1-a1", "Kb1-a2", "Kb1-b2", "Kb1-c1",
                     "Kf1-e1", "Kf1-e2", "Kf1-f2", "Kf1-g1", "Kf1-g2"]


# -- a sweep's attack past its landing (game.rs 1088-1092) ---------------------------------


@cache
def _floored_sweeper() -> Engine:
    """The sweeping bishops (`archive/chess-sweeping-bishops.json`) on a bare board, the
    bishop sweeping only to capture, from its second square on (``min_range: 2``), and
    unable to take a rook."""
    spec = _load("chess-sweeping-bishops", "archive")
    for rule in ("castling", "en_passant", "promotion"):
        spec["rules"].pop(rule)
    spec["setup"] = {"mirror": False, "placements": [
        {"piece": "K", "squares": ["h1"]}, {"piece": "B", "squares": ["a1"]},
    ]}
    bishop = next(piece for piece in spec["pieces"] if piece["id"] == "B")
    bishop["moves"][0].update(min_range=2, mode="capture")
    bishop["captures"] = ["P", "N", "B", "Q", "K"]
    return _engine(spec)


@pytest.mark.parametrize(("beyond", "check", "sweeps"), [
    # c3 empty (a capture-only landing, passed over), d4 a pawn it may take: check.
    ({"d4": ("P", 1)}, True, ["Ba1xd4 (sweeps b2)"]),
    # Nothing beyond to land on.
    ({}, False, []),
    # A rook it may not take on d4 ends the ray before the pawn on e5.
    ({"d4": ("R", 1), "e5": ("P", 1)}, False, []),
    ({"c3": ("R", 1), "d4": ("P", 1)}, False, []),
])
def test_a_sweep_attacks_the_square_it_may_not_land_on_by_landing_beyond(
    beyond: dict[str, tuple[str, int]], check: bool, sweeps: list[str],
) -> None:
    """Counted by hand: Black's king on b2, one square from White's bishop on a1, is a
    square the bishop may not land on (its floor is two). It is in check exactly when a
    capture landing further along the diagonal would sweep it, and the bishop's sweeps say
    the same."""
    engine = _floored_sweeper()
    placement = {"h1": ("K", 0), "a1": ("B", 0), "b2": ("K", 1), **beyond}
    assert engine.in_check(engine.make_state(placement, side_to_move=1), 1) is check
    white = engine.make_state(placement)
    assert [text for text in engine.move_texts(white) if text.startswith("B")] == sweeps


# -- castling's corners (game.rs 1432, 1443, 3727, 3759-3764, 3793) ------------------------


@cache
def _file_castling(queenside: str, kingside: str) -> Engine:
    spec = _load("chess")
    spec["rules"]["castling"] = {
        "king": "K", "rook": "R", "king_file": {"queenside": queenside, "kingside": kingside},
    }
    return _engine(spec)


ROOKS = {"e1": ("K", 0), "a1": ("R", 0), "h1": ("R", 0), "e8": ("K", 1)}


@pytest.mark.parametrize(("attacker", "castles"), [
    ({}, ["Ke1-c1 (castle)", "Ke1-g1 (castle)"]),
    ({"f8": ("R", 1)}, ["Ke1-c1 (castle)"]),
    ({"d8": ("R", 1)}, ["Ke1-g1 (castle)"]),
])
def test_file_castling_never_crosses_an_attacked_square(
    attacker: dict[str, tuple[str, int]], castles: list[str],
) -> None:
    """Chess960's file castling (king to c1 or g1): a black rook on f8 attacks f1, which the
    king crosses to g1, so only the long castling stands; one on d8 takes the long one."""
    engine = _file_castling("c", "g")
    assert _castles(engine, engine.make_state({**ROOKS, **attacker})) == castles


def test_a_file_castling_whose_rook_lands_off_the_board_is_not_offered() -> None:
    """``king_file.queenside: "h"`` puts the queenside rook (one file past the king, by the
    default) on a ninth file. The validator lets the spelling through; the core offers no
    castling it cannot play, so only the kingside one stands."""
    engine = _file_castling("h", "g")
    assert _castles(engine, engine.make_state(ROOKS)) == ["Ke1-g1 (castle)"]


def test_a_castling_king_never_steps_past_the_board_s_edge() -> None:
    """``king_steps: 3`` from c1: towards h1 the king lands on f1, towards a1 it would land
    on the file before a, so that castling is not offered."""
    spec = _load("chess")
    spec["rules"]["castling"] = {"king": "K", "rook": "R", "king_steps": 3}
    spec["setup"]["placements"] = [
        {"piece": "R", "squares": ["a1", "h1"]}, {"piece": "K", "squares": ["c1"]},
        {"piece": "P", "squares": ["a2", "b2"]},
    ]
    engine = _engine(spec)
    state = engine.make_state({"c1": ("K", 0), "a1": ("R", 0), "h1": ("R", 0), "e8": ("K", 1)})
    assert _castles(engine, state) == ["Kc1-f1 (castle)"]


@pytest.mark.parametrize(("files", "target", "rook_file"), [
    (None, "g1", 0),          # step castling (chess): `castling_partner` walks the rank
    (("c", "g"), "g1", 7),    # file castling: it tries both of the rook's sides
])
def test_a_castling_word_with_no_rook_is_refused(
    files: tuple[str, str] | None, target: str, rook_file: int,
) -> None:
    """A castling word handed to `apply` on a board with no rook to castle with: the core
    finds no partner and refuses it as an ordinary exception, rather than playing a king
    move half of a castling."""
    engine = Engine(_load("chess")) if files is None else _file_castling(*files)
    state = engine.make_state({"e1": ("K", 0), "e8": ("K", 1)})
    word = encode_move(_sq(engine, "e1"), _sq(engine, target), MOVE_CASTLE, rook_file)
    with pytest.raises(ValueError, match="no partner rook"):
        engine.apply(state, word)


# -- circe beside Andernach, the home taken (game.rs 1746-1747) ----------------------------


def test_a_switching_capture_whose_victim_has_no_free_home_still_switches() -> None:
    """Circe with Andernach's switch: White's knight takes Black's pawn on d5. The pawn's
    home, d7, holds a black bishop, so the pawn is gone rather than reborn; the knight still
    changes sides where it landed. Counted by hand."""
    spec = _load("circe", "archive")
    spec["rules"]["strike"]["switches_sides"] = ["P", "N", "B", "R", "Q"]
    engine = _engine(spec)
    state = engine.make_state(
        {"e1": ("K", 0), "e8": ("K", 1), "c3": ("N", 0), "d5": ("P", 1), "d7": ("B", 1)},
        moved=("d5",),
    )
    after = engine.apply(state, _between(engine, state, "c3", "d5"))
    assert _on(engine, after) == {
        "e1": ("K", 0), "e8": ("K", 1), "d5": ("N", 1), "d7": ("B", 1),
    }
    assert after.key == engine._compute_hash(after)


# -- a capture into an unlimited hand slot (game.rs 2296) ----------------------------------


def test_a_capture_of_an_unlimited_stock_leaves_the_hand_alone() -> None:
    """Gardner with drops from captures and an unlimited pawn stock: taking a pawn adds no
    pawn to a hand that never runs out, and taking a knight adds a knight. Counted by hand."""
    spec = _load("gardner")
    spec["rules"]["drops"] = {"from_captures": True, "stock": {"P": "unlimited"}}
    engine = _engine(spec)
    hands = engine.initial_state().hands
    assert hands == ((1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0))
    state = engine.make_state(
        {"a1": ("K", 0), "e5": ("K", 1), "b2": ("N", 0), "c4": ("P", 1), "d3": ("N", 1)},
        moved=("c4",),
    ).with_(hands=hands)
    state = state.with_(key=engine._compute_hash(state))
    assert engine.apply(state, engine.find_move(state, "Nb2xc4")).hands == hands
    assert engine.apply(state, engine.find_move(state, "Nb2xd3")).hands == (
        (1, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0),
    )


# -- attacks through a portal and a fold (game.rs 1227, 1248, 1318, 1332) ------------------


def _no_royal_capture(spec: dict[str, Any], piece: str = "R") -> dict[str, Any]:
    """``spec`` with ``piece`` able to take everything but the king (a capture matrix)."""
    spec = copy.deepcopy(spec)
    for entry in spec["pieces"]:
        if entry["id"] == piece:
            entry["captures"] = [other["id"] for other in spec["pieces"] if other["id"] != "K"]
    return spec


@pytest.mark.parametrize(("matrix", "check"), [(False, True), (True, False)])
def test_a_check_through_a_portal_needs_a_piece_that_may_take_the_royal(
    matrix: bool, check: bool,
) -> None:
    """Wormhole (Gardner, a pair on b3 and d3): Black's rook on b5 goes in at b3 and comes out
    of d3 moving down, onto White's king on d2, so it gives check; a rook that may not take a
    king gives none. The portal oracle agrees."""
    spec = _no_royal_capture(wormhole()) if matrix else wormhole()
    engine = _engine(spec)
    state = engine.make_state(
        {"b3": ("1o", WILD), "d3": ("1b", WILD), "d2": ("K", 0), "e5": ("K", 1),
         "b5": ("R", 1)},
        moved=("b5",),
    )
    assert engine.in_check(state, 0) is check
    assert PortalOracle(spec).attacked_through(
        _position(engine, state), 1, _sq(engine, "d2"), _unmoved(state),
    ) is check


def test_a_portal_whose_far_end_is_missing_leads_nowhere() -> None:
    """A hand-built Wormhole position with the blue end missing: the orange end on b3 leads
    to an end on no square, so nothing goes in at b3 and nothing attacks through it. The
    core answers without walking off its tables, and agrees with the oracle."""
    spec = wormhole()
    engine = _engine(spec)
    state = engine.make_state(
        {"b3": ("1o", WILD), "d2": ("K", 0), "e5": ("K", 1), "b5": ("R", 1)}, moved=("b5",),
    )
    assert not engine.in_check(state, 0)
    assert not PortalOracle(spec).attacked_through(
        _position(engine, state), 1, _sq(engine, "d2"), _unmoved(state),
    )
    rook = sorted(text for text in engine.move_texts(state.with_(side_to_move=1))
                  if text.startswith("R"))
    assert rook == ["Rb5-a5", "Rb5-b4", "Rb5-c5", "Rb5-d5"]


@pytest.mark.parametrize(("spec_of", "check"), [
    (lambda: _gardner_folds(), True),
    # Orange (b3) admits only White's pieces: Black's rook may not go in.
    (lambda: _gardner_folds(passage={"0": "orange_to_blue", "1": "blue_to_orange"}), False),
    # Black's rook may go in, but may not take a king.
    (lambda: _no_royal_capture(_gardner_folds()), False),
], ids=["open", "closed-to-black", "may-not-take-the-king"])
def test_a_check_through_a_fold_needs_an_end_that_admits_a_piece_that_may_take(
    spec_of: Any, check: bool,
) -> None:
    """Gardner's folds on b3 and d3, White's king standing in the d3 fold: Black's rook on b5
    lands on the empty b3 fold and goes through onto d3, so it gives check, unless the b3 end
    admits no black piece or the rook may not take a king. The fold oracle agrees."""
    spec = spec_of()
    engine = _engine(spec)
    state = engine.make_state({"d3": ("K", 0), "e5": ("K", 1), "b5": ("R", 1)}, moved=("b5",))
    assert engine.in_check(state, 0) is check
    assert FoldOracle(spec).attacked(
        _position(engine, state), _folds(engine, state), 1, _sq(engine, "d3"), _unmoved(state),
    ) is check


# -- a round's circe victim with no home (rounds.rs 947) -----------------------------------


@cache
def _circe_rounds_without_a_black_queen() -> Engine:
    """`tests/_simultaneous_circe.json` with Black's queen left out of the setup, so a black
    queen has no home square (`compile._compile_rebirth` writes -1)."""
    spec = _load("_simultaneous_circe", "tests")
    white = [("P", ["a2", "b2", "c2", "d2", "e2"]), ("R", ["a1"]), ("N", ["b1"]),
             ("B", ["c1"]), ("Q", ["d1"]), ("K", ["e1"])]
    black = [("P", ["a4", "b4", "c4", "d4", "e4"]), ("R", ["a5"]), ("N", ["b5"]),
             ("B", ["c5"]), ("K", ["e5"])]
    spec["setup"] = {"mirror": False, "placements": [
        *({"piece": piece, "squares": squares} for piece, squares in white),
        *({"piece": piece, "squares": squares, "player": 1} for piece, squares in black),
    ]}
    return _engine(spec)


def test_a_round_s_circe_victim_with_no_home_is_gone() -> None:
    """Counted by hand and held to `simultaneous_oracle.py`: White's knight takes Black's
    queen on d3 while Black's king steps to e4. A black queen has no home in this game, so it
    is simply gone; a1, empty, stays empty."""
    engine = _circe_rounds_without_a_black_queen()
    state = engine.make_state(
        {"b1": ("K", 0), "c1": ("N", 0), "d3": ("Q", 1), "e5": ("K", 1)}, moved=("d3",),
    )
    first, second = _word(engine, state, 0, "Nc1xd3"), _word(engine, state, 1, "Ke5-e4")
    after = engine.apply_round(state, first, second)
    assert after == resolve_round(engine, state, first, second)
    assert _on(engine, after) == {"b1": ("K", 0), "d3": ("N", 0), "e4": ("K", 1)}


# -- a resting entry on an empty square (rest.rs 248) --------------------------------------


def test_a_resting_entry_on_an_empty_square_is_dropped_at_the_boundary() -> None:
    """Gardner, every piece resting two turns: a position handed over with White resting a
    piece on c3, which is empty, and the knight on b1. The empty square's entry names no
    piece, so it is dropped as the position is read; after Black's move White rests the
    knight alone, and Black the king that moved."""
    spec = _load("gardner")
    for piece in spec["pieces"]:
        piece["rest"] = 2
    engine = _engine(spec)
    state = engine.make_state({"a1": ("K", 0), "e5": ("K", 1), "b1": ("N", 0)}, side_to_move=1)
    state = state.with_(rest=(((_sq(engine, "c3"), 2), (_sq(engine, "b1"), 2)), ()))
    state = state.with_(key=engine._compute_hash(state))
    after = engine.apply(state, engine.find_move(state, "Ke5-d5"))
    assert after.rest == (((_sq(engine, "b1"), 2),), ((_sq(engine, "d5"), 2),))


@pytest.mark.parametrize("entry", [(-1, 1), (25, 1), ("a1", 0), ("a1", 3), "third"])
def test_a_resting_entry_the_rule_cannot_hold_is_dropped_at_the_boundary(entry: Any) -> None:
    """`rest.rs` 248 (`rest_lists_from`). Gardner, every piece resting two turns, so a list
    holds two entries of at most two turns: White hands over its knight on b1 resting, after
    an entry off the board (either side), one resting no turn, one resting longer than any
    piece rests, or (``third``) a third entry past the two the list holds. Each is dropped
    as the position is read: White's moves, what the rest takes and the key are those of the
    position with the knight alone resting (the king keeps its moves, the knight has none)."""
    spec = _load("gardner")
    for piece in spec["pieces"]:
        piece["rest"] = 2
    engine = _engine(spec)
    state = engine.make_state({"a1": ("K", 0), "e5": ("K", 1), "b1": ("N", 0), "c1": ("B", 0)})
    knight = (_sq(engine, "b1"), 2)
    given: tuple[tuple[int, int], ...]
    clean: tuple[tuple[int, int], ...]
    if entry == "third":
        given = (knight, (_sq(engine, "c1"), 1), (_sq(engine, "a1"), 1))
        clean = (knight, (_sq(engine, "c1"), 1))
    else:
        square = _sq(engine, entry[0]) if isinstance(entry[0], str) else entry[0]
        given, clean = ((square, entry[1]), knight), (knight,)
    told = state.with_(rest=(given, ()))
    kept = state.with_(rest=(clean, ()))
    assert engine.core.compute_hash(_pack(told)) == engine.core.compute_hash(_pack(kept))
    assert engine.legal_moves(told) == engine.legal_moves(kept)
    assert engine.rest_turn(told, 0) == engine.rest_turn(kept, 0)
    moved = {engine.describe_move(told, m)[:3] for m in engine.legal_moves(told)}
    assert "Ka1" in moved and not any(text.startswith("N") for text in moved)


# -- a sweep word no sweep makes (tables.rs `sweep_path`'s two empty answers) ---------------
#
# Until CS6 these walked `sweep_path`'s fallbacks by handing `apply` a forged sweep word. Since
# CS6 (R12) `apply` refuses any word the position does not offer, so a forged word never
# reaches `make` and those two lines are reachable from no caller; the tests hold the refusal.


def test_a_sweep_word_in_a_game_without_sweeps_is_refused() -> None:
    """A sweep word handed to chess, which has no sweep table: refused, not played."""
    engine = Engine(_load("chess"))
    state = engine.initial_state()
    forged = encode_move(_sq(engine, "e2"), _sq(engine, "e4"), MOVE_SWEEP)
    with pytest.raises(ValueError, match="not a move this position offers"):
        engine.apply(state, forged)


def test_a_sweep_word_to_a_square_no_sweep_reaches_is_refused() -> None:
    """In the sweeping bishops, a sweep word from the bishop on c1 to d3, a square none of its
    rays reach: refused; the real sweeps through d2 are offered."""
    engine = Engine(_load("chess-sweeping-bishops", "archive"))
    state = engine.make_state({"e1": ("K", 0), "e8": ("K", 1), "c1": ("B", 0), "d2": ("P", 0)})
    forged = encode_move(_sq(engine, "c1"), _sq(engine, "d3"), MOVE_SWEEP)
    with pytest.raises(ValueError, match="not a move this position offers"):
        engine.apply(state, forged)
    assert "Bc1xe3 (sweeps d2)" in engine.move_texts(state)
