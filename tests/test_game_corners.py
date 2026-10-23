"""F10b group 3 (Q752b): `rust/src/game.rs`'s corners no test reached, held to an oracle or a
hand count.

`docs/plan-tests.md` F10b. First a fold's own move (`make_fold`, schema 24) beside the rules
whose bookkeeping it must keep though it moves no piece: a check limit (one royal and several),
a goal left, a rest, a long turn and the straight-back bar. Then the corners of the move
generator and its filters that no committed game reaches. Each case is a Gardner test spec
(`test_schema_twenty_four._gardner_folds`: a king and a pawn a side and a fold pair on b3 and
d3) with the one rule added, played from a hand-built position; what the move must leave is
counted by hand and, where it is an attack, asked of `tests/fold_oracle.py` too.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from fold_oracle import FoldOracle
from rest_oracle import apply as rest_apply
from rest_oracle import as_rest, legal
from test_schema_nineteen import _position, _sparse, _unmoved, _with_portals
from test_schema_twenty_four import _folds, _gardner_folds

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.engine.moves import MOVE_EN_PASSANT, MOVE_FOLD, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import validate_spec

ROOT = Path(__file__).resolve().parents[1]
BOTH ={"moved_by": {"orange": [0, 1], "blue": [0, 1]}}


def _folded(pair: dict[str, Any] | None = None, kings: tuple[str, ...] = (),
            rest: bool = False, **rules: Any) -> tuple[Engine, dict[str, Any]]:
    """Gardner's folds (`_gardner_folds`, either player moving either end unless ``pair``
    says otherwise) with ``rules`` added, ``kings`` more royal kings in the setup, and every
    piece resting one turn after it moves when ``rest``."""
    spec = _gardner_folds(**(pair if pair is not None else BOTH))
    spec["rules"].update(rules)
    for square in kings:
        spec["setup"]["placements"].append({"piece": "K", "squares": [square]})
    if rest:
        for piece in spec["pieces"]:
            piece["rest"] = 1
    assert validate_spec(spec) == []
    return Engine(spec), spec


def _mid_game(engine: Engine, state: GameState) -> GameState:
    """``state`` past the opening turn of a two-move-turn game (whose first turn is one
    move), White to make both moves of a turn, with its key made again."""
    state = state.with_(remaining=2, turns_taken=1)
    return state.with_(key=engine._compute_hash(state))


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


# -- a fold's own move -------------------------------------------------------------------


def test_a_fold_move_that_uncovers_a_check_counts_it() -> None:
    """Counted by hand under a check limit: White moves the fold end on d3 under Black's king
    on d4, so White's rook on b1, landing on the near end b3, would go through and take the
    king: the fold move delivered a check, though it moved no piece. The oracle agrees the
    king is attacked."""
    engine, spec = _folded(check_limit=3)
    state = engine.make_state({"e1": ("K", 0), "d4": ("K", 1), "b1": ("R", 0)})
    after = engine.apply(state, engine.find_move(state, "1b:d3-d4"))
    assert after.checks == (1, 0)
    assert FoldOracle(spec).attacked(
        _position(engine, after), _folds(engine, after), 0, _sq(engine, "d4"), _unmoved(after),
    )
    quiet = engine.apply(state, engine.find_move(state, "1b:d3-c3"))
    assert quiet.checks == (0, 0)
    assert after.key == engine._compute_hash(after)


def test_a_fold_move_checks_one_of_several_royals() -> None:
    """The same with two black kings (a side with several royals): the fold goes under the
    one on d4, the other on a5 stays safe, and the check is counted once."""
    engine, _ = _folded(kings=("c1",), check_limit=3)
    state = engine.make_state(
        {"e1": ("K", 0), "c1": ("K", 0), "d4": ("K", 1), "a5": ("K", 1), "b1": ("R", 0)},
    )
    after = engine.apply(state, engine.find_move(state, "1b:d3-d4"))
    assert after.checks == (1, 0)
    quiet = engine.apply(state, engine.find_move(state, "1b:d3-c3"))
    assert quiet.checks == (0, 0)


def test_a_fold_move_after_a_goal_left_clears_it() -> None:
    """`reach.on: "leave"` (a rook leaving its corner): White's rook leaves a1, so White has
    left a goal and the game is decided. A fold move played after it leaves nothing: the goal
    was left by the move before, so the position the fold move makes is no longer decided."""
    engine, _ = _folded(reach={"piece": "R", "squares": ["a1"], "mirror": True, "result": "win",
                            "on": "leave"})
    state = engine.make_state({"e1": ("K", 0), "e5": ("K", 1), "a1": ("R", 0)})
    left = engine.apply(state, engine.find_move(state, "Ra1-a2"))
    assert left.goal_left_by == 0
    verdict = engine.terminal(left)
    assert verdict is not None and (verdict.winner, verdict.reason) == (0, "reached")
    after = engine.apply(left, engine.find_move(left, "1o:b3-b4"))
    assert after.goal_left_by == -1
    assert engine.terminal(after) is None
    assert after.key == engine._compute_hash(after)


def test_a_fold_move_counts_the_rests_down() -> None:
    """Every piece rests one turn after it moves: seeded walks that play a fold's own move
    whenever one is offered half the time, the rest lists and the legal lists held to
    `tests/rest_oracle.py` at every ply, and some fold move made while a piece rested."""
    engine, _ = _folded(rest=True)
    start = engine.make_state({
        "e1": ("K", 0), "e5": ("K", 1), "a1": ("R", 0), "a5": ("R", 1), "c1": ("N", 0),
        "c5": ("N", 1),
    })
    counted = 0
    for seed in range(4):
        rng = random.Random(seed)
        state = start
        for _ in range(30):
            if engine.terminal(state) is not None:
                break
            moves = engine.legal_moves(state)
            assert moves == legal(engine, state)
            folds = [m for m in moves if decode_move(m)[2] == MOVE_FOLD]
            move = rng.choice(folds if folds and rng.random() < 1 / 2 else moves)
            counted += decode_move(move)[2] == MOVE_FOLD and any(state.rest)
            after, lists = rest_apply(engine, state, move)
            assert after.rest == as_rest(lists)
            assert after.key == engine._compute_hash(after)
            state = after
    assert counted > 0


def test_a_fold_move_in_a_long_turn_keeps_the_turn() -> None:
    """Two moves a turn: a fold move first leaves White to move with one move left; a second
    passes the turn. The key is the slow definition's each time."""
    engine, _ = _folded(turn={"moves": 2})
    state = _mid_game(engine, engine.make_state({"e1": ("K", 0), "e5": ("K", 1)}))
    first = engine.apply(state, engine.find_move(state, "1o:b3-b4"))
    assert (first.side_to_move, first.remaining) == (0, 1)
    assert first.key == engine._compute_hash(first)
    second = engine.apply(first, engine.find_move(first, "1b:d3-d4"))
    assert second.side_to_move == 1
    assert second.key == engine._compute_hash(second)


def test_the_straight_back_bar_is_set_only_for_whoever_can_move_the_end() -> None:
    """`no_straight_back` on a fold only White moves, in two-move turns: White's fold move
    as the first move of its turn bars the way straight back for White's second; as the last
    move it bars nothing, since Black moves next and Black never moves the end."""
    engine, _ = _folded(pair={"moved_by": {"orange": [0]}, "no_straight_back": True},
                        turn={"moves": 2})
    state = _mid_game(engine, engine.make_state({"e1": ("K", 0), "e5": ("K", 1)}))
    first = engine.apply(state, engine.find_move(state, "1o:b3-b4"))
    assert first.portal_bar == (_sq(engine, "b4"), _sq(engine, "b3"))
    assert "1o:b4-b3" not in engine.move_texts(first)
    last = engine.apply(first, engine.find_move(first, "1o:b4-c4"))
    assert last.portal_bar == (-1, -1)
    assert last.key == engine._compute_hash(last)


# -- the generator, the filters and the verdict on rarer rule pairs -------------------------


def _gardner(**rules: Any) -> dict[str, Any]:
    spec = json.loads((ROOT / "rulesets" / "gardner.json").read_text(encoding="utf-8"))
    spec["name"] = "Gardner (F10b test spec)"
    spec["rules"].update(rules)
    return spec


def _two_kings(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with a second king a side in the setup (on c1 and c5, the bishop's squares),
    so a side has several royals and every royal reading is the several-royal one."""
    placements = [p for p in spec["setup"]["placements"] if p["piece"] != "B"]
    for placement in placements:
        if placement["piece"] == "K":
            placement["squares"] = ["e1", "c1"]
    spec["setup"]["placements"] = placements
    return spec


def test_a_pawn_drop_that_forks_two_kings_into_mate_is_refused() -> None:
    """Counted by hand, two kings a side and shogi's rule that a pawn drop may not mate: a
    White pawn dropped on d3 attacks both Black kings (c4 and e4); either taking it stands
    beside White's king on d2, and no other move answers both checks, so the drop mates and
    is refused. The same game without the rule offers it, and nothing else differs."""
    placement = {"d2": ("K", 0), "a5": ("K", 0), "c4": ("K", 1), "e4": ("K", 1),
                 "c1": ("B", 0), "d5": ("P", 1)}
    texts = {}
    for no_mate in (True, False):
        drops: dict[str, Any] = {"from_captures": True}
        if no_mate:
            drops["no_mate"] = ["P"]
        engine = Engine(_two_kings(_gardner(drops=drops)))
        pawn = engine.spec.piece_ids.index("P")
        hand = tuple(int(piece_type == pawn) for piece_type in range(engine.spec.n_types))
        state = engine.make_state(placement).with_(hands=(hand, (0,) * len(hand)))
        texts[no_mate] = set(engine.move_texts(state))
    assert texts[False] - texts[True] == {"P@d3"}


def test_a_check_against_one_of_two_kings_is_counted_and_forbidden() -> None:
    """Two kings a side. Under a check limit, White's rook move that attacks one Black king
    counts one check, and a quiet one none; under racing kings' rule (`forbid_check`) the
    checking move is refused, though the generator offers it."""
    placement = {"e1": ("K", 0), "c1": ("K", 0), "a5": ("K", 1), "e5": ("K", 1),
                 "b2": ("R", 0)}
    counted = Engine(_two_kings(_gardner(check_limit=3)))
    state = counted.make_state(placement)
    after = counted.apply(state, counted.find_move(state, "Rb2-b5"))
    assert after.checks == (1, 0)
    quiet = counted.apply(state, counted.find_move(state, "Rb2-b3"))
    assert quiet.checks == (0, 0)
    racing = Engine(_two_kings(_gardner(forbid_check=True)))
    state = racing.make_state(placement)
    texts = racing.move_texts(state)
    assert "Rb2-b5" not in texts and "Rb2-b3" in texts
    assert "Rb2-b5" in {racing.describe_move(state, m) for m in racing.pseudo_legal_moves(state)}


def test_a_fold_move_that_gives_check_is_refused_under_forbid_check() -> None:
    """Racing kings' rule beside folds: moving the end on d3 under Black's king on d4 lets
    White's rook through to it, so that fold move gives check and is refused; moving it to
    c3 gives none and stays, and the trial leaves the folds where they stood."""
    engine, _ = _folded(forbid_check=True)
    state = engine.make_state({"e1": ("K", 0), "d4": ("K", 1), "b1": ("R", 0)})
    texts = engine.move_texts(state)
    assert "1b:d3-d4" not in texts and "1b:d3-c3" in texts
    assert "Rb1-b3~d3" not in texts and "Rb1-b2" in texts  # through d3, beside the king
    assert state.folds == (_sq(engine, "b3"), _sq(engine, "d3"))


def test_a_fold_move_strands_nothing_under_forbid_immobility() -> None:
    """Shogi's rule that no move may strand a piece reads a fold's own move as moving no
    piece: every fold move the generator offers is legal."""
    engine, _ = _folded(forbid_immobility=True)
    state = engine.make_state({"e1": ("K", 0), "e5": ("K", 1), "a2": ("P", 0)})
    folds = [m for m in engine.pseudo_legal_moves(state) if decode_move(m)[2] == MOVE_FOLD]
    assert folds
    assert set(folds) <= set(engine.legal_moves(state))


def test_a_promoting_transit_is_not_stranding() -> None:
    """A pawn pushed into the end on b3 comes out of the end on d4 one step up, on d5, the
    last rank, where it must promote. Under `forbid_immobility` an unpromoted pawn there
    could never move, but the transit carries its promotion, which is read as the piece it
    becomes: all four promotions stay legal. Counted by hand."""
    spec = _sparse(_with_portals("gardner", [{"orange": "b3", "blue": "d4"}]))
    spec["rules"]["forbid_immobility"] = True
    engine = Engine(spec)
    state = engine.make_state({
        "e1": ("K", 0), "a5": ("K", 1), "b2": ("P", 0), "b3": ("1o", WILD), "d4": ("1b", WILD),
    })
    texts = set(engine.move_texts(state))
    assert {f"Pb2-b3>d5={piece}" for piece in "QRBN"} <= texts, texts


def test_a_sliding_capture_takes_en_passant() -> None:
    """Counted by hand: pawns that step twice from their first square and capture by a
    one-square diagonal *slide* (where chess writes a leap). Black's pawn steps d5-d3, opening
    the window on d4; White's pawn on c3 slides onto d4, taking it en passant, and the
    window's victim on d3 leaves the board."""
    spec = _gardner(en_passant=True)
    spec["board"]["height"] = 6
    spec["pieces"][0]["moves"] = [
        {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
        {"kind": "slide", "vectors": [[0, 1]], "range": 2, "mode": "move",
         "condition": {"first_move": True}},
        {"kind": "slide", "vectors": [[1, 1]], "symmetry": "lateral", "range": 1,
         "mode": "capture"},
    ]
    engine = Engine(spec)
    state = engine.make_state(
        {"a1": ("K", 0), "a6": ("K", 1), "c3": ("P", 0), "d5": ("P", 1)}, side_to_move=1,
    )
    state = engine.apply(state, engine.find_move(state, "Pd5-d3"))
    taking = [m for m in engine.legal_moves(state) if decode_move(m)[:3] == (
        _sq(engine, "c3"), _sq(engine, "d4"), MOVE_EN_PASSANT)]
    assert len(taking) == 1
    after = engine.apply(state, taking[0])
    assert after.board[_sq(engine, "d3")] == 0
    assert after.board[_sq(engine, "d4")] == state.board[_sq(engine, "c3")]


def test_a_first_move_sweep_is_gone_once_the_piece_has_moved() -> None:
    """A rook that may also sweep two squares diagonally while unmoved: offered from its
    first square, gone once it has moved (the same square, marked moved)."""
    spec = _gardner()
    rook = next(piece for piece in spec["pieces"] if piece["id"] == "R")
    rook["moves"].append({"kind": "sweep", "vectors": [[1, 1]], "symmetry": "rotate4",
                          "range": 2, "condition": {"first_move": True}})
    engine = Engine(spec)
    placement = {"e1": ("K", 0), "e5": ("K", 1), "c3": ("R", 0)}
    fresh = set(engine.move_texts(engine.make_state(placement)))
    moved = set(engine.move_texts(engine.make_state(placement, moved=("c3",))))
    assert "Rc3-a5" in fresh and "Rc3-a5" not in moved
    assert moved < fresh


def test_territory_short_of_its_target_decides_nothing() -> None:
    """`territory.win_at` 20, counted by hand: nineteen squares coloured White's decide
    nothing, twenty win. Without a `win_at` (a colouring scored, beside a check limit that
    makes the game read its endings) the whole board coloured decides nothing."""
    engine = Engine(_gardner(territory={"mode": "permanent", "win_at": 20}))
    state = engine.make_state({"e1": ("K", 0), "e5": ("K", 1)})
    for coloured, verdict in ((0, None), (19, None), (20, (0, "coloured"))):
        painted = state.with_(colour=bytes([1] * coloured + [0] * (25 - coloured)))
        outcome = engine.terminal(painted)
        assert (None if outcome is None else (outcome.winner, outcome.reason)) == verdict
    unaimed = Engine(_gardner(territory={"mode": "permanent"}, scoring={"count": "territory"},
                              check_limit=3))
    state = unaimed.make_state({"e1": ("K", 0), "e5": ("K", 1)})
    assert unaimed.terminal(state.with_(colour=bytes([1] * 25))) is None


def test_a_missing_royal_never_faces_the_other() -> None:
    """`royal_facing: "forbidden"` (xiangqi's flying general) with Black's king gone (a
    hand-built position): no two royals face, so nothing is refused for it and White's king
    may step onto Black's back file as freely as anywhere."""
    engine = Engine(_gardner(royal_facing="forbidden"))
    alone = engine.make_state({"d1": ("K", 0), "a5": ("R", 1)})
    faced = engine.make_state({"d1": ("K", 0), "a5": ("R", 1), "e5": ("K", 1)})
    assert "Kd1-e1" in engine.move_texts(alone)
    assert "Kd1-e1" not in engine.move_texts(faced)


def _apply_count(engine: Engine, state: GameState, depth: int) -> int:
    """Perft by `Engine.apply`, a fresh position each move, so no move is ever unmade."""
    if depth == 0:
        return 1
    return sum(_apply_count(engine, engine.apply(state, move), depth - 1)
               for move in engine.legal_moves(state))


def test_unmaking_a_fold_move_restores_the_goal_and_the_rests() -> None:
    """The core's perft makes and unmakes every move; counting by `apply` makes each from a
    copy. They agree three plies deep, fold moves among them, beside a goal left (whose
    record a fold move clears and its unmaking puts back) and beside rests."""
    for rules in (
        {"reach": {"piece": "R", "squares": ["a1"], "mirror": True, "result": "win",
                   "on": "leave"}},
        {"rest": True},
    ):
        engine, _ = _folded(**rules)
        state = engine.make_state({"e1": ("K", 0), "e5": ("K", 1), "a1": ("R", 0),
                                   "a5": ("R", 1)})
        assert engine.perft(state, 3) == _apply_count(engine, state, 3)


def test_per_type_extinction_reads_the_side_in_mid_turn_first() -> None:
    """Counted by hand, two moves a turn and each of knight and bishop counted on its own:
    both sides are short of a knight. In the middle of White's turn (one of its two moves
    made) White made the last move, so White is read first and loses; at the turn's end
    Black made the last move and loses."""
    engine = Engine(_gardner(
        extinction={"pieces": ["N", "B"], "min_count": 1, "result": "loss", "per_type": True},
        turn={"moves": 2},
    ))
    state = engine.make_state({"e1": ("K", 0), "e5": ("K", 1), "c1": ("B", 0), "c5": ("B", 1)})
    for remaining, loser in ((1, 0), (2, 1)):
        outcome = engine.terminal(state.with_(remaining=remaining, turns_taken=1))
        assert outcome is not None
        assert (outcome.winner, outcome.reason) == (1 - loser, "extinction")


def test_own_kinds_never_count_against_a_side_that_started_with_none() -> None:
    """`per_type: "own"`: White starts with a knight and a bishop, Black with neither, so
    Black's list is empty and Black is never short. With White's knight gone White is short
    and loses, whichever side is read first."""
    spec = _gardner(extinction={"pieces": ["N", "B"], "min_count": 1, "result": "loss",
                                "per_type": "own"})
    spec["setup"] = {"mirror": False, "placements": [
        {"piece": "K", "squares": ["e1"], "player": 0},
        {"piece": "N", "squares": ["b1"], "player": 0},
        {"piece": "B", "squares": ["c1"], "player": 0},
        {"piece": "P", "squares": ["a2", "b2"], "player": 0},
        {"piece": "K", "squares": ["e5"], "player": 1},
        {"piece": "R", "squares": ["a5"], "player": 1},
        {"piece": "P", "squares": ["a4", "b4"], "player": 1},
    ]}
    assert validate_spec(spec) == []
    engine = Engine(spec)
    assert engine.terminal(engine.initial_state()) is None
    for side in (0, 1):
        state = engine.make_state(
            {"e1": ("K", 0), "c1": ("B", 0), "e5": ("K", 1), "a5": ("R", 1)},
            side_to_move=side, ply=3,
        )
        outcome = engine.terminal(state)
        assert outcome is not None and (outcome.winner, outcome.reason) == (1, "extinction")


def test_perft_s_depth_zero_and_a_stopped_count() -> None:
    """Perft counts the position itself at depth zero, stopped or not; and a count that
    stops at decided positions is the same on two threads as on one, and smaller than the
    count that plays on (a game a single check decides)."""
    engine = Engine(_gardner(check_limit=1))
    start = engine.initial_state()
    assert engine.perft(start, 0) == 1 == engine.perft(start, 0, stop_at_terminal=True)
    stopped = engine.perft(start, 4, stop_at_terminal=True)
    assert engine.perft(start, 4, threads=2, stop_at_terminal=True) == stopped
    assert stopped < engine.perft(start, 4)

