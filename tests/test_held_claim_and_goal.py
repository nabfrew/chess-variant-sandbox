"""SR5k (`docs/plan-schema-next.md` §2.11 rows O10, O11): a held piece claims and arrives for
the mover.

The rulings (2026-10-28): a held piece (nobody's, moved by the player with the majority beside
it) claims territory for the player who moved it (O10); its arrival wins for that player, read
by the goal's own rule (`reach.piece`, `on`, `safe`) as if the mover owned the piece (O11).

The probe (both pairs fed past the validator on SR5i's build `dcf708e41a22ce12`; a 6x6 board, a
condottiere nobody owns held through two guards): a claim already went to the mover, and a
departure under `on: "leave"` already won for it, since SR5d's `Game::arrival_owner` reads a
nobody's piece as the mover's. An arrival did not: the ending's reach arm scans the two players'
pieces on the board, and the piece standing on the goal is nobody's, so a held arrival decided
nothing, `safe` or not. So `make` reads a held arrival off the move (`Game::held_arrival`) into
the goal field `on: "leave"` already keeps (`goal_left_by`, behind `Tables::goal_event`), the
goal's `piece` and `safe` read as for an owned arrival, and the ending decides it in reach's
slot.

Readings the rows left open, settled on the plan's principles: the arrival is the move's, not
the board's -- the piece standing on the goal afterwards is nobody's and decides nothing
(`control`: moving it does not make it yours), so an arrival `safe` refuses never wins later;
onto the other player's goal it decides nothing (a goal is its owner's, and the mover owns the
piece for the move); a held piece taken off the board by its own capture's blast arrives nowhere.

Hand-built 6x6 specs, not chess. The walks check the incremental key against `_compute_hash`
each ply, and `perft` against an `apply` count puts every colour and goal field back.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import MOVE_FOREIGN, move_from, move_to
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_eighty_nine_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
# The first player's goal, written for it; the second player's is its mirror, d3.
GOAL = ("d4", "d3")
# The squares the mover's two guards stand on, next to the condottiere on c3.
GUARDS = (("b2", "b3"), ("b4", "c4"))


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _name(square: int) -> str:
    return "abcdef"[square % WIDTH] + str(square // WIDTH + 1)


def _spec(**rules: Any) -> dict[str, Any]:
    return copy.deepcopy({
        "name": "the held claim (SR5k test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "N", "name": "guard", "value": 3, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "C", "name": "condottiere", "value": 3, "moves": [KING],
             "control": "majority"},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "N", "squares": ["b1"]},
            {"piece": "C", "player": "nobody", "squares": ["c3"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 80, **rules},
    })


def _goal(**extra: Any) -> dict[str, Any]:
    return _spec(reach={"squares": [GOAL[0]], **extra})


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _position(
    engine: Engine, side: int, held_on: str = "c3", guards: tuple[str, str] | None = None,
    **extra: tuple[str, int],
) -> GameState:
    """The condottiere on ``held_on``, held by ``side`` through two guards; the kings on a1
    and f6."""
    first, second = guards or GUARDS[side]
    placement = {
        first: ("N", side), second: ("N", side), held_on: ("C", WILD),
        "a1": ("K", 0), "f6": ("K", 1),
    }
    placement.update(extra)
    return engine.make_state(placement, side_to_move=side, moved=list(placement))


def _held(engine: Engine, state: GameState, origin: str, target: str) -> int:
    (move,) = [
        m for m in engine.legal_moves(state)
        if (move_from(m), move_to(m)) == (_sq(origin), _sq(target))
    ]
    assert move & MOVE_FOREIGN
    return move


def _claimed(state: GameState) -> dict[str, int]:
    return {_name(square): colour for square, colour in enumerate(state.colour) if colour}


# --------------------------------------------------------------------------- O10: the claim


class TestAHeldMoveClaimsForTheMover:
    @pytest.mark.parametrize("mode", ["permanent", "repaint"])
    @pytest.mark.parametrize("side", [0, 1])
    def test_the_square_it_lands_on(self, side: int, mode: str) -> None:
        engine = _engine(_spec(territory={"mode": mode}))
        state = _position(engine, side)
        target = "c4" if side == 0 else "d2"
        after = engine.apply(state, _held(engine, state, "c3", target))
        assert _claimed(after) == {target: side + 1}
        assert after.key == engine._compute_hash(after)

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_repainting_claim_takes_the_other_players_square(self, side: int) -> None:
        engine = _engine(_spec(territory={"mode": "repaint"}))
        target = "d3"
        state = _position(engine, side)
        state = state.with_(colour=bytes(2 - side if i == _sq(target) else 0
                                         for i in range(WIDTH * WIDTH)))
        state = state.with_(key=engine._compute_hash(state))
        after = engine.apply(state, _held(engine, state, "c3", target))
        assert _claimed(after) == {target: side + 1}
        assert after.key == engine._compute_hash(after)

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_claim_counts_toward_the_win(self, side: int) -> None:
        engine = _engine(_spec(territory={"mode": "permanent", "win_at": 1}))
        state = _position(engine, side)
        after = engine.apply(state, _held(engine, state, "c3", "d3"))
        outcome = engine.terminal(after)
        assert outcome is not None and outcome.winner == side


# --------------------------------------------------------------------------- O11: the goal


class TestAHeldArrivalIsTheMovers:
    @pytest.mark.parametrize("side", [0, 1])
    def test_onto_the_movers_goal_it_wins(self, side: int) -> None:
        engine = _engine(_goal())
        state = _position(engine, side)
        after = engine.apply(state, _held(engine, state, "c3", GOAL[side]))
        outcome = engine.terminal(after)
        assert outcome is not None and (outcome.winner, outcome.reason) == (side, "reached")

    @pytest.mark.parametrize("side", [0, 1])
    def test_onto_the_other_players_goal_it_decides_nothing(self, side: int) -> None:
        engine = _engine(_goal())
        state = _position(engine, side)
        after = engine.apply(state, _held(engine, state, "c3", GOAL[1 - side]))
        assert engine.terminal(after) is None

    @pytest.mark.parametrize("named, wins", [("C", True), ("N", False)])
    def test_the_goals_piece_is_read(self, named: str, wins: bool) -> None:
        engine = _engine(_goal(piece=named))
        state = _position(engine, 0)
        outcome = engine.terminal(engine.apply(state, _held(engine, state, "c3", GOAL[0])))
        assert (outcome is not None) is wins

    @pytest.mark.parametrize("side", [0, 1])
    def test_under_safe_a_threatened_arrival_decides_nothing_ever(self, side: int) -> None:
        """The other player's rook bears on the goal: no win, and none later when the rook
        moves away, since the piece standing there is nobody's."""
        engine = _engine(_goal(safe=True))
        rook = "d6" if side == 0 else "d1"
        state = _position(engine, side, **{rook: ("R", 1 - side)})
        after = engine.apply(state, _held(engine, state, "c3", GOAL[side]))
        assert engine.terminal(after) is None
        away = "e6" if side == 0 else "e1"
        later = engine.apply(after, next(
            m for m in engine.legal_moves(after)
            if (move_from(m), move_to(m)) == (_sq(rook), _sq(away))
        ))
        assert engine.terminal(later) is None

    @pytest.mark.parametrize("side", [0, 1])
    def test_under_safe_an_unthreatened_arrival_wins(self, side: int) -> None:
        engine = _engine(_goal(safe=True))
        state = _position(engine, side)
        outcome = engine.terminal(engine.apply(state, _held(engine, state, "c3", GOAL[side])))
        assert outcome is not None and outcome.winner == side

    @pytest.mark.parametrize("side", [0, 1])
    def test_leaving_the_movers_goal_wins(self, side: int) -> None:
        engine = _engine(_goal(on="leave"))
        guards = ("c3", "c4") if side == 0 else ("c2", "c3")
        state = _position(engine, side, held_on=GOAL[side], guards=guards)
        off = "e4" if side == 0 else "e3"
        outcome = engine.terminal(engine.apply(state, _held(engine, state, GOAL[side], off)))
        assert outcome is not None and (outcome.winner, outcome.reason) == (side, "reached")

    @pytest.mark.parametrize("side", [0, 1])
    def test_leaving_the_other_players_goal_decides_nothing(self, side: int) -> None:
        engine = _engine(_goal(on="leave"))
        guards = ("c2", "c3") if side == 0 else ("c3", "c4")
        state = _position(engine, side, held_on=GOAL[1 - side], guards=guards)
        off = "e3" if side == 0 else "e4"
        assert engine.terminal(engine.apply(state, _held(engine, state, GOAL[1 - side], off))) is None

    def test_an_own_arrival_still_wins(self) -> None:
        """The first player's own guard onto its goal: the board's arrival, unchanged."""
        engine = _engine(_goal())
        state = _position(engine, 0, d5=("N", 0))
        move = next(m for m in engine.legal_moves(state)
                    if (move_from(m), move_to(m)) == (_sq("d5"), _sq("d4")))
        outcome = engine.terminal(engine.apply(state, move))
        assert outcome is not None and outcome.winner == 0


# --------------------------------------------------------------------------- the lift


class TestTheLift:
    @pytest.mark.parametrize("rules", [
        {"territory": {"mode": "repaint"}},
        {"reach": {"squares": ["d4"]}},
        {"reach": {"squares": ["d4"], "on": "leave"}},
        {"reach": {"squares": ["d4"], "safe": True}},
    ])
    def test_the_pairs_are_valid(self, rules: dict[str, Any]) -> None:
        errors = validate_spec(_spec(**rules))
        assert errors == []
        assert not {"O10", "O11"} & set(refused_ids(errors))

    def test_the_inventory_names_them(self) -> None:
        both = _spec(territory={"mode": "repaint"}, reach={"squares": ["d4"]})
        assert _schema_eighty_nine_uses(both) == {
            "pieces.control.majority.territory", "pieces.control.majority.reach",
        }
        del both["pieces"][3]["control"]
        assert _schema_eighty_nine_uses(both) == set()

    def test_describe_says_it(self) -> None:
        text = describe_spec(_spec(territory={"mode": "repaint"}, reach={"squares": ["d4"],
                                                                         "safe": True}))
        assert "A held piece claims the square it lands on for the player who moved it" in text
        assert "moving it onto that player's goal wins for that player, if the other" in text
        leave = describe_spec(_goal(on="leave"))
        assert "taking it off that player's goal wins for that player" in leave
        assert "A held piece" not in describe_spec(_spec())

    def test_fairy_stockfish_refuses_it(self) -> None:
        with pytest.raises(Untranslatable):
            translate(_goal())


# --------------------------------------------------------------------------- the walks


_WALKED = {
    "territory": {"territory": {"mode": "repaint"}},
    "arrive": {"reach": {"squares": ["d4", "e4"], "safe": True}},
    "leave": {"reach": {"squares": ["d4", "e4"], "on": "leave"}},
}


def _busy(engine: Engine) -> GameState:
    placement = {
        "a1": ("K", 0), "f6": ("K", 1), "b2": ("N", 0), "e5": ("N", 1), "b5": ("R", 1),
        "e2": ("R", 0), "c3": ("C", WILD), "d4": ("C", WILD),
    }
    return engine.make_state(placement, moved=list(placement))


class TestTheWalks:
    @pytest.mark.parametrize("rules", sorted(_WALKED))
    def test_walks_keep_the_key(self, rules: str) -> None:
        engine = _engine(_spec(**_WALKED[rules]))
        held = 0
        for seed in range(8):
            rng = random.Random(seed)
            state = _busy(engine)
            for _ in range(60):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                foreign = [m for m in moves if m & MOVE_FOREIGN]
                move = rng.choice(foreign if foreign and rng.random() < 0.6 else moves)
                held += bool(move & MOVE_FOREIGN)
                after = engine.apply(state, move)
                assert after.key == engine._compute_hash(after), (rules, seed)
                state = after
        assert held > 0, rules

    @pytest.mark.parametrize("rules", sorted(_WALKED))
    def test_perft_counts_what_apply_counts(self, rules: str) -> None:
        engine = _engine(_spec(**_WALKED[rules]))
        state = _busy(engine)

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(
                count(engine.apply(position, m), depth - 1) for m in engine.legal_moves(position)
            )

        assert engine.perft(state, 3) == count(state, 3)
