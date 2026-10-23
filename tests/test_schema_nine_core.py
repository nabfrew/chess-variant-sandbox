"""Schema 9 in the core: the phase, the third owner, the tally, and what a piece becomes.

The Python half is `tests/test_schema_nine.py` and `tests/test_schema_nine_promotion.py`;
this file is what `rust/src/` had to learn, specified in `docs/archive/plan-schema-6-9.md`
C6.

**Two rules that fail in opposite ways, again.** A wild piece changes *what moves exist* --
it blocks a pawn, offers itself as a capture, and threatens the square it will step to --
so a perft sees most of it, and the ladder below is pinned with its arithmetic spelled
out. What no perft can see is the phase itself: a boar walking a route changes the
position after the move without changing anybody's move list, exactly as a strike does,
so the phase is asserted square by square and the tally is asserted number by number.

**And one thing a perft is actively blind to.** The check filter now reads a third
owner's attack block, which is a change to a *legality* filter: being wrong in the
permissive direction leaves a king able to walk onto a boar's next square, and the move
list still looks like a move list. `TestAWildPieceIsAnAttacker` is the test that can see
it, and the sharpest case in it is a king eaten in the mover's own phase -- where the
post-phase question "is my royal attacked?" answers *no*, because there is nothing left
on the board to attack.

**And the case those tests could not see**, because they ask about the royal's own square
and this one is about everybody else's: `TestWhatThePhaseDiscovers`. Asking the right
question of the right board is not enough if the filter is allowed to skip most of the
moves, and the shortcut that skips them assumes nothing moves but the mover. That one was
found by a played game rather than by any test here (`docs/reference/schema-9.md`), which
is the third time in this bump that the permissive direction has been found by a
measurement.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import fuzz
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _name(square: int, width: int = 8) -> str:
    return chr(ord("a") + square % width) + str(square // width + 1)


def _boar(chess_spec: dict[str, Any], **rules: Any) -> dict[str, Any]:
    """The stage-C showcase: chess plus one mirrored wild boar
    (`docs/archive/plan-schema-6-9.md` C8)."""
    spec = copy.deepcopy(chess_spec)
    spec["schema_version"] = 9
    spec["name"] = "boar"
    spec["pieces"].append(
        {
            "id": "W",
            "name": "boar",
            "value": 3,
            "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}],
        }
    )
    spec["rules"]["npcs"] = [
        {"piece": "W", "route": ["c3", "c4", "d4", "d3"], "bounty": 3, "mirror": True}
    ]
    spec["rules"]["scoring"] = {"count": "points"}
    spec["rules"].update(rules)
    return spec


def _lone(chess_spec: dict[str, Any], route: list[str], **entry: Any) -> dict[str, Any]:
    """One boar on one route, no mirror: the shape most of these tests want."""
    spec = _boar(chess_spec)
    spec["rules"]["npcs"] = [{"piece": "W", "route": route, "mirror": False, **entry}]
    return spec


def _wild_squares(engine: Engine, state: Any) -> list[str]:
    return [
        _name(square, engine.spec.width)
        for square, value in enumerate(state.board)
        if value and engine.spec.owner_of(value) == WILD
    ]


def _move(engine: Engine, state: Any, origin: str, target: str) -> int:
    """The one legal move from `origin` to `target`, by name."""
    width = engine.spec.width
    found = [
        m
        for m in engine.legal_moves(state)
        if _name(m & 0xFF, width) == origin and _name((m >> 8) & 0xFF, width) == target
    ]
    assert len(found) == 1, f"{origin}-{target}: {len(found)} moves"
    return found[0]


def _targets(engine: Engine, state: Any, origin: str) -> set[str]:
    width = engine.spec.width
    return {
        _name((m >> 8) & 0xFF, width)
        for m in engine.legal_moves(state)
        if _name(m & 0xFF, width) == origin
    }


# ------------------------------------------------------------------------------- the phase


class TestThePhase:
    def test_every_wild_piece_walks_one_square_of_its_route_per_move(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """One move is one step, and the route cycles."""
        engine = Engine(_boar(chess_spec))
        state = engine.initial_state()
        assert _wild_squares(engine, state) == ["c3", "c6"]
        walked = []
        for origin, target in [
            ("a2", "a3"), ("a7", "a6"), ("h2", "h3"), ("h7", "h6"),
            ("a3", "a4"), ("a6", "a5"), ("h3", "h4"), ("h6", "h5"),
        ]:
            state = engine.apply(state, _move(engine, state, origin, target))
            walked.append(_wild_squares(engine, state))
        assert walked == [
            ["c4", "c5"], ["d4", "d5"], ["d3", "d6"], ["c3", "c6"],
            ["c4", "c5"], ["d4", "d5"], ["d3", "d6"], ["c3", "c6"],
        ]

    def test_the_squares_are_listed_before_any_of_them_moves(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """The ascending tie-break, and it is load-bearing rather than tidy.

        `c3` is square 18 and `c4` is square 26, so the first step of this route goes
        *up* the board. A single ascending scan that stepped the boar as it met it would
        meet the same piece again at 26 and step it on to `d4` (27) in the same phase --
        two squares for one move, and no move count anywhere would notice.
        """
        engine = Engine(_lone(chess_spec, ["c3", "c4", "d4", "d3"]))
        assert _square("c3") < _square("c4") < _square("d4")
        state = engine.apply(engine.initial_state(), _move(engine, engine.initial_state(), "a2", "a3"))
        assert _wild_squares(engine, state) == ["c4"]

    def test_it_eats_what_it_walks_into_and_the_piece_is_gone(
        self, chess_spec: dict[str, Any]
    ) -> None:
        engine = Engine(_lone(chess_spec, ["c3", "c4"]))
        state = engine.make_state(
            {"a1": ("K", 0), "h8": ("K", 1), "c4": ("R", 0), "c3": ("W", WILD)},
            moved=("c3",),
        )
        after = engine.apply(state, _move(engine, state, "a1", "b1"))
        assert _wild_squares(engine, after) == ["c4"]
        assert after.board[_square("c3")] == 0
        # Gone, not taken: no hand receives it, and this ruleset has none anyway.
        assert after.hands == ((), ())

    def test_it_waits_in_front_of_a_piece_its_type_may_not_take(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """A boar that declares `captures: []` is a moving wall -- `boar-tame`'s arm."""
        spec = _lone(chess_spec, ["c3", "c4"])
        next(p for p in spec["pieces"] if p["id"] == "W")["captures"] = []
        engine = Engine(spec)
        state = engine.make_state(
            {"a1": ("K", 0), "h8": ("K", 1), "c4": ("R", 0), "c3": ("W", WILD)},
            moved=("c3",),
        )
        after = engine.apply(state, _move(engine, state, "a1", "b1"))
        assert _wild_squares(engine, after) == ["c3"]
        assert after.board[_square("c4")] != 0

    def test_a_one_square_route_is_a_piece_that_never_moves(
        self, chess_spec: dict[str, Any]
    ) -> None:
        engine = Engine(_lone(chess_spec, ["c3"]))
        state = engine.initial_state()
        for _ in range(4):
            moves = engine.legal_moves(state)
            state = engine.apply(state, moves[0])
            assert _wild_squares(engine, state) == ["c3"]

    def test_the_phase_closes_the_en_passant_window_it_steps_into(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """A boar standing where an en passant capture would land closes the window.

        The capture is generated onto `ep_target` and removes the piece on `ep_victim`;
        with a boar now on either, playing it would write over the boar without counting
        it. The control below is the same position with the route one file away.
        """
        engine = Engine(_lone(chess_spec, ["d3", "e3"]))
        state = engine.initial_state()
        after = engine.apply(state, _move(engine, state, "e2", "e4"))
        assert _wild_squares(engine, after) == ["e3"]
        assert after.ep_target == -1 and after.ep_victim == -1

        control = Engine(_lone(chess_spec, ["a3", "b3"]))
        state = control.initial_state()
        after = control.apply(state, _move(control, state, "e2", "e4"))
        assert after.ep_target == _square("e3") and after.ep_victim == _square("e4")

    def test_a_quiet_step_is_reversible_and_a_meal_is_not(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """What lets the repetition rule catch a game in which only the boar moves."""
        engine = Engine(_lone(chess_spec, ["c3", "c4"]))
        quiet = engine.make_state(
            {"a1": ("K", 0), "h8": ("K", 1), "c3": ("W", WILD)}, moved=("a1", "h8", "c3")
        ).with_(halfmove=7)
        assert engine.apply(quiet, _move(engine, quiet, "a1", "b1")).halfmove == 8

        meal = engine.make_state(
            {"a1": ("K", 0), "h8": ("K", 1), "c4": ("R", 0), "c3": ("W", WILD)},
            moved=("a1", "h8", "c4", "c3"),
        ).with_(halfmove=7)
        assert engine.apply(meal, _move(engine, meal, "a1", "b1")).halfmove == 0


# --------------------------------------------------------------- the phase, across the boundary


class TestWhatTheBoundaryReports:
    """Q74: `apply` hands the phase back, and the position it hands back did not move.

    The phase runs after the mover's move inside the same `make`, so one move makes two
    board changes and only the sum survives in the position. A caller that has to *show*
    the ply was left diffing two boards against the compiled route table; it asks now.
    """

    def test_apply_reports_every_step_the_phase_took(self, chess_spec: dict[str, Any]) -> None:
        """`(from, to, eaten)` per wild piece that walked, oldest step first.

        The steps are checked against the board they produced rather than against a
        transcript: what left is empty, what arrived holds the piece, and the order is
        the ascending scan the phase itself runs.
        """
        engine = Engine(_boar(chess_spec))
        state = engine.initial_state()
        after, phase = engine.apply_with_phase(state, _move(engine, state, "a2", "a3"))

        assert [(_name(s.origin), _name(s.target), s.eaten) for s in phase] == [
            ("c3", "c4", 0), ("c6", "c5", 0)
        ]
        assert [s.origin for s in phase] == sorted(s.origin for s in phase)
        for step in phase:
            assert after.board[step.origin] == 0
            assert engine.spec.owner_of(after.board[step.target]) == WILD

    def test_a_quiet_ruleset_reports_a_phase_of_nothing(self, chess_spec: dict[str, Any]) -> None:
        engine = Engine(chess_spec)
        state = engine.initial_state()
        assert engine.apply_with_phase(state, engine.legal_moves(state)[0])[1] == ()

    def test_reporting_the_phase_moved_no_position(self, chess_spec: dict[str, Any]) -> None:
        """Returning the log must not perturb the game it is a log of.

        A seeded game on the showcase's own shape, pinned by the sum of its position keys
        and by the board it ends on. The log is written by `make` either way -- `unmake`
        has always walked it back -- so this is the guard that says handing it across the
        boundary stayed a *read*. Q74 took the same 60-ply reading on `archive/wolves`
        before and after the change and got the same key and the same board.
        """
        engine = Engine(_boar(chess_spec))
        rng = random.Random(9)
        state = engine.initial_state()
        keys, steps = [], 0
        for _ in range(80):
            moves = engine.legal_moves(state)
            if not moves or engine.terminal(state, moves) is not None:
                break
            state, phase = engine.apply_with_phase(state, rng.choice(sorted(moves)))
            keys.append(state.key)
            steps += len(phase)

        assert len(keys) == 80, "the game runs the whole way, or the pinning is of a stub"
        assert steps == 85, "and the phase did something in most of those plies"
        # Re-pinned by Q1008a: this inline variant names no stored game, so its identity is
        # its normal-form hash now and its keys are drawn from that; the board below, the
        # same game move for move, did not move. Was 0xF28BDCA625FD4BD3 / 0x25E6A0C27298763E.
        assert sum(keys) % (1 << 64) == 0x27D2F28D4A103CF0
        assert state.key == 0x31F22A6465C02E99
        assert bytes(state.board).hex() == (
            "0000000605000400000000040000000000000008000001010100010000080000"
            "000c000000000b0800011500000000000800000009000000000b000d00000003"
        )
        assert state.points == (0.0, 3.0), "one boar eaten, and the bounty paid for it"


# -------------------------------------------------------------------- a wild piece attacks


class TestAWildPieceIsAnAttacker:
    def test_the_only_illegal_king_move_is_onto_the_square_the_boar_steps_to(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """The test C6 asks for by name, and the one no perft can stand in for.

        The boar stands on `c3` and its route's next square is `c2`. Every square around
        the king on `b1` is legal except `c2`: stepping there is not being attacked
        afterwards -- it is being *eaten* in the mover's own phase, so the post-phase
        board has no king to ask about at all.
        """
        engine = Engine(_lone(chess_spec, ["c3", "c2"]))
        state = engine.make_state(
            {"b1": ("K", 0), "h8": ("K", 1), "c3": ("W", WILD)}, moved=("c3",)
        )
        assert _targets(engine, state, "b1") == {"a1", "a2", "b2", "c1"}
        # `c3` is missing from that set because taking the boar is a capture the king may
        # make -- and it is refused for a different reason: after the capture nothing
        # steps, and the king on c3 is not attacked. It is absent because the *king* is
        # on b1 and c3 is a knight's move away, not a king's.
        assert _square("c2") not in {
            (m >> 8) & 0xFF for m in engine.legal_moves(state)
        }

    def test_a_royal_standing_where_a_boar_steps_next_is_in_check(
        self, chess_spec: dict[str, Any]
    ) -> None:
        engine = Engine(_lone(chess_spec, ["c3", "c2"]))
        state = engine.make_state(
            {"c2": ("K", 0), "h8": ("K", 1), "c3": ("W", WILD)}, moved=("c3",)
        )
        assert engine.in_check(state, 0)
        assert not engine.in_check(state, 1)

    def test_a_side_with_no_way_out_of_a_boars_check_is_checkmated(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """Checkmate by an actor nobody owns, through the one definition of `in_check`.

        The route steps a knight's move -- `c3` to `a2` -- which the DSL allows and which
        is what makes the mate constructible at all: a boar one *king's* step from the
        square it threatens can always be taken by the king standing there. Nothing
        Black owns attacks `a2`; the boar does, and the rook and three knights cover
        every flight square without adding a check of their own.
        """
        engine = Engine(_lone(chess_spec, ["c3", "a2"]))
        state = engine.make_state(
            {
                "a2": ("K", 0), "h8": ("K", 1), "c3": ("W", WILD),
                "h1": ("R", 1), "b5": ("N", 1), "d3": ("N", 1), "d4": ("N", 1),
            },
            moved=("c3",),
        )
        assert engine.in_check(state, 0)
        assert engine.legal_moves(state) == []
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.reason == "checkmate" and outcome.winner == 1

    def test_a_king_may_still_take_the_boar_that_threatens_it(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """Taking it is a way out of check, and it pays."""
        engine = Engine(_lone(chess_spec, ["c3", "c2"], bounty=5))
        state = engine.make_state(
            {"c2": ("K", 0), "h8": ("K", 1), "c3": ("W", WILD)}, moved=("c3",)
        )
        after = engine.apply(state, _move(engine, state, "c2", "c3"))
        assert _wild_squares(engine, after) == []
        assert after.points == (5.0, 0.0)

    def test_a_boar_is_nobodys_check_for_a_check_limit(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """`check_limit` counts what a *player* delivers, and only that.

        The one place schema 9's definition of "attacked" is deliberately not asked. The
        boar arrives on `b3` in White's own phase and its next square is the black king's,
        so Black is in check and nobody delivered it. The rook arm is the control: the
        same ruleset counts a player's check the way it always did.
        """
        spec = _lone(chess_spec, ["a3", "b3", "b4"])
        spec["rules"]["check_limit"] = 3
        engine = Engine(spec)
        state = engine.make_state(
            {"h1": ("K", 0), "b4": ("K", 1), "a3": ("W", WILD), "h5": ("R", 0)},
            moved=("a3",),
        )
        assert not engine.in_check(state, 1)

        boar = engine.apply(state, _move(engine, state, "h1", "g1"))
        assert engine.in_check(boar, 1)
        assert boar.checks == (0, 0)

        rook = engine.apply(state, _move(engine, state, "h5", "h4"))
        assert rook.checks == (1, 0)


class TestWhatThePhaseDiscovers:
    """The filter has to run for *every* move, because the phase moves under all of them.

    `TestAWildPieceIsAnAttacker` covers the two cases C6 named, and both are about the
    royal's own square: a royal eaten in the mover's phase, and a royal left standing
    where a boar steps next. Both are caught however the move list is filtered, because
    the second makes `in_check` true before the move and the first is only reachable by
    moving the royal -- and a royal move is verified by the pin shortcut in any case.

    What the shortcut cannot see is a check the phase *discovers* somewhere else on the
    board. Its premise is that a move which does not touch the royal's lines cannot
    change whether the royal is attacked; a wild piece walking after every ply makes that
    premise false, and the move that discovers the check need have nothing to do with
    either the royal or the boar. Found in a played game rather than by a test
    (`docs/reference/schema-9.md`): in one `boar-lone` position 16 of White's 21 legal
    moves left White's own royal attacked once the phase had run.

    Two routes to it, and neither is the boar's departure alone -- which is why the fix
    is the shortcut and not a case: a wild piece stepping *off* a square unblocks a line,
    and a wild piece stepping *onto* one carries its own threat to a new square.
    """

    def test_a_move_that_lets_the_phase_unblock_a_line_is_illegal(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """The boar on `c3` is what stops the queen on `a5` from seeing `e1`.

        White is not in check, so nothing about the rook on `h1` is suspect: its origin
        is not the royal's square, it is not pinned by anything, and it is neither a drop
        nor an en passant. It is illegal all the same, because the phase steps the boar
        to `c4` and the diagonal opens. Only the king's four squares off the diagonal
        survive.
        """
        engine = Engine(_lone(chess_spec, ["c3", "c4"]))
        state = engine.make_state(
            {
                "e1": ("K", 0), "e8": ("K", 1), "a5": ("Q", 1), "h1": ("R", 0),
                "c3": ("W", WILD),
            },
            moved=("c3", "e1", "h1"),
        )
        assert not engine.in_check(state, 0)

        assert _targets(engine, state, "h1") == set()
        assert _targets(engine, state, "e1") == {"d1", "e2", "f1", "f2"}
        _no_legal_move_leaves_the_mover_attacked(engine, state)

    def test_a_move_that_leaves_the_boar_pointing_at_the_royal_is_illegal(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """The other direction: the phase does not open a line, it moves the threat.

        A wild piece's threat is the one square it steps to next, so a boar on `c3`
        threatens `c2` and the king on `b1` is not in check. After the phase the boar
        stands on `c2` and threatens `b1`. The rook's move is illegal for a reason that
        exists only after the phase has run, and the king's own moves are legal because
        they are the ones the shortcut already verified.
        """
        engine = Engine(_lone(chess_spec, ["c3", "c2", "b1"]))
        state = engine.make_state(
            {"b1": ("K", 0), "e8": ("K", 1), "h1": ("R", 0), "c3": ("W", WILD)},
            moved=("c3", "b1", "h1"),
        )
        assert not engine.in_check(state, 0)

        assert _targets(engine, state, "h1") == set()
        # `c2` is missing for the older reason and the king has to leave anyway: stepping
        # onto the boar's next square is being eaten in the mover's own phase, which is
        # the case `TestAWildPieceIsAnAttacker` pins.
        assert _targets(engine, state, "b1") == {"a1", "a2", "b2", "c1"}
        _no_legal_move_leaves_the_mover_attacked(engine, state)

    def test_the_position_the_campaign_found(self, chess_spec: dict[str, Any]) -> None:
        """The game that found it, replayed move word by move word.

        `archive/boar-lone.json` at seed 11, `random` against
        `minimax:fit=16a4528b0d53,nodes=100`, from the move list in the run: the lone
        boar is back on `c3` after eight phases of its four-square route, blocking the
        black queen on `a5` from the white king on `e1`. The game ended `royal_captured`
        under `win: checkmate`, which is the illegal-position guard firing on a position
        the filter should have made unreachable -- one ply in 87,257 across the whole
        schema-9 campaign, and the guard is why that measurement stands.

        The route and the piece are `boar-lone`'s, built here rather than loaded, so the
        archived spec keeps its hash and this test keeps its own copy of the position.
        Five moves survive: block on `b4` or `d2`, take the boar so nothing steps, or
        step the king off the diagonal.
        """
        engine = Engine(_lone(chess_spec, ["c3", "c4", "d4", "d3"]))
        state = engine.initial_state()
        for move in [4097, 11573, 5389, 9782, 6923, 8754, 4867, 8251]:
            assert move in engine.legal_moves(state)
            state = engine.apply(state, move)
        assert _wild_squares(engine, state) == ["c3"]
        assert not engine.in_check(state, 0)

        assert {engine.describe_move(state, m) for m in engine.legal_moves(state)} == {
            "Bc1-d2", "Ke1-d1", "Ke1-f2", "Pb2-b4", "Pb2xc3",
        }
        _no_legal_move_leaves_the_mover_attacked(engine, state)


def _no_legal_move_leaves_the_mover_attacked(engine: Engine, state: Any) -> None:
    """The invariant the legality filter exists to keep, asked of the post-phase board.

    `engine.apply` runs the phase, so asking `in_check` of the position a legal move
    leaves behind is asking exactly the question C1 says legality is asked of -- and it
    is a question about one implementation, not an agreement between two.
    """
    mover = state.side_to_move
    left_attacked = [
        engine.describe_move(state, m)
        for m in engine.legal_moves(state)
        if engine.in_check(engine.apply(state, m), mover)
    ]
    assert left_attacked == []


# -------------------------------------------------------------------------------- the tally


class TestTheTally:
    def test_taking_a_wild_piece_pays_its_bounty(self, chess_spec: dict[str, Any]) -> None:
        engine = Engine(_boar(chess_spec))
        state = engine.initial_state()
        after = engine.apply(state, _move(engine, state, "b2", "c3"))
        assert after.points == (3.0, 0.0)
        assert _wild_squares(engine, after) == ["c5"]

    def test_a_phase_step_pays_nobody(self, chess_spec: dict[str, Any]) -> None:
        """A boar eating a player's piece is nobody's gain."""
        engine = Engine(_lone(chess_spec, ["c3", "c4"], bounty=3))
        state = engine.make_state(
            {"a1": ("K", 0), "h8": ("K", 1), "c4": ("R", 0), "c3": ("W", WILD)},
            moved=("c3",),
        )
        after = engine.apply(state, _move(engine, state, "a1", "b1"))
        assert after.points == (0.0, 0.0)

    def test_the_clock_compares_the_two_tallies(self, chess_spec: dict[str, Any]) -> None:
        engine = Engine(_lone(chess_spec, ["c3", "c4"], bounty=3))
        state = engine.make_state(
            {"a1": ("K", 0), "h8": ("K", 1), "c3": ("W", WILD)}, moved=("c3",)
        ).with_(ply=engine.spec.max_plies - 1)
        drawn = engine.terminal(state.with_(ply=engine.spec.max_plies))
        assert drawn is not None and drawn.winner is None
        scored = engine.terminal(
            state.with_(ply=engine.spec.max_plies, points=(3.0, 0.0))
        )
        assert scored is not None and scored.reason == "scored" and scored.winner == 0

    def test_the_tally_needs_no_zobrist_term(self, chess_spec: dict[str, Any]) -> None:
        """Two positions differing only in the tally hash the same, and that is right.

        A bounty is paid only when a wild piece is taken, which is a material change,
        which clears the repetition window -- so the repetition rule never compares two
        positions across one. The `hands` argument and the permanent-`colour` argument,
        for the third time.
        """
        engine = Engine(_boar(chess_spec))
        state = engine.initial_state()
        assert engine.hash(state) == engine.hash(state.with_(points=(3.0, 0.0)))

        paid = engine.apply(state, _move(engine, state, "b2", "c3"))
        assert paid.repetition == (paid.key,)


# -------------------------------------------------------------------------------- the perft


class TestWhatPerftCounts:
    """`perft(d)` counts sequences of `d` *player* moves, each carrying the phase that
    follows it, so a leaf is a position after `d` moves and `d` phases. The phase adds no
    move to anybody's list; what it does is move the board on under them.
    """

    def test_the_showcase_ladder(self, chess_spec: dict[str, Any]) -> None:
        """**Twenty, and `docs/archive/plan-schema-6-9.md` C6 predicted twenty-one.**

        The prediction was "twenty chess moves, less `Nb1-c3` onto the boar's square,
        plus `b2xc3` and `d2xc3` taking it". Two of those three clauses are wrong and
        they nearly cancel. `Nb1-c3` does not go away: the boar is not White's, so the
        knight takes it, and a capture is one move exactly as the quiet move was. What
        does go away is `c2-c3` **and** `c2-c4`: a pawn does not capture forwards, so a
        boar standing on `c3` blocks its own file. So it is 20 - 2 + 2 = 20, and the
        showcase's opening move count coincides with chess's by arithmetic rather than
        by accident.
        """
        engine = Engine(_boar(chess_spec))
        state = engine.initial_state()
        opening = {
            f"{_name(m & 0xFF)}-{_name((m >> 8) & 0xFF)}" for m in engine.legal_moves(state)
        }
        chess = Engine(chess_spec)
        plain = {
            f"{_name(m & 0xFF)}-{_name((m >> 8) & 0xFF)}"
            for m in chess.legal_moves(chess.initial_state())
        }
        assert plain - opening == {"c2-c3", "c2-c4"}
        assert opening - plain == {"b2-c3", "d2-c3"}
        assert [engine.perft(state, depth) for depth in (1, 2, 3)] == [20, 380, 8224]

    def test_the_phase_adds_no_move_to_anybodys_list(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """A boar on a route no piece can reach leaves every count alone."""
        engine = Engine(_lone(chess_spec, ["d5", "e5"]))
        chess = Engine(chess_spec)
        state, plain = engine.initial_state(), chess.initial_state()
        assert engine.perft(state, 1) == chess.perft(plain, 1) == 20
        # The second ply differs, and only because the *board* moved under Black: the
        # boar steps `d5`-`e5` in White's phase, which frees `d7-d5` and blocks `e7-e5`.
        assert engine.perft(state, 2) == 380

    def test_make_and_unmake_agree_with_a_walk_over_apply(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """The undo log, checked the only way the boundary allows.

        `perft` walks with make/unmake inside the core and reuses one position; `apply`
        builds a new state per move and can carry no damage forward. A phase step the
        undo log put back wrongly shows up here as a leaf count that disagrees.
        """
        engine = Engine(_boar(chess_spec))
        state = engine.initial_state()

        def walk(position: Any, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(
                walk(engine.apply(position, m), depth - 1)
                for m in engine.legal_moves(position)
            )

        assert engine.perft(state, 3) == walk(state, 3)


# ----------------------------------------------------------- what a piece becomes on arrival


class TestPromotionOnEntry:
    """Schema 9's second key in the core: `Tables::arriving_value` and its call sites."""

    @pytest.fixture
    def fall(self) -> dict[str, Any]:
        spec = load_spec(ROOT / "archive/hierarchy-fall.json")
        spec["schema_version"] = 9
        spec["rules"]["promote_on_entry"] = True
        return spec

    @pytest.fixture
    def dobutsu(self) -> dict[str, Any]:
        spec = load_spec(ROOT / "rulesets/dobutsu.json")
        spec["schema_version"] = 9
        spec["rules"]["promote_on_entry"] = True
        return spec

    def test_a_dropped_chick_on_the_far_rank_stands_as_a_hen(
        self, dobutsu: dict[str, Any]
    ) -> None:
        """The drop branch of `make`, and the hand still gives up what was dropped."""
        engine = Engine(dobutsu)
        width = engine.spec.width
        state = engine.make_state({"a1": ("L", 0), "c4": ("L", 1)})
        chick = engine.spec.piece_ids.index("C")
        hen = engine.spec.piece_ids.index("H")
        hand = [0] * engine.spec.n_types
        hand[chick] = 1
        state = state.with_(hands=(tuple(hand), tuple([0] * engine.spec.n_types)))

        far = [
            m
            for m in engine.legal_moves(state)
            if (m >> 16) & 0xF == 4 and ((m >> 8) & 0xFF) // width == 3
        ]
        assert far, "no drop onto the far rank"
        after = engine.apply(state, far[0])
        landed = after.board[(far[0] >> 8) & 0xFF]
        assert engine.spec.type_of(landed) == hen
        assert after.hands[0][chick] == 0 and after.hands[0][hen] == 0

    @staticmethod
    def _demotion(engine: Engine, king: str = "h1") -> Any:
        """A black knight on `d1` about to be struck by a white rook on `d8`.

        `hierarchy-fall`'s chain ends `N -> P`, so the survivor is a black *pawn* -- and
        `d1` is where a black pawn promotes, which is the whole point. With the white
        king on `h1` the queen the flag produces gives check along the first rank, which
        is what the legality filter has to be able to see. `h4` is the same position with
        the king on the one nearby square a queen on `d1` does not attack, so the strike
        is legal there and what it leaves can be read off the board.
        """
        return engine.make_state(
            {king: ("K", 0), "a8": ("K", 1), "d1": ("N", 1), "d8": ("R", 0)},
            moved=("d1", "d8", king),
        )

    def test_a_strike_that_demotes_onto_the_far_rank_leaves_a_queen(
        self, fall: dict[str, Any]
    ) -> None:
        """`strike_writes` under `position: stay`: the survivor lands where it stood."""
        without = copy.deepcopy(fall)
        del without["rules"]["promote_on_entry"]
        plain = Engine(without)
        state = self._demotion(plain, "h4")
        after = plain.apply(state, _move(plain, state, "d8", "d1"))
        landed = after.board[_square("d1")]
        assert plain.spec.type_of(landed) == plain.spec.piece_ids.index("P")

        # The same strike under the flag: the pawn it leaves has entered play inside its
        # own promotion zone, so what stands there is the first target that rule lists.
        engine = Engine(fall)
        state = self._demotion(engine, "h4")
        after = engine.apply(state, _move(engine, state, "d8", "d1"))
        landed = after.board[_square("d1")]
        assert engine.spec.type_of(landed) == engine.spec.piece_ids.index("Q")
        assert engine.spec.owner_of(landed) == 1

    def test_the_demotion_that_arrives_as_a_queen_still_resets_the_clock(
        self, fall: dict[str, Any]
    ) -> None:
        """`material_changed` compares the captured value against the *survivor's*.

        A knight demoted into a pawn that arrives as a queen is material moving, and the
        no-progress clock has to see it -- confirmed rather than reasoned about, because
        a knight and a queen are worth different amounts and so are a knight and a pawn.
        """
        engine = Engine(fall)
        state = self._demotion(engine, "h4").with_(halfmove=9)
        assert engine.apply(state, _move(engine, state, "d8", "d1")).halfmove == 0

    def test_the_legality_filter_sees_what_arrives(self, fall: dict[str, Any]) -> None:
        """A strike whose survivor arrives as a queen giving check is an illegal strike.

        `strike_writes`-in-the-filter, applied to the new table: without
        `arriving_value` in the filter the survivor stands there as the pawn the chain
        produced, checks nothing, and a king is captured on the next ply.
        """
        without = copy.deepcopy(fall)
        del without["rules"]["promote_on_entry"]
        engine, plain = Engine(fall), Engine(without)
        assert "d1" not in _targets(engine, self._demotion(engine), "d8")
        assert "d1" in _targets(plain, self._demotion(plain), "d8")

    def test_the_flag_changes_the_tree_only_where_something_arrives(
        self, fall: dict[str, Any]
    ) -> None:
        """Nothing about *generation* changes, so the two ladders agree from the opening.

        `perft` counts the same moves -- a drop region and a promoting type's ordinary
        moves are asked of the dropped and the moving type, neither of which the flag
        touches -- and the trees can only differ once a struck piece actually lands
        inside its own promotion zone, which from the chess array is far past any depth
        worth walking. So the divergence is pinned where it happens instead: in
        `_demotion`, at depth 1, by exactly the one strike the flag makes illegal.
        """
        without = copy.deepcopy(fall)
        del without["rules"]["promote_on_entry"]
        flagged, plain = Engine(fall), Engine(without)
        one, other = flagged.initial_state(), plain.initial_state()
        assert [(flagged.perft(one, d), plain.perft(other, d)) for d in range(1, 5)] == [
            (20, 20), (400, 400), (8902, 8902), (197281, 197281)
        ]
        assert plain.perft(self._demotion(plain), 1) - 1 == flagged.perft(
            self._demotion(flagged), 1
        )

    def test_the_fuzz_oracles_run_clean_on_both_keys(
        self, chess_spec: dict[str, Any], fall: dict[str, Any]
    ) -> None:
        """All three of schema 9's oracles, live, on rulesets that exercise them."""
        assert fuzz.walk(_boar(chess_spec), seed=11, plies=12) > 0
        assert fuzz.walk(fall, seed=11, plies=12) > 0
