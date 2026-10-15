"""Engine correctness.

Most assertions here are written against arbitrary specs. The ones that use chess do
so because chess has published reference numbers -- perft counts are the only truly
independent check available on a move generator -- and those tests say so in their
names. A test that needs chess for any other reason is testing the wrong thing.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine import compile as compile_module
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_PROMOTION,
    MOVE_ROTATE,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.schema import _capture_is_possible, validate_spec

ROOT = Path(__file__).resolve().parents[1]

# Reference node counts for the standard opening position, from the chess programming
# literature. These catch castling, en passant, promotion, and check evasion together;
# no other single number does.
CHESS_PERFT = [(1, 20), (2, 400), (3, 8902), (4, 197281)]


@pytest.mark.parametrize(("depth", "expected"), CHESS_PERFT)
def test_chess_perft_matches_published_reference(chess: Engine, depth: int, expected: int) -> None:
    assert chess.perft(chess.initial_state(), depth) == expected


@pytest.mark.slow
def test_chess_perft_depth_five_matches_published_reference(chess: Engine) -> None:
    assert chess.perft(chess.initial_state(), 5) == 4865609


def test_apply_does_not_mutate_its_argument(chess: Engine) -> None:
    state = chess.initial_state()
    before = state.board
    chess.apply(state, chess.legal_moves(state)[0])
    assert state.board is before


def test_hash_is_stable_across_engine_instances(chess_spec: dict) -> None:
    one, two = Engine(chess_spec), Engine(chess_spec)
    assert one.hash(one.initial_state()) == two.hash(two.initial_state())


def test_hash_distinguishes_side_to_move(chess: Engine) -> None:
    state = chess.initial_state()
    assert chess._compute_hash(state) != chess._compute_hash(state.with_(side_to_move=1))


def test_incremental_hash_agrees_with_a_full_recompute(chess: Engine) -> None:
    """The search reads the cached key at every node; a drift here corrupts everything.

    Walked over a real playout rather than a handful of positions, because the update
    has a separate path per move type and castling and en passant are the ones that
    touch squares the move itself does not name.
    """
    rng = random.Random(20260819)
    for _ in range(40):
        state = chess.initial_state()
        for _ in range(60):
            moves = chess.legal_moves(state)
            if chess.terminal(state, moves) is not None:
                break
            state = chess.apply(state, rng.choice(moves))
            assert state.key == chess._compute_hash(state)


class TestCastling:
    def test_both_sides_available_from_the_home_position(self, chess: Engine) -> None:
        state = chess.make_state(
            {"e1": ("K", 0), "a1": ("R", 0), "h1": ("R", 0), "e8": ("K", 1)}
        )
        castles = sorted(t for t in chess.move_texts(state) if "castle" in t)
        assert castles == ["Ke1-c1 (castle)", "Ke1-g1 (castle)"]

    def test_rook_lands_on_the_square_the_king_crossed(self, chess: Engine) -> None:
        state = chess.make_state({"e1": ("K", 0), "h1": ("R", 0), "e8": ("K", 1)})
        after = chess.apply(state, chess.find_move(state, "Ke1-g1 (castle)"))
        assert chess.spec.piece_at(after.board, chess._square_index("g1")) == (0, 5)
        assert chess.spec.piece_at(after.board, chess._square_index("f1")) == (0, 3)
        assert after.board[chess._square_index("h1")] == 0

    def test_unavailable_once_the_king_has_moved(self, chess: Engine) -> None:
        state = chess.make_state(
            {"e1": ("K", 0), "h1": ("R", 0), "e8": ("K", 1)}, moved=["e1"]
        )
        assert not [t for t in chess.move_texts(state) if "castle" in t]

    def test_blocked_by_an_occupied_square_between(self, chess: Engine) -> None:
        state = chess.make_state(
            {"e1": ("K", 0), "h1": ("R", 0), "g1": ("N", 0), "e8": ("K", 1)}
        )
        assert not [t for t in chess.move_texts(state) if "castle" in t]

    def test_forbidden_while_in_check(self, chess: Engine) -> None:
        state = chess.make_state(
            {"e1": ("K", 0), "h1": ("R", 0), "e8": ("R", 1), "a8": ("K", 1)}
        )
        assert not [t for t in chess.move_texts(state) if "castle" in t]

    def test_forbidden_through_an_attacked_square(self, chess: Engine) -> None:
        state = chess.make_state(
            {"e1": ("K", 0), "h1": ("R", 0), "f8": ("R", 1), "a8": ("K", 1)}
        )
        assert not [t for t in chess.move_texts(state) if "castle" in t]


class TestEnPassant:
    def test_double_step_creates_the_capture_and_it_expires(self, chess: Engine) -> None:
        start = chess.make_state(
            {"e5": ("P", 0), "d7": ("P", 1), "e1": ("K", 0), "e8": ("K", 1)}, side_to_move=1
        )
        after_double = chess.apply(start, chess.find_move(start, "Pd7-d5"))
        assert "Pe5xd6 e.p." in chess.move_texts(after_double)

        # A quiet move by White, a quiet reply by Black, and the chance is gone.
        quiet = chess.apply(after_double, chess.find_move(after_double, "Ke1-f1"))
        later = chess.apply(quiet, chess.find_move(quiet, "Ke8-f8"))
        assert not [t for t in chess.move_texts(later) if "e.p." in t]

    def test_capture_removes_the_passed_pawn(self, chess: Engine) -> None:
        start = chess.make_state(
            {"e5": ("P", 0), "d7": ("P", 1), "e1": ("K", 0), "e8": ("K", 1)}, side_to_move=1
        )
        after_double = chess.apply(start, chess.find_move(start, "Pd7-d5"))
        after_capture = chess.apply(after_double, chess.find_move(after_double, "Pe5xd6 e.p."))
        assert after_capture.board[chess._square_index("d5")] == 0
        assert chess.spec.piece_at(after_capture.board, chess._square_index("d6")) == (0, 0)

    def test_single_step_creates_no_capture(self, chess: Engine) -> None:
        start = chess.make_state(
            {"e5": ("P", 0), "d7": ("P", 1), "e1": ("K", 0), "e8": ("K", 1)}, side_to_move=1
        )
        after_single = chess.apply(start, chess.find_move(start, "Pd7-d6"))
        assert after_single.ep_target == -1


class TestAHopAndTheEnPassantWindow:
    """The two halves of Q121, which read as one defect and are not one.

    The compiler built a window from a hop and the generator's hop arm could not take in
    passing, so the obvious story was that the one fed the other. It does not. A hop lands
    *past* its screen, so landing on the second square of its ray means the screen stands
    on the first -- the window a hop opens is on an occupied square and no arm can take
    it, whether or not the hop arm knows how. The compiler stopped building those.

    The arm's own gap is separate and real: a window a *slide* opened is on an empty
    square, and a hop that may capture can land there. That is the second test.
    """

    @staticmethod
    def _hop_double_step() -> dict:
        """A promoting piece whose only forward move hops two, over a screen on b2.

        No `en_passant` -- with it the spec is now refused, which is the third test.
        """
        return {
            "schema_version": 1,
            "name": "hop double step",
            "board": {"width": 5, "height": 5},
            "pieces": [
                {"id": "K", "royal": True, "moves": [
                    {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"},
                ]},
                {"id": "P", "moves": [
                    {"kind": "hop", "vectors": [[0, 1]], "mode": "move"},
                    {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral",
                     "mode": "capture"},
                ]},
                {"id": "S", "moves": [
                    {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"},
                ]},
                {"id": "Q", "moves": [
                    {"kind": "slide", "vectors": [[1, 1]], "symmetry": "dihedral8"},
                ]},
            ],
            "setup": {"placements": [
                {"piece": "K", "squares": ["a1"]},
                {"piece": "P", "squares": ["b1"]},
                {"piece": "S", "squares": ["b2"]},
            ]},
            "rules": {"promotion": {"piece": "P", "rank": "last", "to": ["Q"]}},
        }

    def test_a_hop_opens_no_window_because_it_crosses_its_own_screen(self) -> None:
        """The reason, played out. b1-b3 is a hop, so b2 holds the piece it jumped."""
        engine = Engine(self._hop_double_step())
        state = engine.make_state(
            {"a1": ("K", 0), "b1": ("P", 0), "b2": ("S", 0), "a5": ("K", 1)}
        )
        after = engine.apply(state, engine.find_move(state, "Pb1-b3"))
        assert after.board[engine._square_index("b2")] != 0, "the screen is still there"
        assert after.ep_target == -1

    def test_a_hop_takes_in_passing_on_a_window_a_slide_opened(
        self, chess_spec: dict
    ) -> None:
        """Chess as scaffold, plus one capture-mode hop on the pawn.

        Black's d7-d5 leaves d6 empty and d5 taken, which is the pawn on d4's screen. It
        cannot reach d6 any other way -- the push is blocked and its diagonals are c5 and
        e5 -- so the move below is the hop arm's alone.
        """
        spec = copy.deepcopy(chess_spec)
        for piece in spec["pieces"]:
            if piece["id"] == "P":
                piece["moves"].append(
                    {"kind": "hop", "vectors": [[0, 1]], "mode": "capture"}
                )
        assert validate_spec(spec) == []

        # The other half, on a spec that validates: the hop reaches two squares along its
        # ray and adds no window, because the compiler no longer reads one off a hop.
        def windows(source: dict) -> int:
            table = compile_spec(source).ep_passes
            return sum(len(square) for frame in table for square in frame)

        assert windows(spec) == windows(chess_spec) == 16

        engine = Engine(spec)
        start = engine.make_state(
            {"d4": ("P", 0), "d7": ("P", 1), "e1": ("K", 0), "e8": ("K", 1)},
            side_to_move=1,
        )
        after_double = engine.apply(start, engine.find_move(start, "Pd7-d5"))
        assert "Pd4xd6 e.p." in engine.move_texts(after_double)

        after = engine.apply(after_double, engine.find_move(after_double, "Pd4xd6 e.p."))
        assert after.board[engine._square_index("d5")] == 0
        assert engine.spec.piece_at(after.board, engine._square_index("d6")) == (0, 0)

    def test_en_passant_on_a_hop_only_promoter_is_refused(self) -> None:
        """`can_double_step` and `_ep_passes` agree, so the declaration is now inert."""
        spec = self._hop_double_step()
        spec["rules"]["en_passant"] = True
        errors = validate_spec(spec)
        assert [e for e in errors if "rules.en_passant can generate no capture" in e]


class TestTheCaptureMatrixBindsTheCaptureInPassing:
    """Schema 3's matrix reaches the one arm of movegen that cannot read its own target.

    Every other capture is refused by `may_take` against the piece standing on the square
    being moved to. An en passant capture lands on an **empty** square and takes the piece
    on `ep_victim`, so those three tests see `occupant == 0`, take the quiet-move branch,
    and never ask the matrix at all -- which let `captures: []` on every piece take a piece
    in passing (Q118). The spec below is chess only as a scaffold: what is asserted is that
    one cell of the matrix decides one move, on a spec that has both features at once, and
    no committed ruleset has both.

    It is also the premise `spec.schema::_capture_is_possible` rests on, and through it the
    three inert-rule refusals of `must_capture`, `explode_on_capture` and `drops` -- see
    `docs/reference/bugs-investigations.md`, "Rules that declare nothing", cases 3, 4 and 5. The second test states
    that coupling directly, so a fourth arm added to the generator un-proves it loudly
    rather than silently.
    """

    @staticmethod
    def _window(engine: Engine):
        """White pawn on e5 and a black knight on f6, after Black double-steps past it.

        Two captures for the one pawn from the one position: `f6` by the ordinary arm and
        `d6` in passing. Both are the same descriptor's capture mode, so the pair separates
        *the matrix binds this move* from *en passant has been switched off*.
        """
        start = engine.make_state(
            {
                "e5": ("P", 0), "d7": ("P", 1), "f6": ("N", 1),
                "e1": ("K", 0), "e8": ("K", 1),
            },
            side_to_move=1,
        )
        return engine.apply(start, engine.find_move(start, "Pd7-d5"))

    def test_a_forbidden_pair_loses_the_capture_and_keeps_the_others(
        self, chess_spec: dict
    ) -> None:
        """The pawn may take everything except a pawn, so only the e.p. capture goes."""
        permitted = Engine(chess_spec)
        offered = permitted.move_texts(self._window(permitted))
        assert "Pe5xd6 e.p." in offered and "Pe5xf6" in offered

        spec = copy.deepcopy(chess_spec)
        spec["schema_version"] = 3
        for piece in spec["pieces"]:
            if piece["id"] == "P":
                piece["captures"] = [p["id"] for p in spec["pieces"] if p["id"] != "P"]
        assert validate_spec(spec) == []

        forbidden = Engine(spec)
        offered = forbidden.move_texts(self._window(forbidden))
        assert "Pe5xf6" in offered, "the matrix still lets a pawn take a knight"
        assert "Pe5xd6 e.p." not in offered, (
            "a pawn that may not take a pawn may not take one in passing either"
        )

    def test_a_spec_that_can_take_nothing_generates_no_capture_at_all(
        self, chess_spec: dict
    ) -> None:
        """The sentence three inert-rule refusals are written on, asserted of the core.

        `_capture_is_possible` reads the matrix and the move modes and answers *nothing
        can ever be taken*; `_check_inert_rules` then refuses `must_capture`,
        `explode_on_capture` and a `drops` that fills from captures. If any arm of the
        generator can still take a piece, all three refuse a game that works -- so this
        holds the two readings together rather than trusting that they agree.
        """
        spec = copy.deepcopy(chess_spec)
        spec["schema_version"] = 3
        for piece in spec["pieces"]:
            piece["captures"] = []
        assert validate_spec(spec) == []
        assert not _capture_is_possible(spec["pieces"], {p["id"] for p in spec["pieces"]})

        engine = Engine(spec)
        after_double = self._window(engine)
        assert after_double.ep_target != -1, "the window is open; the capture is refused"
        assert not [t for t in engine.move_texts(after_double) if "x" in t]

        # And over a whole game rather than one position: a playout that can take nothing
        # must end with every piece it started with.
        state = engine.initial_state()
        material = sum(1 for square in state.board if square)
        rng = random.Random(11)
        for _ in range(120):
            if engine.terminal(state) is not None:
                break
            moves = engine.legal_moves(state)
            if not moves:
                break
            state = engine.apply(state, rng.choice(moves))
        assert sum(1 for square in state.board if square) == material


class TestAPawnNeverTakesItsOwnInPassing:
    """Q531: the en passant arm asks whose piece stands on `ep_victim`.

    Every other capture reads its target's owner through `may_land`; the en passant arm
    lands on an empty square, so it never did. In a one-move turn the side that opened the
    window has always handed the move over, so the victim is the opponent's and no
    library game can tell the difference (perft on the published chess positions is
    unchanged). It differs in two places: a long turn, where the window a double step
    opens is still open on the same player's next move (`docs/bugs.md`), and a position
    handed in from outside. Chess below is only a scaffold for the second.
    """

    @staticmethod
    def _passing(texts: list[str]) -> list[str]:
        return [text for text in texts if "e.p." in text]

    def test_a_handed_in_window_on_the_movers_own_pawn_takes_nothing(
        self, chess_spec: dict
    ) -> None:
        engine = Engine(chess_spec)

        def window(victim_owner: int):
            return engine.make_state(
                {
                    "e5": ("P", 0), "d5": ("P", victim_owner),
                    "e1": ("K", 0), "e8": ("K", 1),
                },
                moved=("e5", "d5"),
                ep_target="d6",
                ep_victim="d5",
            )

        assert self._passing(engine.move_texts(window(1))) == ["Pe5xd6 e.p."]
        assert self._passing(engine.move_texts(window(0))) == []

    def test_a_long_turn_does_not_take_its_own_double_step(self, monkeypatch) -> None:
        """The fuzzer's spec, past the validator that refuses it (Q341).

        White's first move double-steps b2-b4 and leaves White to move again, with the
        window on b3 open and the pawn on a2 beside it. Before Q531 `Pa2xb3 e.p.` was
        legal and, with the `strike` rule, panicked the core on the self-capture. The
        guard stays: whether the opponent may answer a double step made on a turn's
        first move is still the open design question `docs/bugs.md` names.
        """
        spec = json.loads(
            (Path(__file__).parent / "_en_passant_in_a_long_turn.json").read_text(
                encoding="utf-8"
            )
        )
        assert validate_spec(spec), "the validator still refuses the pair"
        monkeypatch.setattr(compile_module, "validate_or_raise", lambda spec: None)
        engine = Engine(compile_spec(spec))

        state = engine.initial_state()
        state = engine.apply(state, engine.find_move(state, "Pb2-b4"))
        assert state.side_to_move == 0, "the turn has a second move"
        assert state.ep_target != -1, "the window is open"
        assert self._passing(engine.move_texts(state)) == []


class TestPromotion:
    def test_every_declared_target_is_offered(self, chess: Engine) -> None:
        state = chess.make_state({"a7": ("P", 0), "e1": ("K", 0), "e8": ("K", 1)})
        promotions = sorted(t for t in chess.move_texts(state) if "=" in t)
        assert promotions == ["Pa7-a8=B", "Pa7-a8=N", "Pa7-a8=Q", "Pa7-a8=R"]

    def test_the_chosen_piece_appears_on_the_board(self, chess: Engine) -> None:
        state = chess.make_state({"a7": ("P", 0), "e1": ("K", 0), "e8": ("K", 1)})
        after = chess.apply(state, chess.find_move(state, "Pa7-a8=N"))
        assert chess.spec.piece_at(after.board, chess._square_index("a8")) == (0, 1)


class TestTerminalDetection:
    def test_back_rank_mate_is_a_win(self, chess: Engine) -> None:
        state = chess.make_state(
            {
                "a1": ("R", 0),
                "e1": ("K", 0),
                "h8": ("K", 1),
                "g7": ("P", 1),
                "h7": ("P", 1),
            }
        )
        after = chess.apply(state, chess.find_move(state, "Ra1-a8"))
        outcome = chess.terminal(after)
        assert outcome is not None
        assert (outcome.winner, outcome.reason) == (0, "checkmate")

    def test_no_moves_without_check_is_a_draw(self, chess: Engine) -> None:
        state = chess.make_state(
            {"a8": ("K", 1), "b6": ("K", 0), "c7": ("Q", 0)}, side_to_move=1
        )
        assert chess.legal_moves(state) == []
        outcome = chess.terminal(state)
        assert outcome is not None and outcome.is_draw and outcome.reason == "stalemate"

    def test_moves_leaving_royalty_attacked_are_filtered(self, chess: Engine) -> None:
        state = chess.make_state({"e1": ("K", 0), "e8": ("R", 1), "a1": ("R", 0)})
        assert chess.in_check(state)
        assert sorted(chess.move_texts(state)) == ["Ke1-d1", "Ke1-d2", "Ke1-f1", "Ke1-f2"]

    def test_the_no_progress_clock_draws(self, chess: Engine) -> None:
        state = chess.make_state(
            {"e1": ("K", 0), "e8": ("K", 1)},
            halfmove=chess.spec.no_progress_draw,
        )
        outcome = chess.terminal(state)
        assert outcome is not None and outcome.reason == "no_progress"

    def test_the_ply_limit_draws(self, chess: Engine) -> None:
        state = chess.make_state({"e1": ("K", 0), "e8": ("K", 1)}, ply=chess.spec.max_plies)
        outcome = chess.terminal(state)
        assert outcome is not None and outcome.reason == "ply_limit"


class TestTheMovementTablesAreIndexedByFrame:
    """`[frame][type][square]`, one frame per (player, facing), player-major.

    A **frame** is one way of reading a piece's movement vectors. There was exactly one
    per player until schema 8, which is why the first index used to be the player -- and
    that index has always secretly meant *which way is forward*. Schema 8 is where the
    two stop coinciding, and schema 9 will generalise it again for a third owner.

    The ordering is the part worth pinning. `player * n_facings + facing` keeps one
    army's board values contiguous, so every "is this piece mine?" test in the core stays
    a range comparison; the other spelling interleaves them and turns each of those into
    a silent bug. And with one facing it is the identity on the player, which is what
    makes every ruleset before schema 8 compile to the tables it always did.
    """

    def test_a_frame_is_the_player_when_nothing_turns(self) -> None:
        from chess_variant_sandbox.engine.compile import frame_index

        assert frame_index(0, 0, 1) == 0
        assert frame_index(1, 0, 1) == 1

    def test_frames_are_player_major_so_one_army_stays_contiguous(self) -> None:
        """The encoding argument, as arithmetic rather than as prose."""
        from chess_variant_sandbox.engine.compile import frame_index

        n_facings = 4
        first = [frame_index(0, facing, n_facings) for facing in range(n_facings)]
        second = [frame_index(1, facing, n_facings) for facing in range(n_facings)]
        assert first == [0, 1, 2, 3]
        assert second == [4, 5, 6, 7]
        assert max(first) < min(second)

    def test_the_frame_count_is_the_first_dimension_of_every_movement_table(
        self, chess_spec: dict
    ) -> None:
        """The tables are as long as the frame count says, in the compiler's own terms."""
        from chess_variant_sandbox.engine.compile import compile_spec

        spec = compile_spec(chess_spec)
        assert spec.n_frames == 2
        for table in (spec.leaps, spec.slides, spec.hops,
                      spec.dedup_squares, spec.immobile_squares):
            assert len(table) == spec.n_frames

    def test_the_core_refuses_tables_whose_shape_it_disagrees_with(
        self, chess_spec: dict
    ) -> None:
        """The one way the two sides can disagree about frames, and it is silent.

        A table shorter than the frame count says is read past the end at a high slot,
        and what comes back is whatever bytes follow it -- a plausible wrong game rather
        than a crash. The length *is* the shape, so the core checks it at construction,
        which is the only place either side can be told it was wrong.
        """
        import chess_variant_sandbox._core as core
        from chess_variant_sandbox.engine.bridge import build_tables
        from chess_variant_sandbox.engine.compile import compile_spec

        tables = build_tables(compile_spec(chess_spec))
        core.Core(tables)  # the honest shape is accepted

        with pytest.raises(ValueError, match="slots"):
            core.Core({**tables, "n_frames": 4})


class TestEveryEffectIsInTheKey:
    """The write log, against the slow definition, over everything that can be played.

    `make` used to compute the Zobrist key five different ways -- a full board rescan for
    an explosion, a square-addressed delta for a strike, and three more for drops,
    castling and the ordinary case. One log, one delta now. **This is the only kind of
    test that can see whether that was right.**

    A perft cannot. Nothing the key does changes a move count, so the entire class of
    bug -- a square written by an effect the delta did not know about -- passes every
    perft in the suite. It has happened: an atomic explosion cleared squares the
    incremental update never saw, and the hash quietly disagreed with the position until
    two positions collided in a transposition table. The full recompute in the explosion
    arm was the *response* to that, and the write log is what makes it unnecessary.

    The slow definition is `Engine._compute_hash`, which reads the whole position. If the
    two agree at every ply of a long game on every ruleset in the repository, every
    effect wrote its square into the log.
    """

    @staticmethod
    def _walk(engine: Engine, seed: int, plies: int) -> int:
        """Play a random game, checking the key against the definition at every ply."""
        rng = random.Random(seed)
        state = engine.initial_state()
        played = 0
        for _ in range(plies):
            moves = engine.legal_moves(state)
            if not moves or engine.terminal(state, moves) is not None:
                break
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state), (
                f"the incremental key drifted from the definition at ply {played}"
            )
            played += 1
        return played

    @pytest.mark.slow
    def test_the_key_holds_over_every_committed_and_archived_ruleset(self) -> None:
        """Four seeds and sixty plies each, over all forty-seven of them.

        Long enough to reach the effects: a strike needs something to be takeable, an
        explosion needs a capture, and castling needs the back rank to open. Twenty plies
        of random play on some of these boards reaches none of them.
        """
        from chess_variant_sandbox.measurement import ROOT
        from chess_variant_sandbox.spec import load_spec

        specs = sorted((ROOT / "rulesets").glob("*.json"))
        specs += sorted((ROOT / "archive").glob("*.json"))
        assert len(specs) >= 47, "the corpus shrank; this walk is meant to cover all of it"

        for path in specs:
            engine = Engine(load_spec(path))
            for seed in range(4):
                self._walk(engine, seed, 60)

    def test_the_key_holds_through_an_explosion(self) -> None:
        """The arm that used to rescan the whole board, and the reason it did.

        `atomic` is the ruleset that produced the bug the full recompute was written to
        answer. It is named here rather than left to the corpus walk because it is the
        only committed ruleset that can reach the explosion arm at all, and a slow test
        is not where a regression should be caught.
        """
        from chess_variant_sandbox.measurement import ROOT
        from chess_variant_sandbox.spec import load_spec

        engine = Engine(load_spec(ROOT / "rulesets" / "atomic.json"))
        # Long enough that captures happen: an explosion arm that never fires proves
        # nothing, and random play on chess's board first captures around ply ten.
        assert self._walk(engine, 0, 80) > 20

    @pytest.mark.parametrize("name", [
        "hierarchy-fall", "hierarchy-rise", "hierarchy-rise-stay", "hierarchy-rise-throw",
    ])
    def test_the_key_holds_through_a_strike(self, name: str) -> None:
        """Every position rule a strike game has, because the position rule is what moves
        squares.

        `stay` leaves the attacker where it was and the survivor where it stood; `swap`
        exchanges them; `choose` (`hierarchy-rise-throw`) throws the survivor onto a square
        its player picks. Those write different squares, so they are different deltas, and
        a log that got one right could get another wrong. (`hierarchy-fall-swap`, the 2x2's
        fourth cell, left the library on 2026-10-12, Q868; `hierarchy-rise` covers `swap`.)
        """
        from chess_variant_sandbox.measurement import ROOT
        from chess_variant_sandbox.spec import load_spec

        engine = Engine(load_spec(ROOT / "archive" / f"{name}.json"))
        assert self._walk(engine, 0, 80) > 20

    def test_a_square_written_twice_is_counted_once(self) -> None:
        """The one thing a write log gets wrong if it is not deduplicated.

        A target square can be written three times in one move: the mover arrives,
        promotion replaces it, and a strike or a blast overwrites it again. The delta
        wants what was there before all of that against what is there after all of it, so
        the log keeps the earliest value per square. Keeping the latest -- or keeping
        every write -- XORs an intermediate value into the key that was never on the
        board at the end of a ply, and nothing but this comparison would notice.

        A promoting capture is the shortest move that writes one square twice.
        """
        engine = Engine({
            "schema_version": 2,
            "name": "promoting capture",
            "board": {"width": 5, "height": 5},
            "pieces": [
                {"id": "K", "royal": True,
                 "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]],
                            "symmetry": "rotate4"}]},
                {"id": "P", "moves": [{"kind": "leap", "vectors": [[0, 1]],
                                       "symmetry": "none", "mode": "move"},
                                      {"kind": "leap", "vectors": [[1, 1], [-1, 1]],
                                       "symmetry": "none", "mode": "capture"}]},
                {"id": "Q", "moves": [{"kind": "slide", "vectors": [[1, 0], [1, 1]],
                                       "symmetry": "dihedral8"}]},
            ],
            "setup": {"mirror": False, "placements": [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e1"]},
                {"piece": "P", "player": 0, "squares": ["b4"]},
                {"piece": "Q", "player": 1, "squares": ["c5"]},
            ]},
            "rules": {"win": "royal_capture",
                      "promotion": {"piece": "P", "to": ["Q"], "ranks": [4]}},
        })
        state = engine.initial_state()
        captures = [
            move for move in engine.legal_moves(state)
            if engine.spec.piece_at(state.board, (move >> 8) & 0xFF) is not None
        ]
        assert captures, "the position is meant to offer a promoting capture"
        for move in captures:
            after = engine.apply(state, move)
            assert after.key == engine._compute_hash(after)


class TestIrreversibilityIsDerivedRatherThanDeclared:
    """Q113: which pieces can never return to a square they left, and where that is asked.

    A move that cannot be taken back is *progress*: it clears the repetition window and
    resets the no-progress clock. Until Q113 the core got that from `ep_type`, which is
    set only when `en_passant` is declared -- so whether a game's pawn reset its own clock
    depended on whether the game happened to declare an unrelated capturing rule, and
    dropping a phantom declaration from one committed ruleset moved its index by 0.2038.

    It is now derived from the compiled movement tables: a piece is one-way when its own
    move graph is acyclic. Every test below is written against a purpose-made spec rather
    than against chess, because *a pawn move is progress* is exactly the fact about chess
    this separates out. See `docs/reference/derived-irreversibility.md`.
    """

    @staticmethod
    def _probe(**rules: Any) -> dict[str, Any]:
        """One forward-only grunt each and two kings, on a board with room to walk."""
        return {
            "schema_version": 10,
            "name": "grunt probe", "description": "a piece that only goes one way",
            "board": {"width": 5, "height": 5},
            "pieces": [
                {"id": "G", "name": "grunt", "value": 1,
                 "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1}]},
                {"id": "K", "name": "king", "value": 0, "royal": True,
                 "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
                {"id": "Q", "name": "queen", "value": 9,
                 "moves": [{"kind": "slide", "vectors": [[1, 0], [1, 1]],
                            "symmetry": "dihedral8"}]},
            ],
            "setup": {"mirror": False, "placements": [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "G", "player": 0, "squares": ["c2"]},
                {"piece": "G", "player": 1, "squares": ["c4"]},
            ]},
            "rules": {"win": "royal_capture", **rules},
        }

    @staticmethod
    def _one_way(spec: dict[str, Any]) -> set[str]:
        from chess_variant_sandbox.engine.compile import compile_spec

        compiled = compile_spec(spec)
        return {compiled.piece_ids[index] for index in compiled.one_way_types}

    def test_a_piece_that_only_goes_forward_is_one_way(self) -> None:
        assert self._one_way(self._probe()) == {"G"}

    def test_a_piece_that_can_come_back_is_not(self) -> None:
        """And neither is the king that walks four ways or the queen that rides eight."""
        spec = self._probe()
        spec["pieces"][0]["moves"].append(
            {"kind": "slide", "vectors": [[0, -1]], "range": 1}
        )
        assert self._one_way(spec) == set()

    def test_a_rank_seam_takes_the_grunt_back_where_it_started(self) -> None:
        """The decisive case against deriving this from the declared vectors.

        *One net direction and no facing* still says one-way here and is wrong: the grunt
        marches up the board, crosses the seam and comes home. Nothing in the derivation
        knows what a seam is -- `_compile_square` already put it in the table it walks.
        """
        assert self._one_way(self._probe(wrap={"ranks": "both"})) == set()

    def test_a_file_seam_does_not(self) -> None:
        """And a "any wrap disqualifies it" caveat would get this one wrong instead."""
        assert self._one_way(self._probe(wrap={"files": "both"})) == {"G"}

    def test_a_facing_takes_it_back_without_the_derivation_saying_so(self) -> None:
        """The decision's "no `facing`" clause, as a consequence rather than a clause.

        A piece turns to any of its other orientations in one move, so two rotations at
        one square are a two-cycle in the piece's own graph.
        """
        spec = self._probe()
        spec["pieces"][0]["facing"] = {"axes": 2}
        assert self._one_way(spec) == set()

    def test_a_mandatory_promotion_is_a_sink_and_not_an_edge(self) -> None:
        """A seam the grunt promotes before it can reach does not take it back.

        The promoting move is `MOVE_PROMOTION`, irreversible by kind already, and it ends
        the piece -- so it contributes no edge, and the walk stops short of the seam.
        """
        spec = self._probe(
            wrap={"ranks": "both"},
            promotion={"piece": "G", "to": ["Q"], "rank": "last"},
        )
        assert self._one_way(spec) == {"G"}

    def test_a_promotion_it_may_decline_is_not_a_sink(self) -> None:
        """Declining leaves a `MOVE_NORMAL` behind, and the grunt walks on to the seam."""
        spec = self._probe(
            wrap={"ranks": "both"},
            promotion={"piece": "G", "to": ["Q"], "rank": "last", "mandatory": False},
        )
        assert self._one_way(spec) == set()

    def test_a_piece_that_cannot_move_at_all_is_still_rather_than_one_way(self) -> None:
        """Vacuous acyclicity is not a claim worth making, and nothing would read it."""
        spec = self._probe(drops={"from_captures": False, "stock": {"S": "unlimited"}})
        spec["pieces"].append({"id": "S", "name": "stone", "moves": []})
        assert self._one_way(spec) == {"G"}

    def test_en_passant_no_longer_decides_it(self) -> None:
        """The separation itself, as the assertion that would have failed before Q113.

        Two specs alike but for the capturing rule. The grunt is one-way either way now;
        before, only the declaring one's grunt reset the two clocks.
        """
        with_ep = self._probe(
            promotion={"piece": "G", "to": ["Q"], "rank": "last"}, en_passant=True
        )
        with_ep["pieces"][0]["moves"].append(
            {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2,
             "mode": "move", "condition": {"from_rank": 1}}
        )
        without = copy.deepcopy(with_ep)
        del without["rules"]["en_passant"]
        assert self._one_way(with_ep) == self._one_way(without) == {"G"}

    def test_a_one_way_move_clears_the_repetition_window_and_a_reversible_one_does_not(
        self,
    ) -> None:
        """The derived table reaching the core, asserted through the engine.

        The grunt's step and the king's step are both quiet `MOVE_NORMAL` moves onto an
        empty square, and nothing but the derived flag separates them. `state.repetition`
        is `history[window..]` -- everything that survived the last irreversible move --
        so an irreversible move leaves exactly the position it made.

        The *no-progress clock* is a second reset and this does not touch it. Q135 took
        that one off `promotion` and onto the same flag;
        `TestTheNoProgressClockIsKeyedOnTheSameFlag` below holds it.
        """
        engine = Engine(self._probe())

        def find(state: Any, origin: str, target: str) -> int:
            from chess_variant_sandbox.engine.moves import move_from, move_to

            squares = tuple(
                engine.spec.square_index(ord(name[0]) - ord("a"), int(name[1]) - 1)
                for name in (origin, target)
            )
            return next(
                move for move in engine.legal_moves(state)
                if (move_from(move), move_to(move)) == squares
            )

        # Four king steps out and back: nothing irreversible, so the window keeps them.
        state = engine.initial_state()
        for origin, target in (("a1", "a2"), ("e5", "e4"), ("a2", "a1"), ("e4", "e5")):
            state = engine.apply(state, find(state, origin, target))
        kept = len(state.repetition)
        assert kept > 1, "four reversible moves are meant to leave a window to clear"

        walked_on = engine.apply(state, find(state, "c2", "c3"))
        assert len(walked_on.repetition) == 1, "the grunt cannot come back: progress"

        stepped_back = engine.apply(state, find(state, "a1", "a2"))
        assert len(stepped_back.repetition) == kept + 1, "the king can come back"


class TestTheNoProgressClockIsKeyedOnTheSameFlag:
    """Q135: what resets the count of plies without progress.

    The clock used to read `material_changed || is_promoting(piece_type) || <colour>`, so
    *declaring a promotion rule* was what made a piece's every move count as progress --
    the quiet ones, and a rotation, which the window right above it calls reversible. A
    one-way piece with no promotion rule made no progress at all. That is *a pawn move
    resets the fifty-move clock*, a fact about chess, wearing the key of an unrelated rule
    about what a piece becomes at the far end.

    It reads the derived one-way flag now, plus `kind == MOVE_PROMOTION` for the one thing
    the old term was rightly carrying. Every assertion below is against a purpose-made
    spec. See section 9 of `docs/reference/derived-irreversibility.md`.
    """

    @staticmethod
    def _probe(
        placements: list[dict[str, Any]] | None = None, **rules: Any
    ) -> dict[str, Any]:
        """Two kings that can walk in circles and a grunt that cannot."""
        return {
            "schema_version": 10,
            "name": "clock probe", "description": "a piece that only goes one way",
            "board": {"width": 5, "height": 5},
            "pieces": [
                {"id": "G", "name": "grunt", "value": 1,
                 "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1}]},
                {"id": "K", "name": "king", "value": 0, "royal": True,
                 "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
                {"id": "Q", "name": "queen", "value": 9,
                 "moves": [{"kind": "slide", "vectors": [[1, 0], [1, 1]],
                            "symmetry": "dihedral8"}]},
            ],
            "setup": {"mirror": False, "placements": placements or [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "G", "player": 0, "squares": ["c2"]},
                {"piece": "G", "player": 1, "squares": ["c4"]},
            ]},
            "rules": {"win": "royal_capture", "no_progress_draw": 100,
                      "max_plies": 400, **rules},
        }

    @staticmethod
    def _find(engine: Engine, state: Any, origin: str, target: str) -> int:
        squares = tuple(
            engine.spec.square_index(ord(name[0]) - ord("a"), int(name[1]) - 1)
            for name in (origin, target)
        )
        return next(
            move for move in engine.legal_moves(state)
            if (move_from(move), move_to(move)) == squares
        )

    def _walk(self, engine: Engine, steps: tuple[tuple[str, str], ...]) -> Any:
        """Play `steps` from the initial position, picking each move by its two squares."""
        state = engine.initial_state()
        for origin, target in steps:
            state = engine.apply(state, self._find(engine, state, origin, target))
        return state

    # Two king steps first in every walk below, so the clock is off zero before the move
    # under test and a reset is visible as a reset rather than as a starting value.
    OPENING = (("a1", "a2"), ("e5", "e4"))

    def test_a_one_way_piece_that_promotes_into_nothing_still_makes_progress(self) -> None:
        """`rulesets/dull`'s grunt, which is why the corpus cares.

        No promotion rule anywhere in the spec, so the old key never fired and a game of
        nothing but forward marching drew by no progress while every ply was a step
        nobody could take back.
        """
        engine = Engine(self._probe())
        state = self._walk(engine, self.OPENING + (("c2", "c3"),))
        assert state.halfmove == 0

    def test_a_promoting_piece_that_can_walk_back_does_not(self) -> None:
        """The other direction, which `chess-oriented-pawns` and `shogi` are made of.

        The grunt is given a step backwards, so it can return to the square it left. It
        still promotes at the far rank, and under the old key that alone made its every
        move progress.
        """
        spec = self._probe(promotion={"piece": "G", "to": ["Q"], "rank": "last"})
        spec["pieces"][0]["moves"].append(
            {"kind": "slide", "vectors": [[0, -1]], "range": 1}
        )
        state = self._walk(Engine(spec), self.OPENING + (("c2", "c3"),))
        assert state.halfmove == 3

    def test_the_promotion_itself_resets_it_even_so(self) -> None:
        """The term the old key was rightly carrying, rehoused onto the move.

        The same two-way grunt, one step from the last rank. A piece that changes type
        has spent itself and cannot be spent again, so the move is progress whether or
        not the piece could have walked back -- and that is a fact about the move rather
        than about the type's rule sheet.
        """
        spec = self._probe(
            placements=[
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "G", "player": 0, "squares": ["c4"]},
            ],
            promotion={"piece": "G", "to": ["Q"], "rank": "last"},
        )
        spec["pieces"][0]["moves"].append(
            {"kind": "slide", "vectors": [[0, -1]], "range": 1}
        )
        engine = Engine(spec)
        state = self._walk(engine, self.OPENING)
        promote = self._find(engine, state, "c4", "c5")
        assert move_type(promote) == MOVE_PROMOTION
        assert engine.apply(state, promote).halfmove == 0

    def test_turning_on_the_spot_is_not_progress(self) -> None:
        """`archive/weathervane`'s whole move, and the sharpest of the four.

        A rotation goes nowhere, takes nothing and can be turned straight back -- the
        window has always said so. The old key said the opposite whenever the turning
        piece happened to promote, so two pieces could spin at each other for ever and
        the clock would never leave zero.
        """
        spec = self._probe(promotion={"piece": "G", "to": ["Q"], "rank": "last"})
        spec["pieces"][0]["facing"] = {"axes": 2}
        for placement in spec["setup"]["placements"]:
            if placement["piece"] == "G":
                placement["facing"] = "forward"
        engine = Engine(spec)
        state = self._walk(engine, self.OPENING)
        turn = next(
            move for move in engine.legal_moves(state)
            if move_type(move) == MOVE_ROTATE
        )
        assert engine.apply(state, turn).halfmove == 3


class TestRoyalLossIsDerivedRatherThanDeclared:
    """Q133: when a royal off the board ends the game, and what decides that.

    Until Q133 it was `explode_on_capture`. Declaring a blast switched on *a side whose
    royal is off the board loses at once* -- in `terminal_prefix`, however the royal left,
    and in the legality filter, where a move that removes the opponent's royal became
    legal however exposed it left the mover's own. Neither is what the key is named for.

    It is derived now, from two things the core already had: the **win condition** is
    about the royal, and the side **started with one**. The second clause is not
    defensive. `rulesets/horde.json` is `win: checkmate` with royal pieces and one army
    of thirty-six pawns and no king, so "a side with no royal has lost" would hand every
    game of horde to black at ply zero.

    See `docs/reference/derived-royal-loss.md`.
    """

    @staticmethod
    def _probe(placements: list[dict[str, Any]] | None = None, **rules: Any) -> dict[str, Any]:
        """Two kings and two grunts on a board with room, and no way to take anything."""
        return {
            "schema_version": 10,
            "name": "royal probe", "description": "a game whose royal can leave the board",
            "board": {"width": 5, "height": 5},
            "pieces": [
                {"id": "K", "name": "king", "value": 0, "royal": True,
                 "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]],
                            "symmetry": "rotate4", "mode": "both"}]},
                {"id": "G", "name": "grunt", "value": 1,
                 "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1,
                            "mode": "move"}]},
                {"id": "R", "name": "rook", "value": 5,
                 "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4",
                            "mode": "both"}]},
            ],
            "setup": {"mirror": False, "placements": placements or [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "G", "player": 0, "squares": ["c2"]},
                {"piece": "G", "player": 1, "squares": ["c4"]},
            ]},
            "rules": {"win": "checkmate", **rules},
        }

    @staticmethod
    def _without(engine: Engine, state: Any, square: str) -> Any:
        """The same position with `square` emptied.

        A royal leaves the board by a blast or by the wild phase, and both are rules of
        their own; cutting the square is how to ask what `terminal_prefix` makes of the
        result without dragging one of them in. It is also exactly how the coupling was
        measured in `docs/reference/bugs-investigations.md`, "Rules that declare nothing", case 4.
        """
        board = bytearray(state.board)
        index = engine.spec.square_index(ord(square[0]) - ord("a"), int(square[1]) - 1)
        assert board[index] != 0, f"{square} is already empty"
        board[index] = 0
        return dataclasses.replace(state, board=bytes(board))

    def _verdict(self, spec: dict[str, Any], square: str | None = "e5") -> Any:
        assert validate_spec(spec) == [], "the probe has to be a spec the validator takes"
        engine = Engine(spec)
        state = engine.initial_state()
        if square is not None:
            state = self._without(engine, state, square)
        return engine.terminal_prefix(state)

    def test_a_missing_royal_ends_a_checkmate_game_with_no_explosion_declared(self) -> None:
        """The separation itself: the assertion that would have failed before Q133."""
        outcome = self._verdict(self._probe())
        assert outcome is not None and outcome.winner == 0
        assert outcome.reason == "royal_destroyed"

    def test_declaring_an_explosion_changes_nothing(self) -> None:
        """And the other direction: what the declaration used to buy, it no longer buys."""
        assert self._verdict(self._probe(explode_on_capture={"radius": 1})) == self._verdict(
            self._probe()
        )

    def test_both_royals_standing_is_not_an_ending(self) -> None:
        assert self._verdict(self._probe(), square=None) is None

    def test_the_reason_names_the_win_condition_that_was_resolved(self) -> None:
        """A game won *by capturing* the royal reports a capture; one where it was never
        supposed to be takeable reports its destruction. That split is not new -- it is
        what `Win::RoyalCapture`'s own arm and the explosion clause each said -- but it
        is now one statement rather than two."""
        outcome = self._verdict(self._probe(win="royal_capture"))
        assert outcome is not None and outcome.reason == "royal_captured"

    def test_a_side_that_never_had_a_royal_is_not_losing_for_it(self) -> None:
        """The clause the shorter reading of this rule does not have."""
        kingless = self._probe(
            placements=[
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "G", "player": 0, "squares": ["c2"]},
                {"piece": "G", "player": 1, "squares": ["c4"]},
            ],
            extinction={"pieces": "any", "min_count": 1, "result": "loss"},
        )
        assert self._verdict(kingless, square=None) is None

    def test_horde_is_the_corpus_spec_that_insists_on_it(self) -> None:
        """`win: checkmate`, royal pieces, and one army with no king in the setup.

        Read as "a side with no royal has lost", this ends at ply zero with a win for
        black. It is the only spec in the corpus where a side starts royal-less, and it
        is the reason the derived fact is *the royal it began with*.
        """
        engine = Engine(load_spec(ROOT / "rulesets" / "horde.json"))
        assert engine.terminal_prefix(engine.initial_state()) is None

    def test_a_win_condition_that_is_not_about_the_royal_leaves_it_a_piece(self) -> None:
        """And this one *loses* what the old gate gave it, which is the point.

        A `win: none` game with its own ending may mark a piece royal; `needs_check_filter`
        is false there, so nothing protects it and its loss is not the game's. Declaring
        an explosion used to end such a game the moment the piece left the board.
        """
        for extra in ({}, {"explode_on_capture": {"radius": 1}}):
            spec = self._probe(
                win="none",
                reach={"piece": "G", "squares": ["a5", "b5", "c5", "d5", "e5"],
                       "result": "win"},
                **extra,
            )
            assert self._verdict(spec) is None, extra

    def test_a_kingless_side_is_no_longer_a_game_that_keeps_running(self) -> None:
        """`archive/wolves` is the corpus spec where the clause goes from dead to live.

        Nothing in it stops a wolf landing on a king, so the position is expressible.
        Before Q133 nothing in `terminal_prefix` fired -- the explosion gate was false, and
        the illegal-position guard asks `in_check`, which is `false` because there is no
        royal there to attack -- and the game ran on with one side kingless, unable to be
        checkmated (`generate_legal` skips the check filter when the royal is gone) and
        still able to checkmate.

        The test below cuts the king out by hand because **legal play cannot get there**;
        the test after this one is why.
        """
        engine = Engine(load_spec(ROOT / "archive" / "wolves.json"))
        state = engine.initial_state()
        assert engine.terminal_prefix(state) is None
        outcome = engine.terminal_prefix(self._without(engine, state, "e8"))
        assert outcome is not None and outcome.winner == 0
        assert outcome.reason == "royal_destroyed"

    def test_but_a_wolf_can_never_take_a_king_in_play(self) -> None:
        """Why the two `npcs` specs keep playing exactly the game they played.

        A wild piece's *landing* square is an attack -- `threatened` adds
        `attacks_square(WILD, square)` wherever `has_npcs` -- and `leaves_royal_attacked`
        asks it of the position **after** the phase has walked. So no owner may leave a
        royal where a wolf is about to step, and a ply later the wolf has stepped somewhere
        else. This is the assertion that turns "two specs could change" into "no spec's
        play changes", and it is worth pinning: a wild threat model that stopped looking a
        step ahead would make the clause above live in ordinary games.
        """
        engine = Engine(load_spec(ROOT / "archive" / "wolves.json"))
        spec = engine.spec
        state = engine.initial_state()
        board = bytearray(state.board)

        wolf = spec.piece_ids.index("W")
        king = spec.piece_ids.index("K")
        side_stride = spec.n_facings * spec.n_types
        black_king = next(
            value for value in board
            if value and (value - 1) % spec.n_types == king
            and (value - 1) // side_stride == 1
        )
        for square, value in enumerate(board):
            if value == black_king:
                board[square] = 0

        def name(square: int) -> str:
            return f"{chr(ord('a') + square % spec.width)}{square // spec.width + 1}"

        wolves = {
            square for square, value in enumerate(board)
            if value and (value - 1) % spec.n_types == wolf
        }
        assert wolves, "wolves is meant to be the spec with wild pieces on the board"

        in_check_on = {
            name(square)
            for square in range(spec.n_squares)
            if board[square] == 0
            and engine.in_check(
                dataclasses.replace(
                    state, board=bytes(board[:square] + bytearray([black_king]) + board[square + 1:])
                ),
                1,
            )
        }
        # The four squares the wolves step to next, and no royal may be left on one.
        assert {"b4", "b5", "g4", "g5"} <= in_check_on

    # ------------------------------------------------------------ the legality half

    @staticmethod
    def _wolf_probe(black_king: str) -> dict[str, Any]:
        """Player 0 in check, with one quiet move that does not answer it.

        A wild piece walks a two-square route after every move. Put the opponent's king on
        its next square and the quiet move wins the game outright, so what it leaves the
        mover's own king exposed to cannot matter; put the king out of reach and the same
        move is the illegal one it looks like.

        A **setup**, not a position reached by play: the test above shows that no owner may
        leave a royal on a wolf's landing square, so this one has to be dealt rather than
        played into. That is the point -- the guard is about what the rules make possible,
        not about what a game happens to walk into.
        """
        return {
            "schema_version": 10,
            "name": "wolf probe", "description": "a wild piece that can eat a king",
            "board": {"width": 5, "height": 5},
            "pieces": [
                {"id": "K", "name": "king", "value": 0, "royal": True,
                 "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]],
                            "symmetry": "rotate4", "mode": "both"}]},
                {"id": "G", "name": "grunt", "value": 1,
                 "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1,
                            "mode": "move"}]},
                {"id": "R", "name": "rook", "value": 5,
                 "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4",
                            "mode": "both"}]},
                {"id": "W", "name": "wolf", "value": 3,
                 "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]],
                            "symmetry": "rotate4", "mode": "both"}]},
            ],
            "setup": {"mirror": False, "placements": [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": [black_king]},
                {"piece": "G", "player": 0, "squares": ["a3"]},
                {"piece": "R", "player": 1, "squares": ["e1"]},
            ]},
            "rules": {
                "win": "checkmate",
                "npcs": [{"piece": "W", "route": ["c3", "c4"], "mirror": False}],
                "max_plies": 200,
            },
        }

    def _quiet_move(self, black_king: str) -> tuple[Engine, Any, int | None]:
        from chess_variant_sandbox.engine.moves import move_from, move_to

        spec = self._wolf_probe(black_king)
        assert validate_spec(spec) == [], "the probe has to be a spec the validator takes"
        engine = Engine(spec)
        state = engine.initial_state()
        assert engine.in_check(state), "the probe is about a move made while in check"
        squares = tuple(
            engine.spec.square_index(ord(name[0]) - ord("a"), int(name[1]) - 1)
            for name in ("a3", "a4")
        )
        move = next(
            (m for m in engine.legal_moves(state)
             if (move_from(m), move_to(m)) == squares),
            None,
        )
        return engine, state, move

    def test_a_move_that_takes_the_last_royal_is_legal_however_it_leaves_the_mover(
        self,
    ) -> None:
        """No explosion rule anywhere: the wild phase is the whole mechanism."""
        engine, state, move = self._quiet_move("c4")
        assert move is not None, "the grunt's step wins the game and must be legal"
        outcome = engine.terminal_prefix(engine.apply(state, move))
        assert outcome is not None and outcome.winner == 0
        assert outcome.reason == "royal_destroyed"

    def test_and_is_the_illegal_move_it_looks_like_when_it_takes_nothing(self) -> None:
        """The same position with the king one square out of the wolf's way."""
        _, _, move = self._quiet_move("e5")
        assert move is None, "a quiet move that leaves the mover in check is illegal"


class TestBareRoyalIsARoyalStandingAlone:
    """Q134: the rule needs a royal, and an empty side has not got one.

    `schema_two_ending`'s arm asked whether *no piece of mine is non-royal*, which is
    vacuously true for a side with nothing left. So on a spec with no royal piece
    `bare_royal` quietly became an annihilation ending, and on one with a royal it
    answered for a side that had already lost everything including the royal the rule
    is named for. `docs/reference/bugs-investigations.md` ("Rules that declare nothing", case 7) measured the first half on `dull.json`.

    This is the third of the three second meanings and it separates the other way from
    Q113's and Q133's. Those two had to **derive** the second rule, because no spec
    stated it. A spec that wants *a side with nothing left has lost* already has two
    ways to say so -- `win: "annihilation"` and `extinction` -- and the core reads both
    before this arm, so the second meaning is **removed** rather than rehoused.

    See `docs/reference/derived-royal-loss.md` §9.
    """

    @staticmethod
    def _cleared(engine: Engine, state: Any, side: int) -> Any:
        """The same position with every piece of ``side`` taken off, and ``side`` to move.

        Staged rather than played: what reaches this position is a rule of its own, and
        cutting the squares is how to ask what the verdict is without dragging one in.
        It is how case 7 of "Rules that declare nothing" in
        `docs/reference/bugs-investigations.md` measured the coupling.
        """
        spec = engine.spec
        stride = spec.n_facings * spec.n_types
        board = bytearray(state.board)
        for square, value in enumerate(board):
            if value and (value - 1) // stride == side:
                board[square] = 0
        assert board != state.board, "the side was meant to have pieces to clear"
        return dataclasses.replace(state, board=bytes(board), side_to_move=side)

    @staticmethod
    def _dull(**rules: Any) -> dict[str, Any]:
        """`rulesets/dull.json` with an ending that is not annihilation.

        `win: "annihilation"` answers in `terminal_prefix` before `schema_two_ending` is
        reached, which is what masked this everywhere in the corpus. `win: "none"` beside
        `stalemate: "loss"` is the spec saying *the ending here is running out of moves*,
        and it is what case 7 measured on.
        """
        spec = load_spec(ROOT / "rulesets" / "dull.json")
        spec["schema_version"] = 2  # `win: "none"` and `bare_royal` are both schema 2
        spec["rules"] = {**spec["rules"], "win": "none", "stalemate": "loss", **rules}
        return spec

    def test_a_side_cleared_off_the_board_is_not_bared(self) -> None:
        """Case 7's measurement, with the corrected reading as the assertion.

        Declaring the rule used to change both halves of the verdict: `bare_royal` and a
        draw where the spec states a stalemate loss for the side with nothing to play.

        The rule has to be put on the *compiled* spec by hand, because `compile_spec`
        validates and case 7's own clause refuses this spelling. That is why the coupling
        was invisible from the DSL side: no spec the validator takes could ask for it,
        and the core answered it anyway.
        """
        compiled = compile_spec(self._dull())
        undeclared = Engine(compiled)
        declared = Engine(dataclasses.replace(compiled, bare_royal="draw"))
        position = self._cleared(undeclared, undeclared.initial_state(), 1)

        outcome = undeclared.terminal(position)
        assert outcome is not None
        assert (outcome.winner, outcome.reason) == (0, "stalemate_loss")
        assert declared.terminal(position) == outcome

    def test_the_refusal_of_that_spelling_is_now_a_true_statement(self) -> None:
        """`_check_inert_rules` says the rule *can never fire* with no piece marked royal.

        It was right and for the wrong reason -- the rule fired, on the annihilation the
        arm read by accident. With the arm asking for a royal, no board on a royal-less
        spec can satisfy it, so the sentence is a proof rather than a claim.
        """
        errors = validate_spec(self._dull(bare_royal="draw"))
        assert any("bare_royal" in error and "never fire" in error for error in errors)

    @staticmethod
    def _royals(**rules: Any) -> dict[str, Any]:
        """A king and a grunt a side, and no way for either to take anything."""
        return {
            "schema_version": 2,
            "name": "bare probe", "description": "a game a king can be stripped down to",
            "board": {"width": 5, "height": 5},
            "pieces": [
                {"id": "K", "name": "king", "value": 0, "royal": True,
                 "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]],
                            "symmetry": "rotate4", "mode": "both"}]},
                {"id": "G", "name": "grunt", "value": 1,
                 "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1,
                            "mode": "move"}]},
            ],
            "setup": {"mirror": False, "placements": [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
                {"piece": "G", "player": 0, "squares": ["c2"]},
                {"piece": "G", "player": 1, "squares": ["c4"]},
            ]},
            "rules": {"win": "checkmate", "bare_royal": "loss", **rules},
        }

    def _verdict(self, spec: dict[str, Any], empty: str | None = None) -> Any:
        assert validate_spec(spec) == [], "the probe has to be a spec the validator takes"
        engine = Engine(spec)
        state = engine.initial_state()
        if empty is not None:
            index = engine.spec.square_index(ord(empty[0]) - ord("a"), int(empty[1]) - 1)
            board = bytearray(state.board)
            assert board[index] != 0, f"{empty} is already empty"
            board[index] = 0
            state = dataclasses.replace(state, board=bytes(board))
        return engine.terminal_prefix(state)

    def test_a_royal_left_with_no_company_is_the_ending_the_rule_names(self) -> None:
        """The rule itself, which the separation must not take away."""
        outcome = self._verdict(self._royals(), empty="c4")
        assert outcome is not None
        assert (outcome.winner, outcome.reason) == (0, "bare_royal")

    def test_a_royal_with_company_is_not_bared(self) -> None:
        assert self._verdict(self._royals()) is None

    def test_a_side_with_nothing_at_all_is_not_bared_in_a_royal_game_either(self) -> None:
        """The same reading where the spec *does* mark a piece royal.

        `win: "none"` is the one win condition that leaves a royal an ordinary piece, so
        `royal_loss_ends_it` is false and Q133's missing-royal clause stays out of the
        way -- which makes this the only shape where the empty side reaches this arm at
        all. It used to answer `bare_royal` for a side whose king was gone with
        everything else.
        """
        engine = Engine(self._royals(win="none"))
        stripped = self._cleared(engine, engine.initial_state(), 1)
        assert engine.terminal_prefix(stripped) is None


class TestDirectionWalkOrder:
    """The compiled tables are a function of the spec, not of the interpreter.

    `compile._descriptor_directions` hands back a `set` of `(dx, dy)` tuples, and a set
    of tuples iterates in hash order. CPython builds a tuple's hash from `Py_uhash_t`,
    which is 64 bits on this machine and 32 under Emscripten, so that walk order is a
    property of the interpreter. A `lame` leap is appended to the slide table as its
    direction comes off the walk, rather than merged under a sorted key the way every
    other primitive is, so before the sort landed the walk order reached a table: xiangqi
    compiled a permuted `slide_squares` under Pyodide and its games parted from the
    server's at ply 1 (`docs/reference/browser-core-probe.md`).

    **A different `PYTHONHASHSEED` proves nothing here**, because a tuple of small ints
    hashes from the ints' own hashes and those are not seeded. The two tests below stand
    other orders in hash order's place instead.
    """

    def spec(self) -> dict[str, Any]:
        """A lame leaper with eight directions, and a royal so the game is a game."""
        return {
            "schema_version": 2,
            "name": "lame-order",
            "board": {"width": 5, "height": 5},
            "pieces": [
                {"id": "K", "royal": True,
                 "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]],
                            "symmetry": "rotate4"}]},
                {"id": "N",
                 "moves": [{"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8"}]},
            ],
            "setup": {"mirror": True, "placements": [
                {"piece": "K", "squares": ["a1"]},
                {"piece": "N", "squares": ["c3"]},
            ]},
            "rules": {"win": "checkmate"},
        }

    def test_the_walk_order_reaches_the_slide_table(self) -> None:
        """The order, pinned against the spec, and what makes the next test not vacuous.

        A lame leap compiles to a two-square slide -- the leg it can be blocked at, then
        the target -- so each pair below is one of the eight directions `dihedral8` makes
        of `(1, 2)`, read from c3 on a 5 by 5 board. They come out in ascending direction
        order, `(-2, -1)` first and `(2, 1)` last, which is a property of the spec and of
        nothing else. Take the sort away and this list follows the interpreter's tuple
        hash instead.
        """
        compiled = compile_spec(self.spec())
        knight = 1
        c3 = 2 * 5 + 2
        assert [entry[0] for entry in compiled.slides[0][knight][c3]] == [
            (11, 5),    # (-2, -1), leg b3
            (11, 15),   # (-2,  1), leg b3
            (7, 1),     # (-1, -2), leg c2
            (17, 21),   # (-1,  2), leg c4
            (7, 3),     # ( 1, -2), leg c2
            (17, 23),   # ( 1,  2), leg c4
            (13, 9),    # ( 2, -1), leg d3
            (13, 19),   # ( 2,  1), leg d3
        ]

    def test_two_different_walk_orders_compile_to_one_table(self) -> None:
        """The property itself, proved without reaching the interpreter's hash.

        Hash order cannot be changed from a test, so this puts two *other* orders in its
        place -- the sorted directions reversed, and rotated by one -- and requires both
        compiles to equal the unpatched one. Reversing and rotating eight directions give
        two different sequences, and no third sequence equals both, so if the sort at the
        walk goes then at least one of these two compiles differs from the unpatched one,
        whatever hash order happens to be on the machine running the test.
        """
        spec = self.spec()
        reference = compile_spec(spec)
        original = compile_module._descriptor_directions

        for permute in (lambda items: items[::-1], lambda items: items[1:] + items[:1]):
            def walk(vectors: Any, symmetry: str, facing: int, _p: Any = permute) -> Any:
                return _p(sorted(original(vectors, symmetry, facing)))

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(compile_module, "_descriptor_directions", walk)
                assert compile_spec(spec) == reference


# ------------------------------------------------ Q922: which descriptors can double-step


_SLIDE = {"kind": "slide", "vectors": [[0, 1]]}


@pytest.mark.parametrize(
    ("piece", "can"),
    [
        ({"moves": []}, False),
        ({"moves": [{"kind": "leap", "vectors": [[0, 2]]}]}, False),
        ({"moves": [{"kind": "hop", "vectors": [[0, 1]]}]}, False),
        ({"moves": [{"kind": "sweep", "vectors": [[0, 1]]}]}, False),
        ({"moves": [{"kind": "lame", "vectors": [[1, 2]]}]}, True),
        ({"moves": [{"kind": "teleport", "vectors": [[0, 1]]}]}, True),  # a kind not taught
        ({"moves": [_SLIDE]}, True),                                      # unlimited range
        ({"moves": [{**_SLIDE, "range": 1}]}, False),
        ({"moves": [{**_SLIDE, "range": 2}]}, True),
        ({"moves": [{**_SLIDE, "range": 2, "min_range": 2}]}, True),
        ({"moves": [{**_SLIDE, "range": 2, "min_range": 3}]}, False),
        ({"moves": [{**_SLIDE, "range": 1, "min_range": 3}]}, False),
        ({"moves": [{**_SLIDE, "range": 2, "mode": "capture"}]}, False),
        ({"moves": [{**_SLIDE, "range": 2, "mode": "move"}]}, True),
        ({"moves": [{**_SLIDE, "range": True}]}, True),                   # malformed: refused elsewhere
        ({"moves": [{**_SLIDE, "range": "2"}]}, True),
        ({"moves": [{**_SLIDE, "range": 1, "min_range": "1"}]}, True),
        ({"moves": [{**_SLIDE, "range": 1, "mode": "capture"}, _SLIDE]}, True),
        ({"moves": [{"kind": "leap", "vectors": [[0, 1]]}, {**_SLIDE, "range": 1}]}, False),
        ({"moves": ["junk"]}, True),
        ({"moves": "slide"}, True),
        ({}, True),
    ],
)
def test_a_piece_can_double_step_when_a_slide_has_a_second_square(
    piece: dict, can: bool
) -> None:
    from chess_variant_sandbox.spec.schema import can_double_step

    assert can_double_step(piece) is can
