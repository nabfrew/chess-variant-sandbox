"""The result check (Q404): both engines asked who has won, at the same positions.

`verify` compares move lists and never asks who won, so a translation that got a win or
draw rule wrong passes it. `chess_variant_sandbox/fairy/results.py` asks both engines instead: ours
through `Engine.terminal`, theirs through `pyffish`. These tests put each result key the
check covers into a position where it decides the game and ask both engines about it.
They also show the check failing on a mistranslation that `verify` passes.

Q411 made the two engines reset the no-progress clock on the same moves where an option
says so, and noted the two cases no option covers. Those tests are here too, with the
premise that the option is what makes the difference. Q412 made our core look for mate
before the clocks, so a mate on the clock's last ply is now a mate in both wherever their
royal is in the king slot, and the note moved to atomic, where it is a draw in theirs.

Every test here that needs `pyffish` skips without it. Each asks it a few questions at
about 20 ms apiece, so all are fast except the corpus-wide run, which is marked slow.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import pytest
from fairy_shogi import SHOGI_FORWARD, shogi_forward

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy import position
from chess_variant_sandbox.fairy.results import (
    RESULT_KEYS,
    Ending,
    admits,
    check_results,
    in_play,
    verdicts,
)
from chess_variant_sandbox.fairy.variant import Variant, translate
from chess_variant_sandbox.spec import load_spec

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def _committed(name: str) -> Path:
    """A ruleset's file, in `rulesets/` or, for one moved to the Library (Q512), `archive/`."""
    path = RULESETS / f"{name}.json"
    return path if path.exists() else RULESETS.parent / "archive" / f"{name}.json"

# The ten `rules` keys that decide a result, wholly or in part (Q96).
RESULT_RULES = {
    "checkmate",
    "stalemate",
    "max_plies",
    "no_progress_draw",
    "repetition_draw",
    "extinction",
    "reach",
    "check_limit",
    "win",
    "explode_on_capture",
}

FOOLS_MATE = ["f2f3", "e7e5", "g2g4", "d8h4"]

# Every committed ruleset that translates: `tests/test_fairy.py`'s `TRANSLATES`, whose
# test holds the list to the corpus. `sardines` is the one in `archive/` (moved there by
# Q512). Shogi was refused from Q438 until Q454 put `on_leaving` on its entries; the
# nearest shogi that asks neither of its two rules (`fairy_shogi`) stays beside it, for its
# shape alone: a hand that holds a piece that is not one-way, and one-way pieces off the
# pawn slot.
TRANSLATES = (
    "almost", "antichess", "atomic", "bastion", "berolina", "capablanca", "chess",
    "courier", "crazyhouse", "dobutsu", "dull", "gardner", "horde", "kingofthehill",
    "knightmate", "losalamos", "makruk", "racingkings", "sardines", "shatranj",
    "shogi", SHOGI_FORWARD, "threecheck", "xiangqi",
)


def _pyffish() -> Any:
    return pytest.importorskip("pyffish", reason="pyffish is an optional dev dependency")


def _spec(name: str, **rules: Any) -> dict[str, Any]:
    """A committed ruleset, with some rules replaced at a schema version that has them.

    ``SHOGI_FORWARD`` is not a file: it is the nearest shogi that translates (Q438).
    """
    spec = shogi_forward() if name == SHOGI_FORWARD else load_spec(_committed(name))
    if rules:
        spec["rules"].update(rules)
        spec["schema_version"] = max(spec.get("schema_version", 1), 2)
    return spec


def _referee(written: Variant) -> Any:
    from chess_variant_sandbox.fairy.verify import PyffishBackend

    _pyffish()
    backend = PyffishBackend()
    backend.load(written, "")
    return backend


class _Both:
    """One ruleset in both engines: our core, its translation, and `pyffish` loaded with it."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.engine = Engine(spec)
        self.variant = translate(spec)
        self.referee = _referee(self.variant)

    def play(self, moves: list[str], state: GameState | None = None) -> GameState:
        """``moves`` played in our engine, each looked up among its legal moves."""
        state = state or self.engine.initial_state()
        for uci in moves:
            legal = self.engine.legal_moves(state)
            move = position.move_of(
                self.engine.spec, state, legal, uci, self.variant.chars,
                self.variant.promotion_suffix,
            )
            state = self.engine.apply(state, move)
        return state

    def place(self, pieces: dict[str, tuple[str, int]], side_to_move: int) -> GameState:
        return self.engine.make_state(pieces, side_to_move=side_to_move)

    def holding(self, piece: str) -> GameState:
        """The opening, with one ``piece`` in the first player's hand."""
        opening = self.engine.initial_state()
        hand = [0] * self.engine.spec.n_types
        hand[self.engine.spec.piece_ids.index(piece)] = 1
        return opening.with_(hands=(tuple(hand), opening.hands[1]))

    def ask(
        self, state: GameState, start: GameState | None = None, moves: list[str] | None = None
    ) -> tuple[Ending, Ending]:
        return verdicts(self.engine, self.variant, self.referee, state, start, moves or [])

    def one_short(self, state: GameState) -> GameState:
        """``state`` with the no-progress clock one ply short of our limit."""
        return state.with_(halfmove=self.engine.spec.no_progress_draw - 1)


def _agree(ours: Ending, theirs: Ending, winner: int | None, reason: str) -> None:
    """Both say the game is over, for ``winner``, and ours says it is over by ``reason``."""
    assert ours == Ending(over=True, winner=winner, how=reason), ours
    assert theirs.over and theirs.winner == winner, theirs


class TestTheCheckCoversEachResultKey:
    """One position per key, where the key decides the game, asked of both engines."""

    def test_every_ending_of_ours_names_one_of_the_ten_keys(self) -> None:
        assert set(RESULT_KEYS.values()) == RESULT_RULES

    def test_a_mate_is_a_loss_in_both_and_by_their_moves_too(self) -> None:
        both = _Both(_spec("chess"))
        state = both.play(FOOLS_MATE)
        _agree(*both.ask(state), winner=1, reason="checkmate")
        _agree(*both.ask(state, both.engine.initial_state(), FOOLS_MATE), 1, "checkmate")

    def test_a_mate_that_wins_is_a_win_in_both(self) -> None:
        both = _Both(_spec("chess", checkmate="win"))
        _agree(*both.ask(both.play(FOOLS_MATE)), winner=0, reason="checkmate")

    @pytest.mark.parametrize(
        ("value", "winner", "reason"),
        [("draw", None, "stalemate"), ("loss", 0, "stalemate_loss"), ("win", 1, "stalemate_win")],
    )
    def test_a_stalemate_ends_the_same_way_in_both(
        self, value: str, winner: int | None, reason: str
    ) -> None:
        """The second player, to move, has no move and is not in check."""
        both = _Both(_spec("chess", stalemate=value))
        state = both.place({"h8": ("K", 1), "f7": ("Q", 0), "g6": ("K", 0)}, side_to_move=1)
        _agree(*both.ask(state), winner=winner, reason=reason)

    def test_the_ply_limit_is_ours_alone_and_the_translation_says_so(self) -> None:
        both = _Both(_spec("chess"))
        limit = both.engine.spec.max_plies
        opening = both.engine.initial_state()
        ours, theirs = both.ask(opening.with_(ply=limit - 1))
        assert not ours.over and not theirs.over
        at_limit = opening.with_(ply=limit)
        ours, theirs = both.ask(at_limit)
        assert ours == Ending(over=True, winner=None, how="ply_limit")
        assert not theirs.over
        assert "max_plies" in both.variant.approximates
        assert admits(both.engine, both.variant, at_limit, ours, theirs) == "max_plies"

    def test_the_no_progress_clock_runs_out_on_the_same_ply(self) -> None:
        """A quiet move from one ply short draws in both; a pawn move resets both."""
        both = _Both(_spec("chess"))
        short = both.engine.initial_state().with_(halfmove=99)
        assert not any(end.over for end in both.ask(short))
        _agree(*both.ask(short.with_(halfmove=100)), winner=None, reason="no_progress")
        _agree(*both.ask(both.play(["g1f3"], short), short, ["g1f3"]), None, "no_progress")
        ours, theirs = both.ask(both.play(["e2e4"], short), short, ["e2e4"])
        assert not ours.over and not theirs.over

    def test_an_odd_no_progress_clock_is_rounded_up_there_and_admitted(self) -> None:
        both = _Both(_spec("chess", no_progress_draw=99))
        state = both.engine.initial_state().with_(halfmove=99)
        ours, theirs = both.ask(state)
        assert ours.how == "no_progress" and not theirs.over
        assert "no_progress_draw" in both.variant.approximates
        assert admits(both.engine, both.variant, state, ours, theirs) == "no_progress_draw"

    @pytest.mark.parametrize("fold", [3, 4])
    def test_a_repetition_is_counted_over_the_same_moves_by_both(self, fold: int) -> None:
        """The check's own probe: two knights back and forth from the opening."""
        spec = _spec("chess", repetition_draw=fold)
        result = check_results(spec, _referee(translate(spec)), games=0)
        assert result.ok, result.summary()
        assert result.agreed.get("repetition") == 1
        assert "repetition_draw" not in result.unreached

    @pytest.mark.parametrize(
        ("name", "pieces", "side_to_move", "winner"),
        [
            # Running out wins in antichess, and the second player has nothing left.
            ("antichess", {"b1": ("N", 0)}, 1, 1),
            # Running out loses in horde, and the first player has nothing left.
            ("horde", {"e8": ("K", 1)}, 0, 1),
        ],
    )
    def test_running_out_ends_the_same_way_in_both(
        self, name: str, pieces: dict[str, tuple[str, int]], side_to_move: int, winner: int
    ) -> None:
        both = _Both(_spec(name))
        _agree(*both.ask(both.place(pieces, side_to_move)), winner, "extinction")

    def test_arriving_wins_in_both(self) -> None:
        both = _Both(_spec("kingofthehill"))
        state = both.place({"e4": ("K", 0), "e8": ("K", 1)}, side_to_move=1)
        _agree(*both.ask(state), winner=0, reason="reached")

    @pytest.mark.parametrize("side_to_move", [0, 1])
    def test_a_safe_arrival_wins_in_both_whoever_is_to_move(self, side_to_move: int) -> None:
        both = _Both(_spec("dobutsu"))
        state = both.place({"b4": ("L", 0), "a2": ("L", 1)}, side_to_move=side_to_move)
        _agree(*both.ask(state), winner=0, reason="reached")

    def test_an_attacked_arrival_wins_in_neither(self) -> None:
        both = _Both(_spec("dobutsu"))
        state = both.place({"b4": ("L", 0), "a2": ("L", 1), "c4": ("G", 1)}, side_to_move=1)
        ours, theirs = both.ask(state)
        assert not ours.over and not theirs.over

    def test_the_last_check_wins_in_both_and_by_their_count_too(self) -> None:
        """Two checks already given, and the third played by each engine itself."""
        both = _Both(_spec("threecheck"))
        before = both.play(["e2e4", "f7f6"]).with_(checks=(2, 0))
        after = both.play(["d1h5"], before)
        _agree(*both.ask(after), winner=0, reason="check_limit")
        _agree(*both.ask(after, before, ["d1h5"]), winner=0, reason="check_limit")

    @pytest.mark.parametrize(
        ("name", "pieces", "reason"),
        [
            ("dobutsu", {"a1": ("G", 0), "a3": ("L", 1)}, "royal_captured"),
            ("dull", {"a5": ("G", 1)}, "annihilated"),
        ],
    )
    def test_the_win_condition_ends_the_same_way_in_both(
        self, name: str, pieces: dict[str, tuple[str, int]], reason: str
    ) -> None:
        """The first player has lost its royal, or every piece, and is to move."""
        both = _Both(_spec(name))
        _agree(*both.ask(both.place(pieces, side_to_move=0)), winner=1, reason=reason)

    def test_a_blast_that_takes_the_royal_wins_in_both_and_by_their_moves_too(self) -> None:
        """1.Nf3 a6 2.Ne5 a5 3.Nxf7, and the black king beside f7 goes with it."""
        both = _Both(_spec("atomic"))
        moves = ["g1f3", "a7a6", "f3e5", "a6a5", "e5f7"]
        state = both.play(moves)
        _agree(*both.ask(state), winner=0, reason="royal_destroyed")
        _agree(*both.ask(state, both.engine.initial_state(), moves), 0, "royal_destroyed")


def _opening(both: _Both) -> GameState:
    return both.engine.initial_state()


def _back_rank_mate(both: _Both) -> tuple[GameState, GameState]:
    """1.Ra8, a mate by a quiet move, played on the ply our clock runs out."""
    before = both.place(
        {"h8": ("K", 1), "g7": ("P", 1), "h7": ("P", 1), "a1": ("R", 0), "c1": ("K", 0)},
        side_to_move=0,
    )
    short = both.one_short(before)
    return short, both.play(["a1a8"], short)


# A quiet move or drop of a one-way piece, one ply before our clock runs out: the
# ruleset, the position before it, and the move. Our clock resets on each (Q404).
ONE_WAY_MOVES = {
    "dull's only piece": ("dull", _opening, "b1b2"),
    "berolina's pawn": ("berolina", _opening, "a2b3"),
    "shogi's pawn": (SHOGI_FORWARD, _opening, "a3a4"),
    "shogi's lance": (SHOGI_FORWARD, _opening, "a1a2"),
    "shogi's knight": (SHOGI_FORWARD, lambda both: both.play(["c3c4", "c7c6"]), "b1c3"),
    "dobutsu's chick": (
        "dobutsu",
        lambda both: both.place({"b1": ("L", 0), "b4": ("L", 1), "a2": ("C", 0)}, 0),
        "a2a3",
    ),
    "a crazyhouse pawn dropped": ("crazyhouse", lambda both: both.holding("P"), "P@e4"),
    "chess's pawn, their pawn slot": ("chess", _opening, "e2e4"),
}

# A drop of a piece that is not one-way, one ply before our clock runs out.
TWO_WAY_DROPS = {
    "a crazyhouse knight": ("crazyhouse", "N", "N@e4"),
    "a shogi silver": (SHOGI_FORWARD, "S", "S@e5"),
    "a dobutsu giraffe": ("dobutsu", "G", "G@a2"),
}


class TestWhichMovesResetTheNoProgressClock:
    """Q404 found the two engines resetting the clock on different moves; Q411 closed it.

    Ours resets on a capture, a promotion and any move or drop of a one-way piece.
    Theirs resets on a capture, a promotion, a move of an `nMoveRuleTypes` piece (by
    default its pawn slot) and any drop. The translation writes our one-way pieces into
    `nMoveRuleTypes`. What no option reaches is noted: a drop of any other piece, and a
    mate on the clock's last ply where their royal is pseudo-royal. The premises are
    pinned too, so a build that changes either rule fails a test here, and the record's
    corpus report should be run again.
    """

    @pytest.mark.parametrize("case", list(ONE_WAY_MOVES), ids=list(ONE_WAY_MOVES))
    def test_a_one_way_piece_resets_both_clocks(self, case: str) -> None:
        name, before, move = ONE_WAY_MOVES[case]
        both = _Both(_spec(name))
        short = both.one_short(before(both))
        after = both.play([move], short)
        assert after.halfmove == 0
        ours, theirs = both.ask(after, short, [move])
        assert not ours.over and not theirs.over, (ours, theirs)

    @pytest.mark.parametrize(
        ("name", "line", "move"),
        [
            ("dull", "nMoveRuleTypes = G\n", "b1b2"),
            (SHOGI_FORWARD, "nMoveRuleTypes = LNP\n", "a3a4"),
        ],
    )
    def test_without_the_line_a_one_way_piece_resets_ours_and_not_theirs(
        self, name: str, line: str, move: str
    ) -> None:
        """The premise: their default does not reset on these pieces (Q404's finding)."""
        spec = _spec(name)
        engine = Engine(spec)
        written = translate(spec)
        without = _mistranslated(written, line, "")
        short = engine.initial_state().with_(halfmove=engine.spec.no_progress_draw - 1)
        spelled = position.move_of(
            engine.spec, short, engine.legal_moves(short), move, written.chars,
            written.promotion_suffix,
        )
        after = engine.apply(short, spelled)
        ours, theirs = verdicts(engine, without, _referee(without), after, short, [move])
        assert not ours.over
        assert theirs == Ending(over=True, winner=None, how="a draw may be claimed")

    @pytest.mark.parametrize("case", list(TWO_WAY_DROPS), ids=list(TWO_WAY_DROPS))
    def test_a_drop_that_is_not_one_way_resets_theirs_and_not_ours_as_noted(
        self, case: str
    ) -> None:
        """No option reaches this, so the translation notes it and the check admits it."""
        name, piece, move = TWO_WAY_DROPS[case]
        both = _Both(_spec(name))
        short = both.one_short(both.holding(piece))
        after = both.play([move], short)
        ours, theirs = both.ask(after, short, [move])
        assert ours == Ending(over=True, winner=None, how="no_progress")
        assert not theirs.over
        assert admits(both.engine, both.variant, after, ours, theirs) == "no_progress_draw"
        assert any(repr(piece) in note and "dropping" in note for note in both.variant.notes)

    def test_a_mate_on_the_clocks_last_ply_is_a_mate_in_both(self) -> None:
        """Both look for mate before the clock: theirs lets a mated side not claim (Q412).

        Q411's premise, flipped. Until Q412 ours asked the clocks before the move list,
        so this was a draw here and a mate there, and the 18 translations with a king
        slot carried a note admitting it. Now nothing is admitted: they agree.
        """
        both = _Both(_spec("chess"))
        short, after = _back_rank_mate(both)
        assert after.halfmove == both.engine.spec.no_progress_draw
        for ours, theirs in (both.ask(after), both.ask(after, short, ["a1a8"])):
            _agree(ours, theirs, winner=0, reason="checkmate")
            assert theirs == Ending(over=True, winner=0, how="no legal move")
        assert "checkmate" not in both.variant.approximates
        assert not [note for note in both.variant.notes if "checkmate" in note]

    def test_a_pseudo_royal_is_never_mated_there_so_their_clock_draws_our_mate_as_noted(
        self,
    ) -> None:
        """Atomic's royal is not in their king slot, so they see no check and claim first.

        Before Q412 both engines drew this, the clock coming first in each. Now ours
        finds the mate, and the difference is noted and admitted in its own shape: theirs
        with the clocks wound back calls it the same mate.
        """
        both = _Both(_spec("atomic"))
        short, after = _back_rank_mate(both)
        for ours, theirs in (both.ask(after), both.ask(after, short, ["a1a8"])):
            assert ours == Ending(over=True, winner=0, how="checkmate")
            assert theirs == Ending(over=True, winner=None, how="a draw may be claimed")
            assert admits(both.engine, both.variant, after, ours, theirs, both.referee) == (
                "checkmate"
            )
            # Their wound-back answer is theirs to give, so without them nothing is.
            assert admits(both.engine, both.variant, after, ours, theirs) == ""
        # With the clock wound back, it is the same mate in both.
        _agree(*both.ask(after.with_(halfmove=0)), winner=0, reason="checkmate")
        assert "checkmate" in both.variant.approximates


def _mistranslated(good: Variant, old: str, new: str) -> Variant:
    """``good`` with one line of its definition changed, under a name of its own.

    A name of its own because loading a definition under a name already registered does
    not replace it, so the broken variant would silently be the good one. `pyffish` is
    one per process, so the name is also this change's alone: `test_fairy.py` breaks
    chess under ``<name>broken``, and sharing that name made whichever test ran second
    check the other's definition.
    """
    name = good.name + "q404" + hashlib.sha1(f"{old}|{new}".encode()).hexdigest()[:6]
    config = good.config.replace(f"[{good.name}]", f"[{name}]")
    assert old in config
    return dataclasses.replace(good, name=name, config=config.replace(old, new))


class _Substituting:
    """A `PyffishBackend` that loads a mistranslated variant in place of the real one.

    Both `verify` and `check_results` translate the spec themselves; this is where the
    broken definition gets in, the same way for both.
    """

    def __init__(self, old: str, new: str) -> None:
        from chess_variant_sandbox.fairy.verify import PyffishBackend

        _pyffish()
        self._backend = PyffishBackend()
        self._change = (old, new)

    def load(self, written: Variant, config_path: str) -> None:
        self._backend.load(_mistranslated(written, *self._change), config_path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


class TestAMistranslatedResultIsCaughtHereAndNotByVerify:
    """The check has to be able to fail where `verify` cannot, or it adds nothing."""

    def test_a_flipped_stalemate_passes_verify_and_fails_the_result_check(self) -> None:
        """Antichess's stalemate wins; written as a loss, every move is still the same.

        Its second game, the one that takes an ending when offered, ends in a stalemate
        with seed 0, so two games are enough to reach one.
        """
        from chess_variant_sandbox.fairy.verify import verify

        spec = _spec("antichess")
        flipped = _Substituting("stalemateValue = win", "stalemateValue = loss")
        moves = verify(spec, flipped, "", depth=1, games=1, plies=20)
        assert moves.ok, moves.summary()

        result = check_results(spec, flipped, games=2)
        assert not result.ok
        keys = {difference.key for difference in result.differences}
        assert keys == {"stalemate"}, result.summary()
        assert "stalemate_win" in result.differences[0].ours

    def test_the_real_translation_passes_the_same_check(self) -> None:
        spec = _spec("antichess")
        result = check_results(spec, _referee(translate(spec)), games=2)
        assert result.ok, result.summary()
        assert result.agreed.get("stalemate_win", 0) >= 1

    def test_an_admission_is_no_wider_than_its_note(self) -> None:
        """Chess admits ``max_plies``, in its note's shape only.

        A mate that theirs scores the other way is reported, and so is a no-progress
        clock twice as long: the note is not about theirs playing on past our limit
        (Q411). Until Q412 chess admitted ``checkmate`` too, in the shape of a mate on the
        clock's last ply, and the flipped mate was reported through it.
        """
        spec = _spec("chess")
        both = _Both(spec)
        assert both.variant.approximates == frozenset({"max_plies"})

        flipped = _mistranslated(
            both.variant, "nMoveRule = 50", "nMoveRule = 50\ncheckmateValue = win"
        )
        mated = both.play(FOOLS_MATE)
        ours, theirs = verdicts(both.engine, flipped, _referee(flipped), mated)
        assert ours.how == "checkmate" and theirs.over and theirs.winner == 0
        assert admits(both.engine, flipped, mated, ours, theirs) == ""
        result = check_results(
            spec, _Substituting("nMoveRule = 50", "nMoveRule = 50\ncheckmateValue = win"), games=2
        )
        assert {difference.key for difference in result.differences} == {"checkmate"}

        doubled = _Substituting("nMoveRule = 50", "nMoveRule = 100")
        result = check_results(spec, doubled, games=0)
        assert not result.ok
        assert {difference.key for difference in result.differences} == {"no_progress_draw"}
        assert check_results(spec, both.referee, games=0).ok

    def test_a_mate_their_clock_draws_is_admitted_only_as_the_same_mate(self) -> None:
        """Atomic admits ``checkmate``: a mate here that their clock draws, and no other.

        A mate that theirs scores the other way is reported, at the clock's last ply as
        well as before it, though the key is admitted: wound back, theirs has to call it
        the same mate, not merely a mate (Q412).
        """
        both = _Both(_spec("atomic"))
        assert "checkmate" in both.variant.approximates
        flipped = _mistranslated(
            both.variant, "nMoveRule = 50", "nMoveRule = 50\ncheckmateValue = win"
        )
        referee = _referee(flipped)
        _, after = _back_rank_mate(both)
        for state in (after, after.with_(halfmove=0)):
            ours, theirs = verdicts(both.engine, flipped, referee, state)
            assert ours == Ending(over=True, winner=0, how="checkmate")
            assert not ours.same(theirs), theirs
            assert admits(both.engine, flipped, state, ours, theirs, referee) == ""
        # The real translation's answer at the clock is admitted, as the note says.
        ours, theirs = both.ask(after)
        assert admits(both.engine, both.variant, after, ours, theirs, both.referee) == (
            "checkmate"
        )

    def test_a_flipped_stalemate_is_caught_at_the_position_itself(self) -> None:
        """The same mistranslation in chess, asked at a constructed stalemate."""
        spec = _spec("chess")
        engine = Engine(spec)
        broken = _mistranslated(translate(spec), "stalemateValue = draw", "stalemateValue = loss")
        state = engine.make_state({"h8": ("K", 1), "f7": ("Q", 0), "g6": ("K", 0)}, side_to_move=1)
        ours, theirs = verdicts(engine, broken, _referee(broken), state)
        assert ours == Ending(over=True, winner=None, how="stalemate")
        assert theirs.over and theirs.winner == 0
        assert not ours.same(theirs)


class TestWhatTheTranslationAdmits:
    """`Variant.approximates` is written beside the note it stands for (no engine)."""

    @pytest.mark.parametrize("name", TRANSLATES)
    def test_the_ply_limit_is_admitted_everywhere(self, name: str) -> None:
        assert "max_plies" in translate(_spec(name)).approximates

    def test_an_odd_clock_and_a_reach_that_does_not_win_are_admitted(self) -> None:
        assert "no_progress_draw" in translate(_spec("chess", no_progress_draw=99)).approximates
        reach = dict(_spec("kingofthehill")["rules"]["reach"], result="draw")
        assert "reach" in translate(_spec("kingofthehill", reach=reach)).approximates

    @pytest.mark.parametrize(
        ("name", "line"),
        [
            ("chess", None), ("xiangqi", None), ("racingkings", None),
            ("berolina", "P"), ("dobutsu", "C"), ("dull", "G"), (SHOGI_FORWARD, "LNP"),
        ],
    )
    def test_the_one_way_pieces_are_written_where_they_are_not_the_pawn_slot(
        self, name: str, line: str | None
    ) -> None:
        """Their default is the pawn slot, which is the whole one-way set in chess.

        Xiangqi and racing kings have no one-way piece and nothing in the pawn slot.
        """
        config = translate(_spec(name)).config.splitlines()
        written = [text for text in config if text.startswith("nMoveRuleTypes")]
        assert written == ([] if line is None else [f"nMoveRuleTypes = {line}"])

    def test_a_drop_that_is_not_one_way_is_noted_where_a_hand_can_hold_one(self) -> None:
        for name in TRANSLATES:
            noted = "no_progress_draw" in translate(_spec(name)).approximates
            assert noted is (name in {"crazyhouse", "dobutsu", "shogi", SHOGI_FORWARD}), name

    def test_a_mate_on_the_clocks_last_ply_is_noted_only_where_the_royal_is_pseudo_royal(
        self,
    ) -> None:
        """Only where a mate is possible and their royal is never in check: atomic.

        Until Q412 it was the other way round: every translation with a mate carried the
        note except atomic.
        """
        noted = set()
        for name in TRANSLATES:
            spec = _spec(name)
            if "checkmate" in translate(spec).approximates:
                noted.add(name)
                assert "checkmate" in in_play(Engine(spec).spec), name
                assert "explode_on_capture" in spec["rules"], name
        assert noted == {"atomic"}

    def test_every_admitted_key_is_a_result_rule_with_a_note(self) -> None:
        for spec in (
            _spec("chess", no_progress_draw=99), _spec("dobutsu"), _spec(SHOGI_FORWARD),
            _spec("crazyhouse"),
        ):
            written = translate(spec)
            assert written.approximates <= RESULT_RULES
            assert len(written.notes) >= len(written.approximates)

    def test_the_ply_limit_admits_only_what_ours_says_without_it(self) -> None:
        """Their answer at our limit must be ours with the clocks wound back."""
        spec = _spec("chess")
        engine, written = Engine(spec), translate(spec)
        state = engine.initial_state().with_(ply=engine.spec.max_plies)
        ours = Ending.ours(engine.terminal(state))
        assert admits(engine, written, state, ours, Ending(over=False)) == "max_plies"
        won = Ending(over=True, winner=0, how="ended")
        assert admits(engine, written, state, ours, won) == ""
        assert admits(engine, written, state.with_(ply=0), Ending(over=False), won) == ""

    def test_the_keys_in_play_follow_the_rules(self) -> None:
        assert "checkmate" in in_play(Engine(_spec("chess")).spec)
        assert "checkmate" not in in_play(Engine(_spec("racingkings")).spec)
        dobutsu = in_play(Engine(_spec("dobutsu")).spec)
        assert {"win", "reach"} <= set(dobutsu) and "checkmate" not in dobutsu
        assert "explode_on_capture" in in_play(Engine(_spec("atomic")).spec)


@pytest.mark.slow
@pytest.mark.parametrize("name", TRANSLATES)
def test_every_translation_agrees_on_the_result_or_differs_only_as_noted(name: str) -> None:
    """The whole claim, at `chess-variant-sandbox fairy`'s own depth, one ruleset at a time.

    Every translation agrees since Q411, apart from differences its notes admit. Against
    `pyffish` 0.0.90 the corpus run admits the ply limit in all 22 and a non-one-way drop
    in crazyhouse and dobutsu. Q411's run also admitted a mate on the clock's last ply in
    shogi, which since Q412 is a mate in both. Shogi was refused from Q438 to Q454, when
    the nearest shogi that translates was checked in its place; since Q454 both are.
    Shogi's translation notes the ply limit and a non-one-way drop, and a run by hand in
    Q454 agreed at 66 positions against `pyffish` 0.0.90, admitting the ply limit. A ruleset that starts to differ fails here, and the
    record's corpus table is then out of date.
    """
    spec = _spec(name)
    written = translate(spec)
    result = check_results(spec, _referee(written))
    assert result.ok, result.summary()
    assert "max_plies" in result.admitted
    assert set(result.admitted) <= written.approximates


class TestTheStockEnginesNewTranslationsEndAsOurs:
    """Q703: the results of what crossed in place of a refusal (`general-fairy-fork.md` §1.3)."""

    @staticmethod
    def _perpetual() -> dict[str, Any]:
        return load_spec(RULESETS.parent / "archive" / "perpetual.json")

    def test_no_repetition_is_in_play_under_never(self) -> None:
        assert "repetition_draw" not in in_play(Engine(self._perpetual()).spec)
        assert "repetition_draw" in in_play(Engine(_spec("chess")).spec)

    def test_a_position_repeated_past_three_folds_goes_on_in_both(self) -> None:
        spec = self._perpetual()
        result = check_results(spec, _referee(translate(spec)), games=0)
        assert result.ok, result.summary()
        assert "repetition_draw" not in result.unreached

    def test_a_fold_rule_left_on_there_is_caught(self) -> None:
        """The probe has to be able to fail: the default three folds, written back in."""
        spec = self._perpetual()
        result = check_results(spec, _Substituting("nFoldRule = 0", "nFoldRule = 3"), games=0)
        assert not result.ok
        assert {difference.key for difference in result.differences} == {"repetition_draw"}

    def test_five_checks_are_counted_alike(self) -> None:
        spec = _spec("threecheck", check_limit=5)
        result = check_results(spec, _referee(translate(spec)), games=2)
        assert result.ok, result.summary()
        assert result.agreed.get("check_limit", 0) >= 1


@pytest.mark.slow
@pytest.mark.parametrize("name", ["perpetual", "quarry-vaulting", "three-stones-takeable"])
def test_each_new_translation_agrees_on_the_result(name: str) -> None:
    """Q703's games at the `fairy` command's result depth (`DEFAULT_GAMES`)."""
    import copy

    if name == "perpetual":
        spec = load_spec(RULESETS.parent / "archive" / "perpetual.json")
    elif name == "quarry-vaulting":
        spec = load_spec(RULESETS.parent / "archive" / "quarry.json")
        spec["rules"]["obstacles"]["blocks_leaps"] = False
    else:
        spec = copy.deepcopy(_spec("chess"))
        spec["schema_version"] = 10
        spec["pieces"].append({"id": "S", "name": "stone", "value": 1, "moves": []})
        spec["rules"]["drops"] = {"from_captures": False, "stock": {"S": 3}}
    written = translate(spec)
    result = check_results(spec, _referee(written))
    assert result.ok, result.summary()
    assert set(result.admitted) <= written.approximates
