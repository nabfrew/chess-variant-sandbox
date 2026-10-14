"""The Fairy-Stockfish translation, and whether it plays the game it claims to.

Three layers, and each one is checked against something outside this repository
wherever it can be.

**Betza, against the parser it is written for.** `chess_variant_sandbox/fairy/betza.py` is a port of
`piece.cpp::from_betza` written backwards, so the honest test is to hand the notation to
Fairy-Stockfish and ask what it thinks the piece does. That needs a Fairy-Stockfish, and
the one used here is `pyffish` -- the same rules code as a library, installable from
PyPI, needing no binary and no ``PATH``. It is an optional dependency and every test
that needs it skips without it.

**The whole translation, against both engines at once.** `verify` plays our engine and
theirs over the same tree and the same random games. That is the check that matters and
the only one that could have caught what it did catch: a promotion the engine declined
to make, a double step from a rank nobody wrote, a pawn pushed off the end of the board.
None of it is visible by reading a variant definition.

**The parts that need no engine at all**, which is most of the refusals. A ruleset that
wraps has no counterpart over there whatever version is installed, and saying so is a
fact about the two rule systems rather than about a build.

The committed expectations are deliberately *lower* bounds -- "these rulesets translate"
and "these do not" -- rather than a fixed count. Which rulesets *verify* depends on the
engine build, and pinning that number would make the test a statement about somebody's
installation. `chess-variant-sandbox fairy` is where the current answer lives.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from fairy_shogi import SHOGI_ARRIVING, SHOGI_FORWARD, shogi_arriving, shogi_forward
from retired_specs import REBUILT, hex_glinski_promoting

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.fairy import betza, position, variant
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.complexity import expand_symmetry

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def _pyffish() -> Any:
    """The library, or a skip. Imported lazily so collection never needs it."""
    return pytest.importorskip("pyffish", reason="pyffish is an optional dev dependency")


def _spec(name: str) -> dict[str, Any]:
    """A committed ruleset, or one of the two shogis `fairy_shogi` derives from it."""
    if name == SHOGI_FORWARD:
        return shogi_forward()
    if name == SHOGI_ARRIVING:
        return shogi_arriving()
    return load_spec(RULESETS / f"{name}.json")


# Rulesets whose rules have no counterpart at all, with the word that has to appear in
# the refusal. Independent of which engine is installed: these are facts about the two
# rule systems, and an engine that grew a board without edges would still not have our
# capture matrix.
NO_COUNTERPART = {
    "torus": "wrap",
    "cylinder": "wrap",
    "rps-v1": "captures",
    "rps-v2": "territory",
    # Was `territory` until the user removed that rule from the spec (`c947850`,
    # 2026-09-06): the capture cycle is what is left with no counterpart, and it is
    # still the reason this ruleset cannot cross.
    "intransitive": "captures",
    # Q438: its silver, bishop and rook may promote leaving the zone over there, which no
    # option forbids. The committed shogi was this game until Q454 put `on_leaving` on
    # its entries, and it translates since; the game it was is derived (`fairy_shogi`).
    SHOGI_ARRIVING: "out of the promotion zone",
}

# Rulesets that must at least produce a variant definition. Every one is a published
# game Fairy-Stockfish has its own implementation of, so a failure here is ours.
MUST_TRANSLATE = (
    "chess",
    "crazyhouse",
    "atomic",
    "antichess",
    "kingofthehill",
    "racingkings",
    "threecheck",
    "horde",
    "makruk",
    "shatranj",
    "losalamos",
    "capablanca",
    "gardner",
)


# Every committed ruleset that translates, and therefore every one whose translation
# has to be *right*. Checked against `pyffish` rather than against an installed binary:
# which of these a given engine can play depends on its version, and pinning that would
# make the test a statement about somebody's machine. See `chess-variant-sandbox fairy` for that.
TRANSLATES = (
    "almost",
    "antichess",
    "atomic",
    "bastion",
    "berolina",
    "capablanca",
    "chess",
    "courier",
    "crazyhouse",
    "dobutsu",
    "dull",
    "gardner",
    "horde",
    "kingofthehill",
    "knightmate",
    "losalamos",
    "makruk",
    "racingkings",
    "sardines",
    "shatranj",
    # Since Q454: schema 15's `on_leaving` on every promotion entry.
    "shogi",
    "threecheck",
    "xiangqi",
)


class TestBetzaSaysWhatItMeans:
    """The notation this project writes, read back by the engine it is written for."""

    @staticmethod
    def _reachable(sf: Any, notation: str, size: int = 9, origin: str = "e5") -> set[tuple[int, int]]:
        """Where a lone piece with this notation can go, as offsets from ``origin``.

        Two details of the probe are load-bearing and both were wrong first time.

        **The variant gets a name of its own per notation.** Loading a configuration
        that reuses a name does not replace the variant already registered under it, so
        every case after the first one silently measured the first one's piece.

        **The two kings stand off every line through the origin.** On b1 and h9 they are
        `(-3, -4)` and `(+3, +4)` away, which is no atom and no multiple of one, so
        neither blocks a ray nor stands in check. A king on the long diagonal -- a1, the
        obvious corner -- stops a bishop four squares short and makes the test assert
        something about the corner rather than about the notation.
        """
        name = "probe" + hashlib.sha1(notation.encode()).hexdigest()[:10]
        board = {origin: "P", "b1": "K", "h9": "k"}
        rows = []
        for rank in range(size, 0, -1):
            row, empty = "", 0
            for file in range(size):
                square = f"{chr(ord('a') + file)}{rank}"
                if square in board:
                    row += (str(empty) if empty else "") + board[square]
                    empty = 0
                else:
                    empty += 1
            rows.append(row + (str(empty) if empty else ""))
        fen = "/".join(rows) + " w - - 0 1"
        sf.load_variant_config(
            f"[{name}]\nmaxRank = {size}\nmaxFile = {size}\n"
            f"customPiece1 = p:{notation}\nking = k\nstartFen = {fen}\ncastling = false\n"
        )
        origin_file, origin_rank = ord(origin[0]) - ord("a"), int(origin[1:])
        return {
            (ord(move[2]) - ord("a") - origin_file, int(move[3:]) - origin_rank)
            for move in sf.legal_moves(name, fen, [])
            if move.startswith(origin)
        }

    @pytest.mark.parametrize(
        ("vectors", "symmetry", "kind"),
        [
            ([[1, 0]], "rotate4", "leap"),
            ([[1, 1]], "rotate4", "leap"),
            ([[1, 2]], "dihedral8", "leap"),
            ([[2, 2]], "rotate4", "leap"),
            ([[1, 1]], "lateral", "leap"),
            ([[0, 1]], "none", "leap"),
            ([[1, 0]], "lateral", "leap"),
            ([[1, 0], [1, 1]], "rotate4", "leap"),
        ],
    )
    def test_a_leap_reaches_exactly_the_squares_it_was_asked_for(
        self, vectors: list[list[int]], symmetry: str, kind: str
    ) -> None:
        sf = _pyffish()
        wanted: set[tuple[int, int]] = set()
        for vector in vectors:
            wanted |= expand_symmetry((vector[0], vector[1]), symmetry)
        notation = betza.term(wanted, distance=1)
        assert notation is not None
        assert self._reachable(sf, notation) == wanted

    @pytest.mark.parametrize(
        ("vectors", "symmetry"), [([[1, 0]], "rotate4"), ([[1, 1]], "rotate4")]
    )
    def test_a_ride_reaches_along_the_ray_and_stops_at_the_edge(
        self, vectors: list[list[int]], symmetry: str
    ) -> None:
        sf = _pyffish()
        directions: set[tuple[int, int]] = set()
        for vector in vectors:
            directions |= expand_symmetry((vector[0], vector[1]), symmetry)
        notation = betza.term(directions, distance=0)
        assert notation is not None
        reached = self._reachable(sf, notation)
        # Four squares each way on a 9x9 board from the middle, and nothing off a ray.
        assert reached == {
            (dx * step, dy * step) for dx, dy in directions for step in range(1, 5)
        }

    def test_a_limited_ride_stops_where_it_was_told_to(self) -> None:
        sf = _pyffish()
        directions = expand_symmetry((1, 0), "rotate4")
        notation = betza.term(directions, distance=2)
        assert notation is not None
        assert self._reachable(sf, notation) == {
            (dx * step, dy * step) for dx, dy in directions for step in (1, 2)
        }

    def test_a_direction_the_piece_does_not_have_is_refused_rather_than_widened(self) -> None:
        """A vector with no Betza atom has no nearest match.

        ``(1, 4)`` is a leap no atom covers. Answering with the nearest one would be a
        different piece, which is the failure this whole module is arranged around.
        """
        assert betza.term({(1, 4), (-1, 4)}, distance=1) is None

    def test_a_lame_leap_is_only_offered_where_the_engine_implements_one(self) -> None:
        assert betza.term(expand_symmetry((1, 2), "dihedral8"), lame=True) == "nN"
        # A lame *wazir* parses and then generates nothing over there, so it is refused.
        assert betza.term(expand_symmetry((1, 0), "rotate4"), lame=True) is None

    def test_one_modifier_is_written_short_and_two_are_written_long(self) -> None:
        """The spelling matters because `BUILTIN_PIECES` matches on the text.

        ``ffW`` and ``fW`` mean the same thing to the parser, and only one of them is
        how Fairy-Stockfish spells its own pieces -- so the long form silently cost
        every built-in match, and a translated shogi played eight custom pieces.
        """
        assert betza.term({(0, 1)}, distance=1) == "fW"
        assert betza.term({(0, 1), (1, 0), (-1, 0)}, distance=1) == "fsW"
        assert betza.term({(0, 1), (0, -1)}, distance=1) == "vW"


class TestWhatCannotCross:
    """Refusals that are true of the rules rather than of an installation."""

    @pytest.mark.parametrize(("name", "word"), sorted(NO_COUNTERPART.items()))
    def test_a_rule_with_no_counterpart_is_refused_by_name(self, name: str, word: str) -> None:
        with pytest.raises(Untranslatable) as caught:
            translate(_spec(name))
        assert any(word in reason for reason in caught.value.reasons)

    def test_every_reason_is_a_sentence_rather_than_a_key(self) -> None:
        """A refusal is read by a person deciding whether to change the ruleset."""
        with pytest.raises(Untranslatable) as caught:
            translate(_spec("rps-v5"))
        assert len(caught.value.reasons) >= 2
        for reason in caught.value.reasons:
            assert " " in reason and len(reason) > 20

    def test_a_board_too_big_is_refused_before_anything_else_is_tried(self) -> None:
        """Chess on a tall board: valid, playable here, and past what Fairy will take.

        The board is the only thing changed. An earlier version also cut the placements
        down to a lone king, which made the spec itself incoherent once the validator
        learned to refuse a castling with no rook to castle with and a promotion for a
        piece nobody can play -- so `translate` raised `SpecError` from its own
        `compile_spec` and the test passed for the wrong reason waiting to happen.
        """
        spec = _spec("chess")
        spec["board"] = {"width": 8, "height": 12}
        with pytest.raises(Untranslatable) as caught:
            translate(spec)
        assert any("12x10" in reason for reason in caught.value.reasons)


def _shaped(name: str, **rules: Any) -> dict[str, Any]:
    """A committed ruleset with some rules replaced, at a schema version that has them."""
    spec = _spec(name)
    spec["rules"].update(rules)
    spec["schema_version"] = max(spec.get("schema_version", 1), 2)
    return spec


def _royal_mode_before_q403(compiled: Any, reasons: list[str]) -> str:
    """`_royal_mode` as it read before Q403: the same models, and no refusal.

    Only for the premise tests below, which have to show the definition that used to be
    written in order to show why it is refused now.
    """
    if compiled.win != "checkmate":
        return "extinct"
    return "pseudo" if compiled.explosion is not None else "king"


def _written_fen(engine: Engine, written: Any, state: Any) -> str:
    """``state`` as the opponent is handed it, with the arguments the agent passes."""
    return position.fen(
        engine.spec, state, written.chars, written.king, drops=written.drops,
        check_limit=engine.spec.check_limit, castling=written.castling,
    )


def _played(engine: Engine, written: Any, state: Any, uci: str) -> Any:
    """``state`` after the one legal move of ours that ``uci`` names."""
    legal = engine.legal_moves(state)
    return engine.apply(
        state,
        position.move_of(engine.spec, state, legal, uci, written.chars, written.promotion_suffix),
    )


def _both_after(spec: dict[str, Any], written: Any, moves: list[str]) -> tuple[Any, Any, str, str]:
    """Play ``moves`` in both engines: our engine and state, our FEN, and theirs."""
    sf = _pyffish()
    sf.load_variant_config(written.config)
    engine = Engine(spec)
    state = engine.initial_state()
    start = _written_fen(engine, written, state)
    for uci in moves:
        state = _played(engine, written, state, uci)
    return engine, state, _written_fen(engine, written, state), sf.get_fen(written.name, start, moves)


class TestAResultRuleOnlyTheKingSlotCarriesIsRefusedOutsideIt:
    """Two result rules Fairy-Stockfish reads through its king slot (Q96, refused by Q403).

    Both specs are valid and both used to translate. Neither changes a legal move, so
    `verify` could pass either while the two engines played for different results --
    which is why each is refused by name rather than left to verification.

    * ``win: "none"`` beside a royal. Losing the royal does not end our game; the king
      slot and an extinction rule, the only ways over there to hold one, both make it a
      loss.
    * ``check_limit`` beside a royal outside the king slot: ``royal_capture``, ``none``
      or a blast rule. Fairy-Stockfish counts no check against it. Q96 named the first
      two; the probe for Q403 found the blast rule's pseudo-royal has the same hole.
    """

    # (ruleset, the rules replaced, the start of each reason, in order)
    REFUSED = (
        ("threecheck", {"win": "royal_capture"}, ("rules.check_limit beside rules.win",)),
        ("threecheck", {"win": "none"}, ('rules.win "none"', "rules.check_limit beside rules.win")),
        ("atomic", {"check_limit": 3}, ("rules.check_limit beside rules.explode_on_capture",)),
        ("chess", {"win": "none", "stalemate": "loss"}, ('rules.win "none"',)),
    )

    # The same rules one step away, each still in the model it had: a check count in the
    # king slot, a royal capture or a blast with no count, `none` with no royal at all.
    NEIGHBOURS = (
        ("threecheck", {}, "king"),
        ("dobutsu", {}, "extinct"),
        ("chess", {"win": "royal_capture"}, "extinct"),
        ("atomic", {}, "pseudo"),
        ("antichess", {}, "none"),
    )

    # The three shapes whose check count used to be written for a royal outside the slot.
    UNCOUNTED = (
        ("threecheck", {"win": "royal_capture"}),
        ("threecheck", {"win": "none"}),
        ("atomic", {"check_limit": 3}),
    )

    # Q96's position: 1.e4 f6 2.Qh5, a check by the first player on the second's royal.
    CHECK = ["e2e4", "f7f6", "d1h5"]

    @pytest.mark.parametrize(("name", "rules", "starts"), REFUSED)
    def test_the_shape_is_refused_by_name_and_for_nothing_else(
        self, name: str, rules: dict[str, Any], starts: tuple[str, ...]
    ) -> None:
        """The spec is valid; the refusal is the translator's, and it is the whole of it."""
        from chess_variant_sandbox import fairy

        spec = _shaped(name, **rules)
        assert validate_spec(spec) == []
        with pytest.raises(Untranslatable) as caught:
            translate(spec)
        reasons = caught.value.reasons
        assert len(reasons) == len(starts), reasons
        for reason, start in zip(reasons, starts, strict=True):
            assert reason.startswith(start), reason
        assert fairy.supports(spec) is False
        assert fairy.why_not(spec) == reasons

    @pytest.mark.parametrize(("name", "rules", "royalty"), NEIGHBOURS)
    def test_the_neighbouring_shapes_still_translate_in_the_model_they_had(
        self, name: str, rules: dict[str, Any], royalty: str
    ) -> None:
        spec = _shaped(name, **rules) if rules else _spec(name)
        written = translate(spec)
        assert written.royalty == royalty
        assert written.check_counting is bool(spec["rules"].get("check_limit"))

    def test_a_check_on_the_king_slot_is_counted_the_same_by_both(self) -> None:
        """The accepted case, played: threecheck's counter moves on both sides."""
        spec = _spec("threecheck")
        _, state, ours, theirs = _both_after(spec, translate(spec), self.CHECK)
        assert list(state.checks) == [1, 0]
        assert " 2+3 " in ours
        assert theirs == ours

    @pytest.mark.parametrize(("name", "rules"), UNCOUNTED)
    def test_the_premise_a_royal_outside_the_slot_is_never_counted_as_checked(
        self, monkeypatch: pytest.MonkeyPatch, name: str, rules: dict[str, Any]
    ) -> None:
        """What the refused definition did, so a build that counts these is noticed.

        If this fails, Fairy-Stockfish has started counting checks on a royal outside
        its king slot and the refusal may be worth revisiting against that build.
        """
        spec = _shaped(name, **rules)
        monkeypatch.setattr(variant, "_royal_mode", _royal_mode_before_q403)
        written = translate(spec)
        assert written.royalty != "king" and written.check_counting
        _, state, ours, theirs = _both_after(spec, written, self.CHECK)
        assert list(state.checks) == [1, 0]
        assert " 2+3 " in ours
        assert " 3+3 " in theirs

    def test_the_premise_losing_a_royal_ends_the_game_over_there(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What the refused definition did with `win: "none"`: the royal taken ends it.

        Ours plays on, which is the rule; theirs has no move and calls the game over.
        """
        sf = _pyffish()
        spec = _shaped("chess", win="none", stalemate="loss")
        monkeypatch.setattr(variant, "_royal_mode", _royal_mode_before_q403)
        written = translate(spec)
        moves = [*self.CHECK, "a7a6", "h5e8"]
        engine, state, ours, theirs = _both_after(spec, written, moves)
        assert theirs == ours, "the two engines disagree about the board, not the result"
        assert engine.terminal(state) is None
        assert engine.legal_moves(state)
        start = position.fen(
            engine.spec, engine.initial_state(), written.chars, written.king,
            castling=written.castling,
        )
        assert sf.legal_moves(written.name, start, moves) == []
        assert sf.is_immediate_game_end(written.name, start, moves)[0] is True


class TestTheTranslationIsWellFormed:
    """Properties of the generated definition that need no engine."""

    @pytest.mark.parametrize("name", MUST_TRANSLATE)
    def test_a_published_variant_translates(self, name: str) -> None:
        """Translates, opens with its own name, and gives every piece one slot and one
        letter (Q754 merged the two tests that translated the same variant)."""
        result = translate(_spec(name))
        assert result.config.startswith(f"[{result.name}]")
        assert "startFen = " in result.config
        assert len(set(result.chars)) == len(result.chars)
        defined = [
            line.split(" = ")[0]
            for line in result.config.splitlines()
            if " = " in line and line.split(" = ")[1][:1].isalpha()
        ]
        assert len(defined) == len(set(defined)), "a slot is used twice"

    # One test per ruleset that must translate (Q749): the lower bound is each of them
    # still translating, and the walk over every ruleset was 2 s in the gate.
    @pytest.mark.parametrize("name", sorted(TRANSLATES))
    def test_the_set_of_rulesets_that_translate_has_not_shrunk(self, name: str) -> None:
        """A ruleset that quietly stops translating is an opponent quietly withdrawn.

        Lower bound rather than equality: a new ruleset that translates is good news and
        should not fail a test. One that *stops* is a regression, and nothing else would
        report it -- the roster simply offers one fewer opponent.
        """
        try:
            translate(load_spec(RULESETS / f"{name}.json"))
        except Untranslatable as refused:
            pytest.fail(f"{name} no longer translates: {refused}")

    def test_the_variant_name_is_the_mechanics_hash(self) -> None:
        """Two rulesets called the same thing must not load over one another."""
        chess = translate(_spec("chess"))
        renamed = _spec("chess")
        renamed["name"] = "not chess"
        assert translate(renamed).name == chess.name
        assert translate(_spec("gardner")).name != chess.name

    def test_chess_translates_to_the_engine_s_own_pieces(self) -> None:
        """Every chess piece is one Fairy-Stockfish already has.

        Not cosmetic. Castling only works with the built-in rook, the drop and blast
        rules are written against the built-in pawn by name, and a custom piece carries
        no piece values -- so a chess made of custom pieces is a weaker opponent that
        cannot castle.
        """
        config = translate(_spec("chess")).config
        for line in ("pawn = p", "knight = n", "bishop = b", "rook = r", "queen = q"):
            assert line in config

    def test_a_ruleset_with_no_promotion_clears_the_promotion_zone(self) -> None:
        """The zone defaults to the far rank, and a pawn there is pushed off the board.

        `dull` is a board of pawns with nothing to promote into, and the engine offered
        it a move to a rank that does not exist.
        """
        config = translate(_spec("dull")).config
        assert "promotionRegionWhite = -" in config
        assert "promotionRegionBlack = -" in config

    def test_a_ruleset_with_no_double_step_turns_the_option_off(self) -> None:
        """Emptying the region is not enough; the built-in pawn brings a double step."""
        config = translate(_spec("shatranj")).config
        assert "doubleStep = false" in config

    def test_the_promotion_rank_is_written_beside_the_regions(self) -> None:
        """Two spellings of one rule, because they belong to different engine versions."""
        config = translate(_spec("chess")).config
        assert "promotionRank = 8" in config
        assert "promotionRegionWhite = *8" in config


class TestPositionsAndMoves:
    """The FEN and the move text, which are the only things the opponent ever sees."""

    def test_the_opening_position_round_trips_through_the_fen(self) -> None:
        spec = _spec("chess")
        engine = Engine(spec)
        result = translate(spec)
        text = position.fen(
            engine.spec,
            engine.initial_state(),
            result.chars,
            result.king,
            castling=result.castling,
        )
        assert text == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_a_castling_right_is_derived_from_what_has_not_moved(self) -> None:
        """The `unmoved` mask *is* the right, so the FEN field is read off it.

        A rook that has moved loses its side of the castling field and the other three
        letters stay. Getting this wrong is silent: the opponent simply never castles,
        and every game it plays is still legal.
        """
        spec = _spec("chess")
        engine = Engine(spec)
        result = translate(spec)
        placement = {
            "e1": ("K", 0), "a1": ("R", 0), "h1": ("R", 0),
            "e8": ("K", 1), "a8": ("R", 1), "h8": ("R", 1),
        }
        both = engine.make_state(placement)
        assert " KQkq " in position.fen(
            engine.spec, both, result.chars, result.king, castling=True
        )
        moved = engine.make_state(placement, moved=["h1"])
        assert " Qkq " in position.fen(
            engine.spec, moved, result.chars, result.king, castling=True
        )

    def test_a_hand_is_written_into_the_fen(self) -> None:
        spec = _spec("crazyhouse")
        engine = Engine(spec)
        result = translate(spec)
        assert result.drops
        text = position.fen(
            engine.spec,
            engine.initial_state(),
            result.chars,
            result.king,
            drops=True,
            castling=True,
        )
        assert "[]" in text

    def test_every_legal_move_has_a_spelling_of_its_own(self) -> None:
        """Two moves sharing a spelling is a move the opponent could not have named."""
        for name in ("chess", SHOGI_FORWARD, "crazyhouse", "makruk"):
            spec = _spec(name)
            engine = Engine(spec)
            result = translate(spec)
            state = engine.initial_state()
            spellings = [
                position.uci_of(engine.spec, state, move, result.chars, result.promotion_suffix)
                for move in engine.legal_moves(state)
            ]
            assert len(set(spellings)) == len(spellings), name

    def test_a_move_the_engine_names_is_looked_up_rather_than_built(self) -> None:
        spec = _spec("chess")
        engine = Engine(spec)
        result = translate(spec)
        state = engine.initial_state()
        legal = engine.legal_moves(state)
        found = position.move_of(engine.spec, state, legal, "e2e4", result.chars)
        assert engine.describe_move(state, found) == "Pe2-e4"

    def test_a_move_we_do_not_have_is_refused_rather_than_guessed(self) -> None:
        spec = _spec("chess")
        engine = Engine(spec)
        result = translate(spec)
        state = engine.initial_state()
        with pytest.raises(position.MoveMismatch):
            position.move_of(engine.spec, state, engine.legal_moves(state), "e2e5", result.chars)

    def test_shogi_spells_a_promotion_with_a_plus(self) -> None:
        """Several promoting pieces make the relation a map, and a map has nothing to choose."""
        assert translate(_spec(SHOGI_FORWARD)).promotion_suffix == "plus"
        assert translate(_spec("chess")).promotion_suffix == "piece"


class TestTheTwoEnginesAgree:
    """The check the rest of this rests on, played rather than read."""

    @staticmethod
    def _verify(name: str, **kwargs: Any) -> Any:
        from chess_variant_sandbox.fairy.verify import PyffishBackend, verify

        _pyffish()
        return verify(_spec(name), PyffishBackend(), "", **kwargs)

    @pytest.mark.parametrize("name", ["chess", "gardner", "losalamos", "shatranj"])
    def test_a_published_variant_plays_the_same_game(self, name: str) -> None:
        result = self._verify(name, depth=2, games=2, plies=40)
        assert result.ok, result.summary()
        assert result.positions > 100

    @pytest.mark.slow
    @pytest.mark.parametrize("name", TRANSLATES)
    def test_every_ruleset_that_translates_plays_the_same_game(self, name: str) -> None:
        """The whole claim, one ruleset at a time.

        Against the *library* rather than a binary, deliberately. Which rulesets play
        against somebody's installed engine is a fact about their installation -- the
        newest tagged release is four years older than this and six of these diverge on
        it -- but whether the translation is *correct* is a fact about this repository,
        and that is what belongs in a test.
        """
        result = self._verify(name, depth=3, games=3, plies=60)
        assert result.ok, result.summary()

    def test_a_ruleset_with_no_translation_is_reported_rather_than_raised(self) -> None:
        result = self._verify("torus", depth=1, games=0, plies=0)
        assert not result.ok
        assert any("wrap" in reason for reason in result.reasons)

    def test_a_broken_translation_is_caught_rather_than_played(self) -> None:
        """The check has to be able to fail, or it is not a check.

        The mutation is one that really happened: a chess whose pawns have lost their
        double step. It is a legal game, it is a different one, nothing about the
        configuration looks wrong, and every game played under it would have been rated
        against a ladder playing something else.
        """
        from chess_variant_sandbox.fairy.verify import PyffishBackend

        _pyffish()
        spec = _spec("chess")
        engine = Engine(spec)
        good = translate(spec)
        broken = variant.Variant(
            # A name of its own: loading a configuration under a name already registered
            # does not replace it, so a broken variant sharing the good one's name would
            # simply be the good one and the test would pass without checking anything.
            name=good.name + "broken",
            config=(
                good.config.replace(f"[{good.name}]", f"[{good.name}broken]")
                .replace("doubleStep = true", "doubleStep = false")
            ),
            chars=good.chars,
            king=good.king,
            drops=good.drops,
            check_counting=good.check_counting,
            castling=good.castling,
            royalty=good.royalty,
            promotion_suffix=good.promotion_suffix,
        )
        backend = PyffishBackend()
        backend.load(broken, "")
        text = position.fen(
            engine.spec,
            engine.initial_state(),
            good.chars,
            good.king,
            castling=good.castling,
        )
        ours = engine.perft(engine.initial_state(), 1)
        assert backend.perft(text, 1) != ours

        # and the real one, through the same backend, agrees
        backend.load(good, "")
        assert backend.perft(text, 1) == ours


class TestTheOpponentIsOnlyOfferedWhereItWorks:
    """The roster, which must not depend on an engine being installed."""

    def test_a_ruleset_with_no_translation_is_never_offered(self) -> None:
        from chess_variant_sandbox.web import opponents

        assert opponents.fairy_stockfish_for(_spec("torus")) == []

    def test_no_spec_means_no_opponent_rather_than_an_error(self) -> None:
        from chess_variant_sandbox.web import opponents

        assert opponents.fairy_stockfish_for(None) == []

    def test_the_library_may_narrow_by_translation_and_never_offer_on_it(self) -> None:
        """The filter is `supports`; the opponent is `playable`. Both, and not one.

        A card cannot ask the second -- it costs an engine start per ruleset and needs a
        binary that is discovered rather than shipped -- so the listing carries the cheap
        answer and the picker keeps the expensive one. This is the pair held apart: every
        ruleset the card marks as translating is one the roster is still free to refuse,
        and `torus` is refused by both.
        """
        from chess_variant_sandbox.web import library, opponents

        chess, torus = _spec("chess"), _spec("torus")

        assert library._fairy(chess)["fairyTranslates"] is True
        assert library._fairy(torus)["fairyTranslates"] is False
        assert any("edges that join" in reason for reason in library._fairy(torus)["fairyReasons"])

        # The card said yes about chess; the roster is under no obligation to agree, and
        # on a machine with no engine it does not.
        offered = opponents.fairy_stockfish_for(chess)
        assert offered == [] or offered[0].store == "fairy"
        assert opponents.fairy_stockfish_for(torus) == []

    def test_the_agent_is_registered_and_takes_a_node_budget(self) -> None:
        from chess_variant_sandbox.agents import available_agents
        from chess_variant_sandbox.agents.base import parse_agent
        from chess_variant_sandbox.agents.fairy_stockfish import DEFAULT_NODES

        assert "fairystockfish" in available_agents()
        config = parse_agent(f"fairystockfish:nodes={DEFAULT_NODES}")
        assert config.options["nodes"] == DEFAULT_NODES

    def test_the_agent_has_no_native_form(self) -> None:
        """The core must not think it can run this player.

        `native_spec` answering anything but ``None`` would send the whole game into the
        core, which has never heard of this agent. It is the one property that makes the
        extension point work.
        """
        from chess_variant_sandbox.agents.base import native_spec

        assert native_spec("fairystockfish:nodes=1000") is None

    def test_a_pairing_with_one_native_side_keeps_that_side_in_the_core(self) -> None:
        """Otherwise the outside opponent drags the whole ladder into Python with it.

        `play_game_python` builds *both* players from `agents/`, and those are oracles:
        the picker's rating run on the smallest board took 283 seconds, almost all of it
        our own top rung searching 200,000 nodes an interpreter at a time.
        """
        from chess_variant_sandbox.agents.base import native_spec

        assert native_spec("minimax:nodes=6400") is not None
        assert native_spec("fairystockfish:nodes=1000") is None


class TestTheTagTheOptionAndTheNoteReadOneRecord:
    """Three surfaces, two predicates, one verdict -- and the difference is stated.

    The card's tag reads `supports`; the picker's option and the sentence under it read
    `verdict`, one memoised record per ruleset. They differ on purpose: a listing cannot
    start an engine per card, and the answer would be about a PATH. What was wrong was
    that the difference went unsaid -- `sardines` was tagged, offered nobody, and the
    page could not say whether that was a missing download or a build that plays the
    game differently, though the roster had just found out which.
    """

    @staticmethod
    def _with_verdict(monkeypatch, spec: dict[str, Any], ok: bool, reasons: list[str]):
        """Put a verdict in the memo without an engine, and take it out again."""
        from chess_variant_sandbox import fairy
        from chess_variant_sandbox.fairy.verify import Verification
        from chess_variant_sandbox.spec.canonical import spec_hash

        monkeypatch.setattr(fairy, "available", lambda: True)
        monkeypatch.setattr(fairy, "engine_name", lambda: "Fairy-Stockfish test build")
        key = spec_hash(spec)
        monkeypatch.setitem(
            fairy._PLAYABLE,
            key,
            Verification(
                ruleset_hash=key, name=str(spec["name"]), ok=ok, reasons=reasons,
                depth=2, positions=7, fen="k7/8/8/8/8/8/8/K7 w - - 0 1",
                ours=[] if ok else ["a1a2"], theirs=[] if ok else ["a1b1"],
            ),
        )

    def test_a_build_that_plays_differently_is_named_with_its_position(
        self, monkeypatch
    ) -> None:
        """The sardines shape: translates, refused by the installed build, and said so."""
        from chess_variant_sandbox.web import library, opponents

        spec = _spec("chess")
        self._with_verdict(monkeypatch, spec, ok=False, reasons=["the move lists differ"])

        assert library._fairy(spec)["fairyTranslates"] is True  # the tag
        assert opponents.fairy_stockfish_for(spec) == []  # the option
        check = opponents.fairy_check(spec)  # the note
        assert check["engine"] == "Fairy-Stockfish test build"
        assert check["ok"] is False
        assert check["reasons"] == ["the move lists differ"]
        assert check["fen"] == "k7/8/8/8/8/8/8/K7 w - - 0 1"
        assert (check["ours"], check["theirs"]) == (["a1a2"], ["a1b1"])

    def test_an_agreeing_build_offers_the_opponent_and_the_note_says_ok(
        self, monkeypatch
    ) -> None:
        from chess_variant_sandbox.web import opponents

        spec = _spec("chess")
        self._with_verdict(monkeypatch, spec, ok=True, reasons=[])

        offered = opponents.fairy_stockfish_for(spec)
        assert len(offered) == 1 and offered[0].store == "fairy"
        assert opponents.fairy_check(spec)["ok"] is True

    def test_no_engine_means_nothing_was_asked_and_the_note_says_that(
        self, monkeypatch
    ) -> None:
        """A missing download is not a disagreement, and must not read as one."""
        from chess_variant_sandbox import fairy
        from chess_variant_sandbox.web import opponents

        monkeypatch.setattr(fairy, "available", lambda: False)
        check = opponents.fairy_check(_spec("chess"))
        assert check["engine"] is None
        assert check["ok"] is None and check["reasons"] == []
        assert opponents.fairy_stockfish_for(_spec("chess")) == []

    def test_a_ruleset_with_no_translation_gets_no_verdict(self, monkeypatch) -> None:
        """The library's `fairyReasons` already says why; this does not restate it."""
        from chess_variant_sandbox import fairy
        from chess_variant_sandbox.web import opponents

        monkeypatch.setattr(fairy, "available", lambda: True)
        monkeypatch.setattr(fairy, "engine_name", lambda: "Fairy-Stockfish test build")
        check = opponents.fairy_check(_spec("torus"))
        assert check["ok"] is None and check["reasons"] == []

    def test_checked_is_the_ok_only_view_of_the_verdict(self, monkeypatch) -> None:
        from chess_variant_sandbox import fairy

        spec = _spec("chess")
        self._with_verdict(monkeypatch, spec, ok=False, reasons=["no"])
        assert fairy.checked(spec) is None
        verdict = fairy.verdict(spec)
        assert verdict is not None and verdict.ok is False
        self._with_verdict(monkeypatch, spec, ok=True, reasons=[])
        assert fairy.checked(spec) is fairy.verdict(spec)

    # One test per ruleset (Q749): the walk was one test of 2.2 s in the gate.
    @pytest.mark.parametrize("path", sorted(RULESETS.glob("*.json")), ids=lambda path: path.stem)
    def test_every_committed_card_is_tagged_exactly_where_the_translator_says_yes(
        self, path: Path
    ) -> None:
        """The survey, the cheap half: the tag *is* `supports`, ruleset for ruleset."""
        from chess_variant_sandbox import fairy
        from chess_variant_sandbox.web import library

        spec = load_spec(path)
        card = library._fairy(spec)
        assert card["fairyTranslates"] is fairy.supports(spec), path.name
        assert (card["fairyReasons"] == []) is fairy.supports(spec), path.name

    @pytest.mark.slow
    def test_every_committed_ruleset_is_offered_exactly_where_the_engine_agreed(
        self,
    ) -> None:
        """The survey, the expensive half, against whichever engine is installed.

        One engine start per translating ruleset, so it is slow and it is a statement
        about this machine -- which is the point: on the 2021 release six translating
        rulesets are refused (berolina, dobutsu, horde, knightmate, sardines, xiangqi)
        and every one of them must be refused by the option *and* explained by the note,
        from one record.
        """
        from chess_variant_sandbox import fairy
        from chess_variant_sandbox.web import opponents

        if not fairy.available():
            pytest.skip("no Fairy-Stockfish binary is installed")
        for path in sorted(RULESETS.glob("*.json")):
            spec = load_spec(path)
            offered = bool(opponents.fairy_stockfish_for(spec))
            check = opponents.fairy_check(spec)
            assert check["engine"], path.name
            if not fairy.supports(spec):
                assert not offered and check["ok"] is None, path.name
                continue
            assert offered is (check["ok"] is True), path.name
            if not offered:
                assert check["reasons"], path.name


class TestARuleNobodyConsideredIsRefusedRatherThanDropped:
    """The catch-all, and the reason it is worth more than the named refusals.

    A named refusal is a rule somebody held the two systems up against. A rule nobody
    did takes no branch at all: it is dropped, and the definition then states a different
    game while claiming to state this one. That is the failure `verify.py` cannot see --
    verification compares *move lists*, so a dropped rule that only decides who wins
    leaves both engines offering identical moves everywhere while playing for different
    results, and the check that exists to hold the two readings together reports success.

    Simulated by growing the schema's own key set, because that is the only way the case
    arises: the validator refuses an unknown key today, so a rule can reach the
    translator unconsidered only once a schema version has been added without a line
    here. Schema 10 is queued.
    """

    EXTRA = "teleport"

    @pytest.fixture
    def _schema_ten(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A schema that knows one more rule than this translator does."""
        from chess_variant_sandbox.spec import schema

        monkeypatch.setattr(schema, "_RULES_KEYS", schema._RULES_KEYS | {self.EXTRA})

    def _with_extra_rule(self) -> dict[str, Any]:
        spec = _spec("chess")
        spec["rules"][self.EXTRA] = {"piece": "N", "squares": ["a1"]}
        return spec

    def test_a_rule_the_translator_has_never_been_through_is_refused(
        self, _schema_ten: None
    ) -> None:
        with pytest.raises(Untranslatable) as caught:
            translate(self._with_extra_rule())
        assert any(f"rules.{self.EXTRA}" in reason for reason in caught.value.reasons)

    def test_the_refusal_names_the_new_rule_and_objects_to_nothing_else(
        self, _schema_ten: None
    ) -> None:
        """A guard that refused broadly would cost rulesets that translate today.

        Chess is otherwise translatable, so the whole refusal has to be the one rule
        nobody has been through. What the drop looked like is the other half of the
        same fact: the rule changes the mechanics hash and therefore the variant's
        *name*, and not one line the engine would play by -- so the only evidence of a
        different game was a hash, and no reader of a config compares two of those.
        """
        with pytest.raises(Untranslatable) as caught:
            translate(self._with_extra_rule())
        assert len(caught.value.reasons) == 1
        assert caught.value.reasons[0].startswith(f"rules.{self.EXTRA}: ")

    def test_a_key_inside_a_rule_that_does_cross_is_refused_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same hole one level down, and where the next schema version lands.

        `rules.explode_on_capture` crosses as `blastOnCapture`, so a key added inside it
        is dropped inside a rule that otherwise looks translated -- which is worse to
        read than a rule that is missing altogether.

        The placeholder was `by` until schema 10 made it real, which is the guard doing
        its job: `by` and `of` now sit in `_RULE_OBJECT_KEYS` with a branch of their own
        (`test_a_narrowed_explosion_is_refused` below), and the hypothetical key has to
        be one nobody has claimed. It is named for what it stands for rather than for a
        feature anybody plans, so that the next bump does not silently retire this test
        by implementing it.
        """
        from chess_variant_sandbox.spec import schema

        unclaimed = "unconsidered_key"
        assert unclaimed not in schema._EXPLODE_KEYS
        monkeypatch.setattr(
            schema, "_EXPLODE_KEYS", schema._EXPLODE_KEYS | {unclaimed}
        )
        spec = _spec("atomic")
        spec["rules"]["explode_on_capture"][unclaimed] = ["Q"]
        with pytest.raises(Untranslatable) as caught:
            translate(spec)
        assert any(
            f"explode_on_capture.{unclaimed}" in reason
            for reason in caught.value.reasons
        )

    def test_the_refusal_reaches_the_report_a_person_reads(self, _schema_ten: None) -> None:
        """`chess-variant-sandbox fairy` prints `why_not`, and the picker asks `supports`."""
        from chess_variant_sandbox import fairy

        spec = self._with_extra_rule()
        assert fairy.supports(spec) is False
        assert any(f"rules.{self.EXTRA}" in reason for reason in fairy.why_not(spec))

    def test_the_refusal_reaches_verify(self, _schema_ten: None) -> None:
        """The refusal has to arrive as a *verdict*, not as a passing verification.

        `verify` is the thing that would otherwise say yes. It must return a failed
        `Verification` carrying the reason, and it must not put the ruleset to an engine
        at all -- so the backend here records being asked. Recorded rather than raised
        because `verify` swallows a backend's exceptions into a failed verification,
        which would make an engine that was asked look like one that was not.
        """
        from chess_variant_sandbox.fairy.verify import verify

        class _NeverAsked:
            """Records being asked rather than raising: `verify` swallows exceptions."""

            asked = False

            def load(self, variant: Any, config_path: str) -> None:
                self.asked = True

            def divide(self, fen: str, depth: int) -> dict[str, int]:
                self.asked = True
                return {}

            def perft(self, fen: str, depth: int) -> int:
                self.asked = True
                return 0

        backend = _NeverAsked()
        result = verify(self._with_extra_rule(), backend, "")
        assert result.ok is False
        assert any(f"rules.{self.EXTRA}" in reason for reason in result.reasons)
        assert backend.asked is False, "a ruleset with no translation reached an engine"

    def test_every_rule_the_schema_knows_is_in_exactly_one_list(self) -> None:
        """The tripwire, and the reason the catch-all is not enough on its own.

        The catch-all refuses at runtime, which shows up as one ruleset quietly losing
        an opponent. This fails in the test suite instead, with the key's name in it, on
        the commit that adds the rule -- which is where the decision is cheap.
        """
        from chess_variant_sandbox.spec import schema

        considered = (
            variant._RULES_TRANSLATED | variant._RULES_REFUSED | set(variant._RULES_IRRELEVANT)
        )
        assert considered == schema._RULES_KEYS, (
            f"new in the schema: {sorted(schema._RULES_KEYS - considered)}; "
            f"gone from the schema: {sorted(considered - schema._RULES_KEYS)}"
        )
        overlap = variant._RULES_TRANSLATED & variant._RULES_REFUSED
        assert not overlap, f"a rule cannot both cross and be refused: {sorted(overlap)}"

    def test_every_key_of_a_rule_that_crosses_is_in_the_table(self) -> None:
        from chess_variant_sandbox.spec import schema

        schema_keys = {
            "castling": schema._CASTLING_KEYS,
            "drops": schema._DROPS_KEYS,
            "explode_on_capture": schema._EXPLODE_KEYS,
            "extinction": schema._EXTINCTION_KEYS,
            "obstacles": schema._OBSTACLES_KEYS,
            "promotion": schema._PROMOTION_KEYS,
            "reach": schema._REACH_KEYS,
        }
        assert set(schema_keys) == set(variant._RULE_OBJECT_KEYS)
        for rule, known in schema_keys.items():
            assert set(known) == set(variant._RULE_OBJECT_KEYS[rule]), rule

    def test_the_catch_all_refuses_nothing_the_corpus_already_writes(self) -> None:
        """The guard is only worth having if it costs no ruleset that was fine.

        Every rule key and sub-key in the committed corpus has to be in one of the
        lists, or the guard would withdraw an opponent that had been playing.
        """
        keys: set[str] = set()
        nested: set[str] = set()
        for path in sorted(RULESETS.glob("*.json")):
            rules = load_spec(path).get("rules", {})
            keys |= set(rules)
            for rule, known in variant._RULE_OBJECT_KEYS.items():
                value = rules.get(rule)
                for entry in value if isinstance(value, list) else [value]:
                    if isinstance(entry, dict):
                        nested |= {f"{rule}.{key}" for key in set(entry) - known}
        considered = (
            variant._RULES_TRANSLATED | variant._RULES_REFUSED | set(variant._RULES_IRRELEVANT)
        )
        assert keys <= considered, sorted(keys - considered)
        assert not nested, sorted(nested)


def _first_condition(spec: dict[str, Any]) -> dict[str, Any]:
    """The first move condition in a spec: chess's pawn double step, `from_rank`."""
    condition: dict[str, Any] = next(
        move["condition"]
        for piece in spec["pieces"]
        for move in piece["moves"]
        if move.get("condition")
    )
    return condition


class TestAKeyOutsideTheRulesIsRefusedRatherThanDropped:
    """Q94's guard, carried to every level of the spec below and beside `rules` (Q97).

    Before Q97 a key added under `setup`, in a piece or in a move descriptor took no
    branch in the translator and was dropped. `verify` sees little of two of those
    levels: the opening array reaches the other engine only as the `startFen` the
    translator writes, and a hand only as the pocket `position.py` writes from our state.

    Simulated the way Q94's tests simulate a new rule, by growing the schema's own key
    set for one level, because the validator refuses an unknown key today and that is
    the only way the case arises.
    """

    UNCLAIMED = "unconsidered_key"

    # The schema's key set for each level of `variant._KEYS_BELOW_RULES`, and where in
    # chess to put a key at that level.
    LEVELS: dict[str, tuple[str, Any]] = {
        "spec": ("_SPEC_KEYS", lambda spec: spec),
        "board": ("_BOARD_KEYS", lambda spec: spec["board"]),
        "setup": ("_SETUP_KEYS", lambda spec: spec["setup"]),
        "placement": ("_PLACEMENT_KEYS", lambda spec: spec["setup"]["placements"][0]),
        "piece": ("_PIECE_KEYS", lambda spec: spec["pieces"][0]),
        "move": ("_MOVE_KEYS", lambda spec: spec["pieces"][0]["moves"][0]),
        "move.condition": ("_CONDITION_KEYS", _first_condition),
    }

    def _grown(self, monkeypatch: pytest.MonkeyPatch, level: str) -> dict[str, Any]:
        """Chess with one key nobody has been through, valid under a grown schema."""
        from chess_variant_sandbox.spec import schema

        attribute, where = self.LEVELS[level]
        assert self.UNCLAIMED not in getattr(schema, attribute)
        monkeypatch.setattr(
            schema, attribute, set(getattr(schema, attribute)) | {self.UNCLAIMED}
        )
        spec = _spec("chess")
        where(spec)[self.UNCLAIMED] = 1
        assert validate_spec(spec) == []
        return spec

    def test_every_key_the_schema_allows_is_in_its_level_s_table(self) -> None:
        """The tripwire: fails on the commit that adds a key to the schema unlisted."""
        from chess_variant_sandbox.spec import schema

        assert set(self.LEVELS) == set(variant._KEYS_BELOW_RULES)
        for level, (attribute, _) in self.LEVELS.items():
            allowed = set(getattr(schema, attribute))
            listed = set(variant._KEYS_BELOW_RULES[level])
            assert listed == allowed, (
                f"{level}: new in the schema {sorted(allowed - listed)}, "
                f"gone from it {sorted(listed - allowed)}"
            )

    def test_every_key_says_what_becomes_of_it_and_why(self) -> None:
        verdicts = {variant._TRANSLATED, variant._REFUSED, variant._IRRELEVANT}
        for level, table in variant._KEYS_BELOW_RULES.items():
            for key, (verdict, why) in table.items():
                assert verdict in verdicts, f"{level}.{key}: {verdict!r}"
                assert " " in why and len(why) > 10, f"{level}.{key} gives no reason"

    @pytest.mark.parametrize("level", sorted(LEVELS))
    def test_a_new_key_at_any_level_is_refused_by_name_and_for_nothing_else(
        self, monkeypatch: pytest.MonkeyPatch, level: str
    ) -> None:
        with pytest.raises(Untranslatable) as caught:
            translate(self._grown(monkeypatch, level))
        assert len(caught.value.reasons) == 1, caught.value.reasons
        assert caught.value.reasons[0].startswith(f"{level}.{self.UNCLAIMED}: ")

    @pytest.mark.parametrize("level", sorted(LEVELS))
    def test_without_the_guard_the_key_is_dropped_without_a_word(
        self, monkeypatch: pytest.MonkeyPatch, level: str
    ) -> None:
        """What the hole looked like, at every level: plain chess, but for the name.

        A key inside the mechanics moves the hash and so the variant's name, and not one
        line the engine plays by. A key at the top level is outside the hash, and the
        definition is chess's byte for byte.
        """
        spec = self._grown(monkeypatch, level)
        monkeypatch.setattr(variant, "_check_unconsidered_keys", lambda source, reasons: None)
        dropped = translate(spec).config.splitlines()[1:]
        assert dropped == translate(_spec("chess")).config.splitlines()[1:]

    def test_the_catch_all_refuses_nothing_the_corpus_writes(self) -> None:
        """Every key in every committed and archived ruleset is in its level's table."""
        paths = [*RULESETS.glob("*.json"), *(RULESETS.parent / "archive").glob("*.json")]
        assert len(paths) > 60
        for path in sorted(paths):
            reasons: list[str] = []
            variant._check_unconsidered_keys(load_spec(path), reasons)
            assert reasons == [], (path.name, reasons)


def _archived(name: str) -> dict[str, Any]:
    if name in REBUILT:  # retired by Q870 for `pinwheel`; rebuilt from the hexagon
        return REBUILT[name]()
    return load_spec(RULESETS.parent / "archive" / f"{name}.json")


class TestAHexagonIsRefusedAsAHexagon:
    """Schema 16's `board.shape` (Q281a5), refused by name off `compiled.absent`.

    The compiler folds a hexagon's corners into `obstacles`, so before this refusal the
    native twin of the walled Glinski board was refused as a board with walls, for a rule
    it does not have. Neither case needs `pyffish`: the refusal is the translator's own.
    """

    SHAPE = "board.shape: Fairy-Stockfish has no hexagonal board"

    def _reasons(self, spec: dict[str, Any]) -> list[str]:
        with pytest.raises(Untranslatable) as caught:
            translate(spec)
        return caught.value.reasons

    def test_the_twin_is_refused_as_a_hexagon_and_not_for_walls(self) -> None:
        reasons = self._reasons(_archived("hex-glinski-hexagon"))
        assert self.SHAPE in reasons
        assert not any(reason.startswith("rules.obstacles") for reason in reasons)

    def test_the_walled_parent_is_refused_for_its_board_and_not_as_a_hexagon(self) -> None:
        """Its thirty walls cross since Q703 (`*` in the FEN); its 11x11 board does not."""
        reasons = self._reasons(hex_glinski_promoting())
        assert self.SHAPE not in reasons
        assert not any(reason.startswith("rules.obstacles") for reason in reasons)
        assert any("11x11" in reason for reason in reasons)

    def test_a_wall_inside_a_hexagon_is_refused_as_a_hexagon(self) -> None:
        spec = _archived("hex-glinski-hexagon")
        spec["rules"]["obstacles"] = {"squares": ["f6"]}
        assert validate_spec(spec) == []
        reasons = self._reasons(spec)
        assert self.SHAPE in reasons

    def test_a_joined_hexagon_names_both_the_shape_and_the_seam(self) -> None:
        reasons = self._reasons(_archived("hex-glinski-flanks"))
        assert self.SHAPE in reasons
        assert any(reason.startswith("rules.wrap") for reason in reasons)

    def test_a_grid_written_out_refuses_nothing(self) -> None:
        """Inert: chess with `"shape": "grid"` is a new hash and the same definition."""
        spec = _spec("chess")
        spec["board"]["shape"] = "grid"
        spec["schema_version"] = 16
        assert validate_spec(spec) == []
        written = translate(spec).config.splitlines()[1:]
        assert written == translate(_spec("chess")).config.splitlines()[1:]


def _with_min_range(spec: dict[str, Any], piece_id: str, min_range: int) -> dict[str, Any]:
    for piece in spec["pieces"]:
        if piece["id"] == piece_id:
            for move in piece["moves"]:
                move["min_range"] = min_range
    return spec


class TestARideThatSkipsItsFirstSquaresIsAskedOfTheCompiler:
    """`min_range`, the one movement key Betza cannot spell (Q97).

    The notation rides from the first square, so a move that starts further out is
    written as one that does not. That is the same piece only where another of its moves
    reaches the squares skipped, and the compiler is asked whether it does: the spec is
    compiled again with `min_range` left out and the tables compared. Before Q97 this was
    left to `verify`, which sees it only in a position that uses the move.
    """

    def test_the_premise_the_notation_never_reads_min_range(self) -> None:
        double_step = {
            "kind": "slide", "vectors": [[0, 1]], "range": 2, "mode": "move",
            "condition": {"from_rank": 1},
        }
        written = [
            variant._descriptor_term(
                dict(double_step, min_range=skip), orientation="mirror", where="", reasons=[]
            )
            for skip in (1, 2)
        ]
        assert written[0] == written[1] == "mifW2"

    @pytest.mark.parametrize("name", ["chess", "horde", "berolina", "sardines"])
    def test_a_double_step_beside_a_single_step_is_the_same_piece(self, name: str) -> None:
        from chess_variant_sandbox.engine.compile import compile_spec

        spec = _spec(name)
        assert any(m.get("min_range", 1) > 1 for p in spec["pieces"] for m in p["moves"])
        reasons: list[str] = []
        variant._check_skipped_squares(spec, compile_spec(spec), reasons)
        assert reasons == []

    def test_a_skip_nothing_else_covers_is_refused_by_name_and_for_nothing_else(self) -> None:
        """A bishop that may not stop on the square next to it.

        It used to translate as a plain bishop, and only a walked position with that
        square empty would have shown it.
        """
        spec = _with_min_range(_spec("chess"), "B", 2)
        assert validate_spec(spec) == []
        with pytest.raises(Untranslatable) as caught:
            translate(spec)
        assert len(caught.value.reasons) == 1, caught.value.reasons
        assert caught.value.reasons[0].startswith("move.min_range: ")

    def test_a_single_step_spelled_as_a_leap_is_the_same_piece_too(self) -> None:
        """The neighbour a table comparison might have refused, and does not.

        Equal tables are enough, not required, so a pawn that makes the same moves from
        differently written descriptors could be refused. The likeliest such spelling is
        a single step written as a leap. The compiler folds the leap's square out of the
        slide ray whether or not `min_range` is there, so the tables agree and chess
        written that way still translates, into the engine's own pawn.
        """
        spec = _spec("chess")
        pawn = next(p for p in spec["pieces"] if p["id"] == "P")
        single = next(m for m in pawn["moves"] if m.get("range") == 1)
        single["kind"] = "leap"
        del single["range"]
        assert validate_spec(spec) == []
        assert "pawn = p" in translate(spec).config.splitlines()


def _board_and_hand(fen: str) -> tuple[str, str]:
    """A FEN's board, marks included, and its hand sorted.

    The two engines list a hand in different orders, and they disagree about clocks
    and an en passant square nobody can take, so those are left out.
    """
    board, _, rest = fen.partition("[")
    return board, "".join(sorted(rest.partition("]")[0]))


def _with_promoted_from(spec: dict[str, Any], piece_id: str, base: str) -> dict[str, Any]:
    for piece in spec["pieces"]:
        if piece["id"] == piece_id:
            piece["promoted_from"] = base
    assert validate_spec(spec) == []
    return spec


class TestACapturedPromotedPieceReachesTheHandAsOursDoes:
    """What a captured promoted piece becomes, in both engines (Q97 found it, Q406 closed it).

    Ours demotes by type: a captured piece enters the captor's hand as its
    `promoted_from`. Fairy-Stockfish demotes by history: a piece it promoted itself, or
    one a FEN marks as promoted, is demoted, and any other is not. So `position.fen`
    marks each piece whose type demotes (`variant.board_letters`): ``+P`` where the
    promotion is written as shogi's map and ``H~`` where it is one promoting piece. Where
    no piece demotes, as in crazyhouse, the translation writes ``dropLoop`` so that a
    piece the engine promotes itself is not demoted either.

    The moves it is asked for never depended on this: every position it is asked about
    carries our hand, which is also why `verify` cannot see any of it. What the marks
    change is its search. If a test here fails, the build has changed what it demotes
    or how it reads a mark, and the record (Q97, Q406) is worth revisiting against it.
    """

    # Dobutsu: 1.Lb1a2 Cb3xb2 2.La2a3 Cb2b1=H 3.Gc1xb1. The second player's chick
    # promotes on b1, and the first player's giraffe takes the hen.
    PROMOTE = ["b1a2", "b3b2", "a2a3", "b2b1h"]
    TAKE = "c1b1"

    # One game each, ending with a promoted piece taken, and the hand that leaves.
    LINES = {
        "dobutsu": ([*PROMOTE, TAKE], "Cc"),
        # The c-pawn walks up and promotes taking on c7, and the knight takes the tokin.
        SHOGI_FORWARD: (
            ["c3c4", "g7g6", "c4c5", "g6g5", "c5c6", "g5g4", "c6c7+", "b9c7"], "Pp"
        ),
        # The a-pawn takes twice, promotes taking the b8 knight, and the rook takes the
        # queen, which goes into the second player's hand as a queen.
        "crazyhouse": (
            ["a2a4", "b7b5", "a4b5", "a7a6", "b5a6", "g8f6", "a6a7", "c8b7", "a7b8q", "a8b8"],
            "NPPq",
        ),
    }

    def test_a_piece_it_promotes_itself_is_demoted_as_ours_is(self) -> None:
        spec = _spec("dobutsu")
        _, _, ours, theirs = _both_after(spec, translate(spec), [*self.PROMOTE, self.TAKE])
        assert "[Cc]" in ours
        assert theirs == ours

    def test_a_promoted_piece_it_is_handed_is_marked_and_demoted_there(self) -> None:
        """The hen is written as its engine writes a hen it promoted, and taken as a chick."""
        sf = _pyffish()
        spec = _spec("dobutsu")
        written = translate(spec)
        _, _, handed, theirs = _both_after(spec, written, self.PROMOTE)
        assert "h~" in handed.split("[")[0], "the hen is written promoted"
        assert handed == theirs
        _, _, ours, _ = _both_after(spec, written, [*self.PROMOTE, self.TAKE])
        assert "[Cc]" in ours
        assert sf.get_fen(written.name, handed, [self.TAKE]) == ours

    def test_a_promotion_ours_keeps_is_kept_there(self) -> None:
        """Crazyhouse has no `promoted_from`, so a queen from promotion stays a queen."""
        from chess_variant_sandbox.engine.compile import compile_spec

        sf = _pyffish()
        spec = _spec("crazyhouse")
        compiled = compile_spec(spec)
        assert compiled.demotions == tuple(range(compiled.n_types))
        written = translate(spec)
        assert "dropLoop = true" in written.config.splitlines()
        sf.load_variant_config(written.config)
        promotes = "1r2k3/P7/8/8/8/8/8/4K3[] w - - 0 1"
        assert sf.get_fen(written.name, promotes, ["a7a8q", "b8a8"]).startswith(
            "r3k3/8/8/8/8/8/8/4K3[q]"
        )

    @pytest.mark.parametrize("name", sorted(LINES))
    def test_every_ply_reaches_our_board_and_hand_over_there(self, name: str) -> None:
        """Each ply of the game is applied over there twice, and both must reach ours.

        Once from our FEN of the position before it, which is how the opponent is handed
        a promoted piece, and once from the opening through the engine's own history,
        which is how its search promotes one. The board is compared marks and all, so our
        FEN is the one it would write.
        """
        sf = _pyffish()
        spec = _spec(name)
        written = translate(spec)
        sf.load_variant_config(written.config)
        engine = Engine(spec)
        state = engine.initial_state()
        start = _written_fen(engine, written, state)
        moves, hand = self.LINES[name]
        ours = _board_and_hand(start)
        for ply, uci in enumerate(moves, 1):
            before = _written_fen(engine, written, state)
            state = _played(engine, written, state, uci)
            ours = _board_and_hand(_written_fen(engine, written, state))
            handed = sf.get_fen(written.name, before, [uci])
            assert _board_and_hand(handed) == ours, (ply, uci, "from our FEN")
            own = sf.get_fen(written.name, start, moves[:ply])
            assert _board_and_hand(own) == ours, (ply, uci, "through its own history")
        assert ours[1] == hand

    def test_each_demoting_piece_is_marked_in_the_shape_its_promotion_is_written(self) -> None:
        from chess_variant_sandbox.engine.compile import compile_spec

        shogi = compile_spec(_spec(SHOGI_FORWARD))
        letters = dict(
            zip(shogi.piece_ids, variant.board_letters(shogi, translate(shogi).chars), strict=True)
        )
        assert {key: value for key, value in letters.items() if key != value} == {
            "Q": "+P", "X": "+L", "Y": "+N",
        }
        dobutsu = compile_spec(_spec("dobutsu"))
        chars = translate(dobutsu).chars
        assert variant.board_letters(dobutsu, chars) == ("C", "H~", "E", "G", "L")
        crazyhouse = compile_spec(_spec("crazyhouse"))
        chars = translate(crazyhouse).chars
        assert variant.board_letters(crazyhouse, chars) == chars

    def test_the_premise_each_shape_reads_only_its_own_mark(self) -> None:
        """Why there are two spellings: ``+C`` means nothing where no map promotes C."""
        sf = _pyffish()
        spec = _spec("dobutsu")
        written = translate(spec)
        _, _, handed, _ = _both_after(spec, written, self.PROMOTE)
        assert sf.validate_fen(handed, written.name) == 1
        assert sf.validate_fen(handed.replace("h~", "+c"), written.name) < 0

    def test_nothing_is_marked_where_captures_do_not_go_to_a_hand(self) -> None:
        from chess_variant_sandbox.engine.compile import compile_spec

        spec = _spec(SHOGI_FORWARD)
        del spec["rules"]["drops"]
        assert validate_spec(spec) == []
        compiled = compile_spec(spec)
        assert compiled.demotions != tuple(range(compiled.n_types))
        written = translate(spec)
        assert variant.board_letters(compiled, written.chars) == written.chars
        assert "dropLoop" not in written.config

    def test_the_hands_in_the_corpus_need_no_note(self) -> None:
        for name in self.LINES:
            assert not [note for note in translate(_spec(name)).notes if "hand" in note], name
        assert "dropLoop" not in translate(_spec(SHOGI_FORWARD)).config
        assert "dropLoop" not in translate(_spec("dobutsu")).config

    def test_a_demotion_that_cannot_be_told_is_noted_and_left_unmarked(self) -> None:
        """A hen that demotes to a giraffe: ``~`` would demote it to the chick.

        Every move the opponent plays is still legal here, because it is handed our
        pocket, so this is a note rather than a refusal.
        """
        from chess_variant_sandbox.engine.compile import compile_spec

        spec = _with_promoted_from(_spec("dobutsu"), "H", "G")
        compiled = compile_spec(spec)
        written = translate(spec)
        assert variant.board_letters(compiled, written.chars) == written.chars
        noted = [note for note in written.notes if "hand" in note]
        assert noted == [
            "a captured 'H' goes into the hand as 'G' here and as 'H' there, which cannot "
            "be told it is promoted; the opponent's search expects the wrong piece in hand",
            "a 'H' promoted from 'C' goes into the hand as 'G' here and as 'C' there; the "
            "opponent's search expects the wrong piece in hand",
        ]

    def test_a_promotion_that_keeps_its_piece_beside_one_that_does_not_is_noted(self) -> None:
        """A queen that demotes to a pawn, beside a rook, bishop and knight that do not.

        The queen is marked, the opening ones too, because ours demotes every queen.
        ``dropLoop`` would stop that, so it is not written, and the engine's own
        promotions to the other three are demoted there and not here.
        """
        from chess_variant_sandbox.engine.compile import compile_spec

        spec = _with_promoted_from(_spec("crazyhouse"), "Q", "P")
        compiled = compile_spec(spec)
        written = translate(spec)
        assert variant.board_letters(compiled, written.chars)[compiled.piece_ids.index("Q")] == "Q~"
        assert "dropLoop" not in written.config
        assert "startFen = rnbq~kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ~KBNR[] w KQkq - 0 1" in (
            written.config.splitlines()
        )
        noted = [note for note in written.notes if "hand" in note]
        assert [note.split(" goes")[0] for note in noted] == [
            f"a {piece!r} promoted from 'P'" for piece in ("N", "B", "R")
        ]


class TestTheCompiledSpecCarriesItsSource:
    """Translating needs the descriptors, and the tables have dissolved them."""

    def test_a_compiled_ruleset_can_be_translated_without_the_json(self) -> None:
        from chess_variant_sandbox.engine.compile import compile_spec

        compiled = compile_spec(_spec("chess"))
        assert json.loads(compiled.source)["name"] == "chess"
        assert translate(compiled).name == translate(_spec("chess")).name

    def test_the_source_is_not_part_of_what_a_ruleset_is(self) -> None:
        """It must not move a hash, or every calibration run is orphaned."""
        from chess_variant_sandbox.engine.compile import compile_spec

        assert compile_spec(_spec("chess")).hash == "4c1922ffae26"


def _as_before_q438(monkeypatch: pytest.MonkeyPatch) -> None:
    """The translation as it read before Q438: shogi's promotion refusal switched off.

    Only for the premise test below, which has to play the definition that used to be
    written in order to show why it is refused now. The drop-mate refusal came off in
    Q453, so shogi's pawn crosses without switching anything.
    """
    monkeypatch.setattr(variant, "_check_leaving_the_zone", lambda *args, **kwargs: None)


def _on_leaving(spec: dict[str, Any], pieces: set[str]) -> dict[str, Any]:
    """``spec`` with ``on_leaving: true`` on the promotion entries of ``pieces``.

    Schema 15's key, written in the JSON since Q452 (B1) added it. Before then these
    tests put the flag on the compiled entry by patching the compiler, because the
    schema refused the key.
    """
    spec = copy.deepcopy(spec)
    spec["schema_version"] = 15
    entries = spec["rules"]["promotion"]
    for entry in entries if isinstance(entries, list) else [entries]:
        if entry["piece"] in pieces:
            entry["on_leaving"] = True
    return spec


def _with_pawn_in_hand(engine: Engine, state: Any) -> Any:
    """``state`` with one pawn in the second player's hand."""
    hand = [0] * engine.spec.n_types
    hand[engine.spec.piece_ids.index("P")] = 1
    return state.with_(hands=(tuple([0] * engine.spec.n_types), tuple(hand)))


def _without_drop_mate(written: Any) -> Any:
    """The same translation, with `verify`'s tolerance switched off."""
    from dataclasses import replace

    return replace(written, drop_mate="")


class TestAPromotionOrADropTheirOptionsCannotSayIsRefused:
    """Shogi's two rules, as their options say them (Q438, then Q453).

    Q432's rating run on shogi stopped with `MoveMismatch` on ``e3d4+``: their bishop or
    silver promoted on a move *out* of its zone. Real shogi allows that, and our rule did
    not: ours promoted on arriving in the zone, and theirs on a move that starts or ends
    there. Schema 15 (Q452, Q450's B1) adds ``on_leaving`` to a promotion entry, and a map
    entry that carries it passes. The committed shogi carries it on every entry since B3
    (Q454), so it translates; the game it replaced, promoting on arrival only, is derived
    (`SHOGI_ARRIVING`) and stays refused. The list shape promotes on arriving only, so the
    flag is refused there wherever the piece can leave its zone.

    The second rule, a pawn dropped to mate, is a move that loses over there and no move
    here. The game is the same (Q450, §2), so the refusal came off, and `verify` accepts
    that one extra move of theirs where their engine says it mates. If a premise test
    fails, the build has changed that rule, and the record (Q438) is worth revisiting.
    """

    ZONE = "rules.promotion: 'B', 'R' and 'S' can move out of the promotion zone"
    LISTED = "rules.promotion.on_leaving: 'P' can move out of the promotion zone"
    DROP = "drops.no_mate: "

    # 1.Pc4 Pg6 2.Bxh8 Sxh8 3.Pe4 B*e3 4.Pa4: the second player's bishop, dropped into
    # its own zone and so unpromoted, with six moves out of it.
    DROPPED = ["c3c4", "g7g6", "b2h8", "g9h8", "e3e4", "B@e3", "a3a4"]
    LEAVING = {"e3d4+", "e3c5+", "e3b6+", "e3f4+", "e3g5+", "e3h6+"}

    # A pawn dropped on a2 mates the king on a1.
    MATED = {"a1": ("K", 0), "b3": ("G", 1), "c2": ("S", 1), "e9": ("K", 1)}

    def test_shogi_promoting_on_arrival_only_is_refused_for_its_promotion(self) -> None:
        """The game B3 replaced: no ``on_leaving``, so the map refusal holds, and only it."""
        from chess_variant_sandbox import fairy

        spec = _spec(SHOGI_ARRIVING)
        assert not any("on_leaving" in rule for rule in spec["rules"]["promotion"])
        assert spec["rules"]["drops"]["no_mate"] == ["P"]
        with pytest.raises(Untranslatable) as caught:
            translate(spec)
        reasons = caught.value.reasons
        assert len(reasons) == 1, reasons
        assert reasons[0].startswith(self.ZONE), reasons[0]
        assert fairy.supports(spec) is False
        assert fairy.why_not(spec) == reasons
        assert fairy.playable(spec) is False

    def test_shogi_as_committed_translates(self) -> None:
        """Since B3 (Q454): the flag on all six entries, and nothing left to refuse."""
        from chess_variant_sandbox import fairy

        spec = _spec("shogi")
        assert all(rule.get("on_leaving") is True for rule in spec["rules"]["promotion"])
        assert fairy.supports(spec) is True
        assert fairy.why_not(spec) == []

    def test_the_map_passes_where_every_piece_that_leaves_carries_the_flag(self) -> None:
        """The committed shogi, the flag on every entry: the map, the drops, the drop mate."""
        written = translate(_spec("shogi"))
        lines = written.config.splitlines()
        assert "promotedPieceType = P:Q L:X N:Y S:Z B:H R:D" in lines
        assert "mandatoryPiecePromotion = false" in lines
        assert "shogiPawnDropMateIllegal = true" in lines
        assert written.promotion_suffix == "plus"
        assert written.drop_mate == "P"

    def test_the_flag_is_needed_only_on_a_piece_that_can_leave(self) -> None:
        """The forward pieces cannot leave, so the three that can are the whole ask."""
        translate(_on_leaving(_spec(SHOGI_ARRIVING), {"S", "B", "R"}))
        with pytest.raises(Untranslatable) as caught:
            translate(_on_leaving(_spec(SHOGI_ARRIVING), {"B", "R"}))
        assert len(caught.value.reasons) == 1, caught.value.reasons
        assert caught.value.reasons[0].startswith("rules.promotion: 'S' can move out")

    def test_mandatory_promotion_is_refused_on_the_same_moves(self) -> None:
        """There every move from the zone must promote, and here a move out of it cannot."""
        spec = _spec(SHOGI_ARRIVING)
        for rule in spec["rules"]["promotion"]:
            rule["mandatory"] = True
        assert validate_spec(spec) == []
        with pytest.raises(Untranslatable) as caught:
            translate(spec)
        assert [reason[: len(self.ZONE)] for reason in caught.value.reasons] == [self.ZONE]
        written = translate(_on_leaving(spec, {"S", "B", "R"}))
        assert "mandatoryPiecePromotion = true" in written.config.splitlines()

    def test_the_list_refuses_the_flag_where_its_piece_can_leave(self) -> None:
        """Makruk's pawn promotes on the sixth rank and moves on to the seventh.

        One promoting piece is written as a list, and a piece in ``promotionPawnTypes``
        promotes on arriving only, as ours does without the flag. Chess's pawn cannot
        move from its zone at all, so the compiler drops the flag there and the
        definition is chess's under another name (the name is the hash).
        """
        makruk = translate(_spec("makruk")).config
        assert "promotionPieceTypes = " in makruk and "promotedPieceType" not in makruk
        chess = translate(_spec("chess")).config
        with pytest.raises(Untranslatable) as caught:
            translate(_on_leaving(_spec("makruk"), {"P"}))
        assert [reason[: len(self.LISTED)] for reason in caught.value.reasons] == [
            self.LISTED
        ]
        flagged = translate(_on_leaving(_spec("chess"), {"P"})).config
        assert flagged.splitlines()[1:] == chess.splitlines()[1:]

    def test_the_flag_in_a_spec_is_read_off_the_compiled_entry(self) -> None:
        """Schema 15 made the key real, so a spec that writes it is translated, not refused.

        Until B1 (Q452) this asserted the opposite: the schema had no ``on_leaving`` and a
        spec writing it was refused as an unknown key. B1 added it to
        `schema._PROMOTION_KEYS`, and `test_every_key_of_a_rule_that_crosses_is_in_the_table`
        made it join `variant._RULE_OBJECT_KEYS` in the same change. What the flag means
        is read off `PromotionRule.on_leaving`, never the JSON: an older stamp is still
        refused by the validator, and a flag written ``false`` is the game without it.
        """
        from chess_variant_sandbox.spec.schema import SpecError

        assert "on_leaving" in variant._RULE_OBJECT_KEYS["promotion"]
        spec = _on_leaving(_spec("shogi"), {"P", "L", "N", "S", "B", "R"})
        translate(spec)
        spec["schema_version"] = 14
        with pytest.raises(SpecError, match="added in schema 15"):
            translate(spec)
        declined = _spec("shogi")
        declined["schema_version"] = 15
        for rule in declined["rules"]["promotion"]:
            rule["on_leaving"] = False
        with pytest.raises(Untranslatable) as caught:
            translate(declined)
        assert caught.value.reasons[0].startswith(self.ZONE)

    def test_the_flag_gives_ours_the_moves_theirs_has_leaving_the_zone(self) -> None:
        """Q432's ``e3d4+`` again, with the flag on: both engines now list the same moves.

        The premise test below shows the six moves out of the zone that theirs has and
        ours had not. With ``on_leaving`` on every entry ours has them too, and the drop
        mate is the only difference `verify` still tolerates, which this position lacks.
        """
        sf = _pyffish()
        spec = _spec("shogi")
        written = translate(spec)
        engine, state, ours_fen, theirs_fen = _both_after(spec, written, self.DROPPED)
        assert _board_and_hand(theirs_fen) == _board_and_hand(ours_fen)
        ours = {
            position.uci_of(engine.spec, state, move, written.chars, written.promotion_suffix)
            for move in engine.legal_moves(state)
        }
        theirs = set(sf.legal_moves(written.name, ours_fen, []))
        assert ours >= self.LEAVING
        assert ours == theirs

    def test_a_pawn_drop_that_mates_is_written_and_no_longer_refused(self) -> None:
        """`no_mate` naming the piece in the shogiPawn slot crosses, and nothing else does."""
        spec = _spec(SHOGI_FORWARD)
        assert translate(spec).drop_mate == ""
        spec["rules"]["drops"]["no_mate"] = ["P"]
        written = translate(spec)
        assert "shogiPawnDropMateIllegal = true" in written.config.splitlines()
        assert written.drop_mate == "P"
        for named in (["L"], ["P", "L"]):
            spec["rules"]["drops"]["no_mate"] = named
            with pytest.raises(Untranslatable) as caught:
                translate(spec)
            assert [reason[: len(self.DROP)] for reason in caught.value.reasons] == [
                self.DROP
            ]
            assert "shogiPawn slot" in caught.value.reasons[0]

    def test_verify_accepts_only_their_mating_drop(self) -> None:
        """The tolerance, with a stand-in engine that says which drops mate.

        Accepted: every extra move of theirs is a drop of `Variant.drop_mate` and their
        engine says it mates. Anything else leaves the lists to differ as they stand.
        """
        from chess_variant_sandbox.fairy.verify import _tolerated

        class _Says:
            def __init__(self, mates: set[str]) -> None:
                self.mates = mates
                self.asked: list[str] = []

            def mated_after(self, fen: str, move: str) -> bool:
                self.asked.append(move)
                return move in self.mates

        spec = _spec(SHOGI_FORWARD)
        spec["rules"]["drops"]["no_mate"] = ["P"]
        written = translate(spec)
        ours = ["a1b1", "P@b2"]
        says = _Says({"P@a2"})
        assert _tolerated(written, says, "", ours, [*ours, "P@a2"]) == {"P@a2"}
        # Their engine says it does not mate.
        assert _tolerated(written, _Says(set()), "", ours, [*ours, "P@a2"]) == set()
        # One extra move that is not the drop, beside one that is.
        assert _tolerated(written, says, "", ours, [*ours, "P@a2", "a1a2"]) == set()
        # A drop of another piece.
        assert _tolerated(written, _Says({"L@a2"}), "", ours, [*ours, "L@a2"]) == set()
        # Ours has a move theirs lacks.
        assert _tolerated(written, says, "", ours, ["a1b1", "P@a2"]) == set()
        # No `no_mate`, so no tolerance and no question asked.
        plain = _Says({"P@a2"})
        assert _tolerated(translate(_spec(SHOGI_FORWARD)), plain, "", ours, [*ours, "P@a2"]) == set()
        assert plain.asked == []

    def test_verify_agrees_where_their_pawn_drop_mates(self) -> None:
        """The position the premise below plays, now put to `verify` itself."""
        from chess_variant_sandbox.fairy.verify import PyffishBackend, Verification, _agree_here

        _pyffish()
        spec = _spec(SHOGI_FORWARD)
        spec["rules"]["drops"]["no_mate"] = ["P"]
        written = translate(spec)
        backend = PyffishBackend()
        backend.load(written, "")
        engine = Engine(spec)
        state = _with_pawn_in_hand(engine, engine.make_state(self.MATED, side_to_move=1))
        result = Verification(ruleset_hash=engine.spec.hash, name="", ok=True)
        assert _agree_here(engine, written, backend, state, result), result.reasons
        assert result.drop_mates == 1
        # Without the tolerance the same position differs by exactly that drop.
        bare = Verification(ruleset_hash=engine.spec.hash, name="", ok=True)
        assert not _agree_here(
            engine, _without_drop_mate(written), backend, state, bare
        )
        assert bare.reasons == ["the move lists differ: they have ['P@a2']"]

    def test_the_premise_their_piece_promotes_leaving_its_zone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q432's ``e3d4+``, from a position: theirs has it and ours, on arrival only, not."""
        sf = _pyffish()
        _as_before_q438(monkeypatch)
        spec = _spec(SHOGI_ARRIVING)
        written = translate(spec)
        engine, state, ours_fen, theirs_fen = _both_after(spec, written, self.DROPPED)
        assert _board_and_hand(theirs_fen) == _board_and_hand(ours_fen)
        ours = {
            position.uci_of(engine.spec, state, move, written.chars, written.promotion_suffix)
            for move in engine.legal_moves(state)
        }
        theirs = set(sf.legal_moves(written.name, ours_fen, []))
        assert theirs - ours == self.LEAVING
        assert ours <= theirs
        assert "e3d4" in ours and "e3d2+" in ours

    def test_the_premise_their_pawn_dropped_to_mate_is_a_move_that_loses(self) -> None:
        """A pawn dropped on a2 mates the king on a1: a move there, and none here."""
        sf = _pyffish()
        spec = _spec(SHOGI_FORWARD)
        spec["rules"]["drops"]["no_mate"] = ["P"]
        written = translate(spec)
        sf.load_variant_config(written.config)
        engine = Engine(spec)
        state = _with_pawn_in_hand(engine, engine.make_state(self.MATED, side_to_move=1))
        text = _written_fen(engine, written, state)
        ours = {
            position.uci_of(engine.spec, state, move, written.chars, written.promotion_suffix)
            for move in engine.legal_moves(state)
        }
        theirs = set(sf.legal_moves(written.name, text, []))
        assert theirs - ours == {"P@a2"}
        assert ours <= theirs
        assert sf.legal_moves(written.name, text, ["P@a2"]) == []
        assert sf.gives_check(written.name, text, ["P@a2"])
        assert sf.game_result(written.name, text, ["P@a2"]) > 0, "the mated side wins"

    @pytest.mark.slow
    def test_the_nearest_shogi_that_translates_plays_the_same_game(self) -> None:
        """The map, the drops and the demotions still cross where neither rule is asked.

        Whole games rather than 60 plies, since a piece reaches its zone after about a
        hundred (record, Q438).
        """
        from chess_variant_sandbox.fairy.verify import PyffishBackend, verify

        _pyffish()
        result = verify(_spec(SHOGI_FORWARD), PyffishBackend(), "", depth=2, games=3, plies=300)
        assert result.ok, result.summary()


class TestCliPliesDefault:
    """`chess-variant-sandbox fairy --plies`'s default, against the walk it needs (Q442).

    A 60-ply walk found Q438's shogi refusal (a promotion on a move out of the zone,
    median first difference at ply 108) in only 4 of 40 random games. At 400 plies 38 of
    40 did (record, Q442). The picker's own check (`fairy.playable`, `fairy.checked`,
    `fairy.verdict`) stays short on purpose -- it runs while a menu is drawn, and depth
    stands in for length there -- so this only pins the command's own default.
    """

    def test_the_cli_default_is_four_hundred_plies(self) -> None:
        from chess_variant_sandbox import cli

        args = cli._build_parser().parse_args(["fairy", "rulesets/chess.json"])
        assert args.plies == 400

    def test_the_pickers_own_check_keeps_its_present_length(self) -> None:
        import inspect

        from chess_variant_sandbox import fairy

        for name in ("playable", "checked", "verdict"):
            default = inspect.signature(getattr(fairy, name)).parameters["plies"].default
            assert default == 40, name


class TestTheReasonsAPageShowsAreInTheDesignersWords:
    """Q681 (`plan-understanding.md` W5b): the translator's reasons open on a path into
    the game's text, which the `fairy` command and these tests match on. What a page shows
    (`fairyReasons`, `fairyCheck.reasons`) names each rule by its card's title, one reason
    for one, and the guard enforces *a path into the game's text* over it. Q728: and says
    Fairy-Stockfish's own terms (*Betza*, *blastOnCapture*, *king slot*) as what they mean."""

    SAID = {
        "rules.wrap: Fairy-Stockfish has no board without edges":
            "the rule 'edges that join': Fairy-Stockfish has no board without edges",
        "board.shape: Fairy-Stockfish has no hexagonal board":
            "the board's shape: Fairy-Stockfish has no hexagonal board",
        "rules.check_limit is 4; the check-counting rule there counts to three":
            "the rule 'winning by giving checks' set to 4; the check-counting rule there"
            " counts to three",
        'rules.repetition_draw "never": no variant option there removes it':
            "the rule 'drawing by repetition' set to \"never\": no setting there"
            " removes it",
        "rules.explode_on_capture.by / .of: blastOnCapture is a switch there":
            "the rule 'captures that explode', 'only when taken by' and 'only when"
            " taking': explosions there are on or off for the whole game",
        "piece.captures: not a property a piece can have there":
            "a piece's 'may take only': not a property a piece can have there",
        "move.min_range: Betza has no ride that starts past its first square":
            "a way of moving's 'no nearer than': a sliding move there cannot begin beyond"
            " the first square",
        "the win condition and rules.extinction both end the game":
            "the win condition and the rule 'losing by running out of a piece' both end"
            " the game",
    }

    def test_each_shape_of_reason_names_the_rule_by_its_title(self) -> None:
        from chess_variant_sandbox.fairy import plain_reasons

        assert plain_reasons(list(self.SAID)) == list(self.SAID.values())

    def test_fairy_stockfishs_own_terms_are_said_as_what_they_mean(self) -> None:
        """Q728: every reason the translator writes in Fairy-Stockfish's vocabulary reads
        without it once a page has it, and the `fairy` command's spelling is kept."""
        import site_words

        from chess_variant_sandbox.fairy import plain_reasons

        raw = [
            "rules.explode_on_capture.by / .of: blastOnCapture is a switch there, so every "
            "capture explodes and no option says which ones do",
            "piece.facing: a piece that turns has no counterpart; a Betza notation describes "
            "one piece and a facing makes it several",
            "move.min_range: Betza has no ride that starts past its first square",
            "piece 'x' move 1: no Betza notation for a hop along these vectors",
            "piece 'x': a sweep captures what it passes over, and Betza has no such piece",
            "more than one royal piece: there is one king slot",
            "30 piece types; Fairy-Stockfish has 25 custom slots and a king",
            "26 pieces need a custom slot and there are 25",
            "rules.bare_royal: no counterpart in the current variant options",
            "rules.check_limit beside x: Fairy-Stockfish counts checks only against its king "
            "slot, so a royal outside it is never counted as checked",
            'rules.win "none" beside a royal piece: every way Fairy-Stockfish has of holding '
            "a royal (the king slot or an extinction rule) makes losing it a loss",
            "rules.foo: this translator has never been through it",
            "rules.foo.bar: rules.foo does cross, and this part of it has never been through "
            "this translator",
            "the board is 20x20; Fairy-Stockfish's large-board build stops at 12x10",
        ]
        said = plain_reasons(raw)
        assert len(said) == len(raw)
        for reason in said:
            assert not site_words.hits(reason, ("Fairy-Stockfish's own terms",)), reason
        assert said[0].endswith("explosions there are on or off for the whole game, so every"
                                " capture explodes and nothing says which ones do")

    def test_every_library_games_reasons_hold_no_path(self) -> None:
        import re

        import site_words

        from chess_variant_sandbox import fairy
        from chess_variant_sandbox.web import library

        refused = 0
        for path in sorted((Path(__file__).resolve().parents[1] / "rulesets").glob("*.json")):
            spec = load_spec(path)
            raw, shown = fairy.why_not(spec), library._fairy(spec)["fairyReasons"]
            assert len(shown) == len(raw), path.name
            refused += bool(raw)
            for reason in shown:
                assert not site_words.hits(
                    reason, ("a path into the game's text", "Fairy-Stockfish's own terms")), reason
                assert not re.search(r"\b[a-z]+_[a-z_]+\b", reason), reason
        assert refused >= 5, "the library's refusals are what this reads"


# ------------------------------------------------- what the stock engine can already say


def _with_walls(blocks_leaps: bool = False) -> dict[str, Any]:
    """`quarry`'s four walls (c4, f4 and their mirrors) on chess, vaulted or not."""
    spec = _archived("quarry")
    spec["rules"]["obstacles"]["blocks_leaps"] = blocks_leaps
    return spec


def _h_file_walls_without_knights() -> dict[str, Any]:
    """Walls on h4 and h5 that `blocks_leaps`, and nothing they can stop.

    No knight, so nothing leaps further than one square, and a one-square step passes
    over nothing (`compile._lame_path`, Q880). Before Q880 a diagonal step's
    floor-divided "leg" was the square beside it, and the h file kept that off the board.
    """
    spec = copy.deepcopy(_spec("chess"))
    spec["schema_version"] = 6
    spec["rules"]["obstacles"] = {"squares": ["h4"], "mirror": True, "blocks_leaps": True}
    spec["pieces"] = [piece for piece in spec["pieces"] if piece["id"] != "N"]
    spec["setup"]["placements"] = [
        row for row in spec["setup"]["placements"] if row["piece"] != "N"
    ]
    spec["rules"]["promotion"]["to"] = [p for p in spec["rules"]["promotion"]["to"] if p != "N"]
    return spec


def _five_checks() -> dict[str, Any]:
    spec = _spec("threecheck")
    spec["rules"]["check_limit"] = 5
    return spec


def _three_stones() -> dict[str, Any]:
    """Chess plus three stones a side: a piece with no moves, from a finite stock.

    `archive/three-stones` without its capture matrix, which is refused on its own
    account; here a stone can be taken like anything else.
    """
    spec = _spec("chess")
    spec["schema_version"] = 10
    spec["pieces"].append({"id": "S", "name": "stone", "value": 1, "moves": []})
    spec["rules"]["drops"] = {"from_captures": False, "stock": {"S": 3}}
    return spec


class TestWhatTheStockEngineCanAlreadySay:
    """Q703 (`reference/general-fairy-fork.md` §1.3): refusals the engine did not need.

    Each translation is held to `verify`'s legal-move agreement on a game that uses it,
    and the refusals that stay are the ones where the two readings part.
    """

    @staticmethod
    def _verify(spec: dict[str, Any], **kwargs: Any) -> Any:
        from chess_variant_sandbox.fairy.verify import PyffishBackend, verify

        _pyffish()
        return verify(spec, PyffishBackend(), "", **kwargs)

    def test_never_is_no_fold_rule_with_a_note_on_the_search(self) -> None:
        written = translate(_archived("perpetual"))
        assert "nFoldRule = 0" in written.config
        assert str(0xFFFF_FFFF) not in written.config
        assert any("search still scores" in note for note in written.notes)
        assert "repetition_draw" not in written.approximates

    def test_a_count_of_checks_is_the_fens(self) -> None:
        written = translate(_five_checks())
        assert "checkCounting = true" in written.config
        assert " 5+5 " in written.config

    def test_a_wall_is_a_star_in_every_fen(self) -> None:
        written = translate(_with_walls())
        start = written.config.split("startFen = ")[1].splitlines()[0]
        assert start.split()[0].split("/")[3:5] == ["2*2*2", "2*2*2"]

    def test_a_piece_with_no_moves_is_an_empty_notation(self) -> None:
        written = translate(_three_stones())
        stone = written.chars[-1].lower()
        assert f"= {stone}:\n" in written.config
        assert f"[{'S' * 3}{'s' * 3}]".upper() in written.config.upper()

    def test_quarry_stays_refused_for_the_leaps_its_walls_stop(self) -> None:
        with pytest.raises(Untranslatable) as caught:
            translate(_archived("quarry"))
        assert [r for r in caught.value.reasons if r.startswith("rules.obstacles")] == [
            "rules.obstacles.blocks_leaps: a leap there passes over a wall, and here a wall "
            "stops a leap this game makes"
        ]

    def test_blocks_leaps_that_stops_nothing_refuses_nothing(self) -> None:
        """Asked of the compiler: no leap here passes over an h-file wall."""
        translate(_h_file_walls_without_knights())

    def test_a_wall_beside_a_hop_is_refused(self) -> None:
        spec = _with_walls()
        rook = next(piece for piece in spec["pieces"] if piece["id"] == "R")
        rook["moves"].append(
            {"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "capture"}
        )
        assert validate_spec(spec) == []
        with pytest.raises(Untranslatable) as caught:
            translate(spec)
        assert any(r.startswith("rules.obstacles beside a hop (R)") for r in caught.value.reasons)

    def test_omega_is_refused_for_its_board_alone(self) -> None:
        assert fairy_why_not(_archived("omega")) == [
            "the board is 12x12; Fairy-Stockfish's large-board build stops at 12x10"
        ]

    def test_an_unlimited_stock_stays_refused(self) -> None:
        spec = _three_stones()
        spec["rules"]["drops"]["stock"]["S"] = "unlimited"
        with pytest.raises(Untranslatable) as caught:
            translate(spec)
        assert any('stock "unlimited"' in r for r in caught.value.reasons)
        assert not any("no moves" in r for r in caught.value.reasons)

    @pytest.mark.parametrize(
        "build",
        [
            lambda: _archived("perpetual"),
            _five_checks,
            _with_walls,
            _h_file_walls_without_knights,
            _three_stones,
        ],
        ids=["perpetual", "five-checks", "walls", "walls-blocking-nothing", "three-stones"],
    )
    def test_each_plays_the_same_game(self, build: Any) -> None:
        result = self._verify(build(), depth=1, games=1, plies=40)
        assert result.ok, result.summary()

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "build",
        [
            lambda: _archived("perpetual"),
            _five_checks,
            _with_walls,
            _h_file_walls_without_knights,
            _three_stones,
        ],
        ids=["perpetual", "five-checks", "walls", "walls-blocking-nothing", "three-stones"],
    )
    def test_each_plays_the_same_game_to_the_fairy_commands_depth(self, build: Any) -> None:
        """The `fairy` command's own check: depth 3 and three 400-ply games."""
        result = self._verify(build(), depth=3, games=3, plies=400)
        assert result.ok, result.summary()


def fairy_why_not(spec: dict[str, Any]) -> list[str]:
    from chess_variant_sandbox.fairy import why_not

    return why_not(spec)
