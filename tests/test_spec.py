"""Spec validation, hashing, and complexity measurement.

Validation is the trust boundary for LLM-proposed rules, so these tests are mostly
about what gets *rejected*. A validator that accepts too much does not fail here --
it fails three hours later in a tournament, on a game that was never playable.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from chess_variant_sandbox.spec import SpecError, measure_complexity, spec_hash, validate_spec
from chess_variant_sandbox.spec import schema as schema_module
from chess_variant_sandbox.spec.complexity import expand_symmetry
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import MAX_MOVES_PER_TURN, SCHEMA_VERSION, validate_or_raise

MINIMAL = {
    "schema_version": 1,
    "name": "minimal",
    "board": {"width": 4, "height": 4},
    "pieces": [
        {
            "id": "K",
            "royal": True,
            "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
        }
    ],
    "setup": {"mirror": True, "placements": [{"piece": "K", "squares": ["a1"]}]},
    "rules": {"win": "checkmate"},
}


def _fixture(name: str) -> dict:
    """A spec kept beside these tests rather than inline, because it is too big to read."""
    return json.loads((Path(__file__).parent / name).read_text(encoding="utf-8"))


def test_the_minimal_spec_is_valid() -> None:
    assert validate_spec(MINIMAL) == []


def test_chess_is_valid(chess_spec: dict) -> None:
    assert validate_spec(chess_spec) == []


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        ({"board": {"width": 1, "height": 4}}, "between 2 and"),
        ({"board": {"width": 4, "height": 4, "depth": 2}}, "unknown keys"),
        ({"pieces": []}, "non-empty array"),
        ({"name": ""}, "non-empty string"),
        # One past the current version. It moves with SCHEMA_VERSION on purpose: what
        # is under test is that an *unsupported* number is refused rather than assumed.
        ({"schema_version": SCHEMA_VERSION + 1}, "not supported"),
        ({"setup": {"placements": [{"piece": "Z", "squares": ["a1"]}]}}, "not a declared piece"),
        ({"setup": {"placements": [{"piece": "K", "squares": ["z9"]}]}}, "not a square"),
        ({"rules": {"win": "conquest"}}, "rules.win must be one of"),
        ({"rules": {"win": "checkmate", "max_plies": 0}}, "must be between"),
    ],
)
def test_malformed_specs_are_rejected(mutation: dict, fragment: str) -> None:
    spec = copy.deepcopy(MINIMAL)
    spec.update(mutation)
    errors = validate_spec(spec)
    assert any(fragment in error for error in errors), errors


def test_unknown_keys_are_rejected_rather_than_ignored() -> None:
    """A proposal that invents a field believes the engine supports something it does not."""
    spec = copy.deepcopy(MINIMAL)
    spec["pieces"][0]["moves"][0]["teleport"] = True
    assert any("unknown keys" in error for error in validate_spec(spec))


def test_every_problem_is_reported_not_just_the_first() -> None:
    spec = copy.deepcopy(MINIMAL)
    spec["board"] = {"width": 99, "height": 0}
    spec["name"] = 5
    assert len(validate_spec(spec)) >= 3


class TestCoherence:
    def test_checkmate_requires_royalty(self) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["pieces"][0]["royal"] = False
        assert any("no piece is marked royal" in e for e in validate_spec(spec))

    def test_annihilation_rejects_royalty(self) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["rules"] = {"win": "annihilation"}
        assert any("annihilation" in e for e in validate_spec(spec))

    def test_en_passant_requires_a_promoting_piece(self) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["rules"] = {"win": "checkmate", "en_passant": True}
        assert any("en_passant requires" in e for e in validate_spec(spec))

    def test_en_passant_is_refused_beside_a_turn_of_more_than_one_move(self) -> None:
        """The spec the fuzzer found, and why the pair is still refused.

        The core writes the en-passant window per move and flips the side only when the
        turn runs out, so a player with two moves could answer their own double step and
        take their own pawn in passing -- and with a `strike` rule beside it the core
        panicked on an empty own-contact table. This fixture is the smallest spec that
        did it. Q531 fixed the self-capture in the core
        (`tests/test_engine.py::TestAPawnNeverTakesItsOwnInPassing`); the refusal stays
        because whether the opponent may answer a double step made on a turn's first move
        is a design question nobody has answered. `docs/bugs.md` holds the reasoning.
        """
        spec = _fixture("_en_passant_in_a_long_turn.json")
        errors = validate_spec(spec)
        assert any(
            "rules.en_passant beside a turn longer than one move" in e for e in errors
        ), errors

        # The pair is the only thing wrong with it: drop either half and it validates,
        # so this pins the combination rather than some other defect in the fixture.
        without_turn = copy.deepcopy(spec)
        del without_turn["rules"]["turn"]
        assert validate_spec(without_turn) == []
        without_ep = copy.deepcopy(spec)
        del without_ep["rules"]["en_passant"]
        assert validate_spec(without_ep) == []

    def test_a_ramp_that_reaches_two_moves_is_a_long_turn_too(self) -> None:
        """`moves: 1, ramp: [2]` is a player moving twice in the opening, and counts."""
        spec = _fixture("_en_passant_in_a_long_turn.json")
        spec["rules"]["turn"] = {"moves": 1, "ramp": [2]}
        assert any(
            "rules.en_passant beside a turn longer than one move" in e
            for e in validate_spec(spec)
        )
        spec["rules"]["turn"] = {"moves": 1, "ramp": [1]}
        assert validate_spec(spec) == []

    def test_castling_must_protect_royalty(self) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["pieces"].append(
            {"id": "R", "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]}
        )
        spec["rules"]["castling"] = {"king": "R", "rook": "R"}
        assert any("not a royal piece" in e for e in validate_spec(spec))

    def test_min_range_may_not_exceed_range(self) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["pieces"][0]["moves"][0] = {
            "kind": "slide",
            "vectors": [[0, 1]],
            "range": 2,
            "min_range": 3,
        }
        assert any("min_range" in e for e in validate_spec(spec))

    @pytest.mark.parametrize(
        ("kind", "key"),
        [("leap", "range"), ("leap", "min_range"), ("lame", "range"), ("lame", "min_range")],
    )
    def test_a_distance_on_a_leap_is_refused(self, kind: str, key: str) -> None:
        """A leap is one jump per vector, so a distance on one is a spec that means two things.

        `engine/compile.py::_compile_square` read `range` off a leap and never used it,
        so `[[1, 1]]` with `range: 7` compiled to a one-square diagonal step; `fairy/`
        read the same key and translated a rider limited to seven. Two readings of one
        rule is exactly what `verify.py` exists to catch, and the honest fix is upstream
        of both: refuse the key, and say how to spell the reach that was meant.
        """
        spec = copy.deepcopy(MINIMAL)
        spec["schema_version"] = 2  # `lame` is schema 2
        spec["pieces"][0]["moves"][0] = {"kind": kind, "vectors": [[1, 2]], key: 2}
        errors = validate_spec(spec)
        assert any(f"{key} means nothing on a {kind}" in e for e in errors), errors
        # The same descriptor without the distance is a leaper, and is fine.
        del spec["pieces"][0]["moves"][0][key]
        assert validate_spec(spec) == []


class TestInertRulesAreRefused:
    """A rule that is charged for and declares nothing is refused by name.

    The enumeration behind these is `docs/reference/bugs-investigations.md`, "Rules that declare nothing": every
    case was measured inert -- the nearest baseline with and without the declaration,
    six seeds of greedy against greedy, identical in plies, reason and winner -- before
    it was refused. Each test here asserts both halves, because a refusal that fires on
    the rule *doing something* is worse than no refusal at all: the spec is charged for
    the declaration by `complexity._RULE_COSTS` and `rule_mass` is the denominator of
    `rule_yield`, so getting this wrong deflates the ratio the index rewards in the
    other direction.

    The last two tests are the boundary, and they are the ones to read first when a
    later case is added: an inert *spelling* is legal and free, and a colouring nothing
    reads is not inert at all.
    """

    @staticmethod
    def _base() -> dict:
        """`MINIMAL`, stamped high enough for the rules under test to be declared."""
        spec = copy.deepcopy(MINIMAL)
        spec["schema_version"] = 3  # `captures` is schema 3; the rest are schema 2
        return spec

    @pytest.mark.parametrize(
        ("rule", "value", "fragment"),
        [
            ("must_capture", True, "rules.must_capture can never bind"),
            ("explode_on_capture", {"radius": 1}, "rules.explode_on_capture can never fire"),
            ("drops", {"from_captures": True}, "rules.drops can never fill a hand"),
        ],
    )
    def test_a_rule_about_capturing_needs_a_capture_to_be_possible(
        self, rule: str, value: object, fragment: str
    ) -> None:
        """`captures: []` on every piece is one of the two spellings of "nothing is taken".

        A hand fills only in `make`, only on a capture, and only under `from_captures`,
        so "no capture is possible" is one test serving all three rules -- and it serves
        a wild piece too, which eats under this same matrix.
        """
        spec = self._base()
        spec["pieces"][0]["captures"] = []  # never takes, and still blocks
        spec["rules"][rule] = value
        errors = validate_spec(spec)
        assert any(fragment in error for error in errors), errors
        # The other spelling: every descriptor `mode: "move"`.
        del spec["pieces"][0]["captures"]
        spec["pieces"][0]["moves"][0]["mode"] = "move"
        assert any(fragment in error for error in validate_spec(spec))
        # And with something to take, the same declaration is a rule.
        del spec["pieces"][0]["moves"][0]["mode"]
        assert validate_spec(spec) == []

    def test_drops_that_can_never_fill_a_hand_are_refused(self) -> None:
        """`{"from_captures": false}` compiles to a real `DropRule` and is charged 2.0.

        `{}` is a different thing: falsy, skipped by `_count_special_rules`, compiled to
        `None`, and therefore free -- so it stays legal, on the grounds
        `"en_passant": false` does.
        """
        spec = self._base()
        spec["rules"]["drops"] = {"from_captures": False}
        errors = validate_spec(spec)
        assert any("never anything to drop" in error for error in errors), errors
        spec["rules"]["drops"] = {}
        assert validate_spec(spec) == []
        spec["rules"]["drops"] = {"from_captures": True}
        assert validate_spec(spec) == []

    @pytest.mark.parametrize(
        ("rule", "value"),
        [
            ("bare_royal", "loss"),
            ("check_limit", 3),
            ("forbid_check", True),
            ("royal_facing", "forbidden"),
        ],
    )
    def test_a_rule_about_a_royal_piece_needs_one(self, rule: str, value: object) -> None:
        """Four rules charged 1.0 each for a sentence about a piece that is not there.

        The core already guards `forbid_check` by name -- `if t.forbid_check &&
        !t.royal_types.is_empty()` -- which is the engine saying the rule is nothing
        while the validator charges for it.
        """
        spec = self._base()
        spec["pieces"][0]["royal"] = False
        spec["rules"] = {"win": "annihilation", rule: value}
        errors = validate_spec(spec)
        assert any(
            f"rules.{rule}" in error and "no piece is marked royal" in error
            for error in errors
        ), errors
        # With a royal piece to be about, each is a rule.
        spec["pieces"][0]["royal"] = True
        spec["rules"] = {"win": "checkmate", rule: value}
        assert validate_spec(spec) == []

    def test_a_check_limit_beside_forbid_check_can_never_be_reached(self) -> None:
        """If no legal move may give check, the counter never advances.

        The explosion arm is the whole of the guard rather than a nicety: the
        `forbid_check` filter plays the move *without* the explosion and says so in its
        own comment, while `delivered` is computed after the blast -- so under an
        explosion a line the blast opens still ticks the counter, and the pair is not
        inert.
        """
        spec = self._base()
        spec["rules"] = {"win": "checkmate", "check_limit": 3, "forbid_check": True}
        errors = validate_spec(spec)
        assert any('beside "forbid_check"' in error for error in errors), errors
        spec["rules"] = {"win": "checkmate", "check_limit": 3}
        assert validate_spec(spec) == []
        spec["rules"] = {
            "win": "checkmate",
            "check_limit": 3,
            "forbid_check": True,
            "explode_on_capture": {"radius": 1},
        }
        assert validate_spec(spec) == []

    def _promoting(self, **move: object) -> dict:
        """`_base()` plus a promoting piece whose one move is `move`, and en passant.

        The board is 5x5 so the promotion rank is not the rank the piece starts on, and
        the piece is placed so nothing else in `_check_inert_rules` has anything to say
        about it. `_base()`'s stamp is what admits the `lame` case, which is schema 2.
        """
        spec = self._base()
        spec["board"] = {"width": 5, "height": 5}
        spec["pieces"].append({"id": "P", "moves": [dict(move)]})
        spec["setup"]["placements"].append({"piece": "P", "squares": ["b1"]})
        spec["pieces"].append(
            {"id": "G", "moves": [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}]}
        )
        spec["rules"]["promotion"] = {"piece": "P", "rank": "last", "to": ["G"]}
        spec["rules"]["en_passant"] = True
        return spec

    @pytest.mark.parametrize(
        ("move", "refused"),
        [
            # A leap passes over nothing at all, whatever its vectors say: `_ep_passes`
            # reads a *ray*, and the compiler used to halve a leap's delta for a
            # midpoint, which is a question with no answer.
            ({"kind": "leap", "vectors": [[0, 2]]}, True),
            ({"kind": "slide", "vectors": [[0, 1]], "range": 1}, True),
            # Range 2 is the pawn's own double step; unlimited and 3 reach further.
            ({"kind": "slide", "vectors": [[0, 1]], "range": 2}, False),
            ({"kind": "slide", "vectors": [[0, 1]], "range": 3}, False),
            ({"kind": "slide", "vectors": [[0, 1]]}, False),
            # `min_range` decides whether the *second* square of the ray carries a mode,
            # so 2 is the last value that does and 3 is the first that does not.
            ({"kind": "slide", "vectors": [[0, 1]], "range": 4, "min_range": 2}, False),
            ({"kind": "slide", "vectors": [[0, 1]], "range": 4, "min_range": 3}, True),
            # En passant answers a piece slipping *past* another, so the second square
            # has to carry MODE_MOVE: a capture-only descriptor never does.
            ({"kind": "slide", "vectors": [[0, 1]], "mode": "capture"}, True),
            ({"kind": "slide", "vectors": [[0, 1]], "mode": "move"}, False),
            # A hop has a ray, and it still never answers: it lands past its screen, so
            # reaching the ray's second square means a piece stands on the first, and the
            # window would open on an occupied square (Q121).
            ({"kind": "hop", "vectors": [[0, 1]], "range": 1}, True),
            ({"kind": "hop", "vectors": [[0, 1]], "range": 3}, True),
            # A lame leap compiles to a two-square ray whose first square must be empty.
            ({"kind": "lame", "vectors": [[1, 2]]}, False),
        ],
    )
    def test_en_passant_needs_a_two_square_move_to_answer(
        self, move: dict, refused: bool
    ) -> None:
        """The rule answers a double step and nothing else, so it needs one to exist.

        `compile._ep_passes` builds from the promoting piece's own slides -- the second
        square of a ray, where that square carries `MODE_MOVE` -- so with no such move
        the table is empty and the spec is charged 1.0 for nothing. Decided by the user
        on 2026-09-07: *"En passant is incompatible with anything but a standard pawn
        and should be refused."*
        """
        errors = validate_spec(self._promoting(**move))
        fired = [e for e in errors if "rules.en_passant can generate no capture" in e]
        assert bool(fired) == refused, errors
        if not refused:
            assert errors == []

    def test_the_en_passant_refusal_is_sound_rather_than_exact(self) -> None:
        """It reads descriptors, never geometry, so a board that empties the table is
        still accepted.

        A four-square slide on a board two ranks deep compiles to `ep_passes` with no
        entries -- the ray runs off the edge before its second square -- and this
        accepts it anyway. That direction is the design: `_castling_ranks` refuses to
        keep a second `_compile_square` in the validator for the same reason, and
        refusing *fewer* specs than the compiled table would is the only error this may
        make.
        """
        from chess_variant_sandbox.engine.compile import compile_spec

        spec = self._promoting(kind="slide", vectors=[[0, 1]], range=4)
        spec["board"] = {"width": 5, "height": 2}
        spec["rules"]["promotion"]["rank"] = 1
        assert validate_spec(spec) == []
        compiled = compile_spec(spec)
        assert sum(len(sq) for frame in compiled.ep_passes for sq in frame) == 0

    def test_the_en_passant_refusal_leaves_a_malformed_spec_alone(self) -> None:
        """A spec already collecting errors is never also told a rule of its is empty.

        The two refusals that come first -- no `promotion` at all, and more than one --
        are what make *the* promoting piece a single piece, and this clause has nothing
        to say until they are satisfied. `_capture_is_possible` takes the same line, for
        the same reason: an inert-rule refusal has to be a fact rather than a knock-on
        from a typo somewhere else.
        """
        spec = self._promoting(kind="slide", vectors=[[0, 1]], range=1)
        del spec["rules"]["promotion"]
        errors = validate_spec(spec)
        assert any("rules.en_passant requires rules.promotion" in e for e in errors)
        assert not any("can generate no capture" in e for e in errors), errors

        spec = self._promoting(kind="slide", vectors=[[0, 1]], range=1)
        spec["pieces"].append({"id": "S", "moves": [{"kind": "leap", "vectors": [[0, 1]]}]})
        spec["setup"]["placements"].append({"piece": "S", "squares": ["c1"]})
        spec["rules"]["promotion"] = [
            {"piece": "P", "rank": "last", "to": ["G"]},
            {"piece": "S", "rank": "last", "to": ["G"]},
        ]
        errors = validate_spec(spec)
        assert any("needs a single promoting piece" in e for e in errors)
        assert not any("can generate no capture" in e for e in errors), errors

    def test_castling_needs_a_rook_starting_on_the_king_s_rank(self) -> None:
        """Both pieces must be unmoved, and `unmoved` is only ever cleared.

        `generate_castling` looks for the rook along the king's own rank, and the bit is
        set once at ply zero from `setup.placements`; `make` clears it for every square
        it writes, so a rook arriving by move, drop, promotion or strike is never a
        partner. That is the narrow case -- a castling the *walls* block is
        `_check_obstacle_coherence`'s, and one nobody ever plays is the tournament's.
        """
        spec = self._base()
        spec["board"] = {"width": 5, "height": 5}
        spec["pieces"].append(
            {"id": "R", "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]}
        )
        spec["setup"]["placements"] = [
            {"piece": "K", "squares": ["c1"]},
            {"piece": "R", "squares": ["a2"]},  # a rank no king ever starts on
        ]
        spec["rules"]["castling"] = {"king": "K", "rook": "R"}
        errors = validate_spec(spec)
        assert any("rules.castling can never happen" in error for error in errors), errors
        # On the king's rank it is a rule again.
        spec["setup"]["placements"][1]["squares"] = ["a1"]
        assert validate_spec(spec) == []

    def test_a_rule_about_a_piece_that_can_never_exist_is_refused(self) -> None:
        """*Cannot exist* is a fact about the spec; *never gets there* is a game tree.

        The routes into play are a closed set -- placed at ply zero, a promotion target,
        a strike chain's `to`, out of a hand, a wild piece -- which is what makes this
        one decidable while "a promotion no pawn reaches" is not.
        """
        spec = self._base()
        spec["pieces"].append(
            {"id": "P", "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1}]}
        )
        spec["pieces"].append(
            {"id": "G", "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]}
        )
        spec["setup"]["placements"].append({"piece": "P", "squares": ["b1"]})
        spec["rules"]["reach"] = {"piece": "G", "squares": ["d1"], "mirror": False}
        errors = validate_spec(spec)
        assert any(
            "rules.reach is about 'G'" in error and "can never stand on the board" in error
            for error in errors
        ), errors
        # A promotion target is a route into play, so G can exist and the rule is real.
        spec["rules"]["promotion"] = {"piece": "P", "rank": "last", "to": ["G"]}
        assert validate_spec(spec) == []

    def test_a_promotion_for_a_piece_that_can_never_exist_is_refused(self) -> None:
        spec = self._base()
        spec["pieces"].append(
            {"id": "P", "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1}]}
        )
        spec["pieces"].append(
            {"id": "G", "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1}]}
        )
        spec["setup"]["placements"].append({"piece": "P", "squares": ["b1"]})
        spec["rules"]["promotion"] = {"piece": "G", "rank": "last", "to": ["P"]}
        errors = validate_spec(spec)
        assert any(
            "rules.promotion is about 'G'" in error for error in errors
        ), errors
        # Place it and the rule is about a piece somebody can play.
        spec["setup"]["placements"].append({"piece": "G", "squares": ["c1"]})
        assert validate_spec(spec) == []

    @pytest.mark.parametrize("listed", [False, True], ids=["one rule", "a list of rules"])
    def test_a_promotion_rank_past_the_board_is_refused(self, listed: bool) -> None:
        """A lone `rank` is bounded by the board, as each entry of a `ranks` band was.

        `schema.promotion_squares` skips a rank off the board, so a zone stated as one is
        empty: the rule is charged for and can never fire. `tunnel-horizontal` stated
        `rank: 9` on an 8-high board and validated clean (Q102 found it) until Q434
        retired the game. The refusal uses the band's words, so the two spellings of a
        zone past the board read alike. It is a refusal of a *new* spec only, and
        `tests/test_new_spec_refusals.py` holds where it runs and where it does not.
        """
        spec = self._base()
        spec["pieces"].append(
            {"id": "P", "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1}]}
        )
        spec["pieces"].append(
            {"id": "G", "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]}
        )
        spec["setup"]["placements"].append({"piece": "P", "squares": ["b1"]})
        where = "rules.promotion[0]" if listed else "rules.promotion"

        def promoting(**zone: object) -> dict:
            rule = {"piece": "P", "to": ["G"], **zone}
            spec["rules"]["promotion"] = [rule] if listed else rule
            return spec

        height = spec["board"]["height"]
        assert validate_spec(promoting(rank=height)) == [
            f"{where}.rank {height} is not a rank on this board"
        ]
        assert validate_spec(promoting(ranks=[height])) == [
            f"{where}.ranks entry {height} is not a rank on this board"
        ]
        # The last rank on the board, however it is spelled, is a zone.
        assert validate_spec(promoting(rank=height - 1)) == []
        assert validate_spec(promoting(rank="last")) == []
        assert validate_spec(promoting(ranks=[height - 1])) == []

    def test_an_inert_spelling_stays_legal(self) -> None:
        """The three spellings that are the standing proof schemas 4-6 were inert bumps.

        `_RULE_ACTIVE` charges nothing for any of them, `tests/test_schema_six.py`,
        `test_wrap.py`, `test_schema_two.py` and `test_measurement.py` assert them, and
        `fairy._check_unsupported_rules` reads the *compiled* spec precisely so a spec
        carrying one still translates rather than refusing. A refusal here would delete
        all three and buy no elegance, because none of them costs anything.
        """
        spec = self._base()
        spec["schema_version"] = 6
        spec["rules"].update(
            {
                "turn": {"moves": 1},
                "wrap": {"files": "none"},
                "strike": {"effect": "capture"},
                "stalemate": "draw",
                "checkmate": "loss",
                "royal_facing": "allowed",
                "en_passant": False,
                "must_capture": False,
                "drops": {},
            }
        )
        assert validate_spec(spec) == []

    def test_a_colouring_nothing_scores_is_not_inert(self) -> None:
        """`territory` at `win_at: 0` looks like scenery and measured DIFFERENT on every seed.

        Under `permanent` a claim is `irreversible`, so it clears the repetition window
        and resets the no-progress clock from move one; under `repaint` the colouring
        enters the Zobrist key, so two positions coloured differently are no longer the
        same position. The same trap catches `explode_on_capture.immune` covering every
        piece: immunity protects *neighbours*, and the capturer and the victim go
        regardless.
        """
        spec = self._base()
        spec["schema_version"] = 4
        for mode in ("permanent", "repaint"):
            spec["rules"]["territory"] = {"mode": mode, "win_at": 0}
            assert validate_spec(spec) == []
        del spec["rules"]["territory"]
        spec["rules"]["explode_on_capture"] = {"radius": 1, "immune": ["K"]}
        assert validate_spec(spec) == []


def test_validate_or_raise_carries_every_error() -> None:
    spec = copy.deepcopy(MINIMAL)
    spec["board"] = {"width": 0, "height": 0}
    with pytest.raises(SpecError) as excinfo:
        validate_or_raise(spec)
    assert len(excinfo.value.errors) >= 2


class TestHashing:
    def test_flavour_text_does_not_change_the_hash(self) -> None:
        """Renaming a game must not let it be re-evaluated as if it were new."""
        renamed = copy.deepcopy(MINIMAL)
        renamed["name"] = "totally different game"
        renamed["description"] = "a bold new direction"
        assert spec_hash(renamed) == spec_hash(MINIMAL)

    def test_mechanics_do_change_the_hash(self) -> None:
        altered = copy.deepcopy(MINIMAL)
        altered["board"]["width"] = 5
        assert spec_hash(altered) != spec_hash(MINIMAL)

    def test_key_order_does_not_change_the_hash(self) -> None:
        reordered = {k: MINIMAL[k] for k in reversed(list(MINIMAL))}
        assert spec_hash(reordered) == spec_hash(MINIMAL)


class TestSettlingTheVersionStamp:
    """`schema_version` is a claim about which features a spec uses, and it is inside the
    mechanics hash -- so the stamp is not a preference, it is part of the game's identity.
    `settle_schema_version` decides it from the validator rather than from a table, in
    both directions, which is what makes two people who wrote the same game get one hash.
    """

    def test_a_stamp_higher_than_the_spec_needs_comes_down(self) -> None:
        """The half that already existed: an add-and-remove pair of operators has to be
        able to return a candidate to its parent, and it could not while the stamp stayed
        where the add left it."""
        from chess_variant_sandbox.spec.transform import settle_schema_version

        raised = copy.deepcopy(MINIMAL)
        raised["schema_version"] = 9
        assert settle_schema_version(raised)["schema_version"] == 1
        assert spec_hash(raised) == spec_hash(MINIMAL)

    def test_a_stamp_lower_than_the_spec_needs_goes_up(self) -> None:
        """The half that was missing. A spec declaring 1 and using a schema-6 rule is not
        a version-1 spec: it is a spec whose stamp is wrong, and refusing it asks a person
        to hand-edit a number the validator can work out."""
        from chess_variant_sandbox.spec.transform import settle_schema_version

        walled = copy.deepcopy(MINIMAL)
        walled["rules"]["obstacles"] = {"squares": ["c3"]}
        assert validate_spec(walled), "a schema-1 stamp on a schema-6 rule should refuse"
        assert settle_schema_version(walled)["schema_version"] == 6
        assert validate_spec(walled) == []

    def test_it_is_the_lowest_version_that_validates_and_not_the_newest(self) -> None:
        """The stamp says what the spec *uses*, so settling must not simply reach for
        `SCHEMA_VERSION`: that would put every hand-built game on a different hash from
        the identical one somebody else wrote against an older schema."""
        from chess_variant_sandbox.spec.transform import settle_schema_version

        for start in (None, 1, 4, 9):
            spec = copy.deepcopy(MINIMAL)
            spec["rules"]["obstacles"] = {"squares": ["c3"]}
            if start is None:
                spec.pop("schema_version")
            else:
                spec["schema_version"] = start
            assert settle_schema_version(spec)["schema_version"] == 6, start

    def test_a_spec_no_version_can_save_is_left_alone(self) -> None:
        """Invalid for a reason that is not the stamp. Moving the number would only change
        which complaint the validator makes, and the caller's spec is not the place to
        record a guess."""
        from chess_variant_sandbox.spec.transform import settle_schema_version

        broken = copy.deepcopy(MINIMAL)
        broken["schema_version"] = 4
        broken["pieces"] = []
        assert settle_schema_version(broken)["schema_version"] == 4

    def test_two_versions_of_one_game_settle_to_one_hash(self) -> None:
        """The whole point, and the waste `test_a_known_game_is_not_rediscovered` exists
        to prevent: the same mechanics written by two people who guessed differently about
        the stamp are one game, not two."""
        from chess_variant_sandbox.spec.transform import settle_schema_version

        one = copy.deepcopy(MINIMAL)
        one["schema_version"] = 2
        two = copy.deepcopy(MINIMAL)
        two["schema_version"] = 7
        assert spec_hash(one) != spec_hash(two)
        assert spec_hash(settle_schema_version(one)) == spec_hash(settle_schema_version(two))


class TestSymmetryExpansion:
    @pytest.mark.parametrize(
        ("vector", "symmetry", "count"),
        [
            ((1, 0), "none", 1),
            ((1, 1), "lateral", 2),
            ((1, 0), "rotate4", 4),
            ((1, 1), "rotate4", 4),
            ((1, 2), "dihedral8", 8),
            ((1, 0), "dihedral8", 4),
            ((1, 1), "dihedral8", 4),
        ],
    )
    def test_group_sizes(self, vector: tuple[int, int], symmetry: str, count: int) -> None:
        assert len(expand_symmetry(vector, symmetry)) == count

    def test_knight_expansion_is_the_eight_familiar_directions(self) -> None:
        assert expand_symmetry((1, 2), "dihedral8") == {
            (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        }


class TestComplexity:
    def test_chess_counts(self, chess_spec: dict) -> None:
        complexity = measure_complexity(chess_spec)
        assert complexity.piece_types == 6
        assert complexity.move_descriptors == 8
        assert complexity.conditional_moves == 1  # the pawn's double step
        assert complexity.special_rules == 3  # promotion, en passant, castling

    def test_exceptions_cost_more_than_directions(self, chess_spec: dict) -> None:
        """Rule mass has to reflect what a player must remember, not what a parser reads."""
        without_specials = copy.deepcopy(chess_spec)
        without_specials["rules"] = {"win": "checkmate"}
        assert measure_complexity(without_specials).rule_mass < measure_complexity(chess_spec).rule_mass


class TestDescription:
    """The prose. Read by the judge, by the proposer, and now by a player mid-game.

    It had no tests, and the cost of that showed up as a rock-paper-scissors game
    rendered as three identically-moving pieces with no capture relation, won by
    annihilation, with its territory rule missing -- a description of no game at all.
    The failure is silent in a way an engine bug is not: everything plays correctly and
    only the sentence is wrong.
    """

    def test_the_block_is_composed_of_the_parts(self, chess_spec: dict) -> None:
        """One statement of what a rule says, two renderings of it.

        The page shows the parts and the model reads the block, so a rule stated in one
        and not the other is a rule two readers disagree about.
        """
        from chess_variant_sandbox.spec.describe import (
            describe_pieces,
            describe_rules,
            describe_setup,
            describe_spec,
        )

        block = describe_spec(chess_spec)
        for piece in describe_pieces(chess_spec):
            assert f"{piece.id} - {piece.name}" in block
            for move in piece.moves:
                assert move in block
        for line in describe_setup(chess_spec) + describe_rules(chess_spec):
            assert line in block

    @pytest.mark.parametrize("name", ["rps-v1", "rps-v2", "rps-v4", "rps-v5",
                                      "rps-hierarchy-chess"])
    def test_a_restricted_capture_relation_is_stated_as_a_relation(self, name: str) -> None:
        """Which piece beats which is the shape of the game.

        A reader who has to assemble it out of six separate piece entries has not been
        told it, so it is a rule as well as a note -- and the note is what tells you
        that an unlisted piece still blocks a line rather than being ignored.
        """
        from pathlib import Path

        from chess_variant_sandbox.spec import load_spec
        from chess_variant_sandbox.spec.describe import describe_pieces, describe_rules

        spec = load_spec(Path(__file__).resolve().parents[1] / "rulesets" / f"{name}.json")
        relation = next(
            line for line in describe_rules(spec) if line.startswith("What a piece may")
        )
        for piece in spec["pieces"]:
            captures = piece["captures"]
            assert f"{piece['id']} takes " in relation
            for victim in captures:
                assert victim in relation.split(f"{piece['id']} takes ")[1].split(";")[0]

        # Empty and absent are different, and hierarchy chess has both.
        notes = {p.id: p.notes for p in describe_pieces(spec)}
        for piece in spec["pieces"]:
            text = " ".join(notes[piece["id"]])
            assert ("never capture anything" in text) == (piece["captures"] == [])

    def test_territory_and_scoring_reach_the_page(self) -> None:
        from pathlib import Path

        from chess_variant_sandbox.spec import load_spec
        from chess_variant_sandbox.spec.describe import describe_rules

        rulesets = Path(__file__).resolve().parents[1] / "rulesets"
        permanent = " ".join(describe_rules(load_spec(rulesets / "rps-v2.json")))
        assert "claims it for you" in permanent
        assert "never be taken back" in permanent
        assert "41 of the 81 squares wins" in permanent

        # The same rule with the opposite property, which is the whole difference
        # between v2 and v4 and is invisible in a spec diff of one word.
        repaint = " ".join(describe_rules(load_spec(rulesets / "rps-v4.json")))
        assert "claims it back" in repaint
        assert "never be taken back" not in repaint

        counted = " ".join(describe_rules(load_spec(rulesets / "rps-v5.json")))
        assert "when the clock runs out" in counted.lower()
        assert "ground each player has claimed" in counted

    def test_a_region_only_condition_leaves_no_dangling_clause(self) -> None:
        """Every xiangqi soldier read `... [2 directions] - ` and stopped there.

        The condition fell through every branch and the joiner was appended anyway.
        """
        from pathlib import Path

        from chess_variant_sandbox.spec import load_spec
        from chess_variant_sandbox.spec.describe import describe_pieces

        spec = load_spec(Path(__file__).resolve().parents[1] / "rulesets" / "xiangqi.json")
        soldier = next(piece for piece in describe_pieces(spec) if piece.id == "S")
        assert not any(move.rstrip().endswith("-") for move in soldier.moves)
        assert any("only from" in move for move in soldier.moves)
        # The palace and the river are rules, and they were stated nowhere.
        general = next(piece for piece in describe_pieces(spec) if piece.id == "K")
        assert any("may never leave" in note for note in general.notes)
        horse = next(piece for piece in describe_pieces(spec) if piece.id == "H")
        assert any("stopped by a piece" in move for move in horse.moves)

    def test_a_hop_says_whose_piece_it_leaps_and_where_it_lands(self, chess_spec: dict) -> None:
        """`screen` and `screen_range` have no committed ruleset, so they get a probe."""
        from chess_variant_sandbox.spec.describe import describe_pieces

        spec = copy.deepcopy(chess_spec)
        spec["schema_version"] = 3
        spec["pieces"][1]["moves"] = [
            {"kind": "hop", "vectors": [[1, 0]], "screen": "own"},
            {"kind": "hop", "vectors": [[1, 1]], "screen": "enemy", "screen_range": 1},
            {"kind": "hop", "vectors": [[0, 1]], "screen_range": 3},
        ]
        moves = describe_pieces(spec)[1].moves
        assert "over one of your own pieces" in moves[0]
        assert "over one enemy piece, landing on the square just past it" in moves[1]
        assert "landing at most three squares past it" in moves[2]

    def test_a_longer_turn_is_the_first_thing_said(self, chess_spec: dict) -> None:
        """It changes what "a turn" means, so every rule after it reads differently."""
        from chess_variant_sandbox.spec.describe import describe_rules

        spec = copy.deepcopy(chess_spec)
        spec["schema_version"] = 4
        spec["rules"]["turn"] = {"moves": 2, "ramp": [1]}
        lines = describe_rules(spec)
        assert lines[1].startswith("A turn is 2 moves")
        assert "taking 1 move before 2" in lines[1]

        # Stating the default explicitly is the game that was already there, and it
        # must not grow a sentence saying so.
        spec["rules"]["turn"] = {"moves": 1}
        assert not any("A turn is" in line for line in describe_rules(spec))

    def test_the_second_army_is_derived_the_way_the_spec_says(self) -> None:
        """Mirrored and rotated are different boards, and both said "mirrors"."""
        from pathlib import Path

        from chess_variant_sandbox.spec import load_spec
        from chess_variant_sandbox.spec.describe import describe_setup

        rulesets = Path(__file__).resolve().parents[1] / "rulesets"
        assert any("rotated 180" in line
                   for line in describe_setup(load_spec(rulesets / "shogi.json")))
        assert any("mirrors the first" in line
                   for line in describe_setup(load_spec(rulesets / "chess.json")))
        assert any("stated separately" in line
                   for line in describe_setup(load_spec(rulesets / "horde.json")))


RULESET_PATHS = sorted(
    (__import__("pathlib").Path(__file__).resolve().parents[1] / "rulesets").glob("*.json")
)


@pytest.mark.parametrize("path", RULESET_PATHS, ids=lambda p: p.stem)
def test_every_rule_a_player_is_charged_for_is_a_rule_they_are_told(path) -> None:
    """The invariant that would have caught two schema bumps' worth of silence.

    `complexity._RULE_COSTS` already decides which keys are exceptions a player has to
    be taught -- that is what rule mass *is* -- so the honest check is that the same
    keys are the ones the prose says something about. Removing an active rule has to
    change the description; if it does not, the description never mentioned it.

    This is the test that makes a schema bump cheap to get right: add a rule to the cost
    table without teaching `describe.py`, and this fails by name rather than three
    months later on a page that quietly described a different game.
    """
    from chess_variant_sandbox.spec import load_spec
    from chess_variant_sandbox.spec.complexity import _RULE_COSTS, measure_complexity
    from chess_variant_sandbox.spec.describe import describe_spec

    spec = load_spec(path)
    baseline = describe_spec(spec)
    charged = measure_complexity(spec).special_rules

    for key in list(spec.get("rules", {})):
        if key not in _RULE_COSTS:
            continue
        without = copy.deepcopy(spec)
        del without["rules"][key]
        # Only rules the complexity model actually charges for: a spec that spells out
        # a default is the game that was already there, and saying so would be noise.
        if measure_complexity(without).special_rules == charged:
            continue
        assert describe_spec(without) != baseline, (
            f"{path.stem}: rules.{key} is charged as a special rule and is stated nowhere"
        )


def _stated_properties(piece: dict[str, object]) -> list[str]:
    """The piece properties a ruleset states, of the four `describe` must say.

    A property spelled out at its default is the absence of the property, and saying so
    would be noise -- `"royal": false` is every piece that is not a king. `captures` is the
    exception: an empty list is a real statement, and a piece that may take nothing at all
    is not a piece with no restriction.
    """
    return [
        key
        for key in ("captures", "region", "promoted_from", "royal")
        if key in piece and (piece[key] or key == "captures")
    ]


# Only the rulesets that state one: a ruleset that states none has nothing to check, and
# until Q745 it was a case whose loop ran over nothing (antichess and dull).
STATING_PATHS = [
    path
    for path in RULESET_PATHS
    if any(
        _stated_properties(piece)
        for piece in json.loads(path.read_text(encoding="utf-8"))["pieces"]
    )
]


@pytest.mark.parametrize("path", STATING_PATHS, ids=lambda p: p.stem)
def test_every_piece_property_is_stated(path) -> None:
    """The same check one level down, where `captures` and `region` went missing."""
    from chess_variant_sandbox.spec import load_spec
    from chess_variant_sandbox.spec.describe import describe_spec

    spec = load_spec(path)
    baseline = describe_spec(spec)
    for index, piece in enumerate(spec["pieces"]):
        for key in _stated_properties(piece):
            without = copy.deepcopy(spec)
            del without["pieces"][index][key]
            assert describe_spec(without) != baseline, (
                f"{path.stem}: {piece['id']}.{key} is stated nowhere"
            )


class TestTheModelContractMatchesTheValidator:
    """`spec/jsonschema.py` is generated from the validator's constants, so it must
    cover them exactly.

    Both directions are silent when they break, and the quieter one is why this exists.
    A key the schema has and the validator does not produces a proposal rejected for a
    field the model was told to use. A key the validator has and the schema does not is
    worse: `strict` writes `additionalProperties: false`, so the model *cannot emit it*.
    `rules.turn` shipped that way -- built, tested, documented, and unreachable by the
    only thing in the project that writes new rulesets.

    The sibling of `TestDescription`'s coverage tests. That one asks whether a rule
    reaches the reader; this one asks whether it reaches the writer.
    """

    def test_every_rule_the_validator_accepts_can_be_proposed(self) -> None:
        rules = set(rule_spec_schema()["properties"]["rules"]["properties"])
        assert rules == set(schema_module._RULES_KEYS)

    def test_every_piece_and_move_key_can_be_proposed(self) -> None:
        pieces = rule_spec_schema()["properties"]["pieces"]["items"]["properties"]
        assert set(pieces) == set(schema_module._PIECE_KEYS)
        assert set(pieces["moves"]["items"]["properties"]) == set(schema_module._MOVE_KEYS)

    def test_every_board_and_promotion_key_can_be_proposed(self) -> None:
        """Schema 14 added `board.lattice` and `promotion.squares`, which neither test above
        reaches."""
        properties = rule_spec_schema()["properties"]
        assert set(properties["board"]["properties"]) == set(schema_module._BOARD_KEYS)
        promotion = properties["rules"]["properties"]["promotion"]["anyOf"][0]
        assert set(promotion["properties"]) == set(schema_module._PROMOTION_KEYS)

    def test_every_wrap_key_can_be_proposed(self) -> None:
        """Schema 16 added a hexagon's three pairs of sides under `rules.wrap`."""
        wrap = rule_spec_schema()["properties"]["rules"]["properties"]["wrap"]
        assert set(wrap["properties"]) == set(schema_module._WRAP_KEYS)

    def test_only_appearance_and_provenance_are_withheld(self) -> None:
        """The two blocks outside the mechanics hash, withheld for stated reasons.

        Written as an equality rather than a subset so that a third top-level block
        forces the question of whether a model should be writing it.
        """
        top = set(rule_spec_schema()["properties"])
        assert set(schema_module._SPEC_KEYS) - top == {"appearance", "provenance"}
        assert not top - set(schema_module._SPEC_KEYS)

    def test_a_longer_turn_survives_the_round_trip(self) -> None:
        """The schema describes it, the validator accepts it, and the cap is shared."""
        spec = copy.deepcopy(MINIMAL)
        spec["schema_version"] = SCHEMA_VERSION
        spec["rules"]["turn"] = {"moves": 2, "ramp": [1]}
        validate_or_raise(spec)

        turn = rule_spec_schema()["properties"]["rules"]["properties"]["turn"]
        assert turn["properties"]["moves"]["maximum"] == MAX_MOVES_PER_TURN
        assert turn["properties"]["ramp"]["items"]["maximum"] == MAX_MOVES_PER_TURN


# --- The schema's bounds, pinned (Q736; the survivors of `spec/jsonschema.py`) -------------
#
# `rule_spec_schema` is generated from the validator's constants, and the key-set tests above
# never read a bound. Every number below is written out here, so a bound that drifts by one
# fails against this table rather than against itself.

_BOUND_KEYS = ("minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength")


def _bounds_in(node: object, path: str = "") -> dict[str, dict[str, int]]:
    """Every node of the schema that carries a bound, by its path, with the bounds."""
    found: dict[str, dict[str, int]] = {}
    if isinstance(node, dict):
        carried = {key: node[key] for key in _BOUND_KEYS if key in node}
        if carried:
            found[path] = carried
        for key, value in node.items():
            if key != "description":
                found.update(_bounds_in(value, f"{path}/{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.update(_bounds_in(value, f"{path}[{index}]"))
    return found


_P = "/properties"
_PIECE = f"{_P}/pieces/items{_P}"
_MOVE = f"{_PIECE}/moves/items{_P}"
_RULES = f"{_P}/rules{_P}"
_PROMO = f"{_RULES}/promotion/anyOf[0]{_P}"
_PROMO_LIST = f"{_RULES}/promotion/anyOf[1]/items{_P}"
_LANDING_STRIKE = f"{_RULES}/strike{_P}/landing{_P}/vectors"
_LANDING_CONTACT = f"{_RULES}/contacts{_P}/landing{_P}/vectors"


def _vector_bounds(at: str, many: dict[str, int]) -> dict[str, dict[str, int]]:
    return {
        at: many,
        f"{at}/items": {"minItems": 2, "maxItems": 2},
        f"{at}/items/items": {"minimum": -8, "maximum": 8},
    }


_AT_LEAST_ONE = {"minItems": 1}

#: The whole table of bounds, as the validator states them. Kept literal on purpose.
FROZEN_BOUNDS: dict[str, dict[str, int]] = {
    f"{_P}/name": {"minLength": 1},
    f"{_P}/board{_P}/width": {"minimum": 2, "maximum": 16},
    f"{_P}/board{_P}/height": {"minimum": 2, "maximum": 16},
    f"{_P}/pieces": {"minItems": 1, "maxItems": 24},
    f"{_PIECE}/id": {"minLength": 1, "maxLength": 2},
    f"{_PIECE}/value": {"minimum": 0, "maximum": 100},
    # Schema 28 (Q943a): a speed is a rank of at least 1; 0 is the default left out.
    f"{_PIECE}/speed": {"minimum": 1},
    f"{_PIECE}/rest": {"minimum": 1, "maximum": 4},
    # Q925: no minItems (a drop-stock piece or an unmoved portal type has no moves) and no
    # cap on vectors (the validator has none; `archive/hex-glinski-hexagon.json` lists 12).
    f"{_PIECE}/moves": {"maxItems": 12},
    **_vector_bounds(f"{_MOVE}/vectors", _AT_LEAST_ONE),
    f"{_MOVE}/range": {"minimum": 0, "maximum": 16},
    f"{_MOVE}/min_range": {"minimum": 1, "maximum": 16},
    f"{_MOVE}/screen_range": {"minimum": 0, "maximum": 16},
    f"{_MOVE}/condition{_P}/from_rank": {"minimum": 0},
    f"{_MOVE}/condition{_P}/region": _AT_LEAST_ONE,
    f"{_PIECE}/region": _AT_LEAST_ONE,
    f"{_P}/setup{_P}/placements": _AT_LEAST_ONE,
    f"{_P}/setup{_P}/placements/items{_P}/squares": _AT_LEAST_ONE,
    f"{_PROMO}/ranks": _AT_LEAST_ONE,
    f"{_PROMO}/ranks/items": {"minimum": 0},
    f"{_PROMO}/squares": _AT_LEAST_ONE,
    f"{_PROMO}/to": _AT_LEAST_ONE,
    f"{_PROMO}/to_by_player{_P}/0": _AT_LEAST_ONE,
    f"{_PROMO}/to_by_player{_P}/1": _AT_LEAST_ONE,
    f"{_RULES}/promotion/anyOf[1]": _AT_LEAST_ONE,
    f"{_PROMO_LIST}/ranks": _AT_LEAST_ONE,
    f"{_PROMO_LIST}/ranks/items": {"minimum": 0},
    f"{_PROMO_LIST}/squares": _AT_LEAST_ONE,
    f"{_PROMO_LIST}/to": _AT_LEAST_ONE,
    f"{_PROMO_LIST}/to_by_player{_P}/0": _AT_LEAST_ONE,
    f"{_PROMO_LIST}/to_by_player{_P}/1": _AT_LEAST_ONE,
    f"{_RULES}/castling{_P}/king_steps": {"minimum": 1, "maximum": 4},
    f"{_RULES}/check_limit": {"minimum": 1, "maximum": 20},
    f"{_RULES}/territory{_P}/win_at": {"minimum": 0},
    f"{_RULES}/turn{_P}/moves": {"minimum": 1, "maximum": 4},
    f"{_RULES}/turn{_P}/ramp": {"minItems": 1, "maxItems": 8},
    f"{_RULES}/turn{_P}/ramp/items": {"minimum": 1, "maximum": 4},
    f"{_RULES}/obstacles{_P}/squares": _AT_LEAST_ONE,
    f"{_RULES}/strike{_P}/chain": _AT_LEAST_ONE,
    f"{_RULES}/strike{_P}/switches_sides": _AT_LEAST_ONE,
    **_vector_bounds(_LANDING_STRIKE, _AT_LEAST_ONE),
    f"{_RULES}/contacts{_P}/links": _AT_LEAST_ONE,
    **_vector_bounds(_LANDING_CONTACT, _AT_LEAST_ONE),
    f"{_RULES}/npcs": _AT_LEAST_ONE,
    f"{_RULES}/npcs/items{_P}/route": _AT_LEAST_ONE,
    f"{_RULES}/npcs/items{_P}/also_at": _AT_LEAST_ONE,
    f"{_RULES}/npcs/items{_P}/bounty": {"minimum": 0},
    f"{_RULES}/portals": {"minItems": 1, "maxItems": 4},
    f"{_RULES}/portals/items{_P}/ends": {"minItems": 3, "maxItems": 8},
    f"{_RULES}/portals/items{_P}/ends/items{_P}/to": {"minimum": 0, "maximum": 7},
    f"{_RULES}/scoring{_P}/region": _AT_LEAST_ONE,
    f"{_RULES}/reach{_P}/squares": _AT_LEAST_ONE,
    f"{_RULES}/extinction{_P}/pieces/anyOf[1]": _AT_LEAST_ONE,
    f"{_RULES}/extinction{_P}/min_count": {"minimum": 1, "maximum": 64},
    f"{_RULES}/drops{_P}/region/additionalProperties": _AT_LEAST_ONE,
    f"{_RULES}/drops{_P}/stock/additionalProperties/anyOf[0]": {"minimum": 1, "maximum": 32},
    f"{_RULES}/explode_on_capture{_P}/radius": {"minimum": 0, "maximum": 3},
    f"{_RULES}/max_plies": {"minimum": 2, "maximum": 10000},
    f"{_RULES}/repetition_draw/anyOf[0]": {"minimum": 2, "maximum": 10},
    f"{_RULES}/no_progress_draw": {"minimum": 2, "maximum": 1000},
}


def _schema_errors(instance: object, schema: dict, path: str = "$") -> list[str]:
    """Where `instance` breaks `schema`: the vocabulary the generated schema uses, bounds
    included (`tests/test_loop.py::_violations` stops at types, enums and keys)."""
    if "anyOf" in schema:
        if any(not _schema_errors(instance, arm, path) for arm in schema["anyOf"]):
            return []
        return [f"{path}: matches no arm"]
    if "const" in schema:
        return [] if instance == schema["const"] else [f"{path}: not {schema['const']!r}"]
    kinds = schema.get("type")
    if isinstance(kinds, list):
        return _schema_errors(instance, {**schema, "anyOf": [{"type": k} for k in kinds]}, path)
    bad: list[str] = []
    if kinds == "object":
        if not isinstance(instance, dict):
            return [f"{path}: expected an object"]
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties")
        bad += [f"{path}.{k}: required" for k in schema.get("required", []) if k not in instance]
        for key, value in instance.items():
            if key in properties:
                bad += _schema_errors(value, properties[key], f"{path}.{key}")
            elif isinstance(extra, dict):
                bad += _schema_errors(value, extra, f"{path}.{key}")
            elif extra is False:
                bad.append(f"{path}.{key}: no such property")
        return bad
    if kinds == "array":
        if not isinstance(instance, list):
            return [f"{path}: expected an array"]
        if len(instance) < schema.get("minItems", 0):
            bad.append(f"{path}: fewer than {schema['minItems']} items")
        if len(instance) > schema.get("maxItems", len(instance)):
            bad.append(f"{path}: more than {schema['maxItems']} items")
        for index, item in enumerate(instance):
            bad += _schema_errors(item, schema.get("items", {}), f"{path}[{index}]")
        return bad
    if "enum" in schema and instance not in schema["enum"]:
        bad.append(f"{path}: {instance!r} not in the enum")
    numeric = isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if kinds == "integer" and not (isinstance(instance, int) and not isinstance(instance, bool)):
        bad.append(f"{path}: expected an integer")
    elif kinds == "number" and not numeric:
        bad.append(f"{path}: expected a number")
    elif kinds == "string" and not isinstance(instance, str):
        bad.append(f"{path}: expected a string")
    elif kinds == "boolean" and not isinstance(instance, bool):
        bad.append(f"{path}: expected a boolean")
    if numeric:
        if "minimum" in schema and instance < schema["minimum"]:
            bad.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            bad.append(f"{path}: above {schema['maximum']}")
    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            bad.append(f"{path}: shorter than {schema['minLength']}")
        if len(instance) > schema.get("maxLength", len(instance)):
            bad.append(f"{path}: longer than {schema['maxLength']}")
    return bad


def _as_proposed(spec: dict) -> dict:
    """A spec as the model writes it: no `appearance` or `provenance`, the current version."""
    proposed = copy.deepcopy(spec)
    for withheld in ("appearance", "provenance"):
        proposed.pop(withheld, None)
    proposed["schema_version"] = SCHEMA_VERSION
    return proposed


def _all_nodes(node: dict):
    """Every schema node under `node`, following `properties`, `items`, a map's value schema
    and `anyOf` (a property *named* description is a name, not a node's description)."""
    yield node
    children = list(node.get("properties", {}).values())
    children += [node["items"]] if isinstance(node.get("items"), dict) else []
    children += (
        [node["additionalProperties"]] if isinstance(node.get("additionalProperties"), dict) else []
    )
    children += node.get("anyOf", [])
    for child in children:
        yield from _all_nodes(child)


class TestTheSchemaPinsEveryBound:
    def test_the_bounds_are_exactly_the_frozen_table(self) -> None:
        """Every min/max in the generated schema, by path. A drift of one in any of them,
        or a bound that appears or goes, shows here."""
        assert _bounds_in(rule_spec_schema()) == FROZEN_BOUNDS

    def test_the_clock_bounds_are_the_validators(self) -> None:
        """The three clocks the schema states are the three the validator holds."""
        bounds = schema_module.CLOCK_BOUNDS
        assert {
            k: tuple(bounds[k]) for k in ("max_plies", "repetition_draw", "no_progress_draw")
        } == {
            "max_plies": (2, 10000),
            "repetition_draw": (2, 10),
            "no_progress_draw": (2, 1000),
        }
        rules = rule_spec_schema()["properties"]["rules"]["properties"]
        assert (rules["max_plies"]["minimum"], rules["max_plies"]["maximum"]) == (2, 10000)
        arm = rules["repetition_draw"]["anyOf"][0]
        assert (arm["minimum"], arm["maximum"]) == (2, 10)
        assert (rules["no_progress_draw"]["minimum"], rules["no_progress_draw"]["maximum"]) == (
            2,
            1000,
        )

    def test_chess_satisfies_the_schema(self, chess_spec: dict) -> None:
        """The checker is not simply refusing everything."""
        assert _schema_errors(_as_proposed(chess_spec), rule_spec_schema()) == []

    @pytest.mark.parametrize(
        ("label", "edit", "valid"),
        [
            ("a board of 2 wide", lambda s: s["board"].update(width=2, height=2), True),
            ("a board 1 wide", lambda s: s["board"].update(width=1), False),
            ("an empty name", lambda s: s.update(name=""), False),
            ("a one-letter name", lambda s: s.update(name="x"), True),
            ("no pieces", lambda s: s.update(pieces=[]), False),
            ("a three-letter piece id", lambda s: s["pieces"][1].update(id="Kni"), False),
            ("an empty piece id", lambda s: s["pieces"][1].update(id=""), False),
            ("a two-letter piece id", lambda s: s["pieces"][1].update(id="Kn"), True),
            ("a negative value", lambda s: s["pieces"][1].update(value=-1), False),
            ("a value of 101", lambda s: s["pieces"][1].update(value=101), False),
            ("a value of 100", lambda s: s["pieces"][1].update(value=100), True),
            (
                "a vector of one number",
                lambda s: s["pieces"][1]["moves"][0].update(vectors=[[1]]),
                False,
            ),
            (
                "a vector of three numbers",
                lambda s: s["pieces"][1]["moves"][0].update(vectors=[[1, 0, 0]]),
                False,
            ),
            (
                "a vector of -8",
                lambda s: s["pieces"][1]["moves"][0].update(vectors=[[-8, 1]]),
                True,
            ),
            ("a vector of 9", lambda s: s["pieces"][1]["moves"][0].update(vectors=[[9, 1]]), False),
            (
                "a move with no vectors",
                lambda s: s["pieces"][1]["moves"][0].update(vectors=[]),
                False,
            ),
            (
                "a move of eight vectors",
                lambda s: s["pieces"][1]["moves"][0].update(vectors=[[1, i] for i in range(8)]),
                True,
            ),
            (
                "a move of nine vectors",
                lambda s: s["pieces"][1]["moves"][0].update(vectors=[[1, i] for i in range(9)]),
                True,
            ),
            ("an unlimited range", lambda s: s["pieces"][1]["moves"][0].update(range=0), True),
            ("a negative range", lambda s: s["pieces"][1]["moves"][0].update(range=-1), False),
            ("a min_range of 1", lambda s: s["pieces"][1]["moves"][0].update(min_range=1), True),
            ("a min_range of 0", lambda s: s["pieces"][1]["moves"][0].update(min_range=0), False),
            (
                "a cannon's screen_range",
                lambda s: s["pieces"][1]["moves"][0].update(screen_range=0),
                True,
            ),
            (
                "a negative screen_range",
                lambda s: s["pieces"][1]["moves"][0].update(screen_range=-1),
                False,
            ),
            (
                "a from_rank of 0",
                lambda s: s["pieces"][1]["moves"][0].update(condition={"from_rank": 0}),
                True,
            ),
            (
                "a negative from_rank",
                lambda s: s["pieces"][1]["moves"][0].update(condition={"from_rank": -1}),
                False,
            ),
            (
                "a piece with one move",
                lambda s: s["pieces"][1].update(moves=s["pieces"][1]["moves"][:1]),
                True,
            ),
            # Q925: the validator admits `moves: []` for a drop-stock piece or an unmoved
            # portal type, a cross-reference no bound can state, so the schema admits it.
            ("a piece with no moves", lambda s: s["pieces"][1].update(moves=[]), True),
            ("an empty placement", lambda s: s["setup"]["placements"][0].update(squares=[]), False),
            ("no placements", lambda s: s["setup"].update(placements=[]), False),
            ("one castling step", lambda s: s["rules"]["castling"].update(king_steps=1), True),
            ("four castling steps", lambda s: s["rules"]["castling"].update(king_steps=4), True),
            ("no castling steps", lambda s: s["rules"]["castling"].update(king_steps=0), False),
            ("five castling steps", lambda s: s["rules"]["castling"].update(king_steps=5), False),
            (
                "a territory win_at of 0",
                lambda s: s["rules"].update(territory={"mode": "repaint", "win_at": 0}),
                True,
            ),
            (
                "a negative win_at",
                lambda s: s["rules"].update(territory={"mode": "repaint", "win_at": -1}),
                False,
            ),
            ("a turn of one move", lambda s: s["rules"].update(turn={"moves": 1}), True),
            ("a turn of no moves", lambda s: s["rules"].update(turn={"moves": 0}), False),
            ("a ramp of one", lambda s: s["rules"].update(turn={"moves": 2, "ramp": [1]}), True),
            (
                "a ramp with a zero",
                lambda s: s["rules"].update(turn={"moves": 2, "ramp": [0]}),
                False,
            ),
            ("an empty ramp", lambda s: s["rules"].update(turn={"moves": 2, "ramp": []}), False),
            (
                "a ramp of eight",
                lambda s: s["rules"].update(turn={"moves": 2, "ramp": [1] * 8}),
                True,
            ),
            (
                "a ramp of nine",
                lambda s: s["rules"].update(turn={"moves": 2, "ramp": [1] * 9}),
                False,
            ),
            ("an empty promotion list", lambda s: s["rules"].update(promotion=[]), False),
            (
                "a promotion rank of 0",
                lambda s: s["rules"].update(promotion={"piece": "P", "ranks": [0], "to": ["Q"]}),
                True,
            ),
            (
                "a negative promotion rank",
                lambda s: s["rules"].update(promotion={"piece": "P", "ranks": [-1], "to": ["Q"]}),
                False,
            ),
            (
                "no promotion ranks",
                lambda s: s["rules"].update(promotion={"piece": "P", "ranks": [], "to": ["Q"]}),
                False,
            ),
            (
                "no promotion targets",
                lambda s: s["rules"].update(promotion={"piece": "P", "ranks": [7], "to": []}),
                False,
            ),
            (
                "one promotion target",
                lambda s: s["rules"].update(promotion={"piece": "P", "ranks": [7], "to": ["Q"]}),
                True,
            ),
            (
                "a list of one promotion",
                lambda s: s["rules"].update(promotion=[{"piece": "P", "ranks": [7], "to": ["Q"]}]),
                True,
            ),
            (
                "targets by player, none for the second",
                lambda s: s["rules"].update(
                    promotion={"piece": "P", "ranks": [7], "to_by_player": {"0": ["Q"], "1": []}}
                ),
                False,
            ),
            (
                "targets by player, none for the first",
                lambda s: s["rules"].update(
                    promotion={"piece": "P", "ranks": [7], "to_by_player": {"0": [], "1": ["Q"]}}
                ),
                False,
            ),
            (
                "targets by player, one each",
                lambda s: s["rules"].update(
                    promotion={"piece": "P", "ranks": [7], "to_by_player": {"0": ["Q"], "1": ["Q"]}}
                ),
                True,
            ),
            (
                "an extinction of one type",
                lambda s: s["rules"].update(extinction={"pieces": ["P"]}),
                True,
            ),
            (
                "an extinction of none",
                lambda s: s["rules"].update(extinction={"pieces": []}),
                False,
            ),
            (
                "an extinction min_count of 64",
                lambda s: s["rules"].update(extinction={"pieces": ["P"], "min_count": 64}),
                True,
            ),
            (
                "an extinction min_count of 65",
                lambda s: s["rules"].update(extinction={"pieces": ["P"], "min_count": 65}),
                False,
            ),
            (
                "an extinction min_count of 0",
                lambda s: s["rules"].update(extinction={"pieces": ["P"], "min_count": 0}),
                False,
            ),
            ("a stock of one", lambda s: s["rules"].update(drops={"stock": {"P": 1}}), True),
            ("a stock of none", lambda s: s["rules"].update(drops={"stock": {"P": 0}}), False),
            (
                "a blast of radius 3",
                lambda s: s["rules"].update(explode_on_capture={"radius": 3}),
                True,
            ),
            (
                "a blast of radius 4",
                lambda s: s["rules"].update(explode_on_capture={"radius": 4}),
                False,
            ),
            (
                "a negative blast",
                lambda s: s["rules"].update(explode_on_capture={"radius": -1}),
                False,
            ),
            (
                "a blast of radius 0",
                lambda s: s["rules"].update(explode_on_capture={"radius": 0}),
                True,
            ),
            (
                "an empty obstacle list",
                lambda s: s["rules"].update(obstacles={"squares": []}),
                False,
            ),
            ("one obstacle", lambda s: s["rules"].update(obstacles={"squares": ["e4"]}), True),
            (
                "an npc with a negative bounty",
                lambda s: s["rules"].update(npcs=[{"piece": "P", "route": ["e4"], "bounty": -1}]),
                False,
            ),
            (
                "an npc with no bounty to pay",
                lambda s: s["rules"].update(npcs=[{"piece": "P", "route": ["e4"], "bounty": 0}]),
                True,
            ),
            ("no npcs", lambda s: s["rules"].update(npcs=[]), False),
            (
                "a strike of one link",
                lambda s: s["rules"].update(
                    strike={"effect": "demote", "chain": [{"from": "P", "to": "X"}]}
                ),
                True,
            ),
            (
                "a strike of no links",
                lambda s: s["rules"].update(strike={"effect": "demote", "chain": []}),
                False,
            ),
            (
                "contacts of one link",
                lambda s: s["rules"].update(contacts={"links": [{"mover": "P", "onto": "P"}]}),
                True,
            ),
            ("contacts of no links", lambda s: s["rules"].update(contacts={"links": []}), False),
            (
                "a landing of eight",
                lambda s: s["rules"].update(
                    strike={
                        "effect": "capture",
                        "position": "choose",
                        "landing": {"vectors": [[1, i] for i in range(8)]},
                    }
                ),
                True,
            ),
            (
                "a landing of nine",
                lambda s: s["rules"].update(
                    strike={
                        "effect": "capture",
                        "position": "choose",
                        "landing": {"vectors": [[1, i] for i in range(9)]},
                    }
                ),
                True,
            ),
            (
                "a landing of none",
                lambda s: s["rules"].update(
                    strike={"effect": "capture", "position": "choose", "landing": {"vectors": []}}
                ),
                False,
            ),
            (
                "a contact landing of eight",
                lambda s: s["rules"].update(
                    contacts={
                        "links": [{"mover": "P", "onto": "P"}],
                        "position": "choose",
                        "landing": {"vectors": [[1, i] for i in range(8)]},
                    }
                ),
                True,
            ),
            (
                "a contact landing of none",
                lambda s: s["rules"].update(
                    contacts={
                        "links": [{"mover": "P", "onto": "P"}],
                        "position": "choose",
                        "landing": {"vectors": []},
                    }
                ),
                False,
            ),
        ],
    )
    def test_each_edge_of_a_bound_is_accepted_or_refused(
        self, chess_spec: dict, label: str, edit, valid: bool
    ) -> None:
        """The schema and the bound it states agree on both sides of each edge."""
        spec = _as_proposed(chess_spec)
        spec["pieces"].append(
            {"id": "X", "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]}
        )
        # Index 1 is the piece the edits aim at; make it the extra one.
        spec["pieces"][1], spec["pieces"][-1] = spec["pieces"][-1], spec["pieces"][1]
        edit(spec)
        assert (_schema_errors(spec, rule_spec_schema()) == []) is valid, label


# --- What the validator admits, the model contract admits (Q925) -------------------------
#
# The schema may be looser than `validate_spec` (it cannot state a cross-reference), never
# stricter: a stricter contract hides a game the validator would play from every proposal,
# and nothing fails. Q734m found two such places, a cap of 8 vectors and `minItems: 1` on
# `moves`; every committed and archived spec is the cheap net for the next one.

_REPO = Path(__file__).resolve().parent.parent
_LIBRARY = sorted(
    p for d in ("rulesets", "archive") for p in (_REPO / d).glob("*.json")
)


class TestTheContractAdmitsWhatTheValidatorAdmits:
    @pytest.mark.parametrize("path", _LIBRARY, ids=lambda p: f"{p.parent.name}/{p.stem}")
    def test_every_library_spec_satisfies_the_schema(self, path: Path) -> None:
        spec = json.loads(path.read_text(encoding="utf-8"))
        assert validate_spec(spec) == [], path.name
        assert _schema_errors(_as_proposed(spec), rule_spec_schema()) == [], path.name

    def test_the_library_still_holds_both_of_q925s_cases(self) -> None:
        """The net above only catches what the library uses: keep it using both points."""
        names = {p.name for p in _LIBRARY}
        assert {"hex-glinski-hexagon.json", "stones.json"} <= names
        glinski = json.loads((_REPO / "archive/hex-glinski-hexagon.json").read_text("utf-8"))
        assert max(len(m["vectors"]) for p in glinski["pieces"] for m in p["moves"]) > 8
        stones = json.loads((_REPO / "archive/stones.json").read_text("utf-8"))
        assert any(p["moves"] == [] for p in stones["pieces"])

    def test_a_piece_with_no_moves_is_the_validators_call(self, chess_spec: dict) -> None:
        """Without a stock the validator refuses `moves: []`; the schema leaves that to it."""
        spec = _as_proposed(chess_spec)
        spec["pieces"].append({"id": "S", "name": "stone", "value": 1, "moves": []})
        assert _schema_errors(spec, rule_spec_schema()) == []
        assert any("moves must be a non-empty array" in e for e in validate_spec(spec))


class TestTheSchemaClosesItsObjects:
    def test_the_default_schema_is_strict(self) -> None:
        """`rule_spec_schema()` closes every object it writes, so an invented key is refused."""
        schema = rule_spec_schema()
        objects = [n for n in _all_nodes(schema) if n.get("type") == "object"]
        assert objects
        closed = [n for n in objects if n.get("additionalProperties") is False]
        # Only the maps (drops' region and stock) carry a value schema instead of False.
        assert len(objects) - len(closed) == 2
        assert all(isinstance(n["additionalProperties"], dict) for n in objects if n not in closed)

    def test_a_loose_schema_closes_nothing(self) -> None:
        loose = rule_spec_schema(strict=False)
        objects = [n for n in _all_nodes(loose) if n.get("type") == "object"]
        assert objects
        assert all(n.get("additionalProperties") is not False for n in objects)

    def test_an_invented_key_is_refused_and_a_known_one_is_not(self, chess_spec: dict) -> None:
        spec = _as_proposed(chess_spec)
        assert _schema_errors(spec, rule_spec_schema()) == []
        spec["pieces"][0]["colour"] = "red"
        assert _schema_errors(spec, rule_spec_schema()) == ["$.pieces[0].colour: no such property"]
        assert _schema_errors(spec, rule_spec_schema(strict=False)) == []

    def test_a_description_is_text_where_it_is_given_and_absent_elsewhere(self) -> None:
        """`_object` writes a description only when it is given one, never a null."""
        schema = rule_spec_schema()
        for node in _all_nodes(schema):
            if "description" in node:
                assert isinstance(node["description"], str) and node["description"]
        facing = schema["properties"]["pieces"]["items"]["properties"]["facing"]
        assert "orientation" in facing["description"]
        # An object given no description has none, rather than a null one.
        assert "description" not in schema["properties"]["board"]


# --- Q736: the survivors of `spec/canonical.py`, `schema.py` and `reachability.py` ---------


def _with(base: dict, **changes: object) -> dict:
    spec = copy.deepcopy(base)
    spec.update(changes)
    return spec


def _leaper(piece_id: str, **extra: object) -> dict:
    return {"id": piece_id, "moves": [{"kind": "leap", "vectors": [[1, 0]]}], **extra}


class TestNonAsciiMechanicsAreHashedAsUtf8:
    def test_a_non_ascii_name_is_written_raw_not_escaped(self) -> None:
        from chess_variant_sandbox.spec.canonical import canonical_json

        spec = {"pieces": [{"id": "P", "name": "Ölfant"}]}
        assert canonical_json(spec) == '{"pieces":[{"id":"P","name":"Ölfant"}]}'
        assert "\\u00d6" not in canonical_json(spec)

    def test_the_hash_is_that_of_the_utf8_blob(self) -> None:
        import hashlib

        spec = {"board": {"width": 4, "height": 4}, "pieces": [{"id": "P", "name": "Ölfant"}]}
        blob = '{"board":{"height":4,"width":4},"pieces":[{"id":"P","name":"Ölfant"}]}'
        assert spec_hash(spec) == hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class TestTheLimitsAreEnforcedAtTheirEdge:
    """The numeric bounds of `spec/schema.py`, each from both sides."""

    def test_a_board_of_sixteen_is_allowed_and_seventeen_is_not(self) -> None:
        assert validate_spec(_with(MINIMAL, board={"width": 16, "height": 16})) == []
        errors = validate_spec(_with(MINIMAL, board={"width": 17, "height": 17}))
        assert "board.width must be between 2 and 16, got 17" in errors
        assert "board.height must be between 2 and 16, got 17" in errors

    @pytest.mark.parametrize(("count", "fine"), [(24, True), (25, False)])
    def test_piece_types_stop_at_twenty_four(self, count: int, fine: bool) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["pieces"] += [_leaper(f"A{chr(65 + i)}") for i in range(count - 1)]
        errors = validate_spec(spec)
        assert ("at most 24 piece types" in " ".join(errors)) is not fine
        if fine:
            assert errors == []

    @pytest.mark.parametrize(("count", "fine"), [(12, True), (13, False)])
    def test_a_piece_holds_at_most_twelve_move_descriptors(self, count: int, fine: bool) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["pieces"][0]["moves"] = [{"kind": "leap", "vectors": [[1, 0]]}] * count
        errors = validate_spec(spec)
        assert ("pieces[0].moves may hold at most 12 descriptors" in errors) is not fine
        if fine:
            assert errors == []

    @pytest.mark.parametrize(("reach", "fine"), [(8, True), (9, False), (-8, True), (-9, False)])
    def test_a_vector_component_stops_at_eight_either_way(self, reach: int, fine: bool) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["board"] = {"width": 16, "height": 16}
        spec["pieces"][0]["moves"] = [{"kind": "leap", "vectors": [[reach, 0]]}]
        errors = validate_spec(spec)
        assert any("components must be within +/-8" in e for e in errors) is not fine
        if fine:
            assert errors == []


class TestParseSquareStaysOnTheBoard:
    @pytest.mark.parametrize(
        ("square", "expected"),
        [
            ("a1", (0, 0)),
            ("h8", (7, 7)),
            ("a0", None),
            ("a9", None),
            ("i1", None),
            ("A1", (0, 0)),
            ("", None),
            ("a", None),
            ("1a", None),
            ("ab", None),
        ],
    )
    def test_the_edges_of_an_eight_by_eight_board(self, square: str, expected) -> None:
        from chess_variant_sandbox.spec.schema import parse_square

        assert parse_square(square, 8, 8) == expected

    def test_a_route_with_no_readable_square_does_not_end_the_list(self) -> None:
        from chess_variant_sandbox.spec.schema import npc_routes

        rules = {"npcs": [{"piece": "R", "route": ["z9"]}, {"piece": "R", "route": ["b2", "c2"]}]}
        assert npc_routes(rules, 5, 5) == [("rules.npcs[1]", (6, 7))]

    def test_a_placement_on_rank_zero_or_past_the_top_is_refused(self) -> None:
        for square in ("a0", "a5"):
            spec = copy.deepcopy(MINIMAL)
            spec["setup"]["placements"][0]["squares"] = [square]
            assert any("is not a square on this board" in e for e in validate_spec(spec)), square
        spec = copy.deepcopy(MINIMAL)
        spec["setup"]["placements"][0]["squares"] = ["a4"]
        assert validate_spec(spec) == []


class TestTheReachMirrorDefault:
    def test_a_reach_with_no_mirror_reflects_by_rank(self) -> None:
        from chess_variant_sandbox.spec.schema import reach_reflection

        assert reach_reflection({"squares": ["a8"]}) == "mirror"
        assert reach_reflection({"squares": ["a8"], "mirror": True}) == "mirror"
        assert reach_reflection({"squares": ["a8"], "mirror": "rotate"}) == "rotate"
        assert reach_reflection({"squares": ["a8"], "mirror": False}) is None
        assert reach_reflection("not a reach") is None

    def test_the_rotate_spelling_was_added_in_schema_seven(self) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["rules"] = {
            "win": "reach",
            "reach": {"piece": "K", "squares": ["d4"], "mirror": "rotate"},
        }
        spec["schema_version"] = 6
        assert any("was added in schema 7" in e for e in validate_spec(spec))
        spec["schema_version"] = 7
        assert not any("was added in schema 7" in e for e in validate_spec(spec))


class TestAppearanceAndProvenanceAreChecked:
    @pytest.mark.parametrize(
        ("appearance", "fragment"),
        [
            ({"icons": {"Z": "x"}}, "appearance.icons names unknown piece 'Z'"),
            ({"icons": {"K": ""}}, "appearance.icons['K'] must be a non-empty string"),
            ({"icons": {"K": "  "}}, "appearance.icons['K'] must be a non-empty string"),
            ({"icons": {"K": ["a", ""]}}, "appearance.icons['K'] must be a non-empty string"),
            ({"icons": {"K": ["a"]}}, "as a list must hold exactly two entries"),
            ({"icons": ["K"]}, "appearance.icons must be an object"),
            ("icons", "appearance must be an object"),
        ],
    )
    def test_a_bad_icon_is_refused(self, appearance: object, fragment: str) -> None:
        errors = validate_spec(_with(MINIMAL, appearance=appearance))
        assert any(fragment in e for e in errors), errors

    @pytest.mark.parametrize(
        "appearance",
        [{}, {"icons": {}}, {"icons": {"K": "crown"}}, {"icons": {"K": ["white", "black"]}}],
    )
    def test_a_good_appearance_is_not(self, appearance: dict) -> None:
        assert validate_spec(_with(MINIMAL, appearance=appearance)) == []

    @pytest.mark.parametrize("parents", ["abc", [1], [None], {"a": 1}])
    def test_parents_must_be_a_list_of_hashes(self, parents: object) -> None:
        errors = validate_spec(_with(MINIMAL, provenance={"parents": parents}))
        assert "provenance.parents must be a list of ruleset hashes" in errors

    @pytest.mark.parametrize("parents", [[], ["abc123def456"]])
    def test_a_list_of_hashes_is_fine(self, parents: list) -> None:
        assert validate_spec(_with(MINIMAL, provenance={"parents": parents})) == []


class TestPieceCoherenceReportsRatherThanCrashing:
    def test_a_royal_missing_from_an_otherwise_placed_setup_is_named(self) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["pieces"].append(_leaper("P"))
        spec["setup"]["placements"] = [{"piece": "P", "squares": ["a2"]}]
        errors = validate_spec(spec)
        assert any("no royal piece appears in setup.placements" in e for e in errors), errors

    def test_a_piece_without_an_id_is_reported_not_a_keyerror(self) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["pieces"].append({"value": 1, "moves": [{"kind": "leap", "vectors": [[1, 0]]}]})
        errors = validate_spec(spec)
        assert any("id" in e for e in errors), errors

    def test_a_promoted_from_must_name_a_declared_piece(self) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["pieces"].append(_leaper("N", promoted_from="Z"))
        errors = validate_spec(spec)
        assert any("promoted_from 'Z', which is not a declared piece" in e for e in errors)
        spec["pieces"][1]["promoted_from"] = "K"
        assert not any("promoted_from" in e for e in validate_spec(spec))

    def test_a_non_object_board_leaves_no_width_for_a_file_to_fit(self) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["board"] = "x"
        spec["rules"]["castling"] = {
            "king": "K",
            "rook": "K",
            "king_file": {"kingside": "a", "queenside": "a"},
            "rook_file": {"kingside": "a", "queenside": "a"},
        }
        errors = validate_spec(spec)
        assert "rules.castling.king_file.kingside 'a' is not a file on this board" in errors

    def test_a_refused_board_leaves_no_height_for_a_promotion_rank_to_miss(self) -> None:
        """On a board that is itself refused, the promotion rank is not also reported."""
        spec = copy.deepcopy(MINIMAL)
        spec["pieces"].append(_leaper("N"))
        spec["board"] = {"width": 99, "height": 4}
        spec["rules"]["promotion"] = {"piece": "K", "rank": 1, "to": ["N"]}
        spec["rules"]["castling"] = {
            "king": "K",
            "rook": "K",
            "king_file": {"kingside": "a", "queenside": "a"},
            "rook_file": {"kingside": "a", "queenside": "a"},
        }
        errors = validate_spec(spec)
        assert any("board.width must be between" in e for e in errors)
        assert not any("is not a rank on this board" in e for e in errors), errors
        # ... and no file fits it either, so the castling files are reported as off the board.
        assert "rules.castling.king_file.kingside 'a' is not a file on this board" in errors


class TestPieceIdsAreOneOrTwoCharacters:
    @pytest.mark.parametrize("piece_id", ["K", "NN", "n1"])
    def test_one_or_two_characters_are_fine(self, piece_id: str) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["pieces"][0]["id"] = piece_id
        spec["setup"]["placements"][0]["piece"] = piece_id
        assert validate_spec(spec) == []

    @pytest.mark.parametrize("piece_id", ["", "ABC", 5, None])
    def test_anything_else_is_refused_without_raising(self, piece_id: object) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["pieces"][0]["id"] = piece_id
        errors = validate_spec(spec)
        assert "pieces[0].id must be a string of 1-2 characters" in errors


class TestCanDoubleStep:
    @staticmethod
    def _ask(*moves: dict) -> bool:
        from chess_variant_sandbox.spec.schema import can_double_step

        return can_double_step({"moves": list(moves)})

    def test_a_capture_only_slide_before_the_double_step_does_not_hide_it(self) -> None:
        capture = {"kind": "slide", "vectors": [[1, 1]], "range": 1, "mode": "capture"}
        advance = {"kind": "slide", "vectors": [[0, 1]], "range": 2}
        assert self._ask(capture, advance) is True
        assert self._ask(capture) is False

    def test_a_malformed_distance_counts_as_qualifying(self) -> None:
        assert self._ask({"kind": "slide", "vectors": [[0, 1]], "range": True}) is True
        assert self._ask({"kind": "slide", "vectors": [[0, 1]], "min_range": "x"}) is True

    def test_a_single_step_never_qualifies(self) -> None:
        assert self._ask({"kind": "slide", "vectors": [[0, 1]], "range": 1}) is False
        assert self._ask({"kind": "leap", "vectors": [[0, 2]]}) is False


class TestACaseThirteenRefusalFollowsEveryRoute:
    """`rules.reach` about a piece that can never stand is refused (inert-rule case 13).

    The routes into play are placement, a promotion, a strike link, a contact, a hand and a
    wild piece; a piece stands if any of them reaches it from a piece that stands.
    """

    @staticmethod
    def _errors(
        *, strike=None, contacts=None, drops=None, npcs=None, promotion=None, placed=(), extra=()
    ) -> list[str]:
        spec = copy.deepcopy(MINIMAL)
        spec["schema_version"] = SCHEMA_VERSION
        spec["board"] = {"width": 8, "height": 8}
        spec["pieces"] += [_leaper("X"), _leaper("Y"), _leaper("P"), *extra]
        spec["setup"]["placements"] += [{"piece": p, "squares": [s]} for p, s in placed]
        spec["rules"] = {"win": "reach", "reach": {"piece": "Y", "squares": ["d4"]}}
        for key, value in (
            ("strike", strike),
            ("contacts", contacts),
            ("drops", drops),
            ("npcs", npcs),
            ("promotion", promotion),
        ):
            if value is not None:
                spec["rules"][key] = value
        return validate_spec(spec)

    @staticmethod
    def _refused(errors: list[str]) -> bool:
        return any("rules.reach is about 'Y', which can never stand" in e for e in errors)

    def test_with_no_route_it_is_refused(self) -> None:
        assert self._refused(self._errors())

    def test_a_placement_is_a_route(self) -> None:
        assert not self._refused(self._errors(placed=[("Y", "d1")]))

    def test_a_strike_link_from_a_piece_that_stands_is_a_route(self) -> None:
        strike = {"effect": "demote", "chain": [{"from": "P", "to": "Y"}]}
        assert not self._refused(self._errors(strike=strike, placed=[("P", "a2")]))

    def test_a_strike_link_from_a_piece_that_never_stands_is_not(self) -> None:
        strike = {"effect": "demote", "chain": [{"from": "X", "to": "Y"}]}
        assert self._refused(self._errors(strike=strike, placed=[("P", "a2")]))

    def test_a_contact_between_standing_pieces_is_a_route(self) -> None:
        contacts = {"links": [{"mover": "P", "onto": "K", "mover_to": "Y"}]}
        assert not self._refused(self._errors(contacts=contacts, placed=[("P", "a2")]))

    def test_a_contact_onto_a_piece_that_never_stands_is_not(self) -> None:
        contacts = {"links": [{"mover": "P", "onto": "X", "mover_to": "Y"}]}
        assert self._refused(self._errors(contacts=contacts, placed=[("P", "a2")]))

    def test_a_contact_from_a_piece_that_never_stands_is_not(self) -> None:
        contacts = {"links": [{"mover": "X", "onto": "P", "mover_to": "Y"}]}
        assert self._refused(self._errors(contacts=contacts, placed=[("P", "a2")]))

    def test_a_dead_link_does_not_hide_a_live_one_after_it(self) -> None:
        contacts = {
            "links": [
                {"mover": "X", "onto": "P", "mover_to": "X"},
                {"mover": "P", "onto": "K", "mover_to": "Y"},
            ]
        }
        assert not self._refused(self._errors(contacts=contacts, placed=[("P", "a2")]))

    def test_a_promotion_target_is_a_route_only_from_a_piece_that_stands(self) -> None:
        def promotion(piece: str) -> dict:
            return {"piece": piece, "rank": "last", "to": ["Y"]}

        assert not self._refused(self._errors(placed=[("P", "a2")], promotion=promotion("P")))
        assert self._refused(self._errors(placed=[("P", "a2")], promotion=promotion("X")))

    def test_a_stocked_piece_is_a_route(self) -> None:
        errors = self._errors(placed=[("P", "a2")], drops={"stock": {"Y": 1}})
        assert not self._refused(errors)

    def test_a_wild_piece_is_a_route(self) -> None:
        npcs = [{"piece": "Y", "route": ["e4", "e5"]}]
        spec_errors = self._errors(placed=[("P", "a2")], npcs=npcs)
        assert not self._refused(spec_errors)

    def test_a_captured_piece_returns_to_a_hand_as_its_base(self) -> None:
        promoted = _leaper("Z", promoted_from="Y")
        kwargs = {"placed": [("Z", "a2")], "extra": [promoted]}
        assert not self._refused(self._errors(drops={"from_captures": True}, **kwargs))
        assert self._refused(self._errors(drops={"from_captures": False}, **kwargs))
        assert self._refused(self._errors(**kwargs))


class TestRoutesAreReadFromMalformedEntries:
    """`validate_spec` collects errors rather than stopping, so the reachability walk is fed
    entries that are not objects and must not crash on them."""

    @pytest.mark.parametrize(
        "change",
        [
            {"setup": {"placements": ["x"]}},
            {"rules": {"win": "reach", "reach": {"piece": "K", "squares": ["d4"]}, "npcs": ["x"]}},
            {
                "rules": {
                    "win": "reach",
                    "reach": {"piece": "K", "squares": ["d4"]},
                    "strike": {"effect": "demote", "chain": ["x"]},
                }
            },
            {
                "rules": {
                    "win": "reach",
                    "reach": {"piece": "K", "squares": ["d4"]},
                    "contacts": {"links": ["x"]},
                }
            },
        ],
    )
    def test_a_non_object_entry_is_an_error_not_an_exception(self, change: dict) -> None:
        spec = copy.deepcopy(MINIMAL)
        spec["schema_version"] = SCHEMA_VERSION
        spec.update(copy.deepcopy(change))
        assert validate_spec(spec)
