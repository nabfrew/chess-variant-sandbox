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
