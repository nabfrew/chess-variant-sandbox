"""How a piece moves, said the way a player would say it (`spec/describe.py`, UX A7).

The page used to print the descriptor -- *may only capture along (+1,+1) mirrored left
and right, jumping directly, ignoring anything in between [2 directions]* -- which is a
true statement of the rule and tells a newcomer nothing. These tests hold the sentence
that replaced it.

Three things they are watching for, and all three are silent failures:

* **A wrong sentence is worse than a vector.** A descriptor no phrase fits has to fall
  back to the descriptor, not to the nearest phrase that almost fits, so
  ``test_a_descriptor_no_phrase_fits_falls_back`` names what falls back today.
* **A modifier that changes the game has to change the sentence.** Dropping
  ``min_range`` or ``screen`` from a descriptor is a different piece, and a sentence
  that reads the same for both is a page lying about the game
  (``TestEveryModifierIsSaid``).
* **Nothing here knows any game.** The sentences are built out of the spec's vectors and
  the schema's own keys, so no piece name and no game name appears in the module.

`tests/test_spec.py::TestDescription` holds the rest of the rendering -- the rules, the
setup, and the coverage checks that every rule reaches a reader.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine, compile_spec
from chess_variant_sandbox.spec.describe import (
    RULE_ALSO_CALLED,
    RULE_WORDS,
    PieceProse,
    _Aim,
    _board_words,
    _describe_promotion,
    _landing_phrase,
    _move_descriptor,
    _move_sentence,
    capturing_types,
    describe_changes,
    describe_pieces,
    describe_rules,
    describe_setup,
    move_sentence,
    rule_choices,
    rule_words,
)
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import promotion_squares
from chess_variant_sandbox.spec.transform import HEX_STEP_LANDING, KING_STEP_LANDING

ROOT = Path(__file__).resolve().parents[1]
RULESETS = sorted(
    list((ROOT / "rulesets").glob("*.json")) + list((ROOT / "archive").glob("*.json"))
)


def _spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _moves(path: Path) -> dict[str, tuple[str, ...]]:
    return {piece.id: piece.moves for piece in describe_pieces(_spec(path))}


@pytest.fixture(scope="module")
def chess_moves() -> dict[str, tuple[str, ...]]:
    return _moves(ROOT / "rulesets" / "chess.json")


@pytest.fixture(scope="module")
def hex_moves() -> dict[str, tuple[str, ...]]:
    return _moves(ROOT / "archive" / "hex-glinski-hexagon.json")


def _said(move: dict[str, Any], hexy: bool = False) -> str:
    """One descriptor's sentence, failing rather than falling back silently."""
    sentence = _move_sentence(move, hexy)
    assert sentence is not None, f"no phrase fits {move}"
    return sentence


# --------------------------------------------------------------- the calibration game


class TestTheBaselineReadsAsSentences:
    """The six pieces of `rulesets/chess.json`, in full.

    Written out rather than pattern-matched because the whole box is *does this read
    like something a player would say*, and only the sentence itself answers that.
    """

    def test_the_pawn(self, chess_moves: dict[str, tuple[str, ...]]) -> None:
        assert chess_moves["P"] == (
            "one square forward, not to take",
            "exactly two squares forward, stopped by anything in the way, not to take, "
            "and only from its own rank 2",
            "one square diagonally forward, either way, only to take",
        )

    def test_the_knight(self, chess_moves: dict[str, tuple[str, ...]]) -> None:
        assert chess_moves["N"] == (
            "leaps two and one, in all eight directions, jumping over whatever is in "
            "between",
        )

    def test_the_lines(self, chess_moves: dict[str, tuple[str, ...]]) -> None:
        """Three pieces, one grammar: a distance, a direction, and what stops it."""
        assert chess_moves["B"] == (
            "any distance along all four diagonals, stopped by anything in the way",
        )
        assert chess_moves["R"] == (
            "any distance forward, back, left and right, stopped by anything in the way",
        )
        # Two vectors under one symmetry, said as the eight directions they come to
        # rather than as two halves a reader has to add up.
        assert chess_moves["Q"] == (
            "any distance in all eight directions, stopped by anything in the way",
        )

    def test_the_king(self, chess_moves: dict[str, tuple[str, ...]]) -> None:
        """A one-square line has nothing in between, so it says nothing about blocking."""
        assert chess_moves["K"] == ("one square in all eight directions",)


# ------------------------------------------------------------------ kind by kind


class TestEveryMoveKind:
    """The five kinds of `schema.MOVE_KINDS`, each said as what it does."""

    def test_a_slide_carries_its_reach_and_what_stops_it(self) -> None:
        line = {"kind": "slide", "vectors": [[0, 1]]}
        assert _said(line) == "any distance forward, stopped by anything in the way"
        assert _said({**line, "range": 3}) == (
            "up to three squares forward, stopped by anything in the way"
        )
        assert _said({**line, "range": 3, "min_range": 3}) == (
            "exactly three squares forward, stopped by anything in the way"
        )
        assert _said({**line, "range": 4, "min_range": 2}) == (
            "two to four squares forward, stopped by anything in the way"
        )
        assert _said({**line, "min_range": 2}) == (
            "two squares or more forward, stopped by anything in the way"
        )
        assert _said({**line, "range": 1}) == "one square forward"

    def test_a_leap_says_it_ignores_what_it_passes(self) -> None:
        assert _said({"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}) == (
            "leaps two and one, in all eight directions, jumping over whatever is in "
            "between"
        )
        # A single step passes over nothing, so it is a step and says so.
        assert _said({"kind": "leap", "vectors": [[0, 1]]}) == "one square forward"
        assert _said({"kind": "leap", "vectors": [[2, 2]], "symmetry": "rotate4"}) == (
            "two squares along all four diagonals, jumping over whatever is in between"
        )

    def test_a_lame_leap_names_the_square_that_stops_it(self) -> None:
        """Not "the path must be clear": one square stops it and that is the rule."""
        assert _said({"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8"}) == (
            "leaps two and one, in all eight directions, stopped by a piece standing on "
            "the square it passes"
        )

    def test_a_hop_says_whose_piece_it_jumps_and_where_it_lands(self) -> None:
        line = {"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4"}
        assert _said(line) == (
            "jumps forward, back, left and right, over exactly one piece of either side, "
            "landing anywhere beyond it"
        )
        assert "over one of your own pieces" in _said({**line, "screen": "own"})
        assert "over one enemy piece" in _said({**line, "screen": "enemy"})
        assert _said({**line, "screen_range": 1}).endswith(
            "landing on the square just past it"
        )
        assert _said({**line, "screen_range": 3}).endswith(
            "landing at most three squares past it"
        )
        # Without a `screen_range` the descriptor's own reach is the landing, counted
        # from the screen out.
        assert _said({**line, "range": 2}).endswith("landing at most two squares past it")
        assert _said({**line, "range": 2, "min_range": 2}).endswith(
            "landing exactly two squares past it"
        )

    def test_a_sweep_says_it_takes_its_own_pieces_too(self) -> None:
        assert _said({"kind": "sweep", "vectors": [[1, 1]], "symmetry": "rotate4"}) == (
            "any distance along all four diagonals, sweeping away every piece it passes "
            "over, its own included; a piece it may not capture stops it"
        )


class TestEveryDirectionHasAWord:
    """The symmetries, and the shapes a set of directions comes to."""

    @pytest.mark.parametrize(
        ("vectors", "symmetry", "words"),
        [
            ([[0, 1]], "none", "forward"),
            ([[0, -1]], "none", "back"),
            ([[1, 0]], "none", "right"),
            ([[-1, 0]], "none", "left"),
            ([[1, 1]], "none", "diagonally forward and right"),
            ([[1, 0]], "lateral", "left and right"),
            ([[1, 1]], "lateral", "diagonally forward, either way"),
            ([[1, -1]], "lateral", "diagonally back, either way"),
            ([[1, 0]], "rotate4", "forward, back, left and right"),
            ([[1, 1]], "rotate4", "along all four diagonals"),
            ([[1, 0], [1, 1]], "rotate4", "in all eight directions"),
            # The square's full symmetry group over a straight step is still four
            # directions, because reflecting an orthogonal gives nothing new.
            ([[1, 0]], "dihedral8", "forward, back, left and right"),
            ([[1, 1]], "dihedral8", "along all four diagonals"),
            # Three steps with no name of their own, listed.
            ([[0, 1], [1, 0], [-1, 0]], "none", "forward, left or right"),
        ],
    )
    def test_a_set_of_steps_is_named_as_a_shape(
        self, vectors: list[list[int]], symmetry: str, words: str
    ) -> None:
        move = {"kind": "leap", "vectors": vectors, "symmetry": symmetry}
        assert _said(move) == f"one square {words}"

    @pytest.mark.parametrize(
        ("symmetry", "words"),
        [
            ("none", "two forward and one right"),
            ("lateral", "two forward and one to either side"),
            ("rotate4", "two and one, turned all four ways"),
            ("dihedral8", "two and one, in all eight directions"),
        ],
    )
    def test_a_jump_off_every_line_is_named_by_its_shape(
        self, symmetry: str, words: str
    ) -> None:
        move = {"kind": "leap", "vectors": [[1, 2]], "symmetry": symmetry}
        assert _said(move).startswith(f"leaps {words},")

    def test_a_reach_spelled_out_one_vector_per_distance_is_said_once(self) -> None:
        """`archive/hierarchy-extended.json` writes a line as seven separate leaps."""
        move = {
            "kind": "leap",
            "symmetry": "rotate4",
            "vectors": [[step, 0] for step in range(1, 8)],
        }
        assert _said(move) == (
            "leaps up to seven squares forward, back, left and right, jumping over "
            "whatever is in between"
        )

    def test_a_gap_in_that_reach_is_not_papered_over(self) -> None:
        """Two and four squares out is not *up to four*, so it falls back instead."""
        move = {"kind": "leap", "symmetry": "rotate4", "vectors": [[2, 0], [4, 0]]}
        assert _move_sentence(move) is None


class TestTheModifiers:
    """What a descriptor says on top of where it goes."""

    def test_the_three_modes(self) -> None:
        line = {"kind": "slide", "vectors": [[0, 1]], "range": 1}
        assert _said({**line, "mode": "move"}) == "one square forward, not to take"
        assert _said({**line, "mode": "capture"}) == "one square forward, only to take"
        # Both is what a reader assumes, so saying it would be noise.
        assert _said({**line, "mode": "both"}) == "one square forward"

    def test_the_three_conditions(self) -> None:
        line = {"kind": "slide", "vectors": [[0, 1]], "range": 1}
        assert _said({**line, "condition": {"first_move": True}}) == (
            "one square forward, and only if it has never moved"
        )
        assert _said({**line, "condition": {"from_rank": 1}}) == (
            "one square forward, and only from its own rank 2"
        )
        assert _said({**line, "condition": {"region": ["a1", "b1"]}}) == (
            "one square forward, and only from a1, b1"
        )

    def test_a_facing_piece_says_forward_means_where_it_points(self) -> None:
        """Its vectors are read from its own nose, so the word has to be qualified."""
        piece = {
            "id": "p",
            "name": "turner",
            "facing": {"axes": 4},
            "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"}],
        }
        prose = describe_pieces({"board": {"width": 8, "height": 8}, "pieces": [piece]})[0]
        assert prose.moves == ("one square forward, not to take",)
        assert any("forward above means whichever way it points" in n for n in prose.notes)


class TestTheHexBoard:
    """Schema 14's lattice, where the square board's compass does not hold.

    A hex rank is drawn half a cell to the right of the one below it, so ``+y`` points
    forward *and right* and no cell has a neighbour straight ahead. Naming those six
    directions with the square board's words would be a wrong sentence, so they get
    their own, and anything the six-direction words do not cover falls back.
    """

    def test_the_lines_count_six_and_twelve(self, hex_moves: dict[str, tuple[str, ...]]) -> None:
        assert hex_moves["R"] == (
            "any distance in all six directions, stopped by anything in the way",
        )
        assert hex_moves["B"] == (
            "any distance along all six diagonals, stopped by anything in the way",
        )
        assert hex_moves["Q"] == (
            "any distance in all twelve directions, stopped by anything in the way",
        )
        assert hex_moves["K"] == ("one square in all twelve directions",)

    def test_the_jump_is_named_by_where_it_reaches(
        self, hex_moves: dict[str, tuple[str, ...]]
    ) -> None:
        assert hex_moves["N"] == (
            "leaps to the twelve cells three steps out that lie along no line, jumping "
            "over whatever is in between",
        )

    def test_single_directions_follow_the_drawing(
        self, hex_moves: dict[str, tuple[str, ...]]
    ) -> None:
        assert hex_moves["P"] == (
            "one square forward-right, not to take",
            "exactly two squares forward-right, stopped by anything in the way, not to "
            "take, and only if it has never moved",
            "one square forward-left or right, only to take",
        )

    def test_the_square_boards_words_are_not_borrowed(self) -> None:
        """The same vectors mean different directions on the two lattices."""
        move = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
        assert _said(move) == "any distance forward, back, left and right, stopped by " \
                              "anything in the way"
        assert _move_sentence(move, hexy=True) is None


# ------------------------------------------------------- what it refuses to say


class TestTheFallback:
    """A descriptor no phrase fits gets the vectors, and the reader is told the rule."""

    def test_a_descriptor_no_phrase_fits_falls_back(self) -> None:
        # Lines of one square and of two at once: sayable in English, and not by these
        # rules, so the descriptor stands rather than a sentence that rounds it off. (A line
        # stepping two squares at a time alone is a rider since Q1007a, and is said.)
        move = {"kind": "slide", "vectors": [[1, 0], [2, 2]], "symmetry": "rotate4"}
        assert _move_sentence(move) is None
        piece = {"id": "x", "name": "x", "moves": [move]}
        prose = describe_pieces({"board": {"width": 8, "height": 8}, "pieces": [piece]})[0]
        assert prose.moves == (_move_descriptor(move),)
        assert prose.descriptors == (_move_descriptor(move),)

    def test_the_descriptor_travels_beside_every_sentence(self) -> None:
        """The page keeps it behind the (i), so there is one per move and it is the rule."""
        for piece in describe_pieces(_spec(ROOT / "rulesets" / "chess.json")):
            assert len(piece.descriptors) == len(piece.moves)
            for descriptor in piece.descriptors:
                assert "(" in descriptor and ")" in descriptor
            assert piece.as_dict()["descriptors"] == list(piece.descriptors)

    @pytest.mark.parametrize("path", RULESETS, ids=lambda p: p.stem)
    def test_every_committed_move_is_phrased(self, path: Path) -> None:
        """Nothing in `rulesets/` or `archive/` falls back today.

        The list of exceptions is empty on purpose: a descriptor that starts falling
        back is either a new primitive this module has not been taught or a phrase that
        stopped fitting, and both want a name here rather than a vector on the page.
        """
        spec = _spec(path)
        hexy = (spec.get("board") or {}).get("lattice") == "hex"
        for piece in spec["pieces"]:
            for move in piece.get("moves") or []:
                assert _move_sentence(move, hexy) is not None, (
                    f"{path.stem}: {piece['id']} has a descriptor with no phrase: {move}"
                )


class TestALineAlongAnObliqueStep:
    """A slide whose repeat is a two-and-one offset, said in steps rather than squares.

    The designer's teach-a-move offered every other reading of a click and dropped this
    one, because the vocabulary had no phrase for it (stage 2 of `plan-designer-ux.md`).
    A jump-shaped aim is a *shape*, so the distance counts offsets and the words must
    not say "squares", which would be a wrong sentence about how far the piece goes.
    """

    def test_an_unbounded_line_counts_steps(self) -> None:
        move = {"kind": "slide", "vectors": [[1, 2]], "symmetry": "dihedral8"}
        said = _move_sentence(move)
        # Q1007a: a rider, counted in jumps, and stopped only where it would land.
        assert said == ("any number of jumps of two and one, in all eight directions, "
                        "stopped only by a piece on a square it would land on")

    @pytest.mark.parametrize(
        ("extra", "opening"),
        [
            ({"range": 2}, "up to two jumps of"),
            ({"range": 3, "min_range": 2}, "two to three jumps of"),
            ({"range": 2, "min_range": 2}, "exactly two jumps of"),
            ({"range": 0, "min_range": 3}, "three jumps or more of"),
        ],
    )
    def test_the_reach_is_counted_in_steps(self, extra: dict, opening: str) -> None:
        move = {"kind": "slide", "vectors": [[1, 2]], "symmetry": "rotate4", **extra}
        said = _move_sentence(move)
        assert said is not None and said.startswith(opening), said
        assert "squares" not in said

    def test_a_line_one_offset_long_is_the_leap_and_is_not_said_twice(self) -> None:
        """Nothing stands between the two ends of an oblique step, so a slide of one is
        the leap. Offering both would be two sentences for one rule."""
        one = {"kind": "slide", "vectors": [[1, 2]], "symmetry": "dihedral8", "range": 1}
        assert _move_sentence(one) is None
        leap = dict(one, kind="leap")
        del leap["range"]
        assert _move_sentence(leap) is not None

    def test_a_sweep_along_one_is_said_too(self) -> None:
        move = {"kind": "sweep", "vectors": [[2, 1]], "symmetry": "lateral"}
        said = _move_sentence(move)
        assert said is not None and said.startswith("any number of steps of one forward")

    def test_a_jump_of_unrelated_offsets_still_has_no_line(self) -> None:
        """`up to two squares forward` is a set of jumps, and no line runs along it."""
        move = {"kind": "slide", "vectors": [[0, 1], [0, 2]], "symmetry": "none"}
        assert _move_sentence(move) is None


class TestEveryRuleHasWordsOfItsOwn:
    """A title and an opening sentence per rule key, for a menu offering twenty-eight.

    Thirteen rule keys carry a schema `description` and fifteen carry none, so a menu
    built from the schema alone could only show the key (finding 2.8). These are the
    package's sentences, and the designer places what it is given.
    """

    def test_every_rule_the_schema_knows_has_words(self) -> None:
        keys = set(rule_spec_schema(strict=True)["properties"]["rules"]["properties"])
        assert set(RULE_WORDS) == keys

    def test_a_title_is_words_and_never_the_key(self) -> None:
        for key, (title, says) in RULE_WORDS.items():
            assert title and "_" not in title, key
            assert title == title.lower(), key
            assert says.endswith(".") and says[0].isupper(), key
            assert len(says.split()) >= 6, key

    def test_the_payload_is_plain_data(self) -> None:
        words = rule_words()
        assert words["strike"]["title"] == "what a taken piece becomes"
        assert json.loads(json.dumps(words)) == words


class TestOneSpellingAtATime:
    """Sibling fields that say the same thing, grouped so a card can show one.

    The promotion card offered `rank`, `ranks` and `squares` at once with an *empty
    list* tick beside each. Only one of them is ever read, and which one is the
    compiler's answer -- so the groups are ordered to match it.
    """

    def test_every_group_names_keys_the_schema_has(self) -> None:
        rules = rule_spec_schema(strict=True)["properties"]["rules"]["properties"]
        for path, groups in rule_choices().items():
            node = rules[path.split(".")[1].removesuffix("[]")]
            # A promotion is one rule or a list of them; either branch has the keys.
            holders = node.get("anyOf", [node])
            for group in groups:
                for option in group["options"]:
                    for key in option["keys"]:
                        assert any(
                            key in (holder.get("properties") or {})
                            or key in ((holder.get("items") or {}).get("properties") or {})
                            for holder in holders
                        ), f"{path}: {key}"

    def test_no_key_is_in_two_options_of_one_group(self) -> None:
        for path, groups in rule_choices().items():
            for group in groups:
                keys = [key for option in group["options"] for key in option["keys"]]
                assert len(keys) == len(set(keys)), path

    def test_the_last_option_is_the_one_the_compiler_reads(self) -> None:
        """Held against `schema.promotion_squares`, which resolves the zone."""
        promotion = {"piece": "P", "rank": 0, "ranks": [1], "squares": ["a3"], "to": ["Q"]}
        first, _second = promotion_squares(promotion, 3, 3)
        assert first == frozenset({6}), "squares wins, so it is the group's last option"
        zone = rule_choices()["rules.promotion"][0]
        assert zone["options"][-1]["keys"] == ["squares"]
        assert zone["options"][0]["keys"] == ["rank"]


class TestEveryModifierIsSaid:
    """Drop a key that changes the game and the sentence has to change with it.

    The sibling of `test_spec.py`'s rule and piece-property coverage, one level further
    down. It is the check that makes a new move key cheap to get right: add it to the
    schema without teaching this module, and this fails by name rather than three months
    later on a page describing a different piece.
    """

    BASE: dict[str, Any] = {
        "kind": "hop",
        "vectors": [[1, 0], [1, 1]],
        "symmetry": "rotate4",
        "range": 3,
        "min_range": 2,
        "mode": "capture",
        "screen": "enemy",
        "condition": {"first_move": True},
    }

    @pytest.mark.parametrize(
        "key",
        ["vectors", "symmetry", "range", "min_range", "mode", "condition", "screen"],
    )
    def test_removing_a_key_changes_the_sentence(self, key: str) -> None:
        without = copy.deepcopy(self.BASE)
        if key == "vectors":
            without["vectors"] = [[1, 0]]
        else:
            del without[key]
        assert _said(without) != _said(self.BASE), f"{key} is said nowhere"

    def test_screen_range_overrides_the_reach_and_says_so(self) -> None:
        """It is the one key that means a different thing from `range`."""
        capped = {**self.BASE, "screen_range": 2}
        assert _said(capped) != _said(self.BASE)
        assert "two squares past it" in _said(capped)

    def test_a_kind_is_not_the_same_sentence_as_another(self) -> None:
        line = {"vectors": [[1, 2]], "symmetry": "dihedral8"}
        said = {kind: _said({**line, "kind": kind}) for kind in ("leap", "lame")}
        assert said["leap"] != said["lame"]
        span = {"vectors": [[1, 1]], "symmetry": "rotate4"}
        assert _said({**span, "kind": "slide"}) != _said({**span, "kind": "sweep"})


# Whole words only. "looking" holds "king" and "lancer" holds "lance", and a substring
# test that failed on those would be a test about English rather than about pieces.
_GAME_WORDS = (
    "pawn", "knight", "bishop", "rook", "queen", "king", "lance", "cannon", "chariot",
    "elephant", "advisor", "soldier", "chess", "shogi", "xiangqi", "glinski",
)
_NAMED = re.compile(r"\b(" + "|".join(_GAME_WORDS) + r")s?\b")


def _emittable_strings(path: Path) -> list[str]:
    """Every string literal the module can put in front of a reader.

    Docstrings are dropped and comments never reach the AST, so what is left is the
    prose itself. The reasoning in this project's comments names real games on purpose
    -- *the xiangqi horse is the reason* is why a primitive exists -- and holding a
    comment to the same rule as a phrase table would cost the reasoning to buy nothing.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    documented = {
        node.body[0].value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node not in documented
    ]


# Four strings predate this test, all about `rules.castling` and `rules.strike`.
#
# `rules.castling` spells its two participants `king` and `rook` in the schema itself, so
# those literals are key lookups and the sentence around them uses the schema's nouns for
# them; the pieces a reader sees are whatever the spec put behind those keys. The strike
# landing phrase is an idiom for the eight neighbours, it is asserted word for word in
# `test_schema_twelve.py`, and `loop/operators.py` writes the same phrase about the same
# rule -- so moving it is a change to three files about a rule this box does not touch.
_OLDER_PROSE = frozenset({
    "king",
    "king_file",
    "kingside",
    "queenside",
    "rook",
    "any empty square a king's step from where it stood",
    "Castling does not use the join; king and rook stay on their rank.",
})

# The words of `RULE_ALSO_CALLED` (Q995e) are what a player types to find a rule, the name
# of a game or a piece they bring from elsewhere (*king of the hill*, *bare king*). Nothing
# says them to anyone, so they are search words and not prose.
_SEARCH_WORDS = frozenset(word for names in RULE_ALSO_CALLED.values() for word in names)


def test_no_game_and_no_piece_is_named_in_the_prose() -> None:
    """Rule 5, on the module whose whole job is to talk about pieces.

    A phrase table with "knight" in it would read beautifully for one game and describe
    a different piece in the next, which is the failure the rule exists to stop.
    """
    for text in _emittable_strings(ROOT / "chess_variant_sandbox" / "spec" / "describe.py"):
        if text in _OLDER_PROSE or text in _SEARCH_WORDS:
            continue
        found = _NAMED.search(text.lower())
        assert found is None, f"describe.py says {found.group()!r} in {text!r}"


@pytest.mark.parametrize("path", RULESETS, ids=lambda p: p.stem)
def test_no_sentence_borrows_a_word_from_a_game(path: Path) -> None:
    """The same rule read off the output, which is where it would actually show."""
    for piece in describe_pieces(_spec(path)):
        for sentence in piece.moves:
            found = _NAMED.search(sentence.lower())
            assert found is None, f"{path.stem}: {piece.id} is said with {found!r}"


# ------------------------------------------------------------------- the landing phrase


class TestLandingPhrase:
    """Where `position: "choose"` may place a survivor (`_landing_phrase`, Q20/Q484).

    Two whole landings are named as idioms rather than read out as raw vectors; anything
    else falls back to the vectors so a reader is told the truth instead of the nearest
    phrase that almost fits.
    """

    def test_the_king_step_landing_is_named(self) -> None:
        assert _landing_phrase({"landing": KING_STEP_LANDING}) == (
            "any empty square a king's step from where it stood"
        )

    def test_the_hex_neighbours_landing_is_named(self) -> None:
        """`transform.HEX_STEP_LANDING` (Q484) is the six cells sharing a side with the
        struck one -- the king step's own reading on a hex lattice -- spelled out as six
        vectors under symmetry "none" because a hex board refuses every other symmetry.
        Read out as raw vectors it used to say six pairs of numbers; it is now named the
        way the king step is."""
        assert _landing_phrase({"landing": HEX_STEP_LANDING}) == (
            "any of the six empty cells touching where it stood"
        )

    def test_an_unrecognised_landing_still_falls_back_to_its_vectors(self) -> None:
        landing = {"vectors": [[2, 0]], "symmetry": "none"}
        assert _landing_phrase({"landing": landing}) == (
            "any empty square one of these steps from where it stood, forward being "
            "the attacker's: [2, 0]"
        )

    def test_no_landing_falls_back_to_the_attacker_s_choosing(self) -> None:
        assert _landing_phrase({}) == "an empty square of the attacker's choosing"


# ------------------------------------------------------------ a piece that never moves


STONES = ROOT / "archive" / "stones.json"
THREE_STONES = ROOT / "archive" / "three-stones.json"


def _never_moves(spec: dict[str, Any]) -> tuple[str, ...]:
    """The notes of the one piece in ``spec`` with no moves."""
    (notes,) = [piece.notes for piece in describe_pieces(spec) if not piece.moves]
    return notes


def _anything_takes_it(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with every ``captures`` list dropped: Q131's stones before their repair."""
    broken = copy.deepcopy(spec)
    for piece in broken["pieces"]:
        piece.pop("captures", None)
    return broken


def _with_a_quiet_wild_piece(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` plus a wild piece whose own moves never capture, walking four squares.

    It has no ``captures`` list, so its type may take anything, and it eats on its
    route whatever its moves say.
    """
    walked = copy.deepcopy(spec)
    walked["pieces"].append({
        "id": "W", "name": "wanderer", "value": 3,
        "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4",
                   "mode": "move"}],
    })
    walked["rules"]["npcs"] = [{"piece": "W", "route": ["c4", "d4", "e4", "f4"]}]
    return walked


class TestAPieceThatNeverMoves:
    """What a moveless piece is told about leaving its square (Q132, Q426).

    It used to be told *"stays there until it is taken"* whatever the rest of the spec
    said, so the library page, the judge and the design page's hint all read that the
    repaired stones can be taken -- the one thing their repair (Q131) made untrue.
    """

    @pytest.mark.parametrize("path", [STONES, THREE_STONES], ids=lambda p: p.stem)
    def test_a_stone_nothing_may_take_stays_for_the_whole_game(self, path: Path) -> None:
        spec = _spec(path)
        compiled = compile_spec(spec)
        stone = compiled.piece_ids.index("X")
        # The trap the reading has to avoid: the stone's own `captures` is absent, so
        # its column is not empty -- a stone is *permitted* to take a stone.
        assert compiled.capture_allowed[stone][stone]
        assert not capturing_types(compiled)[stone]

        notes = _never_moves(spec)
        assert not any("until it is taken" in note for note in notes), notes
        assert "nothing can capture it" in notes[0]
        assert "for the rest of the game" in notes[0]

    @pytest.mark.parametrize("path", [STONES, THREE_STONES], ids=lambda p: p.stem)
    def test_a_stone_anything_may_take_is_said_to_be_taken(self, path: Path) -> None:
        assert _never_moves(_anything_takes_it(_spec(path)))[0] == (
            "never moves: once it is on a square it stays there until it is taken")

    def test_one_piece_that_may_take_it_is_enough(self) -> None:
        spec = _spec(STONES)
        next(piece for piece in spec["pieces"] if piece["id"] == "Q")["captures"].append("X")
        assert "until it is taken" in _never_moves(spec)[0]

    def test_a_wild_piece_that_walks_takes_it_whatever_its_moves_say(self) -> None:
        spec = _with_a_quiet_wild_piece(_spec(STONES))
        compiled = compile_spec(spec)
        assert capturing_types(compiled)[compiled.piece_ids.index("W")]
        assert "until it is taken" in _never_moves(spec)[0]

    def test_a_blast_that_can_reach_it_is_said(self) -> None:
        spec = _spec(STONES)
        spec["rules"]["explode_on_capture"] = {"immune": ["K"]}
        notes = _never_moves(spec)
        assert "nothing can capture it" in notes[0]
        assert "until an explosion beside it destroys it" in notes[0]

        spec["rules"]["explode_on_capture"] = {"immune": ["K", "X"]}
        assert "for the rest of the game" in _never_moves(spec)[0]


# ------------------------------------------------------------ which ending is asked first


@pytest.mark.parametrize(
    "name", ["cannonade", "marchfield", "reckoning", "wolves-bounty"])
def test_checkmate_is_said_to_come_before_the_scored_clock(name: str) -> None:
    """Q412's leftover. Since `9e4f098` the core looks for mate before both clocks, so a
    mate on the ply a clock runs out stands; the four archived games that score the
    clock and are won by checkmate said the old order. A stalemate on that ply is still
    the clock's, so where it decides the game it stays after the clock."""
    lines = describe_rules(_spec(ROOT / "archive" / f"{name}.json"))
    (order,) = [line for line in lines if line.startswith("When more than one")]
    assert order.index("checkmate") < order.index("the clock, decided on the score"), order
    if "having no legal move" in order:
        assert order.index("the clock") < order.index("having no legal move"), order


# ------------------------------------------------------------- what a fork changed


class TestWhatAForkChanged:
    """`describe_changes`: a fork's differences from its parent, in the parts' own words.

    The calibration game is the parent only because it is the spec every reader knows;
    nothing below depends on which game it is.
    """

    @staticmethod
    def _fork() -> tuple[dict[str, Any], dict[str, Any]]:
        parent = _spec(ROOT / "rulesets" / "chess.json")
        child = copy.deepcopy(parent)
        child["name"], child["description"] = "a fork", "renamed, which is not a rule"
        child["pieces"][1]["moves"][0]["vectors"] = [[1, 3]]
        child["rules"]["max_plies"] = 300
        return parent, child

    def test_one_move_and_one_rule_are_named_and_nothing_else(self) -> None:
        parent, child = self._fork()
        piece = child["pieces"][1]
        lines = describe_changes(parent, child)
        assert len(lines) == 2, lines
        moved, ruled = lines
        # The piece by name, its old move and its new one, each in the sentence the
        # rules panel already shows for it.
        old = describe_pieces(parent)[1].moves[0]
        new = describe_pieces(child)[1].moves[0]
        assert old != new
        assert moved.startswith(piece["name"]) and old in moved and new in moved
        # The rule as one replacement, not a removal and an addition.
        (was,) = set(describe_rules(parent)) - set(describe_rules(child))
        (now,) = set(describe_rules(child)) - set(describe_rules(parent))
        assert was in ruled and now in ruled and "300" in now

    def test_an_unchanged_fork_changed_nothing(self) -> None:
        parent, _ = self._fork()
        child = copy.deepcopy(parent)
        child["name"] = "only renamed"
        assert describe_changes(parent, child) == []

    def test_an_added_and_a_removed_piece_are_said(self) -> None:
        parent, _ = self._fork()
        child = copy.deepcopy(parent)
        gone = child["pieces"].pop()
        child["pieces"].append({**copy.deepcopy(gone), "id": "Z", "name": "zed"})
        lines = describe_changes(parent, child)
        assert any(line.startswith("A new piece, zed") for line in lines), lines
        assert f"{gone['name']} is gone." in lines


# ------------------------------------------------ Q737: the words a small spec must produce

DULL = ROOT / "rulesets" / "dull.json"


def _small(**rules: Any) -> dict[str, Any]:
    """A one-piece game (`G`, 5 by 5) with ``rules`` merged over its own."""
    spec = _spec(DULL)
    spec["rules"].update(rules)
    return spec


def _move(kind: str, vectors: list[list[int]], **rest: Any) -> dict[str, Any]:
    return {"kind": kind, "vectors": vectors, **rest}


class TestTheProseValuesAreFrozen:
    @pytest.mark.parametrize("value", [
        PieceProse("A", "a", False, ("m",)),
        _Aim("words", True, 1),
    ], ids=["PieceProse", "_Aim"])
    def test_the_dataclass_is_frozen_and_hashable(self, value: Any) -> None:
        hash(value)
        field = dataclasses.fields(value)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field, "other")


class TestBoardSetupAndClocks:
    @pytest.mark.parametrize("width, article", [(9, "a"), (8, "an"), (7, "a"), (11, "an"), (18, "an"), (12, "a")])
    def test_a_plain_hex_grid_takes_the_article_its_number_wants(
        self, width: int, article: str
    ) -> None:
        board = {"width": width, "height": width, "lattice": "hex"}
        assert _board_words({"board": board, "rules": {}}) == (
            f"a hexagonal board of {width * width} cells, {article} {width} by {width} "
            "hex grid; every cell touches six others")

    def test_a_setup_without_mirror_mirrors_the_first_army(self) -> None:
        spec = _spec(DULL)
        del spec["setup"]["mirror"]
        assert describe_setup(spec) == [
            "G on a1, b1, c1, d1, e1",
            "The second player's army mirrors the first across the board.",
        ]

    def test_castling_without_steps_moves_two_squares(self) -> None:
        lines = describe_rules(_small(castling={"king": "K", "rook": "R"}))
        assert ("K may move 2 squares toward an unmoved R, which jumps to the square it "
                "crossed.") in lines

    def test_a_spec_without_clocks_gets_the_default_ones(self) -> None:
        spec = _spec(DULL)
        for key in ("no_progress_draw", "max_plies"):
            del spec["rules"][key]
        assert describe_rules(spec)[-1] == (
            "The game is drawn after 3 repetitions, 100 turns without a capture or other "
            "lasting change, or 400 turns in all.")


class TestWhatAForkChangedExactly:
    def test_a_new_piece_that_moves_is_named_with_its_moves(self) -> None:
        parent = _spec(DULL)
        child = copy.deepcopy(parent)
        child["pieces"].append(
            {"id": "Z", "name": "zed", "value": 1, "moves": [_move("leap", [[0, 1]])]})
        assert describe_changes(parent, child) == [
            "A new piece, zed, moving “one square forward”."]

    def test_a_new_piece_without_moves_is_named_alone(self) -> None:
        parent = _spec(DULL)
        child = copy.deepcopy(parent)
        child["pieces"].append({"id": "Z", "name": "zed", "value": 1})
        assert describe_changes(parent, child) == ["A new piece, zed."]

    def test_a_new_piece_does_not_hide_a_later_change(self) -> None:
        parent = _spec(DULL)
        child = copy.deepcopy(parent)
        child["pieces"].insert(0, {
            "id": "Z", "name": "zed", "value": 1,
            "moves": [_move("leap", [[0, 1]], mode="move")]})
        child["pieces"][1]["name"] = "grunter"
        assert describe_changes(parent, child) == [
            "A new piece, zed, moving “one square forward, not to take”.",
            "grunt is now called grunter.",
        ]


class TestCapturingTypesOfAWalker:
    @staticmethod
    def _walker(route: list[str]) -> Any:
        spec = _spec(DULL)
        spec["schema_version"] = 25
        for move in spec["pieces"][0]["moves"]:
            move["mode"] = "move"
        spec["rules"]["npcs"] = [{"piece": "G", "route": route}]
        return compile_spec(spec)

    def test_the_first_declared_type_walking_a_route_captures(self) -> None:
        assert capturing_types(self._walker(["c3", "c4"])) == (True,)

    def test_a_one_square_route_never_steps_and_captures_nothing(self) -> None:
        assert capturing_types(self._walker(["c3"])) == (False,)


class TestAMovelessPieceWithOtherNotes:
    def test_the_moveless_sentence_comes_first(self) -> None:
        spec = _spec(DULL)
        piece = spec["pieces"][0]
        piece.update({"moves": [], "captures": [], "region": ["a1", "b1"]})
        assert describe_pieces(spec)[0].notes == (
            "never moves: once it is on a square it stays there until it is taken",
            "may never leave a1, b1",
            "may never capture anything, and still blocks a line like any piece",
        )


class TestTheDescriptorOfAMove:
    @pytest.mark.parametrize("move, text", [
        (_move("leap", [[1, 2]], symmetry="dihedral8"),
         "moves and captures along (+1,+2) in all eight directions, jumping directly, "
         "ignoring anything in between [8 directions]"),
        (_move("slide", [[0, 1]]),
         "moves and captures along (+0,+1), any distance until blocked"),
        (_move("slide", [[0, 1]], range=2, min_range=2),
         "moves and captures along (+0,+1), exactly 2 squares, path must be clear"),
        (_move("slide", [[0, 1]], range=1, min_range=1),
         "moves and captures along (+0,+1), exactly 1 square, path must be clear"),
        (_move("slide", [[0, 1]], range=3),
         "moves and captures along (+0,+1), up to 3 squares"),
        (_move("sweep", [[0, 1]]),
         "moves and captures along (+0,+1), jumping any distance along the line, and "
         "every piece it passes over, its own included, is removed; a piece it may not "
         "capture stops it"),
        (_move("sweep", [[0, 1]], range=2, min_range=2),
         "moves and captures along (+0,+1), jumping exactly 2 squares along the line, "
         "and every piece it passes over, its own included, is removed; a piece it may "
         "not capture stops it"),
        (_move("sweep", [[0, 1]], range=3),
         "moves and captures along (+0,+1), jumping up to 3 squares along the line, and "
         "every piece it passes over, its own included, is removed; a piece it may not "
         "capture stops it"),
        (_move("hop", [[0, 1]]),
         "moves and captures along (+0,+1), leaping exactly one piece of either side, "
         "then any distance until blocked"),
        (_move("hop", [[0, 1]], screen_range=1),
         "moves and captures along (+0,+1), leaping exactly one piece of either side, "
         "landing immediately past it"),
        (_move("hop", [[0, 1]], screen_range=2),
         "moves and captures along (+0,+1), leaping exactly one piece of either side, "
         "landing at most 2 squares past it"),
        (_move("hop", [[0, 1]], screen="own", screen_range=1),
         "moves and captures along (+0,+1), leaping one of your own pieces, landing "
         "immediately past it"),
        (_move("hop", [[0, 1]], screen="enemy", range=3),
         "moves and captures along (+0,+1), leaping one enemy piece, then up to 3 squares"),
    ])
    def test_the_descriptor_says_the_vectors_the_reach_and_the_direction_count(
        self, move: dict[str, Any], text: str
    ) -> None:
        assert _move_descriptor(move) == text


class TestSayingNumbersAndSteps:
    @pytest.mark.parametrize("count, word", [
        (2, "two"), (3, "three"), (8, "eight"), (9, "nine"), (10, "ten"), (11, "eleven"),
        (12, "twelve"), (13, "13"),
    ])
    def test_an_exact_slide_says_its_length_in_words_up_to_twelve(
        self, count: int, word: str
    ) -> None:
        move = _move("slide", [[0, 1]], range=count, min_range=count)
        assert move_sentence(move) == f"exactly {word} squares forward, stopped by anything in the way"

    @pytest.mark.parametrize("vector, words", [
        ([-1, 1], "diagonally forward and left"),
        ([1, 1], "diagonally forward and right"),
        ([-1, -1], "diagonally back and left"),
        ([1, -1], "diagonally back and right"),
        ([0, 1], "forward"),
        ([0, -1], "back"),
        ([-1, 0], "left"),
        ([1, 0], "right"),
    ])
    def test_each_single_step_has_its_own_name(self, vector: list[int], words: str) -> None:
        assert move_sentence(_move("leap", [vector])) == f"one square {words}"

    def test_forward_and_back_is_one_family(self) -> None:
        assert move_sentence(_move("slide", [[0, 1], [0, -1]], range=3)) == (
            "up to three squares forward and back, stopped by anything in the way")
        assert move_sentence(_move("slide", [[0, 1], [0, -1]])) == (
            "any distance forward and back, stopped by anything in the way")

    def test_four_named_steps_that_are_no_family_are_said_per_vector(self) -> None:
        assert move_sentence(_move("leap", [[0, 1], [1, 0], [-1, 0], [1, 1]])) == (
            "one square forward, and right, and left, and diagonally forward and right")

    def test_a_run_of_two_distances_is_up_to_the_longer(self) -> None:
        assert move_sentence(_move("leap", [[0, 1], [0, 2]], symmetry="rotate4")) == (
            "leaps up to two squares forward, back, left and right, jumping over "
            "whatever is in between")

    def test_three_hex_sides_are_one_phrase(self) -> None:
        assert move_sentence(_move("leap", [[-1, 1], [0, 1], [1, 0]]), hexy=True) == (
            "one square forward-left or forward-right or right")


class TestWhereTheVectorsAreTheBetterAnswer:
    def test_a_one_step_oblique_sweep_keeps_its_vector_reading(self) -> None:
        move = _move("sweep", [[1, 2]], symmetry="dihedral8", range=1)
        assert move_sentence(move) == _move_descriptor(move)
        assert "exactly 1 square along the line" in move_sentence(move)

    def test_a_one_step_oblique_slide_is_the_leap_and_a_longer_one_is_not(self) -> None:
        move = _move("slide", [[1, 2]], symmetry="dihedral8", range=1)
        assert move_sentence(move) == (
            "leaps two and one, in all eight directions, jumping over whatever is in between")
        move["range"] = 2
        assert move_sentence(move) == (
            "up to two jumps of two and one, in all eight directions, stopped only by a piece o"
            "n a square it would land on")

    def test_a_one_step_slide_with_a_longer_aim_keeps_its_vector_reading(self) -> None:
        move = _move("slide", [[0, 2]], range=1)
        assert move_sentence(move) == (
            "moves and captures along (+0,+2), exactly 1 square, path must be clear")

    def test_a_hop_that_is_not_one_square_a_step_keeps_its_vector_reading(self) -> None:
        for move in (_move("hop", [[0, 2]], symmetry="rotate4"),
                     _move("hop", [[1, 2]], symmetry="dihedral8")):
            assert move_sentence(move) == _move_descriptor(move)
            assert "leaping exactly one piece" in move_sentence(move)

    def test_an_unlimited_oblique_line_with_a_minimum_says_so(self) -> None:
        move = _move("slide", [[1, 2]], symmetry="dihedral8", min_range=2)
        assert move_sentence(move) == (
            "two jumps or more of two and one, in all eight directions, stopped only by a piece o"
            "n a square it would land on")
        del move["min_range"]
        assert move_sentence(move) == (
            "any number of jumps of two and one, in all eight directions, stopped only by a piece o"
            "n a square it would land on")


class TestTurnsPromotionsAndWildPieces:
    @pytest.mark.parametrize("turn, tail", [
        ({"moves": 3, "ramp": [1]}, "taking 1 move before 3"),
        ({"moves": 3, "ramp": [2]}, "taking 2 moves before 3"),
        ({"moves": 3, "ramp": [1, 2]}, "taking 1, 2 moves before 3"),
    ])
    def test_the_ramp_is_said_in_the_right_number(self, turn: dict[str, Any], tail: str) -> None:
        line = next(
            line for line in describe_rules(_small(turn=turn)) if line.startswith("A turn"))
        assert line == (
            "A turn is 3 moves rather than one. The opening turns are shorter, "
            f"{tail} for the rest of the game.")

    @staticmethod
    def _squares(count: int) -> list[str]:
        return [f"{file}{rank}" for file in "abcde" for rank in range(1, 6)][:count]

    def test_a_square_zone_lists_sixteen_and_summarises_seventeen(self) -> None:
        sixteen = self._squares(16)
        assert _describe_promotion({"piece": "G", "squares": sixteen, "to": ["Q"]}) == (
            f"G reaching {', '.join(sixteen)} becomes one of Q")
        seventeen = self._squares(17)
        assert _describe_promotion({"piece": "G", "squares": seventeen, "to": ["Q"]}) == (
            f"G reaching {', '.join(seventeen[:16])}, ... (17 squares) becomes one of Q")

    @pytest.mark.parametrize("ranks, where", [([7], "rank 8"), ([6, 7], "ranks 7, 8")])
    def test_ranks_are_one_based_and_plural_from_two(self, ranks: list[int], where: str) -> None:
        assert _describe_promotion({"piece": "G", "ranks": ranks, "to": ["Q"]}) == (
            f"G reaching {where} becomes one of Q")

    def test_a_per_player_list_promotes_into_the_first_of_each(self) -> None:
        rules = _small(promotion={
            "piece": "G", "rank": "last",
            "to_by_player": {"0": ["Q", "R"], "1": ["Q", "R"]},
        }, promote_on_entry=True)
        (line,) = [line for line in describe_rules(rules) if line.startswith("A piece that enters")]
        assert "to the first piece listed (G to Q), whether" in line

    def test_a_two_square_wild_route_is_walked_and_back(self) -> None:
        spec = _small(npcs=[{"piece": "G", "route": ["d4", "d5"]}])
        (line,) = [line for line in describe_rules(spec) if line.startswith("A wild")]
        assert ("It starts on d4 and, after every move, steps along d4, d5 and back to d4."
                ) in line

    @pytest.mark.parametrize("captures, gives_check", [
        (None, True),
        (["K"], True),
        (["P", "K"], True),
        (["P"], False),
        ([], False),
    ], ids=["anything", "the king", "pawn and king", "a pawn only", "nothing"])
    def test_the_check_a_wild_piece_gives_is_the_one_the_core_gives(
        self, captures: list[str] | None, gives_check: bool
    ) -> None:
        """A wild piece checks a royal only if its type may capture it (Q994, Q994w)."""
        wild: dict[str, Any] = {
            "id": "W", "name": "wolf", "value": 3,
            "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4",
                       "mode": "move"}],
        }
        if captures is not None:
            wild["captures"] = captures
        spec = {
            "schema_version": 9,  # captures (3) and rules.npcs (9)
            "name": "wolfcheck",
            "description": "A king beside a wild piece that walks onto its square.",
            "board": {"width": 5, "height": 5},
            "pieces": [
                {"id": "P", "name": "pawn", "value": 1,
                 "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1,
                            "mode": "move"}]},
                {"id": "K", "name": "king", "value": 0, "royal": True,
                 "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]],
                            "symmetry": "rotate4", "mode": "both"}]},
                wild,
            ],
            "setup": {"mirror": True, "placements": [
                {"piece": "K", "squares": ["a1"]}, {"piece": "P", "squares": ["e1"]}]},
            "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 60,
                      "npcs": [{"piece": "W", "route": ["c3", "c4"]}]},
        }
        # The wild piece starts on c3 and steps to c4 next: put player 0's king there.
        engine = Engine(compile_spec(spec))
        state = engine.initial_state()
        compiled = engine.spec
        width = compiled.width
        board = bytearray(state.board)
        king = compiled.piece_ids.index("K")
        stride = compiled.n_facings * compiled.n_types
        mine = next(
            value for value in board
            if value and (value - 1) % compiled.n_types == king and (value - 1) // stride == 0
        )
        for square, value in enumerate(board):
            if value == mine:
                board[square] = 0
        board[3 * width + 2] = mine  # c4
        state = dataclasses.replace(state, board=bytes(board))
        assert engine.in_check(state, 0) is gives_check

        (line,) = [line for line in describe_rules(spec) if line.startswith("A wild")]
        assert ("is in check from it" in line and "not in check" not in line) is gives_check
        if not gives_check and captures:
            assert "not in check from it" in line

    @pytest.mark.parametrize("npc, value, worth", [
        ({}, None, "Taking it is worth 1 points."),
        ({}, 3, "Taking it is worth 3 points."),
        ({"bounty": 0}, 3, "Taking it is worth nothing."),
        ({"bounty": 2.5}, 3, "Taking it is worth 2.5 points."),
    ])
    def test_the_bounty_defaults_to_the_value_then_one_and_zero_is_nothing(
        self, npc: dict[str, Any], value: int | None, worth: str
    ) -> None:
        spec = _small(npcs=[{"piece": "G", "route": ["d4"], **npc}])
        if value is None:
            del spec["pieces"][0]["value"]
        else:
            spec["pieces"][0]["value"] = value
        (line,) = [line for line in describe_rules(spec) if line.startswith("A wild")]
        assert "It stands on d4 and never moves." in line
        assert line.endswith(f" {worth}")


class TestTheRemainingRuleDefaults:
    def test_extinction_defaults_to_holding_fewer_than_one(self) -> None:
        assert "Holding fewer than 1 of G is a loss." in describe_rules(
            _small(extinction={"pieces": ["G"]}))

    def test_drops_without_from_captures_say_captured_pieces_change_sides(self) -> None:
        assert ("Captured pieces change sides and may be played back onto the board."
                ) in describe_rules(_small(drops={"no_mate": ["G"]}))

    def test_an_empty_explosion_says_nothing(self) -> None:
        assert not any("destroys" in line for line in describe_rules(_small(explode_on_capture={})))

    def test_a_territory_goal_of_one_square_is_a_way_to_win(self) -> None:
        territory = {"mode": "final_rank", "win_at": 1}
        (order,) = [line for line in describe_rules(_small(territory=territory))
                    if line.startswith("When more than one")]
        assert "then holding enough ground" in order
        for win_at in (0, None):
            if win_at is None:
                del territory["win_at"]
            else:
                territory["win_at"] = win_at
            assert not any(
                line.startswith("When more than one")
                for line in describe_rules(_small(territory=territory)))
