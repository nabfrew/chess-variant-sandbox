"""Counting the rules two specs differ in.

Two things are under test and they pull in opposite directions. A *spelling* change must
cost nothing -- renaming pieces, reordering lists, writing a symmetry out by hand -- and
a *rule* change must cost exactly one. Everything below is one of those two claims.

The specs are written here rather than read from `rulesets/`, because what is being
checked is what the DSL says and not what any game happens to be. The three places a
committed ruleset appears are named: the two spelling nulls sweep the whole corpus, and
`docs/archive/plan-game-graph.md`'s own acceptance names chess as the library's hub.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path

import pytest

from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.relate import (
    KINDS,
    RULE_KINDS,
    RULE_WORDS,
    SETTING_WEIGHT,
    SIMILARITY_RATIO,
    _orientation_played,
    _played,
    _restated,
    _setup_cells,
    identity_key,
    mechanisms,
    normal_form,
    piece_identity,
    relate,
    relate_forms,
    rules_involved,
    similarity,
    units,
)
from chess_variant_sandbox.spec.schema import _RULES_KEYS, SpecError
from chess_variant_sandbox.spec.transform import rename_everything, reorder_pieces

# A two-piece game on a small board, with a promoting walker that can double-step, so
# that promotion, en passant and castling all have something to be about. Everything
# below is a copy of this with one thing changed.
BASE: dict = {
    "schema_version": 2,
    "name": "base",
    "board": {"width": 6, "height": 6},
    "pieces": [
        {
            "id": "K",
            "name": "chief",
            "royal": True,
            "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
        },
        {
            "id": "W",
            "name": "walker",
            "value": 1,
            "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 2, "mode": "move"},
                {"kind": "leap", "vectors": [[1, 1], [-1, 1]], "mode": "capture"},
            ],
        },
        {
            "id": "R",
            "name": "tower",
            "value": 5,
            "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}],
        },
    ],
    "setup": {
        "mirror": True,
        "placements": [
            {"piece": "K", "squares": ["c1"]},
            {"piece": "R", "squares": ["a1", "f1"]},
            {"piece": "W", "squares": ["a2", "b2", "c2", "d2", "e2", "f2"]},
        ],
    },
    "rules": {
        "win": "checkmate",
        "promotion": {"piece": "W", "rank": 5, "to": ["R"]},
        "en_passant": True,
    },
}


def changed(**rules) -> dict:
    """`BASE` with these `rules` keys set, or removed where the value is `None`."""
    spec = copy.deepcopy(BASE)
    for key, value in rules.items():
        if value is None:
            spec["rules"].pop(key, None)
        else:
            spec["rules"][key] = value
    return spec


def piece(spec: dict, pid: str) -> dict:
    return next(entry for entry in spec["pieces"] if entry["id"] == pid)


def corpus_specs() -> list[tuple[str, dict]]:
    root = Path(__file__).resolve().parents[1]
    found = []
    for folder in ("rulesets", "archive"):
        for path in sorted((root / folder).glob("*.json")):
            found.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
    return found


def test_the_base_spec_is_valid() -> None:
    assert validate_spec(BASE) == []


# ------------------------------------------------------------- every rule has a kind


def test_every_rules_key_is_classified() -> None:
    """A schema bump that adds a `rules` key fails here until the key has a kind.

    A link with no kind is a link the map cannot colour and a chip cannot hide, and the
    failure it causes is silent: the rule still counts, it is just filed nowhere. The
    `new-move-primitive` skill carries the step that clears this.
    """
    assert set(RULE_KINDS) == set(_RULES_KEYS)
    assert set(RULE_WORDS) == set(_RULES_KEYS)
    assert set(RULE_KINDS.values()) <= set(KINDS)
    # And the words say what the rule is about rather than restating its key.
    assert all(words and words != key for key, words in RULE_WORDS.items())


def test_a_piece_id_is_renamed_where_the_schema_says_one_goes_and_nowhere_else() -> None:
    """The probe renamed by matching the string anywhere; this renames by position.

    A piece may be given any id the schema accepts, so a rule whose *value* happens to
    spell one -- a square, an ending, a count word -- would be rewritten as a reference to
    that piece under a string match. Here the rule's value is `"e4"` in one game and
    `"d4"` in the other while a piece is called `e4`, and it still reads as one rule about
    that rule rather than as a piece that vanished.
    """
    spec = copy.deepcopy(BASE)
    piece(spec, "R")["id"] = "e4"
    for placement in spec["setup"]["placements"]:
        if placement["piece"] == "R":
            placement["piece"] = "e4"
    spec["rules"]["promotion"] = {"piece": "W", "rank": 5, "to": ["e4"]}
    spec["rules"]["reach"] = {"squares": ["e4"], "piece": "K"}
    other = copy.deepcopy(spec)
    other["rules"]["reach"] = {"squares": ["d4"], "piece": "K"}
    relation = relate(spec, other)
    assert relation.units == 1
    assert relation.differences[0].unit == "rules.reach"


def test_every_rules_key_that_names_a_piece_has_a_position_row() -> None:
    """The position map against the validator's own refusals, key by key.

    `schema.py` checks each place a piece id may appear inline, so the map here is a
    second statement of the same fact and could drift from it. This asks the validator: a
    rule that names a piece the spec does not declare must be refused, and any key the
    validator refuses for that reason and the map does not carry is a hole.
    """
    from chess_variant_sandbox.spec.relate import _PIECE_POSITIONS

    naming = {
        "castling": {"king": "K", "rook": "ZZ"},
        "promotion": {"piece": "ZZ", "rank": 5, "to": ["R"]},
        "reach": {"squares": ["c6"], "piece": "ZZ"},
        "extinction": {"pieces": ["ZZ"]},
        "explode_on_capture": {"radius": 1, "immune": ["ZZ"]},
        "drops": {"from_captures": True, "no_doubled": ["ZZ"]},
        "npcs": [{"piece": "ZZ", "route": ["c3", "c4"]}],
        "strike": {"effect": "convert", "chain": [{"from": "ZZ", "to": "R"}]},
        "contacts": {"links": [{"mover": "ZZ", "onto": "R", "mover_to": "K"}]},
    }
    for key, value in naming.items():
        assert key in _PIECE_POSITIONS, key
        # And the validator agrees the key is a place a piece id goes.
        problems = validate_spec(changed(**{key: value}))
        assert any("ZZ" in problem for problem in problems), (key, problems)


# ----------------------------------------------------------------- spelling is free


def test_renaming_every_piece_costs_nothing() -> None:
    assert units(BASE, rename_everything(copy.deepcopy(BASE))) == 0


def test_declaring_the_pieces_in_another_order_costs_nothing() -> None:
    assert units(BASE, reorder_pieces(copy.deepcopy(BASE))) == 0


def test_writing_a_symmetry_out_by_hand_costs_nothing() -> None:
    spec = copy.deepcopy(BASE)
    piece(spec, "K")["moves"] = [
        {"kind": "leap", "vectors": [[1, 0], [0, 1], [-1, 0], [0, -1],
                                     [1, 1], [-1, 1], [1, -1], [-1, -1]]}
    ]
    assert units(BASE, spec) == 0


def test_splitting_one_move_into_two_that_say_the_same_costs_nothing() -> None:
    spec = copy.deepcopy(BASE)
    spec["pieces"][2]["moves"] = [
        {"kind": "slide", "vectors": [[1, 0], [-1, 0]]},
        {"kind": "slide", "vectors": [[0, 1], [0, -1]]},
    ]
    assert units(BASE, spec) == 0


def test_reordering_a_placement_list_costs_nothing() -> None:
    spec = copy.deepcopy(BASE)
    for placement in spec["setup"]["placements"]:
        placement["squares"] = list(reversed(placement["squares"]))
    assert units(BASE, spec) == 0


def test_writing_a_default_costs_nothing() -> None:
    assert units(BASE, changed(stalemate="draw", max_plies=400)) == 0


def test_the_schema_stamp_is_not_a_rule() -> None:
    spec = copy.deepcopy(BASE)
    spec["schema_version"] = 12
    spec["name"] = "something else"
    spec["description"] = "a sentence nobody plays"
    assert units(BASE, spec) == 0


def test_a_spelling_twin_shares_its_identity_key() -> None:
    assert identity_key(BASE) == identity_key(rename_everything(copy.deepcopy(BASE)))
    assert identity_key(BASE) != identity_key(changed(stalemate="loss"))


_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _relettered(spec: dict) -> dict | None:
    """``spec`` with every piece id replaced by a letter it does not use, everywhere it
    appears; ``None`` where there are too few letters or the rewrite is not a valid spec."""
    ids = [str(entry["id"]) for entry in spec.get("pieces", [])]
    spare = [c.upper() for c in _LETTERS if c.upper() not in ids and c not in ids]
    if len(spare) < len(ids):
        return None
    twin = _rewrite(copy.deepcopy(spec), dict(zip(ids, spare, strict=False)))
    return None if validate_spec(twin) else twin


# One test per committed spec (Q749): the walk was one test of 6 s in the gate, the specs
# that rotate costing two compiles each in `units` and `similarity`.
@pytest.mark.parametrize(
    ("stem", "spec"), [pytest.param(stem, spec, id=stem) for stem, spec in corpus_specs()]
)
def test_no_committed_spec_moves_under_a_relettering(stem: str, spec: dict) -> None:
    """The three spelling nulls over the whole corpus, which is where they were measured.

    A re-lettering is the strongest of them: every piece id replaced at once, everywhere
    it appears. `docs/reference/game-graph-probe.md` ran these by hand beside the probe
    and they are the first thing this module owes.
    """
    moved: list[str] = []
    for make in (rename_everything, reorder_pieces):
        if units(spec, make(copy.deepcopy(spec))):
            moved.append(f"{stem} under {make.__name__}")
    twin = _relettered(spec)
    if twin is not None and (units(spec, twin) or similarity(spec, twin).index):
        moved.append(f"{stem} under a re-lettering")
    assert not moved


def test_the_relettering_reaches_most_of_the_corpus() -> None:
    """The walk above re-letters a spec only where letters suffice and the rewrite is
    valid; it must still reach over fifty of them, or it has stopped asking."""
    assert sum(_relettered(spec) is not None for _, spec in corpus_specs()) > 50


def _rewrite(value, mapping: dict[str, str]):
    if isinstance(value, dict):
        return {mapping.get(k, k): _rewrite(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, mapping) for item in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


# ------------------------------------------------------------- one rule costs one rule


@pytest.mark.parametrize(
    ("rules", "unit", "kind"),
    [
        ({"stalemate": "loss"}, "rules.stalemate", "ending"),
        ({"must_capture": True}, "rules.must_capture", "capture"),
        ({"en_passant": None}, "rules.en_passant", "capture"),
        ({"max_plies": 120}, "rules.max_plies", "clock"),
        ({"drops": {"from_captures": True}}, "rules.drops", "hand"),
        ({"turn": {"moves": 2}}, "rules.turn", "turn"),
    ],
)
def test_one_rule_key_is_one_rule(rules: dict, unit: str, kind: str) -> None:
    relation = relate(BASE, changed(**rules))
    assert relation.units == 1
    assert relation.differences[0].unit == unit
    assert relation.differences[0].kind == kind


def test_a_wrapped_edge_is_one_board_rule() -> None:
    relation = relate(BASE, changed(wrap={"files": "both"}))
    assert relation.units == 1
    assert relation.differences[0].unit == "board.wrap"
    assert relation.differences[0].kind == "board"


def test_one_pieces_movement_is_one_rule() -> None:
    spec = copy.deepcopy(BASE)
    piece(spec, "R")["moves"][0]["range"] = 3
    relation = relate(BASE, spec)
    assert relation.units == 1
    assert relation.differences[0].unit == "piece.moves"


def test_a_reshuffled_setup_is_one_rule_that_says_how_big_it_is() -> None:
    spec = copy.deepcopy(BASE)
    spec["setup"]["placements"][1]["squares"] = ["b1", "e1"]
    relation = relate(BASE, spec)
    assert relation.units == 1
    assert relation.differences[0].unit == "setup.cells"
    assert relation.differences[0].size == 8  # two squares, two armies, before and after


def test_who_may_take_whom_is_one_rule_however_many_pieces_declare_it() -> None:
    spec = copy.deepcopy(BASE)
    for entry in spec["pieces"]:
        entry["captures"] = ["W"]
    relation = relate(BASE, spec)
    assert relation.units == 1
    assert relation.differences[0].unit == "piece.captures"
    assert relation.differences[0].kind == "capture"
    assert relation.differences[0].size == 3


def test_a_replaced_piece_is_one_rule_whatever_it_is_called() -> None:
    """The `almost` shape: one piece swapped for another under a different letter.

    Twelve JSON leaves and one rule. Pieces pair by movement before they pair by letter,
    so the two towers pair and the new piece is the difference.
    """
    spec = copy.deepcopy(BASE)
    spec["pieces"][2] = {
        "id": "C",
        "name": "rider",
        "value": 3,
        "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}],
    }
    for placement in spec["setup"]["placements"]:
        if placement["piece"] == "R":
            placement["piece"] = "C"
    spec["rules"]["promotion"] = {"piece": "W", "rank": 5, "to": ["C"]}
    relation = relate(BASE, spec)
    assert relation.units == 1
    assert relation.differences[0].unit == "piece.moves"


def test_the_count_does_not_depend_on_which_spec_is_asked_about_first() -> None:
    spec = changed(must_capture=True)
    assert units(BASE, spec) == units(spec, BASE) == 1


def test_the_same_pair_reads_the_same_twice() -> None:
    spec = changed(stalemate="loss")
    assert relate(BASE, spec).as_dict() == relate(BASE, spec).as_dict()


# --------------------------------------------------------- a consequence folds into its cause


def test_a_new_piece_takes_the_squares_it_displaced() -> None:
    """A split: some of one piece's copies become a piece of their own.

    One edit. Without the fold it reads as the new piece plus a changed setup, because
    the squares it stands on used to hold something else.
    """
    spec = copy.deepcopy(BASE)
    spec["pieces"].append({
        "id": "B",
        "name": "runner",
        "value": 3,
        "moves": [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}],
    })
    walkers = next(p for p in spec["setup"]["placements"] if p["piece"] == "W")
    walkers["squares"] = ["a2", "b2", "c2"]
    spec["setup"]["placements"].append({"piece": "B", "squares": ["d2", "e2", "f2"]})
    relation = relate(BASE, spec)
    assert relation.units == 1
    assert relation.differences[0].unit == "piece.added"


def test_a_removed_piece_takes_the_rules_that_name_it() -> None:
    """Castling names its rook by id, so removing that piece takes the rule with it.

    `transform.drop_piece` does exactly this, because a rule pointing at nothing is a
    spec the validator refuses -- so the removal and the rule are one edit.
    """
    spec = copy.deepcopy(BASE)
    spec["rules"]["castling"] = {"king": "K", "rook": "R"}
    stripped = copy.deepcopy(spec)
    stripped["pieces"] = [p for p in stripped["pieces"] if p["id"] != "R"]
    stripped["setup"]["placements"] = [
        p for p in stripped["setup"]["placements"] if p["piece"] != "R"
    ]
    stripped["rules"].pop("castling")
    stripped["rules"]["promotion"] = {"piece": "W", "rank": 5, "to": ["K"]}
    relation = relate(spec, stripped)
    assert [difference.unit for difference in relation.differences] == [
        "piece.removed", "rules.promotion"
    ]


def test_a_piece_that_arrives_with_its_own_rule_is_one_edit() -> None:
    """A stone reaches the board only through a hand, so the hand comes with it."""
    spec = copy.deepcopy(BASE)
    spec["schema_version"] = 10
    spec["pieces"].append({"id": "S", "name": "stone", "moves": []})
    spec["rules"]["drops"] = {"from_captures": False, "stock": {"S": -1}}
    relation = relate(BASE, spec)
    assert relation.units == 1
    assert relation.differences[0].unit == "piece.added"


def test_a_promotion_rank_follows_the_board() -> None:
    """A resize has to settle where the walk ends, and that is part of the resize."""
    spec = copy.deepcopy(BASE)
    spec["board"] = {"width": 6, "height": 8}
    spec["rules"]["promotion"] = {"piece": "W", "rank": 7, "to": ["R"]}
    relation = relate(BASE, spec)
    assert relation.units == 1
    assert relation.differences[0].unit == "board.size"


def test_a_promotion_moved_on_the_same_board_is_still_a_rule() -> None:
    spec = changed(promotion={"piece": "W", "rank": 3, "to": ["R"]})
    relation = relate(BASE, spec)
    assert relation.units == 1
    assert relation.differences[0].unit == "rules.promotion"


def test_en_passant_goes_with_the_double_step() -> None:
    """A walker that leaps cannot be caught in passing, so the rule is not a second edit.

    It is compared where both games could bring it into play, which is what keeps the
    plain case below at one rule.
    """
    spec = copy.deepcopy(BASE)
    piece(spec, "W")["moves"][0] = {"kind": "leap", "vectors": [[0, 1]], "mode": "move"}
    spec["rules"].pop("en_passant")
    relation = relate(BASE, spec)
    assert relation.units == 1
    assert relation.differences[0].unit == "piece.moves"


def test_switching_en_passant_off_on_its_own_is_a_rule() -> None:
    assert relate(BASE, changed(en_passant=None)).differences[0].unit == "rules.en_passant"


def test_a_piece_priced_differently_is_a_rule_of_its_own() -> None:
    spec = copy.deepcopy(BASE)
    piece(spec, "R")["value"] = 9
    relation = relate(BASE, spec)
    assert relation.units == 1
    assert relation.differences[0].unit == "piece.value"
    assert relation.differences[0].kind == "valuation"


def test_a_price_that_follows_a_changed_piece_is_not_a_second_rule() -> None:
    spec = copy.deepcopy(BASE)
    piece(spec, "R")["value"] = 9
    piece(spec, "R")["moves"][0]["range"] = 2
    relation = relate(BASE, spec)
    assert relation.units == 1
    assert relation.differences[0].unit == "piece.moves"


# ------------------------------------------------------------------- what a link says


def test_every_difference_carries_a_kind_a_unit_and_a_sentence() -> None:
    relation = relate(BASE, changed(must_capture=True, max_plies=90))
    assert relation.units == 2
    for difference in relation.differences:
        assert difference.kind in KINDS
        assert difference.unit
        assert difference.sentence
    assert "; " in relation.sentence


def test_two_specs_that_differ_in_nothing_have_no_link_to_draw() -> None:
    assert relate(BASE, copy.deepcopy(BASE)).differences == ()
    assert relate_forms(normal_form(BASE), normal_form(BASE)).units == 0


class TestAMatchedPairIsNamedFromBothSides:
    """F9 of `docs/reference/first-visit-study.md`, which Q595 reproduced and Q861 fixed.

    Pieces are matched by what they do, so a pair can carry two names. A sentence is
    about the second game, so it names the second game's piece as *its* and the first's
    beside it; naming one alone named the page's own piece in a row about the other
    game. Each kind is asked from both sides of the pair, and with the names the same.
    """

    @staticmethod
    def edited(edit, renamed: bool) -> dict:
        spec = copy.deepcopy(BASE)
        if renamed:
            piece(spec, "R")["name"] = "castle"
        edit(piece(spec, "R"))
        return spec

    @staticmethod
    def said(first: dict, second: dict, unit: str) -> str:
        relation = relate(first, second)
        found = [one for one in relation.differences if one.unit == unit]
        assert len(found) == 1, relation.differences
        return found[0].sentence

    EDITS = {
        "piece.moves": lambda entry: entry["moves"][0].__setitem__("range", 2),
        "piece.royal": lambda entry: entry.__setitem__("royal", True),
        "piece.value": lambda entry: entry.__setitem__("value", 9),
        "piece.facing": lambda entry: entry.__setitem__("facing", {"axes": 4}),
        "piece.other": lambda entry: entry.__setitem__("region", ["a1", "b1", "c1"]),
    }

    DIFFERING = {
        "piece.moves": (
            "its castle moves differently from the tower",
            "its tower moves differently from the castle",
        ),
        "piece.royal": (
            "its castle is royal where the tower is not",
            "its tower is not royal where the castle is",
        ),
        "piece.value": (
            "its castle is priced differently from the tower",
            "its tower is priced differently from the castle",
        ),
        "piece.facing": (
            "its castle turns differently from the tower",
            "its tower turns differently from the castle",
        ),
        "piece.other": (
            "its castle may stand in other places than the tower",
            "its tower may stand in other places than the castle",
        ),
    }

    SAME = {
        "piece.moves": "the tower moves differently",
        "piece.royal": "whether the tower is royal",
        "piece.value": "the tower is priced differently",
        "piece.facing": "whether the tower turns",
        "piece.other": "where the tower may stand",
    }

    def test_every_per_piece_kind_is_covered(self) -> None:
        assert set(self.EDITS) == set(self.DIFFERING) == set(self.SAME)

    @pytest.mark.parametrize("unit", sorted(DIFFERING))
    def test_differing_names_are_both_said_from_either_side(self, unit: str) -> None:
        draft = self.edited(self.EDITS[unit], renamed=True)
        under_draft, under_base = self.DIFFERING[unit]
        assert self.said(BASE, draft, unit) == under_draft
        assert self.said(draft, BASE, unit) == under_base

    @pytest.mark.parametrize("unit", sorted(SAME))
    def test_one_name_is_said_once_from_either_side(self, unit: str) -> None:
        draft = self.edited(self.EDITS[unit], renamed=False)
        assert self.said(BASE, draft, unit) == self.SAME[unit]
        assert self.said(draft, BASE, unit) == self.SAME[unit]

    def test_a_name_differing_only_in_case_is_one_name(self) -> None:
        draft = self.edited(self.EDITS["piece.value"], renamed=False)
        piece(draft, "R")["name"] = "Tower"
        assert self.said(BASE, draft, "piece.value") == self.SAME["piece.value"]

    def test_a_renamed_spelling_does_not_borrow_the_other_spellings_words(self) -> None:
        """Renaming moves no identity key, so the pair cache is keyed on the names too."""
        renamed = self.edited(self.EDITS["piece.value"], renamed=True)
        same = self.edited(self.EDITS["piece.value"], renamed=False)
        assert identity_key(renamed) == identity_key(same)
        assert self.said(BASE, renamed, "piece.value") == self.DIFFERING["piece.value"][0]
        assert self.said(BASE, same, "piece.value") == self.SAME["piece.value"]
        assert self.said(BASE, renamed, "piece.value") == self.DIFFERING["piece.value"][0]

    def test_the_reproduction_reads_right_on_both_pages(self) -> None:
        """Q595's case on its own terms: a knight renamed and made a (2,2) leaper."""
        root = Path(__file__).resolve().parents[1]
        chess = json.loads((root / "rulesets" / "chess.json").read_text(encoding="utf-8"))
        draft = copy.deepcopy(chess)
        knight = next(entry for entry in draft["pieces"] if entry["id"] == "N")
        original = knight["name"]
        knight["name"] = f"diagonal {original}"
        knight["moves"] = [{"kind": "leap", "vectors": [[2, 2]], "symmetry": "rotate4"}]
        # The draft's page lists chess under it: the sentences are about chess.
        assert self.said(draft, chess, "piece.moves") == (
            f"its {original} moves differently from the diagonal {original}"
        )
        # Chess's page lists the draft under it: the sentences are about the draft.
        assert self.said(chess, draft, "piece.moves") == (
            f"its diagonal {original} moves differently from the {original}"
        )


# --------------------------------------------------------------- the library's own hub


@pytest.mark.slow
def test_the_committed_corpus_has_one_hub() -> None:
    """`docs/archive/plan-game-graph.md`'s acceptance, named by game because the plan names it.

    Chess is where the library's variants hang from, and 26 of the other 83 games differ
    from it in exactly one rule. The probe measured 21 before the folds; the
    twenty-second was `wolves-immune`, which was chess plus a wild piece and the one
    pair the probe's operator sweep reached and its diff did not; since Q876 retired it,
    `pitchfork` is chess plus wild pieces, one rule from it.

    The pin read 22 of 70 from Q326 (2026-09-18). It reads 26 of 83 at Q878 (2026-10-13),
    re-derived by running today's `relate` over the library as of the old pin (which still
    gives 22, so `relate` moved nothing): twenty of the twenty-two stayed, two left, six joined. Left (Q868, Q876): `hierarchy-fall-swap`, retired as a near copy, and
    `wolves-immune`, retired for `pitchfork`, which is the one that took its place. Joined
    (new games, each chess plus one rule, so each belongs): `rifle`, `circe`, `andernach` and
    `kamikaze` (Q540's capture-rule showcases, 2026-10-09), `pawn-knockdown` (Q787,
    2026-10-09) and `pitchfork` (Q876). No neighbour is there that should not be.
    """
    specs = corpus_specs()
    forms = {stem: normal_form(spec) for stem, spec in specs}
    mine = forms["chess"]
    near = sorted(
        stem for stem, form in forms.items()
        if stem != "chess" and relate_forms(mine, form).units == 1
    )
    assert len(near) == 26, near
    assert "pitchfork" in near
    # Nothing else in the library comes close to that degree.
    others = max(
        sum(
            1 for other, form in forms.items()
            if other != stem and relate_forms(forms[stem], form).units == 1
        )
        for stem in forms
        if stem != "chess"
    )
    assert others < 10


# ---------------------------------------------------------------- the similarity index
#
# `units` counts steps; the index ranks. The tests split the same way the index does:
# what one difference in each tier costs, checked against specs written here, and then
# the pre-registered pairs of `docs/archive/plan-game-graph.md` §9.4, which are library games
# because a pre-registration is only worth anything against the corpus it was written
# for. The bands and the claims below are copied from §9.4 and were written before any
# tier count had been read.

RULE = SIMILARITY_RATIO**3
MECHANISM = SIMILARITY_RATIO**2
PIECE = SIMILARITY_RATIO


def standing(pid: str, squares: list[str]) -> dict:
    """`BASE` with one piece standing somewhere else."""
    spec = copy.deepcopy(BASE)
    for placement in spec["setup"]["placements"]:
        if placement["piece"] == pid:
            placement["squares"] = squares
    return spec


def test_a_game_is_not_apart_from_itself() -> None:
    reading = similarity(BASE, copy.deepcopy(BASE))
    assert reading.index == 0
    assert reading.counts == dict.fromkeys(reading.counts, 0)


def test_two_spellings_of_one_game_are_not_apart() -> None:
    for make in (rename_everything, reorder_pieces):
        assert similarity(BASE, make(copy.deepcopy(BASE))).index == 0


def test_a_rule_only_one_game_involves_is_the_heaviest_difference() -> None:
    reading = similarity(BASE, changed(check_limit=3))
    assert reading.counts == {
        "rules": 1, "settings": 0, "mechanisms": 0, "pieces": 0, "squares": 0, "books": 0
    }
    assert reading.index == RULE
    assert reading.sentence == "1 rule"


def test_a_rule_both_games_involve_and_set_differently_is_half_of_one() -> None:
    first = changed(promotion={"piece": "W", "rank": 5, "to": ["R"]})
    second = changed(promotion={"piece": "W", "rank": 4, "to": ["R"]})
    reading = similarity(first, second)
    assert reading.counts["rules"] == 0
    assert reading.counts["settings"] == 1
    assert reading.index == RULE * SETTING_WEIGHT
    assert reading.sentence == "no rules, 1 rule set differently"


def test_a_movement_mechanism_outweighs_the_piece_it_makes() -> None:
    """A tier's difference shows in the tier below it, and both are counted.

    Giving one piece a move nobody had adds one mechanism to the game and makes that
    piece a different piece. The index charges the mechanism and the identity, and the
    gradient keeps the lower charge small beside the upper one.
    """
    spec = copy.deepcopy(BASE)
    piece(spec, "R")["moves"].append(
        {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}
    )
    reading = similarity(BASE, spec)
    assert reading.counts == {
        "rules": 0, "settings": 0, "mechanisms": 1, "pieces": 1, "squares": 0, "books": 0
    }
    assert reading.index == MECHANISM + PIECE
    assert reading.sentence == "no rules, 1 mechanism, 1 piece"


def test_what_a_piece_is_counts_whether_or_not_it_moves_differently() -> None:
    """Royalty is part of an identity, so a crowned piece is a different piece."""
    spec = copy.deepcopy(BASE)
    piece(spec, "R")["royal"] = True
    reading = similarity(BASE, spec)
    assert reading.counts["mechanisms"] == 0
    assert reading.counts["pieces"] == 1
    assert reading.index == PIECE


def test_a_piece_only_one_game_has_takes_its_squares_with_it() -> None:
    """Otherwise a changed piece would carry every square it stands on into tier 4."""
    spec = copy.deepcopy(BASE)
    piece(spec, "W")["moves"][0]["range"] = 3
    reading = similarity(BASE, spec)
    assert reading.counts["squares"] == 0
    assert reading.counts["pieces"] == 1


def test_one_starting_square_is_the_lightest_difference() -> None:
    reading = similarity(BASE, standing("K", ["d1"]))
    assert reading.counts == {
        "rules": 0, "settings": 0, "mechanisms": 0, "pieces": 0, "squares": 4, "books": 0
    }
    # One square of each army empties and one fills.
    assert reading.index == 4
    assert reading.sentence == "no rules, 4 squares"


def test_a_price_is_weighed_beside_a_starting_square() -> None:
    spec = copy.deepcopy(BASE)
    piece(spec, "R")["value"] = 9
    reading = similarity(BASE, spec)
    assert reading.counts["books"] == 1
    assert reading.index == 1
    assert reading.sentence == "no rules, 1 clock or price"


def test_a_clock_is_not_a_rule_the_game_involves() -> None:
    """The three draw clocks are the harness's, so they stay out of tier 1."""
    spec = changed(max_plies=90)
    assert rules_involved(normal_form(spec)) == rules_involved(normal_form(BASE))
    reading = similarity(BASE, spec)
    assert reading.counts["rules"] == 0
    assert reading.counts["books"] == 1


def test_the_index_does_not_depend_on_which_spec_is_asked_about_first() -> None:
    others = [
        changed(check_limit=3),
        changed(max_plies=90),
        standing("K", ["d1"]),
        rename_everything(copy.deepcopy(BASE)),
    ]
    for spec in others:
        assert similarity(BASE, spec) == similarity(spec, BASE)


def test_a_compound_piece_adds_no_mechanism() -> None:
    """Tier 2 is what a game can do, not how its pieces are packaged."""
    spec = copy.deepcopy(BASE)
    spec["pieces"].append({
        "id": "Q",
        "name": "compound",
        "value": 9,
        "moves": [
            {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"},
            {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"},
        ],
    })
    assert mechanisms(normal_form(spec)) == mechanisms(normal_form(BASE))
    assert similarity(BASE, spec).counts["mechanisms"] == 0


def test_a_piece_identity_leaves_out_the_letter_the_name_and_the_price() -> None:
    spec = copy.deepcopy(BASE)
    piece(spec, "R")["name"] = "something else"
    piece(spec, "R")["value"] = 9
    assert piece_identity(normal_form(spec), "R") == piece_identity(normal_form(BASE), "R")


# ------------------------------------------------- the index over the committed corpus


def library() -> dict[str, dict]:
    return dict(corpus_specs())


# §9.4's bands, in the order a person who holds the note's priorities sorts them.
NEAR = [
    ("chess", "perpetual"),
    ("chess", "almost"),
    ("chess", "knightmate"),
    ("hierarchy-rise", "hierarchy-rise-stay"),
    ("rps-v2", "rps-v4"),
]
MIDDLE = [
    ("chess", "kingofthehill"),
    ("chess", "atomic"),
    ("chess", "crazyhouse"),
    ("chess", "berolina"),
    ("chess", "gardner"),
    ("chess", "capablanca"),
    ("shatranj", "makruk"),
]
FAR = [
    ("chess", "shogi"),
    ("chess", "xiangqi"),
    ("chess", "dull"),
    ("shogi", "rps-v1"),
    ("xiangqi", "dobutsu"),
    ("chess", "intransitive"),
]

# Each row claims the first pair is nearer than the second.
ORDERED = [
    (("chess", "almost"), ("chess", "berolina")),
    (("chess", "berolina"), ("chess", "kingofthehill")),
    (("shogi", "dobutsu"), ("shogi", "chess")),
    (("hierarchy-rise", "hierarchy-fall"), ("chess", "hierarchy-rise")),
    (("chess", "knightmate"), ("chess", "gardner")),
    (("crazyhouse", "shogi"), ("chess", "shogi")),
    (("courier", "shatranj"), ("courier", "chess")),
]

# §9.4's table: the six counts for every pair it lists, as
# (rules, settings, mechanisms, pieces, squares, books).
COUNTED: dict[tuple[str, str], tuple[int, ...]] = {
    ("chess", "perpetual"): (0, 0, 0, 0, 0, 1),
    ("chess", "almost"): (0, 0, 0, 1, 0, 0),
    ("chess", "knightmate"): (0, 0, 0, 2, 0, 0),
    ("courier", "shatranj"): (0, 0, 1, 3, 29, 1),
    ("hierarchy-rise", "hierarchy-rise-stay"): (0, 1, 0, 0, 0, 0),
    ("rps-v2", "rps-v4"): (0, 1, 0, 0, 0, 0),
    ("hierarchy-rise", "hierarchy-fall"): (0, 1, 0, 0, 0, 0),
    ("chess", "capablanca"): (0, 1, 0, 2, 17, 0),
    ("chess", "berolina"): (0, 0, 6, 1, 0, 0),
    ("chess", "kingofthehill"): (1, 0, 0, 0, 0, 0),
    ("chess", "atomic"): (1, 0, 0, 0, 0, 0),
    ("chess", "crazyhouse"): (1, 0, 0, 0, 0, 0),
    ("chess", "hierarchy-rise"): (1, 0, 0, 0, 0, 0),
    ("shatranj", "makruk"): (1, 1, 2, 1, 34, 1),
    ("chess", "gardner"): (2, 0, 1, 1, 7, 1),
    ("chess", "courier"): (3, 1, 2, 5, 11, 2),
    ("chess", "dull"): (4, 0, 6, 6, 1, 2),
    ("shogi", "dobutsu"): (4, 2, 4, 7, 25, 2),
    # Under r1 this row and the three below naming shogi or dobutsu counted one more square:
    # their `orientation: rotate` read as a setup difference. Their pieces and setups read
    # left and right alike, so under r2 (Q503c2) the compiler builds the same game
    # mirrored, and the rotation is spelling. Rotating `wolves` still counts: it is a rule.
    ("shogi", "crazyhouse"): (4, 2, 8, 7, 1, 3),
    ("chess", "xiangqi"): (5, 0, 11, 6, 3, 2),
    ("chess", "shogi"): (5, 1, 8, 7, 1, 3),
    ("shogi", "rps-v1"): (6, 0, 6, 10, 0, 3),
    ("xiangqi", "dobutsu"): (6, 0, 7, 7, 1, 1),
    # One more book than §9.4 counted since Q696 (2026-10-04): the pair was written when
    # both games drew after 100 capture-free plies; *Intransitive* now waits 200, so the
    # draw clock is a third book difference. The game changed, not `relate`.
    ("chess", "intransitive"): (7, 0, 6, 6, 1, 3),
}

# The same table's index column, which only means anything at the ratio it was taken at.
INDICES: dict[tuple[str, str], int] = {
    ("chess", "perpetual"): 1,
    ("chess", "almost"): 8,
    ("chess", "knightmate"): 16,
    ("courier", "shatranj"): 118,
    ("chess", "capablanca"): 289,
    ("chess", "berolina"): 392,
    ("chess", "kingofthehill"): 512,
    ("shatranj", "makruk"): 939,
    ("chess", "gardner"): 1104,
    ("chess", "courier"): 1973,
    ("chess", "dull"): 2483,
    ("shogi", "dobutsu"): 2899,
    ("shogi", "crazyhouse"): 3132,  # 3133 under r1, as the three below; see COUNTED
    ("chess", "xiangqi"): 3317,
    ("chess", "shogi"): 3388,  # 3389 under r1
    ("shogi", "rps-v1"): 3539,  # 3540 under r1
    ("xiangqi", "dobutsu"): 3578,  # 3579 under r1
    ("chess", "intransitive"): 4020,  # 4019 before Q696's draw clock; see COUNTED
}

_SHIPPED = (SIMILARITY_RATIO, SETTING_WEIGHT) == (8, 0.5)
_AT_THE_RATIO = pytest.mark.skipif(
    not _SHIPPED, reason="§9.4 is a check at the ratio it was pre-registered on"
)


def test_the_counts_are_what_the_pre_registered_table_recorded() -> None:
    """§9.4's counts, which are the index's reading and carry no weighting.

    The pairs are library games because the check was written against the library, and
    the table was fixed before any of it was computed. A number here that moves is either
    a fault in `relate` or a change to what §9.2 says a tier is.
    """
    specs = library()
    for (first, second), expected in COUNTED.items():
        reading = similarity(specs[first], specs[second])
        assert tuple(reading.counts.values()) == expected, f"{first} / {second}"


@_AT_THE_RATIO
def test_the_index_is_what_the_pre_registered_table_recorded() -> None:
    specs = library()
    for (first, second), expected in INDICES.items():
        assert similarity(specs[first], specs[second]).index == expected, f"{first}/{second}"


@_AT_THE_RATIO
def test_the_bands_and_the_ordered_claims_pass_their_marks() -> None:
    """§9.4's three pass marks, which are what the ratio was chosen against.

    Every near pair below every far pair, 76 of the 84 comparisons across neighbouring
    bands, and six of the seven claims. The shipped ratio passes all three outright, so
    the assertions are the marks and not the run.
    """
    specs = library()

    def index(pair: tuple[str, str]) -> float:
        return similarity(specs[pair[0]], specs[pair[1]]).index

    def held(low: list[tuple[str, str]], high: list[tuple[str, str]]) -> int:
        return sum(1 for a in low for b in high if index(a) < index(b))

    assert held(NEAR, FAR) == len(NEAR) * len(FAR)
    neighbouring = held(NEAR, MIDDLE) + held(MIDDLE, FAR)
    assert neighbouring >= 76, neighbouring
    assert sum(1 for a, b in ORDERED if index(a) < index(b)) >= 6


# One test per committed spec (Q749): the walk was one test of 2.6 s in the gate.
@pytest.mark.parametrize(
    ("stem", "spec"), [pytest.param(stem, spec, id=stem) for stem, spec in corpus_specs()]
)
def test_no_committed_spec_is_apart_from_its_own_respelling(stem: str, spec: dict) -> None:
    """The spelling nulls again, on the index rather than on the count.

    A respelling is the one thing the index must never see, and it is where a tier built
    out of sets rather than out of `relate`'s differences could go wrong on its own.
    """
    for make in (rename_everything, reorder_pieces):
        assert similarity(spec, make(copy.deepcopy(spec))).index == 0, stem


def test_the_index_module_names_no_piece_and_no_game() -> None:
    """Rule 5, over the code rather than over the prose that explains it.

    The comments reach for examples from the library, as the rest of the module does.
    What may not happen is a name reaching the arithmetic, so this reads the module with
    its docstrings and comments stripped and looks for every piece name and every
    ruleset stem the corpus has.
    """
    source = (Path(__file__).resolve().parents[1] / "chess_variant_sandbox/spec/relate.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            body[0] = ast.Pass()
    code = ast.unparse(tree).lower()

    # `rules.castling` names its two seats in the schema itself, and the corpus has
    # pieces called that. A schema key is not a piece name.
    schema_words = {"king", "rook"}
    named = set()
    for stem, spec in corpus_specs():
        named.add(stem)
        named.update(
            str(entry["name"]).lower() for entry in spec.get("pieces", []) if entry.get("name")
        )
    found = sorted(
        word
        for word in named - schema_words
        if re.search(rf"\b{re.escape(word)}\b", code)
    )
    assert not found, found


# ------------------------------------------- the orientation question, once and a half
#
# Q866: `_orientation_played` compiled a rotating spec and its restatement whole, and now
# asks `compile.compile_two_readings`, which builds the first player's frames once and
# stops at the first second-player square that differs. The two whole compiles are kept
# here as the oracle the shared one is held to, over every committed spec that rotates.


def _two_whole_compiles(spec: dict) -> str:
    """`_orientation_played` as it was before Q866: both specs compiled whole."""
    from chess_variant_sandbox.engine.bridge import canonical_tables
    from chess_variant_sandbox.engine.compile import compile_spec

    board = spec["board"]
    cells = _setup_cells(spec, board["width"], board["height"], "rotate")
    try:
        same = _played(canonical_tables(compile_spec(spec))) == _played(
            canonical_tables(compile_spec(_restated(spec, "rotate", cells)))
        )
    except Exception:  # noqa: BLE001 -- as the original: any failure keeps it as written
        return "rotate"
    return "mirror" if same else "rotate"


def _question(spec: dict) -> str:
    board = spec["board"]
    cells = _setup_cells(spec, board["width"], board["height"], "rotate")
    return _orientation_played(spec, "rotate", cells)


_ROTATING = [
    pytest.param(
        stem, spec, id=stem, marks=() if stem == "dobutsu" else pytest.mark.slow
    )
    for stem, spec in corpus_specs()
    if (spec.get("setup") or {}).get("orientation") == "rotate"
]


def _one_sided(spec: dict) -> dict:
    """``spec`` with its first piece stepping forward and to the right only, which the
    two orientations read differently for the second player: a rule, not a spelling."""
    twin = copy.deepcopy(spec)
    twin["pieces"][0]["moves"] = [{"kind": "leap", "vectors": [[1, 1]], "mode": "both"}]
    return twin


def test_the_rotating_corpus_is_what_the_shared_compile_is_held_over() -> None:
    assert len(_ROTATING) >= 5, "the oracle below would hold the shared compile to nothing"


@pytest.mark.parametrize(("stem", "spec"), _ROTATING)
def test_the_shared_compile_answers_as_the_two_whole_ones_did(stem: str, spec: dict) -> None:
    from chess_variant_sandbox.engine.bridge import canonical_tables
    from chess_variant_sandbox.engine.compile import compile_spec, compile_two_readings

    assert _question(spec) == _two_whole_compiles(spec), stem
    board = spec["board"]
    restated = _restated(
        spec, "rotate", _setup_cells(spec, board["width"], board["height"], "rotate")
    )
    try:
        compiled, stated = compile_two_readings(spec, restated)
    except SpecError:
        # A restatement the validator refuses (a hex lattice must rotate) is refused by
        # both, before any frame is built, and the question keeps "rotate".
        with pytest.raises(SpecError):
            compile_spec(restated)
        return
    assert canonical_tables(compiled) == canonical_tables(compile_spec(spec)), stem
    if stated is not None:
        # Zobrist keys included: a restated compile that shares is `compile_spec`'s.
        assert canonical_tables(stated) == canonical_tables(compile_spec(restated)), stem


def test_a_one_sided_piece_is_answered_as_the_two_whole_compiles_answer_it() -> None:
    spec = _one_sided(library()["dobutsu"])
    assert validate_spec(spec) == []
    assert _question(spec) == _two_whole_compiles(spec) == "rotate"


def _square_builds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts `_compile_square_full` calls: the compile's unit of work, once per frame,
    type and square."""
    from chess_variant_sandbox.engine import compile as compiler

    calls = [0]
    real = compiler._compile_square_full

    def counted(*args: object, **kwargs: object) -> object:
        calls[0] += 1
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(compiler, "_compile_square_full", counted)
    return calls


@pytest.mark.fresh_compile
def test_the_question_builds_the_first_players_frames_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rotation that is spelling: every second-player square is built under both
    readings and compared, and the first player's are built once, so the question costs
    one and a half compiles where it cost two."""
    from chess_variant_sandbox.engine.compile import compile_spec

    spec = library()["dobutsu"]
    calls = _square_builds(monkeypatch)
    compile_spec(spec)
    whole = calls[0]
    calls[0] = 0
    assert _question(spec) == "mirror"
    assert calls[0] * 2 == whole * 3


@pytest.mark.fresh_compile
def test_the_question_stops_at_the_first_square_that_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rotation that is a rule: the restated compile ends inside the second player's
    first piece, at the first square whose entries the two readings disagree on."""
    from chess_variant_sandbox.engine.compile import compile_spec

    spec = _one_sided(library()["dobutsu"])
    calls = _square_builds(monkeypatch)
    compile_spec(spec)
    whole = calls[0]
    calls[0] = 0
    assert _question(spec) == "rotate"
    board = spec["board"]
    assert whole < calls[0] <= whole + board["width"] * board["height"]
