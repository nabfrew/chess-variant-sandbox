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
    _reading,
    _restated,
    _setup_cells,
    anchored,
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
from chess_variant_sandbox.spec.schema import _RULES_KEYS, SpecError, format_square, parse_square
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
    # pieces called that; `pieces[].spy` (schema 27) is a key, and the showcases' spy is
    # named for it. A schema key is not a piece name.
    # `rules.duck` (generation 42) is a key too, and the shepherds duck's duck is named for it.
    schema_words = {"king", "rook", "spy", "duck"}
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


# ------------------------------------------------- what the module promises beyond a count
#
# Q738: the survivors of a mutation sweep of `relate.py`. Each test below names the claim
# it holds down; none reads a constant of the module to compare it with itself.


def _extra(spec: dict, pid: str, moves: list[dict], **fields) -> dict:
    """`spec` with one more piece that is placed nowhere (a promotion target only)."""
    out = copy.deepcopy(spec)
    out["pieces"].append({"id": pid, "name": f"extra {pid}", "moves": moves, **fields})
    return out


def _promoting_to(spec: dict, *targets: str) -> dict:
    out = copy.deepcopy(spec)
    out["rules"]["promotion"] = {"piece": "W", "rank": 5, "to": ["R", *targets]}
    return out


def _units_of(first: dict, second: dict) -> list[str]:
    return [difference.unit for difference in relate(first, second).differences]


@pytest.mark.parametrize(
    "rules",
    [
        {"win": "checkmate"},
        {"stalemate": "draw"},
        {"max_plies": 400},
        {"repetition_draw": 3},
        {"no_progress_draw": 100},
    ],
)
def test_writing_any_of_the_five_defaults_costs_nothing(rules: dict) -> None:
    assert units(BASE, changed(**rules)) == 0


def test_a_value_that_is_not_the_default_is_a_rule() -> None:
    for rules in ({"repetition_draw": 2}, {"repetition_draw": 4}, {"no_progress_draw": 99}):
        assert _units_of(BASE, changed(**rules)) == [f"rules.{next(iter(rules))}"]


def test_the_values_the_comparison_hands_back_cannot_be_changed_in_place() -> None:
    """Pair results, normal forms and readings are cached and shared between callers."""
    from dataclasses import FrozenInstanceError

    relation = relate(BASE, changed(stalemate="loss"))
    held = [
        (relation.differences[0], "kind"),
        (relation, "differences"),
        (normal_form(BASE), "size"),
        (similarity(BASE, changed(check_limit=3)), "rules"),
        (_reading(normal_form(BASE)), "rules"),
    ]
    for value, attribute in held:
        with pytest.raises(FrozenInstanceError):
            setattr(value, attribute, None)


def test_the_comparison_values_are_hashable_where_they_are_declared_so() -> None:
    relation = relate(BASE, changed(stalemate="loss"))
    assert len({relation, relate(BASE, changed(stalemate="loss"))}) == 1
    assert len({relation.differences[0]}) == 1
    reading = similarity(BASE, changed(check_limit=3))
    assert hash(reading) == hash(similarity(BASE, changed(check_limit=3)))
    # A normal form is hashed by identity: it holds dicts, and a field-wise hash would raise.
    assert len({normal_form(BASE), normal_form(BASE)}) == 1
    assert {normal_form(BASE): 1}[normal_form(BASE)] == 1


def _reordered(value):
    """Every dict in `value` rebuilt with its keys in the opposite order."""
    if isinstance(value, dict):
        return {key: _reordered(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reordered(item) for item in value]
    return value


def test_a_normal_form_is_memoised_on_what_the_spec_says_not_on_how_it_is_ordered() -> None:
    spec = copy.deepcopy(BASE)
    assert list(_reordered(spec)) != list(spec)
    assert normal_form(spec) is normal_form(_reordered(spec))


def test_an_identity_key_is_sixteen_hex_digits() -> None:
    key = identity_key(BASE)
    assert len(key) == 16
    assert int(key, 16) >= 0
    assert identity_key(changed(stalemate="loss")) != key


def _rotated_with_a_wall() -> dict:
    spec = copy.deepcopy(BASE)
    spec["schema_version"] = 6
    spec["setup"]["orientation"] = "rotate"
    spec["rules"]["obstacles"] = {"squares": ["b3"], "mirror": True}
    return spec


def test_a_rotated_army_of_pieces_that_read_left_and_right_alike_is_mirrored_play() -> None:
    """With a wall in play the restated spec has to carry the wall square by square, on
    this board's width and not the default one, and not mirrored a second time."""
    spec = _rotated_with_a_wall()
    assert validate_spec(spec) == []
    assert normal_form(spec).orientation == "mirror"
    assert _orientation_played(spec, "rotate", _setup_cells(spec, 6, 6, "rotate")) == "mirror"


def test_a_rotated_wall_on_a_non_square_board_is_still_spelling() -> None:
    spec = _rotated_with_a_wall()
    spec["board"] = {"width": 5, "height": 7}
    spec["rules"]["promotion"] = {"piece": "W", "rank": 6, "to": ["R"]}
    spec["setup"]["placements"] = [
        {"piece": "K", "squares": ["c1"]},
        {"piece": "R", "squares": ["a1", "e1"]},
        {"piece": "W", "squares": ["a2", "b2", "c2", "d2", "e2"]},
    ]
    spec["rules"]["obstacles"] = {"squares": ["b4"], "mirror": True}
    assert validate_spec(spec) == []
    assert normal_form(spec).orientation == "mirror"


def test_a_wall_that_only_one_army_sees_keeps_the_rotation() -> None:
    spec = _rotated_with_a_wall()
    piece(spec, "W")["moves"][0] = {"kind": "slide", "vectors": [[1, 1]], "mode": "move"}
    assert normal_form(spec).orientation == "rotate"


def _extra_pieces_pairing_case(first: list, second: list, promotes: tuple[str, str]) -> tuple:
    """Two games that differ only in a pair of unplaced extras, and in which one the
    promotion names, so a pairing that crosses the extras costs one more rule."""
    one = BASE
    two = BASE
    for pid, moves, fields in first:
        one = _extra(one, pid, moves, **fields)
    for pid, moves, fields in second:
        two = _extra(two, pid, moves, **fields)
    return _promoting_to(one, promotes[0]), _promoting_to(two, promotes[1])


def _orth(kind: str = "slide", **fields) -> list[dict]:
    return [{"kind": kind, "vectors": [[1, 0]], "symmetry": "rotate4", **fields}]


def test_the_closest_pairing_goes_by_kind_and_direction() -> None:
    """A slider pairs with the slider that moves the same way, whatever the order declared.

    P (slide) and Q (leap) against V (leap) and U (slide), declared the other way round and
    differing from them in range and condition only: the promotion names P in one game and
    U in the other, so the pairing that crosses them costs a third rule.
    """
    first, second = _extra_pieces_pairing_case(
        [("P", _orth(range=2), {}), ("Q", _orth("leap"), {})],
        [("V", _orth("leap", condition={"unmoved": True}), {}), ("U", _orth(range=3), {})],
        ("P", "U"),
    )
    assert _units_of(first, second) == ["piece.moves", "piece.moves"]


def test_the_closest_pairing_prefers_the_larger_share_of_directions() -> None:
    """An orthogonal slider pairs with the orthogonal slider and not with the one that
    also goes diagonally, though that one is declared first (a share is the directions
    they have in common over all the directions either has, not a product of sizes)."""
    first, second = _extra_pieces_pairing_case(
        [("P", _orth(range=2), {}),
         ("Q", [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4",
                 "range": 2}], {})],
        [("U", [{"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4",
                 "range": 3}], {}),
         ("V", _orth(range=3), {})],
        ("P", "V"),
    )
    assert _units_of(first, second) == ["piece.moves", "piece.moves"]


def test_a_stone_pairs_with_a_stone_before_a_royal_stone() -> None:
    """Two pieces that move nowhere have everything in common; a stone and a mover do not."""
    first, second = _extra_pieces_pairing_case(
        [("P", [], {}), ("Q", _orth(range=2), {})],
        [("U", _orth(range=3), {}), ("V", [], {"royal": True})],
        ("P", "V"),
    )
    assert sorted(_units_of(first, second)) == ["piece.moves", "piece.royal"]


def test_writing_the_default_minimum_range_costs_nothing() -> None:
    spec = copy.deepcopy(BASE)
    piece(spec, "R")["moves"][0]["min_range"] = 1
    assert units(BASE, spec) == 0
    for other in (0, 2):
        piece(spec, "R")["moves"][0]["min_range"] = other
        assert _units_of(BASE, spec) == ["piece.moves"]


def test_a_move_that_needs_a_condition_differs_from_one_that_needs_its_opposite() -> None:
    first, second = copy.deepcopy(BASE), copy.deepcopy(BASE)
    piece(first, "R")["moves"][0]["condition"] = {"unmoved": True}
    piece(second, "R")["moves"][0]["condition"] = {"unmoved": False}
    assert _units_of(first, second) == ["piece.moves"]
    assert _units_of(BASE, first) == ["piece.moves"]


# ----------------------------------------------- a piece only one game has leaves a map


def _stocked(spec: dict, **drops) -> dict:
    out = copy.deepcopy(spec)
    out["schema_version"] = 20
    out["rules"]["drops"] = drops
    return out


def test_a_removed_piece_leaves_the_stock_it_was_listed_in() -> None:
    """The piece sorts before the one that stays, so a fold that stops at the first gone
    entry would lose the rest of the map."""
    mover = _orth(range=2)
    first = _stocked(_extra(BASE, "A", mover, value=2), stock={"A": 3, "R": 2})
    second = _stocked(BASE, stock={"R": 2})
    assert validate_spec(first) == [] and validate_spec(second) == []
    assert _units_of(first, second) == ["piece.removed"]
    assert _units_of(second, first) == ["piece.added"]


def test_a_removed_piece_leaves_the_list_that_named_only_it() -> None:
    """A list left empty by the removal leaves too, and the entries after it are still read."""
    first = _stocked(_extra(BASE, "X", _orth(range=2), value=2), no_doubled=["X"], stock={"R": 2})
    second = _stocked(BASE, stock={"R": 2})
    assert validate_spec(first) == []
    assert _units_of(first, second) == ["piece.removed"]
    assert _units_of(second, first) == ["piece.added"]


# ------------------------------------------------------------------ anchoring the armies


def test_an_army_that_is_only_the_second_players_is_still_anchored() -> None:
    cells = {(1, 2, 3, "X", "")}
    assert anchored(cells, 8, 8, "mirror") == {(1, 0, 0, "X", "")}
    assert anchored(set(), 8, 8, "mirror") == frozenset()


def test_each_army_is_slid_to_its_own_corner_by_the_squares_it_leaves_empty() -> None:
    cells = {(0, 3, 2, "X", ""), (0, 5, 4, "Y", ""), (1, 4, 6, "X", ""), (1, 2, 5, "Y", "")}
    # Mirror: the second army reads (x, h - 1 - y), then both are slid to (0, 0).
    assert anchored(cells, 8, 8, "mirror") == {
        (0, 0, 0, "X", ""), (0, 2, 2, "Y", ""),
        (1, 2, 0, "X", ""), (1, 0, 1, "Y", ""),
    }


def _moved_setup(spec: dict, files: int, ranks: int, width: int, height: int) -> dict:
    out = copy.deepcopy(spec)
    out["board"] = {"width": width, "height": height}
    for placement in out["setup"]["placements"]:
        squares = [parse_square(name, 6, 6) for name in placement["squares"]]
        placement["squares"] = [format_square(x + files, y + ranks) for x, y in squares]
    out["rules"]["promotion"] = {"piece": "W", "rank": height - 1, "to": ["R"]}
    return out


def test_a_setup_that_moves_with_the_board_is_not_a_setup_difference() -> None:
    """The same armies one file right and two ranks up on a bigger board: the board is
    the one thing that changed, however far from a1 the armies start."""
    spec = _moved_setup(BASE, 1, 2, 8, 10)
    assert validate_spec(spec) == []
    relation = relate(BASE, spec)
    assert [difference.unit for difference in relation.differences] == ["board.size"]
    assert relation.differences[0].size == 8 * 10 - 6 * 6


def test_the_size_of_a_resized_board_is_said_in_files_and_ranks() -> None:
    first, second = copy.deepcopy(BASE), copy.deepcopy(BASE)
    first["board"] = {"width": 6, "height": 8}
    second["board"] = {"width": 7, "height": 9}
    for spec, top in ((first, 7), (second, 8)):
        spec["rules"]["promotion"] = {"piece": "W", "rank": top, "to": ["R"]}
    relation = relate(first, second)
    assert [difference.unit for difference in relation.differences] == ["board.size"]
    assert relation.differences[0].sentence == "its board is 7x9 rather than 6x8"
    assert relation.differences[0].size == 15
    assert relate(second, first).differences[0].sentence == "its board is 6x8 rather than 7x9"
    assert relate(second, first).differences[0].size == 15


def test_a_game_with_no_promotion_has_no_en_passant_to_fire() -> None:
    assert normal_form(BASE).en_passant_fires is True
    assert normal_form(changed(promotion=None, en_passant=None)).en_passant_fires is False
    assert normal_form(changed(promotion=None)).en_passant_fires is False


# ------------------------------------------------------------- the piece matching, by stage


def _leaper(*vectors: list[int]) -> list[dict]:
    return [{"kind": "leap", "vectors": list(vectors), "symmetry": "rotate4"}]


KNIGHT = _leaper([2, 1])
DIAGONAL = [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}]
TWO_STEP = _leaper([2, 0])


def _game_with(*extras: tuple[str, list[dict], dict], **rules) -> dict:
    spec = changed(**rules)
    for pid, moves, fields in extras:
        spec = _extra(spec, pid, moves, **fields)
    return spec


def test_a_piece_that_keeps_its_letter_and_movement_is_paired_with_itself() -> None:
    """Two leapers of one shape, valued 5 and 3 and declared the other way round in the
    second game, are still the same two pieces: the price of each travels with its letter."""
    first = _game_with(
        ("X", KNIGHT, {}), ("Y", DIAGONAL, {}),
        ("P", TWO_STEP, {"value": 5}), ("Q", TWO_STEP, {"value": 3}),
    )
    second = _game_with(
        ("X", DIAGONAL, {}), ("T", KNIGHT, {}),
        ("Q", TWO_STEP, {"value": 3}), ("P", TWO_STEP, {"value": 5}),
        stalemate="loss",
    )
    assert _units_of(first, second) == ["rules.stalemate"]


def test_of_several_pieces_that_move_alike_the_first_one_declared_is_taken() -> None:
    """Letters differ, so only movement can pair them, and the first that fits is the
    one that is taken: P (5) takes U (5) and Q (3) takes V (3)."""
    first = _game_with(
        ("X", KNIGHT, {}), ("Y", DIAGONAL, {}),
        ("P", TWO_STEP, {"value": 5}), ("Q", TWO_STEP, {"value": 3}),
    )
    second = _game_with(
        ("X", DIAGONAL, {}), ("T", KNIGHT, {}),
        ("U", TWO_STEP, {"value": 5}), ("V", TWO_STEP, {"value": 3}),
        stalemate="loss",
    )
    assert _units_of(first, second) == ["rules.stalemate"]


def test_a_letter_decides_a_pairing_that_movement_cannot() -> None:
    """P and Q keep their letters and both change range, so no movement is shared; the
    letters pair them, and the promotion that names P is then the same rule in both."""
    first, second = _extra_pieces_pairing_case(
        [("P", _orth(range=1), {}), ("Q", _orth(range=2), {})],
        [("Q", _orth(range=4), {}), ("P", _orth(range=3), {})],
        ("P", "P"),
    )
    assert _units_of(first, second) == ["piece.moves", "piece.moves"]


# --------------------------------------------------------------------------- the caches


def test_the_pair_cache_holds_more_than_the_last_pair() -> None:
    pair_a = (BASE, changed(stalemate="loss"))
    pair_b = (BASE, changed(must_capture=True))
    held = relate(*pair_a)
    relate(*pair_b)
    assert relate(*pair_a) is held


def test_the_reading_cache_holds_more_than_the_last_form() -> None:
    first, second = normal_form(BASE), normal_form(changed(stalemate="loss"))
    held = _reading(first)
    _reading(second)
    assert _reading(first) is held


# ------------------------------------------------------------------ the facing of a piece


def _facing(axes: int, placed: str) -> dict:
    spec = copy.deepcopy(BASE)
    spec["schema_version"] = 20
    piece(spec, "W")["facing"] = {"axes": axes}
    for placement in spec["setup"]["placements"]:
        if placement["piece"] == "W":
            placement["facing"] = placed
    return spec


def test_a_facing_written_on_a_placement_goes_with_the_facing_rule_when_that_differs() -> None:
    first, second = _facing(4, "right"), _facing(2, "forward")
    assert validate_spec(first) == [] and validate_spec(second) == []
    assert _units_of(first, second) == ["piece.facing"]
    assert _units_of(second, first) == ["piece.facing"]


def test_a_facing_written_on_a_placement_is_a_setup_difference_when_the_rule_is_alone() -> None:
    first, second = _facing(4, "right"), _facing(4, "forward")
    assert _units_of(first, second) == ["setup.cells"]
    assert _units_of(second, first) == ["setup.cells"]


# ---------------------------------------------------------------- who may take whom, said


def _takers(*pids: str) -> dict:
    spec = copy.deepcopy(BASE)
    for entry in spec["pieces"]:
        if entry["id"] in pids:
            entry["captures"] = ["W"]
    return spec


def test_who_may_take_whom_counts_the_pieces_that_declare_it_in_the_singular_and_plural() -> None:
    one = relate(BASE, _takers("R")).differences[0]
    assert (one.sentence, one.size) == ("who may take whom, declared on 1 piece", 1)
    three = relate(BASE, _takers("K", "W", "R")).differences[0]
    assert (three.sentence, three.size) == ("who may take whom, declared on 3 pieces", 3)
    two = relate(BASE, _takers("K", "R")).differences[0]
    assert two.sentence == "who may take whom, declared on 2 pieces"


# ----------------------------------------------------------- a split read from the other side


def test_taking_back_a_split_is_one_piece_removed() -> None:
    split = copy.deepcopy(BASE)
    split["pieces"].append({
        "id": "B", "name": "runner", "value": 3,
        "moves": [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}],
    })
    walkers = next(p for p in split["setup"]["placements"] if p["piece"] == "W")
    walkers["squares"] = ["a2", "b2", "c2"]
    split["setup"]["placements"].append({"piece": "B", "squares": ["d2", "e2", "f2"]})
    assert _units_of(split, BASE) == ["piece.removed"]
    assert _units_of(BASE, split) == ["piece.added"]


def test_a_promotion_rank_that_follows_the_board_does_not_hide_the_next_rule() -> None:
    spec = copy.deepcopy(BASE)
    spec["board"] = {"width": 6, "height": 8}
    spec["rules"]["promotion"] = {"piece": "W", "rank": 7, "to": ["R"]}
    spec["rules"]["stalemate"] = "loss"
    assert _units_of(BASE, spec) == ["board.size", "rules.stalemate"]


# ------------------------------------------------------------ the index, in whole numbers


def test_the_tiers_weigh_what_the_design_note_says_they_weigh() -> None:
    """Eight squares to a piece, eight pieces to a mechanism, eight mechanisms to a rule,
    written out here as numbers so that the ratio is pinned and not read back."""
    assert SIMILARITY_RATIO == 8
    assert SETTING_WEIGHT == 0.5
    assert similarity(BASE, changed(check_limit=3)).index == 512
    moved = changed(promotion={"piece": "W", "rank": 4, "to": ["R"]})
    assert similarity(BASE, moved).index == 256
    spec = copy.deepcopy(BASE)
    piece(spec, "R")["moves"][0]["range"] = 3
    reading = similarity(BASE, spec)
    assert (reading.counts["mechanisms"], reading.counts["pieces"]) == (2, 1)
    assert reading.index == 2 * 64 + 8
    assert similarity(BASE, changed(max_plies=100)).index == 1
    assert similarity(BASE, standing("W", ["a2", "b2", "c2", "d2", "e2"])).index == 2


# ------------------------------------------------------------------ the rules a game involves


def _wrapped() -> dict:
    return changed(wrap={"files": "both"})


def _walled() -> dict:
    spec = copy.deepcopy(BASE)
    spec["schema_version"] = 6
    spec["rules"]["obstacles"] = {"squares": ["b3"]}
    return spec


def _hex() -> dict:
    spec = copy.deepcopy(BASE)
    spec["schema_version"] = 20
    spec["board"] = {"width": 6, "height": 6, "lattice": "hex"}
    return spec


def _capturing() -> dict:
    return _takers("R")


@pytest.mark.parametrize(
    ("spec", "extra"),
    [
        (BASE, set()),
        (_wrapped(), {"board.wrap"}),
        (_walled(), {"board.walls"}),
        (_hex(), {"board.lattice.hex"}),
        (_capturing(), {"piece.captures"}),
    ],
    ids=["none", "wrap", "walls", "hex", "captures"],
)
def test_the_rules_a_game_involves_are_its_rules_and_the_board_things_that_act_as_rules(
    spec: dict, extra: set[str]
) -> None:
    kept = {"promotion", "en_passant"}
    assert rules_involved(normal_form(BASE)) == kept
    assert rules_involved(normal_form(spec)) == kept | extra


# ------------------------------------------------ Q922: the memo tables are bounded


class TestTheMemoTablesStayWithinTheirBound:
    """`_PAIRS` and `_READINGS` empty themselves rather than grow without limit, and what
    they answer after emptying is what they answered before. The bound is a constant of
    two hundred thousand and five thousand; a test that fills that would be a minute of
    work, so these shrink it. Real behaviour, since the bound is what holds a long-running
    refresher's memory down, and an off-by-one in it lets one entry too many in."""

    VARIANTS = (
        {"en_passant": None},
        {"win": "extinction"},
        {"win": "royal_capture"},
        {"en_passant": None, "win": "extinction"},
        {"stalemate": "win"},
    )

    @staticmethod
    def _specs() -> list[dict]:
        return [BASE] + [changed(**variant) for variant in TestTheMemoTablesStayWithinTheirBound.VARIANTS]

    def test_the_pair_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from chess_variant_sandbox.spec import relate as module

        specs = self._specs()
        expected = [module.relate(specs[0], other).differences for other in specs[1:]]
        monkeypatch.setattr(module, "_PAIRS", {})
        monkeypatch.setattr(module, "_PAIRS_KEPT", 2)
        for round_ in range(2):
            got = []
            for number, other in enumerate(specs[1:]):
                got.append(module.relate(specs[0], other).differences)
                assert len(module._PAIRS) <= 2, round_
                if round_ == 0:  # every call is a miss: fill to the bound, empty, fill again
                    assert len(module._PAIRS) == number % 2 + 1
            assert got == expected

    def test_the_reading_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from chess_variant_sandbox.spec import relate as module

        specs = self._specs()
        forms = [module.normal_form(spec) for spec in specs]
        expected = [module.similarity_forms(forms[0], other) for other in forms[1:]]
        monkeypatch.setattr(module, "_READINGS", {})
        monkeypatch.setattr(module, "_READINGS_KEPT", 2)
        for round_ in range(2):
            got, sizes = [], []
            for other in forms[1:]:
                got.append(module.similarity_forms(forms[0], other))
                sizes.append(len(module._READINGS))
                assert sizes[-1] <= 2, round_
            assert got == expected
            if round_ == 0:
                # Each call reads the first spec (held once, lost whenever the table
                # empties) and a new one. The table fills to two, then empties when the
                # next new one arrives with it full, leaving that one alone.
                assert sizes == [2, 1, 1, 1, 1]
