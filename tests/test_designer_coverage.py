"""Every key the validator accepts has a control in the designer, or a reason it has none.

Q995a (`docs/reference/designer-coverage.md`). The user could not find where to make a game
in rounds, and the question behind it was whether the designer keeps up with the rule
language at all. It does by construction for the rule cards, because `web/cards.py` builds
them from `spec/jsonschema.py`; `tests/test_designer.py` holds the top level of that. Three
gaps were left, and this file closes them:

- **The keys inside a rule.** `tests/test_spec.py` holds the model's schema to the
  validator's key sets for the rules, a piece, a move, the board, promotion and wrap only.
  A key added to `_SIMULTANEOUS_KEYS` (or drops', or reach's) and not to the schema would
  have no card field, and nothing failed. Here every nested key set the validator keeps is
  compared with the card the designer draws.
- **The hand-written form.** The name, the board, the army's placement and the piece and
  move rows are not generated, so no generator can notice when the validator gains a key
  there. Each of those keys is named here with the control that sets it, read in the page's
  source.
- **What is left out on purpose** is a list, each with its reason, so leaving a key out is
  a decision someone wrote down rather than an omission.

It reads source only: the validator's constants, the card model and the page's script text.
Whether a control is easy to *find* is not a thing a test can hold; the record has that.
"""

from __future__ import annotations

from typing import Any

import pytest
from page_text import DESIGN

from chess_variant_sandbox.spec import schema
from chess_variant_sandbox.web import cards

# The keys the hand-written form sets, each with a string the control is drawn with in the
# page's source (`web/static/page/design.js`, with `cards.js`, `teaching.js`, `measure.js`).
HAND: dict[str, str] = {
    # The game itself.
    "name": 'id="dName"',
    "description": 'id="dDesc"',
    "board": "boardFormHtml(",
    "pieces": 'id="dAddPiece"',
    "setup": "placementEditor(",
    "rules": "rulesCardsHtml(",
    "appearance": 'data-field="icon"',
    "appearance.icons": 'data-field="icon"',
    # The board's form (`boardFormHtml`).
    "board.width": 'data-bfield="width"',
    "board.height": 'data-bfield="height"',
    "board.lattice": 'data-bfield="lattice"',
    "board.shape": 'data-bfield="shape"',
    # Where they start.
    "setup.mirror": 'id="dMirror"',
    "setup.orientation": 'id="dOrientation"',
    "setup.placements": 'id="dAddPlacement"',
    "setup.placements[].piece": 'data-pfield="piece"',
    "setup.placements[].player": 'data-pfield="player"',
    "setup.placements[].squares": 'data-pfield="squares"',
    # A piece's own row (`pieceEditor`).
    "pieces[].id": 'data-field="id"',
    "pieces[].name": 'data-field="name"',
    "pieces[].value": 'data-field="value"',
    "pieces[].royal": 'data-field="royal"',
    "pieces[].moves": "data-addmove=",
    # A way of moving (`moveEditor`).
    "pieces[].moves[].kind": "pick('kind',",
    "pieces[].moves[].vectors": "box('vectors',",
    "pieces[].moves[].symmetry": "pick('symmetry',",
    "pieces[].moves[].range": "box('range',",
    "pieces[].moves[].min_range": "box('min_range',",
    "pieces[].moves[].mode": "pick('mode',",
    "pieces[].moves[].condition.from_rank": "box('from_rank',",
}

# Keys with no control in the designer, on purpose. The game's text (the JSON box under the
# form) still carries every one of them; none is a rule a person designs.
NOT_IN_DESIGNER: dict[str, str] = {
    "schema_version": (
        "settled by the server to the lowest version the draft validates under "
        "(`designing.settled`); a person never chooses it"
    ),
    "provenance": "where a game came from, not how it plays; the keys below say who writes each",
    "provenance.parents": (
        "written by the designer itself when a draft is built on a game (`forkOf`, "
        "`stampParent`), never typed"
    ),
    "provenance.origin": "the library's record of who made a filed game; a draft's maker is its saver",
    "provenance.author": "the library's record of who made a filed game; a draft's maker is its saver",
    "provenance.year": "the library's record of a traditional game's date",
    "provenance.source": "the library's record of where a traditional game's rules were read",
    "provenance.history": "the library's note about a game's past, which the site does not draw",
    "provenance.theme": "the library's note on a game's subject, set when a game is filed",
    "provenance.exhibit": "the maintainer's mark on a game kept for what it shows, not a draft's",
    "provenance.added": "stamped when a game joins the library, never by a designer",
    "provenance.preview": "the maintainer's mark on a game shown as a preview, not a draft's",
}

# Every key set the validator keeps for a rule that holds keys of its own, by the path from
# `rules` to the card that draws it. A rule card with fields of its own and no row here is
# refused below, so a new rule cannot arrive without its keys being compared.
RULE_KEY_SETS: dict[tuple[str, ...], set[str]] = {
    ("promotion",): schema._PROMOTION_KEYS,
    ("castling",): schema._CASTLING_KEYS,
    ("territory",): schema._TERRITORY_KEYS,
    ("turn",): schema._TURN_KEYS,
    ("obstacles",): schema._OBSTACLES_KEYS,
    ("strike",): schema._STRIKE_KEYS,
    ("strike", "chain"): schema._STRIKE_LINK_KEYS,
    ("strike", "landing"): schema._LANDING_KEYS,
    ("contacts",): schema._CONTACTS_KEYS,
    ("contacts", "links"): schema._CONTACT_LINK_KEYS,
    ("contacts", "landing"): schema._LANDING_KEYS,
    ("wrap",): schema._WRAP_KEYS,
    ("npcs",): schema._NPC_KEYS,
    # A portal group is a pair or a map, and one card offers both spellings.
    ("portals",): schema._PORTAL_KEYS | schema._PORTAL_ENDS_KEYS,
    ("portals", "ends"): schema._PORTAL_END_KEYS,
    ("simultaneous",): schema._SIMULTANEOUS_KEYS,
    ("scoring",): schema._SCORING_KEYS,
    ("reach",): schema._REACH_KEYS,
    ("extinction",): schema._EXTINCTION_KEYS,
    ("drops",): schema._DROPS_KEYS,
    ("explode_on_capture",): schema._EXPLODE_KEYS,
}


def _inner(field: dict[str, Any]) -> list[dict[str, Any]]:
    """The fields a card, a list of cards, or a choice between spellings draws inside."""
    control = field["control"]
    if control == "card":
        return list(field["fields"])
    if control == "list":
        return _inner(field["item"])
    if control == "either":
        return [inner for branch in field["branches"] for inner in _inner(branch["field"])]
    return []


def _at(fields: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    field = next(f for f in fields if f["key"] == path[0])
    for key in path[1:]:
        field = next(f for f in _inner(field) if f["key"] == key)
    return field


def _keys(fields: list[dict[str, Any]]) -> set[str]:
    return {f["key"] for f in fields}


def _validator_paths() -> set[str]:
    """Every key path the validator accepts outside `rules`, and the rules' own names."""
    paths = set(schema._SPEC_KEYS)
    paths |= {f"board.{key}" for key in schema._BOARD_KEYS}
    paths |= {f"setup.{key}" for key in schema._SETUP_KEYS}
    paths |= {f"setup.placements[].{key}" for key in schema._PLACEMENT_KEYS}
    paths |= {f"pieces[].{key}" for key in schema._PIECE_KEYS}
    paths |= {f"pieces[].facing.{key}" for key in schema._FACING_KEYS}
    paths |= {f"pieces[].moves[].{key}" for key in schema._MOVE_KEYS}
    paths |= {f"pieces[].moves[].condition.{key}" for key in schema._CONDITION_KEYS}
    paths |= {f"appearance.{key}" for key in schema._APPEARANCE_KEYS}
    paths |= {f"provenance.{key}" for key in schema._PROVENANCE_KEYS}
    paths |= {f"rules.{key}" for key in schema._RULES_KEYS}
    return paths


def _generated_paths(model: dict[str, Any]) -> set[str]:
    """What the card model draws a field for, by the same paths."""
    paths = {f"rules.{key}" for key in _keys(model["rules"])}
    paths |= {f"pieces[].{key}" for key in _keys(model["piece"])}
    paths |= {f"pieces[].facing.{key}" for key in _keys(_inner(_at(model["piece"], ("facing",))))}
    paths |= {f"pieces[].moves[].{key}" for key in _keys(model["move"])}
    paths |= {
        f"pieces[].moves[].condition.{key}"
        for key in _keys(_inner(_at(model["move"], ("condition",))))
    }
    paths |= {f"setup.placements[].{key}" for key in _keys(model["placement"])}
    # `condition` is shared: the move row owns `condition.from_rank` (in HAND) and the card
    # the rest, which is why the card leaves that key out.
    return paths


class TestEveryValidatorKeyHasAControl:
    def test_every_key_is_drawn_or_left_out_on_purpose(self) -> None:
        model = cards.card_model()
        generated = _generated_paths(model)
        hand = set(HAND)
        left_out = set(NOT_IN_DESIGNER)
        accepted = _validator_paths()
        unplaced = accepted - generated - hand - left_out
        assert not unplaced, (
            f"the validator accepts {sorted(unplaced)} and the designer draws no control for "
            "them: add one, or name them in NOT_IN_DESIGNER with the reason"
        )
        assert not left_out - accepted, (
            f"NOT_IN_DESIGNER names keys the validator has dropped: {sorted(left_out - accepted)}"
        )
        assert not hand - accepted, (
            f"HAND names keys the validator has dropped: {sorted(hand - accepted)}"
        )
        assert not hand & left_out, sorted(hand & left_out)
        assert not generated & left_out, sorted(generated & left_out)

    @pytest.mark.parametrize("path", sorted(HAND))
    def test_each_hand_control_is_in_the_page(self, path: str) -> None:
        assert HAND[path] in DESIGN, f"{path}: the page no longer draws {HAND[path]!r}"

    def test_the_hand_form_is_what_the_card_model_leaves_out(self) -> None:
        """`cards.OWNED` is the generator's list of what the form draws; this file's is the
        page's. One list must not move without the other."""
        prefixes = {"piece": "pieces[].", "move": "pieces[].moves[].",
                    "placement": "setup.placements[]."}
        for name, prefix in prefixes.items():
            named = {path.removeprefix(prefix) for path in HAND
                     if path.startswith(prefix) and "[]" not in path.removeprefix(prefix)}
            assert named == set(cards.OWNED[name]), name


class TestEveryKeyInsideARuleHasAField:
    @pytest.mark.parametrize("path", sorted(RULE_KEY_SETS), ids=".".join)
    def test_the_card_holds_the_validators_keys(self, path: tuple[str, ...]) -> None:
        model = cards.card_model()
        drawn = _keys(_inner(_at(model["rules"], path)))
        assert drawn == set(RULE_KEY_SETS[path]), (
            f"rules.{'.'.join(path)}: the validator accepts {sorted(RULE_KEY_SETS[path])}, "
            f"the designer draws {sorted(drawn)}"
        )

    def test_every_rule_with_keys_of_its_own_is_compared(self) -> None:
        model = cards.card_model()
        holding = {(field["key"],) for field in model["rules"] if _inner(field)}
        assert holding <= set(RULE_KEY_SETS), (
            f"rules {sorted(holding - set(RULE_KEY_SETS))} hold keys of their own that no "
            "validator key set here is compared with: add the row"
        )
