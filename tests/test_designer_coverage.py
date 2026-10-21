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

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from page_text import DESIGN
from test_designer import _ADD_RULES, _kit_setup, _run_page

from chess_variant_sandbox.site import routes
from chess_variant_sandbox.spec import schema, validate_spec
from chess_variant_sandbox.web import cards, designing

# The designer's form, kept where a test can read it after `drawDesigner` has drawn it.
_CAPTURE_FORM = """
const FORM = { innerHTML: '', querySelector: () => null, querySelectorAll: () => [] };
const _byId = document.getElementById;
document.getElementById = id => (id === 'form' ? FORM : _byId(id));
function drawnForm() {
  try { drawDesigner(); } catch (error) { /* the wiring needs a browser; the markup is drawn */ }
  return FORM.innerHTML;
}
"""

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
    # `rest` is schema 29's retired switch (`pieces[].rest` replaced it): the shape check lets it
    # through only so that its one refusal is the retirement's, and no designer draws it.
    ("simultaneous",): schema._SIMULTANEOUS_KEYS - {"rest"},
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


# ---------------------------------------------------------------------------------------------
# Q995b: the four fixes of the record's §3, each held here.

CHESS = json.loads((Path(__file__).resolve().parent.parent / "rulesets" / "chess.json")
                   .read_text(encoding="utf-8"))

# A path written the way the validator writes one, anywhere in a sentence a person reads.
_RAW_PATH = re.compile(r"\b(?:rules|pieces|setup|board)(?:\[\d*\]|\.[a-z_]+)")


def _problems(spec: dict[str, Any]) -> list[str]:
    return validate_spec(designing.settled(spec))


class TestRoundsIsOneStepFromChess:
    """Fix 1. From a copy of chess, rounds used to be three round trips, each conflict found
    by fixing the one before (F2). Now one choice makes every change, says each in words,
    and leaves a game."""

    def test_rounds_from_chess_makes_every_change_and_is_a_game(self) -> None:
        answer = designing.take_turns(CHESS, "rounds")
        assert answer["errors"] == []
        made = answer["draft"]
        assert isinstance(made["rules"]["simultaneous"], dict)
        assert _problems(made) == []
        assert made["rules"]["win"] == "royal_capture"
        assert "castling" not in made["rules"] and "en_passant" not in made["rules"]
        said = " ".join(answer["changes"])
        assert len(answer["changes"]) == 3, answer["changes"]
        for words in ("checkmate", "castling", "passing"):
            assert words in said, words
        assert not _RAW_PATH.search(said), said
        assert CHESS["rules"].get("castling"), "the draft asked about is not changed in place"

    def test_the_way_back_clears_what_only_rounds_reads(self) -> None:
        rounds = designing.take_turns(CHESS, "rounds")["draft"]
        rounds["pieces"][1]["speed"] = 1
        rounds["pieces"][1]["beats"] = ["P"]
        assert _problems(rounds) == []
        back = designing.take_turns(rounds, "alternate")
        assert back["errors"] == []
        assert "simultaneous" not in back["draft"]["rules"]
        assert "speed" not in back["draft"]["pieces"][1]
        assert "beats" not in back["draft"]["pieces"][1]
        assert _problems(back["draft"]) == []
        assert len(back["changes"]) == 2, back["changes"]

    def test_several_moves_a_turn_takes_out_what_it_must(self) -> None:
        answer = designing.take_turns(CHESS, "several", {"moves": 2})
        assert answer["errors"] == []
        assert answer["draft"]["rules"]["turn"] == {"moves": 2}
        assert "en_passant" not in answer["draft"]["rules"]
        assert _problems(answer["draft"]) == []

    def test_a_conflict_with_no_change_of_ours_refuses_in_words(self) -> None:
        """A sweep can't be played in rounds, and taking a piece's way of moving away is
        not a turn-taking choice's to make: the draft stays, and the reason is given."""
        sweeping = copy.deepcopy(CHESS)
        sweeping["pieces"][2]["moves"][0]["kind"] = "sweep"
        assert _problems(sweeping) == []
        answer = designing.take_turns(sweeping, "rounds")
        assert answer["draft"] is None
        assert any("sweep" in line for line in answer["errors"]), answer["errors"]
        for line in answer["errors"]:
            assert not _RAW_PATH.search(line), line

    @pytest.mark.parametrize("way", designing.TURN_WAYS)
    def test_each_way_from_every_template_is_a_game(self, way: str) -> None:
        for template in designing.templates():
            answer = designing.take_turns(template["spec"], way)
            assert answer["errors"] == [], (template["key"], answer["errors"])
            assert _problems(answer["draft"]) == [], template["key"]

    def test_a_fresh_designer_says_round(self) -> None:
        """The word a newcomer searched the page for (F1). Rendered as the page draws its
        first panel, from the first template, and read as visible text."""
        spec = designing.templates()[0]["spec"]
        drawn = _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(spec)};" + _CAPTURE_FORM,
            "drawnForm()",
        )
        visible = re.sub(r"<[^>]+>", " ", drawn)
        first = visible.split("The pieces")[0]
        assert re.search(r"\bin rounds\b", first), "the first panel names rounds"
        assert "how the players take turns" in first

    def test_the_page_asks_the_server_for_the_changes(self) -> None:
        """Rule 3: the page proposes a way and draws the answer; which changes it needs is
        the validator's, through `designing.take_turns`."""
        assert "'/api/design/turns'" in DESIGN
        assert routes.find("POST", "/api/design/turns") is not None


class TestEveryReasonAtOnceInPlainWords:
    """Fix 2. A refused rule said the validator's first sentence, raw."""

    def test_rounds_from_chess_names_all_three_conflicts(self) -> None:
        answer = designing.refusals(CHESS, {"simultaneous": {}})
        reasons = answer["refused"]["simultaneous"]
        assert isinstance(reasons, list) and len(reasons) == 3, reasons
        said = " ".join(reasons)
        for words in ("checkmate", "castling", "passing"):
            assert words in said, words
        for line in reasons:
            assert not _RAW_PATH.search(line), line
            assert "_" not in line, line

    def test_every_refusal_on_every_template_is_plain(self) -> None:
        candidates = {"simultaneous": {}, "castling": {"king": "K", "rook": "R"},
                      "en_passant": True, "forbid_check": True, "turn": {"moves": 2}}
        for template in designing.templates() + [{"key": "chess", "spec": CHESS}]:
            answer = designing.refusals(template["spec"], candidates)
            for key, reasons in answer["refused"].items():
                assert isinstance(reasons, list) and reasons, (template["key"], key)
                for line in reasons:
                    assert not _RAW_PATH.search(line), (template["key"], key, line)

    def test_the_menu_lists_every_reason_under_its_rule(self) -> None:
        drawn = _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(CHESS)};"
            " RULE_REFUSED = { simultaneous: ['first reason.', 'second reason.'] };",
            "ruleMenuHtml()",
        )
        assert "first reason." in drawn and "second reason." in drawn
        assert drawn.index("first reason.") < drawn.index("second reason.")


# The five sentences the validator writes about a field of the whole army (`pieces[].key`),
# each with a draft that raises it. Built on chess in rounds, where speed and the spy mean
# something, except where the point is that they don't.
def _in_rounds() -> dict[str, Any]:
    return designing.take_turns(CHESS, "rounds")["draft"]


def _army_cases() -> list[tuple[str, dict[str, Any]]]:
    no_rounds = copy.deepcopy(CHESS)
    no_rounds["pieces"][1]["speed"] = 1
    gap = _in_rounds()
    gap["pieces"][1]["speed"] = 2
    # A speed on a piece nobody can hold orders nothing.
    level = _in_rounds()
    level["pieces"].append({"id": "X", "name": "runner", "value": 1, "speed": 1,
                            "moves": [{"kind": "leap", "vectors": [[0, 1]]}]})
    spy_alone = copy.deepcopy(CHESS)
    spy_alone["pieces"][1]["spy"] = True
    gained = _in_rounds()
    gained["pieces"][1]["spy"] = True
    gained["rules"]["promotion"]["to"] = ["Q", "R", "B", "N"]
    return [
        ("pieces[].speed orders the two moves of a round", no_rounds),
        ("pieces[].speed: speeds are ranks", gap),
        ("pieces[].speed never orders a round", level),
        ("pieces[].spy: a spy decides when a game plays in rounds", spy_alone),
        ("pieces[].spy beside", gained),
    ]


class TestAPieceFieldIsNamedWhereItIs:
    """Fix 3. A problem about `pieces[].speed` or `pieces[].spy` opened *Where the game comes
    from* and named neither the piece nor the field (F3)."""

    @pytest.mark.parametrize("index", range(5))
    def test_each_army_sentence_names_the_piece_and_the_field(self, index: int) -> None:
        opening, spec = _army_cases()[index]
        raised = [line for line in _problems(spec) if line.startswith(opening)]
        assert raised, f"the validator no longer says {opening!r}: {_problems(spec)}"
        for line in raised:
            said = designing.plain_problem(line, spec)
            assert not said.startswith("Where the game comes from"), said
            assert not _RAW_PATH.search(said), said
            assert "piece" in said.lower(), said

    def test_speed_without_rounds_says_where_rounds_are_chosen(self) -> None:
        opening, spec = _army_cases()[0]
        line = next(p for p in _problems(spec) if p.startswith(opening))
        said = designing.plain_problem(line, spec)
        assert said.startswith("Piece 2 (knight)"), said
        assert "in rounds" in said and "how the players take turns" in said

    def test_several_pieces_are_named_together(self) -> None:
        spec = copy.deepcopy(CHESS)
        spec["pieces"][1]["speed"] = 1
        spec["pieces"][2]["speed"] = 1
        line = next(p for p in _problems(spec) if p.startswith("pieces[].speed"))
        said = designing.plain_problem(line, spec)
        assert said.startswith("Pieces 2 (knight) and 3 (bishop)"), said

    def test_a_version_sentence_names_the_field(self) -> None:
        spec = copy.deepcopy(CHESS)
        spec["pieces"][1]["speed"] = 1
        said = designing.plain_problem(
            'the spec uses pieces.speed, added in schema 28; declare "schema_version": 28 '
            "to use it", spec)
        assert "where the game comes from" not in said.lower(), said
        assert "speed" in said and "knight" in said, said


# A refused default, as the validator words one.
_DEFAULT_REFUSED = r"\b(?:false|0|\"[a-z_]+\"|'[a-z_]+') is the default\b|\bmust be true\b"

# Where each generated root's fields go in a spec, for setting one alone.
_ROOTS = {"piece": ("pieces", 0), "move": ("pieces", 0, "moves", 0),
          "placement": ("setup", "placements", 0)}


def _menus(fields: list[dict[str, Any]], path: tuple[Any, ...] = ()) -> list[tuple[tuple, dict]]:
    """Every menu in a field tree, by its path; a list's first item stands for every item."""
    out: list[tuple[tuple, dict]] = []
    for field in fields:
        here = (*path, field["key"])
        if field["control"] == "menu":
            out.append((here, field))
        elif field["control"] == "card":
            out += _menus(field["fields"], here)
        elif field["control"] == "list" and field["item"]["control"] == "card":
            out += _menus(field["item"]["fields"], (*here, 0))
    return out


def _set(spec: dict[str, Any], path: tuple[Any, ...], value: Any) -> bool:
    """Set one value inside a spec, through what is already there; False where the path
    passes through nothing (a list with no item, a part the card did not add)."""
    node: Any = spec
    for step in path[:-1]:
        if isinstance(step, int):
            if not isinstance(node, list) or len(node) <= step:
                return False
            node = node[step]
        else:
            if not isinstance(node, dict):
                return False
            node = node.setdefault(step, {})
    if not isinstance(node, dict):
        return False
    node[path[-1]] = value
    return True


class TestNoMenuOffersARefusedDefault:
    """Fix 4 (the record's §4.3). Three menus offered a *no* the validator refuses as the
    default written out (F4). Every option of every menu the card model draws is set alone
    on each template, with its rule added as the page adds it, and on chess in rounds; the
    validator must not refuse it as a default."""

    def _built(self) -> dict[str, dict[str, Any]]:
        held = {
            t["key"]: {"spec": t["spec"], "preview": designing.check(t["spec"])["preview"]}
            for t in designing.templates()
        }
        rounds = _in_rounds()
        held["chess in rounds"] = {"spec": rounds, "preview": designing.check(rounds)["preview"]}
        return _run_page(_kit_setup() + f"const TEMPLATES = {json.dumps(held)};" + _ADD_RULES,
                         "addEveryRule(TEMPLATES)")

    def test_no_option_is_refused_as_the_default(self) -> None:
        model = cards.card_model()
        refused: list[str] = []
        for template, rules in self._built().items():
            for rule in model["rules"]:
                with_rule = rules[rule["key"]]
                for path, menu in _menus([rule]):
                    for option in menu["options"]:
                        trial = copy.deepcopy(with_rule)
                        if not _set(trial["rules"], path, option["value"]):
                            continue
                        refused += [f"{template}: rules.{'.'.join(map(str, path))} = "
                                    f"{option['value']!r}: {line}"
                                    for line in _problems(trial)
                                    if re.search(_DEFAULT_REFUSED, line)
                                    and f".{path[-1]}" in line]
            base = rules[model["rules"][0]["key"]]
            for root, where in _ROOTS.items():
                for path, menu in _menus(model[root]):
                    for option in menu["options"]:
                        trial = copy.deepcopy(base)
                        if not _set(trial, (*where, *path), option["value"]):
                            continue
                        refused += [f"{template}: {root}.{'.'.join(map(str, path))} = "
                                    f"{option['value']!r}: {line}"
                                    for line in _problems(trial)
                                    if re.search(_DEFAULT_REFUSED, line)
                                    and f".{path[-1]}" in line]
        assert not refused, "\n".join(refused)

    @pytest.mark.parametrize("path", sorted(cards.TRUE_ONLY))
    def test_each_true_only_switch_is_refused_as_false(self, path: str) -> None:
        """`cards.TRUE_ONLY` says what the schema does not; it must keep agreeing with the
        validator, or it would hide a *no* somebody may choose."""
        spec = _in_rounds()
        spec["rules"]["contacts"] = {"position": "stay", "immune": ["K"],
                                     "links": [{"mover": "N", "onto": "P", "onto_to": "B"}]}
        dotted = path.replace("[]", ".0").split(".")
        steps = tuple(int(step) if step.isdigit() else step for step in dotted)
        assert _set(spec, steps, False)
        key = steps[-1]
        assert any(re.search(_DEFAULT_REFUSED, line) and f".{key}" in line
                   for line in _problems(spec)), (path, _problems(spec))

    def test_the_true_only_menus_offer_yes_alone(self) -> None:
        model = cards.card_model()
        trees = {"rules": model["rules"], "pieces[]": model["piece"]}
        found = {}
        for prefix, fields in trees.items():
            for path, menu in _menus(fields):
                dotted = prefix + "." + ".".join(
                    "[]" if isinstance(step, int) else str(step) for step in path)
                found[dotted.replace(".[]", "[]")] = [o["value"] for o in menu["options"]]
        for path in cards.TRUE_ONLY:
            assert found[path] == [True], (path, found.get(path))


# ------------------------------------------------------------------ Q995c: fixes 5, 6, 8 and 9


def _template(key: str) -> dict[str, Any]:
    return copy.deepcopy(next(t["spec"] for t in designing.templates() if t["key"] == key))


def _skirmish_in_rounds() -> dict[str, Any]:
    made = designing.take_turns(_template("skirmish"), "rounds")["draft"]
    assert made is not None
    return copy.deepcopy(made)


def _draw(spec: dict[str, Any], expression: str, extra: str = "") -> Any:
    return _run_page(_kit_setup() + f"DRAFT = {json.dumps(spec)};" + _CAPTURE_FORM + extra,
                     expression)


def _decode(text: str) -> str:
    return (text.replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">")
            .replace("&#39;", "'").replace("&amp;", "&"))


def _drawn_paths(html: str, root: str) -> list[str]:
    """The top key of every control drawn for `root`, once per control: a field, or the
    *add* button of a part that holds nothing yet."""
    found = re.findall(r'data-croot="' + re.escape(root) + r'" data-cpath="([^"]*)"', html)
    found += re.findall(r'data-cadd="([^"]*)" data-croot-add="' + re.escape(root) + '"', html)
    return [json.loads(_decode(path))[0] for path in found]


def _outside_folds(html: str) -> str:
    """The card as it stands closed: every `<details>` taken out."""
    return re.sub(r"<details\b.*?</details>", "", html, flags=re.S)


def _visible(html: str) -> str:
    return re.sub(r"\s+", " ", _decode(re.sub(r"<[^>]+>", " ", html)))


class TestRoundsOnThePieceCard:
    """Fix 5 (F5). Speed, *beats* and the spy were the fourth to sixth of seven fields in a
    fold at the foot of each piece card, and drawn in games that refuse them."""

    def test_in_rounds_speed_is_set_without_opening_a_fold(self) -> None:
        card = _draw(_skirmish_in_rounds(), "pieceEditor(DRAFT.pieces[0], 0)")
        closed = _drawn_paths(_outside_folds(card), "piece.0")
        for key in (*cards.PIECE_ROW["rounds"], *cards.PIECE_ROW["always"]):
            assert key in closed, (key, closed)
        assert 'placeholder="0, the slowest"' in card

    @pytest.mark.parametrize("rounds", [False, True])
    def test_every_field_is_drawn_once(self, rounds: bool) -> None:
        spec = _skirmish_in_rounds() if rounds else _template("skirmish")
        card = _draw(spec, "pieceEditor(DRAFT.pieces[0], 0)")
        drawn = _drawn_paths(card, "piece.0")
        assert len(drawn) == len(set(drawn)), drawn
        every = {field["key"] for field in cards.card_model()["piece"]}
        hidden = set() if rounds else set(cards.PIECE_ROW["rounds"])
        assert set(drawn) == every - hidden, (set(drawn), every - hidden)

    def test_out_of_rounds_the_three_fold_behind_one_line(self) -> None:
        card = _draw(_template("skirmish"), "pieceEditor(DRAFT.pieces[0], 0)")
        drawn = _drawn_paths(card, "piece.0")
        for key in cards.PIECE_ROW["rounds"]:
            assert key not in drawn, key
        assert "rest" in _drawn_paths(_outside_folds(card), "piece.0")
        said = _visible(card)
        assert "come with playing in rounds" in said, said
        assert "how the players take turns" in said
        assert "data-toturns" in card

    def test_a_speed_already_written_stays_in_sight(self) -> None:
        spec = _template("skirmish")
        spec["pieces"][0]["speed"] = 2
        card = _draw(spec, "pieceEditor(DRAFT.pieces[0], 0)")
        assert "speed" in _drawn_paths(_outside_folds(card), "piece.0")

    def test_a_set_speed_shows_on_the_head_and_in_the_tray(self) -> None:
        spec = _skirmish_in_rounds()
        spec["pieces"][0]["speed"] = 2
        head, tray, picker = _draw(
            spec, "[pieceEditor(DRAFT.pieces[0], 0).split('data-uploads')[0],"
                  " paletteHtml(), pieceTrayHtml()]")
        for drawn in (head, tray, picker):
            assert "speed 2" in _decode(drawn), drawn
        plain = _draw(_skirmish_in_rounds(), "paletteHtml()")
        assert "speed" not in plain

    def test_the_row_agrees_with_the_validator(self) -> None:
        """A `rounds` key is refused in a game not in rounds and taken in one; an `always`
        key is taken in both."""
        values = {"speed": 1, "spy": True, "rest": 1, "beats": ["P"]}
        for key in (*cards.PIECE_ROW["rounds"], *cards.PIECE_ROW["always"]):
            for rounds in (False, True):
                # Chess's first piece: two kinds of piece, so a speed of 1 is a rank above
                # the rest, and one no promotion makes, so it may be a spy.
                spec = _in_rounds() if rounds else copy.deepcopy(CHESS)
                spec["pieces"][0][key] = values[key]
                problems = _problems(spec)
                if key in cards.PIECE_ROW["rounds"] and not rounds:
                    assert any(key in line for line in problems), (key, problems)
                else:
                    assert problems == [], (key, rounds, problems)


class TestFindARuleByTyping:
    """Fix 6 (F6). *Add a rule* was a long menu with no search, each name cut off by its
    library count."""

    def test_every_other_name_belongs_to_a_rule(self) -> None:
        keys = {field["key"] for field in cards.card_model()["rules"]}
        assert set(cards.RULE_ALSO_CALLED) <= keys, set(cards.RULE_ALSO_CALLED) - keys

    @pytest.mark.parametrize(("word", "key"), [
        ("rounds", "simultaneous"), ("atomic", "explode_on_capture"),
        ("three-check", "check_limit"), ("circe", "strike"), ("walls", "obstacles"),
        ("crazyhouse", "drops"), ("Wild", "npcs"),
    ])
    def test_a_word_finds_its_rule(self, word: str, key: str) -> None:
        found = _draw(_template("kingly"), "ruleOptionsHtml()",
                      f"RULE_FIND = {json.dumps(word)};")
        offered = re.findall(r'<option value="([a-z_]+)"', found)
        assert key in offered, (word, offered)
        every = _draw(_template("kingly"), "ruleOptionsHtml()")
        assert len(offered) < len(re.findall(r'<option value="([a-z_]+)"', every))

    def test_one_match_is_chosen(self) -> None:
        found = _draw(_template("kingly"), "ruleOptionsHtml()", "RULE_FIND = 'atomic';")
        assert re.search(r'<option value="explode_on_capture"[^>]*selected', found), found

    def test_nothing_found_says_so(self) -> None:
        said = _draw(_template("kingly"), "ruleFoundHtml()", "RULE_FIND = 'zzzz';")
        assert "No rule" in _visible(said), said

    def test_a_refused_match_says_where_it_is_chosen(self) -> None:
        said = _draw(CHESS, "ruleFoundHtml()",
                     "RULE_FIND = 'rounds'; RULE_REFUSED = { simultaneous: ['a reason.'] };")
        words = _visible(said)
        assert "how the players take turns" in words, words
        assert "data-toturns" in said

    def test_a_match_already_in_the_game_says_so(self) -> None:
        said = _draw(_skirmish_in_rounds(), "ruleFoundHtml()", "RULE_FIND = 'rounds';")
        assert "already in this game" in _visible(said), said

    def test_the_library_count_moves_under_the_choice(self) -> None:
        menu = _draw(_template("kingly"), "ruleMenuHtml()")
        assert "in the library)" not in menu
        assert "find one: rounds" in menu
        chosen = _draw(_template("kingly"), "ruleChosenHtml('drops')")
        assert "in the library" in _visible(chosen), chosen


class TestOnlyAHopsFieldsOnAHop:
    """Fix 8 (F8). *More about this move* offered a hop's two fields on a leap or a slide."""

    def _move(self, move: dict[str, Any]) -> list[str]:
        spec = _template("skirmish")
        spec["pieces"][0]["moves"] = [move]
        drawn = _draw(spec, "moveEditor(DRAFT.pieces[0].moves[0], 0, 0)")
        return _drawn_paths(drawn, "move.0.0")

    def test_a_leap_has_no_hop_fields(self) -> None:
        drawn = self._move({"kind": "leap", "vectors": [[0, 1]]})
        for key in cards.MOVE_KIND_ONLY:
            assert key not in drawn, (key, drawn)
        assert "condition" in drawn

    def test_a_hop_has_them(self) -> None:
        drawn = self._move({"kind": "hop", "vectors": [[0, 1]]})
        for key in cards.MOVE_KIND_ONLY:
            assert key in drawn, (key, drawn)

    def test_one_already_written_stays(self) -> None:
        drawn = self._move({"kind": "slide", "vectors": [[0, 1]], "screen": "own"})
        assert "screen" in drawn and "screen_range" not in drawn, drawn

    @pytest.mark.parametrize("key", sorted(cards.MOVE_KIND_ONLY))
    def test_the_validator_refuses_each_on_every_other_kind(self, key: str) -> None:
        value = {"screen": "own", "screen_range": 1}[key]
        for kind in schema.MOVE_KINDS:
            spec = _template("skirmish")
            spec["pieces"][0]["moves"][0]["kind"] = kind
            base = _problems(spec)
            spec["pieces"][0]["moves"][0][key] = value
            added = [line for line in _problems(spec) if line not in base]
            if kind in cards.MOVE_KIND_ONLY[key]:
                assert not any(key in line for line in added), (kind, added)
            else:
                assert any(key in line for line in added), (kind, added)


class TestPolish:
    """Fix 9 (F7): a bare value in a menu, and a save under a template's name."""

    def test_each_bare_value_reads_as_a_choice(self) -> None:
        menus = {"rules." + ".".join(map(str, path)): menu
                 for path, menu in _menus(cards.card_model()["rules"])}
        for (path, value), (label, says) in cards.OPTION_WORDS.items():
            option = next(o for o in menus[path]["options"] if o["value"] == value)
            assert (option["label"], option["says"]) == (label, says)
        for bare in ("initiative", "impossible", "own", "collision", "score"):
            labels = [o["label"] for menu in menus.values() for o in menu["options"]
                      if o["value"] == bare]
            assert labels and bare not in labels, (bare, labels)

    def test_the_words_are_a_visitors(self) -> None:
        import site_words

        for (path, _), (label, says) in cards.OPTION_WORDS.items():
            for text in (label, says):
                assert not site_words.hits(text, tuple(site_words.PHRASES)), (path, text)
                assert not re.search(r"\b[a-z]+_[a-z_]+\b", text), (path, text)

    _SAVING = """
      DESIGN_KIT.templates = TEMPLATES;
      let ASKED = 0;
      api = () => { ASKED += 1; return new Promise(() => {}); };
    """

    def _saving(self, name: str | None, then: str) -> Any:
        spec = _template("skirmish")
        if name is not None:
            spec["name"] = name
        setup = f"const TEMPLATES = {json.dumps(designing.templates())};" + self._SAVING
        return _draw(spec, then, setup)

    def test_a_save_under_a_templates_name_asks_for_one(self) -> None:
        asked, saved = self._saving(None, "(saveDraft(), [ASKED, DRAFT_SAVED])")
        assert asked == 0
        assert "the name of the template it started from" in _visible(saved), saved
        assert 'id="dSaveName"' in saved

    def test_a_name_of_its_own_saves_at_once(self) -> None:
        asked, _ = self._saving("a game of my own", "(saveDraft(), [ASKED, DRAFT_SAVED])")
        assert asked == 1

    def test_keeping_the_name_saves(self) -> None:
        asked = self._saving(None, "(saveDraft(), keepNameAndSave(), ASKED)")
        assert asked == 1

    def test_a_copy_saved_under_its_games_name_asks(self) -> None:
        """The game a copy came from is the commonest clash: a copy of chess named back
        *chess*. Once saved under a name, the name is the draft's own."""
        whose = self._saving(
            "Chess", "(LIBRARY = [{ name: 'chess', hash: 'aaaa' }], DRAFT_FROM = 'aaaa',"
                     " [borrowedName(), (NAME_KEPT = 'chess', borrowedName())])")
        assert whose == ["game", ""]

    def test_a_new_name_is_written_then_saved(self) -> None:
        asked, name = self._saving(
            None, "(saveDraft(), saveUnderName('river crossing'), [ASKED, DRAFT.name])")
        assert (asked, name) == (1, "river crossing")
