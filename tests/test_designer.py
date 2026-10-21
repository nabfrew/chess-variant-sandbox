"""The designer's boards as inputs: a way of moving drawn and fitted, pieces put down.

The fitter and the placement writer are page functions, so they run in node against the
page's own script. Every claim about what a fitted descriptor *does* is checked against
`designing.move_map`, which is the engine's answer and the only one that counts.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import node_session
import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.loop import gates
from chess_variant_sandbox.spec import SpecError, describe, load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import _RULES_KEYS
from chess_variant_sandbox.web import cards, designing

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"

_STUB = """
  const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                             set: () => true });
  global.document = { getElementById: () => el, querySelector: () => null,
                      querySelectorAll: () => [], addEventListener: () => {},
                      createElement: () => el, body: el };
  global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                    addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
  global.location = { pathname: '/game', hash: '' };
  global.fetch = () => new Promise(() => {});
"""


def _run_page(setup: str, expression: str) -> Any:
    """Evaluate `expression` (which may be a promise) against the page's own functions."""
    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    body = _STUB + script.group(1) + "\n" + setup + "\n"
    body += "Promise.resolve(" + expression + ").then(v => console.log(JSON.stringify(v)));"
    done = node_session.run(body)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + (ord(name[0]) - ord("a"))


def _offers(
    spec: dict,
    piece: int,
    origin: str,
    clicks: list[str],
    width: int = 8,
    **asked: Any,
) -> list[dict]:
    """What the server says a click on those squares could mean.

    The same click on the same draft is asked by several tests, and each answer is a
    handful of engine probes, so an answer is kept by its question (Q377) and handed out
    as a copy.
    """
    question = json.dumps([spec, piece, origin, clicks, width, asked], sort_keys=True)
    if question not in _ANSWERED:
        _ANSWERED[question] = designing.teach(
            spec,
            piece,
            _square(origin, width),
            [_square(one, width) for one in clicks],
            **asked,
        )["offers"]
    return copy.deepcopy(_ANSWERED[question])


_ANSWERED: dict[str, list[dict]] = {}


def _probe(base: dict, moves: list[dict], square: int) -> dict:
    spec = json.loads(json.dumps(base))
    spec["pieces"][0]["moves"] = moves
    return designing.move_map(spec, 0, square=square)


# One piece on an empty 8x8 board, whose moves each probe replaces.
_BASE = {
    "schema_version": 1,
    "name": "drawing",
    "board": {"width": 8, "height": 8},
    "pieces": [
        {"id": "P", "name": "piece", "value": 1,
         "moves": [{"kind": "leap", "vectors": [[0, 1]]}]},
        {"id": "K", "name": "king", "royal": True, "value": 0,
         "moves": [{"kind": "leap", "vectors": [[0, 1], [1, 1]], "symmetry": "dihedral8"}]},
    ],
    "setup": {"mirror": True, "placements": [
        {"piece": "K", "squares": ["e1"]}, {"piece": "P", "squares": ["a2"]},
    ]},
    "rules": {"win": "royal_capture", "max_plies": 100},
}


class TestWhatAClickCouldMean:
    """One click, and every way of moving the rule language has that gets there.

    The readings are the server's: `designing.teach` proposes descriptors, asks the
    engine which squares each one reaches and says each one back through
    `spec/describe.py`. The page holds no fitter of its own any more, because two
    answers to one question is how a board comes to disagree with the game.
    """

    def _said(self, offers: list[dict]) -> list[str]:
        return [offer["says"] for offer in offers]

    def test_the_exact_reading_comes_first(self) -> None:
        offers = _offers(_BASE, 0, "d4", ["e6"])
        assert offers[0]["exact"] and offers[0]["extra"] == []
        assert offers[0]["move"] == {"kind": "leap", "vectors": [[1, 2]]}

    def test_one_knights_square_offers_the_whole_knight(self) -> None:
        """Demonstrate and generalise, which is what B2 of the plan is for: a person
        clicks one square and the designer offers the move that square belongs to."""
        offers = _offers(_BASE, 0, "d4", ["e6"])
        turned = next(
            offer for offer in offers
            if offer["move"].get("symmetry") == "dihedral8"
        )
        assert turned["move"] == {
            "kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8",
        }
        assert len(turned["moves"]) == 8 and len(turned["extra"]) == 7
        assert "in all eight directions" in turned["says"]

    def test_a_knight_in_the_corner_still_offers_it(self) -> None:
        """The board clips what can be clicked; the offer has to allow for the edge."""
        offers = _offers(_BASE, 0, "a1", ["b3"])
        assert any(
            offer["move"] == {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}
            for offer in offers
        )

    def test_one_diagonal_square_offers_the_bishop(self) -> None:
        offers = _offers(_BASE, 0, "d4", ["f6"])
        said = self._said(offers)
        assert "two squares diagonally forward and right, jumping over whatever is in "\
               "between" in said
        assert "any distance along all four diagonals, stopped by anything in the "\
               "way" in said

    def test_what_blocks_it_is_the_one_thing_a_click_cannot_say(self) -> None:
        """A leap and a blockable leap reach the same squares on an empty board and are
        different rules, so both are offered and each sentence says which it is."""
        kinds = {offer["move"]["kind"] for offer in _offers(_BASE, 0, "d4", ["f6"])}
        assert {"leap", "lame", "slide"} <= kinds
        blockable = next(
            offer for offer in _offers(_BASE, 0, "d4", ["f6"])
            if offer["move"]["kind"] == "lame"
        )
        assert "stopped by a piece standing on the square it passes" in blockable["says"]

    def test_a_sweep_is_offered_for_exactly_what_was_clicked(self) -> None:
        """Every line has a sweep, and offering them all would bury the readings a
        person meant under a rule they have probably never heard of."""
        offers = _offers(_BASE, 0, "d4", ["f6"])
        sweeps = [offer for offer in offers if offer["move"]["kind"] == "sweep"]
        assert sweeps and all(offer["exact"] for offer in sweeps)

    def test_a_click_that_means_only_a_take_says_so(self) -> None:
        """A10: a click means move *or* take unless somebody says otherwise, and what
        they said reaches the descriptor and the sentence together."""
        both = _offers(_BASE, 0, "d2", ["c3"])[0]
        assert "mode" not in both["move"]
        assert both["moves"] == both["takes"]
        only = _offers(_BASE, 0, "d2", ["c3"], mean="capture")[0]
        assert only["move"]["mode"] == "capture"
        assert only["moves"] == [] and only["takes"] == both["takes"]
        assert only["says"].endswith("only to take")

    def test_only_from_this_rank_reaches_the_descriptor_and_the_sentence(self) -> None:
        pinned = _offers(_BASE, 0, "d2", ["d4"], mean="move", from_rank=True)
        first = pinned[0]
        assert first["move"]["condition"] == {"from_rank": 1}
        assert "only from its own rank 2" in first["says"]

    def test_every_offer_is_what_the_engine_says_it_is(self) -> None:
        """The proposing is this module's; the claim is the engine's. Each descriptor
        offered really reaches the squares the offer carries, asked again of the whole
        draft rather than of the stripped board the offers were probed on."""
        for offer in _offers(_BASE, 0, "d4", ["f6"]):
            answered = _probe(_BASE, [offer["move"]], _square("d4"))
            assert answered["moves"] == offer["moves"], offer["says"]

    def test_every_offer_is_a_draft_that_validates(self) -> None:
        for offer in _offers(_BASE, 0, "d4", ["e6"]) + _offers(_BASE, 0, "d2", ["d3"]):
            spec = json.loads(json.dumps(_BASE))
            spec["pieces"][0]["moves"] = [offer["move"]]
            assert validate_spec(designing.settled(spec)) == [], offer["says"]

    def test_two_clicks_narrow_the_exact_reading(self) -> None:
        one = _offers(_BASE, 0, "d4", ["e6"])[0]
        two = _offers(_BASE, 0, "d4", ["e6", "f5"])[0]
        assert one["move"]["vectors"] == [[1, 2]]
        assert two["exact"] and sorted(two["move"]["vectors"]) == [[1, 2], [2, 1]]

    def test_a_hex_board_is_offered_the_hexagons_turns(self) -> None:
        """A hex lattice takes only the hexagon's symmetries (schema 18b), so one step is
        offered as itself and turned six ways, and no square turn is offered. That is the
        schema's rule rather than the designer's."""
        hexed = load_spec(Path(__file__).resolve().parent.parent / "archive"
                          / "hex-glinski-hexagon.json")
        width = hexed["board"]["width"]
        offers = _offers(hexed, 0, "f6", ["f7"], width=width)
        assert offers, "a hex board offered nothing at all"
        turns = {offer["move"].get("symmetry", "none") for offer in offers}
        assert turns <= {"none", "rotate6", "dihedral12"}
        assert "rotate6" in turns
        assert any("forward-right" in offer["says"] for offer in offers)
        assert any("in all six directions" in offer["says"] for offer in offers)

    def test_a_square_nothing_reaches_is_answered_with_nothing(self) -> None:
        """A leap past the schema's widest vector, which no descriptor can write."""
        wide = json.loads(json.dumps(_BASE))
        wide["board"] = {"width": 16, "height": 3}
        assert _offers(wide, 0, "a1", ["p3"], width=16) == []

    def test_the_click_the_piece_is_standing_on_is_not_a_target(self) -> None:
        assert _offers(_BASE, 0, "d4", ["d4"]) == []


class TestTheChessPawnIsTaughtWithoutTypingAVector:
    """The exit test for teach-a-move: the chess pawn, built by clicking squares and
    picking sentences, plays exactly as `rulesets/chess.json`'s pawn plays."""

    def _pick(self, offers: list[dict], says: str) -> dict:
        found = [offer for offer in offers if offer["says"] == says]
        assert found, f"no reading said {says!r}; got {[o['says'] for o in offers]}"
        return found[0]["move"]

    def test_the_pawn(self) -> None:
        step = self._pick(
            _offers(_BASE, 0, "d4", ["d5"], mean="move"),
            "one square forward, not to take",
        )
        takes = self._pick(
            _offers(_BASE, 0, "d4", ["c5"], mean="capture"),
            "one square diagonally forward, either way, only to take",
        )
        double = self._pick(
            _offers(_BASE, 0, "d2", ["d4"], mean="move", from_rank=True),
            "exactly two squares forward, stopped by anything in the way, not to take,"
            " and only from its own rank 2",
        )
        spec = json.loads(json.dumps(_BASE))
        spec["pieces"][0]["moves"] = [step, takes, double]
        assert validate_spec(designing.settled(spec)) == []

        chess = load_spec(RULESETS / "chess.json")
        for origin in ("d2", "d4", "a2", "h5"):
            taught = designing.move_map(spec, 0, square=_square(origin))
            real = designing.move_map(chess, 0, square=_square(origin))
            assert taught["moves"] == real["moves"], origin
            assert taught["takes"] == real["takes"], origin


class TestTheSentenceSaysWhatWasTaught:
    """The offers' sentences are `spec/describe.py`'s, so a piece taught in the editor
    and the same piece read on its game's page are said in one voice."""

    def test_the_pawns_double_step(self) -> None:
        said = describe.plain_move_sentence({
            "kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2,
            "mode": "move", "condition": {"from_rank": 1},
        })
        assert said == ("exactly two squares forward, stopped by anything in the way,"
                        " not to take, and only from its own rank 2")

    def test_the_knight(self) -> None:
        said = describe.plain_move_sentence(
            {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}
        )
        assert said == ("leaps two and one, in all eight directions, jumping over"
                        " whatever is in between")

    def test_a_line_along_an_oblique_step_is_offered(self) -> None:
        """A piece that slides two-and-one over and over.

        The vocabulary had no phrase for it through stage 2, so one click on a
        two-and-one square could not be read as a line at all and the move row was the
        only way to write one. `describe` counts the line in steps rather than in
        squares, which is what the engine counts a slide's range in.
        """
        oblique = {"kind": "slide", "vectors": [[1, 2]], "symmetry": "dihedral8"}
        assert describe.plain_move_sentence(oblique) == (
            "any number of steps of two and one, in all eight directions,"
            " stopped by anything in the way")
        assert [
            offer for offer in _offers(_BASE, 0, "d4", ["e6"])
            if offer["move"] == oblique
        ]

    def test_a_descriptor_with_no_phrase_is_not_said_at_all(self) -> None:
        """A line stepping two squares at a time. The designer would rather offer one
        reading fewer than one nobody can read, so this one is never an offer."""
        stretched = {"kind": "slide", "vectors": [[2, 2]], "symmetry": "rotate4"}
        assert describe.plain_move_sentence(stretched) is None
        assert describe.move_sentence(stretched).startswith("moves and captures along")
        assert not [
            offer for offer in _offers(_BASE, 0, "d4", ["f6"])
            if offer["move"] == stretched
        ]

def _place(spec: dict, placing: dict, squares: list[str], width: int = 8) -> dict:
    """Put PLACING down on each square in turn, and return the setup that results.

    The form redraw and the re-check that follow a placement need a real DOM and a
    server; what is under test is the setup `placeAt` writes, so both are stubbed.
    """
    preview = {"width": spec["board"]["width"], "height": spec["board"]["height"]}
    clicks = "".join(f"placeAt({_square(s, width)});" for s in squares)
    return _run_page(
        "drawDesigner = () => {}; checkDraft = () => {};"
        f"DRAFT = {json.dumps(spec)}; DRAFT_CHECK = {{ preview: {json.dumps(preview)} }};"
        f" PLACING = {json.dumps(placing)}; {clicks}",
        "DRAFT.setup",
    )


class TestPiecesArePutDownOnTheBoard:
    def test_a_click_writes_a_square_and_a_second_click_lifts_it(self) -> None:
        setup = _place(_BASE, {"piece": 0, "player": 0}, ["b2", "c2"])
        assert {"piece": "P", "squares": ["a2", "b2", "c2"]} in setup["placements"]
        lifted = _place(_BASE, {"piece": 0, "player": 0}, ["a2"])
        assert all(p["piece"] != "P" for p in lifted["placements"])

    def test_under_a_mirror_the_far_side_is_written_as_its_reflection(self) -> None:
        """Lifting a black pawn on a7 lifts the white one on a2 it is the image of."""
        lifted = _place(_BASE, {"erase": True}, ["a7"])
        assert all(p["piece"] != "P" for p in lifted["placements"])

    def test_a_rotated_board_reflects_the_file_too(self) -> None:
        spec = json.loads(json.dumps(_BASE))
        spec["setup"]["orientation"] = "rotate"
        lifted = _place(spec, {"erase": True}, ["h7"])
        assert all(p["piece"] != "P" for p in lifted["placements"])
        kept = _place(spec, {"erase": True}, ["a7"])
        assert {"piece": "P", "squares": ["a2"]} in kept["placements"]

    def test_without_a_mirror_the_second_player_gets_rows_of_its_own(self) -> None:
        spec = json.loads(json.dumps(_BASE))
        spec["setup"]["mirror"] = False
        setup = _place(spec, {"piece": 1, "player": 1}, ["e8"])
        assert {"piece": "K", "player": 1, "squares": ["e8"]} in setup["placements"]
        spec["setup"] = setup
        assert validate_spec(designing.settled(spec)) == []
        board = designing.check(spec)["preview"]["board"]
        assert board[_square("e8")] != 0 and board[_square("e1")] != 0

    def test_putting_a_piece_on_an_occupied_square_replaces_it(self) -> None:
        setup = _place(_BASE, {"piece": 1, "player": 0}, ["a2"])
        assert {"piece": "K", "squares": ["e1", "a2"]} in setup["placements"]
        assert all(p["piece"] != "P" for p in setup["placements"])


class TestABrokenDraftKeepsItsBoard:
    """A half-made edit leaves the draft invalid, and the board is the tool the repair
    needs. The last position a check could draw stays, marked stale, and still takes
    clicks -- now on the workbench board in the middle, with the errors beside it."""

    def _preview(self) -> dict:
        kingly = next(t for t in designing.templates() if t["key"] == "kingly")
        return designing.check(kingly["spec"])["preview"]

    def test_the_board_stays_and_the_errors_stand_beside_it(self) -> None:
        setup = (
            "DRAFT = { name: 'x', pieces: [] }; DRAFT_MAP = null; DRAWING = null;"
            " const side = { innerHTML: '', querySelectorAll: () => [] };"
            " const board = { innerHTML: '', querySelectorAll: () => [] };"
            " document.getElementById = id => id === 'verdictSide' ? side"
            "   : id === 'boardSide' ? board : { onclick: null };"
            " wireBoards = () => {}; wirePalette = () => {};"
            f" DRAFT_PREVIEW = {json.dumps(self._preview())};"
            " DRAFT_CHECK = { valid: false, errors: ['no royal piece appears in setup'] };"
        )
        drawn = _run_page(
            setup, "(drawVerdict(), { board: board.innerHTML, says: side.innerHTML })"
        )
        assert 'data-board="start"' in drawn["board"]
        assert "benchBoard stale" in drawn["board"]
        assert "no royal piece appears in setup" in drawn["says"]
        assert "Not a game yet" in drawn["says"]

    def test_a_piece_can_be_put_down_while_the_draft_is_not_a_game(self) -> None:
        """The way out of walk 1.1's dead end: the king goes back on the board that the
        queen displaced it from, with the draft invalid the whole time."""
        kingly = next(t for t in designing.templates() if t["key"] == "kingly")
        spec = json.loads(json.dumps(kingly["spec"]))
        preview = designing.check(spec)["preview"]
        # The king lifted, which is the state the walk got stuck in.
        spec["setup"]["placements"] = [
            p for p in spec["setup"]["placements"] if p["piece"] != "K"
        ]
        king = next(i for i, p in enumerate(spec["pieces"]) if p["id"] == "K")
        setup = _run_page(
            "drawDesigner = () => {}; checkDraft = () => {};"
            f"DRAFT = {json.dumps(spec)}; DRAFT_PREVIEW = {json.dumps(preview)};"
            " DRAFT_CHECK = { valid: false, errors: ['no royal piece appears'] };"
            f" PLACING = {{ piece: {king}, player: 0 }}; placeAt({_square('d1', 6)});",
            "DRAFT.setup",
        )
        assert {"piece": "K", "squares": ["d1"]} in setup["placements"]
        spec["setup"] = setup
        assert designing.check(spec)["valid"]


ARCHIVE = RULESETS.parent / "archive"
LIBRARY = sorted(RULESETS.glob("*.json")) + sorted(ARCHIVE.glob("*.json"))


def _escaped(text: str) -> str:
    """One sentence as `page/core.esc` writes it into the markup."""
    for raw, entity in (("&", "&amp;"), ('"', "&quot;"), ("'", "&#39;"),
                        ("<", "&lt;"), (">", "&gt;")):
        text = text.replace(raw, entity)
    return text


def _kit_setup(corpus: list[dict] | None = None) -> str:
    """What `GET /api/design` hands the page, as the page's own code reads it."""
    kit = {
        "cards": cards.card_model(corpus or []),
        "ruleWords": describe.rule_words(),
        "ruleChoices": describe.rule_choices(),
        "palette": designing.piece_palette(corpus or []),
    }
    return f"DESIGN_KIT = {json.dumps(kit)};"


# Every generated root of one spec, rendered the way the page renders it, then rebuilt
# from nothing but the control records the render pushed.
_ROUND_TRIP = """
function roundTripRoot(root) {
  const target = rootTarget(root);
  const out = [];
  const html = root === 'rules' ? rulesCardsHtml(out) : rootHtml(root, target, out);
  // Read each path back the way a browser would: up to the closing quote, entities
  // decoded, then parsed. A quote left unescaped ends the attribute early.
  const decode = s => s.replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
  const drawn = new Set([...html.matchAll(/data-cpath="([^"]*)"/g)].map(m => {
    try { return JSON.stringify(JSON.parse(decode(m[1]))); } catch (e) { return null; }
  }));
  const missing = out.filter(c => !drawn.has(JSON.stringify(c.path)))
    .map(c => root + ' ' + c.path.join('.'));
  const rebuilt = mergeGenerated(JSON.parse(JSON.stringify(target)), buildControls(out),
                                 rootFields(root));
  Object.keys(target).forEach(key => { delete target[key]; });
  Object.assign(target, rebuilt);
  return missing;
}
function roundTrip(specs) {
  const result = {};
  for (const [name, spec] of Object.entries(specs)) {
    DRAFT = JSON.parse(JSON.stringify(spec));
    const roots = ['rules'];
    (DRAFT.pieces || []).forEach((piece, i) => {
      roots.push('piece.' + i);
      (piece.moves || []).forEach((_, j) => roots.push('move.' + i + '.' + j));
    });
    ((DRAFT.setup || {}).placements || []).forEach((_, i) => roots.push('place.' + i));
    const missing = roots.flatMap(roundTripRoot);
    result[name] = { spec: DRAFT, missing };
  }
  return result;
}
"""

# The hand-written board and move row, read the way a browser reads them: an input's
# value, and a menu's selected option or else its first. Each reading is written back
# through the form's own writer. A key the spec holds that no control draws is missing.
_HAND_TRIP = """
function readHand(html, attr) {
  const decode = s => s.replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  const named = tag => (new RegExp(attr + '="([^"]*)"').exec(tag) || [])[1];
  const out = [];
  for (const [, tag, body] of html.matchAll(/<select([^>]*)>([\\s\\S]*?)<\\/select>/g)) {
    if (named(tag) === undefined) continue;
    const options = [...body.matchAll(/<option value="([^"]*)"([^>]*)>/g)];
    const chosen = options.find(o => /\\bselected\\b/.test(o[2])) || options[0];
    out.push([named(tag), chosen ? decode(chosen[1]) : '']);
  }
  for (const [, tag] of html.matchAll(/<input([^>]*)>/g)) {
    if (named(tag) === undefined) continue;
    out.push([named(tag), decode((/\\svalue="([^"]*)"/.exec(tag) || [])[1] || '')]);
  }
  return out;
}
function handTrip(specs) {
  const result = {};
  const generated = new Set(rootFields('move.0.0').map(f => f.key));
  for (const [name, spec] of Object.entries(specs)) {
    DRAFT = JSON.parse(JSON.stringify(spec));
    const missing = [];
    const board = DRAFT.board;
    const boardControls = readHand(boardFormHtml(board), 'data-bfield');
    const written = new Set(boardControls.flatMap(([field]) => BOARD_FIELD_KEYS[field] || [field]));
    Object.keys(board).filter(key => !written.has(key))
      .forEach(key => missing.push('board.' + key));
    boardControls.forEach(([field, value]) => writeBoardField(board, field, value));
    DRAFT.pieces.forEach((piece, i) => (piece.moves || []).forEach((move, j) => {
      const controls = readHand(moveEditor(move, i, j), 'data-mfield');
      Object.keys(move).filter(key => !generated.has(key)
        && !controls.some(([field]) => field === key))
        .forEach(key => missing.push(`pieces.${i}.moves.${j}.${key}`));
      controls.forEach(([field, value]) => writeMoveField(move, field, value));
    }));
    result[name] = { spec: DRAFT, missing };
  }
  return result;
}
"""


class TestARuleIsOfferedInWords:
    """Stage 3, A8. A card's title and its first sentence are the package's, and every
    rule has both -- so the menu says what a rule does rather than naming its key."""

    def _kingly(self) -> dict:
        return next(t for t in designing.templates() if t["key"] == "kingly")["spec"]

    def test_a_card_is_titled_in_words_and_opens_with_a_sentence(self) -> None:
        spec = self._kingly()
        html = _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(spec)};", "rulesCardsHtml([])"
        )
        for key in spec["rules"]:
            title, says = describe.RULE_WORDS[key]
            assert f"<b>{_escaped(title)}</b>" in html, key
            assert _escaped(says) in html, key
            # Two rules are called what their key spells; the rest never show one.
            assert title == key or f"<b>{key}</b>" not in html, key

    def test_the_menu_offers_a_title_and_a_sentence_for_what_is_absent(self) -> None:
        spec = self._kingly()
        html = _run_page(_kit_setup() + f"DRAFT = {json.dumps(spec)};", "ruleMenuHtml()")
        for key in ("npcs", "territory", "wrap", "strike", "turn", "scoring"):
            title, says = describe.RULE_WORDS[key]
            # The title alone: the library's count is said under a chosen rule (Q995c).
            assert f">{_escaped(title)}</option>" in html, key
            assert _escaped(says) in html, key

    def test_every_rule_reaches_the_menu_or_a_card_by_its_title(self) -> None:
        """Both halves of the panel together name all twenty-eight, and none by key."""
        spec = self._kingly()
        drawn = _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(spec)};",
            "rulesCardsHtml([]) + ruleMenuHtml()",
        )
        for key in _RULES_KEYS:
            assert _escaped(describe.RULE_WORDS[key][0]) in drawn, key


class TestOneSpellingOfARuleAtATime:
    """A8's third part. The promotion card showed `rank`, `ranks` and `squares` at once
    with an *empty list* tick beside each; only one of the three is ever read."""

    def _card(self, promotion: dict) -> str:
        spec = next(t for t in designing.templates() if t["key"] == "kingly")["spec"]
        spec = json.loads(json.dumps(spec))
        spec["rules"]["promotion"] = promotion
        return _run_page(_kit_setup() + f"DRAFT = {json.dumps(spec)};",
                         "rulesCardsHtml([])")

    def test_only_the_spelling_in_use_is_drawn(self) -> None:
        html = self._card({"piece": "P", "rank": "last", "to": ["R"]})
        assert '"promotion","rank"' in html.replace("&quot;", '"')
        for absent in ('"promotion","ranks"', '"promotion","squares"'):
            assert absent not in html.replace("&quot;", '"')

    def test_the_spelling_the_compiler_reads_is_the_one_shown(self) -> None:
        """Both written: `squares` wins in `schema.promotion_squares`, so it is drawn."""
        html = self._card({"piece": "P", "rank": 0, "squares": ["a6"], "to": ["R"]})
        plain = html.replace("&quot;", '"')
        assert '"promotion","squares"' in plain
        assert '"promotion","rank"' not in plain

    def _switched(self, promotion: dict, group: int, option: int) -> dict:
        spec = next(t for t in designing.templates() if t["key"] == "kingly")["spec"]
        spec = json.loads(json.dumps(spec))
        spec["rules"]["promotion"] = promotion
        return _run_page(
            _kit_setup() + "drawDesigner = () => {}; checkDraft = () => {};"
            f"DRAFT = {json.dumps(spec)};"
            f" DRAFT_CHECK = {{ preview: {json.dumps(designing.check(spec)['preview'])} }};"
            f" switchChoice('rules', ['promotion'], {group}, {option});",
            "DRAFT",
        )

    def test_switching_a_spelling_drops_the_others(self) -> None:
        after = self._switched({"piece": "P", "rank": "last", "to": ["R"]}, 0, 1)
        assert "rank" not in after["rules"]["promotion"]

    def test_the_zone_stays_a_game_through_every_spelling(self) -> None:
        """A promotion zone nobody has filled in yet writes no key at all and falls back
        to the last rank, so a spelling can be tried without the draft going invalid
        under the hand."""
        for option in range(3):
            after = self._switched({"piece": "P", "rank": "last", "to": ["R"]}, 0, option)
            assert validate_spec(designing.settled(after)) == [], option

    def test_the_spelling_taken_is_the_one_drawn_before_it_is_filled_in(self) -> None:
        """It holds no key yet, so the pick is remembered rather than written."""
        spec = next(t for t in designing.templates() if t["key"] == "kingly")["spec"]
        spec = json.loads(json.dumps(spec))
        spec["rules"]["promotion"] = {"piece": "P", "rank": "last", "to": ["R"]}
        html = _run_page(
            _kit_setup() + "drawDesigner = () => {}; checkDraft = () => {};"
            f"DRAFT = {json.dumps(spec)};"
            f" DRAFT_CHECK = {{ preview: {json.dumps(designing.check(spec)['preview'])} }};"
            " switchChoice('rules', ['promotion'], 0, 2);",
            "rulesCardsHtml([])",
        )
        plain = html.replace("&quot;", '"')
        assert '"promotion","squares"' in plain
        assert '"promotion","rank"' not in plain

    def test_the_rules_own_settle_still_fills_what_it_knows(self) -> None:
        """Switching to *the same list for both players* leaves `to` holding the piece
        the draft promotes into, rather than the empty list a bare default would give."""
        after = self._switched({"piece": "P", "to_by_player": {"0": ["R"], "1": ["R"]}},
                               1, 0)
        promotion = after["rules"]["promotion"]
        assert promotion["to"] and "to_by_player" not in promotion
        assert validate_spec(designing.settled(after)) == []


class TestARuleThatDoesNotGoInThisGame:
    """A9. The validator is asked before the menu offers a rule, so the wolf's eight
    wasted actions become a sentence beside a rule that is not on offer."""

    def _candidates(self, spec: dict) -> dict:
        return _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(spec)};"
            f" DRAFT_CHECK = {{ preview: {json.dumps(designing.check(spec)['preview'])} }};",
            "ruleCandidates()",
        )

    def test_the_skirmish_cannot_take_the_rules_it_has_no_royalty_for(self) -> None:
        spec = next(t for t in designing.templates() if t["key"] == "skirmish")["spec"]
        answer = designing.refusals(spec, self._candidates(spec))
        assert answer["asked"] is True
        assert sorted(answer["refused"]) == sorted(_CANNOT["skirmish"])
        # Every reason, in the designer's words (Q995b): never the validator's path.
        for key, reasons in answer["refused"].items():
            assert isinstance(reasons, list) and reasons, key
            for line in reasons:
                assert not re.search(r"\brules\.[a-z_]+", line), (key, line)
        assert any("no piece is marked royal" in line
                   for line in answer["refused"]["forbid_check"])

    def test_a_draft_that_is_not_a_game_is_asked_nothing(self) -> None:
        """Every rule would be refused for the draft's own errors, which greys the whole
        menu and says nothing about any rule."""
        spec = next(t for t in designing.templates() if t["key"] == "kingly")["spec"]
        broken = json.loads(json.dumps(spec))
        broken["setup"]["placements"] = []
        answer = designing.refusals(broken, {"wrap": {"files": "both"}})
        assert answer == {"refused": {}, "asked": False}

    def test_a_refused_rule_leaves_the_menu_and_takes_its_reason_with_it(self) -> None:
        spec = next(t for t in designing.templates() if t["key"] == "skirmish")["spec"]
        drawn = _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(spec)};"
            " RULE_REFUSED = { forbid_check: ['no piece is marked royal'] };",
            "ruleMenuHtml()",
        )
        title = describe.RULE_WORDS["forbid_check"][0]
        assert f">{title} (" not in drawn
        assert "no piece is marked royal" in drawn
        assert "1 rule" in drawn


class TestAPieceOutOfTheLibrary:
    """D1. Every distinct piece the library holds, with its moves attached."""

    def _palette(self) -> list[dict]:
        return designing.piece_palette(cards.corpus_specs(RULESETS, ARCHIVE))

    def test_one_entry_per_distinct_piece_with_the_games_that_use_it(self) -> None:
        palette = self._palette()
        keys = [entry["key"] for entry in palette]
        assert len(keys) == len(set(keys))
        for entry in palette:
            assert entry["games"], entry["name"]
            assert len(entry["says"]) == len(entry["piece"]["moves"])
            assert entry["piece"]["moves"]

    def test_two_games_that_spell_a_piece_the_same_share_one_entry(self) -> None:
        """A palette row is a piece's mechanics, never its letter or its name, so the
        library's fifty-odd games collapse to far fewer distinct pieces."""
        palette = self._palette()
        specs = cards.corpus_specs(RULESETS, ARCHIVE)
        total = sum(len(spec.get("pieces") or []) for spec in specs)
        assert len(palette) < total / 3, (len(palette), total)
        mechanics = {
            json.dumps([entry["lattice"], entry["piece"]["moves"],
                        entry["piece"].get("royal"), entry["piece"].get("facing")],
                       sort_keys=True)
            for entry in palette
        }
        assert len(mechanics) == len(palette)
        assert [entry for entry in palette if len(entry["games"]) > 20]

    def test_what_a_piece_names_elsewhere_is_left_behind(self) -> None:
        """`captures`, `promoted_from` and `region` name pieces and squares of the game
        the piece came from, so a draft that took one whole would be invalid at once."""
        for entry in self._palette():
            assert not ({"captures", "promoted_from", "region"} & set(entry["piece"]))

    def test_a_hex_piece_is_offered_only_to_a_hex_draft(self) -> None:
        spec = next(t for t in designing.templates() if t["key"] == "kingly")["spec"]
        square = json.loads(json.dumps(spec))
        hexy = json.loads(json.dumps(spec))
        hexy["board"]["lattice"] = "hex"
        kit = _kit_setup(cards.corpus_specs(RULESETS, ARCHIVE))
        for draft, lattice in ((square, "square"), (hexy, "hex")):
            offers = _run_page(kit + f"DRAFT = {json.dumps(draft)};", "paletteOffers()")
            assert offers, lattice
            assert {entry["lattice"] for entry in offers} == {lattice}

    def test_taking_one_adds_it_with_a_letter_the_draft_has_free(self) -> None:
        spec = next(t for t in designing.templates() if t["key"] == "kingly")["spec"]
        kit = _kit_setup(cards.corpus_specs(RULESETS, ARCHIVE))
        palette = self._palette()
        taken = next(e for e in palette if e["lattice"] == "square"
                     and e["id"] in {p["id"] for p in spec["pieces"]})
        after = _run_page(
            kit + "drawDesigner = () => {}; checkDraft = () => {};"
            " loadMoveMap = () => {};"
            f"DRAFT = {json.dumps(spec)}; takePalettePiece({json.dumps(taken['key'])});",
            "DRAFT.pieces",
        )
        added = after[-1]
        assert added["moves"] == taken["piece"]["moves"]
        assert added["id"] != taken["id"]
        assert len({p["id"] for p in after}) == len(after)
        spec = json.loads(json.dumps(spec))
        spec["pieces"] = after
        assert validate_spec(designing.settled(spec)) == []


class TestOneNotationForADirection:
    """A landing's vectors used to be one box per pair with *add one* for the next, while
    the move row wrote the same thing as `1,0 1,1`. One notation, and the shape of the
    field says which -- a list whose items are pairs of numbers is a list of directions."""

    def _landing(self, vectors: list[list[int]]) -> str:
        spec = next(t for t in designing.templates() if t["key"] == "kingly")["spec"]
        spec = json.loads(json.dumps(spec))
        spec["rules"]["strike"] = {
            "effect": "demote", "chain": [{"from": "R", "to": "P"}],
            "position": "choose", "landing": {"vectors": vectors},
        }
        return _run_page(_kit_setup() + f"DRAFT = {json.dumps(spec)};",
                         "rulesCardsHtml([])")

    def test_a_vector_list_is_one_box_in_the_move_rows_notation(self) -> None:
        html = self._landing([[1, 0], [1, 1]])
        plain = html.replace("&quot;", '"')
        assert 'value="1,0 1,1"' in html
        assert '"strike","landing","vectors"' in plain
        # No box per pair and no *add one* for the next.
        assert 'data-cpush="["strike","landing","vectors"]"' not in plain
        assert '"strike","landing","vectors",0' not in plain

    def test_what_is_typed_comes_back_as_pairs(self) -> None:
        built = _run_page(
            _kit_setup() + "DRAFT = { pieces: [{ id: 'P' }], rules: {} };",
            "controlValue({ control: 'vectors', required: true,"
            " raw: { text: '1,0 0,1 -1,-1', empty: false } })",
        )
        assert built == [[1, 0], [0, 1], [-1, -1]]

    def test_a_token_that_is_not_a_pair_is_left_out(self) -> None:
        """Read exactly as `writeMoveField` reads the move row's box, so the same typing
        means the same thing in both places."""
        built = _run_page(
            _kit_setup() + "DRAFT = { pieces: [{ id: 'P' }], rules: {} };",
            "[controlValue({ control: 'vectors', required: true,"
            "   raw: { text: '1,0 nope 2', empty: false } }),"
            " (writeMoveField(MOVE = { kind: 'leap' }, 'vectors', '1,0 nope 2'),"
            "  MOVE.vectors)]",
        )
        assert built == [[[1, 0]], [[1, 0]]]


class TestTheStillBoardDrawsAHand:
    """A drops game's hand is part of its opening position. The workbench board drew only
    the squares, so crazyhouse looked like a game with no hand at all."""

    def _preview(self, stem: str) -> dict:
        return designing.check(load_spec(RULESETS / stem))["preview"]

    def test_a_game_with_no_drops_has_no_strip(self) -> None:
        preview = self._preview("chess.json")
        assert preview["handsUsed"] is False
        drawn = _run_page(
            f"const P = {json.dumps(preview)};", "handStripHtml(P)"
        )
        assert drawn == ""

    def test_a_hand_that_fills_from_captures_says_so_while_it_is_empty(self) -> None:
        preview = self._preview("crazyhouse.json")
        assert preview["handsUsed"] is True
        assert preview["handsFromCaptures"] is True
        assert not any(any(hand) for hand in preview["hands"])
        drawn = _run_page(f"const P = {json.dumps(preview)};", "handStripHtml(P)")
        assert "Both hands are empty" in drawn
        assert "played back onto the board" in drawn

    def test_a_stocked_hand_is_counted_out(self) -> None:
        preview = dict(self._preview("crazyhouse.json"))
        held = [0] * preview["nTypes"]
        held[0] = 3
        preview["hands"] = [held, [0] * preview["nTypes"]]
        drawn = _run_page(f"const P = {json.dumps(preview)};", "handStripHtml(P)")
        assert f"{preview['pieceIds'][0]} &times;3" in drawn
        assert preview["sides"][0] in drawn

    def test_the_lone_board_holds_nothing(self) -> None:
        """A diagram of what a descriptor means is one piece on an empty board."""
        preview = self._preview("crazyhouse.json")
        drawn = _run_page(
            f"const P = {json.dumps(preview)};",
            "handStripHtml(loneBoard(P, 0, P.board.find(v => v) || 1))",
        )
        assert drawn == ""


class TestRuleCardsComeFromTheSchema:
    """Phase 2: the cards are generated, and nothing in the schema is left without one."""

    def test_every_rule_key_has_a_field(self) -> None:
        """The next `SCHEMA_VERSION` bump cannot forget the designer."""
        model = cards.card_model()
        assert {f["key"] for f in model["rules"]} == set(_RULES_KEYS)
        assert set(model["ruleOrder"]) == set(_RULES_KEYS)

    def test_every_rule_key_is_drawn_as_a_card(self) -> None:
        html = _run_page(
            _kit_setup() + "DRAFT = { pieces: [{ id: 'K', name: 'king' }], rules: {} };"
            " cardModel().rules.forEach(f => { DRAFT.rules[f.key] = fieldDefault(f); });",
            "rulesCardsHtml([])",
        )
        for key in _RULES_KEYS:
            assert f'data-rule="{key}"' in html, key

    def test_every_other_key_is_the_hand_form_or_a_generated_field(self) -> None:
        """Parity for the objects the hand-written form shares with generated extras."""
        schema = rule_spec_schema()
        piece = schema["properties"]["pieces"]["items"]
        move = piece["properties"]["moves"]["items"]
        placement = schema["properties"]["setup"]["properties"]["placements"]["items"]
        model = cards.card_model()
        for name, node in (("piece", piece), ("move", move), ("placement", placement)):
            generated = {f["key"] for f in model[name]}
            owned = {key.split(".")[0] for key in cards.OWNED[name]}
            assert set(node["properties"]) == generated | owned, name
        # `condition` is shared: the move row owns `from_rank`, the card owns the rest.
        condition = next(f for f in model["move"] if f["key"] == "condition")
        assert condition["partial"]
        assert {f["key"] for f in condition["fields"]} == {"first_move", "region"}

    def test_a_string_field_nobody_classified_is_refused(self) -> None:
        with pytest.raises(KeyError, match="neither PIECE_PATHS nor TEXT_PATHS"):
            cards.field_for({"type": "string"}, "rules.somewhere_new", "somewhere_new", True)

    def test_enum_options_carry_the_schemas_own_sentences(self) -> None:
        model = cards.card_model()
        win = next(f for f in model["rules"] if f["key"] == "win")
        assert win["control"] == "menu"
        says = {o["value"]: o.get("says") for o in win["options"]}
        assert says["checkmate"] == "trap the royal piece."
        # A description that does not introduce every value stays one sentence for all.
        mode = next(f for f in model["rules"] if f["key"] == "territory")["fields"][0]
        assert all("says" not in o for o in mode["options"]) and mode["says"]

    def test_integers_are_bounded_by_the_schema(self) -> None:
        model = cards.card_model()
        limit = next(f for f in model["rules"] if f["key"] == "check_limit")
        assert (limit["control"], limit["minimum"], limit["maximum"]) == ("int", 1, 20)
        html = _run_page(
            _kit_setup() + "DRAFT = { pieces: [], rules: { win: 'none', check_limit: 3 } };",
            "rulesCardsHtml([])",
        )
        assert re.search(r'data-cctl="int"[^>]*min="1" max="20"', " ".join(html.split()))

    def test_rare_rules_are_the_ones_the_library_never_states(self) -> None:
        corpus = [load_spec(path) for path in LIBRARY]
        model = cards.card_model(corpus)
        counts = model["ruleCounts"]
        assert model["rare"] == [k for k in model["ruleOrder"] if counts[k] == 0]
        assert list(counts.values()) == sorted(counts.values(), reverse=True)

    def test_a_square_list_is_picked_on_the_board(self) -> None:
        picked = _run_page(
            _kit_setup() + "drawDesigner = () => {}; checkDraft = () => {}; drawBoardSide = () => {};"
            " DRAFT = { pieces: [], rules: { win: 'none', reach: { squares: ['e4'] } } };"
            " DRAFT_CHECK = { preview: { width: 8, height: 8 } };"
            " PICKING = { root: 'rules', path: ['reach', 'squares'], label: 'squares' };"
            f" pickSquare({_square('d4')}); pickSquare({_square('e4')});"
            f" pickSquare({_square('h8')});",
            "DRAFT.rules.reach.squares",
        )
        assert picked == ["d4", "h8"]


# Every rule added to every template, and every optional sub-card added inside a rule,
# through the page's own `fieldDefault`. The preview is the engine's opening position,
# which is where the corner piece and the free squares come from.
_ADD_RULES = """
function addEveryRule(templates) {
  const out = {};
  for (const [key, held] of Object.entries(templates)) {
    out[key] = {};
    for (const field of cardModel().rules) {
      DRAFT = JSON.parse(JSON.stringify(held.spec));
      DRAFT_CHECK = { preview: held.preview };
      DRAFT.rules[field.key] = fieldDefault(field, [field.key], 'rules');
      if (RULE_COMPANIONS[field.key]) RULE_COMPANIONS[field.key](DRAFT.rules);
      out[key][field.key] = DRAFT;
    }
  }
  return out;
}

function addEverySubCard(templates, adds) {
  const out = {};
  for (const [key, held] of Object.entries(templates)) {
    out[key] = {};
    for (const path of adds) {
      DRAFT = JSON.parse(JSON.stringify(held.spec));
      DRAFT_CHECK = { preview: held.preview };
      const rule = cardModel().rules.find(f => f.key === path[0]);
      DRAFT.rules[path[0]] = fieldDefault(rule, [path[0]], 'rules');
      const last = path[path.length - 1];
      if (typeof last === 'number') {
        const list = fieldAt('rules', path.slice(0, -1));
        getIn(DRAFT.rules, path.slice(0, -1))
          .push(fieldDefault(list.item, path, 'rules'));
      } else {
        addPart(DRAFT.rules, 'rules', path);
      }
      out[key][path.join('.')] = DRAFT;
    }
  }
  return out;
}
"""

# Where a rule cannot be made valid on a template, and the words that say why. Each is a
# fact about the game the template is -- no royal piece, nothing that promotes, one kind
# of piece -- and not about the card, which is the line the first slice draws.
# Schema 19 (Q771): a portal card arrives with no squares, because where a pair stands is
# P3's portal tool's to place on the board (`docs/archive/plan-portals.md` D2, Q772); until then
# the validator says so, on every template.
_PORTAL_SQUARES = "must be a square on the board"
_CANNOT = {
    "skirmish": {
        "bare_royal": "no piece is marked royal",
        "check_limit": "no piece is marked royal",
        "forbid_check": "no piece is marked royal",
        "en_passant": "requires rules.promotion",
        "promote_on_entry": "needs rules.promotion",
        "promotion": "may not include the promoting piece itself",
        "contacts": "an ordinary capture written the long way",
        "portals": _PORTAL_SQUARES,
    },
    "kingly": {
        "en_passant": "the promoting piece it is defined against, has none",
        "promote_on_entry": "nothing enters play except by moving",
        "portals": _PORTAL_SQUARES,
    },
    "race": {
        "bare_royal": "no piece is marked royal",
        "check_limit": "no piece is marked royal",
        "forbid_check": "no piece is marked royal",
        "en_passant": "requires rules.promotion",
        "promote_on_entry": "needs rules.promotion",
        "portals": _PORTAL_SQUARES,
    },
}

_SUB_CANNOT = {
    "skirmish": {
        "strike.chain": "points 'S' at itself",
        "contacts.landing": "an ordinary capture written the long way",
        "promotion": "may not include the promoting piece itself",
    },
    # Q538 refused a chain under the strike card's default effect `capture`; the chain
    # card now sets a transforming effect and `stay` as it arrives (Q541, `addPart`).
    "kingly": {},
    "race": {},
}


class TestANewCardArrivesValid:
    """2.11: castling arrived naming a common piece as king, a chain link pointed a piece
    at itself, a landing arrived as an empty pair. Each cost an error before the person
    had typed anything. A card now reads the draft for the part it needs filled."""

    def _templates(self) -> dict[str, dict]:
        return {
            t["key"]: {"spec": t["spec"], "preview": designing.check(t["spec"])["preview"]}
            for t in designing.templates()
        }

    def _setup(self) -> str:
        return _kit_setup() + f"const TEMPLATES = {json.dumps(self._templates())};"

    def test_every_rule_added_to_every_template(self) -> None:
        model = cards.card_model()
        addable = [k for k in model["ruleOrder"] if k not in model["fixed"]]
        built = _run_page(self._setup() + _ADD_RULES, "addEveryRule(TEMPLATES)")
        for template, rules in built.items():
            refused = {}
            for key in addable:
                problems = validate_spec(designing.settled(rules[key]))
                if problems:
                    refused[key] = problems[0]
            assert sorted(refused) == sorted(_CANNOT[template]), template
            for key, says in _CANNOT[template].items():
                assert says in refused[key], (template, key, refused[key])

    def test_every_sub_card_added_inside_its_rule(self) -> None:
        adds = [["strike", "chain"], ["strike", "landing"], ["contacts", "landing"],
                ["npcs", 1], ["promotion"]]
        built = _run_page(
            self._setup() + f"const ADDS = {json.dumps(adds)};" + _ADD_RULES,
            "addEverySubCard(TEMPLATES, ADDS)",
        )
        for template, made in built.items():
            refused = {}
            for where, spec in made.items():
                problems = validate_spec(designing.settled(spec))
                if problems:
                    refused[where] = problems[0]
            assert sorted(refused) == sorted(_SUB_CANNOT[template]), (template, refused)
            for where, says in _SUB_CANNOT[template].items():
                assert says in refused[where], (template, where, refused[where])

    def test_castling_names_the_royal_piece_and_a_piece_on_a_corner(self) -> None:
        """Neither letter is written down anywhere. The royal flag and the opening
        position say which pieces those are, so another game's letters carry over."""
        chess = load_spec(RULESETS / "chess.json")
        without = json.loads(json.dumps(chess))
        del without["rules"]["castling"]
        card = _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(without)};"
            f" DRAFT_CHECK = {json.dumps({'preview': designing.check(without)['preview']})};",
            "fieldDefault(cardModel().rules.find(f => f.key === 'castling'),"
            " ['castling'], 'rules')",
        )
        assert (card["king"], card["rook"]) == ("K", "R")

    def test_a_landing_arrives_as_the_step_the_draft_already_writes(self) -> None:
        """Not eight offsets worked out here: a hex board's neighbours are not the
        square board's, and the page is not where that is known."""
        hexy = load_spec(ARCHIVE / "hex-glinski-hexagon.json")
        royal = next(p for p in hexy["pieces"] if p.get("royal"))
        landing = _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(hexy)}; DRAFT_CHECK = null;",
            "fieldDefault(cardModel().rules.find(f => f.key === 'strike')"
            ".fields.find(f => f.key === 'landing'), ['strike', 'landing'], 'rules')",
        )
        assert landing["vectors"] == royal["moves"][0]["vectors"]
        assert landing.get("symmetry") == royal["moves"][0].get("symmetry")


class TestTheLibraryRoundTripsThroughTheForm:
    """Load a ruleset into the form, write it back out, and the spec is unchanged.

    The generated roots -- the rules, and the extras of every piece, move and placement --
    are rebuilt from the control records their render pushed, which is the same build
    `writeRoot` runs over live elements. Every record's path must also appear in the HTML.
    The hand-written fields are not rebuilt: they are only written when edited.
    """

    def test_every_committed_ruleset_round_trips(self) -> None:
        specs = {f"{p.parent.name}/{p.stem}": load_spec(p) for p in LIBRARY}
        setup = _kit_setup(list(specs.values())) + f"const SPECS = {json.dumps(specs)};"
        result = _run_page(setup + _ROUND_TRIP, "roundTrip(SPECS)")
        unchanged = [name for name, spec in specs.items() if result[name]["spec"] == spec]
        undrawn = {name: r["missing"] for name, r in result.items() if r["missing"]}
        assert not undrawn
        assert unchanged == list(specs), sorted(set(specs) - set(unchanged))
        assert len(unchanged) == len(LIBRARY) >= 60

    def test_every_hand_written_board_and_move_field_reads_back(self) -> None:
        """The kind menu, the rest of the move row and the board, read as a browser reads
        them and written back through the form's writers.

        A menu with no option for a value shows its first option, so a sweep drew as a
        slide and a hex board had no control at all. Both showcases are named, so the
        corpus losing either one fails here rather than passing on what is left.
        """
        specs = {f"{p.parent.name}/{p.stem}": load_spec(p) for p in LIBRARY}
        assert {"archive/chess-sweeping-bishops", "archive/pinwheel"} <= set(specs)
        setup = _kit_setup() + f"const SPECS = {json.dumps(specs)};"
        result = _run_page(setup + _HAND_TRIP, "handTrip(SPECS)")
        assert {"archive/hex-glinski-hexagon", "archive/pinwheel"} <= set(specs)
        assert {name: r["missing"] for name, r in result.items() if r["missing"]} == {}
        assert sorted(name for name, r in result.items() if r["spec"] != specs[name]) == []


# Every piece of every ruleset given a letter nothing else uses, through the page's own
# rename. The draft that comes back has to be the same game under new letters.
_RENAME = """
function freeLetter(used) {
  const pool = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
  return pool.find(c => !used.has(c)) || pool.flatMap(a => pool.map(b => a + b))
    .find(c => !used.has(c));
}
function renameEveryPiece(specs) {
  const out = {};
  for (const [name, spec] of Object.entries(specs)) {
    DRAFT = JSON.parse(JSON.stringify(spec));
    const used = new Set(DRAFT.pieces.map(p => p.id));
    DRAFT.pieces.forEach(piece => {
      const from = piece.id;
      used.delete(from);
      const to = freeLetter(used);
      used.add(to);
      piece.id = to;
      renamePiece(from, to);
    });
    out[name] = DRAFT;
  }
  return out;
}
"""


class TestAChangeCanBeTakenBack:
    """2.3: every experiment used to be made on the only copy. Snapshots of the whole
    draft, so a rule half added and the piece it named come back together."""

    _STEPS = (
        "drawDesigner = () => {}; checkDraft = () => {}; loadMoveMap = () => {};"
        " const seen = [];"
        " const note = () => seen.push(JSON.parse(JSON.stringify(DRAFT)));"
    )

    def test_a_change_is_undone_and_redone_whole(self) -> None:
        seen = _run_page(
            self._STEPS
            + " DRAFT = { name: 'a', pieces: [] }; snapshotDraft();"
            " DRAFT.name = 'b'; snapshotDraft();"
            " DRAFT.pieces.push({ id: 'P' }); snapshotDraft();"
            " stepDraft(false); note(); stepDraft(false); note(); stepDraft(true); note();",
            "seen",
        )
        assert [draft["name"] for draft in seen] == ["b", "a", "b"]
        assert [len(draft["pieces"]) for draft in seen] == [0, 0, 0]

    def test_a_change_after_an_undo_drops_what_was_ahead(self) -> None:
        ahead = _run_page(
            self._STEPS
            + " DRAFT = { name: 'a' }; snapshotDraft(); DRAFT.name = 'b'; snapshotDraft();"
            " stepDraft(false); DRAFT.name = 'c'; snapshotDraft();",
            "[DRAFT.name, DRAFT_REDO.length, DRAFT_UNDO.length]",
        )
        assert ahead == ["c", 0, 1]

    def test_the_draft_waits_out_a_reload(self) -> None:
        held = _run_page(
            "const store = {};"
            " window.sessionStorage = { setItem: (k, v) => { store[k] = v; },"
            " getItem: k => (k in store ? store[k] : null) };"
            " DRAFT = { name: 'kept', pieces: [] }; DRAFT_FROM = 'abc'; keepDraft();"
            " DRAFT = null; DRAFT_FROM = '';",
            "heldDraft()",
        )
        assert held["spec"]["name"] == "kept" and held["from"] == "abc"

    def test_storage_that_is_not_offered_is_not_a_failure(self) -> None:
        """A private window, or a reader who blocked it: the editor works without it."""
        held = _run_page(
            "window.sessionStorage = { setItem: () => { throw new Error('no'); },"
            " getItem: () => { throw new Error('no'); } };"
            " DRAFT = { name: 'x' }; keepDraft();",
            "heldDraft()",
        )
        assert held is None

    def test_the_template_menu_asks_nothing(self) -> None:
        """Undo covers it, so the dialog in front of a template is gone."""
        from page_text import PAGE

        assert "Start again from" not in PAGE
        # Three dialogs on the whole page, and each is in front of something with no way
        # back: the corpus replay, which spends minutes, deleting a game's rules (Q303),
        # and (Q993b) leaving a try at a proven position before its end, which abandons
        # the game. A fourth would be a dialog in front of something undo already covers.
        assert PAGE.count("confirm(") == 3
        assert "Delete ' + game.name + ' for good?" in PAGE


class TestAChangedLetterIsCarried:
    """2.2: a letter is a piece's identity, and every rule that names the piece names the
    letter. The form carries a change of it now, so neither reproduction in 1.2 dead-ends."""

    def _renamed(self, spec: dict, index: int, letter: str) -> dict:
        return _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(spec)};"
            f" const piece = DRAFT.pieces[{index}], from = piece.id;"
            f" piece.id = {json.dumps(letter)}; renamePiece(from, piece.id);",
            "DRAFT",
        )

    def test_the_skirmishs_soldier_can_be_renamed(self) -> None:
        """1.2's first reproduction: S becomes R, and the placement follows it."""
        skirmish = next(t for t in designing.templates() if t["key"] == "skirmish")
        renamed = self._renamed(skirmish["spec"], 0, "R")
        assert renamed["setup"]["placements"][0]["piece"] == "R"
        assert validate_spec(designing.settled(renamed)) == []

    def test_the_placement_menu_offers_the_new_letter(self) -> None:
        """The second reproduction was the menu, not the spec: the row could not be
        pointed at the letter the piece now has."""
        skirmish = next(t for t in designing.templates() if t["key"] == "skirmish")
        renamed = self._renamed(skirmish["spec"], 0, "R")
        html = _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(renamed)};",
            "placementEditor(DRAFT.setup.placements[0], 0)",
        )
        assert re.search(r'<option value="R" selected', " ".join(html.split()))
        assert 'value="S"' not in html

    def test_the_rules_that_name_a_piece_follow_it(self) -> None:
        chess = load_spec(RULESETS / "chess.json")
        pawn = self._renamed(chess, 0, "Z")
        assert pawn["rules"]["promotion"]["piece"] == "Z"
        king = self._renamed(chess, 5, "X")
        assert king["rules"]["castling"]["king"] == "X"
        assert king["rules"]["win"] == "checkmate"
        for renamed in (pawn, king):
            assert validate_spec(designing.settled(renamed)) == []

    def test_every_library_game_survives_every_letter_changing(self) -> None:
        """The tree says which fields hold a piece, so this covers captures, chains,
        promotion lists, immunities and the drawings filed under a letter at once."""
        from chess_variant_sandbox.engine import Engine

        specs = {f"{p.parent.name}/{p.stem}": load_spec(p) for p in LIBRARY}
        setup = _kit_setup() + f"const SPECS = {json.dumps(specs)};"
        renamed = _run_page(setup + _RENAME, "renameEveryPiece(SPECS)")
        for name, spec in specs.items():
            after = renamed[name]
            assert [p["id"] for p in after["pieces"]] != [p["id"] for p in spec["pieces"]]
            assert validate_spec(designing.settled(after)) == [], name
        # And it is the same game under other letters. Compiled for the rules that name
        # a piece hardest -- a capture matrix, a strike chain, wild pieces, drops and a
        # promotion list -- rather than for all sixty, which costs six seconds.
        played = ["rulesets/intransitive", "archive/hierarchy-fall", "archive/wolves",
                  "rulesets/shogi", "rulesets/chess"]
        assert set(played) <= set(specs)
        for name in played:
            before, now = Engine(specs[name]), Engine(designing.settled(renamed[name]))
            # The pieces are declared in the same order, so nothing moved on the board.
            assert now.initial_state().board == before.initial_state().board, name
            assert len(now.legal_moves(now.initial_state())) == len(
                before.legal_moves(before.initial_state())
            ), name


class TestTheFormOffersWhatTheSchemaAccepts:
    """Each hand-written menu is the schema's own list, so the next kind is not forgotten."""

    def test_each_menu_holds_exactly_the_schemas_values(self) -> None:
        from chess_variant_sandbox.spec.schema import (
            HEX_SYMMETRIES,
            LATTICES,
            MODES,
            MOVE_KINDS,
            SQUARE_SYMMETRIES,
            SYMMETRIES,
        )

        html = _run_page(
            _kit_setup() + "DRAFT = { pieces: [] };",
            "moveEditor({ kind: 'slide', vectors: [[0, 1]] }, 0, 0)"
            " + boardFormHtml({ width: 8, height: 8 })",
        )
        # Sets, because a menu leads with its default and the schema's tuple need not.
        menus = {
            name: set(re.findall(r'<option value="([^"]*)"', body))
            for name, body in re.findall(
                r'<select[^>]*data-[mb]field="(\w+)"[^>]*>(.*?)</select>', html, re.S
            )
        }
        # The *turned* menu offers each lattice its own (schema 18b), so the schema's
        # whole list is the square board's menu and the hex board's together.
        hexed = _run_page(
            _kit_setup() + "DRAFT = { pieces: [], board: { width: 7, height: 7, lattice: 'hex' } };",
            "moveEditor({ kind: 'slide', vectors: [[0, 1]] }, 0, 0)",
        )
        hex_turns = set(re.findall(
            r'<option value="([^"]*)"',
            re.search(r'<select[^>]*data-mfield="symmetry"[^>]*>(.*?)</select>', hexed, re.S)
            .group(1),
        ))
        assert menus == {
            "kind": set(MOVE_KINDS), "symmetry": set(SQUARE_SYMMETRIES),
            "mode": set(MODES), "lattice": set(LATTICES),
        }
        assert hex_turns == set(HEX_SYMMETRIES)
        assert menus["symmetry"] | hex_turns == set(SYMMETRIES)


    def test_a_hex_draft_is_offered_no_square_turn(self) -> None:
        """Q681: the square's turns are refused on hexagons. A hex draft's *turned* menu
        offers *as written* and the hexagon's two (schema 18b), plus a move's own turn
        while it still has one, so the menu shows what is there; a square draft, the
        same the other way round."""

        def turned(board: str, move: str) -> list[str]:
            html = _run_page(_kit_setup() + f"DRAFT = {{ pieces: [], board: {board} }};",
                             f"moveEditor({move}, 0, 0)")
            body = re.search(r'<select[^>]*data-mfield="symmetry"[^>]*>(.*?)</select>',
                             html, re.S)
            assert body is not None
            return re.findall(r'<option value="([^"]*)"', body.group(1))

        hexes = "{ width: 7, height: 7, lattice: 'hex' }"
        assert turned(hexes, "{ kind: 'slide', vectors: [[0, 1]] }") == [
            "none", "rotate6", "dihedral12"]
        assert turned(hexes, "{ kind: 'slide', vectors: [[0, 1]], symmetry: 'rotate4' }") == [
            "none", "rotate4", "rotate6", "dihedral12"]
        squares = "{ width: 8, height: 8 }"
        assert turned(squares, "{ kind: 'slide', vectors: [[0, 1]] }") == [
            "none", "lateral", "rotate4", "dihedral8"]
        assert turned(squares, "{ kind: 'slide', vectors: [[0, 1]], symmetry: 'rotate6' }") == [
            "none", "lateral", "rotate4", "dihedral8", "rotate6"]

def _replayed(engine: Any, moves: list[int]) -> Any:
    state = engine.initial_state()
    for move in moves:
        state = engine.apply(state, move)
    return state


class TestShowMePlaysTheRuleAtWork:
    """Phase 3: the engine plays games, and the demonstration stops where a rule mattered."""

    def test_the_last_move_shown_is_illegal_without_the_rule(self) -> None:
        from chess_variant_sandbox.engine import Engine

        chess = load_spec(RULESETS / "chess.json")
        demo = designing.demonstrate(chess, "en_passant")
        moves = demo["moves"]
        assert moves and len(moves) - demo["start"] <= designing.SHOW_BEFORE + 1
        without = Engine(designing.without_rule(designing.settled(chess), "en_passant"))
        assert moves[-1] not in without.legal_moves(_replayed(without, moves[:-1]))
        assert "only legal because of this rule" in demo["says"]

    def test_the_same_draft_shows_the_same_moves(self) -> None:
        chess = load_spec(RULESETS / "chess.json")
        assert designing.demonstrate(chess, "castling") == designing.demonstrate(
            chess, "castling"
        )

    def test_a_rule_that_changes_nothing_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Searching fewer games reaches the same refusal: this is about what is said once
        none of them showed the rule, not about how many were searched (Q377)."""
        monkeypatch.setattr(designing, "SHOW_GAMES", 4)
        chess = load_spec(RULESETS / "chess.json")
        with pytest.raises(designing.DesignError, match="never changed a move"):
            designing.demonstrate(chess, "stalemate")

    def test_a_rule_the_draft_cannot_lose_says_so(self) -> None:
        chess = load_spec(RULESETS / "chess.json")
        with pytest.raises(designing.DesignError, match="not a game"):
            designing.demonstrate(chess, "promotion")
        with pytest.raises(designing.DesignError, match="no 'pieces played back onto the board' rule"):
            designing.demonstrate(chess, "drops")

    def test_every_removable_card_has_a_show_me_and_win_has_none(self) -> None:
        chess = load_spec(RULESETS / "chess.json")
        html = _run_page(_kit_setup() + f"DRAFT = {json.dumps(chess)};", "rulesCardsHtml([])")
        for key in chess["rules"]:
            assert (f'data-cshow="{key}"' in html) == (key != "win"), key


class TestTheUnfinishedGameIsOffered:
    def test_the_board_opens_where_the_game_ended(self) -> None:
        from chess_variant_sandbox.engine import Engine

        dull = load_spec(RULESETS / "dull.json")
        found = designing.unfinished(dull)
        engine = Engine(designing.settled(dull))
        assert engine.terminal(_replayed(engine, found["moves"])) is not None
        assert found["start"] == len(found["moves"])
        assert found == designing.unfinished(dull)

    def test_the_gate_offers_it_only_when_the_draft_does_not_finish(self) -> None:
        setup = (
            "DRAFT = { name: 'x', pieces: [] }; DRAFT_MAP = null; DRAWING = null;"
            " const side = { innerHTML: '' }; document.getElementById ="
            " id => id === 'verdictSide' ? side : { onclick: null };"
            " drawBoardSide = () => {}; identityHtml = () => ''; wireBoards = () => {};"
        )
        failing = "{ valid: true, playable: false, playableReasons: ['r'], ruleLines: [] }"
        passing = "{ valid: true, playable: true, openingMoves: 20, ruleLines: [] }"
        shown = _run_page(
            setup,
            f"[{failing}, {passing}].map(c => {{ DRAFT_CHECK = c; drawVerdict();"
            " return side.innerHTML; })",
        )
        assert 'id="dStuck"' in shown[0] and 'id="dStuck"' not in shown[1]
        assert all('id="dWatch"' in html for html in shown)


class TestTheCheckSaysWhatToDo:
    """2.9: the editor listed a stamp it has no control for beside every real error, and
    called two different counters one clock."""

    def _skirmish(self) -> dict:
        template = next(t for t in designing.templates() if t["key"] == "skirmish")
        return json.loads(json.dumps(template["spec"]))

    def test_the_version_sentence_goes_while_another_error_stands(self) -> None:
        """1.2's third error: a capture list naming a piece that no longer exists, told
        alongside *declare "schema_version": 3*. The form has no version control."""
        draft = self._skirmish()
        draft["schema_version"] = 1
        draft["pieces"][0]["captures"] = ["A"]
        raw = validate_spec(designing.settled(draft))
        assert any('"schema_version":' in problem for problem in raw)
        errors = designing.check(draft)["errors"]
        assert errors
        assert not any('"schema_version":' in error for error in errors)
        assert "Piece 1 (soldier), 'may take only': 'A' isn't one of the game's pieces." in errors

    def test_a_stamp_the_draft_cannot_settle_still_stands(self) -> None:
        """Alone it is the refusal itself, so it is dropped rather than suppressed."""
        only_version = ["the spec uses captures, added in schema 3; declare "
                        '"schema_version": 3 to use it']
        assert designing._worth_saying(only_version) == only_version
        assert designing._worth_saying([*only_version, "a real one"]) == ["a real one"]

    def test_run_out_of_clock_names_the_counter_that_fired(self) -> None:
        """Walk 1.3 raised the move limit twice before finding it was the other one."""
        stalling = {"no_progress_draw": 2}, {"max_plies": 4, "no_progress_draw": 1000}
        said = []
        for tweak in stalling:
            draft = self._skirmish()
            draft["rules"].update(tweak)
            reasons = designing.check(draft)["playableReasons"]
            said.append(next(r for r in reasons if gates.STALLED in r))
        assert "were stopped by the no-progress rule (now 2 turns)" in said[0], said[0]
        assert "were stopped by the turn limit (now 4 turns)" in said[1], said[1]
        # The counter that did not fire is not named, so there is one number to change.
        counted = [line.split(". Of ", 1)[1] for line in said]
        assert "turn limit" not in counted[0] and "no-progress rule" not in counted[1]

    def test_a_reason_that_is_not_about_the_clock_is_left_alone(self) -> None:
        from chess_variant_sandbox.engine import Engine

        dull = designing.settled(load_spec(RULESETS / "dull.json"))
        plain = "one side wins 90% more often than the other even when both move at random"
        assert designing._with_the_counter_named(plain, Engine(dull), dull) == plain


class TestADraftOpensPartWayIn:
    def test_the_server_replays_the_demonstration_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        from chess_variant_sandbox.web.server import App

        chess = load_spec(RULESETS / "chess.json")
        app = App(RULESETS, tmp_path)
        shown = app.post("/api/design/play", {"spec": chess, "show": {"rule": "en_passant"}})
        assert shown["seats"] == ["human", "human"] and shown["ruleset"]["scratch"]
        assert len(shown["replay"]) == shown["ply"] + 1
        assert shown["replay"][-1]["board"] == shown["board"]
        assert shown["demo"]["start"] == max(0, shown["ply"] - designing.SHOW_BEFORE - 1)
        with pytest.raises(ValueError, match="show must be"):
            app.post("/api/design/play", {"spec": chess, "show": "everything"})
        assert not any(tmp_path.iterdir())

    def test_the_board_steps_through_every_replayed_beat(self) -> None:
        snapshot = {
            "id": "s",
            "board": [0, 2, 1],
            "history": [
                {"text": "a", "phase": []},
                {"text": "b", "phase": [{"from": 0, "to": 1, "eaten": 0}]},
            ],
            "replay": [
                {"board": [1, 0, 0], "colour": [], "hands": [], "sideToMove": 0},
                {"board": [0, 1, 0], "colour": [], "hands": [], "sideToMove": 1},
                {"board": [0, 2, 1], "colour": [], "hands": [], "sideToMove": 0},
            ],
        }
        beats = _run_page(
            f"record({json.dumps(snapshot)});",
            "TIMELINE.map(b => [b.kind, b.plies, b.board])",
        )
        assert [kind for kind, _, _ in beats] == ["start", "move", "move", "phase"]
        assert [plies for _, plies, _ in beats] == [0, 1, 2, 2]
        # The wild-piece phase is unmade from the ply's end, so its first beat stands between.
        assert beats[2][2] == [2, 0, 1] and beats[3][2] == [0, 2, 1]


class TestTheMoveRowHasTheFieldsThePawnNeeds:
    def test_min_range_and_from_rank_are_shown_from_the_spec(self) -> None:
        chess = load_spec(RULESETS / "chess.json")
        html = _run_page(
            f"DRAFT = {json.dumps(chess)}; DRAFT_PIECE = 0;",
            "moveEditor(DRAFT.pieces[0].moves[1], 0, 1)",
        )
        assert re.search(r'data-mfield="min_range"[^>]*value="2"', " ".join(html.split()))
        # Counted from one on screen, as the board labels ranks: the spec's 1 is rank 2.
        assert re.search(r'data-mfield="from_rank"[^>]*value="2"', " ".join(html.split()))


# ------------------------------------------------------- starting from a game that exists
#
# `docs/archive/plan-game-graph.md` G6. Two ways of making a draft out of games that are
# already in the library, both of them the server's answer and neither of them the page's.

_REMIX: dict[str, Any] = {
    "schema_version": 2,
    "name": "remix base",
    "board": {"width": 6, "height": 6},
    "pieces": [
        {"id": "K", "name": "chief", "royal": True,
         "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
        {"id": "W", "name": "walker", "value": 1,
         "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 2, "mode": "move"},
                   {"kind": "leap", "vectors": [[1, 1], [-1, 1]], "mode": "capture"}]},
        {"id": "R", "name": "tower", "value": 5,
         "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
    ],
    "setup": {"mirror": True, "placements": [
        {"piece": "K", "squares": ["c1"]},
        {"piece": "R", "squares": ["a1", "f1"]},
        {"piece": "W", "squares": ["a2", "b2", "c2", "d2", "e2", "f2"]},
    ]},
    "rules": {"win": "checkmate"},
}


def _remix_spec(**rules: Any) -> dict[str, Any]:
    made = json.loads(json.dumps(_REMIX))
    made["rules"].update(rules)
    return made


def _relettered(spec: dict[str, Any], swap: dict[str, str]) -> dict[str, Any]:
    """The same game under other letters, which is the same game and another spelling."""
    made = json.loads(json.dumps(spec))
    for piece in made["pieces"]:
        piece["id"] = swap.get(piece["id"], piece["id"])
    for row in made["setup"]["placements"]:
        row["piece"] = swap.get(row["piece"], row["piece"])
    return made


class TestTheDesignStepsOnOffer:
    """*And now try*: which of the invention loop's own edits make a game out of this one.

    The words are the operator's own and the draw is seeded, so the list a person reads
    and the draft the press gives them are the same child.
    """

    def test_every_step_listed_makes_a_valid_game_that_is_not_this_one(self) -> None:
        from chess_variant_sandbox.spec import relate

        spec = _remix_spec()
        steps = designing.design_steps(spec)
        assert len(steps) > 5, "the operator set should reach a game this ordinary"
        mine = relate.identity_key(spec)
        for step in steps:
            made = designing.remix(spec, step["operator"], step["seed"])
            assert not validate_spec(made["spec"])
            assert relate.identity_key(made["spec"]) != mine

    def test_the_same_game_offers_the_same_steps_twice(self) -> None:
        """Seeds are fixed, so a second press is not a second lottery."""
        spec = _remix_spec()
        assert designing.design_steps(spec) == designing.design_steps(spec)

    def test_a_step_drawn_again_is_the_child_that_was_listed(self) -> None:
        spec = _remix_spec()
        step = designing.design_steps(spec)[0]
        first = designing.remix(spec, step["operator"], step["seed"])
        again = designing.remix(spec, step["operator"], step["seed"])
        assert first["hash"] == again["hash"]
        assert first["words"] == step["words"]

    def test_the_draft_says_in_words_what_was_changed(self) -> None:
        spec = _remix_spec()
        step = designing.design_steps(spec)[0]
        made = designing.remix(spec, step["operator"], step["seed"], name="the parent")
        # The operator's own sentence, word for word, in the draft itself rather than
        # only on the page, so what is saved says what it is.
        assert step["words"] in made["spec"]["description"]
        assert "Made from the parent in one design step." in made["spec"]["description"]
        assert made["spec"]["name"] == "the parent variant"

    def test_the_draft_is_a_game_that_opens(self) -> None:
        spec = _remix_spec()
        step = designing.design_steps(spec)[0]
        made = designing.remix(spec, step["operator"], step["seed"])
        assert made["openingMoves"] > 0
        assert designing.check(made["spec"])["valid"]

    def test_the_parents_own_parents_do_not_travel(self) -> None:
        spec = _remix_spec()
        spec["provenance"] = {"parents": ["0123456789ab"]}
        step = designing.design_steps(spec)[0]
        made = designing.remix(spec, step["operator"], step["seed"])
        assert "provenance" not in made["spec"]

    def test_a_step_nobody_offers_is_refused(self) -> None:
        with pytest.raises(designing.DesignError, match="no design step called"):
            designing.remix(_remix_spec(), "turn-it-inside-out", 0)

    def test_a_step_that_does_not_apply_here_is_refused(self) -> None:
        """A list older than the game it was taken on, which is what a stale page holds."""
        from chess_variant_sandbox.loop.operators import OPERATORS

        spec = _remix_spec()
        offered = {step["operator"] for step in designing.design_steps(spec)}
        missing = [one.name for one in OPERATORS if one.name not in offered]
        assert missing, "every operator applied, so there is no refusal to check"
        with pytest.raises(designing.DesignError, match="doesn't apply to this draft any more"):
            designing.remix(spec, missing[0], 0)


class TestTheDesignStepsOnADraft:
    """Stage 4, C1 and C2. The same steps, offered on the draft in the editor.

    A draft is not a game the library holds: it has no hash anybody could name, it is
    usually somebody's unfinished spelling of something, and it may not validate at all.
    The two differences from a step pressed on a saved game are held here, and everything
    else is `design_steps` and `remix` unchanged.
    """

    def test_a_draft_is_offered_the_steps_that_apply_to_it(self) -> None:
        spec = _remix_spec()
        answer = designing.draft_steps(spec)
        assert answer["why"] == ""
        assert [step["operator"] for step in answer["steps"]] == [
            step["operator"] for step in designing.design_steps(spec)
        ]

    def test_every_template_offers_steps_that_are_games(self) -> None:
        """The three starting points a newcomer meets, each with something to press."""
        for template in designing.templates():
            answer = designing.draft_steps(template["spec"])
            assert len(answer["steps"]) > 5, template["key"]
            for step in answer["steps"]:
                made = designing.draft_remix(
                    template["spec"], step["operator"], step["seed"]
                )
                assert not validate_spec(made["spec"]), step["operator"]
                assert made["openingMoves"] > 0, step["operator"]

    def test_a_draft_that_is_not_a_game_says_why_rather_than_offering_nothing(
        self,
    ) -> None:
        """An operator edits a spec the validator accepts, so on a broken draft every
        draw fails -- and a bare empty list would read as *this game has no changes*."""
        broken = _remix_spec()
        broken["setup"]["placements"][0]["piece"] = "Q"
        answer = designing.draft_steps(broken)
        assert answer["steps"] == []
        assert answer["why"] == "Where they start, row 1: 'Q' isn't one of the game's pieces."

    def test_the_same_press_on_the_same_draft_makes_the_same_game(self) -> None:
        """Seeds are fixed, so pressing a step is not a lottery and is not a clock."""
        spec = _remix_spec()
        step = designing.draft_steps(spec)["steps"][0]
        first = designing.draft_remix(spec, step["operator"], step["seed"])
        again = designing.draft_remix(spec, step["operator"], step["seed"])
        assert first["hash"] == again["hash"]
        assert first["words"] == step["words"] == again["words"]

    def test_a_step_pressed_on_a_draft_keeps_the_drafts_own_name(self) -> None:
        """Three presses on a saved game's remix would otherwise leave a draft called
        *chess variant variant variant*."""
        spec = _remix_spec()
        spec["name"] = "my game"
        spec["description"] = "what I am trying"
        step = designing.draft_steps(spec)["steps"][0]
        made = designing.draft_remix(spec, step["operator"], step["seed"])
        assert made["spec"]["name"] == "my game"
        assert made["spec"]["description"] == "what I am trying"
        # What changed is still said, and it is the operator's own sentence.
        assert made["words"] == step["words"]

    def test_pressing_twice_walks_rather_than_renaming(self) -> None:
        """A step's child is a different game from the draft it was pressed on, and that
        is the whole promise. Whether it may be the game one press back is
        `test_a_step_does_not_walk_back_to_the_draft_it_was_pressed_on`'s."""
        spec = _remix_spec()
        spec["name"] = "my game"
        first = designing.draft_remix(spec, *_first_step(spec))
        second = designing.draft_remix(first["spec"], *_first_step(first["spec"]))
        assert second["spec"]["name"] == "my game"
        assert second["hash"] != first["hash"]

    def test_a_step_does_not_walk_back_to_the_draft_it_was_pressed_on(self) -> None:
        """Q532. Several operators toggle a rule -- one puts a wall up, and pressed on the
        walled game takes it down -- so the list offered right after a press used to hold
        a step back to the draft that press was made on: undo, under another name. The
        editor sends that draft as `away_from`, and no step offered lands on it."""
        from chess_variant_sandbox.loop.operators import OPERATORS
        from chess_variant_sandbox.spec import relate

        by_name = {operator.name: operator for operator in OPERATORS}
        spec = designing.settled(_remix_spec())
        start = relate.identity_key(spec)

        def lands_back(made: dict[str, Any], steps: list[dict[str, Any]]) -> list[str]:
            back = []
            for step in steps:
                drawn = designing._drawn(made, by_name[step["operator"]], step["seed"], set())
                if drawn is not None and relate.identity_key(drawn[1]) == start:
                    back.append(step["operator"])
            return back

        walked = 0
        for step in designing.draft_steps(spec)["steps"]:
            # The child the press makes, uncompiled: `remix` is this draw and a compile.
            drawn = designing._drawn(spec, by_name[step["operator"]], step["seed"], {start})
            assert drawn is not None
            made = drawn[1]
            # Without the draft it came from, a step on the list walks back...
            if not lands_back(made, designing.draft_steps(made)["steps"]):
                continue
            walked += 1
            # ...and with it, none does.
            assert lands_back(made, designing.draft_steps(made, [spec])["steps"]) == [], (
                step["operator"]
            )
            if walked == 3:
                break
        assert walked, "no step on this spec toggles, so the test proves nothing"

    def test_a_further_step_is_added_to_a_remixs_description(self) -> None:
        """Q532. A draft remixed from a saved game says *Made from X in one design step*
        and what that step did; a second press kept that sentence as it was, so the draft
        described the first change alone. The frame is this module's own sentence, so it
        is read back and counted. A description somebody wrote is theirs and is kept."""
        spec = _remix_spec()
        first = designing.design_steps(spec)[0]
        made = designing.remix(spec, first["operator"], first["seed"], name="base")["spec"]
        second = designing.draft_steps(made, [spec])["steps"][0]
        again = designing.draft_remix(made, second["operator"], second["seed"])["spec"]
        assert again["description"] == (
            f"Made from base in two design steps. {first['words']}. {second['words']}."
        )

        assert designing._one_step_further(
            "Made from base in nine design steps. a. b.", "c"
        ).startswith("Made from base in ten design steps. a. b. c.")
        assert designing._one_step_further(
            "Made from base in ten design steps. a.", "b"
        ) == "Made from base in 11 design steps. a. b."
        assert designing._one_step_further(
            "Made from base in 11 design steps. a.", "b"
        ) == "Made from base in 12 design steps. a. b."
        assert designing._one_step_further("my own words", "b") == "my own words"
        assert designing._one_step_further("", "b") == ""

    def test_the_route_passes_the_draft_to_walk_away_from(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The editor's `back` is the draft its last press was made on."""
        from chess_variant_sandbox.web.server import App

        seen: list[Any] = []
        monkeypatch.setattr(
            designing, "draft_steps",
            lambda spec, away_from=(): seen.append(away_from) or {"steps": [], "why": ""},
        )
        # The handler reads nothing off the app, so it is called unbound.
        App._draft_steps_or_one(None, {"spec": {"a": 1}, "back": {"b": 2}})  # type: ignore[arg-type]
        App._draft_steps_or_one(None, {"spec": {"a": 1}})  # type: ignore[arg-type]
        assert seen == [[{"b": 2}], []]

    def test_a_saved_games_remix_still_names_itself_after_its_parent(self) -> None:
        """The map's *and now try* is the other caller and nothing here moved it."""
        spec = _remix_spec()
        step = designing.design_steps(spec)[0]
        made = designing.remix(spec, step["operator"], step["seed"], name="chess")
        assert made["spec"]["name"] == "chess variant"
        assert "Made from chess in one design step." in made["spec"]["description"]

    def test_the_draft_has_to_be_an_object(self) -> None:
        for call in (
            lambda: designing.draft_steps("chess"),
            lambda: designing.draft_remix("chess", "join-the-edges", 0),
        ):
            with pytest.raises(designing.DesignError, match="could not be read"):
                call()


def _first_step(spec: dict[str, Any]) -> tuple[str, int]:
    step = designing.draft_steps(spec)["steps"][0]
    return step["operator"], step["seed"]


class TestAndNowTryInTheEditor:
    """The panel: what it lists, what a press does to the draft, and taking it back.

    Run against the page's own functions, with `api` standing in for the server -- what
    the server answers is `TestTheDesignStepsOnADraft`'s.
    """

    STEPS = [
        {"operator": "join-the-edges", "seed": 3, "words": "the left and right edges "
         "are joined, so a piece leaving one arrives at the other"},
        {"operator": "soften-the-capture", "seed": 0, "words": "the R may no longer "
         "take the P"},
    ]

    def _panel(self, held: dict[str, Any], said: str = "null") -> str:
        setup = (
            "DRAFT = { name: 'a draft', pieces: [] };"
            f"TRY_OPEN = true; TRY = {json.dumps(held)}; TRY_SAID = {said};"
        )
        return str(_run_page(setup, "tryHtml()"))

    def test_before_it_is_asked_the_panel_is_one_button(self) -> None:
        drawn = str(_run_page("DRAFT = { name: 'a draft' }; TRY_OPEN = false;",
                              "tryHtml()"))
        assert 'id="dTryAsk"' in drawn
        assert "data-trystep" not in drawn

    def test_each_step_is_a_button_carrying_the_operators_own_sentence(self) -> None:
        drawn = self._panel(
            {"state": "ready", "steps": self.STEPS, "why": "", "failed": "",
             "making": ""}
        )
        for step in self.STEPS:
            assert f'data-trystep="{step["operator"]}"' in drawn
            assert f'data-seed="{step["seed"]}"' in drawn
            assert step["words"] in drawn
        assert "2 changes each make a different game" in drawn

    def test_a_draft_that_is_not_a_game_says_the_validators_reason(self) -> None:
        drawn = self._panel(
            {"state": "ready", "steps": [], "why": "setup.placements[0].piece 'Q' is "
             "not a declared piece id", "failed": "", "making": ""}
        )
        assert "not a game" in drawn
        assert "not a declared piece id" in drawn
        assert "data-trystep" not in drawn

    def test_what_the_last_press_changed_stands_above_the_list(self) -> None:
        drawn = self._panel(
            {"state": "asking"},
            said=json.dumps({"key": "x", "words": "the edges are joined"}),
        )
        assert "the edges are joined. Undo takes it back." in drawn

    def test_a_press_sends_the_draft_the_step_and_the_seed(self) -> None:
        asked = _run_page(_TRY_PRESS, "takeStep('join-the-edges', 3).then(() => asked)")
        assert asked["path"] == "/api/design/steps"
        assert asked["body"]["step"] == "join-the-edges"
        assert asked["body"]["seed"] == 3
        assert asked["body"]["spec"]["name"] == "a draft"

    def test_the_list_after_a_press_names_the_draft_it_was_pressed_on(self) -> None:
        """Q532. So the server can leave out a step back to it (`away_from`), and the
        panel says a change that only takes the last one back is left out."""
        after = _run_page(
            _TRY_PRESS,
            "takeStep('join-the-edges', 3).then(() => { TRY = null; refreshTry();"
            " return asked; })",
        )
        assert after["body"]["back"] == {
            "name": "a draft", "pieces": [], "rules": {"win": "checkmate"}}
        # Before any press there is nothing to walk away from.
        first = _run_page(_TRY_PRESS, "(TRY = null, refreshTry(), asked)")
        assert "back" not in first["body"]
        drawn = self._panel(
            {"state": "ready", "steps": self.STEPS, "why": "", "failed": "",
             "making": "", "back": True}
        )
        assert "only take back the last one is left out" in drawn

    def test_a_press_leaves_the_made_draft_and_says_what_it_changed(self) -> None:
        after = _run_page(
            _TRY_PRESS,
            "takeStep('join-the-edges', 3).then(() => "
            "({ draft: DRAFT, said: TRY_SAID }))",
        )
        assert after["draft"]["rules"]["wrap"] == {"axes": ["file"]}
        assert after["said"]["words"] == "the edges are joined"

    def test_the_undo_the_first_slice_built_takes_the_step_back(self) -> None:
        """A press is a change to the draft like every other change in the editor, so it
        goes behind the same Ctrl+Z rather than behind a way back of its own."""
        after = _run_page(
            _TRY_PRESS,
            "takeStep('join-the-edges', 3).then(() => { snapshotDraft(); "
            "stepDraft(false); return { draft: DRAFT, said: TRY_SAID }; })",
        )
        assert "wrap" not in (after["draft"].get("rules") or {})
        assert after["draft"]["name"] == "a draft"
        # And the sentence about what that press changed goes with it.
        assert after["said"] is None


# One press, with the server's answer stood in for: a draft, a step that joins the board's
# edges, and everything the editor would redraw stubbed out.
_TRY_PRESS = """
  DRAFT = { name: 'a draft', pieces: [], rules: { win: 'checkmate' } };
  DRAFT_SNAP = JSON.stringify(DRAFT);
  DRAFT_UNDO = []; DRAFT_REDO = [];
  TRY_OPEN = true;
  TRY = { key: DRAFT_SNAP, state: 'ready', steps: [], why: '', failed: '', making: '' };
  TRY_SAID = null;
  let asked = null;
  api = (path, body) => {
    asked = { path, body };
    return Promise.resolve({ remix: {
      spec: { name: 'a draft', pieces: [], rules: { win: 'checkmate',
                                                    wrap: { axes: ['file'] } } },
      words: 'the edges are joined' } });
  };
  drawDesigner = () => {}; drawTry = () => {}; loadMoveMap = () => {};
  drawUndo = () => {}; keepDraft = () => {};
  checkDraft = () => { snapshotDraft(); };
"""


class TestTheGameInBetween:
    """*Make the game in between*: a gap's two ends, and the game nobody made.

    A gap is two or three rules apart with nothing between. Where every rule that differs
    is one the spec declares, carrying one of them across builds the game that is one step
    from each end.
    """

    def test_two_rules_apart_gives_a_game_one_rule_from_each(self) -> None:
        from chess_variant_sandbox.spec import relate

        first = _remix_spec(check_limit=2)
        second = _remix_spec(no_progress_draw=12)
        made = designing.in_between(first, second, ("first", "second"))
        assert "spec" in made, made.get("why")
        assert made["units"] == [1, 1]
        assert relate.units(made["spec"], first) == 1
        assert relate.units(made["spec"], second) == 1

    def test_the_draft_is_a_game_that_opens(self) -> None:
        made = designing.in_between(
            _remix_spec(check_limit=2), _remix_spec(no_progress_draw=12)
        )
        assert not validate_spec(made["spec"])
        assert made["openingMoves"] > 0

    def test_it_says_which_rule_came_from_which_end(self) -> None:
        from chess_variant_sandbox.spec import relate

        first = _remix_spec(check_limit=2)
        second = _remix_spec(no_progress_draw=12)
        made = designing.in_between(first, second, ("first", "second"))
        # The walk is fixed: the first end, and the rules in the order `relate` lists
        # them, so two calls name the same rule and the same end.
        differences = relate.relate(first, second).differences
        assert made["on"] == 0
        assert made["take"] == differences[0].unit.split(".", 1)[1]
        # The phrase is `relate`'s own, so the map and the draft cannot name one rule two
        # ways.
        assert made["words"] == relate.RULE_WORDS[made["take"]]
        assert made["spec"]["description"].startswith("first with ")

    def test_three_rules_apart_gives_a_game_one_rule_from_one_end(self) -> None:
        from chess_variant_sandbox.spec import relate

        first = _remix_spec(check_limit=2, no_progress_draw=12, max_plies=120)
        second = _remix_spec()
        made = designing.in_between(first, second, ("first", "second"))
        assert sorted(made["units"]) == [1, 2]
        assert relate.units(made["spec"], first) + relate.units(made["spec"], second) == 3

    def test_the_same_pair_answers_the_same_way_twice(self) -> None:
        first, second = _remix_spec(check_limit=2), _remix_spec(no_progress_draw=12)
        assert designing.in_between(first, second) == designing.in_between(first, second)

    def test_a_gap_that_differs_in_more_than_declared_rules_offers_nothing(self) -> None:
        first = _remix_spec(check_limit=2)
        second = _remix_spec()
        second["board"] = {"width": 7, "height": 6}
        made = designing.in_between(first, second)
        assert "spec" not in made
        assert "not only their rules" in made["why"]

    def test_a_carried_rule_the_validator_refuses_offers_nothing(self) -> None:
        """Two games whose pieces carry other letters, so a rule cannot simply move.

        Both games declare a promotion and a goal, and each names its own letters. The
        rules differ, so the pair is a gap; carrying either across writes a letter the
        other game has no piece under, and the validator says so. That is an answer and
        not an error.
        """
        first = _remix_spec(
            promotion={"piece": "W", "rank": "last", "to": ["R"]},
            reach={"piece": "W", "squares": ["a6", "f6"], "mirror": True, "result": "win"},
        )
        second = _relettered(_remix_spec(), {"W": "P", "R": "T"})
        second["rules"].update({
            "promotion": {"piece": "P", "rank": 4, "to": ["T"]},
            "reach": {"piece": "P", "squares": ["c6"], "mirror": True, "result": "win"},
        })
        assert not validate_spec(first) and not validate_spec(second)
        made = designing.in_between(first, second)
        assert "spec" not in made
        assert "rules that don't fit together" in made["why"]

    def test_a_carry_that_gives_one_of_the_ends_back_offers_nothing(self) -> None:
        """Two games whose rules read the same and mean different things.

        The letters of two pieces are exchanged, so both files write the same promotion
        and the same goal and each says something the other does not. Carrying a rule
        across writes the text that is already there, and `relate.identity_key` says the
        result is the game it started from.
        """
        from chess_variant_sandbox.spec import relate

        rules = {
            "promotion": {"piece": "W", "rank": "last", "to": ["R"]},
            "reach": {"piece": "W", "squares": ["a6", "f6"], "mirror": True,
                      "result": "win"},
        }
        first = _remix_spec(**rules)
        second = _relettered(first, {"W": "R", "R": "W"})
        second["rules"] = json.loads(json.dumps(first["rules"]))
        assert not validate_spec(second)
        assert relate.units(first, second) == 2
        made = designing.in_between(first, second)
        assert "spec" not in made
        assert "one of these two games back" in made["why"]

    def test_the_corpus_gap_the_plan_names(self) -> None:
        """`docs/archive/plan-game-graph.md` G6's acceptance, on the two files it names."""
        from chess_variant_sandbox.spec import relate

        first = load_spec(RULESETS / "rps-v2.json")
        second = load_spec(RULESETS / "rps-v3.json")
        assert relate.units(first, second) == 2
        made = designing.in_between(first, second, (first["name"], second["name"]))
        assert made["units"] == [1, 1]
        assert made["take"] == "reach"
        assert not validate_spec(made["spec"])
        assert made["hash"] not in (spec_hash(first), spec_hash(second))


class TestPlayingADraftInPlace:
    """Stage 5 of `docs/plan-designer-ux.md`, option B4.

    The board column gets a fourth tab, and the draft is played on the board already in
    front of the person rather than on a page that replaces the editor. The board itself
    is `page/board.js`'s, mounted here, so what is held below is the wiring: which seats
    are offered, that a game in hand owns the column, and that the first edit ends it.
    Walked in a browser at 1400 px and at 375 px, on chess, on the hex draft and on
    crazyhouse, against the machine, by hand for both sides, and watched.
    """

    def test_the_column_offers_a_fourth_tab(self) -> None:
        drawn = str(_run_page("DRAFT_MODE = 'setup';", "benchTabsHtml()"))
        for mode in ("setup", "teach", "goes", "play"):
            assert f'data-bench="{mode}"' in drawn
        assert drawn.count("data-bench") == 4

    def test_the_seats_are_the_ladders_own_rungs(self) -> None:
        """A budget is part of an agent's identity, so a seat at one nobody rated is a
        player nobody measured. These are the two the verdict column already seats."""
        from chess_variant_sandbox.web import opponents

        seated = _run_page("", "['machine', 'hand', 'watch'].map(benchSeats)")
        rungs = dict(opponents.LADDER)
        assert seated[1] == ["human", "human"]
        for seats in seated:
            for seat in seats:
                assert seat == "human" or seat in rungs, seat

    def test_the_invitation_offers_the_machine_and_both_sides_by_hand(self) -> None:
        drawn = str(_run_page(
            "DRAFT_CHECK = { valid: true }; BENCH_ENDED = '';", "benchPlayHtml()"
        ))
        assert 'data-seats="machine"' in drawn and 'data-seats="hand"' in drawn
        assert "Nothing is kept" in drawn

    def test_a_draft_that_is_not_a_game_is_not_offered_a_seat(self) -> None:
        drawn = str(_run_page(
            "DRAFT_CHECK = { valid: false, errors: ['x'] };", "benchPlayHtml()"
        ))
        assert "data-seats" not in drawn
        assert "A draft can be played as soon as these rules" in drawn

    def test_editing_the_draft_ends_the_game_in_hand(self) -> None:
        """The acceptance line. A game in hand was played under the rules as they stood
        when it started, and the sentence stands where the board was."""
        ended = _run_page(
            "BENCH_PLAY = { key: JSON.stringify({ name: 'before' }), id: 'g1' };"
            " DRAFT = { name: 'after' }; SESSION = { id: 'g1' }; TIMELINE = [1, 2];"
            " snapshotDraft();",
            "({ playing: !!BENCH_PLAY, said: BENCH_ENDED, session: SESSION,"
            " timeline: TIMELINE.length })",
        )
        assert ended["playing"] is False
        assert "so it is over" in ended["said"]
        # The board goes with it, so a reply still on its way has nothing to land on:
        # `maybeAgent` drops one whose session is not the one on screen.
        assert ended["session"] is None
        assert ended["timeline"] == 0

    def test_a_draft_that_has_not_moved_keeps_its_game(self) -> None:
        held = _run_page(
            "DRAFT = { name: 'same' };"
            " BENCH_PLAY = { key: JSON.stringify(DRAFT), id: 'g1' };"
            " SESSION = { id: 'g1' }; snapshotDraft();",
            "({ playing: !!BENCH_PLAY, said: BENCH_ENDED })",
        )
        assert held["playing"] is True
        assert held["said"] == ""

    def test_a_game_in_hand_owns_the_column(self) -> None:
        """Every check redraws this column, and a game is a board that has to survive
        between two of its own moves -- so the redraw stops at the panel it would
        replace."""
        kept = _run_page(
            "const side = { innerHTML: 'the game', querySelectorAll: () => [] };"
            " document.getElementById = id => id === 'boardSide' ? side : null;"
            " DRAFT_MODE = 'play'; PICKING = null;"
            " BENCH_PLAY = { key: '{}', id: 'g1' }; drawBoardSide();",
            "({ column: side.innerHTML, playing: !!BENCH_PLAY })",
        )
        assert kept == {"column": "the game", "playing": True}

    def test_taking_the_board_back_ends_the_game(self) -> None:
        """Another tab, or a rule card asking for squares, is the editor asking for the
        board back. The game goes, and the person is told where it went."""
        gone = _run_page(
            "const side = { innerHTML: '', querySelectorAll: () => [] };"
            " document.getElementById = id => id === 'boardSide' ? side : null;"
            " DRAFT = {}; DRAFT_MODE = 'setup'; PICKING = null;"
            " DRAFT_CHECK = null; DRAFT_PREVIEW = null;"
            " BENCH_PLAY = { key: '{}', id: 'g1' }; SESSION = { id: 'g1' };"
            " wirePalette = () => {}; wirePieceTray = () => {}; wireBoards = () => {};"
            " drawBoardSide();",
            "({ playing: !!BENCH_PLAY, said: BENCH_ENDED,"
            " column: side.innerHTML.indexOf('data-bench') >= 0 })",
        )
        assert gone["playing"] is False
        assert "back to the editor" in gone["said"]
        assert gone["column"] is True

    def test_the_board_given_up_takes_its_game_with_it(self) -> None:
        """`leaveBoard` is `page/board.js`'s, called from here. A reply still on its way
        lands on nothing afterwards, because `maybeAgent` drops one whose session is not
        the one on screen."""
        left = _run_page(
            "SESSION = { id: 'x' }; TIMELINE = [1, 2]; TIMELINE_FOR = 'x';"
            " AT = { board: [] }; CURSOR = 1; PAUSED = true; BOARD_INLINE = true;"
            " leaveBoard();",
            "({ session: SESSION, timeline: TIMELINE.length, at: AT, cursor: CURSOR,"
            " paused: PAUSED, inline: BOARD_INLINE })",
        )
        assert left == {"session": None, "timeline": 0, "at": None, "cursor": None,
                        "paused": False, "inline": False}

    def test_and_now_try_is_in_the_column_a_phone_meets_first(self) -> None:
        """Stage 4 left the panel in the verdict column, which stacks last: on the hex
        draft at 375 px it sat 11,000 px down while only the action bar was pinned. The
        board column is `order: 1` stacked, so the panel goes to the foot of it --
        measured in a browser at 375 px, y=768 on an 11,508 px page.
        """
        from page_text import DESIGN

        verdict = DESIGN[DESIGN.index("function drawVerdict("):]
        verdict = verdict[: verdict.index("\nfunction ", 1)]
        assert "tryPanelHtml()" not in verdict
        column = DESIGN[DESIGN.index("function drawBoardSide("):]
        column = column[: column.index("\nfunction ", 1)]
        assert "tryPanelHtml()" in column


# ------------------------------------------------------ what a newcomer trips on (Q532)
#
# `docs/loose-ends.md` §2: one person walking the designer on 2026-09-19, re-walked at
# 1400 and 375 px in a browser on 2026-09-29. What was cheap to fix is held here.


class TestTwoWaysOfPlayingSayWhichIsWhich:
    """The workbench tab and the verdict's button both said *play it*. One plays on the
    board beside the editor with no move list, the other opens the board page."""

    def test_the_tab_and_the_buttons_read_differently(self) -> None:
        said = _run_page(
            "DRAFT = { name: 'a draft', pieces: [] };"
            " DRAFT_CHECK = { valid: true };",
            "({ tab: BENCH_MODES.find(m => m[0] === 'play')[1], away: PLAY_AWAY,"
            " watch: WATCH_AWAY, invite: benchPlayHtml() })",
        )
        assert said["tab"] == "play it here"
        assert said["away"] == "play it on its own page"
        assert said["watch"] == "watch it on its own page"
        # The invitation under the tab says what this board lacks and where it is.
        assert "no move list" in said["invite"]
        assert said["away"] in said["invite"]

    def test_the_verdicts_buttons_carry_those_words(self) -> None:
        from page_text import DESIGN

        verdict = DESIGN[DESIGN.index("function drawVerdict("):]
        verdict = verdict[: verdict.index("\nfunction ", 1)]
        assert '<button id="dPlay"' in verdict
        assert "${awayLabel(PLAY_AWAY)}</button>" in verdict
        assert "${awayLabel(WATCH_AWAY)}</button>" in verdict
        assert "play it, without saving" not in DESIGN

    def test_the_board_played_here_is_not_framed_twice(self) -> None:
        """The play board brings its own panel; inside the workbench's that was two
        frames of padding, and a 271 px board on a 375 px phone."""
        from chess_variant_sandbox.web.page.style import STYLE

        assert "#benchPlay .panel { padding: 0; border: 0;" in STYLE


# A draft with one piece standing and a palette holding one more.
_FRESH = """
  DRAFT = { name: 'a draft', board: { width: 6, height: 6 },
            pieces: [{ id: 'S', name: 'soldier', moves: [] }],
            setup: { mirror: true, placements: [{ piece: 'S', squares: ['a1'] }] } };
  DESIGN_KIT = { palette: [{ key: 'k1', id: 'N', name: 'jumper',
                             piece: { id: 'N', name: 'jumper', moves: [] } }] };
  DRAFT_MODE = 'teach'; PICKING = null; PLACING = null; TEACHING = null;
  drawDesigner = () => {}; checkDraft = () => {}; loadMoveMap = () => {};
"""


class TestAPiecesArrivalSaysItIsNotPlaced:
    """A piece taken from the palette arrived in the list of pieces and on no square,
    and nothing said so."""

    def test_a_palette_piece_is_picked_in_the_tray_and_said_to_be_off_the_board(
        self,
    ) -> None:
        after = _run_page(
            _FRESH + "takePalettePiece('k1');",
            "({ mode: DRAFT_MODE, placing: PLACING, card: freshNoteHtml(1),"
            " under: paletteNoteHtml(), other: freshNoteHtml(0) })",
        )
        assert after["mode"] == "setup"
        assert after["placing"] == {"piece": 1, "player": 0}
        assert "not on the board yet" in after["card"]
        assert "not on the board yet" in after["under"]
        assert after["other"] == ""

    def test_the_note_goes_once_a_square_holds_it(self) -> None:
        after = _run_page(
            _FRESH + "takePalettePiece('k1');"
            " DRAFT.setup.placements.push({ piece: 'N', squares: ['b1'] });",
            "({ card: freshNoteHtml(1), under: paletteNoteHtml() })",
        )
        assert after["card"] == ""
        assert "not on the board yet" not in after["under"]

    def test_a_piece_that_starts_off_the_board_is_not_nagged_about(self) -> None:
        """A piece that arrives by promotion or from a hand stands nowhere at the start
        and is meant to: only the piece just added is said to be missing."""
        after = _run_page(
            _FRESH + "DRAFT.pieces.push({ id: 'P', name: 'promoted', moves: [] });",
            "({ card: freshNoteHtml(1), under: paletteNoteHtml() })",
        )
        assert after["card"] == ""
        assert "not on the board yet" not in after["under"]


class TestAMoveMapOfABrokenDraftIsARefusal:
    """Q541: ``POST /api/design/moves`` answered 500 for a draft the validator refuses (a
    palette king taken into an annihilation draft). The page coped; the log filled with
    stack traces. It is a 400 carrying the validator's sentence."""

    def _broken(self) -> dict[str, Any]:
        skirmish = next(t for t in designing.templates() if t["key"] == "skirmish")
        spec = copy.deepcopy(skirmish["spec"])
        spec["pieces"].append({
            "id": "K", "name": "king", "royal": True, "value": 0,
            "moves": [{"kind": "leap", "vectors": [[0, 1], [1, 1]], "symmetry": "dihedral8"}],
        })
        assert validate_spec(designing.settled(spec))
        return spec

    def test_the_move_map_raises_a_design_error(self) -> None:
        with pytest.raises(designing.DesignError, match="aren't a complete game yet"):
            designing.move_map(self._broken(), 1)

    def test_the_route_answers_400(self) -> None:
        import threading
        from http.client import HTTPConnection

        from chess_variant_sandbox.web import library
        from chess_variant_sandbox.web.server import (
            REQUEST_HEADER,
            REQUEST_VALUE,
            App,
            Server,
            handler_for,
        )

        server = Server(("127.0.0.1", 0), handler_for(App(RULESETS, library.RUNS_DIR)))
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02},
                         daemon=True).start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=30)
            connection.request("POST", "/api/design/moves",
                               json.dumps({"spec": self._broken(), "piece": 1}),
                               {"Content-Type": "application/json",
                                REQUEST_HEADER: REQUEST_VALUE})
            response = connection.getresponse()
            answer = json.loads(response.read())
            assert response.status == 400, answer
            assert "annihilation" in answer["error"]
        finally:
            server.shutdown()
            server.server_close()


class TestAMoveMapOfADraftTheCompilerRefusesIsARefusal:
    """Q792: a draft the validator passes and the compiler refuses (a wall set that
    bricks a player in) raised ``SpecError`` out of ``Engine(spec)`` into a 500, five
    times on the live site on 2026-10-05. It is a 400 carrying the compiler's sentence."""

    @staticmethod
    def _bricked() -> dict[str, Any]:
        return {
            "schema_version": 6,
            "name": "bricked",
            "board": {"width": 5, "height": 5},
            "pieces": [
                {"id": "R", "name": "rook", "value": 5, "royal": True,
                 "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
            ],
            "setup": {"mirror": False, "placements": [
                {"piece": "R", "player": 0, "squares": ["c3"]},
                {"piece": "R", "player": 1, "squares": ["a1"]},
            ]},
            "rules": {"win": "royal_capture",
                      "obstacles": {"squares": ["c2", "c4", "b3", "d3"]}},
        }

    def test_the_draft_validates_and_does_not_compile(self) -> None:
        assert validate_spec(designing.settled(self._bricked())) == []
        with pytest.raises(SpecError):
            Engine(designing.settled(self._bricked()))

    def test_the_move_map_raises_a_design_error(self) -> None:
        with pytest.raises(designing.DesignError, match="walls player 0 in|walls"):
            designing.move_map(self._bricked(), 0)

    def test_the_route_answers_400(self) -> None:
        import threading
        from http.client import HTTPConnection

        from chess_variant_sandbox.web import library
        from chess_variant_sandbox.web.server import (
            REQUEST_HEADER,
            REQUEST_VALUE,
            App,
            Server,
            handler_for,
        )

        server = Server(("127.0.0.1", 0), handler_for(App(RULESETS, library.RUNS_DIR)))
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02},
                         daemon=True).start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=30)
            connection.request("POST", "/api/design/moves",
                               json.dumps({"spec": self._bricked(), "piece": 0}),
                               {"Content-Type": "application/json",
                                REQUEST_HEADER: REQUEST_VALUE})
            response = connection.getresponse()
            answer = json.loads(response.read())
            assert response.status == 400, answer
            assert "complete game" in answer["error"]
            assert "walls" in answer["error"]
        finally:
            server.shutdown()
            server.server_close()


class TestAnObliqueLineIsReadBack:
    """Q541: a template given a slide along ``[1, 2]`` (a nightrider) validated, and no
    line of *the rules, in English* named it: the rule lines are every rule beyond
    movement, and the panel had no piece lines. And a nightrider cut to one step was said
    in vectors, *exactly 1 square, path must be clear*, which it is neither."""

    @pytest.mark.parametrize("symmetry", ["none", "lateral", "rotate4", "dihedral8"])
    @pytest.mark.parametrize("reach", [0, 1, 3])
    def test_every_symmetry_and_reach_has_a_sentence(self, symmetry: str, reach: int) -> None:
        kingly = next(t for t in designing.templates() if t["key"] == "kingly")
        spec = copy.deepcopy(kingly["spec"])
        rider = next(p for p in spec["pieces"] if not p.get("royal") and p["id"] != "P")
        rider["moves"] = [{"kind": "slide", "vectors": [[1, 2]], "symmetry": symmetry,
                           "range": reach}]
        check = designing.check(spec)
        assert check["valid"], check["errors"]
        line = next(line for line in check["pieceLines"] if line.startswith(rider["id"] + " "))
        assert "along (" not in line and "path must be clear" not in line, line
        assert "two" in line and "one" in line, line
        if reach == 1:
            assert "leaps" in line, line
        else:
            assert "steps of" in line and "stopped by anything in the way" in line, line

    def test_the_panel_draws_the_piece_lines(self) -> None:
        from page_text import PAGE

        assert "(check.pieceLines || [])" in PAGE
        assert "how each piece moves" in PAGE


class TestASpellingWaitsOutAReload:
    """Q541: a spelling taken on a rule card and not yet filled in writes no key, so it
    lived in the page (`CHOSEN_SPELLING`) and a reload drew the card in its first spelling
    again. It is kept beside the draft in the reload's entry and taken back up with it."""

    _STORE = (
        "const store = {};"
        " window.sessionStorage = { setItem: (k, v) => { store[k] = v; },"
        " getItem: k => (k in store ? store[k] : null) };"
    )

    def _page(self, then: str) -> str:
        spec = next(t for t in designing.templates() if t["key"] == "kingly")["spec"]
        spec = json.loads(json.dumps(spec))
        spec["rules"]["promotion"] = {"piece": "P", "rank": "last", "to": ["R"]}
        return _run_page(
            _kit_setup() + self._STORE + "drawDesigner = () => {}; checkDraft = () => {};"
            f"DRAFT = {json.dumps(spec)};"
            f" DRAFT_CHECK = {{ preview: {json.dumps(designing.check(spec)['preview'])} }};"
            " switchChoice('rules', ['promotion'], 0, 2);"
            # The reload: the page's state is gone, and only the tab's storage is left.
            " DRAFT = null; CHOSEN_SPELLING = {};" + then,
            "rulesCardsHtml([])",
        )

    def test_the_spelling_comes_back_with_the_draft(self) -> None:
        plain = self._page(" takeDraft(heldDraft(), {});").replace("&quot;", '"')
        assert '"promotion","squares"' in plain
        assert '"promotion","rank"' not in plain

    def test_a_template_starts_with_no_spelling_held(self) -> None:
        """Nothing held: the template, and every card in its first spelling."""
        spec = next(t for t in designing.templates() if t["key"] == "kingly")["spec"]
        template = json.loads(json.dumps(spec))
        template["rules"]["promotion"] = {"piece": "P", "to": ["R"]}
        plain = self._page(
            f" takeDraft(null, {json.dumps(template)});"
        ).replace("&quot;", '"')
        assert '"promotion","squares"' not in plain


class TestTheMoveRowStartsShort:
    """Q545 (`docs/loose-ends.md` §2, the user's *"A"*): the four controls most moves
    need on the row, and how far, how near and from which rank folded under *more*."""

    def test_a_plain_move_folds_the_rest(self) -> None:
        html = _run_page("", "moveEditor({ kind: 'slide', vectors: [[0, 1]] }, 0, 0)")
        row, folded = html.split('<details class="moveMore"', 1)
        for kind in ("kind", "symmetry", "vectors", "mode"):
            assert f'data-mfield="{kind}"' in row, kind
        for kind in ("range", "min_range", "from_rank"):
            assert f'data-mfield="{kind}"' in folded and f'data-mfield="{kind}"' not in row
        assert folded.startswith(">"), "folded, not open"

    @pytest.mark.parametrize("move", [
        {"kind": "slide", "vectors": [[0, 1]], "range": 2},
        {"kind": "slide", "vectors": [[0, 1]], "min_range": 2},
        {"kind": "leap", "vectors": [[0, 1]], "condition": {"from_rank": 1}},
    ])
    def test_a_move_that_says_one_of_them_shows_it(self, move: dict) -> None:
        html = _run_page("", f"moveEditor({json.dumps(move)}, 0, 0)")
        assert '<details class="moveMore" open>' in html


class TestThePhoneIsToldOnce:
    """Q545: on a narrow screen the designer says once that a computer is easier."""

    def _said(self, narrow: bool, times: int, storage: str) -> Any:
        return _run_page(
            f"window.localStorage = {storage};"
            f" window.matchMedia = () => ({{ matches: {str(narrow).lower()} }});",
            f"Array.from({{length: {times}}}, () => phoneNoteHtml().includes('computer'))",
        )

    _KEPT = ("(() => { const kept = {}; return { getItem: k => kept[k] || null,"
             " setItem: (k, v) => { kept[k] = v; } }; })()")

    def test_a_phone_is_told_once(self) -> None:
        assert self._said(True, 2, self._KEPT) == [True, False]

    def test_a_computer_is_never_told(self) -> None:
        assert self._said(False, 1, self._KEPT) == [False]

    def test_storage_refused_still_tells_it(self) -> None:
        refused = "{ getItem: () => { throw new Error('refused'); } }"
        assert self._said(True, 2, refused) == [True, True]


class TestTheDesignersLayout:
    """Q580: F10 and F11 of `docs/reference/first-visit-study.md`, and the phone pass's
    *Not a game yet* panel that took half the screen (§1.2a step 5). The behaviour was
    walked in Chromium at 375 x 812 and 1400 x 900; what is held here is what holds it."""

    def _kingly(self) -> dict:
        return copy.deepcopy(next(t for t in designing.templates() if t["key"] == "kingly")["spec"])

    def test_every_typed_move_is_read_back(self) -> None:
        """F10: a step typed as numbers had no read-back. One sentence per editor row, the
        game page's own (`describe.move_sentence`), in the editor's order."""
        spec = self._kingly()
        lines = designing.check(spec)["moveLines"]
        assert [len(said) for said in lines] == [len(p["moves"]) for p in spec["pieces"]]
        for piece, said in zip(spec["pieces"], lines, strict=True):
            for move, sentence in zip(piece["moves"], said, strict=True):
                assert sentence == describe.move_sentence(move)

    def test_a_draft_that_is_not_a_game_still_reads_its_rows_back(self) -> None:
        """A half-typed row reads as nothing, and the rest of the draft still reads."""
        spec = self._kingly()
        spec["pieces"][0]["moves"][0]["vectors"] = []
        spec["pieces"][1]["moves"][0]["vectors"] = [[1, "x"]]
        answer = designing.check(spec)
        assert answer["valid"] is False
        assert answer["moveLines"][0][0] == "" and answer["moveLines"][1][0] == ""
        assert answer["moveLines"][0][1] == describe.move_sentence(spec["pieces"][0]["moves"][1])

    def test_a_row_shows_only_its_own_drafts_sentence(self) -> None:
        said = _run_page(
            "DRAFT = { pieces: [] };"
            " MOVE_SAID = { key: JSON.stringify(DRAFT), lines: [['one step forward']] };",
            "[moveSaid(0, 0), (DRAFT = { pieces: [{}] }, moveSaid(0, 0))]",
        )
        assert said == ["That reads: one step forward.", ""]

    def test_not_a_game_yet_is_one_line_and_the_rest_one_tap_away(self) -> None:
        """The panel complained about a setting and covered it. A heading, the first thing
        to fix, and every reason folded under one summary."""
        errors = ["the first thing wrong", "the second thing wrong", "a third"]
        drawn = _run_page(
            "DRAFT = { name: 'x', pieces: [] }; DRAWING = null;"
            " const side = { innerHTML: '', querySelectorAll: () => [] };"
            " document.getElementById = id => id === 'verdictSide' ? side : null;"
            f" DRAFT_CHECK = {{ valid: false, errors: {json.dumps(errors)} }};",
            "(drawVerdict(), side.innerHTML)",
        )
        head, folded = drawn.split('<details class="notGameAll"', 1)
        assert "Not a game yet" in head and "the first thing wrong" in head
        assert "the second thing wrong" not in head
        assert "all 3 things to fix" in folded
        for error in errors:
            assert f"<li>{error}</li>" in folded

    def test_the_sticking_columns_have_no_scrollbar_of_their_own(self) -> None:
        """F11: the wheel over the verdict column scrolled the column. It sticks by its
        foot instead, from an offset `fitSticky` sets, and never scrolls on its own."""
        from page_text import PAGE

        at = PAGE.index(".designer .stick {")
        rule = PAGE[at : PAGE.index("}", at)]
        assert "overflow" not in rule and "max-height" not in rule
        assert "top: var(--stick-top, 12px)" in rule
        offset = _run_page(
            "window.innerHeight = 900;"
            " const tall = { offsetHeight: 2224, style: { setProperty(k, v) { this[k] = v; } } };"
            " const short = { offsetHeight: 642, style: { setProperty(k, v) { this[k] = v; } } };"
            " document.querySelectorAll = () => [tall, short];"
            " window.getComputedStyle = () => ({ position: 'sticky' });",
            "(fitSticky(), [tall.style['--stick-top'], short.style['--stick-top']])",
        )
        assert offset == ["-1336px", "12px"]

    def test_where_to_start_stands_above_the_board(self) -> None:
        """F10: the template picker was in the form, under the board on a phone."""
        from page_text import PAGE

        start = PAGE.index("async function showDesigner(from)")
        body = PAGE[start : PAGE.index("\nfunction ", start + 1)]
        assert body.index("${startFromHtml(kit)}") < body.index('<div class="designer bench"')
        draw = PAGE.index("function drawDesigner()")
        assert 'id="dTemplate"' not in PAGE[draw : PAGE.index("\nfunction ", draw + 1)]
        assert PAGE.count('id="dTemplate"') == 1

    def test_the_setting_just_changed_stays_above_the_bar(self) -> None:
        """On a phone the bar at the foot grows when a change makes the draft no game; the
        page scrolls by what it covers of the control just changed, and no further."""
        moved = _run_page(
            "const bar = { getBoundingClientRect: () => ({ top: 700 }) };"
            " const field = { getBoundingClientRect: () => ({ bottom: 720 }) };"
            " const form = { contains: node => node === field };"
            " document.getElementById = id => id === 'dActions' ? bar : id === 'form' ? form : null;"
            " document.activeElement = field;"
            " window.getComputedStyle = () => ({ position: 'fixed' });"
            " const scrolled = []; window.scrollBy = (x, y) => scrolled.push(y);",
            "(keepEditInView(), field.getBoundingClientRect = () => ({ bottom: 600 }),"
            " keepEditInView(), scrolled)",
        )
        assert moved == [28]
