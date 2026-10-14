"""Q851, Q853, Q854, Q855 and Q856: the designer's wild pieces, as the wolves walk found them.

`docs/reference/designer-case-study-wolves.md` §3, fixes 2, 4, 5, 6 and 7. Each class is one
fix's acceptance, on a draft made from `archive/wolves` (two wild wolves, W, mirrored, on
a4's and h4's five-square patrols):

- Fix 2: picking the tray's W says it is a player's W and where the wild one is written;
  *where it goes* on W names a4's walk instead of saying none stands on the board; and a
  tray click while a walk is being picked puts a piece down and leaves the walk alone.
- Fix 4: two wolves on one circuit, written as two entries, are refused in one sentence
  that names both and the squares, and points at *more on this walk*.
- Fix 5: the walk picker names whose walk it is, draws each click at once from the draft,
  and adds *a whole row* from the end nearer the last square picked, so the user's circuit
  is two line presses and two clicks. (That the board stays in view is the look's,
  `designer-wild-pick`.)
- Fix 6: *+ wild piece* under the tray starts a wild piece of a chosen type, the draft's
  own or the library's, on the square clicked, and opens its walk's picker.
- Fix 7: the wild pieces card says what the rule is in one paragraph, and *add one* after
  a wolf adds a wolf.

The page's half runs the page's own functions in node (`test_designer._run_page`).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from test_designer import _kit_setup, _run_page

from chess_variant_sandbox.spec.describe import RULE_WORDS
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.web import designing

ROOT = Path(__file__).resolve().parents[1]
WOLVES: dict[str, Any] = json.loads((ROOT / "archive" / "wolves.json").read_text(encoding="utf-8"))
CIRCUIT = [f"{f}4" for f in "abcdefgh"] + [f"{f}5" for f in "hgfedcba"]


def _wolves() -> dict[str, Any]:
    return copy.deepcopy(WOLVES)


def _index(spec: dict[str, Any], piece: str) -> int:
    return [p["id"] for p in spec["pieces"]].index(piece)


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + (ord(name[0]) - ord("a"))


# The redraws and the re-check need a DOM and a server; what is under test is what the
# page writes into the draft and says, so they are stubbed.
_QUIET = ("drawDesigner = () => {}; checkDraft = () => {}; drawBoardSide = () => {};"
          " loadMoveMap = () => {}; drawVerdict = () => {}; endBenchGame = () => {};"
          " keepDraft = () => {}; drawUndo = () => {};")


def _page(spec: dict[str, Any], steps: str, expression: str) -> Any:
    answer = designing.check(spec)
    assert answer["valid"], answer["errors"]
    setup = (_QUIET + _kit_setup() + f"DRAFT = {json.dumps(spec)};"
             f" DRAFT_CHECK = {json.dumps(answer)}; DRAFT_MODE = 'setup';" + steps)
    return _run_page(setup, expression)


# --------------------------------------------------------------------------- fix 2

class TestTheTraysWildType:
    def test_picking_w_says_it_is_a_player_s_and_links_the_card(self) -> None:
        spec = _wolves()
        note = _page(spec, f"PLACING = {{ piece: {_index(spec, 'W')}, player: 0 }};",
                     "paletteNoteHtml()")
        assert "data-wildnote" in note
        assert "is also a wild piece in this game" in note
        assert "one for each player" in note
        assert "data-wildcard" in note and ">wild pieces</a> card" in note

    def test_a_piece_no_wild_one_is_made_of_says_nothing_more(self) -> None:
        spec = _wolves()
        note = _page(spec, f"PLACING = {{ piece: {_index(spec, 'N')}, player: 0 }};",
                     "paletteNoteHtml()")
        assert "data-wildnote" not in note and "data-wildcard" not in note

    def test_a_tray_click_while_picking_places_a_piece_and_leaves_the_walk(self) -> None:
        """Step 10: W picked in the tray while *its walk* was being picked; the click on h5
        took h5 out of the walk. Now the tray takes the board, and the click puts W down."""
        spec = _wolves()
        w = _index(spec, "W")
        steps = (
            "PICKING = { root: 'rules', path: ['npcs', 0, 'route'], label: 'its walk' };"
            f" const tray = {{ dataset: {{ placing: '{w}.0' }} }};"
            " wirePalette({ querySelectorAll: () => [tray] }); tray.onclick();"
            " const svg = {};"
            " wireBoards({ querySelectorAll: q => (q.includes('start') ? [svg] : []),"
            " querySelector: () => null });"
            f" svg.onclick({{ target: {{ closest: () => ({{ dataset: {{ square: '{_square('d3')}' }} }}) }} }});"
        )
        picking, placing, npcs, placements = _page(
            spec, steps, "[PICKING, PLACING, DRAFT.rules.npcs, DRAFT.setup.placements]")
        assert picking is None
        assert placing == {"piece": w, "player": 0}
        assert npcs == WOLVES["rules"]["npcs"]
        assert any("d3" in p.get("squares", []) for p in placements if p["piece"] == "W"), \
            placements

    def test_the_tray_takes_the_board_back_to_set_up(self) -> None:
        spec = _wolves()
        steps = ("DRAFT_MODE = 'goes';"
                 " PICKING = { root: 'rules', path: ['npcs', 0, 'route'], label: 'its walk' };"
                 " endPickingForTray();")
        assert _page(spec, steps, "[PICKING, DRAFT_MODE, PICK_LINE]") == [None, "setup", ""]


class TestWhereTheWildTypeGoes:
    def _key(self, spec: dict[str, Any], piece: str, start: int = -1) -> str:
        standing = [] if start < 0 else [start]
        steps = (f"DRAFT_PIECE = {_index(spec, piece)}; DRAFT_MAP = {{ live: true, from: {start},"
                 f" width: 8, standing: {json.dumps(standing)}, moves: [], takes: [] }};")
        return _page(spec, steps, "moveKeyHtml()")

    def test_w_names_a4_s_walk_instead_of_none_on_the_board(self) -> None:
        said = self._key(_wolves(), "W")
        assert "nothing for it to do" not in said
        assert "the one on a4, with a mirrored one, walks a4, b5, c4, b3, a3" in said
        assert "the one on h4, with a mirrored one, walks h4, g5, f4, g3, h3" in said
        assert "data-wildcard" in said and "No player's <b>wolf</b> stands on" in said

    def test_a_long_walk_is_shortened_and_its_other_starts_named(self) -> None:
        spec = _wolves()
        spec["rules"]["npcs"] = [{"piece": "W", "route": CIRCUIT, "also_at": ["h5"]}]
        said = self._key(spec, "W")
        assert "the ones on a4 and h5 walk a4, b4, c4 &hellip; a5" in said
        assert "(16 squares)" in said

    def test_a_piece_no_wild_one_is_made_of_reads_as_before(self) -> None:
        said = self._key(_wolves(), "N")
        assert "nothing for it to do yet" in said and "data-wildcard" not in said

    def test_a_player_s_w_on_the_board_is_told_apart_from_the_wild_one(self) -> None:
        said = self._key(_wolves(), "W", _square("d3"))
        assert "The marks show every move" in said and "data-wildgoes" in said


# --------------------------------------------------------------------------- fix 4

class TestOverlappingWalks:
    def test_two_wolves_on_one_circuit_are_one_sentence(self) -> None:
        spec = _wolves()
        spec["rules"]["npcs"] = [{"piece": "W", "route": CIRCUIT},
                                 {"piece": "W", "route": CIRCUIT[8:] + CIRCUIT[:8]}]
        answer = designing.check(spec)
        assert not answer["valid"]
        (said,) = answer["errors"]
        assert said.startswith("Wild pieces 1 and 2 both walk h5 … h4 (16 squares).")
        assert "use 'more on this walk'" in said
        assert "fact about the square" not in said and "rules.npcs" not in said

    def test_a_mirror_on_itself_is_named_as_the_mirror(self) -> None:
        spec = _wolves()
        spec["rules"]["npcs"] = [{"piece": "W", "route": CIRCUIT, "mirror": True}]
        (said,) = designing.check(spec)["errors"]
        assert said.startswith("Wild pieces 1 and wild pieces 1's mirror both walk")
        assert "(mirrored)" not in said and "number 1:" not in said
        assert said.endswith("use 'more on this walk' instead of the mirror.")

    def test_one_sentence_per_pair_and_other_problems_kept(self) -> None:
        spec = _wolves()
        spec["rules"]["npcs"] = [{"piece": "W", "route": ["a4", "b4", "c4"]},
                                 {"piece": "W", "route": ["c4", "d4"]},
                                 {"piece": "W", "route": ["a4", "a3"]}]
        problems = ["name must be a non-empty string",
                    "rules.npcs[1] and rules.npcs[0] both walk c4; routes may not share a "
                    "square, because ...",
                    "rules.npcs[2] and rules.npcs[0] both walk a4; routes may not share a "
                    "square, because ..."]
        said = designing.plain_problems(problems, spec)
        assert said[0] == "Give the game a name."
        assert [s.split(".")[0] for s in said[1:]] == [
            "Wild pieces 1 and 2 both walk c4", "Wild pieces 1 and 3 both walk a4"]

    def test_the_validator_s_own_sentence_is_unchanged(self) -> None:
        """The proposing LLM repairs a draft from the validator's words, so only the
        designer's are collapsed."""
        from chess_variant_sandbox.spec import validate_spec

        spec = _wolves()
        spec["rules"]["npcs"] = [{"piece": "W", "route": CIRCUIT},
                                 {"piece": "W", "route": CIRCUIT[8:] + CIRCUIT[:8]}]
        raw = [p for p in validate_spec(spec) if "both walk" in p]
        assert len(raw) == 16 and all(designing._WALK_OVERLAP.match(p) for p in raw)


# --------------------------------------------------------------------------- fix 7

class TestTheWildPiecesCard:
    def test_one_intro_paragraph(self) -> None:
        html = _page(_wolves(), "", "rulesCardsHtml([])")
        card = re.search(r'<div class="ruleCard" data-rule="npcs">(.*?)<div class="subCard">',
                         html, re.S)
        assert card is not None
        assert card.group(1).count('class="muted small cardSays"') == 1
        assert RULE_WORDS["npcs"][1] in card.group(1)
        schema_words = rule_spec_schema()["properties"]["rules"]["properties"]["npcs"]
        assert schema_words["description"] not in html

    def test_add_one_after_a_wolf_adds_a_wolf(self) -> None:
        steps = (" const push = { dataset: { cpush: '[\"npcs\"]', crootPush: 'rules' } };"
                 " wireCards({ querySelectorAll: q => (q === '[data-cpush]' ? [push] : []),"
                 " querySelector: () => null });"
                 " push.onclick({ stopPropagation() {} });")
        npcs = _page(_wolves(), steps, "DRAFT.rules.npcs")
        assert len(npcs) == 3
        assert npcs[2]["piece"] == "W"
        assert npcs[2]["route"] and npcs[2]["route"][0] != "a3"

    def test_the_first_one_still_starts_from_the_plain_default(self) -> None:
        spec = _wolves()
        del spec["rules"]["npcs"]
        piece = _page(spec, "", "fieldDefault(fieldAt('rules', ['npcs', 0]), ['npcs', 0], 'rules').piece")
        assert piece == spec["pieces"][0]["id"]


# --------------------------------------------------------------------------- fix 5

def _one_wolf(route: list[str]) -> dict[str, Any]:
    """Wolves with one wild wolf, no mirror, on ``route``: the circuit's starting point."""
    spec = _wolves()
    spec["rules"]["npcs"] = [{"piece": "W", "route": route}]
    return spec


_PICK_ROUTE = "PICKING = Object.assign({ root: 'rules', path: ['npcs', 0, 'route'] }," \
              " pickNamed('rules', ['npcs', 0, 'route'], 'its walk'));"


def _line(kind: str, square: str) -> str:
    return f" PICK_LINE = '{kind}'; pickLine({_square(square)});"


class TestTheWalkPicker:
    def test_the_card_s_button_names_the_entry_and_its_piece(self) -> None:
        """Step 8: two walks were both *its walk*; the note names whose."""
        steps = (" const pick = { dataset: { cpick: '[\"npcs\",1,\"route\"]', crootPick: 'rules',"
                 " clabel: 'its walk' } };"
                 " wireCards({ querySelectorAll: q => (q === '[data-cpick]' ? [pick] : []),"
                 " querySelector: () => null });"
                 " pick.onclick({ preventDefault() {} });")
        picking, note = _page(_wolves(), steps, "[PICKING, pickNoteHtml()]")
        assert picking["label"] == "wild pieces 2's walk" and picking["piece"] == "W"
        assert picking["fromCard"] is True
        assert "<b>wild pieces 2&#39;s walk, W</b>" in note or "<b>wild pieces 2's walk, W</b>" in note
        assert "data-pickdone" in note and "data-pickline" in note

    def test_the_users_circuit_in_two_line_presses_and_two_clicks(self) -> None:
        """Step 9: `a4 ... h4`, then row 5 joins from h5, the end nearer h4."""
        spec = _one_wolf(["a4"])
        steps = _PICK_ROUTE + _line("row", "a4") + _line("row", "e5")
        route, said = _page(spec, steps, "[DRAFT.rules.npcs[0].route, BENCH_SAID]")
        assert route == CIRCUIT
        assert said == "Added row 5 to wild pieces 1's walk, h5 to a5."
        done = _one_wolf(route)
        assert designing.check(done)["valid"]

    def test_a_line_starts_from_the_clicked_end_of_an_empty_list(self) -> None:
        spec = _one_wolf(["a4"])
        steps = _PICK_ROUTE + " DRAFT.rules.npcs[0].route = [];" + _line("row", "h4")
        assert _page(spec, steps, "DRAFT.rules.npcs[0].route") == [f"{f}4" for f in "hgfedcba"]

    def test_a_column_joins_from_its_nearer_end(self) -> None:
        spec = _one_wolf(["a4"])
        steps = _PICK_ROUTE + _line("column", "b5")
        route = _page(spec, steps, "DRAFT.rules.npcs[0].route")
        # a4 is three ranks from b1 and four from b8: the column joins from b1 upwards.
        assert route == ["a4"] + [f"b{r}" for r in range(1, 9)]
        steps = _PICK_ROUTE + " DRAFT.rules.npcs[0].route = ['a6'];" + _line("column", "b2")
        assert _page(spec, steps, "DRAFT.rules.npcs[0].route") == \
            ["a6"] + [f"b{r}" for r in range(8, 0, -1)]

    def test_a_whole_line_already_there_is_taken_out(self) -> None:
        spec = _one_wolf(CIRCUIT)
        steps = _PICK_ROUTE + _line("row", "c5")
        assert _page(spec, steps, "DRAFT.rules.npcs[0].route") == CIRCUIT[:8]

    def test_enter_on_the_board_takes_the_line_too(self) -> None:
        spec = _one_wolf(["a4"])
        steps = _PICK_ROUTE + f" PICK_LINE = 'row'; benchAct({_square('a4')});"
        assert _page(spec, steps, "DRAFT.rules.npcs[0].route") == CIRCUIT[:8]

    def test_each_click_is_said_under_the_board(self) -> None:
        spec = _one_wolf(["a4"])
        steps = _PICK_ROUTE + f" pickSquare({_square('b4')}); const one = BENCH_SAID;"             f" pickSquare({_square('a4')});"
        assert _page(spec, steps, "[one, BENCH_SAID]") == [
            "Added b4 to wild pieces 1's walk, after a4.", "Took a4 out of wild pieces 1's walk."]

    def test_each_click_redraws_the_board_at_once(self) -> None:
        """Step 7: the board waited for the server's check; now it is drawn on the click."""
        spec = _one_wolf(["a4"])
        steps = (_PICK_ROUTE + " let drawn = 0; drawBoardSide = () => { drawn += 1; };"
                 f" pickSquare({_square('b4')});" + _line("row", "a5"))
        assert _page(spec, steps, "drawn") == 2

    def test_the_walk_is_drawn_from_the_draft_before_the_check(self) -> None:
        """The wolf on a4 taken out of its walk: the drawing walks b5 ... a3 at once, and
        no wolf is drawn on a4 any more (the picture a click behind, step 7)."""
        spec = _wolves()
        steps = _PICK_ROUTE + f" pickSquare({_square('a4')});"
        walk, board, stride = _page(
            spec, steps, "(p => [p.walks[0], p.board, p.sideStride || p.nTypes])"
                         "(pickedWalk(lastPreview()))")
        assert walk["entry"] == 0 and not walk["mirrored"]
        assert walk["squares"] == [_square(s) for s in ("b5", "c4", "b3", "a3")]
        assert walk["starts"] == [_square("b5")]
        assert walk["name"] == "wild pieces 1"
        assert not board[_square("a4")]
        assert board[_square("h4")] > 2 * stride   # the other walk's wolf stays

    def test_the_set_up_board_draws_the_pending_walk(self) -> None:
        spec = _one_wolf(["a4"])
        steps = _PICK_ROUTE + _line("row", "a4")
        html = _page(spec, steps, "benchBoardHtml(lastPreview(), 'setup')")
        for a, b in zip(CIRCUIT[:8], CIRCUIT[1:8] + ["a4"], strict=True):
            assert f'data-step="{_square(a)}&gt;{_square(b)}"' in html \
                or f'data-step="{_square(a)}>{_square(b)}"' in html, (a, b)

    def test_the_answer_once_back_is_drawn_unchanged(self) -> None:
        spec = _wolves()
        steps = _PICK_ROUTE
        same = _page(spec, steps, "pickedWalk(lastPreview()) === lastPreview()")
        assert same is True

    def test_the_note_comes_first_under_the_board_while_picking(self) -> None:
        spec = _wolves()
        said = _page(spec, _PICK_ROUTE, "benchSaysHtml('setup')")
        assert said.startswith('<div class="small pickNote" data-picknote')
        assert "data-palettenote" not in said

    def test_done_picking_ends_it(self) -> None:
        steps = _PICK_ROUTE + " PICK_LINE = 'row'; donePicking();"
        assert _page(_wolves(), steps, "[PICKING, PICK_LINE]") == [None, ""]


# --------------------------------------------------------------------------- fix 6

CHESS: dict[str, Any] = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))


def _chess_page(steps: str, expression: str) -> Any:
    """A draft of chess, with the library's wolf on offer (`DESIGN_KIT.palette`)."""
    spec = copy.deepcopy(CHESS)
    answer = designing.check(spec)
    assert answer["valid"], answer["errors"]
    setup = (_QUIET + _kit_setup([WOLVES, CHESS]) + f"DRAFT = {json.dumps(spec)};"
             f" DRAFT_CHECK = {json.dumps(answer)}; DRAFT_MODE = 'setup';"
             " const wolfKey = DESIGN_KIT.palette.find(e => e.name === 'wolf').key;" + steps)
    return _run_page(setup, expression)


class TestAWildPieceFromTheTray:
    def test_the_row_sits_under_the_tray(self) -> None:
        tray = _chess_page("", "paletteHtml()")
        assert tray.index("data-portalrow") < tray.index("data-wildrow")
        assert "+ wild piece</button>" in tray

    def test_pressed_it_offers_the_draft_s_pieces_but_its_king_and_the_library(self) -> None:
        html = _chess_page(" startWildTool();", "wildToolHtml()")
        offered = re.findall(r'data-wildtype="([^"]+)"', html)
        assert sorted(offered) == sorted(p["id"] for p in CHESS["pieces"] if not p.get("royal"))
        assert "data-wildlib" in html and ">wolf</option>" in html

    def test_a_wild_wolf_on_a4_with_a_three_square_walk(self) -> None:
        """The acceptance: from chess, *+ wild piece*, the library's wolf, a4, b4, b3."""
        steps = (" startWildTool(); wildFromLibrary(wolfKey);"
                 f" wildAt({_square('a4')}); pickSquare({_square('b4')});"
                 f" pickSquare({_square('b3')});")
        npcs, picking, wolf, tool = _chess_page(
            steps, "[DRAFT.rules.npcs, PICKING, DRAFT.pieces[DRAFT.pieces.length - 1], WILD_TOOL]")
        assert npcs == [{"piece": "W", "route": ["a4", "b4", "b3"]}]
        assert picking["path"] == ["npcs", 0, "route"]
        assert picking["label"] == "wild pieces 1's walk" and picking["piece"] == "W"
        assert not picking.get("fromCard")
        assert wolf["id"] == "W" and wolf["name"] == "wolf" and tool is None
        made = copy.deepcopy(CHESS)
        made["pieces"].append(wolf)
        made["rules"]["npcs"] = npcs
        assert designing.check(made)["valid"], designing.check(made)["errors"]

    def test_a_draft_s_own_piece_can_be_the_type(self) -> None:
        knight = next(p["id"] for p in CHESS["pieces"] if p.get("name") == "knight")
        steps = (f" startWildTool(); WILD_TOOL.piece = '{knight}'; wildAt({_square('d5')});")
        assert _chess_page(steps, "DRAFT.rules.npcs") == [{"piece": knight, "route": ["d5"]}]

    def test_an_occupied_square_is_refused_in_words(self) -> None:
        steps = (" startWildTool(); wildFromLibrary(wolfKey);"
                 f" wildAt({_square('a2')});")
        npcs, said, tool = _chess_page(steps, "[DRAFT.rules.npcs || null, BENCH_SAID, WILD_TOOL]")
        assert npcs is None and "A piece already stands on a2" in said and tool == {"piece": "W"}

    def test_no_type_yet_asks_for_one(self) -> None:
        steps = f" startWildTool(); wildAt({_square('a4')});"
        npcs, said = _chess_page(steps, "[DRAFT.rules.npcs || null, BENCH_SAID]")
        assert npcs is None and "Pick what the wild piece is first" in said

    def test_the_tray_and_escape_put_it_away(self) -> None:
        assert _chess_page(" startWildTool(); endPickingForTray();", "WILD_TOOL") is None
        assert _chess_page(" startWildTool(); benchEscape();", "WILD_TOOL") is None

    def test_beside_portals_it_is_refused(self) -> None:
        steps = " DRAFT.rules.portals = [{ piece: 'P' }];"
        row = _chess_page(steps, "wildRowHtml()")
        assert "disabled" in row and "can&#39;t be used together with portals" in row \
            or "can't be used together with portals" in row
