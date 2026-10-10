"""Portals in the designer, as P3 of `docs/plan-portals.md` builds them (Q772c, D1-D10).

What can be held without a browser. The server's half: the pair's card in sentences (D4),
the refusals in the designer's words and never the validator's paths (D7), the warnings
when a setting makes a portal useless, with N4's exits read off the compiled table, whether
*+ portal pair* can add another pair (D2), and the lone board that keeps the portals (N5)
with the engine's transits on it. The page's half, run in node against the page's own
script: the tray's row, putting a pair down and the refusals of a click (D2), moving,
removing and renumbering, the card's choices, the keyboard on the bench board and on the
board played inside the designer (D9), *a whole row* (D3) and what an undo says (D8). The
look -- that the card reads, that the glyph and the legs are where they should be at 1400
and 375 -- is the look check's (`docs/reference/designer-portals-look.md`).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from test_designer import _kit_setup, _run_page
from test_portal_board import _EXITS, _SNAPSHOT
from test_web_landing import _HARNESS

from chess_variant_sandbox.web import designing

ROOT = Path(__file__).resolve().parent.parent

# Six files by seven ranks, a wall across rank 4, a king and a rook a side: the walled
# board of Divide, small.
_GAME: dict[str, Any] = {
    "schema_version": 6,
    "name": "doors",
    "board": {"width": 6, "height": 7},
    "pieces": [
        {"id": "K", "name": "king", "royal": True, "value": 0,
         "moves": [{"kind": "leap", "vectors": [[0, 1], [1, 1]], "symmetry": "dihedral8"}]},
        {"id": "R", "name": "rook", "value": 5,
         "moves": [{"kind": "slide", "vectors": [[0, 1]], "symmetry": "rotate4"}]},
    ],
    "setup": {"mirror": True, "placements": [
        {"piece": "K", "squares": ["c1"]}, {"piece": "R", "squares": ["a1"]},
    ]},
    "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 200,
              "obstacles": {"squares": ["a4", "b4", "c4", "d4", "e4", "f4"]}},
}


def _with(*pairs: dict[str, Any], moves: list[Any] | None = None) -> dict[str, Any]:
    spec = copy.deepcopy(_GAME)
    spec["pieces"].append({"id": "O", "name": "portal", "value": 0, "moves": moves or []})
    spec["rules"]["portals"] = [{"piece": "O", **pair} for pair in pairs]
    return spec


def _square(name: str, width: int = 6) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


# ------------------------------------------------------------------ the server's half

class TestTheCardInSentences:
    def test_a_new_pair_says_both_ways_and_nobody_moves_it(self) -> None:
        answer = designing.check(_with({"orange": "b2", "blue": "e6"}))
        assert answer["valid"], answer["errors"]
        (card,) = answer["portals"]["cards"]
        assert (card["pair"], card["orange"], card["blue"]) == (1, "b2", "e6")
        assert card["ways"] == ["White's and Black's pieces can go in at either end and come "
                                "out of the other."]
        assert card["moves"] == "Nobody moves these portals."
        assert card["solid"] == [] and card["warnings"] == []

    def test_one_way_per_player_is_a_sentence_each_and_says_which_end_is_solid(self) -> None:
        spec = _with({"orange": "b2", "blue": "e6",
                      "passage": {"0": "orange_to_blue", "1": "closed"}})
        (card,) = designing.check(spec)["portals"]["cards"]
        assert card["same"] is False
        assert card["ways"] == [
            "White's pieces can go in at the orange end and come out by the blue end, not back.",
            "Black's pieces can't use these portals: to them both ends are solid blocks, like a "
            "wall.",
        ]
        assert card["solid"][0].startswith("The blue end is solid to White's pieces")
        assert card["solid"][1].startswith("To Black's pieces both ends are solid")

    def test_who_moves_it_names_each_end_and_how_a_portal_moves(self) -> None:
        spec = _with({"orange": "b2", "blue": "e6", "moved_by": {"orange": [0, 1]}},
                     moves=[designing.PORTAL_STEP])
        (card,) = designing.check(spec)["portals"]["cards"]
        assert card["moves"].startswith("Either player may spend a turn moving the orange end. "
                                        "Nobody moves the blue end. A portal moves one square in "
                                        "all eight directions, always onto an empty square.")
        assert "open the portal on the piece list" in card["moves"]

    def test_the_bar_on_going_straight_back_is_said_with_who_moves_it(self) -> None:
        """Schema 20 (Q807): the card says the bar where it says who may move the pair."""
        spec = _with({"orange": "b2", "blue": "e6", "moved_by": {"orange": [0, 1]},
                      "no_straight_back": True}, moves=[designing.PORTAL_STEP])
        answer = designing.check(spec)
        assert answer["valid"], answer["errors"]
        (card,) = answer["portals"]["cards"]
        assert card["noStraightBack"] is True
        assert card["moves"].endswith("A portal that has just been moved can't be moved "
                                      "straight back where it came from on the very next move.")
        spec["rules"]["portals"][0].pop("no_straight_back")
        (card,) = designing.check(spec)["portals"]["cards"]
        assert card["noStraightBack"] is False and "straight back" not in card["moves"]

    def test_the_card_never_shows_the_spec_s_words(self) -> None:
        spec = _with({"orange": "b2", "blue": "e6",
                      "passage": {"0": "blue_to_orange", "1": "both_ways"},
                      "moved_by": {"blue": [1]}}, moves=[designing.PORTAL_STEP])
        (card,) = designing.check(spec)["portals"]["cards"]
        said = " ".join(card["ways"] + card["solid"] + [card["moves"]])
        for word in ("passage", "moved_by", "free agent", "blue_to_orange", "owner", "mode",
                     "player 0", "player 1"):
            assert word not in said


class TestTheWarnings:
    def test_a_pair_nobody_can_use_is_two_solid_blocks(self) -> None:
        spec = _with({"orange": "b2", "blue": "e6", "passage": {"0": "closed", "1": "closed"}})
        (card,) = designing.check(spec)["portals"]["cards"]
        assert card["warnings"] == ["Nobody can go through these portals, so they are just two "
                                    "solid blocks. If that is what you want, a wall does the "
                                    "same and is simpler."]

    def test_an_end_with_no_exits_is_read_off_the_compiled_table(self) -> None:
        """N4: the blue end on f7 in a pocket of walls (e6, e7, f6): every square beside it
        is off the board or a wall, and somebody can go in at the orange end."""
        spec = _with({"orange": "b2", "blue": "f7"})
        spec["rules"]["obstacles"]["squares"] += ["e6", "e7", "f6"]
        spec["setup"]["placements"] = [{"piece": "K", "squares": ["c1"]},
                                       {"piece": "R", "squares": ["a1"]}]
        answer = designing.check(spec)
        assert answer["valid"], answer["errors"]
        (card,) = answer["portals"]["cards"]
        assert card["exits"] == {"orange": 8, "blue": 0}
        assert card["warnings"] == ["Nothing can come out of the blue end: every square beside "
                                    "it is off the board or a wall."]

    def test_no_exit_is_no_warning_where_nobody_comes_out_there(self) -> None:
        spec = _with({"orange": "b2", "blue": "f7",
                      "passage": {"0": "blue_to_orange", "1": "blue_to_orange"}})
        spec["rules"]["obstacles"]["squares"] += ["e6", "e7", "f6"]
        (card,) = designing.check(spec)["portals"]["cards"]
        assert card["exits"]["blue"] == 0 and card["warnings"] == []


# Each refusal of D7's table: a draft that earns it, and the sentence the designer shows.
_REFUSALS = [
    ("the portal piece on the board",
     lambda s: s["setup"]["placements"].append({"piece": "O", "squares": ["f1"]}),
     "The portal piece is only for portals: it can't start on the board, be promoted to, or "
     "be taken. Remove it from where they start."),
    ("a portal move that captures",
     lambda s: s["pieces"][-1].update(moves=[{"kind": "leap", "vectors": [[1, 0]],
                                              "symmetry": "rotate4"}]),
     'Portals can\'t take anything, so every way a portal moves must be "move only".'),
    ("a portal move that differs for the two players",
     lambda s: s["pieces"][-1].update(moves=[{"kind": "leap", "vectors": [[0, 1]],
                                              "mode": "move"}]),
     "Anyone may move a portal, so its moves must be the same both ways up."),
    ("an end somebody may move, and the portal piece has no move",
     lambda s: s["rules"]["portals"][0].update(moved_by={"orange": [0]}),
     "Pair 1's orange end may be moved by White, but the portal piece has no way to move. "
     'Give it a move, or set the end to "nobody".'),
    ("an end on a wall",
     lambda s: s["rules"]["portals"][0].update(blue="c4"),
     "Pair 1's blue end is on a wall (c4). Move it to an empty square."),
    ("an end on a starting piece",
     lambda s: s["rules"]["portals"][0].update(orange="a1"),
     "Pair 1's orange end is on a1, where a piece starts. A portal needs an empty square: "
     "move it, or take the piece off first."),
    ("two ends on one square",
     lambda s: s["rules"]["portals"].append({"piece": "O", "orange": "b2", "blue": "d6"}),
     "Pair 2's orange end is on b2, which is pair 1's orange end. Two portals can't share a "
     "square."),
    ("beside a refused rule",
     lambda s: s["rules"].update(explode_on_capture={"radius": 1}),
     "Portals can't be used together with the rule 'captures that explode' yet. Take off one "
     "of them."),
    ("a bar on going straight back with nobody to move the pair",
     lambda s: s["rules"]["portals"][0].update(no_straight_back=True),
     "Pair 1 says a portal can't be moved straight back, but nobody may move its portals. "
     "Let somebody move them, or untick 'not straight back'."),
    ("more than four pairs",
     lambda s: s["rules"]["portals"].extend(
         {"piece": "O", "orange": o, "blue": b}
         for o, b in (("a2", "a6"), ("c2", "c6"), ("d2", "d6"), ("f2", "f5"))),
     "A game can have at most four pairs of portals."),
]


class TestTheRefusalsInWords:
    @pytest.mark.parametrize(("why", "make", "says"), _REFUSALS, ids=[r[0] for r in _REFUSALS])
    def test_each_refusal_reads_as_the_designer_says_it(self, why: str, make: Any,
                                                        says: str) -> None:
        spec = _with({"orange": "b2", "blue": "e6"})
        make(spec)
        answer = designing.check(spec)
        assert not answer["valid"]
        assert says in answer["errors"], answer["errors"]
        # Never the validator's code: no path, no field name, no player's number.
        for line in answer["errors"]:
            assert "rules.portals" not in line and "moved_by" not in line, line

    def test_a_refusal_about_an_end_names_its_pair_for_the_card(self) -> None:
        spec = _with({"orange": "b2", "blue": "c4"})
        errors = designing.check(spec)["portals"]["errors"]
        assert {"says": "Pair 1's blue end is on a wall (c4). Move it to an empty square.",
                "pair": 1, "end": "blue", "square": "c4"} in errors

    def test_the_hexagon_and_too_many_kinds_of_piece_have_their_sentences(self) -> None:
        assert designing.portal_problem(
            "rules.portals is refused on a hex lattice in this schema: x")["says"].startswith(
            "Portals can't be used on a hexagon board yet")
        said = designing.portal_problem("at most 24 piece types, and rules.portals adds two per "
                                        "pair: 21 declared and 4 portal ends")
        assert said == {"says": "This game has too many kinds of piece for another portal "
                                "pair: each pair counts as two.", "beside": True}


class TestWhetherAnotherPairCanGoIn:
    def test_a_plain_draft_takes_a_pair(self) -> None:
        assert designing.portal_tool(copy.deepcopy(_GAME)) == {"open": True, "why": ""}

    def test_four_pairs_is_the_most(self) -> None:
        spec = _with(*({"orange": o, "blue": b} for o, b in (("a2", "a6"), ("b2", "b6"),
                                                              ("c2", "c6"), ("d2", "d6"))))
        assert designing.portal_tool(spec) == {
            "open": False, "why": "A game can have at most four pairs of portals."}

    def test_beside_wild_pieces_the_button_says_why_not(self) -> None:
        wolves = json.loads((ROOT / "archive" / "wolves.json").read_text(encoding="utf-8"))
        tool = designing.portal_tool(wolves)
        assert tool == {"open": False, "why": "Portals can't be used together with the rule "
                                              "'wild pieces' yet. Take off one of them."}


class TestTheLoneBoardKeepsThePortals:
    def test_the_engine_s_transits_are_on_the_map(self) -> None:
        """N5, D5: the rook alone on b1 among the portals, pair 1 from b2 to e6. Going up into
        b2 it comes out one step above e6, on e7: a transit the engine names, with its way
        out, and the squares it reaches straight beside it."""
        spec = _with({"orange": "b2", "blue": "e6"})
        rook = 1
        moved = designing.move_map(spec, rook, _square("b1"))
        assert moved["from"] == _square("b1")
        (through,) = moved["transits"]
        assert (through["to"], through["portal"], through["portalOut"]) == (
            _square("e7"), _square("b2"), _square("e6"))
        assert through["exitDir"] == 2  # up, in `PORTAL_DIRECTIONS` order
        assert _square("e7") in moved["moves"] and _square("e7") not in moved["direct"]
        assert _square("a1") in moved["direct"]

    def test_the_piece_never_stands_on_a_portal(self) -> None:
        spec = _with({"orange": "b2", "blue": "e6"})
        assert designing.move_map(spec, 1, _square("b2"))["from"] != _square("b2")

    def test_where_it_goes_reads_the_opening_position(self) -> None:
        ferry = json.loads((ROOT / "archive" / "ferry.json").read_text(encoding="utf-8"))
        pawn = [piece["id"] for piece in ferry["pieces"]].index("P")
        live = designing.move_map(ferry, pawn, _square("c2"), alone=False)
        assert {(t["to"], t["portal"]) for t in live["transits"]} >= {
            (_square("d5"), _square("b3"))}
        assert _square("d5") in live["takes"]

    def test_a_game_without_portals_has_maps_as_they_were(self) -> None:
        assert "transits" not in designing.move_map(copy.deepcopy(_GAME), 1, _square("b1"))


# ------------------------------------------------------------------ the page's half

def _preview(spec: dict[str, Any]) -> dict[str, Any]:
    answer = designing.check(spec)
    preview = answer["preview"]
    assert preview is not None, answer["errors"]
    return preview


# The redraws and the re-check need a real DOM and a server; what is under test is what the
# page's portal code writes into the draft and says, so they are stubbed.
_QUIET = ("drawDesigner = () => {}; checkDraft = () => {}; drawBoardSide = () => {};"
          " loadMoveMap = () => {}; drawPortalCards = () => {}; showPairCard = () => {};"
          " endBenchGame = () => {};")


def _page(spec: dict[str, Any], steps: str, expression: str, check: dict | None = None) -> Any:
    answer = check or {"valid": True, "preview": _preview(spec), "portals": {
        "cards": [], "errors": [], "tool": {"open": True, "why": ""}}}
    setup = (_QUIET + f"DRAFT = {json.dumps(spec)}; DRAFT_CHECK = {json.dumps(answer)};"
             f" DRAFT_MODE = 'setup';" + steps)
    return _run_page(setup, expression)


def _clicks(*names: str) -> str:
    return "".join(f"portalClick({_square(name)});" for name in names)


class TestTheTraysRow:
    def test_the_row_offers_a_pair_and_hides_the_portal_piece(self) -> None:
        html = _page(_with({"orange": "b2", "blue": "e6"}), "", "paletteHtml()")
        assert "+ portal pair" in html and 'data-portalchip="0"' in html
        assert 'data-placing="2.0"' not in html  # the portal piece is never put down by hand
        assert 'data-placing="0.0"' in html and 'data-placing="1.0"' in html

    def test_at_four_pairs_the_button_is_off_and_says_why(self) -> None:
        spec = _with(*({"orange": o, "blue": b} for o, b in (("a2", "a6"), ("b2", "b6"),
                                                              ("c2", "c6"), ("d2", "d6"))))
        html = _page(spec, "", "portalRowHtml()")
        assert "disabled" in html and "A game can have at most four pairs of portals." in html

    def test_the_server_s_refusal_disables_it_too(self) -> None:
        check = {"valid": True, "preview": _preview(copy.deepcopy(_GAME)), "portals": {
            "cards": [], "errors": [], "tool": {"open": False, "why": "Portals can't be used "
                                                "on a hexagon board yet."}}}
        html = _page(copy.deepcopy(_GAME), "", "portalRowHtml()", check)
        assert 'disabled title="Portals can&#39;t be used on a hexagon board yet."' in html \
            or "disabled title=\"Portals can't be used on a hexagon board yet.\"" in html


class TestPuttingAPairDown:
    def test_two_clicks_make_a_pair_and_the_unseen_portal_piece(self) -> None:
        said, pieces, pairs = _page(copy.deepcopy(_GAME),
                                    "startPortalTool();" + _clicks("b2", "e6"),
                                    "[BENCH_SAID, DRAFT.pieces, DRAFT.rules.portals]")
        assert pairs == [{"piece": "O", "orange": "b2", "blue": "e6"}]
        assert pieces[-1] == {"id": "O", "name": "portal", "value": 0, "moves": []}
        assert said.startswith("Pair 1 is on b2 and e6, and picked")
        assert "among the rules" in said and "beside the board" not in said
        # The draft the page wrote is a game, with the pair where it was put.
        spec = copy.deepcopy(_GAME)
        spec["pieces"], spec["rules"]["portals"] = pieces, pairs
        assert designing.check(spec)["valid"]

    def test_a_pair_the_check_has_not_drawn_yet_is_drawn_meanwhile(self) -> None:
        """The cold reader saw two empty rings where a pair had just gone down: until the
        check's board has the pair, the bench draws its ends itself, orange and blue."""
        pending = _page(copy.deepcopy(_GAME), "startPortalTool();" + _clicks("b2", "e6"),
                        "benchMarks('setup').pending")
        assert pending == [{"square": _square("b2"), "pair": 1, "end": "orange"},
                           {"square": _square("e6"), "pair": 1, "end": "blue"}]
        drawn = _page(_with({"orange": "b2", "blue": "e6"}), "", "benchMarks('setup').pending")
        assert drawn == []

    def test_the_status_line_says_what_the_next_click_does(self) -> None:
        first, second = _page(copy.deepcopy(_GAME), "startPortalTool(); const a = BENCH_SAID;"
                              + _clicks("b2"), "[a, BENCH_SAID]")
        assert first == "Click an empty square for the orange end of pair 1."
        assert second == "Click an empty square for the blue end of pair 1."

    @pytest.mark.parametrize(("where", "says"), [
        ("a1", "a1 has a rook on it. A portal needs an empty square: pick another, or take "
               "the rook off first."),
        ("c4", "c4 is a wall: nothing can stand there, a portal included."),
        ("b2", "That is where the orange end is. The blue end goes on a different square."),
        ("d2", "That is pair 1's orange end. Two portals can't share a square."),
    ])
    def test_a_refused_click_says_why_and_puts_nothing_down(self, where: str, says: str) -> None:
        spec = _with({"orange": "d2", "blue": "d6"})
        said, pairs, flash = _page(spec, "startPortalTool();" + _clicks("b2", where),
                                   "[BENCH_SAID, DRAFT.rules.portals.length, BENCH_FLASH]")
        assert said == says and pairs == 1 and flash == _square(where)

    def test_escape_or_undo_cancels_a_placement_and_leaves_nothing(self) -> None:
        for cancel in ("benchKey('Escape');", "stepDraft(false);"):
            tool, pairs, said = _page(copy.deepcopy(_GAME), "startPortalTool();"
                                      + _clicks("b2") + cancel,
                                      "[PORTAL_TOOL, (DRAFT.rules.portals || []).length, BENCH_SAID]")
            assert tool is None and pairs == 0 and said == "Put away: no pair was added."


class TestMovingRemovingAndRenumbering:
    def test_an_end_is_lifted_and_put_on_an_empty_square(self) -> None:
        lifted, pairs, said = _page(_with({"orange": "b2", "blue": "e6"}),
                                    _clicks("e6") + "const up = PORTAL_LIFT;" + _clicks("f6"),
                                    "[up, DRAFT.rules.portals, BENCH_SAID]")
        assert lifted == {"pair": 0, "end": "blue"}
        assert pairs[0]["blue"] == "f6" and said == "Pair 1's blue end is on f6 now."

    def test_the_end_clicked_again_is_put_back(self) -> None:
        lift, said = _page(_with({"orange": "b2", "blue": "e6"}), _clicks("e6", "e6"),
                           "[PORTAL_LIFT, BENCH_SAID]")
        assert lift is None and said == "Pair 1's blue end stays on e6."

    def test_take_off_takes_the_whole_pair_and_renumbers_the_rest(self) -> None:
        spec = _with({"orange": "b2", "blue": "e6"}, {"orange": "c2", "blue": "d6"})
        pairs, said = _page(spec, "PLACING = { erase: true };" + _clicks("e6"),
                            "[DRAFT.rules.portals, BENCH_SAID]")
        assert pairs == [{"piece": "O", "orange": "c2", "blue": "d6"}]
        assert said == "Took off pair 1, both ends. Undo brings it back. The pair that was 2 " \
                       "is now 1."

    def test_the_last_pair_taken_off_takes_the_portal_piece_too(self) -> None:
        rules, pieces = _page(_with({"orange": "b2", "blue": "e6"}), "removePair(0);",
                              "[DRAFT.rules, DRAFT.pieces.map(p => p.id)]")
        assert "portals" not in rules and pieces == ["K", "R"]

    def test_make_this_pair_1_moves_it_to_the_front(self) -> None:
        spec = _with({"orange": "b2", "blue": "e6"}, {"orange": "c2", "blue": "d6"})
        pairs, said = _page(spec, "renumberPair(1, 0);", "[DRAFT.rules.portals, BENCH_SAID]")
        assert [p["orange"] for p in pairs] == ["c2", "b2"]
        assert said.startswith("Pair 2 is now pair 1; the pair that was 1 is now 2.")


class TestTheCardsChoices:
    def test_a_way_for_both_and_for_one_player(self) -> None:
        both, one = _page(_with({"orange": "b2", "blue": "e6"}),
                          "setPassage(0, null, 'orange_to_blue'); const a = "
                          "JSON.parse(JSON.stringify(DRAFT.rules.portals[0]));"
                          " setPassage(0, 1, 'both_ways');", "[a, DRAFT.rules.portals[0]]")
        assert both["passage"] == {"0": "orange_to_blue", "1": "orange_to_blue"}
        assert one["passage"] == {"0": "orange_to_blue"}

    def test_somebody_moving_an_end_gives_the_portal_piece_its_step(self) -> None:
        pair, piece = _page(_with({"orange": "b2", "blue": "e6"}),
                            "setMovers(0, 'orange', [0, 1]);",
                            "[DRAFT.rules.portals[0], DRAFT.pieces[2]]")
        assert pair["moved_by"] == {"orange": [0, 1]}
        assert piece["moves"] == [designing.PORTAL_STEP]
        # Back to nobody: the key goes, and the step stays for the designer to keep or drop.
        pair = _page(_with({"orange": "b2", "blue": "e6", "moved_by": {"orange": [0]}}),
                     "setMovers(0, 'orange', []);", "DRAFT.rules.portals[0]")
        assert "moved_by" not in pair

    def test_the_bar_is_a_tick_while_somebody_moves_the_pair(self) -> None:
        """Schema 20 (Q807): *not straight back* is offered only where somebody moves the
        pair, writes the flag, and goes with the last mover, as the validator asks."""
        spec = _with({"orange": "b2", "blue": "e6", "moved_by": {"orange": [0, 1]}},
                     moves=[designing.PORTAL_STEP])
        html = _page(spec, "", "portalCardsHtml([])", designing.check(spec))
        assert 'data-pback="0"' in html and "Not straight back" in html
        unmoved = _with({"orange": "b2", "blue": "e6"})
        assert 'data-pback="0"' not in _page(unmoved, "", "portalCardsHtml([])",
                                             designing.check(unmoved))
        on, off = _page(spec, "setNoStraightBack(0, true); const a = "
                        "JSON.parse(JSON.stringify(DRAFT.rules.portals[0]));"
                        " setMovers(0, 'orange', []);", "[a, DRAFT.rules.portals[0]]")
        assert on["no_straight_back"] is True
        assert "no_straight_back" not in off and "moved_by" not in off

    def test_the_card_draws_the_four_ways_and_the_server_s_sentences(self) -> None:
        spec = _with({"orange": "b2", "blue": "e6"})
        check = designing.check(spec)
        html = _page(spec, "", "portalCardsHtml([])", check)
        for name in ("both ways", "one way: orange to blue", "one way: blue to orange", "no way"):
            assert f"<span>{name}</span>" in html
        assert "White&#39;s and Black&#39;s pieces" in html or "White's and Black's pieces" in html
        assert "Who may move these portals?" in html and "either player" in html
        assert "rules.portals" not in html

    def test_an_edit_to_another_rule_card_keeps_the_pairs(self) -> None:
        """The form's round trip: the portals' card records the rule as one held value."""
        spec = _with({"orange": "b2", "blue": "e6"})
        built = _page(spec, _kit_setup() + " const out = []; rulesCardsHtml(out);",
                      "buildControls(out)")
        assert built["portals"] == spec["rules"]["portals"]

    def test_the_portal_piece_s_move_menus_offer_only_what_a_portal_may_do(self) -> None:
        html = _page(_with({"orange": "b2", "blue": "e6", "moved_by": {"orange": [0]}},
                           moves=[designing.PORTAL_STEP]), "",
                     "moveEditor(DRAFT.pieces[2].moves[0], 2, 0)")
        assert "move only" in html and "move or take" not in html and "take only" not in html
        assert 'value="sweep"' not in html and 'value="none"' not in html and "four ways" in html


class TestTheKeyboard:
    def test_arrows_and_enter_place_a_pair_with_no_pointer(self) -> None:
        """D9: from the middle of the board, the arrows walk to b2 and Enter puts the orange
        end down, then to e6 and Enter puts the blue; Esc cancels nothing then."""
        start = _square("c4")  # the middle of six by seven: file 3 rounds down to c... d4
        steps = ("startPortalTool(); benchKey('ArrowUp');"  # the first arrow lands mid-board
                 "const mid = BENCH_FOCUS;")
        mid = _page(copy.deepcopy(_GAME), steps, "mid")
        assert mid == 3 * 6 + 3  # d4, the middle of six by seven
        walk = ("startPortalTool(); benchKey('ArrowUp');"
                " benchKey('ArrowLeft'); benchKey('ArrowLeft'); benchKey('ArrowDown');"
                " benchKey('ArrowDown'); benchKey('Enter');"
                " benchKey('ArrowRight'); benchKey('ArrowRight'); benchKey('ArrowRight');"
                " for (let i = 0; i < 4; i++) benchKey('ArrowUp'); benchKey('Enter');")
        pairs = _page(copy.deepcopy(_GAME), walk, "DRAFT.rules.portals")
        assert pairs == [{"piece": "O", "orange": "b2", "blue": "e6"}]
        assert start != mid

    def test_the_keys_lift_move_and_take_off_a_pair(self) -> None:
        to_e6 = "benchKey('ArrowUp'); benchKey('ArrowRight'); benchKey('ArrowUp'); benchKey('ArrowUp');"
        moved = _page(_with({"orange": "b2", "blue": "e6"}),
                      to_e6 + "benchKey('Enter'); benchKey('ArrowRight'); benchKey('Enter');",
                      "DRAFT.rules.portals[0].blue")
        assert moved == "f6"
        gone = _page(_with({"orange": "b2", "blue": "e6"}),
                     "PLACING = { erase: true };" + to_e6 + "benchKey('Enter');",
                     "(DRAFT.rules.portals || []).length")
        assert gone == 0

    def test_escape_puts_a_lifted_end_back(self) -> None:
        lift, pair = _page(_with({"orange": "b2", "blue": "e6"}),
                           _clicks("e6") + "benchKey('Escape');",
                           "[PORTAL_LIFT, DRAFT.rules.portals[0]]")
        assert lift is None and pair["blue"] == "e6"

    def test_the_board_walks_over_cells_it_does_not_have(self) -> None:
        """The arrows stop at the edge rather than leave the board."""
        at = _page(copy.deepcopy(_GAME), "BENCH_FOCUS = 0; benchKey('ArrowLeft');"
                   " benchKey('ArrowDown');", "BENCH_FOCUS")
        assert at == 0


class TestAWholeRow:
    def test_a_row_joins_the_list_and_leaves_it_again(self) -> None:
        spec = copy.deepcopy(_GAME)
        steps = ("structural = (root, change) => change(rootTarget(root));"
                 " PICKING = { root: 'rules', path: ['obstacles', 'squares'], label: 'walls' };"
                 f" PICK_LINE = 'row'; pickLine({_square('b5')});"
                 " const once = DRAFT.rules.obstacles.squares.slice(); const said = BENCH_SAID;"
                 f" PICK_LINE = 'row'; pickLine({_square('e5')});")
        once, said, twice, line = _page(spec, steps,
                                        "[once, said, DRAFT.rules.obstacles.squares, PICK_LINE]")
        assert once == ["a4", "b4", "c4", "d4", "e4", "f4", "a5", "b5", "c5", "d5", "e5", "f5"]
        assert said == "Added row 5 to walls."
        assert twice == ["a4", "b4", "c4", "d4", "e4", "f4"] and line == ""

    def test_a_column_is_its_file(self) -> None:
        steps = ("structural = (root, change) => change(rootTarget(root));"
                 " PICKING = { root: 'rules', path: ['obstacles', 'squares'], label: 'walls' };"
                 f" PICK_LINE = 'column'; pickLine({_square('f2')});")
        squares = _page(copy.deepcopy(_GAME), steps, "DRAFT.rules.obstacles.squares")
        assert squares[6:] == ["f1", "f2", "f3", "f5", "f6", "f7"]


class TestWhatAnUndoSays:
    @pytest.mark.parametrize(("before", "after", "says"), [
        ([], [{"orange": "b4", "blue": "e3"}], "Pair 1 is back, on b4 and e3."),
        ([{"orange": "b2", "blue": "e6"}], [], "Pair 1 is off the board again."),
        ([{"orange": "b2", "blue": "e6"}], [{"orange": "b2", "blue": "f6"}],
         "Pair 1's ends are back on b2 and f6."),
        ([{"orange": "b2", "blue": "e6"}], [{"orange": "b2", "blue": "e6", "passage": {"0": "closed"}}],
         "Pair 1's card is back as it was."),
    ])
    def test_the_status_line_says_what_came_back(self, before: list, after: list,
                                                 says: str) -> None:
        draft = lambda pairs: {"rules": {"portals": pairs}}  # noqa: E731
        said = _page(copy.deepcopy(_GAME), "",
                     f"portalUndoWords({json.dumps(draft(before))}, {json.dumps(draft(after))})")
        assert said == says


class TestTheChooserByKeyboard:
    """D9 on the board played in the designer (*play it here*): the cylinder wormhole's rook
    on a4, which reaches d4 both ways round, picked and sent into the portal by keys alone,
    and the chooser answered by a number key."""

    def _run(self, steps: str, expression: str) -> Any:
        from test_web import _in_page

        setup = (f"SESSION = {json.dumps(_SNAPSHOT)};\n" + _HARNESS
                 + _EXITS % {"a4": 24, "d4": 27} + "BOARD_INLINE = true;\n")
        return _in_page(setup + steps, expression)

    def test_the_keys_pick_the_rook_open_the_chooser_and_take_a_way(self) -> None:
        keys = ("const press = k => boardKeys({ key: k, preventDefault() {}, stopPropagation() {} });"
                " BOARD_FOCUS = 24; press('Enter');"
                " press('ArrowRight'); press('ArrowRight'); press('ArrowRight'); press('Enter');")
        stage, focus = self._run(keys, "[promotionStage(), BOARD_FOCUS]")
        assert stage == "exit" and focus == 27
        sent = self._run(keys + " key('2');",
                         "CALLS.map(c => textOf(c[1].move))")
        assert sent == ["Ra4-d4>e5"]

    def test_the_board_s_own_page_keeps_its_arrows_for_the_game(self) -> None:
        moved = self._run("BOARD_INLINE = false; BOARD_FOCUS = 24;"
                          " boardKeys({ key: 'ArrowRight', preventDefault() {}, stopPropagation() {} });",
                          "BOARD_FOCUS")
        assert moved == 24
