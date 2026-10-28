"""The designer writes a hexagon and its seams (Q281c, schema 16).

`docs/game-ideas.md` *Q281 specified*, *How the board draws*, the designer's list: the
board form offers a rhombus or a hexagon and one *cells on a side* control, the wrap card
shows a hexagon its three pairs of sides and nothing else, the tray puts pieces only on
cells, and `teach` offers the offset through each copy of the board the draft joins.

The page's half runs in node against the page's own script, as `test_designer.py` does;
the server's half is `designing.teach` and its probe, asked here directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from retired_specs import hex_glinski_flanks, hex_glinski_flanks_one_way, hex_glinski_promoting
from test_designer import _HAND_TRIP, _ROUND_TRIP, _kit_setup, _run_page

from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.schema import absent_squares
from chess_variant_sandbox.web import cards, designing

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "archive"
RULESETS = ROOT / "rulesets"


def _committed(name: str) -> Path:
    """A ruleset's file, in `rulesets/` or, for one moved to the Library (Q512), `archive/`."""
    path = RULESETS / f"{name}.json"
    return path if path.exists() else ARCHIVE / f"{name}.json"


def _square(name: str, width: int) -> int:
    return (int(name[1:]) - 1) * width + (ord(name[0]) - ord("a"))


def _name(square: int, width: int) -> str:
    return f"{chr(ord('a') + square % width)}{square // width + 1}"


# ------------------------------------------------------------------ the board form


def _board_after(board: dict[str, Any], *writes: tuple[str, Any]) -> dict[str, Any]:
    steps = "".join(
        f"writeBoardField(B, {json.dumps(field)}, {json.dumps(value)});" for field, value in writes
    )
    return _run_page(f"const B = {json.dumps(board)}; {steps}", "B")


class TestTheBoardFormOffersAHexagon:
    def test_choosing_the_hexagon_makes_the_frame_one(self) -> None:
        """Glinski's 11 by 11 rhombus is six cells a side, and stays that size."""
        board = _board_after({"width": 11, "height": 11, "lattice": "hex"}, ("shape", "hexagon"))
        assert board == {"width": 11, "height": 11, "lattice": "hex", "shape": "hexagon"}
        assert validate_spec({**load_spec(ARCHIVE / "hex-glinski-hexagon.json"),
                              "board": board}) == []

    def test_a_board_of_any_size_becomes_an_odd_square_frame(self) -> None:
        board = _board_after({"width": 6, "height": 8, "lattice": "hex"}, ("shape", "hexagon"))
        assert (board["width"], board["height"]) == (7, 7)

    @pytest.mark.parametrize(("side", "across"), [(2, 3), (3, 5), (6, 11), (8, 15)])
    def test_cells_on_a_side_writes_both_sizes(self, side: int, across: int) -> None:
        board = _board_after(
            {"width": 11, "height": 11, "lattice": "hex", "shape": "hexagon"}, ("side", str(side))
        )
        assert (board["width"], board["height"]) == (across, across)

    @pytest.mark.parametrize(("side", "across"), [("1", 3), ("12", 15), ("0", 3), ("", 5)])
    def test_cells_on_a_side_is_held_to_the_schemas_hexagons(self, side: str, across: int) -> None:
        board = _board_after(
            {"width": 5, "height": 5, "lattice": "hex", "shape": "hexagon"}, ("side", side)
        )
        assert (board["width"], board["height"]) == (across, across)

    def test_the_rhombus_takes_the_shape_away(self) -> None:
        board = _board_after(
            {"width": 11, "height": 11, "lattice": "hex", "shape": "hexagon"}, ("shape", "grid")
        )
        assert board == {"width": 11, "height": 11, "lattice": "hex"}

    def test_squares_take_the_hexagon_away_too(self) -> None:
        """The validator allows a hexagon on the hex lattice only (refusal 2)."""
        board = _board_after(
            {"width": 11, "height": 11, "lattice": "hex", "shape": "hexagon"},
            ("lattice", "square"),
        )
        assert board == {"width": 11, "height": 11}

    def test_grid_written_out_stays_written(self) -> None:
        """A different hash, as `lattice: "square"` is, so a load changes nothing."""
        board = {"width": 11, "height": 11, "lattice": "hex", "shape": "grid"}
        assert _board_after(board, ("shape", "grid"), ("width", "11")) == board

    def test_a_hexagon_draws_one_size_and_a_rhombus_two(self) -> None:
        drawn = _run_page(
            "", "[boardFormHtml({width: 11, height: 11, lattice: 'hex', shape: 'hexagon'}),"
            " boardFormHtml({width: 11, height: 11, lattice: 'hex'}),"
            " boardFormHtml({width: 8, height: 8})]",
        )
        hexagon, rhombus, square = drawn
        assert 'data-bfield="side"' in hexagon and 'data-bfield="width"' not in hexagon
        assert 'value="6"' in hexagon and "a hexagon</option>" in hexagon
        assert 'data-bfield="width"' in rhombus and 'data-bfield="shape"' in rhombus
        assert 'data-bfield="shape"' not in square and "a hexagon" not in square


# ------------------------------------------------------------------ the wrap card


def _wrap_card(spec: dict[str, Any]) -> str:
    return _run_page(
        _kit_setup() + f"DRAFT = {json.dumps(spec)}; DRAFT_CHECK = null;",
        "rulesCardsHtml([])",
    )


def _paths(html: str) -> set[str]:
    import re

    return {
        ".".join(json.loads(raw.replace("&quot;", '"')))
        for raw in re.findall(r'data-cpath="([^"]*)"', html)
    }


class TestTheWrapCardShowsTheBoardsOwnSides:
    def test_the_server_says_which_keys_are_whose(self) -> None:
        keys = cards.card_model()["shapeKeys"]["rules.wrap"]
        assert keys == {
            "grid": ["files", "files_by_player", "ranks", "ranks_by_player"],
            "hexagon": ["n_s", "n_s_by_player", "ne_sw", "ne_sw_by_player",
                        "nw_se", "nw_se_by_player"],
        }

    def test_a_hexagon_gets_three_rows_and_no_axis(self) -> None:
        html = _wrap_card(hex_glinski_flanks())
        paths = _paths(html)
        assert {"wrap.n_s", "wrap.ne_sw", "wrap.nw_se"} <= paths
        assert not {"wrap.files", "wrap.ranks"} & paths
        for group in ("the top and bottom sides", "the upper-right and lower-left sides",
                      "the upper-left and lower-right sides"):
            assert group in html
        assert "the side edges" not in html and "the top and bottom edges" not in html

    def test_the_rows_are_in_the_boards_words(self) -> None:
        html = _wrap_card(hex_glinski_flanks())
        for words in ("top and bottom", "not joined", "both ways",
                      "one way, out by the top", "one way, out by the lower left",
                      "one way, out by the upper left"):
            assert words in html, words
        assert ">n<" not in html and ">sw<" not in html

    @pytest.mark.parametrize("name", ["cylinder", "onewaytorus"])
    def test_a_rectangle_keeps_its_two_axes_and_its_words(self, name: str) -> None:
        html = _wrap_card(load_spec(_committed(name)))
        paths = {path for path in _paths(html) if path.startswith("wrap.")}
        assert paths and all(path.split(".")[1].startswith(("files", "ranks")) for path in paths)
        assert "the side edges" in html and "the top and bottom sides" not in html
        assert "not joined" not in html

    def test_a_key_of_the_other_shape_the_draft_holds_is_still_drawn(self) -> None:
        """What the draft says stays on screen beside the validator's refusal of it."""
        spec = load_spec(ARCHIVE / "hex-glinski-hexagon.json")
        spec["rules"]["wrap"] = {"files": "both"}
        assert "wrap.files" in _paths(_wrap_card(spec))

    def test_changing_the_shape_drops_the_other_shapes_seams(self) -> None:
        spec = load_spec(RULESETS / "cylinder.json")
        out = _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(spec)};"
            " DRAFT.rules.wrap.n_s = 'both'; DRAFT.board.lattice = 'hex';"
            " writeBoardField(DRAFT.board, 'shape', 'hexagon'); fitWrapToShape();"
            " const hexagon = JSON.parse(JSON.stringify(DRAFT.rules.wrap));"
            " writeBoardField(DRAFT.board, 'shape', 'grid'); fitWrapToShape();",
            "[hexagon, DRAFT.rules.wrap === undefined]",
        )
        assert out == [{"n_s": "both"}, True]


# ------------------------------------------------------------------ a draft made that way

# The designer's own writers, in the order a person meets them: the board form, the
# orientation menu, the tray over the first two ranks (every square, corners included),
# then *add a rule*, the wrap card's per-player switch for the top and bottom, and a value
# in each of the three rows.
_MAKE = """
function make(start, preview) {
  drawDesigner = () => {}; checkDraft = () => {}; loadMoveMap = () => {};
  DRAFT = start;
  writeBoardField(DRAFT.board, 'lattice', 'hex'); fitWrapToShape();
  writeBoardField(DRAFT.board, 'shape', 'hexagon'); fitWrapToShape();
  writeBoardField(DRAFT.board, 'side', '4');
  DRAFT.setup.orientation = 'rotate';
  DRAFT.setup.placements = [];
  DRAFT_CHECK = { preview };
  PLACING = { piece: 0, player: 0 };
  for (let square = 0; square < 2 * preview.width; square++) placeAt(square);
  const wrap = cardModel().rules.find(f => f.key === 'wrap');
  DRAFT.rules.wrap = fieldDefault(wrap, ['wrap'], 'rules');
  const group = choiceGroups('rules.wrap')
    .findIndex(g => g.options.some(o => o.keys.includes('n_s')));
  switchChoice('rules', ['wrap'], group, 1);
  const out = [];
  rulesCardsHtml(out);
  const say = { 'wrap.n_s_by_player.0': 'n', 'wrap.n_s_by_player.1': 's',
                'wrap.ne_sw': 'both', 'wrap.nw_se': 'se' };
  out.forEach(control => {
    const at = control.path.join('.');
    if (at in say) control.raw = JSON.stringify(say[at]);
  });
  mergeGenerated(DRAFT.rules, buildControls(out), rootFields('rules'));
  return DRAFT;
}
"""


@pytest.fixture(scope="module")
def made() -> dict[str, Any]:
    start = designing.blank_spec()
    start["pieces"][0]["moves"] = [{
        "kind": "leap", "vectors": [[0, 1], [1, 0], [-1, 1], [0, -1], [-1, 0], [1, -1]],
    }]
    board = {"width": 7, "height": 7, "lattice": "hex", "shape": "hexagon"}
    preview = {"width": 7, "height": 7, "board": [0] * 49,
               "absent": sorted(absent_squares(board)), "obstacles": []}
    return _run_page(
        _kit_setup() + _MAKE, f"make({json.dumps(start)}, {json.dumps(preview)})"
    )


class TestADraftMadeThatWay:
    def test_it_is_a_hexagon_with_its_three_seams(self, made) -> None:
        assert made["board"] == {"width": 7, "height": 7, "lattice": "hex", "shape": "hexagon"}
        assert made["rules"]["wrap"] == {
            "n_s_by_player": {"0": "n", "1": "s"}, "ne_sw": "both", "nw_se": "se",
        }

    def test_the_tray_put_pieces_only_on_cells(self, made) -> None:
        absent = {_name(s, 7) for s in absent_squares(made["board"])}
        placed = [name for row in made["setup"]["placements"] for name in row["squares"]]
        # Ranks 1 and 2 of a side-four hexagon: four cells, then five.
        assert sorted(placed) == sorted(["d1", "e1", "f1", "g1", "c2", "d2", "e2", "f2", "g2"])
        assert not absent & set(placed)

    def test_it_validates_and_plays(self, made) -> None:
        assert validate_spec(designing.settled(made)) == []
        answer = designing.check(made)
        assert answer["valid"], answer["errors"]
        assert answer["preview"]["absent"] == sorted(absent_squares(made["board"]))
        assert answer["preview"]["wrap"]["copies"]

    def test_it_round_trips_through_the_form_byte_for_byte(self, made) -> None:
        text = json.dumps(made)
        kept = _run_page(
            _kit_setup() + _ROUND_TRIP + _HAND_TRIP + f"const SPECS = {{made: {text}}};",
            "[roundTrip(SPECS).made, handTrip(SPECS).made]",
        )
        for trip in kept:
            assert trip["missing"] == []
            assert json.dumps(trip["spec"]) == text
        assert spec_hash(designing.settled(kept[0]["spec"])) == spec_hash(designing.settled(made))


# ------------------------------------------------------------------ teaching across a seam


def _teach(spec: dict[str, Any], origin: str, clicks: list[str]) -> list[dict[str, Any]]:
    width = spec["board"]["width"]
    return designing.teach(
        spec, 0, _square(origin, width), [_square(one, width) for one in clicks]
    )["offers"]


class TestTeachReachesAcrossTheSeam:
    def test_a_step_off_the_lower_right_comes_in_on_the_upper_left(self) -> None:
        """The spec's second worked crossing: `k3` stepping right lands on `a8`. A raw
        difference of file and rank is ten files left and five ranks up."""
        offers = _teach(hex_glinski_flanks(), "k3", ["a8"])
        assert offers and offers[0]["exact"]
        assert offers[0]["move"] == {"kind": "leap", "vectors": [[1, 0]]}
        assert offers[0]["says"] == "one square right"

    def test_the_same_click_on_the_unjoined_twin_offers_no_crossing(self) -> None:
        offers = _teach(load_spec(ARCHIVE / "hex-glinski-hexagon.json"), "k3", ["a8"])
        assert all(offer["move"]["vectors"] != [[1, 0]] for offer in offers)

    def test_a_one_way_seam_is_crossed_only_the_way_the_first_player_may(self) -> None:
        """`nw_se_by_player {"0": "se"}`: out by the lower right, never the upper left."""
        spec = hex_glinski_flanks_one_way()
        assert spec["rules"]["wrap"]["nw_se_by_player"]["0"] == "se"
        across = _teach(spec, "k3", ["a8"])
        assert across and across[0]["move"] == {"kind": "leap", "vectors": [[1, 0]]}
        back = _teach(spec, "a8", ["k3"])
        assert all(offer["move"]["vectors"] != [[-1, 0]] for offer in back)

    def test_every_offset_comes_from_the_compilers_copies(self) -> None:
        spec = designing.settled(hex_glinski_flanks())
        geometry = designing._geometry(json.dumps(spec, sort_keys=True))
        joined = {one.side: one.vector for one in geometry.copies if one.permission[0]}
        assert set(joined) == {"nw", "se"}
        ways = designing._offsets_across(geometry.copies, (10, 2), (0, 7))
        assert ways == {side: (0 + v[0] - 10, 7 + v[1] - 2) for side, v in joined.items()}
        assert ways["se"] == (1, 0)

    def test_a_board_with_edges_is_offered_what_it_always_was(self) -> None:
        """No seam, no copies, so no crossing is added to the candidates."""
        spec = designing.settled(load_spec(RULESETS / "chess.json"))
        geometry = designing._geometry(json.dumps(spec, sort_keys=True))
        assert geometry.copies == ()
        offsets = [(1, 2), (-7, 0)]
        across = [designing._offsets_across(geometry.copies, (3, 3), (4, 5))] * 2
        assert designing._candidate_moves(offsets, "square", across) == (
            designing._candidate_moves(offsets, "square")
        )


class TestASquareSeamIsCrossedTheShortWay:
    """Q281c3: a cylinder or a torus offers the step across its seam, not the long way
    round, from copies the compiler lists (`CompiledSpec.grid_copies`)."""

    def test_the_compiler_lists_a_joined_rectangles_copies(self) -> None:
        def listed(name: str) -> dict[str, tuple[tuple[int, int], tuple[bool, bool]]]:
            compiled = compile_spec(load_spec(_committed(name)))
            assert compiled.copies == ()
            return {one.side: (one.vector, one.permission) for one in compiled.grid_copies}

        assert listed("chess") == {}
        assert listed("cylinder") == {
            "left": ((-8, 0), (True, True)), "right": ((8, 0), (True, True))}
        torus = listed("torus")
        assert len(torus) == 8
        assert torus["top"] == ((0, 12), (True, True))
        assert torus["left+top"] == ((-8, 12), (True, True))
        assert listed("onewaytorus")["left"] == ((-8, 0), (True, False))
        hexagon = compile_spec(hex_glinski_flanks())
        assert hexagon.grid_copies == () and hexagon.copies

    def test_a_step_off_the_a_file_comes_in_on_the_h_file(self) -> None:
        offers = _teach(load_spec(RULESETS / "cylinder.json"), "a4", ["h4"])
        assert offers and offers[0]["exact"]
        assert offers[0]["move"] == {"kind": "leap", "vectors": [[-1, 0]]}

    def test_a_torus_corner_is_one_diagonal_step(self) -> None:
        offers = _teach(load_spec(RULESETS / "torus.json"), "a1", ["h12"])
        assert offers and offers[0]["exact"]
        assert offers[0]["move"] == {"kind": "leap", "vectors": [[-1, -1]]}

    def test_a_one_way_seam_is_crossed_only_the_way_the_first_player_may(self) -> None:
        spec = load_spec(ARCHIVE / "onewaytorus.json")
        assert spec["rules"]["wrap"]["files_by_player"]["0"] == "left"
        across = _teach(spec, "a4", ["h4"])
        assert across and across[0]["move"] == {"kind": "leap", "vectors": [[-1, 0]]}
        back = _teach(spec, "h4", ["a4"])
        assert all(offer["move"]["vectors"] != [[1, 0]] for offer in back)


# ------------------------------------------------------------------ a board change


def _reshaped(spec: dict[str, Any], *writes: tuple[str, Any]) -> dict[str, Any]:
    """What the page does: the board controls, the check, and the squares it says are off
    taken out by the page's own `dropOffBoard`. Returns the draft and the words shown."""
    draft = {**spec, "board": _board_after(spec["board"], *writes)}
    answer = designing.check(draft)
    assert not answer["valid"]
    return _run_page(
        f"const D = {json.dumps(draft)};"
        f"const said = dropOffBoard(D, {json.dumps(answer['offBoard'])});",
        "({ draft: D, said })",
    )


class TestABoardChangeKeepsAValidDraft:
    """Q281c3: what a board change leaves off the board goes, and the page says what."""

    def test_a_walled_rhombus_made_a_hexagon_loses_its_corner_walls(self) -> None:
        """Glinski's rhombus walls its corners; on a hexagon those are not cells at all."""
        spec = hex_glinski_promoting()
        done = _reshaped(spec, ("shape", "hexagon"))
        assert "obstacles" not in done["draft"]["rules"]
        assert done["said"].startswith("The board change took away the walls at a1, a2")
        assert done["said"].endswith(": off the hexagon.")
        assert validate_spec(designing.settled(done["draft"])) == []
        assert designing.check(done["draft"])["valid"]

    def test_the_server_names_only_what_is_off(self) -> None:
        """A wall the hexagon still has a cell for is kept, and a valid draft has none."""
        spec = hex_glinski_promoting()
        spec["rules"]["obstacles"]["squares"].append("f6")
        draft = {**spec, "board": _board_after(spec["board"], ("shape", "hexagon"))}
        (walls,) = designing.off_board(draft)
        assert "f6" not in walls["squares"] and len(walls["squares"]) == 15
        assert designing.off_board(load_spec(ARCHIVE / "hex-glinski-hexagon.json")) == []

    def test_fewer_cells_on_a_side_take_the_army_and_promotion_squares_with_them(self) -> None:
        spec = load_spec(ARCHIVE / "hex-glinski-hexagon.json")
        done = _reshaped(spec, ("side", "5"))
        rows = {row["piece"]: row["squares"] for row in done["draft"]["setup"]["placements"]}
        assert "j1" not in rows["P"] and "b5" in rows["P"]
        assert "k6" not in done["draft"]["rules"]["promotion"]["squares"]
        assert "the P at j1" in done["said"] and "the promotion squares a11" in done["said"]
        assert not any("not a square on this board" in problem
                       for problem in validate_spec(designing.settled(done["draft"])))

    def test_a_narrower_rectangle_drops_the_file_it_lost(self) -> None:
        done = _reshaped(load_spec(RULESETS / "chess.json"), ("width", "7"))
        assert all(not name.startswith("h")
                   for row in done["draft"]["setup"]["placements"] for name in row["squares"])
        assert done["said"].endswith(": off the board.")


# ------------------------------------------------------------------ a square game made hexagons
#
# Q640, the user's phone pass (`first-visit-study.md` §1.2a item 5): *changing to hexagonal
# didn't work; changing the cells didn't change the board; changing the board made the
# board go away completely.* Every library game and template reads the square lattice
# (a move's symmetry, the mirrored army, a rank zone), which the validator refuses on a hex
# one, so making the cells hexagons made no game; an invalid draft drew the last board a
# check had made, and with none in hand (a reload) no board at all.


def _made_hex(spec: dict[str, Any]) -> dict[str, Any]:
    """A draft with only its lattice control turned to hexagons, as the page writes it."""
    return {**spec, "board": _board_after(spec["board"], ("lattice", "hex"))}


def _vector_sets(spec: dict[str, Any]) -> dict[str, list[frozenset[tuple[int, int]]]]:
    return {
        piece["id"]: [frozenset(map(tuple, move["vectors"])) for move in piece["moves"]]
        for piece in spec["pieces"]
    }


class TestASquareGameMadeHexagonsIsAGame:
    def test_chess_made_hexagons_moves_as_glinski_does(self) -> None:
        """The hex group standing where the square's stood gives Glinski's own rook,
        bishop, knight, queen and king (`archive/hex-glinski-hexagon.json`)."""
        made = designing.onto_hex(_made_hex(load_spec(RULESETS / "chess.json")))["spec"]
        glinski = _vector_sets(load_spec(ARCHIVE / "hex-glinski-hexagon.json"))
        ours = _vector_sets(made)
        for piece in "RBNQK":
            assert set().union(*ours[piece]) == set().union(*glinski[piece]), piece
        assert all(move.get("symmetry", "none") == "none"
                   for piece in made["pieces"] for move in piece["moves"])
        assert made["setup"]["orientation"] == "rotate"

    def test_every_library_game_made_hexagons_validates_but_what_has_no_twin(self) -> None:
        """Pinned from below: what has no hex twin stays a refusal in words."""
        refused = {}
        for path in sorted(RULESETS.glob("*.json")):
            made = designing.onto_hex(_made_hex(load_spec(path)))["spec"]
            problems = validate_spec(designing.settled(made))
            if problems:
                refused[path.stem] = problems
        assert len(refused) <= 2, sorted(refused)
        for problems in refused.values():
            # SR1a: a hex lame is refused only until it states its legs.
            assert all("must state its legs" in line or "facing.axes 4" in line
                       or "schema_version" in line for line in problems), problems

    def test_a_rank_zone_becomes_the_cells_it_was(self) -> None:
        spec = load_spec(RULESETS / "chess.json")
        made = designing.onto_hex(_made_hex(spec))["spec"]
        assert made["rules"]["promotion"]["squares"] == [f"{f}8" for f in "abcdefgh"]
        assert "rank" not in made["rules"]["promotion"]

    def test_the_answer_says_what_was_rewritten(self) -> None:
        said = designing.onto_hex(_made_hex(designing.blank_spec()))["said"]
        assert said.startswith("For hexagons, the moves of the soldier")
        assert "turned half round" in said

    def test_a_square_draft_and_a_hex_one_come_back_as_they_went(self) -> None:
        square = load_spec(RULESETS / "chess.json")
        assert designing.onto_hex(square) == {"spec": square, "said": ""}
        glinski = load_spec(ARCHIVE / "hex-glinski-hexagon.json")
        assert designing.onto_hex(glinski) == {"spec": glinski, "said": ""}

    def test_the_lattice_control_asks_the_server_and_takes_its_answer(self) -> None:
        """`ontoHex`: the draft the server rewrote replaces the one sent, and its sentence
        stands under the board controls. A draft edited before the answer is left alone."""
        start = _made_hex(designing.blank_spec())
        answer = designing.onto_hex(start)
        out = _run_page(
            "drawDesigner = () => {}; checkDraft = () => {}; loadMoveMap = () => {};"
            f" const START = {json.dumps(start)}; const ANSWER = {json.dumps(answer)};"
            " let asked = null;"
            " api = (path, body) => { asked = path; return Promise.resolve(ANSWER); };"
            " async function both() {"
            "   DRAFT = JSON.parse(JSON.stringify(START)); await ontoHex();"
            "   const taken = { draft: DRAFT, said: BOARD_REMOVED, asked };"
            "   DRAFT = JSON.parse(JSON.stringify(START)); BOARD_REMOVED = '';"
            "   const late = ontoHex(); DRAFT.name = 'typed meanwhile'; await late;"
            "   return { taken, late: { name: DRAFT.name, setup: DRAFT.setup,"
            "                          said: BOARD_REMOVED } };"
            " }",
            "both()",
        )
        assert out["taken"]["asked"] == "/api/design/hex"
        assert out["taken"]["draft"] == answer["spec"]
        assert out["taken"]["said"] == answer["said"]
        assert out["late"]["name"] == "typed meanwhile"
        assert out["late"]["setup"] == start["setup"] and out["late"]["said"] == ""

    def test_the_board_controls_send_a_hex_draft_through_the_server(self) -> None:
        """The lattice and the shape menus both; a size box does not."""
        from page_text import PAGE

        handler = PAGE[PAGE.index("form.querySelectorAll('[data-bfield]')"):]
        handler = handler[:handler.index("wireMoveRows(form)")]
        assert "if (reshaped && DRAFT.board.lattice === 'hex') return ontoHex();" in handler


class TestTheBoardIsAlwaysTheDrafts:
    def test_an_invalid_hex_draft_is_drawn_on_its_own_board(self) -> None:
        """Squares made hexagons and then a hexagon, with no rewrite: still not a game,
        and the board drawn is the hexagon, its corners absent and its army standing."""
        start = designing.blank_spec()
        draft = {**start, "board": _board_after(
            start["board"], ("lattice", "hex"), ("shape", "hexagon"))}
        draft["setup"] = {**draft["setup"], "placements": [
            {"piece": "S", "squares": ["c1", "d1", "e1"]}]}
        answer = designing.check(draft)
        assert not answer["valid"]
        preview = answer["preview"]
        assert preview["standIn"] is True
        assert (preview["width"], preview["height"], preview["lattice"]) == (5, 5, "hex")
        assert preview["absent"] == sorted(absent_squares(draft["board"]))
        first = [square for square, held in enumerate(preview["board"]) if held]
        assert {_name(s, 5) for s in first} >= {"c1", "d1", "e1"}

    def test_a_resized_square_draft_is_drawn_at_its_new_size(self) -> None:
        draft = load_spec(RULESETS / "chess.json")
        draft["board"] = _board_after(draft["board"], ("width", "10"))
        draft["pieces"][0]["moves"][0]["symmetry"] = "sideways"
        answer = designing.check(draft)
        assert not answer["valid"]
        assert answer["preview"]["width"] == 10 and answer["preview"]["standIn"]

    def test_the_army_is_drawn_where_it_can_stand(self) -> None:
        """A square off the board, or no square at all, is left out of the drawing."""
        draft = designing.blank_spec()
        draft["setup"] = {"mirror": True, "placements": [
            {"piece": "S", "squares": ["b1", "g1", "z9"]}]}
        answer = designing.check(draft)
        assert not answer["valid"]
        held = [square for square, piece in enumerate(answer["preview"]["board"]) if piece]
        assert [_name(s, 6) for s in held] == ["b1", "b6"]

    def test_a_hexagon_grown_round_its_army_is_drawn_empty_at_its_new_size(self) -> None:
        """Three a side to six puts `c1`-`e1` on absent corners; the page takes them off
        and the draft has no army at all, which no spec may be. The board still draws."""
        draft = designing.onto_hex(_made_hex(designing.blank_spec()))["spec"]
        draft["board"] = _board_after(draft["board"], ("shape", "hexagon"), ("side", "6"))
        draft["setup"]["placements"] = []
        answer = designing.check(draft)
        assert not answer["valid"]
        preview = answer["preview"]
        assert (preview["width"], preview["standIn"]) == (11, True)
        assert preview["absent"] == sorted(absent_squares(draft["board"]))
        assert not any(preview["board"])

    def test_a_valid_draft_has_no_stand_in(self) -> None:
        assert "standIn" not in designing.check(designing.blank_spec())["preview"]

    def test_the_column_draws_the_stand_in_after_a_reload(self) -> None:
        """No earlier board in hand (`DRAFT_PREVIEW` null, as after a reload): the
        column draws the draft's own board and says whose it is."""
        start = designing.blank_spec()
        draft = {**start, "board": _board_after(start["board"], ("lattice", "hex"))}
        check = designing.check(draft)
        drawn = _run_page(
            _kit_setup()
            + "const side = { innerHTML: '', querySelectorAll: () => [] };"
            " document.getElementById = id => id === 'boardSide' ? side : null;"
            f" DRAFT = {json.dumps(draft)}; DRAFT_CHECK = {json.dumps(check)};"
            " DRAFT_PREVIEW = null; DRAFT_MODE = 'setup'; PICKING = null; BENCH_PLAY = null;"
            " wirePalette = () => {}; wirePieceTray = () => {}; wireBoards = () => {};"
            " wireBenchPlay = () => {}; wireTry = () => {}; drawBoardSide();",
            "side.innerHTML",
        )
        assert "The board appears as soon as" not in drawn
        assert "benchBoard" in drawn and "benchBoard stale" not in drawn
        assert "This draft's board and" in drawn


# ------------------------------------------------------------------ the probe's cost


def _torus(radius: int) -> dict[str, Any]:
    """A hexagon with all three pairs joined, a king and a slider on it."""
    width = 2 * radius + 1
    six = [[0, 1], [1, 0], [-1, 1], [0, -1], [-1, 0], [1, -1]]
    return {
        "schema_version": 16, "name": "torus probe", "description": "",
        "board": {"width": width, "height": width, "lattice": "hex", "shape": "hexagon"},
        "pieces": [
            {"id": "K", "name": "king", "value": 0, "royal": True,
             "moves": [{"kind": "leap", "vectors": six}]},
            {"id": "R", "name": "rook", "value": 5, "moves": [{"kind": "slide", "vectors": six}]},
        ],
        "setup": {"mirror": True, "orientation": "rotate", "placements": [
            {"piece": "K", "squares": [_name(radius, width)]},
            {"piece": "R", "squares": [_name(radius + 1, width)]},
        ]},
        "rules": {"win": "royal_capture", "max_plies": 100,
                  "wrap": {"n_s": "both", "ne_sw": "both", "nw_se": "both"}},
    }


class TestTheProbeOnATorusIsBounded:
    """On a torus every line through a cell is the whole board, so the draft's own
    compile is the cost of a click (about 1.7 s for Glinski's army here, against about
    20 ms for a one-piece probe). Counted rather than timed: engines built per click."""

    @pytest.fixture
    def built(self, monkeypatch) -> list[str]:
        designing._geometry.cache_clear()
        designing._probe_reach_of.cache_clear()
        names: list[str] = []
        real = designing.Engine

        def counting(spec: Any, *args: Any, **kwargs: Any) -> Any:
            names.append(spec.get("name"))
            return real(spec, *args, **kwargs)

        monkeypatch.setattr(designing, "Engine", counting)
        return names

    def test_a_click_builds_the_draft_once_and_at_most_the_limit_of_probes(self, built) -> None:
        spec = _torus(3)
        width = 7
        origin = _square("d4", width)
        designing.teach(spec, 1, origin, [_square("e4", width)])
        assert built.count("torus probe") == 1
        first = built.count("probe")
        assert 0 < first <= designing.TEACH_LIMIT

    def test_the_next_click_on_the_same_draft_compiles_no_draft(self, built) -> None:
        spec = _torus(3)
        width = 7
        origin = _square("d4", width)
        designing.teach(spec, 1, origin, [_square("e4", width)])
        before = len(built)
        designing.teach(spec, 1, origin, [_square("e4", width)])
        assert len(built) == before, "a repeated click compiled something"
        designing.teach(spec, 1, origin, [_square("e4", width), _square("f4", width)])
        added = built[before:]
        assert "torus probe" not in added
        assert len(added) <= designing.TEACH_LIMIT

    def test_a_torus_slider_probe_answers_every_other_cell(self) -> None:
        """What one probe costs is one piece's slide over the whole board: a lone rook
        on the R = 3 torus reaches all 36 other cells, 3R² + 3R, and no more."""
        spec = designing.settled(_torus(3))
        geometry = designing._geometry(json.dumps(spec, sort_keys=True))
        probe = designing._probe_spec(spec, geometry)
        reach = designing._probe_reach(
            probe, {"kind": "slide", "vectors": [[1, 0]]}, _square("d4", 7)
        )
        assert len(reach) == 36


# ------------------------------------------------------------------ room for a hexagon


class TestAHexagonGetsTheRoom:
    """Q545: a hex draft's board takes the larger share of a computer's columns."""

    def _classes(self, lattice: str) -> list[str]:
        return _run_page(
            "const classes = new Set(['designer', 'bench']);"
            " const designer = { classList: { toggle: (c, on) => on ? classes.add(c)"
            "                                                   : classes.delete(c) } };"
            " const side = { innerHTML: '', querySelectorAll: () => [],"
            "                closest: () => designer };"
            " document.getElementById = id => id === 'boardSide' ? side : null;"
            " DRAFT = {}; DRAFT_CHECK = null;"
            f" DRAFT_PREVIEW = {{ width: 3, height: 3, lattice: {json.dumps(lattice)},"
            "   board: [0, 0, 0, 0, 0, 0, 0, 0, 0], absent: [], obstacles: [] };"
            " DRAFT_MODE = 'play'; PICKING = null; BENCH_PLAY = null;"
            " benchPlayHtml = () => ''; wirePalette = () => {}; wirePieceTray = () => {};"
            " wireBoards = () => {}; wireBenchPlay = () => {}; wireTry = () => {};"
            " drawBoardSide();",
            "[...classes]",
        )

    def test_a_hex_board_marks_the_columns(self) -> None:
        assert "hexBench" in self._classes("hex")
        assert "hexBench" not in self._classes("square")

    def test_the_wide_columns_are_a_computers_only(self) -> None:
        from chess_variant_sandbox.web.page.style import STYLE

        at = STYLE.index(".designer.bench.hexBench { grid-template-columns")
        assert STYLE[:at].rstrip().endswith("@media (min-width: 1200px) {")
        assert "minmax(440px, 1fr)" in STYLE[at:].split("}")[0]
