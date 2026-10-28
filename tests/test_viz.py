"""The replay page: piece icons where the library has one, spec letters where not."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import node_session
import pytest
from retired_specs import REBUILT, hex_glinski_promoting

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.viz import render_position
from chess_variant_sandbox.viz.heading import directional_types
from chess_variant_sandbox.viz.pieces import ASSET_ROOT, icon_for, icons_for, side_names

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def _payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    start = text.index("const DATA = ") + len("const DATA = ")
    end = text.index(";\n", start)
    return json.loads(text[start:end])


def _drawn_angle(data: dict, square: int) -> int:
    """The angle the page draws the piece on `square` at, by the page's own arithmetic.

    A declared oracle for the drawing rather than for the rules. It decodes a board byte
    the way `drawPiece` does -- owner off `sideStride`, facing off `nTypes` -- and then
    takes the heading only where `directional` says the type has a facing. Everything
    else is upright, except a single-colour icon's ownership 180.
    """
    value = data["boards"][0][square]
    player = (value - 1) // data["sideStride"]
    kind = (value - 1) % data["nTypes"]
    facing = ((value - 1) % data["sideStride"]) // data["nTypes"]
    if data["directional"][kind]:
        return data["facingAngles"][player][facing]
    icon = (data["pieceIcons"] or [])[kind][player] if data["pieceIcons"] else None
    return 180 if icon and icon["rotate"] else 0


def test_icons_are_inlined_not_linked(chess_spec: dict, tmp_path: Path) -> None:
    """The page has to stay one file that opens off a disk with no assets beside it."""
    engine = Engine(chess_spec)
    page = render_position(chess_spec, engine.initial_state(), tmp_path / "chess.html")
    data = _payload(page)

    assert len(data["pieceIcons"]) == len(data["pieceNames"])
    for pair in data["pieceIcons"]:
        for icon in pair:
            assert icon is not None
            assert icon["href"].startswith("data:image/svg+xml;base64,")
    assert data["credits"], "icons carry attribution requirements"


def test_letters_when_the_library_has_no_icon(chess_spec: dict, tmp_path: Path) -> None:
    """An invented piece is the ordinary case, and it falls back rather than failing."""
    spec = copy.deepcopy(chess_spec)
    spec["pieces"][1]["name"] = "schnauzer"
    engine = Engine(spec)
    page = render_position(spec, engine.initial_state(), tmp_path / "invented.html")
    data = _payload(page)

    index = data["pieceNames"].index("schnauzer")
    assert data["pieceIcons"][index] == [None, None]
    assert data["pieceIcons"][0] != [None, None]


def test_a_wild_piece_is_drawn_as_neither_players(chess_spec: dict, tmp_path: Path) -> None:
    """Schema 9's third owner, on the replay page.

    The board bytes are the only thing that crosses, so the page has to be able to read
    an owner off one: a wild piece's value is `2 * sideStride + type + 1`, and the
    JavaScript's `floor((value - 1) / sideStride)` gets 2 for it. What it must not do is
    reach into an icon pair with that index -- there are two drawings, one per army --
    so a wild piece is the lettered disc, in the third tint the page defines for it.
    """
    spec = copy.deepcopy(chess_spec)
    spec["schema_version"] = 9
    spec["pieces"].append(
        {"id": "W", "name": "boar", "value": 3,
         "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]}
    )
    spec["rules"]["npcs"] = [{"piece": "W", "route": ["c3", "c4"], "mirror": True}]
    engine = Engine(spec)
    state = engine.initial_state()
    page = render_position(spec, state, tmp_path / "boar.html")
    data = _payload(page)

    stride = data["sideStride"]
    wild = [value for value in state.board if value and (value - 1) // stride == 2]
    assert len(wild) == 2, "two boars on the board the page was handed"

    text = page.read_text(encoding="utf-8")
    assert "--p2:" in text and "--p2-ink:" in text
    assert "pair[piece.player > 1 ? 0 : piece.player]" in text  # PI4: drawn, not lettered


def test_icons_can_be_turned_off(chess_spec: dict, tmp_path: Path) -> None:
    engine = Engine(chess_spec)
    page = render_position(
        chess_spec, engine.initial_state(), tmp_path / "letters.html", icons=False
    )
    data = _payload(page)
    assert data["pieceIcons"] == []
    assert data["credits"] == []


@pytest.mark.parametrize("path", sorted(RULESETS.glob("*.json")), ids=lambda p: p.stem)
def test_committed_rulesets_draw_every_piece(path: Path) -> None:
    """A published game the library covers should never fall back to its letters.

    Falling back is correct for an *invented* piece and wrong for a rook. Seven of the
    schema-2 rulesets encode their pieces by id alone -- ``name`` sits inside ``pieces``
    and so inside the mechanics hash, which is why they cannot simply be named -- and
    every one of them played as bare letters until an ``appearance.icons`` block said
    what they look like. This is the check that noticed nothing.
    """
    spec = load_spec(path)
    if icon_for("rook", 0) is None:  # the library is optional
        pytest.skip("icon library not fetched")

    pieces = spec["pieces"]
    icons = icons_for(
        [piece.get("name", piece["id"]) for piece in pieces],
        [piece["id"] for piece in pieces],
        (spec.get("appearance") or {}).get("icons"),
    )
    bare = [piece["id"] for piece, pair in zip(pieces, icons, strict=True) if not any(pair)]
    assert not bare, f"{path.stem} draws {bare} as letters; name them in appearance.icons"


def test_single_colour_sets_rotate_for_the_second_player() -> None:
    """Shogi and xiangqi tell the players apart by orientation, not by colour."""
    white, black = icons_for(["gold"])[0]
    if white is None:  # the library is optional; a checkout may not have fetched it
        return
    assert white["href"] == black["href"]
    assert (white["rotate"], black["rotate"]) == (False, True)


def test_two_colour_sets_do_not_rotate() -> None:
    white = icon_for("rook", 0)
    black = icon_for("rook", 1)
    if white is None:
        return
    assert white["href"] != black["href"]
    assert not white["rotate"] and not black["rotate"]


def test_the_sides_are_named_after_what_is_drawn() -> None:
    """What to call the two players, which is a fact about the icons and not about chess.

    A single-colour set distinguishes them by turning a piece around, so both armies are
    the same colour and calling one of them black would send a player looking for pieces
    the opponent also has. All-or-nothing: one rotated piece is enough, because the names
    label a side and dobutsu rotates two of its five.
    """
    assert side_names(icons_for(["rook", "king"])) == ("white", "black")
    # No library at all: every piece falls back to a letter, drawn light against dark.
    assert side_names([[None, None], [None, None]]) == ("white", "black")

    gold = icons_for(["gold"])
    if gold[0][1] is None:
        return  # the library is optional; a checkout may not have fetched it
    assert side_names(gold) == ("first", "second")
    assert side_names(icons_for(["rook", "gold"])) == ("first", "second")


def test_lookup_is_by_name_and_case_insensitive() -> None:
    assert icon_for("King", 0) == icon_for("king", 0)
    assert icon_for("chariot", 0) == icon_for("rook", 0)  # an alias, not a chess name


@pytest.mark.parametrize("path", sorted(RULESETS.glob("*.json")), ids=lambda p: p.stem)
def test_no_two_pieces_in_one_ruleset_draw_alike(path: Path) -> None:
    """A board whose pieces are indistinguishable is a board nobody can play.

    The test above catches a piece with *no* icon; this catches the other half, which
    is worse because it looks fine. Courier's schleich and its pawn both fell back to
    ``pawn`` -- so did bastion's archer and its footman -- and shogi's promoted bishop
    drew as a knight, which is not a fallback at all but a wrong answer stated
    confidently. All three rendered a full board every time.
    """
    spec = load_spec(path)
    if icon_for("rook", 0) is None:  # the library is optional
        pytest.skip("icon library not fetched")

    pieces = spec["pieces"]
    icons = icons_for(
        [piece.get("name", piece["id"]) for piece in pieces],
        [piece["id"] for piece in pieces],
        (spec.get("appearance") or {}).get("icons"),
    )
    seen: dict[str, str] = {}
    for piece, pair in zip(pieces, icons, strict=True):
        stem = pair[0]["stem"] if pair[0] else None
        if stem is None:
            continue
        assert stem not in seen, (
            f"{path.stem}: {piece['id']} and {seen[stem]} both draw as {stem}"
        )
        seen[stem] = piece["id"]


def test_the_rps_pieces_are_drawn_rather_than_stood_in_for() -> None:
    """Three icons this repository drew, because Commons has no picture of a rock.

    They wore a dabbaba, a crescent and a giraffe: three fairy glyphs picked for being
    distinguishable from each other, which is a placeholder and not a drawing. The
    corpus they belong to is the one place in the library where what a piece *is*
    carries the whole game, so a stand-in costs more here than anywhere else.
    """
    for name in ("rock", "paper", "scissors"):
        for player in (0, 1):
            icon = icon_for(name, player)
            assert icon is not None, f"no icon for {name}"
            assert icon["stem"].startswith(name)
        assert icon_for(name, 0)["href"] != icon_for(name, 1)["href"]


# The pictures the first player wears: the light half of every pair, and every
# single-colour file (shogi), which serves both seats.
_SVG = ASSET_ROOT / "svg"
_LIGHT_SIDE = sorted(
    path
    for path in _SVG.glob("*.svg")
    if path.stem.endswith("-white") or not path.stem.endswith("-black")
)
_SHAPES = {"path", "circle", "ellipse", "rect", "polygon"}


def _light_fills(svg: Path) -> list[str]:
    """The opaque light fills a picture paints its shapes with, as a browser resolves them.

    A declared oracle for the drawing. Fill comes from the element's own ``style``, then
    its ``fill`` attribute, then a ``<style>`` class rule, then its parent; SVG's own
    default is black. A fill under half opacity counts as see-through.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(svg.read_bytes())
    sheet = " ".join(node.text or "" for node in root.iter() if node.tag.endswith("style"))
    classes = {
        name: body for name, body in re.findall(r"\.([\w-]+)\s*\{([^}]*)\}", sheet)
    }

    def declared(node: ET.Element, prop: str) -> str | None:
        style = node.get("style", "")
        found = re.search(rf"(?:^|;)\s*{prop}\s*:\s*([^;]+)", style)
        if found:
            return found.group(1).strip()
        if node.get(prop) is not None:
            return str(node.get(prop)).strip()
        for name in node.get("class", "").split():
            found = re.search(rf"(?:^|;|\s){prop}\s*:\s*([^;]+)", classes.get(name, ""))
            if found:
                return found.group(1).strip()
        return None

    def light(colour: str) -> bool:
        colour = colour.lower()
        named = {"white": "#ffffff"}
        colour = named.get(colour, colour)
        if not re.fullmatch(r"#[0-9a-f]{3}|#[0-9a-f]{6}", colour):
            return False
        digits = colour[1:]
        if len(digits) == 3:
            digits = "".join(c * 2 for c in digits)
        r, g, b = (int(digits[i : i + 2], 16) / 255 for i in (0, 2, 4))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b >= 0.6

    out: list[str] = []

    def walk(node: ET.Element, fill: str, opacity: float) -> None:
        if node.tag.split("}")[-1] in {"defs", "style", "title", "metadata"}:
            return
        fill = declared(node, "fill") or fill
        for prop in ("opacity", "fill-opacity"):
            value = declared(node, prop)
            if value is not None:
                opacity *= float(value)
        if node.tag.split("}")[-1] in _SHAPES and light(fill) and opacity >= 0.5:
            out.append(fill)
        for child in node:
            walk(child, fill, opacity)

    walk(root, "#000000", 1.0)
    return out


@pytest.mark.skipif(not _LIGHT_SIDE, reason="icon library not fetched")
@pytest.mark.parametrize("svg", _LIGHT_SIDE, ids=lambda p: p.stem)
def test_the_first_players_pictures_are_filled_not_see_through(svg: Path) -> None:
    """A white piece is a white body inside its outline, never the square showing through.

    The Quivira glyphs (alfil, camel, wazir) arrive as a font's white pieces: black
    outline, nothing inside. A white elephant on the watering hole board was the square
    seen through a black line, beside lettered discs filled white (the user, 2026-10-19:
    "the elephant svg isn't white, it's transparant"). Every picture the first player
    wears paints some shape in an opaque light colour; a hollow one paints none.
    """
    assert _light_fills(svg), f"{svg.name} fills nothing light: its body is see-through"


def test_the_elephant_the_first_player_wears_is_filled() -> None:
    """The watering hole's elephant, by name, as the board looks it up."""
    icon = icon_for("elephant", 0)
    if icon is None:
        pytest.skip("icon library not fetched")
    assert _light_fills(_SVG / f"{icon['stem']}.svg")


class TestWhichPiecesPoint:
    """A drawing turns a piece if and only if the compiler gave that piece a facing.

    Both renderers used to turn every one of the second player's icons, because they
    read `facing_headings()[player]` for every piece and that table is `((0,), (180,))`
    on a ruleset where nothing turns. It answers *which way does this facing point*, so
    asking it about a piece with no facing gets the army's answer rather than the
    piece's, and a bishop stood on its head to say something about the seat.

    A facing is the whole test. An icon arrives pointing up, zero degrees is up relative
    to the board, and a piece with no facing keeps that for both armies -- so a chess
    pawn is upright for black even though black advances the other way.
    """

    @staticmethod
    def _turning(name: str) -> dict[str, bool]:
        path = RULESETS / name
        if not path.exists():  # moved to the Library (Q512)
            path = path.parent.parent / "archive" / name
        compiled = Engine(load_spec(path)).spec
        return dict(zip(compiled.piece_ids, directional_types(compiled), strict=True))

    def test_nothing_turns_on_a_chess_board(self) -> None:
        """The calibration baseline has no facing anywhere, so no icon moves. A pawn
        advances one way for white and the other for black, and that is a fact about the
        seat rather than about the piece: the drawing stays as it came."""
        assert not any(self._turning("chess.json").values())

    def test_nothing_turns_on_racing_kings_either(self) -> None:
        """No pawn and no facing, so the board is upright end to end."""
        assert not any(self._turning("racingkings.json").values())

    def test_only_the_piece_with_a_facing_turns(self) -> None:
        """A facing is a direction by construction -- the validator refuses one whose
        turn changes no move -- and it moves during the game rather than following the
        seat. Here the pawn has one and the other five pieces do not."""
        compiled = Engine(load_spec(ARCHIVE / "chess-oriented-pawns.json")).spec
        assert self._turning("chess-oriented-pawns.json") == {
            "P": True, "N": False, "B": False, "R": False, "Q": False, "K": False
        }
        # The heading is `facing_headings()`, never `facing * 90`: under `mirror` the
        # second player's vectors are reflected, so the same facing index turns the
        # other way round on screen.
        assert compiled.facing_headings()[0] != compiled.facing_headings()[1]

    def test_a_pawn_turned_east_is_drawn_at_ninety_degrees(self, tmp_path: Path) -> None:
        """The facing the game changes, on the page. `Pa2^right` spends a whole move
        turning the a2 pawn to face east, and the replay page draws it lying on its
        side -- 90 degrees clockwise from the up its icon arrived at.

        Black's pawns face their own forward, which is down the board, so they are drawn
        at 180. That is the facing being told truthfully rather than the seat's 180 that
        used to land on every black piece: black's knights and king stay upright.
        """
        spec = load_spec(ARCHIVE / "chess-oriented-pawns.json")
        engine = Engine(spec)
        state = engine.initial_state()
        moves = engine.legal_moves(state)
        turn = moves[engine.move_texts(state).index("Pa2^right")]
        page = render_position(spec, engine.apply(state, turn), tmp_path / "east.html")
        data = _payload(page)
        assert _drawn_angle(data, square=8) == 90, "the a2 pawn now faces east"
        assert _drawn_angle(data, square=9) == 0, "b2, still facing up the board"
        assert _drawn_angle(data, square=48) == 180, "a7, black's pawn facing down"
        for square in (1, 4, 57, 60):
            assert _drawn_angle(data, square) == 0, "a knight and a king have no facing"

    def test_shogi_turns_by_ownership_and_not_by_heading(self) -> None:
        """The single-colour set. No shogi piece has a facing, so `directional` is false
        end to end and every rotation on that board is `icon.rotate` -- one tile drawing
        for both armies, turned 180 for the second to say whose it is."""
        spec = load_spec(RULESETS / "shogi.json")
        compiled = Engine(spec).spec
        assert not any(self._turning("shogi.json").values())
        art = icons_for(
            compiled.piece_names,
            compiled.piece_ids,
            (spec.get("appearance") or {}).get("icons"),
        )
        assert all(pair[1]["rotate"] for pair in art)
        assert not any(pair[0]["rotate"] for pair in art)

    def test_the_replay_page_is_told_which_pieces_point(
        self, chess_spec: dict, tmp_path: Path
    ) -> None:
        """The renderer is handed the answer rather than working one out. Board bytes
        and this list are all it has; nothing on the page reads a move."""
        engine = Engine(chess_spec)
        page = render_position(chess_spec, engine.initial_state(), tmp_path / "point.html")
        data = _payload(page)
        assert len(data["directional"]) == len(data["pieceIds"])
        assert data["directional"] == [False] * len(data["pieceIds"])


# --------------------------------------------------------- a board without edges


def _wrapped(expression: str) -> object:
    """Evaluate one expression against the shared wrapped-board module, in node.

    The module is arithmetic and string building with no DOM anywhere in it, which is
    the point of testing it rather than the drawing: what a shift does to a square, and
    what path a move through the seam takes, are answers a test can have without a
    browser. Skipped where node is absent rather than approximated.
    """
    from chess_variant_sandbox.viz.wrapped import WRAPPED

    done = node_session.run(WRAPPED + "\nconsole.log(JSON.stringify(" + expression + "));")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


_LABELS = r"""
(() => {
  const cell = 100 / 8;
  const view = boardView(8, 8, { files: FILES, ranks: RANKS }, SHIFT, FLIP);
  const found = [];
  for (let sq = 0; sq < 64; sq++) {
    const place = squareCell(sq, view);
    const xy = cellXY(place[0], place[1], cell, view);
    const html = coordsHtml(sq, xy[0], xy[1], cell, view);
    const each = /x="([-\d.]+)" y="([-\d.]+)"[\s\S]*?>([^<]*)<\/text>/g;
    let hit;
    while ((hit = each.exec(html)) !== null) {
      found.push({ x: +hit[1], y: +hit[2], text: hit[3] });
    }
  }
  const pick = (kind, key) => found.filter(f => kind.test(f.text))
    .sort((a, b) => a[key] - b[key]).map(f => f.text).join('');
  return { files: pick(/[a-z]/, 'x'), ranks: pick(/[0-9]/, 'y') };
})()
"""


def _labels(files: bool, ranks: bool, shift: str, flip: str = "false") -> object:
    """The labels the board draws, in the order they appear across it and down it."""
    return _wrapped(
        _LABELS.replace("FILES", str(files).lower())
        .replace("RANKS", str(ranks).lower())
        .replace("SHIFT", shift)
        .replace("FLIP", flip)
    )


class TestTheViewShiftsAcrossTheSeam:
    """A wrapping board has no boundary of its own, so the one it draws is a choice.

    The stored board runs file ``a`` to file ``h``, but on a wrapping axis that is one
    cut through a surface with no seam in it. Everything here follows: the view carries
    an offset, the outline follows it, the labels follow the outline, and offset zero is
    only the frame the squares happen to be named in.
    """

    def test_a_shift_is_a_modulo_and_nothing_else(self) -> None:
        """Every square moves by the same offset, and none is lost or doubled."""
        seen = _wrapped("""
          (() => {
            const view = boardView(8, 8, { files: true, ranks: true },
                                   { files: 3, ranks: 2 }, false);
            const cells = [];
            for (let sq = 0; sq < 64; sq++) cells.push(squareCell(sq, view).join(','));
            return { a1: squareCell(0, view), d3: squareCell(19, view),
                     distinct: new Set(cells).size, corner: cellSquare(0, 0, view) };
          })()
        """)
        assert seen["a1"] == [5, 6], "a1 is five files and six ranks into the repeat"
        assert seen["d3"] == [0, 0], "the square the shift names sits in the corner"
        assert seen["corner"] == 19, "and reading the corner back gives d3 again"
        assert seen["distinct"] == 64, "a shift is a permutation of the board"

    def test_a_shift_of_a_whole_board_is_no_shift_at_all(self) -> None:
        assert _wrapped("""
          (() => {
            const axes = { files: true, ranks: true };
            const none = boardView(8, 8, axes, { files: 0, ranks: 0 }, false);
            const lap = boardView(8, 8, axes, { files: 8, ranks: -16 }, false);
            return [none.files.shift === lap.files.shift,
                    none.ranks.shift === lap.ranks.shift];
          })()
        """) == [True, True]

    def test_an_axis_with_edges_ignores_its_offset(self) -> None:
        """A cylinder's ranks are ordinary ranks, and nothing shifts them."""
        assert _wrapped("""
          (() => {
            const view = boardView(8, 8, { files: true, ranks: false },
                                   { files: 2, ranks: 5 }, false);
            return [view.files.shift, view.ranks.shift, view.ranks.rings,
                    squareCell(0, view)];
          })()
        """) == [2, 0, 0, [6, 0]]

    def test_a_board_with_edges_sets_its_names_outside(self) -> None:
        """V4 (plan-visual.md §3.3): a file's letter under the bottom edge and a rank's
        number left of the left edge, each at the middle of its square's edge and
        pointing straight out, the right way round on a turned board too."""
        seen = _wrapped("""(() => {
          const read = flip => {
            const view = boardView(8, 8, {}, null, flip);
            return { outside: labelsOutside(view), labels: edgeLabels(12.5, view) };
          };
          return [read(false), read(true)];
        })()""")
        for flipped, found in zip((False, True), seen, strict=True):
            assert found["outside"] is True
            files = [label for label in found["labels"] if label["text"].isalpha()]
            ranks = [label for label in found["labels"] if label["text"].isdigit()]
            across = "".join(label["text"] for label in sorted(files, key=lambda f: f["x"]))
            down = "".join(label["text"] for label in sorted(ranks, key=lambda r: r["y"]))
            assert across == ("hgfedcba" if flipped else "abcdefgh")
            assert down == ("12345678" if flipped else "87654321")
            assert all((f["y"], f["dx"], f["dy"]) == (100, 0, 1) for f in files)
            assert all((r["x"], r["dx"], r["dy"]) == (0, -1, 0) for r in ranks)
            assert sorted(f["x"] for f in files) == [6.25 + 12.5 * i for i in range(8)]

    def test_a_wrapping_board_keeps_its_names_on_its_squares(self) -> None:
        """Its margin is the ghost copies, so the names stay on the naming frame."""
        seen = _wrapped("""[
          labelsOutside(boardView(8, 8, { files: true, ranks: false }, null, false)),
          labelsOutside(boardView(8, 8, { files: false, ranks: true }, null, false))]""")
        assert seen == [False, False]

    def test_the_labels_move_with_the_shift(self) -> None:
        """The label is the square's own name, so a shifted board reads round.

        This is what says the board has been shifted rather than renamed. The bottom
        row still names the squares the engine names in the move list, and it starts
        wherever the reader has put the outline.
        """
        assert _labels(True, False, "{ files: 0, ranks: 0 }") == {
            "files": "abcdefgh", "ranks": "87654321",
        }
        assert _labels(True, False, "{ files: 3, ranks: 0 }") == {
            "files": "defghabc", "ranks": "87654321",
        }
        assert _labels(True, True, "{ files: 0, ranks: 2 }") == {
            "files": "abcdefgh", "ranks": "21876543",
        }

    def test_turning_the_board_round_reverses_both_and_renames_neither(self) -> None:
        assert _labels(True, False, "{ files: 3, ranks: 0 }", flip="true") == {
            "files": "cbahgfed", "ranks": "12345678",
        }

    def test_the_outline_marks_the_axes_that_repeat_and_no_others(self) -> None:
        """A cylinder's top and bottom are real edges; boxing them in would say seam."""
        assert _wrapped("""
          (() => {
            const count = axes => (outlineHtml(12.5, boardView(8, 8, axes, null, false))
              .match(/<line/g) || []).length;
            return [count({}), count({ files: true }), count({ ranks: true }),
                    count({ files: true, ranks: true })];
          })()
        """) == [0, 2, 2, 4]

    def test_the_ghost_extension_is_a_few_rings_that_fade_outwards(self) -> None:
        """Two or three rows and columns, each paler than the one inside it, and the
        outermost running off the frame rather than stopping at it."""
        seen = _wrapped("""
          (() => {
            const view = boardView(8, 8, { files: true, ranks: false }, null, false);
            const cells = ghostCells(view);
            const depths = {};
            for (const cell of cells) depths[cell.depth] = (depths[cell.depth] || 0) + 1;
            return { rings: view.files.rings, margin: view.files.margin,
                     ranks: view.ranks.rings, depths: depths,
                     fades: [1, 2, 3].map(ghostFade),
                     beside: cells.filter(c => c.depth === 1 && c.u === 8)
                                  .map(c => c.square).sort((a, b) => a - b) };
          })()
        """)
        assert seen["rings"] == 3 and seen["ranks"] == 0
        assert seen["margin"] < seen["rings"], "the palest ring is cut by the frame"
        assert seen["depths"] == {"1": 16, "2": 16, "3": 16}
        assert seen["fades"] == sorted(seen["fades"], reverse=True)
        assert seen["fades"][0] < 0.5, "a ghost is paler than the board at once"
        assert seen["beside"] == [0, 8, 16, 24, 32, 40, 48, 56], "the a-file, again"

    def test_a_board_with_edges_is_drawn_exactly_as_it_always_was(self) -> None:
        """`flatView` is the whole of what a still preview of a spec needs."""
        assert _wrapped("""
          (() => {
            const view = flatView(8, 8, false);
            return [viewBox(view, 12.5), squareXY(0, 12.5, view),
                    squareXY(63, 12.5, view), ghostCells(view).length,
                    outlineHtml(12.5, view)];
          })()
        """) == ["0 0 100 100", [0, 87.5], [87.5, 0], 0, ""]


class TestAHexBoardIsDrawnAsHexagons:
    """Schema 14's one geometry seam: `cellXY` shears the ranks and a cell is a hexagon.

    Every renderer places a square through `cellXY` and draws a cell-sized mark through
    `cellShape`, so these are asked of the shared module and hold for all three boards.
    """

    _GEOMETRY = r"""
    (() => {
      const cell = 10;
      const view = flatView(11, 11, false, 'hex');
      const turned = flatView(11, 11, true, 'hex');
      const centre = (u, v, vw) => {
        const c = cellXY(u, v, cell, vw);
        return [c[0] + cell / 2, c[1] + cell / 2];
      };
      const here = centre(5, 5, view);
      const gaps = steps => steps.map(([du, dv]) => {
        const there = centre(5 + du, 5 + dv, view);
        return +Math.hypot(here[0] - there[0], here[1] - there[1]).toFixed(6);
      });
      const neighbours = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]];
      return {
        neighbours: gaps(neighbours),
        diagonals: gaps([[1, 1], [-1, -1], [2, -1], [-2, 1], [1, -2], [-1, 2]]),
        shades: neighbours.map(([du, dv]) =>
          cellShade(5 + du, 5 + dv, view) !== cellShade(5, 5, view)),
        halfTurn: [cellXY(0, 0, cell, turned), cellXY(10, 10, cell, view)],
        box: viewBox(view, cell).split(' ').map(Number),
        hexCell: cellShape(0, 0, cell, view, 'fill="x"').slice(0, 10),
        squareCell: cellShape(0, 0, cell, flatView(8, 8, false), 'fill="x"'),
        ring: squareBadge(0, 0, cell, [{ kind: 'zone', sides: [0] }], view).slice(0, 10),
        squareShade: cellShade(0, 0, flatView(8, 8, false)),
      };
    })()
    """

    def test_the_six_neighbours_touch_and_the_six_diagonals_are_one_further(self) -> None:
        seen = _wrapped(self._GEOMETRY)
        assert seen["neighbours"] == [10.0] * 6
        assert seen["diagonals"] == [pytest.approx(10 * 3 ** 0.5)] * 6

    def test_no_two_touching_cells_share_a_shade(self) -> None:
        assert _wrapped(self._GEOMETRY)["shades"] == [True] * 6

    def test_a_turned_board_is_the_half_turn(self) -> None:
        first, second = _wrapped(self._GEOMETRY)["halfTurn"]
        assert first == pytest.approx(second)

    def test_the_frame_is_wider_by_the_shear_and_taller_by_the_points(self) -> None:
        x, y, width, height = _wrapped(self._GEOMETRY)["box"]
        pad = 10 * (1 / 3 ** 0.5 - 0.5)
        assert (x, width) == (0, 160)
        assert y == pytest.approx(-pad)
        assert height == pytest.approx(10 * (10 * 3 ** 0.5 / 2 + 1) + 2 * pad)

    def test_a_hex_cell_and_its_zone_ring_are_paths_and_a_square_cell_is_not(self) -> None:
        seen = _wrapped(self._GEOMETRY)
        assert seen["hexCell"] == '<path d="M'
        assert seen["ring"] == '<path d="M'
        assert seen["squareCell"] == '<rect x="0" y="0" width="10" height="10" fill="x"/>'
        assert seen["squareShade"] == "var(--dark)"

    def test_the_replay_is_sent_the_lattice(self, tmp_path: Path) -> None:
        hexed = hex_glinski_promoting()
        page = render_position(hexed, Engine(hexed).initial_state(), tmp_path / "hex.html")
        assert _payload(page)["lattice"] == "hex"
        chess = load_spec(RULESETS / "chess.json")
        page = render_position(chess, Engine(chess).initial_state(), tmp_path / "chess.html")
        assert _payload(page)["lattice"] == "square"


class TestThePathAMoveThroughTheSeamTakes:
    """Two squares do not say which way round a piece went, and the board has to.

    The core supplies the pair and, per player, which edges that player may leave by.
    The step drawn is the shortest one consistent with both, which is presentation
    rather than legality -- a wrong guess makes a piece take a silly path and cannot
    make an illegal move look legal.
    """

    def test_a_board_with_edges_takes_the_direct_step(self) -> None:
        assert _wrapped(
            "wrapStep(0, 7, 8, 8, { files: [false, false], ranks: [false, false] })"
        ) == {"files": 7, "ranks": 0, "wrapped": False}

    def test_the_seam_wins_only_where_it_is_shorter(self) -> None:
        seen = _wrapped("""
          (() => {
            const both = { files: [true, true], ranks: [true, true] };
            return { short: wrapStep(0, 7, 8, 8, both),
                     tie: wrapStep(0, 4, 8, 8, both),
                     still: wrapStep(9, 9, 8, 8, both) };
          })()
        """)
        assert seen["short"] == {"files": -1, "ranks": 0, "wrapped": True}
        assert seen["tie"] == {"files": 4, "ranks": 0, "wrapped": False}, (
            "four either way is not shorter either way, so the board says nothing"
        )
        assert seen["still"] == {"files": 0, "ranks": 0, "wrapped": False}

    def test_a_one_way_seam_is_the_movers_to_cross(self) -> None:
        """The same square pair, drawn two ways, because two players have two rules.

        On a one-way cylinder White leaves by the a-file and Black by the h-file. A
        rook going a1 to h1 crossed the seam for the player allowed to, and slid the
        long way round the board for the player who was not.
        """
        seen = _wrapped("""
          (() => {
            const rule = { files: [[true, false], [false, true]], ranks: [[], []] };
            return { white: wrapStep(0, 7, 8, 8, wrapAllow(rule, 0)),
                     black: wrapStep(0, 7, 8, 8, wrapAllow(rule, 1)),
                     board: wrapStep(0, 7, 8, 8, wrapAllow(rule, null)) };
          })()
        """)
        assert seen["white"] == {"files": -1, "ranks": 0, "wrapped": True}
        assert seen["black"] == {"files": 7, "ranks": 0, "wrapped": False}
        assert seen["board"] == {"files": -1, "ranks": 0, "wrapped": True}, (
            "the union is the board's shape, which is what a wild piece crosses"
        )

    def test_the_two_axes_are_decided_apart(self) -> None:
        """A knight leaps out of one corner of a torus and in at the next."""
        assert _wrapped("""
          wrapStep(63, 8, 8, 8, { files: [true, true], ranks: [true, true] })
        """) == {"files": 1, "ranks": 2, "wrapped": True}

    def test_what_the_piece_passed_over_is_read_through_the_seam(self) -> None:
        """A rook sliding round the cylinder is still a slide, and still blocked."""
        assert _wrapped("""
          (() => {
            const clear = new Array(64).fill(0);
            const blocked = new Array(64).fill(0);
            blocked[0] = 1;                       // a1, between g1 and b1 through the seam
            const step = { files: 3, ranks: 0 };  // g1 -> b1, the short way round
            return [pathHops(clear, 6, step, 8, 8), pathHops(blocked, 6, step, 8, 8),
                    pathHops(clear, 6, { files: 1, ranks: 2 }, 8, 8)];
          })()
        """) == [False, True, True], "clear slide, blocked slide, leap"

    def test_the_step_becomes_a_translation_the_flip_agrees_with(self) -> None:
        assert _wrapped("""
          (() => {
            const step = { files: 1, ranks: 2 };
            return [stepXY(step, 12.5, flatView(8, 8, false)),
                    stepXY(step, 12.5, flatView(8, 8, true))];
          })()
        """) == [[12.5, -25], [-12.5, 25]]

    def test_the_crossing_hands_over_from_one_copy_to_the_other(self) -> None:
        """Two copies, one translation apart, and never both at full strength: a king
        in two places at once is the one thing a board must not say."""
        seen = _wrapped("[crossOpacity(true), crossOpacity(false)]")
        assert [frame["opacity"] for frame in seen[0]] == [1, 1, 0, 0]
        assert [frame["opacity"] for frame in seen[1]] == [0, 0, 1, 1]
        assert seen[0][1]["offset"] < seen[0][2]["offset"], "the handover has width"


class TestTheReplayDrawsTheSameBoard:
    """The offline replay and the play board are two programs drawing one game."""

    def test_both_carry_the_one_module(self, chess_spec: dict, tmp_path: Path) -> None:
        from page_text import PAGE

        from chess_variant_sandbox.viz.wrapped import WRAPPED

        engine = Engine(chess_spec)
        page = render_position(chess_spec, engine.initial_state(), tmp_path / "one.html")
        assert page.read_text(encoding="utf-8").count(WRAPPED) == 1
        assert PAGE.count(WRAPPED) == 1

    def test_the_replay_draws_the_seam_and_offers_the_same_toggle(
        self, chess_spec: dict, tmp_path: Path
    ) -> None:
        """A replay is a file somebody is sent, so it has to answer the same two
        questions the play board does: where the seam is, and where a piece promotes."""
        page = render_position(
            chess_spec, Engine(chess_spec).initial_state(), tmp_path / "both.html"
        ).read_text(encoding="utf-8")
        assert "outlineHtml(CELL, FRAME, DATA.wrap)" in page
        assert "barrierHtml(CELL, FRAME, DATA.wrap)" in page
        assert 'id="marks"' in page and "function drawMarks(" in page
        assert "zoneToggleHtml(ZONES)" in page
        assert "squareMarks(DATA, { zone: !ZONES })" in page

    def test_the_seam_crosses_as_the_compiled_rule_and_not_as_edge_names(
        self, chess_spec: dict, tmp_path: Path
    ) -> None:
        """``rules.wrap`` spells its permissions as edge names and ``engine/compile.py``
        is the one place that means anything by them. What reaches a renderer is the
        resolved pair per player, so neither page parses the DSL."""
        engine = Engine(chess_spec)
        plain = _payload(
            render_position(chess_spec, engine.initial_state(), tmp_path / "plain.html")
        )
        assert plain["wrap"] is None, "a board with edges says so, and skips all of it"

        spec = load_spec(ARCHIVE / "onewaycylinder.json")
        seen = _payload(
            render_position(
                spec, Engine(spec).initial_state(), tmp_path / "oneway.html"
            )
        )
        assert seen["wrap"] == {
            "files": [[True, False], [False, True]],
            "ranks": [[False, False], [False, False]],
        }
        assert "left" not in json.dumps(seen["wrap"])


# ------------------------------------------ a square that ends the game when reached


class TestASquareThatEndsTheGameCrossesAsCompiledSquares:
    """``rules.reach`` ends the game on arrival, and a board that did not draw it asked
    a player to find the winning square by reading the rules panel instead.

    What crosses is the *compiled* rule. The spec names its squares and mirrors them by
    a keyword, and ``engine/compile.py`` is the one place that means anything by either.
    """

    def test_a_game_with_no_such_rule_carries_no_squares(
        self, chess_spec: dict, tmp_path: Path
    ) -> None:
        from chess_variant_sandbox.viz.wrapped import goal_payload

        assert goal_payload(Engine(chess_spec).spec.reach) is None
        page = render_position(
            chess_spec, Engine(chess_spec).initial_state(), tmp_path / "none.html"
        )
        assert _payload(page)["goals"] is None

    def test_the_squares_are_the_compiled_ones_and_not_the_spelt_ones(self) -> None:
        """King of the hill names four centre squares and mirrors none of them, so both
        sides race for the same four. Nothing on the far side ever sees ``"d4"``."""
        from chess_variant_sandbox.viz.wrapped import goal_payload

        spec = load_spec(RULESETS / "kingofthehill.json")
        centre = [27, 28, 35, 36]
        assert goal_payload(Engine(spec).spec.reach) == [centre, centre]
        assert "d4" in json.dumps(spec["rules"]["reach"]["squares"])

    def test_a_mirrored_rule_sends_each_side_to_its_own_end(self) -> None:
        """Dobutsu names rank 4 and mirrors it, so the first player wins on the far
        rank and the second on the near one. Two sets, which is what makes a marker
        have to say whose square it is."""
        from chess_variant_sandbox.viz.wrapped import goal_payload

        spec = load_spec(RULESETS / "dobutsu.json")
        assert goal_payload(Engine(spec).spec.reach) == [[9, 10, 11], [0, 1, 2]]

    def test_a_rotated_rule_lands_on_the_opposite_corner(self) -> None:
        """``mirror: "rotate"`` turns the board rather than folding it, which on a nine
        by nine sends the second player's goal to the corner diagonally opposite."""
        from chess_variant_sandbox.viz.wrapped import goal_payload

        spec = load_spec(RULESETS / "intransitive.json")
        assert goal_payload(Engine(spec).spec.reach) == [[80], [0]]


class TestTheMarkerNamesASideOnlyWhenTheSidesDiffer:
    """The module in node with no DOM: which squares are marked, and which of them
    carry a bead naming the player who wins there."""

    def test_a_game_with_no_such_rule_marks_nothing(self) -> None:
        assert _wrapped("goalMarks(null)") == {}
        assert _wrapped("goalMarks(undefined)") == {}
        assert _wrapped("squareMarks({})") == {}
        assert _wrapped("squareMarks(undefined)") == {}

    def test_one_goal_both_sides_race_for_names_neither(self) -> None:
        """Racing kings sends both kings to rank 8. A bead saying "and this one is
        yours as well" on every square of it would only repeat the ring."""
        seen = _wrapped("goalMarks([[56, 57], [56, 57]])")
        assert sorted(seen) == ["56", "57"]
        assert all(sides == [] for sides in seen.values())

    def test_two_ends_of_the_board_each_name_their_own_side(self) -> None:
        assert _wrapped("goalMarks([[9, 10, 11], [0, 1, 2]])") == {
            "9": [0],
            "10": [0],
            "11": [0],
            "0": [1],
            "1": [1],
            "2": [1],
        }

    def test_a_square_both_sides_win_on_carries_both(self) -> None:
        """A board mirrored about a middle rank can send both sides to one square while
        the rest of the two sets stay apart."""
        seen = _wrapped("goalMarks([[4, 8], [4, 0]])")
        assert seen["4"] == [0, 1]
        assert seen["8"] == [0] and seen["0"] == [1]

    def test_a_square_carries_a_list_so_a_second_kind_can_join_it(self) -> None:
        """More than one rule can single out the same square -- a goal square inside a
        promotion zone -- so a kind is added to what is there rather than replacing it.
        A square nothing marks is absent, never an empty list."""
        seen = _wrapped("""
          (() => {
            const marks = squareMarks({ goals: [[4], [0]] });
            addMarks(marks, 'zone', { 4: [], 9: [1] });
            return { four: marks[4], nine: marks[9], zero: marks[0],
                     absent: marks[7] === undefined };
          })()
        """)
        assert seen["four"] == [{"kind": "goal", "sides": [0]}, {"kind": "zone", "sides": []}]
        assert seen["nine"] == [{"kind": "zone", "sides": [1]}]
        assert seen["zero"] == [{"kind": "goal", "sides": [1]}]
        assert seen["absent"], "a square no rule singles out holds nothing at all"

    def test_a_second_kind_nests_inside_the_first(self) -> None:
        """Two rings on one square are two insets, so neither is drawn over the other.
        A kind's inset is what places its ring and its beads alike.

        A made-up kind rather than either real one, because what is being tested is the
        mechanism a third kind will arrive through.
        """
        seen = _wrapped("""
          (() => {
            MARK_KINDS.spare = { colour: 'var(--zone)', inset: 0.2 };
            const html = squareBadge(0, 0, 100, [{ kind: 'goal', sides: [0] },
                                                 { kind: 'spare', sides: [1] }]);
            const rings = [...html.matchAll(/<rect x="([-\\d.]+)"/g)].map(m => +m[1]);
            const beads = [...html.matchAll(/<circle cx="([-\\d.]+)"/g)].map(m => +m[1]);
            return { rings: rings, beads: beads,
                     colours: [...new Set([...html.matchAll(/var\\(--[a-z0-9]+\\)/g)]
                       .map(m => m[0]))] };
          })()
        """)
        assert seen["rings"] == [8.5, 20.0], "the second kind sits further in"
        assert seen["beads"][0] > seen["beads"][1], "and its bead follows its own ring"
        assert "var(--zone)" in seen["colours"], "a kind brings its own colour"

    def test_the_badge_is_a_ring_with_one_bead_per_side(self) -> None:
        """A ring, because a fill is how the board says *last move* and *ground held*
        already and because the marker has to survive a piece standing on it. The bead
        is the player's own colour, which is how a board with no icons draws them."""
        seen = _wrapped("""
          (() => {
            const count = sides => {
              const html = squareBadge(0, 0, 100, [{ kind: 'goal', sides: sides }]);
              return { rings: (html.match(/<rect /g) || []).length,
                       beads: (html.match(/<circle /g) || []).length,
                       first: html.indexOf('var(--p0)') >= 0,
                       second: html.indexOf('var(--p1)') >= 0,
                       goal: html.indexOf('var(--goal)') >= 0 };
            };
            return { bare: count([]), one: count([1]), both: count([0, 1]) };
          })()
        """)
        assert seen["bare"] == {
            "rings": 1,
            "beads": 0,
            "first": False,
            "second": False,
            "goal": True,
        }
        assert seen["one"]["beads"] == 1 and seen["one"]["second"]
        assert not seen["one"]["first"], "a bead names the side that wins there"
        assert seen["both"]["beads"] == 2

    def test_the_marker_stays_inside_its_own_square(self) -> None:
        """It is drawn in the cell, so anything of it reaching past the cell would land
        on the neighbour -- and on a ghost copy the neighbour is a different square."""
        extent = _wrapped(r"""
          (() => {
            const html = squareBadge(400, 700, 100,
                                     [{ kind: 'goal', sides: [0, 1] }]);
            let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
            const see = (a, b, c, d) => {
              x0 = Math.min(x0, a); y0 = Math.min(y0, b);
              x1 = Math.max(x1, c); y1 = Math.max(y1, d);
            };
            const rects =
              /<rect x="([-\d.]+)" y="([-\d.]+)"\s+width="([-\d.]+)" height="([-\d.]+)"/g;
            const beads = /<circle cx="([-\d.]+)" cy="([-\d.]+)"\s+r="([-\d.]+)"/g;
            let hit;
            while ((hit = rects.exec(html)) !== null) {
              see(+hit[1], +hit[2], +hit[1] + +hit[3], +hit[2] + +hit[4]);
            }
            while ((hit = beads.exec(html)) !== null) {
              see(+hit[1] - +hit[3], +hit[2] - +hit[3],
                  +hit[1] + +hit[3], +hit[2] + +hit[3]);
            }
            return [x0, y0, x1, y1];
          })()
        """)
        assert extent[0] >= 400 and extent[1] >= 700
        assert extent[2] <= 500 and extent[3] <= 800


# ------------------------------------------------ a seam the rules put somewhere


_SEAM_LINES = r"""
(() => {
  const rule = RULE;
  const view = boardView(8, 8, wrapAxes(rule), { files: FILES, ranks: RANKS }, FLIP);
  const html = barrierHtml(12.5, view, rule);
  return [...html.matchAll(/<line x1="([-\d.]+)" y1="([-\d.]+)"/g)]
    .map(hit => (DOWN ? +hit[2] : +hit[1])).sort((a, b) => a - b);
})()
"""


_ONE_WAY = ("{ files: [[true, false], [false, true]], "
            "ranks: [[false, false], [false, false]] }")


def _seam_lines(rule: str, files: int = 0, ranks: int = 0, flip: str = "false",
                down: str = "false") -> object:
    """Where the barrier is drawn on its axis, in order, for one view."""
    return _wrapped(
        _SEAM_LINES.replace("RULE", rule)
        .replace("FILES", str(files))
        .replace("RANKS", str(ranks))
        .replace("FLIP", flip)
        .replace("DOWN", down)
    )


class TestTheSeamTheRulesPutSomewhere:
    """A wrapping board privileges no boundary -- unless the rules put one there.

    A seam every player may cross both ways is nowhere in particular: file ``a`` is
    beside file ``h`` exactly as ``c`` is beside ``d``, and the outline is the only
    frame there is. A seam somebody may cross one way and not the other is *somewhere*,
    because that line is the only place on its axis where a piece meets a rule. So the
    board draws it where it really is, the shift carries it with the board rather than
    with the frame, and the outline stands down on that axis.
    """

    def test_only_a_directional_seam_is_a_fact(self) -> None:
        seen = _wrapped("""
          (() => {
            const bare = { files: [[], []], ranks: [[], []] };
            const both = { files: [[true, true], [true, true]], ranks: [[], []] };
            const oneway = { files: [[true, false], [false, true]], ranks: [[], []] };
            const barred = { files: [[true, true], [false, false]], ranks: [[], []] };
            return { none: seamAxes(null), bare: seamAxes(bare), both: seamAxes(both),
                     oneway: seamAxes(oneway), barred: seamAxes(barred) };
          })()
        """)
        assert seen["none"] == {"files": False, "ranks": False}
        assert seen["bare"] == {"files": False, "ranks": False}
        assert seen["both"] == {"files": False, "ranks": False}, (
            "a seam either side crosses either way is nowhere in particular"
        )
        assert seen["oneway"] == {"files": True, "ranks": False}
        assert seen["barred"] == {"files": True, "ranks": False}, (
            "a side that may not cross at all meets something there and nowhere else"
        )

    def test_the_outline_gives_way_on_the_axis_that_has_a_seam(self) -> None:
        """Two boundaries on one axis, one fixed and one movable, would leave a reader
        deciding which of them the rules meant -- and only one of them is a rule."""
        seen = _wrapped("""
          (() => {
            const view = boardView(8, 8, { files: true, ranks: true }, null, false);
            const count = rule =>
              (outlineHtml(12.5, view, rule).match(/<line/g) || []).length;
            const both = { files: [[true, true], [true, true]],
                           ranks: [[true, true], [true, true]] };
            const oneway = { files: [[true, false], [false, true]],
                             ranks: [[true, true], [true, true]] };
            return [count(undefined), count(both), count(oneway)];
          })()
        """)
        assert seen == [4, 4, 2], "the files axis keeps no outline once its seam is real"

    def test_the_barrier_is_welded_to_the_board_and_the_frame_is_not(self) -> None:
        """The seam is the low edge's own line -- file ``a``'s -- so shifting the view
        carries it across the frame a square at a time while the outline stays where the
        frame is. Every copy of it inside the frame is drawn, because every copy of it
        is the same line: at the naming offset both edges of the repeat are that line.
        """
        assert _seam_lines(_ONE_WAY) == [0, 100], "both edges of the repeat are one seam"
        assert _seam_lines(_ONE_WAY, files=3) == [62.5], (
            "three files along, the seam is five squares into the frame"
        )
        assert _seam_lines(_ONE_WAY, files=1) == [-12.5, 87.5]
        assert _seam_lines(_ONE_WAY, files=8) == _seam_lines(_ONE_WAY), (
            "a shift of a whole board is no shift at all, for the seam as for the rest"
        )

    def test_turning_the_board_round_mirrors_the_seam_and_moves_it_nowhere(self) -> None:
        assert _seam_lines(_ONE_WAY, files=3, flip="true") == [37.5], (
            "the same line, read from the other end of a hundred-unit board"
        )

    def test_a_seam_nobody_is_stopped_at_is_not_drawn(self) -> None:
        assert _wrapped("""
          (() => {
            const both = { files: [[true, true], [true, true]],
                           ranks: [[true, true], [true, true]] };
            return [barrierHtml(12.5, boardView(8, 8, wrapAxes(both), null, false), both),
                    barrierHtml(12.5, flatView(8, 8, false), null)];
          })()
        """) == ["", ""]


_SEAM_MARKS = r"""
(() => {
  const rule = RULE;
  const view = boardView(8, 8, wrapAxes(rule), { files: FILES, ranks: RANKS }, FLIP);
  const html = barrierHtml(12.5, view, rule);
  return [...html.matchAll(/<title>([^<]*)<\/title>([\s\S]*?)<\/g>/g)].map(hit => ({
    words: hit[1],
    arrows: [...hit[2].matchAll(/<polygon points="([-\d.]+),([-\d.]+)/g)]
      .map(m => (DOWN ? +m[2] : +m[1])),
    fill: (hit[2].match(/fill="(var\(--[a-z0-9]+\))"/) || [])[1],
  }));
})()
"""


def _seam_marks(rule: str, files: int = 0, ranks: int = 0, flip: str = "false",
                down: str = "false") -> object:
    """What each side's marker on the seam says: its words, its arrows, its colour."""
    return _wrapped(
        _SEAM_MARKS.replace("RULE", rule)
        .replace("FILES", str(files))
        .replace("RANKS", str(ranks))
        .replace("FLIP", flip)
        .replace("DOWN", down)
    )


class TestTheSeamSaysWhichWayCrossingIsAllowed:
    """Two squares do not say which way round a piece went and neither does a line.

    The marker is an arrowhead per edge that side may leave by, pointing the way the
    piece goes, filled in that side's own colour. A side that may not cross at all gets
    a bare disc, which is a wall for them and nowhere else on the board.
    """

    def test_each_side_gets_the_one_way_it_may_go(self) -> None:
        """One-way cylinder chess: White leaves by the a-file and Black by the h-file,
        so the two arrows on the one line point opposite ways."""
        seen = _seam_marks(_ONE_WAY, files=3)
        assert [mark["words"] for mark in seen] == [
            "the first player leaves by the a-file",
            "the second player leaves by the h-file",
        ]
        assert [mark["fill"] for mark in seen] == ["var(--p0)", "var(--p1)"]
        assert len(seen[0]["arrows"]) == 1 and len(seen[1]["arrows"]) == 1
        assert seen[0]["arrows"][0] < 62.5 < seen[1]["arrows"][0], (
            "off the a-file is towards the h-file, which is the other way for Black"
        )

    def test_turning_the_board_round_turns_the_arrows_with_it(self) -> None:
        """The same answer `stepXY` gives a step of one square: a board turned round
        runs its files the other way, and an arrow that did not would send a reader to
        the edge the piece does not leave by."""
        seen = _seam_marks(_ONE_WAY, files=3, flip="true")
        assert seen[0]["arrows"][0] > 37.5 > seen[1]["arrows"][0]

    def test_a_rank_seam_points_up_and_down_rather_than_across(self) -> None:
        """One-way tunnel chess, where the first player leaves by the top."""
        rule = ("{ files: [[false, false], [false, false]], "
                "ranks: [[false, true], [true, false]] }")
        seen = _seam_marks(rule, ranks=3, down="true")
        assert [mark["words"] for mark in seen] == [
            "the first player leaves by rank 8",
            "the second player leaves by rank 1",
        ]
        assert seen[0]["arrows"][0] < 37.5 < seen[1]["arrows"][0], (
            "up the screen is towards the high rank, and down is towards the low one"
        )

    def test_every_copy_of_the_seam_carries_the_same_markers(self) -> None:
        """At the naming offset both edges of the repeat are the seam. A copy drawn
        bare beside a copy carrying arrows would invite a reader to go looking for the
        difference between two drawings of one line."""
        seen = _seam_marks(_ONE_WAY)
        assert len(seen) == 4, "two copies in the frame, two sides on each"
        assert [mark["words"] for mark in seen[:2]] == [
            mark["words"] for mark in seen[2:]
        ]

    def test_a_seam_both_sides_read_alike_names_neither(self) -> None:
        """One marker in the seam's own colour, because a pair saying the same thing
        twice would only repeat the line -- `goalMarks`'s reading again."""
        rule = ("{ files: [[true, false], [true, false]], "
                "ranks: [[false, false], [false, false]] }")
        seen = _seam_marks(rule, files=3)
        assert len(seen) == 1
        assert seen[0]["words"] == "either side leaves by the a-file"
        assert seen[0]["fill"] == "var(--seam)"

    def test_a_side_that_may_not_cross_gets_a_disc_and_no_arrow(self) -> None:
        rule = ("{ files: [[true, true], [false, false]], "
                "ranks: [[false, false], [false, false]] }")
        seen = _seam_marks(rule, files=3)
        assert [mark["words"] for mark in seen] == [
            "the first player crosses this seam either way",
            "the second player may not cross this seam",
        ]
        assert len(seen[0]["arrows"]) == 2 and seen[1]["arrows"] == []


# ------------------------------------------------------ the squares a piece promotes on


class TestThePromotionZoneCrossesAsCompiledSquares:
    """``rules.promotion`` states its zone once from the first player's side, and
    ``spec.schema.promotion_squares`` reflects it. What crosses is the resolved pair of
    square sets, so neither renderer ever reads ``"last"``.
    """

    def test_a_game_with_no_promotion_carries_no_squares(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.viz.wrapped import zone_payload

        spec = load_spec(RULESETS / "dull.json")
        engine = Engine(spec)
        assert zone_payload(engine.spec.promotions) is None
        page = render_position(spec, engine.initial_state(), tmp_path / "none.html")
        assert _payload(page)["zones"] is None

    def test_the_band_is_the_compiled_ranks_and_not_the_word(
        self, chess_spec: dict
    ) -> None:
        from chess_variant_sandbox.viz.wrapped import zone_payload

        engine = Engine(chess_spec)
        assert zone_payload(engine.spec.promotions) == [
            list(range(56, 64)),
            list(range(0, 8)),
        ]
        assert chess_spec["rules"]["promotion"]["rank"] == "last", (
            "which is a word, and stays on this side of the boundary"
        )

    def test_a_zone_several_ranks_deep_is_the_whole_band(self) -> None:
        """Shogi promotes in the last three ranks, and promotes six pieces there. The
        mark belongs to a square rather than to a piece, so it is one band."""
        from chess_variant_sandbox.viz.wrapped import zone_payload

        spec = load_spec(RULESETS / "shogi.json")
        engine = Engine(spec)
        assert len(engine.spec.promotions) > 1, "six rules, and one band"
        first, second = zone_payload(engine.spec.promotions)
        assert {square // 9 for square in first} == {6, 7, 8}
        assert {square // 9 for square in second} == {0, 1, 2}
        assert len(first) == 27 and len(second) == 27

    def test_a_goal_square_can_be_a_promotion_square(self) -> None:
        """Dobutsu's chick promotes on the rank its lion wins on, which is the case the
        two kinds of mark have to share a square for."""
        from chess_variant_sandbox.viz.wrapped import goal_payload, zone_payload

        spec = load_spec(RULESETS / "dobutsu.json")
        engine = Engine(spec)
        assert zone_payload(engine.spec.promotions) == [
            [9, 10, 11],
            [0, 1, 2],
        ]
        assert goal_payload(engine.spec.reach) == [[9, 10, 11], [0, 1, 2]]


class TestThePromotionZoneIsAMarkBehindAToggle:
    """The module in node with no DOM: which squares the band marks, how it shares a
    square with a goal ring, and what the reader's toggle takes off the board.
    """

    def test_a_game_with_no_promotion_marks_nothing(self) -> None:
        assert _wrapped("zoneMarks(null)") == {}
        assert _wrapped("squareMarks({ zones: null })") == {}

    def test_two_ends_of_the_board_each_name_their_own_side(self) -> None:
        """Chess mirrors its band, so each end belongs to one player and the mark says
        which; a game promoting both sides in one place names neither."""
        assert _wrapped("zoneMarks([[56, 57], [0, 1]])") == {
            "56": [0], "57": [0], "0": [1], "1": [1],
        }
        seen = _wrapped("zoneMarks([[56, 57], [56, 57]])")
        assert all(sides == [] for sides in seen.values())

    def test_the_zone_joins_a_goal_square_rather_than_replacing_it(self) -> None:
        seen = _wrapped("""
          (() => {
            const marks = squareMarks({ goals: [[9], [0]], zones: [[9, 10], [0, 1]] });
            return { nine: marks[9], ten: marks[10], absent: marks[5] === undefined };
          })()
        """)
        assert seen["nine"] == [
            {"kind": "goal", "sides": [0]},
            {"kind": "zone", "sides": [0]},
        ], "the goal ring first, so it is the outer one"
        assert seen["ten"] == [{"kind": "zone", "sides": [0]}]
        assert seen["absent"], "a square no rule singles out holds nothing at all"

    def test_the_zone_is_hatched_ground_with_no_beads_under_the_goal_ring(
        self,
    ) -> None:
        """Q577 (the first-visit study's F1): the zone was a ring with a bead per side,
        and all five readers read it as something else -- castling rights, *special*,
        a pick, purple dots. A ring is a target and a tint is *this game, now*, so the
        zone is ground: the whole cell filled with a hatch, no bead. Whose zone it is
        moved to the legend line. A goal square inside a zone keeps its ring and bead."""
        seen = _wrapped(r"""
          (() => {
            const html = squareBadge(0, 0, 100, [{ kind: 'goal', sides: [0] },
                                                 { kind: 'zone', sides: [1] }], null, 'b');
            return {
              rings: [...html.matchAll(/<rect x="([-\d.]+)"[^>]*stroke=/g)].length,
              hatch: [...html.matchAll(/fill="url\(#(mark-hatch-zone-b)\)"/g)].length,
              beads: [...html.matchAll(/<circle [^>]*stroke="(var\(--[a-z]+\))"/g)]
                .map(m => m[1]),
              kinds: [...html.matchAll(/data-mark="([a-z]+)"/g)].map(m => m[1]),
            };
          })()
        """)
        assert seen["hatch"] == 1, "the zone fills its cell with its own pattern"
        assert seen["rings"] == 1, "the goal is the one ring"
        assert seen["beads"] == ["var(--goal)"], "the goal's bead, and none for the zone"
        assert seen["kinds"] == ["goal", "zone"]

    def test_a_drawing_defines_the_hatch_only_for_the_kinds_it_carries(self) -> None:
        """A pattern is an element with an id, and an id is the document's: each drawing
        carries its own under its own tag. One with no zone defines nothing, so nothing on
        it paints in the zone's colour."""
        seen = _wrapped("""
          (() => ({
            zoned: markDefs(squareMarks({ zones: [[56], [0]] }), 'still3'),
            bare: markDefs(squareMarks({ goals: [[4], [4]] }), 'still3'),
            key: markKey('zone'),
          }))()
        """)
        assert 'id="mark-hatch-zone-still3"' in seen["zoned"]
        assert "var(--zone)" in seen["zoned"]
        assert seen["bare"] == ""
        assert 'id="mark-hatch-zone-key"' in seen["key"]
        assert "url(#mark-hatch-zone-key)" in seen["key"], "the key is the mark, drawn small"

    def test_a_check_is_a_glow_on_the_squares_it_is_given(self) -> None:
        """The check is the position's, not the rules': `squareMarks` draws it on the
        squares it is handed (the board's `checkSquares`), first, so it is the layer under
        every ring, and a still or a replay that hands none draws none."""
        seen = _wrapped("""
          (() => {
            const marks = squareMarks({ zones: [[4], [60]], checked: [4] });
            return { four: marks[4], none: squareMarks({ zones: [[4], [60]] })[4],
                     html: squareBadge(0, 0, 100, marks[4], null, 'b') };
          })()
        """)
        assert seen["four"] == [{"kind": "check", "sides": []}, {"kind": "zone", "sides": [0]}]
        assert seen["none"] == [{"kind": "zone", "sides": [0]}]
        assert 'fill="var(--alarm)"' in seen["html"]
        assert seen["html"].index('data-mark="check"') < seen["html"].index('data-mark="zone"')

    def test_every_mark_kind_says_what_it_is(self) -> None:
        """The legend line is generated from `MARK_KINDS` (plan-visual.md S3), so a kind
        cannot be drawn without words for it."""
        rows = _wrapped("MARK_KINDS")
        for kind, row in rows.items():
            assert row.get("colour", "").startswith("var(--"), kind
            assert row.get("name") and row.get("label"), f"{kind} has no words for the legend"

    def test_the_toggle_takes_a_kind_off_whole(self) -> None:
        """Off rather than faint: a square already carries the last move, a selection
        and a piece."""
        seen = _wrapped("""
          (() => {
            const source = { goals: [[4], [4]], zones: [[56], [0]] };
            return { on: squareMarks(source),
                     off: squareMarks(source, { zone: true }),
                     neither: squareMarks(source, { goal: true, zone: true }) };
          })()
        """)
        assert sorted(seen["on"]) == ["0", "4", "56"]
        assert sorted(seen["off"]) == ["4"], "the goal survives its neighbour's toggle"
        assert seen["neither"] == {}

    def test_the_toggle_remembers_and_shows_the_zone_when_it_cannot(self) -> None:
        """The default is on -- a promotion zone is a rule nothing else on the board
        shows -- and a browser that refuses site data throws on the property itself
        rather than returning nothing, which must not stop the board drawing."""
        seen = _wrapped("""
          (() => {
            const answers = {};
            global.window = { localStorage: { getItem: () => null, setItem: () => {} } };
            answers.fresh = zonesShown();
            window.localStorage.getItem = () => '0';
            answers.put_away = zonesShown();
            window.localStorage.getItem = () => '1';
            answers.brought_back = zonesShown();
            window.localStorage = { get getItem() { throw new Error('refused'); } };
            answers.refused = zonesShown();
            rememberZones(true);
            answers.survived = true;
            return answers;
          })()
        """)
        assert seen == {
            "fresh": True,
            "put_away": False,
            "brought_back": True,
            "refused": True,
            "survived": True,
        }

    def test_the_control_is_one_button_both_surfaces_share(self) -> None:
        seen = _wrapped("[zoneToggleHtml(true), zoneToggleHtml(false)]")
        assert 'aria-pressed="true"' in seen[0] and "u-c-zone" in seen[0]
        assert 'aria-pressed="false"' in seen[1] and "u-c-zone" not in seen[1]
        assert all("promotion zone" in html for html in seen)


# ------------------------------------------------- a replay steps through the beats

_REPLAY_STUB = r"""
  const __nodes = {};
  const __el = id => __nodes[id] || (__nodes[id] = {
    id: id, innerHTML: '', textContent: '', dataset: {}, disabled: false, onclick: null,
    classList: { toggle() {} }, setAttribute() {}, addEventListener() {},
    querySelector() { return null; }, querySelectorAll() { return []; }, appendChild() {},
  });
  global.document = { getElementById: __el, querySelectorAll: () => [],
                      addEventListener() {}, title: '' };
  global.window = { matchMedia: () => ({ matches: true }) };
"""

# What a step shows: the beat, the squares its drawing marks, and its board.
_STEP = r"""
(() => {
  const drawn = cellHtml;
  let marks = [];
  cellHtml = (board, marked, square, u, v, depth) => {
    if (depth === 0 && marked.has(square)) marks.push(square);
    return drawn(board, marked, square, u, v, depth);
  };
  const seen = [];
  for (const index of STEPS) {
    marks = [];
    goto(index);
    seen.push({ at: at, kind: BEATS[at].kind, ply: BEATS[at].ply,
                marks: marks.sort((a, b) => a - b), board: BEATS[at].board });
  }
  return { beats: BEATS.map(b => b.kind), rows: __el('moves').innerHTML, seen: seen };
})()
"""


def _replayed(spec: dict, tmp_path: Path, steps: list[int], seed: int = 0) -> tuple:
    """A random game of `spec`, rendered, and the replay page stepped to each beat in
    `steps` by its own `goto`, in node. Returns the payload and what the page did."""
    from chess_variant_sandbox.tournament.play import play_game
    from chess_variant_sandbox.viz.html import render_game

    record = play_game(Engine(spec), ("random", "random"), seed=seed)
    out = render_game(spec, record, tmp_path / "replay.html")
    script = re.search(r"<script>(.*)</script>", out.read_text(encoding="utf-8"), re.S)
    assert script is not None
    expression = _STEP.replace("STEPS", json.dumps(steps))
    done = node_session.run(
        _REPLAY_STUB + script.group(1) + "\nconsole.log(JSON.stringify(" + expression + "));"
    )
    assert done.returncode == 0, done.stderr
    return _payload(out), json.loads(done.stdout)


class TestTheReplayStepsThroughTheWildPiecesTurn:
    """Q75. The replay redraws per step and marks what changed, and a step was a ply --
    so in a wolves game four wolves moved at once with nothing marking where any of them
    came from, and a wolf's capture was a piece vanishing beside somebody else's move.

    A step is now a beat, split by the play board's own `plyBeats` from the phase the
    core logged, and the move list gives the phase a row in the play board's words. A
    game without wild pieces has one beat per ply and replays as it did.
    """

    WOLVES = RULESETS.parent / "archive" / "wolves.json"

    def test_a_ply_with_a_phase_is_two_steps(self, tmp_path: Path) -> None:
        data, page = _replayed(load_spec(self.WOLVES), tmp_path, [1, 2])
        phased = [bool(move["phase"]) for move in data["moves"]]
        assert phased[0], "the wolves walk after the very first move"
        assert page["beats"] == ["start"] + [
            kind for has in phased for kind in (["move", "phase"] if has else ["move"])
        ]

        between, after = page["seen"]
        first = data["moves"][0]
        # The move's beat marks the mover's two squares and stands the wolves where they
        # started; the phase's marks every square a wolf left or arrived on.
        assert between["kind"] == "move"
        assert between["marks"] == sorted([first["from"], first["to"]])
        for step in first["phase"]:
            assert between["board"][step["from"]] == data["boards"][0][step["from"]]
        assert after["kind"] == "phase" and after["ply"] == 1
        assert after["marks"] == sorted(
            {square for step in first["phase"] for square in (step["from"], step["to"])}
        )
        assert after["board"] == data["boards"][1]

    def test_the_move_list_gives_the_phase_a_row(self, tmp_path: Path) -> None:
        data, page = _replayed(load_spec(self.WOLVES), tmp_path, [])
        rows = re.findall(
            r'<div class="(move[^"]*)" data-beat="(\d+)">(.*?)</div>', page["rows"], re.S
        )
        assert [int(beat) for _, beat, _ in rows] == list(range(1, len(page["beats"])))
        phases = [text for classes, _, text in rows if "phase" in classes]
        assert len(phases) == sum(1 for move in data["moves"] if move["phase"])
        assert all(text.startswith("and the wolf on ") for text in phases), phases[:3]

    def test_an_ordinary_game_is_one_step_a_ply(self, chess_spec: dict, tmp_path: Path) -> None:
        data, page = _replayed(chess_spec, tmp_path, [1, 2])
        assert page["beats"] == ["start"] + ["move"] * len(data["moves"])
        assert "phase" not in page["rows"]
        for seen in page["seen"]:
            move = data["moves"][seen["at"] - 1]
            assert seen["marks"] == sorted([move["from"], move["to"]])
            assert seen["board"] == data["boards"][seen["at"]]

    def test_the_split_and_the_words_are_one_copy(self, tmp_path: Path) -> None:
        """Both surfaces carry `plyBeats` and `phaseWords` through the shared module and
        neither holds a copy of its own."""
        from page_text import PAGE

        spec = load_spec(self.WOLVES)
        page = render_position(spec, Engine(spec).initial_state(), tmp_path / "one.html")
        for text in (page.read_text(encoding="utf-8"), PAGE):
            assert text.count("function plyBeats(") == 1
            assert text.count("function phaseWords(") == 1


# ------------------------------------------------ a board cut to a hexagon (Q281b)

ARCHIVE = Path(__file__).resolve().parents[1] / "archive"
HEXAGONS = ("hex-glinski-hexagon", "hex-glinski-flanks", "hex-glinski-flanks-one-way")


def _hexagon(name: str, wrap: dict | None = None) -> dict:
    """What every board payload sends for a schema 16 showcase: its size, its walls and
    absent corners, and its compiled wrap rule -- the showcase's own, or `wrap` in its
    place."""
    from chess_variant_sandbox.viz.wrapped import walls_payload, wrap_payload

    if name in REBUILT:  # retired by Q870 for `pinwheel`; rebuilt from the hexagon
        spec = REBUILT[name]()
    else:
        spec = json.loads((ARCHIVE / f"{name}.json").read_text("utf-8"))
    if wrap is not None:
        spec["rules"]["wrap"] = wrap
    compiled = Engine(spec).spec
    return {"W": compiled.width, "H": compiled.height, **walls_payload(compiled),
            "wrap": wrap_payload(compiled.wrap, compiled.copies)}


def _cut(name: str, body: str, wrap: dict | None = None) -> object:
    """Evaluate `body` with `P` the showcase's payload and `view`, `turned` and `still`
    its views: the play board, the same turned round, and a still preview."""
    return _wrapped(f"""(() => {{
      const P = {json.dumps(_hexagon(name, wrap))};
      const cut = boardCut(P.absent, P.wrap);
      const view = boardView(P.W, P.H, wrapAxes(P.wrap), null, false, 'hex', cut);
      const turned = boardView(P.W, P.H, wrapAxes(P.wrap), null, true, 'hex', cut);
      const still = flatView(P.W, P.H, false, 'hex', cut);
      {body}
    }})()""")


class TestAHexagonIsDrawnAsAHexagon:
    """Q281b: `board.shape: "hexagon"` drawn as one, on every board that draws the
    shared module, asserted against the three showcases in `archive/`."""

    @pytest.mark.parametrize("name", HEXAGONS)
    def test_the_payload_keeps_the_corners_apart_from_the_walls(self, name: str) -> None:
        sent = _hexagon(name)
        assert sent["obstacles"] == []
        assert len(sent["absent"]) == 30

    def test_the_walled_twin_still_sends_its_corners_as_walls(self) -> None:
        from chess_variant_sandbox.viz.wrapped import walls_payload

        spec = hex_glinski_promoting()
        sent = walls_payload(Engine(spec).spec)
        assert (len(sent["obstacles"]), sent["absent"]) == (30, [])

    def test_a_wrap_carries_the_pairs_and_the_six_copies(self) -> None:
        assert _hexagon("hex-glinski-hexagon")["wrap"] is None
        wrap = _hexagon("hex-glinski-flanks-one-way")["wrap"]
        assert wrap["nw_se"] == [[False, True], [True, False]]
        assert "files" not in wrap and "ranks" not in wrap
        by_side = {copy["side"]: copy for copy in wrap["copies"]}
        assert sorted(by_side) == ["n", "ne", "nw", "s", "se", "sw"]
        assert by_side["se"]["permission"] == [True, False]
        assert by_side["nw"]["permission"] == [False, True]
        assert (by_side["se"]["pair"], by_side["se"]["high"]) == ("nw_se", True)

    @pytest.mark.parametrize("name", HEXAGONS)
    def test_absent_cells_are_not_drawn_and_the_frame_is_eleven_cells_wide(
        self, name: str
    ) -> None:
        seen = _cut(name, """
          let cells = 0;
          for (let s = 0; s < P.W * P.H; s++) if (!cellAbsent(s, view)) cells++;
          return { cells: cells, still: viewBox(still, 10).split(' ').map(Number),
                   box: viewBox(view, 10).split(' ').map(Number),
                   corner: cellAbsent(0, view), centre: cellAbsent(60, view) };
        """)
        assert (seen["cells"], seen["corner"], seen["centre"]) == (91, True, False)
        joined = name != "hex-glinski-hexagon"
        marked = name == "hex-glinski-flanks-one-way"
        margin = 2.4 * 10 if joined else 0
        assert seen["box"][2] == pytest.approx(110 + 2 * margin)
        # A still draws no ghosts, and opens only by an arrowhead's reach for a marker.
        reach = 10 * (0.47 + 0.05) if marked else 0
        assert seen["still"][2] == pytest.approx(110 + 2 * reach)

    def test_labels_run_up_the_lower_left_along_the_bottom_and_up_the_upper_left(
        self,
    ) -> None:
        seen = _cut("hex-glinski-hexagon", """
          const read = vw => {
            const found = {};
            for (let s = 0; s < P.W * P.H; s++) {
              if (cellAbsent(s, vw)) continue;
              const xy = squareXY(s, 10, vw);
              const text = coordsHtml(s, xy[0], xy[1], 10, vw)
                .match(/>([a-z0-9]+)<\\/text>/g);
              if (text) found[squareName(s % P.W, Math.floor(s / P.W))] =
                text.map(t => t.slice(1, -7));
            }
            return found;
          };
          return { upright: read(view), turned: read(turned) };
        """)
        upright = seen["upright"]
        labels = [label for found in upright.values() for label in found]
        assert sorted(x for x in labels if x.isalpha()) == list("abcdefghijk")
        assert sorted(int(x) for x in labels if x.isdigit()) == list(range(1, 12))
        assert upright["c4"] == ["c", "4"]
        assert upright["a6"] == ["a", "6"]
        for square in ("f1", "g1", "h1", "i1", "j1", "k1"):
            assert upright[square][0] == square[0]
        for rank in range(7, 12):
            assert upright[f"a{rank}"] == [str(rank)]
        # Turned round, every name is still on exactly one cell.
        turned = [label for found in seen["turned"].values() for label in found]
        assert sorted(turned) == sorted(labels)

    def test_outside_the_board_the_names_are_the_same_names_on_the_same_edges(self) -> None:
        """V4 sets the names off a board with nothing round it into a margin; it must be
        the same names `coordsHtml` puts in the cells, each beside its own cell, pointing
        away from the board."""
        seen = _cut("hex-glinski-hexagon", """
          const inside = vw => {
            const found = [];
            for (let s = 0; s < P.W * P.H; s++) {
              if (cellAbsent(s, vw)) continue;
              const xy = squareXY(s, 10, vw);
              (coordsHtml(s, xy[0], xy[1], 10, vw).match(/>([a-z0-9]+)<\\/text>/g) || [])
                .forEach(t => found.push(t.slice(1, -7)));
            }
            return found.sort();
          };
          const outside = vw => edgeLabels(10, vw);
          return { outside: [labelsOutside(view), labelsOutside(turned)],
                   upright: [inside(view), outside(view)],
                   turned: [inside(turned), outside(turned)] };
        """)
        assert seen["outside"] == [True, True], "an unjoined hexagon has nothing round it"
        for inside, outside in (seen["upright"], seen["turned"]):
            assert sorted(label["text"] for label in outside) == inside
            for label in outside:
                assert abs((label["dx"] ** 2 + label["dy"] ** 2) - 1) < 1e-9
                assert label["dx"] <= 1e-9, "a hexagon's names stand to its lower left"

    def test_an_unjoined_hexagon_has_no_ghosts_and_no_seams(self) -> None:
        seen = _cut("hex-glinski-hexagon", """
          return [ghostCells(view).length, outlineHtml(10, view, P.wrap),
                  barrierHtml(10, view, P.wrap)];
        """)
        assert seen == [0, "", ""]

    @pytest.mark.parametrize("name", HEXAGONS[1:])
    def test_ghosts_lie_only_across_the_joined_sides(self, name: str) -> None:
        seen = _cut(name, """
          const ghosts = ghostCells(view);
          return { count: ghosts.length,
                   sides: [...new Set(ghosts.map(g =>
                     copyAt(g.u, g.v, view, cut.copies).copy.side))].sort(),
                   depths: [...new Set(ghosts.map(g => g.depth))].sort(),
                   real: ghosts.every(g => !cellAbsent(g.square, view)),
                   faded: ghostFade(3) < ghostFade(1) };
        """)
        assert seen["count"] > 0
        assert seen["sides"] == ["nw", "se"]
        assert seen["depths"] == [1, 2, 3]
        assert seen["real"] and seen["faded"]

    def test_every_side_is_found_and_each_is_the_same_length(self) -> None:
        seen = _cut("hex-glinski-flanks", """
          const sides = cutSides(10, view);
          return Object.keys(sides).sort().map(side => sides[side].edges.length);
        """)
        assert len(seen) == 6
        assert len(set(seen)) == 1

    def test_a_side_both_players_cross_both_ways_is_a_repeat_line(self) -> None:
        seen = _cut("hex-glinski-flanks", """
          return [outlineHtml(10, view, P.wrap), barrierHtml(10, view, P.wrap),
                  outlineHtml(10, still, P.wrap)];
        """)
        outline, barrier, still = seen
        assert outline.count("<path") == 2 and 'stroke="var(--ink)"' in outline
        assert barrier == ""
        assert still == outline

    def test_a_one_way_side_is_a_barrier_with_its_arrows(self) -> None:
        seen = _cut("hex-glinski-flanks-one-way", """
          return [outlineHtml(10, view, P.wrap), barrierHtml(10, view, P.wrap)];
        """)
        outline, barrier = seen
        assert outline == ""
        assert barrier.count('stroke="var(--seam)"\n      stroke-width="1"') == 2
        # The players differ, so each side carries a marker per player.
        assert barrier.count("<title>") == 4
        assert barrier.count("<polygon") == 4
        assert "the first player leaves by the lower-right side" in barrier
        assert "the second player leaves by the upper-left side" in barrier

    def test_a_crossing_move_goes_through_the_copy_the_mover_may_leave_into(
        self,
    ) -> None:
        # a6 is (0, 5); the cell across the upper-left side from it is k1, (10, 0).
        seen = _cut("hex-glinski-flanks-one-way", """
          const a6 = 5 * P.W, k1 = 10;
          return { first: wrapStep(a6, k1, P.W, P.H, wrapAllow(P.wrap, 0)),
                   second: wrapStep(a6, k1, P.W, P.H, wrapAllow(P.wrap, 1)),
                   back: wrapStep(k1, a6, P.W, P.H, wrapAllow(P.wrap, 0)),
                   near: wrapStep(60, 61, P.W, P.H, wrapAllow(P.wrap, 1)),
                   across: stepXY({ files: 1, ranks: 0 }, 10, view),
                   up: stepXY({ files: 0, ranks: 1 }, 10, view),
                   upTurned: stepXY({ files: 0, ranks: 1 }, 10, turned) };
        """)
        assert seen["first"] == {"files": 10, "ranks": -5, "wrapped": False}
        assert seen["second"] == {"files": -1, "ranks": 0, "wrapped": True}
        assert seen["back"] == {"files": 1, "ranks": 0, "wrapped": True}
        assert seen["near"] == {"files": 1, "ranks": 0, "wrapped": False}
        assert seen["across"] == pytest.approx([10, 0])
        assert seen["up"] == pytest.approx([5, -10 * 3 ** 0.5 / 2])
        assert seen["upTurned"] == pytest.approx([-5, 10 * 3 ** 0.5 / 2])

    @pytest.mark.parametrize("name", HEXAGONS)
    def test_the_replay_sends_the_corners_apart_and_skips_them(
        self, name: str, tmp_path: Path
    ) -> None:
        spec = REBUILT[name]() if name in REBUILT else json.loads(
            (ARCHIVE / f"{name}.json").read_text("utf-8")
        )
        page = render_position(spec, Engine(spec).initial_state(), tmp_path / "hex.html")
        data = _payload(page)
        assert (data["obstacles"], len(data["absent"])) == ([], 30)
        text = page.read_text(encoding="utf-8")
        assert "const CUT = boardCut(DATA.absent, DATA.wrap);" in text
        assert "if (cellAbsent(square, FRAME)) continue;" in text


# A hexagon with every pair joined and one of them a barrier: the only kind on which two
# ghost copies touch across a barrier, so the only kind with one between ghosts to draw.
_ALL_JOINED_ONE_WAY = {"n_s": "both", "ne_sw": "both",
                       "nw_se_by_player": {"0": "se", "1": "nw"}}
_TORUS = {"n_s": "both", "ne_sw": "both", "nw_se": "both"}


class TestAHexagonsSeamRunsOnAcrossItsGhosts:
    """Q519: a rectangle's barrier runs on through its ghost rings (`seamHtml` spans the
    margin), so a hexagon's runs on along the sides where two ghost copies meet: bare and
    at full strength there, as a rectangle's is, with its markers left on the board."""

    def test_the_sides_between_ghosts_are_the_third_pair(self) -> None:
        seen = _cut("hex-glinski-flanks", """
          const sides = cutGhostSides(10, view);
          return Object.keys(sides).sort().map(pair => [pair, sides[pair].edges.length]);
        """, _ALL_JOINED_ONE_WAY)
        assert [pair for pair, _ in seen] == ["n_s", "ne_sw", "nw_se"]
        assert all(count > 0 for _, count in seen)
        assert len({count for _, count in seen}) == 1, "the six meetings are alike"

    def test_a_barrier_runs_on_from_the_boards_corners_without_its_markers(self) -> None:
        seen = _cut("hex-glinski-flanks", r"""
          const html = barrierHtml(10, view, P.wrap);
          const ends = d => [...d.matchAll(/[ML]([-\d.e]+) ([-\d.e]+)/g)]
            .map(m => [+m[1], +m[2]]);
          const board = Object.values(cutSides(10, view)).flatMap(side => ends(sidePath(side)));
          const ghost = [...html.matchAll(/<path d="([^"]*)"[^>]*data-ghost-seam="([a-z_]+)"/g)]
            .map(m => ({ pair: m[2], ends: ends(m[1]) }));
          const touches = ghost.flatMap(g => g.ends).filter(p =>
            board.some(q => Math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-6)).length;
          return { ghost: ghost.map(g => g.pair), touches: touches,
                   titles: (html.match(/<title>/g) || []).length,
                   still: barrierHtml(10, still, P.wrap).includes('data-ghost-seam') };
        """, _ALL_JOINED_ONE_WAY)
        assert seen["ghost"] == ["nw_se"], "only the barrier runs on; a repeat line does not"
        # Copy n's lower-right side is copy ne's upper-left: it leaves the board at the
        # corner between the board's n and ne sides, and its half-turn at s and sw.
        assert seen["touches"] == 2, "the line leaves the board at two of its corners"
        assert seen["titles"] == 4, "no marker on a ghost's side"
        assert not seen["still"], "a still has no ghosts, so no seam between them"

    @pytest.mark.parametrize(("name", "wrap", "outline"), [
        # Slow (Q749): compiling a hexagon with all three pairs joined, which nothing else
        # here builds, is 2.9 s of the test's 3.
        pytest.param("hex-glinski-flanks", _TORUS, 6, marks=pytest.mark.slow),
        ("hex-glinski-flanks-one-way", None, 0),
        ("hex-glinski-flanks", None, 2),
    ])
    def test_nothing_is_added_where_no_barrier_lies_between_ghosts(
        self, name: str, wrap: dict | None, outline: int
    ) -> None:
        seen = _cut(name, """
          return [outlineHtml(10, view, P.wrap).split('<path').length - 1,
                  barrierHtml(10, view, P.wrap).includes('data-ghost-seam')];
        """, wrap)
        assert seen == [outline, False]


def test_the_chancellor_is_a_tower_with_a_knights_head() -> None:
    """The chancellor (rook + knight) is drawn tower first (first-visit study N2).

    Commons' compound is a whole knight on a sliver of upturned battlements, and a
    first-time visitor read it as a knight. The library's own drawing is Cburnett's rook,
    shortened, with his knight's head on its battlements, so both parts are in the file:
    the rook's crenellated top and the knight's head. It is recorded in `fetch.py` as
    drawn here, so a rebuild keeps it rather than fetching Commons' one again, and it is
    credited to the artist whose parts it is made of, under his licence.
    """
    from importlib.util import module_from_spec, spec_from_file_location

    from chess_variant_sandbox.viz import pieces

    if icon_for("chancellor", 0) is None:  # the library is optional
        pytest.skip("icon library not fetched")
    found = spec_from_file_location("fetch", pieces.ASSET_ROOT / "fetch.py")
    assert found is not None and found.loader is not None
    fetch = module_from_spec(found)
    found.loader.exec_module(fetch)
    assert "chancellor" not in fetch.CBURNETT

    battlements = "L 15,16.5 L 15,18.5 L 20,18.5 L 20,16.5 L 25,16.5 L 25,18.5 L 30,18.5 L 30,16.5"
    head = "C 14.5,6.5 16.5,10 16.5,10 L 18.5,10"
    manifest = json.loads((pieces.ASSET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    for colour in ("white", "black"):
        stem = f"chancellor-{colour}"
        drawing = (pieces.ASSET_ROOT / "svg" / f"{stem}.svg").read_text(encoding="utf-8")
        assert battlements in drawing and head in drawing, stem
        assert stem in fetch.DERIVED
        entry = manifest[stem]
        assert (entry["author"], entry["licence"]) == ("Cburnett", "CC BY-SA 3.0")
        assert entry["modified"].startswith("drawn for this repository from Cburnett's")
        assert entry["bytes"] == len(drawing.encode("utf-8"))
    assert icon_for("chancellor", 0)["href"] != icon_for("knight", 0)["href"]


def test_the_archbishop_is_a_mitre_with_a_knights_head() -> None:
    """The archbishop (bishop + knight) is drawn mitre first, as the chancellor is tower first.

    Commons' compound is a whole knight standing on a bishop's foot, the chancellor's
    mistake again (first-visit study N2). The library's own drawing is Cburnett's bishop,
    at nine tenths, with his knight's head rising from behind the mitre's point, so both
    parts are in the file: the mitre with its slit and the knight's head. It is recorded in
    `fetch.py` as drawn here, so a rebuild keeps it rather than fetching Commons' one again,
    and it is credited to the artist whose parts it is made of, under his licence.
    """
    from importlib.util import module_from_spec, spec_from_file_location

    from chess_variant_sandbox.viz import pieces

    if icon_for("archbishop", 0) is None:  # the library is optional
        pytest.skip("icon library not fetched")
    found = spec_from_file_location("fetch", pieces.ASSET_ROOT / "fetch.py")
    assert found is not None and found.loader is not None
    fetch = module_from_spec(found)
    found.loader.exec_module(fetch)
    assert "archbishop" not in fetch.CBURNETT

    mitre = "C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26"
    slit = "M 22.5,15.5 L 22.5,20.5 M 20,18 L 25,18"
    head = "C 14.5,6.5 16.5,10 16.5,10 L 18.5,10"
    manifest = json.loads((pieces.ASSET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    for colour in ("white", "black"):
        stem = f"archbishop-{colour}"
        drawing = (pieces.ASSET_ROOT / "svg" / f"{stem}.svg").read_text(encoding="utf-8")
        assert mitre in drawing and slit in drawing and head in drawing, stem
        # the head is the smaller part, at the chancellor's 0.6; the bishop near whole
        assert "matrix(0.6,0,0,0.6," in drawing and "matrix(0.9,0,0,0.9," in drawing, stem
        assert stem in fetch.DERIVED
        entry = manifest[stem]
        assert (entry["author"], entry["licence"]) == ("Cburnett", "CC BY-SA 3.0")
        assert entry["modified"].startswith("drawn for this repository from Cburnett's")
        assert "bishop" in entry["modified"]
        assert entry["bytes"] == len(drawing.encode("utf-8"))
    credits = (pieces.ASSET_ROOT / "CREDITS.md").read_text(encoding="utf-8")
    assert "| `archbishop-white.svg` | Cburnett |" in credits
    hrefs = {name: icon_for(name, 0)["href"] for name in ("archbishop", "knight", "bishop", "chancellor")}
    assert len(set(hrefs.values())) == 4, hrefs


class TestTheReplayDrawsAWildPieceAsTheBoardsDo:
    """PI4. PI3 put a wild piece's own drawing on the third owner's disc on every page board;
    the offline replay still lettered it. Same three types as
    `test_web.py::TestAWildPieceWearsItsDrawing`: 0 has a drawing, 1 has none, 2 is the duck.
    """

    ICONS = "[[{href:'wolf0'},{href:'wolf1'}],[null,null],[{href:'duck0'},{href:'duck1'}]]"

    def _drawn(self, player: int, kind: int) -> str:
        replay = Path(__file__).parents[1] / "chess_variant_sandbox" / "viz" / "replay.js"
        source = replay.read_text(encoding="utf-8")
        start = source.index("function iconFor")
        end = source.index("// One square, wherever in the grid")
        head = (
            "const DATA = { nTypes: 3, pieceIds: ['W','Z','U'], duck: 2,"
            f" pieceIcons: {self.ICONS} }}; const CELL = 10, FRAME = {{}};"
            "function portalGlyph(){} function portalPairs(){} function portalAimFor(){}"
            "function portalPartner(){} function portalSquareIn(){}"
            "let BEATS = null, at = 0;"
        )
        call = f"drawPiece({{ player: {player}, type: {kind}, facing: 0 }}, 0, 0, false)"
        done = node_session.run(
            head + source[start:end] + f"\nconsole.log(JSON.stringify({call}));"
        )
        assert done.returncode == 0, done.stderr
        return str(json.loads(done.stdout))

    def test_a_wild_piece_with_a_drawing_wears_the_first_seats_on_the_third_disc(self) -> None:
        drawn = self._drawn(2, 0)
        assert 'href="wolf0"' in drawn and "wolf1" not in drawn
        assert 'data-wild="1"' in drawn and "var(--p2)" in drawn and "<text" not in drawn

    def test_a_wild_piece_with_no_drawing_is_still_the_lettered_disc(self) -> None:
        drawn = self._drawn(2, 1)
        assert "<image" not in drawn and 'data-wild="1"' not in drawn
        assert ">Z</text>" in drawn and "var(--p2)" in drawn

    def test_the_duck_stands_bare(self) -> None:
        drawn = self._drawn(2, 2)
        assert 'href="duck0"' in drawn and 'data-wild="1"' not in drawn and "<text" not in drawn

    def test_a_players_piece_is_untouched(self) -> None:
        drawn = self._drawn(1, 0)
        assert 'href="wolf1"' in drawn and 'data-wild="1"' not in drawn


# ------------------------------------------------------- the purse in the offline replay


# The replay's own script in node, on a stub of the few elements it touches: it steps to
# each beat in `ORDER` and prints what each seat's purse line says there.
_PURSE_DOM = r"""
const made = {};
const element = id => made[id] || (made[id] = {
  id, innerHTML: '', textContent: '', hidden: false, disabled: false, dataset: {},
  classList: { toggle() {} }, setAttribute() {}, addEventListener() {},
  querySelectorAll: () => [], appendChild() {}, onclick: null,
});
const seats = [0, 1].map(p => ({ dataset: { purse: String(p) }, textContent: '' }));
global.window = { matchMedia: () => ({ matches: true }), localStorage: null };
global.document = {
  title: '', getElementById: element, addEventListener() {},
  querySelectorAll: sel => (sel === '[data-purse]' ? seats : []),
};
"""


class TestTheReplayDrawsThePurse:
    """Q1049: the offline replay shows each seat's purse at the position on screen, worded as
    the play board words it, and the counts are the core's replayed state, never the page's
    arithmetic (CLAUDE.md rule 3)."""

    HERE = Path(__file__).parent

    def _game(self) -> tuple[dict, object]:
        from chess_variant_sandbox.tournament.play import play_game

        spec = json.loads((self.HERE / "_purse_market.json").read_text(encoding="utf-8"))
        # Greedy against greedy buys three times in thirty-odd plies and ends in a mate.
        return spec, play_game(Engine(spec), ("greedy", "greedy"), 0)

    def test_each_ply_carries_the_cores_purses(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.engine.moves import MOVE_BUY, move_type
        from chess_variant_sandbox.tournament.play import replay
        from chess_variant_sandbox.viz.html import render_game

        spec, record = self._game()
        engine = Engine(spec)
        data = _payload(render_game(spec, record, tmp_path / "purse.html"))
        start = spec["rules"]["purse"]["start"]
        assert data["purses"][0] == [start, start]
        assert data["purses"] == [list(s.purse) for s in replay(engine, record)]
        assert data["sides"] == ["white", "black"]
        # The oracle for a buy: the buyer's purse falls by the market's price, the other's holds.
        price = dict(zip(engine.spec.purse.market, engine.spec.purse.price, strict=True))
        states = replay(engine, record)
        buys = [i for i, m in enumerate(record.moves) if move_type(m) == MOVE_BUY]
        assert buys, "the game this test reads buys something"
        for i in buys:
            mover = states[i].side_to_move
            bought = engine.spec.type_of(states[i + 1].board[(record.moves[i] >> 8) & 0xFF])
            assert data["purses"][i + 1][mover] == data["purses"][i][mover] - price[bought]
            assert data["purses"][i + 1][1 - mover] == data["purses"][i][1 - mover]

    def test_a_game_without_a_purse_carries_none(
        self, chess_spec: dict, tmp_path: Path
    ) -> None:
        data = _payload(
            render_position(chess_spec, Engine(chess_spec).initial_state(), tmp_path / "c.html")
        )
        assert "purses" not in data and data["purse"] is None

    def test_the_seats_say_the_purse_at_every_ply_forward_and_back(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.viz.html import render_game

        spec, record = self._game()
        text = render_game(spec, record, tmp_path / "purse.html").read_text(encoding="utf-8")
        script = text[text.index("<script>") + len("<script>"): text.rindex("</script>")]
        data = _payload(tmp_path / "purse.html")
        n = len(data["purses"])
        # Forward through every ply, then back, then a jump: each step shows its own ply.
        order = list(range(n)) + list(range(n - 1, -1, -1)) + [n - 1, 0]
        probe = (
            "\nconst seen = [];\n"
            f"for (const at of {json.dumps(order)}) {{\n"
            "  goto(BEATS.findIndex(b => b.ply === at));\n"
            "  seen.push(seats.map(s => s.textContent));\n"
            "}\n"
            "console.log(JSON.stringify({ seen, seats: made.seats.innerHTML,"
            " hidden: made.seats.hidden }));\n"
        )
        done = node_session.run(_PURSE_DOM + script + probe)
        assert done.returncode == 0, done.stderr
        out = json.loads(done.stdout.strip().splitlines()[-1])

        def words(k: int) -> str:
            return f"purse: {k} coin{'' if k == 1 else 's'}"

        assert out["seen"] == [[words(c) for c in data["purses"][at]] for at in order]
        assert not out["hidden"]
        assert "<b>white</b>" in out["seats"] and "<b>black</b>" in out["seats"]
        assert 'data-purse="0"' in out["seats"] and "buy or hire" in out["seats"]
