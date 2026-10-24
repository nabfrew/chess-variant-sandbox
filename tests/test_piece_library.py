"""The piece library (plan-visual.md §7, V9): the grammar, the sort, the legibility, the
licences, the catalogue and the designer's palette.

The drawings are display data and nothing else: a picture never changes a game's rules or
its identity, so the last class holds a spec's hash still when the drawing it names moves.
The rendering checks read the files through resvg (the `site` extra) and are skipped where
it is not installed, as `assets/pieces/check.py` says it cannot check rather than passing.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
from page_text import PAGE

from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.viz.pieces import icon_for
from chess_variant_sandbox.web import piece_library

PIECES = Path(__file__).resolve().parents[1] / "assets" / "pieces"
MANIFEST = json.loads((PIECES / "manifest.json").read_text(encoding="utf-8"))
CATALOGUE = json.loads((PIECES / "catalogue.json").read_text(encoding="utf-8"))

# The threshold the sort computes, the familiar set's worst pair (bishop and pawn), pinned so
# a change to the measure or to a standard file is seen rather than absorbed.
THRESHOLD = 0.947


def _module(name: str) -> ModuleType:
    if str(PIECES) not in sys.path:
        sys.path.insert(0, str(PIECES))
    spec = importlib.util.spec_from_file_location(name, PIECES / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


draw = _module("draw")
check = _module("check")
catalogue = _module("catalogue")


def _rendering() -> None:
    pytest.importorskip("resvg_py")


class TestTheGrammar:
    def test_the_files_are_what_draw_py_writes(self) -> None:
        """The drawings are written by `draw.py`, never edited by hand."""
        for slug in draw.DRAWN:
            for seat in draw.SEATS:
                on_disk = (PIECES / "svg" / f"{slug}-{seat}.svg").read_text(encoding="utf-8")
                assert on_disk == draw.svg(slug, seat), f"{slug}-{seat}: run draw.py"

    def test_about_sixty_drawings(self) -> None:
        assert len(draw.DRAWN) >= 55
        assert {d.category for d in draw.DRAWN.values()} <= set(draw.CATEGORIES)

    def test_every_drawing_keeps_cburnetts_grammar(self) -> None:
        """A 45 x 45 frame, white or black fill, a black outline, and one stroke width:
        a grown head's stroke is divided by its scale, so it draws 1.5 like the rest."""
        for slug in draw.DRAWN:
            for seat, (fill, detail) in draw.SEATS.items():
                text = draw.svg(slug, seat)
                assert 'viewBox="0 0 45 45"' in text
                colours = set(re.findall(r'(?:fill|stroke)="(#[0-9a-f]{6}|none)"', text))
                assert colours <= {fill, detail, draw.INK, "none"}, (slug, seat, colours)
                for path in re.findall(r"<path [^>]*/>", text):
                    width = float(re.search(r'stroke-width="([\d.]+)"', path).group(1))
                    found = re.search(r"scale\(([\d.]+)\)", path)
                    scale = float(found.group(1)) if found else 1.0
                    assert abs(width * scale - draw.STROKE) < 0.02, (slug, path[:60])

    def test_every_drawing_stands_on_the_one_base(self) -> None:
        for slug in draw.DRAWN:
            assert draw.marks(draw.DRAWN[slug])[0] == draw.BASE[0], slug


class TestTheLicences:
    def test_each_drawing_is_credited_by_what_it_is_made_of(self) -> None:
        """Original work is CC0; a drawing on Cburnett's knight head keeps his CC BY-SA 3.0
        and names him, in the manifest and in the credits `fetch.py` writes."""
        credits = (PIECES / "CREDITS.md").read_text(encoding="utf-8")
        for slug, drawing in draw.DRAWN.items():
            for seat in draw.SEATS:
                row = MANIFEST[f"{slug}-{seat}"]
                if drawing.cburnett:
                    assert (row["author"], row["licence"]) == ("Cburnett", "CC BY-SA 3.0")
                    assert "Cburnett's knight" in row["modified"]
                else:
                    assert row["licence"] == "CC0", slug
                    assert "Cburnett's grammar" in row["modified"]
                assert f"`{slug}-{seat}.svg`" in credits
                assert row["bytes"] == (PIECES / row["file"]).stat().st_size

    def test_the_file_says_its_licence_too(self) -> None:
        for slug, drawing in draw.DRAWN.items():
            text = draw.svg(slug, "white")
            assert ("CC BY-SA 3.0" in text) == drawing.cburnett
            assert ("released CC0" in text) != drawing.cburnett

    def test_the_western_family_is_one_grammar(self) -> None:
        """Every western file is one of Cburnett's set, drawn from his pieces, or drawn here
        in his grammar. The fairy fonts' alfil, camel and wazir (Quivira's glyphs) were the
        last second style, and are drawn in the grammar now (Q557b)."""
        for slug in ("alfil", "camel", "wazir"):
            assert slug in draw.DRAWN and not draw.DRAWN[slug].cburnett, slug
        for stem, row in MANIFEST.items():
            if stem.startswith(check.OWN_FAMILIES):
                continue
            source = row.get("source", "")
            assert not source or re.search(r"/File:Chess_\w+t45\.svg$", source), stem

    def test_every_licence_is_free(self) -> None:
        allowed = {"CC0", "CC BY-SA 3.0", "CC BY-SA 4.0", "Public domain", "GFDL"}
        assert {row["licence"] for row in MANIFEST.values()} <= allowed


class TestTheCatalogue:
    def test_it_is_what_catalogue_py_writes(self) -> None:
        assert catalogue.build(MANIFEST) == CATALOGUE, "run catalogue.py"

    def test_every_drawing_is_found_in_one_category(self) -> None:
        assert set(CATALOGUE["pieces"]) == set(catalogue.slugs(MANIFEST))
        assert CATALOGUE["categories"] == ["leapers", "riders", "compounds", "royals",
                                           "pawns", "oddities", "shogi", "xiangqi"]
        for slug, entry in CATALOGUE["pieces"].items():
            assert entry["category"] in CATALOGUE["categories"], slug
            assert entry["name"] and isinstance(entry["words"], list), slug

    def test_the_families_stay_their_own(self) -> None:
        for slug, entry in CATALOGUE["pieces"].items():
            family = slug.split("-")[0]
            if family in ("shogi", "xiangqi"):
                assert entry["category"] == family, slug
            else:
                assert entry["category"] not in ("shogi", "xiangqi"), slug


class TestTheSort:
    """Told apart by silhouette at 24 px (S7), the rule fixed in plan-visual.md §7."""

    def test_the_threshold_is_the_familiar_sets_worst_pair(self) -> None:
        _rendering()
        masks = {s + "-white": check.silhouette(s + "-white", MANIFEST) for s in check.STANDARD}
        assert round(check.threshold(masks), 3) == THRESHOLD

    def test_no_two_western_pieces_are_alike(self) -> None:
        _rendering()
        limit, pairs = check.sort_test(MANIFEST)
        assert round(limit, 3) == THRESHOLD
        assert not pairs, pairs[:5]

    def test_the_chancellor_is_told_from_the_knight(self) -> None:
        """The first pair the sort must separate (plan-visual.md §7)."""
        _rendering()
        chancellor = check.silhouette("chancellor-white", MANIFEST)
        knight = check.silhouette("knight-white", MANIFEST)
        assert check.overlap(chancellor, knight) < THRESHOLD - 0.2

    def test_the_sort_covers_the_whole_western_library(self) -> None:
        western = check.pieces(MANIFEST)
        assert len(western) == len([s for s in CATALOGUE["pieces"]
                                    if not s.startswith(check.OWN_FAMILIES)])

    def test_a_copy_reads_as_one(self) -> None:
        """The measure itself: a silhouette overlaps itself fully and the empty one not."""
        _rendering()
        knight = check.silhouette("knight-white", MANIFEST)
        assert check.overlap(knight, knight) == 1.0
        assert check.overlap(knight, 0) == 0.0


class TestTheDrawingsAreLegible:
    def test_every_black_piece_reads_on_the_worst_dark_square(self) -> None:
        """Black pieces at least 4.5 : 1 on the dark square under its worst streak, in
        both themes (S8 holds the board's side; this holds the pieces')."""
        _rendering()
        rows = check.contrast(MANIFEST)
        assert len(rows) == 2 * len([s for s in MANIFEST if s.endswith("-black")
                                     and not s.startswith(check.OWN_FAMILIES)])
        low = [row for row in rows if row[0] < check.BODY_FLOOR]
        assert not low, low

    def test_every_drawing_draws_inside_its_frame(self) -> None:
        """Nothing a drawing paints touches the frame's edge at 45 px."""
        _rendering()
        for slug in draw.DRAWN:
            rows = check.render(slug + "-white", 45, MANIFEST)
            edge = [rows[y][x * 4 + 3] for y in (0, 44) for x in range(45)]
            edge += [rows[y][x * 4 + 3] for y in range(45) for x in (0, 44)]
            assert not any(edge), slug

    def test_the_contact_sheet_shows_both_squares_in_both_themes(self) -> None:
        from chess_variant_sandbox.web.page import tokens

        sheet = (PIECES / "index.html").read_text(encoding="utf-8")
        dark = {**tokens.LIGHT, **tokens.DARK}
        for colour in (tokens.LIGHT["sq-lt"], tokens.LIGHT["sq-dk"], dark["sq-lt"],
                       dark["sq-dk"]):
            assert colour in sheet
        assert sheet.count("<figure>") == len(CATALOGUE["pieces"])
        for cls in ("light-lt", "light-dk", "dark-lt", "dark-dk"):
            assert sheet.count(f'class="{cls}"') == 2 * len(CATALOGUE["pieces"])


class TestThePalette:
    def test_it_offers_every_drawing_with_its_picture(self) -> None:
        offered = piece_library.palette()
        assert [c["key"] for c in offered["categories"]] == CATALOGUE["categories"]
        assert {p["slug"] for p in offered["pieces"]} == set(CATALOGUE["pieces"])
        for piece in offered["pieces"]:
            assert "<svg" in piece["svg"] and "<script" not in piece["svg"]
            assert icon_for(piece["slug"], 0) is not None, piece["slug"]

    def test_it_is_a_part_of_the_design_kit_fetched_apart(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.web.server import App

        for name in ("rulesets", "runs", "archive"):
            (tmp_path / name).mkdir()
        app = App(tmp_path / "rulesets", tmp_path / "runs", archive_dir=tmp_path / "archive")
        assert app.get("/api/design", {"part": "pieces"}) == {
            "pieceLibrary": piece_library.palette()}
        assert "pieceLibrary" not in app.get("/api/design")

    def test_the_page_searches_sorts_and_writes_a_slug(self) -> None:
        for needle in ("/api/design?part=pieces", "data-drawfor", "data-drawkind",
                       "data-drawing", 'id="dDrawFind"', "function wearDrawing",
                       "DRAFT.appearance.icons[piece.id] = iconSeats(index) ? [slug, slug] : slug"):
            assert needle in PAGE, needle


class TestAPictureIsNotARule:
    def test_naming_a_new_drawing_moves_no_hash(self) -> None:
        """A game is its mechanics: dressing a piece in any drawing of the library leaves
        the spec's hash where it was."""
        spec = json.loads((Path(__file__).resolve().parents[1] / "rulesets" / "chess.json")
                          .read_text(encoding="utf-8"))
        before = spec_hash(spec)
        for slug in ("kirin", "duck", "zebra", "shogi-gold"):
            dressed = copy.deepcopy(spec)
            dressed.setdefault("appearance", {}).setdefault("icons", {})["N"] = slug
            assert spec_hash(dressed) == before, slug
