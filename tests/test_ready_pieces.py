"""The ready-made pieces (PF): `assets/pieces/ready.json`, the designer's pieces picked whole.

Every drawing in the library is either a ready piece or left out with a reason. Each ready
piece compiles, alone and beside the game-wide rule it says it needs, and its moves are the
reading `docs/plan-pieces-moves.md` §1 gives its drawing, in the spelling the survey's
scratch specs proved with `validate` (§1.4's `gen.py`), or the committed game's own moves for
the shogi and xiangqi drawings. A piece named here is a fixture of the catalogue's, not of
the engine's: nothing under `engine/` reads this file.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from page_text import PAGE

from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.web import piece_library

ROOT = Path(__file__).resolve().parents[1]
PIECES = ROOT / "assets" / "pieces"
READY = json.loads((PIECES / "ready.json").read_text(encoding="utf-8"))
CATALOGUE = json.loads((PIECES / "catalogue.json").read_text(encoding="utf-8"))

# The keys an entry may hold: the piece's own (what travels into a draft) and what is said.
_PIECE_KEYS = {"name", "value", "royal", "moves", "rest", "sight", "facing"}
_ENTRY_KEYS = _PIECE_KEYS | {"id", "reading", "description", "lattice", "rules", "rules_say",
                             "note"}

# PF3: a description is one or two sentences of 60 to 200 characters, short enough to read
# at a glance on a phone beside the grid, long enough to say what, how and from where.
_DESCRIPTION_CHARS = (60, 200)
# Words a visitor is not asked to know (plain-words.md); a piece's own name may still hold one,
# and an inventor's name (Ralph Betza) is a name, not his notation.
_JARGON = ("xbetza", "leaper", "rider", "orthogonal", "royal", "compound", "hopper", "rules.",
           "@", "spec", "json", "lattice", "vector", "symmetry")


# ---------------------------------------------------------------- the survey's spellings
# `gen.py`'s helpers, as §1.4 says to rebuild them.

def leap(v: list[list[int]], sym: str = "rotate4", **kw: Any) -> dict[str, Any]:
    return {"kind": "leap", "vectors": v, "symmetry": sym, **kw}


def slide(v: list[list[int]], sym: str = "rotate4", **kw: Any) -> dict[str, Any]:
    return {"kind": "slide", "vectors": v, "symmetry": sym, **kw}


W = leap([[1, 0]]); F = leap([[1, 1]]); D = leap([[2, 0]]); A = leap([[2, 2]])  # noqa: E702
H = leap([[3, 0]]); G = leap([[3, 3]]); N = leap([[1, 2]], "dihedral8")  # noqa: E702
C = leap([[1, 3]], "dihedral8"); Z = leap([[2, 3]], "dihedral8")  # noqa: E702
GI = leap([[1, 4]], "dihedral8"); AN = leap([[3, 4]], "dihedral8")  # noqa: E702
K = leap([[1, 0], [1, 1]]); R = slide([[1, 0]]); B = slide([[1, 1]])  # noqa: E702
Q = slide([[1, 0], [1, 1]]); NN = slide([[1, 2]], "dihedral8")  # noqa: E702
PAWN = [slide([[0, 1]], "none", range=1, mode="move"),
        slide([[0, 1]], "none", range=2, min_range=2, mode="move", condition={"from_rank": 1}),
        leap([[1, 1]], "lateral", mode="capture")]
GH = {"kind": "hop", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "screen_range": 1}
KING_STEPS = [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"},
              {"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4", "mode": "both"}]

# §1's table, drawing by drawing. The new rules' rows are §4.2 (`moves: []`), §5.2 (the
# three paths) and §6.2 (the arrow, the archers' bishop of §6.7).
SURVEY: dict[str, list[dict[str, Any]]] = {
    "alfil": [A], "antelope": [AN], "bison": [C, Z], "camel": [C], "dabbaba": [D],
    "dragon": [N] + PAWN, "fad": [F, A, D], "ferz": [F], "frog": [F, H], "giraffe": [GI],
    "kirin": [F, D], "knight": [N], "phoenix": [W, A], "squirrel": [A, D, N],
    "threeleaper": [H], "tripper": [G], "waffle": [W, A], "wazir": [W], "zebra": [Z],
    "snowflake": [slide([[1, 0]], "rotate6")],
    "arrow": [B], "bishop": [B], "grasshopper": [GH], "inverted-queen": [GH],
    "nightrider": [NN], "queen": [Q], "rook": [R], "saltire-rider": [B], "ship": [A],
    "short-rook": [slide([[1, 0]], range=4)], "spear": [slide([[0, 1]], "none")],
    "trident": [slide([[0, 1], [1, 1]], "lateral")], "unicorn": [B, NN],
    "rose": [{"kind": "slide", "vectors": [[1, 2]], "symmetry": "dihedral8", "path": "circle"}],
    "ring-rider": [{"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4",
                    "path": "circle"}],
    "bolt": [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4", "path": "zigzag"}],
    "amazon": [Q, N], "archbishop": [B, N], "centaur": [K, N], "champion": [W, A, D],
    "chancellor": [R, N], "crowned-bishop": [B, W], "crowned-rook": [R, F], "moon": [F, C],
    "queen-star": [Q, N], "rook-diamond": [R, A],
    "commoner": [K], "emperor": [K], "inverted-king": [K], "king": [K], "man": [K],
    "prince": [K], "royal-knight": [N], "shield-king": [K],
    "pawn": PAWN,
    # §1: the xiangqi soldier; its sideways step needs the river's squares, left to the
    # designer and said in the note (`test_a_traditional_piece_moves_as_its_committed_game`).
    "soldier": [leap([[0, 1]], "none")],
    "sergeant": [leap([[0, 1]], "none"), leap([[1, 1]], "lateral"),
                 slide([[0, 1]], "none", range=2, min_range=2, mode="move",
                       condition={"from_rank": 1})],
    "anchor": [], "bomb": [], "crown": [], "eye": [K],
    "ghost": [leap([[0, d] for d in range(1, 8)] + [[d, d] for d in range(1, 8)])],
    "hourglass": [Q], "rock": KING_STEPS, "paper": KING_STEPS, "scissors": KING_STEPS,
}
ROYAL = {"king", "emperor", "inverted-king", "prince", "royal-knight", "shield-king", "crown"}

# §1.3's hand counts: the piece on d3, kings on a1 and h8, an enemy copy on b8.
HAND_COUNTS = {
    "alfil": 4, "camel": 6, "zebra": 5, "giraffe": 4, "antelope": 3, "threeleaper": 3,
    "tripper": 2, "kirin": 8, "squirrel": 16, "bison": 11, "fad": 12, "champion": 12,
    "moon": 10, "rook": 14, "short-rook": 13, "nightrider": 12, "unicorn": 23, "amazon": 33,
    "crowned-rook": 18, "rook-diamond": 18, "ghost": 25,
}


def _committed(game: str) -> dict[str, dict[str, Any]]:
    """The committed game's pieces by the drawing each wears."""
    spec = json.loads((ROOT / "rulesets" / f"{game}.json").read_text(encoding="utf-8"))
    icons = spec["appearance"]["icons"]
    found = {icons[piece["id"]]: piece for piece in spec["pieces"]}
    if game == "xiangqi":
        found.update({slug.replace("pictorial-", ""): piece for slug, piece in list(found.items())})
    return found


TRADITIONAL = {**_committed("shogi"), **_committed("xiangqi")}
# §1: "yes (committed)", berolina.json's pawn.
BEROLINA = [p for p in json.loads((ROOT / "rulesets" / "berolina.json").read_text(
    encoding="utf-8"))["pieces"] if p["id"] == "P"][0]["moves"]
SURVEY_COMMITTED = {"berolina": BEROLINA}
# PF2: the pointing pawn is this project's own, drawn after the survey, so no section-1 row holds it:
# archive/weathervane.json's pawn does (name, value, moves and facing, exactly as spelled there).
WEATHERVANE_PAWN = json.loads((ROOT / "archive" / "weathervane.json").read_text(
    encoding="utf-8"))["pieces"][0]
OWN = {"facing-pawn": WEATHERVANE_PAWN}


def _resolve(value: Any, self_id: str, royals: list[str]) -> Any:
    """The designer's substitution: '@self' the piece's letter, '@royals' the royal letters."""
    if value == "@self":
        return self_id
    if value == "@royals":
        return list(royals)
    if isinstance(value, list):
        return [_resolve(item, self_id, royals) for item in value]
    if isinstance(value, dict):
        return {key: _resolve(item, self_id, royals) for key, item in value.items()}
    return value


def _spec(slug: str, with_rules: bool = True) -> dict[str, Any]:
    """The survey's scratch position: the piece on d3 and b8, royal kings on a1 and h8."""
    entry = READY["pieces"][slug]
    piece = {key: copy.deepcopy(entry[key]) for key in _PIECE_KEYS if key in entry}
    piece["id"] = "X"
    hexy = entry.get("lattice") == "hex"
    king = {"id": "K", "name": "king", "royal": True, "value": 0,
            "moves": [leap([[1, 0]], "rotate6")] if hexy else [K]}
    pieces = [piece]
    placements = [{"piece": "X", "player": 0, "squares": ["d3"]},
                  {"piece": "X", "player": 1, "squares": ["a7" if hexy else "b8"]}]
    if not piece.get("royal"):
        pieces.append(king)
        placements += [{"piece": "K", "player": 0, "squares": ["a1"]},
                       {"piece": "K", "player": 1, "squares": ["g7" if hexy else "h8"]}]
    if piece.get("royal") and not piece["moves"]:
        # A side whose every piece never moves can never move (§4.3, refusal 1).
        pieces.append({"id": "M", "name": "man", "value": 3, "moves": [K]})
        placements += [{"piece": "M", "player": 0, "squares": ["a1"]},
                       {"piece": "M", "player": 1, "squares": ["h8"]}]
    royals = [p["id"] for p in pieces if p.get("royal")]
    rules: dict[str, Any] = {"win": "checkmate", "max_plies": 200}
    if len(royals) > 1 or (piece.get("royal") and not piece["moves"]):
        rules["win"] = "royal_capture"
    if with_rules:
        rules.update(_resolve(entry.get("rules", {}), "X", royals))
    board: dict[str, Any] = {"width": 7, "height": 7, "lattice": "hex"} if hexy else {
        "width": 8, "height": 8}
    setup: dict[str, Any] = {"mirror": False, "placements": placements}
    if hexy:
        setup["orientation"] = "rotate"
    return {"name": f"ready-{slug}", "board": board, "pieces": pieces, "setup": setup,
            "rules": rules}


class TestTheCatalogue:
    def test_every_drawing_is_ready_or_left_out_with_a_reason(self) -> None:
        ready, left = set(READY["pieces"]), set(READY["left_out"])
        assert ready | left == set(CATALOGUE["pieces"])
        assert not ready & left
        assert all(isinstance(why, str) and why for why in READY["left_out"].values())

    def test_each_entry_holds_only_a_piece_and_its_words(self) -> None:
        for slug, entry in READY["pieces"].items():
            assert set(entry) <= _ENTRY_KEYS, (slug, set(entry) - _ENTRY_KEYS)
            assert entry["name"] and entry["reading"] and isinstance(entry["moves"], list)
            assert len(entry["id"]) == 1 and entry["id"].isupper(), slug
            # A ready piece names no square (a board's) and no other piece (a game's).
            text = json.dumps({k: v for k, v in entry.items() if k in _PIECE_KEYS})
            assert '"region"' not in text and '"captures"' not in text, slug
            assert "promoted_from" not in text, slug

    def test_a_rule_it_needs_is_said_in_words(self) -> None:
        for slug, entry in READY["pieces"].items():
            assert ("rules" in entry) == ("rules_say" in entry), slug

    @pytest.mark.parametrize("slug", sorted(SURVEY) + sorted(SURVEY_COMMITTED))
    def test_its_moves_are_the_surveys(self, slug: str) -> None:
        entry = READY["pieces"][slug]
        assert entry["moves"] == {**SURVEY, **SURVEY_COMMITTED}[slug]
        assert bool(entry.get("royal")) == (slug in ROYAL)

    @pytest.mark.parametrize("slug", sorted(OWN))
    def test_an_own_piece_is_its_games_spelling(self, slug: str) -> None:
        """The project's own drawings: the piece as the game that plays it spells it."""
        entry, piece = READY["pieces"][slug], OWN[slug]
        for key in ("id", "name", "value", "moves", "facing"):
            assert entry[key] == piece[key], (slug, key)
        assert bool(entry.get("royal")) == bool(piece.get("royal"))
        assert "weathervane" in entry["reading"] and "own" in entry["reading"]
        assert "rules" not in entry  # weathervane's pawn needs no rule of the whole game

    @pytest.mark.parametrize("slug", sorted(TRADITIONAL))
    def test_a_traditional_piece_moves_as_its_committed_game(self, slug: str) -> None:
        """The shogi and xiangqi drawings: the committed game's moves, less a square."""
        piece = TRADITIONAL[slug]
        want = [m for m in piece["moves"] if "region" not in (m.get("condition") or {})]
        entry = READY["pieces"][slug]
        assert entry["moves"] == want
        assert bool(entry.get("royal")) == bool(piece.get("royal"))
        if len(want) < len(piece["moves"]) or piece.get("region"):
            assert "note" in entry, f"{slug} leaves squares out without saying so"

    def test_the_survey_and_the_committed_games_cover_every_ready_piece(self) -> None:
        assert set(READY["pieces"]) == (set(SURVEY) | set(SURVEY_COMMITTED) | set(TRADITIONAL)
                                        | set(OWN))
        assert "note" in READY["pieces"]["soldier"]


class TestEachIsDescribed:
    """PF3: every ready piece says in a visitor's words what it is, how it moves and whence."""

    @pytest.mark.parametrize("slug", sorted(READY["pieces"]))
    def test_it_has_a_short_plain_description(self, slug: str) -> None:
        import re

        from site_words import hits

        entry = READY["pieces"][slug]
        said = entry.get("description", "")
        low, high = _DESCRIPTION_CHARS
        assert low <= len(said) <= high, (slug, len(said))
        assert said[0].isupper() and said.endswith("."), slug
        assert 1 <= len(re.findall(r"[.!?](?:\s|$)", said)) <= 2, slug
        own = entry["name"].lower()
        for word in _JARGON:
            assert word not in said.lower().replace(own, ""), (slug, word)
        assert hits(said) == [], slug

    @pytest.mark.parametrize("slug", sorted(READY["pieces"]))
    def test_its_words_agree_with_its_moves(self, slug: str) -> None:
        entry = READY["pieces"][slug]
        said = entry["description"].lower()
        assert ("never moves" in said) == (entry["moves"] == []), slug
        assert ("not losing it" in said) == bool(entry.get("royal")), slug
        if entry.get("rest"):
            assert "rest" in said, slug
        if entry.get("rules"):
            assert "rule on" in said, slug  # the rule it needs is a condition, said as one


class TestEachCompiles:
    @pytest.mark.parametrize("slug", sorted(READY["pieces"]))
    def test_it_validates_and_compiles(self, slug: str) -> None:
        spec = _spec(slug)
        assert validate_spec(spec) == [], slug
        compile_spec(spec)

    def test_the_arrow_needs_the_strike_and_the_bomb_the_explosion(self) -> None:
        """§6.2 and §4.3: the two pieces whose reading is half a game-wide rule."""
        arrow, bomb = READY["pieces"]["arrow"], READY["pieces"]["bomb"]
        assert arrow["rules"] == {"strike": {"effect": "capture", "position": "stay",
                                             "immune": "@royals", "by": ["@self"]}}
        assert bomb["rules"] == {"explode_on_capture": {"radius": 0, "of": ["@self"]}}
        assert _spec("arrow")["rules"]["strike"]["by"] == ["X"]
        # Without its rule the arrow is a bishop: a different game, so a different name.
        from chess_variant_sandbox.spec import spec_hash
        assert spec_hash(_spec("arrow")) != spec_hash(_spec("arrow", with_rules=False))

    @pytest.mark.parametrize("slug", sorted(HAND_COUNTS))
    def test_the_engine_gives_the_surveys_hand_count(self, slug: str) -> None:
        engine = Engine(_spec(slug))
        state = engine.initial_state()
        names = [engine.describe_move(state, m) for m in engine.legal_moves(state)]
        assert len([n for n in names if n.startswith("Xd3")]) == HAND_COUNTS[slug]


class TestTheDesignerOffersThem:
    def test_the_palette_carries_each_ready_piece_on_its_drawing(self) -> None:
        offered = {row["slug"]: row for row in piece_library.palette()["pieces"]}
        for slug, entry in READY["pieces"].items():
            ready = offered[slug]["ready"]
            assert ready["piece"]["moves"] == entry["moves"]
            assert ready["piece"]["name"] == entry["name"]
            assert ready["piece"].get("facing") == entry.get("facing")
            assert len(ready["says"]) == len(entry["moves"])
            assert ready["lattice"] == entry.get("lattice", "square")
            assert ready["description"] == entry["description"]
        for slug in READY["left_out"]:
            assert offered[slug].get("ready") is None

    def test_the_page_picks_one_whole_and_says_what_it_needs(self) -> None:
        for needle in ("data-readypiece", "function takeReadyPiece", "data-readyrules",
                       "function readyNeedsHtml", "'@self'", "'@royals'",
                       "data-readyabout", "function placeReadyAbout", "ready.description"):
            assert needle in PAGE, needle
