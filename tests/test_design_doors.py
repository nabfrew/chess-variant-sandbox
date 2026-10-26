"""A door per rule into the designer (LP7, `docs/plan-landing.md` §5).

`#/design?rule=<key>` opens the designer on a copy of chess with that rule added, through the
designer's own add paths. These hold the route together from its three sides: every word of
the landing's wall and every story's card links a key that `designing.doors` resolves to a
card the designer draws (the server's half), the kit carries the doors and the copy they
start from (the route), and the page reads the key off the address and takes the door, or
says it has none (the page's half, read as text). That each door opens with its card present
and the draft checked is the browser's, LP7's acceptance.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import unquote

import pytest
from page_text import PAGE

from chess_variant_sandbox.spec.canonical import load_spec, spec_hash
from chess_variant_sandbox.spec.schema import MOVE_KINDS
from chess_variant_sandbox.web import cards, designing, landing, stories
from chess_variant_sandbox.web.server import App

ROOT = Path(__file__).resolve().parents[1]
CHESS = load_spec(ROOT / "rulesets" / "chess.json")
MODEL = cards.card_model()
DOORS = designing.doors(MODEL, {"hash": spec_hash(CHESS), "name": "chess", "spec": CHESS})
WORDS = [(block, word) for block in landing.BLOCKS for word in block.words]


def _card_of(door: dict[str, object]) -> str:
    """The designer's card a door opens on, by what the card model calls it."""
    kind = door["kind"]
    if kind in ("rule", "turns"):
        rule = str(door["rule"])
        assert rule in MODEL["ruleOrder"], door  # *add a rule* has it
        return rule
    if kind == "piece" or (kind == "turns" and "field" in door):
        assert door["field"] in {field["key"] for field in MODEL["piece"]}, door
        return f"pieces[].{door['field']}"
    if kind == "move":
        assert door["move"] in MOVE_KINDS, door
        return str(door["move"])
    if kind == "sight":
        return "pieces[].sight"
    assert kind == "board" and door["lattice"] == "hex", door
    return "board.lattice"


@pytest.mark.parametrize(("block", "word"), WORDS,
                         ids=[f"{block.hue}:{word.said}" for block, word in WORDS])
def test_every_word_of_the_wall_resolves_to_a_card(block: landing.Block,
                                                   word: landing.Word) -> None:
    door = DOORS["doors"][word.key]
    assert door["said"].strip()
    _card_of(door)
    if word.key.startswith("rules.") and word.key not in designing.DOOR_TURNS:
        assert door == {"kind": "rule", "rule": word.key.removeprefix("rules."),
                        "said": word.card}  # the card's own label, as the wall says it


def test_every_story_card_is_a_word_of_the_wall() -> None:
    keys = {word.key for _, word in WORDS}
    for story in stories.STORIES + stories.EXAMPLES:
        assert story.card.door in keys, story.key
        assert story.card.door in DOORS["doors"], story.key


def test_a_door_that_adds_to_a_piece_names_a_piece_that_is_not_royal() -> None:
    piece = CHESS["pieces"][designing.door_piece(CHESS)]
    assert piece.get("royal") is not True
    assert piece["value"] == max(p["value"] for p in CHESS["pieces"] if not p.get("royal"))
    assert designing.door_piece({"pieces": []}) == 0


def test_no_door_for_a_key_the_designer_has_no_control_for() -> None:
    assert "rules.nonsense" not in DOORS["doors"]
    assert "pieces[].moves" not in DOORS["doors"]


class TestTheRoute:
    @pytest.fixture
    def app(self, tmp_path: Path) -> App:
        for name in ("rulesets", "runs", "archive"):
            (tmp_path / name).mkdir()
        shutil.copy(ROOT / "rulesets" / "chess.json", tmp_path / "rulesets" / "chess.json")
        return App(tmp_path / "rulesets", tmp_path / "runs", archive_dir=tmp_path / "archive")

    def test_the_kit_carries_the_doors_and_their_copy_of_chess(self, app: App) -> None:
        kit = app.get("/api/design")
        assert kit["doors"]["start"] == {"hash": spec_hash(CHESS), "name": "chess"}
        assert kit["doors"]["doors"] == DOORS["doors"]
        # The copy the page asks for is a game this library answers.
        assert app.get("/api/design/" + spec_hash(CHESS))["spec"]["pieces"]

    def test_without_chess_the_doors_start_from_a_template(self, tmp_path: Path) -> None:
        for name in ("rulesets", "runs", "archive"):
            (tmp_path / name).mkdir()
        app = App(tmp_path / "rulesets", tmp_path / "runs", archive_dir=tmp_path / "archive")
        assert app.get("/api/design")["doors"]["start"] is None


class TestTheLinks:
    def test_every_word_carries_its_key_and_the_page_links_its_door(self) -> None:
        wall = landing._wall_html()
        for _, word in WORDS:
            assert f'data-door="{word.key}">' in wall, word.said
            # `landing.js` spells the link with `encodeURIComponent`, which agrees with
            # `door_link` on every character a key is made of.
            assert re.fullmatch(r"[a-z_.\[\]]+", word.key), word.key
            assert unquote(landing.door_link(word.key).removeprefix("#/design?rule=")) == word.key
        assert wall.count("data-door=") == len(WORDS)
        assert "door.href = '#/design?rule=' + encodeURIComponent(word.dataset.door);" in PAGE
        assert f"const WALL_DOOR_SAID = '{landing.DOOR_SAID}';" in PAGE


class TestThePage:
    def test_the_router_reads_the_key_off_the_address(self) -> None:
        assert "showDesigner(parts[1] || '', routeQuery.get('rule'))" in PAGE
        assert "split('?')[0]" in PAGE

    def test_the_designer_takes_the_door_through_its_own_add_paths(self) -> None:
        assert "async function openDoor(kit, key)" in PAGE
        # *add a rule*'s default and companion, and the server's refusals for the fallback.
        assert "DRAFT.rules[key] = fieldDefault(field, [key], 'rules');" in PAGE
        assert "RULE_COMPANIONS[key]" in PAGE
        assert "api('/api/design/rules'," in PAGE
        # rounds and hexagons are the controls' own answers.
        assert "chooseTurnWay(door.way)" in PAGE
        assert "document.getElementById('dLattice')" in PAGE

    def test_an_unknown_key_opens_the_designer_as_it_was_with_a_sentence(self) -> None:
        assert "DOOR_SAID = { missing: key };" in PAGE
        assert "so the designer opens as it was" in PAGE
