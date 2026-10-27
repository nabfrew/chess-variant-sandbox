"""A door per rule into the designer (LP7, `docs/archive/plan-landing.md` §5).

`#/design?rule=<key>` opens the designer on a copy of chess with that rule added, through the
designer's own add paths. These hold the route together from its three sides: every word of
the landing's wall and every story's card links a key that `designing.doors` resolves to a
card the designer draws (the server's half), the kit carries the doors and the copy they
start from (the route), and the page reads the key off the address and takes the door, or
says it has none (the page's half, read as text). That each door opens with its card present
and the draft checked is the browser's, LP7's acceptance.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

import node_session
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
        plain = {key: door[key] for key in ("kind", "rule", "said")}
        assert plain == {"kind": "rule", "rule": word.key.removeprefix("rules."),
                         "said": word.card}  # the card's own label, as the wall says it
        # Beyond those, only the parts the server chose where the default is no game.
        assert set(door) - set(plain) <= {"card", "with", "withSaid"}, door


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
        assert "showDesigner(parts[1] || '', routeQuery.get('rule')," in PAGE
        assert "split('?')[0]" in PAGE

    def test_the_designer_takes_the_door_through_its_own_add_paths(self) -> None:
        assert "async function openDoor(kit, key)" in PAGE
        # *add a rule*'s default and companion, and the server's refusals for the fallback.
        assert ": fieldDefault(field, [key], 'rules');" in PAGE
        # Where the default is no game, the server's choice the door carries (Q1036).
        assert "JSON.parse(JSON.stringify(door.card))" in PAGE
        assert "door.value !== undefined ? JSON.parse(JSON.stringify(door.value))" in PAGE
        assert "RULE_COMPANIONS[key]" in PAGE
        assert "api('/api/design/rules'," in PAGE
        # rounds and hexagons are the controls' own answers.
        assert "chooseTurnWay(door.way)" in PAGE
        assert "document.getElementById('dLattice')" in PAGE

    def test_an_unknown_key_opens_the_designer_as_it_was_with_a_sentence(self) -> None:
        assert "DOOR_SAID = { missing: key };" in PAGE
        assert "so the designer opens as it was" in PAGE


# Q1036: every door opens on a draft that is a game. The page's own `openDoor` makes each
# door's draft in node, as a click from the landing does, and every question it asks the
# server is answered by this library's `App`: a question not yet answered is recorded, the
# run answers it and goes again, until the page asks nothing new. Rounds and hexagons are
# the controls' own answers (`doorDrawn`), so their two control paths run after the door.
# Each draft then goes through the designer's check, `designing.check`, which is what the
# page shows and what `/api/design/moves` needs to answer.
_DOOR_STUB = """
  const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                             set: () => true });
  global.document = { getElementById: () => el, querySelector: () => null,
                      querySelectorAll: () => [], addEventListener: () => {},
                      createElement: () => el, body: el };
  global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                    addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
  global.location = { pathname: '/game', hash: '' };
  global.history = { replaceState: () => {} };
  global.fetch = () => new Promise(() => {});
"""

_DOOR_WALK = """
const ASKED = [];
api = async (path, body) => {
  const asked = path + ' ' + JSON.stringify(body === undefined ? null : body);
  if (Object.prototype.hasOwnProperty.call(ANSWERS, asked)) {
    return JSON.parse(JSON.stringify(ANSWERS[asked]));
  }
  ASKED.push([path, body === undefined ? null : body]);
  return path === '/api/design/rules' ? { refused: {} } : null;
};
drawDesigner = () => {}; drawTurns = () => {}; checkDraft = () => {};
loadMoveMap = () => {}; drawBoardSide = () => {};
async function walkDoors(keys) {
  const drafts = {};
  for (const key of keys) {
    DRAFT = null;
    const door = await openDoor(DESIGN_KIT, key);
    if (door.kind === 'turns') { TURN_ASKING = ''; await chooseTurnWay(door.way); }
    if (door.kind === 'board') {
      writeBoardField(DRAFT.board, 'lattice', door.lattice);
      fitWrapToShape();
      await ontoHex();
    }
    drafts[key] = DRAFT;
  }
  return { drafts, asked: ASKED };
}
"""


def _door_drafts(app: App) -> dict[str, dict[str, object]]:
    """Each door's draft as the page makes it, every question answered by `app`."""
    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    kit = app.get("/api/design")
    answers: dict[str, object] = {}
    for _ in range(8):
        body = (_DOOR_STUB + script.group(1) + "\n" + f"DESIGN_KIT = {json.dumps(kit)};\n"
                + f"const ANSWERS = {json.dumps(answers)};\n" + _DOOR_WALK
                + f"walkDoors({json.dumps(sorted(kit['doors']['doors']))})"
                + ".then(v => console.log(JSON.stringify(v)));")
        done = node_session.run(body)
        assert done.returncode == 0, done.stderr
        walked = json.loads(done.stdout)
        if not walked["asked"]:
            return dict(walked["drafts"])
        for path, asked in walked["asked"]:
            # JSON.stringify's spelling, which the page's lookup reads.
            key = path + " " + json.dumps(asked, separators=(",", ":"), ensure_ascii=False)
            answers[key] = app.get(path) if asked is None else app.post(path, asked)
    raise AssertionError("the page kept asking new questions")


@pytest.mark.slow
class TestEveryDoorOpensAGame:
    def test_every_doors_draft_passes_the_designers_check(self, tmp_path: Path) -> None:
        for name in ("rulesets", "runs", "archive"):
            (tmp_path / name).mkdir()
        shutil.copy(ROOT / "rulesets" / "chess.json", tmp_path / "rulesets" / "chess.json")
        app = App(tmp_path / "rulesets", tmp_path / "runs", archive_dir=tmp_path / "archive")
        drafts = _door_drafts(app)
        assert set(drafts) == set(DOORS["doors"])
        failed = {}
        for key, draft in drafts.items():
            answer = designing.check(draft)
            if not answer.get("valid") or answer.get("errors"):
                failed[key] = answer.get("errors")
        assert not failed, failed


class TestTheChosenParts:
    """What the server chooses for the four doors whose card's default is no game, read
    against chess (fast; the walk above is the proof that each is then a game)."""

    def test_the_portals_stand_a_half_turn_apart_off_the_middle(self) -> None:
        assert DOORS["doors"]["rules.portals"]["card"] == [{"orange": "c4", "blue": "f5"}]

    def test_promoting_on_arrival_comes_with_a_hand_that_feeds_it(self) -> None:
        door = DOORS["doors"]["rules.promote_on_entry"]
        assert door["card"] is True
        assert door["with"] == designing.DOOR_ENTRY_FEED
        assert door["withSaid"] == ["pieces played back onto the board"]

    def test_only_on_these_squares_keeps_a_piece_to_its_own_half(self) -> None:
        door = DOORS["doors"]["pieces[].region"]
        piece = CHESS["pieces"][door["piece"]]
        assert piece.get("royal") is not True
        assert door["value"] and all(int(square[1:]) <= 4 for square in door["value"])

    def test_a_facing_goes_on_a_piece_it_changes(self) -> None:
        door = DOORS["doors"]["pieces[].facing"]
        assert door["value"] == designing.DOOR_FACING
        # Not the piece worth most, whose moves read the same both ways.
        assert door["piece"] != designing.door_piece(CHESS)

    def test_a_template_start_chooses_nothing(self) -> None:
        bare = designing.doors(MODEL)["doors"]
        assert "card" not in bare["rules.portals"]
        assert "value" not in bare["pieces[].region"]
