"""Lines of sight on the page, in the designer and in the analysis pass (schema 30, Q999c,
`docs/plan-passive.md` §4.6 to §4.8).

What is held here, by behaviour:

- **The play page asks the core.** A position of a game with a sight carries the core's effect
  bits (`Engine.effects`) and, for each piece with a sight, the squares it sees
  (`Engine.sight_of`); each history entry and each replayed position carries the effects its
  ply left, so a step back draws its own marks; a game without a sight carries none of it.
- **The marks.** A frozen, a disarmed and a shielded piece each wear a badge of their own
  (freeze contains disarm, so a piece under both wears the snowflake), drawn on the board
  from the position on screen, each with its legend line; a piece's sight is tinted by effect
  on hover or tap, its own square ringed, and only on the position in play.
- **The cards.** Each sight is said on its piece in `describe`'s words, and the line under a
  tapped piece says them.
- **The designer's sight rows.** Every committed sight test spec read into the rows and back
  out through the row's own writer is the file; the menus offer the schema's values; a ray is
  the absent key and an aura `true`, never `false`; the refusals are said under the rows, and
  each row is read back in words; a renamed letter and the hexagons carry a sight with them.
- **The analysis tallies.** Absent from a game without a sight, so its `analysis.json` is the
  bytes it was; present, one entry per side-turn (a ply, or each seat of a round), each the
  core's answer on the board the choice was made on.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
from test_designer import _kit_setup, _run_page
from test_portal_board import _game
from test_web import _in_page

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.describe import sight_sentence
from chess_variant_sandbox.spec.schema import SIGHT_EFFECTS, SIGHT_MAX, SIGHT_WHOSE, piece_sights
from chess_variant_sandbox.tournament import analysis as analysis_pass
from chess_variant_sandbox.tournament.analysis import SIGHT_SERIES, CorpusAnalysis, GameAnalysis
from chess_variant_sandbox.tournament.play import play_game
from chess_variant_sandbox.web import cards, designing
from chess_variant_sandbox.web.library import Game, engine_for, pieces
from chess_variant_sandbox.web.play import Session, effects_payload, sights_payload

HERE = Path(__file__).parent
CHESS = HERE.parent / "rulesets" / "chess.json"
SIGHT_SPECS = sorted(HERE.glob("_sight_*.json"))


def _spec(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"_sight_{name}.json").read_text(encoding="utf-8"))


def _chess() -> dict[str, Any]:
    return json.loads(CHESS.read_text(encoding="utf-8"))


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + "abcdefgh".index(name[0])


# The wardens' position the look check also draws: the warden on d2 freezes the black queen
# beside it on d3 and shields its own king on e1, which stands in the black rook's open file.
WARDENS = {"e1": ("K", 0), "d2": ("W", 0), "a2": ("P", 0),
           "e8": ("K", 1), "d3": ("Q", 1), "e5": ("R", 1)}


def _session(spec: dict[str, Any], placement: dict[str, tuple[str, int]] | None = None) -> Session:
    engine = engine_for(spec)
    state = engine.make_state(placement) if placement else engine.initial_state()
    return Session(id="s1", game=_game(spec), seats=("human", "human"), seed=0, state=state)


# ------------------------------------------------------------------ the server's half


class TestTheServer:
    def test_a_game_without_a_sight_carries_none_of_it(self) -> None:
        session = _session(_chess())
        snap = session.snapshot()
        assert "effects" not in snap and "sights" not in snap
        engine = session.engine
        snap = session.play(engine.legal_moves(session.state)[0])
        assert all("effects" not in entry for entry in snap["history"])
        assert all("effects" not in position for position in designing.timeline(session))
        assert effects_payload(engine, session.state) == {}
        assert sights_payload(engine, session.state) == []

    def test_the_position_carries_the_cores_effects_and_sights(self) -> None:
        session = _session(_spec("wardens"), WARDENS)
        engine, state = session.engine, session.state
        snap = session.snapshot()
        assert snap["effects"] == list(engine.effects(state).bits)
        assert snap["effects"][_square("d3")] == 1  # the queen beside the warden: frozen
        assert snap["effects"][_square("e1")] == 4  # the king: shielded
        # Shielded, so the rook down the open file gives no check.
        assert snap["inCheck"] is False
        # Every piece with a sight, and nothing else, with exactly the core's squares.
        carriers = {_square(name) for name, (kind, _) in WARDENS.items() if kind == "W"}
        assert {entry["square"] for entry in snap["sights"]} == carriers
        for entry in snap["sights"]:
            assert entry["seen"] == [list(pair) for pair in engine.sight_of(state, entry["square"])]
        seen = dict(map(tuple, snap["sights"][0]["seen"]))
        assert seen[_square("d3")] & 1 and seen[_square("e1")] & 4

    def test_a_step_back_has_the_effects_its_ply_left(self) -> None:
        session = _session(_spec("madrasi"))
        engine = session.engine
        for _ in range(6):
            session.play(engine.legal_moves(session.state)[0])
        snap = session.snapshot()
        states = [*session.states[1:], session.state]
        for entry, state in zip(snap["history"], states, strict=True):
            bits = engine.effects(state).bits
            assert entry["effects"] == [[sq, b] for sq, b in enumerate(bits) if b]
        timeline = designing.timeline(session)
        for position, state in zip(timeline, [*session.states, session.state], strict=True):
            bits = engine.effects(state).bits
            assert position["effects"] == [[sq, b] for sq, b in enumerate(bits) if b]

    def test_each_card_says_its_sights_in_describes_words(self) -> None:
        spec = _spec("wardens")
        game = Game(hash="w", name=spec["name"], description="", path=Path(), source="test",
                    spec=spec)
        names = {piece["id"]: piece["name"] for piece in spec["pieces"]}
        sights = piece_sights(spec)
        for piece in pieces(game):
            if piece["id"] not in sights:
                assert "sights" not in piece
                continue
            assert piece["sights"] == [
                {"effect": sight.effect, "words": sight_sentence(sight, names)}
                for sight in sights[piece["id"]]
            ]
            # The card's own notes say each of them too.
            for said in piece["sights"]:
                assert any(note.startswith(said["words"]) for note in piece["notes"])


# ------------------------------------------------------------------ the page's half

_BOARD = """
let DRAWN = '';
const BOARD_SVG = { setAttribute() {}, getAttribute: () => null, querySelector: () => null,
  querySelectorAll: () => [], appendChild() {}, closest: () => null, parentNode: null,
  set innerHTML(v) { DRAWN = v; }, get innerHTML() { return DRAWN; } };
const LINE = { hidden: true, innerHTML: '' };
const deep = () => new Proxy(function () {}, {
  get: (t, k) => k === Symbol.toPrimitive ? (() => '') : k === 'style' || k === 'dataset' ? {} : deep(),
  set: () => true, apply: () => deep() });
document.getElementById = id => id === 'board' ? BOARD_SVG : id === 'pieceSays' ? LINE : deep();
document.createElementNS = () => deep();
"""


def _drawn(snapshot: dict[str, Any], steps: str, expression: str) -> Any:
    return _in_page(f"SESSION = {json.dumps(snapshot)};\n" + _BOARD + steps + "drawBoard();",
                    expression)


def _marks(html: str) -> list[tuple[str, str | None]]:
    return re.findall(r'data-mark="(frozen|disarmed|shielded)"(?: data-player="(\d)")?', html)


class TestThePage:
    def test_each_effect_wears_a_badge_of_its_own(self) -> None:
        kinds = _in_page("", "[1, 2, 3, 4, 5, 6, 7, 0].map(effectKinds)")
        assert kinds == [["frozen"], ["disarmed"], ["frozen"], ["shielded"],
                         ["frozen", "shielded"], ["disarmed", "shielded"],
                         ["frozen", "shielded"], []]
        for bits, kind in ((1, "frozen"), (2, "disarmed"), (4, "shielded")):
            badge = _in_page("", f"effectBadges({bits}, 0, 0, 10, 1)")
            assert _marks(badge) == [(kind, "1")], badge
        legend = _in_page("", "legendEntries(new Set(['frozen', 'disarmed', 'shielded', 'seen']),"
                              " null, '', null)")
        said = " ".join(legend)
        for words in ("frozen: it cannot move, and gives no check",
                      "disarmed: it may move but cannot capture",
                      "shielded: it cannot be captured", "what a piece sees"):
            assert words in said

    def test_the_board_draws_the_cores_marks_and_names_them(self) -> None:
        snap = _session(_spec("wardens"), WARDENS).snapshot()
        drawn = _drawn(snap, "", "[DRAWN, effectLegendKinds(effectsShown())]")
        assert sorted(_marks(drawn[0])) == [("frozen", "1"), ("shielded", "0")]
        assert sorted(drawn[1]) == ["frozen", "seen", "shielded"]
        # A game without a sight draws none, and its legend names none.
        plain = _drawn(_session(_chess()).snapshot(), "",
                       "[DRAWN, effectLegendKinds(effectsShown())]")
        assert _marks(plain[0]) == [] and plain[1] == []

    def test_a_piece_shows_what_it_sees_on_the_position_in_play_only(self) -> None:
        snap = _session(_spec("wardens"), WARDENS).snapshot()
        warden = _square("d2")
        seen = next(entry["seen"] for entry in snap["sights"] if entry["square"] == warden)
        drawn = _drawn(snap, "", f"[sightHtml({warden}, sightOf({warden}), 12.5),"
                                 f" sightOf({_square('e1')})]")
        tinted = re.findall(r'data-mark="seen" data-effect="(\w+)"', drawn[0])
        assert len(tinted) == len(seen)
        assert drawn[0].count('data-source="1"') == 1
        # The king has no sight; and stepping back draws no sight at all.
        assert drawn[1] is None
        back = _drawn(snap, "AT = { board: SESSION.board, effects: [] };",
                      f"sightOf({warden})")
        assert back is None

    def test_a_step_back_reads_its_own_effects(self) -> None:
        pairs = _in_page("", "[effectPairs([0, 1, 0, 4]), effectPairs([[1, 1], [3, 4]]),"
                             " effectPairs(undefined)]")
        assert pairs == [[[1, 1], [3, 4]], [[1, 1], [3, 4]], []]
        session = _session(_spec("madrasi"))
        engine = session.engine
        for _ in range(4):
            snap = session.play(engine.legal_moves(session.state)[0])
        beat = _in_page(f"const S = {json.dumps(snap)};",
                        "beatEntry('ply', S.board, null, S, 3, null, null).effects")
        assert beat == snap["history"][2].get("effects", [])

    def test_the_line_under_a_tapped_piece_says_its_sights(self) -> None:
        spec = _spec("wardens")
        session = _session(spec, WARDENS)
        snap = session.snapshot()
        detail = {"pieces": pieces(Game(hash="w", name=spec["name"], description="",
                                        path=Path(), source="test", spec=spec))}
        steps = (f"DETAIL[SESSION.ruleset.hash] = {json.dumps(detail)};"
                 f" SIGHT_TAP = {{ square: {_square('d2')}, ply: SESSION.ply }};")
        line = _drawn(snap, steps, "LINE.innerHTML")
        warden = next(piece for piece in detail["pieces"] if piece["id"] == "W")
        for sight in warden["sights"]:
            assert sight["words"].replace("'", "&#39;") in line or sight["words"] in line


# ------------------------------------------------------------------ the designer's half

# Each sight row read the way a browser reads it -- a menu's selected option, an input's value,
# the ticked boxes -- and written back through the row's own writer.
_SIGHT_TRIP = """
function readRow(html) {
  const decode = s => s.replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  const named = (tag, attr) => (new RegExp(attr + '="([^"]*)"').exec(tag) || [])[1];
  const out = [];
  for (const [, tag, body] of html.matchAll(/<select([^>]*)>([\\s\\S]*?)<\\/select>/g)) {
    const options = [...body.matchAll(/<option value="([^"]*)"([^>]*)>/g)];
    const chosen = options.find(o => /\\bselected\\b/.test(o[2])) || options[0];
    out.push([named(tag, 'data-sfield'), chosen ? decode(chosen[1]) : '']);
  }
  const ticked = [];
  for (const [, tag] of html.matchAll(/<input([^>]*)>/g)) {
    if (/type="checkbox"/.test(tag)) {
      if (/\\schecked\\b/.test(tag)) ticked.push(decode(named(tag, 'data-pid')));
      continue;
    }
    out.push([named(tag, 'data-sfield'), decode((/\\svalue="([^"]*)"/.exec(tag) || [])[1] || '')]);
  }
  out.push(['affects', ticked]);
  return out;
}
function sightTrip(specs) {
  const result = {};
  for (const [name, spec] of Object.entries(specs)) {
    DRAFT = JSON.parse(JSON.stringify(spec));
    DRAFT.pieces.forEach((piece, i) => (piece.sight || []).forEach((sight, k) => {
      readRow(sightEditor(sight, i, k)).forEach(([field, value]) =>
        writeSightField(sight, field, value));
    }));
    result[name] = DRAFT;
  }
  return result;
}
"""


def _designer(expression: str, setup: str = "") -> Any:
    return _run_page(_kit_setup() + setup, expression)


class TestTheDesigner:
    def test_every_sight_spec_round_trips_through_the_rows(self) -> None:
        specs = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in SIGHT_SPECS}
        assert len(specs) >= 9
        # Every value the rows can hold, written out: the defaults spelled, an aura, a ray,
        # and a list of pieces out of the pieces' own order.
        spelled = _spec("wardens")
        spelled["pieces"][0]["sight"] = [
            {"vectors": [[1, 1]], "symmetry": "lateral", "range": 0, "min_range": 1,
             "effect": "disarm", "whose": "any", "affects": ["Q", "P"], "through": True},
            {"vectors": [[0, 1]], "range": 3, "min_range": 2, "effect": "shield", "whose": "own"},
        ]
        specs["spelled"] = spelled
        back = _designer("sightTrip(" + json.dumps(specs) + ")", _SIGHT_TRIP)
        for name, spec in specs.items():
            assert back[name] == spec, name

    def test_the_menus_offer_the_schemas_values(self) -> None:
        kit = cards.card_model()["sight"]
        assert kit["effects"] == list(SIGHT_EFFECTS) and kit["whose"] == list(SIGHT_WHOSE)
        assert kit["max"] == SIGHT_MAX
        html = _designer("(DRAFT = " + json.dumps(_spec("wardens")) + ", "
                         "sightEditor(DRAFT.pieces[6].sight[0], 6, 0))")
        menus = dict(re.findall(r'data-sfield="(\w+)">(.*?)</select>', html, re.S))
        values = {key: re.findall(r'<option value="([^"]*)"', body) for key, body in menus.items()}
        assert values["effect"] == list(SIGHT_EFFECTS)
        assert values["whose"] == list(SIGHT_WHOSE)
        assert values["through"] == ["ray", "aura"]
        # Every field is labelled in `spec.plain`'s words.
        for key in cards.SIGHT_FIELDS:
            assert f'data-sfor="{key}"' in html or key == "affects", key
        assert "only these pieces" in html and "what it does" in html
        # The generated stand-in is gone: the piece's generated fields hold no sight.
        assert "sight" not in {field["key"] for field in cards.card_model()["piece"]}

    def test_a_ray_is_the_absent_key_and_an_aura_true(self) -> None:
        written = _designer(
            "(() => { const s = { through: true }, t = {}, u = { affects: ['P'] };"
            " writeSightField(s, 'through', 'ray'); writeSightField(t, 'through', 'aura');"
            " writeSightField(u, 'affects', []); return [s, t, u]; })()")
        assert written == [{}, {"through": True}, {}]

    def test_a_new_line_of_sight_is_a_game_and_a_full_piece_takes_no_more(self) -> None:
        new = _designer("NEW_SIGHT")
        spec = _chess()
        spec["schema_version"] = 30
        queen = next(piece for piece in spec["pieces"] if piece["id"] == "Q")
        queen["sight"] = [new]
        assert validate_spec(spec) == []
        add = _designer("[sightAddHtml({}, 2), sightAddHtml({ sight: [{}, {}, {}, {}] }, 2),"
                        " sightsHtml({ id: 'K' }, 0)]")
        assert 'data-addsight="2"' in add[0] and "disabled" not in add[0]
        assert "disabled" in add[1]
        assert add[2] == ""

    def test_the_check_reads_each_row_back_and_says_its_refusals(self) -> None:
        spec = _spec("wardens")
        answer = designing.check(spec)
        assert answer["valid"]
        warden = next(i for i, piece in enumerate(spec["pieces"]) if piece["id"] == "W")
        names = {piece["id"]: piece["name"] for piece in spec["pieces"]}
        assert answer["sightLines"][warden] == [
            sight_sentence(sight, names) for sight in piece_sights(spec)["W"]]
        broken = copy.deepcopy(spec)
        broken["pieces"][warden]["sight"][0]["effect"] = "melt"
        broken["pieces"][warden]["sight"].append({"vectors": [[1, 0]], "effect": "shield",
                                                  "whose": "own", "affects": ["Z"]})
        answer = designing.check(broken)
        assert not answer["valid"]
        under = answer["sightProblems"][warden]
        assert under and all(line in answer["errors"] for line in under)
        assert any("line of sight 1" in line and "what it does" in line for line in under)
        assert any("line of sight 3" in line for line in under)
        assert all(not lines for i, lines in enumerate(answer["sightProblems"]) if i != warden)
        # The page shows the check's words under the rows of the draft they are of.
        html = _designer(f"(DRAFT = {json.dumps(broken)}, SIGHT_SAID = {{ key: JSON.stringify(DRAFT),"
                         f" lines: [], problems: {json.dumps(answer['sightProblems'])} }},"
                         f" sightsHtml(DRAFT.pieces[{warden}], {warden}))")
        assert html.count('class="error small sightError"') == len(under)

    def test_a_renamed_letter_and_the_hexagons_carry_a_sight(self) -> None:
        spec = _spec("madrasi")
        renamed = _designer("(DRAFT = " + json.dumps(spec) + ", renamePiece('Q', 'X'),"
                            " DRAFT.pieces.find(p => p.id === 'Q').sight)")
        assert renamed[0]["affects"] == ["X"]
        spec = _spec("wardens")
        spec["board"]["lattice"] = "hex"
        made = designing.onto_hex(spec)
        warden = next(piece for piece in made["spec"]["pieces"] if piece["id"] == "W")
        assert all(sight.get("symmetry", "none") == "none" and len(sight["vectors"]) >= 6
                   for sight in warden["sight"])
        assert "lines of sight of the warden" in made["said"]


# ------------------------------------------------------------------ the analysis tallies

# The fields an analysis held before schema 30, which a game without a sight still writes,
# and only these (a game in rounds adds `rounds`, a spy game `spy`).
_BEFORE = ["seed", "agents", "near_best", "legal_counts", "moves_by_type", "captures_by_type",
           "capture_counts", "revisited_plies", "walked_plies", "starting_instances",
           "moved_instances", "position_keys"]


def _analysed(spec: dict[str, Any], seed: int = 1) -> tuple[Engine, Any, GameAnalysis]:
    engine = Engine(spec)
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    record = play_game(engine, ("greedy", "random"), seed)
    found = analysis_pass._analyse_game(engine, evaluator, record,
                                        analysis_pass._epsilon_for(engine))
    return engine, record, found


class TestTheTallies:
    def test_a_game_without_a_sight_writes_the_bytes_it_did(self, tmp_path: Path) -> None:
        _, _, found = _analysed(_chess())
        assert found.sight is None
        data = found.as_dict()
        assert list(data) == _BEFORE
        before = {f.name: getattr(found, f.name) for f in fields(found) if f.name in _BEFORE}
        before["agents"] = list(found.agents)
        path = CorpusAnalysis(games=[found]).save(tmp_path / "analysis.json")
        assert path.read_text(encoding="utf-8") == json.dumps(
            {"games": [before], "blunders": []}, indent=2)

    def test_each_turn_is_the_cores_answer_on_its_board(self) -> None:
        engine, record, found = _analysed(_spec("madrasi"))
        tally = found.sight
        assert tally is not None and list(tally) == list(SIGHT_SERIES)
        assert all(len(series) == record.plies for series in tally.values())
        state = engine.initial_state()
        for ply, move in enumerate(record.moves):
            effects = engine.effects(state)
            mine = [b for sq, b in enumerate(effects.bits)
                    if b and engine.spec.owner_of(state.board[sq]) == state.side_to_move]
            row = {key: tally[key][ply] for key in SIGHT_SERIES}
            assert row == {
                "side": state.side_to_move, "removed_freeze": effects.frozen,
                "removed_disarm": effects.disarmed, "removed_shield": effects.shielded,
                "binds": int(effects.binds), "frozen": sum(1 for b in mine if b & 1),
                "disarmed": sum(1 for b in mine if b & 2),
                "shielded": sum(1 for b in mine if b & 4), "suppressed": int(effects.suppressed),
            }, ply
            state = engine.apply(state, move)
        assert sum(tally["binds"]) > 0
        assert GameAnalysis.from_dict(json.loads(json.dumps(found.as_dict()))) == found

    @pytest.mark.parametrize("name", ["rounds", "spy"])
    def test_a_round_counts_each_seat(self, name: str) -> None:
        _, record, found = _analysed(_spec(name), seed=2)
        tally = found.sight
        assert tally is not None
        assert len(tally["side"]) == len(record.moves)
        if name == "rounds":
            assert tally["side"] == [0, 1] * (len(record.moves) // 2)
        assert set(tally["side"]) <= {0, 1}


class TestTheMapReadsASight:
    def test_a_game_with_a_sight_has_words_and_its_affects_read_by_what_they_name(self) -> None:
        """The map's relation (`spec/likeness.py`) read a sight's entries as piece ids and
        failed on every game with one, which took its game page down (`near`, Q999c)."""
        from chess_variant_sandbox.spec.likeness import words_of

        for path in SIGHT_SPECS:
            assert words_of(json.loads(path.read_text(encoding="utf-8")))
        spec = _spec("madrasi")
        other = copy.deepcopy(spec)
        queen = next(piece for piece in other["pieces"] if piece["id"] == "Q")
        queen["sight"][0]["affects"] = ["R"]
        assert words_of(spec) != words_of(other)
