"""The `command` effect on the page, in the designer and in the analysis tallies (Q1007d,
`docs/plan-schema-31.md` §3.9).

What is held here, by behaviour:

- **The play page asks the core.** A position of a game whose sight commands carries the
  core's effect bits with the command bits (8, and 16 under ``captures_own``) on each
  commanded piece; its legal list holds the commanded moves with their origins on the
  opponent's pieces and their text with the ``~``; a commanded move made from the board is
  the core's own move, its history entry and a replay say it with the ``~``.
- **The marks.** A commanded piece wears a badge of its own (a marionette's cross and
  strings) on the corner the rest's sign has, which a command is refused beside; a frozen
  piece is commanded by nobody, so it wears the snowflake alone; the legend names it; a
  command sight's hover is tinted in the command's colour.
- **The board picks the opponent's piece.** A tap on a commanded enemy piece picks it by
  the legal list's origins, never by its colour, and a tap on one of its squares sends the
  core's commanded move.
- **The designer.** *Commands it* is on the effect menu; *may make them take their own side*
  (``captures_own``) is a box shown only beside it, writing ``true`` or no key; changing the
  effect away from command takes the key with it; every command spec round-trips through
  the rows.
- **The tallies.** `COMMAND_SERIES` beside `SIGHT_SERIES` in a game whose sight commands, each
  the core's answer or the record's move; absent byte for byte from every other game,
  a sight game without a command included.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from test_designer import _kit_setup, _run_page
from test_portal_board import _game
from test_sight_page import _SIGHT_TRIP, _drawn
from test_web import _in_page

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import SIGHT_COMMAND, SIGHT_COMMAND_OWN
from chess_variant_sandbox.engine.moves import MOVE_COMMANDED, move_from
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.tournament import analysis as analysis_pass
from chess_variant_sandbox.tournament.analysis import (
    COMMAND_SERIES,
    SIGHT_SERIES,
    GameAnalysis,
)
from chess_variant_sandbox.tournament.play import play_game
from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.viz.html import render_game
from chess_variant_sandbox.web import cards, designing
from chess_variant_sandbox.web.library import engine_for
from chess_variant_sandbox.web.play import Session

HERE = Path(__file__).parent
COMMAND_SPECS = sorted(HERE.glob("_command_*.json"))


def _spec(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"_command_{name}.json").read_text(encoding="utf-8"))


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + "abcdefgh".index(name[0])


def _session(spec: dict[str, Any], placement: dict[str, tuple[str, int]] | None = None,
             side: int = 0) -> Session:
    engine = engine_for(spec)
    state = engine.make_state(placement) if placement else engine.initial_state()
    if side:
        state = state.with_(side_to_move=side)
    return Session(id="s1", game=_game(spec), seats=("human", "human"), seed=0, state=state)


# K1's opening position: the white knights command the black pieces a knight's move away,
# none at the start; after 1. Nc3 Nf6 2. Nd5 the knight on d5 commands c7 and e7.
# K2's: the white king beside a black queen, which it commands and may turn on its side.
K2_QUEEN = {"e1": ("K", 0), "e2": ("Q", 1), "b5": ("B", 1), "e8": ("K", 1), "a2": ("P", 0)}


def _k1_after(session: Session, texts: list[str]) -> None:
    engine = session.engine
    for text in texts:
        move = next(m for m in engine.legal_moves(session.state)
                    if engine.describe_move(session.state, m) == text)
        session.play(move)


# ------------------------------------------------------------------ the server's half


class TestTheServer:
    def test_the_position_carries_the_command_bits_and_the_commanded_moves(self) -> None:
        session = _session(_spec("k1"))
        _k1_after(session, ["Nb1-c3", "Ng8-f6", "Nc3-d5"])
        engine, state = session.engine, session.state
        snap = session.snapshot()
        assert snap["effects"] == list(engine.effects(state).bits)
        # It is black's turn: the black knight on f6 commands nothing of white's yet, but
        # white's knight on d5 commands black's c7 and e7 pawns, for white's next turn.
        assert snap["effects"][_square("c7")] & SIGHT_COMMAND
        assert snap["effects"][_square("e7")] & SIGHT_COMMAND
        assert not snap["effects"][_square("c7")] & SIGHT_COMMAND_OWN
        session.play(next(m for m in engine.legal_moves(state)
                          if engine.describe_move(state, m) == "Nf6-g8"))
        snap = session.snapshot()
        commanded = [m for m in snap["moves"] if m["move"] & MOVE_COMMANDED]
        assert commanded, "white may move the pawns its knight commands"
        assert {m["from"] for m in commanded} == {_square("c7"), _square("e7")}
        assert all(m["text"].startswith("~") for m in commanded)
        assert all(m["kind"] == "normal" for m in commanded)
        assert len(commanded) == session.engine.effects(session.state).commanded

    def test_captures_own_sets_the_second_bit(self) -> None:
        session = _session(_spec("k2"), K2_QUEEN)
        snap = session.snapshot()
        bits = snap["effects"][_square("e2")]
        assert bits & SIGHT_COMMAND and bits & SIGHT_COMMAND_OWN
        texts = [m["text"] for m in snap["moves"] if m["move"] & MOVE_COMMANDED]
        assert "~Qe2xb5" in texts  # the queen made to take its own bishop

    def test_a_commanded_move_is_said_with_its_mark_in_history_and_replay(
        self, tmp_path: Path
    ) -> None:
        session = _session(_spec("k2"), K2_QUEEN)
        move = next(m for m in session.snapshot()["moves"] if m["text"] == "~Qe2xb5")
        snap = session.play(move["move"])
        last = snap["history"][-1]
        assert last["text"] == "~Qe2xb5" and last["player"] == 0
        assert last["from"] == _square("e2") and last["to"] == _square("b5")
        # The piece on b5 is still black's queen, moved by white.
        assert session.engine.spec.owner_of(snap["board"][_square("b5")]) == 1
        assert len(designing.timeline(session)) == 2
        # A replay of a game from its record says it the same way.
        spec = _spec("k1")
        k1 = _session(spec)
        _k1_after(k1, ["Nb1-c3", "Ng8-f6", "Nc3-d5", "Nf6-g8"])
        commanded = next(m for m in k1.snapshot()["moves"] if m["move"] & MOVE_COMMANDED)
        k1.play(commanded["move"])
        assert k1.snapshot()["history"][-1]["text"] == commanded["text"]
        record = GameRecord(ruleset_hash=k1.game.hash, ruleset_name=spec["name"], seed=0,
                            agents=("human", "human"), moves=list(k1.played), winner=None,
                            reason="unfinished", plies=len(k1.played))
        page = render_game(spec, record, tmp_path / "replay.html").read_text(encoding="utf-8")
        assert json.dumps(commanded["text"]) in page

    def test_a_sight_game_without_a_command_carries_no_command_bit(self) -> None:
        spec = json.loads((HERE / "_sight_madrasi.json").read_text(encoding="utf-8"))
        session = _session(spec)
        snap = session.snapshot()
        assert not any(bits & (SIGHT_COMMAND | SIGHT_COMMAND_OWN) for bits in snap["effects"])
        assert not any(m["move"] & MOVE_COMMANDED for m in snap["moves"])


# ------------------------------------------------------------------ the page's half


def _marks(html: str) -> list[tuple[str, str | None]]:
    return re.findall(r'data-mark="(frozen|disarmed|shielded|commanded)"(?: data-player="(\d)")?',
                      html)


class TestThePage:
    def test_a_commanded_piece_wears_a_badge_of_its_own(self) -> None:
        kinds = _in_page("", "[8, 24, 9, 10, 12, 26, 0].map(effectKinds)")
        # Freeze wins over command (a frozen piece is moved by nobody); disarm and shield sit
        # beside it, each on its own corner.
        assert kinds == [["commanded"], ["commanded"], ["frozen"], ["disarmed", "commanded"],
                         ["shielded", "commanded"], ["disarmed", "commanded"], []]
        badge = _in_page("", "effectBadges(8, 0, 0, 10, 1)")
        assert _marks(badge) == [("commanded", "1")], badge
        # On the top-right corner, the rest's, which a command is refused beside; the others
        # are on the left.
        assert re.search(r'data-mark="commanded"[^>]*>\s*<circle cx="8\.000" cy="2\.000"', badge)
        legend = " ".join(_in_page("", "legendEntries(new Set(['commanded']), null, '', null)"))
        assert "commanded: the other player may move it too" in legend
        # A command sight's hover is tinted in its own colour.
        assert _in_page("", "[effectTint(8), effectTint(24), effectTint(9), effectTint(4)]") == [
            "commanded", "commanded", "frozen", "shielded"]

    def test_the_board_draws_the_commanded_badges_and_names_them(self) -> None:
        snap = _session(_spec("k2"), K2_QUEEN).snapshot()
        drawn = _drawn(snap, "", "[DRAWN, effectLegendKinds(effectsShown())]")
        assert _marks(drawn[0]) == [("commanded", "1")]
        assert sorted(drawn[1]) == ["commanded", "seen"]
        # The king's sight, hovered, tints the queen's square in the command's colour.
        king = _square("e1")
        tinted = _drawn(snap, "", f"sightHtml({king}, sightOf({king}), 12.5)")
        assert 'data-effect="commanded"' in tinted

    def test_a_tap_on_a_commanded_enemy_piece_picks_it_and_a_tap_on_its_square_moves_it(
        self,
    ) -> None:
        snap = _session(_spec("k2"), K2_QUEEN).snapshot()
        queen, bishop = _square("e2"), _square("b5")
        move = next(m for m in snap["moves"] if m["text"] == "~Qe2xb5")
        steps = ("let SENT = null; send = m => { SENT = m; };"
                 f" clickSquare({queen}); const PICKED = SELECTED; clickSquare({bishop});")
        picked, sent = _drawn(snap, steps, "[PICKED, SENT]")
        assert picked == queen
        assert sent == move["move"]


# ------------------------------------------------------------------ the designer's half


def _designer(expression: str, setup: str = "") -> Any:
    return _run_page(_kit_setup() + setup, expression)


class TestTheDesigner:
    def test_the_menu_offers_command_and_its_box_only_beside_it(self) -> None:
        kit = cards.card_model()["sight"]
        assert "command" in kit["effects"]
        assert "captures_own" in kit["words"]
        spec = _spec("k2")
        king = next(i for i, piece in enumerate(spec["pieces"]) if piece["id"] == "K")
        html = _designer(f"(DRAFT = {json.dumps(spec)}, sightEditor(DRAFT.pieces[{king}].sight[0],"
                         f" {king}, 0))")
        assert re.search(r'<option value="command" selected>commands it</option>', html)
        box = re.search(r'<input\s+type="checkbox"[^>]*data-sfield="captures_own"[^>]*>', html)
        assert box and " checked" in box.group(0)
        assert 'data-sfor="captures_own"' in html
        k1 = _spec("k1")
        knight = next(i for i, piece in enumerate(k1["pieces"]) if piece["id"] == "N")
        html = _designer(f"(DRAFT = {json.dumps(k1)}, sightEditor(DRAFT.pieces[{knight}].sight[0],"
                         f" {knight}, 0))")
        box = re.search(r'<input\s+type="checkbox"[^>]*data-sfield="captures_own"[^>]*>', html)
        assert box and " checked" not in box.group(0)
        # Beside any other effect the box is not there.
        frozen = copy.deepcopy(k1["pieces"][knight]["sight"][0])
        frozen["effect"] = "freeze"
        html = _designer(f"(DRAFT = {json.dumps(k1)}, sightEditor({json.dumps(frozen)},"
                         f" {knight}, 0))")
        assert "captures_own" not in html

    def test_every_command_spec_round_trips_through_the_rows(self) -> None:
        specs = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in COMMAND_SPECS}
        assert len(specs) >= 4
        back = _designer("sightTrip(" + json.dumps(specs) + ")", _SIGHT_TRIP)
        for name, spec in specs.items():
            assert back[name] == spec, name

    def test_captures_own_is_true_or_no_key_and_goes_with_the_command(self) -> None:
        written = _designer(
            "(() => { const a = { effect: 'command' }, b = { effect: 'command',"
            " captures_own: true }, c = { effect: 'command', captures_own: true },"
            " d = { effect: 'command', captures_own: false };"
            " writeSightField(a, 'captures_own', true);"
            " writeSightField(b, 'captures_own', false);"
            " writeSightField(c, 'effect', 'freeze');"
            " writeSightField(d, 'captures_own', false); return [a, b, c, d]; })()")
        # A file's `false` (the off value) read through an unticked box is left as written.
        assert written == [{"effect": "command", "captures_own": True}, {"effect": "command"},
                           {"effect": "freeze"}, {"effect": "command", "captures_own": False}]

    def test_a_command_drawn_in_the_designer_is_a_game(self) -> None:
        spec = _spec("k2")
        assert validate_spec(spec) == []
        answer = designing.check(spec)
        assert answer["valid"]
        king = next(i for i, piece in enumerate(spec["pieces"]) if piece["id"] == "K")
        assert "their own side" in answer["sightLines"][king][0]
        # The validator's refusal, said under the row: a royal commanded.
        broken = copy.deepcopy(spec)
        broken["pieces"][king]["sight"][0]["affects"] = ["K"]
        answer = designing.check(broken)
        assert not answer["valid"]
        under = answer["sightProblems"][king]
        assert under and any("royal" in line for line in under)


# ------------------------------------------------------------------ the analysis tallies


def _analysed(spec: dict[str, Any], seed: int = 1) -> tuple[Engine, Any, GameAnalysis]:
    engine = Engine(spec)
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    record = play_game(engine, ("greedy", "random"), seed)
    found = analysis_pass._analyse_game(engine, evaluator, record,
                                        analysis_pass._epsilon_for(engine))
    return engine, record, found


class TestTheTallies:
    def test_a_sight_game_without_a_command_writes_the_keys_it_did(self) -> None:
        spec = json.loads((HERE / "_sight_madrasi.json").read_text(encoding="utf-8"))
        _, _, found = _analysed(spec)
        assert found.sight is not None and list(found.sight) == list(SIGHT_SERIES)

    def test_each_turn_is_the_cores_answer_and_the_records_move(self) -> None:
        for name, seed in (("k1", 1), ("k2", 2)):
            engine, record, found = _analysed(_spec(name), seed)
            tally = found.sight
            assert tally is not None and list(tally) == [*SIGHT_SERIES, *COMMAND_SERIES]
            assert all(len(series) == record.plies for series in tally.values())
            state = engine.initial_state()
            for ply, move in enumerate(record.moves):
                effects = engine.effects(state)
                mine = [b for sq, b in enumerate(effects.bits)
                        if b and engine.spec.owner_of(state.board[sq]) == state.side_to_move]
                made = bool(move & MOVE_COMMANDED)
                row = {key: tally[key][ply] for key in COMMAND_SERIES}
                assert row == {
                    "commanded": sum(1 for b in mine if b & SIGHT_COMMAND),
                    "commanded_own": sum(1 for b in mine if b & SIGHT_COMMAND_OWN),
                    "command_offered": effects.commanded,
                    "command_made": int(made),
                    "turncoat": int(made and state.board[move >> 8 & 0xFF] != 0),
                }, (name, ply)
                if made:
                    # A commanded move starts on the opponent's piece.
                    assert engine.spec.owner_of(state.board[move_from(move)]) != \
                        state.side_to_move
                state = engine.apply(state, move)
            assert sum(tally["command_offered"]) > 0, name
            assert GameAnalysis.from_dict(json.loads(json.dumps(found.as_dict()))) == found
