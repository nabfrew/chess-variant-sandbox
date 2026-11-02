"""N3: a game's small endings, from its persisted record to the page (`web/endings.py`).

The records are built here in N2's shape (`runs/N2/n2_summary.py`), for an invented game, so
the sentences, the basis stamp and every missing-data case are held without a solver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chess_variant_sandbox.web import endings
from chess_variant_sandbox.web.page import game as drawing
from tests.page_text import PAGE

BUILD = "00112233aabbccdd"
DIGEST = "abcdef012345"
SPEC: dict[str, Any] = {
    "name": "an invented game",
    "pieces": [
        {"id": "S", "name": "sparrow"},
        {"id": "O", "name": "owl"},
        {"id": "E", "name": "eagle"},
        {"id": "C", "name": "crown", "royal": True},
    ],
    "rules": {},
}


def _mover(won: int, lost: int, drawn: int, win: int = 0, loss: int = 0) -> dict[str, int]:
    return {"won": won, "lost": lost, "drawn": drawn, "longest_win": win, "longest_loss": loss}


def _set(name: str, material: list[list[int]], movers: list[dict[str, int]],
         forced: list[bool]) -> dict[str, Any]:
    return {
        "set": name, "material": [*material, []],
        "won": sum(m["won"] for m in movers), "lost": sum(m["lost"] for m in movers),
        "drawn": sum(m["drawn"] for m in movers), "mover": movers, "forced": forced,
    }


def _record(**over: Any) -> dict[str, Any]:
    sets = [
        _set("C v C", [[3], [3]], [_mover(0, 0, 50), _mover(0, 0, 50)], [False, False]),
        # The eagle wins from 90% of positions, in 31 plies: 16 moves.
        _set("EC v C", [[2, 3], [3]], [_mover(90, 0, 10, win=31), _mover(0, 90, 10, loss=30)],
             [True, False]),
        _set("C v EC", [[3], [2, 3]], [_mover(0, 90, 10, loss=30), _mover(90, 0, 10, win=31)],
             [False, True]),
        # The owl forces nothing.
        _set("OC v C", [[1, 3], [3]], [_mover(0, 0, 100), _mover(0, 0, 100)], [False, False]),
        # The sparrow's ending was too big to solve.
        {"set": "SC v C", "material": [[0, 3], [3], []], "error": "over the cap"},
        # A two-piece set and a closure set: neither is a one-piece ending, so unread.
        _set("EOC v C", [[1, 2, 3], [3]], [_mover(1, 0, 1, win=3), _mover(0, 1, 1)],
             [True, False]),
    ]
    stalls = [
        {"seed": 0, "record": 1, "plies": 120,
         "labels": {"won": 4, "cursed": 0, "drawn": 2, "absent": 0}},
        {"seed": 0, "record": 2, "plies": 120,
         "labels": {"won": 0, "cursed": 0, "drawn": 9, "absent": 0}},
        {"seed": 1000, "record": 3, "plies": 99,
         "labels": {"won": 0, "cursed": 0, "drawn": 0, "absent": 0}},
    ]
    found = {
        "hash": DIGEST, "name": "an invented game", "k": 1, "royals": True, "clock": 100,
        "sets": sets, "stalls": stalls,
        "basis": {"build": BUILD, "commit": "38fa5da6", "date": "2026-11-02",
                  "seeds": [0, 1000, 2000, 3000], "cap": 8000000},
    }
    found.update(over)
    return found


def _write(runs: Path, record: dict[str, Any], digest: str = DIGEST) -> None:
    (runs / endings.DIRECTORY).mkdir(parents=True, exist_ok=True)
    (runs / endings.DIRECTORY / f"{digest}.json").write_text(json.dumps(record), "utf-8")


def test_a_record_reads_as_plain_facts_with_its_basis(tmp_path: Path) -> None:
    _write(tmp_path, _record())
    sheet = endings.sheet_for(tmp_path, DIGEST, SPEC, built=BUILD)
    assert sheet is not None
    assert sheet["status"] == "current"
    assert sheet["lines"] == [
        "A crown and an owl against a lone crown: a win cannot be forced.",
        "A crown and an eagle against a lone crown: a win can be forced from 90% of "
        "positions, in at most 16 moves.",
        "With a sparrow added there are too many positions to check, so those endings are "
        "not worked out.",
    ]
    assert sheet["stalls"] == (
        "Of the computer players' 3 games here that were stopped for going on too long "
        "(over four runs), 2 ended with at most one other piece besides the crowns, and in 1 "
        "of those one side had a win it could force and did not take it.")
    assert sheet["stamp"] == "Worked out by computer on 2 November 2026, checking every position."
    assert sheet["measuredAt"] == "2026-11-02" and sheet["build"] == BUILD
    # Facts only: no draw rule, no adjudication, no advice.
    words = " ".join([*sheet["lines"], sheet["stalls"], sheet["intro"]]).lower()
    for never in ("tablebase", "adjudicat", "should", "draw rule"):
        assert never not in words


def test_another_build_marks_the_sheet_provisional(tmp_path: Path) -> None:
    _write(tmp_path, _record())
    sheet = endings.sheet_for(tmp_path, DIGEST, SPEC, built="ffffffffffffffff")
    assert sheet is not None and sheet["status"] == "provisional"
    assert sheet["stamp"].startswith("Provisional: worked out on 2 November 2026")
    html = drawing.endings_html(sheet)
    assert '<span class="tag stale">provisional</span>' in html


def test_a_win_from_a_rare_few_positions_is_said_so_and_one_sided_wins_name_the_side(
    tmp_path: Path,
) -> None:
    record = _record()
    record["sets"][3] = _set("OC v C", [[1, 3], [3]],
                             [_mover(1, 0, 999, win=9), _mover(0, 1, 999, loss=8)],
                             [True, False])
    record["sets"].append(_set("C v OC", [[3], [1, 3]],
                               [_mover(0, 0, 100), _mover(0, 0, 100)], [False, False]))
    _write(tmp_path, record)
    sheet = endings.sheet_for(tmp_path, DIGEST, SPEC, built=BUILD)
    assert sheet is not None
    owl = next(line for line in sheet["lines"] if "owl" in line)
    assert "the first player can force a win and the second cannot" in owl
    assert "only from a rare few positions (fewer than 1 in 100), in at most 5 moves" in owl


def test_bare_royals_are_said_only_where_a_win_can_be_forced(tmp_path: Path) -> None:
    record = _record()
    record["sets"][0] = _set("C v C", [[3], [3]],
                             [_mover(5, 5, 90, win=9, loss=8), _mover(5, 5, 90, win=9, loss=8)],
                             [True, True])
    _write(tmp_path, record)
    sheet = endings.sheet_for(tmp_path, DIGEST, {**SPEC, "rules": {"duck": {"piece": "D"}}},
                              built=BUILD)
    assert sheet is not None
    # A duck game's turn is two plies, so 9 plies are the winner's third move.
    assert sheet["lines"][0] == (
        "A crown against a lone crown, with nothing else on the board: a win can be forced "
        "from 5% of positions, in at most 3 moves.")


def test_no_record_or_nothing_to_say_is_no_sheet(tmp_path: Path) -> None:
    assert endings.sheet_for(tmp_path, DIGEST, SPEC) is None  # no record at all
    assert endings.sheet_for(tmp_path, "not a hash", SPEC) is None
    cases = {
        "aaaaaaaaaaaa": {"refused": "a capture hand"},
        "bbbbbbbbbbbb": {"royals": False},
        "cccccccccccc": {"sets": [{"set": "C v C", "material": [[3], [3], []],
                                   "error": "over the cap"}]},
        "dddddddddddd": {"sets": [_record()["sets"][0], _record()["sets"][4]]},  # only too big
    }
    for digest, over in cases.items():
        _write(tmp_path, _record(hash=digest, **over), digest)
        assert endings.sheet_for(tmp_path, digest, SPEC, built=BUILD) is None, digest
    # A record naming another game, or a piece the spec does not have, says nothing.
    _write(tmp_path, _record(hash="eeeeeeeeeeee"), "ffffffffffff")
    assert endings.sheet_for(tmp_path, "ffffffffffff", SPEC, built=BUILD) is None
    _write(tmp_path, _record())
    assert endings.sheet_for(tmp_path, DIGEST, {**SPEC, "pieces": SPEC["pieces"][:2]},
                             built=BUILD) is None
    (tmp_path / endings.DIRECTORY / "eeeeeeeeeeee.json").write_text("{", "utf-8")
    assert endings.sheet_for(tmp_path, "eeeeeeeeeeee", SPEC) is None


def _drawn(sheet: dict[str, Any] | None, summary: bool = True) -> drawing.Drawn:
    spec = {**SPEC, **({"summary": ["A game of birds."]} if summary else {})}
    return drawing.Drawn(hash=DIGEST, name="an invented game", description="Longer words.",
                         spec=spec, score=None, stale=False, near=(), indexable=True,
                         endings=sheet)


def test_the_server_drawn_page_folds_the_sheet_under_the_full_rules(tmp_path: Path) -> None:
    _write(tmp_path, _record())
    sheet = endings.sheet_for(tmp_path, DIGEST, SPEC, built=BUILD)
    for summary in (True, False):
        html = drawing.view(_drawn(sheet, summary))
        rules = html.index("Full rules")
        assert html.index(endings.TITLE) > rules
        assert "A crown and an eagle against a lone crown" in html
        assert "Worked out by computer on 2 November 2026" in html
        assert "provisional" not in html


def test_the_server_drawn_page_without_a_sheet_says_nothing_of_endings() -> None:
    for summary in (True, False):
        html = drawing.view(_drawn(None, summary))
        assert endings.TITLE not in html and "endings" not in html


def test_the_page_draws_the_servers_sheet_under_the_full_rules() -> None:
    assert "function endingsHtml(sheet)" in PAGE
    assert "${endingsHtml(game.endings)}" in PAGE
    # The script adds no words of its own beyond the provisional tag.
    assert "if (!sheet || !(sheet.lines || []).length) return '';" in PAGE
