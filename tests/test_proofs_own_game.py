"""A proven position must show what makes its game its own (Q1084).

The user, 2026-11-01: proven positions that hold only standard chess pieces are removed
(except in chess itself); each must include one of the elements that make its game special.
`puzzles.distinction` reads that from the spec against a reference game, and `puzzles.shows`
asks the engine what a position and its proven line show. The reference here is
`test_puzzles.SPEC`, a 5x5 probe, so nothing below names chess's pieces: a game is the probe
with a piece added, or with a goal square added, and the probe itself is "the reference".
"""

from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import puzzles
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.site import proven
from tests.test_puzzles import SMALL, SPEC

CROSSER = {"id": "X", "name": "crosser", "value": 3,
           "moves": [{"kind": "leap", "vectors": [[2, 0]], "symmetry": "rotate4"}]}


def _with_piece() -> dict[str, Any]:
    """The probe with a piece it lacks: a two-square leaper a side."""
    spec = copy.deepcopy(SPEC)
    spec["name"] = "probe with a crosser"
    spec["pieces"].append(CROSSER)
    spec["setup"]["placements"] += [{"piece": "X", "player": 0, "squares": ["e1"]},
                                    {"piece": "X", "player": 1, "squares": ["a5"]}]
    return spec


def _with_goal() -> dict[str, Any]:
    """The probe with a rule it lacks and no new piece: a royal reaching c3 wins."""
    spec = copy.deepcopy(SPEC)
    spec["name"] = "probe with a hill"
    spec["rules"]["reach"] = {"piece": "K", "squares": ["c3"], "mirror": False,
                              "result": "win"}
    return spec


def _proved(engine: Engine, board: dict[str, tuple[str, int]]) -> tuple[Any, dict[str, Any]]:
    state = engine.make_state(board, side_to_move=0)
    row = puzzles.verdict(engine, state, nodes=SMALL)
    assert row["result"] == puzzles.WON
    return state, row


# The royal is taken in one by the stepper: nothing in it but the probe's own pieces.
TAKEN = {"a1": ("K", 0), "c3": ("S", 0), "c4": ("K", 1), "e4": ("T", 1)}


def test_the_reference_itself_keeps_every_position() -> None:
    seen = puzzles.distinction(SPEC, SPEC)
    assert seen.same
    engine = Engine(SPEC)
    state, row = _proved(engine, TAKEN)
    assert puzzles.shows(engine, row, seen, nodes=SMALL, state=state)


def test_chess_is_the_reference_and_so_keeps_every_position() -> None:
    """Read off `rulesets/chess.json` because it is the reference the library is sifted
    against, not for anything chess does."""
    reference = puzzles.reference_spec()
    [(_digest, _name, spec)] = [game for game in puzzles.library(".")
                                if game[2] == reference]
    assert puzzles.distinction(spec, reference).same


def test_a_piece_the_reference_lacks_is_read_off_the_spec() -> None:
    spec = _with_piece()
    seen = puzzles.distinction(spec, SPEC)
    assert not seen.same
    names = [Engine(spec).spec.piece_names[index] for index in seen.pieces]
    assert names == ["crosser"]


def test_a_position_of_only_the_references_pieces_is_refused() -> None:
    spec = _with_piece()
    engine = Engine(spec)
    state, row = _proved(engine, TAKEN)
    assert puzzles.shows(engine, row, puzzles.distinction(spec, SPEC), nodes=SMALL,
                         state=state) == []


def test_a_position_with_the_special_piece_is_kept() -> None:
    spec = _with_piece()
    engine = Engine(spec)
    state, row = _proved(engine, {**TAKEN, "e1": ("X", 0)})
    said = puzzles.shows(engine, row, puzzles.distinction(spec, SPEC), nodes=SMALL,
                         state=state)
    assert said and "crosser" in said[0]


def test_a_rule_only_games_line_that_uses_the_rule_is_kept() -> None:
    spec = _with_goal()
    seen = puzzles.distinction(spec, SPEC)
    assert not seen.pieces and [rule for rule, _twin in seen.twins] == ["reach"]
    engine = Engine(spec)
    state, row = _proved(engine, {"b3": ("K", 0), "e5": ("K", 1), "a5": ("T", 1)})
    said = puzzles.shows(engine, row, seen, nodes=SMALL, state=state)
    assert said and "reach" in said[0]


def test_a_rule_only_games_line_that_does_not_use_the_rule_is_refused() -> None:
    spec = _with_goal()
    engine = Engine(spec)
    state, row = _proved(engine, {"a1": ("K", 0), "d4": ("S", 0), "d5": ("K", 1),
                                  "a5": ("T", 1)})
    assert puzzles.key_move(row)["text"].startswith("S")
    assert puzzles.shows(engine, row, puzzles.distinction(spec, SPEC), nodes=SMALL,
                         state=state) == []


# ------------------------------------------------------------ the record (selection)


def _row(ident: str, ply: int) -> dict[str, Any]:
    scores = [{"move": n, "text": f"m{n}", "proof": proof, **({"ends_in": 3} if proof == "won"
                                                               else {})}
              for n, proof in enumerate(["won", "open", "open"])]
    return {"id": ident, "seed": 0, "game": 0, "ply": ply, "result": puzzles.WON,
            "ends_in": 3, "scores": scores, "moves": []}


def test_the_sift_sets_aside_and_deletes_nothing() -> None:
    data = {"positions": [{**_row("a", 1), "shows": ["x"]}, {**_row("b", 2), "shows": []},
                          _row("c", 3)], "counts": {}, "basis": {}}
    puzzles.sifted(data, 7, "built", datetime.date(2026, 11, 1))
    assert [row["id"] for row in data["positions"]] == ["a", "c"]
    assert [row["id"] for row in data["set_aside"]] == ["b"]
    assert data["counts"]["set_aside"] == 1 and puzzles.is_sifted(data)
    # A later sift that finds it shows something brings it back.
    data["set_aside"][0]["shows"] = ["y"]
    puzzles.sifted(data, 7, "built")
    assert [row["id"] for row in data["positions"]] == ["a", "b", "c"]
    assert data["set_aside"] == []


def test_refilter_keeps_the_old_record_and_sets_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    digest = Engine(SPEC).spec.hash
    old = {"format": puzzles.FORMAT, "ruleset_hash": digest, "name": "probe",
           "basis": {}, "counts": {}, "positions": [_row("a", 1), _row("b", 2)]}
    puzzles.write(puzzles.record_path(runs, digest), old)
    answers = {"a": ["something"], "b": []}
    monkeypatch.setattr(puzzles, "shows", lambda _engine, row, _seen, nodes: answers[row["id"]])
    day = datetime.date(2026, 11, 1)
    said = puzzles.refilter(SPEC, runs, workers=1, today=day,
                            reference=_with_piece())
    assert (said["asked"], said["offered"], said["set_aside"]) == (2, 1, 1)
    data = json.loads(puzzles.record_path(runs, digest).read_text(encoding="utf-8"))
    assert [row["id"] for row in data["positions"]] == ["a"]
    assert [row["id"] for row in data["set_aside"]] == ["b"]
    kept = puzzles.work_dir(runs, digest) / "record-2026-11-01.json"
    assert json.loads(kept.read_text(encoding="utf-8")) == old


def test_an_unsifted_record_is_owed_a_re_filter_and_then_is_current(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    digest = Engine(SPEC).spec.hash
    puzzles.write(puzzles.record_path(runs, digest),
                  {"format": puzzles.FORMAT, "ruleset_hash": digest, "name": "probe",
                   "basis": {"proof_version": puzzles.PROOF_VERSION}, "counts": {},
                   "positions": []})
    before = puzzles.standing(digest, "probe", SPEC, runs, seeds=())
    assert before.needs == puzzles.FILTER
    puzzles.refilter(SPEC, runs, workers=1, reference=SPEC)
    after = puzzles.standing(digest, "probe", SPEC, runs, seeds=())
    assert not after.owed, after.reason


# ------------------------------------------------------------------ the site


def test_a_game_left_with_none_says_why_and_counts_what_was_set_aside() -> None:
    record = proven.Record(game="g", made="2026-11-01", basis={}, counts={"won": 3},
                           positions=(), dropped=0, too_easy=1, set_aside=2)
    section = proven.summary(record, None)
    assert section["count"] == 0 and section["none"] == proven.NONE_SHOWN
    assert "2 more were set aside" in section["basis"]
    assert proven.page(record, None)["none"] == proven.NONE_SHOWN
    plain = proven.Record(game="g", made="", basis={}, counts={}, positions=(), dropped=0,
                          too_easy=1)
    assert proven.summary(plain, None)["none"] == proven.NONE_LEFT
