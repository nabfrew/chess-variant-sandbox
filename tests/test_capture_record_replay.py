"""R-L4b (`docs/reference/capture-effects-design.md` §4): the stored records of every game with a
capture entry, and chess, replay move by move on this core (`tests/capture_record_replay.py`).

Slow: it reads the main checkout's `runs/`, and skips where that holds no record of these games
(a cloud worker, a fresh clone), so the gate re-runs it where they are.
"""

from __future__ import annotations

import capture_record_replay as replay
import pytest

pytestmark = pytest.mark.slow


def test_every_capture_game_s_records_replay_on_this_core() -> None:
    runs = replay.main_runs()
    if not runs.is_dir() or not any(runs.glob("*-*/games.jsonl")):
        pytest.skip(f"no stored records under {runs}")
    assert replay.main(["--runs", str(runs), "--workers", "2", "--limit", "8"]) == 0


def test_the_chosen_games_are_the_capture_games_and_chess() -> None:
    names = {path.stem for path in replay.chosen_files()}
    assert {"chess", "atomic", "crazyhouse", "circe", "einstein", "succession", "the-herd"} <= names
    assert "dohyo" not in names  # a push with no capture entry
    assert {"breather", "watering-hole"} <= names  # R-L4b5: every game in rounds
