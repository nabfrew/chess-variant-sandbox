"""The speed history's record is append-only, and refuses an unleased reading (R-L5a).

`.claude/scripts/speed_history.py` writes `docs/reference/speed-history.csv`. These pin the two
promises its record rests on, on a temp file and with no bench: an old row is never rewritten,
and a row not taken under `quiet` is refused unless it is marked so.
"""

from __future__ import annotations

from pathlib import Path

import private_paths
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ".claude/scripts/speed_history.py"  # a maintainer's tool: not in the public copy
history = private_paths.script(SCRIPT, "speed_history")
pytestmark = private_paths.needs(SCRIPT)


def _row(game: str, median: str) -> dict[str, str]:
    return {"date": "2026-11-14T00:00Z", "kind": "measured", "game": game, "median_mlps": median, "quiet": "yes"}


def test_append_keeps_every_earlier_byte(tmp_path: Path) -> None:
    record = tmp_path / "speed-history.csv"
    history.append_rows(record, [_row("a", "1.00"), _row("b", "2.00")])
    before = record.read_bytes()
    history.append_rows(record, [_row("a", "1.10")])
    after = record.read_bytes()
    assert after.startswith(before)
    assert after.count(b"\n") == before.count(b"\n") + 1
    rows = history.read_rows(record)
    assert [(row["game"], row["median_mlps"]) for row in rows] == [("a", "1.00"), ("b", "2.00"), ("a", "1.10")]
    assert list(rows[0]) == list(history.FIELDS)


def test_append_refuses_a_file_it_did_not_start(tmp_path: Path) -> None:
    record = tmp_path / "speed-history.csv"
    record.write_text("game,median\nchess,1\n", encoding="utf-8")
    with pytest.raises(history.RecordError):
        history.append_rows(record, [_row("a", "1.00")])
    assert record.read_text(encoding="utf-8") == "game,median\nchess,1\n"


def test_append_refuses_a_hand_edit_without_a_final_newline(tmp_path: Path) -> None:
    record = tmp_path / "speed-history.csv"
    history.append_rows(record, [_row("a", "1.00")])
    record.write_bytes(record.read_bytes().rstrip(b"\n"))
    with pytest.raises(history.RecordError):
        history.append_rows(record, [_row("b", "2.00")])


def test_append_refuses_an_unknown_field(tmp_path: Path) -> None:
    with pytest.raises(history.RecordError):
        history.append_rows(tmp_path / "x.csv", [{"game": "a", "speed": "1"}])
    assert not (tmp_path / "x.csv").exists()


def test_a_reading_not_under_quiet_is_refused_unless_marked() -> None:
    assert history.quiet_mark("quiet", False) == "yes"
    assert history.quiet_mark("quiet", True) == "yes"
    for profile in (None, "heavy", "build", "suite"):
        with pytest.raises(history.RecordError):
            history.quiet_mark(profile, False)
        assert history.quiet_mark(profile, True) == "no"


# --- the registered bar (R-L5c1; docs/reference/common-path-plan.md §4) ---------------------


def _reading(base: dict[str, list[float]], stage: dict[str, list[float]], leaves: int = 100) -> dict:
    def arm(rounds: dict[str, list[float]]) -> dict:
        return {"fingerprint": "f00d", "games": {g: {"rounds": r, "leaves": leaves, "hash": "h"}
                                                 for g, r in rounds.items()}}
    return {"base": arm(base), "stage": arm(stage)}


def test_the_bar_is_the_registered_one() -> None:
    assert history.BATCH_BASE == "34fb96d6a"
    assert history.bar_tolerance("chess") == 1.5
    assert {history.bar_tolerance(name) for name, *_ in history.GAMES if name != "chess"} == {4.0}
    assert history.BAR_ROUNDS == 31


def test_chess_passes_at_its_tolerance_and_misses_past_it() -> None:
    at = history.grade("chess", [36.0, 36.3, 36.5], [35.40, 35.75, 35.9], [])
    assert at.base[0] == 36.3 and at.stage == (35.75, 35.40, 35.9)
    assert at.change == -1.52 and at.verdict == "miss"
    edge = history.grade("chess", [100.0], [98.5], [])
    assert edge.change == -1.5 and edge.verdict == "pass"  # compared at 0.01%: float noise never misses
    assert history.grade("chess", [100.0], [98.49], []).verdict == "miss"
    assert history.grade("chess", [100.0], [103.0], []).change == 3.0


def test_the_other_games_get_four_percent() -> None:
    assert history.grade("atomic", [10.0], [9.6], []).verdict == "pass"
    assert history.grade("atomic", [10.0], [9.59], []).verdict == "miss"
    assert history.grade("dohyo", [50.0, 49.0, 51.0], [47.0, 48.0, 46.0], []).verdict == "miss"  # -4.08%


def test_the_median_decides_not_the_best_round() -> None:
    graded = history.grade("chess", [36.0, 36.0, 36.0, 36.0, 36.0], [30.0, 30.0, 36.0, 37.0, 38.0], [])
    assert graded.stage[0] == 36.0 and graded.verdict == "pass"


def test_a_game_that_uses_the_rule_is_shown_not_graded() -> None:
    graded = history.grade("dohyo", [50.0], [40.0], ["dohyo"])
    assert graded.verdict == "exempt" and graded.change == -20.0
    assert "not graded" in graded.line()
    assert history.bar_passes([graded])


def test_a_reading_is_graded_game_by_game_and_one_miss_misses() -> None:
    result = _reading({"chess": [36.0], "atomic": [17.0]}, {"chess": [35.9], "atomic": [16.0]})
    graded = history.grade_reading(result, "base", "stage", ["chess", "atomic"], [])
    assert [g.verdict for g in graded] == ["pass", "miss"]
    assert not history.bar_passes(graded)
    assert "MISS on atomic" in history.bar_verdict(graded)
    assert "R-L5 debt" in history.bar_verdict(graded) and "R-L5 draw" in history.bar_verdict(graded)
    assert history.bar_verdict(graded[:1]) == "bar: PASS"
    # R-L6e: a debt holds the next rule stage that edits the same function, never the replay.
    verdict = history.bar_verdict(graded)
    assert "the next rule stage that edits the same function does not start" in verdict
    assert "replay is not held" in verdict and "replay does not start" not in verdict


def test_the_bar_refuses_arms_that_play_different_games() -> None:
    result = _reading({"chess": [36.0]}, {"chess": [36.0]})
    result["stage"]["games"]["chess"]["leaves"] = 101
    with pytest.raises(history.RecordError, match="different games"):
        history.grade_reading(result, "base", "stage", ["chess"], [])


def test_the_bar_refuses_an_arm_without_the_game() -> None:
    result = _reading({"chess": [36.0]}, {})
    result["stage"]["games"]["chess"] = {"error": "ValidationError: unknown key", "rounds": [], "leaves": None}
    with pytest.raises(history.RecordError, match="no reading of chess"):
        history.grade_reading(result, "base", "stage", ["chess"], [])


def test_the_bar_refuses_a_reading_not_under_quiet() -> None:
    history.require_quiet("quiet")
    for profile in (None, "heavy", "build", "suite"):
        with pytest.raises(history.RecordError, match="not `quiet`"):
            history.require_quiet(profile)


def test_uses_may_not_name_chess_or_a_stranger() -> None:
    assert history.check_uses(["dohyo", "atomic"]) == ["dohyo", "atomic"]
    with pytest.raises(history.RecordError, match="common by nature"):
        history.check_uses(["chess"])
    with pytest.raises(history.RecordError, match="not history games"):
        history.check_uses(["shogi"])


def test_bar_and_compare_refuse_before_building_or_timing_without_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    def never(*_: object, **__: object) -> None:
        raise AssertionError("built or timed without quiet")

    monkeypatch.setattr(history, "held_profile", lambda: "heavy")
    monkeypatch.setattr(history, "prepare_arm", never)
    monkeypatch.setattr(history, "take_reading", never)
    assert history.main(["bar"]) == 2
    assert history.main(["compare", "--arm", "a=", "--arm", "b=", "--bar", "a"]) == 2
    assert history.main(["compare", "--arm", "a=", "--arm", "b=", "--bar", "a", "--not-quiet"]) == 2


def test_bar_rows_append_the_stage_with_its_grade(tmp_path: Path) -> None:
    result = _reading({"chess": [36.0, 36.3, 36.5], "dohyo": [48.0]}, {"chess": [36.1, 36.2, 36.4], "dohyo": [40.0]},
                      leaves=4865609)
    graded = history.grade_reading(result, "base", "stage", ["chess", "dohyo"], ["dohyo"])
    rows = history.bar_rows(graded, result["stage"], date="2026-11-14T09:00Z", commit="abc123def",
                            source="f00d", base_commit="34fb96d6a", profile_note="pgo: fresh", note="trial")
    record = tmp_path / "speed-history.csv"
    history.append_rows(record, rows)
    back = history.read_rows(record)
    assert [(r["game"], r["median_mlps"], r["rounds"], r["quiet"], r["pgo"]) for r in back] == [
        ("chess", "36.20", "3", "yes", "yes"), ("dohyo", "40.00", "1", "yes", "yes")]
    assert back[0]["fingerprint_current"] == "yes" and back[0]["leaves"] == "4865609"
    assert back[0]["note"].startswith("bar against 34fb96d6a (base 36.30): -0.3%, pass (at most 1.5% slower)")
    assert back[0]["note"].endswith("; pgo: fresh; trial")
    assert "uses the stage's rule, not graded" in back[1]["note"]


def test_an_arm_is_cached_by_what_builds_and_loads_the_core(monkeypatch: pytest.MonkeyPatch) -> None:
    trees = {"a": {"rust": "r1", "chess_variant_sandbox": "p1", "pyproject.toml": "t", "uv.lock": "l"},
             "b": {"rust": "r1", "chess_variant_sandbox": "p1", "pyproject.toml": "t", "uv.lock": "l"},
             "c": {"rust": "r1", "chess_variant_sandbox": "p2", "pyproject.toml": "t", "uv.lock": "l"}}

    def git(*args: str) -> str:
        commit, path = args[1].split(":", 1)
        return trees[commit][path]

    monkeypatch.setattr(history, "_git", git)
    assert history.arm_key("a") == history.arm_key("b")  # a docs commit shares its parent's arm
    assert history.arm_key("a") != history.arm_key("c")  # a compiler change does not


def test_an_arm_profiled_by_another_recipe_is_another_arm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(history, "_git", lambda *args: "object")
    train = tmp_path / "train.py"
    train.write_text('RECIPE = "serial-1"\n', encoding="utf-8")
    monkeypatch.setattr(history, "TRAIN_SCRIPT", train)
    serial = history.arm_key("a")
    train.write_text('RECIPE = "serial-2"\n', encoding="utf-8")
    assert history.arm_key("a") != serial  # R-L5e: the recipe is part of the arm


def test_the_recipe_is_read_off_train_py() -> None:
    assert history.recipe_of('X = 1\nRECIPE = "serial-1"  # note\n') == "serial-1"
    assert history.recipe_of("PERFT = []\n") == "threaded"  # a train.py from before R-L5e


def test_a_committed_profile_serves_only_when_current_and_by_this_recipe() -> None:
    stamp = {"source": "s", "recipe": "serial-1"}
    assert history.committed_profile_serves(True, stamp, "serial-1")
    assert not history.committed_profile_serves(False, stamp, "serial-1")  # stale source or toolchain
    assert not history.committed_profile_serves(True, {"source": "s"}, "serial-1")  # threaded, unstamped
    assert not history.committed_profile_serves(True, stamp, "serial-2")


def test_the_training_runs_on_one_thread() -> None:
    """R-L5e (speed-history.md §4d): threaded training lost counter updates and built different
    code each time; the recipe the bar and the committed profile use is serial."""
    text = (ROOT / "rust" / "pgo" / "train.py").read_text(encoding="utf-8")
    assert history.recipe_of(text).startswith("serial")
    assert '"--workers", "1"' in text and '"--threads", "1"' in text
    assert '"--workers", "4"' not in text


def test_ten_named_games_that_exist() -> None:
    assert len(history.GAMES) == 10
    assert len({name for name, *_ in history.GAMES}) == 10
    for _, file, depth, reps in history.GAMES:
        assert (ROOT / file).is_file()
        assert depth >= 1 and reps >= 1


def test_others_busy_is_the_machine_less_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """R-L5f (speed-history.md §4g): a round logs what else ran, all CPUs' busy less its one CPU."""
    monkeypatch.setattr(history.os, "cpu_count", lambda: 16)
    assert history.others_busy((0, 0), (1500, 1600)) == pytest.approx(0.0, abs=0.11)
    assert history.others_busy((0, 0), (800, 1600)) == pytest.approx(43.8, abs=0.06)
    assert history.others_busy(None, (1, 2)) is None
    assert history.others_busy((5, 5), (5, 5)) is None


def _fake_round(seen: list[str]):  # type: ignore[no-untyped-def]
    def run(games, core, tree):  # type: ignore[no-untyped-def]
        seen.append(core)
        assert Path(core).is_file()
        return {"fingerprint": "f", "core": core, "package": "p", "address": f"0x{len(seen):x}0000",
                "games": {name: {"hash": "h", "leaves": 10, "times": [1e-6]} for name, *_ in games}}
    return run


def test_a_fresh_copy_each_round_is_a_new_file_and_is_removed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """R-L5f: `fresh_copy` loads each arm from a new copy of its core every round, so each round
    draws the file's load placement anew; the copies go when the reading ends."""
    core = tmp_path / "_core.pyd"
    core.write_bytes(b"core bytes")
    seen: list[str] = []
    monkeypatch.setattr(history, "run_round", _fake_round(seen))
    games = [("chess", "rulesets/chess.json", 1, 1)]
    result = history.take_reading({"a": (str(core), None), "b": (str(core), None)}, games, 3, fresh_copy=True)
    assert len(seen) == 6 and len(set(seen)) == 6 and str(core) not in seen
    assert not any(Path(path).exists() for path in seen)
    assert core.read_bytes() == b"core bytes"
    assert len(result["a"]["addresses"]) == 3 and len(result["a"]["others_busy"]) == 3
    assert result["a"]["games"]["chess"]["rounds"] == pytest.approx([10.0] * 3)


def test_a_graded_reading_beside_other_load_is_refused() -> None:
    """R-L5f: `quiet` checks the CPU only when taken; a graded reading with more than one round
    run beside other load (others' CPU over 15%) is refused, not graded."""
    calm = {"a": {"others_busy": [5.0, 16.0, 7.0]}, "b": {"others_busy": [None, 9.0]}}
    history.check_unloaded(calm)  # one loaded round is allowed
    loaded = {"a": {"others_busy": [5.0, 16.0]}, "b": {"others_busy": [31.5]}}
    assert history.loaded_rounds(loaded) == 2
    with pytest.raises(history.RecordError, match="other load"):
        history.check_unloaded(loaded)


def test_without_fresh_copy_every_round_loads_the_one_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    core = tmp_path / "_core.pyd"
    core.write_bytes(b"x")
    seen: list[str] = []
    monkeypatch.setattr(history, "run_round", _fake_round(seen))
    history.take_reading({"a": (str(core), None)}, [("chess", "rulesets/chess.json", 1, 1)], 2)
    assert seen == [str(core), str(core)]
