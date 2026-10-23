"""The command line's commands that no test ran (archive/plan-tests F9, Q751b).

Each is driven through `cli.main([...])` in-process and judged by what it prints, what it
writes and what it exits with. The ones whose work is a model call, a server or a
campaign (`verdict`, `evolve`, `serve`, `site serve`, `promote`, `premise`'s playing) have
the one expensive call replaced by a recorder, so what is held is the command: how its
flags reach that call, and what it does with the answer. Nothing here calls the API.

`heuristics` and `values` have their own commands tested in `test_heuristics.py` and
`test_valuations.py`; `fairy` in `test_f9_fairy.py`; the `site` commands that need a
database in `test_f9_site_commands.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_variant_sandbox import cli, corpus, premise
from chess_variant_sandbox.agents import available_agents
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.llm.judge import Judgement
from chess_variant_sandbox.loop import LoopConfig
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.tournament import TournamentConfig, run_tournament
from chess_variant_sandbox.tournament.play import play_game
from chess_variant_sandbox.tournament.record import read_records
from chess_variant_sandbox.web.human import HumanRecord

REPO = Path(__file__).resolve().parent.parent
RULESETS = REPO / "rulesets"
DULL = RULESETS / "dull.json"


@pytest.fixture
def at_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    """`rulesets/` is found relative to the working directory, as the commands find it."""
    monkeypatch.chdir(REPO)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small finished run of `dull`: the games and the tournament file."""
    played = run_tournament(
        load_spec(DULL),
        TournamentConfig(agents=("random", "greedy"), games_per_pairing=1, seed=0, workers=1),
    )
    return Path(played.save(tmp_path_factory.mktemp("f9") / "run"))


def say(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestAgents:
    def test_lists_every_registered_kind_and_how_options_are_spelled(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = say(capsys, "agents")
        assert code == 0
        listed = [line.strip() for line in out.splitlines() if line.startswith("  ")]
        assert listed == list(available_agents())
        assert {"random", "greedy", "minimax", "mcts"} <= set(listed)
        assert "kind:key=value,key=value" in out
        assert "budget is always in nodes" in out


class TestBench:
    def test_reports_the_leaf_count_the_engine_counts_and_the_games_it_played(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        engine = Engine(load_spec(DULL))
        leaves = engine.perft(engine.initial_state(), 3)
        code, out, _ = say(capsys, "bench", str(DULL), "--depth", "3", "--games", "7",
                           "--threads", "1")
        assert code == 0
        assert out.startswith(f"dull [{spec_hash(load_spec(DULL))}]  (1 thread(s))")
        assert re.search(rf"perft\(3\)\s+{leaves:,} leaves in [\d.]+s", out)
        match = re.search(r"random playouts\s+(\d+) games, ([\d,]+) plies in", out)
        assert match and match.group(1) == "7" and int(match.group(2).replace(",", "")) > 0


class TestPlay:
    def test_prints_every_move_of_the_game_the_engine_plays_and_its_ending(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        engine = Engine(load_spec(DULL))
        record = play_game(engine, ("greedy", "random"), 3)
        code, out, _ = say(capsys, "play", str(DULL), "--agents", "greedy", "random",
                           "--seed", "3")
        assert code == 0
        numbered = [line for line in out.splitlines() if re.match(r"\d+\.", line)]
        assert len(numbered) == record.plies
        state = engine.initial_state()
        first = engine.describe_move(state, record.moves[0])
        assert numbered[0] == f"1. {first}"
        assert numbered[1].startswith("1... ")
        who = "draw" if record.winner is None else f"player {record.winner} ({record.agents[record.winner]})"
        assert f"{who} by {record.reason} in {record.plies} plies" in out

    def test_quiet_prints_only_the_ending(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out, _ = say(capsys, "play", str(DULL), "--seed", "1", "--quiet")
        assert not re.search(r"^\d+\.", out, re.M)
        assert " plies (" in out

    def test_the_same_seed_is_the_same_game(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, one, _ = say(capsys, "play", str(DULL), "--seed", "5")
        _, two, _ = say(capsys, "play", str(DULL), "--seed", "5")
        strip = lambda text: re.sub(r"\(\d+ ms\)", "", text)  # noqa: E731 - the clock differs
        assert strip(one) == strip(two)


class TestSmoke:
    def test_a_game_that_passes_prints_its_stats_and_exits_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = say(capsys, "smoke", str(DULL), "--games", "6", "--workers", "1")
        assert code == 0
        assert out.splitlines()[0] == f"dull [{spec_hash(load_spec(DULL))}]: passed"
        assert re.search(r"^  games\s+6\.0$", out, re.M)
        assert "mean_plies" in out

    def test_a_game_decided_before_it_begins_is_rejected_with_its_reasons_and_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        spec = load_spec(DULL)
        spec["rules"]["max_plies"] = 4
        short = tmp_path / "short.json"
        short.write_text(json.dumps(spec), encoding="utf-8")
        code, out, _ = say(capsys, "smoke", str(short), "--games", "6", "--workers", "1")
        assert code == 1
        assert out.splitlines()[0].endswith(": REJECTED")
        assert "  - 100% of random games ended on the turn limit" in out
        assert "forced_rate" in out


class TestAnalyse:
    def test_replays_a_run_and_writes_analysis_beside_it(
        self, run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        copy = tmp_path / "run"
        copy.mkdir()
        (copy / "games.jsonl").write_bytes((run / "games.jsonl").read_bytes())
        code, out, _ = say(capsys, "analyse", str(copy), "--ruleset", str(DULL),
                           "--blunders", "1", "--workers", "1")
        assert code == 0
        records = list(read_records(copy / "games.jsonl"))
        # only a game both of whose seats chose is analysed: the random mover's is not
        games = sum(1 for r in records if all(a.split(":")[0] != "random" for a in r.agents))
        assert 0 < games < len(records)
        assert re.search(rf"^dull: {games} games analysed, \d+ positions scored, "
                         r"\d+ blunder trials in [\d.]+s$", out, re.M)
        assert f"wrote {copy / 'analysis.json'}" in out
        written = json.loads((copy / "analysis.json").read_text(encoding="utf-8"))
        assert len(written["games"]) == games

    def test_a_games_file_writes_next_to_it_and_the_spec_is_found_by_hash(
        self, run: Path, tmp_path: Path, at_the_repository: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        target = tmp_path / "elsewhere"
        target.mkdir()
        games = target / "games.jsonl"
        games.write_bytes((run / "games.jsonl").read_bytes())
        code, out, _ = say(capsys, "analyse", str(games), "--blunders", "0", "--workers", "1")
        assert code == 0 and (target / "analysis.json").exists()
        assert out.startswith("dull: ")

    def test_a_run_with_no_games_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "games.jsonl").write_text("", encoding="utf-8")
        code, _, err = say(capsys, "analyse", str(tmp_path))
        assert code == 2 and err.startswith("error: no games found in ")


class TestFindRuleset:
    def test_finds_a_committed_game_by_its_content_hash(self, at_the_repository: None) -> None:
        found = cli._find_ruleset(spec_hash(load_spec(DULL)))
        assert found["name"] == "dull"

    def test_a_renamed_copy_is_the_same_game_and_files_that_are_not_specs_are_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rulesets").mkdir()
        (tmp_path / "rulesets" / "broken.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "rulesets" / "list.json").write_text("[1, 2]", encoding="utf-8")
        copy = load_spec(DULL)
        copy["name"] = "dull by another name"
        (tmp_path / "rulesets" / "zz.json").write_text(json.dumps(copy), encoding="utf-8")
        assert cli._find_ruleset(spec_hash(copy))["name"] == "dull by another name"

    def test_the_design_runs_archive_is_searched_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "runs" / "archive" / "deep").mkdir(parents=True)
        spec = load_spec(DULL)
        (tmp_path / "runs" / "archive" / "deep" / "x.json").write_text(
            json.dumps(spec), encoding="utf-8")
        assert cli._find_ruleset(spec_hash(spec))["name"] == "dull"

    def test_an_unknown_hash_says_where_it_looked_and_how_to_point_at_the_spec(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="no ruleset with hash abc123 under rulesets/ "
                                                   "or runs/archive/; pass --ruleset"):
            cli._find_ruleset("abc123")


class TestViz:
    def test_writes_a_page_for_the_chosen_game_next_to_the_run(
        self, run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        copy = tmp_path / "run"
        copy.mkdir()
        (copy / "games.jsonl").write_bytes((run / "games.jsonl").read_bytes())
        code, out, _ = say(capsys, "viz", str(copy), "--ruleset", str(DULL), "--game", "1",
                           "--letters")
        assert code == 0
        page = copy / "game-1.html"
        assert out.strip() == f"wrote {page}"
        html = page.read_text(encoding="utf-8")
        assert html.lstrip().lower().startswith("<!doctype html")
        assert "dull" in html

    def test_out_names_the_file_and_the_spec_is_found_by_the_records_hash(
        self, run: Path, tmp_path: Path, at_the_repository: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        target = tmp_path / "board.html"
        code, out, _ = say(capsys, "viz", str(run / "games.jsonl"), "--out", str(target))
        assert code == 0 and target.exists() and out.strip() == f"wrote {target}"

    def test_a_game_index_outside_the_run_is_refused_with_the_range(
        self, run: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        count = len(list(read_records(run / "games.jsonl")))
        code, _, err = say(capsys, "viz", str(run), "--ruleset", str(DULL), "--game", str(count))
        assert code == 2
        assert f"game index {count} outside 0..{count - 1}" in err

    def test_an_empty_run_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "games.jsonl").write_text("", encoding="utf-8")
        code, _, err = say(capsys, "viz", str(tmp_path))
        assert code == 2 and "no games found in" in err


class TestMutants:
    def test_writes_one_spec_per_mutant_and_a_manifest_that_says_what_each_expects(
        self, tmp_path: Path, at_the_repository: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out_dir = tmp_path / "mutants"
        code, out, _ = say(capsys, "mutants", "--out", str(out_dir))
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest, "no mutant was built"
        files = sorted(p.stem for p in out_dir.glob("*.json") if p.name != "manifest.json")
        assert files == sorted(row["name"] for row in manifest)
        assert f"\n{len(manifest)} mutants written to {out_dir}" in out
        for row in manifest:
            spec = json.loads((out_dir / f"{row['name']}.json").read_text(encoding="utf-8"))
            assert row["hash"] == spec_hash(spec)
            assert row["base"] and row["expectation"] and row["rationale"]
            assert row["valid"] is (not row["errors"])
        assert code == (0 if all(row["valid"] for row in manifest) else 1)

    def test_an_invalid_mutant_is_flagged_and_turns_the_exit_code(
        self, tmp_path: Path, at_the_repository: None, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chess_variant_sandbox.metrics import mutants

        real = mutants.build_mutants

        def broken(bases: dict[str, Any], only: tuple[str, ...] = ()) -> Any:
            built = real(bases, only)
            mutant, spec = built[0]
            spec = dict(spec)
            spec["board"] = {"width": 0, "height": 0}
            return [(mutant, spec), *built[1:]]

        monkeypatch.setattr(mutants, "build_mutants", broken)
        code, out, _ = say(capsys, "mutants", "--out", str(tmp_path / "m"))
        assert code == 1
        assert "  INVALID: " in out
        manifest = json.loads((tmp_path / "m" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest[0]["valid"] is False and manifest[0]["errors"]


class TestEvolve:
    def test_the_flags_become_the_loops_configuration_and_nothing_is_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import chess_variant_sandbox.loop as loop

        seen: list[LoopConfig] = []
        monkeypatch.setattr(loop, "evolve", lambda config: seen.append(config))
        code = cli.main([
            "evolve", "--generations", "3", "--agents", "random", "minimax:nodes=100",
            "minimax:nodes=400", "--games", "2", "--workers", "1", "--seed", "9",
            "--no-judge", "--mechanical-rate", "1.0", "--from", "a.json", "b.json",
            "--batch", "4", "--promote", "2", "--out", str(tmp_path),
        ])
        assert code == 0
        (config,) = seen
        assert config.generations == 3
        assert config.agents == ("random", "minimax:nodes=100", "minimax:nodes=400")
        assert (config.games_per_pairing, config.workers, config.seed) == (2, 1, 9)
        assert config.judge is False and config.mechanical_rate == 1.0
        assert config.seeds == (Path("a.json"), Path("b.json"))
        assert (config.batch, config.promote, config.directory) == (4, 2, tmp_path)

    def test_the_defaults_leave_the_loops_own_roster_and_judge_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import chess_variant_sandbox.loop as loop

        seen: list[LoopConfig] = []
        monkeypatch.setattr(loop, "evolve", lambda config: seen.append(config))
        assert cli.main(["evolve"]) == 0
        assert seen[0].agents == LoopConfig.agents and seen[0].seeds == ()
        assert seen[0].judge is True


class TestServe:
    def test_the_local_application_is_started_with_the_flags_it_was_given(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import chess_variant_sandbox.web as web

        seen: list[dict[str, Any]] = []
        monkeypatch.setattr(web, "serve", lambda **kwargs: seen.append(kwargs))
        assert cli.main([
            "serve", "--port", "9", "--host", "0.0.0.0", "--rulesets", "r", "--runs", "u",
            "--archive", "a", "--heuristics", "h", "--browser", "b", "--open",
        ]) == 0
        assert seen == [{
            "host": "0.0.0.0", "port": 9, "ruleset_dir": Path("r"), "runs_dir": Path("u"),
            "archive_dir": Path("a"), "open_browser": True, "heuristics_dir": Path("h"),
            "browser_dir": Path("b"),
        }]

    def test_by_default_it_is_localhost_and_does_not_open_a_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import chess_variant_sandbox.web as web

        seen: list[dict[str, Any]] = []
        monkeypatch.setattr(web, "serve", lambda **kwargs: seen.append(kwargs))
        assert cli.main(["serve"]) == 0
        assert seen[0]["host"] == "127.0.0.1" and seen[0]["open_browser"] is False
        assert seen[0]["port"] == 8765

    def test_the_public_site_is_started_with_the_same_directories_and_its_own_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import chess_variant_sandbox.site as site

        seen: list[dict[str, Any]] = []
        monkeypatch.setattr(site, "serve", lambda **kwargs: seen.append(kwargs))
        assert cli.main(["site", "serve", "--port", "7", "--runs", "u"]) == 0
        assert seen == [{
            "host": "127.0.0.1", "port": 7, "ruleset_dir": Path("rulesets"),
            "runs_dir": Path("u"), "archive_dir": corpus.ARCHIVE_DIR,
            "heuristics_dir": Path("heuristics"), "browser_dir": Path("build") / "browser",
        }]
        assert cli.main(["site", "serve"]) == 0
        assert seen[1]["port"] == 8766


class TestVerdict:
    VERDICT = Judgement(
        verdict="Quiet, and nothing in it to decide.",
        strengths=["short"], weaknesses=["no choices", "no tension"],
        suspected_degeneracy="every game is the same draw",
    )

    @pytest.fixture
    def judged(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        import chess_variant_sandbox.llm as llm

        calls: list[dict[str, Any]] = []

        def judge(spec: dict[str, Any], records: list[Any], sample: int) -> Judgement:
            calls.append({"spec": spec["name"], "games": len(records), "sample": sample})
            return self.VERDICT

        monkeypatch.setattr(llm, "judge_ruleset", judge)
        return calls

    def test_prints_the_prose_and_writes_it_beside_the_run_under_the_games_hash(
        self, run: Path, tmp_path: Path, judged: list[dict[str, Any]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        copy = tmp_path / "run"
        copy.mkdir()
        (copy / "games.jsonl").write_bytes((run / "games.jsonl").read_bytes())
        code, out, _ = say(capsys, "verdict", str(copy), "--ruleset", str(DULL), "--sample", "2")
        assert code == 0
        digest = spec_hash(load_spec(DULL))
        assert out.startswith(f"dull [{digest}]\n\nQuiet, and nothing in it to decide.")
        assert "  s: short" in out and "  w: no choices" in out and "  w: no tension" in out
        assert "suspected degeneracy: every game is the same draw" in out
        assert judged == [{"spec": "dull", "games": len(list(read_records(copy / "games.jsonl"))),
                           "sample": 2}]
        written = json.loads((copy / "verdict.json").read_text(encoding="utf-8"))
        assert written == {"ruleset_hash": digest, **self.VERDICT.as_dict()}
        assert f"wrote {copy / 'verdict.json'}" in out

    def test_a_games_file_is_judged_and_nothing_is_written(
        self, run: Path, tmp_path: Path, judged: list[dict[str, Any]],
        capsys: pytest.CaptureFixture[str], at_the_repository: None,
    ) -> None:
        code, out, _ = say(capsys, "verdict", str(run / "games.jsonl"))
        assert code == 0 and "wrote" not in out
        assert not (run / "verdict.json").exists()
        assert judged[0]["sample"] > 0

    def test_a_judgement_with_no_suspicion_prints_no_suspicion(
        self, run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        at_the_repository: None,
    ) -> None:
        import chess_variant_sandbox.llm as llm

        monkeypatch.setattr(llm, "judge_ruleset", lambda *a, **k: Judgement(verdict="Fine."))
        _, out, _ = say(capsys, "verdict", str(run / "games.jsonl"))
        assert "suspected degeneracy" not in out

    def test_a_run_with_no_games_is_refused_before_a_model_is_asked(
        self, tmp_path: Path, judged: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "games.jsonl").write_text("", encoding="utf-8")
        code, _, err = say(capsys, "verdict", str(tmp_path))
        assert code == 2 and "no games found" in err and judged == []


class TestPromote:
    def test_a_game_already_in_the_corpus_is_refused_by_name(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = say(capsys, "promote", spec_hash(load_spec(DULL)), "--reason", "x",
                           "--runs", "no-such-runs")
        assert code == 2 and err.startswith("error: dull is already in ")

    def test_an_unknown_hash_is_refused(self, tmp_path: Path,
                                        capsys: pytest.CaptureFixture[str]) -> None:
        code, _, err = say(capsys, "promote", "feedfeedfeed", "--runs", str(tmp_path))
        assert code == 2 and "no game with hash feedfeedfeed" in err

    def test_an_admitted_game_is_announced_with_where_it_went_and_what_it_owes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        candidate = SimpleNamespace(name="lancers", hash="abc123abc123")
        asked: list[dict[str, Any]] = []
        monkeypatch.setattr(corpus, "find", lambda digest, runs_dir: candidate)

        def promote(found: Any, **kwargs: Any) -> Path:
            asked.append({"found": found, **kwargs})
            return tmp_path / "lancers.json"

        monkeypatch.setattr(corpus, "promote", promote)
        code, out, _ = say(capsys, "promote", "abc123abc123", "--reason", "it pins a hop",
                           "--name", "Lancers", "--runs", "R", "--out", str(tmp_path))
        assert code == 0
        assert asked == [{
            "found": candidate, "reason": "it pins a hop", "ruleset_dir": tmp_path,
            "runs_dir": Path("R"), "name": "Lancers",
        }]
        assert f"promoted lancers [abc123abc123] -> {tmp_path / 'lancers.json'}" in out
        assert "rulesets-corpus.md" in out


class TestHumans:
    @pytest.fixture
    def played(self, tmp_path: Path) -> Path:
        """Two finished sessions and one abandoned, and a picker table for the game."""
        record = HumanRecord(tmp_path)
        for session, seat, winner, opponent in (
            ("a", 0, 0, "random"), ("b", 1, None, "minimax:nodes=100"),
        ):
            record.note_start(session, "h1", "lancers", opponent, seat, 0)
            record.note_result(session, "h1", "lancers", opponent, seat, 0, 10, winner, "capture")
        record.note_start("c", "h1", "lancers", "random", 0, 0)
        opponents = tmp_path / "opponents"
        opponents.mkdir()
        (opponents / "h1.json").write_text(json.dumps({"standings": [
            {"name": "random", "rating": -200.0}, {"name": "minimax:nodes=100", "rating": 150.0},
            {"name": "ignored", "rating": "n/a"}]}), encoding="utf-8")
        return tmp_path

    def test_with_nothing_recorded_it_says_where_to_play_and_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = say(capsys, "humans", "--runs", str(tmp_path))
        assert code == 1
        assert f"no human games recorded under {tmp_path / 'human' / 'games.jsonl'}" in out
        assert "serve --open" in out

    def test_json_is_the_summary_per_ruleset_placed_on_the_picker_ladder(
        self, played: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = say(capsys, "humans", "--runs", str(played), "--json")
        assert code == 0
        [summary] = json.loads(out)
        assert (summary["hash"], summary["name"]) == ("h1", "lancers")
        assert (summary["started"], summary["finished"], summary["abandoned"]) == (3, 2, 1)
        assert summary["score"] == 0.75
        by = {row["agent"]: row for row in summary["byOpponent"]}
        assert by["random"]["rating"] == -200.0 and by["random"]["score"] == 1.0
        assert by["minimax:nodes=100"]["rating"] == 150.0 and by["minimax:nodes=100"]["score"] == 0.5
        assert summary["anchor"]["top"] == "minimax:nodes=100"

    def test_the_table_says_played_left_score_and_the_gap_to_the_top_rung(
        self, played: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = say(capsys, "humans", "--runs", str(played))
        assert code == 0
        assert out.startswith("3 sessions across 1 rulesets, from ")
        [row] = [line for line in out.splitlines() if re.match(r"  lancers\s+\d", line)]
        cells = row.split()
        assert cells[:4] == ["lancers", "2", "1", "0.750"]
        _, as_json, _ = say(capsys, "humans", "--runs", str(played), "--json")
        anchor = json.loads(as_json)[0]["anchor"]
        assert cells[4] == f"{anchor['rating']:.0f}"  # where the person sits
        assert cells[5] == "150"  # the top rung's rating
        assert cells[6] == f"{anchor['top_above_human']:+.0f}"  # how far above them it is
        assert "'left' is games begun and never finished" in out
        assert "you scored 0.500   their elo 150" in out
        assert "you scored 1.000   their elo -200" in out

    def test_a_ruleset_with_no_picker_table_has_no_anchor_and_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        record = HumanRecord(tmp_path)
        record.note_result("a", "h2", "bare", "random", 0, 0, 4, 1, "capture")
        _, out, _ = say(capsys, "humans", "--runs", str(tmp_path))
        [row] = [line for line in out.splitlines() if re.match(r"  bare\s+\d", line)]
        assert row.split()[1:] == ["1", "0", "0.000", "--", "--", "--"]
        assert "their elo --" in out

    def test_one_ruleset_can_be_asked_for_alone(
        self, played: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        HumanRecord(played).note_result("z", "h9", "other", "random", 0, 0, 4, 0, "capture")
        _, out, _ = say(capsys, "humans", "--runs", str(played), "--ruleset", "h9", "--json")
        assert [entry["hash"] for entry in json.loads(out)] == ["h9"]


@pytest.fixture(scope="module")
def warm_premise() -> None:
    """The first basis of a process digests the engine binary: paid here, not by a test."""
    premise.PremiseBasis.current(spec_hash(load_spec(DULL)), load_spec(DULL))


@pytest.mark.usefixtures("warm_premise")
class TestPremise:
    """`premise` reads what is on file, or plays it; playing is replaced by a stand-in."""

    DIGEST = spec_hash(load_spec(DULL))

    @pytest.fixture(autouse=True)
    def a_built_core(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "stale_build", lambda: None)

    def _payload(self, verdict: str = premise.PASSES, fairy_played: bool = False) -> dict[str, Any]:
        spec = load_spec(DULL)
        basis = premise.PremiseBasis.current(self.DIGEST, spec)
        part = {"candidate": basis.player, "reference": "r", "score": 0.4, "error": 0.02,
                "games": 32, "passes": True}
        return {
            "hash": self.DIGEST, "name": "dull", "verdict": verdict, "ceiling": part,
            "fairy": ({"played": True, **part, "reference": "fairystockfish:nodes=1"}
                      if fairy_played else {"played": False, "why": "no binary here"}),
            "basis": basis.as_dict(),
        }

    def run(self, capsys: pytest.CaptureFixture[str], tmp_path: Path, *argv: str
            ) -> tuple[int, str]:
        code = cli.main(["premise", str(DULL), "--runs", str(tmp_path), *argv])
        return code, capsys.readouterr().out

    def test_check_with_nothing_on_file_says_so_and_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out = self.run(capsys, tmp_path, "--check")
        assert code == 1
        assert f"[{self.DIGEST}] not checked yet" in out
        assert "1 of 1 unchecked or stale" in out

    def test_check_with_a_current_verdict_on_file_reads_it_and_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        premise.write(self._payload(), tmp_path)
        code, out = self.run(capsys, tmp_path, "--check")
        assert code == 0 and "0 of 1 unchecked or stale" in out
        found = premise.read(self.DIGEST, tmp_path, load_spec(DULL))
        assert found is not None and found.recorded in out

    def test_check_with_a_stale_verdict_says_why_and_that_the_last_one_failed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        payload = self._payload(premise.FAILS)
        payload["basis"] = {**payload["basis"], "player": "minimax:nodes=1"}
        premise.write(payload, tmp_path)
        code, out = self.run(capsys, tmp_path, "--check")
        assert code == 1
        found = premise.read(self.DIGEST, tmp_path, load_spec(DULL))
        assert found is not None and not found.current and found.reasons
        assert "stale: " + "; ".join(found.reasons) + " (last verdict failed)" in out

    def test_a_missing_verdict_is_played_written_and_reported_with_its_scores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        asked: list[Any] = []
        monkeypatch.setattr(premise, "check",
                            lambda spec, workers: asked.append(workers) or self._payload())
        code, out = self.run(capsys, tmp_path, "--workers", "3")
        assert code == 0 and asked == [3]
        assert (tmp_path / self.DIGEST / premise.PREMISE_FILE).exists()
        assert f"[{self.DIGEST}] {premise.PASSES}: 0.400 +/- 0.020 against r" in out
        assert "Fairy-Stockfish not played: no binary here" in out

    def test_a_played_fairy_half_is_reported_beside_the_ceiling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(premise, "check",
                            lambda spec, workers: self._payload(fairy_played=True))
        _, out = self.run(capsys, tmp_path)
        assert "; 0.400 +/- 0.020 against fairystockfish:nodes=1" in out

    def test_a_current_verdict_is_not_played_again_unless_forced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        premise.write(self._payload(), tmp_path)
        played: list[int] = []
        monkeypatch.setattr(premise, "check",
                            lambda spec, workers: played.append(1) or self._payload())
        self.run(capsys, tmp_path)
        assert played == []
        self.run(capsys, tmp_path, "--force")
        assert played == [1]

    def test_a_core_behind_its_sources_refuses_to_play(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(cli, "stale_build", lambda: "the core is behind rust/")
        monkeypatch.setattr(premise, "check", lambda spec, workers: pytest.fail("played"))
        code, out = self.run(capsys, tmp_path)
        assert code == 1 and "the core is behind rust/" in out

    def test_a_game_that_cannot_be_found_is_reported_and_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli.main(["premise", "feedfeedfeed", "--runs", str(tmp_path)])
        out = capsys.readouterr().out
        assert code == 1 and out.startswith("  feedfeedfeed: no game with hash feedfeedfeed")

    def test_a_check_that_cannot_be_played_is_reported_and_exits_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def refuse(spec: Any, workers: int) -> None:
            raise ValueError("no searching player in the roster")

        monkeypatch.setattr(premise, "check", refuse)
        code, out = self.run(capsys, tmp_path)
        assert code == 1 and "no searching player in the roster" in out
        assert not (tmp_path / self.DIGEST).exists()


class TestMeasurePooled:
    """One pooled game: what `_measure_candidate` said is kept, and a refusal is a line."""

    def test_it_runs_on_one_core_without_force_and_returns_what_was_said(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[dict[str, Any]] = []

        def measure(candidate: Any, runs: Path, weights: str, **kwargs: Any) -> None:
            seen.append({"candidate": candidate, "runs": runs, "weights": weights, **kwargs})
            kwargs["say"]("measured")

        monkeypatch.setattr(cli, "_measure_candidate", measure)
        candidate = SimpleNamespace(hash="abc")
        lines, refusal = cli._measure_pooled(candidate, Path("R"), "v12", (0, 1))  # type: ignore[arg-type]
        assert (lines, refusal) == (["measured"], "")
        [call] = seen
        assert (call["workers"], call["force"], call["adopt"], call["seeds"]) == (1, False, None, (0, 1))
        assert (call["runs"], call["weights"], call["candidate"]) == (Path("R"), "v12", candidate)

    def test_a_refusal_comes_back_as_a_line_with_what_had_been_said_before_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def measure(candidate: Any, runs: Path, weights: str, **kwargs: Any) -> None:
            kwargs["say"]("started")
            raise ValueError("cannot measure")

        monkeypatch.setattr(cli, "_measure_candidate", measure)
        lines, refusal = cli._measure_pooled(SimpleNamespace(hash="abc"), Path("R"), "v12", None)  # type: ignore[arg-type]
        assert lines == ["started"] and refusal == "  abc: cannot measure"
