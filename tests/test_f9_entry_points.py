"""Entry points and drivers that nothing in the suite ran (archive/plan-tests F9, Q751b).

`python -m ...fuzz`, `python -m ...history deliver`, the page's native stand-in
(`web/page/measure.drive`), the refresher's `git worktree` creation and its rebuild, and the
designer's two helpers that put a refusal and a piece's live moves into words and squares.
Each is driven the way its caller drives it: through `main(argv)`, with a recording stand-in
where the real thing is a measurement, a build or a hundred games.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import corpus, fuzz, history, refresher
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec.plain import plain_problems
from chess_variant_sandbox.spec.schema import SpecError
from chess_variant_sandbox.web import designing
from chess_variant_sandbox.web.page import measure

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"


def load(name: str) -> dict[str, Any]:
    path = RULESETS / f"{name}.json"
    if not path.exists():  # moved to the Library (Q512)
        path = RULESETS.parent / "archive" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


class TestFuzzMain:
    def test_it_walks_the_seeds_it_is_given_and_prints_the_coverage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert fuzz.main(["--cases", "2", "--start", "6", "--plies", "3", "--limit", "2"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("2 specs generated, ")
        assert "rule pairs seen active together:" in out

    def test_the_arguments_reach_the_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[Any, ...]] = []

        class Quiet:
            def report(self, limit: int) -> str:
                seen.append(("limit", limit))
                return "report"

        def fake_run(seeds: Any, plies: int) -> Quiet:
            seen.append((list(seeds), plies))
            return Quiet()

        monkeypatch.setattr(fuzz, "run", fake_run)
        assert fuzz.main(["--cases", "2", "--start", "10", "--plies", "7", "--limit", "5"]) == 0
        assert seen == [([10, 11], 7), ("limit", 5)]

    def test_an_oracle_failure_is_printed_to_stderr_and_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def fail(seeds: Any, plies: int) -> None:
            raise fuzz.OracleFailure("seed 3: the key drifted")

        monkeypatch.setattr(fuzz, "run", fail)
        assert fuzz.main(["--cases", "1"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == "FAILED\nseed 3: the key drifted\n"


class TestHistoryMain:
    @pytest.mark.parametrize("argv", [[], ["deliver"], ["deliver", "a"], ["copy", "a", "b"]])
    def test_anything_but_deliver_with_two_directories_is_a_usage_error(
        self, argv: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert history.main(argv) == 2
        assert "usage: python -m chess_variant_sandbox.history deliver" in capsys.readouterr().err

    def test_deliver_passes_both_directories_the_hashes_and_the_convergence_judge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake(source: Path, runs: Path, digests: Any, judge: Any, say: Any) -> list[str]:
            calls.append({"source": source, "runs": runs, "digests": digests,
                          "judge": judge, "say": say})
            return []

        monkeypatch.setattr(history, "deliver", fake)
        assert history.main(["deliver", "from", "to", "abc", "def"]) == 0
        assert history.main(["deliver", "from", "to"]) == 0
        first, second = calls
        assert (first["source"], first["runs"]) == (Path("from"), Path("to"))
        assert first["digests"] == ["abc", "def"]
        assert second["digests"] is None, "no hashes means every game"
        assert first["judge"] is corpus.judge_convergence
        assert first["say"] is print

    def test_delivering_between_two_empty_trees_delivers_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        assert history.main(["deliver", str(tmp_path / "a"), str(tmp_path / "b")]) == 0
        assert capsys.readouterr().err == ""


class TestPageDriver:
    """`measure.drive` goes through `gather_request` only, so a scripted gatherer shows it."""

    def test_every_unit_is_run_and_its_reply_goes_back_under_its_phase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[dict[str, Any]] = []
        script = iter([
            {"result": None, "state": "S1", "units": [
                {"phase": "fit", "request": "a"}, {"phase": "fit", "request": "b"},
                {"phase": "tournament", "request": "c"}]},
            {"result": None, "state": "S2", "units": [{"phase": "reading", "request": "d"}]},
            {"result": {"index": 0.5}, "state": "S3", "units": []},
        ])

        def gather(text: str, cut: int) -> str:
            asked.append({"asked": json.loads(text), "cut": cut})
            return json.dumps(next(script))

        monkeypatch.setattr(measure.slices, "gather_request", gather)
        ran: list[str] = []

        def run(request: str) -> str:
            ran.append(request)
            return request.upper()

        spec = {"name": "x"}
        assert measure.drive(spec, run=run, cut=9) == {"index": 0.5}
        assert ran == ["a", "b", "c", "d"], "units run in the order they were dealt"
        assert asked[0] == {"asked": {"spec": spec, "state": None}, "cut": 9}
        assert asked[1]["asked"] == {
            "state": "S1", "replies": {"fit": ["A", "B"], "tournament": ["C"]}}
        assert asked[2]["asked"] == {"state": "S2", "replies": {"reading": ["D"]}}
        assert {entry["cut"] for entry in asked} == {9}

    def test_a_result_on_the_first_answer_runs_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            measure.slices, "gather_request",
            lambda text, cut: json.dumps({"result": {"done": True}, "state": "", "units": []}),
        )

        def never(_request: str) -> str:
            raise AssertionError("no unit was dealt")

        assert measure.drive({}, run=never) == {"done": True}


class TestRefresherWorktree:
    @pytest.fixture
    def main(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        config = tmp_path / "gitconfig"
        config.write_text("[user]\n\tname = t\n\temail = t@t\n[commit]\n\tgpgsign = false\n",
                          encoding="utf-8")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
        monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        root = tmp_path / "main"
        root.mkdir()
        git(root, "init", "-q", "-b", "main")
        (root / "a.txt").write_text("a", encoding="utf-8")
        git(root, "add", "-A")
        git(root, "commit", "-qm", "one")
        return root

    def machine(self, main: Path) -> refresher.GitMachine:
        return refresher.GitMachine(
            main_checkout=main, worktree=main / ".claude" / "worktrees" / "refresher"
        )

    def test_create_makes_the_branch_from_main_and_locks_the_worktree(self, main: Path) -> None:
        machine = self.machine(main)
        assert not machine.exists()
        machine.create()
        assert machine.exists()
        assert git(machine.worktree, "rev-parse", "--abbrev-ref", "HEAD").strip() == refresher.BRANCH
        assert machine.head() == git(main, "rev-parse", "main").strip()
        listed = git(main, "worktree", "list", "--porcelain")
        block = next(b for b in listed.split("\n\n") if "refresher" in b.splitlines()[0])
        assert any(line.startswith("locked") for line in block.splitlines()), (
            "prune_worktrees.py must never take the refresher's own checkout"
        )

    def test_create_starts_from_main_even_when_the_main_checkout_is_on_another_branch(
        self, main: Path
    ) -> None:
        git(main, "checkout", "-q", "-b", "elsewhere")
        (main / "c.txt").write_text("c", encoding="utf-8")
        git(main, "add", "-A")
        git(main, "commit", "-qm", "not on main")
        machine = self.machine(main)
        machine.create()
        assert machine.head() == git(main, "rev-parse", "main").strip()
        assert machine.head() != git(main, "rev-parse", "HEAD").strip()

    def test_create_reuses_a_branch_that_already_exists_where_it_stands(self, main: Path) -> None:
        git(main, "branch", refresher.BRANCH)
        (main / "b.txt").write_text("b", encoding="utf-8")
        git(main, "add", "-A")
        git(main, "commit", "-qm", "two")
        machine = self.machine(main)
        machine.create()
        # the old branch was left where it was; the worktree is behind main and says so
        assert machine.head() != git(main, "rev-parse", "main").strip()
        assert machine.behind()

    def test_rebuild_reinstalls_the_package_in_the_worktree_and_says_nothing_on_success(
        self, tmp_path: Path
    ) -> None:
        calls: list[tuple[list[str], Path]] = []

        def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append((command, kwargs["cwd"]))
            return subprocess.CompletedProcess(command, 0, "", "")

        machine = refresher.GitMachine(main_checkout=tmp_path, worktree=tmp_path / "w", run=run)
        assert machine.rebuild() is None
        assert calls == [
            (["uv", "sync", "--extra", "dev", "--reinstall-package", "chess-variant-sandbox"],
             tmp_path / "w")
        ]

    def test_a_failed_rebuild_reports_the_tail_of_what_it_said(self, tmp_path: Path) -> None:
        def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(command, 1, "stdout", "x" * 500 + "the cause")

        machine = refresher.GitMachine(main_checkout=tmp_path, worktree=tmp_path, run=run)
        said = machine.rebuild()
        assert said is not None and said.startswith("the rebuild failed: ")
        assert said.endswith("the cause")
        assert len(said) == len("the rebuild failed: ") + 400

    def test_a_failed_rebuild_with_no_stderr_falls_back_to_stdout(self, tmp_path: Path) -> None:
        def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(command, 1, "  from stdout \n", "")

        machine = refresher.GitMachine(main_checkout=tmp_path, worktree=tmp_path, run=run)
        assert machine.rebuild() == "the rebuild failed: from stdout"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout


class TestDesignerErrors:
    def test_a_spec_error_is_one_plain_sentence_per_problem(self) -> None:
        problems = ["pieces must list at least one piece", "board.width must be an integer"]
        said = designing._errors_of(SpecError(problems))
        assert said == plain_problems(problems, None)
        assert len(said) == 2

    def test_any_other_error_is_split_into_its_lines_and_stripped_of_list_dashes(self) -> None:
        said = designing._errors_of(ValueError("- first problem\n\n  - second problem \n"))
        assert said == plain_problems(["first problem", "second problem"], None)

    def test_an_error_with_no_text_is_still_one_message(self) -> None:
        assert len(designing._errors_of(RuntimeError(""))) == 1

    def test_the_spec_lets_a_problem_name_the_piece_it_is_about(self) -> None:
        spec = load("dull")
        problem = "setup.placements[0].squares entry 'z9' is not a square on this board"
        [said] = designing._errors_of(SpecError([problem]), spec)
        assert said == plain_problems([problem], spec)[0]
        assert "z9" in said


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(load("gardner"))


class TestLiveMap:
    """What a piece may do from the opening position, against the engine's own list."""

    def expected(self, engine: Engine, origin: int) -> tuple[set[int], set[int]]:
        state = engine.initial_state()
        spec = engine.spec
        moves = {designing.move_to_square(m) for m in engine.legal_moves(state)
                 if designing.move_from_square(m) == origin}
        takes = {t for t in moves
                 if (found := spec.piece_at(state.board, t)) is not None
                 and found[0] != state.side_to_move}
        return moves, takes

    def test_the_squares_are_the_destinations_of_that_pieces_legal_moves(
        self, engine: Engine
    ) -> None:
        state = engine.initial_state()
        spec = engine.spec
        # a piece type that has a legal move from the opening position
        movers = {
            spec.piece_at(state.board, designing.move_from_square(m))[1]: designing.move_from_square(m)
            for m in engine.legal_moves(state)
        }
        assert movers, "the opening position has no moves"
        for piece, origin in movers.items():
            live = designing._live_map(engine, piece, origin)
            moves, takes = self.expected(engine, live["from"])
            assert live["live"] is True
            assert live["moves"] == sorted(moves)
            assert live["takes"] == sorted(takes)
            assert live["moveOnly"] == sorted(moves - takes)
            assert live["captureOnly"] == sorted(takes - moves)
            assert origin in live["standing"] and live["from"] == origin
            assert live["width"] == spec.width and live["height"] == spec.height
            assert live["value"] == state.board[origin]

    @pytest.mark.parametrize("name", ["dobutsu", "xiangqi", "sardines"])
    def test_a_move_onto_an_enemy_is_a_take_and_is_not_a_quiet_move_there(self, name: str) -> None:
        """These three open with a piece in contact with an enemy one."""
        engine = Engine(load(name))
        state = engine.initial_state()
        spec = engine.spec
        found = 0
        for move in engine.legal_moves(state):
            origin = designing.move_from_square(move)
            if origin < 0:
                continue
            piece = spec.piece_at(state.board, origin)[1]
            live = designing._live_map(engine, piece, origin)
            moves, takes = self.expected(engine, origin)
            assert live["takes"] == sorted(takes) and live["moves"] == sorted(moves)
            assert set(live["moveOnly"]).isdisjoint(live["takes"])
            assert set(live["captureOnly"]) <= set(live["takes"])
            found += bool(takes)
        assert found, "none of these openings has a capture available"

    def test_a_square_where_the_piece_does_not_stand_falls_back_to_one_where_it_does(
        self, engine: Engine
    ) -> None:
        state = engine.initial_state()
        spec = engine.spec
        origin = designing.move_from_square(engine.legal_moves(state)[0])
        piece = spec.piece_at(state.board, origin)[1]
        live = designing._live_map(engine, piece, -1)
        assert live["from"] == live["standing"][0] and live["from"] in live["standing"]
        assert live["moves"], "the fallback square is one the piece can move from"

    def test_a_piece_that_is_not_on_the_board_has_no_square_and_no_moves(
        self, engine: Engine
    ) -> None:
        # the horde opens with the mover holding only pawns
        engine = Engine(load("horde"))
        state = engine.initial_state()
        spec = engine.spec
        present = {spec.piece_at(state.board, s)[1] for s in range(spec.width * spec.height)
                   if spec.piece_at(state.board, s) is not None
                   and spec.piece_at(state.board, s)[0] == state.side_to_move}
        absent = next(t for t in range(spec.n_types) if t not in present)
        live = designing._live_map(engine, absent, 0)
        assert live["from"] == -1 and live["standing"] == [] and live["moves"] == []

    def test_move_map_not_alone_is_this_answer(self, engine: Engine) -> None:
        state = engine.initial_state()
        origin = designing.move_from_square(engine.legal_moves(state)[0])
        piece = engine.spec.piece_at(state.board, origin)[1]
        through = designing.move_map(load("gardner"), piece, origin, alone=False)
        assert through == designing._live_map(Engine(load("gardner")), piece, origin)
