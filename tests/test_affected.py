"""What a change reaches: the judgement `--skip-if-green` leans on when a stamp is another tree's.

Each case is a small tree held in memory, so a rule is pinned by its own example; what the
rules do on this repository's record is `docs/reference/affected-gate.md`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import affected
import pytest


def graph(files: dict[str, str]) -> affected.Graph:
    return affected.Graph({path: affected.parse(path, text) for path, text in files.items()})


PACKAGE = {
    "chess_variant_sandbox/__init__.py": "",
    "chess_variant_sandbox/base.py": "VALUE = 1\n",
    "chess_variant_sandbox/middle.py": "from chess_variant_sandbox import base\n",
    "chess_variant_sandbox/top.py": "from . import middle\n",
    "chess_variant_sandbox/far.py": "from chess_variant_sandbox import top\n",
    "tests/test_base.py": "from chess_variant_sandbox import base\n",
    "tests/test_middle.py": "from chess_variant_sandbox import middle\n",
    "tests/test_far.py": "from chess_variant_sandbox import far\n",
    "tests/test_alone.py": "def test_nothing():\n    pass\n",
}


def reached(files: dict[str, str], *paths: str) -> set[str]:
    reach = graph(files).reach(paths)
    assert reach.everything is None, reach.everything
    return set(reach.tests)


class TestImports:
    def test_importers_are_followed_two_steps_and_no_further(self) -> None:
        # base <- middle <- top (a relative import) <- far <- test_far: four steps from base
        assert reached(PACKAGE, "chess_variant_sandbox/base.py") == {
            "tests/test_base.py", "tests/test_middle.py",
        }

    def test_a_changed_test_file_reaches_itself(self) -> None:
        assert reached(PACKAGE, "tests/test_alone.py") == {"tests/test_alone.py"}

    def test_a_deleted_module_reaches_what_still_imports_it(self) -> None:
        files = {k: v for k, v in PACKAGE.items() if k != "chess_variant_sandbox/base.py"}
        assert "tests/test_base.py" in reached(files, "chess_variant_sandbox/base.py")

    def test_a_helper_beside_the_tests_is_imported_by_its_bare_name(self) -> None:
        files = {**PACKAGE, "tests/helper.py": "X = 1\n", "tests/test_helped.py": "import helper\n"}
        assert reached(files, "tests/helper.py") == {"tests/test_helped.py"}


class TestNames:
    def test_a_module_named_in_a_monkeypatch_is_reached(self) -> None:
        files = {
            **PACKAGE,
            "tests/test_patch.py": (
                "def test_it(monkeypatch):\n"
                '    monkeypatch.setattr("chess_variant_sandbox.far.top", None)\n'
            ),
        }
        assert "tests/test_patch.py" in reached(files, "chess_variant_sandbox/far.py")

    def test_a_data_file_is_reached_through_what_names_its_directory(self) -> None:
        files = {
            **PACKAGE,
            "chess_variant_sandbox/corpus.py": 'from pathlib import Path\nROOT = Path("rulesets")\n',
            "tests/test_corpus.py": "from chess_variant_sandbox import corpus\n",
        }
        assert reached(files, "rulesets/new.json") == {"tests/test_corpus.py"}

    def test_a_directory_counts_as_a_whole_path_component_only(self) -> None:
        files = {**PACKAGE, "tests/test_words.py": 'X = "rulesetsy"\nY = "docs/rulesets.md"\n'}
        reach = graph(files).reach(["rulesets/new.json"])
        assert reach.everything == "nothing names rulesets/new.json"

    def test_a_docstring_names_nothing(self) -> None:
        files = {**PACKAGE, "tests/test_prose.py": '"""Reads rulesets/new.json, they say."""\n'}
        assert graph(files).reach(["rulesets/new.json"]).everything is not None

    def test_a_test_that_globs_source_is_reached_by_any_python_change(self) -> None:
        files = {**PACKAGE, "tests/test_scan.py": 'FILES = "chess_variant_sandbox/*.py"\n'}
        assert "tests/test_scan.py" in reached(files, "chess_variant_sandbox/base.py")
        assert "tests/test_scan.py" not in reached(files, "tests/test_alone.py")

    def test_a_test_that_greps_source_through_a_helper_is_reached_too(self) -> None:
        """`test_green_stamp.py` reads every tracked `*.py` through `green_stamp.unread`."""
        files = {
            **PACKAGE,
            "tests/grepper.py": 'def grep(name):\n    return ["git", "grep", name, "--", "*.py"]\n',
            "tests/test_grep.py": "import grepper\n",
        }
        assert "tests/test_grep.py" in reached(files, "chess_variant_sandbox/base.py")


class TestEveryTest:
    @pytest.mark.parametrize(
        "path", ["rust/src/lib.rs", "pyproject.toml", "uv.lock", "tests/conftest.py", "deploy/unnamed.sh"]
    )
    def test_what_every_test_can_see(self, path: str) -> None:
        reach = graph(PACKAGE).reach(["chess_variant_sandbox/base.py", path])
        assert reach.everything is not None
        assert reach.tests == graph(PACKAGE).tests

    def test_a_file_that_does_not_parse(self) -> None:
        files = {**PACKAGE, "chess_variant_sandbox/broken.py": "def (:\n"}
        assert graph(files).reach(["chess_variant_sandbox/broken.py"]).everything is not None

    def test_a_note_or_picture_nothing_names_reaches_nothing(self) -> None:
        assert reached(PACKAGE, "docs/reference/shot.png", ".claude/rules/x.md") == set()

    def test_what_conftest_imports_reaches_the_tests_using_its_fixtures(self) -> None:
        files = {
            **PACKAGE,
            "tests/conftest.py": (
                "import pytest\nfrom chess_variant_sandbox import base\n\n"
                "@pytest.fixture\ndef engine():\n    return base\n"
            ),
            "tests/test_uses.py": "def test_it(engine):\n    pass\n",
        }
        assert reached(files, "chess_variant_sandbox/base.py") == {
            "tests/test_base.py", "tests/test_middle.py", "tests/test_uses.py",
        }


def test_parsing_is_cached_by_blob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A gate parses only what changed: the whole tree takes 5-15 s, a warm cache 0.1 s."""
    root = tmp_path / "repository"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_one.py").write_text("import helper\n", encoding="utf-8")
    (root / "tests" / "helper.py").write_text("X = 1\n", encoding="utf-8")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "x"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    cache = tmp_path / "cache.json"
    cache.write_text('{"version": "an older reader", "entries": {}}', encoding="utf-8")

    first = affected.graph_of(str(root), tree, cache=cache)
    assert json.loads(cache.read_text(encoding="utf-8"))["version"] == affected.VERSION

    def no_parsing(path: str, text: str) -> None:
        raise AssertionError(f"parsed {path} again")

    monkeypatch.setattr(affected, "parse", no_parsing)
    again = affected.graph_of(str(root), tree, cache=cache)
    assert again.parsed == first.parsed
    assert again.reach(["tests/helper.py"]).tests == {"tests/test_one.py"}
