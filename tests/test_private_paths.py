"""No exported test reads a path the public copy lacks, except through `private_paths` (Q1210).

The public repository is an export of this one (`docs/reference/public-fork.md`). Q1209 found
seven exported tests that read `.claude/scripts/` or `docs/` and so failed there, and the
export's own run found six more. Each now reads the path through `tests/private_paths.py` and
carries its `needs` mark, which skips only where the path is absent. This file holds that:

- every path a mark names is here, so in this checkout every marked test runs;
- the mechanism's idea of "private" is the exporter's;
- a scan of every exported test file refuses a join from the repository root onto a private
  path, and a file that reads one through the mechanism without any `needs` mark.

The scan is a reading of source text, so it sees the spellings the suite uses (`ROOT`,
`REPO`, `REPOSITORY`, `TREE`, `parents[1]`, `parent.parent`); a path built some other way
is still caught by the export's own suite, which is where Q1209 found the seven.
"""

from __future__ import annotations

import re

import private_paths
import pytest

REPOSITORY = private_paths.REPOSITORY
EXPORTER = ".claude/scripts/export_public.py"
export_public = private_paths.script(EXPORTER, "export_public_for_private_paths")

# A join from the repository root onto a literal: `ROOT / ".claude" / "scripts"`,
# `Path(__file__).resolve().parents[1] / "docs"`, `REPOSITORY / "deploy/remote"`.
RAW_JOIN = re.compile(
    r"(?:\b(?:ROOT|REPO|REPOSITORY|TREE)\b|parents\[1\]|parent\.parent)\s*\)?\s*/\s*"
    r'"([^"\n]+)"(?:\s*/\s*"([^"\n]+)")?'
)
THROUGH = re.compile(r"private_paths\.(?:path|script)\(")
MARKED = re.compile(r"private_paths\.needs\(")
NAMED = re.compile(r"private_paths\.needs\(([^)]*)\)")


def findings(source: str) -> list[str]:
    """What the scan refuses in one test file's source."""
    found = []
    for match in RAW_JOIN.finditer(source):
        relative = "/".join(part for part in match.groups() if part)
        # A bare folder (`ROOT / "docs"`) is read as the folder, `docs/`.
        folder = relative.rstrip("/") + "/"
        if relative not in private_paths.EXPORTED and (
                private_paths.private(relative) or private_paths.private(folder)):
            line = source.count("\n", 0, match.start()) + 1
            found.append(f"line {line}: {match.group(0)!r} joins the root onto {relative}")
    if THROUGH.search(source) and not MARKED.search(source):
        found.append("reads a private path through private_paths with no needs(...) mark")
    return found


def exported_tests() -> list[str]:
    paths = sorted(p.relative_to(REPOSITORY).as_posix() for p in (REPOSITORY / "tests").rglob("*.py"))
    return [path for path in paths
            if export_public.allowed(path) and not export_public.never(path)
            # The mechanism, and this file, whose planted examples are what the scan refuses.
            and path not in ("tests/private_paths.py", "tests/test_private_paths.py")]


class TestTheMechanism:
    def test_it_skips_only_where_the_path_is_absent(self) -> None:
        here = private_paths.needs("tests/private_paths.py", "tests/")
        gone = private_paths.needs("tests/private_paths.py", "no/such/path.py")
        assert here.args == (False,)
        assert gone.args == (True,)
        assert gone.kwargs["reason"].endswith("no/such/path.py")

    def test_a_missing_script_fails_a_test_that_lacks_the_mark(self) -> None:
        missing = private_paths.script("no/such/script.py")
        with pytest.raises(RuntimeError, match=r'needs\("no/such/script.py"\)'):
            missing.main()

    def test_private_is_what_the_export_leaves_behind(self) -> None:
        for path in (".claude/scripts/x.py", "deploy/remote/Caddyfile", "docs/plan-x.md",
                     "docs_tests/test_x.py", "CLAUDE.md"):
            assert private_paths.private(path), path
        for path in ("docs/changelog.md", "tests/test_x.py", "rulesets/chess.json"):
            assert not private_paths.private(path), path


@private_paths.needs(EXPORTER)
class TestTheExporterAgrees:
    def test_every_private_root_is_never_exported(self) -> None:
        for root in private_paths.PRIVATE:
            probe = f"{root}x.py" if root.endswith("/") else root
            assert export_public.never(probe), root
        for path in private_paths.EXPORTED:
            assert export_public.allowed(path) and not export_public.never(path), path

    def test_the_mechanism_and_this_file_are_exported(self) -> None:
        for path in ("tests/private_paths.py", "tests/test_private_paths.py"):
            assert export_public.allowed(path) and not export_public.never(path), path


@private_paths.needs(EXPORTER)
class TestTheScan:
    def test_it_refuses_a_raw_join_and_an_unmarked_read(self) -> None:
        assert findings('SCRIPT = ROOT / ".claude" / "scripts" / "x.py"\n')
        assert findings('p = Path(__file__).resolve().parents[1] / "docs" / "reference"\n')
        assert findings('DEPLOY = Path(__file__).resolve().parent.parent / "deploy"\n')
        assert findings('x = REPOSITORY / "deploy/remote/Caddyfile"\n')
        assert findings('x = private_paths.path("deploy/x")\n')

    def test_it_passes_the_mechanism_the_changelog_and_a_scratch_tree(self) -> None:
        assert not findings('x = private_paths.path("deploy/x")\n'
                            '@private_paths.needs("deploy/x")\n')
        assert not findings('x = REPOSITORY / "docs" / "changelog.md"\n')
        assert not findings('x = tmp_path / ".claude" / "scripts"\n')
        assert not findings('x = ROOT / "rulesets" / "chess.json"\n')

    def test_no_exported_test_reads_a_private_path_unmarked(self) -> None:
        tests = exported_tests()
        assert "tests/test_clean_runs.py" in tests and "tests/test_deploy.py" not in tests
        bad = {path: found for path in tests
               if (found := findings((REPOSITORY / path).read_text(encoding="utf-8")))}
        assert bad == {}, (
            "an exported test reads a path the public copy lacks: read it through "
            "tests/private_paths.py with its needs(...) mark, or keep the file out of the "
            "export (docs/reference/public-fork.md section 2)"
        )

    def test_every_path_a_mark_names_is_here(self) -> None:
        """So in this checkout every marked test runs and asserts what it always did."""
        named = {literal for path in exported_tests()
                 for args in NAMED.findall((REPOSITORY / path).read_text(encoding="utf-8"))
                 for literal in re.findall(r'"([^"]+)"', args)}
        assert len(named) >= 10, named
        assert private_paths.absent(*sorted(named)) == []
