"""The session's one `node` answers as `node file.js` would.

About twenty helpers across the suite run the page's JavaScript through
`tests/node_session.py` instead of starting a node per check (Q377b). What they read is the
exit code and what was printed, so that is what is held here against a real `node` run on
the same source: ordinary output, output after a timer and a promise, and the three ways a
script fails. stderr is not compared -- node prints a failure's source line and version,
and no helper reads stderr except to show it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import node_session
import pytest

CASES = {
    "output": "let seen = { a: [1, 2], b: 'x' }; console.log(seen, 3); console.log('two');",
    "later": (
        "setTimeout(() => console.log('timer'), 20);"
        "Promise.resolve().then(() => console.log('promise'));"
        "console.log('now');"
    ),
    "a throw": "console.log('before'); throw new Error('boom');",
    "a throw in a timer": (
        "setTimeout(() => { throw new Error('later'); }, 5);"
        "setTimeout(() => console.log('never'), 60);"
    ),
    "a rejection nobody handles": "console.log('before'); Promise.reject(new Error('unheld'));",
    "a syntax error": "console.log('never'); let = ;",
}


@pytest.mark.parametrize("source", CASES.values(), ids=CASES.keys())
def test_it_answers_as_a_fresh_node_would(source: str, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    path = tmp_path / "check.js"
    path.write_text(source, encoding="utf-8")
    fresh = subprocess.run([node, str(path)], capture_output=True, text=True, encoding="utf-8")
    shared = node_session.run(source)
    assert (shared.returncode, shared.stdout) == (fresh.returncode, fresh.stdout)


def test_one_check_cannot_see_what_the_last_one_left() -> None:
    """A context per check is what makes one process stand in for many."""
    node_session.run("global.leftover = 1; var declared = 2;")
    seen = node_session.run("console.log(typeof leftover, typeof declared);")
    assert seen.stdout == "undefined undefined\n"


def test_a_check_only_parses() -> None:
    """`check=True` is `node --check`: a syntax error fails, a script that would throw
    does not, because nothing runs."""
    assert node_session.run("let = ;", check=True).returncode == 1
    assert node_session.run("throw new Error('not run');", check=True).returncode == 0
