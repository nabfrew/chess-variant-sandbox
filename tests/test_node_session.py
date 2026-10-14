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


# A job that asks the node itself how much heap is live after a full collection. `require`
# is the host's, so `v8` and `vm` are reachable from a check; `gc` is made reachable the way
# `--expose-gc` would, in a context of its own.
_LIVE_HEAP = (
    "const v8 = require('v8'); v8.setFlagsFromString('--expose-gc');"
    "require('vm').runInNewContext('gc')();"
    "console.log(v8.getHeapStatistics().used_heap_size);"
)


def test_a_check_s_context_is_let_go_when_it_ends() -> None:
    """Q749: thirty checks of a megabyte each leave no more live heap than one did.

    V8's compilation cache held every compiled script, and with it the context it was
    compiled into, for the life of the node; `test_web.py` alone grew it to a gigabyte.
    `NODE_FLAGS` turns the cache off. Without that flag the thirty leave 30 MB live.
    """
    node_session.release()
    node_session.run("console.log(1);")
    before = int(node_session.run(_LIVE_HEAP).stdout)
    padding = "// " + "x" * (1 << 20) + "\n"
    for each in range(30):
        assert node_session.run(f"{padding}console.log({each});").stdout == f"{each}\n"
    after = int(node_session.run(_LIVE_HEAP + "//again").stdout)
    assert after - before < 16 * 2**20, f"{(after - before) >> 20} MB still live after 30 checks"


def test_a_released_node_is_started_again_by_the_next_check() -> None:
    node_session.run("console.log('started');")
    assert node_session.release()
    assert not node_session.release()
    assert node_session.run("console.log('again');").stdout == "again\n"
