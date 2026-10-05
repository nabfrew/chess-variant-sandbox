"""Every script a page ships, and which names in it nothing declares.

The page, the site's pages and the offline replay run JavaScript that lives in plain files
(Python strings until Q513), and nothing lints it. Two slips in its history were a name nobody declared
(`docs/reference/page-javascript.md` §3): the router calling a view that did not exist yet,
and the admin page calling a helper only another page's script defined. `node --check`
passes both, since an undefined name is an error only when the line runs, and so did every
test that did not walk that path. `undefined` is the check that catches them: ESLint's
`no-undef` over each assembled script, with no npm package. The walk is
`tests/undefined_names.js`; the parser is the acorn node ships for its own REPL.

`scripts()` is what both it and the parse checks read, so a new page with a script is
parsed and walked without anyone adding it to a list: the site's pages are found by calling
every public page function in `site.pages`, which is how `/suggestions` went unparsed.
Skips where node is absent, as `node_session.run` does.
"""

from __future__ import annotations

import inspect
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import page_text
import pytest

WALKER = Path(__file__).with_name("undefined_names.js")

def script_of(page: str) -> str:
    """Every script an HTML page runs, in order: one global scope, as served. A `src` is
    read from the file its server answers it with (`page_text.script_of`)."""
    return page_text.script_of(page)


def site_pages() -> dict[str, str]:
    """Each page `site.pages` renders, by its path, whether or not it carries a script.

    Every public function the module defines is a page named by its path; each takes
    strings only (a handle, an invitation code), and any string renders it.
    """
    from chess_variant_sandbox.site import pages

    found = {}
    for name, function in vars(pages).items():
        if name.startswith("_") or not inspect.isfunction(function):
            continue
        if function.__module__ != pages.__name__:
            continue
        # A keyword-only parameter (the reader's account id, for their mark) is left at
        # its default: positionally, every page takes strings.
        named = [
            parameter for parameter in inspect.signature(function).parameters.values()
            if parameter.kind != inspect.Parameter.KEYWORD_ONLY
        ]
        if name == "member":
            # The one page drawn whole from a reading rather than from strings (R1): a
            # reading with a challenge, so its script is among those parsed and walked.
            found["/players/ada"] = function(MEMBER_READING, "bo", viewer=2)
            continue
        found["/" + name] = function(*["ada"] * len(named))
    return found


MEMBER_READING: dict[str, Any] = {
    "user": 1, "handle": "ada", "mark": None, "since": "October 2026", "finished": None,
    "own": False,
    "challenge": {"paused": False, "pausedSays": "",
                  "games": [{"game": "0123456789ab", "name": "A game"}]},
    "recent": [], "designed": [], "milestones": [], "badges": [], "favourites": [],
    "playsWith": {"people": [], "computers": []},
}
"""A member's page's reading as `site.members.read` hands it over, for a page with no store."""


def scripts() -> dict[str, str]:
    """Every page script there is, labelled: the application's page, each site page that
    has one, and the offline replay with a payload spliced in where it writes one."""
    from page_text import PAGE

    from chess_variant_sandbox.viz import html as replay

    found = {"the page": script_of(PAGE)}
    for path, page in site_pages().items():
        if "<script" in page:
            found[path] = script_of(page)
    found["the replay"] = script_of(replay._PAGE.replace("__DATA__", "{}"))
    return found


def undefined(sources: dict[str, str]) -> dict[str, Any]:
    """For each labelled script, `[{"name", "line"}]`: each name it uses that nothing
    declares, at its first use. `{"error"}` in place of the list where it does not parse.
    One node for all of them."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    done = subprocess.run(
        [node, "--expose-internals", str(WALKER)],
        input=json.dumps(sources),
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    if done.returncode != 0 and "internal/deps/acorn" in done.stderr:
        pytest.skip("this node does not carry acorn where the walk looks for it")
    assert done.returncode == 0, done.stderr
    answer: dict[str, Any] = json.loads(done.stdout)
    return answer
