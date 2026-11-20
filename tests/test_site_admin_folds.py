"""AD1 (the owner, 2026-11-20): `/admin`'s long tables fold shut, their count on the fold.

*"The admin page has an enourmous list of games that is not useful, makes it take ages to
scroll down to the bottom."* The games table, and the two other tables a row a game or a
row a re-measurement long, now sit in a `<details>` closed by default; every row is still
drawn inside it, so every game stays one click away, and the summary carries the count.
The drawing is run in node over the stubbed DOM `tests/test_site.py` keeps, so what the
summary says is under test, not merely that a line is in the page.
"""

from __future__ import annotations

import json
import re
from typing import Any

import page_text
import pytest
from test_site import _DOM, _in_node, _script_of

from chess_variant_sandbox.site import pages as site_pages

FOLDS = (
    # (the fold, its summary, the table body inside it, the table's own heading)
    ("games-fold", "games-summary", "games", "<h2>Games</h2>"),
    ("score-history-fold", "score-history-summary", "score-history", "<h2>Earlier scores</h2>"),
    ("score-volatility-fold", "score-volatility-summary", "score-volatility",
     "<h2>How much scores have moved</h2>"),
    ("limits-fold", "limits-summary", "limits", "<h2>What the site limits</h2>"),
)

PAGE = page_text.inlined(site_pages.admin("boss"))


@pytest.mark.parametrize(("fold", "summary", "body", "heading"), FOLDS)
def test_each_long_table_sits_in_a_fold_closed_by_default(
    fold: str, summary: str, body: str, heading: str
) -> None:
    opened = re.search(rf'<details class="adminFold" id="{fold}"[^>]*>', PAGE)
    assert opened is not None, fold
    assert " open" not in opened.group(0)
    start = opened.start()
    end = PAGE.index("</details>", start)
    inside = PAGE[start:end]
    assert f'<summary id="{summary}">' in inside
    assert f'<tbody id="{body}"></tbody>' in inside
    # Under its own heading and explanation, which stay readable with the fold shut.
    assert PAGE.index(heading) < start


def _drawn(runner: str) -> Any:
    return _in_node(_DOM + _script_of(site_pages.admin("boss")), runner)


def _game(index: int) -> dict[str, Any]:
    return {"hash": f"{index:064x}", "name": f"game {index}", "origin": "corpus",
            "owner": None, "visibility": "public", "matches": 0, "state": "listed",
            "gone": False, "killed": False}


def test_the_games_fold_counts_every_game_and_draws_every_row() -> None:
    rows = [_game(index) for index in range(250)]
    said = _drawn(
        f"games({json.dumps(rows)});"
        "console.log(JSON.stringify({"
        " summary: document.getElementById('games-summary').textContent,"
        " rows: document.getElementById('games').children.length}));"
    )
    assert said == {"summary": "All games (250)", "rows": 250}


def test_an_empty_games_fold_says_nought() -> None:
    said = _drawn("games([]); console.log(JSON.stringify("
                  "document.getElementById('games-summary').textContent));")
    assert said == "All games (0)"


def test_the_replaced_scores_fold_counts_rows_not_games() -> None:
    """One row a replaced score: two games with three and one replacements are four rows."""
    entry = {"index": 0.5, "runs": 4, "weights_version": "v15", "measured": "2026-11-01",
             "replaced": "2026-11-02", "players": "p", "moved": [], "reasons": [],
             "convergence": None}
    games = [{"hash": "a" * 64, "name": "a", "entries": [entry] * 3},
             {"hash": "b" * 64, "name": "b", "entries": [entry]}]
    said = _drawn(
        f"scoreHistory({json.dumps(games)});"
        "console.log(JSON.stringify({"
        " summary: document.getElementById('score-history-summary').textContent,"
        " rows: document.getElementById('score-history').children.length}));"
    )
    assert said == {"summary": "Replaced scores (4)", "rows": 4}


def test_the_scored_games_fold_counts_its_games() -> None:
    games = [{"hash": f"{index:064x}", "name": str(index), "revisions": 2,
              "weights_version": "v15", "spread": 0.01, "largest_step": 0.01,
              "interval_half": 0.02, "said": "steady"} for index in range(7)]
    said = _drawn(
        f"scoreVolatility({json.dumps(games)});"
        "console.log(JSON.stringify("
        "document.getElementById('score-volatility-summary').textContent));"
    )
    assert said == "Scored games (7)"


def test_the_limits_fold_counts_its_limits() -> None:
    rows = [{"label": f"limit {index}", "value": index} for index in range(3)]
    said = _drawn(f"limits({json.dumps(rows)}); console.log(JSON.stringify("
                  "document.getElementById('limits-summary').textContent));")
    assert said == "Every limit (3)"
