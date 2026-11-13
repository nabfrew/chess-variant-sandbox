"""What the score has been tested against, `/score-tests` (R-S9 of `docs/roadmap-2026-11.md`).

A public page that lists every test of the score as it was read: what was asked, what was
predicted beforehand, what was found, hit or miss, and the day, with misses drawn exactly
as hits are. It is the strongest thing the site can say to a sceptical visitor, so it says
what each test does and does not show, beside the result.

**The rows are data.** Each test is one entry of `chess_variant_sandbox/web/score_tests.json`,
oldest first; a later test's task appends its own entry and touches nothing here.
`tests/test_score_tests_page.py` refuses an entry missing a field and checks that the page
draws every field of every entry. Numbers in an entry are copied from the record it names,
never worked out here: this page computes nothing, reads no score and no store.

**Two drawings of the frame, one of the page**, as `rules_page.py` does: `body()` is the
page; the site wears it in its own pages' shell (`site/pages.py`'s `score_tests`), and the
local application answers with `document()`, the same body under the application's header.
Both are constants for a process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from importlib import resources
from typing import Any

from chess_variant_sandbox.web.page import assets, front, shell, stylesheet_links

PATH = "/score-tests"

TITLE = front.titled("What the score has been tested against")

HEADING = "What the score has been tested against"

DESCRIPTION = (
    f"Every test of the score given to chess variants {front.FREE}: what was asked, "
    "predicted and found, hits and misses alike."
)
"""What a search result says (under `game.DESCRIPTION_LIMIT`, never another page's)."""

DATA = "score_tests.json"
"""The rows, beside this package's `web/` modules."""

FIELDS = ("id", "read", "question", "asked", "predicted", "found", "outcome", "tally",
          "shows", "record")
"""Every field an entry must carry, each a non-empty string."""

OUTCOMES = {
    "hit": "Hit",
    "miss": "Miss",
    "mixed": "Hits and misses",
    "not graded": "Not graded",
}
"""What an entry's `outcome` may be, and the words the page shows for it."""

LEAD = (
    "Every game here carries a score: how interesting it looks when the computer plays it, "
    "out of 1. A number like that is only worth what it has been tested against. This page "
    "lists each test of the score as it was read, newest first, with the misses shown as "
    "plainly as the hits."
)

HOW = (
    "Each test was written down before its numbers were computed: the question, what would "
    "count as a hit and what as a miss, and a prediction. The project's own AI agents "
    "registered and ran every test, and nobody outside the project has checked them; the "
    "agent that wrote a test's prediction is called its author below. In "
    "some tests the rule is a pass mark for the score; in others it grades a prediction "
    "about how the score would behave, so a miss there means the prediction was wrong, "
    "which can be good or bad news for the score. Each test says which. No threshold was "
    "moved after the numbers were seen."
)

WORDS = (
    "Some words used below. A <strong>95% interval</strong> is the range the number would "
    "fall in nineteen times in twenty if the test were run again on games like these. "
    "<strong>Likeness to chess</strong> is a number from 0 to 1 worked out from a game's "
    "rules alone: how many of its rarer ingredients it shares with chess. Several tests ask "
    "whether games unlike chess score lower; that the score marks such games down is a "
    "question these tests are asking, not a fact this page assumes. The "
    "<strong>checks for a broken game</strong> are separate from the number: they rule a "
    "game out (for example, when one side nearly always wins) rather than lower its score."
)


@dataclass(frozen=True)
class Test:
    """One test as read: an entry of the data file."""

    id: str
    read: str
    question: str
    asked: str
    predicted: str
    found: str
    outcome: str
    tally: str
    shows: str
    record: str


def problems(entry: Any) -> list[str]:
    """What is wrong with one entry of the data file, in words; empty when nothing is."""
    if not isinstance(entry, dict):
        return ["an entry is not an object"]
    said = [f"{entry.get('id', '?')}: no {field}" for field in FIELDS
            if not isinstance(entry.get(field), str) or not entry[field].strip()]
    extra = sorted(set(entry) - set(FIELDS))
    if extra:
        said.append(f"{entry.get('id', '?')}: unknown fields {extra}")
    if entry.get("outcome") not in OUTCOMES and isinstance(entry.get("outcome"), str):
        said.append(f"{entry.get('id', '?')}: outcome {entry['outcome']!r} is not one of "
                    f"{sorted(OUTCOMES)}")
    read = entry.get("read")
    if isinstance(read, str) and read.strip() and not _is_day(read):
        said.append(f"{entry.get('id', '?')}: read {read!r} is not a YYYY-MM-DD day")
    return said


def _is_day(text: str) -> bool:
    parts = text.split("-")
    return (len(parts) == 3 and [len(p) for p in parts] == [4, 2, 2]
            and all(p.isdigit() for p in parts) and 1 <= int(parts[1]) <= 12
            and 1 <= int(parts[2]) <= 31)


def raw() -> dict[str, Any]:
    """The data file as written."""
    text = resources.files("chess_variant_sandbox.web").joinpath(DATA).read_text("utf-8")
    loaded = json.loads(text)
    assert isinstance(loaded, dict)
    return loaded


@lru_cache(maxsize=1)
def tests() -> tuple[Test, ...]:
    """Every test in the file's order, oldest first. A broken entry stops the page rather
    than drawing a row with a hole in it."""
    entries = raw().get("tests")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{DATA} holds no tests")
    wrong = [said for entry in entries for said in problems(entry)]
    if wrong:
        raise ValueError(f"{DATA}: " + "; ".join(wrong))
    return tuple(Test(**{field: entry[field].strip() for field in FIELDS})
                 for entry in entries)


_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December")


def day(read: str) -> str:
    """`2026-11-13` as `13 November 2026`."""
    year, month, date = read.split("-")
    return f"{int(date)} {_MONTHS[int(month) - 1]} {year}"


def newest_first(held: tuple[Test, ...]) -> list[Test]:
    """The tests newest first: by day, and within a day the later entry first."""
    order = list(reversed(held))
    return sorted(order, key=lambda test: test.read, reverse=True)


def anchor(test: Test) -> str:
    """A test's id on the page, from its reference: `R-S10` is `#test-r-s10`."""
    return "test-" + "".join(c if c.isalnum() else "-" for c in test.id.lower())


def _outcome(test: Test) -> str:
    return (f'<span class="testOutcome" data-outcome="{escape(test.outcome, quote=True)}">'
            f"{escape(OUTCOMES[test.outcome])}</span>")


def _counts(held: tuple[Test, ...]) -> str:
    """One sentence: how many tests, and how many of each outcome."""
    count = {key: sum(1 for test in held if test.outcome == key) for key in OUTCOMES}
    parts = [f"{count['hit']} {'hit' if count['hit'] == 1 else 'hits'}",
             f"{count['miss']} {'miss' if count['miss'] == 1 else 'misses'}",
             f"{count['mixed']} with hits and misses"]
    if count["not graded"]:
        parts.append(f"{count['not graded']} not graded")
    newest = max(test.read for test in held)
    return (f"{len(held)} tests read, the latest on {day(newest)}: "
            + ", ".join(parts[:-1]) + " and " + parts[-1] + ".")


def _summary(shown: list[Test]) -> str:
    """The table at the top: every test in a line, linked to its own section."""
    rows = "".join(
        f'<tr><td class="testWhen">{escape(day(test.read))}</td>'
        f'<td><a href="#{anchor(test)}">{escape(test.question)}</a></td>'
        f"<td>{_outcome(test)}</td></tr>"
        for test in shown
    )
    return (
        '<div class="testTableWrap"><table class="testTable">'
        '<thead><tr><th class="testWhen">Read</th><th>The question</th><th>Result</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _section(test: Test) -> str:
    """One test, in full."""
    parts = (
        ("What was asked", test.asked),
        ("Predicted beforehand", test.predicted),
        ("What was found", test.found),
        ("What it does and does not show", test.shows),
    )
    said = "".join(f"<dt>{escape(name)}</dt><dd>{escape(words)}</dd>" for name, words in parts)
    return (
        f'<section class="testOne" id="{anchor(test)}">'
        f"<h2>{escape(test.question)}</h2>"
        f'<p class="testMeta">{_outcome(test)} '
        f'<span class="muted">Read {escape(day(test.read))}</span></p>'
        f'<p class="testTally">{escape(test.tally)}</p>'
        f'<dl class="testParts">{said}</dl>'
        f'<p class="testRef muted small">Reference {escape(test.id)} · the full record, '
        f"with every number and the script that made it, is "
        f"<code>{escape(test.record)}</code> in the project's source.</p>"
        "</section>"
    )


def body() -> str:
    """The page inside its frame."""
    held = tests()
    shown = newest_first(held)
    return (
        '<div class="testsPage">'
        f"<h1>{escape(HEADING)}</h1>"
        f'<p class="lead">{escape(LEAD)}</p>'
        f"<p>{escape(HOW)}</p>"
        f'<p class="muted">{WORDS}</p>'
        f'<p class="testCounts">{escape(_counts(held))}</p>'
        + _summary(shown)
        + "".join(_section(test) for test in shown)
        + '<p class="muted small">How the score is put together is on '
        '<a href="/#/score">how the score works</a>, and how the scores have moved under '
        f'each version of it on <a href="{front.HISTORY_PATH}">the score history</a>.</p>'
        "</div>"
    )


STYLE = """
  /* What the score has been tested against (R-S9): a reading measure, a table of every
     test, then each test in full. A hit and a miss are drawn alike: the word is the mark. */
  main.scoreTests .testsPage { max-width: 46rem; }
  main.scoreTests .testsPage h1 { font-size: 28px; line-height: 34px; margin: 0 0 10px; }
  main.scoreTests .testsPage .lead { margin: 0 0 12px; }
  main.scoreTests .testsPage p { margin: 0 0 12px; }
  main.scoreTests .testCounts { font-weight: 600; }
  main.scoreTests .testTableWrap { margin: 0 0 24px; }
  main.scoreTests table.testTable { width: 100%; border-collapse: collapse; font-size: 14px;
              line-height: 20px; }
  main.scoreTests table.testTable th { text-align: left; color: var(--muted); font-weight: 600;
              border-bottom: 1px solid var(--line); padding: 4px 8px 4px 0; }
  main.scoreTests table.testTable td { border-bottom: 1px solid var(--line); padding: 6px 8px 6px 0;
              vertical-align: top; }
  main.scoreTests td.testWhen { white-space: nowrap; color: var(--muted); }
  main.scoreTests .testOutcome { display: inline-block; font-weight: 600; font-size: 13px;
              line-height: 18px; padding: 1px 8px; border: 1px solid var(--edge);
              border-radius: var(--r-ctl); background: var(--sunk); color: var(--ink);
              white-space: nowrap; }
  main.scoreTests section.testOne { border-top: 1px solid var(--line); padding-top: 18px;
              margin-top: 22px; scroll-margin-top: 12px; }
  main.scoreTests section.testOne h2 { font-size: 20px; line-height: 26px; margin: 0 0 8px; }
  main.scoreTests p.testMeta { display: flex; flex-wrap: wrap; gap: 4px 12px;
              align-items: center; margin: 0 0 8px; }
  main.scoreTests p.testTally { font-weight: 600; }
  main.scoreTests dl.testParts { margin: 0 0 10px; }
  main.scoreTests dl.testParts dt { font-weight: 600; margin: 12px 0 2px; }
  main.scoreTests dl.testParts dd { margin: 0; }
  main.scoreTests p.testRef code { overflow-wrap: anywhere; }
  @media (max-width: 560px) {
    /* A phone: each test's own section carries its day, so the table drops the column and
       the question takes the width. */
    main.scoreTests table.testTable th.testWhen, main.scoreTests td.testWhen { display: none; }
    main.scoreTests .testOutcome { white-space: normal; }
  }
"""
"""The page's own rules, beside the application's stylesheet on both drawings."""

SHEET = assets.sheet("score-tests", STYLE)
LOCAL_SHEET = assets.sheet(
    "score-tests-local", STYLE + "  header h1 a { color: var(--ink); border-bottom: none; }\n"
)


def document() -> str:
    """The whole page as the local application answers `/score-tests`: the body under the
    application's header and footer, as `rules_page.document` draws `/rules`."""
    links = " · ".join(
        f'<a href="{where}">{words}</a>'
        for where, words in ((front.GAMES_PATH, "games"), ("/", "about"),
                             ("/#/score", "the score"),
                             (front.HISTORY_PATH, "score history"),
                             ("/#/whats-new", "what's new"))
    )
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(TITLE)}</title>\n{front.head_lines(TITLE, DESCRIPTION)}"
        f"{shell.head_links()}{stylesheet_links(LOCAL_SHEET)}</head>\n"
        '<body>\n<div class="wrap">\n  <header class="top">\n'
        f'    <h1 class="brand"><a href="/">{shell.brand()}</a></h1>\n'
        f'    <span class="muted tagline">{shell.TAGLINE}</span>\n'
        '    <span class="grow"></span>\n'
        f"    {shell.sections('/')}\n"
        '    <span class="muted small" id="who"></span>\n  </header>\n'
        f'<main class="scoreTests">\n{body()}\n</main>\n{shell.footer(links)}\n</div>\n'
        "</body>\n</html>\n"
    )


@lru_cache(maxsize=1)
def local_bytes() -> bytes:
    """`document()` as the local server sends it, encoded once."""
    return document().encode("utf-8")
