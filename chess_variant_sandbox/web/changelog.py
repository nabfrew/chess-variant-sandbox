"""`docs/changelog.md`, read as data for the *what's new* page.

The file is written by hand, newest first: a `##` heading per day, a `###` heading and a
paragraph or two per entry, with `*emphasis*` and `` `inline code` `` inside. This module
turns that into blocks and inline runs. **It produces no markup at all** -- every piece of
the file arrives at the browser as a string with a label saying what kind of string it is,
and `web/static/page/whatsnew.js` builds an element per piece and sets its `textContent`. So a
day's entry that holds a tag holds a tag on screen; there is no escaping step to get wrong
and no path by which the file can become part of the document.

What is supported is what the file uses and nothing else: the two heading levels, blank
line separated paragraphs, emphasis, inline code and the horizontal rule. **Every other
line is a paragraph of plain text**, printed as it stands -- a list item keeps its dash, a
table row its pipes, a link its brackets. That is the safe direction: an unsupported
construct reads as itself rather than as something this parser guessed at. Widening it is
a case in `runs()` or `parse()` here, never a special case in the page.

The file's own header -- the title and the note to whoever writes an entry, everything
above the first `---` -- is not shown. It is instructions to the maintainer, and the page
is for a reader.

An entry whose first paragraph is exactly `*For maintainers.*` (`MAINTAINERS`) is the
maintainer's news -- a command, a folder, a decision taken -- and is parsed with
`maintainers: true`. The local application shows it as written; the site skips it
(`for_visitors`, W8 of `docs/archive/plan-understanding.md` §7). The file's header says so to
whoever writes an entry.

Read on each request rather than cached. It is 30 KB and parses in about a millisecond,
and a changelog that needed a restart to show today's entry would be one the deploy could
silently leave behind.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

# Relative, like `web/library.py`'s `rulesets` and `runs`, and for the same reason: the
# box runs the site with `/opt/chess-variant-sandbox` as its working directory and `deploy/bootstrap.sh`
# ships the ref's tracked files there, `docs/` included. The fallback is for a process
# started somewhere else in a source checkout; an installed wheel has no `docs/` beside
# the package, which is what the missing-file sentence is for.
CHANGELOG = Path("docs") / "changelog.md"

_HEADING = re.compile(r"^(#+)\s+(.*)$")
_RULE = re.compile(r"^-{3,}$")

# `code` before `emphasis`, so a star inside backticks stays a star. Emphasis takes no
# leading space, which is what keeps a lone `*` in prose from opening a run that swallows
# the rest of the line.
_INLINE = re.compile(r"`([^`]+)`|\*([^*\s][^*]*)\*")

# The line that marks an entry as the maintainer's, alone as the entry's first paragraph. The
# file's own emphasis, so it reads as a sentence on GitHub and in the local application.
MAINTAINERS = "*For maintainers.*"

_MONTHS = ("January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December")


def source(path: Path | None = None) -> Path:
    """Where the changelog is, for whoever is asking."""
    if path is not None:
        return path
    if CHANGELOG.exists():
        return CHANGELOG
    beside = Path(__file__).resolve().parents[2] / CHANGELOG
    return beside if beside.exists() else CHANGELOG


def payload(path: Path | None = None) -> dict[str, Any]:
    """The whole changelog as days of entries, or a sentence saying why it is not here.

    `note` is what the page shows instead of the entries. A server without the file is a
    normal state -- an installed package has no `docs/` -- so it is a sentence and never
    an error.
    """
    try:
        text = source(path).read_text(encoding="utf-8")
    except OSError:
        return {"days": [], "note": "This server has no copy of the changelog to show."}
    days = parse(text)
    return {"days": days, "note": "" if days else "The changelog holds no entries yet."}


def parse(text: str) -> list[dict[str, Any]]:
    """The file's body as `[{date, words, entries: [{heading, blocks}]}]`, in file order.

    Newest first, because the file is written that way; nothing here sorts, so an entry
    filed out of order shows where it was filed rather than where a parser thought it
    belonged.
    """
    days: list[dict[str, Any]] = []
    paragraph: list[str] = []

    def day() -> dict[str, Any]:
        if not days:
            days.append({"date": "", "words": "", "entries": []})
        return days[-1]

    def blocks() -> list[dict[str, Any]]:
        entries = day()["entries"]
        if not entries:
            entries.append({"heading": "", "blocks": []})
        return entries[-1]["blocks"]

    def close() -> None:
        if paragraph:
            blocks().append({"kind": "paragraph", "runs": runs(" ".join(paragraph))})
            paragraph.clear()

    for line in body(text).splitlines():
        stripped = line.strip()
        heading = _HEADING.match(stripped)
        level = len(heading.group(1)) if heading else 0
        if not stripped:
            close()
        elif _RULE.match(stripped):
            close()
            blocks().append({"kind": "rule"})
        elif level == 2 and heading is not None:
            close()
            days.append(_day(heading.group(2).strip()))
        elif level == 3 and heading is not None:
            close()
            day()["entries"].append({"heading": heading.group(2).strip(), "blocks": []})
        elif stripped == MAINTAINERS and not paragraph and _opening(days):
            day()["entries"][-1]["maintainers"] = True
            paragraph.append(stripped)
        else:
            # Everything else, `#` and `####` included, is a line of prose.
            paragraph.append(stripped)
    close()
    return days


def _opening(days: list[dict[str, Any]]) -> bool:
    """Whether the next paragraph is the first of an entry that has a heading."""
    entries = days[-1]["entries"] if days else []
    return bool(entries) and bool(entries[-1]["heading"]) and not entries[-1]["blocks"]


def for_visitors(payload: dict[str, Any]) -> dict[str, Any]:
    """The changelog as the site serves it: no entry marked `MAINTAINERS`, and no day
    that held only such entries."""
    days = []
    for day in payload.get("days") or []:
        entries = [entry for entry in day["entries"] if not entry.get("maintainers")]
        if entries:
            days.append({**day, "entries": entries})
    note = str(payload.get("note") or "")
    if not days and not note:
        note = "The changelog holds no entries yet."
    return {**payload, "days": days, "note": note}


def body(text: str) -> str:
    """The file below its header.

    The header is the maintainer's instructions to themselves and ends at the first
    `---`. Without one -- a file somebody rewrote -- the first day heading is where the
    entries start, and a file with neither is shown whole.
    """
    lines = text.splitlines()
    for at, line in enumerate(lines):
        if _RULE.match(line.strip()):
            return "\n".join(lines[at + 1:])
    for at, line in enumerate(lines):
        if line.startswith("## "):
            return "\n".join(lines[at:])
    return text


def runs(text: str) -> list[dict[str, str]]:
    """One paragraph as `text`, `code` and `em` pieces, in order.

    Nothing is escaped, because nothing becomes markup: `t` says which element the page
    makes and `s` is that element's text.
    """
    pieces: list[dict[str, str]] = []

    def plain(said: str) -> None:
        if said:
            pieces.append({"t": "text", "s": said})

    at = 0
    for found in _INLINE.finditer(text):
        plain(text[at:found.start()])
        code, emphasis = found.group(1), found.group(2)
        pieces.append({"t": "code", "s": code} if code is not None
                      else {"t": "em", "s": emphasis or ""})
        at = found.end()
    plain(text[at:])
    return pieces


def _day(heading: str) -> dict[str, Any]:
    """A day heading as its own text and as a date a reader reads.

    `2026-09-18` is how the file is written and `18 September 2026` is what the page
    prints. A heading that is not a date keeps its own words -- the file decides what a
    day is called, and a parser that insisted on a format would drop the heading.
    """
    try:
        when = date.fromisoformat(heading)
    except ValueError:
        return {"date": "", "words": heading, "entries": []}
    return {
        "date": heading,
        "words": f"{when.day} {_MONTHS[when.month - 1]} {when.year}",
        "entries": [],
    }
