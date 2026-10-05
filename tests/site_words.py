"""The words a visitor to the public site must never be shown, and how a test finds them.

`docs/plan-understanding.md` §7, W0. The audit of §3 found the site speaking in the
maintainer's voice: a path into `runs/`, *git*, *the command line*, an agent's string,
*plies*, *the yardstick*. Each word task of §7 removes some of it and, when it does, moves
the phrase from `NOT_YET` to `ENFORCED`, so a later change cannot put it back.

A phrase is a regular expression and the flags it is read with. `hits(text)` is the list of
phrases (by name) that `text` contains. Two rules keep it about prose and not about code:

- `prose(value)` is whether a string is a sentence a person reads: two words or more. A
  hash, a key, a route or an `href` is data, and the phrase *a twelve-hex hash in prose*
  means a hash inside a sentence, never a game's hash on its own.
- A page's text is its visible words (`visible`), never its markup or its scripts: `git`
  must not be found in `digit` or in a class name.

The collection (what the site serves, who serves it) is each test's own; this module is the
list and the matching, shared by `test_candidacy.py` and `test_site_words.py`.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterator
from typing import Any

# name -> (regular expression, flags). The name is what a failure prints.
PHRASES: dict[str, tuple[str, int]] = {
    "runs/": (r"runs/", re.I),
    "git": (r"\bgit\b", re.I),
    "command line": (r"command line", re.I),
    "chess-variant-sandbox ": (r"chess-variant-sandbox ", 0),
    "PATH": (r"\bPATH\b", 0),
    "minimax:": (r"minimax:", re.I),
    "nodes=": (r"nodes=", re.I),
    "a twelve-hex hash": (r"\b[0-9a-f]{12}\b", 0),
    "plies": (r"\bplies\b", re.I),
    "weights v": (r"weights v", re.I),
    "checkout": (r"checkout", re.I),
    "corpus": (r"corpus", re.I),
    "schema": (r"schema", re.I),
    "candidate": (r"candidate", re.I),
    "yardstick": (r"yardstick", re.I),
    "leaderboard": (r"leaderboard", re.I),
    "sandbox mode": (r"sandbox mode", re.I),
    # Q579 (F7, F8; rows P1-P21, S1-S12): an opponent in the machine's terms, and the
    # profile's jargon.
    "searches N positions": (r"\bsearch(?:es|ing)? [\d,]+ positions", re.I),
    "lookahead": (r"lookahead", re.I),
    "search depth": (r"search depth", re.I),
    "playouts": (r"playouts", re.I),
    "positions a move": (r"positions a move", re.I),
    "an engine somebody else wrote": (r"engine somebody else wrote", re.I),
    "rungs": (r"\brungs?\b", re.I),
    "tops beaten": (r"\btops beaten\b", re.I),
    "behaviour grid": (r"behaviou?r grid", re.I),
    "idle sweep": (r"idle sweep", re.I),
    # W6 (rows S13-S25): the site's own pages and its server sentences. *maintainer* is
    # *the person who runs the site* there (`/admin` keeps it, and the guard does not read
    # `/admin`); *beat the top* is *beat the strongest*; *a game that counts* is *a rated
    # game*; *the sweep* is *resigned after thirty days without a move*.
    "maintainer": (r"\bmaintainer", re.I),
    "beat the top": (r"\bbeat(?:en|s)? the top\b", re.I),
    # Bare: *a game that counts towards the ratings* says what it counts for, and stays.
    "a game that counts": (r"\bgames? that counts?\b(?! towards)", re.I),
    "the sweep": (r"\bthe sweep\b", re.I),
    # W2 (rows B1-B23, M1-M13): the library's cards and the map. A visitor's words are
    # *likeness*, *the slider*, *a small change to the rules*, *games played by people*,
    # *ruled out* and *not played here yet*.
    "similarity index": (r"similarity index", re.I),
    "the cut": (r"\b(?:the|this) cut\b", re.I),
    "design step": (r"\bdesign steps?\b", re.I),
    "fifty-five features": (r"fifty-five features", re.I),
    "human games": (r"\bhuman games?\b", re.I),
    "the gate says": (r"the gate says", re.I),
    "nobody's played this": (r"nobody's played this", re.I),
    "revisions deep": (r"revisions deep", re.I),
    "translates to Fairy-Stockfish": (r"translates to Fairy-Stockfish", re.I),
    # W5a (rows D1-D39): the designer's own words. A visitor reads *saved as your draft*,
    # *a computer player*, *working it out*, *the game's text* and no rule-language version.
    "rule set version": (r"rule set version", re.I),
    "the engine answered": (r"the engine answered", re.I),
    "searching player": (r"\bsearching player", re.I),
    "not a ruleset yet": (r"not a ruleset yet", re.I),
    "not an object": (r"is not an object", re.I),
    "play-tested": (r"play-tested", re.I),
    "makes nothing of this game": (r"makes nothing of this game", re.I),
    "asking the engine": (r"asking the engine", re.I),
    # Q563 (L5; rows G15, G24, D23, D35, G44): the refusing sentences. The site said why it
    # withholds a route in its own terms (*a tournament the site does not offer yet*, *does
    # not measure or nominate games*) and called a measured score a *reading*.
    "does not offer yet": (r"does not offer yet", re.I),
    "measure or nominate": (r"measure or nominate", re.I),
    "a reading from a computer": (
        r"\breadings? from (?:your|its designer's|their own) computer|check a reading", re.I),
    "refresh it": (r"\brefresh it\b", re.I),
    # W5b (rows D40-D50, M7): the rules' own complaints. A problem names its place the way
    # the designer draws it (*Where they start, row 1*, *Piece 2 (knight), way of moving
    # 1*, *the rule 'the turn limit'*), never by its path in the game's text; a gate reason
    # speaks of *sides*, *turns* and *rules that don't fit together*; a rule card's field
    # is never its key; a small change to the rules names the rule, not its key.
    "a path into the game's text": (
        r"\b(?:pieces|placements|moves|npcs|links|chain|promotion)\[\d+\]"
        r"|\b(?:rules|setup|board)\.[a-z_]+\b(?!\.json)", 0),
    "does not compile": (r"does not compile", re.I),
    "run out of clock": (r"run out of clock", re.I),
    "one seat": (r"\bone seat\b", re.I),
    "under random play": (r"under random play", re.I),
    "the clock cut short": (r"clock cut short", re.I),
    "a rule's key": (r"\b(?:max plies|no progress draw|explode on capture|onto to|mover to"
                     r"|promote on entry)\b", re.I),
    "a declared piece": (r"\bdeclared piece", re.I),
    "a non-empty string": (r"non-empty string", re.I),
    # Q647 (`plan-confidence.md` T2): the strength check's own terms. The trust line says
    # what the evidence found in a visitor's words; the self-duel's *ceiling*, a duel's
    # *floor* and a check's *verdict* stay on `/admin` and the local application. *Not a
    # verdict*, the purpose's own words for what a score is not (`plan-understanding.md`
    # §1.1), is not one.
    "ceiling": (r"\bceiling", re.I),
    "floor": (r"\bfloors?\b", re.I),
    "verdict": (r"(?<!not a )\bverdicts?\b", re.I),
}

# What the site serves today that still holds a phrase, one entry per phrase with the rows of
# `docs/plan-understanding.md` §3.3 that remove it and the task of §7 that owns them.
# **Empty when the audit is done.** A word task that removes a phrase from everything the
# guard reads deletes its entry here, which enforces it; the entry's own test fails the day
# the phrase is gone from everything read and the entry is still here. Every entry is a
# promise somebody else keeps, so it names where it is kept and what still serves it.
NOT_YET: dict[str, str] = {}
# *a path into the game's text* was the last entry: Fairy-Stockfish's reasons named a rule
# by its path until Q681 put them in the designer's words (`fairy.words`).

ENFORCED = tuple(name for name in PHRASES if name not in NOT_YET)


def hits(text: str, phrases: tuple[str, ...] | None = None) -> list[str]:
    """The names of `phrases` (by default `ENFORCED`) found in `text`."""
    return [name for name in (ENFORCED if phrases is None else phrases)
            if re.search(PHRASES[name][0], text, PHRASES[name][1])]


def prose(value: str) -> bool:
    """Whether `value` is a sentence: two words or more, at least one with a letter."""
    words = value.split()
    return len(words) >= 2 and any(re.search(r"[A-Za-z]", word) for word in words)


def strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Every prose string in a JSON value, with where it is (`.a.b[2]`)."""
    if isinstance(value, str):
        if prose(value):
            yield path, value
    elif isinstance(value, dict):
        for key, inner in value.items():
            yield from strings(inner, f"{path}.{key}")
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            yield from strings(inner, f"{path}[{index}]")


_BLOCKS = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.S | re.I)
_ATTRIBUTES = re.compile(r'\b(?:title|aria-label|placeholder|alt)="([^"]*)"')


def visible(page: str) -> str:
    """A page's words as one string: its text, its `title`s and `aria-label`s, no markup."""
    page = _BLOCKS.sub(" ", page)
    attributes = " ".join(_ATTRIBUTES.findall(page))
    text = re.sub(r"<[^>]+>", " ", page)
    return " ".join(html.unescape(f"{text} {attributes}").split())


def literals(script: str) -> Iterator[str]:
    """The prose in a script's quoted strings and template texts.

    Good enough for the short scripts of the site's own pages: a quoted run, with a
    template's `${...}` holes cut out, kept when it reads as a sentence. It does not parse
    JavaScript, so a regular expression or a comment with a quote in it can give a false
    string; the answer to that is to word the comment differently.
    """
    for match in re.finditer(r"'((?:[^'\\\n]|\\.)*)'|\"((?:[^\"\\\n]|\\.)*)\"|`((?:[^`\\]|\\.)*)`",
                             script, re.S):
        text = next(group for group in match.groups() if group is not None)
        text = re.sub(r"\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = " ".join(html.unescape(text).split())
        if prose(text):
            yield text
