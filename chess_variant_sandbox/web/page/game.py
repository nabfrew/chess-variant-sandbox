"""A game's own address, drawn on the server (G3 of `docs/plan-growth.md`).

Every game used to live behind a fragment, `/#/game/<hash>`, and everything after `#` never
reaches a server: to a crawler or a link unfurler every game was one untitled page. So a game
has an address of its own, `/game/<hash>`, and the server answers it with the single page
(`page`) carrying, before any script runs:

- the game's name in `<title>`, a description (the first sentence of its own, under 160
  characters) and a canonical address in the head; `noindex` where the game is not listed;
- in `#view`, its name, board, description, its score with what the score rests on (the run
  count, the scoring rules' version and the day; a provisional one marked as provisional, as
  everywhere), and links to the games most like it, each at its own address.

The router replaces `#view` once it draws, so a person with script sees the page they always
saw; this is what a reader without script, a crawler and an unfurler get. The hash stays in
the address because it is the game's identity: a rename moves the name, never the address.

Nothing here reads a store or a file. The caller hands over what is to be drawn (`Drawn`),
which is what makes the page cheap and testable: `site/game_page.py` reads the store and
the library, and this draws.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

from chess_variant_sandbox.web.page import front, game_picture, page
from chess_variant_sandbox.web.page import picture as site_picture
from chess_variant_sandbox.web.page.shell import NAME

# The address. The hash is `spec.canonical.HASH_LENGTH` lowercase hex characters today; the
# pattern takes a range so the address outlives a change of length, and a lookup decides.
PREFIX = front.GAMES_PATH + "/"
ROUTE = re.compile(r"^/game/([0-9a-f]{8,64})$")

# A search result shows about this many characters of a description.
DESCRIPTION_LIMIT = 160

# How many near games the page names: the most alike first, in the order the game's page
# and the map list them (`library.neighbours`, M5 of `docs/plan-map.md`).
NEAR_SHOWN = 12

_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
)
_COUNT_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten")


def path(digest: str) -> str:
    """A game's own address on this server."""
    return PREFIX + digest


def digest_of(asked: str) -> str | None:
    """The hash a path names, or `None` for any other path."""
    found = ROUTE.match(asked)
    return None if found is None else found.group(1)


@dataclass(frozen=True)
class Near:
    """One game near this one, as a link: its hash and its name."""

    hash: str
    name: str


@dataclass(frozen=True)
class Drawn:
    """Everything the page draws, and nothing it would have to look up.

    `score` is `library.Game.score` (the latest estimate with its `status`, `revision`,
    `measured_at` and `seeds`, or `None`), `stale` whether a measurement exists that is no
    longer current, `indexable` whether the game is listed (an unlisted draft, a retired or a
    resting game opens by its link and says `noindex`), `base` the site's own address,
    without a trailing slash, for the canonical (empty: the canonical is the path alone), and
    `picture` the path of the game's own link-preview picture where one is made (G6,
    `site/game_pictures.py`; empty: the site's picture stands in).
    """

    hash: str
    name: str
    description: str
    spec: Mapping[str, Any]
    score: Mapping[str, Any] | None
    stale: bool
    near: tuple[Near, ...]
    indexable: bool
    base: str = ""
    picture: str = ""


# ------------------------------------------------------------------------ the words


def paragraphs(description: str) -> list[str]:
    """A description's paragraphs: blank lines part them, other line breaks are spaces."""
    return [
        " ".join(block.split())
        for block in re.split(r"\n\s*\n", description or "")
        if block.strip()
    ]


def first_sentence(text: str, limit: int = DESCRIPTION_LIMIT) -> str:
    """The first sentence of `text`, cut at a word to fit `limit` characters with an ellipsis.

    A sentence ends at `.`, `!` or `?` followed by a space; a number like *0.5* has none.
    """
    flat = " ".join((text or "").split())
    found = re.match(r"^(.+?[.!?])(?:\s|$)", flat)
    sentence = found.group(1) if found else flat
    if len(sentence) <= limit:
        return sentence
    cut = sentence[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return cut + "…"


def board_words(spec: Mapping[str, Any]) -> str:
    """*8 × 8 board · 6 kinds of piece*, from the rules, or as much of it as they say."""
    found = spec.get("board")
    board: Mapping[str, Any] = found if isinstance(found, Mapping) else {}
    width, height = board.get("width"), board.get("height")
    said: list[str] = []
    if isinstance(width, int) and isinstance(height, int):
        said.append(f"{width} × {height} board")
    kinds = spec.get("pieces")
    if isinstance(kinds, Sequence) and not isinstance(kinds, str) and kinds:
        said.append(f"{len(kinds)} kind{'' if len(kinds) == 1 else 's'} of piece")
    return " · ".join(said)


def meta_description(drawn: Drawn) -> str:
    """The search result's line: the description's first sentence, or one said for it."""
    said = paragraphs(drawn.description)
    if said:
        return first_sentence(said[0])
    return first_sentence(
        f"{drawn.name or 'An unnamed game'}, a chess-like game: play it in the browser "
        "against the computer, with no account."
    )


def _day(iso: Any) -> str:
    found = re.match(r"^(\d{4})-(\d{2})-(\d{2})", str(iso or ""))
    if not found:
        return ""
    return f"{int(found.group(3))} {_MONTHS[int(found.group(2)) - 1]} {found.group(1)}"


def _rules_version(revision: Any) -> str:
    found = re.match(r"^v(\d+)$", str(revision or ""))
    return "scoring rules version " + (found.group(1) if found else str(revision or ""))


def _runs(seeds: Any) -> str:
    count = len(seeds) if isinstance(seeds, Sequence) and not isinstance(seeds, str) else 0
    if count == 0:
        return ""
    if count == 1:
        return "from one run"
    word = _COUNT_WORDS[count] if count < len(_COUNT_WORDS) else str(count)
    return f"the average of {word} runs"


def score_line(score: Mapping[str, Any] | None, stale: bool) -> str:
    """The score in a sentence, with what it rests on, in the site's words
    (`docs/reference/plain-words.md` §2, §3): never a bare number, never a provisional one
    as current, and *out of date* when there is no number to show."""
    index = None if score is None else score.get("index")
    if score is not None and isinstance(index, (int, float)):
        runs = _runs(score.get("seeds"))
        revision = score.get("revision") or score.get("weights_version")
        day = _day(score.get("measured_at"))
        number = f"Score {index:.3f} out of 1"
        if score.get("status") == "provisional":
            stamp = ", ".join(part for part in (
                _rules_version(revision) if revision else "", day) if part)
            return (
                f"{number}{', ' + runs if runs else ''}, provisional: worked out before the "
                "latest change to how games are scored"
                f"{f' ({stamp})' if stamp else ''}. It will be measured again."
            )
        basis = [part for part in (runs, f"under {_rules_version(revision)}" if revision
                                   else "") if part]
        return (
            f"{number}{', ' + ', '.join(basis) if basis else ''}"
            f"{f', measured {day}' if day else ''}."
        )
    if score is not None and _runs(score.get("seeds")):
        return ("No score: too few of its runs could be scored, so it has no score rather "
                "than a low one.")
    if stale:
        return ("Out of date: measured before the computer players or the scoring last "
                "changed. No score is shown until it is measured again.")
    return "Not measured yet, so it has no score."


SCORE_MEANS = "The score is how interesting a game looks when the computer plays it, out of 1."


# ------------------------------------------------------------------------- the page


def title(name: str) -> str:
    """The tab's and the search result's title; `core.js`'s `gameTitle` says the same."""
    return f"{name or 'An unnamed game'}: play it online · {NAME}"


def canonical(drawn: Drawn) -> str:
    return drawn.base + path(drawn.hash)


def picture(drawn: Drawn) -> str:
    """The game's link-preview picture, a whole address where the site has one: its own
    starting position (G6, `game_picture.py`) once that is made, the site's picture until
    then, so the tag never names a picture that is not there."""
    if drawn.picture:
        return drawn.base + drawn.picture
    return front.picture_address(drawn.base)


def picture_alt(drawn: Drawn) -> str:
    """What `picture` shows, for a reader who cannot see it."""
    return game_picture.alt(drawn.name) if drawn.picture else site_picture.ALT


def head(drawn: Drawn) -> str:
    """The lines the head carries beyond the title (`front.head_lines`)."""
    return front.head_lines(title(drawn.name), meta_description(drawn), canonical(drawn),
                            drawn.indexable, image=picture(drawn),
                            image_alt=picture_alt(drawn))


def view(drawn: Drawn) -> str:
    """The game's page as a reader without script reads it; the router draws over it."""
    name = escape(drawn.name or "An unnamed game")
    about = "".join(
        f'<p class="gameAbout">{escape(block)}</p>' for block in paragraphs(drawn.description)
    )
    board = board_words(drawn.spec)
    near = drawn.near[:NEAR_SHOWN]
    near_html = (
        '<h3>Games like this one</h3>\n<ul class="drawnNear">'
        + "".join(
            f'<li><a href="{escape(path(one.hash), quote=True)}">'
            f'{escape(one.name or "An unnamed game")}</a></li>'
            for one in near
        )
        + "</ul>"
    ) if near else ""
    return (
        '<article class="drawnGame" id="drawnGame">\n'
        '<a class="crumb" href="/game">&larr; all games</a>\n'
        f'<h2 style="margin:10px 0 0">{name}</h2>\n'
        + (f'<div class="row muted small" style="margin-top:10px">{escape(board)}</div>\n'
           if board else "")
        + about
        + f'\n<p class="drawnScore">{escape(score_line(drawn.score, drawn.stale))} '
        f'<span class="muted">{escape(SCORE_MEANS)}</span></p>\n'
        '<p>Play it here in the browser, against the computer, with no account needed.</p>\n'
        '<noscript><p class="muted">Playing needs JavaScript switched on.</p></noscript>\n'
        f"{near_html}\n</article>"
    )


def draw(drawn: Drawn) -> str:
    """The whole page for a game's own address."""
    return page(title(drawn.name), head(drawn), view(drawn))
