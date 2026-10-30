"""What the user page's workshop is sent, read on the server (UP1, `docs/archive/plan-user-page.md` §5).

The profile's *Your games* names each of a person's games with its level and its next step
(`stages.yours`). The page the user chose (§4, direction B, *the workshop*) draws each as a
card: its board, the four levels with its own lit, the rungs from a draft to a score with
what is known of each, the game it was made from with that game's level and score, and the
change from it in the designer's words. Beside it, *Waiting on you* draws each waiting game
as a small board with its last move. Every one of those is an answer, and this module is
where the server gives it, so the page works nothing out (rule 3).

- **The board** is the listing's own still (`library.still`), each piece's drawing named
  by its stem and sent once for the whole payload (`pieceArt`), as `/api/library` sends it.
- **The track** is `stages.track` of the level `stages.yours` already decided.
- **The rungs** are the designer's (`candidacy.rungs`), in the site's words, each with what
  the store knows of it: the save's check (`screen_passed`), the save itself, the score
  attached from the designer's computer (`drafts.attached`, `provisional.checking`), and the
  level. The store keeps the screen's verdict and not its counts (`0016_drafts.sql`); UP1
  says the rung's words alone rather than add a column, because a count stored at save
  would be missing for every draft saved before it, and the words are honest for all.
- **The parent** is the row's `parent`: its name, its level (`stages.read`, or where its
  file is for a game the store has no row for) and its score with the basis that score
  carries (status, revision, date, runs, re-certification). A provisional score keeps its
  status, so a page cannot draw it bare.
- **The change** is `designing._changes_from`, the designer's own sentences for the same
  pair, so the profile and the designer cannot word one change two ways.
- **A waiting game's board** (`position`) is the match replayed through the engine
  (`matches.session`), drawn as a still is (`card_rules.preview` on the state it reached),
  with its `lastMove` the session's own history entry. A challenge has no moves yet, so its
  board is the game's opening still and its `lastMove` is `None`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.site import drafts, kept, matches, provisional, stages
from chess_variant_sandbox.web import candidacy, card_rules, library
from chess_variant_sandbox.web.designing import _changes_from, settled

DONE, FAILED, RUNNING, AHEAD = "done", "failed", "running", "ahead"
"""A rung's state, in the marks the designer's ladder draws (`design.js` `RUNG_MARKS`)."""

SCREEN_UNKNOWN = "not known: the check stopped at this step or the next"
"""The smoke rung of a draft whose save's check did not pass. The store keeps one verdict
for both playing rungs, so which of the two stopped it is not known, and is not guessed."""

PUBLISHED_STEPS = (stages.ON_TRIAL, stages.SHORTLISTED, stages.RESTING, stages.RETIRED)
"""Levels a designed game reaches only through *Publish*, which needs a checked score."""

SCORED_BY_SITE = (stages.LIBRARY, stages.BASELINE, stages.PREVIEW)
"""Levels where the score is the site's own measurement, not one from the designer."""


MEASURED = "Measured"
"""The workshop's one column that is not a level (UP2, §3.2 of the plan): a draft whose
score from the designer's computer the site has checked (`drafts.attached`), the step a
draft waits on before *Publish*. Its level is still Draft."""

COLUMNS = (stages.DRAFT, MEASURED, stages.ON_TRIAL, stages.SHORTLISTED, stages.LIBRARY)
"""The workshop's columns, the path every game takes, in order. Always drawn, so the path
reads whole; a level off the path (Resting, Retired, Preview) is a column after them only
while it holds a game."""

COLUMN_SAYS = {
    stages.DRAFT: "yours, and anybody you send its link; checked as you edit",
    MEASURED: drafts.READING_CHECKED,
    stages.ON_TRIAL: "listed for anybody to play and vote on",
    stages.SHORTLISTED: "members vote on it at the month's round",
    stages.LIBRARY: "scored by the site and listed for everyone",
    stages.RESTING: "off the listing until it is played again; its link still works",
    stages.RETIRED: "not offered for new games; its page and games stay",
    stages.PREVIEW: "a new kind of game, still being tested",
}
"""One short line under each column's word, the level's sentence cut to fit a column."""

FEATURED = {
    stages.ON_TRIAL: "Your game on trial",
    stages.SHORTLISTED: "Your game, shortlisted",
    stages.LIBRARY: "Your game in the library",
}
"""The kicker over the game the page shows large beside what waits (§3.2): the first game
in the last of these columns that holds one, the member's game furthest along the path."""


@dataclass(frozen=True)
class Kept:
    """One of a person's games as the store holds it, with what its rungs read."""

    hash: str
    name: str
    spec: dict[str, Any]
    parent: str
    screened: bool
    level: str
    waking: bool
    reading: str
    """`checked`, `checking` or `none`: the score from the designer's computer, for a game
    that has not been published yet (a draft, or a resting game that may come back)."""


def kept_rows(
    connection: Any, owner: int, current: provisional.Current = provisional.one_run
) -> dict[str, Kept]:
    """This person's games, as `stages.yours` lists them (not taken off, not deleted)."""
    columns = ", ".join(stages.COLUMNS)
    with connection.cursor() as cursor:
        cursor.execute(
            f"select name, spec, parent, screen_passed, {columns} from games"
            " where owner = %s and revoked is null and deleted is null",
            (owner,),
        )
        found = cursor.fetchall()
    rows: dict[str, Kept] = {}
    for name, spec, parent, screened, *rest in found:
        row = dict(zip(stages.COLUMNS, rest, strict=True))
        level = stages.stage(row)
        waking = level == stages.RESTING and row.get("woken") is None
        digest = str(row["hash"])
        reading = "none"
        if level == stages.DRAFT or waking:
            if drafts.attached(connection, digest, current) is not None:
                reading = "checked"
            elif provisional.checking(connection, digest):
                reading = "checking"
        rows[digest] = Kept(
            hash=digest, name=str(name), spec=dict(spec or {}), parent=str(parent or ""),
            screened=bool(screened), level=level, waking=waking, reading=reading,
        )
    return rows


def rungs(found: Kept) -> list[dict[str, Any]]:
    """The designer's rungs in the site's words, each with what is known of it.

    `state` is one of `DONE`, `FAILED`, `RUNNING`, `AHEAD`; `says` is the known answer in
    words; `next` marks the first rung not done, which is the designer's next step.
    """
    playable = found.screened
    if found.level in SCORED_BY_SITE:
        scored = (DONE, "scored by the site")
    elif found.level in PUBLISHED_STEPS and not found.waking or found.reading == "checked":
        scored = (DONE, drafts.READING_CHECKED)
    elif found.reading == "checking":
        scored = (RUNNING, drafts.READING_CHECKING)
    else:
        scored = (AHEAD, f"not yet: {drafts.READING_TICK}")
    known = {
        # A saved game passed the schema and the compiler: a save refuses anything else.
        "validate": (DONE, "yes"),
        "smoke": (DONE, "yes") if playable else (AHEAD, SCREEN_UNKNOWN),
        "screen": (DONE, "it passes the playability check") if playable
        else (FAILED, "it has not passed the playability check"),
        "save": (DONE, "yes, as your game"),
        "canonical": scored,
        "nominate": (DONE, "already one of them") if found.level == stages.BASELINE
        else (AHEAD, "decided by the person who runs the site"),
    }
    shown: list[dict[str, Any]] = []
    waiting = True
    for rung in candidacy.rungs():
        words = {**rung, **(rung.get("site") or {})}
        state, says = known.get(str(rung["key"]), (AHEAD, ""))
        first = waiting and state != DONE
        waiting = waiting and not first
        shown.append({"key": rung["key"], "name": words["name"], "state": state,
                      "says": says, "next": first})
    return shown


def column(found: Kept) -> str:
    """The workshop column a game stands in: its level, except a draft whose score from
    the designer's computer the site has checked, which stands in *Measured*, and a
    Baseline game, which is in the library's column under its own level word."""
    if found.level == stages.DRAFT and found.reading == "checked":
        return MEASURED
    if found.level == stages.BASELINE:
        return stages.LIBRARY
    return found.level


def step(found: Kept, said: str) -> str:
    """A game's next step in words, for its card in the workshop.

    A draft's is the designer's next rung: the playability check where the save's check
    did not pass, then a score measured on the designer's computer, then *Publish*; a draft
    whose designer pressed *Publish* while the site checks says so (`said`, the server's
    `stages.WAITING`). Every other level's is `said`, `stages.yours`' own next step.
    """
    if found.level != stages.DRAFT or said == stages.WAITING:
        return said
    if not found.screened:
        return "Next: make it pass the playability check, in the designer."
    if found.reading == "checked":
        return "Next: publish it from the designer to put it on trial."
    if found.reading == "checking":
        return f"Next: {drafts.READING_CHECKING}."
    return f"Next: {drafts.READING_TICK}, from the designer."


MEASURE = "measure"
"""`stepAt` for a draft whose next step is the score from the designer's computer: the
designer's `&at=measure`, which opens the draft at its *measure it here* (UP7)."""

MEASURE_NOTE = "Measuring takes a few minutes, and it is easier on a computer."
"""The one line under that step (UP7, `user-page-cold-read.md` finding 2): a phone reader
otherwise guessed it, from the designer's own phone note."""


def step_at(found: Kept, said: str) -> str:
    """Where in the designer a game's next step is taken: `MEASURE`, or the empty string
    for a step that is not the measuring one (`step` says them all in words)."""
    if found.level != stages.DRAFT or said == stages.WAITING:
        return ""
    return MEASURE if found.screened and found.reading not in ("checked", "checking") else ""


def featured(word: str, count: int) -> str:
    """The kicker over the game shown large (`FEATURED`), saying when it is one of several
    at its level: *One of your 2 games on trial* (UP7, the cold read's finding 3)."""
    kicker = FEATURED.get(word, "")
    if not kicker or count < 2:
        return kicker
    return f"One of your {count} games {FEATURED_SEVERAL[word]}"


FEATURED_SEVERAL = {
    stages.ON_TRIAL: "on trial",
    stages.SHORTLISTED: "shortlisted",
    stages.LIBRARY: "in the library",
}
"""`FEATURED`'s place, for a kicker over one of several games there."""


def columns(groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The workshop's columns, each with its word, its line and the hashes standing in it
    in `stages.yours`' order. Empty when the person has made no game: a path with nothing
    on it is not drawn."""
    held: dict[str, list[str]] = {}
    for group in groups:
        for game in group["games"]:
            if "column" in game:
                held.setdefault(str(game["column"]), []).append(str(game["hash"]))
    if not held:
        return []
    order = [*COLUMNS, *(level for level in stages.LEVELS if level not in COLUMNS)]
    order += sorted(set(held) - set(order))
    return [
        {"word": word, "says": COLUMN_SAYS.get(word, stages.SAYS.get(word, "")),
         "games": held.get(word, []), "feature": featured(word, len(held.get(word, [])))}
        for word in order
        if word in COLUMNS or held.get(word)
    ]


def _stems(shown: Mapping[str, Any], art: dict[str, str]) -> dict[str, Any]:
    """A drawn position with each piece's drawing named by its stem, as the listing's
    stills are (`card_rules.still`), the drawings gathered into `art`."""
    named: list[list[dict[str, Any] | None]] = []
    for pair in shown.get("icons") or []:
        row: list[dict[str, Any] | None] = []
        for icon in pair:
            if icon is None:
                row.append(None)
                continue
            art[str(icon["stem"])] = str(icon["href"])
            row.append({"stem": icon["stem"], "rotate": bool(icon.get("rotate"))})
        named.append(row)
    return {**shown, "icons": named}


def still(game: library.Game | None, art: dict[str, str]) -> dict[str, Any] | None:
    """A game's opening still, the listing's own, its drawings gathered into `art`."""
    if game is None:
        return None
    art.update(library.piece_art(game))
    return library.still(game)


def score(game: library.Game | None) -> dict[str, Any] | None:
    """A game's score with the basis it carries, or `None` where there is none to show.

    The fields are the card's (`library.Game.summary`), so a provisional number arrives
    with `status` `provisional`, its revision and its date, and is never bare.
    """
    if game is None or game.status == library.NONE:
        return None
    found = game.score or {}
    return {
        "index": found.get("index"),
        "status": game.status,
        "revision": found.get("revision"),
        "measuredAt": found.get("measured_at"),
        "certified": found.get("certified"),
        "viable": found.get("viable"),
        "measured": found.get("measured") is not False,
        "weights": found.get("weights_version"),
        "seeds": list(found.get("seeds") or []),
        "totalsUsed": found.get("totals_used"),
    }


def parent(
    connection: Any,
    digest: str,
    known: Mapping[str, library.Game],
    stored: Callable[[str], library.Game | None],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The game this one was made from, as the card names it, and its spec; or two `None`s.

    Its level is its row's (`stages.read`); a game the store has no row for reads where its
    file is. A parent whose rules are gone keeps its name and has no score and no spec.
    """
    if not digest:
        return None, None
    game = known.get(digest) or stored(digest)
    row = stages.read(connection, [digest]).get(digest)
    if row is not None:
        level = str(row["level"])
    elif game is not None:
        level = stages.local(game.source)
    else:
        level = ""
    with connection.cursor() as cursor:
        cursor.execute("select name from games where hash = %s", (digest,))
        named = cursor.fetchone()
    name = game.name if game is not None else (str(named[0]) if named else "")
    return (
        {"hash": digest, "name": name, "level": level,
         "levelGloss": stages.GLOSSES.get(level, ""), "score": score(game)},
        None if game is None else game.spec,
    )


def change(parent_spec: dict[str, Any] | None, spec: dict[str, Any]) -> list[str] | None:
    """The change from the parent in the designer's words (`designing._changes_from`)."""
    if parent_spec is None:
        return None
    try:
        return _changes_from(parent_spec, settled(spec))
    except Exception:  # noqa: BLE001 - as the check: the parent is only a comparison
        return None


def designs(
    connection: Any,
    owner: int,
    groups: Iterable[dict[str, Any]],
    known: Mapping[str, library.Game],
    runs_dir: Any,
    art: dict[str, str],
    current: provisional.Current = provisional.one_run,
) -> None:
    """Add each game's `still`, `track`, `column`, `step` (with `stepAt` and `stepNote`),
    `rungs`, `parent` and `change` to `stages.yours`' groups, in place, gathering the
    drawings into `art`."""
    rows = kept_rows(connection, owner, current)

    def stored(digest: str) -> library.Game | None:
        return game_for(connection, digest, {}, runs_dir)

    for group in groups:
        for game in group["games"]:
            found = rows.get(str(game["hash"]))
            if found is None:
                continue
            own = known.get(found.hash) or _from_row(
                found.hash, found.name, found.spec, runs_dir
            )
            made_from, parent_spec = parent(connection, found.parent, known, stored)
            game["still"] = still(own, art)
            game["track"] = stages.track(found.level)
            game["column"] = column(found)
            game["step"] = step(found, str(game.get("next") or ""))
            game["stepAt"] = step_at(found, str(game.get("next") or ""))
            game["stepNote"] = MEASURE_NOTE if game["stepAt"] == MEASURE else ""
            game["rungs"] = rungs(found)
            game["parent"] = made_from
            game["change"] = change(parent_spec, found.spec)


def position(
    connection: Any, match_id: str, art: dict[str, str]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """A running match's position as it stands, drawn as a still, and its last move.

    Two `None`s for a match whose moves no longer replay (`matches.MatchError`): the page
    names the game and draws no board, as it does for one whose rules are gone.
    """
    try:
        session = matches.session(connection, match_id)
    except (matches.MatchError, KeyError, ValueError):
        return None, None
    engine = session.engine
    shown = card_rules.preview(engine, session.state, session.game.spec)
    last = session.history[-1] if session.history else None
    return _stems(shown, art), last


def waiting(
    connection: Any,
    entries: Iterable[dict[str, Any]],
    art: dict[str, str],
    shut: Iterable[str] = (),
) -> None:
    """Add `position` and `lastMove` to each running game's entry, in place. A game whose
    rules the store will not hand back (`shut`) gets neither drawn."""
    closed = set(shut)
    for entry in entries:
        if entry["game"] in closed:
            entry["position"], entry["lastMove"] = None, None
            continue
        entry["position"], entry["lastMove"] = position(connection, str(entry["id"]), art)


def challenged(
    connection: Any,
    entries: Iterable[dict[str, Any]],
    known: Mapping[str, library.Game],
    runs_dir: Any,
    art: dict[str, str],
) -> None:
    """Add a challenge's board, its game's opening still, and a `lastMove` of `None`."""
    for entry in entries:
        entry["position"] = still(game_for(connection, str(entry["game"]), known, runs_dir), art)
        entry["lastMove"] = None


def game_for(
    connection: Any, digest: str, known: Mapping[str, library.Game], runs_dir: Any
) -> library.Game | None:
    """A game from the tree's listing, else from the store's copy of its rules
    (`kept`, as `SiteApp.game` falls back), else `None` for rules that are gone."""
    if digest in known:
        return known[digest]
    found = kept.rules_of(connection, digest)
    return None if found is None else _from_row(digest, found[0], found[1], runs_dir)


def _from_row(
    digest: str, name: str, spec: dict[str, Any], runs_dir: Any
) -> library.Game | None:
    """`kept.game_from`, or `None` for rules that no longer compile: a profile must
    survive one unloadable row, as the listing does (`library.still`), and that game's
    card then has no board and no score rather than the page failing."""
    try:
        return kept.game_from(digest, name, spec, runs_dir)
    except Exception:  # noqa: BLE001 - one unloadable row is not the page's failure
        return None
