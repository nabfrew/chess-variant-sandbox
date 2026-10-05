"""A draft saved on the site, as a `games` row an admin can take down.

Phase 2's first row of `docs/plan-website.md`: the designer saves into the `games` table
with a visibility and a fork parent, and the same hash saved twice is one row. The files
under `runs/designed/` are still written, because the designer's progress page and the
library read them. The row is what makes the draft the site's business -- a match needs
it (`matches.start`), and so does the per-game kill switch (`admin.kill_game`).

## Who sees a saved draft

Nobody who does not have its link. Since Q503a (`docs/plan-game-pipeline.md` §5) a save
keeps a **Draft**: the row is `unlisted` whatever the playability check says, and the
check's verdict is recorded beside it as `screen_passed`. An unlisted draft opens by its
link and plays in sandbox mode like any other game (§6 of `plan-website.md`: *unlisted
games by link only*), and the listing leaves it out.

A game goes **On trial** -- public, listed, playable by anybody -- only when its designer
presses *Publish* (`publish`). That needs the check passed and a spot-checked reading from
the designer's own computer attached (answer Q5), and it is capped per person
(`quota.PUBLISHED_AT_ONCE`, `quota.PUBLISHES_A_DAY`). The reading is the game's standing
provisional result, spot-checked on today's basis (`attached`, Q503e): the designer
measures the draft in their browser, the site walks every game (tier one) and replays a
sample exactly (tier two, `site/spotcheck.py`), and only then does *Publish* open. A draft
without one is refused with `NEEDS_A_READING`. An account a spot check flagged cannot
publish at all (`provisional.FLAGGED`) until an admin clears it.

**A copy is not a new game** (Q503c, option E2 of `docs/archive/plan-designer-ux.md` §3).
A save refuses a rename by hash, but the same rules spelled another way -- pieces in
another order, a symmetry written out, a default written in -- hash differently. So
*Publish* also asks `spec/relate.identity_key`, the digest of the rules' normal form,
against every game in the Library (`archive/`), the Baseline (`rulesets/`) and on trial,
and refuses a match by that game's name (`twin`). The key only ever *refuses*: it never
lets a draft borrow a reading, a rating or a card from the game it matches. Move order
follows spelling, so two spellings under one seed play different games, and a measurement
belongs to the hash it was taken under.

The designed games that a save had listed before this went back to Draft
(`0016_drafts.sql`), and `returned` names them for their designer's profile.

The site's listing carries a card from `runs/` only when the store holds a public row for
it that nobody has killed. A draft with files and no row is the site's to ignore: it cannot
be played as a match or taken down, so it is not offered. The corpus keeps its own rule,
because `chess-variant-sandbox site load` writes every game in `rulesets/` and `archive/` as public.

## The first saver keeps the row

A save whose hash the store already holds leaves that row as it is: its owner, its
visibility and its kill stamp. Saving somebody's killed rules again does not bring them
back, and a rename is not a new game. The one thing a later save can add is a pass: the
check is a verdict on the rules, and the rules are the hash, so a pass by anybody's save
is recorded on the row and a later failure does not take it back.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chess_variant_sandbox.site import links, provisional, quota, resting, stages, terms
from chess_variant_sandbox.spec import relate

DESIGNED = "designed"
"""A row's origin when a person built it on the site. The loader's are `corpus` and
`archive`, and `load.LOADED_ORIGINS` leaves every other origin alone."""

PUBLIC = "public"
UNLISTED = "unlisted"

MIGRATION = "0016_drafts.sql"
"""The migration that took saved games off the listing. When it ran is when they went."""

NEEDS_A_READING = (
    "Publishing needs a score measured on your computer and spot-checked by the site: "
    "measure it on this computer below, and once the site has checked it, a few minutes "
    "later, it can be published. Until then the game stays a draft: anybody with its link "
    "can open and play it."
)
"""What a press of *Publish* is told while the draft carries no spot-checked score on
today's basis, and what the designer prints beside its disabled button. One sentence in one
place, so the two cannot differ."""

READING_TICK = "a score measured on your computer, checked by the site"
"""The reading's tick on *Publish*: what a publish needs, met or not."""

CORPUS_SOURCES = ("committed", "archived")
"""What `library.discover` calls a card read from `rulesets/` or `archive/`."""


@dataclass(frozen=True)
class Kept:
    """The row a save left behind, whether this save wrote it or found it."""

    hash: str
    visibility: str
    created: bool
    revoked: bool
    screen_passed: bool = False

    @property
    def listed(self) -> bool:
        return self.visibility == PUBLIC and not self.revoked


def keep(
    connection: Any,
    digest: str,
    spec: Mapping[str, Any],
    owner: int,
    parent: str = "",
    passed: bool = False,
) -> Kept:
    """Write the draft's row, or find the one already there. Nothing commits.

    A new row is `unlisted` whatever the check said, and `passed` -- the playability
    check's verdict -- is recorded as `screen_passed`. `parent` is kept only when the store
    holds that game, because `games.parent` references `games.hash`. A fork of a draft
    nobody saved on the site has no parent row.

    The draft's links are written in the same transaction, so a draft's page lists its
    neighbours from the moment it is saved rather than from the next deploy. Comparing it
    against the seventy games the library holds costs about 50 milliseconds, which is why
    it is here rather than in the check's child process (`site/links.py`). Saving the same
    rules again writes no link either, because the pairs already carry this reading.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select visibility, revoked, deleted, screen_passed from games where hash = %s",
            (digest,),
        )
        found = cursor.fetchone()
    if found is not None:
        # A deleted game gets no links back. Its rules were taken away for good and a
        # link is a sentence about them, so saving the same rules again must not write
        # the sentence back -- the same reason `load` never writes a deleted row again.
        if found[2] is None:
            links.write(connection, digest, dict(spec))
        screened = bool(found[3])
        if passed and not screened and found[2] is None:
            with connection.cursor() as cursor:
                cursor.execute(
                    "update games set screen_passed = true where hash = %s", (digest,)
                )
            screened = True
        return Kept(
            digest,
            str(found[0]),
            created=False,
            revoked=found[1] is not None,
            screen_passed=screened,
        )

    forked = parent if parent and _held(connection, parent) else None
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into games"
            " (hash, spec, name, owner, visibility, parent, origin, screen_passed)"
            " values (%s, %s::jsonb, %s, %s, %s, %s, %s, %s)",
            (
                digest,
                json.dumps(dict(spec)),
                str(spec.get("name") or digest),
                owner,
                UNLISTED,
                forked,
                DESIGNED,
                passed,
            ),
        )
    links.write(connection, digest, dict(spec))
    return Kept(digest, UNLISTED, created=True, revoked=False, screen_passed=passed)


def _held(connection: Any, digest: str) -> bool:
    with connection.cursor() as cursor:
        cursor.execute("select 1 from games where hash = %s", (digest,))
        return cursor.fetchone() is not None


def why(kept: Kept, screened: Mapping[str, Any]) -> str:
    """What the save says about who can see the game, in one or two sentences."""
    if kept.revoked:
        return (
            "These rules have been taken off the site, so they are kept but not listed "
            "and cannot be played here."
        )
    if kept.listed:
        return "On trial: listed with the other games, where anybody can open it and play it."
    if kept.screen_passed:
        return (
            "Saved as a draft: only people with its link can open it. It passes the "
            "playability check, so it can be published once a score measured on your "
            "computer has been spot-checked by the site."
        )
    reasons = "; ".join(str(reason) for reason in screened.get("reasons") or ())
    said = f" ({reasons})" if reasons else ""
    return (
        f"Saved as a draft: only people with its link can open it. It did not pass the "
        f"playability check{said}, so it cannot be published until a version of it does."
    )


def listed(connection: Any) -> set[str]:
    """The hashes the store lets the listing show: public, not killed, retired, deleted or
    resting (Q503i: a trial game nobody plays keeps its page and leaves the listing).

    A draft is never retired -- the loader stamps corpus rows only -- but a corpus game
    that left the tree can still reach a card through a design run's lineage, and the
    listing should not carry it back in under `runs/`. A deleted game has no rules left
    at all, and its card would be drawn from whatever copy of the spec is lying about in
    `runs/`, which is the one thing a delete is for.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash from games where visibility = %s and revoked is null"
            " and retired is null and deleted is null and rested is null",
            (PUBLIC,),
        )
        return {str(row[0]) for row in cursor.fetchall()}


def offered(
    cards: Iterable[Mapping[str, Any]], shown: Collection[str], gone: Collection[str]
) -> list[Mapping[str, Any]]:
    """The library's cards the site may list.

    A corpus card is dropped only when it has been killed. Any other card needs a row in
    `shown`, which `listed` reads, so a killed or unlisted draft and a draft with no row
    are all left out.
    """
    kept = []
    for card in cards:
        digest = str(card["hash"])
        if str(card.get("source")) in CORPUS_SOURCES:
            if digest not in gone:
                kept.append(card)
        elif digest in shown:
            kept.append(card)
    return kept


class PublishError(ValueError):
    """A press of *Publish* refused, said the way its designer needs to hear it.

    A `ValueError`, so the server answers 400 with the sentence and the designer prints it
    beside the button.
    """


LEVELS = {
    stages.LIBRARY: "in the Library",
    stages.BASELINE: "a Baseline game",
    stages.SHORTLISTED: "shortlisted",
    stages.ON_TRIAL: "on trial",
}
"""Where a game the draft copies stands, in the refusal's words, by `stages.stage`'s level
and in the order a twin is looked for. A designed row is a twin only while it is listed."""


@dataclass(frozen=True)
class Twin:
    """A game on the site whose rules are the draft's under another spelling.

    Its hash, name and level, and nothing else: the key finds a game to name, never a
    reading to share (`twin`).
    """

    hash: str
    name: str
    level: str

    @property
    def said(self) -> str:
        return (
            f"These rules play the same as {self.name} ({self.level}); they are only "
            f"written differently, so it can't be published as a new game. Keep "
            f"tinkering, or open {self.name}."
        )


def twin(connection: Any, digest: str, spec: Mapping[str, Any]) -> Twin | None:
    """The Library, Baseline or trial game these rules copy, or `None`.

    A copy is `relate.identity_key` equal: the two specs' normal forms agree, so they differ
    in nothing but spelling. Every live row in those three levels is asked -- not killed,
    not deleted, not retired from the corpus -- except the draft's own. Library first, then
    Baseline, then trial, then by name, so a draft matching several is told the same one
    each time.

    **The key never shares a measurement.** It refuses and names; it never hands a draft
    the other game's reading, rating or card. Move order follows spelling, so two spellings
    under one seed play different games, and every number stays with the hash it was taken
    under.
    """
    mine = relate.identity_key(dict(spec))
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash, name, origin, spec, visibility, nominated, rested from games"
            " where hash <> %s and revoked is null and deleted is null and retired is null"
            " and (origin = any(%s) or (origin = %s and visibility = %s))",
            (digest, [stages.ARCHIVE, stages.CORPUS], DESIGNED, PUBLIC),
        )
        rows = [
            (found, name, theirs, stages.stage(
                {"origin": origin, "visibility": visibility, "nominated": nominated,
                 "rested": rested}
            ))
            for found, name, origin, theirs, visibility, nominated, rested
            in cursor.fetchall()
        ]
    order = list(LEVELS)
    for found, name, theirs, level in sorted(
        rows,
        key=lambda row: (
            order.index(row[3]) if row[3] in order else len(order), str(row[1]), str(row[0])
        ),
    ):
        if level in LEVELS and isinstance(theirs, dict) and relate.identity_key(theirs) == mine:
            return Twin(str(found), str(name), LEVELS[level])
    return None


def attached(
    connection: Any, digest: str, current: provisional.Current = provisional.one_run
) -> Mapping[str, Any] | None:
    """The spot-checked reading on the current basis attached to this draft, or `None`.

    Since Q503e, a read of `provisional` (`provisional.attached`): the result the
    designer's browser measured through `POST /api/provisional`, walked by tier one and
    spot-checked by tier two, whose stamp matches what the server would stamp now. `current`
    is that basis, which a test replaces with a smaller roster's.
    """
    return provisional.attached(connection, digest, current)


@dataclass(frozen=True)
class Published:
    """What a press of *Publish* did."""

    hash: str
    already: bool

    @property
    def said(self) -> str:
        if self.already:
            return "Already on trial: listed with the other games."
        return "Published. It is on trial: others can find it, play it and vote on it."


def publish(
    connection: Any,
    digest: str,
    user: int,
    reading: Mapping[str, Any] | None,
    agreed: int | None = None,
) -> Published:
    """Put this person's draft on trial, or refuse in a sentence. Nothing commits.

    The refusals come in the order a designer can act on them: whose game it is, whether
    a spot check has flagged the account (`provisional.FLAGGED`), whether it passed the check, whether it copies a game already on the site (`twin`), the two
    caps, the reading, and last the terms (Q722).

    `agreed` is the version of the terms the designer ticked the box for, `None` when the
    box was not ticked. Anything but `terms.VERSION` is refused: no box is
    `terms.NOT_ACCEPTED`, an older version is `terms.STALE`. A publish that goes through
    stamps `terms_accepted` and `terms_version` on the row, a first publish and a resting
    game's return alike, since each is a publish the designer agreed to. Games published
    before Q722 keep their nulls.

    **A resting game comes back the same way, once** (Q503i, `resting`): every gate
    above, then `rested` cleared, `woken` stamped and `published` restarted. One that has
    come back before is refused before any gate. A copy is told it is a copy before anything it could wait
    out. `reading` is what `attached`
    found; the server passes it rather than this reading it, so a test can hand one in. A
    killed or deleted game never reaches here: the server answers 404 first.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select owner, origin, visibility, screen_passed, spec, rested, woken,"
            " nominated from games where hash = %s for update",
            (digest,),
        )
        found = cursor.fetchone()
    if found is None:
        raise KeyError(f"no game {digest}")
    owner, origin, visibility, screened, spec, rested, woken, nominated = found
    if str(origin) != DESIGNED:
        raise PublishError("This game came with the site. There is nothing to publish.")
    if owner is None or int(owner) != user:
        raise PublishError("Only the person who saved this game can publish it.")
    waking = str(visibility) == PUBLIC and rested is not None and nominated is None
    if str(visibility) == PUBLIC and not waking:
        return Published(digest, already=True)
    if provisional.flagged(connection, user):
        raise PublishError(provisional.FLAGGED)
    if waking and not resting.wakes(woken):
        raise PublishError(
            "It has come back from resting once already, so it stays resting. Its page "
            "and link still work."
        )
    if not screened:
        raise PublishError(
            "It has not passed the playability check, so it cannot be published. Save a "
            "version that passes, and publish that."
        )
    copied = twin(connection, digest, spec or {})
    if copied is not None:
        raise PublishError(copied.said)
    quota.check_publish(connection, user)
    if reading is None:
        raise PublishError(NEEDS_A_READING)
    if agreed is None:
        raise PublishError(terms.NOT_ACCEPTED)
    if agreed != terms.VERSION:
        raise PublishError(terms.STALE)
    with connection.cursor() as cursor:
        if waking:
            cursor.execute(
                "update games set rested = null, woken = now(), published = now(),"
                " terms_accepted = now(), terms_version = %s where hash = %s",
                (terms.VERSION, digest),
            )
        else:
            cursor.execute(
                "update games set visibility = %s, published = now(),"
                " terms_accepted = now(), terms_version = %s where hash = %s",
                (PUBLIC, terms.VERSION, digest),
            )
    return Published(digest, already=False)


def standing(
    connection: Any,
    digest: str,
    user: int | None,
    current: provisional.Current = provisional.one_run,
) -> dict[str, Any] | None:
    """Where a designed game stands for *Publish*, as the designer draws it, or `None`.

    `None` for a game that is not a designed row. `ticks` are the two things a publish
    needs, each met or not with its sentence, and only its designer is shown them; `open`
    is whether the button works. `waiting` is why it does not, for the disabled button to
    say: `provisional.FLAGGED` for an account a spot check flagged, the same-game refusal
    when the rules copy a game on the site (`twin`, named in `twin` for the page to link),
    else `NEEDS_A_READING` while no spot-checked reading on today's basis is attached.
    `current` is that basis (`attached`).
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select owner, origin, visibility, screen_passed, revoked, deleted, spec,"
            " nominated, retired, rested, woken from games where hash = %s",
            (digest,),
        )
        found = cursor.fetchone()
    if found is None or str(found[1]) != DESIGNED:
        return None
    (owner, origin, visibility, screened, revoked, deleted, spec, nominated, retired,
     rested, woken) = found
    mine = owner is not None and user is not None and int(owner) == user
    level = stages.stage(
        {"origin": origin, "visibility": visibility, "nominated": nominated,
         "retired": retired, "rested": rested}
    )
    # A resting game its designer may still bring back (Q503i) stands where a draft does:
    # the two ticks and the button, pressed through the same `publish`.
    waking = level == stages.RESTING and resting.wakes(woken)
    on_trial = (
        level != stages.DRAFT and not waking and revoked is None and deleted is None
    )
    reading = None if on_trial else attached(connection, digest, current)
    ticks = [
        {
            "met": bool(screened),
            "says": "it passes the playability check"
            if screened
            else "it has not passed the playability check",
        },
    ]
    if level == stages.ON_TRIAL:
        says = "On trial: listed with the other games, where anybody can open and play it."
    elif level == stages.RESTING:
        says = resting.said(rested, woken, mine)
    elif on_trial:
        says = stages.SAYS.get(level, "")
    elif not mine:
        says = "A draft: only people with its link can open it, and only its designer can publish it."
    else:
        says = (
            "Saved as a draft: only people with its link can open it. Publish puts it on "
            "trial: others can find it, play it and vote."
        )
    shown = mine and not on_trial
    blocked = shown and user is not None and provisional.flagged(connection, user)
    copied = twin(connection, digest, spec or {}) if shown else None
    if blocked:
        waiting = provisional.FLAGGED
    elif copied is not None:
        waiting = copied.said
    elif shown and reading is None:
        waiting = NEEDS_A_READING
    else:
        waiting = ""
    # The reading's tick names what a publish needs, met or not; how to meet it is the
    # line under the button (`NEEDS_A_READING`), said once (Q537).
    ticks.append({"met": reading is not None, "says": READING_TICK})
    return {
        # The one word, from `stages.stage` like every other page's (Q503b).
        "level": level,
        "mine": mine,
        # Whether the page draws the *publish* button at all: its designer, on a draft.
        # Sent rather than worked out from `level`, so no page script holds a level word.
        "publishable": mine and (level == stages.DRAFT or waking),
        # What the box beside *publish* says and stands for (Q722): the one sentence, the
        # version a tick agrees to, and where the whole terms are. Only with the button.
        "terms": {"version": terms.VERSION, "says": terms.DEDICATION, "href": terms.PATH}
        if mine and (level == stages.DRAFT or waking)
        else None,
        "ticks": ticks if shown else [],
        "open": shown
        and not blocked
        and copied is None
        and all(bool(tick["met"]) for tick in ticks),
        "says": says,
        "waiting": waiting,
        "twin": {"hash": copied.hash, "name": copied.name} if copied is not None else None,
    }


@dataclass(frozen=True)
class Returned:
    """A game a save had listed, which `0016_drafts.sql` put back to Draft."""

    hash: str
    name: str
    when: datetime


def returned(connection: Any, owner: int) -> list[Returned]:
    """This person's games that were listed until saves stopped listing, for the profile.

    Told apart by when, not by a stamp, because the migration stamps nothing: a designed
    row that passed the check, was saved before `MIGRATION` ran, is unlisted and was never
    published is exactly a row the migration moved (a save before it listed every pass).
    A draft saved since is never here. Empty where the ledger does not hold the migration.
    """
    with connection.cursor() as cursor:
        cursor.execute("select to_regclass('schema_migrations')")
        ledger = cursor.fetchone()
        if ledger is None or ledger[0] is None:
            return []
        cursor.execute("select applied from schema_migrations where name = %s", (MIGRATION,))
        ran = cursor.fetchone()
        if ran is None:
            return []
        cursor.execute(
            "select hash, name from games where owner = %s and origin = %s"
            " and visibility = %s and screen_passed and published is null"
            " and revoked is null and deleted is null and created < %s"
            " order by created, hash",
            (owner, DESIGNED, UNLISTED, ran[0]),
        )
        return [Returned(str(row[0]), str(row[1]), ran[0]) for row in cursor.fetchall()]
