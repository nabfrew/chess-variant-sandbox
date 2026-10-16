"""Everything the repository already knows, as rows the site can query.

Task 1.6 of `docs/plan-website.md`, against §3's *What "migrate key data" means*. The
files stay the truth. `canonical.json` is the one file a score and a ladder come from, it
is never hand-written, and the CLI reads `runs/` with no database in reach -- so the
database cannot become where a measurement lives. It holds what the site queries, and an
index over the files that this rebuilds. *Migrate* means load, idempotently, by hash, and
never move.

Four sources, four tables and one seeding:

    rulesets/ and archive/          -> games          61 specs, keyed by mechanics hash
    runs/<hash>/canonical.json      -> measurements   the summary, its basis, its state
    runs/<hash>/index-<w>.json      -> measurements   both ends of the interval
    runs/opponents/<hash>.json      -> measurements   the whole ladder, and its top rung
                                    -> ratings        seated through `rating.reseed`
    runs/human/games.jsonl          -> matches        the maintainer's own games, history
    the games just written          -> links          how near each pair is (`site/links.py`)

The links are the one table whose source is the store rather than the tree, because a
draft saved on the site is a game the map and a game page can be near to and has no file
in `rulesets/`. G7 of `docs/archive/plan-game-graph.md` is the row; the reading is
`spec/relate.py`'s and this only stores it.

## Idempotent by reading first

Every write here reads what is there and compares before it writes, so a second run
reports zero added and zero updated and leaves every `updated` timestamp where it was.
That is row 1.6's own acceptance, and it is a property of the comparison rather than of
`on conflict do nothing` -- the loader has to be able to *correct* a row when the file
behind it moved, and a blind upsert would rewrite all 61 every deploy.

## What it will not do

**It never deletes.** `runs/` is gitignored, so a checkout's copy is not the box's copy,
and a load run from a partial one must not erase the box's index. A row whose file has
gone stays where it is.

**A game whose spec has left the tree is retired, not removed.** `rulesets/` and
`archive/` are tracked, so the deployed tree is the whole account of what the library
holds, and a spec superseded there should stop being offered. Measurements, ratings, votes
and match rows all reference the row, and a match already played is a fact, so `retire`
stamps `games.retired` instead: off the listing, refused for a new match, and still
readable by hash with its past games intact. A spec that comes back under the same
mechanics hash is un-retired by `load_games`, which treats `retired = null` as part of a
present game's desired row.

**A deleted game is never written again.** A retired game's creator or an admin can
delete its rules outright (`admin.delete_game`, migration `0010`), and that has no way
back. So a spec whose row is deleted is dropped from this load before anything is
written, and reported: the tree still holds the file, and a load that quietly resurrected
the rules would undo a person's deletion on the next deploy.

**It never invents a basis.** A measurement with no `basis` block is loaded as state
`replay`, a game with no `canonical.json` as `missing`, and neither carries a headline.
The headline is written only when the stored basis matches what a measurement taken now
would rest on, which is `corpus.read_measurement`'s judgement and not a second one --
`.claude/rules/viz-and-web.md`'s *a stale measurement shows neither* is the same rule one
layer down, made structural here by there being no number in the column to show. The two
ends of the interval follow the headline exactly: written when the measurement reads
current, null in every other state. The column is what ranks, badges and nominates, so it
holds current numbers only; the provisional number a reader sees while a score is measured
again (EV2, `docs/plan-evolving-score.md` §5) comes off the files through
`library.Game.score`, marked, and never from here.

**It never rates the maintainer's games.** The rows from the local application land in
leaderboard mode so the anchor can read them (`matches.human_games`), and carry
`source = 'local-app'` and `top = false`, so `rating.finished` drops them from the fold
and no reading of the *Unbeaten* tag can mistake one for a clear. The user's instruction
of 2026-09-06 is that their own results are not collected as data; §5 keeps them as
history on the profile, and those two columns are how task 1.5 tells them apart.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chess_variant_sandbox import corpus
from chess_variant_sandbox.measurement import Measurement
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.site import badges, links, rating, resting, rounds
from chess_variant_sandbox.site.matches import LEADERBOARD, LOCAL_APP
from chess_variant_sandbox.spec.canonical import load_spec, spec_hash
from chess_variant_sandbox.web.human import HumanGame, HumanRecord
from chess_variant_sandbox.web.library import preview_sentence

# ------------------------------------------------------------------- what a row says

MAINTAINER = "maintainer"
"""The handle the corpus is owned by and the local application's games are played by.

§3 asks for *the maintainer* and names nobody, so this is the documented default and
`--maintainer` overrides it. One row, created on the first load and found by handle after
that.
"""

MAINTAINER_EMAIL = "maintainer@example.invalid"
"""A placeholder in a reserved domain, never a real address.

`users.email` is not null because the sign-in link is sent to it, and the loader has no
business writing somebody's address into a table. Task 1.3 is where a real one lands.
"""

MAINTAINER_ROLE = "admin"

CORPUS = "corpus"
ARCHIVED = "archive"
DESIGNED = "designed"
PREVIEW = "preview"
"""A file in `archive/` that says it is a preview (`provenance.preview`, Q931): a new kind of
game still being tested, listed in its own tab and played as practice. Its row is written
and corrected like an archived game's, and nothing that scores reads it: no measurement row,
no seeded ratings, no badges (`load`), no links (`links.specs_of`)."""
LOADED_ORIGINS = (CORPUS, ARCHIVED, PREVIEW)
"""The origins this module owns. A row carrying either may be corrected from the
files; a row carrying anything else belongs to whoever uploaded it, and §3's *the same
rules uploaded twice are one row, first uploader credited* is that row staying put. A draft
saved on the site is written at save by `site/drafts.py`, so this loader never reads
`runs/designed/`."""

PUBLIC = "public"
"""What a corpus game's visibility is. The library is what everybody came for; a spec
committed to the repository is already public in the plainest sense."""

CURRENT = "current"
MISSING = "missing"
"""The two states `Staleness.needs` has no word for. `current` is its empty string, and
`missing` is a game with a ladder and no measurement at all."""


# --------------------------------------------------------------------- what a run did


@dataclass
class Change:
    """What one table's load did.

    `written` is zero on a second run over unmoved files, which is row 1.6's acceptance
    and what the tests assert.
    """

    added: int = 0
    updated: int = 0
    unchanged: int = 0

    @property
    def written(self) -> int:
        return self.added + self.updated

    @property
    def rows(self) -> int:
        return self.added + self.updated + self.unchanged

    def __str__(self) -> str:
        return f"{self.rows} row(s): {self.added} added, {self.updated} updated"


@dataclass
class Report:
    """What a whole load did, table by table, with what it could not place.

    `skipped` is sentences rather than counts, because every one of them is a file in the
    repository that a person can go and look at.
    """

    games: Change = field(default_factory=Change)
    retired: Change = field(default_factory=Change)
    moved: Change = field(default_factory=Change)
    rested: Change = field(default_factory=Change)
    measurements: Change = field(default_factory=Change)
    ratings: Change = field(default_factory=Change)
    matches: Change = field(default_factory=Change)
    links: links.Written = field(default_factory=links.Written)
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def written(self) -> int:
        return (
            self.games.written
            + self.retired.written
            + self.moved.written
            + self.rested.written
            + self.measurements.written
            + self.ratings.written
            + self.matches.written
            + self.links.written
        )

    def summary(self) -> str:
        """Several lines, and the first says outright when nothing changed."""
        head = "nothing changed" if not self.written else f"{self.written} row(s) written"
        lines = [
            head,
            f"  games        {self.games}",
            f"  retired      {self.retired}",
            f"  moved        {self.moved}",
            f"  rested       {self.rested}",
            f"  measurements {self.measurements}",
            f"  ratings      {self.ratings}",
            f"  matches      {self.matches}",
            f"  links        {self.links}",
        ]
        lines += [f"  {note}" for note in self.notes]
        lines += [f"  skipped: {reason}" for reason in self.skipped]
        return "\n".join(lines)


# --------------------------------------------------------------------- the four sources


def read_specs(directory: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Every spec in one directory, by mechanics hash, and what would not parse.

    A directory that is not there is empty rather than an error, because `archive/` is
    optional in a checkout somebody trimmed and a deploy that stopped on it would ship
    nothing. Filenames are human names, not content hashes, so two files can collide on
    mechanics hash; the first in sorted order wins and the second is reported.
    """
    found: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    if not Path(directory).is_dir():
        return found, problems
    for path in sorted(Path(directory).glob("*.json")):
        try:
            spec = load_spec(path)
        except (OSError, ValueError) as error:
            problems.append(f"{path.name} did not parse as a spec ({error})")
            continue
        if not isinstance(spec, dict):
            problems.append(f"{path.name} is not a spec object")
            continue
        digest = spec_hash(spec)
        if digest in found:
            problems.append(f"{path.name} repeats the hash {digest}")
            continue
        found[digest] = spec
    return found, problems


def read_ladder(runs_dir: Path, digest: str) -> dict[str, Any] | None:
    """The picker ladder for one game, as it stands on disk.

    The whole file, because §3 stores every rung: the site's table is seeded from it
    (`rating.reseed`) and the page names the rungs. A file that will not parse is no ladder, which leaves
    the game unrated and offering sandbox only -- the same answer as never having run the
    picker on it.
    """
    path = Path(runs_dir) / "opponents" / f"{digest}.json"
    if not path.exists():
        return None
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def top_rung(ladder: dict[str, Any] | None) -> tuple[str | None, float | None]:
    """The ladder's own top rung and its own rating, on the ladder's own scale.

    Not the seed's number on the site's scale, and the difference matters twice. The
    column sits beside the file this row is an index over, so it should say what the file
    says. And the seeded value depends on the shift `rating.seeds_from_ladder` reads off
    the live table, which human play moves -- a column computed from that would change
    under a second load and this loader would stop being idempotent.

    The live top is a different question with a different answer, and §5 reads it off
    `ratings` through `rating.top`.
    """
    if ladder is None:
        return None, None
    standings = rating.rungs(ladder)
    if not standings:
        return None, None
    best = max(standings, key=lambda rung: rung.rating)
    return best.name, best.rating


# -------------------------------------------------------------------------- the people


def maintainer_id(
    connection: Any, handle: str = MAINTAINER, email: str = MAINTAINER_EMAIL
) -> tuple[int, bool]:
    """The maintainer's user id, creating the row the first time. Returns whether it did.

    `password` is left null, so this account cannot sign in until task 1.3 gives it one.
    That is the right state for a row the loader invented, because it exists to own the
    corpus and to sit in the seat of games already played, and neither is a login.
    """
    with connection.cursor() as cursor:
        cursor.execute("select id from users where handle = %s", (handle,))
        found = cursor.fetchone()
        if found is not None:
            return int(found[0]), False
        cursor.execute(
            "insert into users (handle, email, role) values (%s, %s, %s) returning id",
            (handle, email, MAINTAINER_ROLE),
        )
        created = cursor.fetchone()
    return int(created[0]), True


# --------------------------------------------------------------------------- the games


_GAME_FIELDS = "spec, name, owner, visibility, origin"

_GAME_STATE = f"{_GAME_FIELDS}, retired"


def load_games(
    connection: Any,
    specs: dict[str, tuple[dict[str, Any], str]],
    owner: int,
    moved: Change | None = None,
    notes: list[str] | None = None,
) -> Change:
    """Put the library in `games`, one row per mechanics hash.

    `specs` maps a hash to its spec and the origin it was found under, so a game
    promoted out of `archive/` into `rulesets/` since the last load has its origin
    corrected rather than duplicated -- the hash did not move, because the filename is
    the hash and a promotion copies the mechanics unchanged.

    **A move between the Library and the Baseline is remembered** (Q503n). When the
    corrected origin is the other of this module's two, a `level_moves` row records it
    with the maintainer's reason (`reason_for`), counted in `moved` and named in `notes`,
    for the game page's *Where it stands* and for *What's new* (`stages.said`,
    `stages.whats_new`). This loader is the only writer of an origin on the site; no route
    changes one (`tests/test_site_resting.py`).

    A row this module did not write is left exactly as it is. A user who uploaded the
    same rules first keeps the credit, which is §3's rule, and the loader has no way to
    tell a deliberate rename on the site from a stale one in the repository.

    A present spec's desired row carries `retired = null`, so a game that comes back
    under the same hash is put back on the listing by the same comparison that corrects
    its name. `revoked` is untouched here: an admin took that game down deliberately, and
    a deploy is not an argument against it.
    """
    change = Change()
    # Q503h. The games a round confirmed, with the maintainer's reason. Once one's spec is
    # in the tree its `designed` row is taken over here, and its designer stays its owner
    # on this load and every later one.
    confirmed = rounds.confirmed(connection)
    for digest, (spec, origin) in sorted(specs.items()):
        name = str(spec.get("name") or digest)
        with connection.cursor() as cursor:
            cursor.execute(f"select {_GAME_STATE} from games where hash = %s", (digest,))
            found = cursor.fetchone()
        kept = digest in confirmed and found is not None and found[2] is not None
        desired = (spec, name, found[2] if kept else owner, PUBLIC, origin, None)
        if found is None:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"insert into games (hash, {_GAME_FIELDS})"
                    f" values (%s, %s::jsonb, %s, %s, %s, %s)",
                    (digest, json.dumps(spec), name, owner, PUBLIC, origin),
                )
            change.added += 1
            continue
        adopted = str(found[4]) == DESIGNED and digest in confirmed
        if (str(found[4]) not in LOADED_ORIGINS and not adopted) or tuple(found) == desired:
            change.unchanged += 1
            continue
        with connection.cursor() as cursor:
            cursor.execute(
                "update games set spec = %s::jsonb, name = %s, owner = %s,"
                " visibility = %s, origin = %s, retired = null where hash = %s",
                (json.dumps(spec), name, desired[2], PUBLIC, origin, digest),
            )
        change.updated += 1
        was = str(found[4])
        if was != origin and PREVIEW in (was, origin):
            # Q931. A preview joining the library, or a game shown as a preview again, is
            # said in the load's own output; the Library and the Baseline's moves table
            # (`stages.whats_new`) is about those two levels and no other.
            if notes is not None:
                where = "joined the Library from preview" if origin == ARCHIVED else (
                    "became a preview")
                notes.append(f"{name!r} ({digest}) {where}")
        elif was != origin:
            reason = confirmed[digest] if adopted else reason_for(found[0], spec)
            with connection.cursor() as cursor:
                cursor.execute(
                    "insert into level_moves (game, was, became, reason)"
                    " values (%s, %s, %s, %s)",
                    (digest, was, origin, reason),
                )
            if moved is not None:
                moved.added += 1
            if notes is not None:
                where = "joined the Baseline" if origin == CORPUS else "left the Baseline"
                if adopted:
                    where = "joined the " + (
                        "Baseline" if origin == CORPUS else "Library"
                    ) + " from its round"
                notes.append(
                    f"{name!r} ({digest}) {where}"
                    + (f": {reason}" if reason else ", with no reason in its provenance")
                )
    return change


_SENTENCE = re.compile(r"^(.+?[.!?])(?:\s|$)", re.DOTALL)


def reason_for(before: Any, after: dict[str, Any]) -> str:
    """The maintainer's reason for a move, from the spec's `provenance.history`.

    `chess-variant-sandbox promote --reason` writes its reason as the history's first
    sentence (`corpus._provenance`), and `docs/reference/baseline-procedure.md` asks the
    maintainer to do the same by hand for a move the other way. So the reason is that
    first sentence, **when the move changed the history**; a file moved without a word
    keeps the history it had, and then there is no reason to show rather than a stale one.
    """

    def history(spec: Any) -> str:
        if not isinstance(spec, dict):
            return ""
        provenance = spec.get("provenance")
        if not isinstance(provenance, dict):
            return ""
        return str(provenance.get("history") or "").strip()

    now = history(after)
    if not now or now == history(before):
        return ""
    first = _SENTENCE.match(now)
    return (first.group(1) if first else now).strip()


def shortlisted_specs(
    connection: Any, known: set[str]
) -> dict[str, tuple[dict[str, Any], str]]:
    """Every shortlisted player-made game's rules, from its row, for its measurement.

    Given the origin `archive`, so `load_measurements` owes it four runs (Q503h). A game
    whose spec is in the tree already is `known`, and read from there.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash, spec from games where origin = %s and nominated is not null"
            " and spec is not null and deleted is null",
            (DESIGNED,),
        )
        rows = cursor.fetchall()
    return {
        str(digest): (spec, ARCHIVED)
        for digest, spec in rows
        if str(digest) not in known and isinstance(spec, dict)
    }


def _deleted(connection: Any) -> set[str]:
    """The hashes whose rules a creator or an admin removed for good (migration `0010`).

    Read here rather than through `site.admin` so the loader keeps its one import of the
    store's shape, which is its own SQL.
    """
    with connection.cursor() as cursor:
        cursor.execute("select hash from games where deleted is not null")
        return {str(row[0]) for row in cursor.fetchall()}


def retire(
    connection: Any, specs: dict[str, tuple[dict[str, Any], str]]
) -> tuple[Change, list[str]]:
    """Stamp `retired` on every library row whose spec is in no directory read.

    Returns the count and a sentence per game newly retired, because a deploy that takes
    a game off the listing should say which one in its own output.

    A deleted row is left out: its rules are gone, which is further off the listing than
    retirement reaches, and stamping it would only move its first stamp about.

    Three things are never stamped. A row whose origin this module does not own is
    somebody's own game -- a draft saved on the site has no spec in the tree by
    definition, so retiring by absence would take down every draft. A row already retired
    keeps its first stamp, so the column answers *when it left* rather than *when the
    last deploy ran*. And nothing at all is stamped when `specs` is empty: a checkout
    with no `rulesets/` is a trimmed or half-shipped tree, not a corpus that has been
    emptied, and the loader never lets one erase the box's library.
    """
    change = Change()
    if not specs:
        return change, []
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash, name, retired from games"
            " where origin = any(%s) and hash <> all(%s) and deleted is null",
            (list(LOADED_ORIGINS), list(specs)),
        )
        absent = cursor.fetchall()

    notes: list[str] = []
    for digest, name, stamped in sorted(absent, key=lambda row: str(row[0])):
        if stamped is not None:
            change.unchanged += 1
            continue
        with connection.cursor() as cursor:
            cursor.execute(
                "update games set retired = now() where hash = %s", (digest,)
            )
        change.updated += 1
        notes.append(
            f"retired {name!r} ({digest}): its spec is in neither rulesets/ nor archive/,"
            f" so it is off the listing and refused for a new match; its matches stay"
        )
    return change, notes


# -------------------------------------------------------------------- the measurements


_MEASUREMENT_FIELDS = (
    "directory, state, basis, headline, ladder, top, top_rating, ci_low, ci_high"
)


def load_measurements(
    connection: Any,
    specs: dict[str, tuple[dict[str, Any], str]],
    runs_dir: Path,
    weights: str = DEFAULT_WEIGHTS,
) -> Change:
    """Index `runs/` for every game in the library: the summary, its state, the ladder.

    A row exists for a game that has either file. The two halves are independent on
    purpose. A canonical measurement goes stale when the code or the weights move, and
    `corpus.read_measurement` is the one place that decides so; a picker ladder has no
    basis at all and is read as a seed, so a core rebuild leaves it alone exactly as §5
    says it leaves the rating table alone.

    Staleness is judged against *this* checkout's code, whatever `--root` says about
    where the files are, because the question a state answers is what a measurement taken
    now, here, would rest on.
    """
    change = Change()
    for digest, (spec, origin) in sorted(specs.items()):
        # How many runs this rung's score is expected to rest on. Both origins this
        # module owns are library games, so both read `corpus.expected_seeds`, which is
        # the one place that knob lives -- naming the seeds here would put a second
        # answer beside it.
        expected = corpus.expected_seeds(
            "archived" if origin == ARCHIVED else "committed"
        )
        measurement = corpus.read_measurement(digest, runs_dir, weights, spec, expected)
        ladder = read_ladder(runs_dir, digest)
        if measurement is None and ladder is None:
            continue

        # `<hash>`, the measurement's own directory. The row joins on it to open
        # `index-<weights>.json` beside the file, which in a directory of its own is the
        # measurement's rather than one run's.
        directory = corpus.measurement_dir(digest, runs_dir).name
        if measurement is None:
            state, basis, headline = MISSING, None, None
            ci_low, ci_high = None, None
        else:
            state = measurement.staleness.needs or CURRENT
            basis = measurement.basis.as_dict() if measurement.basis else None
            headline = _headline(measurement) if measurement.current else None
            ci_low, ci_high = _range(measurement) if measurement.current else (None, None)
        top, top_rating = top_rung(ladder)
        desired = (directory, state, basis, headline, ladder, top, top_rating,
                   ci_low, ci_high)

        with connection.cursor() as cursor:
            cursor.execute(
                f"select {_MEASUREMENT_FIELDS} from measurements where hash = %s", (digest,)
            )
            found = cursor.fetchone()
        if found is None:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"insert into measurements (hash, {_MEASUREMENT_FIELDS})"
                    f" values (%s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s, %s, %s)",
                    (digest, directory, state, _dump(basis), headline, _dump(ladder),
                     top, top_rating, ci_low, ci_high),
                )
            change.added += 1
            continue
        if tuple(found) == desired:
            change.unchanged += 1
            continue
        with connection.cursor() as cursor:
            cursor.execute(
                "update measurements set directory = %s, state = %s, basis = %s::jsonb,"
                " headline = %s, ladder = %s::jsonb, top = %s, top_rating = %s,"
                " ci_low = %s, ci_high = %s, updated = now() where hash = %s",
                (directory, state, _dump(basis), headline, _dump(ladder), top,
                 top_rating, ci_low, ci_high, digest),
            )
        change.updated += 1
    return change


def _headline(measurement: Measurement) -> float | None:
    """The stored index, as a number or not at all."""
    found = measurement.stored.get("index")
    return None if found is None else float(found)


def _range(measurement: Measurement) -> tuple[float | None, float | None]:
    """Both ends of the resampled interval, from the index file beside `canonical.json`.

    The summary keeps the headline and its `sd`; the interval itself is written to
    `index-<weights>.json` in the same directory, which is the file `web/library` expands
    into the score view. The site needs the two ends in the row because U1 tests a game's
    whole range inside a vote's transaction, and a transaction should not be opening
    files.

    Absent is not an error and never a zero. A measurement taken on a corpus too small to
    resample has no interval, and a run directory can have lost the file while keeping
    the summary; both answer *no range*, which is what a badge that needs one refuses on.
    """
    version = str(measurement.stored.get("weights_version") or "")
    path = measurement.directory / f"index-{version}.json"
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    interval = found.get("interval") if isinstance(found, dict) else None
    if not isinstance(interval, dict):
        return None, None
    return _number(interval.get("ci_low")), _number(interval.get("ci_high"))


def _number(found: Any) -> float | None:
    try:
        return None if found is None else float(found)
    except (TypeError, ValueError):
        return None


def _dump(payload: dict[str, Any] | None) -> str | None:
    return None if payload is None else json.dumps(payload)


# -------------------------------------------------------------------------- the ladders


def seed_ratings(connection: Any, digests: list[str], runs_dir: Path) -> Change:
    """Seat each game's computer players from its picker ladder (`rating.reseed`).

    The reseed rule of `docs/plan-leaderboard.md` §2.3 (L3). A player the table has not
    seen is *added*. A computer row no person has played is still only its seed, so a
    newer ladder's seat replaces it: *updated*. A row people have played keeps the number
    they gave it, and a row already at its seat is left as it is: *unchanged*. So a second
    load over the same files writes nothing, a re-rated ladder refreshes every computer
    number nobody has played against, and what people decided stays.

    The first seeding of a table centres on the ladder's five standard players
    (`rating.anchor`, §2.4), so the first load after L3 also moves the unplayed rows of a
    table seeded under the old rule (1500 at the whole roster's mean) onto the new one.
    """
    change = Change()
    for digest in sorted(digests):
        ladder = read_ladder(runs_dir, digest)
        if ladder is None:
            continue
        seated = rating.reseed(connection, digest, ladder)
        change.added += len(seated.added)
        change.updated += len(seated.reseeded)
        change.unchanged += seated.kept
    return change


# ------------------------------------------------------------------- the maintainer's games


_MATCH_FIELDS = (
    'game, mode, seat0_user, seat0_agent, seat1_user, seat1_agent, seed, result,'
    ' reason, "left", started, finished, source, top'
)


def load_matches(
    connection: Any, runs_dir: Path, owner: int, known: set[str]
) -> tuple[Change, list[str]]:
    """The local application's games, as history on the maintainer's profile.

    §3: loaded with `source = local-app` and the maintainer in the seat, rated by
    nothing, counted by no leaderboard and clearing no *Unbeaten* tag, because they were
    played with undo on hand and predate the modes. Leaderboard mode, because that is
    what `matches.human_games` reads and the anchor is what these rows were kept for; the
    two columns that keep them out of everything else are `source` and `top`.

    One thing is lost rather than faked. The JSONL never held a move list, only a ply
    count, so `moves` stays empty and a loaded row reads as nought plies. A reading of
    mean length must exclude `source = 'local-app'` rows, which it must anyway -- they
    are not site play.
    """
    change = Change()
    skipped: list[str] = []
    missing: set[str] = set()
    for game in HumanRecord(runs_dir).games():
        if game.ruleset_hash not in known:
            missing.add(game.ruleset_hash)
            continue
        _put_match(connection, game, owner, change)
    for digest in sorted(missing):
        skipped.append(f"human games on {digest}, which is in neither rulesets/ nor archive/")
    return change, skipped


def _put_match(connection: Any, game: HumanGame, owner: int, change: Change) -> None:
    """One `HumanGame` as a row, written only when it is not already there unchanged.

    A row that is there and differs is corrected only when it is one of ours, because the
    JSONL is append-only and a game finished after the last load is exactly the row that
    has to move, while a match played on the site is never this file's business.
    """
    seat = int(game.human_seat)
    # `matches.result` is seat 0's score and `HumanGame.result` is the person's, so the
    # two agree when the person sat first and are complements when they did not.
    result = None if game.result is None else (
        float(game.result) if seat == 0 else 1.0 - float(game.result)
    )
    if seat == 0:
        players: tuple[Any, Any, Any, Any] = (owner, None, None, game.opponent)
    else:
        players = (None, game.opponent, owner, None)
    desired = (
        game.ruleset_hash,
        LEADERBOARD,
        *players,
        int(game.seed),
        result,
        str(game.reason or ""),
        False,
        _moment(game.started),
        _moment(game.finished),
        LOCAL_APP,
        False,
    )

    with connection.cursor() as cursor:
        cursor.execute(f"select {_MATCH_FIELDS} from matches where id = %s", (game.session,))
        found = cursor.fetchone()
    if found is None:
        with connection.cursor() as cursor:
            cursor.execute(
                f"insert into matches (id, {_MATCH_FIELDS}) values"
                f" (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (game.session, *desired),
            )
        change.added += 1
        return
    if found[12] != LOCAL_APP or tuple(found) == desired:
        change.unchanged += 1
        return
    with connection.cursor() as cursor:
        cursor.execute(
            'update matches set game = %s, mode = %s, seat0_user = %s, seat0_agent = %s,'
            ' seat1_user = %s, seat1_agent = %s, seed = %s, result = %s, reason = %s,'
            ' "left" = %s, started = %s, finished = %s, source = %s, top = %s'
            " where id = %s",
            (*desired, game.session),
        )
    change.updated += 1


def _moment(stamp: float | None) -> datetime | None:
    """An epoch second as the database keeps it. UTC, because a row has one instant."""
    return None if stamp is None else datetime.fromtimestamp(float(stamp), tz=timezone.utc)


# ------------------------------------------------------------------------- the whole load


def load(
    connection: Any,
    root: Path = Path("."),
    handle: str = MAINTAINER,
    weights: str = DEFAULT_WEIGHTS,
) -> Report:
    """Read the checkout at `root` into the store. Nothing commits; the caller decides.

    The order is the foreign keys' order. A measurement references a game and so does a
    match, so the library goes first, and the ratings are seeded before the matches so
    that a reading of the maintainer's rows has a table to read them against.

    Retirement comes straight after the library, on the same set of specs the library was
    written from, so one load answers *what is here* and *what has left* from one reading
    of the tree.

    The links come after the matches and read the store rather than `specs`, so a draft
    somebody saved before this table existed gets its pairs on the next deploy. They are
    the run's one expensive step -- 2,415 comparisons and about 1.4 seconds on a library
    of seventy -- and a second load compares none of them.

    The badges that sit on a game go last, because a load is one of the three events that
    can complete U1: a game with its people and its ratings already in place earns it on
    the load that finds its measurement current. Games nobody has rated are dropped from
    that pass before anything else is read, so it costs a query or two over the library.
    """
    report = Report()
    ruleset_dir = Path(root) / "rulesets"
    archive_dir = Path(root) / "archive"
    runs_dir = Path(root) / "runs"

    specs: dict[str, tuple[dict[str, Any], str]] = {}
    for directory, origin in ((archive_dir, ARCHIVED), (ruleset_dir, CORPUS)):
        found, problems = read_specs(directory)
        report.skipped.extend(problems)
        # `rulesets/` second, so a game in both directories reads as committed. That is
        # `corpus.find`'s own ladder order and the same reason: a promoted game is
        # committed, whatever copy of it is still lying about.
        for digest, spec in found.items():
            # Q931: an archived file that says it is a preview is loaded as one.
            specs[digest] = (
                spec,
                PREVIEW if origin == ARCHIVED and preview_sentence(spec) is not None
                else origin,
            )

    # A deleted game is not loaded again, whatever the tree says. The rules were removed
    # on purpose and the delete has no way back, so a spec that reappears under the same
    # mechanics hash is left on disk and named here rather than written into the row.
    # Dropped before the library is written, so every pass below sees one set of specs.
    for digest in sorted(_deleted(connection) & set(specs)):
        specs.pop(digest)
        report.notes.append(
            f"skipped {digest}: that game was deleted, and a deleted game is never "
            f"written again; remove its spec from the tree if it should not be there"
        )

    owner, created = maintainer_id(connection, handle)
    report.games = load_games(connection, specs, owner, report.moved, report.notes)
    report.retired, retirements = retire(connection, specs)
    report.notes.extend(retirements)
    # Q931. A preview is in no scored table: its measurement stays on disk, where its own
    # page reads it with its label, and never becomes a headline the store ranks, badges or
    # nominates on; its computer players are seated from no ladder, so it is practice only.
    scored = {digest: entry for digest, entry in specs.items() if entry[1] != PREVIEW}
    report.measurements = load_measurements(connection, scored, runs_dir, weights)
    # Q503h. A shortlisted game enters a round once its four runs are current, and the
    # maintainer measures it by hand (`measure`) until Q503g's pipeline uploads them. Its
    # rules are the store's, so its runs are indexed from the row, owed four seeds as a
    # Library game's are.
    waiting = shortlisted_specs(connection, set(specs))
    if waiting:
        more = load_measurements(connection, waiting, runs_dir, weights)
        report.measurements.added += more.added
        report.measurements.updated += more.updated
        report.measurements.unchanged += more.unchanged
    report.ratings = seed_ratings(connection, list(scored), runs_dir)
    report.matches, missing = load_matches(connection, runs_dir, owner, set(specs))
    report.skipped.extend(missing)
    report.links = links.fill(connection, links.specs_of(connection))
    granted = badges.grant_for_games(connection, list(scored))
    if granted:
        report.notes.append(f"granted {len(granted)} badge(s) on the library's games")
    # Q503i. The trial games nobody plays leave the listing. Here because every deploy
    # runs this load, and it is one `update` that stamps nothing on a second run.
    for rest in resting.rest(connection):
        report.rested.updated += 1
        report.notes.append(rest.said)
    # Q503h. Close a round whose week is over and open this month's, as `site sweep` does.
    report.notes.extend(rounds.sweep(connection).lines)
    if created:
        report.notes.append(f"created the maintainer {handle!r}, who cannot sign in yet")
    return report
