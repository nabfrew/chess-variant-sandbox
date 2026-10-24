"""Finding the games, and everything known about them.

A ruleset on disk is a spec file. What a person wants to see is a *game*: what it is
called, what it is about, how it plays, how it scored and where it came from. Those live
in four different places -- the spec, `spec/describe.py`, whatever `runs/` holds for that
hash, and a design run's lineage -- and this module is the one place that joins them.

Nothing here computes anything. Scores are read from the index files a scoring run wrote;
if a ruleset has never been scored it simply has no score, which is reported as absent
rather than as zero. Verdicts are read the same way, from whatever ``chess-variant-sandbox verdict``
left behind -- and they are prose, never a number, because the one place a second
opinion is allowed to act is the ordering the interval leaves undefined.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.loop.archive import Archive, Entry, provisional_sentence
from chess_variant_sandbox.loop.grid import (
    CELLS,
    axis_vocabulary,
    bins_of,
    describe_axes,
    describe_cell,
    descriptor,
    label,
)
from chess_variant_sandbox.measurement import ROOT as MEASUREMENT_ROOT
from chess_variant_sandbox.measurement import Measurement, built_fingerprint, tables_cache_path
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.metrics.glossary import explain
from chess_variant_sandbox.metrics.index import IndexResult
from chess_variant_sandbox.metrics.scoring import WeightsError
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.spec.canonical import HASH_LENGTH
from chess_variant_sandbox.spec.complexity import measure_complexity
from chess_variant_sandbox.spec.describe import describe_rules, describe_setup, describe_spec
from chess_variant_sandbox.spec.schema import is_spy_game
from chess_variant_sandbox.viz.pieces import icons_for, side_names
from chess_variant_sandbox.web import card_rules, sitescore

RULESET_DIR = Path("rulesets")
RUNS_DIR = Path("runs")
ARCHIVE_DIR = Path("archive")

# An estimate's three states (`docs/archive/plan-evolving-score.md` §2.1), as `Game.status` and
# every payload spell them.
CURRENT = "current"
PROVISIONAL = "provisional"
NONE = "none"

# A file in `archive/` that says it is a preview (`provenance.preview`, Q931): its own
# source, beside *committed* and *archived*, so that no list of the library's sources holds
# it. See `preview`.
PREVIEW = "preview"


@dataclass
class Game:
    """One ruleset, with everything the interface knows about it."""

    hash: str
    name: str
    description: str
    path: Path
    source: str  # committed | archived | preview | designed | evolved | retired
    spec: dict[str, Any]
    # One measurement, or none. Score and ladder are *views* onto it rather than two
    # fields that happen to sit beside each other, because as two fields they were
    # filled by two independent scans of `runs/` and could -- and did -- come from
    # different runs: a fresh Elo table rendered next to a year-old gate, as one row.
    measurement: Measurement | None = None
    verdict: dict[str, Any] | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    # What the design run that invented this game scored it, on the run's own smaller
    # roster. Kept strictly apart from `measurement` and never merged into it: a loop
    # scores on four agents at three games a pairing so a generation fits in the time it
    # has, and putting that number in the same column as `chess` is the one comparison
    # this project is built to refuse. Shown, because for an evolved candidate it is the
    # only number there is -- and labelled, because it is not the corpus's number.
    design_score: dict[str, Any] | None = None

    @property
    def stale(self) -> bool:
        return self.measurement is not None and not self.measurement.current

    @property
    def status(self) -> str:
        """Where this game's estimate stands: ``current``, ``provisional`` or ``none``.

        *Current* is worked out under the scorer this checkout runs, *provisional* is the
        latest estimate worked out under an earlier revision of it, and *none* is no
        estimate that can be shown: never measured, or a stale file that cannot state
        its basis or names another ruleset (`docs/archive/plan-evolving-score.md` §2.1). The basis stamp decides
        which (`Measurement.current`); a provisional number is shown, labelled, and never
        as current.
        """
        measurement = self.measurement
        if measurement is None:
            return NONE
        if measurement.current:
            return CURRENT
        # Provisional only where the stale file is this game's own past estimate and can
        # say what it rests on. A file with no basis cannot state its basis, and one whose
        # basis names another ruleset is somebody else's number: neither is shown.
        if measurement.basis is None or measurement.basis.ruleset_hash != self.hash:
            return NONE
        return PROVISIONAL

    @property
    def score(self) -> dict[str, Any] | None:
        """The game's latest estimate, current or provisional, or ``None`` with none.

        The policy in one place (`CLAUDE.md`, *A score carries its basis*). A stale
        measurement is no longer withheld: its stored number comes back, carrying
        ``status`` (`provisional`), the ``revision`` it was worked out under, the date it
        was ``measured_at`` and its ``staleness``, and every reader draws it with that
        mark and never as current. What a provisional number may not do is enter a
        comparison, a band or a ladder: `current_score` is the door for those, and it is
        ``None`` unless the estimate is current.

        Memoised on the instance, because a listing asks each game for its score five or
        six times -- once per key on the card -- and each ask used to read the run's
        index file off disk again. Forty games times six reads is the difference between
        a listing that appears and one that arrives.
        """
        if not hasattr(self, "_score_cache"):
            self._score_cache: dict[str, Any] | None = self._read_score()
        return self._score_cache

    @property
    def current_score(self) -> dict[str, Any] | None:
        """`score` where it is current, and ``None`` where it is provisional or absent.

        What every comparison reads: the bands, a neighbour's gap, the map's ramp, a
        draft's parent. A provisional estimate is never compared with a current one
        (`docs/archive/plan-evolving-score.md` §3).
        """
        return current_only(self.score)

    def _read_score(self) -> dict[str, Any] | None:
        if self.measurement is None or self.status == NONE:
            return None
        # Joined with the scoring run's own index file, from the same directory and the
        # same weights version, because `canonical.json` holds the headline and not the
        # components under it -- and the detail view is entirely about what is
        # under it. This is a *join*, not a second scan: the file is named by the
        # measurement's own weights version and read from the measurement's own
        # directory, so it cannot be another run's the way the old two scans could.
        detail = _index_detail(self.measurement)
        joined = {**detail, **self.measurement.as_dict()}
        # `scoreHtml` reads `total`; an index file that has gone missing leaves the
        # headline the canonical file recorded rather than an empty number.
        joined.setdefault("total", joined.get("index"))
        # What the estimate is and what it rests on, on every score, so a reader can never
        # draw a provisional number without its mark: the status, the scorer revision
        # (the weights version, beside the `basis` the file already carries) and the day
        # it was worked out.
        from chess_variant_sandbox import history

        joined["status"] = self.status
        basis = self.measurement.basis
        joined["revision"] = (
            basis.weights_version if basis is not None
            else self.measurement.stored.get("weights_version")
        )
        joined["measured_at"] = history.file_date(
            self.measurement.directory / "canonical.json"
        )
        # Where a sample re-certified the games on code that did not play them (C6,
        # `compute-economy.md` §3.5): how large it was, so the page can say so beside the
        # number (`certifiedMarkHtml`). `None` on a score its games were played for.
        joined["certified"] = certified_payload(basis)
        return joined

    @property
    def staleness(self) -> dict[str, Any] | None:
        if self.measurement is None:
            return None
        return self.measurement.staleness.as_dict()

    @property
    def premise(self) -> dict[str, Any] | None:
        """The strength-premise check beside the score (`premise.py`, Q502d), or ``None``.

        Read from the measurement's own directory, so a game with no score has none. It is
        no longer drawn on its own: the trust line (`trust_line`, Q647) says it among the
        rest of the evidence, and its maintainer's `words`, with the duels' numbers, ride
        in that line's `numbers`. Memoised as `score` is.
        """
        if not hasattr(self, "_premise_cache"):
            self._premise_cache: dict[str, Any] | None = None
            if self.measurement is not None:
                from chess_variant_sandbox import premise

                found = premise.read(self.hash, self.measurement.directory.parent, self.spec)
                self._premise_cache = None if found is None else found.as_dict()
        return self._premise_cache

    @property
    def trust(self) -> Any:
        """How far to trust the score (`trust.read`, Q646): a `trust.Trust`, or ``None``
        for a game never measured. Read from the measurement's own directory, as
        `premise` is, without people's counts (the site folds those in with
        `trust.with_people`), and memoised as `score` is."""
        if not hasattr(self, "_trust_cache"):
            self._trust_cache: Any = None
            if self.measurement is not None:
                from chess_variant_sandbox import trust

                self._trust_cache = trust.read(
                    self.hash, self.measurement.directory.parent, self.spec
                )
        return self._trust_cache

    @property
    def history(self) -> list[dict[str, Any]]:
        """The measurements this game's score replaced, newest first (`history.py`, Q649).

        Past values with their bases, dates and the reason each was replaced: the
        maintainer's view, never a headline. `score` is untouched by it, and the site sends
        a visitor none of it while `history.SHOW_WAS_TO_VISITORS` is ``False``. Read from
        the measurement's own directory, as `premise` is, and memoised as `score` is.
        """
        if not hasattr(self, "_history_cache"):
            self._history_cache: list[dict[str, Any]] = []
            if self.measurement is not None:
                from chess_variant_sandbox import history

                self._history_cache = history.maintainer(
                    history.read(self.hash, self.measurement.directory.parent)
                )
        return self._history_cache

    @property
    def ratings(self) -> dict[str, Any] | None:
        """The ladder, from the same measurement as the score or not at all.

        Withheld when stale for the same reason the score is: "how hard is this opponent"
        is answered by an Elo table, and a table fitted under a different engine answers
        it about a player that no longer exists.
        """
        if self.measurement is None or not self.measurement.current:
            return None
        standings = self.measurement.standings
        if not standings:
            return None
        from chess_variant_sandbox.web.opponents import table_name

        return {
            # Each row named for a person, never by its agent string (G17): the ladder that
            # scored a game is not the picker, so the page's roster names none of its rows.
            "standings": [
                {**row, "label": row.get("label") or table_name(str(row.get("name", "")))}
                if isinstance(row, dict) else row
                for row in standings
            ],
            "games": self.measurement.stored.get("games"),
            "source": self.measurement.directory.name,
            # Every seed the ladder was fitted over. A four-run ladder is the per-agent
            # mean of four fits and a one-run ladder is one fit, so the count is part of
            # the answer rather than a detail beside it.
            "seeds": list(self.measurement.stored.get("seeds") or []),
            "gamesPerPairing": self.measurement.stored.get("games_per_pairing"),
        }

    def summary(self) -> dict[str, Any]:
        """The card: enough to choose from a list, not enough to study.

        It also has to be enough to *filter and sort* a list by, which is a stronger
        requirement and the reason the numbers below are here rather than only in
        `detail`. A person narrowing forty games to the three they might play asks
        questions -- big board, few pieces, scores well on depth, came out of this run --
        that a card carrying only prose cannot answer, and answering them by fetching
        every game's detail would be forty round trips to draw one screen.
        """
        # Here rather than at the top: the site's package imports this module.
        from chess_variant_sandbox.site import stages

        board = self.spec.get("board", {})
        complexity = measure_complexity(self.spec)
        score = self.score
        return {
            "hash": self.hash,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            # Where the game stands, in the one word every page uses (Q503b). A checkout
            # has no rows, so it is read from where the file is (`stages.local`); the
            # site replaces it with the row's (`stages.stage`).
            "level": stages.local(self.source),
            "levelGloss": stages.GLOSSES.get(stages.local(self.source), ""),
            "board": f"{board.get('width', '?')}x{board.get('height', '?')}",
            # The same board as numbers. "8x8" sorts as a string and 12x8 sorts before
            # 8x8, which is the sort a person reading "board size" does not mean.
            "width": _int(board.get("width")),
            "height": _int(board.get("height")),
            "squares": _int(board.get("width")) * _int(board.get("height")),
            "pieceTypes": complexity.piece_types,
            "ruleMass": round(complexity.rule_mass, 1),
            "specialRules": complexity.special_rules,
            # How each family of the index came out, and each component under it. Both
            # are already in the score; naming them on the card is what lets the list be
            # ordered by "whether thinking pays" rather than only by the total.
            "families": {} if score is None else dict(score.get("families") or {}),
            "components": {} if score is None else dict(score.get("scores") or {}),
            # Where it sits in a design run's tree, flattened to what a grouping needs:
            # the run, how many generations in, and what it was revised from.
            **self.ancestry(),
            "cell": label(descriptor(self.spec)),
            # The same cell in words. `label` is an identifier -- it goes in filenames
            # and archives, and it grows a component only when the grid grows an axis,
            # never cosmetically -- and `1-1-0-0-1-0-0` is not something a person can
            # read, so both travel and the interface decides which to show.
            "cellBins": list(bins_of(descriptor(self.spec))),
            "index": None if score is None else score.get("index"),
            # Whether that number is current or provisional (EV2), what revision of the
            # scorer it was worked out under and when. Every card carries them, so no
            # page can draw a provisional number bare: `scoreTag` marks it.
            "status": self.status,
            "revision": None if score is None else score.get("revision"),
            "measuredAt": None if score is None else score.get("measured_at"),
            # How large a sample re-certified it (C6), or `None`: the card says *from a
            # sample* beside a number its games were not all played for.
            "certified": None if score is None else score.get("certified"),
            "viable": None if score is None else score.get("viable"),
            "weights": None if score is None else score.get("weights_version"),
            # How many runs the number rests on, and which. A mean of four runs and a
            # single run are not the same claim, so the count travels with every number
            # a card can show and the page says it beside each one.
            "seeds": None if score is None else list(score.get("seeds") or []),
            # Each run as the measurement recorded it -- seed, total, sd, verdict,
            # violations, origin. The same list the game's page reads, so the card and
            # the page cannot come to describe one measurement two ways.
            # Each with its refusals in a visitor's words too (`sitescore.runs`).
            "runs": None if score is None else sitescore.runs([
                dict(run) for run in (score.get("runs") or [])
            ]),
            # How many of those runs carried a total. Fewer than `runs` means the mean is
            # over the readable ones, and below the minimum there is no number at all.
            "totalsUsed": None if score is None else score.get("totals_used"),
            # What the runs decided together: the rule, the two counts and the split in
            # words. `runVerdict` rather than `verdict`, because `detail` already spends
            # that key on the model's written opinion and one word for two things is how
            # a page comes to render one as the other.
            "runVerdict": None if score is None else (
                {**score["verdict"], "site": {
                    "split": sitescore.split(score["verdict"], score.get("viable"))}}
                if score.get("verdict") else None),
            # The seat check over the runs pooled, and in particular the lean it prints
            # when it refuses nothing. A game whose strongest players favour one seat
            # more than the check tolerates is still ranked, and a card that showed only
            # the number would not say so.
            "seat": None if score is None else (
                {**score["seat"], "site": {"lean": sitescore.lean(score["seat"])}}
                if score.get("seat") else None),
            # Whether a model has written about this game. Not the text: a card cannot
            # hold three paragraphs, and "has been read by something that would play it
            # again" is a filter in its own right.
            "judged": None if self.verdict is None else bool(
                self.verdict.get("would_play_again")
            ),
            # Never omitted, and never folded into `index`. A card showing a blank score
            # answers "unmeasured"; one showing a blank score *and* a staleness reason
            # answers "measured, but not under the code you are running", and those are
            # different pieces of news with different repairs.
            "staleness": self.staleness,
            # Separate keys, so a page cannot render one as the other by accident.
            "designIndex": None if self.design_score is None else self.design_score.get("index"),
            "designRun": None if self.design_score is None else self.design_score.get("run"),
            "designStatus": None if self.design_score is None else self.design_score.get(
                "status"),
            "designRevision": None if self.design_score is None else self.design_score.get(
                "revision"),
            # The runs `designIndex` rests on, which the card says beside it as it does
            # beside `index`.
            "designSeeds": None if self.design_score is None else list(
                self.design_score.get("seeds") or []
            ),
            # The cheap gate's verdict, which answers a different question from the
            # index and is the one a person about to spend twenty minutes actually
            # needs. `viable` reads the index, so it is absent for anything unscored
            # and it never carries a reason -- and "scored badly" and "cannot be won"
            # are not the same news. `rps-hierarchy-chess` fails this gate by name and
            # the library said nothing, so somebody picking it played an unwinnable
            # game with no indication that it was the ruleset and not them.
            **_playability(self.spec),
            # Whether an engine nobody here wrote has a counterpart for these rules.
            # The cheap half of that question and named after it -- see `_fairy`.
            **_fairy(self.spec),
            # The sentence saying this game is kept on purpose for what it shows, or
            # `None`. Beside the number wherever the number is, so a deliberately broken
            # game at the bottom of a ranking reads as what it is -- see `exhibit`.
            "exhibit": exhibit(self),
            # The sentence saying this game is a preview, or `None` (Q931): a new kind of
            # game still being tested, listed apart and in no scored surface. A preview's
            # card is sent in the listing's `previews`, never its `games`.
            "preview": preview(self),
            # How far to trust the score (Q647, `docs/archive/plan-confidence.md` §2.1): the level word
            # as the card's chip, and its sentence for the chip's (i). `None` where there
            # is no current score, which says so itself: a provisional estimate has no
            # evidence read against the current basis, so it carries no trust level.
            "trust": trust_card(self),
            # The day the game joined the library, `YYYY-MM-DD`, or `None` (Q826): what the
            # games page's *date added* order sorts on. The site puts a member's publish
            # day here for a member's game (`site/server.py`).
            "added": added(self),
        }

    def ancestry(self) -> dict[str, Any]:
        """Which family of games this one belongs to, for grouping a list by lineage.

        Two sources and one shape. An evolved candidate carries its parents in the design
        run's archive; a game *promoted* out of one carries them in its spec's
        provenance, because the archive it came from lives under `runs/` and does not
        survive a clean checkout. Reading only the first would put chess and the eight
        games descended from it in eight different groups from the run that made them.

        ``family`` is the oldest ancestor known, which is what a person means by "show me
        everything that came out of chess". A game with no recorded parent is its own
        family, so grouping by lineage never drops a row.
        """
        declared = dict(self.spec.get("provenance") or {})
        # Two shapes for the same fact. The archive keeps `{hash, name, index}` because
        # it has the whole run in hand; a spec's `provenance.parents` is a list of bare
        # hashes, because that is all the schema allows and all a promoted file can
        # honestly carry. Names for the second come from the library, which has them.
        parents = [
            entry if isinstance(entry, dict) else {"hash": str(entry), "name": ""}
            for entry in (self.lineage.get("parents") or declared.get("parents") or [])
        ]
        first = parents[0] if parents else None
        oldest = parents[-1] if parents else None
        return {
            "run": self.lineage.get("run") or declared.get("run") or "",
            "generation": self.lineage.get("generation", declared.get("generation")),
            "theme": self.lineage.get("theme") or declared.get("theme") or "",
            "parent": None if first is None else {
                "hash": first.get("hash"), "name": first.get("name"),
            },
            "family": {"hash": self.hash, "name": self.name} if oldest is None else {
                "hash": oldest.get("hash"), "name": oldest.get("name"),
            },
            "descent": len(parents),
        }


def certified_payload(basis: Any) -> dict[str, Any] | None:
    """A re-certified score's sample, as every payload carries it (C6), or ``None``.

    ``sample`` games of ``games`` were played again on the code of ``build`` and came out
    the same, which is what keeps the score current; the page draws it as *re-certified
    from a sample of n games*, and a full replay, which writes no certificate, clears it.
    """
    found = getattr(basis, "certified", None)
    if found is None:
        return None
    return {"sample": found.sample, "games": found.games, "build": found.build}


def _stamp(path: Path) -> tuple[str, int, int]:
    """One file as the three facts that say whether it has moved: name, mtime, size.

    A missing file stamps as ``-1, -1`` rather than raising, so a file deleted between
    the walk and the stat is a *changed* listing instead of a failed one.
    """
    try:
        info = path.stat()
    except OSError:
        return (str(path), -1, -1)
    return (str(path), info.st_mtime_ns, info.st_size)


# A game's own directory directly under `runs/`: its measurement, `runs/<hash>/`, or one
# of its runs, `runs/<hash>-<seed>/` (`corpus.measurement_dir`, `corpus.run_dir`).
_GAME_DIR = re.compile(rf"[0-9a-f]{{{HASH_LENGTH}}}(-[0-9]+)?")
# Where a tournament given no `--out` goes (`cli.TOURNAMENT_DIR`).
_ONE_OFFS = "tournament"


def _top_level(root: Path) -> tuple[list[Path], list[Path]]:
    """The directories directly under ``root``: the games' own, and the rest.

    One listing of ``root`` and nothing below it. A symbolic link is neither, as it was to
    the `os.walk` this replaced, which listed a linked directory but never entered it.
    """
    games: list[Path] = []
    others: list[Path] = []
    try:
        with os.scandir(root) as listing:
            entries = list(listing)
    except OSError:
        return games, others
    for entry in entries:
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        (games if _GAME_DIR.fullmatch(entry.name) else others).append(Path(entry.path))
    return games, others


def _file_names(directory: Path) -> list[str]:
    """The names of what is directly in ``directory`` and is not a directory."""
    try:
        with os.scandir(directory) as entries:
            return [entry.name for entry in entries if not entry.is_dir()]
    except OSError:
        return []


@dataclass(frozen=True)
class _Scan:
    """Every file a listing reads, found in one walk, with the stamp of each.

    Built before `discover` reads anything, for two jobs at once. It is the *key* the
    cached listing is compared on, and it is the file list `discover` then works
    through -- so the walk happens once rather than three times.
    """

    roots: tuple[str, str, str]
    canonical: tuple[Path, ...]
    verdicts: tuple[Path, ...]
    lineages: tuple[Path, ...]
    specs: tuple[tuple[Path, str], ...]
    indexes: tuple[Path, ...]
    key: tuple[tuple[str, int, int], ...]
    # The mtime of the newest file stamped, which decides whether this may be held.
    newest: int


def _scan(
    ruleset_dir: Path, runs_dir: Path, archive_dir: Path | None
) -> _Scan:
    """Walk the three roots once and stamp everything a listing depends on.

    **Files, not directories.** `runs/`'s own mtime does not move when a re-score
    rewrites `canonical.json` inside an existing `runs/<hash>/`, so a key made of
    directory mtimes would show yesterday's score for as long as the process lived --
    the bug in `archive/bugs-resolved.md` that `game()` re-reading fixed, reintroduced
    behind a cache. The file's own mtime does move, and that is what is stamped here.

    A game's measurement is `runs/<hash>/canonical.json` and its runs are the
    `runs/<hash>-<seed>/` directories beside it. The walk stamps the `canonical.json` in
    each, so the old single-run files still on disk are stamped too; `_measurements`
    resolves each hash through `corpus.read_measurement`, which opens the measurement
    directory, so a stamped old file can move a cache key and can never become a number.

    **The games' own directories, one level down, and nothing below them** (Q885). The
    walk used to be `os.walk` over the whole of `runs/`: 10,760 directories in the main
    checkout on 2026-10-13, 1.7 s of every cold listing, for the 126 measurements among
    them. The rest is what a measurement leaves behind and no listing reads: the kept
    scores under `runs/<hash>/previous/<n>/`, every replay campaign's `replaced/` copy,
    the campaign directories' own runs, a run's games. Each `canonical.json` there names a
    hash `read_measurement` answers from `runs/<hash>/` (or `runs/<hash>-0/`) anyway, so
    it could only repeat an answer or ask about a game nobody listed (Q874). So a
    directory whose name is a game's (`_GAME_DIR`) is listed, and any other directory
    under `runs/` is only asked for the two files a listing reads by name there, a
    design run's `lineage.jsonl` and a `verdict.json`. A verdict is also looked for in a
    one-off tournament's `runs/tournament/<hash>-<seed>/`; one filed deeper than that,
    inside a campaign, is no longer found, and none is (`verdict --run` is handed a
    game's run directory, which is where `reread_one` looks too).

    The `index-<weights>.json` beside a measurement is stamped too, and only there.
    `Game._read_score` joins it onto the headline, and `chess-variant-sandbox score <run>` writes one
    without touching the `canonical.json` next to it -- so without this the components
    under a score could go on showing the previous pass's working. The hundreds of others
    under design runs are read by nothing here, and stamping them would invalidate every
    listing a generation wrote.

    The one gap left is a design run's `specs/<hash>.json` edited in place without its
    `lineage.jsonl` moving, which nothing does: `Archive.restate`, the one writer that
    edits a spec in place (a saved draft's new description, Q595), rewrites the lineage
    with it.
    """
    canonical: list[Path] = []
    verdicts: list[Path] = []
    lineages: list[Path] = []
    indexes: list[Path] = []
    # A game's strength-premise check (`premise.py`, Q502d), beside its score. Stamped
    # only, never listed: `Game.premise` reads it from the measurement's own directory, and
    # stamping it is what lets a new verdict reach a listing held in the cache.
    premises: list[Path] = []
    runs = Path(runs_dir)
    games, others = _top_level(runs)
    for here in games:
        names = _file_names(here)
        scored = "canonical.json" in names
        for name in names:
            if name == "canonical.json":
                canonical.append(here / name)
            elif name == "verdict.json":
                verdicts.append(here / name)
            elif name == "lineage.jsonl":
                lineages.append(here / name)
            elif scored and name.startswith("index-") and name.endswith(".json"):
                indexes.append(here / name)
            elif name in ("premise.json", "history.jsonl", "convergence.json"):
                # A score's history (Q649) and its convergence record (Q650) are stamped
                # the same way: a replacement that appends a line, or a new record the
                # trust line reads (Q647), reaches a listing held in the cache.
                premises.append(here / name)
    for here in others:
        # Asked by name, never listed: a design run's lineage, and a verdict written
        # beside it (`verdict --run` takes any directory).
        for name, into in (("lineage.jsonl", lineages), ("verdict.json", verdicts)):
            if (here / name).is_file():
                into.append(here / name)
    # A one-off tournament's directory (`cli._tournament_dir`), the other place
    # `verdict --run` is pointed at. Its games are no measurement, so only that.
    for here in _top_level(runs / _ONE_OFFS)[0]:
        if (here / "verdict.json").is_file():
            verdicts.append(here / "verdict.json")

    specs: list[tuple[Path, str]] = [
        (path, "committed") for path in sorted(Path(ruleset_dir).glob("*.json"))
    ]
    if archive_dir is not None:
        specs += [(path, "archived") for path in sorted(Path(archive_dir).glob("*.json"))]

    # The learned evaluator's stores (`nets/<hash>-<procedure>.json`): a new verdict moves
    # a game's trust line (Q647), so it moves the key too.
    from chess_variant_sandbox.agents.nets import STORE_DIR

    nets = sorted(Path(STORE_DIR).glob("*.json"))
    stamped = sorted(canonical), sorted(verdicts), sorted(lineages), sorted(indexes)
    key = tuple(
        _stamp(path)
        for path in [*stamped[0], *stamped[1], *stamped[2], *stamped[3],
                     *sorted(premises), *(entry[0] for entry in specs), *nets]
    )
    return _Scan(
        roots=(str(ruleset_dir), str(runs_dir), str(archive_dir)),
        canonical=tuple(stamped[0]),
        verdicts=tuple(stamped[1]),
        lineages=tuple(stamped[2]),
        specs=tuple(specs),
        indexes=tuple(stamped[3]),
        key=key,
        newest=max((stamp[1] for stamp in key), default=0),
    )


_Key = tuple[tuple[str, int, int], ...]

# The last few listings, by the three roots they were taken from. A few rather than one
# because the tests and the site both hold more than one root triple in a process, and
# bounded because each entry holds every spec in the corpus.
_LISTINGS: dict[tuple[str, str, str], tuple[_Key, list[Game]]] = {}
_LISTINGS_KEPT = 8

# How old the newest stamped file must be before a listing may be cached at all. A
# filesystem whose mtime is coarser than its clock -- ext4's is, by a timer tick -- can
# stamp a write that lands just after a scan with the mtime the scan already read, and a
# cache cannot tell that apart from no write at all. Waiting one tick out costs a listing
# nobody has asked for yet and closes the window; the box's disk is ext4.
_SETTLE_NS = 20_000_000


def _cached(scan: _Scan) -> list[Game] | None:
    """The listing taken from these exact files, or ``None`` if anything has moved."""
    held = _LISTINGS.get(scan.roots)
    if held is None or held[0] != scan.key:
        return None
    return held[1]


def _remember(scan: _Scan, games: list[Game]) -> None:
    if scan.newest + _SETTLE_NS > time.time_ns():
        return
    _LISTINGS.pop(scan.roots, None)
    while len(_LISTINGS) >= _LISTINGS_KEPT:
        _LISTINGS.pop(next(iter(_LISTINGS)))
    _LISTINGS[scan.roots] = (scan.key, games)


def discover(
    ruleset_dir: Path = RULESET_DIR,
    runs_dir: Path = RUNS_DIR,
    archive_dir: Path | None = None,
) -> list[Game]:
    """Every game worth showing, in the order of how far along the ladder it is.

    Three sources, and the order is what resolves them: a game promoted out of a design
    run still has its copy in that run's `specs/`, so committed wins over archived and
    archived over the candidate both of them came from. Otherwise a promoted game would
    keep offering to be promoted.

    Design runs are read through ``Archive`` rather than by globbing their spec
    directory, because the lineage is what makes an evolved ruleset legible: without it
    a generated game is a name, a hash and no answer to "where did this come from".

    The archive is opt-in rather than defaulted, unlike the other two roots: a caller
    that names its own directories must not silently read one it did not name, and this
    one arrived after several callers had already named two.

    **Cached on what it read, not on how long ago it read it.** `_scan` stamps every
    file behind the answer, and an unmoved stamp returns the same games without opening
    anything -- so the listing a person waits for costs a walk rather than four thousand
    file opens. A moved stamp is a full re-read, which is why the freshness the web layer
    promises survives: a re-score rewrites `canonical.json` and the very next listing
    sees it.
    """
    scan = _scan(ruleset_dir, runs_dir, archive_dir)
    held = _cached(scan)
    if held is not None:
        # A copy, so a caller that sorts or filters the list it was handed cannot edit
        # what the next caller is given. The `Game` objects themselves are shared, and
        # everything they memoise is derived from the files just stamped.
        return list(held)

    verdicts = _verdicts(runs_dir, scan.verdicts)
    games: dict[str, Game] = {}

    for path, source in scan.specs:
        try:
            spec = load_spec(path)
        except (OSError, ValueError):
            continue
        digest = spec_hash(spec)
        if digest in games:
            continue
        games[digest] = Game(
            hash=digest,
            name=str(spec.get("name", path.stem)),
            description=str(spec.get("description", "")),
            path=path,
            # An archived file that says it is a preview is filed as one (Q931, `preview`).
            source=(
                PREVIEW
                if source == "archived" and preview_sentence(spec) is not None
                else source
            ),
            spec=spec,
            verdict=verdicts.get(digest),
        )

    for lineage in scan.lineages:
        archive = Archive(lineage.parent)
        by_hash = {entry.hash: entry for entry in archive.entries}
        for entry in archive.entries:
            archived = archive.load_spec(entry.hash)
            if archived is None or entry.hash in games:
                continue
            spec = archived
            games[entry.hash] = Game(
                hash=entry.hash,
                name=entry.name,
                description=str(spec.get("description", "")),
                path=lineage.parent / "specs" / f"{entry.hash}.json",
                # A hand-built game and a mutation are both candidates in a run
                # directory, and they are not the same thing to a reader: one was
                # written on purpose and the other was found. The archive is the only
                # place that knows which, so it says so on the entry.
                source=(
                    "designed"
                    if (entry.detail or {}).get("origin") == "designed"
                    else "evolved"
                ),
                spec=spec,
                verdict=verdicts.get(entry.hash),
                lineage=_lineage(entry, by_hash, lineage.parent.name),
                design_score=_design_score(entry, lineage.parent.name),
            )

    # Every spec this call can name for a hash, gathered above -- committed, archived
    # and lineage alike. Handed to `_measurements` so it can resolve each
    # `canonical.json` against a spec already in hand rather than asking `corpus.find`
    # to re-glob and re-hash the whole corpus once per measurement.
    spec_by_hash = {digest: game.spec for digest, game in games.items()}
    seeds_by_hash = {
        digest: _expected_seeds(game.source) for digest, game in games.items()
    }
    measurements = _measurements(
        runs_dir, spec_by_hash, scan.canonical, seeds_by_hash, only_known=True
    )
    for digest, game in games.items():
        game.measurement = measurements.get(digest)

    order = {"committed": 0, "archived": 1, PREVIEW: 2, "designed": 3, "evolved": 4}
    found = sorted(
        games.values(),
        key=lambda game: (order.get(game.source, 5), game.name.lower()),
    )
    _remember(scan, found)
    return list(found)


def reread_one(game: Game, runs_dir: Path) -> Game:
    """This one game's own measurement and verdict, re-read from its own directory.

    `App.game()`'s fix for performance item 20: it used to answer a request for one
    game by calling `discover()` again in full, so that a measurement which had just
    finished would be seen rather than the stale snapshot the bug in
    `docs/archive/bugs-resolved.md` describes ("nobody has measured this one" printed the
    instant after a tournament that measured it had finished). That kept the freshness at
    the cost of re-reading every *other* game's `canonical.json` too -- a corpus-wide
    `_scan` to answer a question about one hash.

    `corpus.read_measurement` already resolves `runs/<hash>/canonical.json` (or the old
    `runs/<hash>-0/` fallback) directly, at a handful of file opens rather than a walk of
    `runs/`, when a spec is already in hand -- which `game.spec` is, from the caller's
    last full listing. The verdict beside it is read the same bounded way, by
    `_verdict_for`.

    Everything else on the `Game` -- name, description, spec, source, lineage and
    design_score -- is the caller's already-known copy, unchanged. Those come from the
    ruleset file and the design run's archive, not from a tournament that just finished,
    and `discover()` is still what notices when one of *them* moves (an edited name, a
    promoted candidate, a brand-new ruleset); `App.game()` falls back to a full refresh
    when the hash is not one its last listing has ever seen.
    """
    from chess_variant_sandbox import corpus

    measurement = corpus.read_measurement(
        game.hash, runs_dir, spec=game.spec, seeds=_expected_seeds(game.source)
    )
    verdict = _verdict_for(game.hash, runs_dir)
    return replace(game, measurement=measurement, verdict=verdict)


def _expected_seeds(source: str) -> tuple[int, ...]:
    """How many runs this game's measurement owes, by where the listing found it.

    `corpus.expected_seeds` is the one place that answer lives, and it is keyed by the
    ladder's stage rather than by this module's `source`. A design run's game, evolved
    or built in the designer, is a candidate and owes one run; everything else a `Game`
    can be -- committed, archived, or a library game the site has retired -- was the
    library and owes four. Without this every reader here expected four of every game,
    so a candidate measured on its own rung read *out of date* on every screen while
    `corpus.refresh` found it current and did nothing: a *measure it* with no exit (Q488).
    """
    from chess_variant_sandbox import corpus

    stage = "candidate" if source in ("designed", "evolved") else "committed"
    return corpus.expected_seeds(stage)


# Every seed a corpus game's own tournament could be filed under -- `ONE_SEED` for a
# design candidate, `LIBRARY_SEEDS` for a committed or archived one -- so `_verdict_for`
# can check exactly the directories that belong to one hash without asking which kind of
# game it is.
def _verdict_seeds() -> tuple[int, ...]:
    from chess_variant_sandbox import corpus

    return tuple(sorted({seed for group in corpus.KNOWN_SEED_SETS for seed in group}))


def _verdict_for(digest: str, runs_dir: Path) -> dict[str, Any] | None:
    """The verdict written about this one game, read without walking `runs/`.

    `_verdicts` (the full-listing path) finds a verdict anywhere under `runs/`, matched
    by the `ruleset_hash` its content carries rather than by where it sits, because
    `chess-variant-sandbox verdict --run <run>` takes any run directory on the command
    line and nothing enforces which one. In practice that directory is one of this
    game's own -- its measurement directory or one of its seeded tournament directories,
    never an unrelated one -- so checking those few, rather than the whole tree, answers
    the same question at a bounded cost.

    Ties (more than one of this game's directories carrying a verdict) resolve the same
    way `_verdicts` resolves them across the whole tree: most recently written wins.
    """
    from chess_variant_sandbox import corpus

    directories = [corpus.measurement_dir(digest, runs_dir)]
    directories += [
        corpus.run_dir(digest, seed, runs_dir) for seed in _verdict_seeds()
    ]
    found: dict[str, Any] | None = None
    newest = -1
    for directory in directories:
        path = directory / "verdict.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("ruleset_hash") != digest:
            continue
        stamped = _stamp(path)[1]
        if stamped >= newest:
            newest = stamped
            found = data
    return found


def _design_score(entry: Entry, run: str) -> dict[str, Any] | None:
    """What the design run that played this candidate made of it, or ``None`` if none did.

    Three cases, and the card needs all three told apart (`upkeep.scoreTag`). A number
    with its run: the rough pill. **A run with no number**: the candidate was played and
    could not be scored -- since C3 the loop archives such a result with no index rather
    than a placeholder 0.0 (`orchestrator.measure`) -- and the card says *no score*, naming
    the run. Nothing: the candidate never reached a tournament (proposed, invalid,
    duplicate, gated out), and the card says *not measured*.

    The null index stays null. Its spread goes with it: a spread beside no number would
    read as a precision the entry does not have.
    """
    if entry.index is None and entry.status != "evaluated":
        return None
    return {
        "index": entry.index,
        # The same rule as a library score (EV2): a number worked out under an earlier
        # revision of the scorer than this checkout's is provisional, and says which.
        # An entry that records no version cannot say, and is provisional too.
        "status": CURRENT if entry.weights_version == DEFAULT_WEIGHTS else PROVISIONAL,
        "revision": entry.weights_version or None,
        "sd": None if entry.index is None else entry.index_sd,
        "viable": entry.viable,
        "run": run,
        # Empty for an entry written before the archive kept them, and the card then
        # shows no number (`upkeep.scoreTag`).
        "seeds": list(entry.seeds),
    }


def trust_card(game: Game, people: Any = None) -> dict[str, Any] | None:
    """A card's chip (`trust.card`), with people's counts folded in where given."""
    from chess_variant_sandbox import trust

    found = game.trust
    # A preview's score is not yet trusted, and its own sentence says so (Q931).
    if found is None or game.status != CURRENT or is_preview(game):
        return None
    return trust.card(found if people is None else trust.with_people(found, people))


def trust_line(game: Game, people: Any = None, *, maintainer: bool = True) -> dict[str, Any] | None:
    """A game page's trust line (`trust.shown`): the reading, people's counts folded in
    where given, each earlier score as a visitor reads it, and whether Fairy-Stockfish
    could play the rules at all. ``maintainer`` false leaves the numbers out. ``None``
    on a provisional estimate (EV2): trust reads evidence against the current basis,
    and a provisional estimate has none (`docs/archive/plan-evolving-score.md` §5.1)."""
    from chess_variant_sandbox import history, trust

    found = game.trust
    # A preview's score is not yet trusted, and its own sentence says so (Q931).
    if found is None or game.status != CURRENT or is_preview(game):
        return None
    if people is not None:
        found = trust.with_people(found, people)
    entries = [history.Entry.from_dict(line) for line in game.history]
    score = game.score or {}
    return trust.shown(
        found,
        was=history.for_visitors(entries, score.get("index")),
        fairy_translates=_fairy(game.spec).get("fairyTranslates"),
        premise_words=str((game.premise or {}).get("words") or ""),
        maintainer=maintainer,
    )


def detail(game: Game, others: Iterable[Game] | None = None) -> dict[str, Any]:
    """Everything about one game: rules in English, measurements, and where it came from.

    The rules arrive twice, in the two shapes ``spec/describe.py`` produces. ``rules`` is
    the block a model reads; ``setupLines`` and ``ruleLines`` are the same sentences as
    parts, because a player wants one rule at a time next to the board it governs and a
    wall of preformatted text is not that.

    ``others`` is the rest of the library, which names the reference game a piece is
    said to move like. The games near this one are not here: they are the map's
    (`neighbours`, over the map's own games), and the server adds them (`App.near`).
    """
    from chess_variant_sandbox.site import stages

    cell = descriptor(game.spec)
    return {
        **game.summary(),
        # *Where it stands*: the level's sentence and the designer's track, from the
        # same word the card carries (Q503b).
        **stages.shown(stages.local(game.source)),
        "rules": describe_spec(game.spec),
        "setupLines": describe_setup(game.spec),
        "ruleLines": describe_rules(game.spec),
        "spec": game.spec,
        "provenance": provenance(game),
        # The score as stored, with its refusals, its split and its lean also said in a
        # visitor's words (`sitescore`, G29-G31), which the page draws where it is the site.
        "score": sitescore.worded(game.score),
        "scoreGuide": score_guide(game.score),
        # The measurements the score replaced (Q649), the maintainer's view with bases.
        # Never a headline; the site keeps it from a visitor (`SiteApp`'s game page),
        # who reads each one as a *was* line under the trust line's *more*.
        "history": game.history,
        # The trust line under the score (Q647): the level word, one sentence, the *more*
        # list and the maintainer's numbers. The site sends a visitor no `numbers` and
        # folds in people's counts (`SiteApp`'s game page).
        "trust": trust_line(game),
        "verdict": game.verdict,
        "cellDescription": describe_cell(cell),
        "axes": describe_axes(cell),
        "pieces": pieces(game),
        # K1 (`docs/archive/plan-understanding.md` §4.2 step 1): the opening position the page
        # draws before a game starts, the card's `still` with its drawings put back.
        "opening": opening(game),
        # K2: each piece alone on an empty board, the engine's answer, with the reference
        # game's piece it moves like where there is one.
        "pieceMaps": piece_maps(game, reference_of(others)),
        "sides": list(sides(game)),
        "armies": armies(game),
        # Schema 26 (Q918c): whether both players move at once, every round, so the page
        # never says one of them moves first. A spy game (schema 27) starts by turns, and
        # says who moves first, so it is not one.
        "rounds": isinstance((game.spec.get("rules") or {}).get("simultaneous"), dict)
        and not is_spy_game(game.spec),
        "ratings": game.ratings,
        # Which of the fourteen things an opponent can be told to care about are not
        # inert here. Six of them read a rule most rulesets do not declare, and a slider
        # that silently does nothing is worse than one that is not offered.
        "liveTerms": _live_terms(game.spec),
        # What each piece is worth to the rules, which is what a value slider moves
        # relative to. Ruleset-shaped, so it belongs with the game rather than with the
        # design kit the way the fixed vocabulary does.
        "pieceValues": _piece_values(game.spec),
    }


# How many games one page names as like this one: the map's panel's `families.MUTUAL`
# open and the rest folded (`detail.js`), twelve in all, which is also what the page the
# server draws for a game's own address names (`web/page/game.py`'s `NEAR_SHOWN`).
MOST_LINKS = 12


def neighbours(
    game: Game, mapped: Iterable[Game], runs_dir: Path | str
) -> dict[str, Any]:
    """The games most like this one on the map, most alike first, and what each pair is.

    **One meaning of near** (M5 of `docs/archive/plan-map.md`). Which games and in what order is
    `graphing.near`'s, which reads the map built over `mapped` -- the same games, the same
    `spec/likeness.py` relation and its order -- so this page, the map's panel and the
    site's page for a game's own address name the same games in the same order, and the
    first `families.MUTUAL` here are the panel's *games like this one*. `mapped` is the
    map's population as the server draws it (`App.mapped`); a game not on the map (a draft,
    a candidate) is ranked against the map's games by the same relation.

    What each link adds is the reading: how alike the two are and the rarest things both
    have (the map's words), what differs between them -- `spec/relate.py`'s count of rules
    and its sentences, which explain a difference and rank nothing -- and the gap between
    the two scores where both carry a *current* measurement under the same weights.

    **The gap is withheld unless both sides are current.** A missing measurement shows no
    number, and a stale one shows a provisional number (`Game.score`) that never enters a
    comparison: a difference of two numbers one of which is a past value is not a
    separation claim anyone can make. Scores under two weights versions never share a
    leaderboard, so a pair that straddles a version has no gap however fresh both are.

    Answers `{"links", "onMap", "community", "open"}`: `community` is the map's family the
    game is in (`{id, name}`) or `None`, and `open` how many links the page shows before
    folding the rest. About a millisecond a link the first time (`relate` memoises each
    pair) and the map's own cost, which a warm map does not pay again (`graphing._MAPS`).
    """
    from chess_variant_sandbox.spec import relate
    from chess_variant_sandbox.web import families

    found, by_hash = _nearest(game, mapped, runs_dir)
    if found is None:
        return {"links": [], "onMap": False, "community": None, "open": families.MUTUAL}
    mine = relate.normal_form(game.spec)
    links = []
    for row in found.like:
        other = by_hash[str(row["hash"])]
        theirs = relate.normal_form(other.spec)
        links.append(
            _link(
                game, other, row,
                relate.relate_forms(mine, theirs), relate.similarity_forms(mine, theirs),
            )
        )
    return {
        "links": links,
        "onMap": found.on_map,
        "community": found.community,
        "open": families.MUTUAL,
    }


def near_names(
    game: Game, mapped: Iterable[Game], runs_dir: Path | str
) -> list[tuple[str, str]]:
    """`neighbours`' games as `(hash, name)`, in its order, without reading each pair.

    What the page for a game's own address names (`site/game_page.py`): the same list,
    for a reader who is shown names and nothing else.
    """
    found, by_hash = _nearest(game, mapped, runs_dir)
    if found is None:
        return []
    return [(by_hash[str(row["hash"])].hash, by_hash[str(row["hash"])].name) for row in found.like]


def _nearest(
    game: Game, mapped: Iterable[Game], runs_dir: Path | str
) -> tuple[Any, dict[str, Game]]:
    """`graphing.near` over the scored part of `mapped`, and those games by hash.

    `None` for a preview, which has no neighbours and is nobody's (Q931: a link carries the
    gap between two scores, and a preview is in no comparison), and where the map holds
    no game.
    """
    from chess_variant_sandbox.web import graphing

    population = scored(mapped)
    if is_preview(game) or not population:
        return None, {}
    found = graphing.near(game, population, runs_dir, MOST_LINKS)
    return found, {other.hash: other for other in population}


# The same counts as `relate.Similarity.sentence` says them, in a visitor's words (G40 of
# `docs/archive/plan-understanding.md` §3.3): a mechanism is a *way of moving*, a square a
# *starting square*, a draw clock or a piece's price a *limit or price*. Sent beside the
# maintainer's as the similarity's `site`, and drawn where the page is the site.
SITE_LIKENESS: tuple[tuple[str, str, str], ...] = (
    ("rules", "rule", "rules"),
    ("settings", "rule set differently", "rules set differently"),
    ("mechanisms", "way of moving", "ways of moving"),
    ("pieces", "piece", "pieces"),
    ("squares", "starting square", "starting squares"),
    ("books", "limit or price", "limits or prices"),
)


def likeness_words(counts: Mapping[str, int]) -> str:
    """What differs between two games, from `relate`'s counts: *1 rule, 28 starting squares*.

    The first count is always said, as `relate` says it, because *no rules* is the most
    important thing the reading reports; the rest only when they are not zero.
    """
    said = []
    for position, (name, one, many) in enumerate(SITE_LIKENESS):
        number = int(counts.get(name, 0) or 0)
        if number or position == 0:
            said.append(f"{number or 'no'} {one if number == 1 else many}")
    return ", ".join(said)


def _link(
    game: Game,
    other: Game,
    row: Mapping[str, Any],
    relation: Any,
    alike: Any,
) -> dict[str, Any]:
    first = relation.differences[0] if relation.differences else None
    return {
        "hash": other.hash,
        "name": other.name,
        "source": other.source,
        # The map's reading of the pair, which is what orders the list: how alike the two
        # are (0 to 1), the rarest things both have, in words, and whether a line joins
        # them on the map (each among the other's nearest).
        "likeness": row["likeness"],
        "shares": list(row["shares"]),
        "line": bool(row["line"]),
        # What differs, which orders nothing: `relate`'s count of rules apart, and its
        # `similarity` counts said by kind (`{index, counts, words}`, with the site's words).
        "units": relation.units,
        "similarity": {**alike.as_dict(), "site": likeness_words(alike.counts)},
        # The kind of the first difference, which is the whole kind on a one-rule link
        # and the one a mixed link is filed under. Every kind on the link travels too, so
        # a filter never has to guess from the first.
        "kind": "" if first is None else first.kind,
        "kinds": sorted({difference.kind for difference in relation.differences}),
        "sentence": relation.sentence,
        "differences": [difference.as_dict() for difference in relation.differences],
        "gap": _gap(game, other),
        # The neighbour's number is printed in the gap, so its label travels with it.
        "exhibit": exhibit(other),
    }


def _gap(game: Game, other: Game) -> dict[str, Any] | None:
    """Both indexes and the distance between them, where both can be read at all.

    *Clear* is the gap against two combined standard deviations. The spread is the
    within-run figure `canonical.json` stores; `docs/reference/run-to-run-spread.md`
    records a larger one, and the plan's G4 is where the map takes whichever
    `measurement` exposes.
    """
    # Current only: a gap is a separation claim, and a provisional estimate never enters
    # one (`docs/archive/plan-evolving-score.md` §3).
    mine, theirs = current_only(game.score), current_only(other.score)
    if not mine or not theirs:
        return None
    if mine.get("weights_version") != theirs.get("weights_version"):
        return None
    here, there = mine.get("index"), theirs.get("index")
    if here is None or there is None:
        return None
    # How many runs each end is the mean of. Both numbers are printed, so both counts
    # travel, under the card's own key; an end that names none has no number to print.
    seeds, other_seeds = list(mine.get("seeds") or []), list(theirs.get("seeds") or [])
    if not seeds or not other_seeds:
        return None
    spread = (float(mine.get("sd") or 0.0) ** 2 + float(theirs.get("sd") or 0.0) ** 2) ** 0.5
    gap = float(there) - float(here)
    return {
        "index": float(here),
        "otherIndex": float(there),
        "seeds": seeds,
        "otherSeeds": other_seeds,
        "gap": gap,
        "spread": spread,
        "clear": abs(gap) > 2 * spread,
        "weights": mine.get("weights_version"),
    }


def _live_terms(spec: dict[str, Any]) -> list[str]:
    """Never raised. A ruleset the evaluator cannot be built for still has a page."""
    from chess_variant_sandbox.web.designing import live_terms

    try:
        return live_terms(spec)
    except Exception:  # noqa: BLE001 - a detail view must survive one odd spec
        return []


def _piece_values(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Never raised, for the reason above: a detail view must survive one odd spec."""
    from chess_variant_sandbox.web.designing import piece_values

    try:
        return piece_values(spec)
    except Exception:  # noqa: BLE001
        return []


def keeping(
    game: Game,
    runs_dir: Path = RUNS_DIR,
    ruleset_dir: Path = RULESET_DIR,
    archive_dir: Path = ARCHIVE_DIR,
) -> dict[str, Any]:
    """What can still become of this game, and what stands in the way.

    Every judgement here is `corpus`'s, asked rather than restated: the page must not be
    able to offer a promotion the command line would refuse, and the reason a promotion
    is refused is more useful to a person than a greyed-out button.
    """
    from chess_variant_sandbox import corpus

    try:
        candidate = corpus.find(
            game.hash, runs_dir=runs_dir, ruleset_dir=ruleset_dir, archive_dir=archive_dir
        )
    except corpus.CorpusError:
        return {"stage": "unknown", "measurement": None, "blocker": None, "roster": []}

    measurement = corpus.read_measurement(
        game.hash, runs_dir, spec=candidate.spec,
        seeds=corpus.expected_seeds(candidate.stage),
    )
    return {
        "stage": candidate.stage,
        "measurement": None if measurement is None else measurement.as_dict(),
        "blocker": corpus.promotion_blocker(candidate, runs_dir, ruleset_dir),
        # Shown because "tested" means nothing without saying against what, and this
        # ladder is the whole difference between a design run's number and a corpus one.
        # This game's: a net ladder where its learned net is rated (Q502h).
        "roster": list(corpus.canonical_agents(game.hash, spec=candidate.spec)),
        "gamesPerPairing": corpus.CANONICAL_GAMES,
        "lineage": dict(candidate.lineage),
    }


def sides(game: Game) -> tuple[str, str]:
    """What to call the two players of this game."""
    compiled = engine_for(game.spec).spec
    return side_names(
        icons_for(
            compiled.piece_names,
            compiled.piece_ids,
            (game.spec.get("appearance") or {}).get("icons"),
        )
    )


def armies(game: Game) -> list[list[dict[str, Any]]]:
    """What each side starts with: one entry per piece type it owns, most numerous first.

    Which side you take is a real choice in an asymmetric game -- horde is thirty-six
    pawns against an ordinary army -- and until this existed the picker offered the
    choice without saying what either side was.

    Read off the compiled opening position rather than off ``setup.placements``, because
    a spec may state one army and derive the other by mirroring or by rotation. The side
    somebody is choosing is the one that appears on the board, not the one in the file.

    **A wild piece is in neither army** (schema 9). The row is counted -- the board has
    three owners and a table with two rows raised on every ``rules.npcs`` ruleset -- and
    then dropped, because the question this answers is *which seat do I take*, and a boar
    stands on the same squares whichever seat that is. What it is and what it does are
    the pieces list's and the rules prose's to say; the side picker's job is the choice,
    and a third army on a page offering two seats would read as a side nobody can play.
    """
    engine = engine_for(game.spec)
    state = engine.initial_state()
    types = engine.spec.n_types
    # By the army's stride, not the type count: with a facing, `// n_types` is the frame
    # a piece reads its moves from and the second player's pieces would be counted as a
    # third and fourth army that does not exist. And by `n_owners` rather than by 2, so
    # that a wild piece lands in its own row instead of indexing off the end.
    stride = engine.spec.side_stride
    counts = [[0] * types for _ in range(engine.spec.n_owners)]
    for value in state.board:
        if value:
            counts[(value - 1) // stride][(value - 1) % types] += 1
    # A ruleset could open with something already in hand; no committed one does, and an
    # army summary that ignored it would be wrong for the first that did. A hand is a
    # player's by construction -- nothing is ever held for a wild piece.
    for player, hand in enumerate(state.hands):
        for kind, held in enumerate(hand):
            counts[player][kind] += held

    return [
        sorted(
            (
                {
                    "id": engine.spec.piece_ids[kind],
                    "name": engine.spec.piece_names[kind],
                    "count": count,
                }
                for kind, count in enumerate(side)
                if count
            ),
            key=lambda entry: (-int(entry["count"]), str(entry["id"])),
        )
        for side in counts[:2]
    ]


def pieces(game: Game) -> list[dict[str, Any]]:
    """Every piece: what it looks like, and what it does.

    Both seats' icons, not just the first player's. A single-colour set -- shogi,
    xiangqi -- tells the two players apart by rotating the piece rather than recolouring
    it, so a guide that showed one icon per piece would be showing half of some games'
    boards and claiming it was all of them.
    """
    from chess_variant_sandbox.spec.describe import describe_pieces, sight_sentence
    from chess_variant_sandbox.spec.schema import piece_sights

    compiled = engine_for(game.spec).spec
    art = icons_for(
        compiled.piece_names,
        compiled.piece_ids,
        (game.spec.get("appearance") or {}).get("icons"),
    )
    # Schema 28 (Q943b, `docs/archive/plan-speed.md` §4.5): each piece's speed on its card, the
    # compiled table's (0 where none is written); absent in a game without a speed.
    sim = compiled.simultaneous
    speeds = sim.speed if sim is not None else ()
    # Schema 29 (Q983b, `docs/plan-rest.md` §5.6): each piece's rest on its card, the compiled
    # table's (0 where none is written), for the card's pill and the board's line under a
    # picked piece; absent in a game without a rest.
    rests = compiled.rest
    # Schema 30 (Q999c, `docs/archive/plan-passive.md` §4.6): each piece's lines of sight, one
    # sentence each in `describe`'s words (`sight_sentence`) with the effect it carries, for
    # the board's line under a picked piece; absent on a piece that sees nothing. The card's
    # notes already say them, with *see the rules*.
    sights = piece_sights(game.spec)
    names = dict(zip(compiled.piece_ids, compiled.piece_names, strict=True))
    hexy = (game.spec.get("board") or {}).get("lattice") == "hex"
    return [
        {
            **prose.as_dict(),
            **({"speed": speeds[index]} if speeds else {}),
            **({"rest": rests[index]} if rests else {}),
            **(
                {"sights": [
                    {"effect": sight.effect, "words": sight_sentence(sight, names, hexy)}
                    for sight in sights[prose.id]
                ]}
                if prose.id in sights else {}
            ),
            "icon": (art[index][0] or {}).get("href"),
            "icons": [
                None if icon is None else {"href": icon["href"], "rotate": icon["rotate"]}
                for icon in art[index]
            ],
        }
        for index, prose in enumerate(describe_pieces(game.spec))
    ]


def score_guide(score: dict[str, Any] | None) -> dict[str, Any] | None:
    """What the numbers on this score mean, under the weights that produced it.

    Keyed off the score's own ``weights_version`` rather than off the current default,
    because a score records the version it was computed under and versions are never
    edited. Explaining a v3 score with v5's curves would be describing a scoring rule
    that number was never subject to -- and the older versions are exactly the ones
    still sitting in ``runs/`` for the interface to render.
    """
    if not score:
        return None
    try:
        # Each test a game must pass carries its visitor's words too (G29).
        return sitescore.guide(explain(str(score.get("weights_version", ""))))
    except WeightsError:
        # A run written under a version this checkout no longer ships. The number is
        # still real and still worth showing; only the explanation is missing.
        return None


# ------------------------------------------------------- where a game sits in the library


@dataclass(frozen=True)
class Band:
    """How one game's score stands against every other measured game's.

    Three counts and never a place. The interval on a score is wider than most of the
    gaps between neighbours, so *12th of 71* would be noise dressed as a ranking -- the
    band says which games this one is clearly above, which are clearly above it, and
    which the measurement cannot separate it from at all.
    """

    clears: int
    cleared_by: int
    close: int

    @property
    def of(self) -> int:
        """How many games it was compared against."""
        return self.clears + self.cleared_by + self.close

    def as_dict(self) -> dict[str, int]:
        return {
            "clears": self.clears,
            "clearedBy": self.cleared_by,
            "close": self.close,
            "of": self.of,
        }


class _Measured(NamedTuple):
    """One game as the separation rule reads it, and nothing else.

    Hashable and made of plain numbers, so the tuple of these *is* the cache key: it
    differs exactly when a measurement a comparison depends on differs.
    """

    game: str
    weights: str
    headline: float
    ci_low: float
    ci_high: float
    sweeps: float
    bounded: bool


def bands(games: Iterable[Game]) -> dict[str, dict[str, int]]:
    """Every measured game's band, keyed by hash, with the rest of the library absent.

    A game with a stale or missing measurement is not in the answer at all, and the page
    draws nothing for it. That is the corpus rule rather than a page's choice, and it is
    read off `Game.current_score` -- which is `None` in both cases, a provisional estimate
    included -- rather than restated here.

    **The pass is quadratic and it is paid once per measurement, not once per request.**
    The 71 games in the library are 71 x 70 pairwise calls, measured at 5.4 ms; the
    answer is memoised on the tuple of `_Measured`, which is every figure a comparison
    reads and no figure it does not, and a hit costs the 0.15 ms of building that key
    over a listing already in hand. A changed measurement changes its game's tuple and
    the whole table is recomputed; nothing else can invalidate it. Keying on the
    listing's file stamps instead would cost the same and invalidate on a spec edit that
    moves no score.

    **The population is the whole measured library, not the listing a reader is looking
    at.** So a card under a filter, the same card unfiltered and the game's own page all
    say the same three numbers, which they could not if the band followed the listing.

    **One leaderboard, the current version's** (EV3, `docs/archive/plan-evolving-score.md` §3,
    §5.2). Only a score under `DEFAULT_WEIGHTS` enters, so scores under different weights
    never share a band: `_comparable` drops the rest, a provisional estimate included,
    rather than banding each version apart beside the other.
    """
    return {
        digest: band.as_dict() for digest, band in _bands(_comparable(games)).items()
    }


def _comparable(games: Iterable[Game]) -> tuple[_Measured, ...]:
    """Every game with a current measurement, reduced to what a comparison reads.

    `current_only`, never a bare `Game.score`: a provisional estimate is placed in a listing
    by its number and marked, and never enters a band (`docs/archive/plan-evolving-score.md` §5.2).
    A score that calls itself current under another weights version (one built by hand,
    which names no status) is left out too: the band is the current version's.
    """
    found: list[_Measured] = []
    # A preview is in no band, and is no other game's comparison either (Q931, `preview`).
    for game in scored(games):
        score = current_only(game.score)
        if score is None or score.get("weights_version") != DEFAULT_WEIGHTS:
            continue
        headline = score.get("combined")
        if headline is None:
            headline = score.get("total")
        if headline is None:
            continue
        raw = score.get("interval")
        interval: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
        found.append(
            _Measured(
                game=game.hash,
                weights=str(score.get("weights_version", "")),
                headline=float(headline),
                ci_low=float(interval.get("ci_low", 0.0)),
                ci_high=float(interval.get("ci_high", 0.0)),
                sweeps=float(interval.get("deterministic_pairings", 0.0)),
                bounded=bool(interval),
            )
        )
    return tuple(sorted(found))


@lru_cache(maxsize=8)
def _bands(entries: tuple[_Measured, ...]) -> dict[str, Band]:
    """The pairwise pass. Cached on its whole input, which is its whole basis."""
    results = {entry.game: _as_result(entry) for entry in entries}
    found: dict[str, Band] = {}
    for entry in entries:
        mine = results[entry.game]
        above = below = 0
        for other in entries:
            if other.game == entry.game:
                continue
            theirs = results[other.game]
            if mine.separated_from(theirs):
                above += 1
            elif theirs.separated_from(mine):
                below += 1
        found[entry.game] = Band(above, below, len(entries) - 1 - above - below)
    return found


def provisional_count(cards: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """How many of a listing's scores are provisional, or ``None`` when none are.

    The leaderboard's header (EV3, `docs/archive/plan-evolving-score.md` §5.2): *4 of 73 scores
    are provisional while they are measured again.* Read off each card's own `status`,
    so it counts the cards a reader is shown and no others; *of* is every card with a
    number, current or provisional, since a game with no estimate is not a score. When
    nothing is provisional it says nothing at all, rather than *0 of 73*.
    """
    statuses = [card.get("status") for card in cards]
    provisional = sum(1 for status in statuses if status == PROVISIONAL)
    if not provisional:
        return None
    scored = sum(1 for status in statuses if status in (CURRENT, PROVISIONAL))
    return {
        "provisional": provisional,
        "of": scored,
        "sentence": provisional_sentence(provisional, scored),
    }

    return f"{provisional} of {scored} scores are provisional while they are measured again."


def _as_result(entry: _Measured) -> IndexResult:
    """A stored measurement as the type that owns the separation rule.

    The rule is `IndexResult.separated_from` and it is called from where it lives. It
    reads the headline and three numbers off the interval, so the fields it never touches
    are left empty rather than half-filled out of a file this comparison does not read.
    An absent interval is absent here too, and the rule falls back to comparing headlines.
    """
    return IndexResult(
        ruleset_hash=entry.game,
        ruleset_name="",
        weights_version=entry.weights,
        total=entry.headline,
        viable=True,
        violations=[],
        scores={},
        contributions={},
        raw={},
        families={},
        interval=(
            {
                "ci_low": entry.ci_low,
                "ci_high": entry.ci_high,
                "deterministic_pairings": entry.sweeps,
            }
            if entry.bounded
            else None
        ),
    )


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def metric_legend() -> dict[str, Any]:
    """What a list can be ordered by, in the words the glossary already uses.

    The sort menu is the one place in the interface where a component's identifier would
    otherwise be unavoidable -- there is no room on a card for a sentence, and
    ``gradient_persistence`` is not a menu item. So the titles and the one-line
    explanations come from ``metrics/glossary.py``, the same source the score panel
    reads, rather than being restated here where they could drift.

    Under the *default* weights, deliberately, and not under each game's own: this
    labels a menu, not a number. A score taken under older weights still sorts by
    whichever of these components it happens to carry, and shows nothing where it does
    not carry one. So a family or component a version drops leaves the menu the day that
    version becomes the default -- *elegance* and *rule yield* when `v11` does -- while
    `score_guide` still explains both on any older score that carries them.
    """
    try:
        glossary = explain(DEFAULT_WEIGHTS)
    except WeightsError:  # pragma: no cover - the shipped default always loads
        return {"version": DEFAULT_WEIGHTS, "families": {}, "components": {}}
    return {
        "version": glossary["version"],
        "families": {
            name: {"title": name, "means": entry["means"], "weight": entry["weight"]}
            for name, entry in glossary["families"].items()
        },
        "components": {
            name: {
                "title": entry.get("title", name.replace("_", " ")),
                "means": entry.get("what", ""),
                "reading": entry.get("reading", ""),
                "family": entry.get("family", ""),
                "weight": entry.get("weight", 0.0),
                # The site's title and *what* line where the glossary has one (B3), drawn
                # where the page is the site, as the score panel draws them.
                "site": {key: words for key, words in entry.get("site", {}).items()
                         if key in ("title", "what")},
            }
            for name, entry in glossary["components"].items()
        },
    }


def coverage(games: Iterable[Game]) -> dict[str, Any]:
    """Which kinds of game the library holds, in the grid's own vocabulary.

    Deliberately not a ``Grid``. ``loop.grid.build`` keeps one elite per cell out of
    *viable archive entries*, and a library holds everything -- committed, invented,
    scored, unscored, gated out. The question here is different: how much of the
    descriptor space the games on this machine actually cover, and which neighbourhoods
    are crowded. Same axes, same labels, no ranking.
    """
    occupied: dict[tuple[int, ...], list[str]] = {}
    for game in games:
        occupied.setdefault(descriptor(game.spec), []).append(game.name)
    return {
        "filled": len(occupied),
        "cells": CELLS,
        # The legend, once. A cell means nothing without it, and it is the same six
        # questions for every ruleset in the library.
        "axes": axis_vocabulary(),
        "occupied": [
            {
                "cell": label(cell),
                "bins": list(bins_of(cell)),
                "description": describe_cell(cell),
                "games": sorted(names, key=str.lower),
            }
            for cell, names in sorted(occupied.items())
        ],
    }


def provenance(game: Game) -> dict[str, Any]:
    """Where this ruleset came from, from whichever of the two sources has it.

    A published variant carries its history in the spec, because nothing else knows it.
    An evolved one carries it in the archive, because the spec was written by a model
    that must not be asked to describe its own ancestry -- it would make one up, and a
    fabricated lineage is worse than none.
    """
    declared = dict(game.spec.get("provenance") or {})
    if game.lineage:
        declared.setdefault("origin", "evolved")
        declared.update(game.lineage)
    declared.setdefault("origin", "unrecorded")
    return declared


# Where a game's own file is the library's, so its `provenance.added` is the day it joined.
# A design run's candidate and a member's draft are not in the library; a member's file
# could say anything there, and only the site's own publish day speaks for it.
LIBRARY_SOURCES = ("committed", "archived")


def added(game: Game) -> str | None:
    """The day `game` joined the library, `YYYY-MM-DD`, from its file's `provenance.added`
    (Q826; `.claude/scripts/backfill_added.py` wrote the day git first saw each file).
    `None` for a game the library's two directories do not hold."""
    if game.source not in LIBRARY_SOURCES:
        return None
    day = (game.spec.get("provenance") or {}).get("added")
    try:
        return date.fromisoformat(day).isoformat() if isinstance(day, str) else None
    except ValueError:
        return None


# Where a deliberate example may be kept: the two directories the maintainer commits to,
# and a game retired out of them, whose page the site draws from the rules its loader
# copied from those files (`site/kept.py`). A draft and a design run's candidate are not.
EXHIBIT_SOURCES = ("committed", "archived", "retired")


def exhibit(game: Game) -> str | None:
    """The sentence saying this game is kept on purpose for what it shows, or ``None``.

    The user's decision of 2026-09-09 keeps `archive/stones` exactly as it is, as a
    deliberate example of a game that cannot really be played, and obliges a label
    wherever its number is shown -- otherwise the number reads as a score. The sentence
    is the spec's own `provenance.exhibit`, outside the mechanics hash, so the label moves
    no game and no measurement, and a second deliberate example joins by saying so in its
    own file. Nothing here or on the page matches a name.

    Only a ruleset the repository keeps may say it. Provenance is free text a draft could
    carry too, and a game describing *itself* as a deliberate example would be excusing
    its own number; the designer and the operators already drop a parent's provenance, so
    a game forked from an exhibit is not one either.
    """
    if game.source not in EXHIBIT_SOURCES:
        return None
    said = (game.spec.get("provenance") or {}).get("exhibit")
    if not isinstance(said, str) or not said.strip():
        return None
    return said.strip()


def preview_sentence(spec: Mapping[str, Any]) -> str | None:
    """A spec's own `provenance.preview`, stripped, or ``None`` where it says none."""
    provenance = spec.get("provenance")
    said = provenance.get("preview") if isinstance(provenance, Mapping) else None
    if not isinstance(said, str) or not said.strip():
        return None
    return said.strip()


def preview(game: Game) -> str | None:
    """The sentence saying this game is shown as a preview, or ``None`` (Q931).

    The user, 2026-10-15: *"get the showcase games against computer up for testing on the
    site as soon as possible. label them testing/previews if necessary"*. A preview is a new
    kind of game still being tested: it is listed apart from the library under its own
    label and played as practice, and it stays out of every scored surface -- the library's
    listing and its bands, the landing page's shelves, the map, every neighbour and
    comparison, ratings and leaderboards. The sentence is the spec's own
    `provenance.preview`, outside the mechanics hash, so marking a game or taking the mark
    off moves no game and no measurement; taking it off is how a preview joins the library.

    Only a file in `archive/` may say it (`discover` files it as source `PREVIEW`): a
    Baseline game is what every score is measured against, and a draft or a design run's
    candidate is not the maintainer's to show. Being its own source rather than *archived*
    is what keeps a preview out of every list that names the library's two sources
    (`LIBRARY_SOURCES`, the map's, the site's `CORPUS_SOURCES`) without each being told.
    """
    if game.source != PREVIEW:
        return None
    return preview_sentence(game.spec)


def is_preview(game: Any) -> bool:
    """Whether `game` is filed as a preview. Read off `source` where the reader has one, so
    a stand-in carrying only what a comparison reads is never one."""
    return getattr(game, "source", None) == PREVIEW


def scored(games: Iterable[Game]) -> list[Game]:
    """`games` without the previews: the population every scored surface reads (Q931)."""
    return [game for game in games if not is_preview(game)]


def still(game: Game) -> dict[str, Any] | None:
    """A card's thumbnail: the game's opening position, shaped as `check`'s `preview`
    (`designing.opening_preview`), with each piece's drawing named by its stem rather
    than carried (V6, `docs/plan-visual.md` §3.3).

    The drawings are most of a preview's weight and nearly all of them are shared: forty
    games' previews are a megabyte with the drawings in each and thirty kilobytes
    without, beside about sixty distinct drawings. So the listing sends each drawing once
    (`piece_art`) and the page puts it back. None for a spec that does not compile: a
    listing must survive one unloadable file, and a card with no board still reads.
    """
    try:
        shape, _ = _still_cached(json.dumps(game.spec, sort_keys=True))
    except Exception:  # noqa: BLE001 - a listing must survive one unloadable spec
        return None
    return json.loads(shape)


def piece_art(game: Game) -> dict[str, str]:
    """The drawings `still(game)` names, stem to drawing; empty where it has none."""
    try:
        _, art = _still_cached(json.dumps(game.spec, sort_keys=True))
    except Exception:  # noqa: BLE001 - as `still`
        return {}
    return dict(art)


def opening(game: Game) -> dict[str, Any] | None:
    """The game page's start position (K1): `still(game)` with each drawing put back, so
    one cached compile serves the card and the page. None where `still` is None."""
    shape = still(game)
    if shape is None:
        return None
    art = piece_art(game)
    shape["icons"] = [
        [
            None if icon is None or icon["stem"] not in art
            else {"href": art[icon["stem"]], "rotate": icon["rotate"], "stem": icon["stem"]}
            for icon in pair
        ]
        for pair in shape.get("icons") or []
    ]
    return shape


def far_from_chess(game: Game) -> list[str]:
    """Why a game is far from chess, in a visitor's words, or nothing (K3, §4.2): no royal
    piece, won by arriving somewhere, or wild pieces that belong to neither side. Read off
    what the page already shows of it (the pieces' royal mark, the kind card's ending, the
    opening position's owners), so the offer to watch first says what the page says."""
    from chess_variant_sandbox.spec.describe import describe_pieces

    reasons: list[str] = []
    try:
        if not any(piece.royal for piece in describe_pieces(game.spec)):
            reasons.append("it has no royal piece to trap")
        ending = next((axis["bin"] for axis in describe_axes(descriptor(game.spec))
                       if axis["axis"] == "ending"), None)
        if ending == "arrival":
            reasons.append("it is won by getting a piece to a particular place")
        shape = still(game)
        if shape is not None:
            stride = shape.get("sideStride") or shape.get("nTypes") or 1
            types = shape.get("nTypes") or 1
            # A portal end (schema 19) stands in the third owner's block too, and is
            # nobody's piece: the board, not a wild piece (Q772d).
            ends = {end["type"] for end in (shape.get("portals") or {}).get("ends") or []}
            if any(value and (value - 1) // stride > 1 and (value - 1) % types not in ends
                   for value in shape["board"]):
                reasons.append("it has wild pieces that belong to neither side")
    except Exception:  # noqa: BLE001 - a page must survive one unloadable spec
        return []
    return reasons


# The game every other is compared with (`rulesets/chess.json`, the calibration baseline),
# found in the library by its file's name and read as data: its pieces' names are its
# spec's, and nothing here names one.
REFERENCE_FILE = "chess.json"


def reference_of(others: Iterable[Game] | None) -> Game | None:
    """The reference game among `others`, the committed one, or None where it is absent."""
    for game in others or ():
        if game.path.name == REFERENCE_FILE and game.source == "committed":
            return game
    return None


def piece_maps(game: Game, reference: Game | None = None) -> list[dict[str, Any]] | None:
    """Where each piece may go, alone on an empty board, one per piece in `pieces` order (K2,
    `docs/archive/plan-understanding.md` §4.2 step 2).

    Each map is `designing.move_map(alone=True)`, the designer's diagram: the engine's legal
    moves for that piece alone, and a takes probe per square. `like` is the reference game's
    piece whose map is the same (`same_moves`), by that game's own name for it, or None.
    Memoised on the two specs, so a library game's maps are worked out once and a game page
    posts nothing per tap. None where the spec does not compile: the page then draws no
    board under a piece, and says nothing it cannot back.
    """
    blob = json.dumps(game.spec, sort_keys=True)
    other = None if reference is None else json.dumps(reference.spec, sort_keys=True)
    try:
        return json.loads(_maps_cached(blob, other))
    except Exception:  # noqa: BLE001 - a page must survive one unloadable spec
        return None


@lru_cache(maxsize=256)
def _maps_alone(blob: str) -> str:
    """Every piece's map in one game, from one compile (`designing.alone_maps`).

    Nothing on disk can stand in for that compile: the tables cache keeps a fingerprint
    of the compiled tables, not the tables (`measurement.tables_fingerprint`), and
    `engine_for` keeps the last sixteen engines, which a listing over eighty games has
    long turned over. Asked a piece at a time through `move_map`, as before Q890, a game
    compiled once more per piece and the reference game once per game compared with it:
    661 compiles for the 87-game library where 88 do."""
    from chess_variant_sandbox.web import designing

    engine, found = designing.alone_maps(json.loads(blob))
    compiled = engine.spec
    maps = [{key: one[key] for key in ("from", "value", "moves", "takes")} for one in found]
    shape = {"width": compiled.width, "height": compiled.height,
             "comparable": compiled.lattice == "square" and compiled.wrap is None,
             "names": list(compiled.piece_names)}
    return json.dumps({"maps": maps, **shape})


@lru_cache(maxsize=256)
def _maps_cached(blob: str, reference: str | None) -> str:
    mine = json.loads(_maps_alone(blob))
    theirs = None if reference is None else json.loads(_maps_alone(reference))
    names: list[str] = [] if theirs is None else theirs["names"]
    out = []
    for found in mine["maps"]:
        like = None
        if theirs is not None and mine["comparable"] and theirs["comparable"]:
            like = next((names[at] for at, other in enumerate(theirs["maps"])
                         if same_moves(found, mine, other, theirs)), None)
        out.append({**found, "like": like})
    return json.dumps(out)


def _offsets(squares: Iterable[int], start: int, width: int) -> set[tuple[int, int]]:
    return {(to % width - start % width, to // width - start // width) for to in squares}


# How far, in every direction, both boards must reach from the pieces' squares before two
# answers are compared at all.
MIN_REACH = 3


def same_moves(
    one: dict[str, Any], one_board: dict[str, Any],
    two: dict[str, Any], two_board: dict[str, Any],
) -> bool:
    """Whether two engine answers about two pieces (`move_map`, each on its own board) are
    the same piece's: the same steps to move and the same to take, as offsets from where
    each stands. False where either board is too small to tell (`MIN_REACH`).

    Two boards differ in size, so a slider reaches further on the wider one. Inside the
    box both boards reach from their pieces' squares the offsets must be equal; outside it,
    a step either answer has must continue a line the box already holds whole in the other
    answer, with every step before it along that line in its own (a rook running on to
    file l on a twelve-wide board). Nothing here decides a move: both sides are the engine's
    answers, compared.
    """
    from math import gcd

    def reach(board: dict[str, Any], start: int) -> tuple[int, int, int, int]:
        x, y = start % board["width"], start // board["width"]
        return -x, board["width"] - 1 - x, -y, board["height"] - 1 - y

    a, b = reach(one_board, one["from"]), reach(two_board, two["from"])
    box = (max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3]))
    # A box too small to tell a slider from a short step says nothing: on a three-wide
    # board a one-step diagonal and a bishop give the same answer from the middle.
    if min(-box[0], box[1], -box[2], box[3]) < MIN_REACH:
        return False

    def inside(step: tuple[int, int]) -> bool:
        return box[0] <= step[0] <= box[1] and box[2] <= step[1] <= box[3]

    def runs_on(step: tuple[int, int], own: set[tuple[int, int]],
                other: set[tuple[int, int]]) -> bool:
        g = gcd(abs(step[0]), abs(step[1]))
        unit = (step[0] // g, step[1] // g)
        if any((unit[0] * k, unit[1] * k) not in own for k in range(1, g)):
            return False
        last = max((k for k in range(1, g) if inside((unit[0] * k, unit[1] * k))), default=0)
        if last == 0:
            return False
        # The other answer holds the line whole up to the edge of the box.
        if not all((unit[0] * k, unit[1] * k) in other for k in range(1, last + 1)):
            return False
        return not inside((unit[0] * (last + 1), unit[1] * (last + 1)))

    for key in ("moves", "takes"):
        mine = _offsets(one[key], one["from"], one_board["width"])
        theirs = _offsets(two[key], two["from"], two_board["width"])
        if {s for s in mine if inside(s)} != {s for s in theirs if inside(s)}:
            return False
        if not all(runs_on(s, mine, theirs) for s in mine if not inside(s)):
            return False
        if not all(runs_on(s, theirs, mine) for s in theirs if not inside(s)):
            return False
    return True


@lru_cache(maxsize=512)
def _still_cached(blob: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """One opening position per ruleset (`card_rules.still`), memoised on the spec as
    `_playability` is, and kept on disk between processes (`CardAnswers`)."""
    stored = _CARDS.get(_STILL, blob)
    if stored is not None:
        shape, art = stored
        return str(shape), tuple((str(stem), str(href)) for stem, href in art)
    spec = json.loads(blob)
    answer = card_rules.still(spec, lambda: engine_for(spec))
    _CARDS.put(_STILL, blob, [answer[0], [list(pair) for pair in answer[1]]])
    return answer


@lru_cache(maxsize=16)
def _engine_cached(blob: str) -> Engine:
    return Engine(json.loads(blob))


@lru_cache(maxsize=256)
def _smoke_cached(blob: str) -> tuple[bool, tuple[str, ...]]:
    """The cheap gate's verdict, played on the engine `engine_for` compiled
    (`card_rules.smoke`), and kept on disk between processes (`CardAnswers`) unless the
    gate's clock cut it, which is a fact about this machine's load and not the rules."""
    stored = _CARDS.get(_SMOKE, blob)
    if stored is not None:
        passed, reasons = stored
        return bool(passed), tuple(str(reason) for reason in reasons)
    spec = json.loads(blob)
    passed, said, timed_out = card_rules.smoke(spec, lambda: engine_for(spec))
    if not timed_out:
        _CARDS.put(_SMOKE, blob, [passed, list(said)])
    return passed, said


def _playability(spec: dict[str, Any]) -> dict[str, Any]:
    """Whether the cheap gate thinks this game can be played at all, and why not.

    Memoised on the spec, which is content-addressed, so the whole library costs about
    two seconds once and nothing afterwards. A gate that a listing had to re-run per
    request would be a listing nobody waits for -- and it is the *reason* that matters
    here, so caching the boolean alone would not have been enough.

    A spec the gate cannot even compile is reported as unplayable rather than raised:
    a listing that dies on one broken file shows nobody the other twenty-seven.

    A game played in rounds is checked as any game is: the gate plays it in rounds (Q932),
    where Q931 had to leave it unsaid while the gate refused every such game by rule.
    """
    try:
        passed, reasons = _smoke_cached(json.dumps(spec, sort_keys=True))
    except Exception:  # noqa: BLE001 - a listing must survive one unloadable spec
        return {"playable": None, "playableReasons": []}
    return {"playable": passed, "playableReasons": list(reasons)}


@lru_cache(maxsize=256)
def _fairy_cached(blob: str) -> tuple[bool, tuple[str, ...]]:
    """One translation attempt per ruleset, for both halves of the answer
    (`card_rules.fairy`), and kept on disk between processes (`CardAnswers`).

    A cold listing compiled every ruleset three times, for the cheap gate, the card's
    still board and this; the still board asks `engine_for` straight after this
    (`App.library_payload`), so the three share one compile (the first-visit study's N8:
    the first listing after a start took 13.8 s, nearly all of it compiling).
    """
    stored = _CARDS.get(_FAIRY, blob)
    if stored is not None:
        translates, reasons = stored
        return bool(translates), tuple(str(reason) for reason in reasons)
    answer = card_rules.fairy(lambda: engine_for(json.loads(blob)))
    _CARDS.put(_FAIRY, blob, [answer[0], list(answer[1])])
    return answer


# ---------------------------------------------------------------- the card answers' store
#
# Q887. A process's first listing worked out the three answers above for every game, and
# compiling the rules for them was nine tenths of it (`docs/reference/fast-suite-review.md`,
# Q887): 6.7 s of a 7.4 s payload over the 87-game library here, and with the cheap gate's
# playouts on one thread, most of the box's minute after a deploy. The answers are a pure
# function of the spec, the code `card_rules` can reach, the piece drawings and the core,
# so they are kept on disk under exactly that key and read back by the next process.

CARD_ANSWERS_NAME = "library-cards.json"
# Moves when the file's shape does, what a stored answer means, or how the key is read.
# 2 (Q890): files in byte order of their repository path, and no `__pycache__`, so the
# key is the same on Windows and on the box for the same tree.
_CARD_FORMAT = 2
# The module whose import closure is the key: everything an answer is worked out by.
CARD_ENTRY = "chess_variant_sandbox/web/card_rules.py"
_SMOKE, _FAIRY, _STILL = "smoke", "fairy", "still"

# `from chess_variant_sandbox.x import a, b` (or a parenthesised list over several lines)
# and `import chess_variant_sandbox.x`, one statement a line, deferred ones included.
# Anchored at a line's start: `measurement._closure`'s pattern let one import's names run
# on into the next line, which hid every second consecutive import from it (fixed in
# Q889, `measurement.package_imports`).
_FROM_IMPORT = re.compile(
    rb"^[ \t]*from[ \t]+(chess_variant_sandbox(?:\.\w+)*)[ \t]+import[ \t]+"
    rb"(\([^)]*\)|[^\r\n#;]*)",
    re.MULTILINE,
)
_PLAIN_IMPORT = re.compile(rb"^[ \t]*import[ \t]+([^\r\n#;]*)", re.MULTILINE)


def card_answers_path() -> Path:
    """Beside the tables cache (`measurement.tables_cache_path`), so the one variable that
    moves that moves this, and a test run never writes the checkout's `runs/`."""
    return tables_cache_path().with_name(CARD_ANSWERS_NAME)


def _module_path(dotted: str, listing: Callable[[str], frozenset[str]]) -> str | None:
    """The repository path of the source file a dotted package name is, or ``None`` for an
    attribute, in a tree whose directories `listing` names the files of.

    Spelled exactly, against the directory's own listing: on a case-insensitive disk
    `from chess_variant_sandbox.engine import Engine` would otherwise find `Engine.py`,
    the same file as `engine.py` under another name, and which name reached the key
    first would depend on the process's string hashing (it moved the key in 1 of 15
    processes here before this check)."""
    base = dotted.replace(".", "/")
    parent, _, stem = base.rpartition("/")
    for directory, name in ((parent, f"{stem}.py"), (base, "__init__.py")):
        if name in listing(directory):
            return f"{directory}/{name}" if directory else name
    return None


@lru_cache(maxsize=512)
def _module_file(dotted: str) -> Path | None:
    """`_module_path` on this checkout's disk, as a path."""
    found = _module_path(dotted, _disk_listing)
    return None if found is None else MEASUREMENT_ROOT / found


def _disk_listing(directory: str) -> frozenset[str]:
    return _listing(MEASUREMENT_ROOT / directory)


@lru_cache(maxsize=256)
def _listing(directory: Path) -> frozenset[str]:
    """The exact names of the files in one directory, or none for a missing one."""
    try:
        return frozenset(entry.name for entry in os.scandir(directory) if entry.is_file())
    except OSError:
        return frozenset()


def _package_imports(body: bytes) -> set[str]:
    """Every package module a source file's import statements name, with its parents."""
    named: list[str] = []
    for match in _FROM_IMPORT.finditer(body):
        module = match.group(1).decode("ascii")
        names = match.group(2).decode("utf-8", "replace").strip("() \t\r\n")
        named.append(module)
        for name in names.replace("\n", ",").split(","):
            word = name.split("#")[0].strip().split(" ")[0]
            if word.isidentifier():
                named.append(f"{module}.{word}")
    for match in _PLAIN_IMPORT.finditer(body):
        for name in match.group(1).decode("utf-8", "replace").split(","):
            word = name.strip().split(" ")[0]
            if word.startswith("chess_variant_sandbox"):
                named.append(word)
    found: set[str] = set()
    for dotted in named:
        parts = dotted.split(".")
        found.update(".".join(parts[:end]) for end in range(1, len(parts) + 1))
    return found


def _generated(name: str) -> bool:
    """A file Python leaves beside a script it ran, which no answer reads and no tree
    ships: the main checkout's `assets/pieces/__pycache__/` was in the key and the box's
    tree has none (Q890)."""
    return "__pycache__" in name.split("/") or name.endswith((".pyc", ".pyo"))


def card_files_in(
    read: Callable[[str], bytes | None],
    listing: Callable[[str], frozenset[str]],
    assets: Iterable[str],
) -> tuple[tuple[str, bytes], ...]:
    """`card_sources` and their bytes in any tree, by repository path (POSIX, relative).

    ``read`` gives a file's bytes or ``None`` when it cannot, ``listing`` the exact names
    of the files in a directory, ``assets`` every file under the piece drawings. This
    checkout's disk is one tree (`_card_files`); a git ref is another (`deploy/cards.py`,
    which asks what key the box will run before it is deployed).

    Sorted by path in byte order, never by `Path`: Windows compares paths case-blind, so
    the same tree's files came out in another order there than on the box, and the key
    moved between the two for the same bytes (Q890)."""
    found: dict[str, bytes] = {}
    queue = [CARD_ENTRY]
    while queue:
        name = queue.pop()
        if name in found:
            continue
        body = read(name)
        found[name] = b"<unreadable>" if body is None else body
        if body is None:
            continue
        for dotted in _package_imports(body):
            target = _module_path(dotted, listing)
            if target is not None and target not in found:
                queue.append(target)
    for name in assets:
        if name not in found and not _generated(name):
            body = read(name)
            found[name] = b"<unreadable>" if body is None else body
    return tuple(sorted(found.items()))


def _repo_name(path: Path) -> str:
    return (path.relative_to(MEASUREMENT_ROOT).as_posix()
            if path.is_relative_to(MEASUREMENT_ROOT) else path.as_posix())


def _disk_read(name: str) -> bytes | None:
    try:
        return (MEASUREMENT_ROOT / name).read_bytes()
    except OSError:
        return None


@lru_cache(maxsize=1)
def _card_files() -> tuple[tuple[str, bytes], ...]:
    """`card_sources` with each file's bytes, on this checkout's disk, each read once."""
    from chess_variant_sandbox.viz.pieces import ASSET_ROOT

    assets = [_repo_name(path) for path in ASSET_ROOT.rglob("*") if path.is_file()]
    return card_files_in(_disk_read, _disk_listing, assets)


def card_sources() -> tuple[Path, ...]:
    """Every file whose content can decide a card answer: `CARD_ENTRY`'s import closure,
    parent packages included (importing `a.b.c` runs `a/__init__.py` and `a/b/__init__.py`),
    and every file under the piece drawings' directory but what Python generates there.
    More files than the answers read only costs a recompute after an unrelated edit; fewer
    would serve a stale answer."""
    return tuple(MEASUREMENT_ROOT / name for name, _ in _card_files())


def card_key(core: str, files: Iterable[tuple[str, bytes]]) -> str:
    """SHA-256 over the format, a core build fingerprint and ``files`` by repository path
    and exact bytes: `card_answers_key` for any core and tree."""
    import hashlib

    digest = hashlib.sha256(f"library-cards {_CARD_FORMAT}\0core {core}\0".encode())
    for name, body in files:
        digest.update(name.encode("utf-8") + b"\0" + body + b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def card_answers_key() -> str:
    """The key the card answers are stored under in this process: `card_key` of the
    loaded core's build fingerprint and this checkout's `card_sources`. Cached for the
    process: the tree does not move under it."""
    return card_key(built_fingerprint(), _card_files())


class CardAnswers:
    """The three card answers per spec, read from disk once and written back in one go.

    The file holds one key's answers, ``{"sources": key, "cards": {spec digest: {kind:
    answer}}}``, and an answer stored under any other key is never read: a moved key
    means any of them may be wrong. A spec is named by the SHA-256 of the JSON text the
    in-process memos are keyed on, so its name and its drawings are in the name too.

    `save` writes only when something was added, to a temporary file renamed over the
    old one, after folding in what another process stored meanwhile under the same key.
    A `runs/` that cannot be written leaves the answers in memory. Every answer here is
    the code's own, so the worst a race can cost is a recompute.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._fixed = path
        self._lock = threading.Lock()
        self._read_as: tuple[Path, str] | None = None
        self._cards: dict[str, dict[str, Any]] = {}
        self._added = False

    def _path(self) -> Path:
        return self._fixed if self._fixed is not None else card_answers_path()

    @staticmethod
    def _stored(path: Path, key: str) -> dict[str, dict[str, Any]]:
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(held, dict) or held.get("sources") != key:
            return {}
        cards = held.get("cards")
        return {str(k): dict(v) for k, v in cards.items() if isinstance(v, dict)} if (
            isinstance(cards, dict)) else {}

    def _current(self) -> dict[str, dict[str, Any]]:
        """The answers for this path and key, read at most once per pair; under the lock."""
        place = (self._path(), card_answers_key())
        if self._read_as != place:
            self._cards = self._stored(*place)
            self._read_as = place
            self._added = False
        return self._cards

    @staticmethod
    def _name(blob: str) -> str:
        import hashlib

        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, kind: str, blob: str) -> Any | None:
        with self._lock:
            return self._current().get(self._name(blob), {}).get(kind)

    def put(self, kind: str, blob: str, answer: Any) -> None:
        with self._lock:
            self._current().setdefault(self._name(blob), {})[kind] = answer
            self._added = True

    def save(self) -> None:
        with self._lock:
            if not self._added or self._read_as is None:
                return
            path, key = self._read_as
            merged = self._stored(path, key)
            for name, answers in self._cards.items():
                merged[name] = {**merged.get(name, {}), **answers}
            self._cards = merged
            payload = json.dumps({"sources": key, "cards": dict(sorted(merged.items()))},
                                 sort_keys=True, separators=(",", ":"))
            temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(payload, encoding="utf-8")
                os.replace(temporary, path)
            except OSError:
                with contextlib.suppress(OSError):
                    temporary.unlink()
                return
            self._added = False


_CARDS = CardAnswers()


def save_card_answers() -> None:
    """Write the answers this process worked out since the last save, if any; best effort.
    `App.library_payload` calls it once a listing, so a cold listing writes the file once."""
    _CARDS.save()


def work_out_card_answers(games: Iterable[Game]) -> int:
    """The three answers a listing asks of each game's card, worked out or read back
    through the listing's own memos, then saved: the store a listing of these games
    leaves, without the rest of the listing. How a deploy fills the store it ships
    (`deploy/cards.py`, Q890). Returns how many games were asked about."""
    asked = 0
    for game in games:
        _playability(game.spec)
        _fairy(game.spec)
        still(game)
        asked += 1
    save_card_answers()
    return asked


def _fairy(spec: dict[str, Any]) -> dict[str, Any]:
    """Whether Fairy-Stockfish has a counterpart for these rules, and why not.

    The *cheap* half of that question, deliberately, and the key is named after the half
    it answers. ``fairy.supports`` says a variant definition can be written;
    ``fairy.playable`` says the two engines were played against each other and agreed,
    and only the second is enough to *offer* an opponent -- a translation that is nearly
    right is a legal-looking game under rules nobody wrote.

    A card cannot ask the second. It costs an engine start and a second or two of perft
    per ruleset, which is a minute of waiting to draw one listing, and it needs a binary
    that is discovered rather than shipped -- so a card reporting it would answer "no"
    for the whole library on a machine with no engine, which is a statement about a
    ``PATH`` rather than about a game.

    So this narrows and the game's own page confirms: ``opponents.fairy_stockfish_for``
    verifies the one ruleset you opened and remains the only thing that offers the
    player. A card saying the rules translate beside a picker offering nobody is the
    honest shape of "this build cannot play it" -- the card's tag says *translates* for
    that reason, the page says which build and which position from the same verdict
    (``opponents.fairy_check``), and the four things the released build cannot do are
    listed in ``docs/reference/fairy-stockfish.md``.

    Memoised on the spec, which is content-addressed, for the reason ``_playability`` is:
    translating compiles the ruleset, and a listing that paid that per request would be a
    listing nobody waits for.
    """
    from chess_variant_sandbox import fairy

    try:
        translates, reasons = _fairy_cached(json.dumps(spec, sort_keys=True))
    except Exception:  # noqa: BLE001 - a listing must survive one unloadable spec
        return {"fairyTranslates": None, "fairyReasons": []}
    # The page names a rule the way the designer does, never by its path in the game's
    # text (`fairy.words`); the `fairy` command keeps the translator's own spelling.
    return {"fairyTranslates": translates, "fairyReasons": fairy.plain_reasons(list(reasons))}


def engine_for(spec: dict[str, Any]) -> Engine:
    """A compiled engine for this spec, reused across requests.

    Compilation builds a table per (player, piece type, square) and costs milliseconds;
    a page that redraws a board on every click would pay it on every click.
    """
    return _engine_cached(json.dumps(spec, sort_keys=True))


def _lineage(entry: Any, by_hash: dict[str, Any], run: str) -> dict[str, Any]:
    """The chain of parents, walked back as far as the archive goes."""
    ancestors: list[dict[str, Any]] = []
    seen = {entry.hash}
    current = entry.parent
    while current and current in by_hash and current not in seen:
        seen.add(current)
        parent = by_hash[current]
        ancestors.append({"hash": parent.hash, "name": parent.name, "index": parent.index})
        current = parent.parent

    return {
        "run": run,
        "generation": entry.generation,
        "parents": ancestors,
        "theme": (entry.detail or {}).get("theme", ""),
        "notes": entry.notes,
    }


def current_only(score: dict[str, Any] | None) -> dict[str, Any] | None:
    """A score where it is current, else ``None``: the one door into a comparison.

    Read off the score's own `status`, so anything holding a score can ask, and a score
    that names no status (one built by hand, before EV2) is read as it always was.
    """
    if not score or score.get("status", CURRENT) != CURRENT:
        return None
    return score


def _index_detail(measurement: Measurement) -> dict[str, Any]:
    """The full scoring result behind a measurement: components, contributions, interval.

    Absent is not an error. A measurement is complete without it -- the headline, the
    families and the ladder are all in `canonical.json` -- and this is the working that
    the score view expands.
    """
    version = str(measurement.stored.get("weights_version", ""))
    path = measurement.directory / f"index-{version}.json"
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _measurements(
    runs_dir: Path,
    spec_by_hash: dict[str, dict[str, Any]] | None = None,
    paths: Iterable[Path] | None = None,
    seeds_by_hash: dict[str, tuple[int, ...]] | None = None,
    only_known: bool = False,
) -> dict[str, Measurement]:
    """The one canonical-roster measurement for each game, judged against now.

    This replaced two functions, and the pair of them was the defect. `_scores` picked
    an index file by newest weights version; `_ratings` picked a tournament by biggest
    roster. Two independent scans over two file types, with no constraint that they
    land in the same directory -- so a game with a fresh tournament and no fresh score
    rendered this morning's Elo ladder beside a year-old `0.0, gated`, and the page had
    no way to know they were different measurements of different code.

    One file, written by `corpus.measure`, holding score and ladder together with the
    basis both were taken under. A run directory that has only a raw `tournament.json`
    is *not* a measurement, and is deliberately not scavenged into one: what made the
    old behaviour dangerous was inferring a complete answer from partial evidence.

    ``spec_by_hash`` is what `discover` already read off disk -- committed, archived
    and lineage specs, keyed by their own hash. Passed into `corpus.read_measurement`
    as `spec=`, it lets that call skip `corpus.find`, which without a spec in hand
    re-globs and re-hashes every committed and archived spec to answer one lookup. A
    digest this map does not carry (a canonical.json with no spec anywhere this call
    visited) still falls back to `find`, unless ``only_known`` says the map is the
    whole of what the caller will read.

    ``paths`` is `discover`'s own walk of `runs/`, handed over so the tree is not walked
    a second time to find the same files. Absent, this globs for them as it always did.

    ``seeds_by_hash`` is how many runs each game's rung owes, from `_expected_seeds`:
    four for a library game, one for a candidate. A digest it does not carry is read
    against the library's four, `corpus.read_measurement`'s own default.

    **Each digest is resolved once, and with ``only_known`` only the ones the map
    carries** (Q874). The walk finds far more `canonical.json` files than games: every
    replaced score kept under `runs/<hash>/previous/<n>/` and every replay campaign's
    `replaced/` copy names its game's hash too, and `read_measurement` opens
    `runs/<hash>/` whichever file named it, so a second file for a digest is the same
    answer again. A digest no listed game carries -- a game that left the library or had
    its rules edited, with its history still on disk -- went to `corpus.find`, about 60 ms
    each, and `discover` then threw the answer away, since it reads a measurement only
    for a game it lists. In the main checkout on 2026-10-13 that was 1,778 files for 119
    digests, 322 of them for 34 such digests: 19 s of a 26 s listing.
    """
    from chess_variant_sandbox import corpus

    known = spec_by_hash or {}
    seeds_by_hash = seeds_by_hash or {}
    found: dict[str, Measurement] = {}
    seen: set[str] = set()
    if paths is None:
        paths = Path(runs_dir).rglob("canonical.json")
    for path in sorted(paths):
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        digest = stored.get("hash") if isinstance(stored, dict) else None
        if not isinstance(digest, str) or not digest or digest in seen:
            continue
        seen.add(digest)
        if only_known and digest not in known:
            continue
        measurement = corpus.read_measurement(
            digest, runs_dir, spec=known.get(digest),
            seeds=seeds_by_hash.get(digest),
        )
        if measurement is not None:
            found[digest] = measurement
    return found


def _verdicts(
    runs_dir: Path, paths: Iterable[Path] | None = None
) -> dict[str, dict[str, Any]]:
    """Every written verdict under ``runs/``, keyed by the ruleset it is about.

    Last one wins where a ruleset has been judged more than once -- unlike a score,
    there is nothing to compare two verdicts on, so "most recently written" is the only
    honest tie-break and pretending otherwise would invent a ranking out of prose. Two
    written in the same nanosecond fall back to the path, so the answer is the same
    twice running whatever order the filesystem offered them in.

    ``paths`` is `discover`'s walk, as in `_measurements`.
    """
    found: dict[str, dict[str, Any]] = {}
    if paths is None:
        paths = Path(runs_dir).rglob("verdict.json")
    ordered = sorted(sorted(paths), key=lambda p: _stamp(p)[1])
    for path in ordered:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        digest = data.get("ruleset_hash")
        if isinstance(digest, str):
            found[digest] = data
    return found
