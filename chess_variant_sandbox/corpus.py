"""What becomes of a game the design loop invented.

A run leaves its candidates in `runs/`, which is gitignored: the best game a loop ever
found is one `rm -rf runs/` from gone, and there was no code path by which it could
become anything else. Every reference to `rulesets/` in this project reads it. This
module is the three things you can do with a candidate instead, in the order of how much
they commit you:

* **test** -- measure it again on the roster a committed score actually means. A design
  run scores on four agents and three games a pairing because it must fit in a
  generation; the corpus is scored on six budgets and four games a pairing. Those are
  two different quantities and the whole project turns on not mixing them, so a
  candidate cannot be compared with `chess` until it has been measured the same way.
* **archive** -- keep it, in the repository, out of the corpus. `archive/` is committed
  and nothing in `metrics/` or `tests/` reads it, so a game can be kept, played and
  linked to without joining the thing the instruments are calibrated against.
* **promote** -- into `rulesets/`, which is not a trophy case. It is the calibration
  corpus: `tests/test_index_quality.py` holds the gates to accepting every file in it,
  `metrics/calibration.py` reads it as the human band, and the perft ladders are frozen
  against it. A game admitted there changes what the instruments say about every other
  game, which is why this is the only one of the three that demands a stated reason.

The ladder is the point. Testing is free of consequence, archiving is reversible, and
promotion is a claim about the corpus -- so the interface offers all three rather than
one button labelled "keep", and refuses to promote what has not been tested.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from chess_variant_sandbox import history
from chess_variant_sandbox.agents.nets import STORE_DIR, rated_net
from chess_variant_sandbox.loop.archive import Archive
from chess_variant_sandbox.loop.screen import ROUND_BUDGET_FACTOR
from chess_variant_sandbox.measurement import (
    Basis,
    Measurement,
    players_fingerprint,
    stale_build,
    tables_fingerprint,
    warm_fingerprints,
)
from chess_variant_sandbox.measurement import load as load_measurement
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS, load_weights, repeats, score_ruleset
from chess_variant_sandbox.metrics.components import plays_in_rounds, plays_with_spies
from chess_variant_sandbox.metrics.scoring import WeightsError
from chess_variant_sandbox.metrics.seat import seat_details
from chess_variant_sandbox.metrics.seeds import (
    VOTE_POOLED_SEAT,
    RunReading,
    combine,
    minimum_totals,
)
from chess_variant_sandbox.spec.canonical import load_spec, spec_hash
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.spec.spelling import current_spelling
from chess_variant_sandbox.tournament import TournamentConfig
from chess_variant_sandbox.tournament.analysis import CorpusAnalysis, analyse_corpus

# Q900: the runner's `run_tournament` outside a campaign, and resumable from its sealed
# pieces inside one (`--checkpoint DIR measure`, `checkpoint.active`). Until Q900 this was
# the runner's own, so a campaign given to `measure` was opened and ignored, and a
# measurement killed partway replayed every game (`docs/reference/campaign-checkpoints.md`).
from chess_variant_sandbox.tournament.checkpoint import run_tournament

RULESET_DIR = Path("rulesets")
ARCHIVE_DIR = Path("archive")
RUNS_DIR = Path("runs")

# The roster a committed score *means*, which until now existed only in prose. Every
# number in `docs/reference/metrics-and-calibration-record.md` is on this ladder and no other:
# six budgets rather than four, because with four the two halves `gradient_persistence`
# compares shared a point. `cli.DEFAULT_AGENTS` is a different, smaller roster for a
# one-off tournament, and `loop.DEFAULT_AGENTS` is smaller again because a design run
# spends it once per generation -- so a candidate measured by either is not comparable
# with the corpus, and saying that in code is the first half of what `test` is for.
#
# Every searching rung is `minimax:nodes=N` on the built-in objective, the same player
# definition on every board; `random` and `greedy` are the two rungs below search. Where
# a learned net is rated the searching rungs play it instead (`canonical_agents`, Q502h).
#
# **A spy game** (schema 27, Q935d; `docs/plan-spy.md` §3.3, §3.4) is measured on this
# roster too: `minimax` plays every phase of one (its blind rounds as `matrix:nodes=N`
# would), and a stored objective, a net, `matrix:` and `mcts:` are refused on one. Its
# budgets are these, not the round ladder's sixteen-fold ones: a spy game starts
# alternating, where these are the measured rungs, and Q930's finding that its round
# phases need 25,600 nodes to separate players is read beside the score (§3.4).
LADDER_BUDGETS = (100, 200, 400, 800, 1600, 6400)
CANONICAL_AGENTS = ("random", "greedy") + tuple(
    f"minimax:nodes={budget}" for budget in LADDER_BUDGETS
)
CANONICAL_GAMES = 4
CANONICAL_SEED = 0

# The roster a game played in rounds is measured on (schema 26, Q918d;
# `docs/archive/plan-simultaneous.md` §6.2): the same eight places, with the round players where
# the alternating ladder has minimax, which is refused on these games, at
# `ROUND_BUDGET_FACTOR` (16) times its budgets, `matrix:nodes=1600` to `102400` (Q944a,
# `docs/reference/round-roster-budgets.md` §6). At the same budgets the bottom rungs were
# greedy under other names: a round player always finishes greedy's whole one-round matrix
# and spends nothing more when that matrix holds its budget. `loop.screen.round_roster`
# applies the same factor, so one rule says what a budget means in rounds. The round
# players play the built-in objective only, and whether they need their own is read from
# this ladder's monotonicity (the plan's *Q918d, as built*). The stamp names the roster,
# so a round game's score is never read as though the ladder were the alternating one.
ROUND_AGENTS = ("random", "greedy") + tuple(
    f"matrix:nodes={budget * ROUND_BUDGET_FACTOR}" for budget in LADDER_BUDGETS
)

# Where the measuring roster reads a game's learned net from (Q502h). Relative, as
# `RUNS_DIR` is: every command runs from a checkout's root, and the box serves from its.
NETS_DIR = Path(STORE_DIR)


def canonical_agents(
    digest: str, nets_dir: Path | None = None, spec: dict[str, Any] | None = None
) -> tuple[str, ...]:
    """The roster this game's committed score means: `CANONICAL_AGENTS`, its net ladder,
    or for a game played in rounds `ROUND_AGENTS`.

    **A game in rounds** is known by its spec (``rules.simultaneous``), so a caller holding
    one passes it; without it the alternating roster is named, which matches no round
    game's stamp and so reads stale, the safe direction (`current_basis` says the same of
    the tables). **A spy game** (schema 27, ``pieces[].spy``) is known by its spec too, and
    is asked first: it declares ``rules.simultaneous`` but starts alternating, so it takes
    `CANONICAL_AGENTS` and never a net.

    **Where a learned net is rated** (`nets.rated_net`: kept, or accepted and confirmed
    where no generation line runs), every searching rung plays that net at the same
    budget, `minimax:net=<procedure>,nodes=N` for N of `LADDER_BUDGETS`, in place of the
    built-in objective; `random` and `greedy` are unchanged. The user's answers: the
    measuring players *"become smarter where they are trained"* (2026-09-28) and each kept
    generation joins the measuring set at once (2026-10-03, `docs/archive/decisions.md`).
    The whole ladder moves, not one rung, because every skill component asks whether
    thinking harder wins, and a ladder whose top rung alone priced the game differently
    would win for two reasons at once. The shape (eight players, the same budgets) and
    the weights version stay, so every score still sits in one leaderboard; which players
    measured it is in its `Basis.agents` and `Basis.net`.

    Read off `nets/` every call, so a newly kept generation stales its game's score the
    moment its store lands, and the refresher replays it (`docs/archive/plan-evolving-score.md` §6.4).
    """
    if spec is not None and plays_with_spies(spec):
        return CANONICAL_AGENTS
    if spec is not None and plays_in_rounds(spec):
        return ROUND_AGENTS
    procedure = rated_net(digest, NETS_DIR if nets_dir is None else nets_dir)
    if not procedure:
        return CANONICAL_AGENTS
    return ("random", "greedy") + tuple(
        f"minimax:net={procedure},nodes={budget}" for budget in LADDER_BUDGETS
    )


# The seeds a library game is measured on, and the seeds everything else is. A run is a
# tournament from one seed, and `docs/reference/run-to-run-spread.md` measured that four
# of 71 games change their verdict depending on which one -- so a library score is the
# mean of four runs and a visitor's design is one, and the file has the same shape either
# way. The spacing is the fresh campaign's: `runner.schedule` hands seeds out by position
# over 256 games, so bases 1000 apart share no game.
ONE_SEED = (0,)
LIBRARY_SEEDS = (0, 1000, 2000, 3000)

# What a reader expects of a committed or archived game. The four-run files were delivered
# by the campaign in `docs/reference/four-seeds-2026-09.md`, and this constant turned over
# with them. A game measured on fewer seeds than this reads stale by seeds and `measure`
# plays only the ones the file does not name.
EXPECTED_LIBRARY_SEEDS = LIBRARY_SEEDS

# Every seed set this project uses. A file naming anything else is stale by its seeds
# rather than judged on them.
KNOWN_SEED_SETS = (ONE_SEED, LIBRARY_SEEDS)

# How a game's runs decide together when they disagree, from `metrics/seeds.py`'s three.
# It lives here, outside every fingerprint set, and the basis stamps it by value -- so the
# answer to `docs/decisions.md`'s open question is a one-line edit that stales every
# multi-run file by name and costs a re-score. In force since 2026-09-19: a majority vote
# on every check but the seat, which is read once over the game's runs pooled against a
# tolerated share (`docs/reference/seat-effect-size.md` §13). A one-run measurement is
# unmoved by the name, and `measurement.py` stales a file by verdict rule only above one
# seed for that reason.
VERDICT_RULE = VOTE_POOLED_SEAT


def expected_seeds(stage: str) -> tuple[int, ...]:
    """How many runs a game at this rung is expected to have been measured on.

    A committed or archived game is the library and owes `EXPECTED_LIBRARY_SEEDS`.
    Everything else -- a candidate, a visitor's design -- is one run, because a number
    nobody has committed to does not owe four tournaments.
    """
    return EXPECTED_LIBRARY_SEEDS if stage in ("committed", "archived") else ONE_SEED


class CorpusError(ValueError):
    """A refusal with a reason a person can act on.

    A `ValueError` so that `cli.main`'s existing handler prints it as one line rather
    than a stack trace: every refusal here is a thing the person asked for that cannot
    be done yet, and it names what to do instead.
    """


@dataclass(frozen=True)
class Candidate:
    """One game, wherever it currently lives, and how far along the ladder it is."""

    hash: str
    name: str
    spec: dict[str, Any]
    path: Path
    # committed | archived | candidate -- where it is now, not how good it is.
    stage: str
    # The design run it came out of, when it came out of one.
    run: Path | None = None
    lineage: dict[str, Any] = field(default_factory=dict)

    @property
    def tested(self) -> bool:
        """Whether a *current* canonical-roster score exists.

        A stale one does not count and must not. The measurement is still readable --
        `measurement()` returns it, staleness attached, because "how badly out of date"
        is the next question -- but every decision that turns on having been tested
        treats an out-of-date number as no number at all.
        """
        measurement = self.measurement()
        return measurement is not None and measurement.current

    def measurement(self, runs_dir: Path = RUNS_DIR) -> Measurement | None:
        """This game's canonical-roster score, if one has been taken, stale or not."""
        return read_measurement(
            self.hash, runs_dir, spec=self.spec, seeds=expected_seeds(self.stage)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "name": self.name,
            "stage": self.stage,
            "path": str(self.path),
            "run": str(self.run) if self.run else None,
            "lineage": dict(self.lineage),
            "tested": self.tested,
        }


# ------------------------------------------------------------------ finding a game


def find(
    digest: str,
    runs_dir: Path = RUNS_DIR,
    ruleset_dir: Path = RULESET_DIR,
    archive_dir: Path = ARCHIVE_DIR,
) -> Candidate:
    """The game with this content hash, from wherever it is.

    Searched in ladder order so a promoted game reports as committed rather than as the
    candidate it also still is: a design run's `specs/` directory keeps its copy, and
    two answers to "what is this" would make the interface offer to promote a game that
    is already in the corpus.
    """
    for directory, stage in ((ruleset_dir, "committed"), (archive_dir, "archived")):
        path = _committed_paths(directory).get(digest)
        if path is not None:
            spec = load_spec(path)
            return Candidate(
                hash=digest, name=str(spec.get("name", path.stem)), spec=spec,
                path=path, stage=stage,
            )

    for lineage in sorted(Path(runs_dir).glob("*/lineage.jsonl")):
        archive = Archive(lineage.parent)
        archived = archive.load_spec(digest)
        if archived is None:
            continue
        entry = next((e for e in archive.entries if e.hash == digest), None)
        return Candidate(
            hash=digest,
            name=str(archived.get("name", digest)),
            spec=archived,
            path=lineage.parent / "specs" / f"{digest}.json",
            stage="candidate",
            run=lineage.parent,
            lineage=_lineage_of(entry, archive),
        )

    raise CorpusError(
        f"no game with hash {digest} under {ruleset_dir}/, {archive_dir}/ or {runs_dir}/"
    )


def from_file(path: Path, runs_dir: Path = RUNS_DIR) -> Candidate:
    """The game a spec file holds: `find`'s answer where the corpus, the archive or a design
    run holds it, else a candidate read from the file itself.

    The second case is a game nobody has put on the ladder yet -- a test spec, a game being
    written by hand -- measured where it stands (Q918d's round game was the first). It is a
    candidate, so it owes one run, and its measurement lives under ``runs_dir`` as any
    other's does.
    """
    spec = load_spec(path)
    digest = spec_hash(spec)
    try:
        return find(digest, runs_dir=runs_dir)
    except CorpusError:
        pass
    errors = validate_spec(spec)
    if errors:
        raise CorpusError(f"{path} is not a valid game: {'; '.join(errors)}")
    return Candidate(
        hash=digest, name=str(spec.get("name", Path(path).stem)), spec=spec,
        path=Path(path), stage="candidate",
    )


def _committed_paths(directory: Path) -> dict[str, Path]:
    """Content hash to file, for a directory of specs. Not cached: files arrive."""
    found: dict[str, Path] = {}
    if not Path(directory).is_dir():
        return found
    for path in sorted(Path(directory).glob("*.json")):
        try:
            spec = load_spec(path)
        except (OSError, ValueError):
            continue
        if isinstance(spec, dict):
            found.setdefault(spec_hash(spec), path)
    return found


def committed_candidates(
    ruleset_dir: Path = RULESET_DIR, archive_dir: Path = ARCHIVE_DIR
) -> dict[str, Candidate]:
    """Every committed and archived game by hash, each directory read once.

    What `find` returns for each, without re-reading both directories per game: the
    refresher's plan asks it once for every replay it routes.
    """
    found: dict[str, Candidate] = {}
    for directory, stage in ((ruleset_dir, "committed"), (archive_dir, "archived")):
        for digest, path in _committed_paths(Path(directory)).items():
            if digest not in found:
                spec = load_spec(path)
                found[digest] = Candidate(
                    hash=digest, name=str(spec.get("name", path.stem)), spec=spec,
                    path=path, stage=stage,
                )
    return found


def _lineage_of(entry: Any, archive: Archive) -> dict[str, Any]:
    """What the archive knows about where this candidate came from.

    Read from the archive rather than from the spec, and the distinction is the same one
    `web/library.provenance` makes: the spec was written by a model, and a model asked
    to describe its own ancestry would invent one.
    """
    if entry is None:
        return {}
    parent = next((e for e in archive.entries if e.hash == entry.parent), None)
    return {
        "generation": entry.generation,
        "parent": entry.parent,
        "parent_name": parent.name if parent else None,
        "theme": (entry.detail or {}).get("theme") or "",
        "operator": (entry.detail or {}).get("operator") or "",
        "index": entry.index,
        "weights_version": entry.weights_version,
    }


# ------------------------------------------------------------------------- the test


def measurement_dir(digest: str, runs_dir: Path = RUNS_DIR) -> Path:
    """Where this game's one measurement lives: `runs/<hash>`.

    A directory of its own, holding `canonical.json` and the mean's own
    `index-<weights>.json` and no games at all. Two readers join on the measurement's
    directory and open the index file beside the file -- `web/library._index_detail` and
    `site/load._range` -- so in a directory of its own that file is the *mean's*, and both
    show the right components and the right interval with no edit. Widening the seed-0
    directory's file instead would have left an unmigrated reader showing a mean headline
    over one run's components, with nothing failing.
    """
    return Path(runs_dir) / digest


def run_dir(digest: str, seed: int = CANONICAL_SEED, runs_dir: Path = RUNS_DIR) -> Path:
    """Where one run of this game lives: `runs/<hash>-<seed>`.

    Games, the tournament's own record, the analysis pass's file and that run's index
    file. **No stamp of its own**: every claim about a run is in the game's one
    measurement, which `_store` stamps once for all of them. Co-location is trusted here
    as it always has been, and determinism is what makes that safe -- a replay under the
    same basis reproduces the same bytes, and under a different one the game's file is
    already stale.
    """
    return Path(runs_dir) / f"{digest}-{seed}"


def current_basis(
    digest: str,
    weights: str = DEFAULT_WEIGHTS,
    spec: dict[str, Any] | None = None,
    seeds: tuple[int, ...] | None = None,
) -> Basis:
    """What a measurement of this game taken right now would rest on.

    The one place the canonical roster becomes a `Basis`. Everything that reads or
    refuses a stored score compares against this, so "what counts as current" has a
    single definition rather than one per caller.

    `spec` is the game itself, needed for the tables fingerprint and nothing else. It is
    optional because a caller can hold a hash and no spec, and **absent means the tables
    cannot be compared**: the fingerprint is `""`, which matches no stored one, so the
    measurement reads stale. That is the safe direction and it is the same rule an
    absent `basis` already follows -- refusing to guess is the whole mechanism.

    `seeds` is how many runs the reader expects, from `expected_seeds`. It defaults to
    `EXPECTED_LIBRARY_SEEDS` rather than to one seed, so the knob that turns the library
    over to four runs is one constant and not a default repeated at every call site.

    No spec is looked up here. `read_measurement` is where a bare hash is resolved into
    a spec, because that is where the directories to search in are already arguments.

    **The roster is this game's** (`canonical_agents`, Q502h), read from `nets/`.
    """
    agents = canonical_agents(digest, spec=spec)
    return Basis.current(
        ruleset_hash=digest,
        weights_version=weights,
        agents=agents,
        games_per_pairing=CANONICAL_GAMES,
        seeds=EXPECTED_LIBRARY_SEEDS if seeds is None else tuple(seeds),
        tables=tables_fingerprint(spec) if spec is not None else "",
        verdict_rule=VERDICT_RULE,
        known_seed_sets=KNOWN_SEED_SETS,
        # What the runner would hand the core (Q631); `""`, *cannot say*, without a spec.
        players=players_fingerprint(spec, agents) if spec is not None else "",
    )


def read_measurement(
    digest: str,
    runs_dir: Path = RUNS_DIR,
    weights: str = DEFAULT_WEIGHTS,
    spec: dict[str, Any] | None = None,
    seeds: tuple[int, ...] | None = None,
) -> Measurement | None:
    """The stored canonical-roster score for this game, with its staleness attached.

    Returns a stale measurement rather than hiding it. That is the change: the roster
    used to be checked here and a mismatch returned `None`, which told a caller "no
    score" when the truth was "a score about something else". A reader that wants to
    show a number asks `.current`; one that wants to explain the gap reads
    `.staleness.summary()`. Neither can get a stale number by accident, because the
    staleness travels in the same object.

    **`runs/<hash>/canonical.json`, and `runs/<hash>-0/canonical.json` only as a
    fallback.** A file at the old path is always stale and never yields a number: it was
    written before a game could have more than one run, and the index file beside it is
    seed 0's own rather than the measurement's. The old directories stay on disk with
    their games and their older weights anchors, and this is the one path that still
    opens one.
    """
    path = measurement_dir(digest, runs_dir) / "canonical.json"
    old_path = not path.exists()
    if old_path:
        path = run_dir(digest, CANONICAL_SEED, runs_dir) / "canonical.json"
    if spec is None:
        # Found rather than guessed. A reader that hands over only a hash still gets a
        # tables comparison, at the cost of the lookup -- and a hash with no spec behind
        # it in this checkout gets `""`, which reads stale, which is right: nothing here
        # can say what that game compiles to.
        try:
            spec = find(digest, runs_dir=runs_dir).spec
        except CorpusError:
            spec = None
    return load_measurement(
        path, current_basis(digest, weights, spec, seeds), old_path=old_path
    )


# --------------------------------------------------------------- what needs measuring


@dataclass(frozen=True)
class ScoringStatus:
    """One game's standing in the scoring queue: what it has, and what it would cost.

    The unit both readers of "what is out of date" deal in -- `measure --check` and the
    web interface's scoring panel. They used to be two loops that happened to agree;
    they are one now, because the whole point of a staleness check is that it is the
    same answer wherever it is asked.

    ``needs`` is `measurement.Staleness.needs` widened by one case: a game nobody has
    ever measured needs a `replay` too, but "never measured" and "measured under other
    code" are different news, so `state` keeps them apart while `needs` says what to run.
    """

    hash: str
    name: str
    stage: str
    state: str  # current | stale | absent
    needs: str  # "" | rescore | reanalyse | replay
    reason: str
    index: float | None = None
    sd: float | None = None
    viable: bool | None = None
    weights_version: str = ""
    # How many runs the stored file names. Travels with every number, because a mean of
    # four runs and a single run are not the same claim and nothing that shows one should
    # be able to show it without saying which.
    runs: int = 0
    # Where the score is current because a sample re-certified it (C6), what the stamp
    # says of it (`Certificate.words`); empty for a score its games were played for.
    certified: str = ""

    @property
    def current(self) -> bool:
        return self.state == "current"

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "name": self.name,
            "stage": self.stage,
            "state": self.state,
            "needs": self.needs,
            "reason": self.reason,
            "index": self.index,
            "sd": self.sd,
            "viable": self.viable,
            "weights_version": self.weights_version,
            "runs": self.runs,
            "certified": self.certified,
        }


def corpus_hashes(
    ruleset_dir: Path = RULESET_DIR, archive_dir: Path = ARCHIVE_DIR
) -> list[str]:
    """Every game a canonical measurement is *expected* for, committed first.

    The corpus and the archive, and nothing under `runs/`: a design run's candidates are
    measured one at a time by the person looking at one, and sweeping them would spend
    hours on games nobody has decided to keep.
    """
    digests: list[str] = []
    for directory in (ruleset_dir, archive_dir):
        for digest in _committed_paths(Path(directory)):
            if digest not in digests:
                digests.append(digest)
    return digests


def status_of(
    digest: str,
    runs_dir: Path = RUNS_DIR,
    ruleset_dir: Path = RULESET_DIR,
    archive_dir: Path = ARCHIVE_DIR,
    weights: str = DEFAULT_WEIGHTS,
    known: tuple[str, str, dict[str, Any]] | None = None,
) -> ScoringStatus:
    """Whether this game's measurement still holds, without spending anything to find out.

    ``known`` is the game's name, stage and spec where the caller already has them.
    `find` re-globs both spec directories and hashes every file in them, which is nothing
    once and a second and a half when a page asks it forty-two times. The spec joined
    that tuple when the basis grew a tables fingerprint: it is the one thing that has to
    be *compiled* rather than read, so passing it saves the lookup and the cache in
    `measurement.tables_fingerprint` saves the compile.
    """
    if known is None:
        candidate = find(
            digest, runs_dir=runs_dir, ruleset_dir=ruleset_dir, archive_dir=archive_dir
        )
        known = (candidate.name, candidate.stage, candidate.spec)
    name, stage, spec = known
    stored = read_measurement(digest, runs_dir, weights, spec, expected_seeds(stage))
    if stored is None:
        return ScoringStatus(
            hash=digest, name=name, stage=stage,
            state="absent", needs="replay", reason="never measured",
        )
    numbers = dict(stored.stored)
    return ScoringStatus(
        hash=digest,
        name=name,
        stage=stage,
        state="current" if stored.current else "stale",
        needs=stored.staleness.needs,
        reason=stored.staleness.summary(),
        # The number itself only travels when it is still true. A stale measurement is
        # not a number -- `measurement.py` says so in bold -- and a panel that offered
        # one beside the reason it is stale would be the same bug in a smaller box.
        index=numbers.get("index") if stored.current else None,
        sd=numbers.get("sd") if stored.current else None,
        viable=numbers.get("viable") if stored.current else None,
        weights_version=str(numbers.get("weights_version", "")),
        runs=len(numbers.get("runs") or []),
        certified=(
            "re-" + stored.basis.certified.words()
            if stored.basis is not None and stored.basis.certified is not None else ""
        ),
    )


def survey(
    digests: list[str] | None = None,
    runs_dir: Path = RUNS_DIR,
    ruleset_dir: Path = RULESET_DIR,
    archive_dir: Path = ARCHIVE_DIR,
    weights: str = DEFAULT_WEIGHTS,
) -> list[ScoringStatus]:
    """The state of every measurement asked about, in the order asked.

    Games this checkout cannot find are dropped rather than reported as stale: a hash
    with no spec behind it is not an out-of-date measurement, it is a missing game, and
    nothing a re-score would fix.
    """
    if digests is None:
        digests = corpus_hashes(ruleset_dir, archive_dir)

    # Both spec directories read once, rather than once per game inside `find`.
    known: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for directory, stage in ((ruleset_dir, "committed"), (archive_dir, "archived")):
        for digest, path in _committed_paths(Path(directory)).items():
            if digest not in known:
                spec = load_spec(path)
                known[digest] = (str(spec.get("name", path.stem)), stage, spec)

    # A cold tables cache is filled by a process pool before the games are read one by one
    # (Q904); it changes no answer, only when each is computed.
    warm_fingerprints([known[d][2] for d in digests if d in known])
    found: list[ScoringStatus] = []
    for digest in digests:
        try:
            found.append(
                status_of(
                    digest, runs_dir, ruleset_dir, archive_dir, weights,
                    known=known.get(digest),
                )
            )
        except CorpusError:
            continue
    return found


def measure(
    candidate: Candidate,
    runs_dir: Path = RUNS_DIR,
    weights: str = DEFAULT_WEIGHTS,
    workers: int = 0,
    report: Any = None,
    seeds: tuple[int, ...] | None = None,
) -> Measurement:
    """Play this game on the canonical roster, analyse it, score it, and keep all three.

    The same three passes a corpus entry gets, in the same order and at the same
    settings, because the point of the exercise is a number that can sit in a column
    next to `chess`. It costs minutes rather than the seconds a design run spends.

    Unconditional: this always replays, once for each seed. `refresh` is the entry point
    that decides how much work a measurement actually needs, and it calls this only when
    the answer is "all of it".

    `seeds` defaults to what a reader expects of this game -- one run for a candidate,
    `EXPECTED_LIBRARY_SEEDS` for a committed or archived one. Each run's games land in
    its own directory and the game's measurement is written once over all of them, in the
    same shape at one run and at four, so a visitor's score is the case n = 1 rather than
    a second kind of file.
    """
    chosen = tuple(sorted(seeds)) if seeds else expected_seeds(candidate.stage)
    return _measure_seeds(
        candidate, runs_dir, weights, workers, report,
        {seed: "replay" for seed in chosen},
    )


def _measure_seeds(
    candidate: Candidate,
    runs_dir: Path,
    weights: str,
    workers: int,
    report: Any,
    work: dict[int, str],
    totals: dict[int, float | None] | None = None,
    adopt_from: Path | None = None,
    carry: Basis | None = None,
) -> Measurement:
    """Bring every run of one game to the state `work` asks for, then write its file.

    The seeds in ascending order, then one write.

    ``work`` maps each seed to what it owes -- ``replay``, ``reanalyse``, ``rescore`` or
    ``""`` for a run that is read as it stands. ``totals`` is what the stored measurement
    said each run scored, which is how a run read as it stands is checked against the file
    that named it.

    **A certificate is carried, never dropped short of a full replay** (C6). Where the
    stored stamp was re-certified from a sample and some run it names is not played again
    here, the new stamp keeps what played the games and the certificate (``carry``, or the
    stored stamp's own), so a re-score never turns a sampled basis into a played one. A
    measurement that replays every run it holds writes none: that is what clears the label.
    """
    say = report if callable(report) else (lambda _message: None)
    if not work:
        raise CorpusError("a measurement covers at least one seed")

    plays = sorted(seed for seed, need in work.items() if need == "replay")
    passes = [seed for seed, need in work.items() if need == "reanalyse"]

    # Before minutes of tournament, not after. A core built from older sources does not
    # fail -- it plays, with the previous build's rules -- so the measurement would be
    # taken, written, and stamped with a basis describing source it was not built from.
    if plays or passes or adopt_from is not None:
        behind = stale_build()
        if behind is not None:
            raise CorpusError(behind)

    # The score's history (Q649, `docs/archive/plan-confidence.md` §3.2 step 3). Before any
    # run is touched: a measurement under another basis is kept whole under `previous/<n>/`, its
    # runs that are played again moved and the rest copied. Its ledger line is written
    # once the new file is, so a measurement that fails half way leaves no line.
    outgoing = read_measurement(
        candidate.hash, runs_dir, weights, candidate.spec, tuple(sorted(work))
    )
    kept: int | None = None
    measured_on: str | None = None
    if outgoing is not None and not outgoing.current:
        canonical = outgoing.directory / "canonical.json"
        measured_on = history.file_date(canonical)
        held = (
            outgoing.basis.seeds if outgoing.basis is not None
            else tuple(int(run["seed"]) for run in outgoing.stored.get("runs") or []
                       if isinstance(run, dict) and run.get("seed") is not None)
        )
        kept = history.keep(
            candidate.hash, runs_dir, canonical, held,
            moving={seed for seed, need in work.items() if need == "replay"},
        )
        say(f"  the measurement it replaces is kept as {history.PREVIOUS_DIR}/{kept}")

    if carry is None and outgoing is not None and outgoing.basis is not None and (
        outgoing.basis.certified is not None
        and not set(outgoing.basis.seeds) <= {seed for seed, need in work.items()
                                              if need == "replay"}
    ):
        carry = outgoing.basis

    origins: dict[int, str] = {}
    if adopt_from is not None:
        for seed in list(plays):
            origin = _adopt_seed(candidate, seed, adopt_from, runs_dir, workers, say)
            if origin:
                origins[seed] = origin
                work[seed] = "rescore"
                plays.remove(seed)

    weighting = load_weights(weights)
    readings = [
        _run_seed(
            candidate, seed, work[seed], runs_dir, weighting, workers, say,
            origin=origins.get(seed, "played"),
            total=(totals or {}).get(seed),
        )
        for seed in sorted(work)
    ]
    measurement = _store(candidate, runs_dir, readings, weighting, say, carry=carry)
    if kept is not None and outgoing is not None and measurement.basis is not None:
        incoming = {**measurement.stored, "basis": measurement.basis.as_dict()}
        # §3.2 step 4 (Q650): what the replacement says about the players, if anything.
        said = judge_convergence(
            candidate.hash, runs_dir, kept, outgoing.stored, incoming,
            candidate=candidate, now=measurement.basis, workers=workers,
        )
        say(f"  convergence: {said}")
        history.append(candidate.hash, runs_dir, history.entry_for(
            kept, outgoing.stored, incoming["basis"], measured_on,
            outgoing.staleness.needs, outgoing.staleness.reasons, said,
        ))
    return measurement


def rescore_kept(
    candidate: Candidate,
    kept: Path,
    seeds: Any,
    weights: str = DEFAULT_WEIGHTS,
    reanalyse: bool = False,
    workers: int = 0,
) -> list[float | None]:
    """Each kept run's total under the current metrics, from its stored games (Q650).

    The runs a replaced measurement left under `previous/<n>/<hash>-<seed>/`, scored by
    this checkout's scorer at ``weights`` -- and analysed again first when ``reanalyse``
    says the pass moved -- so a convergence read compares two measurements scored by one
    code. **Writes nothing**: the kept runs are a record of what was, and an index file
    written beside them would make them a mixture. A run whose scorer refuses is ``None``,
    as `combine` withholds it.
    """
    from chess_variant_sandbox.tournament import rate, read_records

    weighting = load_weights(weights)
    totals: list[float | None] = []
    for seed in sorted(int(seed) for seed in seeds):
        out = Path(kept) / f"{candidate.hash}-{seed}"
        records = list(read_records(out / "games.jsonl"))
        if not records:
            raise CorpusError(f"{out} has no games to re-score")
        analysis: CorpusAnalysis | None = None
        if not reanalyse and (out / "analysis.json").exists():
            analysis = CorpusAnalysis.load(out / "analysis.json")
        if analysis is None:
            behind = stale_build()
            if behind is not None:
                raise CorpusError(behind)
            analysis = analyse_corpus(
                candidate.spec, records, seed=CANONICAL_SEED, workers=workers
            )
        try:
            result = score_ruleset(
                candidate.spec, records, rate(records), weights=weighting, analysis=analysis
            )
        except WeightsError:
            totals.append(None)
            continue
        totals.append(float(result.headline) if result.scored else None)
    return totals


def judge_convergence(
    digest: str,
    runs_dir: Path,
    number: int,
    outgoing: Mapping[str, Any],
    incoming: Mapping[str, Any],
    candidate: Candidate | None = None,
    now: Basis | None = None,
    workers: int = 0,
) -> str:
    """`convergence.judge` for one replacement, with this checkout's re-score of the kept
    runs; the word for the history line. Never raises: a read that cannot be made is
    *no verdict*, and the measurement it follows stands."""
    from chess_variant_sandbox import convergence
    from chess_variant_sandbox.measurement import _moved

    def rescore(kept: Path, seeds: Any, weights: str, reanalyse: bool) -> list[float | None]:
        game = candidate if candidate is not None else find(digest, runs_dir=runs_dir)
        return rescore_kept(game, kept, seeds, weights, reanalyse, workers)

    moved = None
    if now is not None:
        old = outgoing.get("basis") or {}
        bytes_now = {"engine": now.engine_bytes, "metrics": now.metrics_bytes,
                     "analysis": now.analysis_bytes}
        stamp_now = {"engine": now.engine, "metrics": now.metrics, "analysis": now.analysis}

        def moved(name: str) -> bool:
            return _moved(str(old.get(name) or ""), stamp_now[name], bytes_now[name])

    try:
        return convergence.judge(
            digest, runs_dir, number, outgoing, incoming, rescore=rescore, moved=moved
        ).said
    except (CorpusError, OSError, ValueError, KeyError, WeightsError) as error:
        return f"{convergence.NO_VERDICT}: {error}"


def _adopt_seed(
    candidate: Candidate,
    seed: int,
    adopt_from: Path,
    runs_dir: Path,
    workers: int,
    say: Any,
) -> str:
    """Take a campaign's run for this seed instead of playing it, or say why not.

    Returns what `runs[].origin` should record, or ``""`` when the run is played after
    all. A refusal is never fatal: the seed keeps its `replay`, so the fallback for every
    failure adoption can have is the work that was going to happen anyway.
    """
    from chess_variant_sandbox.tournament import adopt

    source = adopt.campaign_run(adopt_from, candidate.hash, seed)
    if source is None:
        say(f"  seed {seed}: {adopt_from} holds no run to adopt; it is played")
        return ""

    taken = adopt.adopt_run(
        candidate.spec,
        seed,
        source,
        run_dir(candidate.hash, seed, runs_dir),
        TournamentConfig(
            agents=canonical_agents(candidate.hash, spec=candidate.spec),
            games_per_pairing=CANONICAL_GAMES,
            seed=seed,
            workers=workers,
        ),
        workers=workers,
        say=say,
    )
    if not taken.adopted:
        say(f"  seed {seed}: not adopted -- {taken.reason}; it is played")
        return ""
    say(f"  seed {seed}: {taken.origin}")
    return taken.origin


def _run_seed(
    candidate: Candidate,
    seed: int,
    need: str,
    runs_dir: Path,
    weighting: Any,
    workers: int,
    say: Any,
    origin: str = "played",
    total: float | None = None,
) -> RunReading:
    """One run, brought to the state it owes, as a reading for the combiner."""
    out = run_dir(candidate.hash, seed, runs_dir)

    if need != "replay" and not (out / "games.jsonl").exists():
        say(f"  seed {seed}: its games are not on disk, so it is played again")
        need = "replay"

    if need == "replay":
        return _play_seed(candidate, seed, out, weighting, workers, say, origin)

    if not need:
        reading = _read_seed(candidate, seed, out, weighting, total, say, origin)
        if reading is not None:
            return reading
        need = "rescore"

    return _rescore_seed(
        candidate, seed, out, weighting, workers, say,
        reanalyse=need == "reanalyse", origin=origin,
    )


def _play_seed(
    candidate: Candidate,
    seed: int,
    out: Path,
    weighting: Any,
    workers: int,
    say: Any,
    origin: str,
) -> RunReading:
    """Play, analyse and score one seed into its own run directory."""
    agents = canonical_agents(candidate.hash, spec=candidate.spec)
    refusal = _foreign_run(out, seed, agents)
    if refusal:
        raise CorpusError(refusal)

    say(
        f"seed {seed}: {len(agents)} agents, "
        f"{CANONICAL_GAMES} games a pairing"
    )
    tournament = run_tournament(
        candidate.spec,
        TournamentConfig(
            agents=agents,
            games_per_pairing=CANONICAL_GAMES,
            seed=seed,
            workers=workers,
        ),
    )
    tournament.save(out)

    say(f"  {len(tournament.records)} games; the analysis pass")
    analysis = analyse_corpus(
        candidate.spec, tournament.records, seed=CANONICAL_SEED, workers=workers
    )
    analysis.save(out / "analysis.json")

    say("  scoring")
    return _score_run(
        candidate, seed, out, tournament.records, tournament.ratings,
        weighting, analysis, say, origin=origin,
    )


def _foreign_run(out: Path, seed: int, roster: tuple[str, ...] = CANONICAL_AGENTS) -> str:
    """Why this run directory must not be written into, or ``""``.

    `chess-variant-sandbox tournament` writes to `runs/<hash>-<seed>` only with an explicit `--out`, or for
    directories made before 2026-09-24, when Q335 moved its default to `runs/tournament/`; it
    may still hold whatever roster it was asked for. A measurement that replayed over one of
    those would throw away somebody's one-off and stamp what replaced it as canonical, so a
    replay writes only where the directory is absent or already holds a canonical run at this
    seed.
    """
    stored = out / "tournament.json"
    if not out.is_dir() or not (stored.exists() or (out / "games.jsonl").exists()):
        return ""
    try:
        config = json.loads(stored.read_text(encoding="utf-8")).get("config") or {}
    except (OSError, ValueError):
        return f"{out} holds games and no readable tournament.json; move it aside first"

    agents = tuple(str(agent) for agent in config.get("agents") or ())
    if (
        agents != roster
        or int(config.get("games_per_pairing", -1)) != CANONICAL_GAMES
        or int(config.get("seed", -1)) != seed
    ):
        return (
            f"{out} holds a tournament on another roster or seed "
            f"({len(agents)} agents, {config.get('games_per_pairing')} games a pairing, "
            f"seed {config.get('seed')}); move it aside before measuring this game"
        )
    return ""


def _read_seed(
    candidate: Candidate,
    seed: int,
    out: Path,
    weighting: Any,
    total: float | None,
    say: Any,
    origin: str,
) -> RunReading | None:
    """One run as it already stands, or ``None`` when it has to be scored again.

    A run whose games and score are both current is read off its own
    `index-<weights>.json` and its tournament's ladder, which costs a file read against
    the bootstrap's forty seconds. The check that makes that safe is the total: the game's
    file records what each run scored, and if the index file beside the games disagrees
    the two were written by different passes. Neither number is then trusted -- the run is
    scored again from the games, which are the only thing both were derived from.
    """
    index_path = out / f"index-{weighting.version}.json"
    try:
        result = json.loads(index_path.read_text(encoding="utf-8"))
        played = json.loads((out / "tournament.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict) or not isinstance(played, dict):
        return None

    # `IndexResult.scored` is a property, so it is not in the saved JSON. What the
    # property asks is whether the run produced a combination to report, and a
    # placeholder leaves that empty.
    scored = bool(result.get("contributions"))
    headline = result.get("combined")
    if headline is None:
        headline = result.get("total")
    if not scored or headline is None:
        return None

    if total is not None and round(float(headline), 4) != round(float(total), 4):
        say(
            f"  seed {seed}: its index file scores {float(headline):.4f} and the "
            f"measurement recorded {float(total):.4f}, so it is scored again"
        )
        return None

    standings = [
        {
            "name": str(standing.get("name", "")),
            "rating": float(standing.get("rating", 0.0)),
            "games": int(standing.get("games", 0) or 0),
        }
        for standing in (played.get("ratings") or {}).get("standings") or []
    ]
    say(f"  seed {seed}: read as it stands, {float(headline):.4f}")
    return RunReading(
        seed=seed,
        scored=True,
        total=float(headline),
        sd=float(result["interval"]["sd"]) if result.get("interval") else None,
        viable=bool(result.get("viable")),
        violations=tuple(str(item) for item in result.get("violations") or ()),
        unreadable=tuple(str(item) for item in result.get("unreadable") or ()),
        scores=dict(result.get("scores") or {}),
        raw=dict(result.get("raw") or {}),
        contributions=dict(result.get("contributions") or {}),
        families=dict(result.get("families") or {}),
        interval=dict(result["interval"]) if result.get("interval") else None,
        standings=tuple(standings),
        games=int(played.get("games", 0) or 0),
        origin=origin,
        # The seat splits alone, out of the diagnostics this run stored. The pooled seat
        # reading adds them up across the runs, so a run read as it stands has to bring
        # them with it or the pool would be short a run's games.
        seat=seat_details(result.get("diagnostics") or {}),
    )


def _score_run(
    candidate: Candidate,
    seed: int,
    out: Path,
    records: list[Any],
    ratings: Any,
    weighting: Any,
    analysis: CorpusAnalysis | None,
    say: Any,
    origin: str = "played",
) -> RunReading:
    """Score one run, and keep its reading whether or not it produced a number.

    A scorer that raises is a fact about this run and not about the game. Under `v10` a
    run that drew no blunder trials raises `WeightsError`, and eight of the 284 runs on
    disk store a 0.0 that was never a score. Either way the run carries the sentence and
    no number: it votes to refuse, it is never averaged, and the other runs are
    untouched.
    """
    standings = [
        # The game count travels with the rating. An Elo without it is a number with no
        # error bar attached, and the page renders that column: it read `undefined` for
        # every opponent until this was written down.
        {"name": standing.name, "rating": standing.rating, "games": standing.games}
        for standing in ratings.standings
    ]
    try:
        result = score_ruleset(
            candidate.spec, records, ratings, weights=weighting, analysis=analysis,
        )
    except WeightsError as error:
        say(f"  run {seed} has no score: {error}")
        return RunReading.unscored(
            seed, str(error), standings=standings, games=len(records), origin=origin
        )
    result.save(out / f"index-{result.weights_version}.json")
    return RunReading.from_result(
        seed, result, standings=standings, games=len(records), origin=origin
    )


def _store(
    candidate: Candidate,
    runs_dir: Path,
    readings: list[RunReading],
    weighting: Any,
    say: Any,
    carry: Basis | None = None,
) -> Measurement:
    """Write the one file every reader trusts, and hand it back already judged.

    The only writer of a `canonical.json` but `recertify`'s re-stamp, and the only place
    a basis is stamped. It
    combines the runs through `metrics.seeds.combine` -- which is where the mean, the
    interval, the verdict and the mean ladder are defined -- and writes the file and the
    mean's own `index-<weights>.json` into `runs/<hash>/`, a directory that holds no
    games.

    The ladder is written *here*, beside the score, rather than left to be found by
    globbing `tournament.json` files. That is the fix for the failure this whole module
    grew out of: score and standings that came from two independent scans could be two
    different runs, and the page rendering them had no way to know.
    """
    # The weights themselves (the repeats block scores with them), or their version's name.
    if isinstance(weighting, str):
        weighting = load_weights(weighting)
    weights_version = weighting.version
    combined = combine(
        candidate.hash, candidate.name, weights_version, readings, VERDICT_RULE
    )
    stored = {
        "hash": candidate.hash,
        "name": candidate.name,
        "agents": list(canonical_agents(candidate.hash, spec=candidate.spec)),
        "games_per_pairing": CANONICAL_GAMES,
        "seeds": list(combined.seeds),
        "games": combined.games,
        "index": combined.index,
        "totals_used": combined.totals_used,
        "sd": combined.sd,
        "ci_low": combined.ci_low,
        "ci_high": combined.ci_high,
        "viable": combined.viable,
        "verdict": combined.verdict.as_dict(),
        # The seat check over the runs pooled, whether or not it refused: the split of
        # each band, the share its favoured seat took and the lean that share is, so a
        # page can print how close a game came without the number deciding anything.
        "seat": dict(combined.seat),
        "violations": list(combined.violations),
        "weights_version": weights_version,
        "families": dict(combined.families),
        "standings": list(combined.standings),
        "runs": list(combined.runs),
        # How far the score rests on repeated games (Q953a): the share of different games
        # beside it and the runs scored with each game once. It moves no number.
        "repeats": _repeats(candidate, runs_dir, readings, weighting, combined),
    }

    out = measurement_dir(candidate.hash, runs_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"index-{weights_version}.json").write_text(
        json.dumps(combined.detail, indent=2), encoding="utf-8"
    )

    now = current_basis(
        candidate.hash, weights_version, candidate.spec, combined.seeds
    )
    # A carried certificate (C6) keeps what played the games beside it: the runs not
    # played again here are still the older code's, vouched for by the sample.
    basis = now if carry is None else _carried(now, carry)
    measurement = Measurement(
        hash=candidate.hash,
        directory=out,
        stored=stored,
        basis=basis,
        staleness=basis.stale_against(now),
    )
    measurement.save(out / "canonical.json")
    say(_summary(candidate, combined, weights_version))
    said = repeats.summary(stored["repeats"])
    if said:
        say(said)
    return measurement


def _repeats(
    candidate: Candidate,
    runs_dir: Path,
    readings: list[RunReading],
    weighting: Any,
    combined: Any,
) -> dict[str, Any] | None:
    """The `repeats` block, read off the runs' records on disk, or ``None`` without them.

    Every run this measurement combines has its games on disk by now (played, re-scored or
    read as it stands), so the block is the same whichever path each run took.
    """
    from chess_variant_sandbox.tournament import read_records

    runs: list[repeats.Run] = []
    for reading in sorted(readings, key=lambda run: run.seed):
        out = run_dir(candidate.hash, reading.seed, runs_dir)
        games = out / "games.jsonl"
        if not games.exists():
            return None
        records = list(read_records(games))
        analysis_path = out / "analysis.json"
        analysis = CorpusAnalysis.load(analysis_path) if analysis_path.exists() else None
        runs.append(repeats.Run(
            seed=reading.seed, records=records, analysis=analysis,
            counted=reading.scored and reading.total is not None,
        ))
    if not runs:
        return None
    return repeats.reading(
        candidate.spec, runs, canonical_agents(candidate.hash, spec=candidate.spec),
        weighting, combined.index, combined.sd, minimum_totals(len(runs)),
    )


def _summary(candidate: Candidate, combined: Any, weights_version: str) -> str:
    """The measurement in a few lines, for somebody watching a command run."""
    runs = len(combined.runs)
    over = f"mean of {runs} runs" if runs > 1 else "one run"
    if combined.index is None:
        head = (
            f"{candidate.name} [{candidate.hash}]  no index: "
            f"{combined.totals_used} of {runs} runs could be scored"
        )
    else:
        band = ""
        if combined.sd is not None and combined.ci_low is not None:
            band = (
                f" +/-{combined.sd:.3f} "
                f"[{combined.ci_low:.3f}, {combined.ci_high:.3f}]"
            )
        head = (
            f"{candidate.name} [{candidate.hash}]  index={combined.index:.3f}{band} "
            f"({over}, weights {weights_version})"
        )
    lines = [head]
    if not combined.viable:
        lines.append("  NOT VIABLE:")
        lines.extend(f"    - {violation}" for violation in combined.violations)
    if combined.families:
        lines.append("  by family: " + ", ".join(
            f"{name} {value:.2f}" for name, value in sorted(combined.families.items())
        ))
    return "\n".join(lines)


def refresh(
    candidate: Candidate,
    runs_dir: Path = RUNS_DIR,
    weights: str = DEFAULT_WEIGHTS,
    workers: int = 0,
    report: Any = None,
    force: bool = False,
    seeds: tuple[int, ...] | None = None,
    adopt_from: Path | None = None,
) -> Measurement:
    """Bring this game's measurement up to date, doing only the work that is missing.

    The front door, and the answer to having to remember three commands in the right
    order. Four outcomes, and which one you get is read off the stored basis rather
    than guessed:

    * **current** -- nothing runs, and the stored measurement comes straight back.
    * **`rescore`** -- the games still stand, because nothing that decides a game moved.
      The persisted records are re-read and re-scored, which is seconds. This is the
      common case: a weights bump, or an edit to `metrics/`.
    * **`reanalyse`** -- the games still stand but the analysis pass has changed, so the
      `analysis.json` it wrote is about a pass nobody would run now. The pass runs again
      over the stored games, then the re-score. Until this existed an edit to the pass
      read as a `rescore`, which reused the stale file and stamped the result current
      with the pass never having run; `docs/archive/bugs-resolved.md` (`v5`) has the demonstration.
    * **`replay`** -- the tournament is about a match nobody would play now, so all three
      passes run again.

    The middle cases rest on the invariant the whole project rests on: `metrics` reads
    persisted records and never a live engine, so the games do not have to be played
    again to be scored again -- and the analysis pass reads the same records, so they do
    not have to be played again to be analysed again either.

    **How much that saves depends entirely on the ruleset, and on a cheap one it saves
    very little.** What a re-score skips is the tournament and the analysis pass; what it
    still pays is the bootstrap, which resamples the corpus and is the dominant cost of
    scoring anything. Measured on `antichess`: re-score 53s against a full replay of 59s,
    because the bootstrap is 41.8s of both and the tournament only 6.9s at eight workers.
    The saving is large exactly where replaying hurts -- a ruleset whose games are slow --
    and near zero where it does not.

    So this is a correctness mechanism before it is a speed one. The reason not to replay
    games that are still valid is that they are still valid.

    **Seed by seed, and the file's own seeds are never dropped.** A game whose file names
    one run and whose reader expects four owes three tournaments and nothing else, and one
    whose metrics moved owes a re-score of every run it has. The work is decided per seed
    from `Staleness.missing_seeds` and `Staleness.stored_needs`, so the repair costs what
    it says it costs. `seeds` overrides what the reader expects; a seed already in the
    file is measured whether or not it is asked for, because a measurement that quietly
    lost a run would be a smaller claim wearing the same shape.

    `adopt_from` is a campaign root whose runs stand in for tournaments this would
    otherwise play. A run is taken only where this checkout re-derives a sample of it --
    `tournament/adopt.py` -- and one that fails is played as it would have been.
    """
    say = report if callable(report) else (lambda _message: None)
    expected = tuple(sorted(seeds)) if seeds else expected_seeds(candidate.stage)
    stored = read_measurement(
        candidate.hash, runs_dir, weights, candidate.spec, expected
    )

    held: tuple[int, ...] = ()
    totals: dict[int, float | None] = {}
    if stored is not None:
        held = stored.basis.seeds if stored.basis is not None else ()
        totals = {
            int(run["seed"]): run.get("index")
            for run in stored.stored.get("runs") or []
            if isinstance(run, dict) and run.get("seed") is not None
        }

    if stored is not None and stored.current and not force:
        say(f"{candidate.name}: already current")
        return stored

    if stored is not None:
        say(f"{candidate.name}: {stored.staleness.summary()}")

    covered = sorted(set(expected) | set(held))
    if force or stored is None:
        work = {seed: "replay" for seed in covered}
    else:
        owed = stored.staleness.stored_needs
        work = {seed: ("replay" if seed not in held else owed) for seed in covered}

    return _measure_seeds(
        candidate, runs_dir, weights, workers, report, work,
        totals=totals, adopt_from=adopt_from,
    )


def _carried(now: Basis, carry: Basis) -> Basis:
    """``now`` with ``carry``'s played fields and certificate: a re-certified stamp, re-scored."""
    return replace(
        now,
        engine=carry.engine,
        tables=carry.tables,
        players=carry.players,
        agents_source=carry.agents_source,
        certified=carry.certified,
    )


# ---------------------------------------------------------- sampled re-certification

# The two ways a replay can be paid (`replay_route`): a sample of every run that may let
# the stored games stand (C6, `recertify`), or every game played again (`refresh`).
SAMPLE_ROUTE = "sample"
REPLAY_ROUTE = "replay"


@dataclass(frozen=True)
class Route:
    """How a game that owes a replay would be repaired, and why; read, never spent.

    ``route`` is `SAMPLE_ROUTE`, `REPLAY_ROUTE`, or ``""`` where no replay is owed (the
    game is current, or owes a re-score or a re-analysis). `recertify` decides with it and
    the refresher's plan prints it, so the plan and the repair cannot disagree.
    """

    route: str
    reason: str
    stored: Measurement | None = None
    now: Basis | None = None


def replay_route(
    candidate: Candidate,
    runs_dir: Path = RUNS_DIR,
    weights: str = DEFAULT_WEIGHTS,
    stored: Measurement | None = None,
) -> Route:
    """Whether `recertify` would take a sample for this game or replay it (C6's rules).

    A sample stands in only for a replay the stored stamp owes because the code that plays
    games moved (`Basis.certifiable`: the core, the runner, the tables, the players); never
    for a game with no stamp, another roster, net, seed set or ruleset, or a browser's
    games. The answer is the plan's: a run the sample then refuses is still played again
    (`recertify`), which only `recertify` can know.
    """
    expected = expected_seeds(candidate.stage)
    if stored is None:
        stored = read_measurement(candidate.hash, runs_dir, weights, candidate.spec, expected)
    if stored is None or stored.basis is None:
        return Route(REPLAY_ROUTE, "no stamped measurement for a sample to stand in for",
                     stored)
    if stored.current:
        return Route("", "already current", stored)
    if stored.staleness.stored_needs != "replay":
        if stored.staleness.needs == "replay":
            return Route(REPLAY_ROUTE, "only missing runs owe a replay, and a sample "
                         "stands in for no run that was never played", stored)
        return Route("", "owes no replay, so there is nothing for a sample to save", stored)
    now = current_basis(candidate.hash, weights, candidate.spec, expected)
    after = stored.basis.certifiable(now)
    if after.stored_needs == "replay":
        return Route(REPLAY_ROUTE,
                     f"a sample cannot stand in for this -- {after.summary()}", stored, now)
    return Route(SAMPLE_ROUTE, "the code that plays games moved; a sample of every run "
                 "stands in for the replay, and a run it refuses is played again",
                 stored, now)


def recertify(
    candidate: Candidate,
    runs_dir: Path = RUNS_DIR,
    weights: str = DEFAULT_WEIGHTS,
    workers: int = 0,
    report: Any = None,
) -> Measurement:
    """Keep this game's score through a change that should move no game, by a sample (C6).

    `docs/reference/compute-economy.md` §3.5, under the user's answer B (2026-10-02): a
    score kept by a sample is shown *re-certified from a sample* and its stamp carries the
    sample as its basis (`measurement.Certificate`). The repair `measure --recertify` buys
    in place of the replay `refresh` would pay:

    1. **Only a replay is replaced.** A current file, one that owes a re-score or less, and
       one a sample cannot speak for (another roster, net, seed set or ruleset:
       `Basis.certifiable`) go to `refresh` as they always did.
    2. **Every run, by a sample** (`adopt.recertify_run`, `RECERTIFY_PER_AGENT` games with
       each agent first, six blunder trials), in place. A run the sample refuses is
       played again in full, as adopt replayed xiangqi's three; a game none of whose runs
       holds is replayed as `refresh` would.
    3. **The stamp.** The stored stamp keeps what played the games and gains the
       certificate; whatever else it owes (a re-score, a missing seed, the refused runs) is
       then paid by `_measure_seeds`, which carries the certificate through. Where nothing
       else is owed, the file is re-stamped and its numbers are not touched.

    A later replay of every run clears the label (`_measure_seeds`); a later change to
    the code the sample ran on stales the score again, as it would any stamp.
    """
    from chess_variant_sandbox.tournament import adopt

    say = report if callable(report) else (lambda _message: None)
    route = replay_route(candidate, runs_dir, weights)
    stored, now = route.stored, route.now
    if route.route != SAMPLE_ROUTE or stored is None or stored.basis is None or now is None:
        say(f"{candidate.name}: {route.reason}")
        if stored is not None and stored.current:
            return stored
        return refresh(candidate, runs_dir, weights, workers, report)
    behind = stale_build()
    if behind is not None:
        raise CorpusError(behind)
    say(f"{candidate.name}: {stored.staleness.summary()}")
    say("  re-certifying from a sample instead of replaying")

    agents = canonical_agents(candidate.hash, spec=candidate.spec)
    checks = [
        adopt.recertify_run(
            candidate.spec, seed, run_dir(candidate.hash, seed, runs_dir),
            TournamentConfig(
                agents=agents, games_per_pairing=CANONICAL_GAMES, seed=seed,
                workers=workers,
            ),
            workers=workers, say=say,
        )
        for seed in stored.basis.seeds
    ]
    kept = [check for check in checks if check.certified]
    refused = [check.seed for check in checks if not check.certified]
    for check in checks:
        if not check.certified:
            say(f"  seed {check.seed}: not re-certified -- {check.reason}; "
                "it is played again")
    if not kept:
        say("  no run held, so the game is replayed")
        return refresh(candidate, runs_dir, weights, workers, report, force=True)

    sample = sum(check.sampled for check in kept)
    games = sum(check.games for check in kept)
    certified = stored.basis.certify(now, sample, games)
    owed = certified.stale_against(now)
    say(f"  {sample} of {games} games replayed exactly; "
        f"{owed.summary() or 'nothing more is owed'}")
    work = {seed: ("replay" if seed in refused else owed.stored_needs) for seed in stored.basis.seeds}
    for seed in owed.missing_seeds:
        work[seed] = "replay"
    if not any(work.values()):
        return _restamp(stored, certified, now, say)
    totals = {
        int(run["seed"]): run.get("index")
        for run in stored.stored.get("runs") or []
        if isinstance(run, dict) and run.get("seed") is not None
    }
    return _measure_seeds(
        candidate, runs_dir, weights, workers, report, work,
        totals=totals, carry=certified,
    )


def _restamp(stored: Measurement, basis: Basis, now: Basis, say: Any) -> Measurement:
    """Write the stored measurement back with a re-certified stamp; no number moves.

    `_store`'s one sibling. The file's numbers are the ones it held, read from it, and only
    `basis` is new; `Measurement.save` writes it, so the shape is the one `_store` writes.
    """
    measurement = Measurement(
        hash=stored.hash,
        directory=stored.directory,
        stored={key: value for key, value in stored.stored.items() if key != "basis"},
        basis=basis,
        staleness=basis.stale_against(now),
    )
    measurement.save(stored.directory / "canonical.json")
    if basis.certified is not None:
        say(f"  re-stamped: {basis.certified.words()}")
    return measurement


def poolable(
    candidate: Candidate,
    runs_dir: Path = RUNS_DIR,
    weights: str = DEFAULT_WEIGHTS,
    seeds: tuple[int, ...] | None = None,
) -> bool:
    """Whether `refresh` would only re-score or re-analyse games already on disk.

    What `measure` asks before it sends a game through a process pool (R7 of
    `docs/reference/perf-survey-2026-09.md`). Such a game plays nothing and writes only its own directories, so games like it can run side by side; one that owes a
    tournament already uses every core and runs alone, as does anything this cannot vouch
    for. It reads what `refresh` reads and decides nothing for it: the worker calls
    `refresh` itself, so a pooled game takes the serial path's every step.

    Conservative on purpose. A missing seed, a run whose games are gone, or a
    stamp that cannot be read all say no, and the game is repaired serially as before.
    """
    expected = tuple(sorted(seeds)) if seeds else expected_seeds(candidate.stage)
    stored = read_measurement(
        candidate.hash, runs_dir, weights, candidate.spec, expected
    )
    if stored is None or stored.current or stored.basis is None:
        return False
    staleness = stored.staleness
    if staleness.missing_seeds:
        return False
    if staleness.stored_needs not in ("rescore", "reanalyse"):
        return False
    return all(
        (run_dir(candidate.hash, seed, runs_dir) / "games.jsonl").exists()
        for seed in stored.basis.seeds
    )


def _rescore_seed(
    candidate: Candidate,
    seed: int,
    out: Path,
    weighting: Any,
    workers: int,
    say: Any,
    reanalyse: bool = False,
    origin: str = "played",
) -> RunReading:
    """Score the games already in this run directory, without replaying them.

    Rebuilds the ladder from the records too. `rate` is a fit over finished games rather
    than something the tournament held privately, so a re-score produces the same
    standings the tournament did -- which is what lets this path contribute a whole
    reading, ladder included, rather than a partial one.

    ``reanalyse`` runs the analysis pass again over the stored games before scoring,
    rather than reading the `analysis.json` already there. It is set when the basis says
    the pass itself changed; the file on disk is then about a pass nobody would run now,
    and reading it would stamp a measurement current that the pass never produced.

    The pass runs at `CANONICAL_SEED` whatever this run's seed is. Every trial is fixed by
    the pass's seed and the record it is drawn from, and the records differ by run, so the
    trials already differ between runs without the pass's own seed moving.
    """
    from chess_variant_sandbox.tournament import rate, read_records

    say(
        f"seed {seed}: re-analysing and re-scoring the games already on disk"
        if reanalyse
        else f"seed {seed}: re-scoring the games already on disk"
    )
    records = list(read_records(out / "games.jsonl"))
    if not records:
        raise CorpusError(
            f"{out} has no games to re-score; delete it and measure again"
        )

    analysis_path = out / "analysis.json"
    analysis: CorpusAnalysis | None = None
    if analysis_path.exists() and not reanalyse:
        analysis = CorpusAnalysis.load(analysis_path)
    if analysis is None:
        # Four weighted components come only from this pass. Re-scoring without it would
        # silently produce a total resting on twelve measurements where the stored one
        # rested on sixteen, and the two would sit in the same column.
        #
        # The pass drives the engine, so the same refusal `measure` makes applies: a core
        # built from older sources does not fail, it answers with the previous build, and
        # the file written here would carry a basis describing source it was not built
        # from.
        behind = stale_build()
        if behind is not None:
            raise CorpusError(behind)
        say(
            "  the analysis pass again, because the pass itself changed"
            if reanalyse
            else "  the analysis pass, which the stored measurement is missing"
        )
        analysis = analyse_corpus(
            candidate.spec, records, seed=CANONICAL_SEED, workers=workers
        )
        analysis.save(analysis_path)

    return _score_run(
        candidate, seed, out, records, rate(records), weighting, analysis, say,
        origin=origin,
    )


# -------------------------------------------------------------- archive and promote


def archive_game(
    candidate: Candidate,
    note: str = "",
    archive_dir: Path = ARCHIVE_DIR,
    runs_dir: Path = RUNS_DIR,
    name: str = "",
    credit: str = "",
) -> Path:
    """Keep this game in the repository, out of the corpus.

    `credit` is the designer's handle: it becomes `provenance.author` and a sentence
    ("Designed by <handle>.") directly after the note in `provenance.history`.

    No measurement is demanded. Archiving says "this was worth inventing", which is a
    judgement a person is entitled to make about a game whose tournament nobody wants to
    pay for -- and the alternative to an unmeasured game in `archive/` is an unmeasured
    game in a directory git ignores.
    """
    if candidate.stage == "committed":
        raise CorpusError(f"{candidate.name} is already in {RULESET_DIR}/")
    return _write(candidate, archive_dir, note, "archived", runs_dir, name, credit)


def promote(
    candidate: Candidate,
    reason: str,
    ruleset_dir: Path = RULESET_DIR,
    runs_dir: Path = RUNS_DIR,
    name: str = "",
) -> Path:
    """Admit this game to the calibration corpus, with the argument for doing so.

    Three refusals, and each is a thing the corpus would otherwise quietly acquire:

    * an invalid spec -- every committed ruleset validates, and a run's copy can predate
      a schema bump;
    * an untested or non-viable one -- a corpus entry's number has to be on the corpus's
      own roster, and a game the gates disqualify is not a corpus entry at any score;
    * no reason -- `rulesets/` is a set of games that each pin down something no other
      one does, and that sentence is the only part of this a command cannot supply.
    """
    blocker = promotion_blocker(candidate, runs_dir, ruleset_dir)
    if blocker is not None:
        raise CorpusError(blocker)

    if not reason.strip():
        raise CorpusError(
            "promotion needs a reason: what does this game pin down that nothing "
            "already in the corpus does? It is written into the ruleset's provenance "
            "and is the only part of this decision a command cannot make"
        )

    written = _write(candidate, ruleset_dir, reason, "committed", runs_dir, name)

    # A game is on one rung at a time. Left in `archive/` as well, it would be a second,
    # staler copy of a ruleset the corpus now owns -- and the archive's whole meaning is
    # "kept, and not part of what the index is calibrated against".
    if candidate.stage == "archived" and candidate.path.exists():
        candidate.path.unlink()

    return written


def promotion_blocker(
    candidate: Candidate,
    runs_dir: Path = RUNS_DIR,
    ruleset_dir: Path = RULESET_DIR,
) -> str | None:
    """Why this game cannot enter the corpus yet, in one sentence, or ``None``.

    Separate from `promote` so an interface can *show* the answer rather than discover
    it by being refused -- and shared with `promote` so the two cannot disagree about
    what the rule is. The reason a person must supply is deliberately not checked here:
    it is a property of the moment of asking, not of the game, and a button that greys
    itself out until a text box is filled is a worse explanation than the text box.

    **Four runs, `LIBRARY_SEEDS`, and not whatever the reader currently expects.** Four of
    71 library games change their verdict with the seed, and promotion is a claim about
    the corpus rather than a reading of one game -- so the rung before it is the full
    measurement, whether or not the library has caught up. `archive` still asks for
    nothing, and an archived game reads stale until its other three seeds are played.
    """
    if candidate.stage == "committed":
        return f"{candidate.name} is already in {ruleset_dir}/"

    errors = validate_spec(candidate.spec)
    if errors:
        return f"{candidate.name} does not validate: " + "; ".join(errors[:4])

    stored = read_measurement(
        candidate.hash, runs_dir, spec=candidate.spec, seeds=LIBRARY_SEEDS,
    )
    if stored is None:
        return (
            f"{candidate.name} has no score on the corpus roster. Measure it first: "
            f"`uv run chess-variant-sandbox test {candidate.hash}` -- a design run's number is taken "
            "on a smaller ladder and is not comparable with a committed one"
        )
    if not stored.current:
        # The refusal this whole mechanism exists for. Promotion is a claim about the
        # corpus, and admitting a game on a number taken under different weights, a
        # different roster, a different engine or one run instead of four puts a quantity
        # into `rulesets/` that no other entry shares -- which is precisely what the
        # corpus is for not doing.
        return (
            f"{candidate.name}'s score is out of date -- {stored.staleness.summary()}. "
            f"Refresh it: `uv run chess-variant-sandbox test {candidate.hash}`"
        )
    if not stored.stored.get("viable"):
        violations = "; ".join(str(item) for item in stored.stored.get("violations", [])[:3])
        return (
            f"{candidate.name} is not viable on the corpus roster"
            + (f": {violations}" if violations else "")
        )
    return None


def _write(
    candidate: Candidate,
    directory: Path,
    note: str,
    stage: str,
    runs_dir: Path,
    name: str = "",
    credit: str = "",
) -> Path:
    """One spec into a committed directory, with the provenance it can prove.

    The spec is written with `provenance` filled from the *archive*, never from the
    model: origin, the run it came from, what that run was asked to explore, and its
    parent. `provenance` is outside the mechanics hash, so none of this changes what
    the game is -- `canonical.MECHANIC_KEYS` is what a hash covers, and adding a
    provenance block to twenty-two rulesets once moved none of them.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # A name given here renames the *game*, not just the file. `name` is outside the
    # mechanics hash, so this is free -- and the alternative is a corpus entry called
    # "gardner (some of the pawns are now a new piece X with a 1,3 step)", which is what
    # a mutation operator calls its own output.
    raw_title = name.strip() or str(candidate.spec.get("name", candidate.hash))
    title = _normalize_name(raw_title) or candidate.hash
    path = directory / f"{_slug(title)}.json"
    if path.exists():
        existing = load_spec(path)
        if spec_hash(existing) != candidate.hash:
            raise CorpusError(
                f"{path} is already a different game ({spec_hash(existing)}); "
                "rename this one first"
            )

    # In the current spelling (Q1008c), so a game promoted from an older one is written the
    # way the library is and passes `tests/test_spelling.py`'s guard; its name stays.
    spec = current_spelling(candidate.spec)
    spec["name"] = title
    spec["provenance"] = _provenance(candidate, note, stage, runs_dir, credit)
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return path


def _provenance(
    candidate: Candidate, note: str, stage: str, runs_dir: Path, credit: str = ""
) -> dict[str, Any]:
    declared = dict(candidate.spec.get("provenance") or {})
    lineage = candidate.lineage

    # What the previous rung wrote, kept: a game archived and then promoted came out of
    # a design run either way, and the second write no longer has the lineage to say so
    # -- `find` reports it as archived, which is where it now lives. The sentences this
    # function generates are recognisable, so they are added once rather than twice.
    carried = str(declared.get("history") or "").strip()
    history = [_sentence(note)] if note.strip() else []
    credit = " ".join(credit.split())
    if credit:
        # The designer is named, and the reason still leads the history, which is the
        # sentence `site.load.reason_for` reads.
        declared["author"] = credit
        history.append(f"Designed by {credit}.")
    if carried:
        history.append(carried)

    if candidate.run is not None and "Found by" not in carried:
        made = "a mechanical edit" if lineage.get("operator") == "mechanical" else "the design loop"
        where = f"Found by {made} in run {candidate.run.name}"
        if lineage.get("generation") is not None:
            where += f", generation {lineage['generation']}"
        if lineage.get("parent_name"):
            where += f", revising {lineage['parent_name']}"
        history.append(where + ".")

    stored = read_measurement(
        candidate.hash, runs_dir, spec=candidate.spec,
        seeds=expected_seeds(candidate.stage),
    )
    # Only a current measurement becomes provenance, and only one that has a number. A
    # spec written into `rulesets/` carries this sentence forever, and a number that was
    # already out of date when it was inscribed is the most durable form of the problem
    # this module now prevents -- while a game whose runs could not be scored has no
    # number to inscribe at all.
    if (
        stored is not None
        and stored.current
        and stored.stored.get("index") is not None
        and "on the corpus roster" not in carried
    ):
        runs = len(stored.stored.get("runs") or [])
        over = f"mean of {runs} runs" if runs > 1 else "one run"
        history.append(
            f"Scores {stored.stored['index']:.3f} +/- {stored.stored['sd']:.3f} on the "
            f"corpus roster under weights {stored.stored['weights_version']} ({over})."
        )
    elif stored is None and stage == "archived" and not carried:
        history.append("Kept without a corpus-roster measurement.")

    declared.setdefault("origin", "evolved" if candidate.run else "generated")
    if candidate.run is not None:
        declared.setdefault("author", f"chess-variant-sandbox evolve ({candidate.run.name})")
    if lineage.get("theme"):
        declared.setdefault("theme", str(lineage["theme"]))
    if lineage.get("parent"):
        declared.setdefault("parents", [str(lineage["parent"])])
    if history:
        declared["history"] = " ".join(history)
    # The day the game joined the library (Q826), which the landing page's *New* shelf and
    # the games page's *date added* order read. A game already in the library keeps its
    # day, so one archived and later promoted keeps the day it first arrived; a candidate
    # from `runs/` joins today, whatever its own file claims.
    if candidate.stage not in ("archived", "committed") or "added" not in declared:
        declared["added"] = _today()
    return declared


def _today() -> str:
    """Today's calendar day in UTC, `YYYY-MM-DD`: one day for everybody, as `launch.today`."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).date().isoformat()


def _sentence(text: str) -> str:
    """A person's note, punctuated so it does not run into the ones added after it."""
    stripped = text.strip()
    return stripped if stripped.endswith((".", "!", "?")) else stripped + "."


def _normalize_name(name: str) -> str:
    """A proposed name, brought to the corpus's one convention before it is written.

    `tests/test_corpus.py::TestTheCorpusNames` pins the shape: lowercase words
    separated by single spaces, nothing after the last word. A model proposes whatever
    style it likes -- `"Longshot"`, `"Tunnel Chess Horizontal."` -- and this is where
    that gets reduced, so a spec never reaches disk carrying a name the corpus test
    would refuse.

    A word chaining two or more hyphens (`"tunnel-chess-horizontal"`) is unmistakably a
    model's spaces gone missing, so every hyphen in it becomes a space. A single hyphen
    (`"one-way"`) cannot be told apart from a genuine compound word by any rule
    available here -- the corpus already commits `one-way`, `three-check` and
    `half-turn` to meaning exactly that -- so a lone hyphen is kept as given; the
    convention's own check in `tests/test_corpus.py` is the authority on what still
    gets through wrong.
    """
    collapsed = re.sub(r"\s+", " ", name.strip().lower())
    collapsed = re.sub(r"[^a-z0-9]+$", "", collapsed)
    words = [
        word.replace("-", " ") if word.count("-") > 1 else word
        for word in collapsed.split(" ")
    ]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def _slug(name: str) -> str:
    """A filename from a game's name, in the style the corpus already uses.

    Committed rulesets are named for their game -- `chess.json`, `racingkings.json` --
    rather than by hash, because a directory listing is something a person reads. A
    model-proposed name can be anything, so it is reduced rather than trusted.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if len(slug) > 40:
        # At a word boundary, so a truncated name reads as a short name rather than as
        # a broken one -- `gardner-some-of-the-pawns` and not `...-a-new-`.
        slug = slug[:40].rsplit("-", 1)[0]
    return slug.strip("-") or "unnamed"
