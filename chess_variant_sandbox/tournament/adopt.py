"""Runs played elsewhere, taken in once this checkout can reproduce a sample of them.

Three more seeds of every library game were played, analysed and scored by a campaign of
their own, and a campaign directory holds exactly what a run directory holds. Replaying
them would cost about 66 minutes to arrive at the same bytes. Copying them outright would
cost nothing and assert a basis from a log: the campaign's log names the core it ran on,
and the Python half of the `engine` fingerprint was recorded nowhere for these runs.

So a run is adopted only where this checkout reproduces part of it. Sixteen of the 256
games are played again from their own seeds and compared move for move, the same sixteen
are analysed again, and six of the twenty-four blunder trials are re-derived. A run that
fails any step is not adopted, and `corpus` replays that seed instead -- which is why no
step here needs a fallback of its own. `docs/archive/plan-four-seeds.md` §3.2 is the
design.

**The re-derivation goes through `tournament/slices.py`'s units**, which call the runner's
own play loop and the analysis pass's own graders. So what is compared here is what a
replay would write, rather than a second reading of the rules.

**Nothing in this file is fingerprinted, and nothing in it may become so.** It decides no
game and computes no component: it reads a schedule, plays through the runner and copies
three files. A measurement's basis is stamped by `corpus._store` as it always was, and the
sample is what entitles that stamp to cover games this checkout did not just play.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.tournament import slices
from chess_variant_sandbox.tournament.analysis import BLUNDER_SAMPLE, CorpusAnalysis
from chess_variant_sandbox.tournament.record import GameRecord, read_records
from chess_variant_sandbox.tournament.runner import TournamentConfig, schedule

# What a campaign's run directory carries over. The index file is left behind on purpose:
# `metrics` moved after the campaign scored these runs, so `corpus` re-scores every
# adopted run and writes the index file itself.
CARRIED = ("games.jsonl", "tournament.json", "analysis.json")

# How much of a run is re-derived before it is believed. Two games with each roster agent
# in the first seat, so no player's native form goes unchecked, and a quarter of the
# blunder trials, which are the one part of the analysis pass with a stream of its own.
SAMPLE_PER_AGENT = 2
SAMPLE_BLUNDERS = 6

# What a record may differ in and still be the same game. Wall-clock is a fact about the
# machine that played it.
IGNORED_FIELDS = ("duration_ms",)


@dataclass(frozen=True)
class Adoption:
    """What became of one campaign run, in the words `runs[].origin` keeps."""

    seed: int
    adopted: bool
    # Why not, when it was not. Empty on success.
    reason: str = ""
    # What the game's file records about where this run's records came from.
    origin: str = ""
    source: Path | None = None


def campaign_run(source_root: str | Path, digest: str, seed: int) -> Path | None:
    """Where a campaign keeps this game's run at this seed, if it kept one.

    A campaign root is a directory of `<hash>-<seed>` directories, which is the shape
    `runs/` itself has -- so a campaign is adopted from wherever it was left rather than
    being moved into place first.
    """
    path = Path(source_root) / f"{digest}-{seed}"
    return path if (path / "games.jsonl").exists() else None


def adopt_run(
    spec: dict[str, Any],
    seed: int,
    source: Path,
    target: Path,
    config: TournamentConfig,
    workers: int = 0,
    say: Any = None,
) -> Adoption:
    """Check a campaign's run against this checkout, and copy it in if it holds.

    Five steps, in this order, and the first failure is the answer:

    1. refuse to overwrite an existing run directory;
    2. read what the directory claims -- its roster, its games a pairing and its seed;
    3. check every record sits at the seed `runner.schedule` gives its position;
    4. re-derive the sample and compare;
    5. copy the three files in.

    The campaign directory is never written to. `config` is the roster a committed score
    means, at this run's seed, which is both what step 3 checks against and what step 4
    plays with.
    """
    report = say if callable(say) else (lambda _message: None)

    if (target / "games.jsonl").exists():
        return Adoption(
            seed=seed, adopted=False, source=source,
            reason=f"{target} already holds a run; it was left alone",
        )

    records = list(read_records(source / "games.jsonl"))
    tasks = schedule(config)

    reason = _claims(source, config, len(tasks), len(records))
    if reason:
        return Adoption(seed=seed, adopted=False, reason=reason, source=source)

    reason = _seated(records, tasks)
    if reason:
        return Adoption(seed=seed, adopted=False, reason=reason, source=source)

    analysis_path = source / "analysis.json"
    if not analysis_path.exists():
        return Adoption(
            seed=seed, adopted=False, source=source,
            reason=f"{source} has no analysis.json to check",
        )
    stored = CorpusAnalysis.load(analysis_path)

    sample = _sample(records, config)
    report(f"  re-deriving {len(sample)} of {len(records)} games and "
           f"{SAMPLE_BLUNDERS} of {len(stored.blunders)} blunder trials")
    reason = _rederived(spec, records, stored, sample, config, workers)
    if reason:
        return Adoption(seed=seed, adopted=False, reason=reason, source=source)

    target.mkdir(parents=True, exist_ok=True)
    for name in CARRIED:
        shutil.copyfile(source / name, target / name)

    return Adoption(
        seed=seed,
        adopted=True,
        source=source,
        origin=(
            f"adopted from {source.parent.name}, "
            f"{len(sample)} games and {SAMPLE_BLUNDERS} trials re-derived"
        ),
    )


# ------------------------------------------------------------------ what it claims


def _claims(
    source: Path, config: TournamentConfig, expected: int, held: int
) -> str:
    """Whether the directory's own record says it is the run we are looking for.

    A campaign's `tournament.json` names the roster, the games a pairing and the seed it
    was played at. A directory that names anything else is somebody else's tournament and
    is not adopted under this game's basis, whatever its games look like.
    """
    path = source / "tournament.json"
    try:
        stored = json.loads(path.read_text(encoding="utf-8")).get("config") or {}
    except (OSError, ValueError):
        return f"{path} is missing or unreadable"

    agents = [str(agent) for agent in stored.get("agents") or ()]
    if tuple(agents) != tuple(config.agents):
        return (
            f"{source.name} was played on a different roster "
            f"({len(agents)} agents, not {len(config.agents)})"
        )
    if int(stored.get("games_per_pairing", 0)) != config.games_per_pairing:
        return (
            f"{source.name} played {stored.get('games_per_pairing')} games a pairing, "
            f"not {config.games_per_pairing}"
        )
    if int(stored.get("seed", -1)) != config.seed:
        return f"{source.name} was played from seed {stored.get('seed')}, not {config.seed}"
    if held != expected:
        return f"{source.name} holds {held} games and the schedule has {expected}"
    return ""


def _seated(records: list[GameRecord], tasks: list[tuple[int, str, str]]) -> str:
    """Whether every record sits where the schedule put it.

    The schedule hands seeds out by position, so a record's seed and its two agents are
    fixed by its index before any game is played. Checking all 256 costs nothing and is
    what makes the sample a sample of *this* schedule rather than of some ordering of it.
    """
    for index, (record, (seed, first, second)) in enumerate(zip(records, tasks, strict=True)):
        if record.seed != seed or record.agents != (first, second):
            return (
                f"game {index} is {record.agents[0]} against {record.agents[1]} at seed "
                f"{record.seed}, and the schedule puts {first} against {second} at {seed}"
            )
    return ""


def _sample(records: list[GameRecord], config: TournamentConfig) -> list[int]:
    """Which games are played again: two with each roster agent in the first seat.

    Read off the records rather than off the schedule so the choice is the same however a
    roster is ordered, and in record order so a run is re-derived front to back.
    """
    counted: dict[str, int] = {}
    chosen: list[int] = []
    for index, record in enumerate(records):
        first = record.agents[0]
        if first not in config.agents or counted.get(first, 0) >= SAMPLE_PER_AGENT:
            continue
        counted[first] = counted.get(first, 0) + 1
        chosen.append(index)
    return sorted(chosen)


# --------------------------------------------------------------- the re-derivation


def _rederived(
    spec: dict[str, Any],
    records: list[GameRecord],
    stored: CorpusAnalysis,
    sample: list[int],
    config: TournamentConfig,
    workers: int,
) -> str:
    """Play, analyse and blunder the sample again, and say the first thing that differs.

    The games run at this run's own seed and the analysis at seed 0, which is where every
    run's pass and bootstrap sit: a blunder trial is fixed by `(analysis seed, record)`
    and the records differ by run, so the trials differ without the pass's seed moving.
    """
    played = slices.run_unit(
        slices.Unit(phase=slices.TOURNAMENT, part=0, parts=1, games=tuple(sample)),
        spec,
        config=_with_workers(config, workers),
    )
    for index, fresh in zip(sample, slices.parse_records(played), strict=True):
        if _bare(fresh) != _bare(records[index]):
            return f"game {index} at seed {records[index].seed} replays differently here"

    # The pass's own seed, not the run's. `analyse_corpus` is called at seed 0 for every
    # run, so a unit re-deriving its trials must be too.
    at_zero = _with_seed(config, 0, workers)

    graded = slices.analysis_units(records, 1)
    positions = {index: place for place, index in enumerate(graded[0].games)} if graded else {}
    wanted = [index for index in sample if index in positions]
    if wanted and stored.games:
        analysed = slices.run_unit(
            slices.Unit(phase=slices.ANALYSIS, part=0, parts=1, games=tuple(wanted)),
            spec,
            records=[records[index] for index in wanted],
            config=at_zero,
        )
        for index, graded_again in zip(
            wanted, slices.parse_analyses(analysed), strict=True
        ):
            if graded_again.as_dict() != stored.games[positions[index]].as_dict():
                return f"game {index} analyses differently here"

    trials = slices.blunder_units(records, 1, BLUNDER_SAMPLE, at_zero.seed)
    if not trials:
        return "" if not stored.blunders else "this checkout draws no blunder trials here"
    drawn = trials[0].games[:SAMPLE_BLUNDERS]
    derived = slices.parse_blunders(slices.run_unit(
        slices.Unit(phase=slices.BLUNDER, part=0, parts=1, games=tuple(drawn)),
        spec,
        records=[records[index] for index in drawn],
        config=at_zero,
    ))
    # The pass keeps its trials in draw order and drops the games it could not force a
    # move into, so the trials the first six draws produced are the stored list's own
    # first entries.
    if len(stored.blunders) < len(derived):
        return (
            f"{len(derived)} of the first {SAMPLE_BLUNDERS} draws yield a trial here and "
            f"the stored pass kept {len(stored.blunders)}"
        )
    for place, trial in enumerate(derived):
        if trial.as_dict() != stored.blunders[place].as_dict():
            return f"blunder trial {place} at seed {trial.seed} differs here"
    return ""


def _bare(record: GameRecord) -> dict[str, Any]:
    """A record with the fields a machine decides taken out."""
    return {
        key: value
        for key, value in record.as_dict().items()
        if key not in IGNORED_FIELDS
    }


def _with_workers(config: TournamentConfig, workers: int) -> TournamentConfig:
    return _with_seed(config, config.seed, workers)


def _with_seed(config: TournamentConfig, seed: int, workers: int) -> TournamentConfig:
    return slices.canonical_config(
        agents=config.agents,
        games_per_pairing=config.games_per_pairing,
        seed=seed,
        workers=workers,
    )
