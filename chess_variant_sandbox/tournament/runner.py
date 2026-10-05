"""Scheduling and running a tournament.

The schedule includes self-pairings on purpose. An agent playing itself is the only
matchup where strength cancels exactly, which makes it the clean measurement of
first-player advantage -- one of the balance components of the index. Reading that
number off mixed-strength pairings instead would confound "moving first helps" with
"the stronger player happened to move first more often".

Colours are always played both ways: every ordered pair is scheduled, so a pairing
contributes equally to both seats and a seat-dependent bias cannot masquerade as a
rating difference.

Games run in parallel inside the Rust core, on threads rather than processes. That
removed the two costs that used to shape this file -- re-importing the package and
recompiling the ruleset in every worker -- which is why the schedule is no longer
kept serial below a threshold. What did not change is the guarantee that matters: the
seed of every game is fixed by the schedule before any thread starts, so the corpus is
identical at any worker count.

A roster with a player the core cannot run -- Fairy-Stockfish, or a custom agent -- is
split by pairing (Q439). The pairings where both seats are native still go to the core
as one schedule; only the ones with an outside seat play in the Python loop, one game
at a time. That loop hands every native pairing to the core too, a game per call, so
the split moves no game: it only stops one outside player from serialising the whole
schedule. The records come back in schedule order either way.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents import native_spec, parse_agent
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.tournament.play import build_record, eval_values, play_game
from chess_variant_sandbox.tournament.ratings import RatingTable, rate
from chess_variant_sandbox.tournament.record import GameRecord, write_records

# How many progress reports a long run prints. The schedule is handed to the core in
# that many pieces, which costs one thread-pool setup each -- microseconds against a
# chunk of games. A piece is never smaller than the threads playing it, though: below
# twenty games per thread the pieces used to be one or two games each, and a short run
# with progress on played one game at a time on a pool of eight (Q439).
PROGRESS_CHUNKS = 20


# How a fixed number of games is spread over the pairings.
#
# "uniform" is what a round robin means and is the honest baseline: every ordered pair
# gets the same count. "ladder" spends the budget where the noise is instead -- games
# against `random` inform only a gate that passes by a mile, while `skill_gradient`
# carries the largest weight in the index and is fitted from the searching agents alone.
#
# A scheme only earns its complexity by beating "more uniform games" at equal cost, and
# `ALLOCATIONS` is a closed set so that a corpus can always say which one produced it.
ALLOCATIONS = ("uniform", "ladder")

# What a "ladder" schedule does to a pairing's game count, as a multiplier on
# `games_per_pairing`. Integers, because a fractional game is not a thing and rounding
# a rate per pairing would make the total depend on the roster in a way nobody could
# predict from the config.
_LADDER_WEIGHTS = {
    "weak": 1,      # anything involving the random mover
    "mixed": 2,     # a searching agent against greedy
    "ladder": 3,    # two searching agents, which is where the gradient is fitted
}


def _pairing_kind(first: str, second: str) -> str:
    kinds = {first.split(":")[0], second.split(":")[0]}
    if "random" in kinds:
        return "weak"
    if kinds == {"minimax"} or kinds == {"mcts"} or len(kinds) == 1:
        return "ladder"
    return "mixed"


@dataclass(frozen=True)
class TournamentConfig:
    agents: tuple[str, ...]
    games_per_pairing: int = 4
    seed: int = 0
    workers: int = 0
    include_self_play: bool = True
    collect_series: bool = True
    allocation: str = "uniform"
    # Whether both orderings of a pairing play the *same* seeds.
    #
    # Off by default and it must stay that way: every committed run was scheduled with
    # seeds handed out by position, and turning this on would renumber them and replay
    # a different corpus under an unchanged name.
    #
    # On, it is common random numbers. A game is a pure function of
    # ``(ruleset, seed, agents)`` and ``seat_rng`` derives each seat's stream from the
    # seed and the *seat*, so swapping two agents between the seats of one seed leaves
    # both streams where they were. Two players that make the same choices therefore
    # play a move-for-move identical game in both orderings, one scores what the other
    # loses, and the pairing comes out at exactly 0.5 with no variance at all. What is
    # left when they differ is the difference, which is the quantity being measured.
    # See ``tournament/duel.py``, which is the estimator this exists for.
    paired: bool = False

    def __post_init__(self) -> None:
        if self.allocation not in ALLOCATIONS:
            raise ValueError(
                f"allocation must be one of {ALLOCATIONS}, got {self.allocation!r}"
            )

    def games_for(self, first: str, second: str) -> int:
        """How many games this ordered pairing gets."""
        if self.allocation == "uniform":
            return self.games_per_pairing
        return self.games_per_pairing * _LADDER_WEIGHTS[_pairing_kind(first, second)]

    def resolved_workers(self) -> int:
        """Threads to play on. ``0`` means "all but one core"."""
        return self.workers if self.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agents": list(self.agents),
            "games_per_pairing": self.games_per_pairing,
            "seed": self.seed,
            "include_self_play": self.include_self_play,
            "allocation": self.allocation,
            "paired": self.paired,
        }


@dataclass
class TournamentResult:
    ruleset_hash: str
    ruleset_name: str
    config: TournamentConfig
    records: list[GameRecord]
    ratings: RatingTable = field(init=False)

    def __post_init__(self) -> None:
        self.ratings = rate(self.records)

    def save(self, directory: str | Path) -> Path:
        import json

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_records(self.records, directory / "games.jsonl")
        with open(directory / "tournament.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "ruleset_hash": self.ruleset_hash,
                    "ruleset_name": self.ruleset_name,
                    "config": self.config.as_dict(),
                    "games": len(self.records),
                    "ratings": self.ratings.as_dict(),
                },
                handle,
                indent=2,
            )
        return directory


def schedule(config: TournamentConfig) -> list[tuple[int, str, str]]:
    """Every game to be played, as ``(seed, first agent, second agent)``.

    Seeds are assigned by position in the schedule, so the same config always produces
    the same set of games regardless of how they are distributed across workers -- and
    regardless of how many games each pairing was allotted, since the count is decided
    before any seed is handed out.

    ``config.paired`` swaps that for seeds keyed by the unordered pairing, which is the
    same guarantee said a different way: still fixed before any thread starts, still a
    pure function of the config, and additionally matched across the two colours.
    """
    for agent in config.agents:
        parse_agent(agent)  # fail on a bad name before any work is scheduled

    if config.paired:
        return _paired_schedule(config)

    tasks: list[tuple[int, str, str]] = []
    counter = 0
    for first in config.agents:
        for second in config.agents:
            if first == second and not config.include_self_play:
                continue
            for _ in range(config.games_for(first, second)):
                tasks.append((config.seed + counter, first, second))
                counter += 1
    return tasks


def _paired_schedule(config: TournamentConfig) -> list[tuple[int, str, str]]:
    """The same games, with the seeds keyed by the *unordered* pair rather than by row.

    Every unordered pairing gets one block of seeds, and both orderings play it. That is
    the only difference from the schedule above, and it is what makes the two games of a
    seed a matched pair: the same randomness reaches the same seats, so the luck is held
    fixed and the players are what varies.

    Both orderings must want the same number of games or a block would not cover them,
    which is true of both allocations -- ``uniform`` ignores the pairing and
    ``_pairing_kind`` reads an unordered set -- and is asserted rather than assumed,
    because a future allocation that broke it would silently reuse seeds.
    """
    tasks: list[tuple[int, str, str]] = []
    blocks: dict[tuple[str, str], tuple[int, int]] = {}
    cursor = 0
    for first in config.agents:
        for second in config.agents:
            if first == second and not config.include_self_play:
                continue
            key = (first, second) if first <= second else (second, first)
            count = config.games_for(first, second)
            block = blocks.get(key)
            if block is None:
                block = (cursor, count)
                blocks[key] = block
                cursor += count
            elif block[1] != count:
                raise ValueError(
                    f"a paired schedule gives {key} one block of seeds and the two "
                    f"orderings want {block[1]} and {count} games; the allocation is "
                    "not symmetric, so the pair cannot be matched"
                )
            start, _ = block
            for index in range(count):
                tasks.append((config.seed + start + index, first, second))
    return tasks


def run_tournament(
    spec: dict[str, Any],
    config: TournamentConfig,
    progress: bool = False,
    on_progress: Callable[[int], None] | None = None,
) -> TournamentResult:
    """Play the full schedule and rate the results.

    ``on_progress``, where given, is called with how many games are back each time some
    come back: after each piece the core returns, and after each game the Python loop
    plays. The core's part comes back first, and in one piece unless ``progress`` asks
    for twenty, so a caller that prints nothing hears once when the core is done and
    then once per outside game (Q446). It is told, never asked: nothing it does can
    change which games are played or their seeds.
    """
    tasks = schedule(config)
    engine = Engine(spec)
    natives = [native_spec(agent, engine.spec.hash) for agent in config.agents]
    records = _play(engine, config, tasks, natives, progress, on_progress)

    return TournamentResult(
        ruleset_hash=spec_hash(spec),
        ruleset_name=spec["name"],
        config=config,
        records=records,
    )


def _split_by_native_form(
    tasks: list[tuple[int, str, str]],
    agents: tuple[str, ...],
    natives: list[Any],
    core: bool = True,
) -> tuple[list[int], list[int]]:
    """Which games the core can play as one schedule, and which need the Python loop.

    Returned as two lists of positions in ``tasks``, in schedule order, which together
    cover every game once. A game goes to the core when both of its seats have a native
    form; a game with an outside seat -- Fairy-Stockfish, a custom agent -- stays in the
    loop, which plays it through `play_game` and so still runs any native seat in the
    core. With no core at all, everything is the loop's.

    The seeds are already fixed: this reads the schedule and never renumbers it, so a
    game has the seed it had before the split, and the split cannot change what it is.
    """
    native = {agent for agent, spec in zip(agents, natives, strict=True) if spec is not None}
    inside: list[int] = []
    outside: list[int] = []
    for index, (_, first, second) in enumerate(tasks):
        both = core and first in native and second in native
        (inside if both else outside).append(index)
    return inside, outside


def _play(
    engine: Engine,
    config: TournamentConfig,
    tasks: list[tuple[int, str, str]],
    natives: list[Any],
    progress: bool,
    on_progress: Callable[[int], None] | None = None,
) -> list[GameRecord]:
    """Every game of ``tasks``, returned in the order given; the core's part plays first.

    Before Q439 a single player with no native form sent every game here through the
    Python loop, serially, although that loop already handed each native pairing to the
    core one game at a time. So the same games came back, only slower: on the picker's
    rating run, one Fairy-Stockfish seat kept ours waiting for minutes (Q432). Each part
    now goes where it runs fastest, and each record is put back at its own position,
    so a reader of ``games.jsonl`` sees the order the schedule wrote.
    """
    if not tasks:
        return []
    inside, outside = _split_by_native_form(
        tasks, config.agents, natives, core=engine.core is not None
    )
    if not outside:
        return _run_native(engine, config, tasks, natives, progress, on_progress=on_progress)
    if not inside:
        return list(_run_python(
            engine, tasks, config.collect_series, progress, on_progress=on_progress
        ))

    if progress:
        away = sorted(
            {agent for agent, spec in zip(config.agents, natives, strict=True) if spec is None}
        )
        print(
            f"  {len(inside)} games in the core; {len(outside)} with "
            f"{', '.join(away)} one at a time",
            flush=True,
        )
    core_part = _run_native(
        engine, config, [tasks[index] for index in inside], natives, progress,
        total=len(tasks), on_progress=on_progress,
    )
    loop_part = _run_python(
        engine, [tasks[index] for index in outside], config.collect_series, progress,
        done=len(inside), total=len(tasks), on_progress=on_progress,
    )
    placed = dict(zip(inside, core_part, strict=True))
    placed.update(zip(outside, loop_part, strict=True))
    return [placed[index] for index in range(len(tasks))]


def _run_native(
    engine: Engine,
    config: TournamentConfig,
    tasks: list[tuple[int, str, str]],
    natives: list[Any],
    progress: bool,
    total: int | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> list[GameRecord]:
    """``tasks`` as schedules for the core, every seat of which must be native.

    The core is handed only the players that have a native form: an outside one has
    nothing it could convert, and it has no game here to be looked up for. Positions
    index that shorter list, which is the whole roster whenever everybody is native, so
    an all-native schedule is the call it always was. These are always the first games
    back, so ``on_progress`` is told the count as it stands.
    """
    roster = [
        (agent, spec) for agent, spec in zip(config.agents, natives, strict=True)
        if spec is not None
    ]
    position_of = {agent: position for position, (agent, _) in enumerate(roster)}
    specs = [spec for _, spec in roster]
    values = eval_values(engine.spec)
    workers = min(config.resolved_workers(), max(1, len(tasks)))
    size = max(workers, math.ceil(len(tasks) / PROGRESS_CHUNKS)) if progress else len(tasks)
    total = len(tasks) if total is None else total

    records: list[GameRecord] = []
    for start in range(0, len(tasks), size):
        chunk = tasks[start : start + size]
        results = engine.core.run_tournament(
            specs,
            values,
            [(seed, position_of[first], position_of[second]) for seed, first, second in chunk],
            config.collect_series,
            workers,
        )
        records.extend(
            build_record(engine, (first, second), seed, result)
            for (seed, first, second), result in zip(chunk, results, strict=True)
        )
        if progress:
            print(f"  {len(records)}/{total} games", flush=True)
        if on_progress is not None:
            on_progress(len(records))
    return records


def _run_python(
    engine: Engine,
    tasks: list[tuple[int, str, str]],
    collect_series: bool,
    progress: bool,
    done: int = 0,
    total: int | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> Iterator[GameRecord]:
    """``tasks`` one game at a time. ``done`` games of ``total`` were played elsewhere."""
    total = len(tasks) if total is None else total
    for index, (seed, first, second) in enumerate(tasks):
        # At index 0 after the core's part, its own last line already said this.
        if progress and index % 25 == 0 and (index or not done):
            print(f"  {done + index}/{total} games", flush=True)
        record = play_game(engine, (first, second), seed, collect_series=collect_series)
        if on_progress is not None:
            on_progress(done + index + 1)
        yield record
    if progress and tasks:
        print(f"  {done + len(tasks)}/{total} games", flush=True)
