"""Tier one: the legality job, which checks a score a visitor's browser measured.

`docs/archive/plan-client-compute.md` §5 and §8's row *Tier one, the legality job*. A browser runs
the whole package under Pyodide, plays and scores a draft, and sends what it played. The
server trusts none of it. This job re-derives everything that depends on positions, asking
the core at every ply, and it scores the records with its own metrics. The number it
stores is the server's. `provisional` holds the result afterwards, as `legal`, and shows it
to its designer alone.

## What arrives: B0's gathered text, slimmed

A measurement in the browser is B0's units (`tournament/slices.py`), and the text a phase's
units send back, concatenated in part order, is that phase's whole output. The upload is
those three texts and the stamp the browser made. `upload` writes it and `read_upload` reads it, and they are the only
writer and reader, so there is one form and not two:

    {"schema": 2, "basis": {...},
     "phases": {"tournament": "<JSON Lines>", "analysis": ..., "blunder": ...}}

It is *slim* because `upload` drops from each line the fields the walk rebuilds: a game's
per-ply series, and every analysis field but `near_best`. Those are half of `analysis.json`
and most of `games.jsonl`, and the position keys among them are the 64-bit values no
JavaScript number may carry (§3). Every dropped field has a default, so a slim line is
still a line `GameRecord.from_dict` and `GameAnalysis.from_dict` read, and a line that
keeps them is read the same way with the copies ignored. JavaScript neither writes nor
reads the text: Python writes it in the browser and Python reads it here.

## What the job checks, in order

1. **The basis.** `provisional.refusal`: the browser's stamp against the server's one-run
   basis. A stale cached module is refused by its build fingerprint before anything is
   walked, and the page fetches the current module.
2. **The schedule is the server's.** It builds the tournament's schedule itself and
   refuses any difference by name: a missing seed, a game the schedule does not hold, a
   seed played twice, other players, or another order.
3. **The walk asks the core at every ply** (`CLAUDE.md` rule 3). A move after a terminal
   position, a move outside `Engine.legal_moves`, moves that stop before the game ends, and
   a stated winner or reason that differ from the core's verdict are refused, each naming
   the game and the ply. The same walk rebuilds the series and the analysis fields. A
   game in rounds and a spy game are walked a step at a time, each round's two moves
   checked against their own seat's list and resolved together by the core, and the
   round, spy, sight and rest tallies rebuilt as the native pass writes them (Q1000; the
   rest's, `analysis.REST_SERIES`, since Q983b).
4. **The analysis covers the graded games, and the trials are the sampler's.** Which games
   the pass grades and which it forces a blunder into are functions of the records, so the
   server lists both and refuses a submission that covers others.
5. **The score is the server's.** `slices.gather_measurement` scores the walked records with
   the ordinary metrics, and that is the number `provisional.record` stores.

## What stays unproven

That the rostered players chose those moves; the search counters (`nodes`, `depth_sum`,
`searches`); the probe's `near_best` counts, beyond their shape; and each blunder trial's
new winner. Tier two replays a sample of all four. A forger who passes this tier has
written legal games by hand and invented those numbers, which is why a `legal` result is
shown to its designer alone.

## The job

A `jobs` row of kind `legality`. `file` refuses at once what needs no walk (an upload that
cannot be read, a stale basis, a game that is not the sender's, a second job in flight),
keeps the upload under `runs/provisional/uploads/`, and queues the row. `claim` takes the
oldest queued one, and `run` checks it and moves it to `done`, with a provisional result
that is `legal`, or to `refused`, with the refusal's name and sentence in `jobs.error`.
The job runs off the request thread, because §5 costs it at about ten CPU-seconds for
chess and a `ThreadingHTTPServer` thread held that long is a thread nobody else gets; it
runs ahead of every other kind of job.

Nothing here commits but `Checker`. The caller owns the transaction, as `provisional` and
`votes` do, and `sweep` deletes the uploads of finished jobs after the commit.

## The site's door (B10)

`POST /api/provisional` files a submission (`SiteApp._submit`): a refusal that needs no
walk is answered at once with its name, and `stale-build` -- a basis refused by its build
fingerprint alone -- is the one the page repairs, by fetching the current manifest.
`Checker` is the server's one runner: a daemon thread that `serve` starts, which claims,
runs and sweeps, each in its own transaction, and is woken by every filing.
`GET /api/provisional/<job>` is `status`, for the sender alone: the job's state, a
refusal's name and sentence, or the server's own number under its state.

## Tier two follows (Q503e)

A pass files the spot check of its result (`spotcheck.file`), drawn from the very text this
job read, unless a spot check has flagged the sender's account. `Checker` runs it on the
same thread once no legality job is queued, at `nice 19` where the platform has it
(`lower_this_thread`), within `quota.SAMPLE_JOB_CPU` and `quota.SAMPLE_JOB_WALL`. Every job
of either tier records its thread's processor time in `jobs.cpu_seconds`, and `file`
refuses a person whose rolling day of it is spent (`quota.CHECK_CPU_A_DAY`). `status`
carries the check's state as `spotCheck`, so the page knows to keep asking.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

from chess_variant_sandbox import corpus
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine, GameState, SpyPhase
from chess_variant_sandbox.engine.moves import move_from
from chess_variant_sandbox.measurement import Basis, built_fingerprint
from chess_variant_sandbox.metrics.scoring import WeightsError
from chess_variant_sandbox.site import provisional, quota, spine, spotcheck, usage
from chess_variant_sandbox.site.provisional import ProvisionalError
from chess_variant_sandbox.tournament import runner, slices
from chess_variant_sandbox.tournament.analysis import (
    BLUNDER_FRACTION,
    BLUNDER_SAMPLE,
    PLY_STRIDE,
    SPY_COUNTS,
    BlunderTrial,
    CorpusAnalysis,
    GameAnalysis,
    SpyStep,
    blunder_rng,
    count_rest,
    count_round,
    count_sight,
    mover_kind,
    moves_an_instance,
    rest_tally,
    round_blunder,
    round_tallies,
    sight_tally,
    spy_holders,
    spy_loss,
    spy_round_blunder,
    starting_squares,
    takes,
)
from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.tournament.runner import TournamentConfig

# The upload's form is written in the wheel, by the one worker that gathers (B10), so its
# writer and its constants live in `slices`, which a browser imports; `site/` it does not.
SCHEMA = slices.UPLOAD_SCHEMA
"""The upload's own version. A reader that does not know it refuses the upload."""

REBUILT_RECORD_FIELDS = slices.REBUILT_RECORD_FIELDS
REBUILT_ANALYSIS_FIELDS = slices.REBUILT_ANALYSIS_FIELDS

# What a refusal is called. The name is what a test, a log line and the page can key on;
# the sentence beside it names the game and the ply.
BASIS = "basis"
STALE_BUILD = "stale-build"
"""A basis refused by its build fingerprint alone: the browser ran a core built from other
sources, which is a stale cached manifest and wheel. It is named apart because it has a
remedy the page acts on -- fetch the current manifest and measure again (B10)."""
UNREADABLE = "unreadable"
MISSING_SEED = "missing-seed"
UNSCHEDULED = "unscheduled-game"
WRONG_PLAYERS = "wrong-players"
OUT_OF_ORDER = "out-of-order"
ILLEGAL_MOVE = "illegal-move"
AFTER_TERMINAL = "move-after-terminal"
UNFINISHED = "unfinished"
WRONG_OUTCOME = "wrong-outcome"
ANALYSIS = "analysis"
BLUNDER = "blunder"
STORE = "store"
NAMES: tuple[str, ...] = (
    BASIS,
    STALE_BUILD,
    UNREADABLE,
    MISSING_SEED,
    UNSCHEDULED,
    WRONG_PLAYERS,
    OUT_OF_ORDER,
    ILLEGAL_MOVE,
    AFTER_TERMINAL,
    UNFINISHED,
    WRONG_OUTCOME,
    ANALYSIS,
    BLUNDER,
    STORE,
)

# How each playing phase is named in a sentence.
_CALLED = {
    slices.TOURNAMENT: "the tournament",
}


class Refused(ValueError):
    """A submission the server will not score, by name, with the game and ply where known.

    A `ValueError`, like `provisional.ProvisionalError`, so a route answers 400 with the
    sentence. `str()` is the sentence; `logged` is what `jobs.error` keeps.
    """

    def __init__(
        self,
        name: str,
        sentence: str,
        *,
        phase: str | None = None,
        game: int | None = None,
        seed: int | None = None,
        ply: int | None = None,
    ) -> None:
        super().__init__(sentence)
        self.name = name
        self.sentence = sentence
        self.phase = phase
        self.game = game
        self.seed = seed
        self.ply = ply

    @property
    def logged(self) -> str:
        return f"{self.name}: {self.sentence}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sentence": self.sentence,
            "phase": self.phase,
            "game": self.game,
            "seed": self.seed,
            "ply": self.ply,
        }


# ------------------------------------------------------------------ the upload's form


@dataclass(frozen=True)
class Upload:
    """What a browser sent: its stamp and each phase's gathered text."""

    basis: Mapping[str, Any] | None
    phases: Mapping[str, str]


upload = slices.upload
"""The slim upload's writer, `slices.upload`: B0's gathered text for each phase, less
what the walk rebuilds. The browser calls it under Pyodide, in the gathering half's last
step, so the text is Python's from end to end."""


def read_upload(text: str | bytes) -> Upload:
    """The upload as sent, or a refusal saying what could not be read. Nothing is walked."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as error:
        raise Refused(UNREADABLE, "the upload is not JSON") from error
    if not isinstance(data, dict):
        raise Refused(UNREADABLE, "the upload is not one JSON object")
    if data.get("schema") != SCHEMA:
        raise Refused(
            UNREADABLE,
            f"the upload is in form {data.get('schema')!r} and this server reads form "
            f"{SCHEMA}, so the page fetches the current module",
        )
    phases = data.get("phases")
    if not isinstance(phases, dict):
        raise Refused(UNREADABLE, "the upload carries no phases")
    unknown = sorted(str(name) for name in phases if name not in slices.PHASES)
    if unknown:
        raise Refused(UNREADABLE, f"the upload carries phases no measurement has: {unknown}")
    missing = [phase for phase in slices.PHASES if phase not in phases]
    if missing:
        raise Refused(UNREADABLE, f"the upload lacks the phases {missing}")
    if not all(isinstance(value, str) for value in phases.values()):
        raise Refused(UNREADABLE, "each phase is carried as the text its units sent back")
    basis = data.get("basis")
    return Upload(
        basis=basis if isinstance(basis, dict) else None,
        phases={str(name): str(value) for name, value in phases.items()},
    )


# --------------------------------------------------------------------- the check


@dataclass
class Checked:
    """A submission that passed: the server's own copies of everything it re-derived."""

    spec: dict[str, Any]
    roster: TournamentConfig
    records: list[GameRecord]
    analysis: CorpusAnalysis
    # The server's one-run basis: the stamp the submission was held to, and the one stored.
    basis: Basis | None = None


@dataclass
class Scored:
    """The server's number for a checked submission, and the files it stores."""

    headline: float | None
    viable: bool | None
    files: dict[str, str] = field(default_factory=dict)
    unscored: str = ""
    # False only for *not measured* (plan-teacher §10.16.4 point 4).
    measured: bool = True


def refuse_basis(submitted: Mapping[str, Any] | None, current: Basis) -> None:
    """`provisional.refusal` as a `Refused`: `stale-build` when the build alone differs."""
    reason = provisional.refusal(submitted, current)
    if reason is None:
        return
    name = STALE_BUILD if provisional.other_build(submitted, current) else BASIS
    raise Refused(name, f"the result was refused: {reason}")


def check(spec: dict[str, Any], sent: Upload, current: Basis) -> Checked:
    """Walk, rebuild and re-derive a submission, or refuse it by name.

    `current` is the server's basis for this game's one-run measurement, as it would stamp
    its own (`corpus.current_basis(..., seeds=corpus.ONE_SEED)`); its roster, games per
    pairing and seed are the schedule the tournament is held to.
    """
    refuse_basis(sent.basis, current)
    engine = Engine(spec)
    if engine.spec.hash != current.ruleset_hash:
        raise Refused(
            BASIS, f"the basis is for the game {current.ruleset_hash}, not {engine.spec.hash}"
        )
    roster = roster_of(current)

    tournament_sent = _records(sent.phases[slices.TOURNAMENT], slices.TOURNAMENT)
    analyses_sent = _analyses(sent.phases[slices.ANALYSIS])
    trials_sent = _trials(sent.phases[slices.BLUNDER])

    # The schedule, before walking anything.
    same_schedule(slices.TOURNAMENT, tournament_sent, runner.schedule(roster))

    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    walked = [
        walk(
            engine,
            evaluator,
            record,
            phase=slices.TOURNAMENT,
            game=index,
            series=roster.collect_series,
        )
        for index, record in enumerate(tournament_sent)
    ]

    records = [step.record for step in walked]
    analyses = rebuilt_analyses(records, walked, analyses_sent)
    trials = rebuilt_trials(records, walked, trials_sent, roster.seed, engine)
    return Checked(
        spec=spec,
        roster=roster,
        records=records,
        analysis=CorpusAnalysis(games=analyses, blunders=trials),
        basis=current,
    )


def roster_of(current: Basis) -> TournamentConfig:
    """The tournament a one-run basis names: its roster, games a pairing and one seed."""
    if len(current.seeds) != 1:
        raise Refused(
            BASIS,
            f"a browser measures one run, and this basis names the seeds {list(current.seeds)}",
        )
    return slices.canonical_config(
        agents=tuple(current.agents),
        games_per_pairing=current.games_per_pairing,
        seed=current.seeds[0],
    )


def score(checked: Checked, weights: str) -> Scored:
    """The server's own score of the walked records, and the files a result carries.

    `slices.gather_measurement` writes the four files a run directory holds and scores
    them with the ordinary metrics.
    """
    with tempfile.TemporaryDirectory(prefix="legality-") as scratch:
        out = Path(scratch)
        try:
            gathered = slices.gather_measurement(
                checked.spec,
                out,
                checked.records,
                checked.analysis,
                config=checked.roster,
                weights=weights,
            )
            headline: float | None = gathered.index
            viable: bool | None = bool(gathered.result.viable)
            measured = bool(gathered.result.measured)
            unscored = ""
        except WeightsError as error:
            # A run the scorer cannot read is a fact about the run: it carries the
            # sentence and no number, as `corpus._score_run` keeps it.
            headline, viable, unscored, measured = None, None, str(error), True
        files = {
            path.name: path.read_text(encoding="utf-8")
            for path in sorted(out.iterdir())
            if path.is_file()
        }
    return Scored(headline=headline, viable=viable, files=files, unscored=unscored,
                  measured=measured)


# ------------------------------------------------------------------- reading lines


def _lines(text: str, phase: str) -> list[dict[str, Any]]:
    try:
        payloads = slices._parse(text)
    except slices.SliceError as error:
        raise Refused(UNREADABLE, f"{phase}: {error}", phase=phase) from error
    for number, payload in enumerate(payloads, start=1):
        if not isinstance(payload, dict):
            raise Refused(
                UNREADABLE, f"line {number} of {phase} is not one JSON object", phase=phase
            )
    return payloads


def _records(text: str, phase: str) -> list[GameRecord]:
    """A playing phase's games, each held to the shape a record has before it is walked."""
    records = []
    for index, payload in enumerate(_lines(text, phase)):
        try:
            record = GameRecord.from_dict(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise Refused(
                UNREADABLE,
                f"game {index} of {_CALLED[phase]} is not a game record ({error})",
                phase=phase,
                game=index,
            ) from error
        problem = _shape(record)
        if problem:
            raise Refused(
                UNREADABLE,
                f"game {index} of {_CALLED[phase]} {problem}",
                phase=phase,
                game=index,
            )
        records.append(record)
    return records


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _shape(record: GameRecord) -> str:
    """What is wrong with a record's types, or `""`. The walk trusts nothing it reads."""
    if not _is_int(record.seed):
        return "has no whole-number seed"
    if len(record.agents) != 2 or not all(isinstance(a, str) for a in record.agents):
        return "does not name two players"
    if not isinstance(record.moves, list) or not all(_is_int(m) for m in record.moves):
        return "does not list its moves as whole numbers"
    if record.winner is not None and not (_is_int(record.winner) and record.winner in (0, 1)):
        return "names a winner that is neither seat"
    if not isinstance(record.reason, str):
        return "gives no reason for its ending"
    for pair in (record.nodes, record.depth_sum, record.searches):
        if len(pair) != 2 or not all(_is_int(value) for value in pair):
            return "does not count its searches as two whole numbers"
    return ""


def _analyses(text: str) -> list[GameAnalysis]:
    found = []
    for index, payload in enumerate(_lines(text, slices.ANALYSIS)):
        try:
            analysis = GameAnalysis.from_dict(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise Refused(
                UNREADABLE,
                f"analysis {index} is not a game's analysis ({error})",
                phase=slices.ANALYSIS,
                game=index,
            ) from error
        if not _is_int(analysis.seed) or not isinstance(analysis.near_best, list):
            raise Refused(
                UNREADABLE, f"analysis {index} is not a game's analysis", phase=slices.ANALYSIS
            )
        found.append(analysis)
    return found


def _trials(text: str) -> list[BlunderTrial]:
    found = []
    for index, payload in enumerate(_lines(text, slices.BLUNDER)):
        try:
            found.append(BlunderTrial.from_dict(payload))
        except (KeyError, TypeError, ValueError) as error:
            raise Refused(
                UNREADABLE,
                f"blunder trial {index} is not a trial ({error})",
                phase=slices.BLUNDER,
                game=index,
            ) from error
    return found



# ------------------------------------------------------------------ the schedules


def same_schedule(
    phase: str, records: Sequence[GameRecord], tasks: Sequence[tuple[int, str, str]]
) -> None:
    """Refuse, by name, any difference between the games sent and the server's schedule.

    Names the first missing seed before anything else, because a dropped game is the
    forgery this check exists for; then a game the schedule does not hold or holds once
    and was sent twice, then other players at a seed, then the order.
    """
    called = _CALLED[phase]
    wanted = {seed: (first, second) for seed, first, second in tasks}
    sent: dict[int, int] = {}
    for index, record in enumerate(records):
        if record.seed not in wanted:
            raise Refused(
                UNSCHEDULED,
                f"game {index} of {called} has seed {record.seed}, which its schedule does "
                "not hold",
                phase=phase,
                game=index,
                seed=record.seed,
            )
        if record.seed in sent:
            raise Refused(
                UNSCHEDULED,
                f"games {sent[record.seed]} and {index} of {called} both have seed "
                f"{record.seed}",
                phase=phase,
                game=index,
                seed=record.seed,
            )
        sent[record.seed] = index
    for position, (seed, _first, _second) in enumerate(tasks):
        if seed not in sent:
            raise Refused(
                MISSING_SEED,
                f"{called} is missing seed {seed}, game {position} of its schedule",
                phase=phase,
                game=position,
                seed=seed,
            )
    for index, record in enumerate(records):
        if tuple(record.agents) != wanted[record.seed]:
            first, second = wanted[record.seed]
            raise Refused(
                WRONG_PLAYERS,
                f"game {index} of {called} (seed {record.seed}) is "
                f"{record.agents[0]} against {record.agents[1]}, and its schedule has "
                f"{first} against {second}",
                phase=phase,
                game=index,
                seed=record.seed,
            )
    for index, (record, task) in enumerate(zip(records, tasks, strict=True)):
        if record.seed != task[0]:
            raise Refused(
                OUT_OF_ORDER,
                f"game {index} of {called} has seed {record.seed} where its schedule has "
                f"seed {task[0]}",
                phase=phase,
                game=index,
                seed=record.seed,
            )


# ---------------------------------------------------------------------- the walk


class Step(NamedTuple):
    """One step of a game that is not walked a ply at a time (Q1000): a turn's one move or
    a round's two in seat order, the position they were chosen on and each mover's legal
    moves there, the position after, the round's events, and a spy game's phase (`None` in
    a game in rounds, which has no phases)."""

    ply: int
    state: GameState
    moves: tuple[int, ...]
    lists: tuple[list[int], ...]
    after: GameState
    events: tuple[tuple[int, int, int], ...]
    phase: SpyPhase | None

    def as_spy_step(self) -> SpyStep:
        assert self.phase is not None
        return SpyStep(
            self.ply, self.state, self.phase, self.moves, self.lists, self.after, self.events
        )


@dataclass
class Walked:
    """One game walked: the server's copy of it, and what the positions said on the way."""

    record: GameRecord
    # Every analysis field but `near_best`, as `analysis._analyse_game` reads them.
    positions: GameAnalysis
    # The side to move and the number of legal moves at the ply a blunder trial forces
    # its move into, which is all of an alternating game's trial the positions decide.
    blunder_side: int = 0
    blunder_choices: int = 0
    # In a game in rounds or a spy game, the step the trial's ply falls in, or `None` when
    # no step holds it. Whether such a trial forces anything is the probe's reading of that
    # step (`trial_side`), which only a game the sampler draws pays for.
    trial: Step | None = None
    stepped: bool = False


def trial_side(engine: Engine | None, walked: Walked, seed: int) -> int | None:
    """The side a game's blunder trial forces its move into, or `None` where the pass's
    trial (`analysis._one_blunder`) finds none to force.

    An alternating game's is the walk's: a trial needs two legal moves at its ply. A turn of
    a spy game reads the same. A round's is the pass's own reading of its step, every part
    of `_one_blunder` but the playout: `analysis.round_blunder` (a move short of the best
    against the mix, other than the one played) in a game in rounds or a blind round, and
    `analysis.spy_round_blunder` in an announced one, where a blunder forced into the
    announcer is answered by the record's player, asked of the core. Each draws on the
    trial's own stream (`blunder_rng`), as the pass's does, so the same move is drawn.
    `engine` is needed only for a game walked in steps.
    """
    if not walked.stepped:
        return walked.blunder_side if walked.blunder_choices >= 2 else None
    if engine is None:
        raise ValueError("a game walked in steps needs its engine to draw its trial")
    step = walked.trial
    if step is None:
        return None
    if len(step.moves) == 1:
        return step.state.side_to_move if len(step.lists[0]) >= 2 else None
    record = walked.record
    seat = int(record.plies * BLUNDER_FRACTION) - step.ply
    rng = blunder_rng(seed, record)
    if step.phase is not None:
        pair = spy_round_blunder(engine, record, step.as_spy_step(), seat, rng)
        return None if pair is None else seat
    forced = round_blunder(engine, step.state, seat, step.moves[seat], rng)
    return None if forced is None else seat


def walk(
    engine: Engine,
    evaluator: Evaluator,
    record: GameRecord,
    *,
    phase: str,
    game: int,
    series: bool = False,
) -> Walked:
    """Walk one game, asking the core at every ply, and rebuild what the positions decide.

    Python holds the position and the core answers every question: which moves are
    legal, whether the game is over and how it ended. A move after a terminal position,
    a move outside the legal list, moves that stop before a verdict and a verdict other
    than the stated one are refused, naming the game and the ply.

    `series` rebuilds the per-ply `branching`, `material` and `captures` the way
    `play.play_game_python` records them, which `tests/test_backends.py` holds to the
    core's own. Without it the series are empty.

    The analysis fields are `analysis._analyse_game`'s reading of the same positions,
    without the probe search that only `near_best` needs, the sight tallies
    (`analysis.SIGHT_SERIES`) among them where the game has a line of sight.
    `tests/test_site_legality.py` holds this copy to that one field by field.

    A game in rounds and a spy game are walked a step at a time (`_walk_steps`, Q1000).
    """
    if engine.spy or engine.simultaneous:
        return _walk_steps(engine, evaluator, record, phase=phase, game=game, series=series)
    called = _CALLED[phase]
    n_types = engine.spec.n_types
    moves = list(record.moves)
    target = int(len(moves) * BLUNDER_FRACTION)

    state = engine.initial_state()
    starting = starting_squares(engine, state.board)
    moved: set[int] = set()
    seen = {state.key}
    keys: list[int] = []
    revisited = 0
    moves_by_type = [0] * n_types
    captures_by_type = [0] * n_types
    legal_counts: list[int] = []
    capture_counts: list[int] = []
    counting = True
    branching: list[int] = []
    material: list[float] = []
    captures: list[int] = []
    blunder = (0, 0)
    sight = sight_tally(engine)
    rest = rest_tally(engine)

    for ply, move in enumerate(moves):
        legal = engine.legal_moves(state)
        outcome = engine.terminal(state, legal)
        if outcome is not None:
            raise Refused(
                AFTER_TERMINAL,
                f"game {game} of {called} (seed {record.seed}) plays on at ply {ply}, "
                f"after the game ended ({_ending(outcome.winner, outcome.reason)})",
                phase=phase,
                game=game,
                seed=record.seed,
                ply=ply,
            )
        if move not in legal:
            raise Refused(
                ILLEGAL_MOVE,
                f"game {game} of {called} (seed {record.seed}) plays an illegal move at "
                f"ply {ply}",
                phase=phase,
                game=game,
                seed=record.seed,
                ply=ply,
            )
        if ply == target:
            blunder = (state.side_to_move, len(legal))
        if series:
            branching.append(len(legal))
            material.append(round(evaluator.material(state, 0), 4))

        board = state.board
        # The pass's own readings of a move (`analysis.mover_kind`, `moves_an_instance`,
        # `starting_squares`, `takes`), so the two cannot drift apart again as they did
        # over a fold's own move (Q875).
        kind = mover_kind(engine, board, move) if counting else None
        if kind is None:
            # `_analyse_game` stops counting at a move from an empty square, saying the
            # record and the spec have drifted apart. A legal move does not start on one
            # (a fold's own move is read by its end, not its origin), so this is kept only
            # so that the two read every game alike.
            counting = False
        else:
            count_sight(engine, sight, state, state.side_to_move)
            count_rest(engine, rest, state, state.side_to_move)
            if moves_an_instance(move) and move_from(move) in starting:
                moved.add(move_from(move))
            moves_by_type[kind] += 1
            if takes(engine, board, move):
                captures_by_type[kind] += 1
            if ply % PLY_STRIDE == 0 and len(legal) > 1:
                legal_counts.append(len(legal))
                capture_counts.append(sum(1 for m in legal if takes(engine, board, m)))

        successor = engine.apply(state, move)
        if series:
            captures.append(1 if successor.halfmove == 0 else 0)
        state = successor
        if counting:
            keys.append(state.key)
            if state.key in seen:
                revisited += 1
            else:
                seen.add(state.key)

    legal = engine.legal_moves(state)
    outcome = engine.terminal(state, legal)
    if target == len(moves):
        blunder = (state.side_to_move, len(legal))
    _held_to_its_verdict(outcome, record, phase=phase, game=game, plies=len(moves))

    rebuilt = replace(
        record,
        ruleset_hash=engine.spec.hash,
        ruleset_name=engine.spec.name,
        agents=(str(record.agents[0]), str(record.agents[1])),
        moves=moves,
        plies=len(moves),
        branching=branching,
        material=material,
        captures=captures,
    )
    positions = GameAnalysis(
        seed=record.seed,
        agents=rebuilt.agents,
        near_best=[],
        legal_counts=legal_counts,
        moves_by_type=moves_by_type,
        captures_by_type=captures_by_type,
        capture_counts=capture_counts,
        revisited_plies=revisited,
        walked_plies=len(keys),
        starting_instances=len(starting),
        moved_instances=len(moved),
        position_keys=keys,
        sight=sight,
        rest=rest,
    )
    return Walked(
        record=rebuilt,
        positions=positions,
        blunder_side=blunder[0],
        blunder_choices=blunder[1],
    )


def _held_to_its_verdict(
    outcome: Any, record: GameRecord, *, phase: str, game: int, plies: int
) -> None:
    """Refuse a game whose moves stop before the core's verdict, or end in another one."""
    called = _CALLED[phase]
    if outcome is None:
        raise Refused(
            UNFINISHED,
            f"game {game} of {called} (seed {record.seed}) stops after ply "
            f"{plies - 1}, and the game is not over there",
            phase=phase,
            game=game,
            seed=record.seed,
            ply=plies,
        )
    if outcome.winner != record.winner or outcome.reason != record.reason:
        raise Refused(
            WRONG_OUTCOME,
            f"game {game} of {called} (seed {record.seed}) ends in "
            f"{_ending(outcome.winner, outcome.reason)}, and the submission says "
            f"{_ending(record.winner, record.reason)}",
            phase=phase,
            game=game,
            seed=record.seed,
            ply=plies,
        )


def _walk_steps(
    engine: Engine,
    evaluator: Evaluator,
    record: GameRecord,
    *,
    phase: str,
    game: int,
    series: bool,
) -> Walked:
    """`walk` for a game in rounds (schema 26) and a spy game (schema 27), a step at a time.

    A step is a round in a game in rounds; in a spy game it is what the core's phase says,
    a turn of one move or a round of two. A round's two moves are each checked against
    their own seat's legal list on the round's board and resolved together by the core
    (`Engine.apply_round_with_phase`); a record cut inside a round is refused as
    unfinished. The series are the round loop's (`play._play_rounds_python`,
    `_play_spy_python`): once a turn, and twice a round, each seat's own branching. The
    analysis fields are `analysis._analyse_rounds`' and `_analyse_spy`'s reading, the
    round and spy tallies and the sights among them, through the pass's own counters
    (`count_round`, `count_sight`, `spy_holders`, `spy_loss`). A step is sampled where the
    pass samples it, and gives a sample for each seat with a choice when both have a move,
    which is when the pass's reading of a round exists.
    """
    called = _CALLED[phase]
    n_types = engine.spec.n_types
    moves = list(record.moves)
    target = int(len(moves) * BLUNDER_FRACTION)
    spy = engine.spy

    state = engine.initial_state()
    starting = starting_squares(engine, state.board)
    moved: set[int] = set()
    seen = {state.key}
    keys: list[int] = []
    revisited = 0
    moves_by_type = [0] * n_types
    captures_by_type = [0] * n_types
    legal_counts: list[int] = []
    capture_counts: list[int] = []
    branching: list[int] = []
    material: list[float] = []
    captures: list[int] = []
    rounds = round_tallies(engine)
    spies: dict[str, Any] | None = (
        {key: (None if key.startswith("lost_") else 0) for key in SPY_COUNTS} if spy else None
    )
    sight = sight_tally(engine)
    rest = rest_tally(engine)
    trial: Step | None = None

    def refuse(name: str, sentence: str, ply: int) -> Refused:
        return Refused(
            name,
            f"game {game} of {called} (seed {record.seed}) {sentence}",
            phase=phase,
            game=game,
            seed=record.seed,
            ply=ply,
        )

    ply = 0
    while True:
        step_phase = engine.phase(state) if spy else None
        if step_phase is None or step_phase.is_round:
            lists: tuple[list[int], ...] = (
                engine.legal_moves_for(state, 0),
                engine.legal_moves_for(state, 1),
            )
            outcome = engine.terminal(state)
        else:
            lists = (engine.legal_moves(state),)
            outcome = engine.terminal(state, lists[0])
        if ply >= len(moves):
            break
        if outcome is not None:
            raise refuse(
                AFTER_TERMINAL,
                f"plays on at ply {ply}, after the game ended "
                f"({_ending(outcome.winner, outcome.reason)})",
                ply,
            )
        words = tuple(moves[ply:ply + len(lists)])
        if len(words) < len(lists):
            raise refuse(
                UNFINISHED,
                f"stops after ply {len(moves) - 1}, one move into a round of two",
                len(moves),
            )
        for offset, (move, legal) in enumerate(zip(words, lists, strict=True)):
            if move not in legal:
                whose = f" (seat {offset}'s move in a round)" if len(words) == 2 else ""
                raise refuse(
                    ILLEGAL_MOVE, f"plays an illegal move at ply {ply + offset}{whose}",
                    ply + offset,
                )

        board = state.board
        seats = (state.side_to_move,) if len(words) == 1 else (0, 1)
        if series:
            branching.extend(len(legal) for legal in lists)
            value = round(evaluator.material(state, 0), 4)
            material.extend([value] * len(words))
        for seat in seats:
            count_sight(engine, sight, state, seat)
            count_rest(engine, rest, state, seat)
        for move in words:
            kind = mover_kind(engine, board, move)
            if kind is None:  # pragma: no cover - a legal move stands on its origin
                continue
            if moves_an_instance(move) and move_from(move) in starting:
                moved.add(move_from(move))
            moves_by_type[kind] += 1
            if takes(engine, board, move):
                captures_by_type[kind] += 1

        # The pass's sampling: a ply of the step on the stride in a spy game (a round
        # there may start on an odd ply), the round's first ply in a game in rounds.
        sampled = (
            any((ply + k) % PLY_STRIDE == 0 for k in range(len(words)))
            if spy
            else ply % PLY_STRIDE == 0
        )
        if sampled and all(lists):
            for legal in lists:
                if len(legal) > 1:
                    legal_counts.append(len(legal))
                    capture_counts.append(sum(1 for m in legal if takes(engine, board, m)))

        events: tuple[tuple[int, int, int], ...] = ()
        if len(words) == 2:
            after, _, events = engine.apply_round_with_phase(state, words[0], words[1])
        else:
            after = engine.apply(state, words[0])
        if series:
            captures.extend([1 if after.halfmove == 0 else 0] * len(words))
        step = Step(ply, state, words, lists, after, events, step_phase)
        if ply <= target < ply + len(words):
            trial = step

        if len(words) == 2:
            count_round(rounds, words, events)
        if spies is not None and step_phase is not None:
            if len(words) == 1:
                turn = "alternating_plies" if step_phase.name == "alternating" else "reply_turns"
                spies[turn] += 1
            else:
                spies[f"{step_phase.name}_rounds"] += 1
        # One key a turn and the key after a round twice, a revisit counted on both, as
        # every per-ply series is written.
        keys.extend([after.key] * len(words))
        if after.key in seen:
            revisited += len(words)
        else:
            seen.add(after.key)
        if spies is not None and step_phase is not None:
            lost = spy_holders(step_phase, state.side_to_move) - spy_holders(
                engine.phase(after), after.side_to_move
            )
            for player in sorted(lost):
                spies[f"lost_ply_{player}"] = len(keys)
                spies[f"lost_how_{player}"] = spy_loss(engine, step.as_spy_step(), player)

        state = after
        ply += len(words)

    _held_to_its_verdict(outcome, record, phase=phase, game=game, plies=len(moves))
    rebuilt = replace(
        record,
        ruleset_hash=engine.spec.hash,
        ruleset_name=engine.spec.name,
        agents=(str(record.agents[0]), str(record.agents[1])),
        moves=moves,
        plies=len(moves),
        branching=branching,
        material=material,
        captures=captures,
    )
    positions = GameAnalysis(
        seed=record.seed,
        agents=rebuilt.agents,
        near_best=[],
        legal_counts=legal_counts,
        moves_by_type=moves_by_type,
        captures_by_type=captures_by_type,
        capture_counts=capture_counts,
        revisited_plies=revisited,
        walked_plies=len(keys),
        starting_instances=len(starting),
        moved_instances=len(moved),
        position_keys=keys,
        rounds=rounds,
        spy=spies,
        sight=sight,
        rest=rest,
    )
    return Walked(record=rebuilt, positions=positions, trial=trial, stepped=True)


def _ending(winner: int | None, reason: str) -> str:
    who = "a draw" if winner is None else f"a win for seat {winner}"
    return f"{who} by {reason}"


# ---------------------------------------------------------- the analysis and trials


def rebuilt_analyses(
    records: Sequence[GameRecord], walked: Sequence[Walked], sent: Sequence[GameAnalysis]
) -> list[GameAnalysis]:
    """The analysis the server stores: its own reading of the positions, the sent probe.

    The graded games are the pass's (`slices.analysis_units`), in its order, and the
    submission has to cover exactly those. `near_best` is the one field kept from it, and
    only in the shape it can have: one count per sampled ply, each at least one (the best
    move is within the margin of itself) and at most the legal moves there.
    """
    units = slices.analysis_units(records, 1)
    graded = list(units[0].games) if units else []
    if len(sent) != len(graded):
        raise Refused(
            ANALYSIS,
            f"the analysis covers {len(sent)} games, and the pass grades {len(graded)}",
            phase=slices.ANALYSIS,
        )
    found: list[GameAnalysis] = []
    for position, (index, analysis) in enumerate(zip(graded, sent, strict=True)):
        record = records[index]
        if analysis.seed != record.seed or tuple(analysis.agents) != record.agents:
            raise Refused(
                ANALYSIS,
                f"analysis {position} is of seed {analysis.seed}, and the pass's game "
                f"{position} is tournament game {index} (seed {record.seed})",
                phase=slices.ANALYSIS,
                game=index,
                seed=record.seed,
            )
        own = walked[index].positions
        counts = analysis.near_best
        if len(counts) != len(own.legal_counts) or not all(
            _is_int(count) and 1 <= count <= legal
            for count, legal in zip(counts, own.legal_counts, strict=False)
        ):
            raise Refused(
                ANALYSIS,
                f"the probe counts for tournament game {index} (seed {record.seed}) are "
                f"not one count between one and the legal moves at each of its "
                f"{len(own.legal_counts)} sampled plies",
                phase=slices.ANALYSIS,
                game=index,
                seed=record.seed,
            )
        found.append(replace(own, near_best=list(counts)))
    return found


def rebuilt_trials(
    records: Sequence[GameRecord],
    walked: Sequence[Walked],
    sent: Sequence[BlunderTrial],
    seed: int,
    engine: Engine | None = None,
) -> list[BlunderTrial]:
    """The blunder trials the server stores: the sampler's games, the sent new winners.

    Which games get a trial is `_blunder_choices`' draw over the records, and a chosen
    game whose trial finds no move to force has none, as `analysis._one_blunder` rules
    (`trial_side`: fewer than two legal moves at the trial's ply, or in a round no move
    short of the best). The ply, the original winner and the side made to blunder are the
    walk's. The new winner is the one field a trial's playout decides, and tier two
    replays a sample of them. `engine` is needed only for a game walked in steps, whose
    rounds the probe reads.
    """
    units = slices.blunder_units(records, 1, BLUNDER_SAMPLE, seed)
    sides: dict[int, int] = {}
    for index in units[0].games if units else ():
        side = trial_side(engine, walked[index], seed)
        if side is not None:
            sides[index] = side
    chosen = list(sides)
    if len(sent) != len(chosen):
        raise Refused(
            BLUNDER,
            f"the submission has {len(sent)} blunder trials, and the sampler draws "
            f"{len(chosen)} from these games",
            phase=slices.BLUNDER,
        )
    found: list[BlunderTrial] = []
    for position, (index, trial) in enumerate(zip(chosen, sent, strict=True)):
        record = records[index]
        if trial.seed != record.seed or tuple(trial.agents) != record.agents:
            raise Refused(
                BLUNDER,
                f"blunder trial {position} is of seed {trial.seed}, and the sampler's "
                f"trial {position} is tournament game {index} (seed {record.seed})",
                phase=slices.BLUNDER,
                game=index,
                seed=record.seed,
            )
        if trial.new_winner is not None and (
            not _is_int(trial.new_winner) or trial.new_winner not in (0, 1)
        ):
            raise Refused(
                BLUNDER,
                f"blunder trial {position} names a winner that is neither seat",
                phase=slices.BLUNDER,
                game=index,
                seed=record.seed,
            )
        found.append(
            BlunderTrial(
                seed=record.seed,
                agents=record.agents,
                ply=int(record.plies * BLUNDER_FRACTION),
                original_winner=record.winner,
                blundering_side=sides[index],
                new_winner=trial.new_winner,
            )
        )
    return found


# ----------------------------------------------------------------------- the job

KIND = provisional.WALK
"""`jobs.kind` for tier one. It runs ahead of every other kind (§5)."""

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
REFUSED = "refused"
FAILED = "failed"
JOB_STATES: tuple[str, ...] = (QUEUED, RUNNING, DONE, REFUSED, FAILED)
IN_FLIGHT: tuple[str, ...] = (QUEUED, RUNNING)

UPLOADS = "uploads"
"""Where a queued upload waits, under `runs/provisional/`. Never a result's directory."""


class LegalityError(ValueError):
    """A job that cannot be filed or run as asked. A route answers 400 with the sentence."""


@dataclass(frozen=True)
class Finished:
    """How one job ended: `done` with a result, `refused` by name, or `failed`."""

    job: int
    state: str
    result: provisional.Result | None = None
    refusal: Refused | None = None
    error: str = ""
    sample: int | None = None
    """The spot check a pass filed (`spotcheck.file`), or `None` for a flagged account."""


Now = Callable[[str, dict[str, Any]], Basis]
"""The server's basis for a game's one-run measurement, from its hash and its rules."""


def one_run_basis(game: str, spec: dict[str, Any]) -> Basis:
    """What the server would stamp on its own one-run measurement of this game now."""
    return corpus.current_basis(game, spec=spec, seeds=corpus.ONE_SEED)


def upload_path(runs_dir: Path, job: int) -> Path:
    return Path(runs_dir) / provisional.DIRECTORY / UPLOADS / f"{int(job)}.json"


def file(
    connection: Any,
    runs_dir: Path,
    *,
    game: str,
    submitter: int,
    text: str | bytes,
    basis: Now = one_run_basis,
) -> int:
    """Queue a submission's check, refusing at once what needs no walk. Returns the job.

    Refused here, with the sentence the page shows: an upload that cannot be read, a
    stamp the server's basis does not match (a stale cached module first, so the page
    fetches the current one), a game that is not the sender's own, a person whose day of
    checking is spent (`quota.check_checking`), and a second submission while one of
    theirs is still being walked (`quota.JOBS_IN_FLIGHT`). A spot check still waiting does
    not hold a new one back: the new result supersedes the old, whose check then finds
    nothing to replay.

    `basis` is the server's one-run basis for a game, which a test replaces with a
    smaller roster's; `run` takes the same.
    """
    sent = read_upload(text)
    spec = _spec(connection, game)
    current = basis(game, spec)
    refuse_basis(sent.basis, current)
    provisional.check_owner(connection, game, submitter)
    quota.check_checking(connection, submitter)
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) from jobs where kind = %s and requester = %s"
            " and state = any(%s)",
            (KIND, submitter, list(IN_FLIGHT)),
        )
        flying = int(cursor.fetchone()[0])
        if flying >= quota.JOBS_IN_FLIGHT:
            raise LegalityError(
                "a score of yours is still being checked; send this one when it is done"
            )
        cursor.execute(
            "insert into jobs (kind, hash, requester, state) values (%s, %s, %s, %s)"
            " returning id",
            (KIND, game, submitter, QUEUED),
        )
        job = int(cursor.fetchone()[0])
    path = upload_path(runs_dir, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = text if isinstance(text, str) else text.decode("utf-8")
    path.write_text(body, encoding="utf-8", newline="")
    return job


def claim(connection: Any, worker: str = "server") -> int | None:
    """Take the oldest queued legality job, or `None` when there is none.

    One atomic update, which `jobs_queue` indexes, stamped with the server's build as
    every claim is; `skip locked` lets two runners share the queue.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "update jobs set state = %s, claimed = now(), worker = %s, fingerprint = %s"
            " where id = (select id from jobs where kind = %s and state = %s"
            " order by created, id limit 1 for update skip locked) returning id",
            (RUNNING, worker, built_fingerprint(), KIND, QUEUED),
        )
        found = cursor.fetchone()
    return None if found is None else int(found[0])


def run(
    connection: Any,
    runs_dir: Path,
    job: int,
    worker: str = "server",
    basis: Now = one_run_basis,
) -> Finished:
    """Check one filed submission and move its job, and the game's result, on.

    A pass records the server's number and files through `provisional.record`, as
    `legal`, superseding the game's standing result; a refusal records nothing and keeps
    the name and sentence in `jobs.error`. Anything else is a fault here rather than in
    the submission: the job is `failed` with the error, and what was written for it is
    rolled back to a savepoint so no half-recorded result survives the caller's commit.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select kind, hash, requester, state from jobs where id = %s for update", (job,)
        )
        found = cursor.fetchone()
    if found is None:
        raise LegalityError(f"no job numbered {job}")
    kind, game, submitter, state = found
    if kind != KIND:
        raise LegalityError(f"job {job} is a {kind} job, not a legality check")
    if state not in IN_FLIGHT:
        raise LegalityError(f"job {job} is already {state}")
    if state == QUEUED:
        with connection.cursor() as cursor:
            cursor.execute(
                "update jobs set state = %s, claimed = now(), worker = %s,"
                " fingerprint = %s where id = %s",
                (RUNNING, worker, built_fingerprint(), job),
            )

    started = time.thread_time()
    with connection.cursor() as cursor:
        cursor.execute("savepoint legality_job")
    try:
        result, text = _check_and_record(
            connection, runs_dir, job, str(game), int(submitter), basis
        )
    except Refused as refusal:
        _rollback(connection)
        _finish(connection, job, REFUSED, started, error=refusal.logged)
        return Finished(job=job, state=REFUSED, refusal=refusal, error=refusal.logged)
    except ProvisionalError as error:
        stored = Refused(STORE, str(error))
        _rollback(connection)
        _finish(connection, job, REFUSED, started, error=stored.logged)
        return Finished(job=job, state=REFUSED, refusal=stored, error=stored.logged)
    except Exception as error:  # a fault here is recorded on the job, never lost
        _rollback(connection)
        said = f"{type(error).__name__}: {error}"
        _finish(connection, job, FAILED, started, error=said)
        return Finished(job=job, state=FAILED, error=said)
    with connection.cursor() as cursor:
        cursor.execute("release savepoint legality_job")
    # Tier two follows on the same thread, after every queued legality job: a sample drawn
    # from this very text, unless the account is flagged (Q503e, `spotcheck`).
    check = spotcheck.file(
        connection, game=str(game), submitter=int(submitter), result=result.id, text=text
    )
    shown = "no number" if result.headline is None else f"{result.headline:.4f}"
    then = "" if check is not None else "; not spot-checked: the account is flagged"
    _finish(
        connection, job, DONE, started, log=f"provisional result {result.id}: {shown}{then}"
    )
    return Finished(job=job, state=DONE, result=result, sample=check)


def _check_and_record(
    connection: Any, runs_dir: Path, job: int, game: str, submitter: int, basis: Now
) -> tuple[provisional.Result, str]:
    path = upload_path(runs_dir, job)
    if not path.is_file():
        raise LegalityError(f"job {job}'s upload is not under {path.parent}")
    text = path.read_text(encoding="utf-8")
    sent = read_upload(text)
    spec = _spec(connection, game)
    current = basis(game, spec)
    checked = check(spec, sent, current)
    scored = score(checked, current.weights_version)
    return provisional.record(
        connection,
        runs_dir,
        game=game,
        submitter=submitter,
        submitted=sent.basis,
        current=checked.basis or current,
        headline=scored.headline,
        viable=scored.viable,
        files=scored.files,
        job=job,
        measured=scored.measured,
    ), text


def _rollback(connection: Any) -> None:
    with connection.cursor() as cursor:
        cursor.execute("rollback to savepoint legality_job")
        cursor.execute("release savepoint legality_job")


def _finish(
    connection: Any, job: int, state: str, started: float, error: str = "", log: str = ""
) -> None:
    # `clock_timestamp()`, not `now()`: the walk runs inside this transaction, and `now()`
    # is when it began, which would date the end of a minute's walk to its start.
    # `cpu_seconds` is this thread's processor time since `started`, which a person's day
    # of checking sums (`quota.CHECK_CPU_A_DAY`).
    spent = round(max(time.thread_time() - started, 0.0), 3)
    with connection.cursor() as cursor:
        cursor.execute(
            "update jobs set state = %s, finished = clock_timestamp(), error = %s, log = %s,"
            " cpu_seconds = %s where id = %s",
            (state, error or None, log or None, spent, job),
        )


def _spec(connection: Any, game: str) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute("select spec from games where hash = %s", (game,))
        found = cursor.fetchone()
    if found is None:
        raise LegalityError(
            f"no game with the hash {game} is saved on the site: save it, then send its "
            "score again"
        )
    spec = found[0]
    if not isinstance(spec, dict):
        raise LegalityError("this game's rules have been deleted")
    return spec


def sweep(connection: Any, runs_dir: Path) -> list[int]:
    """Delete the uploads no job still waits on, after the commit. Returns their jobs.

    A finished job's upload has served: a pass stored the server's own copies under the
    result, and a refusal's sentence names the game and the ply. An upload whose row is
    gone -- a deleted game cascades its jobs -- goes too.
    """
    where = Path(runs_dir) / provisional.DIRECTORY / UPLOADS
    if not where.is_dir():
        return []
    numbered = {
        int(path.stem): path
        for path in where.glob("*.json")
        if path.stem.isdigit() and path.stem == str(int(path.stem))
    }
    if not numbered:
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            "select id from jobs where id = any(%s) and kind = %s and state = any(%s)",
            (sorted(numbered), KIND, list(IN_FLIGHT)),
        )
        waiting = {int(row[0]) for row in cursor.fetchall()}
    gone = sorted(set(numbered) - waiting)
    for number in gone:
        numbered[number].unlink(missing_ok=True)
    return gone


# ------------------------------------------------------- what the page is told (B10)


def status(
    connection: Any, job: int, requester: int | None, basis: Now = one_run_basis
) -> dict[str, Any]:
    """Where one submission stands, for the person who sent it and nobody else.

    `state` is the job's: `queued`, `running`, `done`, `refused` or `failed`. A refusal
    comes back as its name and sentence, which is how the page tells `stale-build` --
    fetch the current manifest -- from the rest. `done` carries `result`, the payload the
    game's page reads (`provisional.Result.payload`): the server's own number under its
    state, or `None` when that result no longer stands on the current basis. Another
    person's job, or a number that is no job, is the `KeyError` an unknown path gets.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select kind, hash, requester, state, error, created, finished"
            " from jobs where id = %s",
            (int(job),),
        )
        found = cursor.fetchone()
    if (
        found is None
        or found[0] != KIND
        or requester is None
        or found[2] is None
        or int(found[2]) != int(requester)
    ):
        raise KeyError(f"no such submission: {job}")
    _, game, _, state, error, created, finished = found
    answer: dict[str, Any] = {"job": int(job), "game": str(game), "state": str(state)}
    if finished is not None and created is not None:
        answer["seconds"] = round((finished - created).total_seconds(), 1)
    if state == REFUSED:
        name, _, sentence = str(error or "").partition(": ")
        answer["refusal"] = {"name": name, "sentence": sentence or name}
    elif state == FAILED:
        answer["error"] = "the server could not finish checking it; send it again"
    elif state == DONE:
        answer["result"] = None
        with connection.cursor() as cursor:
            cursor.execute(
                "select state, id from provisional where job = %s order by id desc limit 1",
                (int(job),),
            )
            row = cursor.fetchone()
        if row is not None and row[0] in provisional.STANDING:
            shown = provisional.shown(connection, [str(game)], requester, basis)
            answer["result"] = shown.get(str(game))
        elif row is not None:
            answer["superseded"] = str(row[0])
        if row is not None:
            # Tier two's word on it (Q503e): `queued` or `running` while the page should
            # keep asking, then `done`, `mismatch`, `stopped` (out of budget) or `failed`;
            # absent for a flagged account, whose results are never spot-checked.
            answer["spotCheck"] = spotcheck.of_result(connection, int(row[1]))
    return answer


# --------------------------------------------------------------- the one runner


def lower_this_thread() -> bool:
    """Put the calling thread at `nice 19`, where the platform can; whether it did.

    On Linux a thread is a task of its own and `setpriority` on its native id lowers it
    alone, so the site's request threads keep their priority while the checks yield to
    them (`docs/reference/tier-costs-on-the-box.md`: one runner thread, `nice 19`, the
    box's other core the site's). Elsewhere -- Windows has no `setpriority` -- it does
    nothing and says so.
    """
    setpriority = getattr(os, "setpriority", None)
    if setpriority is None:
        return False
    try:
        setpriority(getattr(os, "PRIO_PROCESS", 0), threading.get_native_id(), 19)
    except OSError:
        return False
    return True



class Checker:
    """The server's one thread that runs filed legality jobs, oldest first.

    `connect` is a context-manager factory that yields a connection and commits on the
    way out, as `SiteApp._connection` does, so each claim, each run and each sweep is its
    own transaction: a claimed job is `running` for every other reader before its walk
    starts, and a pass lands with its result or not at all. `drain` is the whole of the
    work, and a test calls it on its own; `start` runs it on a daemon thread, woken by
    `wake` when a submission is filed and every `every` seconds besides.
    """

    def __init__(
        self,
        connect: Callable[[], Any],
        runs_dir: Path,
        basis: Now = one_run_basis,
        worker: str = "server",
        every: float = 60.0,
        cpu_cap: float = quota.SAMPLE_JOB_CPU,
        wall_cap: float = quota.SAMPLE_JOB_WALL,
        rates: bool = True,
        spine_caps: spine.Caps | None = None,
    ) -> None:
        self.connect = connect
        self.runs_dir = Path(runs_dir)
        self.basis = basis
        self.worker = worker
        self.every = every
        # A spot check's caps (`quota.SAMPLE_JOB_CPU`, `SAMPLE_JOB_WALL`); a test lowers them.
        self.cpu_cap = cpu_cap
        self.wall_cap = wall_cap
        # L5: whether this runner rates a member's spine at publish (`site/spine.py`), and
        # under what caps (`quota.SPINE_*`); a test lowers them, or turns the rating off.
        self.rates = rates
        self.spine_caps = spine_caps
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def drain(self) -> list[Finished | spotcheck.Outcome]:
        """Run every queued job, then delete what the finished ones no longer need.

        Tier one first, every time: a spot check is claimed only when no legality job is
        queued, so a submission is never kept waiting behind somebody else's replay. A
        member's spine rating (L5, `rate`) comes last, one game a round, after the waiting
        publishes are settled; it is not in the answer.
        """
        done: list[Finished | spotcheck.Outcome] = []
        while True:
            with self.connect() as connection:
                job = claim(connection, self.worker)
            # A walk and a spot check run in this process, so each takes its own share of
            # the process's clock (`usage.Ledger`, Q970) rather than landing on whoever
            # makes a request meanwhile. A spine rating runs in a child process, which
            # the ledger's clock never sees, so it takes none.
            if job is not None:
                with self.connect() as connection, usage.LEDGER.sharing():
                    done.append(run(connection, self.runs_dir, job, self.worker, self.basis))
                continue
            with self.connect() as connection:
                check = spotcheck.claim(connection, self.worker)
            if check is not None:
                with self.connect() as connection, usage.LEDGER.sharing():
                    done.append(
                        spotcheck.run(
                            connection, self.runs_dir, check, self.worker, self.basis,
                            self.cpu_cap, self.wall_cap,
                        )
                    )
                continue
            # Every check is done, so every waiting *Publish* can be settled, and a game it
            # puts on trial has its spine rating filed. Then one spine job, and round again,
            # so a check filed while it played goes first.
            self.settle()
            if self.rate() is None:
                break
        if done:
            with self.connect() as connection:
                sweep(connection, self.runs_dir)
                provisional.sweep(connection, self.runs_dir)
        return done

    def settle(self) -> list[Any]:
        """Finish every *Publish* that waited for a check now ended, or call it off (Q742).

        On every pass, not only after a job: a press can land just after the check it was
        waiting for ended, and the next pass is then what finishes it. `drafts.settle` is
        idempotent, so a pass that finds nothing to do changes nothing.
        """
        # Imported here: `drafts` is read by modules that never run a check.
        from chess_variant_sandbox.site import drafts

        with self.connect() as connection:
            moved = drafts.settle_all(connection, self.basis)
        for settled in moved:
            print(f"legality: the publish of {settled.hash} {settled.state}")
        return moved

    def rate(self) -> spine.Outcome | None:
        """Rate one member's spine (L5, `site/spine.py`), or `None` when there is none to
        rate, rating is off, or a legality walk or spot check is queued (those go first).

        Its outcome is printed, not returned by `drain`: `drain` answers for the checks of
        people's measurements, and a spine rating is the site's own work.
        """
        if not self.rates:
            return None
        with self.connect() as connection:
            if spine.waiting_ahead(connection):
                return None
            job = spine.claim(connection, self.worker)
        if job is None:
            return None
        rated = spine.run(self.connect, self.runs_dir, job, self.spine_caps)
        print(f"legality: the spine of {rated.hash} {rated.state}: {rated.said}")
        return rated

    def requeue(self) -> int:
        """Put back what a stopped process left `running`; this is the only runner."""
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "update jobs set state = %s, claimed = null, worker = null"
                " where kind = %s and state = %s",
                (QUEUED, KIND, RUNNING),
            )
            return (
                int(cursor.rowcount or 0)
                + spotcheck.requeue(connection)
                + spine.requeue(connection)
            )

    def wake(self) -> None:
        self._wake.set()

    def start(self) -> threading.Thread:
        self.requeue()
        # A start is a deploy, which is when a basis moves: the results it made stale get
        # their `stale_since`, the shortlist's 30-day clock (`provisional.counted`).
        try:
            with self.connect() as connection:
                provisional.mark_stale(connection, self.basis)
        except Exception as error:  # noqa: BLE001 - a stale stamp is not worth the site
            print(f"legality: the stale readings could not be stamped: {error}")
        # L5: a member's game on trial with no computer player rated and no spine job --
        # one published before this existed -- is filed now, oldest publish first.
        if self.rates:
            try:
                with self.connect() as connection:
                    owed = spine.file_owed(connection)
                if owed:
                    print(f"legality: {len(owed)} published game(s) owed a spine rating")
            except Exception as error:  # noqa: BLE001 - nor is a backfill
                print(f"legality: the owed spine ratings could not be filed: {error}")
        self._thread = threading.Thread(target=self._run, name="legality", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        lowered = lower_this_thread()
        print(f"legality: the runner thread {'runs at nice 19' if lowered else 'keeps its priority'}")
        while not self._stopping.is_set():
            try:
                for finished in self.drain():
                    print(f"legality: job {finished.job} {finished.state}")
            except Exception as error:  # noqa: BLE001 - a fault must not end the thread
                print(f"legality: the queue could not be run: {error}")
            self._wake.wait(self.every)
            self._wake.clear()

