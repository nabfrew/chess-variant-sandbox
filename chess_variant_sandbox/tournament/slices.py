"""A measurement cut into units, and the units gathered back into its files.

`corpus.measure` is one call that fits, plays, analyses and scores. It spreads the
tournament over threads inside the core and the analysis over a `ProcessPoolExecutor`,
and a browser has neither. This module is the same measurement expressed as a list of
**units** -- whole pieces of work that can run anywhere, name their games by schedule
index, and hand back text -- with the arithmetic that joins them in the gathers between.

Five phases, in order, and a gather after each:

| phase | a unit plays | its input | the gather |
|---|---|---|---|
| `fit` | a slice of the fit tournament | the spec | harvest the positions and regress |
| `accept` | a slice of the acceptance match | the spec, the provisional vector | the verdict |
| `tournament` | a slice of the canonical schedule | the spec, the verdict | ratings |
| `analysis` | a slice of the graded games | the spec, the verdict, those games | -- |
| `blunder` | a slice of the blunder trials | the spec, the verdict, those games | the score |

**The fit's verdict is an input to every later unit.** `resolve_fit` is what turns
`minimax:fit=<procedure>` into a player, and it refuses to default, so a worker with no
`fitted/` store cannot play the canonical roster until somebody hands it the verdict.
Passing it in the request is what makes a unit runnable on a machine that has only the
text it was sent.

**Nothing here is a second implementation.** The unit runners call `runner._run_native`,
`analysis._analyse_game` and `analysis._one_blunder` unchanged, and the gathers call
`fitting.harvest`, `logistic_fit`, `rate` and `score_ruleset`. Those files are
fingerprinted -- an edit to one owes a replay of every measurement in the corpus -- which
is the whole reason this is a new module rather than a `slices=` argument on the old
ones. What is new here is the cutting, the text format and the joining, and none of the
three decides a game.

**How the work is divided must not change an answer.** Every seed is fixed by
`runner.schedule` before any unit is listed, the blunder trials keep the order
`_blunder_choices` drew them in, and the analyses keep the order of the records. So one
slice and four gather to the same bytes, which is the property
`tests/test_slices.py` measures against `corpus.measure` itself.

**A sliced measurement writes no `canonical.json`.** It writes the four files a run
directory holds -- `games.jsonl`, `tournament.json`, `analysis.json` and the index -- and
returns what it gathered. Whether those records become an official score is a decision
above this module, and for a browser-made result the answer is no until a server has
re-derived it.

**Two text entry points, and a browser needs no others.** `run_request` runs one unit, in
every worker. `gather_request` is one step of the gathers between the phases, in the one
worker that gathers: it takes the replies to the units it dealt last and answers with the
next phase's units, or the result. So the page sends a worker text and never Python.
"""

from __future__ import annotations

import copy
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.agents.features import VOCABULARY
from chess_variant_sandbox.agents.fitted import (
    ACCEPT_GAMES_PER_PAIRING,
    ACCEPT_NODES,
    ACCEPT_SEED,
    FIT_AGENTS,
    FIT_GAMES_PER_PAIRING,
    FIT_SEED,
    FittedObjective,
    logistic_fit,
    procedure_digest,
    register_fit,
)
from chess_variant_sandbox.agents.fitted import unit as unit_vector
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.measurement import Basis
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS, glossary, load_weights, score_ruleset
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.transform import settle_schema_version
from chess_variant_sandbox.tournament import analysis as analysis_pass
from chess_variant_sandbox.tournament import fitting, runner
from chess_variant_sandbox.tournament.analysis import (
    BLUNDER_SAMPLE,
    BlunderTrial,
    CorpusAnalysis,
    GameAnalysis,
)
from chess_variant_sandbox.tournament.ratings import RatingTable
from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.tournament.runner import TournamentConfig, TournamentResult

FIT = "fit"
ACCEPT = "accept"
TOURNAMENT = "tournament"
ANALYSIS = "analysis"
BLUNDER = "blunder"

# The order a measurement runs them in. `analysis` and `blunder` both read the finished
# tournament and neither reads the other, so a coordinator may deal them together.
PHASES = (FIT, ACCEPT, TOURNAMENT, ANALYSIS, BLUNDER)

# Which phases play games and which read finished ones. A reader of a unit needs to know
# which, because only the second kind is sent record text.
PLAYING_PHASES = (FIT, ACCEPT, TOURNAMENT)
READING_PHASES = (ANALYSIS, BLUNDER)


class SliceError(ValueError):
    """A unit that cannot be run, or a gather handed something that does not fit."""


# ------------------------------------------------------------------------ the unit


@dataclass(frozen=True)
class Unit:
    """One piece of a measurement, named by what it covers rather than by a seed.

    `games` holds schedule indices for a playing phase and record indices for a reading
    one, which are the same numbers: the tournament's records come back in schedule
    order. Naming a unit this way is what lets a server check a submitted unit against
    the schedule it built itself -- a seed in the name would be the visitor's word for
    which game this was.
    """

    phase: str
    part: int
    parts: int
    games: tuple[int, ...]

    @property
    def name(self) -> str:
        return f"{self.phase}-{self.part + 1}-of-{self.parts}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "part": self.part,
            "parts": self.parts,
            "games": list(self.games),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Unit:
        phase = str(data["phase"])
        if phase not in PHASES:
            raise SliceError(f"{phase!r} is not one of {PHASES}")
        return cls(
            phase=phase,
            part=int(data["part"]),
            parts=int(data["parts"]),
            games=tuple(int(index) for index in data["games"]),
        )


def _cut(items: Sequence[int], slices: int) -> list[tuple[int, ...]]:
    """Contiguous pieces of nearly equal size, and never an empty one.

    Fewer pieces than asked for where there is less work than that, so a coordinator
    never deals a round trip that carries no games. Contiguous rather than round-robin
    because a gather then joins its texts by concatenating them in part order.
    """
    count = len(items)
    if count == 0:
        return []
    parts = max(1, min(slices, count))
    base, extra = divmod(count, parts)
    pieces: list[tuple[int, ...]] = []
    start = 0
    for part in range(parts):
        size = base + (1 if part < extra else 0)
        pieces.append(tuple(items[start : start + size]))
        start += size
    return pieces


def _units(phase: str, indices: Sequence[int], slices: int) -> list[Unit]:
    pieces = _cut(indices, slices)
    return [
        Unit(phase=phase, part=part, parts=len(pieces), games=games)
        for part, games in enumerate(pieces)
    ]


# ---------------------------------------------------------------- the schedules


def fit_config(workers: int = 0) -> TournamentConfig:
    """The games the fit reads. Constants of the procedure, so nobody may vary them."""
    return TournamentConfig(
        agents=FIT_AGENTS,
        games_per_pairing=FIT_GAMES_PER_PAIRING,
        seed=FIT_SEED,
        workers=workers,
        collect_series=False,
    )


def accept_config(procedure: str = "", workers: int = 0) -> TournamentConfig:
    """The match that turns a vector into a verdict: the fit against what it replaces."""
    digest = procedure or procedure_digest()
    return TournamentConfig(
        agents=(
            f"minimax:fit={digest},nodes={ACCEPT_NODES}",
            f"minimax:nodes={ACCEPT_NODES}",
        ),
        games_per_pairing=ACCEPT_GAMES_PER_PAIRING,
        seed=ACCEPT_SEED,
        workers=workers,
        include_self_play=False,
        collect_series=False,
    )


def canonical_config(
    agents: tuple[str, ...] | None = None,
    games_per_pairing: int | None = None,
    seed: int | None = None,
    workers: int = 0,
) -> TournamentConfig:
    """The roster a committed score means, as a config.

    The defaults are read from `corpus` rather than restated here, and lazily, because
    `corpus` sits above this package and owns what the canonical roster is.
    """
    from chess_variant_sandbox.corpus import CANONICAL_AGENTS, CANONICAL_GAMES, CANONICAL_SEED

    return TournamentConfig(
        agents=CANONICAL_AGENTS if agents is None else agents,
        games_per_pairing=CANONICAL_GAMES if games_per_pairing is None else games_per_pairing,
        seed=CANONICAL_SEED if seed is None else seed,
        workers=workers,
    )


# ------------------------------------------------------------------ listing units


def playing_units(phase: str, config: TournamentConfig, slices: int = 1) -> list[Unit]:
    """The units of a phase that plays games: slices of that phase's schedule."""
    if phase not in PLAYING_PHASES:
        raise SliceError(f"{phase!r} does not play games; it reads them")
    return _units(phase, range(len(runner.schedule(config))), slices)


def analysis_units(records: Sequence[GameRecord], slices: int = 1) -> list[Unit]:
    """Slices of the games the analysis pass grades, in the pass's own order."""
    return _units(ANALYSIS, [index for index, _ in _graded(records)], slices)


def blunder_units(
    records: Sequence[GameRecord],
    slices: int = 1,
    sample: int = BLUNDER_SAMPLE,
    seed: int = 0,
) -> list[Unit]:
    """Slices of the blunder trials, in the order the sampler drew them.

    The order is not tidy and must not be tidied. `_blunder_choices` draws with
    `random.Random.sample`, which returns its picks in draw order, and `analysis.json`
    lists the trials in that order -- so sorting here would change a file the scorer
    reads.

    The sampler is the pass's own and it hands back the very records it was given, so
    the picks are turned into schedule indices by identity rather than by a second copy
    of which games are eligible.
    """
    graded = _graded(records)
    index_of = {id(record): index for index, record in reversed(graded)}
    chosen = analysis_pass._blunder_choices([record for _, record in graded], sample, seed)
    return _units(BLUNDER, [index_of[id(record)] for record in chosen], slices)


def _graded(records: Sequence[GameRecord]) -> list[tuple[int, GameRecord]]:
    """What the analysis pass grades, with each game's index, read from the pass.

    `analyse_corpus` grades the games both of whose seats actually chose, and falls back
    to every game where none did. Both halves are its line, kept here so a unit list and
    the pass cannot disagree about which games there are.
    """
    pairs = list(enumerate(records))
    return [pair for pair in pairs if analysis_pass._searching(pair[1])] or pairs


# ------------------------------------------------------------------ running a unit


def run_unit(
    unit: Unit,
    spec: dict[str, Any],
    fit: FittedObjective | None = None,
    records: Sequence[GameRecord] | None = None,
    config: TournamentConfig | None = None,
) -> str:
    """One unit's work, as the text the gather parses.

    The output is JSON Lines whatever the phase, one object a game or a trial, in the
    unit's own order. Concatenating the parts of a phase in part order therefore gives
    the phase's whole output with no joining rule of its own -- which is what makes the
    worker count invisible in the result.

    `config` is the measurement's roster and is read by a `tournament` unit, which needs
    the schedule its games are indices into, and by a `blunder` unit, which takes the
    sampler's seed from it. The fit and acceptance phases are constants of the procedure
    and ignore it.
    """
    if fit is not None:
        # The one piece of state a unit needs and cannot derive. Registered before the
        # engine is asked for anything, because `native_spec` resolves `fit=` at the
        # moment an agent string becomes a player.
        register_fit(fit, replace=True)

    chosen = config if config is not None else canonical_config()
    if unit.phase in PLAYING_PHASES:
        return _lines(
            record.as_dict()
            for record in _play(spec, _config_for(unit.phase, fit, chosen), unit.games)
        )

    if records is None:
        raise SliceError(f"a {unit.phase} unit needs the records it reads")
    if len(records) != len(unit.games):
        raise SliceError(
            f"{unit.name} covers {len(unit.games)} games and was handed {len(records)}"
        )

    engine = Engine(spec)
    if unit.phase == ANALYSIS:
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        epsilon = analysis_pass._epsilon_for(engine)
        return _lines(
            analysis_pass._analyse_game(engine, evaluator, record, epsilon).as_dict()
            for record in records
        )

    trials = (
        analysis_pass._one_blunder(
            engine, record, analysis_pass.blunder_rng(chosen.seed, record)
        )
        for record in records
    )
    return _lines(trial.as_dict() for trial in trials if trial is not None)


def _config_for(
    phase: str, fit: FittedObjective | None, roster: TournamentConfig
) -> TournamentConfig:
    if phase == FIT:
        return fit_config(roster.workers)
    if phase == ACCEPT:
        return accept_config(fit.procedure if fit is not None else "", roster.workers)
    return roster


def _play(
    spec: dict[str, Any], config: TournamentConfig, games: Sequence[int]
) -> list[GameRecord]:
    """The schedule's games at these indices, played by the ordinary runner.

    `run_tournament` builds the schedule and then dispatches to one of two loops. This
    is that dispatch over a subset of the schedule, and the two loops themselves are the
    runner's own -- so a slice plays what the whole would have played, seeds included.
    """
    from chess_variant_sandbox.agents import native_spec

    tasks = runner.schedule(config)
    out_of_range = [index for index in games if not 0 <= index < len(tasks)]
    if out_of_range:
        raise SliceError(
            f"the schedule has {len(tasks)} games and this unit names {out_of_range[:4]}"
        )
    chosen = [tasks[index] for index in games]
    if not chosen:
        return []

    engine = Engine(spec)
    natives = [native_spec(agent, engine.spec.hash) for agent in config.agents]
    if engine.core is not None and all(native is not None for native in natives):
        return runner._run_native(engine, config, chosen, natives, False)
    return list(runner._run_python(engine, chosen, config.collect_series, False))


def _lines(payloads: Any) -> str:
    """JSON Lines in the record file's own spelling, so a gather can write them out."""
    return "".join(
        json.dumps(payload, separators=(",", ":")) + "\n" for payload in payloads
    )


# ------------------------------------------------------- a unit as text, both ways


def request(
    unit: Unit,
    spec: dict[str, Any],
    fit: FittedObjective | None = None,
    records: Sequence[GameRecord] | None = None,
    config: TournamentConfig | None = None,
) -> str:
    """Everything a worker needs for one unit, as one JSON object.

    Four keys and no fifth. `roster` is what the measurement's config tells a unit --
    the agents, the games a pairing and the seed -- so the fit and acceptance constants
    stay where the procedure digest covers them. The records travel as the JSON Lines a
    reading unit's games would occupy in `games.jsonl`, and only those games, so a unit's
    input is proportional to its work.

    `workers` rides along and is the one field that cannot change an answer: it is how
    many threads the core plays a unit's games on, which is a fact about the machine
    running it. `TournamentConfig.as_dict` leaves it out of what a run records for the
    same reason, and a browser worker has one thread whatever it says.
    """
    chosen = config if config is not None else canonical_config()
    payload: dict[str, Any] = {
        "unit": unit.as_dict(),
        "spec": spec,
        "fit": fit.as_dict() if fit is not None else None,
        "roster": {
            "agents": list(chosen.agents),
            "games_per_pairing": chosen.games_per_pairing,
            "seed": chosen.seed,
            "workers": chosen.workers,
        },
    }
    if unit.phase in READING_PHASES:
        if records is None:
            raise SliceError(f"a {unit.phase} unit needs the records it reads")
        payload["records"] = _lines(records[index].as_dict() for index in unit.games)
    return json.dumps(payload, separators=(",", ":"))


def run_request(payload: str) -> str:
    """Text in, text out: the whole of what a worker does.

    JavaScript moves this string and never looks inside it. Records hold 64-bit position
    keys and a JavaScript number is exact to 53 bits, so the only parser either side is
    Python's.
    """
    data = json.loads(payload)
    unit = Unit.from_dict(data["unit"])
    fit = (
        FittedObjective.from_dict(data["fit"]) if data.get("fit") is not None else None
    )
    records = (
        parse_records(data.get("records", "")) if unit.phase in READING_PHASES else None
    )
    roster = data.get("roster") or {}
    config = canonical_config(
        agents=tuple(str(agent) for agent in roster["agents"]),
        games_per_pairing=int(roster["games_per_pairing"]),
        seed=int(roster["seed"]),
        workers=int(roster.get("workers", 0)),
    )
    return run_unit(unit, data["spec"], fit=fit, records=records, config=config)


def parse_records(text: str) -> list[GameRecord]:
    return [GameRecord.from_dict(payload) for payload in _parse(text)]


def parse_analyses(text: str) -> list[GameAnalysis]:
    return [GameAnalysis.from_dict(payload) for payload in _parse(text)]


def parse_blunders(text: str) -> list[BlunderTrial]:
    return [BlunderTrial.from_dict(payload) for payload in _parse(text)]


def _parse(text: str) -> list[dict[str, Any]]:
    """Every line of a unit's output. A blank line is skipped; a broken one is not.

    `read_records` skips what it cannot parse, because a killed run leaves a truncated
    last line and the games before it are still a corpus. A unit's text is either whole
    or it is a unit that has to be run again, so damage is raised rather than dropped.
    """
    payloads = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SliceError(f"line {number} of a unit's text is not JSON") from error
    return payloads


# ----------------------------------------------------------------- the slim upload
#
# What a browser sends the site when a measurement is done: tier one's form, which
# `site/legality.py` documents and reads (`read_upload`). It is written here and not
# there because the one worker that gathers writes it, under Pyodide, and that worker
# imports this module and nothing under `site/` (B10). `legality.upload` is this function.

UPLOAD_SCHEMA = 1
"""The upload's own version. A reader that does not know it refuses the upload."""

REBUILT_RECORD_FIELDS: tuple[str, ...] = ("branching", "material", "captures")
"""The per-ply series of a game record, which the walk rebuilds from the positions."""

REBUILT_ANALYSIS_FIELDS: tuple[str, ...] = (
    "legal_counts",
    "moves_by_type",
    "captures_by_type",
    "capture_counts",
    "revisited_plies",
    "walked_plies",
    "starting_instances",
    "moved_instances",
    "position_keys",
)
"""Every analysis field but `near_best`, `seed` and `agents`: functions of the positions."""


def upload(
    basis: Basis | dict[str, Any],
    fit: FittedObjective | dict[str, Any],
    phases: dict[str, str],
) -> str:
    """The slim upload: each phase's gathered text, less what the server's walk rebuilds.

    `phases` maps each of `PHASES` to the text its units sent back, concatenated in part
    order -- which is what the gathering half already holds. `basis` is the stamp, a
    `measurement.Basis` or its dict.
    """
    missing = [phase for phase in PHASES if phase not in phases]
    if missing:
        raise ValueError(f"an upload carries every phase; this one lacks {missing}")
    slim = {
        FIT: _slim(phases[FIT], REBUILT_RECORD_FIELDS),
        ACCEPT: _slim(phases[ACCEPT], REBUILT_RECORD_FIELDS),
        TOURNAMENT: _slim(phases[TOURNAMENT], REBUILT_RECORD_FIELDS),
        ANALYSIS: _slim(phases[ANALYSIS], REBUILT_ANALYSIS_FIELDS),
        BLUNDER: _slim(phases[BLUNDER], ()),
    }
    stamp = basis.as_dict() if isinstance(basis, Basis) else dict(basis)
    verdict = fit.as_dict() if isinstance(fit, FittedObjective) else dict(fit)
    return json.dumps(
        {"schema": UPLOAD_SCHEMA, "basis": stamp, "fit": verdict, "phases": slim},
        separators=(",", ":"),
    )


def _slim(text: str, drop: Sequence[str]) -> str:
    """Each line again, in this module's own spelling, without the named fields."""
    lines = []
    for payload in _parse(text):
        for name in drop:
            payload.pop(name, None)
        lines.append(json.dumps(payload, separators=(",", ":")) + "\n")
    return "".join(lines)


# --------------------------------------------------------------------- the gathers


def gather_fit(spec: dict[str, Any], records: Sequence[GameRecord]) -> FittedObjective:
    """Regress the fit games into a provisional verdict, or into a refusal.

    Two refusals never reach the acceptance match, and both are `fit_ruleset`'s: a
    ruleset whose games end inside the opening skip yields no rows at all, and a vector
    of zeros is a player that moves at random. `accepted` on what comes back is
    provisional either way -- `gather_verdict` is what settles it -- and a refusal here
    is already final, which is why `accept_units` for one is empty.
    """
    engine = Engine(spec)
    procedure = procedure_digest()
    samples = fitting.harvest(engine, list(records))
    if not samples:
        return fitting._record(
            engine.spec.hash, procedure, (0.0,) * len(VOCABULARY), False, 0.0, 0, 0, 0.0
        )

    weights, _intercept, loss = logistic_fit(samples)
    vector = unit_vector(weights)
    if not any(vector):
        return fitting._record(
            engine.spec.hash, procedure, vector, False, 0.0, 0, len(samples), loss
        )
    return fitting._record(
        engine.spec.hash, procedure, vector, True, 0.0, 0, len(samples), loss
    )


def needs_a_match(fit: FittedObjective) -> bool:
    """Whether the acceptance phase has anything to play.

    False for the two refusals `gather_fit` settles on its own, both of which leave an
    all-zero vector behind. Their verdict is already final, so the acceptance phase is
    listed with no units at all rather than with units nobody would read.
    """
    return any(fit.vector)


def gather_verdict(
    provisional: FittedObjective, records: Sequence[GameRecord]
) -> FittedObjective:
    """The acceptance match's result, as the verdict every later unit is handed.

    Strictly better, which is `fit_ruleset`'s bar and not a new one: a fitted objective
    that draws with the one it replaces has not earned a seat.
    """
    if not needs_a_match(provisional):
        return provisional
    score = match_score(records, accept_config(provisional.procedure).agents[0])
    return fitting._record(
        provisional.ruleset_hash,
        provisional.procedure,
        provisional.vector,
        score > 0.5,
        score,
        2 * ACCEPT_GAMES_PER_PAIRING,
        provisional.samples,
        provisional.loss,
    )


def match_score(records: Sequence[GameRecord], first: str) -> float:
    """``first``'s score over both colours.

    `heuristics.head_to_head` plays this match and totals it in one call, and this is
    that total taken over records somebody else played. `tests/test_slices.py` holds the
    two to the same number so the copy cannot drift.
    """
    total = 0.0
    played = 0
    for record in records:
        total += record.score_for(0 if record.agents[0] == first else 1)
        played += 1
    return round(total / played, 4) if played else 0.5


def gather_analysis(
    analyses: Sequence[GameAnalysis], blunders: Sequence[BlunderTrial]
) -> CorpusAnalysis:
    return CorpusAnalysis(games=list(analyses), blunders=list(blunders))


@dataclass
class Gathered:
    """What a sliced measurement produced, and where it put it."""

    directory: Path
    spec: dict[str, Any]
    fit: FittedObjective
    records: list[GameRecord]
    ratings: RatingTable
    analysis: CorpusAnalysis
    result: Any

    @property
    def index(self) -> float:
        return float(self.result.headline)


def gather_measurement(
    spec: dict[str, Any],
    directory: str | Path,
    fit: FittedObjective,
    records: Sequence[GameRecord],
    analysis: CorpusAnalysis,
    config: TournamentConfig | None = None,
    weights: str = DEFAULT_WEIGHTS,
) -> Gathered:
    """Write the four files a run directory holds, from gathered units.

    No `canonical.json`. That file is the one every reader trusts, and what it means is
    a decision above this module: a browser-made measurement is provisional until a
    server has re-derived it, so the basis stamp belongs where the re-derivation happens.
    """
    out = Path(directory)
    chosen = config if config is not None else canonical_config()
    tournament = TournamentResult(
        ruleset_hash=spec_hash(spec),
        ruleset_name=spec["name"],
        config=chosen,
        records=list(records),
    )
    tournament.save(out)
    analysis.save(out / "analysis.json")

    result = score_ruleset(
        spec,
        list(records),
        tournament.ratings,
        weights=load_weights(weights),
        analysis=analysis,
    )
    result.save(out / f"index-{result.weights_version}.json")
    return Gathered(
        directory=out,
        spec=spec,
        fit=fit,
        records=list(records),
        ratings=tournament.ratings,
        analysis=analysis,
        result=result,
    )


# ---------------------------------------------------------------------- the driver


def measure_in_slices(
    spec: dict[str, Any],
    directory: str | Path,
    slices: int = 1,
    weights: str = DEFAULT_WEIGHTS,
    config: TournamentConfig | None = None,
    blunder_sample: int = BLUNDER_SAMPLE,
    fit: FittedObjective | None = None,
    report: Any = None,
) -> Gathered:
    """Run every unit here, in order, through the text interface, and gather.

    The reference the identity test measures against, and the shape a coordinator
    follows: list a phase's units, run them, gather, list the next. Running them in this
    process rather than in workers is the only difference, and it is deliberately the
    only one -- each unit still goes out as a request string and comes back as text, so
    a bug that lives in the serialisation cannot hide behind a shared object.

    `fit` is a verdict already in hand, which skips the first two phases. A browser has
    no `fitted/` store and a new game has no verdict, so the ordinary path fits; a caller
    measuring the same game twice, or resuming one, has the verdict already and re-playing
    the fit games would be minutes spent to reach the same vector.
    """
    say = report if callable(report) else (lambda _message: None)
    chosen = config if config is not None else canonical_config()

    if fit is not None:
        verdict = fit
        say("the fit's verdict was handed in; its games are not replayed")
    else:
        say(f"fit games, in {slices} slice(s)")
        fit_records = _run_phase(
            playing_units(FIT, fit_config(), slices), spec, parse_records, chosen
        )
        provisional = gather_fit(spec, fit_records)

        if not needs_a_match(provisional):
            say("  refused before the match: no rows to regress, or a vector of zeros")
            verdict = provisional
        else:
            say(f"acceptance match, in {slices} slice(s)")
            match = _run_phase(
                playing_units(ACCEPT, accept_config(provisional.procedure), slices),
                spec,
                parse_records,
                chosen,
                fit=provisional,
            )
            verdict = gather_verdict(provisional, match)
    say(
        f"  {'accepted' if verdict.accepted else 'refused'} at "
        f"{verdict.accept_score:.4f}"
    )

    say(f"tournament, in {slices} slice(s)")
    records = _run_phase(
        playing_units(TOURNAMENT, chosen, slices), spec, parse_records, chosen, fit=verdict
    )

    say(f"{len(records)} games; the analysis pass, in {slices} slice(s)")
    analyses = _run_phase(
        analysis_units(records, slices),
        spec,
        parse_analyses,
        chosen,
        fit=verdict,
        records=records,
    )
    trials = _run_phase(
        blunder_units(records, slices, blunder_sample, chosen.seed),
        spec,
        parse_blunders,
        chosen,
        fit=verdict,
        records=records,
    )

    say("scoring")
    return gather_measurement(
        spec,
        directory,
        verdict,
        records,
        gather_analysis(analyses, trials),
        config=chosen,
        weights=weights,
    )


def _run_phase(
    units: Sequence[Unit],
    spec: dict[str, Any],
    parse: Any,
    config: TournamentConfig,
    fit: FittedObjective | None = None,
    records: Sequence[GameRecord] | None = None,
) -> list[Any]:
    gathered: list[Any] = []
    for unit in units:
        text = run_request(request(unit, spec, fit=fit, records=records, config=config))
        gathered.extend(parse(text))
    return gathered


# ------------------------------------------------------ the page's gather, as text

# How many units the page cuts every phase into, whatever its worker count, so one worker
# and four are dealt the same units and gather the same bytes (`web/page/measure.py`).
CUT = 16

# The page's name for its fourth step: the two reading phases dealt together, since
# neither reads the other.
READING = "reading"


def gather_request(payload: str, cut: int = CUT) -> str:
    """One step of a measurement the page coordinates: text in, text out.

    `run_request` is what every worker runs; this is what the one worker that gathers runs
    between the phases (`docs/archive/plan-client-compute.md` §8, B5). It is in the wheel so that
    no worker is sent Python: the page and the wheel share this name and JSON, and nothing
    else.

    The first request is ``{"spec": <the draft>, "state": null}``. Every later one is
    ``{"state": <the last answer's state>, "replies": {phase: [text, ...]}}``, a phase's
    replies in part order. The answer is JSON with the next ``units`` to deal and the
    ``state`` to hand back, or, once the score is in, a ``result``. What a step must
    remember goes out in ``state``, which the page hands back unread, so a step is a
    function of its text and nothing else.

    **A draft is always measured from its fit**, because a browser has no `fitted/` and a
    draft is a new game. The spec is settled first (`settle_schema_version`, what
    `designing.settled` calls), so the number is for the game the designer's check hashed.
    """
    data = json.loads(payload)
    if data.get("state") is None:
        spec = settle_schema_version(copy.deepcopy(data["spec"]))
        state: dict[str, Any] = {"spec": spec, "cut": cut, "step": "", "dealt": {}}
    else:
        state = json.loads(data["state"])
    return json.dumps(_gather_step(state, data.get("replies") or {}), separators=(",", ":"))


def _gather_step(state: dict[str, Any], replies: dict[str, Any]) -> dict[str, Any]:
    """The step after ``state['step']``: `measure_in_slices`, one phase at a time."""
    step = state["step"]
    if step not in ("", FIT, ACCEPT, TOURNAMENT, READING):
        raise SliceError(f"a measurement has no step after {step!r}")
    spec = state["spec"]
    cut = int(state["cut"])
    roster = canonical_config()

    if step == "":
        return _deal(state, FIT, playing_units(FIT, fit_config(), cut), roster)

    # The fit's and the match's texts are kept to the end, because the upload carries
    # every phase's text (B10) and the server walks them all.
    texts = state.setdefault("texts", {})
    if step == FIT:
        texts[FIT] = _joined(state, replies, FIT)
        texts[ACCEPT] = ""
        provisional = gather_fit(spec, parse_records(texts[FIT]))
        if not needs_a_match(provisional):
            return _deal_tournament(state, provisional, roster)
        state["provisional"] = provisional.as_dict()
        units = playing_units(ACCEPT, accept_config(provisional.procedure), cut)
        return _deal(state, ACCEPT, units, roster, fit=provisional)

    if step == ACCEPT:
        provisional = FittedObjective.from_dict(state.pop("provisional"))
        texts[ACCEPT] = _joined(state, replies, ACCEPT)
        match = parse_records(texts[ACCEPT])
        return _deal_tournament(state, gather_verdict(provisional, match), roster)

    verdict = FittedObjective.from_dict(state["verdict"])
    if step == TOURNAMENT:
        text = _joined(state, replies, TOURNAMENT)
        records = parse_records(text)
        state["records"] = text
        units = analysis_units(records, cut) + blunder_units(
            records, cut, BLUNDER_SAMPLE, roster.seed
        )
        return _deal(state, READING, units, roster, fit=verdict, records=records)

    # The reading step's replies, and the score. `gather_measurement` writes the four files
    # a run directory holds, and none of them is kept here: the page shows the result, and
    # on the site sends `upload`, from which the server re-derives its own (B10).
    records = parse_records(state["records"])
    texts[ANALYSIS] = _joined(state, replies, ANALYSIS)
    texts[BLUNDER] = _joined(state, replies, BLUNDER)
    analysis = gather_analysis(parse_analyses(texts[ANALYSIS]), parse_blunders(texts[BLUNDER]))
    with tempfile.TemporaryDirectory() as directory:
        gathered = gather_measurement(spec, directory, verdict, records, analysis, config=roster)
    result = _page_result(gathered, roster)
    # The stamp the site reads: this wheel's basis for the game, played in a browser.
    stamp = Basis.from_dict(result["basis"])
    if stamp is None:
        raise SliceError("this wheel wrote a basis it cannot read back")
    phases = {FIT: texts[FIT], ACCEPT: texts[ACCEPT], TOURNAMENT: state["records"]}
    phases.update({ANALYSIS: texts[ANALYSIS], BLUNDER: texts[BLUNDER]})
    sent = upload(stamp.in_browser(), verdict, phases)
    return {"step": "done", "units": [], "result": result, "upload": sent}


def _deal_tournament(
    state: dict[str, Any], verdict: FittedObjective, roster: TournamentConfig
) -> dict[str, Any]:
    state["verdict"] = verdict.as_dict()
    units = playing_units(TOURNAMENT, roster, int(state["cut"]))
    return _deal(state, TOURNAMENT, units, roster, fit=verdict)


def _deal(
    state: dict[str, Any],
    step: str,
    units: Sequence[Unit],
    roster: TournamentConfig,
    fit: FittedObjective | None = None,
    records: Sequence[GameRecord] | None = None,
) -> dict[str, Any]:
    """The units of one step as the page deals them, and the state to hand back.

    Each unit carries its phase and its part, so the page can put its reply in place, and
    how many games it plays, so the page can count them. The request is `request`'s text,
    which the page passes to a worker without reading.
    """
    state["step"] = step
    state["dealt"] = {unit.phase: unit.parts for unit in units}
    return {
        "state": json.dumps(state, separators=(",", ":")),
        "step": step,
        "units": [
            {
                "phase": unit.phase,
                "part": unit.part,
                "name": unit.name,
                "games": len(unit.games),
                "plays": unit.phase in PLAYING_PHASES,
                "request": request(unit, state["spec"], fit=fit, records=records, config=roster),
            }
            for unit in units
        ],
        "result": None,
    }


def _joined(state: dict[str, Any], replies: dict[str, Any], phase: str) -> str:
    """A phase's replies in part order, as one text, or a refusal naming what is missing.

    Concatenating in part order is the phase's whole output, so the worker count leaves no
    trace. A phase dealt no units -- no blunder trials, say -- is empty.
    """
    parts = int(state["dealt"].get(phase, 0))
    texts = list(replies.get(phase) or [])
    missing = [
        f"{phase}-{part + 1}-of-{parts}"
        for part in range(parts)
        if part >= len(texts) or not isinstance(texts[part], str)
    ]
    if missing or len(texts) != parts:
        raise SliceError(
            f"the replies to {phase} do not fit the units it was dealt: missing "
            f"{', '.join(missing) or 'nothing'}, {len(texts)} sent for {parts}"
        )
    return "".join(texts)


def _page_result(gathered: Gathered, roster: TournamentConfig) -> dict[str, Any]:
    """What the page shows: the index, what it is made of, and what it rests on.

    Every sentence is the glossary's. The basis is the one this wheel would stamp on a
    measurement of this game at this seed, so a reader can see it is the server's basis.
    The page shows it and stores nothing; the site's submission carries it (tier one).
    """
    # Imported here and not at the top: `corpus` reaches this module through
    # `tournament.adopt`, and only the last step of the one worker that gathers needs it,
    # while every playing worker imports this module.
    from chess_variant_sandbox import corpus

    result = gathered.result
    spec = gathered.spec
    digest = spec_hash(spec)
    order = sorted(result.scores, key=lambda key: (-result.contributions.get(key, 0.0), key))
    components = []
    for key in order:
        entry = glossary.MEASUREMENTS.get(key)
        components.append(
            {
                "key": key,
                "title": glossary.title_of(key),
                "what": entry.what if entry else "",
                "units": entry.units if entry else "",
                "score": result.scores[key],
                "raw": result.raw.get(key),
            }
        )
    basis = corpus.current_basis(
        digest, weights=result.weights_version, spec=spec, seeds=(roster.seed,)
    ).as_dict()
    # Which objective the fitted rungs played is this measurement's own fit, never a store:
    # the browser has none, and the server checks the stamp against its own regression of
    # the fit games (`site.provisional.refusal`, Q468).
    basis["fit"] = corpus.player_digest(gathered.fit)
    return {
        "hash": digest,
        "name": spec.get("name", ""),
        "index": result.headline if result.scored else None,
        "viable": result.viable,
        "violations": list(result.violations),
        "unreadable": list(result.unreadable),
        "weights": result.weights_version,
        "interval": result.interval,
        "families": result.families,
        "components": components,
        "games": len(gathered.records),
        "seed": roster.seed,
        "fit": {"accepted": gathered.fit.accepted, "score": gathered.fit.accept_score},
        "basis": basis,
    }
