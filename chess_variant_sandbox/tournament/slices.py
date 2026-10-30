"""A measurement cut into units, and the units gathered back into its files.

`corpus.measure` is one call that plays, analyses and scores. It spreads the
tournament over threads inside the core and the analysis over a `ProcessPoolExecutor`,
and a browser has neither. This module is the same measurement expressed as a list of
**units** -- whole pieces of work that can run anywhere, name their games by schedule
index, and hand back text -- with the arithmetic that joins them in the gathers between.

Three phases, in order, and a gather after each:

| phase | a unit plays | its input | the gather |
|---|---|---|---|
| `tournament` | a slice of the canonical schedule | the spec | ratings |
| `analysis` | a slice of the graded games | the spec, those games | -- |
| `blunder` | a slice of the blunder trials | the spec, those games | the score |

**Nothing here is a second implementation.** The unit runners call `runner._run_native`,
`analysis._analyse_game` and `analysis._one_blunder` unchanged, and the gathers call
`rate` and `score_ruleset`. Those files are fingerprinted -- an edit to one owes a replay
of every measurement in the corpus -- which is the whole reason this is a new module
rather than a `slices=` argument on the old ones. What is new here is the cutting, the text format and the joining, and none of the
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

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.measurement import Basis
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS, glossary, load_weights, score_ruleset
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.spelling import current_spelling
from chess_variant_sandbox.tournament import analysis as analysis_pass
from chess_variant_sandbox.tournament import runner
from chess_variant_sandbox.tournament.analysis import (
    BLUNDER_SAMPLE,
    BlunderTrial,
    CorpusAnalysis,
    GameAnalysis,
)
from chess_variant_sandbox.tournament.ratings import RatingTable
from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.tournament.runner import TournamentConfig, TournamentResult

TOURNAMENT = "tournament"
ANALYSIS = "analysis"
BLUNDER = "blunder"

# The order a measurement runs them in. `analysis` and `blunder` both read the finished
# tournament and neither reads the other, so a coordinator may deal them together.
PHASES = (TOURNAMENT, ANALYSIS, BLUNDER)

# Which phases play games and which read finished ones. A reader of a unit needs to know
# which, because only the second kind is sent record text.
PLAYING_PHASES = (TOURNAMENT,)
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


def roster_for(spec: dict[str, Any]) -> TournamentConfig:
    """The roster a measurement of this game plays: `corpus.canonical_agents`' for it.

    A game in rounds is measured on `corpus.ROUND_AGENTS` (`matrix:nodes=N` where the
    alternating ladder has `minimax:`, which the core refuses on one), and a game with a
    rated net on its net ladder. The basis `_page_result` stamps names the same function's
    answer, so the games a page plays and the roster its stamp names are one roster: the
    page's measurement of a round game used to play `CANONICAL_AGENTS` and fail on its
    first unit (Q998).
    """
    from chess_variant_sandbox.corpus import canonical_agents

    return canonical_config(agents=canonical_agents(spec_hash(spec), spec=spec))


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
    sampler's seed from it.
    """
    chosen = config if config is not None else canonical_config()
    if unit.phase in PLAYING_PHASES:
        return _lines(
            record.as_dict()
            for record in _play(spec, chosen, unit.games)
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
    records: Sequence[GameRecord] | None = None,
    config: TournamentConfig | None = None,
) -> str:
    """Everything a worker needs for one unit, as one JSON object.

    Three keys and no fourth. `roster` is what the measurement's config tells a unit --
    the agents, the games a pairing and the seed. The records travel as the JSON Lines a
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
    return run_unit(unit, data["spec"], records=records, config=config)


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

UPLOAD_SCHEMA = 2
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


def upload(basis: Basis | dict[str, Any], phases: dict[str, str]) -> str:
    """The slim upload: each phase's gathered text, less what the server's walk rebuilds.

    `phases` maps each of `PHASES` to the text its units sent back, concatenated in part
    order -- which is what the gathering half already holds. `basis` is the stamp, a
    `measurement.Basis` or its dict.
    """
    missing = [phase for phase in PHASES if phase not in phases]
    if missing:
        raise ValueError(f"an upload carries every phase; this one lacks {missing}")
    slim = {
        TOURNAMENT: _slim(phases[TOURNAMENT], REBUILT_RECORD_FIELDS),
        ANALYSIS: _slim(phases[ANALYSIS], REBUILT_ANALYSIS_FIELDS),
        BLUNDER: _slim(phases[BLUNDER], ()),
    }
    stamp = basis.as_dict() if isinstance(basis, Basis) else dict(basis)
    return json.dumps(
        {"schema": UPLOAD_SCHEMA, "basis": stamp, "phases": slim},
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


def gather_analysis(
    analyses: Sequence[GameAnalysis], blunders: Sequence[BlunderTrial]
) -> CorpusAnalysis:
    return CorpusAnalysis(games=list(analyses), blunders=list(blunders))


@dataclass
class Gathered:
    """What a sliced measurement produced, and where it put it."""

    directory: Path
    spec: dict[str, Any]
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
    report: Any = None,
) -> Gathered:
    """Run every unit here, in order, through the text interface, and gather.

    The reference the identity test measures against, and the shape a coordinator
    follows: list a phase's units, run them, gather, list the next. Running them in this
    process rather than in workers is the only difference, and it is deliberately the
    only one -- each unit still goes out as a request string and comes back as text, so
    a bug that lives in the serialisation cannot hide behind a shared object.

    With no ``config`` it plays `roster_for(spec)`, the roster `corpus.measure` and the
    page's gather play: a game with a rated net is measured on its net ladder. Until Q1062
    the default was the net-free `canonical_config()`, so the reference measured a netted
    game (dobutsu, since L2c rated its net) on players its own measurement does not use.
    """
    say = report if callable(report) else (lambda _message: None)
    chosen = config if config is not None else roster_for(spec)

    say(f"tournament, in {slices} slice(s)")
    records = _run_phase(
        playing_units(TOURNAMENT, chosen, slices), spec, parse_records, chosen
    )

    say(f"{len(records)} games; the analysis pass, in {slices} slice(s)")
    analyses = _run_phase(
        analysis_units(records, slices),
        spec,
        parse_analyses,
        chosen,
        records=records,
    )
    trials = _run_phase(
        blunder_units(records, slices, blunder_sample, chosen.seed),
        spec,
        parse_blunders,
        chosen,
        records=records,
    )

    say("scoring")
    return gather_measurement(
        spec,
        directory,
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
    records: Sequence[GameRecord] | None = None,
) -> list[Any]:
    gathered: list[Any] = []
    for unit in units:
        text = run_request(request(unit, spec, records=records, config=config))
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

    The spec is tidied first (`spelling.current_spelling`, what `designing.settled` calls),
    so the number is for the game the designer's check hashed.
    """
    data = json.loads(payload)
    if data.get("state") is None:
        spec = current_spelling(data["spec"])
        state: dict[str, Any] = {"spec": spec, "cut": cut, "step": "", "dealt": {}}
    else:
        state = json.loads(data["state"])
    return json.dumps(_gather_step(state, data.get("replies") or {}), separators=(",", ":"))


def _gather_step(state: dict[str, Any], replies: dict[str, Any]) -> dict[str, Any]:
    """The step after ``state['step']``: `measure_in_slices`, one phase at a time."""
    step = state["step"]
    if step not in ("", TOURNAMENT, READING):
        raise SliceError(f"a measurement has no step after {step!r}")
    spec = state["spec"]
    cut = int(state["cut"])
    roster = roster_for(spec)

    if step == "":
        return _deal(state, TOURNAMENT, playing_units(TOURNAMENT, roster, cut), roster)

    # Every phase's text is kept to the end, because the upload carries them all (B10)
    # and the server walks them all.
    texts = state.setdefault("texts", {})
    if step == TOURNAMENT:
        text = _joined(state, replies, TOURNAMENT)
        records = parse_records(text)
        state["records"] = text
        units = analysis_units(records, cut) + blunder_units(
            records, cut, BLUNDER_SAMPLE, roster.seed
        )
        return _deal(state, READING, units, roster, records=records)

    # The reading step's replies, and the score. `gather_measurement` writes the four files
    # a run directory holds, and none of them is kept here: the page shows the result, and
    # on the site sends `upload`, from which the server re-derives its own (B10).
    records = parse_records(state["records"])
    texts[ANALYSIS] = _joined(state, replies, ANALYSIS)
    texts[BLUNDER] = _joined(state, replies, BLUNDER)
    analysis = gather_analysis(parse_analyses(texts[ANALYSIS]), parse_blunders(texts[BLUNDER]))
    with tempfile.TemporaryDirectory() as directory:
        gathered = gather_measurement(spec, directory, records, analysis, config=roster)
    result = _page_result(gathered, roster)
    # The stamp the site reads: this wheel's basis for the game, played in a browser.
    stamp = Basis.from_dict(result["basis"])
    if stamp is None:
        raise SliceError("this wheel wrote a basis it cannot read back")
    phases = {TOURNAMENT: state["records"], ANALYSIS: texts[ANALYSIS], BLUNDER: texts[BLUNDER]}
    sent = upload(stamp.in_browser(), phases)
    return {"step": "done", "units": [], "result": result, "upload": sent}


def _deal(
    state: dict[str, Any],
    step: str,
    units: Sequence[Unit],
    roster: TournamentConfig,
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
                "request": request(unit, state["spec"], records=records, config=roster),
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
    return {
        "hash": digest,
        "name": spec.get("name", ""),
        "index": result.headline if result.scored else None,
        "viable": result.viable,
        "measured": result.measured,
        "violations": list(result.violations),
        "unreadable": list(result.unreadable),
        "weights": result.weights_version,
        "interval": result.interval,
        "families": result.families,
        "components": components,
        "games": len(gathered.records),
        "seed": roster.seed,
        "basis": basis,
    }
