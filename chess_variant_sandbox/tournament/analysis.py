"""Second-pass measurements that need an engine, computed once and persisted.

The index reads persisted records and never touches a live engine. That invariant is
what lets a corpus be re-scored under new weights without re-simulating, and it is not
negotiable. But three of the things most worth knowing about a game cannot be read off
a result list:

* how often the player actually faced a decision, rather than a move list with one
  sane entry in it;
* whether the pieces the rules define are pieces anyone uses;
* whether a single bad move ends the game.

All three need positions, and positions need an engine. So they are computed here, in
a pass that runs after the tournament and writes its findings next to the records.
Metrics then read *those*, and the invariant holds: the engine is upstream of the
persisted file, never inside the scorer.

The pass costs almost nothing to re-run, because a record stores moves and the engine
re-derives every position from them. An existing corpus can be analysed without playing
a single new game -- which is the whole reason these measurements were worth adding at
this point rather than after the next expensive re-run.

**Sampling is deliberate.** Scoring every root move at every ply of every game costs
more than the tournament did. Positions are sampled on a fixed stride, so the
measurement is reproducible and cheap, and the quantity it estimates is a mean over
plies that does not need every ply to be exact.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from chess_variant_sandbox._core import REGRET_ITERATIONS, regret_solve
from chess_variant_sandbox.agents import native_spec
from chess_variant_sandbox.agents.evaluation import WIN_SCORE, Evaluator
from chess_variant_sandbox.engine import Engine, GameState, SpyPhase
from chess_variant_sandbox.engine.compile import POUNCE_ATTACKERS
from chess_variant_sandbox.engine.moves import (
    DUCK_PASS,
    DUCK_PLACE,
    MOVE_BUY,
    MOVE_CASTLE,
    MOVE_DROP,
    MOVE_DUCK,
    MOVE_FOLD,
    MOVE_FOREIGN,
    MOVE_HIRE,
    MOVE_PROMOTION,
    MOVE_PUSH,
    MOVE_ROTATE,
    move_fold,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.tournament.play import eval_values, finish_game
from chess_variant_sandbox.tournament.record import GameRecord

# The components this pass -- and only this pass -- can supply. Named here rather than
# in the scorer so that a weights version asking for one of them can say what to run,
# instead of reporting a missing measurement as an anonymous gap.
ANALYSIS_COMPONENTS = (
    "decision_criticality",
    "piece_utilisation",
    "blunder_recoverability",
    "position_revisit_rate",
    "capture_availability",
    "instance_participation",
)

# Every Nth ply is scored in full. Eight is short enough that a 60-ply game still
# contributes several samples and long enough that the pass stays far cheaper than the
# games it is reading.
PLY_STRIDE = 8

# Node budget for the analysis search. Deliberately fixed and modest: this is not a
# player, it is a probe asking "how far apart were the options here?", and it must mean
# the same thing in every ruleset and at every point in the corpus. It is unrelated to
# any agent's budget and must not be tuned to match one.
PROBE_NODES = 1200
PROBE_MAX_DEPTH = 8

# How many games per corpus get the blunder treatment. Each one is a full extra
# playout, so this is the most expensive knob in the module.
BLUNDER_SAMPLE = 48

# Where the blunder goes: a third of the way in. Early enough that there is a game left
# to play, late enough that the position has some structure to damage.
BLUNDER_FRACTION = 0.34


@dataclass
class GameAnalysis:
    """Engine-derived facts about one game, persisted for the scorer to read."""

    seed: int
    agents: tuple[str, str]
    # Per sampled ply: how many legal moves were within the cheapest piece's value of
    # the best one, and how many there were in total. Kept as counts rather than as a
    # ratio so that the corpus mean is over plies rather than over games.
    near_best: list[int] = field(default_factory=list)
    legal_counts: list[int] = field(default_factory=list)
    # How many times each piece type moved and captured, indexed by type, so the spec's
    # own piece list is the only vocabulary involved.
    moves_by_type: list[int] = field(default_factory=list)
    captures_by_type: list[int] = field(default_factory=list)
    # Per sampled ply, alongside `legal_counts`: how many of those legal moves took
    # something. The armies-never-engage measurement.
    capture_counts: list[int] = field(default_factory=list)
    # Positions whose key had already occurred in this game, and how many plies were
    # walked. "Back and forth and back and forth", counted.
    revisited_plies: int = 0
    walked_plies: int = 0
    # Pieces standing on the opening board, and how many of them ever moved. The axis
    # `moves_by_type` averages away: five pieces sitting in the back are invisible to
    # an entropy over piece *types* as long as their type moved somewhere else.
    starting_instances: int = 0
    moved_instances: int = 0
    # The core's position key after every ply, in order, so a measurement over the
    # positions a game passed through reads the record instead of replaying it. Empty in
    # an analysis written before the field existed.
    position_keys: list[int] = field(default_factory=list)
    # Schema 26 (Q918d): what the rounds' resolutions did, counted over the game
    # (`ROUND_COUNTS`), read off the core's events as the pass replays each round. ``None``
    # for a game that alternates, and then absent from the file, so an alternating game's
    # `analysis.json` is the bytes it always was.
    rounds: dict[str, int] | None = None
    # Schema 27 (Q935d): a spy game's phases (`SPY_COUNTS`): the plies and rounds played in
    # each, and each player's spy loss, its ply and how. ``None`` for every other game, and
    # then absent from the file, as ``rounds`` is.
    spy: dict[str, Any] | None = None
    # Schema 30 (Q999c, `docs/archive/plan-passive.md` §4.8): what the lines of sight did, one entry
    # per side-turn in each of `SIGHT_SERIES`, as the core answers it (`Engine.effects`), and
    # (Q1007d) in each of `COMMAND_SERIES` too where a sight commands.
    # ``None`` for a game without a sight, and then absent from the file, as ``rounds`` is.
    sight: dict[str, list[int]] | None = None
    # Schema 29 (Q983b, `docs/plan-rest.md` §5.4, §7): what the rest did, one entry per
    # side-turn in each of `REST_SERIES`, as the core answers it (`Engine.rest_turn`).
    # ``None`` for a game without a rest, and then absent from the file, as ``rounds`` is.
    rest: dict[str, list[int]] | None = None
    # Generation 39 (N3b, `docs/archive/plan-npc.md` §4.3): what the wild pieces did (`WILD_COUNTS`),
    # read off the phase log's kinds as the pass replays each move, with the pounces' victims
    # by type. ``None`` for a game without wild pieces (and for a game in rounds), and then
    # absent from the file, as ``rounds`` is.
    wild: dict[str, Any] | None = None
    # Generation 42 (N4b, `docs/archive/plan-npc.md` §4.6): what the duck did (`DUCK_COUNTS`, and
    # ``royal_distance``, the duck's king's distance to the mover's royal after each duck move,
    # -1 with none on the board). ``None`` for a game without a duck, and then absent from the
    # file, as ``rounds`` is.
    duck: dict[str, Any] | None = None
    # TC7 (`docs/plan-turncoat.md` §5.8): what the purse did, one entry per side-turn in each
    # of `PURSE_SERIES`, read off the core's states and the record's moves. ``None`` for a
    # game without a purse, and then absent from the file, as ``rounds`` is.
    purse: dict[str, list[int]] | None = None
    # TC7 (§5.8): what the pieces nobody owns did, one entry per side-turn in each of
    # `NEUTRAL_SERIES`, the core's answers. ``None`` for a game without a piece nobody owns,
    # and then absent from the file.
    neutral: dict[str, list[int]] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["agents"] = list(self.agents)
        if self.rounds is None:
            del data["rounds"]
        if self.spy is None:
            del data["spy"]
        if self.sight is None:
            del data["sight"]
        if self.rest is None:
            del data["rest"]
        if self.wild is None:
            del data["wild"]
        if self.duck is None:
            del data["duck"]
        if self.purse is None:
            del data["purse"]
        if self.neutral is None:
            del data["neutral"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameAnalysis:
        payload = dict(data)
        payload["agents"] = tuple(payload["agents"])
        return cls(**payload)


@dataclass
class BlunderTrial:
    """One game replayed with a single random move forced into it."""

    seed: int
    agents: tuple[str, str]
    ply: int
    original_winner: int | None
    blundering_side: int
    new_winner: int | None

    @property
    def recovered(self) -> bool:
        """Whether the side made to blunder still avoided losing."""
        return self.new_winner != (1 - self.blundering_side)

    @property
    def changed(self) -> bool:
        return self.new_winner != self.original_winner

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["agents"] = list(self.agents)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlunderTrial:
        payload = dict(data)
        payload["agents"] = tuple(payload["agents"])
        return cls(**payload)


@dataclass
class CorpusAnalysis:
    """Everything the second pass found, for one ruleset's corpus."""

    games: list[GameAnalysis] = field(default_factory=list)
    blunders: list[BlunderTrial] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "games": [g.as_dict() for g in self.games],
            "blunders": [b.as_dict() for b in self.blunders],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> CorpusAnalysis:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            games=[GameAnalysis.from_dict(g) for g in data.get("games", [])],
            blunders=[BlunderTrial.from_dict(b) for b in data.get("blunders", [])],
        )


def analyse_corpus(
    spec: dict[str, Any],
    records: list[GameRecord],
    blunder_sample: int = BLUNDER_SAMPLE,
    seed: int = 0,
    progress: bool = False,
    workers: int = 0,
) -> CorpusAnalysis:
    """Run the second pass over a finished corpus.

    Parallel by game, for the same reason the tournament is: the probe search dominates
    the cost and every game is independent. Workers compile the ruleset once and keep
    it, because compilation builds a table per (player, piece type, square) triple and
    paying for that per game would cost more than the analysis.
    """
    engine = Engine(spec)
    epsilon = _epsilon_for(engine)

    graded = [r for r in records if _searching(r)] or records
    resolved = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    if resolved == 1 or len(graded) < 8:
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        analyses = [_analyse_game(engine, evaluator, record, epsilon) for record in graded]
        blunders = _blunder_trials(engine, graded, blunder_sample, seed, progress)
        return CorpusAnalysis(games=analyses, blunders=blunders)

    chosen = _blunder_choices(graded, blunder_sample, seed)
    try:
        with ProcessPoolExecutor(
            max_workers=resolved, initializer=_init_worker, initargs=(spec,)
        ) as pool:
            analyses = list(pool.map(_analyse_one, graded, chunksize=2))
            trials = list(pool.map(_blunder_one, [(r, seed) for r in chosen], chunksize=1))
    except BrokenExecutor:
        # A worker died and the child's cause does not survive to be reported. The one
        # case seen so far was a caller without an ``if __name__ == "__main__"`` guard,
        # which on Windows makes every spawned child re-run the caller's own work -- a
        # mistake in the caller, not here, but one that surfaces as a dead pool. A
        # calibration corpus takes hours; finishing serially and saying so is a far
        # better outcome than losing it.
        print(
            "  analysis worker pool died; falling back to a single process",
            flush=True,
        )
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        analyses = [_analyse_game(engine, evaluator, record, epsilon) for record in graded]
        trials = [_one_blunder(engine, record, blunder_rng(seed, record))
                  for record in chosen]

    return CorpusAnalysis(
        games=analyses, blunders=[t for t in trials if t is not None]
    )


def _epsilon_for(engine: Engine) -> float:
    """The value of the cheapest piece: the threshold for "these were alternatives".

    Spec-derived, so it means the same thing in a game with no pawns as in one with
    five kinds of them, and nothing about it assumes a chess piece exists. Asked of the
    *opening position* rather than of the value table, because on a ruleset whose pieces
    are worth what they may still take the two are different numbers -- see
    ``Evaluator.cheapest_piece``.
    """
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    return evaluator.cheapest_piece(engine.initial_state().board)


# Each worker compiles the ruleset once. Mirrors the tournament runner deliberately:
# two modules paying the same cost should pay it the same way.
_WORKER_ENGINE: Engine | None = None
_WORKER_EVALUATOR: Evaluator | None = None
_WORKER_EPSILON = 1.0


def _init_worker(spec: dict[str, Any]) -> None:
    global _WORKER_ENGINE, _WORKER_EVALUATOR, _WORKER_EPSILON
    _WORKER_ENGINE = Engine(spec)
    _WORKER_EVALUATOR = Evaluator.for_engine(_WORKER_ENGINE, mobility_weight=0.0)
    _WORKER_EPSILON = _epsilon_for(_WORKER_ENGINE)


def _analyse_one(record: GameRecord) -> GameAnalysis:
    assert _WORKER_ENGINE is not None and _WORKER_EVALUATOR is not None
    return _analyse_game(_WORKER_ENGINE, _WORKER_EVALUATOR, record, _WORKER_EPSILON)


def _blunder_one(task: tuple[GameRecord, int]) -> BlunderTrial | None:
    assert _WORKER_ENGINE is not None
    record, seed = task
    return _one_blunder(_WORKER_ENGINE, record, blunder_rng(seed, record))


def _blunder_choices(
    records: list[GameRecord], sample: int, seed: int
) -> list[GameRecord]:
    candidates = [r for r in records if r.plies >= 12]
    if not candidates:
        return []
    rng = random.Random(seed ^ 0x5EED)
    return candidates if len(candidates) <= sample else rng.sample(candidates, sample)


def _searching(record: GameRecord) -> bool:
    """Games where both seats were held by a player that actually chose."""
    return all(agent.split(":")[0] != "random" for agent in record.agents)


# Schema 30 (Q999c, `docs/archive/plan-passive.md` §4.8): what the pass records of a game with a line
# of sight, one entry a side-turn (a ply of a game that alternates; each seat's choice in a
# round), every one the core's (`Engine.effects` on the board the choice was made on):
#
# * ``side``: whose turn it was;
# * ``removed_freeze``, ``removed_disarm``, ``removed_shield``: how many of that side's moves
#   the effects removed, by reason (a move barred two ways counts once, under the first);
# * ``binds``: 1 when the effects removed any move at all (*an effect binds*), else 0;
# * ``frozen``, ``disarmed``, ``shielded``: how many of that side's pieces stood under each
#   effect (a piece under two counts in both);
# * ``suppressed``: 1 when that side's royal stood under an attack the effects suppressed (a
#   frozen or disarmed attacker, or a shielded royal), else 0.
#
# A game without a sight has no ``sight`` key at all, so its `analysis.json` is the bytes it was.
SIGHT_SERIES = (
    "side",
    "removed_freeze",
    "removed_disarm",
    "removed_shield",
    "binds",
    "frozen",
    "disarmed",
    "shielded",
    "suppressed",
)

# Q1007d (`docs/archive/plan-schema-31.md` §3.9): what the pass adds for a game whose sight commands,
# one entry a side-turn beside `SIGHT_SERIES`'s (a command is refused beside rounds, so every
# side-turn is a ply), each the core's answer or the record's move:
#
# * ``commanded``, ``commanded_own``: how many of that side's pieces stood under a command
#   (the opponent may move them) and under a command that ``captures_own`` (and may make them
#   take their own side), read as the three piece counts above are;
# * ``command_offered``: how many legal commanded moves that side was offered
#   (`Effects.commanded`);
# * ``command_made``: 1 when the move made was a commanded one (the record's move carries
#   ``MOVE_FOREIGN``), else 0;
# * ``turncoat``: 1 when it was a commanded capture, a piece made to take its own side, else 0.
#
# A game without a command has none of these keys, so its `analysis.json` is the bytes it was:
# a sight game without a command writes `SIGHT_SERIES` alone, as before.
COMMAND_SERIES = (
    "commanded",
    "commanded_own",
    "command_offered",
    "command_made",
    "turncoat",
)

# Generation 37 (N2, `docs/archive/plan-npc.md` §4.2): the third owner's column, for a game in which a
# wild piece sees, one entry a side-turn beside `SIGHT_SERIES`' (and `COMMAND_SERIES`', when a
# player's sight commands): how many of that side's pieces stood under a freeze, a disarm and a
# shield **a wild piece gives** (`Effects.wild`, the core's bits from the wild sights alone),
# read as the piece counts above are. A game without a seeing wild piece has none of these keys,
# so its `analysis.json` is the bytes it was.
WILD_SIGHT_SERIES = ("wild_frozen", "wild_disarmed", "wild_shielded")


def _wild_sees(engine: Engine) -> bool:
    """Whether a wild piece of this game carries a sight: the compiled forward lines hold the
    third owner's frame (`compile._compile_sights`, generation 37)."""
    sight = engine.spec.sight
    return sight is not None and len(sight.lines) > engine.spec.n_frames


# The effect bits as the core reports them (`compile.SIGHT_FREEZE`, `SIGHT_DISARM`,
# `SIGHT_SHIELD`, and Q1007c's `SIGHT_COMMAND` and `SIGHT_COMMAND_OWN`), in the order of the
# piece counts: the first three are `SIGHT_SERIES`', the last two `COMMAND_SERIES`'.
_SIGHT_BITS = (1, 2, 4, 8, 16)


def _commands(engine: Engine) -> bool:
    """Whether any line of sight of this game commands (the compiled flag, Q1007c)."""
    return engine.spec.sight is not None and engine.spec.sight.has_command


# Schema 29 (Q983b, `docs/plan-rest.md` §5.4, §7): what the pass records of a game whose pieces
# rest, one entry a side-turn (a ply of a game that alternates; each seat's choice in a round),
# every one the core's (`Engine.rest_turn` on the board the choice was made on):
#
# * ``side``: whose turn it was;
# * ``binds``: 1 when the rest took away at least one of that side's legal moves (*the rest
#   binds*), else 0;
# * ``lifted``: 1 when the rest was lifted, because it would have left that side no move (a
#   resting piece's move offered), else 0; never 1 beside ``binds``;
# * ``resting``: how many of that side's pieces rested as it chose (its entries in
#   ``GameState.rest``).
#
# A game without a rest has no ``rest`` key at all, so its `analysis.json` is the bytes it was.
REST_SERIES = ("side", "binds", "lifted", "resting")


# TC7 (`docs/plan-turncoat.md` §5.8): what the pass records of a game with a purse, one entry
# a side-turn (the purse is refused beside rounds, so every side-turn is a ply), read off the
# core's states before and after the record's move:
#
# * ``side``: whose turn it was;
# * ``purse``: that side's coins as it chose (`GameState.purse`);
# * ``spent``: the coins its move spent (a buy, hire or bribe takes nothing, so the fall in
#   its purse), else 0;
# * ``earned``: the coins its move earned (a bounty, after the cap), else 0;
# * ``bought``, ``hired``, ``bribed``: 1 when the move was that purchase (a hire of a piece
#   nobody owned, a bribe of the opponent's), else 0.
#
# The totals, the counts and the purse at the end are sums and the last entry's. A game without
# a purse has no ``purse`` key at all, so its `analysis.json` is the bytes it was.
PURSE_SERIES = ("side", "purse", "spent", "earned", "bought", "hired", "bribed")


def _purse_tally(engine: Engine) -> dict[str, list[int]] | None:
    """`PURSE_SERIES` empty, for a game with a purse; ``None`` for every other game."""
    if engine.spec.purse is None:
        return None
    return {key: [] for key in PURSE_SERIES}


def _count_purse(
    engine: Engine, tally: dict[str, list[int]] | None, before: GameState, move: int,
    after: GameState,
) -> None:
    """One side-turn of `PURSE_SERIES`: the side to move on ``before`` making ``move``."""
    if tally is None:
        return
    seat = before.side_to_move
    kind = move_type(move)
    change = after.purse[seat] - before.purse[seat]
    purchase = kind in (MOVE_BUY, MOVE_HIRE)
    # A hire and a bribe are one kind (`MOVE_HIRE`): whose the piece was says which.
    bribe = kind == MOVE_HIRE and engine.spec.owner_of(before.board[move_from(move)]) != 2
    row = (
        seat,
        before.purse[seat],
        -change if purchase else 0,
        0 if purchase else max(change, 0),
        int(kind == MOVE_BUY),
        int(kind == MOVE_HIRE and not bribe),
        int(bribe),
    )
    for key, value in zip(PURSE_SERIES, row, strict=True):
        tally[key].append(value)


# TC7 (§5.8): what the pass records of a game with a piece nobody owns, one entry a side-turn,
# each the core's answer on the board the choice was made on:
#
# * ``side``: whose turn it was;
# * ``held``: how many pieces nobody owns that side held by majority (`Engine.controller_of`);
# * ``neutral_offered``: how many legal moves of a piece nobody owns it was offered (a foreign
#   move from a square of nobody's block);
# * ``neutral_made``: 1 when the move made was one, else 0.
#
# The share of side-turns holding one is the mean of ``held > 0``. A game without a piece
# nobody owns has no ``neutral`` key at all, so its `analysis.json` is the bytes it was.
NEUTRAL_SERIES = ("side", "held", "neutral_offered", "neutral_made")


def _neutral_tally(engine: Engine) -> dict[str, list[int]] | None:
    """`NEUTRAL_SERIES` empty, for a game with a piece nobody owns; ``None`` otherwise."""
    if engine.spec.neutrals is None:
        return None
    return {key: [] for key in NEUTRAL_SERIES}


def _count_neutral(
    engine: Engine, tally: dict[str, list[int]] | None, state: GameState, move: int
) -> None:
    """One side-turn of `NEUTRAL_SERIES`: the side to move on ``state`` making ``move``."""
    if tally is None:
        return
    spec = engine.spec
    seat = state.side_to_move

    def nobodys(square: int) -> bool:
        value = state.board[square]
        return bool(value) and spec.owner_of(value) == 2

    held = sum(
        1 for square in range(len(state.board))
        if nobodys(square) and engine.controller_of(state, square) == seat
    )
    offered = sum(
        1 for legal in engine.legal_moves(state)
        if legal & MOVE_FOREIGN and nobodys(move_from(legal))
    )
    made = bool(move & MOVE_FOREIGN) and nobodys(move_from(move))
    for key, value in zip(NEUTRAL_SERIES, (seat, held, offered, int(made)), strict=True):
        tally[key].append(value)


# Generation 39 (N3b, `docs/archive/plan-npc.md` §4.3): the wild tally, per game. The phase log's
# entries by kind (``steps_eating`` the steps that ate, then every pounce, return and turn),
# the wild pieces players took, and the most standing at once; beside it ``pounced``, the
# pounces' victims by type in the spec's piece order. N1 left returns uncounted; they are here.
# Generation 40 (N3c, §4.4): ``pounces_on_attackers``, the pounces by a type that takes what
# attacks it, read off the compiled kind of the piece standing on the pounce's origin before the
# move (a player's move never puts a wild piece anywhere, and a pouncer only leaves its square).
# Generation 41 (N3d, §4.5): ``copies``, the heads a splitting pouncer grew (each one ate an
# attacker); the heads standing and taken are ``most_standing`` and ``taken_by_players``.
WILD_COUNTS = (
    "steps_eating", "pounces", "pounces_on_attackers", "copies", "returns", "turns",
    "taken_by_players", "most_standing",
)


# Generation 42 (N4b, `docs/archive/plan-npc.md` §4.6): the duck tally, per game. Its moves, how many put
# it down from off the board, how many passed, and after each duck move the duck's king's
# distance to the mover's royal (the lattice's own distance on a hex board; -1 with no royal
# of the mover's standing), which §6.1's prediction reads.
DUCK_COUNTS = ("duck_moves", "placements", "passes")


def _duck_tally(engine: Engine) -> dict[str, Any] | None:
    if engine.spec.duck is None:
        return None
    return {**dict.fromkeys(DUCK_COUNTS, 0), "royal_distance": []}


def _king_distance(engine: Engine, first: int, second: int) -> int:
    width = engine.spec.width
    files = second % width - first % width
    ranks = second // width - first // width
    if engine.spec.lattice == "hex":
        # The hex lattice's own distance (`schema.hex_distance`, outside this file's set).
        return max(abs(files), abs(ranks), abs(files + ranks))
    return max(abs(files), abs(ranks))


def _count_duck(engine: Engine, duck: dict[str, Any], mover: int, move: int, after: GameState) -> None:
    if move_type(move) != MOVE_DUCK:
        return
    duck["duck_moves"] += 1
    how = move_promotion(move)
    duck["placements"] += int(how == DUCK_PLACE)
    duck["passes"] += int(how == DUCK_PASS)
    spec = engine.spec
    at = engine.duck_square(after)
    royals = [
        square for square, value in enumerate(after.board)
        if value and spec.owner_of(value) == mover and spec.type_of(value) in spec.royal_types
    ]
    duck["royal_distance"].append(
        min((_king_distance(engine, at, royal) for royal in royals), default=-1)
        if at is not None else -1
    )


def _wild_tally(engine: Engine) -> dict[str, Any] | None:
    if engine.spec.npcs is None:
        return None
    return {**dict.fromkeys(WILD_COUNTS, 0), "pounced": [0] * engine.spec.n_types}


def _count_wild(
    engine: Engine, wild: dict[str, Any], board: bytes, move: int, after: GameState, phase: Any
) -> None:
    spec = engine.spec
    target = board[move_to(move)]
    if takes(engine, board, move) and target and spec.owner_of(target) == 2:
        wild["taken_by_players"] += 1
    for step in phase:
        if step.kind == "step":
            wild["steps_eating"] += int(step.eaten != 0)
        elif step.kind == "pounce":
            wild["pounces"] += 1
            wild["pounced"][spec.type_of(step.eaten)] += 1
            pouncer = board[step.origin]
            if spec.pounce is not None and spec.pounce.kind[spec.type_of(pouncer)] == POUNCE_ATTACKERS:
                wild["pounces_on_attackers"] += 1
        elif step.kind == "copy":
            wild["copies"] += 1
        elif step.kind == "return":
            wild["returns"] += 1
        elif step.kind == "turn":
            wild["turns"] += 1
    standing = sum(1 for value in after.board if _is_wild_piece(spec, value))
    wild["most_standing"] = max(wild["most_standing"], standing)


def _analyse_game(
    engine: Engine, evaluator: Evaluator, record: GameRecord, epsilon: float
) -> GameAnalysis:
    if engine.spy:
        return _analyse_spy(engine, evaluator, record, epsilon)
    if engine.simultaneous:
        return _analyse_rounds(engine, evaluator, record, epsilon)
    n_types = engine.spec.n_types
    moves_by_type = [0] * n_types
    captures_by_type = [0] * n_types
    near_best: list[int] = []
    legal_counts: list[int] = []
    capture_counts: list[int] = []

    state = engine.initial_state()
    # Which of the pieces standing at the start ever moved. Tracked by origin square,
    # which is the cheap version and is wrong in two bounded ways: a castling rook is
    # not credited, because the move is encoded at the king's square, and a piece
    # captured on its home square followed by an enemy piece moving off that square
    # credits the wrong army. Both are worth a handful of instances per game against a
    # denominator in the tens, and neither is worth a per-ply board diff to fix.
    start_squares = starting_squares(engine, state.board)
    moved_squares: set[int] = set()
    seen_keys = {state.key}
    position_keys: list[int] = []
    revisited = 0
    walked = 0
    sight = sight_tally(engine)
    rest = rest_tally(engine)
    wild = _wild_tally(engine)
    if wild is not None:
        wild["most_standing"] = sum(
            1 for value in state.board if _is_wild_piece(engine.spec, value)
        )
    duck = _duck_tally(engine)
    purse = _purse_tally(engine)
    neutral = _neutral_tally(engine)

    for index, move in enumerate(record.moves):
        board = state.board
        kind = mover_kind(engine, board, move)
        if kind is None:
            # The record and the spec have drifted apart. Stop counting rather than
            # inventing numbers; replay raises on the same condition with a message.
            break
        count_sight(engine, sight, state, state.side_to_move, move)
        count_rest(engine, rest, state, state.side_to_move)
        _count_neutral(engine, neutral, state, move)
        before = state
        if moves_an_instance(move) and move_from(move) in start_squares:
            moved_squares.add(move_from(move))
        moves_by_type[kind] += 1
        if takes(engine, board, move):
            captures_by_type[kind] += 1

        if index % PLY_STRIDE == 0:
            legal = engine.legal_moves(state)
            if len(legal) > 1:
                scores = _score_roots(engine, evaluator, state, legal)
                best = max(scores)
                near_best.append(sum(1 for s in scores if best - s <= epsilon))
                legal_counts.append(len(legal))
                # Same sample, so the two ratios are over the same plies. "Occupied
                # destination" is the same reading of a capture that
                # `captures_by_type` uses two lines up; en passant is missed by both,
                # consistently.
                capture_counts.append(sum(1 for m in legal if takes(engine, board, m)))

        mover = state.side_to_move
        if wild is None:
            state = engine.apply(state, move)
        else:
            state, phase = engine.apply_with_phase(state, move)
            _count_wild(engine, wild, board, move, state, phase)
        if duck is not None:
            _count_duck(engine, duck, mover, move, state)
        _count_purse(engine, purse, before, move, state)
        walked += 1
        position_keys.append(state.key)
        if state.key in seen_keys:
            revisited += 1
        else:
            seen_keys.add(state.key)

    return GameAnalysis(
        seed=record.seed,
        agents=record.agents,
        near_best=near_best,
        legal_counts=legal_counts,
        moves_by_type=moves_by_type,
        captures_by_type=captures_by_type,
        capture_counts=capture_counts,
        revisited_plies=revisited,
        walked_plies=walked,
        starting_instances=len(start_squares),
        moved_instances=len(moved_squares),
        position_keys=position_keys,
        sight=sight,
        rest=rest,
        wild=wild,
        duck=duck,
        purse=purse,
        neutral=neutral,
    )


def _is_wild_piece(spec: Any, value: int) -> bool:
    """A wild piece's byte: the third owner's, and (generation 42) not the duck's."""
    if not value or spec.owner_of(value) != 2:
        return False
    return spec.duck is None or spec.type_of(value) != spec.duck.piece_type


class _ProbeExhausted(Exception):
    """Raised to abandon a probe iteration that ran out of budget."""


class _NodeCounter:
    __slots__ = ("used", "budget")

    def __init__(self, budget: int) -> None:
        self.used = 0
        self.budget = budget


def _score_roots(
    engine: Engine, evaluator: Evaluator, state: GameState, legal: list[int]
) -> list[float]:
    """The probe, in the core where there is one.

    ``rust/src/probe.rs`` is a line-for-line port of ``_root_scores`` below, ragged edge
    included, and the two are held to producing identical vectors -- this pass is the
    dominant cost of scoring a ruleset and it is a search over one position at a time,
    which is exactly the shape the boundary could not serve until ``root_scores``
    existed. The Python version stays as the oracle and as the path for a ruleset the
    core cannot play.

    The ruler is fixed: the derived table from ``eval_values``, never a fitted
    objective, even where the rated players use one. ``docs/reference/decision-probe-ruler.md``.
    It reads the goal terms and, since Q471, the immunity term at the weights
    ``evaluator`` carries, which are the rated players' defaults.

    **A spy game's turn is probed here, in Python** (schema 27, Q935d): its tree may cross
    into rounds, which ``probe.rs`` would play as turns, so ``_probe`` stops at the seam and
    values the round as a leaf. Every move and verdict is still the core's.
    """
    if engine.core is not None and not engine.spy:
        return engine.root_scores(
            state,
            eval_values(engine.spec),
            PROBE_NODES,
            PROBE_MAX_DEPTH,
            evaluator.goal_pieces,
            evaluator.immune_pieces,
        )
    return _root_scores(engine, evaluator, state, legal)


def _root_scores(
    engine: Engine, evaluator: Evaluator, state: GameState, legal: list[int]
) -> list[float]:
    """Evaluate every legal move at the probe budget, with a full window.

    The full window is the point. A playing search evaluates root moves inside an
    alpha-beta window, so everything except the best move comes back as a bound rather
    than a value -- fine for choosing a move, useless for asking how close the
    alternatives were. Reusing those numbers would systematically overstate how many
    real options a position offered, which is exactly the quantity being measured.

    Budget is nodes rather than depth, matching the rule that governs agents: a fixed
    depth is wildly different amounts of work in a game with eight legal moves and one
    with sixty, and this number has to be comparable across rulesets.
    """
    budget = max(len(legal) * 2, PROBE_NODES)
    counter = _NodeCounter(budget)
    scores = [0.0] * len(legal)
    for depth in range(1, PROBE_MAX_DEPTH + 1):
        try:
            for index, move in enumerate(legal):
                # See ``_probe``: the negation is only right when the side changed.
                child = engine.apply(state, move)
                value = _probe(engine, evaluator, child, depth - 1, counter)
                scores[index] = (
                    -value if child.side_to_move != state.side_to_move else value
                )
        except _ProbeExhausted:
            break
        if counter.used >= budget:
            break
    return scores


def _probe(
    engine: Engine, evaluator: Evaluator, state: GameState, depth: int, counter: _NodeCounter
) -> float:
    counter.used += 1
    if counter.used >= counter.budget:
        raise _ProbeExhausted

    if engine.spy and engine.is_round(state):
        # Schema 27 (Q935d): the seam. A spy game's turn may lead into rounds, which a tree
        # of single moves cannot hold, so a round is a leaf here (`docs/plan-spy.md` §3.8).
        return _round_leaf(engine, evaluator, state)

    moves = engine.legal_moves(state)
    outcome = engine.terminal(state, moves)
    if outcome is not None:
        if outcome.is_draw:
            return 0.0
        signed = WIN_SCORE - state.ply
        return signed if outcome.winner == state.side_to_move else -signed
    if depth <= 0:
        return evaluator.evaluate(engine, state, state.side_to_move, len(moves))

    best = float("-inf")
    for move in moves:
        # Negamax negates the child because the side is assumed to alternate. A turn
        # longer than one move makes that conditional: when the same player moves again
        # the child's value is already from this player's point of view. Asked of the
        # states rather than of the turn rule, since a ramp changes the answer ply by
        # ply. ``max`` keeps its first argument on a tie, and ``probe.rs`` agrees.
        child = engine.apply(state, move)
        value = _probe(engine, evaluator, child, depth - 1, counter)
        best = max(best, -value if child.side_to_move != state.side_to_move else value)
    return best


def _blunder_trials(
    engine: Engine,
    records: list[GameRecord],
    sample: int,
    seed: int,
    progress: bool,
) -> list[BlunderTrial]:
    """Force one random move into a game and finish it with the same players.

    The question is not whether the blunder gets punished -- a competent opponent
    punishes everything -- but whether the game is still a game afterwards. A ruleset
    where one arbitrary move always loses is brittle: it rewards not erring rather than
    playing well, and the winner is whoever slips second. One where the result never
    moves is not responding to move quality at all, which is the dull failure wearing a
    different hat. Both ends are bad, so this is scored as a band.
    """
    chosen = _blunder_choices(records, sample, seed)
    if not chosen:
        return []

    trials: list[BlunderTrial] = []
    for number, record in enumerate(chosen):
        if progress and number % 8 == 0:
            print(f"  blunder trial {number}/{len(chosen)}", flush=True)
        trial = _one_blunder(engine, record, blunder_rng(seed, record))
        if trial is not None:
            trials.append(trial)
    return trials


def blunder_rng(seed: int, record: GameRecord) -> random.Random:
    """The stream that picks the move forced into one game.

    Derived per record rather than drawn from one stream walked in order, so that which
    move gets forced into a game does not depend on how many games were analysed before
    it -- or on how the work was divided. This loop and the worker pool used to derive
    it differently, which made ``blunder_recoverability`` depend on ``--workers``: the
    same corpus analysed serially and in parallel picked different moves and could score
    differently. That is the same class of bug as a tournament whose result depends on
    its thread count, and it is ruled out the same way, by fixing every trial's seed
    before any work starts.
    """
    return random.Random((seed ^ 0x5EED) + record.seed)


def _one_blunder(
    engine: Engine, record: GameRecord, rng: random.Random
) -> BlunderTrial | None:
    if engine.spy:
        return _one_spy_blunder(engine, record, rng)
    if engine.simultaneous:
        return _one_round_blunder(engine, record, rng)
    target_ply = int(record.plies * BLUNDER_FRACTION)
    state = engine.initial_state()
    for move in record.moves[:target_ply]:
        if move not in engine.legal_moves(state):
            return None
        state = engine.apply(state, move)

    legal = engine.legal_moves(state)
    if len(legal) < 2:
        return None

    blundering = state.side_to_move
    played = record.moves[target_ply] if target_ply < len(record.moves) else None
    alternatives = [m for m in legal if m != played] or legal
    state = engine.apply(state, rng.choice(alternatives))

    # Same players and the same per-seat streams as the original game, so the only
    # difference between this playout and the recorded one is the move forced in. The
    # players are the *rated* ones: ``finish_game`` runs the core's agents wherever the
    # core implements them, so a trial is not quietly measuring a second implementation.
    resumed = finish_game(engine, record.agents, state, record.seed)

    return BlunderTrial(
        seed=record.seed,
        agents=record.agents,
        ply=target_ply,
        original_winner=record.winner,
        blundering_side=blundering,
        new_winner=resumed.winner,
    )


# ------------------------------------------------------- schema 26: games in rounds
#
# `docs/archive/plan-simultaneous.md` §6.3 (Q918d). A round game's record keeps both moves of each
# round in seat order and every per-ply series twice, so a round is two plies; the pass
# walks it a round at a time through `apply_round`. Two of its measurements need a reading
# of their own, because with no mover there is no single list of options to score:
#
# * **the probe** values *each* player's moves against the opponent's equilibrium mix of
#   the round's matrix game, one round deep, at the probe's fixed ruler (the built-in
#   evaluator, mobility off, the goal and immunity terms at their weights; a finished round
#   at ±(`WIN_SCORE` − ply)). The matrix is solved by the round players' own solver, the
#   core's regret matching+ (`_core.regret_solve`). A move is near the best when it is
#   within the cheapest piece of the best against that mix, as an alternating probe's is;
#   each seat with a choice is one sample, so a sampled round gives two.
# * **a blunder** is a move whose value against the mix falls short of the best by more
#   than that threshold. The trial forces one, picked at random among them, into the
#   seat whose ply the trial falls on, beside the other seat's recorded move (it chose
#   without seeing either), and finishes the game with the same players. A position with
#   no such move offers no blunder and gives no trial.
#
# Everything else the pass reads -- types moved, instances, captures offered, positions
# revisited -- reads both moves of a round on the round's board, where both were chosen.

# What the pass counts of the rounds' resolutions, from the core's events (`Engine.
# apply_round_with_phase`: 0 a collision with its winner or -1, 1 a void move, 2 a tie the
# initiative decided, 3 a pair barred), and a swap read off the two move words.
ROUND_COUNTS = (
    "rounds",
    "collisions",
    "collisions_both_lost",
    "void_moves",
    "initiative_ties",
    "bars",
    "swaps",
)

# Schema 28 (Q943b, `docs/archive/plan-speed.md` §4.4): what speed did to the rounds, from the core's
# events 5 to 7 (an ordered round, a slower move lost with its piece, a slower move never
# played because the faster ended the game) and a void move in an ordered round, which is
# the slower one's. Counted only where a piece declares a speed, so a game without one has
# none of these keys and its `analysis.json` is the bytes it was.
SPEED_COUNTS = (
    "ordered_rounds",
    "slower_void",
    "slower_lost",
    "ended_after_faster",
)

_COLLISION, _VOID, _INITIATIVE, _BARRED = 0, 1, 2, 3
_ORDER, _LOST, _UNPLAYED = 5, 6, 7


# The share of a mix read as the solver's residue rather than as a move the equilibrium
# plays: what the first two of the solver's iterations alone can leave on a move, since
# its average weights iteration t by t. Regret matching+'s average keeps that much of its
# opening iterations on moves the equilibrium abandons, and against a finished round worth
# ±`WIN_SCORE` a residue of 1e-5 is ten points, enough to push a sound move out of reach
# of the cheapest piece. Decided before any round game was measured (Q918d's plan entry).
DUST = 3.0 / (REGRET_ITERATIONS * (REGRET_ITERATIONS + 1) / 2)


@dataclass(frozen=True)
class RoundReading:
    """One round's matrix game read by the probe: each seat's legal moves, the mixes the
    solver found, and each move's value against the opponent's mix, from its own seat."""

    moves: tuple[list[int], list[int]]
    strategies: tuple[list[float], list[float]]
    values: tuple[list[float], list[float]]

    def near_best(self, seat: int, epsilon: float) -> int:
        values = self.values[seat]
        best = max(values)
        return sum(1 for value in values if best - value <= epsilon)

    def blunders(self, seat: int, epsilon: float) -> list[int]:
        """The seat's moves that fall short of its best against the mix by more than
        ``epsilon``, in generation order."""
        values = self.values[seat]
        best = max(values)
        return [
            move for move, value in zip(self.moves[seat], values, strict=True)
            if best - value > epsilon
        ]


def read_matrix(
    moves: tuple[list[int], list[int]], cells: list[float]
) -> RoundReading:
    """A round's matrix game, ``cells`` row-major from player 0's side (rows its moves,
    columns player 1's), solved and read: each seat's moves valued against the other's
    mix with its residue below `DUST` cleared. Sums run in plain loops, in order."""
    rows, cols = len(moves[0]), len(moves[1])
    p, q = regret_solve(cells, rows, cols)
    first, second = _purified(p), _purified(q)
    row_values = []
    for i in range(rows):
        total = 0.0
        for j in range(cols):
            total += cells[i * cols + j] * second[j]
        row_values.append(total)
    col_values = []
    for j in range(cols):
        total = 0.0
        for i in range(rows):
            total += cells[i * cols + j] * first[i]
        col_values.append(-total)
    return RoundReading(moves=moves, strategies=(first, second), values=(row_values, col_values))


def _purified(strategy: list[float]) -> list[float]:
    kept = [weight if weight >= DUST else 0.0 for weight in strategy]
    total = 0.0
    for weight in kept:
        total += weight
    if total <= 0.0:
        return list(strategy)
    return [weight / total for weight in kept]


def read_round(
    engine: Engine,
    evaluator: Evaluator,
    state: GameState,
    moves: tuple[list[int], list[int]] | None = None,
) -> RoundReading | None:
    """The probe on one round's board, or ``None`` when either player has no move."""
    if moves is None:
        moves = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    if not moves[0] or not moves[1]:
        return None
    return read_matrix(moves, _round_cells(engine, evaluator, state, moves))


def _round_cells(
    engine: Engine,
    evaluator: Evaluator,
    state: GameState,
    moves: tuple[list[int], list[int]],
) -> list[float]:
    """Every pair of the round's moves resolved and valued at the probe's ruler, row-major
    from player 0's side (rows its moves, columns player 1's)."""
    cells = []
    for first in moves[0]:
        for second in moves[1]:
            child = engine.apply_round(state, first, second)
            outcome = engine.terminal(child)
            if outcome is None:
                # Mobility is off at the ruler, so the move counts weigh nothing.
                cells.append(evaluator.evaluate_round(engine, child, 0, (0, 0)))
            elif outcome.is_draw:
                cells.append(0.0)
            else:
                signed = WIN_SCORE - child.ply
                cells.append(signed if outcome.winner == 0 else -signed)
    return cells


def replay_rounds(engine: Engine, record: GameRecord) -> list[GameState]:
    """The position before the first round and after every round of a round game's record.

    `play.replay`'s counterpart for a game in rounds: a record holds both moves of each
    round in seat order, and an illegal one means the record and its spec have drifted.
    """
    if len(record.moves) % 2:
        raise ValueError(
            f"game {record.seed} holds {len(record.moves)} moves; a game in rounds holds two "
            "a round"
        )
    state = engine.initial_state()
    states = [state]
    for index in range(0, len(record.moves), 2):
        first, second = record.moves[index], record.moves[index + 1]
        if (
            first not in engine.legal_moves_for(state, 0)
            or second not in engine.legal_moves_for(state, 1)
        ):
            raise ValueError(
                f"round {index // 2} of game {record.seed} is not legal under ruleset "
                f"{record.ruleset_hash}; the record and the spec disagree"
            )
        state = engine.apply_round(state, first, second)
        states.append(state)
    return states


def _is_swap(first: int, second: int) -> bool:
    """Two moves onto each other's origin, read off the words (§2.4: two chases)."""
    if MOVE_DROP in (move_type(first), move_type(second)):
        return False
    return move_to(first) == move_from(second) and move_to(second) == move_from(first)


def _analyse_rounds(
    engine: Engine, evaluator: Evaluator, record: GameRecord, epsilon: float
) -> GameAnalysis:
    """`_analyse_game` for a game in rounds (the reading above `ROUND_COUNTS`)."""
    n_types = engine.spec.n_types
    moves_by_type = [0] * n_types
    captures_by_type = [0] * n_types
    near_best: list[int] = []
    legal_counts: list[int] = []
    capture_counts: list[int] = []
    counts = round_tallies(engine)
    sight = sight_tally(engine)
    rest = rest_tally(engine)

    state = engine.initial_state()
    start_squares = starting_squares(engine, state.board)
    moved_squares: set[int] = set()
    seen_keys = {state.key}
    position_keys: list[int] = []
    revisited = 0
    walked = 0

    for index in range(0, len(record.moves) - 1, 2):
        pair = (record.moves[index], record.moves[index + 1])
        lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
        if pair[0] not in lists[0] or pair[1] not in lists[1]:
            # Drifted from its spec; stop counting, as the alternating walk does.
            break
        for seat in (0, 1):
            count_sight(engine, sight, state, seat)
            count_rest(engine, rest, state, seat)
        board = state.board
        for move in pair:
            kind = mover_kind(engine, board, move)
            if kind is None:  # pragma: no cover - a legal move stands on its origin
                continue
            if moves_an_instance(move) and move_from(move) in start_squares:
                moved_squares.add(move_from(move))
            moves_by_type[kind] += 1
            if takes(engine, board, move):
                captures_by_type[kind] += 1

        # The ply stride, read at the round's first ply: every fourth round.
        if index % PLY_STRIDE == 0:
            reading = read_round(engine, evaluator, state, lists)
            if reading is not None:
                for seat in (0, 1):
                    if len(lists[seat]) > 1:
                        near_best.append(reading.near_best(seat, epsilon))
                        legal_counts.append(len(lists[seat]))
                        capture_counts.append(
                            sum(1 for move in lists[seat] if takes(engine, board, move))
                        )

        state, _, events = engine.apply_round_with_phase(state, pair[0], pair[1])
        count_round(counts, pair, events)

        # Two plies a round, as every per-ply series in the record: the key after the
        # round twice, and a revisit counted on both, so the rate is per round.
        walked += 2
        position_keys.extend((state.key, state.key))
        if state.key in seen_keys:
            revisited += 2
        else:
            seen_keys.add(state.key)

    return GameAnalysis(
        seed=record.seed,
        agents=record.agents,
        near_best=near_best,
        legal_counts=legal_counts,
        moves_by_type=moves_by_type,
        captures_by_type=captures_by_type,
        capture_counts=capture_counts,
        revisited_plies=revisited,
        walked_plies=walked,
        starting_instances=len(start_squares),
        moved_instances=len(moved_squares),
        position_keys=position_keys,
        rounds=counts,
        sight=sight,
        rest=rest,
    )


def round_blunder(
    engine: Engine,
    state: GameState,
    seat: int,
    played: int,
    rng: random.Random,
) -> int | None:
    """The move a trial forces into ``seat`` on this round's board: one of its blunders
    other than the move it played, drawn by ``rng``, or ``None`` when it has none."""
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    reading = read_round(engine, evaluator, state)
    if reading is None or len(reading.moves[seat]) < 2:
        return None
    epsilon = _epsilon_for(engine)
    blunders = [move for move in reading.blunders(seat, epsilon) if move != played]
    return rng.choice(blunders) if blunders else None


def _one_round_blunder(
    engine: Engine, record: GameRecord, rng: random.Random
) -> BlunderTrial | None:
    """`_one_blunder` for a game in rounds: the trial's ply names a round and a seat."""
    target_ply = int(record.plies * BLUNDER_FRACTION)
    target_round, seat = divmod(target_ply, 2)
    start = 2 * target_round
    if start + 1 >= len(record.moves):
        return None
    state = engine.initial_state()
    for index in range(0, start, 2):
        first, second = record.moves[index], record.moves[index + 1]
        if (
            first not in engine.legal_moves_for(state, 0)
            or second not in engine.legal_moves_for(state, 1)
        ):
            return None
        state = engine.apply_round(state, first, second)

    played, other = record.moves[start + seat], record.moves[start + 1 - seat]
    if other not in engine.legal_moves_for(state, 1 - seat):
        return None
    forced = round_blunder(engine, state, seat, played, rng)
    if forced is None:
        return None
    pair = (forced, other) if seat == 0 else (other, forced)
    state = engine.apply_round(state, pair[0], pair[1])
    resumed = finish_game(engine, record.agents, state, record.seed)

    return BlunderTrial(
        seed=record.seed,
        agents=record.agents,
        ply=target_ply,
        original_winner=record.winner,
        blundering_side=seat,
        new_winner=resumed.winner,
    )


# ------------------------------------------------------------- schema 27: the spy
#
# `docs/plan-spy.md` §3.8 (Q935d). A spy game alternates while both players have a spy and
# plays rounds once one is lost. Its record is the flat move list -- one word a turn, two a
# round in seat order -- and every per-ply series once a turn and twice a round, as the
# tournament wrote it; nothing in it says which step is which. The pass asks the engine the
# phase at every step, as the tournament did (`spy_steps`), and reads each step by its kind:
#
# * **a turn** (alternating, or the reply) as an alternating ply. The probe is `_root_scores`
#   in Python, over the core's moves and verdicts, with one difference at the seam: a child
#   that is a round is a leaf (`_round_leaf`, the ruler's `evaluate_round` or the verdict),
#   because a tree of single moves cannot hold a round and `probe.rs` would play one as a
#   turn.
# * **an announced round** one round deep at the ruler, by what each side knew: the
#   announcer's moves by their worst answer (the maximin), the answerer's against the move
#   actually announced (`read_announced`). A move is near the best within the cheapest
#   piece; a blunder falls short by more.
# * **a blind round** as in any round game (`read_round`).
#
# A step is sampled when one of its plies falls on `PLY_STRIDE`: the alternating walk's rule
# for a turn and the round walk's for a round, which in a spy game may start on an odd ply
# (E2). `SPY_COUNTS` is what the phases did, the study's inputs (§5.3).

# Per game: the turns played while both spies stood, whether the reply turn was played (0 or
# 1), the announced and the blind rounds, and for each player the ply after the step that
# took its last spy (the clock's count, as §5.3's first loss reads it) and how (`SPY_LOSSES`),
# both ``None`` while it stands.
SPY_COUNTS = (
    "alternating_plies",
    "reply_turns",
    "announced_rounds",
    "blind_rounds",
    "lost_ply_0",
    "lost_ply_1",
    "lost_how_0",
    "lost_how_1",
)

# How a spy was lost, read off the step's move words, the board it was chosen on and the
# round's events, first match wins: its own move promoted it; it moved and met another
# piece (a collision, or a swap that met); an opposing move ended on the square where it
# stood; otherwise the spec's rule that removes a piece no move names (an explosion, a wild
# piece's step, a strike chain or a contact's merge, in that order when a game has several);
# `other` where none applies.
SPY_LOSSES = ("promotion", "collision", "capture", "blast", "wild", "strike", "other")

_SWAP_MET = 4


class SpyStep(NamedTuple):
    """One step of a spy game's record: the ply it starts on, the position it was chosen
    on and its phase, its moves (one for a turn, two in seat order for a round), each
    mover's legal moves there (the mover's alone for a turn), the position after it and
    the round's events (`Engine.apply_round_with_phase`; empty for a turn)."""

    ply: int
    state: GameState
    phase: SpyPhase
    moves: tuple[int, ...]
    lists: tuple[list[int], ...]
    after: GameState
    events: tuple[tuple[int, int, int], ...]

    @property
    def seats(self) -> tuple[int, ...]:
        """Who made each of the step's moves."""
        return (self.state.side_to_move,) if len(self.moves) == 1 else (0, 1)


def spy_steps(engine: Engine, record: GameRecord, strict: bool = False) -> Iterator[SpyStep]:
    """Walk a spy game's record a step at a time, each step's kind asked of the engine,
    every move checked legal on the board it was chosen on. A record that has drifted from
    its spec stops the walk, or with ``strict`` raises naming the ply."""
    state = engine.initial_state()
    moves = record.moves
    index = 0
    while index < len(moves):
        phase = engine.phase(state)
        events: tuple[tuple[int, int, int], ...] = ()
        lists: tuple[list[int], ...]
        if phase.is_round:
            step = tuple(moves[index:index + 2])
            lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
        else:
            step = (moves[index],)
            lists = (engine.legal_moves(state),)
        if len(step) != len(lists) or any(
            move not in legal for move, legal in zip(step, lists, strict=True)
        ):
            if strict:
                raise ValueError(
                    f"ply {index} of game {record.seed} is not legal under ruleset "
                    f"{record.ruleset_hash}; the record and the spec disagree"
                )
            return
        if len(step) == 2:
            after, _, events = engine.apply_round_with_phase(state, step[0], step[1])
        else:
            after = engine.apply(state, step[0])
        yield SpyStep(index, state, phase, step, lists, after, events)
        state = after
        index += len(step)


def replay_spy(engine: Engine, record: GameRecord) -> list[GameState]:
    """The position before the first step and after every step of a spy game's record.

    `replay_rounds`' counterpart: each step a turn or a round by the phase the engine
    reports, and a move that is not legal where it was chosen means the record and its spec
    have drifted, which raises."""
    states = [engine.initial_state()]
    states.extend(step.after for step in spy_steps(engine, record, strict=True))
    return states


def spy_holders(phase: SpyPhase, side_to_move: int) -> frozenset[int]:
    """Who has a spy in a position of this phase (`docs/plan-spy.md` §1.3's table read
    backwards): both while play alternates, the player not replying in a reply turn, the
    answerer in an announced round, nobody in a blind one."""
    if phase.name == "alternating":
        return frozenset((0, 1))
    if phase.name == "reply":
        return frozenset((1 - side_to_move,))
    if phase.name == "announced":
        return frozenset((1 - phase.announcer,))
    return frozenset()


def _spy_squares(engine: Engine, board: bytes, player: int) -> set[int]:
    """Where ``player``'s spies stand on ``board``: a piece of one of its spy types (the
    board byte is player-major, `player * n_facings * n_types + facing * n_types + type + 1`)."""
    spec = engine.spec
    assert spec.spy is not None
    types = spec.spy.types[player]
    return {
        square
        for square in range(spec.n_squares)
        if board[square] != 0
        and spec.owner_of(board[square]) == player
        and spec.type_of(board[square]) in types
    }


def spy_loss(engine: Engine, step: SpyStep, player: int) -> str:
    """How ``player``'s last spy left the board in ``step`` (`SPY_LOSSES`' reading)."""
    squares = _spy_squares(engine, step.state.board, player)
    moved: set[int] = set()
    for seat, move in zip(step.seats, step.moves, strict=True):
        if seat != player or move_type(move) in (MOVE_DROP, MOVE_FOLD):
            continue
        if move_from(move) not in squares:
            continue
        moved.add(move_from(move))
        if move_type(move) == MOVE_PROMOTION:
            return "promotion"
        if any(
            event in (_COLLISION, _SWAP_MET) and square in (move_to(move), move_from(move))
            for event, square, _player in step.events
        ):
            return "collision"
    standing = squares - moved
    for seat, move in zip(step.seats, step.moves, strict=True):
        if seat != player and move_type(move) != MOVE_DROP and move_to(move) in standing:
            return "capture"
    spec = engine.spec
    if spec.explosion is not None:
        return "blast"
    if spec.npcs is not None:
        return "wild"
    if spec.strike is not None:
        return "strike"
    return "other"


def _round_leaf(engine: Engine, evaluator: Evaluator, state: GameState) -> float:
    """A round reached inside a turn's probe, from the side to move's point of view: its
    verdict if the game has ended, else the ruler's `evaluate_round` (mobility is off at the
    ruler, so the move counts weigh nothing)."""
    outcome = engine.terminal(state)
    if outcome is None:
        return evaluator.evaluate_round(engine, state, state.side_to_move, (0, 0))
    if outcome.is_draw:
        return 0.0
    signed = WIN_SCORE - state.ply
    return signed if outcome.winner == state.side_to_move else -signed


def announced_matrix(
    moves: tuple[list[int], list[int]], cells: list[float], announcer: int, announced: int
) -> RoundReading:
    """An announced round's cells (`_round_cells`, row-major from player 0's side) read by
    what each side knew: the announcer's moves valued by their worst answer, the answerer's
    against ``announced``, each from its own side. The strategies are the pure choices those
    values make: the move announced, and the first best answer to it."""
    cols = len(moves[1])
    answerer = 1 - announcer
    sign = 1.0 if announcer == 0 else -1.0

    def cell(mine: int, theirs: int) -> float:
        row, col = (mine, theirs) if announcer == 0 else (theirs, mine)
        return sign * cells[row * cols + col]

    maximin = []
    for mine in range(len(moves[announcer])):
        worst = float("inf")
        for theirs in range(len(moves[answerer])):
            worst = min(worst, cell(mine, theirs))
        maximin.append(worst)
    shown = moves[announcer].index(announced)
    answers = [-cell(shown, theirs) for theirs in range(len(moves[answerer]))]
    best_answer = answers.index(max(answers))

    values: list[list[float]] = [[], []]
    strategies: list[list[float]] = [[], []]
    values[announcer], values[answerer] = maximin, answers
    strategies[announcer] = [1.0 if i == shown else 0.0 for i in range(len(maximin))]
    strategies[answerer] = [1.0 if i == best_answer else 0.0 for i in range(len(answers))]
    return RoundReading(
        moves=moves,
        strategies=(strategies[0], strategies[1]),
        values=(values[0], values[1]),
    )


def read_announced(
    engine: Engine,
    evaluator: Evaluator,
    state: GameState,
    announcer: int,
    announced: int,
    moves: tuple[list[int], list[int]] | None = None,
) -> RoundReading | None:
    """The probe on one announced round's board (`announced_matrix`), or ``None`` when
    either player has no move."""
    if moves is None:
        moves = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    if not moves[0] or not moves[1]:
        return None
    cells = _round_cells(engine, evaluator, state, moves)
    return announced_matrix(moves, cells, announcer, announced)


def _analyse_spy(
    engine: Engine, evaluator: Evaluator, record: GameRecord, epsilon: float
) -> GameAnalysis:
    """`_analyse_game` for a spy game (the reading above `SPY_COUNTS`)."""
    n_types = engine.spec.n_types
    moves_by_type = [0] * n_types
    captures_by_type = [0] * n_types
    near_best: list[int] = []
    legal_counts: list[int] = []
    capture_counts: list[int] = []
    rounds = round_tallies(engine)
    spy: dict[str, Any] = {
        key: (None if key.startswith("lost_") else 0) for key in SPY_COUNTS
    }

    start = engine.initial_state()
    start_squares = starting_squares(engine, start.board)
    moved_squares: set[int] = set()
    seen_keys = {start.key}
    position_keys: list[int] = []
    revisited = 0
    walked = 0

    sight = sight_tally(engine)
    rest = rest_tally(engine)

    for step in spy_steps(engine, record):
        state, board = step.state, step.state.board
        for seat in step.seats:
            count_sight(engine, sight, state, seat)
            count_rest(engine, rest, state, seat)
        for move in step.moves:
            kind = mover_kind(engine, board, move)
            if kind is None:  # pragma: no cover - a legal move stands on its origin
                continue
            if moves_an_instance(move) and move_from(move) in start_squares:
                moved_squares.add(move_from(move))
            moves_by_type[kind] += 1
            if takes(engine, board, move):
                captures_by_type[kind] += 1

        if any((step.ply + k) % PLY_STRIDE == 0 for k in range(len(step.moves))):
            _sample_spy_step(
                engine, evaluator, step, epsilon, near_best, legal_counts, capture_counts
            )

        after = step.after
        if len(step.moves) == 1:
            spy["alternating_plies" if step.phase.name == "alternating" else "reply_turns"] += 1
            walked += 1
            position_keys.append(after.key)
            if after.key in seen_keys:
                revisited += 1
            else:
                seen_keys.add(after.key)
        else:
            spy[f"{step.phase.name}_rounds"] += 1
            count_round(rounds, step.moves, step.events)
            # Two plies a round, as the round walk counts them.
            walked += 2
            position_keys.extend((after.key, after.key))
            if after.key in seen_keys:
                revisited += 2
            else:
                seen_keys.add(after.key)

        lost = spy_holders(step.phase, state.side_to_move) - spy_holders(
            engine.phase(after), after.side_to_move
        )
        for player in sorted(lost):
            spy[f"lost_ply_{player}"] = walked
            spy[f"lost_how_{player}"] = spy_loss(engine, step, player)

    return GameAnalysis(
        seed=record.seed,
        agents=record.agents,
        near_best=near_best,
        legal_counts=legal_counts,
        moves_by_type=moves_by_type,
        captures_by_type=captures_by_type,
        capture_counts=capture_counts,
        revisited_plies=revisited,
        walked_plies=walked,
        starting_instances=len(start_squares),
        moved_instances=len(moved_squares),
        position_keys=position_keys,
        rounds=rounds,
        spy=spy,
        sight=sight,
        rest=rest,
    )


def _sample_spy_step(
    engine: Engine,
    evaluator: Evaluator,
    step: SpyStep,
    epsilon: float,
    near_best: list[int],
    legal_counts: list[int],
    capture_counts: list[int],
) -> None:
    """The probe on one sampled step of a spy game, appended to the three series: a turn as
    an alternating ply is, a round one sample for each seat with a choice."""
    state, board = step.state, step.state.board
    if len(step.moves) == 1:
        legal = step.lists[0]
        if len(legal) > 1:
            scores = _score_roots(engine, evaluator, state, legal)
            best = max(scores)
            near_best.append(sum(1 for s in scores if best - s <= epsilon))
            legal_counts.append(len(legal))
            capture_counts.append(sum(1 for m in legal if takes(engine, board, m)))
        return
    lists = (step.lists[0], step.lists[1])
    if step.phase.name == "announced":
        announcer = step.phase.announcer
        reading = read_announced(
            engine, evaluator, state, announcer, step.moves[announcer], lists
        )
    else:
        reading = read_round(engine, evaluator, state, lists)
    if reading is None:
        return
    for seat in (0, 1):
        if len(lists[seat]) > 1:
            near_best.append(reading.near_best(seat, epsilon))
            legal_counts.append(len(lists[seat]))
            capture_counts.append(sum(1 for move in lists[seat] if takes(engine, board, move)))


def _one_spy_blunder(
    engine: Engine, record: GameRecord, rng: random.Random
) -> BlunderTrial | None:
    """`_one_blunder` for a spy game: the trial's ply names a step and, in a round, a seat.

    A turn takes a random other move, as an alternating game's trial does; a blind round a
    move short of its best against the mix beside the other seat's recorded move, as a round
    game's does. In an announced round the move forced is short of its best by
    `read_announced`'s reading: into the answerer, beside the move recorded as announced;
    into the announcer, and then the answerer, who sees it, answers as the record's player
    would, asked of the core (`Engine.select_round`) at the record's seed."""
    target_ply = int(record.plies * BLUNDER_FRACTION)
    found = None
    for step in spy_steps(engine, record):
        if step.ply + len(step.moves) > target_ply:
            found = step
            break
    if found is None:
        return None
    state = found.state
    if len(found.moves) == 1:
        legal = found.lists[0]
        if len(legal) < 2:
            return None
        blundering = state.side_to_move
        alternatives = [m for m in legal if m != found.moves[0]] or legal
        state = engine.apply(state, rng.choice(alternatives))
    else:
        blundering = target_ply - found.ply
        pair = spy_round_blunder(engine, record, found, blundering, rng)
        if pair is None:
            return None
        state = engine.apply_round(state, pair[0], pair[1])
    resumed = finish_game(engine, record.agents, state, record.seed)

    return BlunderTrial(
        seed=record.seed,
        agents=record.agents,
        ply=target_ply,
        original_winner=record.winner,
        blundering_side=blundering,
        new_winner=resumed.winner,
    )


# -------------------- the round counters, public (the legality job)
#
# These eleven are the second reader's contract: `site/legality.py` (the legality job, Q1000)
# rebuilds what a browser claims by walking a round game with these same counters. Whoever
# changes a round event changes what that job must reproduce; the job's tests pin it.


def sight_tally(engine: Engine) -> dict[str, list[int]] | None:
    """`SIGHT_SERIES` empty, for a game with a sight, and `COMMAND_SERIES` beside it for one
    whose sight commands; ``None`` for every other game."""
    if engine.spec.sight is None:
        return None
    keys = SIGHT_SERIES + (COMMAND_SERIES if _commands(engine) else ())
    keys += WILD_SIGHT_SERIES if _wild_sees(engine) else ()
    return {key: [] for key in keys}


def count_sight(
    engine: Engine,
    tally: dict[str, list[int]] | None,
    state: GameState,
    seat: int,
    move: int | None = None,
) -> None:
    """One side-turn of `SIGHT_SERIES`: ``seat`` choosing on ``state``, as the core reports the
    effects there. In a round the state names nobody, so it is asked as the seat sees it.
    In a game whose sight commands, `COMMAND_SERIES` too, ``move`` being the move made."""
    if tally is None:
        return
    seen = state if state.side_to_move == seat else state.with_(side_to_move=seat)
    effects = engine.effects(seen)
    spec = engine.spec
    commands = "commanded" in tally
    under = [0] * (len(_SIGHT_BITS) if commands else 3)
    for square, bits in enumerate(effects.bits):
        if bits and spec.owner_of(state.board[square]) == seat:
            for index, bit in enumerate(_SIGHT_BITS[: len(under)]):
                under[index] += int(bool(bits & bit))
    row: tuple[int, ...] = (
        seat,
        effects.frozen,
        effects.disarmed,
        effects.shielded,
        int(effects.binds),
        *under[:3],
        int(effects.suppressed),
    )
    if commands:
        made = move is not None and bool(move & MOVE_FOREIGN)
        row += (
            *under[3:],
            effects.commanded,
            int(made),
            int(made and move is not None and takes(engine, state.board, move)),
        )
    if "wild_frozen" in tally:
        wild = [0, 0, 0]
        for square, bits in enumerate(effects.wild):
            if bits and spec.owner_of(state.board[square]) == seat:
                for index, bit in enumerate(_SIGHT_BITS[:3]):
                    wild[index] += int(bool(bits & bit))
        row += tuple(wild)
    for key, value in zip(tally, row, strict=True):
        tally[key].append(value)


def rest_tally(engine: Engine) -> dict[str, list[int]] | None:
    """`REST_SERIES` empty, for a game whose pieces rest; ``None`` for every other game."""
    if not engine.spec.rest:
        return None
    return {key: [] for key in REST_SERIES}


def count_rest(
    engine: Engine, tally: dict[str, list[int]] | None, state: GameState, seat: int
) -> None:
    """One side-turn of `REST_SERIES`: ``seat`` choosing on ``state``, as the core answers it."""
    if tally is None:
        return
    turn = engine.rest_turn(state, seat)
    row = (seat, int(turn.binds), int(turn.lifted), len(state.rest[seat]))
    for key, value in zip(REST_SERIES, row, strict=True):
        tally[key].append(value)


def mover_kind(engine: Engine, board: bytes, move: int) -> int | None:
    """The internal type a move counts under in `moves_by_type`, or `None` from an empty
    origin, which only a record that has drifted from its spec plays.

    A drop counts under the type it drops. A fold's own move (schema 22, revised at 24)
    counts under its end's internal type: a fold stands on no byte of the board, so its
    origin holds nothing, or a piece standing in it that does not move (Q873, Q844's
    finding: the pass used to stop here). Every other move counts under the type on its
    origin, a momentum end's own move under its end's byte. `site/legality.py::walk` reads
    a move through this too.
    """
    n_types = engine.spec.n_types
    kind = move_type(move)
    if kind in (MOVE_DROP, MOVE_BUY):
        # TC5: a buy, like a drop, is the type it places.
        return move_promotion(move) % n_types
    if kind == MOVE_DUCK:
        # Generation 42: the duck's moves count under its type, placed or passing as moved.
        duck = engine.spec.duck
        assert duck is not None
        return duck.piece_type
    if kind == MOVE_FOLD:
        portals = engine.spec.portals
        assert portals is not None
        return portals.end_types[move_fold(move)]
    occupant = board[move_from(move)]
    return None if occupant == 0 else engine.spec.type_of(occupant)


def moves_an_instance(move: int) -> bool:
    """Whether a move moves the piece on its origin square: a drop or a buy (TC5) moves none,
    a hire or a bribe (TC6) changes its side where it stands, and a fold's own move leaves
    the piece standing in it where it is."""
    return move_type(move) not in (MOVE_DROP, MOVE_BUY, MOVE_FOLD, MOVE_DUCK, MOVE_HIRE)


def starting_squares(engine: Engine, board: bytes) -> set[int]:
    """The squares of the opening board's pieces: the instances `instance_participation`
    counts.

    A portal is not one of them (Q875, `docs/reference/portal-showcase-ideas.md` beside
    finding 1). It is part of the board that pieces move through, held by no player and
    taken by no capture, so a fixed portal is not an idle piece and a moved one is not a
    piece that played. A fold stands on no byte and was never counted; a momentum end
    stands on its byte and is left out here, so the two kinds read alike.
    """
    portals = engine.spec.portals
    spec = engine.spec
    return {
        square
        for square in range(engine.spec.n_squares)
        if board[square] != 0
        and (portals is None or not portals.is_end(spec.type_of(board[square])))
    }


def takes(engine: Engine, board: bytes, move: int) -> bool:
    """Whether a move is read as a capture: it ends on an occupied square, and is not one of
    the three kinds that end on one and take nothing, as the core's `apply` reads them.

    A push (Q1074b5, `docs/plan-push.md` §3.3 C1) always ends on an occupied square, and
    takes only where it throws an enemy piece off the board: `CompiledSpec.push_takes`, the
    reading `Tables::push_takes` and `restrict`'s `takes` share. A push that moves its line
    on the board takes nothing, and nor does one that throws the pusher's own piece off.

    A rotation's target is its own square, which holds the piece that turns (Q875: it was
    read as a capture, so a game whose pieces turn offered one in nearly every position).
    A file-based castling can land the king on its own rook, or on its own square, and
    moves the rook rather than removing it. A fold's own move may end on a piece and leave
    it standing (`docs/archive/plan-portals-2.md` §3R.1 rule 6). A duck's pass (generation 42)
    names the duck's own square. These are the four `agents.rs::capture_gain` and
    `agents/minimax.py` leave out. A hire or a bribe (TC6) names the piece it buys, which
    changes sides and is not taken.
    """
    if move_type(move) == MOVE_PUSH:
        return engine.spec.push_takes(board, move)
    return board[move_to(move)] != 0 and move_type(move) not in (
        MOVE_ROTATE,
        MOVE_CASTLE,
        MOVE_FOLD,
        MOVE_DUCK,
        MOVE_HIRE,
    )


def round_tallies(engine: Engine) -> dict[str, int]:
    """`ROUND_COUNTS` at nought, and `SPEED_COUNTS` beside them where a speed is declared."""
    sim = engine.spec.simultaneous
    keys = ROUND_COUNTS + (SPEED_COUNTS if sim is not None and sim.speed else ())
    return dict.fromkeys(keys, 0)


def count_round(
    counts: dict[str, int], pair: tuple[int, ...], events: tuple[tuple[int, int, int], ...]
) -> None:
    """`ROUND_COUNTS` for one round: the round, a swap read off its words, its events; and
    `SPEED_COUNTS` where the tallies hold them (an ordered round's events, schema 28)."""
    counts["rounds"] += 1
    ordered = any(event == _ORDER for event, _square, _player in events)
    # Two pieces trading squares in an ordered round did not swap: the faster took the
    # slower where it stood (`docs/archive/plan-speed.md` §2.5 B).
    counts["swaps"] += int(not ordered and _is_swap(pair[0], pair[1]))
    for event, _square, player in events:
        if event == _COLLISION:
            counts["collisions"] += 1
            counts["collisions_both_lost"] += int(player < 0)
        elif event == _VOID:
            counts["void_moves"] += 1
            if ordered:
                counts["slower_void"] += 1
        elif event == _ORDER:
            counts["ordered_rounds"] += 1
        elif event == _LOST:
            counts["slower_lost"] += 1
        elif event == _UNPLAYED:
            counts["ended_after_faster"] += 1
        elif event == _INITIATIVE:
            counts["initiative_ties"] += 1
        elif event == _BARRED:
            counts["bars"] += 1


def spy_round_blunder(
    engine: Engine, record: GameRecord, step: SpyStep, seat: int, rng: random.Random
) -> tuple[int, int] | None:
    """The round a spy game's trial plays in place of ``step``, with a blunder forced into
    ``seat`` (`_one_spy_blunder`), in seat order; ``None`` where there is none to force."""
    state = step.state
    played, other = step.moves[seat], step.moves[1 - seat]
    if step.phase.name == "blind":
        forced = round_blunder(engine, state, seat, played, rng)
    else:
        announcer = step.phase.announcer
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        reading = read_announced(engine, evaluator, state, announcer, step.moves[announcer])
        if reading is None or len(reading.moves[seat]) < 2:
            return None
        choices = [m for m in reading.blunders(seat, _epsilon_for(engine)) if m != played]
        forced = rng.choice(choices) if choices else None
        if forced is not None and seat == announcer:
            native = native_spec(record.agents[1 - announcer], engine.spec.hash)
            if native is None:
                return None
            answer, _, _, _ = engine.select_round(
                state, 1 - announcer, native, eval_values(engine.spec), record.seed,
                announced=forced,
            )
            if answer is None:
                return None
            other = answer
    if forced is None:
        return None
    return (forced, other) if seat == 0 else (other, forced)
