"""Train a ruleset's net end to end, judge it against the built-in objective, and store it.

The store is the pair ``agents/nets.py`` reads (``learned-evaluator.md`` §4):
``nets/<ruleset-hash>-<procedure>.net``, the weights, and ``.json`` beside it with the
fields ``nets.load_net`` checks (procedure, ruleset hash, input layout, verdict, the
weights' SHA-256, the build fingerprint) plus what this trainer knows: the recipe, the
data's digest and size, the loss by epoch, the quantization error, and the duel.

**Nothing in the JSON depends on the clock or the machine's load**, so the same data on
the same machine and core writes the same bytes (``tests/test_training.py``). Wall time
and node rate are printed by ``chess-variant-sandbox train``, never stored.

**Verdicts.** ``accepted`` when the net beat the built-in objective at equal nodes by
``ACCEPT_ERRORS`` paired standard errors; ``refused`` otherwise; ``unchecked`` when the
duel was skipped (``--no-duel``). Only ``accepted`` plays: ``NetRecord.playing`` sends the
core no bytes for the other two, and the player is the built-in objective, move for move.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from chess_variant_sandbox.agents.nets import (
    ACCEPTED,
    INPUTS_VERSION,
    OUTPUT_SCALE,
    REFUSED,
    STORE_DIR,
    Net,
    NetError,
    NetRecord,
    register_net,
    store_paths,
)
from chess_variant_sandbox.agents.training.data import (
    Dataset,
    harvest,
    opening_plies_for,
    relabel,
    self_play,
)
from chess_variant_sandbox.agents.training.model import FloatNet, predict, quantize, train
from chess_variant_sandbox.agents.training.procedure import (
    ACCEPT_ERRORS,
    ACCEPT_NODES,
    ACCEPT_OPENING_PLIES,
    ACCEPT_PAIRS,
    ACCEPT_SEED,
    ROUND_ACCEPT_NODES,
    TRAINER_VERSION,
    Parent,
    Procedure,
)
from chess_variant_sandbox.metrics.components import plays_in_rounds

Report = Callable[[str], None]

UNCHECKED = "unchecked"
"""The verdict of a net whose duel was skipped. It never plays."""

# Games played, held out from training, to measure what quantization cost: the float
# net's output against the core's integer one on every position of them.
CHECK_GAMES = 4


@dataclass(frozen=True)
class Acceptance:
    """What the acceptance duel found. ``nodes`` is each side's total over the duel."""

    candidate: str
    reference: str
    score: float
    error: float
    games: int
    identical: int
    candidate_nodes: int
    reference_nodes: int
    accepted: bool
    # Which duel (§8.11): the first pair's seed, the most random opening plies (0 is the
    # start position), and how many distinct openings the pairs played.
    seed: int = ACCEPT_SEED
    opening_plies: int = ACCEPT_OPENING_PLIES
    openings: int = 1
    # The fixed rule's size in seed-pairs. Below `games / 2` only for a curtailed duel
    # (reference/compute-economy.md §4.2): the verdict is still the full duel's, the score the prefix's.
    planned: int = 0

    @property
    def decided_after(self) -> int:
        return self.games // 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "reference": self.reference,
            "seed": self.seed,
            "opening_plies": self.opening_plies,
            "score": round(self.score, 6),
            "error": round(self.error, 6),
            "games": self.games,
            "identical": self.identical,
            "openings": self.openings,
            "candidate_nodes": self.candidate_nodes,
            "reference_nodes": self.reference_nodes,
            "bar": f"score - {ACCEPT_ERRORS:g} x error > 0.5",
            "accepted": self.accepted,
            "decided_after": self.decided_after,
            "planned_pairs": self.planned or self.decided_after,
        }


def accepts(estimates: Sequence[float]) -> bool:
    """The acceptance verdict over a duel's seed-pairs: ``score - 2 x error > 0.5``."""
    from chess_variant_sandbox.tournament.duel import summary

    score, error = summary(estimates)
    return score - ACCEPT_ERRORS * error > 0.5


@dataclass(frozen=True)
class Trained:
    """One training's outcome: the record the player reads, and where it was written."""

    record: NetRecord
    procedure: Procedure
    data: Dataset
    history: tuple[float, ...]
    quantization_error: float
    acceptance: Acceptance | None
    meta_path: Path
    # Seconds, for the report only: never stored.
    seconds: dict[str, float]

    @property
    def player(self) -> str:
        nodes = ROUND_ACCEPT_NODES if self.procedure.rounds else ACCEPT_NODES
        return f"{self.procedure.searcher}:net={self.procedure.digest},nodes={nodes}"


def store_payload(
    record: NetRecord,
    procedure: Procedure,
    data: Dataset,
    history: tuple[float, ...],
    w0_limit: int,
    quantization_error: float,
    acceptance: Acceptance | None,
) -> dict[str, Any]:
    """The JSON beside the weights: ``nets.save_net``'s fields first, then the trainer's."""
    return {
        "procedure": record.procedure,
        "ruleset_hash": record.ruleset_hash,
        "inputs_version": INPUTS_VERSION,
        "verdict": record.verdict,
        "weights_sha256": hashlib.sha256(record.weights).hexdigest(),
        "build": record.build,
        "digest": record.digest,
        "trainer_version": TRAINER_VERSION,
        "recipe": procedure.as_dict(),
        "data": {
            "digest": data.digest,
            "games": data.games,
            "positions": data.positions,
            "most_active": data.most_active,
        },
        "training": {
            "loss_by_epoch": list(history),
            "w0_limit": w0_limit,
            "quantization_error": round(quantization_error, 6),
        },
        "acceptance": None if acceptance is None else acceptance.as_dict(),
    } | lineage(procedure)


def lineage(procedure: Procedure) -> dict[str, Any]:
    """A generation's store names the net it learnt from (Q677); others carry no key, so
    no store written before a generation existed changes a byte."""
    if procedure.parent is None:
        return {}
    return {
        "lineage": {
            "parent": procedure.parent.procedure,
            "parent_weights_sha256": procedure.parent.weights_sha256,
            "teacher": procedure.teacher,
        }
    }


def read_parent(path: str | Path, ruleset_hash: str) -> tuple[Parent, bytes]:
    """A parent net from its weights file, ``<ruleset-hash>-<procedure>.net`` (or the
    ``.json`` beside it), and its bytes.

    The file's name gives the procedure and the ruleset, which must be ``ruleset_hash``;
    the bytes must parse as a net, and where a JSON lies beside them its
    ``weights_sha256`` must be theirs. The verdict is not read: a generation may learn
    from a refused net (archive/plan-teacher §9.3, gardner's parent), which plays only in the
    process that registers it (``register_parent``).
    """
    weights = Path(path).with_suffix(".net")
    stem_hash, _, procedure = weights.stem.partition("-")
    if stem_hash != ruleset_hash:
        raise NetError(f"{weights.name} is a net for ruleset {stem_hash}, not {ruleset_hash}")
    blob = weights.read_bytes()
    Net.parse(blob)
    sha = hashlib.sha256(blob).hexdigest()
    meta = weights.with_suffix(".json")
    if meta.exists():
        recorded = json.loads(meta.read_text(encoding="utf-8")).get("weights_sha256")
        if recorded != sha:
            raise NetError(f"{weights} does not hash to what {meta} records")
    return Parent(procedure=procedure, weights_sha256=sha), blob


def register_parent(ruleset_hash: str, parent: Parent, blob: bytes) -> None:
    """Make ``net=<parent>`` play these bytes in this process (and so in its threads).

    Registered as accepted whatever the store's verdict: the procedure's name carries the
    bytes' hash, so the generation is named by exactly this player. Spawned harvest
    workers register it again from ``data.parent_bytes``.
    """
    if hashlib.sha256(blob).hexdigest() != parent.weights_sha256:
        raise NetError(f"these bytes are not the parent {parent.procedure}'s weights")
    register_net(NetRecord(parent.procedure, ruleset_hash, ACCEPTED, blob))


def write_store(record: NetRecord, payload: dict[str, Any], root: str | Path = STORE_DIR) -> Path:
    """Write the weights and the JSON; the weights are parsed first, so a bad blob never lands."""
    Net.parse(record.weights)
    meta, weights = store_paths(record.ruleset_hash, record.procedure, root)
    meta.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(record.weights)
    meta.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return meta


def quantization_error(spec: dict[str, Any], net: FloatNet, blob: bytes, procedure: Procedure) -> float:
    """Mean ``|float - integer|`` output, in built-in units, over ``CHECK_GAMES`` held-out games.

    The integer outputs are the core's (``Core.net_line``), so this is also the check that
    the core reads the bytes as the net the trainer meant.
    """
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.engine.engine import _pack

    engine = Engine(spec)
    # The held-out games are new games only: a replay set (Q712) is training data, and
    # replaying it here would play the parent's whole schedule to check four games.
    held_out = procedure.with_(games=CHECK_GAMES, seed=procedure.seed + 1, replay=None)
    games = self_play(spec, held_out, workers=1)
    rows: list[list[int]] = []
    sides: list[int] = []
    cores: list[float] = []
    for record in games:
        state = engine.initial_state()
        line = engine.core.net_line(_pack(state), blob, list(record.moves))
        if engine.simultaneous:
            # Q918g: a round game's line is a position a round, every one seat 0's.
            for raw, inputs, _fresh in line:
                rows.append(list(inputs))
                sides.append(0)
                cores.append(raw / OUTPUT_SCALE)
            continue
        for ply, (raw, inputs, _fresh) in enumerate(line):
            rows.append(list(inputs))
            sides.append(int(state.side_to_move))
            cores.append(raw / OUTPUT_SCALE)
            if ply < len(record.moves):
                state = engine.apply(state, record.moves[ply])
    if not rows:
        return 0.0
    floats = predict(net, rows, sides).astype(np.float64)
    return float(np.mean(np.abs(floats - np.asarray(cores, dtype=np.float64))))


def accept(
    spec: dict[str, Any],
    procedure: Procedure,
    pairs: int = ACCEPT_PAIRS,
    nodes: int | None = None,
    workers: int = 0,
    seed: int = ACCEPT_SEED,
    opening_plies: int = ACCEPT_OPENING_PLIES,
    reference: str | None = None,
    curtail: bool = False,
) -> Acceptance:
    """The net against the built-in objective at equal nodes, on paired seeds.

    ``reference`` (Q677) puts another player in the built-in's seat on the same schedule:
    the parent net for archive/plan-teacher §9.1's P, ``minimax:nodes=25600`` for A25. The
    verdict's bar is the same; only the duel against the built-in at equal nodes is an
    acceptance (``train_ruleset`` never passes one).

    The net must already be registered as accepted, or the candidate would be the built-in
    objective under another name (and the duel exactly 0.5).

    With ``opening_plies`` (§8.11) each pair opens as a self-play game does: up to that
    many plies of the core's ``random`` agent, every stream named by the pair's schedule
    seed, so both games of a pair start from one position with the seats swapped and the
    duel samples many positions. 0 is the ordinary runner from the start position.

    ``curtail`` (reference/compute-economy.md §4.2) stops the duel at the first schedule-order prefix of
    seed-pairs whose every completion gives the same verdict, in batches of the worker
    count: the same verdict as the full duel, by construction, with the prefix's score and
    ``decided_after`` of ``planned``. Off by default because the score is not only the
    verdict's: plan-teacher §3's G-S and G-G compare D2 scores between arms, and a prefix's
    score is a different measurement from the 200 pairs those gates were registered on.
    """
    from chess_variant_sandbox.tournament.checkpoint import run_tournament
    from chess_variant_sandbox.tournament.duel import curtailed_duel, grade
    from chess_variant_sandbox.tournament.runner import TournamentConfig

    # Q918g: on a round game both seats are the round player, `matrix:`, at the round
    # duel's budget unless one is named (`ROUND_ACCEPT_NODES`).
    rounds = plays_in_rounds(spec)
    kind = "matrix" if rounds else "minimax"
    if nodes is None:
        nodes = ROUND_ACCEPT_NODES if rounds else ACCEPT_NODES
    candidate = f"{kind}:net={procedure.digest},nodes={nodes}"
    reference = reference or f"{kind}:nodes={nodes}"
    if reference == candidate:
        raise ValueError(f"a duel needs two players; both seats are {candidate}")
    config = TournamentConfig(
        agents=(candidate, reference),
        games_per_pairing=pairs,
        seed=seed,
        workers=workers,
        include_self_play=False,
        collect_series=False,
        paired=True,
    )

    def play(first: int, count: int) -> list[Any]:
        block = replace(config, games_per_pairing=count, seed=first)
        if opening_plies <= 0:
            return list(run_tournament(spec, block).records)
        return _opening_duel(spec, block, opening_plies)

    if curtail:
        curtailed = curtailed_duel(
            spec, candidate, reference, pairs, accepts, seed, workers, play=play
        )
        found, accepted = curtailed.found, curtailed.verdict
        records = list(curtailed.records)
    else:
        records = play(seed, pairs)
        found = grade(records, candidate, reference)
        accepted = accepts(found.pairs)
    # A round game opens with whole rounds, `opening_plies_for // 2` of them (Q918g).
    def opened(seed: int) -> int:
        plies = opening_plies_for(seed, opening_plies)
        return 2 * (plies // 2) if rounds else plies

    openings = 1 if opening_plies <= 0 else len(
        {tuple(r.moves[: opened(r.seed)]) for r in records}
    )
    spent = [0, 0]
    for record in records:
        seat = 0 if record.agents[0] == candidate else 1
        spent[0] += record.nodes[seat]
        spent[1] += record.nodes[1 - seat]
    return Acceptance(
        candidate=candidate,
        reference=reference,
        score=found.score,
        error=found.error,
        games=found.games,
        identical=found.identical,
        candidate_nodes=spent[0],
        reference_nodes=spent[1],
        accepted=accepted,
        seed=seed,
        opening_plies=max(0, opening_plies),
        openings=openings,
        planned=pairs,
    )


def _opening_duel(spec: dict[str, Any], config: Any, most: int) -> list[Any]:
    """The paired schedule's games, each opened as a self-play game is (``data``'s opening).

    The opening is a function of the seed alone, and both games of a pair share the seed,
    so they share the opening. One game per core call on the config's threads; records in
    schedule order, so the answer does not depend on the thread count. Self-play's own
    loop (``data._play_tasks``), so a campaign directory resumes it as it does self-play.
    """
    from chess_variant_sandbox.agents.training.data import _play_tasks
    from chess_variant_sandbox.tournament.runner import schedule

    return _play_tasks(spec, config, schedule(config), most)


def train_ruleset(
    spec: dict[str, Any],
    procedure: Procedure,
    root: str | Path = STORE_DIR,
    workers: int = 0,
    pairs: int = ACCEPT_PAIRS,
    nodes: int | None = None,
    duel: bool = True,
    data: Dataset | None = None,
    report: Report | None = None,
    curtail: bool = False,
) -> Trained:
    """Self-play, harvest, train, quantize, judge, store; returns what was stored.

    ``data`` skips the self-play and the harvest (a dataset saved earlier, or the same
    positions twice for the determinism test). ``curtail`` curtails the acceptance duel
    (``accept``). The store's verdict is registered in this process, so a ``net=`` player
    built after this call plays what the store says.
    """
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.measurement import built_fingerprint

    say = report or (lambda _line: None)
    engine = Engine(spec)
    digest = engine.spec.hash
    n_inputs = int(engine.core.net_inputs())
    seconds: dict[str, float] = {}

    if data is None:
        started = time.perf_counter()
        records = self_play(spec, procedure, workers=workers)
        seconds["self_play"] = time.perf_counter() - started
        say(f"  {len(records)} self-play games in {seconds['self_play']:.1f}s")
        started = time.perf_counter()
        data = harvest(spec, records, procedure, workers=max(1, workers), report=report)
        seconds["harvest"] = time.perf_counter() - started
        say(f"  {data.positions} positions harvested in {seconds['harvest']:.1f}s")
    if data.n_inputs != n_inputs:
        raise NetError(
            f"the data has {data.n_inputs} inputs a position and this ruleset's layout has "
            f"{n_inputs}; it was harvested from another game"
        )

    # The labels this procedure asks for, remade from the data's line (Q658): the harvest's
    # own at td_lambda 0, the TD(λ) blend otherwise. Data saved without a line keeps its
    # targets, and refuses a TD label.
    data = relabel(data, procedure)

    started = time.perf_counter()
    net = train(data, procedure, report=report)
    blob = quantize(net)
    seconds["train"] = time.perf_counter() - started
    error = quantization_error(spec, net, blob, procedure)
    say(f"  trained in {seconds['train']:.1f}s; quantization costs {error:.4f} units a position")

    build = built_fingerprint()
    acceptance: Acceptance | None = None
    verdict = UNCHECKED
    if duel:
        register_net(NetRecord(procedure.digest, digest, ACCEPTED, blob, build))
        started = time.perf_counter()
        acceptance = accept(
            spec, procedure, pairs=pairs, nodes=nodes, workers=workers, curtail=curtail
        )
        seconds["duel"] = time.perf_counter() - started
        verdict = ACCEPTED if acceptance.accepted else REFUSED
    record = NetRecord(procedure.digest, digest, verdict, blob, build)
    register_net(record)
    payload = store_payload(
        record, procedure, data, tuple(net.history), net.w0_limit, error, acceptance
    )
    meta = write_store(record, payload, root)
    return Trained(
        record=record,
        procedure=procedure,
        data=data,
        history=tuple(net.history),
        quantization_error=error,
        acceptance=acceptance,
        meta_path=meta,
        seconds=seconds,
    )
