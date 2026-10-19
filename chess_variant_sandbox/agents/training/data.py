"""Self-play data through the core: positions as the net's active inputs, and their labels.

**Who computes what** (CLAUDE.md rules 2 and 3). The core plays the games, as a whole
schedule (``tournament.runner.run_tournament``). The core reports every position's
active inputs along each game (``Core.net_line``, handed a zero net of the right size:
its outputs are ignored, its input lists are the layout's, from the one definition in
``net.rs``). The core scores the position for the optional search label (the analysis
probe, ``Engine.root_scores``). Python replays each game's moves with ``Engine.apply``
only to hold the position and read whose turn it is; it chooses nothing and judges
nothing.

**The label** is the side to move's expectation in ``[0, 1]``:
``w * result + (1 - w) * sigmoid(probe / scale)``, where ``result`` is the game's
outcome for that side (1, 0.5, 0) and ``probe`` the best root score the probe found, in
built-in units (§2 E's ``λ·result + (1 − λ)·search score``). The probe is not the player
that played the games: it is the full-window search the analysis pass rates decisions
with, at the rated players' goal and immunity weights, the same ruler on every board.

**TD(λ)** (Q658, ``Procedure.td_lambda``): the probe's part of the label may instead be
the TD(λ) value along the game's own line, the probe's later values ending in the result
(``td_labels``). The harvest keeps each sample's game, ply, result and probe value (the
``Dataset``'s line), so the label is remade from saved data (``labels``) without playing
or probing again; at λ 0 that remake is the harvest's label bit for bit.

**The player teacher** (Q665, ``Procedure.label_search = "player"``): ``probe`` is
instead the root value of the playing search itself, ``minimax:nodes=<label_nodes>``,
asked of the core once per game for all its sampled positions (``Core.search_values``),
each position's stream named by the game's seed and the ply (``label_seed``).

**Diverse** (Q637, ``Procedure.opening_plies``): each game may open with a few plies of
the core's ``random`` agent, drawn from streams the game's schedule seed names, so
self-play is not one line replayed (``learned-evaluator.md`` §8.10).

**Round games** (Q918g, ``docs/archive/plan-simultaneous.md`` §6.4): a record's moves are its rounds' in
pairs, a position is a round's board (the start and after each round, ``Core.net_line`` walks
them with ``make_round``), its side is always 0 and its label player 0's expectation (the net
reads seat 0's head and negates it for seat 1), sampled from ply ``skip_plies`` on every
``stride`` rounds. The teacher is the round player's own value of the round
(``Core.round_values``, ``matrix:nodes=<label_nodes>``); the core has no probe for rounds. A
random opening is up to ``opening_plies // 2`` rounds of the core's ``random`` agent, each
seat's move from its own stream (``opening_seed(seed, 2r + seat)``).

**Deterministic.** The schedule fixes every seed before a thread starts, and the chunks
are joined in game order, so the dataset is the same at any worker count
(``tests/test_training.py``).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from chess_variant_sandbox.agents.nets import encode
from chess_variant_sandbox.agents.training.procedure import Parent, Procedure, Replay

Report = Callable[[str], None]


@dataclass(frozen=True)
class Dataset:
    """Positions in compressed rows: position ``i``'s inputs are ``inputs[offsets[i]:offsets[i + 1]]``.

    ``sides`` is who is to move (it picks the output head), ``targets`` the label for that
    side. ``games`` is how many games the positions came from.

    **The line** (Q658, optional; every harvest since writes it, older ``.npz`` files lack
    it): where each sample sits and what its label was made of, so a label can be remade
    from saved data (``labels``, ``relabel``). ``game`` is the game's index in the
    harvest, ``ply`` the sample's ply in it, ``results`` the game's result for the side to
    move, ``values`` the probe's squashed score for that side (float64, so the label is
    rebuilt bit for bit), NaN where the probe gave none and the label was the result.
    Samples of one game are contiguous and in ply order.
    """

    n_inputs: int
    offsets: np.ndarray
    inputs: np.ndarray
    sides: np.ndarray
    targets: np.ndarray
    games: int
    game: np.ndarray | None = None
    ply: np.ndarray | None = None
    results: np.ndarray | None = None
    values: np.ndarray | None = None
    # Q665: which search made ``values`` (``Procedure.label_search``), so a label is never
    # remade from another teacher's values under this one's name. ``None`` for data saved
    # before Q665, which was the probe's; not in the digest.
    teacher: str | None = None

    def __post_init__(self) -> None:
        count = len(self.sides)
        if len(self.offsets) != count + 1 or len(self.targets) != count:
            raise ValueError("a dataset's offsets, sides and targets disagree on its size")
        if count and int(self.offsets[-1]) != len(self.inputs):
            raise ValueError("a dataset's last offset is not the number of inputs")
        line = (self.game, self.ply, self.results, self.values)
        partial = any(column is None or len(column) != count for column in line)
        if partial and any(column is not None for column in line):
            raise ValueError("a dataset's line needs game, ply, results and values, one per sample")

    @property
    def has_line(self) -> bool:
        """Whether the samples carry their game, ply, result and probe value (Q658)."""
        return self.values is not None

    @property
    def positions(self) -> int:
        return len(self.sides)

    @property
    def most_active(self) -> int:
        """The most inputs any one position has on."""
        return int(np.diff(self.offsets).max()) if self.positions else 0

    @property
    def digest(self) -> str:
        """SHA-256 over the arrays as little-endian bytes: what 'the same data' means.

        The arrays the trainer fits (inputs, sides, targets), not the line: the line is
        how the targets were made, and a label remade from it is a new ``targets`` and so
        a new digest. Two harvests of the same games with the same labels share it, with
        the line or without.
        """
        sha = hashlib.sha256()
        sha.update(np.int64(self.n_inputs).astype("<i8").tobytes())
        sha.update(np.int64(self.games).astype("<i8").tobytes())
        sha.update(self.offsets.astype("<i8").tobytes())
        sha.update(self.inputs.astype("<i4").tobytes())
        sha.update(self.sides.astype("<i1").tobytes())
        sha.update(self.targets.astype("<f4").tobytes())
        return sha.hexdigest()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "n_inputs": np.int64(self.n_inputs),
            "games": np.int64(self.games),
            "offsets": self.offsets.astype("<i8"),
            "inputs": self.inputs.astype("<i4"),
            "sides": self.sides.astype("<i1"),
            "targets": self.targets.astype("<f4"),
        }
        if self.has_line:
            assert self.game is not None and self.ply is not None and self.results is not None
            assert self.values is not None
            arrays.update(
                game=self.game.astype("<i4"),
                ply=self.ply.astype("<i4"),
                results=self.results.astype("<f8"),
                values=self.values.astype("<f8"),
            )
        if self.teacher is not None:
            arrays["teacher"] = np.array(self.teacher)
        with target.open("wb") as handle:
            np.savez(handle, **arrays)
        return target

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        with np.load(Path(path)) as data:
            line = "values" in data.files
            return cls(
                n_inputs=int(data["n_inputs"]),
                offsets=data["offsets"].astype(np.int64),
                inputs=data["inputs"].astype(np.int32),
                sides=data["sides"].astype(np.int8),
                targets=data["targets"].astype(np.float32),
                games=int(data["games"]),
                game=data["game"].astype(np.int32) if line else None,
                ply=data["ply"].astype(np.int32) if line else None,
                results=data["results"].astype(np.float64) if line else None,
                values=data["values"].astype(np.float64) if line else None,
                teacher=str(data["teacher"]) if "teacher" in data.files else None,
            )


def joined(parts: Sequence[Dataset]) -> Dataset:
    """``parts`` one after another, as one harvest of their games in that order (E3b).

    Each part's game ids are moved past the games before it, so a game's samples stay one
    run of ids and no TD(λ) line runs from one part's last game into the next one's first
    (``td_labels`` ends a line where the id changes). Harvests of ``link_games``' parts,
    joined in link order, are the whole chained harvest, array for array. The parts must
    agree on the input layout, the teacher and whether they carry the line.
    """
    if not parts:
        raise ValueError("joining needs at least one dataset")
    first = parts[0]
    if any(part.n_inputs != first.n_inputs for part in parts):
        raise ValueError("these datasets are of different input layouts; they cannot be joined")
    if any(part.teacher != first.teacher for part in parts):
        teachers = ", ".join(sorted({str(part.teacher) for part in parts}))
        raise ValueError(f"these datasets were labelled by different teachers ({teachers})")
    if any(part.has_line != first.has_line for part in parts):
        raise ValueError("some of these datasets carry their game line and some do not")
    lengths = np.concatenate([np.diff(part.offsets) for part in parts])
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    game: np.ndarray | None = None
    ply: np.ndarray | None = None
    results: np.ndarray | None = None
    values: np.ndarray | None = None
    if first.has_line:
        before = np.cumsum([0] + [part.games for part in parts[:-1]])
        game = np.concatenate([
            np.asarray(part.game, dtype=np.int64) + int(shift)
            for part, shift in zip(parts, before, strict=True)
        ]).astype(np.int32)
        ply = np.concatenate([np.asarray(part.ply) for part in parts]).astype(np.int32)
        results = np.concatenate([np.asarray(part.results) for part in parts]).astype(np.float64)
        values = np.concatenate([np.asarray(part.values) for part in parts]).astype(np.float64)
    return Dataset(
        n_inputs=first.n_inputs,
        offsets=offsets,
        inputs=np.concatenate([part.inputs for part in parts]).astype(np.int32),
        sides=np.concatenate([part.sides for part in parts]).astype(np.int8),
        targets=np.concatenate([part.targets for part in parts]).astype(np.float32),
        games=sum(part.games for part in parts),
        game=game,
        ply=ply,
        results=results,
        values=values,
        teacher=first.teacher,
    )


def squash(score: float, scale: float) -> float:
    """``sigmoid(score / scale)``, safe at the probe's win scores (about 10^6)."""
    return 0.5 * (1.0 + math.tanh(score / (2.0 * scale)))


def td_labels(
    values: np.ndarray,
    results: np.ndarray,
    sides: np.ndarray,
    game: np.ndarray,
    td_lambda: float,
) -> np.ndarray:
    """The TD(λ) value of every sample along its game's line (Q658, plan-teacher §2.2).

    Walking each game backwards over its *sampled* positions ``1..n``, for the side to
    move at each::

        L_n = (1 - λ) V_n + λ R_n
        L_i = (1 - λ) V_i + λ L'_{i+1}

    where ``V`` is the probe's squashed value, ``R`` the game's result, and ``L'_{i+1}``
    the next sample's ``L`` seen from sample ``i``'s side: ``1 - L`` when the side to move
    changed between them, ``L`` when it did not (a game may give one side two plies, and a
    stride may skip a ply). Unrolled, ``L_i = (1 - λ) Σ_k λ^k V_{i+k} + λ^(n-i) R``: λ 0
    is the probe alone, λ 1 the result alone (the line's only return; nothing is paid on
    the way). A sample whose probe gave nothing (NaN) counts its result as its value.

    A pure function of the arrays and λ, in float64; samples of one game must be
    contiguous and in ply order, as ``harvest`` writes them.
    """
    count = len(values)
    probe = np.where(np.isnan(values), results, values).astype(np.float64)
    out = np.empty(count, dtype=np.float64)
    keep = 1.0 - td_lambda
    for i in range(count - 1, -1, -1):
        if i == count - 1 or game[i + 1] != game[i]:
            later = float(results[i])
        else:
            later = float(out[i + 1]) if sides[i + 1] == sides[i] else 1.0 - float(out[i + 1])
        out[i] = keep * probe[i] + td_lambda * later
    return out


def labels(data: Dataset, procedure: Procedure) -> np.ndarray:
    """``data``'s labels under ``procedure``, remade from its line; float32, as ``targets``.

    ``w x result + (1 - w) x L`` with ``L`` the TD(λ) value (``td_labels``), or the result
    where the probe gave nothing. At ``td_lambda`` 0, ``L`` is the probe's value and this
    is ``harvest``'s own arithmetic, so the labels are ``targets`` bit for bit. Data
    without a line keeps its ``targets`` and refuses a TD label.
    """
    teacher = data.teacher or "probe"
    if procedure.searched and teacher != procedure.teacher:
        raise ValueError(
            f"this data was labelled by the {teacher}; the procedure asks for the "
            f"{procedure.teacher} (label_search): harvest it again"
        )
    if not data.has_line:
        if procedure.td_lambda:
            raise ValueError(
                "a TD(λ) label needs the samples' game line; this data was saved before "
                "harvests kept it (Q658): harvest it again"
            )
        return data.targets
    assert data.values is not None and data.results is not None and data.game is not None
    missing = np.isnan(data.values)
    if procedure.searched and data.positions and bool(missing.all()):
        raise ValueError("this data was harvested without the probe; a search label needs it")
    searched = td_labels(data.values, data.results, data.sides, data.game, procedure.td_lambda)
    weight = procedure.result_weight
    blended = weight * data.results + (1.0 - weight) * searched
    return np.where(missing, data.results, blended).astype(np.float32)


def relabel(data: Dataset, procedure: Procedure) -> Dataset:
    """``data`` with ``procedure``'s labels as its targets (the line is kept, so it can be remade)."""
    from dataclasses import replace

    return replace(data, targets=labels(data, procedure))


def zero_net(n_inputs: int) -> bytes:
    """The smallest store the core accepts for this layout: asked for inputs, not outputs."""
    head = ([0], [0], [0], [0], [0], 0)
    return encode(n_inputs, 1, 1, 1, [0] * n_inputs, [0], [head, head])


def _config(procedure: Procedure, workers: int) -> Any:
    from chess_variant_sandbox.tournament.runner import TournamentConfig

    return TournamentConfig(
        agents=procedure.agents,
        games_per_pairing=procedure.games,
        seed=procedure.seed,
        workers=workers,
        collect_series=False,
    )


def _hashed(*parts: object) -> int:
    """A 64-bit number named by ``parts``: how a game's seed becomes its opening's streams."""
    blob = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


def opening_plies_for(seed: int, most: int) -> int:
    """How many random plies the game at ``seed`` opens with: uniform in ``0..most``."""
    return _hashed("opening-plies", seed) % (most + 1) if most > 0 else 0


def opening_seed(seed: int, ply: int) -> int:
    """The stream the core's ``random`` agent draws opening ply ``ply`` of game ``seed`` from."""
    return _hashed("opening-move", seed, ply)


def diverse_game(
    engine: Any, task: tuple[int, str, str], most: int
) -> Any:
    """One self-play game with a random opening: the core picks every move.

    The opening's moves are the core's ``random`` agent's (``choose_move``), each from its
    own stream named by the schedule seed and the ply; the roster then finishes the game
    from that position at the schedule seed (``finish_game``, the core's ``play_from``).
    Python only holds the position between the two and asks whether it is over.
    """
    from dataclasses import replace

    from chess_variant_sandbox.tournament.play import choose_move, finish_game

    seed, first, second = task
    if engine.simultaneous:
        # Q918g: a round game opens with random rounds, both seats from their own streams.
        state, opening = _round_opening(engine, seed, most)
    else:
        state = engine.initial_state()
        opening = []
        for ply in range(opening_plies_for(seed, most)):
            legal = engine.legal_moves(state)
            if engine.terminal(state, legal) is not None:
                break
            move = choose_move(engine, "random", state, opening_seed(seed, ply))
            if move is None:
                break
            opening.append(move)
            state = engine.apply(state, move)
    record = finish_game(engine, (first, second), state, seed)
    moves = opening + list(record.moves)
    return replace(record, moves=moves, plies=len(moves))


def _round_opening(engine: Any, seed: int, most: int) -> tuple[Any, list[int]]:
    """Q918g: a round game's random opening, ``opening_plies_for(seed, most) // 2`` rounds
    (a round is two plies, ``docs/reference/rounds.md`` §2.6), each seat's move the core's
    ``random`` agent's from its own stream (``opening_seed(seed, 2r + seat)``). The position
    and the moves, each round's two in seat order; it stops where a round is over before
    anyone chooses."""
    from chess_variant_sandbox.agents.base import native_spec
    from chess_variant_sandbox.tournament.play import eval_values

    agent = native_spec("random")
    values = eval_values(engine.spec)
    state = engine.initial_state()
    moves: list[int] = []
    for round_index in range(opening_plies_for(seed, most) // 2):
        if engine.terminal(state) is not None:
            break
        pair = [
            engine.select_round(state, seat, agent, values, opening_seed(seed, 2 * round_index + seat))[0]
            for seat in (0, 1)
        ]
        if pair[0] is None or pair[1] is None:
            break
        moves.extend((pair[0], pair[1]))
        state = engine.apply_round(state, pair[0], pair[1])
    return state, moves


def self_play(
    spec: dict[str, Any], procedure: Procedure, workers: int = 0
) -> list[Any]:
    """The procedure's games, played by the core.

    Without a random opening (``opening_plies`` 0), the ordinary tournament runner plays
    the schedule. With one, the same schedule (its seeds fixed before a thread starts) is
    played a game per core call on ``workers`` threads -- the core releases the
    interpreter for the game, so they run side by side -- and the records come back in
    schedule order, so the games do not depend on the thread count.

    With a replay set (Q712, ``Procedure.replay``) the new games come first, in schedule
    order (``new_game_count`` of them), then the replay schedule's games without its
    built-in-only block (``replay_games``), so a harvest of these records is the union
    E1 trains on, and every arm harvesting them shares its positions. A chained replay set
    (E3b, ``Replay.prior``) adds each link in order, one back and then two back, each
    without its built-in-only block: ``link_games(spec, procedure, k)`` for k = 0, 1, ...
    concatenated are these records.
    """
    # A generation's net seat must be the parent the name carries, or it would play the
    # built-in objective (or another net) under the generation's name; and every replay
    # link's net its own, checked before any game is played.
    parent_bytes(spec, procedure)
    replay_bytes(spec, procedure)
    games = new_games(spec, procedure, workers)
    if procedure.replay is not None:
        games.extend(replay_games(spec, procedure, workers))
    return games


def new_games(spec: dict[str, Any], procedure: Procedure, workers: int = 0) -> list[Any]:
    """The procedure's own schedule played by the core, without any replay set: link 0."""
    from chess_variant_sandbox.tournament.checkpoint import run_tournament
    from chess_variant_sandbox.tournament.runner import schedule

    config = _config(procedure, workers)
    parent_bytes(spec, procedure)
    if procedure.opening_plies <= 0:
        return list(run_tournament(spec, config).records)
    return _play_tasks(spec, config, schedule(config), procedure.opening_plies)


def _play_tasks(spec: dict[str, Any], config: Any, tasks: list[Any], most: int) -> list[Any]:
    """These schedule tasks, each opened by up to ``most`` random plies, in task order."""
    from concurrent.futures import ThreadPoolExecutor

    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.tournament import checkpoint

    engine = Engine(spec)
    # Q732: inside a campaign directory, in sealed pieces a restart skips; the same games.
    resumable = checkpoint.play_games(
        {"ruleset_hash": engine.spec.hash, "opening_plies": most},
        config.agents,
        tasks,
        lambda piece: [diverse_game(engine, task, most) for task in piece],
        config.resolved_workers(),
    )
    if resumable is not None:
        return resumable
    with ThreadPoolExecutor(max_workers=config.resolved_workers()) as pool:
        return list(pool.map(lambda task: diverse_game(engine, task, most), tasks))


def new_game_count(procedure: Procedure) -> int:
    """How many of ``self_play``'s records are the procedure's new games (the rest replay)."""
    from chess_variant_sandbox.tournament.runner import schedule

    return len(schedule(_config(procedure, 1)))


def replay_links(procedure: Procedure) -> tuple[Replay, ...]:
    """The procedure's replay links, nearest first (``Replay.links``); none without a set."""
    return procedure.replay.links if procedure.replay is not None else ()


def link_count(procedure: Procedure) -> int:
    """How many parts ``self_play``'s records come in: the new games, then each link."""
    return 1 + len(replay_links(procedure))


def _replay_link(procedure: Procedure, link: int) -> Replay:
    """Replay link ``link`` (1 one back, 2 two back, ...) of ``procedure``."""
    links = replay_links(procedure)
    if not 1 <= link <= len(links):
        raise ValueError(
            f"replay link {link} is not one of this procedure's "
            f"({'none' if not links else f'1..{len(links)}'}; 0 is the new games)"
        )
    return links[link - 1]


def _link_config(replay: Replay, workers: int) -> Any:
    from chess_variant_sandbox.tournament.runner import TournamentConfig

    return TournamentConfig(
        agents=replay.agents,
        games_per_pairing=replay.games,
        seed=replay.seed,
        workers=workers,
        collect_series=False,
    )


def _names_a_net(agent: str) -> bool:
    from chess_variant_sandbox.agents.training.procedure import net_option

    return net_option(agent) is not None


def _link_tasks(replay: Replay) -> list[tuple[int, str, str]]:
    from chess_variant_sandbox.tournament.runner import schedule

    return [
        task for task in schedule(_link_config(replay, 1))
        if _names_a_net(task[1]) or _names_a_net(task[2])
    ]


def replay_tasks(procedure: Procedure, link: int = 1) -> list[tuple[int, str, str]]:
    """The replay schedule's games that are played: every task of the parent's schedule
    (``tournament.runner.schedule``, its seeds as the parent's were) in which some seat
    names a net. The built-in-only block is left out here, by construction (``Replay``).
    ``link`` picks a chained set's link (E3b): 1 is one generation back, 2 two back."""
    if procedure.replay is None:
        return []
    return _link_tasks(_replay_link(procedure, link))


def _replay_link_games(spec: dict[str, Any], replay: Replay, workers: int) -> list[Any]:
    from chess_variant_sandbox.tournament.checkpoint import run_tournament

    _checked_bytes(spec, replay.net, "replay roster's")
    config = _link_config(replay, workers)
    if replay.opening_plies <= 0:
        return [
            record for record in run_tournament(spec, config).records
            if any(_names_a_net(agent) for agent in record.agents)
        ]
    return _play_tasks(spec, config, _link_tasks(replay), replay.opening_plies)


def replay_games(spec: dict[str, Any], procedure: Procedure, workers: int = 0) -> list[Any]:
    """The replay set's games (plan-teacher §9.13.3): the parent's own training schedule
    played again by the core on this ruleset, without its built-in-only block, in schedule
    order. The replay roster's net must be the bytes its name carries (``replay_bytes``).
    A chained set (E3b, §9.17.4) is each link's games in turn, nearest first.

    Without a random opening the ordinary runner plays the whole schedule and the
    built-in-only games are dropped from its records, so a game is the runner's game as the
    parent's was; with one, only ``replay_tasks`` are played. Either way the games are a
    pure function of the schedule's seeds, whatever the thread count.
    """
    if procedure.replay is None:
        return []
    replay_bytes(spec, procedure)
    games: list[Any] = []
    for replay in procedure.replay.links:
        games.extend(_replay_link_games(spec, replay, workers))
    return games


def link_games(
    spec: dict[str, Any], procedure: Procedure, link: int, workers: int = 0
) -> list[Any]:
    """One part of ``self_play``'s records alone (E3b, plan-teacher §9.17.4): ``link`` 0 is
    the new games, 1 the replay set one generation back, 2 two back, and so on. Harvested
    under a procedure's teacher and joined in link order with their games renumbered
    (``joined``), the parts are the whole harvest of ``self_play``, array for array."""
    if link == 0:
        return new_games(spec, procedure, workers)
    return _replay_link_games(spec, _replay_link(procedure, link), workers)


def label_seed(seed: int, ply: int) -> int:
    """The player teacher's stream at ply ``ply`` of the game at schedule seed ``seed``.

    A pure function of the two (Q665), so a label never depends on which worker asked or
    what it asked before. Under 2^62, so ``seat_rng``'s ``seed << 1`` is the same number
    in the core's u64 as in Python's int.
    """
    return _hashed("label-search", seed, ply) >> 2


def player_agent(procedure: Procedure) -> str:
    """The player whose own search labels the positions under ``label_search`` "player"
    (the built-in objective) or "parent" (the parent net, Q677): minimax, or on a round
    game's recipe the round player, ``matrix`` (Q918g)."""
    kind = procedure.searcher
    if procedure.label_search == "parent":
        assert procedure.parent is not None
        return f"{kind}:net={procedure.parent.procedure},nodes={procedure.label_nodes}"
    return f"{kind}:nodes={procedure.label_nodes}"


def parent_bytes(spec: dict[str, Any], procedure: Procedure) -> bytes:
    """The parent net's weights as this process would play them, checked against its name.

    ``b""`` for a procedure without a parent. Otherwise ``net=<parent>`` must resolve on
    this ruleset (registered in this process, ``training.store.register_parent``, or an
    accepted store under ``nets/``) to bytes whose SHA-256 is the one the procedure
    carries; anything else raises, since the net seat and the label would silently be
    another player.
    """
    if procedure.parent is None:
        return b""
    return _checked_bytes(spec, procedure.parent, "parent")


def replay_bytes(spec: dict[str, Any], procedure: Procedure) -> bytes:
    """The replay roster's net (the parent's parent, Q712) as this process would play it,
    checked against the SHA-256 the replay set carries, as ``parent_bytes`` checks the
    parent. ``b""`` for a procedure without a replay set. A chained set's every link's net
    is checked too (E3b); the bytes returned are the nearest link's."""
    if procedure.replay is None:
        return b""
    found = [_checked_bytes(spec, link.net, "replay roster's") for link in procedure.replay.links]
    return found[0]


def _checked_bytes(spec: dict[str, Any], net: Parent, role: str) -> bytes:
    from chess_variant_sandbox.agents.nets import NetError, net_bytes
    from chess_variant_sandbox.engine import Engine

    ruleset_hash = Engine(spec).spec.hash
    blob = net_bytes({"net": net.procedure}, ruleset_hash)
    if hashlib.sha256(blob).hexdigest() != net.weights_sha256:
        raise NetError(
            f"the {role} net={net.procedure} on {ruleset_hash} does not resolve to the "
            f"weights {net.weights_sha256[:12]}... the procedure names "
            f"({'no net' if not blob else 'other weights'}); register it first "
            f"(training.store.register_parent, or train --parent)"
        )
    return blob


# One game as a worker needs it: its index in the harvest, its moves, seat 0's result,
# and its schedule seed (the player teacher's streams are named by it).
Game = tuple[int, list[int], float, int]

# What a worker hands back: per sample, its inputs, side, label, game, ply, result, value.
Harvested = tuple[
    list[list[int]], list[int], list[float], list[int], list[int], list[float], list[float]
]

# Per worker process: the engine and what every game of it shares.
_WORKER: dict[str, Any] = {}


def _setup(spec: dict[str, Any], procedure: Procedure, parent: bytes = b"") -> None:
    from chess_variant_sandbox.agents.nets import ACCEPTED, NetRecord, register_net
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.tournament.play import eval_values, native_spec

    engine = Engine(spec)
    if parent:
        # A spawned worker starts with an empty registry: the parent's bytes come with
        # the task, already checked against the procedure by `harvest`.
        assert procedure.parent is not None
        register_net(NetRecord(procedure.parent.procedure, engine.spec.hash, ACCEPTED, parent))
    _WORKER.clear()
    _WORKER.update(
        engine=engine,
        procedure=procedure,
        zero=zero_net(engine.core.net_inputs()),
        values=eval_values(engine.spec),
        player=(
            native_spec(player_agent(procedure), engine.spec.hash)
            if procedure.searched and procedure.label_search in ("player", "parent")
            else None
        ),
    )


def _harvest_games(games: Sequence[Game]) -> Harvested:
    from chess_variant_sandbox.agents.evaluation import GOAL_PIECES, IMMUNE_PIECES
    from chess_variant_sandbox.engine.engine import _pack

    engine = _WORKER["engine"]
    procedure: Procedure = _WORKER["procedure"]
    rows: list[list[int]] = []
    sides: list[int] = []
    targets: list[float] = []
    indices: list[int] = []
    plies: list[int] = []
    results: list[float] = []
    values: list[float] = []
    player = _WORKER["player"]
    if engine.simultaneous:
        return _harvest_rounds(games)
    for index, moves, first_score, seed in games:
        start = engine.initial_state()
        line = engine.core.net_line(_pack(start), _WORKER["zero"], list(moves))
        state = start
        # The player teacher's positions, asked in one core call per game after the walk
        # (`Core.search_values`): (row in the lists above, packed state, its seed).
        asked: list[tuple[int, Any, int]] = []
        for ply in range(len(moves)):
            if ply >= procedure.skip_plies and (ply - procedure.skip_plies) % procedure.stride == 0:
                side = int(state.side_to_move)
                result = first_score if side == 0 else 1.0 - first_score
                target = result
                value = math.nan
                if player is not None:
                    asked.append((len(sides), _pack(state), label_seed(seed, ply)))
                elif procedure.searched:
                    scores = engine.root_scores(
                        state,
                        _WORKER["values"],
                        procedure.label_nodes,
                        procedure.label_depth,
                        GOAL_PIECES,
                        IMMUNE_PIECES,
                    )
                    if scores:
                        searched = squash(max(scores), procedure.scale)
                        weight = procedure.result_weight
                        target = weight * result + (1.0 - weight) * searched
                        value = searched
                rows.append(list(line[ply][1]))
                sides.append(side)
                targets.append(target)
                indices.append(index)
                plies.append(ply)
                results.append(result)
                values.append(value)
            state = engine.apply(state, moves[ply])
        if asked:
            answers = engine.core.search_values(
                [packed for _row, packed, _seed in asked],
                player,
                _WORKER["values"],
                [seed for _row, _packed, seed in asked],
            )
            weight = procedure.result_weight
            for (row, _packed, _seed), answer in zip(asked, answers, strict=True):
                if answer is None:
                    continue
                searched = squash(answer[0], procedure.scale)
                targets[row] = weight * results[row] + (1.0 - weight) * searched
                values[row] = searched
    return rows, sides, targets, indices, plies, results, values


def _harvest_rounds(games: Sequence[Game]) -> Harvested:
    """``_harvest_games`` on a round game (Q918g): one sample a round's board, side 0.

    Round ``r`` (its board at ply ``2r``) is a sample when ``2r >= skip_plies`` and it is a
    whole number of ``stride`` rounds past the first such round; the board after the last
    round is never one (the game is over there). The label is player 0's: the result, or
    the round player's own value of the round (``Core.round_values``), squashed, one core
    call a game; under ``round_payoff`` "score" (Q918i) the one-round matrix solved on
    expected score, used as it is."""
    from chess_variant_sandbox.engine.engine import _pack

    engine = _WORKER["engine"]
    procedure: Procedure = _WORKER["procedure"]
    player = _WORKER["player"]
    if procedure.searched and player is None:
        raise ValueError(
            "a round game's positions are labelled by the round player's own value "
            "(label_search 'player'); the core has no probe for rounds"
        )
    out: Harvested = ([], [], [], [], [], [], [])
    rows, sides, targets, indices, plies, results, values = out
    first_round = (procedure.skip_plies + 1) // 2
    for index, moves, first_score, _seed in games:
        start = engine.initial_state()
        line = engine.core.net_line(_pack(start), _WORKER["zero"], list(moves))
        state = start
        asked: list[tuple[int, Any]] = []
        for round_index in range(len(moves) // 2):
            if round_index >= first_round and (round_index - first_round) % procedure.stride == 0:
                if player is not None:
                    asked.append((len(sides), _pack(state)))
                rows.append(list(line[round_index][1]))
                sides.append(0)
                targets.append(first_score)
                indices.append(index)
                plies.append(2 * round_index)
                results.append(first_score)
                values.append(math.nan)
            state = engine.apply_round(state, moves[2 * round_index], moves[2 * round_index + 1])
        if asked:
            # Q918i: a score payoff's value is already player 0's expected score (each cell
            # squashed before the matrix is solved); a units value is squashed here.
            scored = procedure.round_payoff == "score"
            answers = engine.core.round_values(
                [packed for _row, packed in asked],
                player,
                _WORKER["values"],
                score_scale=procedure.scale if scored else None,
            )
            weight = procedure.result_weight
            for (row, _packed), answer in zip(asked, answers, strict=True):
                if answer is None:
                    continue
                searched = answer[0] if scored else squash(answer[0], procedure.scale)
                targets[row] = weight * results[row] + (1.0 - weight) * searched
                values[row] = searched
    return out


def _chunks(games: list[Game], workers: int) -> list[list[Game]]:
    size = max(1, math.ceil(len(games) / (workers * 8)))
    return [games[i : i + size] for i in range(0, len(games), size)]


def harvest(
    spec: dict[str, Any],
    records: Sequence[Any],
    procedure: Procedure,
    workers: int = 1,
    report: Report | None = None,
) -> Dataset:
    """Every sampled position of ``records``, as inputs, side to move and label.

    ``workers`` > 1 spreads the games over processes (the probe is one core call per
    position, and a process each is how they run side by side); the chunks come back in
    game order, so the answer does not depend on it.
    """
    games: list[Game] = [
        (index, list(record.moves), record.score_for(0), int(record.seed))
        for index, record in enumerate(records)
    ]
    parent = parent_bytes(spec, procedure)
    # Each chunk is packed into arrays as it arrives (Q713): a union of 3.7 M positions held
    # as Python lists of ints until the end would need several GB; the arrays are the same.
    parts: list[tuple[np.ndarray, ...]] = []
    from chess_variant_sandbox.tournament import checkpoint

    campaign = checkpoint.current()
    if campaign is not None and games:
        # Q732: in sealed pieces of games a restart skips, concatenated in game order as the
        # chunks below are, so the dataset is the same.
        parts = _harvest_pieces(campaign, spec, games, procedure, parent, workers, report)
        from chess_variant_sandbox.engine import Engine

        n_inputs = int(Engine(spec).core.net_inputs())
    elif workers <= 1 or len(games) < 2:
        _setup(spec, procedure, parent)
        parts.append(_packed(_harvest_games(games)))
        n_inputs = int(_WORKER["engine"].core.net_inputs())
    else:
        chunks = _chunks(games, workers)
        positions = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_setup,
            initargs=(spec, procedure, parent),
        ) as pool:
            for done, part in enumerate(pool.map(_harvest_games, chunks), start=1):
                parts.append(_packed(part))
                positions += len(part[1])
                if report is not None and done % max(1, len(chunks) // 10) == 0:
                    report(f"  harvested {done}/{len(chunks)} chunks, {positions} positions")
        from chess_variant_sandbox.engine import Engine

        n_inputs = int(Engine(spec).core.net_inputs())

    lengths, flat, sides, targets, indices, plies, results, values = (
        np.concatenate([part[column] for part in parts]) for column in range(8)
    )
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return Dataset(
        n_inputs=n_inputs,
        offsets=offsets,
        inputs=flat,
        sides=sides,
        targets=targets,
        games=len(games),
        game=indices,
        ply=plies,
        results=results,
        values=values,
        teacher=procedure.teacher,
    )


def _harvest_packed(games: Sequence[Game]) -> tuple[np.ndarray, ...]:
    """A worker's harvest of ``games``, packed: one campaign piece (Q732)."""
    return _packed(_harvest_games(games))


def _harvest_pieces(
    campaign: Any,
    spec: dict[str, Any],
    games: list[Game],
    procedure: Procedure,
    parent: bytes,
    workers: int,
    report: Report | None,
) -> list[tuple[np.ndarray, ...]]:
    """``harvest``'s packed parts through a campaign directory (Q732), in game order.

    The stage is named by the ruleset, the procedure, the games (moves, results, seeds)
    and the parent's bytes; a piece is a run of games, cut from the game count alone.
    """
    import io
    from concurrent.futures import Executor, ThreadPoolExecutor

    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.tournament import checkpoint

    identity = {
        "ruleset_hash": Engine(spec).spec.hash,
        "procedure": procedure.digest,
        "games": checkpoint.digest([[i, moves, score, seed] for i, moves, score, seed in games]),
        "parent_sha256": hashlib.sha256(parent).hexdigest(),
    }
    stage = campaign.stage("harvest", identity, list(range(len(games))))

    def encode(part: tuple[np.ndarray, ...]) -> bytes:
        buffer = io.BytesIO()
        np.savez(buffer, *part)
        return buffer.getvalue()

    def decode(payload: bytes, piece: list[int]) -> tuple[np.ndarray, ...] | None:
        with np.load(io.BytesIO(payload)) as found:
            part = tuple(found[f"arr_{column}"] for column in range(8))
        if not set(part[4].tolist()) <= set(piece):
            return None
        return part

    pool: Executor
    if workers <= 1:
        _setup(spec, procedure, parent)
        pool = ThreadPoolExecutor(max_workers=1)
    else:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_setup,
            initargs=(spec, procedure, parent),
        )
    step = max(1, len(games) // 10)
    last = [0]

    def on_piece(done: int) -> None:
        if report is not None and (done - last[0] >= step or done == len(games)):
            last[0] = done
            report(f"  harvested {done}/{len(games)} games")

    with pool:
        found: list[tuple[np.ndarray, ...]] = stage.run(
            lambda piece: pool.submit(_harvest_packed, [games[index] for index in piece]),
            encode,
            decode,
            on_piece,
        )
    return found


def _packed(part: Harvested) -> tuple[np.ndarray, ...]:
    """One worker's lists as the arrays a ``Dataset`` holds: row lengths, the flat inputs,
    then sides, targets, game, ply, results and values, in their stored dtypes."""
    rows, sides, targets, indices, plies, results, values = part
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    flat = np.fromiter((i for row in rows for i in row), dtype=np.int32, count=int(lengths.sum()))
    return (
        lengths,
        flat,
        np.asarray(sides, dtype=np.int8),
        np.asarray(targets, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(plies, dtype=np.int32),
        np.asarray(results, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )


def harvest_arms(
    spec: dict[str, Any],
    procedures: Sequence[Procedure],
    workers: int = 1,
    report: Report | None = None,
) -> list[Dataset]:
    """One self-play, harvested once per procedure (plan-teacher §9.1: the child, C and K).

    The procedures must agree on everything that decides the games and the samples
    (``procedure.SELF_PLAY_FIELDS``); they differ in the label. The games are played once,
    by the first; every harvest's positions are checked equal to the first's, byte for
    byte, so the arms differ only in their targets.
    """
    from chess_variant_sandbox.agents.training.procedure import same_positions

    if not procedures:
        raise ValueError("harvest_arms needs at least one procedure")
    first = procedures[0]
    for other in procedures[1:]:
        if not same_positions(first, other):
            raise ValueError(
                f"procedure {other.digest} plays other games or samples other positions "
                f"than {first.digest}; arms share one self-play"
            )
    records = self_play(spec, first, workers=workers)
    say = report or (lambda _line: None)
    say(f"  {len(records)} self-play games, harvested {len(procedures)} times")
    out: list[Dataset] = []
    for procedure in procedures:
        data = harvest(spec, records, procedure, workers=workers, report=report)
        if out:
            base = out[0]
            if not (
                np.array_equal(data.offsets, base.offsets)
                and np.array_equal(data.inputs, base.inputs)
                and np.array_equal(data.sides, base.sides)
            ):
                raise AssertionError("two harvests of one self-play disagree on the positions")
        out.append(data)
    return out
