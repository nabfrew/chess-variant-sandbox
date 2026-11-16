"""One set of weights for many games: θ, its descriptors, its trainer and its materialiser.

R-P3b1's probe (`docs/reference/one-evaluator-plan.md` §3, `one-evaluator-probe.md`). A per-game
net's first layer is a free row of 256 weights for each input of one game's `Layout`. Here a row
is the output of one shared function, ``row = g_θ(descriptor of the input, descriptor of the
game)``, so θ is the same for every game and a game's net is *materialised*: g is run once per
input of that game's layout and the result is written as an ordinary ``CVSNET01`` store, which the
core plays exactly as it plays a per-game net (no core change).

**Descriptors read compiled tables only** (`CompiledSpec`: the move tables, the capture matrix,
the derived values, the goal distances, the promotion squares), never a piece's name, and run
once per game. The input layout is `nets.Layout`'s, unchanged, so a stored harvest row needs no
replay: its input ids are moved into a global id space (each game's block after the one before)
and the descriptor table is the concatenation of the games' tables.

**The model.** g is a network from a descriptor of ``D`` numbers through two ReLU layers of
``hidden`` to ``accumulator`` outputs, bounded by ``tanh`` to the float range a per-game ``W0``
row may take (``model.w0_int_limit(MOST_ACTIVE) / 127`` = 1.0). The accumulator's bias is
``tanh(Γ_game · Wb + bb)``, from the game descriptor alone. The dense heads after the
accumulator (two, by side to move, 32 and 32) are shared. Loss, optimiser, scale and clipping
are `model.py`'s.

**Determinism.** One ``numpy.random.Generator`` from the recipe's seed draws the initial weights
and every epoch's order; every sum over positions is taken in sorted order (``np.add.reduceat``).
The same data and the same BLAS thread count give the same θ and the same bytes on one machine.

**Resumable.** Inside a campaign directory the run saves θ, Adam's moments, the step and the
epoch's random state (`tournament/checkpoint.py`), and a restart resumes to the same bytes.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from chess_variant_sandbox.agents.nets import (
    ACTIVATION_MAX,
    BANK_CAP,
    CHECK_CAP,
    CLOCK_BUCKETS,
    HAND_CAP,
    POINTS_CAP,
    PURSE_BUCKETS,
    PURSE_UNITS,
    TURN_CAP,
    Layout,
)
from chess_variant_sandbox.agents.training import model

Report = Callable[[str], None]

DESCRIPTOR_VERSION = 1
SHARED_VERSION = 1
# The most inputs a position may have on for which the float row bound (1.0) is exact in i16.
MOST_ACTIVE = 257
ROW_LIMIT = model.w0_int_limit(MOST_ACTIVE) / ACTIVATION_MAX

SEGMENTS = (
    "piece", "unmoved", "ep", "colour", "hand", "checks", "points", "halfmove", "ply", "turn",
    "rest", "purse_unit", "purse_bucket", "bank", "parts", "other",
)
_KINDS = ("leaps", "slides", "hops", "sweeps")
_MOVE, _CAPTURE = 1, 2
UNREACHABLE = 255


# ------------------------------------------------------------------ descriptors


@dataclass(frozen=True)
class _TypeStats:
    """What one (frame, type) can do on an empty board, from the move tables."""

    by_kind: tuple[float, ...]  # mean landings per origin over n_squares: kind x (move, capture)
    forward: float
    sideways: float
    back: float
    reach: float  # longest landing over the board's span
    per_square_moves: tuple[float, ...]  # landings from each square over the best square's
    per_square_captures: tuple[float, ...]


def _coords(square: int, width: int) -> tuple[int, int]:
    return square % width, square // width


def _type_stats(spec: Any, frame: int, piece_type: int, player: int) -> _TypeStats:
    s, width = spec.n_squares, spec.width
    span = max(1, max(spec.width, spec.height) - 1)
    counts = np.zeros((len(_KINDS), 2))
    moves = np.zeros(s)
    captures = np.zeros(s)
    forward = sideways = back = 0
    longest = 0
    sign = 1 if player == 0 else -1
    for k, kind in enumerate(_KINDS):
        table = getattr(spec, kind, ())
        if not table or frame >= len(table) or piece_type >= len(table[frame]):
            continue
        for origin, entries in enumerate(table[frame][piece_type]):
            ox, oy = _coords(origin, width)
            for entry in entries:
                targets, masks = entry[0], entry[1]
                for target, mask in zip(targets, masks, strict=False):
                    if target < 0 or target >= s:
                        continue
                    if mask & _MOVE:
                        counts[k, 0] += 1
                        moves[origin] += 1
                    if mask & _CAPTURE:
                        counts[k, 1] += 1
                        captures[origin] += 1
                    tx, ty = _coords(target, width)
                    dy = (ty - oy) * sign
                    if dy > 0:
                        forward += 1
                    elif dy < 0:
                        back += 1
                    else:
                        sideways += 1
                    longest = max(longest, abs(tx - ox), abs(ty - oy))
    total = max(1, forward + sideways + back)
    best_m, best_c = max(1.0, moves.max()), max(1.0, captures.max())
    return _TypeStats(
        by_kind=tuple(float(v) for v in (counts / (s * s)).ravel()),
        forward=forward / total,
        sideways=sideways / total,
        back=back / total,
        reach=min(1.0, longest / span),
        per_square_moves=tuple(float(v) for v in moves / best_m),
        per_square_captures=tuple(float(v) for v in captures / best_c),
    )


def _flag(value: Any) -> float:
    return 1.0 if value else 0.0


def game_descriptor(spec: Any) -> np.ndarray:
    """One fixed-length vector of compiled flags and shares for the whole game."""
    reach = getattr(spec, "reach", None)
    extinction = getattr(spec, "extinction", None)
    territory = getattr(spec, "territory", None)
    turn = getattr(spec, "turn", None)
    win = str(getattr(spec, "win", ""))
    stalemate = str(getattr(spec, "stalemate", ""))
    values = [
        _flag(win == "checkmate" or getattr(spec, "checkmate", "") not in ("", "none")),
        _flag(stalemate == "win"), _flag(stalemate == "loss"), _flag(stalemate == "draw"),
        _flag(extinction), _flag(extinction is not None and getattr(extinction, "result", "") == "win"),
        _flag(reach), _flag(reach is not None and getattr(reach, "count", 1) > 1),
        _flag(territory), _flag(getattr(spec, "scoring", None)), _flag(getattr(spec, "check_limit", 0)),
        _flag(getattr(spec, "bare_royal", "")),
        _flag(getattr(spec, "must_capture", False)), _flag(getattr(spec, "forbid_check", False)),
        _flag(getattr(spec, "drops", None)),
        _flag(getattr(spec, "drops", None) is not None and getattr(spec.drops, "from_captures", False)),
        _flag(getattr(spec, "purse", None)), _flag(getattr(spec, "income", None)),
        _flag(turn is not None and getattr(turn, "moves", 1) > 1), _flag(getattr(spec, "parts", None)),
        _flag(getattr(spec, "duck", None)), _flag(getattr(spec, "npcs", None)),
        _flag(getattr(spec, "portals", None)), _flag(getattr(spec, "push", None)),
        _flag(getattr(spec, "rest", ())), _flag(getattr(spec, "explosion", None)),
        _flag(getattr(spec, "strike", None)), _flag(getattr(spec, "wrap", None)),
        _flag(getattr(spec, "simultaneous", None)), _flag(getattr(spec, "castling", None)),
        _flag(getattr(spec, "en_passant", False)), _flag(getattr(spec, "promotions", ())),
        _flag(getattr(spec, "has_capture_matrix", False)), _flag(getattr(spec, "multi_royal", False)),
        _flag(getattr(spec, "royal_types", ())), _flag(getattr(spec, "obstacles", ())),
        min(1.0, spec.n_squares / 100.0), min(1.0, spec.width / 12.0), min(1.0, spec.height / 12.0),
        min(1.0, spec.n_types / 16.0), _flag(spec.n_facings > 1),
        min(1.0, spec.max_plies / 1000.0), min(1.0, spec.no_progress_draw / 200.0),
    ]
    # What a strike does instead of removal, and where the striker ends: the rule's own
    # parameters, so games that differ only there are told apart.
    strike = getattr(spec, "strike", None)
    effect = str(getattr(strike, "effect", "")) if strike is not None else ""
    position = str(getattr(strike, "position", "")) if strike is not None else ""
    values += [_flag(effect == e) for e in ("capture", "promote", "demote")]
    values.append(_flag(strike is not None and effect not in ("capture", "promote", "demote")))
    values += [_flag(position == q) for q in ("home", "swap", "stay")]
    values.append(_flag(strike is not None and position not in ("home", "swap", "stay")))
    return np.asarray(values, dtype=np.float32)


GAME_WIDTH = 51


def _square_features(spec: Any, square: int, owner: int) -> list[float]:
    width, height = spec.width, spec.height
    x, y = _coords(square, width)
    if owner == 1:
        y = height - 1 - y
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    radius = max(1.0, max(cx, cy))
    centre = 1.0 - max(abs(x - cx), abs(y - cy)) / radius
    edge = _flag(x in (0, width - 1) or y in (0, height - 1))
    reach = getattr(spec, "reach", None)
    goal = 0.0
    if reach is not None and owner in (0, 1):
        goal = _flag(square in reach.squares[owner])
    promo = 0.0
    for rule in getattr(spec, "promotions", ()):
        if owner in (0, 1) and square in rule.squares[owner]:
            promo = 1.0
    return [x / max(1, width - 1), y / max(1, height - 1), centre, edge, goal, promo]


SQUARE_WIDTH = 6


def _owner_features(owner: int, facing: int, n_facings: int) -> list[float]:
    return [_flag(owner == 0), _flag(owner == 1), _flag(owner >= 2), facing / max(1, n_facings)]


OWNER_WIDTH = 4


class _GameTables:
    """Per (owner, facing, type) features of one game, computed once."""

    def __init__(self, spec: Any) -> None:
        from chess_variant_sandbox.agents.evaluation import Evaluator
        from chess_variant_sandbox.engine.compile import frame_index

        self.spec = spec
        t, nf = spec.n_types, spec.n_facings
        self.values = tuple(float(v) for v in Evaluator.for_spec(spec).values)
        top = max([abs(v) for v in self.values] + [1e-9])
        royal = set(spec.royal_types)
        extinction = getattr(spec, "extinction", None)
        reach = getattr(spec, "reach", None)
        drops = getattr(spec, "drops", None)
        allowed = spec.capture_allowed
        counts = np.zeros((2, t))
        for byte in spec.initial_board:
            if byte:
                owner, rest = divmod(byte - 1, nf * t)
                if owner < 2:
                    counts[owner, rest % t] += 1
        army = np.maximum(1.0, counts.sum(axis=1))
        worth = np.array([[counts[p, k] * max(0.0, self.values[k]) for k in range(t)] for p in (0, 1)])
        worth_total = np.maximum(1e-9, worth.sum(axis=1))
        promoted_from = {rule.piece_type for rule in getattr(spec, "promotions", ())}
        self.type_rows: dict[tuple[int, int, int], list[float]] = {}
        self.stats: dict[tuple[int, int, int], _TypeStats] = {}
        for owner in range(spec.n_owners):
            player = min(owner, 1)
            for facing in range(nf):
                frame = frame_index(player, facing, nf)
                for k in range(t):
                    stats = _type_stats(spec, frame, k, player)
                    self.stats[(owner, facing, k)] = stats
                    droppable = drops is not None and (
                        bool(getattr(drops, "from_captures", False))
                        or (k < len(getattr(drops, "stock", ())) and drops.stock[k] > 0)
                        or (k < len(getattr(drops, "unlimited", ())) and drops.unlimited[k])
                    )
                    row = list(stats.by_kind) + [
                        stats.forward, stats.sideways, stats.back, stats.reach,
                        self.values[k] / top,
                        float(worth[player, k] / worth_total[player]),
                        float(counts[player, k] / army[player]),
                        _flag(k in royal),
                        _flag(extinction is not None and (extinction.any_piece or k in extinction.piece_types)),
                        _flag(reach is not None and k in reach.piece_types),
                        float(np.mean([allowed[k][v] for v in range(t)])) if allowed else 1.0,
                        float(np.mean([allowed[a][k] for a in range(t)])) if allowed else 1.0,
                        _flag(k in promoted_from),
                        _flag(droppable),
                    ]
                    self.type_rows[(owner, facing, k)] = row

    def piece_square(self, owner: int, facing: int, k: int, square: int) -> list[float]:
        spec = self.spec
        stats = self.stats[(owner, facing, k)]
        player = min(owner, 1)
        dist, unreachable = 0.0, 0.0
        table = getattr(spec, "goal_distance", ())
        if table:
            d = table[player][k][square]
            if d >= UNREACHABLE:
                unreachable = 1.0
            else:
                dist = d / max(1, spec.goal_diameter)
        promotes = 0.0
        for rule in getattr(spec, "promotions", ()):
            if rule.piece_type == k and square in rule.squares[player]:
                promotes = 1.0
        immobile = getattr(spec, "immobile_squares", ())
        stuck = _flag(immobile and square in immobile[player][k])
        return [stats.per_square_moves[square], stats.per_square_captures[square], dist,
                unreachable, promotes, stuck]


TYPE_WIDTH = 8 + 4 + 3 + 7
PIECE_SQUARE_WIDTH = 6
INPUT_WIDTH = len(SEGMENTS) + TYPE_WIDTH + PIECE_SQUARE_WIDTH + SQUARE_WIDTH + OWNER_WIDTH + 1
DESCRIPTOR_WIDTH = INPUT_WIDTH + GAME_WIDTH


def input_descriptors(spec: Any) -> np.ndarray:
    """``(n_inputs, DESCRIPTOR_WIDTH)``: one descriptor per input of the game's `Layout`."""
    layout = Layout.of(spec)
    tables = _GameTables(spec)
    gamma = game_descriptor(spec)
    assert len(gamma) == GAME_WIDTH
    t, nf, s = spec.n_types, spec.n_facings, spec.n_squares
    zero_type = [0.0] * TYPE_WIDTH
    zero_ps = [0.0] * PIECE_SQUARE_WIDTH
    zero_sq = [0.0] * SQUARE_WIDTH
    rows: list[list[float]] = []

    def put(segment: str, type_row: list[float], ps: list[float], sq: list[float],
            owner: list[float], level: float) -> None:
        tag = [0.0] * len(SEGMENTS)
        tag[SEGMENTS.index(segment)] = 1.0
        rows.append(tag + type_row + ps + sq + owner + [level])

    no_owner = [0.0] * OWNER_WIDTH
    for i in range(layout.n_inputs):
        if i < layout.unmoved:
            square, v = divmod(i, layout.n_values)
            owner, rest = divmod(v, nf * t)
            facing, k = divmod(rest, t)
            put("piece", tables.type_rows[(owner, facing, k)],
                tables.piece_square(owner, facing, k, square),
                _square_features(spec, square, owner), _owner_features(owner, facing, nf), 1.0)
        elif i < layout.ep:
            square = i - layout.unmoved
            byte = spec.initial_board[square] if square < len(spec.initial_board) else 0
            if byte:
                owner, rest = divmod(byte - 1, nf * t)
                facing, k = divmod(rest, t)
                put("unmoved", tables.type_rows[(owner, facing, k)], zero_ps,
                    _square_features(spec, square, owner), _owner_features(owner, facing, nf), 1.0)
            else:
                put("unmoved", zero_type, zero_ps, _square_features(spec, square, 0), no_owner, 1.0)
        elif i < layout.colour:
            square = i - layout.ep
            put("ep", zero_type, zero_ps, _square_features(spec, square, 0), no_owner, 1.0)
        elif i < layout.hands:
            square, c = divmod(i - layout.colour, 2)
            put("colour", zero_type, zero_ps, _square_features(spec, square, c),
                _owner_features(c, 0, nf), 1.0)
        elif i < layout.checks:
            rel = i - layout.hands
            player, rest = divmod(rel, t * HAND_CAP)
            k, count = divmod(rest, HAND_CAP)
            put("hand", tables.type_rows[(player, 0, k)], zero_ps, zero_sq,
                _owner_features(player, 0, nf), (count + 1) / HAND_CAP)
        elif i < layout.points:
            player, count = divmod(i - layout.checks, CHECK_CAP)
            limit = max(1, min(CHECK_CAP, int(getattr(spec, "check_limit", 0) or CHECK_CAP)))
            put("checks", zero_type, zero_ps, zero_sq, _owner_features(player, 0, nf),
                min(1.0, (count + 1) / limit))
        elif i < layout.halfmove:
            player, count = divmod(i - layout.points, POINTS_CAP)
            put("points", zero_type, zero_ps, zero_sq, _owner_features(player, 0, nf),
                (count + 1) / POINTS_CAP)
        elif i < layout.ply:
            put("halfmove", zero_type, zero_ps, zero_sq, no_owner,
                (i - layout.halfmove + 1) / CLOCK_BUCKETS)
        elif i < layout.turn:
            put("ply", zero_type, zero_ps, zero_sq, no_owner, (i - layout.ply + 1) / CLOCK_BUCKETS)
        elif i < layout.initiative:
            put("turn", zero_type, zero_ps, zero_sq, no_owner, (i - layout.turn + 1) / TURN_CAP)
        elif layout.has_rest and layout.rest <= i < layout.void:
            rel = i - layout.rest
            layer, square = divmod(rel, s)
            left, player = divmod(layer, 2)
            put("rest", zero_type, zero_ps, _square_features(spec, square, player),
                _owner_features(player, 0, nf), (left + 1) / max(1, layout.rest_turns))
        elif layout.has_purse and layout.purse <= i < layout.bank:
            player, k = divmod(i - layout.purse, PURSE_UNITS + PURSE_BUCKETS)
            if k < PURSE_UNITS:
                put("purse_unit", zero_type, zero_ps, zero_sq, _owner_features(player, 0, nf),
                    (k + 1) / PURSE_UNITS)
            else:
                put("purse_bucket", zero_type, zero_ps, zero_sq, _owner_features(player, 0, nf),
                    (k - PURSE_UNITS + 1) / PURSE_BUCKETS)
        elif layout.has_bank and layout.bank <= i < layout.parts:
            player, k = divmod(i - layout.bank, BANK_CAP)
            put("bank", zero_type, zero_ps, zero_sq, _owner_features(player, 0, nf), (k + 1) / BANK_CAP)
        elif i >= layout.parts:
            part = i - layout.parts
            put("parts", zero_type, zero_ps, zero_sq, no_owner, (part + 1) / max(1, layout.n_parts))
        else:
            put("other", zero_type, zero_ps, zero_sq, no_owner, 0.0)
    body = np.asarray(rows, dtype=np.float32)
    assert body.shape == (layout.n_inputs, INPUT_WIDTH), body.shape
    return np.hstack([body, np.repeat(gamma[None, :], layout.n_inputs, axis=0)])


# ------------------------------------------------------------------ the data


@dataclass
class ProbeData:
    """Positions of several games in one global input-id space.

    Game ``g``'s inputs are ids ``starts[g] .. starts[g] + n_inputs[g]``; ``descriptors`` has
    one row per global id, ``gammas`` one row per game; ``game`` is each position's game.
    """

    hashes: list[str]
    n_inputs: list[int]
    starts: np.ndarray
    descriptors: np.ndarray
    gammas: np.ndarray
    offsets: np.ndarray
    inputs: np.ndarray
    sides: np.ndarray
    targets: np.ndarray
    game: np.ndarray

    @property
    def positions(self) -> int:
        return len(self.sides)

    @property
    def total_inputs(self) -> int:
        return int(self.descriptors.shape[0])

    @property
    def digest(self) -> str:
        sha = hashlib.sha256()
        sha.update(",".join(self.hashes).encode())
        for array in (self.starts, self.offsets):
            sha.update(array.astype("<i8").tobytes())
        sha.update(self.inputs.astype("<i4").tobytes())
        sha.update(self.sides.astype("<i1").tobytes())
        sha.update(self.targets.astype("<f4").tobytes())
        sha.update(self.game.astype("<i1").tobytes())
        sha.update(self.descriptors.astype("<f4").tobytes())
        sha.update(self.gammas.astype("<f4").tobytes())
        return sha.hexdigest()

    def save(self, path: Any) -> None:
        with open(path, "wb") as handle:
            np.savez(handle, hashes=np.array(self.hashes), n_inputs=np.array(self.n_inputs),
                     starts=self.starts, descriptors=self.descriptors, gammas=self.gammas,
                     offsets=self.offsets, inputs=self.inputs, sides=self.sides,
                     targets=self.targets, game=self.game)


def load_probe(path: Any) -> ProbeData:
    with np.load(path) as data:
        return ProbeData(
            hashes=[str(h) for h in data["hashes"]], n_inputs=[int(n) for n in data["n_inputs"]],
            starts=data["starts"].astype(np.int64), descriptors=data["descriptors"].astype(np.float32),
            gammas=data["gammas"].astype(np.float32), offsets=data["offsets"].astype(np.int64),
            inputs=data["inputs"].astype(np.int32), sides=data["sides"].astype(np.int8),
            targets=data["targets"].astype(np.float32), game=data["game"].astype(np.int8),
        )


def head_slice(data: ProbeData, count: int) -> ProbeData:
    """Every ``positions // count``-th position, for timing runs."""
    step = max(1, data.positions // max(1, count))
    rows = np.arange(0, data.positions, step)[:count]
    lengths = data.offsets[rows + 1] - data.offsets[rows]
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    inputs = np.concatenate([data.inputs[data.offsets[r]:data.offsets[r + 1]] for r in rows])
    return ProbeData(data.hashes, data.n_inputs, data.starts, data.descriptors, data.gammas,
                     offsets, inputs, data.sides[rows], data.targets[rows], data.game[rows])


# ------------------------------------------------------------------ θ and its trainer


@dataclass(frozen=True)
class ProbeRecipe:
    """θ's widths and schedule. In the store's digest, field by field."""

    hidden: int = 256
    accumulator: int = 256
    hidden1: int = 32
    hidden2: int = 32
    epochs: int = 8
    batch: int = 4096
    learning_rate: float = 0.001
    scale: float = 1.0
    seed: int = 0

    @property
    def digest(self) -> str:
        blob = repr(sorted(asdict(self).items())) + f"|d{DESCRIPTOR_VERSION}|s{SHARED_VERSION}"
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


HEAD_KEYS = ("w1", "b1", "w2", "b2", "w3", "b3")


@dataclass
class Theta:
    params: dict[str, np.ndarray]
    history: list[float] = field(default_factory=list)


def initial(recipe: ProbeRecipe, width: int = DESCRIPTOR_WIDTH, gamma: int = GAME_WIDTH) -> Theta:
    rng = np.random.default_rng(recipe.seed)
    h, a, h1, h2 = recipe.hidden, recipe.accumulator, recipe.hidden1, recipe.hidden2

    def uniform(shape: tuple[int, ...], limit: float) -> np.ndarray:
        return rng.uniform(-limit, limit, size=shape).astype(np.float32)

    p: dict[str, np.ndarray] = {
        "g1": uniform((width, h), np.sqrt(6.0 / width)), "gb1": np.zeros(h, np.float32),
        "g2": uniform((h, h), np.sqrt(6.0 / h)), "gb2": np.zeros(h, np.float32),
        # Small, so that the first rows are about the per-game trainer's initial spread.
        "g3": uniform((h, a), 0.1 / np.sqrt(h)), "gb3": np.zeros(a, np.float32),
        "bw": uniform((gamma, a), 0.01), "bb": np.full(a, np.arctanh(0.5), np.float32),
    }
    for side in (0, 1):
        p[f"w1_{side}"] = uniform((h1, a), 1.0 / np.sqrt(a))
        p[f"b1_{side}"] = np.full(h1, 0.5, np.float32)
        p[f"w2_{side}"] = uniform((h2, h1), 1.0 / np.sqrt(h1))
        p[f"b2_{side}"] = np.full(h2, 0.5, np.float32)
        p[f"w3_{side}"] = uniform((h2,), 1.0 / np.sqrt(h2))
        p[f"b3_{side}"] = np.zeros(1, np.float32)
    theta = Theta(p)
    _clip(theta)
    return theta


def _clip(theta: Theta) -> None:
    p = theta.params
    for side in (0, 1):
        for key in ("w1", "w2"):
            np.clip(p[f"{key}_{side}"], -model.DENSE_LIMIT, model.DENSE_LIMIT, out=p[f"{key}_{side}"])
        np.clip(p[f"w3_{side}"], -model.OUTPUT_LIMIT, model.OUTPUT_LIMIT, out=p[f"w3_{side}"])


def _rows(theta: Theta, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """g on descriptor rows ``x``: the W0 rows, and what the backward pass needs."""
    p = theta.params
    z1 = x @ p["g1"] + p["gb1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ p["g2"] + p["gb2"]
    a2 = np.maximum(z2, 0.0)
    out = np.tanh(a2 @ p["g3"] + p["gb3"]) * np.float32(ROW_LIMIT)
    return out.astype(np.float32), (x, a1, a2, out)


def _biases(theta: Theta, gammas: np.ndarray) -> np.ndarray:
    p = theta.params
    return np.tanh(gammas @ p["bw"] + p["bb"]).astype(np.float32)


def _step(theta: Theta, data: ProbeData, rows: np.ndarray, scale: float
          ) -> tuple[float, dict[str, np.ndarray]]:
    p = theta.params
    sentinel = data.total_inputs
    index = model._padded(data.offsets, data.inputs, rows, sentinel)
    valid = index < sentinel
    unique, inverse = np.unique(index[valid], return_inverse=True)
    local = np.full(index.shape, len(unique), dtype=np.int64)
    local[valid] = inverse
    w0, cache = _rows(theta, data.descriptors[unique])
    padded = np.vstack([w0, np.zeros((1, w0.shape[1]), np.float32)])
    games = data.game[rows].astype(np.int64)
    b0 = _biases(theta, data.gammas)
    acc = padded[local].sum(axis=1, dtype=np.float32) + b0[games]
    x0 = np.clip(acc, 0.0, 1.0)
    sides = data.sides[rows]
    targets = data.targets[rows]
    y = np.zeros(len(rows), dtype=np.float32)
    heads = {}
    for side in (0, 1):
        mask = np.flatnonzero(sides == side)
        if not len(mask):
            continue
        z1 = x0[mask] @ p[f"w1_{side}"].T + p[f"b1_{side}"]
        x1 = np.clip(z1, 0.0, 1.0)
        z2 = x1 @ p[f"w2_{side}"].T + p[f"b2_{side}"]
        x2 = np.clip(z2, 0.0, 1.0)
        y[mask] = x2 @ p[f"w3_{side}"] + p[f"b3_{side}"][0]
        heads[side] = (mask, z1, x1, z2, x2)
    prob = model._sigmoid(y / scale)
    diff = prob - targets
    loss = float(np.mean(diff.astype(np.float64) ** 2))
    dy = (2.0 / len(rows)) * diff * prob * (1.0 - prob) / scale
    grads: dict[str, np.ndarray] = {}
    dx0 = np.zeros_like(x0)
    for side in (0, 1):
        if side not in heads:
            for key in HEAD_KEYS:
                grads[f"{key}_{side}"] = np.zeros_like(p[f"{key}_{side}"])
            continue
        mask, z1, x1, z2, x2 = heads[side]
        d = dy[mask]
        grads[f"w3_{side}"] = x2.T @ d
        grads[f"b3_{side}"] = np.array([d.sum()], dtype=d.dtype)
        dz2 = np.outer(d, p[f"w3_{side}"]) * ((z2 > 0.0) & (z2 < 1.0))
        grads[f"w2_{side}"] = dz2.T @ x1
        grads[f"b2_{side}"] = dz2.sum(axis=0)
        dz1 = (dz2 @ p[f"w2_{side}"]) * ((z1 > 0.0) & (z1 < 1.0))
        grads[f"w1_{side}"] = dz1.T @ x0[mask]
        grads[f"b1_{side}"] = dz1.sum(axis=0)
        dx0[mask] = dz1 @ p[f"w1_{side}"]
    dacc = (dx0 * ((acc > 0.0) & (acc < 1.0))).astype(np.float32)
    # The bias: per game, then through tanh and the game layer.
    db0 = np.zeros_like(b0)
    order = np.argsort(games, kind="stable")
    present, starts = np.unique(games[order], return_index=True)
    db0[present] = np.add.reduceat(dacc[order], starts, axis=0)
    dpre = db0 * (1.0 - b0 * b0)
    grads["bw"] = data.gammas.T @ dpre
    grads["bb"] = dpre.sum(axis=0)
    # The rows: each unique input's gradient summed in sorted order, then back through g.
    flat = local[valid]
    positions = np.broadcast_to(np.arange(len(rows))[:, None], index.shape)[valid]
    order = np.argsort(flat, kind="stable")
    flat, positions = flat[order], positions[order]
    _, starts = np.unique(flat, return_index=True)
    drow = np.add.reduceat(dacc[positions], starts, axis=0)
    x, a1, a2, out = cache
    dpre3 = drow * (np.float32(ROW_LIMIT) - out * out / np.float32(ROW_LIMIT))
    grads["g3"] = a2.T @ dpre3
    grads["gb3"] = dpre3.sum(axis=0)
    da2 = (dpre3 @ p["g3"].T) * (a2 > 0.0)
    grads["g2"] = a1.T @ da2
    grads["gb2"] = da2.sum(axis=0)
    da1 = (da2 @ p["g2"].T) * (a1 > 0.0)
    grads["g1"] = x.T @ da1
    grads["gb1"] = da1.sum(axis=0)
    return loss, {k: v.astype(np.float32) for k, v in grads.items()}


class OutOfTime(Exception):
    """The slice's minutes ran out; the newest state is saved."""


def train(data: ProbeData, recipe: ProbeRecipe, report: Report | None = None,
          deadline: float | None = None, saves: Any = None) -> Theta:
    """Fit θ to ``data``. With ``saves`` (a campaign's), resumable; past ``deadline``
    (``time.monotonic``), save and raise ``OutOfTime``."""
    theta = initial(recipe)
    rng = np.random.default_rng(recipe.seed + 1)
    first = {k: np.zeros_like(v) for k, v in theta.params.items()}
    second = {k: np.zeros_like(v) for k, v in theta.params.items()}
    step = 0
    start_epoch, start_begin, start_total = 0, 0, 0.0
    start_rng: dict[str, Any] | None = None
    found = saves.latest() if saves is not None else None
    if found is not None:
        arrays, meta = found
        for key in theta.params:
            theta.params[key] = arrays[f"p_{key}"].copy()
            first[key] = arrays[f"m_{key}"].copy()
            second[key] = arrays[f"v_{key}"].copy()
        step = int(meta["step"])
        theta.history = [float(v) for v in meta["history"]]
        start_epoch, start_begin = int(meta["epoch"]), int(meta["begin"])
        start_total, start_rng = float(meta["total"]), dict(meta["rng"])
        if report is not None:
            report(f"  resumed at epoch {start_epoch + 1}, position {start_begin}, step {step}")

    def save(epoch: int, begin: int, total: float, epoch_rng: dict[str, Any]) -> None:
        arrays: dict[str, np.ndarray] = {}
        for key in theta.params:
            arrays[f"p_{key}"] = theta.params[key]
            arrays[f"m_{key}"] = first[key]
            arrays[f"v_{key}"] = second[key]
        saves.save(arrays, {"step": step, "epoch": epoch, "begin": begin, "total": total,
                            "history": list(theta.history), "rng": epoch_rng,
                            "where": f"epoch {epoch + 1}, position {begin}, step {step}"})

    for epoch in range(start_epoch, recipe.epochs):
        resumed = epoch == start_epoch and start_rng is not None
        if resumed and start_rng is not None:
            rng.bit_generator.state = start_rng
        epoch_rng = dict(rng.bit_generator.state)
        order = rng.permutation(data.positions)
        total = start_total if resumed else 0.0
        began = time.monotonic()
        for begin in range(start_begin if resumed else 0, data.positions, recipe.batch):
            rows = order[begin:begin + recipe.batch]
            loss, grads = _step(theta, data, rows, recipe.scale)
            total += loss * len(rows)
            step += 1
            c1 = 1.0 - model.ADAM_BETA1 ** step
            c2 = 1.0 - model.ADAM_BETA2 ** step
            rate = recipe.learning_rate * np.sqrt(c2) / c1
            for key, grad in grads.items():
                m, v = first[key], second[key]
                m *= model.ADAM_BETA1
                m += (1.0 - model.ADAM_BETA1) * grad
                v *= model.ADAM_BETA2
                v += (1.0 - model.ADAM_BETA2) * grad * grad
                theta.params[key] -= (rate * m / (np.sqrt(v) + model.ADAM_EPSILON)).astype(np.float32)
            _clip(theta)
            done = begin + len(rows)
            if saves is not None and saves.due():
                save(epoch, done, total, epoch_rng)
            if deadline is not None and time.monotonic() > deadline and done < data.positions:
                if saves is not None:
                    save(epoch, done, total, epoch_rng)
                raise OutOfTime(f"epoch {epoch + 1}, position {done}, step {step}")
            if report is not None and step % 500 == 0:
                rate_s = (done - (start_begin if resumed else 0)) / max(1e-9, time.monotonic() - began)
                report(f"  epoch {epoch + 1} position {done}/{data.positions} "
                       f"loss {total / max(1, done):.6f} ({rate_s:.0f} positions/s)")
        theta.history.append(round(total / data.positions, 8))
        start_begin = 0
        if report is not None:
            report(f"  epoch {epoch + 1}/{recipe.epochs}: loss {theta.history[-1]:.6f}")
        if saves is not None:
            save(epoch + 1, 0, 0.0, dict(rng.bit_generator.state))
    return theta


def theta_digest(theta: Theta, recipe: ProbeRecipe, data_digest: str) -> str:
    """The name a materialised store is filed under: θ's bytes, the recipe, the data."""
    sha = hashlib.sha256()
    for key in sorted(theta.params):
        sha.update(key.encode())
        sha.update(theta.params[key].astype("<f4").tobytes())
    sha.update(recipe.digest.encode())
    sha.update(data_digest.encode())
    sha.update(f"d{DESCRIPTOR_VERSION}".encode())
    return sha.hexdigest()[:12]


def materialise(theta: Theta, spec: Any) -> bytes:
    """θ written as one game's ordinary per-game net (``CVSNET01``): a pure function of θ and
    the compiled spec."""
    descriptors = input_descriptors(spec)
    w0, _ = _rows(theta, descriptors)
    b0 = _biases(theta, game_descriptor(spec)[None, :])[0]
    params: dict[str, np.ndarray] = {"w0": w0, "b0": b0}
    for side in (0, 1):
        for key in HEAD_KEYS:
            params[f"{key}_{side}"] = theta.params[f"{key}_{side}"]
    net = model.FloatNet(params=params, w0_limit=model.w0_int_limit(MOST_ACTIVE))
    return model.quantize(net)


def theta_arrays(theta: Theta) -> dict[str, np.ndarray]:
    return {f"p_{key}": value for key, value in theta.params.items()}


def theta_from_arrays(arrays: dict[str, np.ndarray], history: Sequence[float] = ()) -> Theta:
    return Theta({key[2:]: np.asarray(v, dtype=np.float32) for key, v in arrays.items()
                  if key.startswith("p_")}, list(history))
