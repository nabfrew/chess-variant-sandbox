"""The float net the trainer fits, and its quantization to the bytes ``net.rs`` plays.

**The float net is the integer net with the scales divided out** (``learned-evaluator.md``
§3). Every activation lives in ``[0, 1]`` where the core's lives in ``[0, 127]``, so:

| layer | float | integer | scale |
|---|---|---|---|
| accumulator | ``b0 + sum W0 rows``, clipped to [0, 1] | ``i16``, clipped to [0, 127] | ``W0, b0 x 127`` |
| hidden 1, 2 | ``clip(b + W x, 0, 1)`` | ``clip((b + W x) >> 6, 0, 127)`` | ``W x 64``, ``b x 64 x 127`` |
| output | ``y = b3 + W3 x`` in built-in units | ``b3 + W3 x`` in 1/1024 units | ``W3 x 1024 / 127``, ``b3 x 1024`` |

Each weight is kept, during training, inside the range its integer form can hold (``i8``
at +-127 for the dense layers; ``W0`` inside what keeps the ``i16`` accumulator from
wrapping on any position with as many inputs as the busiest one in the data), so
quantizing is rounding and nothing else.

**The loss** is the squared error between ``sigmoid(y / scale)`` and the label, the
side to move's expectation, which is the usual NNUE loss and keeps a won position's
huge probe score from dominating. Side to move picks the head, as in the core.

**Determinism.** One ``numpy.random.Generator`` from ``init_seed`` draws the initial
weights and every epoch's order; the sparse gradient is summed in sorted input order
(``np.add.reduceat``), never by scatter in arrival order. The same data therefore gives
the same bytes on one machine (``tests/test_training.py``); another BLAS or CPU may sum
a matrix product in another order, which is why the store, not the recipe, is the player.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from chess_variant_sandbox.agents.nets import (
    ACTIVATION_MAX,
    OUTPUT_SCALE,
    WEIGHT_SHIFT,
    encode,
)
from chess_variant_sandbox.agents.training.data import Dataset
from chess_variant_sandbox.agents.training.procedure import Procedure

Report = Callable[[str], None]

I16_MAX = 32_767
I8_MAX = 127
I32_LIMIT = (1 << 31) - 1
HIDDEN_SCALE = float(1 << WEIGHT_SHIFT)
# The largest a dense weight or the output weight may be in float, so its i8 fits.
DENSE_LIMIT = I8_MAX / HIDDEN_SCALE
OUTPUT_LIMIT = I8_MAX * ACTIVATION_MAX / OUTPUT_SCALE
BIAS0_LIMIT = 1.0

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

HEAD_KEYS = ("w1", "b1", "w2", "b2", "w3", "b3")


def w0_int_limit(most_active: int) -> int:
    """The largest ``|W0|`` in integer units that keeps every accumulator inside ``i16``.

    ``|b0| <= 127`` and at most ``most_active`` rows are summed, so the sum stays below
    32,767 in magnitude whatever the signs.
    """
    return max(1, min(I8_MAX, (I16_MAX - I8_MAX) // max(1, most_active)))


@dataclass
class FloatNet:
    """The trainer's parameters: ``w0`` (inputs x A), ``b0`` (A), and two heads."""

    params: dict[str, np.ndarray]
    w0_limit: int
    history: list[float] = field(default_factory=list)

    @property
    def accumulator(self) -> int:
        return int(self.params["b0"].shape[0])

    @property
    def hidden1(self) -> int:
        return int(self.params["b1_0"].shape[0])

    @property
    def hidden2(self) -> int:
        return int(self.params["b2_0"].shape[0])

    @property
    def n_inputs(self) -> int:
        return int(self.params["w0"].shape[0])


def initial(n_inputs: int, procedure: Procedure, most_active: int) -> FloatNet:
    """Uniform fan-in weights from ``init_seed``, biases that start the clips half open."""
    rng = np.random.default_rng(procedure.init_seed)
    a, h1, h2 = procedure.accumulator, procedure.hidden1, procedure.hidden2
    spread = 0.5 / np.sqrt(max(1, most_active))

    def uniform(shape: tuple[int, ...], limit: float) -> np.ndarray:
        return rng.uniform(-limit, limit, size=shape).astype(np.float32)

    params: dict[str, np.ndarray] = {
        "w0": uniform((n_inputs, a), spread),
        "b0": np.full(a, 0.5, dtype=np.float32),
    }
    for side in (0, 1):
        params[f"w1_{side}"] = uniform((h1, a), 1.0 / np.sqrt(a))
        params[f"b1_{side}"] = np.full(h1, 0.5, dtype=np.float32)
        params[f"w2_{side}"] = uniform((h2, h1), 1.0 / np.sqrt(h1))
        params[f"b2_{side}"] = np.full(h2, 0.5, dtype=np.float32)
        params[f"w3_{side}"] = uniform((h2,), 1.0 / np.sqrt(h2))
        params[f"b3_{side}"] = np.zeros(1, dtype=np.float32)
    net = FloatNet(params=params, w0_limit=w0_int_limit(most_active))
    _clip(net)
    return net


def _clip(net: FloatNet) -> None:
    """Hold every weight inside what its integer form can store."""
    p = net.params
    w0 = net.w0_limit / ACTIVATION_MAX
    np.clip(p["w0"], -w0, w0, out=p["w0"])
    np.clip(p["b0"], -BIAS0_LIMIT, BIAS0_LIMIT, out=p["b0"])
    for side in (0, 1):
        for key in ("w1", "w2"):
            np.clip(p[f"{key}_{side}"], -DENSE_LIMIT, DENSE_LIMIT, out=p[f"{key}_{side}"])
        np.clip(p[f"w3_{side}"], -OUTPUT_LIMIT, OUTPUT_LIMIT, out=p[f"w3_{side}"])


def _padded(offsets: np.ndarray, inputs: np.ndarray, rows: np.ndarray, sentinel: int) -> np.ndarray:
    """``rows``' inputs as a rectangle, short rows filled with ``sentinel`` (a zero row)."""
    starts = offsets[rows]
    lengths = offsets[rows + 1] - starts
    width = int(lengths.max()) if len(rows) else 0
    out = np.full((len(rows), max(1, width)), sentinel, dtype=np.int64)
    if width:
        column = np.arange(width)
        mask = column[None, :] < lengths[:, None]
        out[mask] = inputs[(starts[:, None] + column[None, :])[mask]]
    return out


@dataclass
class _Pass:
    """One forward pass, kept for the backward one."""

    acc: np.ndarray
    x0: np.ndarray
    heads: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    y: np.ndarray


def _forward(net: FloatNet, index: np.ndarray, sides: np.ndarray) -> _Pass:
    p = net.params
    padding = index >= net.n_inputs
    rows = p["w0"][np.where(padding, 0, index)]
    rows[padding] = 0.0
    acc = rows.sum(axis=1, dtype=rows.dtype) + p["b0"]
    x0 = np.clip(acc, 0.0, 1.0)
    y = np.zeros(len(sides), dtype=acc.dtype)
    heads = {}
    for side in (0, 1):
        mask = np.flatnonzero(sides == side)
        if not len(mask):
            continue
        x = x0[mask]
        z1 = x @ p[f"w1_{side}"].T + p[f"b1_{side}"]
        x1 = np.clip(z1, 0.0, 1.0)
        z2 = x1 @ p[f"w2_{side}"].T + p[f"b2_{side}"]
        x2 = np.clip(z2, 0.0, 1.0)
        y[mask] = x2 @ p[f"w3_{side}"] + p[f"b3_{side}"][0]
        heads[side] = (mask, z1, x1, z2, x2)
    return _Pass(acc=acc, x0=x0, heads=heads, y=y)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (0.5 * (1.0 + np.tanh(0.5 * x))).astype(x.dtype)


def _gradients(
    net: FloatNet, index: np.ndarray, sides: np.ndarray, targets: np.ndarray, scale: float
) -> tuple[float, dict[str, np.ndarray]]:
    p = net.params
    f = _forward(net, index, sides)
    prob = _sigmoid(f.y / scale)
    diff = prob - targets
    loss = float(np.mean(diff.astype(np.float64) ** 2))
    dy = (2.0 / len(sides)) * diff * prob * (1.0 - prob) / scale
    grads: dict[str, np.ndarray] = {}
    dx0 = np.zeros_like(f.x0)
    for side in (0, 1):
        if side not in f.heads:
            for key in HEAD_KEYS:
                grads[f"{key}_{side}"] = np.zeros_like(p[f"{key}_{side}"])
            continue
        mask, z1, x1, z2, x2 = f.heads[side]
        d = dy[mask]
        grads[f"w3_{side}"] = x2.T @ d
        grads[f"b3_{side}"] = np.array([d.sum()], dtype=d.dtype)
        dz2 = np.outer(d, p[f"w3_{side}"]) * ((z2 > 0.0) & (z2 < 1.0))
        grads[f"w2_{side}"] = dz2.T @ x1
        grads[f"b2_{side}"] = dz2.sum(axis=0)
        dz1 = (dz2 @ p[f"w2_{side}"]) * ((z1 > 0.0) & (z1 < 1.0))
        grads[f"w1_{side}"] = dz1.T @ f.x0[mask]
        grads[f"b1_{side}"] = dz1.sum(axis=0)
        dx0[mask] = dz1 @ p[f"w1_{side}"]
    dacc = dx0 * ((f.acc > 0.0) & (f.acc < 1.0))
    grads["b0"] = dacc.sum(axis=0)
    grads["w0"] = _sparse_rows(index, dacc, net.n_inputs)
    return loss, grads


def _sparse_rows(index: np.ndarray, dacc: np.ndarray, n_inputs: int) -> np.ndarray:
    """``W0``'s gradient: each active input's rows summed, in sorted input order."""
    everything: np.ndarray = index.ravel()
    keep = everything < n_inputs
    flat: np.ndarray = everything[keep]
    positions: np.ndarray = np.repeat(np.arange(index.shape[0]), index.shape[1])[keep]
    out = np.zeros((n_inputs, dacc.shape[1]), dtype=dacc.dtype)
    if not len(flat):
        return out
    order = np.argsort(flat, kind="stable")
    flat, positions = flat[order], positions[order]
    unique, starts = np.unique(flat, return_index=True)
    out[unique] = np.add.reduceat(dacc[positions], starts, axis=0)
    return out


def train(
    data: Dataset, procedure: Procedure, report: Report | None = None
) -> FloatNet:
    """Fit the float net to ``data`` under ``procedure``'s widths and schedule."""
    if not data.positions:
        raise ValueError("no positions to train on: every game ended inside the skipped plies")
    net = initial(data.n_inputs, procedure, data.most_active)
    rng = np.random.default_rng(procedure.init_seed + 1)
    first = {key: np.zeros_like(value) for key, value in net.params.items()}
    second = {key: np.zeros_like(value) for key, value in net.params.items()}
    step = 0
    for epoch in range(procedure.epochs):
        order = rng.permutation(data.positions)
        total = 0.0
        for begin in range(0, data.positions, procedure.batch):
            rows = order[begin : begin + procedure.batch]
            index = _padded(data.offsets, data.inputs, rows, data.n_inputs)
            loss, grads = _gradients(
                net, index, data.sides[rows], data.targets[rows], procedure.scale
            )
            total += loss * len(rows)
            step += 1
            correction1 = 1.0 - ADAM_BETA1**step
            correction2 = 1.0 - ADAM_BETA2**step
            rate = procedure.learning_rate * np.sqrt(correction2) / correction1
            for key, grad in grads.items():
                m, v = first[key], second[key]
                m *= ADAM_BETA1
                m += (1.0 - ADAM_BETA1) * grad
                v *= ADAM_BETA2
                v += (1.0 - ADAM_BETA2) * grad * grad
                net.params[key] -= (rate * m / (np.sqrt(v) + ADAM_EPSILON)).astype(np.float32)
            _clip(net)
        net.history.append(round(total / data.positions, 8))
        if report is not None:
            report(f"  epoch {epoch + 1}/{procedure.epochs}: loss {net.history[-1]:.6f}")
    return net


def predict(net: FloatNet, rows: Sequence[Sequence[int]], sides: Sequence[int]) -> np.ndarray:
    """The float net's output, in built-in units, for positions given as input lists."""
    lengths = np.array([len(row) for row in rows], dtype=np.int64)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.array([i for row in rows for i in row], dtype=np.int64)
    index = _padded(offsets, flat, np.arange(len(rows)), net.n_inputs)
    return _forward(net, index, np.asarray(sides, dtype=np.int8)).y


def _ints(values: np.ndarray, factor: float, limit: int) -> list[int]:
    scaled = np.clip(np.rint(values.astype(np.float64) * factor), -limit, limit)
    return [int(v) for v in scaled.astype(np.int64).ravel()]


def quantize(net: FloatNet) -> bytes:
    """The ``CVSNET01`` store of ``net``, in the order ``nets.Net.parse`` reads it."""
    p = net.params
    bias_scale = HIDDEN_SCALE * ACTIVATION_MAX
    output_scale = OUTPUT_SCALE / ACTIVATION_MAX
    heads = []
    for side in (0, 1):
        heads.append(
            (
                _ints(p[f"w1_{side}"], HIDDEN_SCALE, I8_MAX),
                _ints(p[f"b1_{side}"], bias_scale, I32_LIMIT),
                _ints(p[f"w2_{side}"], HIDDEN_SCALE, I8_MAX),
                _ints(p[f"b2_{side}"], bias_scale, I32_LIMIT),
                _ints(p[f"w3_{side}"], output_scale, I8_MAX),
                _ints(p[f"b3_{side}"], OUTPUT_SCALE, I32_LIMIT)[0],
            )
        )
    return encode(
        net.n_inputs,
        net.accumulator,
        net.hidden1,
        net.hidden2,
        _ints(p["w0"], ACTIVATION_MAX, net.w0_limit),
        _ints(p["b0"], ACTIVATION_MAX, I8_MAX),
        heads,
    )
