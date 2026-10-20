"""The map's picture: where each game is drawn, remembered from one map to the next.

`docs/plan-map.md` §4, M3. The relation is `spec/likeness.py`'s and the families are
`web/families.py`'s; this module only places the games, so that games alike are drawn
near each other and a game that joins or leaves moves the others little.

- **t-SNE, exact.** Each game's neighbourhood is a Gaussian over the likeness distance,
  its width chosen so the game has `PERPLEXITY` effective neighbours; the picture is the
  one whose Student-t neighbourhoods match them best (van der Maaten and Hinton 2008).
  The library is about a hundred games, so every pair is computed: no tree, no sampling,
  no random number, in plain Python.
- **Cold**, with no previous picture: started from classical scaling of the distance
  (the two leading axes, by subspace iteration), early exaggeration first, then turned
  so its long axis lies across and reflected so its weight leans right and down. A cold
  draw happens only when there is no previous picture or its `LAYOUT_VERSION` differs.
- **Settled**, from the previous picture (`runs/graph/map.json`): every game it placed
  starts where it was, a new one at the mean of its neighbours' places weighed by its
  neighbourhood. The descent then takes `WARM` plain steps at the early momentum, with
  no exaggeration and **no gains**: gains are how a cold draw crosses the picture, and a
  settled game should only shuffle, not travel (with gains, every settle released a game
  the cold draw had left between two families and sent it 40% of the way across; M3's
  readings in the plan). The result is turned, reflected and moved (Procrustes) onto the
  previous picture, so north stays north and a returning reader finds the families where
  they were. Positions keep t-SNE's own scale, so a settled draw starts at rest; a page
  scales the picture to fit.
- **Kept**, when the previous picture holds exactly these games under this layout and
  likeness: nothing changed, so nothing moves.

**A pure function of what it is given**, the library and the previous picture: games are
sorted by id before anything runs and every sum is taken in that order. Two games with the
same words would never part (nothing pushes two points at one place apart), so each
starting point is nudged by a vanishing amount around a golden-angle spiral, an amount
fixed by the start's own spread.

**The numbers here are the method's, none read off a library**: the plan's perplexity,
and the optimiser's published settings (exaggeration, the momentum switch, the gains,
the cold iteration count; van der Maaten 2014; the learning rate is the larger of the
`n / (4 * exaggeration)` rule of Belkina et al. 2019 and its floor), and `WARM`, a
settle's length, chosen with M3's readings in view (the plan records each count tried).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from operator import mul

from chess_variant_sandbox.spec.likeness import LIKENESS_VERSION, likeness_of_forms
from chess_variant_sandbox.spec.relate import NormalForm
from chess_variant_sandbox.web import families

# Bumped when a change here would draw another picture for the same library and the same
# previous picture. A stored picture of another version is not settled from.
LAYOUT_VERSION = "p1"

# How many neighbours each game's neighbourhood counts, in effect. The plan's.
PERPLEXITY = 10.0

# The optimiser (van der Maaten 2014): early exaggeration and how long it lasts, the
# momentum before and after the switch, the gains' step and floor, the iterations of a
# cold draw, and the learning rate's floor. A settle's length, `WARM`, is M3's own.
EXAGGERATION = 12.0
EXAGGERATED = 250
MOMENTUM = (0.5, 0.8)
GAIN_UP = 0.2
GAIN_DOWN = 0.8
GAIN_FLOOR = 0.01
COLD = 1000
WARM = 100
RATE = 50.0

# The width search's tolerance on a neighbourhood's entropy, and its longest search.
ENTROPY_TOLERANCE = 1e-5
SEARCH = 64

# The start's nudge, as a share of the start's own spread, and the scale a cold start is
# given (small, so exaggeration can gather the neighbourhoods before they spread).
NUDGE = 1e-4
START_SCALE = 1e-2

# Classical scaling's subspace iteration: when the axes have settled, and its longest run.
AXES_TOLERANCE = 1e-9
AXES_LONGEST = 2000

Point = tuple[float, float]

# The golden angle: consecutive nudges point as far from each other as they can.
GOLDEN = math.pi * (3.0 - math.sqrt(5.0))


# ------------------------------------------------------------------------- the build


def build(
    ids: Sequence[str],
    forms: Sequence[NormalForm],
    names: Sequence[str],
    previous: families.Memory | None = None,
) -> families.FamilyMap:
    """`families.build`'s map with each game's place in the picture.

    The picture is `previous`'s unchanged when it holds exactly these games under this
    `LAYOUT_VERSION` and likeness, settled from it when it holds any of them under this
    layout, and drawn cold otherwise.
    """
    found = families.build(ids, forms, names, previous)
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    ordered = [ids[i] for i in order]
    relation = likeness_of_forms([forms[i] for i in order])
    before = (
        previous.positions
        if previous is not None and previous.layout == LAYOUT_VERSION
        else None
    )
    if (
        before is not None
        and previous is not None
        and previous.likeness == LIKENESS_VERSION
        and set(before) == set(ordered)
    ):
        positions = {game: before[game] for game in ordered}
    else:
        positions = place(ordered, relation.matrix(), before)
    return replace(found, positions=positions, layout=LAYOUT_VERSION)


def place(
    ids: Sequence[str],
    distances: Sequence[Sequence[float]],
    previous: Mapping[str, Point] | None = None,
) -> dict[str, Point]:
    """Each game's place, from the distances between them (in `ids`' order).

    Settled from `previous` when it places any of these games, drawn cold otherwise.
    """
    n = len(ids)
    if len(distances) != n or any(len(row) != n for row in distances):
        raise ValueError("the distances must be one square row per game")
    if n == 0:
        return {}
    if n == 1:
        return {ids[0]: (0.0, 0.0)}
    order = sorted(range(n), key=lambda i: ids[i])
    games = [ids[i] for i in order]
    d = [[float(distances[i][j]) for j in order] for i in order]
    conditional = _neighbourhoods(d)
    joint = _joint(conditional)
    known = [i for i, game in enumerate(games) if previous and game in previous]
    if previous and known:
        old = [previous[games[i]] for i in known]
        start = _settled_start(games, conditional, known, previous)
        xs, ys = _descend(joint, _nudged(start), WARM, 0, settling=True)
        xs, ys = _onto([(xs[i], ys[i]) for i in range(n)], known, old)
    else:
        start = _scaled(_classical(d))
        xs, ys = _descend(joint, _nudged(start), COLD, EXAGGERATED)
        xs, ys = _upright(xs, ys)
    return {game: (xs[i], ys[i]) for i, game in enumerate(games)}


# ----------------------------------------------------------------- the neighbourhoods


def _neighbourhoods(d: Sequence[Sequence[float]]) -> list[list[float]]:
    """Each game's conditional neighbourhood: a Gaussian over its squared distances.

    The width is found by bisection so the neighbourhood's entropy is `log(PERPLEXITY)`,
    or as near as the library allows (a game has at most `n - 1` neighbours).
    """
    n = len(d)
    target = math.log(min(PERPLEXITY, n - 1))
    rows: list[list[float]] = []
    for i in range(n):
        squared = [d[i][j] * d[i][j] for j in range(n) if j != i]
        least = min(squared)
        low, high, beta = 0.0, math.inf, 1.0
        shares = [1.0 / len(squared)] * len(squared)
        for _ in range(SEARCH):
            weights = [math.exp(-(s - least) * beta) for s in squared]
            total = sum(weights)
            shares = [w / total for w in weights]
            entropy = -sum(p * math.log(p) for p in shares if p > 0)
            if abs(entropy - target) < ENTROPY_TOLERANCE:
                break
            if entropy > target:
                low = beta
                beta = beta * 2 if high == math.inf else (beta + high) / 2
            else:
                high = beta
                beta = (beta + low) / 2
        row = shares[:i] + [0.0] + shares[i:]
        rows.append(row)
    return rows


def _joint(conditional: Sequence[Sequence[float]]) -> list[list[float]]:
    """The symmetric affinities: each pair's two conditionals averaged over the library."""
    n = len(conditional)
    return [
        [(conditional[i][j] + conditional[j][i]) / (2 * n) for j in range(n)]
        for i in range(n)
    ]


# ------------------------------------------------------------------------ the starts


def _classical(d: Sequence[Sequence[float]]) -> list[Point]:
    """Classical scaling's two leading axes, by subspace iteration from a fixed start.

    The doubly centred squared distances, shifted by a bound on their spectrum so the
    leading eigenvalues are the largest ones and not the most negative; each axis is
    signed so its largest coordinate is positive and scaled by its eigenvalue's root.
    """
    n = len(d)
    squared = [[value * value for value in row] for row in d]
    row_mean = [sum(row) / n for row in squared]
    grand = sum(row_mean) / n
    b = [
        [-(squared[i][j] - row_mean[i] - row_mean[j] + grand) / 2 for j in range(n)]
        for i in range(n)
    ]
    shift = max(sum(abs(value) for value in row) for row in b)
    for i in range(n):
        b[i][i] += shift
    # A fixed start no axis can be orthogonal to by accident of the library: two
    # different smooth waves over the games' order.
    first = [math.cos(GOLDEN * (i + 1)) + 1.0 for i in range(n)]
    second = [math.sin(GOLDEN * (i + 1)) for i in range(n)]
    axes = _orthonormal([first, second])
    for _ in range(AXES_LONGEST):
        moved = _orthonormal([[sum(map(mul, row, axis)) for row in b] for axis in axes])
        change = max(
            1.0 - abs(sum(map(mul, old, new))) for old, new in zip(axes, moved, strict=True)
        )
        axes = moved
        if change < AXES_TOLERANCE:
            break
    # The 2-by-2 problem inside the subspace (Rayleigh-Ritz), solved in closed form.
    images = [[sum(map(mul, row, axis)) for row in b] for axis in axes]
    a11 = sum(map(mul, axes[0], images[0]))
    a12 = sum(map(mul, axes[0], images[1]))
    a22 = sum(map(mul, axes[1], images[1]))
    angle = math.atan2(2 * a12, a11 - a22) / 2
    c, s = math.cos(angle), math.sin(angle)
    u = [c * p + s * q for p, q in zip(axes[0], axes[1], strict=True)]
    v = [-s * p + c * q for p, q in zip(axes[0], axes[1], strict=True)]
    lu = a11 * c * c + 2 * a12 * c * s + a22 * s * s - shift
    lv = a11 * s * s - 2 * a12 * c * s + a22 * c * c - shift
    if lv > lu:
        u, v, lu, lv = v, u, lv, lu
    u = [value * math.sqrt(max(lu, 0.0)) for value in _signed(u)]
    v = [value * math.sqrt(max(lv, 0.0)) for value in _signed(v)]
    return list(zip(u, v, strict=True))


def _orthonormal(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Gram-Schmidt, twice over for rounding; a vector that vanishes is left as it is."""
    found: list[list[float]] = []
    for vector in vectors:
        current = list(vector)
        for _ in range(2):
            for basis in found:
                dot = sum(map(mul, current, basis))
                current = [a - dot * b for a, b in zip(current, basis, strict=True)]
        norm = math.sqrt(sum(map(mul, current, current)))
        found.append([a / norm for a in current] if norm > 0 else current)
    return found


def _signed(vector: list[float]) -> list[float]:
    """The vector, turned so its largest coordinate (the first, on a tie) is positive."""
    largest = max(range(len(vector)), key=lambda i: (abs(vector[i]), -i))
    return [-a for a in vector] if vector[largest] < 0 else vector


def _scaled(points: Sequence[Point]) -> list[Point]:
    """These points centred, at the cold start's small spread."""
    spread = _spread(points)
    if spread == 0:
        return [(0.0, 0.0) for _ in points]
    cx, cy = _centre(points)
    return [((x - cx) / spread * START_SCALE, (y - cy) / spread * START_SCALE) for x, y in points]


def _settled_start(
    games: Sequence[str],
    conditional: Sequence[Sequence[float]],
    known: Sequence[int],
    previous: Mapping[str, Point],
) -> list[Point]:
    """The previous places, and a new game at its neighbourhood's mean over the known.

    A new game whose neighbourhood gives the known games no weight starts at the known
    game nearest it in its neighbourhood's order (the first known, when it has none).
    """
    held = {i: previous[games[i]] for i in known}
    start: list[Point] = []
    for i, _game in enumerate(games):
        if i in held:
            start.append(held[i])
            continue
        weight = sum(conditional[i][j] for j in known)
        if weight > 0:
            x = sum(conditional[i][j] * held[j][0] for j in known) / weight
            y = sum(conditional[i][j] * held[j][1] for j in known) / weight
            start.append((x, y))
        else:
            start.append(held[known[0]])
    return start


def _nudged(points: Sequence[Point]) -> list[Point]:
    """Each point moved a vanishing step along a golden-angle spiral, so none coincide."""
    spread = _spread(points) or 1.0
    step = spread * NUDGE
    return [
        (x + step * math.cos(GOLDEN * i), y + step * math.sin(GOLDEN * i))
        for i, (x, y) in enumerate(points)
    ]


# ------------------------------------------------------------------------- the descent


def _descend(
    joint: Sequence[Sequence[float]],
    start: Sequence[Point],
    iterations: int,
    exaggerated: int,
    settling: bool = False,
) -> tuple[list[float], list[float]]:
    """Gradient descent on t-SNE's cost, with momentum and per-coordinate gains.

    The gradient on game *i* is `4 * sum_j (p_ij - q_ij) * w_ij * (y_i - y_j)`, with `w`
    the Student-t kernel and `q = w / sum(w)`; every pair, every iteration. All games move
    together, from the gradient at the places they were.
    """
    n = len(start)
    xs = [x for x, _ in start]
    ys = [y for _, y in start]
    rate = max(n / (4 * EXAGGERATION), RATE)
    step_x = [0.0] * n
    step_y = [0.0] * n
    gain_x = [1.0] * n
    gain_y = [1.0] * n
    for it in range(iterations):
        boost = EXAGGERATION if it < exaggerated else 1.0
        momentum = MOMENTUM[0] if settling or it < EXAGGERATED else MOMENTUM[1]
        kernels: list[list[float]] = []
        total = 0.0
        for i in range(n):
            xi, yi = xs[i], ys[i]
            row = [
                1.0 / (1.0 + (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj))
                for xj, yj in zip(xs, ys, strict=True)
            ]
            row[i] = 0.0
            total += sum(row)
            kernels.append(row)
        inverse = 1.0 / total
        grad_x = [0.0] * n
        grad_y = [0.0] * n
        for i in range(n):
            pull = [(boost * p - w * inverse) * w for p, w in zip(joint[i], kernels[i], strict=True)]
            weight = sum(pull)
            grad_x[i] = 4.0 * (xs[i] * weight - sum(map(mul, pull, xs)))
            grad_y[i] = 4.0 * (ys[i] * weight - sum(map(mul, pull, ys)))
        for i in range(n):
            if not settling:
                gain_x[i] = _gain(gain_x[i], grad_x[i], step_x[i])
                gain_y[i] = _gain(gain_y[i], grad_y[i], step_y[i])
            step_x[i] = momentum * step_x[i] - rate * gain_x[i] * grad_x[i]
            step_y[i] = momentum * step_y[i] - rate * gain_y[i] * grad_y[i]
        xs = [x + s for x, s in zip(xs, step_x, strict=True)]
        ys = [y + s for y, s in zip(ys, step_y, strict=True)]
        mx, my = sum(xs) / n, sum(ys) / n
        xs = [x - mx for x in xs]
        ys = [y - my for y in ys]
    return xs, ys


def _gain(gain: float, grad: float, step: float) -> float:
    """A coordinate's gain grows while the gradient turns against its step, else shrinks."""
    if _sign(grad) != _sign(step):
        return gain + GAIN_UP
    return max(gain * GAIN_DOWN, GAIN_FLOOR)


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


# ----------------------------------------------------------------- the picture's frame


def _upright(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """A cold picture centred, its long axis across, leaning right and down.

    Its scale is t-SNE's own, so a settled draw starts where the descent came to rest.

    Turned by its principal axes (closed form for two dimensions); reflected on each axis
    so the third moment is not negative, which fixes the picture's handedness from its
    own shape.
    """
    n = len(xs)
    cx, cy = sum(xs) / n, sum(ys) / n
    xs = [x - cx for x in xs]
    ys = [y - cy for y in ys]
    sxx = sum(map(mul, xs, xs))
    syy = sum(map(mul, ys, ys))
    sxy = sum(map(mul, xs, ys))
    angle = math.atan2(2 * sxy, sxx - syy) / 2
    c, s = math.cos(angle), math.sin(angle)
    us = [c * x + s * y for x, y in zip(xs, ys, strict=True)]
    vs = [-s * x + c * y for x, y in zip(xs, ys, strict=True)]
    if sum(u * u * u for u in us) < 0:
        us = [-u for u in us]
    if sum(v * v * v for v in vs) < 0:
        vs = [-v for v in vs]
    return us, vs


def _onto(
    points: Sequence[Point], known: Sequence[int], old: Sequence[Point]
) -> tuple[list[float], list[float]]:
    """These points turned, maybe reflected, and moved to fit `old` at `known`.

    Orthogonal Procrustes in two dimensions, closed form: the rotation (or, when it fits
    better, the reflection) that brings the known games nearest their old places, applied
    to every game. Not scaled: the scale is the descent's own, where it came to rest.
    """
    mine = [points[i] for i in known]
    mx, my = _centre(mine)
    ox, oy = _centre(old)
    a = [(x - mx, y - my) for x, y in mine]
    b = [(x - ox, y - oy) for x, y in old]
    best: tuple[float, float, int] | None = None
    for flip in (1, -1):
        dot = sum(ax * bx + flip * ay * by for (ax, ay), (bx, by) in zip(a, b, strict=True))
        cross = sum(ax * by - flip * ay * bx for (ax, ay), (bx, by) in zip(a, b, strict=True))
        fit = math.hypot(dot, cross)
        if best is None or fit > best[0]:
            best = (fit, math.atan2(cross, dot), flip)
    assert best is not None
    _, angle, flip = best
    c, s = math.cos(angle), math.sin(angle)
    xs: list[float] = []
    ys: list[float] = []
    for x, y in points:
        px, py = x - mx, (y - my) * flip
        xs.append(c * px - s * py + ox)
        ys.append(s * px + c * py + oy)
    return xs, ys


def _centre(points: Sequence[Point]) -> Point:
    n = len(points)
    return sum(x for x, _ in points) / n, sum(y for _, y in points) / n


def _spread(points: Sequence[Point]) -> float:
    """The root-mean-square distance of these points from their centre."""
    cx, cy = _centre(points)
    return math.sqrt(sum((x - cx) ** 2 + (y - cy) ** 2 for x, y in points) / len(points))
