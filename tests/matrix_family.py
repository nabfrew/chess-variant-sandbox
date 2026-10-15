"""The pre-registered matrices that choose `REGRET_ITERATIONS` (`docs/plan-simultaneous.md`,
*Q918b, pre-registered*), shared by the test and the measurement that picked the count."""

from __future__ import annotations

import random
from collections.abc import Callable

from chess_variant_sandbox.agents.matrix import solve

CANDIDATES = (100, 200, 400, 800, 1600, 3200)
SHAPES = ((2, 2), (3, 3), (5, 5), (8, 8), (12, 12), (20, 20), (30, 30), (45, 45), (6, 30), (30, 6))
STAKE = 1_000_000.0

Matrix = tuple[list[float], int, int]


def gap(values: list[float], rows: int, cols: int, p: list[float], q: list[float]) -> float:
    """max over rows of (A q) minus min over columns of (p^T A): zero exactly at an
    equilibrium, and a bound on how far the value is from the game's."""
    best_row = max(
        sum(values[i * cols + j] * q[j] for j in range(cols)) for i in range(rows)
    )
    worst_col = min(
        sum(values[i * cols + j] * p[i] for i in range(rows)) for j in range(cols)
    )
    return best_row - worst_col


def random_family() -> list[Matrix]:
    """(d): five of each shape, integers in [-10, 10], drawn in the listed order."""
    rng = random.Random(918)
    family = []
    for rows, cols in SHAPES:
        for _ in range(5):
            family.append(([float(rng.randint(-10, 10)) for _ in range(rows * cols)], rows, cols))
    return family


def stake_family() -> list[Matrix]:
    """(e): a royal at stake: 20 x 20 in [-3, 3], three columns of rows 0-9 at -1,000,000."""
    rng = random.Random(9180)
    family = []
    for _ in range(5):
        values = [float(rng.randint(-3, 3)) for _ in range(400)]
        for column in rng.sample(range(20), 3):
            for row in range(10):
                values[row * 20 + column] = -STAKE
        family.append((values, 20, 20))
    return family


def _near(found: list[float], exact: list[float], tolerance: float = 0.01) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(found, exact, strict=True))


def exact_cases() -> list[tuple[str, Matrix, list[float], list[float]]]:
    """(a) to (c): matrices whose equilibrium is known exactly."""
    return [
        ("matching pennies", ([1.0, -1.0, -1.0, 1.0], 2, 2), [0.5, 0.5], [0.5, 0.5]),
        (
            "rock, paper, scissors",
            ([0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0], 3, 3),
            [1 / 3] * 3,
            [1 / 3] * 3,
        ),
        ("biased 2x2", ([3.0, -1.0, -2.0, 1.0], 2, 2), [3 / 7, 4 / 7], [2 / 7, 5 / 7]),
    ]


def failures(iterations: int, solver: Callable[..., tuple[list[float], list[float]]] = solve):
    """Every criterion (a) to (e) ``iterations`` misses, by name: ``(a_to_d, e)``."""
    missed = []
    for name, (values, rows, cols), row, col in exact_cases():
        p, q = solver(values, rows, cols, iterations)
        if not (_near(p, row) and _near(q, col)):
            missed.append(name)
    for index, (values, rows, cols) in enumerate(random_family()):
        p, q = solver(values, rows, cols, iterations)
        if gap(values, rows, cols, p, q) > 0.01 * (max(values) - min(values)):
            missed.append(f"random {rows}x{cols} #{index % 5}")
    stake = []
    for index, (values, rows, cols) in enumerate(stake_family()):
        p, q = solver(values, rows, cols, iterations)
        if gap(values, rows, cols, p, q) > 0.06:
            stake.append(f"stake #{index}")
    return missed, stake
