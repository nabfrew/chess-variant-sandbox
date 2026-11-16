"""θ, the shared evaluator (R-P3b1, `agents/training/shared.py`): its descriptors, its
gradient, its resumable trainer, and the materialised file the core reads."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numpy")

from chess_variant_sandbox.agents.nets import Net  # noqa: E402
from chess_variant_sandbox.agents.training import shared  # noqa: E402
from chess_variant_sandbox.engine import Engine  # noqa: E402
from chess_variant_sandbox.spec import load_spec  # noqa: E402

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"
TINY = shared.ProbeRecipe(hidden=8, accumulator=8, hidden1=4, hidden2=4, epochs=2, batch=16, seed=3)


def _tiny_data(seed: int = 0, games: int = 2, inputs: int = 12, positions: int = 40) -> shared.ProbeData:
    rng = np.random.default_rng(seed)
    total = games * inputs
    lengths = rng.integers(1, 5, size=positions)
    offsets = np.zeros(positions + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    game = rng.integers(0, games, size=positions).astype(np.int8)
    flat = np.concatenate([
        np.sort(rng.choice(inputs, size=n, replace=False)) + g * inputs
        for n, g in zip(lengths, game, strict=True)
    ]).astype(np.int32)
    return shared.ProbeData(
        hashes=[f"{g:012x}" for g in range(games)], n_inputs=[inputs] * games,
        starts=np.arange(games, dtype=np.int64) * inputs,
        descriptors=rng.uniform(0, 1, size=(total, shared.DESCRIPTOR_WIDTH)).astype(np.float32),
        gammas=rng.uniform(0, 1, size=(games, shared.GAME_WIDTH)).astype(np.float32),
        offsets=offsets, inputs=flat, sides=rng.integers(0, 2, size=positions).astype(np.int8),
        targets=rng.uniform(0, 1, size=positions).astype(np.float32), game=game,
    )


def test_gradient_matches_finite_differences() -> None:
    data = _tiny_data()
    theta = shared.initial(TINY)
    for key in theta.params:
        theta.params[key] = theta.params[key].astype(np.float64)
    data.descriptors = data.descriptors.astype(np.float64)
    data.gammas = data.gammas.astype(np.float64)
    data.targets = data.targets.astype(np.float64)
    rows = np.arange(data.positions)
    _, grads = shared._step(theta, data, rows, 1.0)
    rng = np.random.default_rng(1)
    for key in ("g1", "g2", "g3", "gb3", "bw", "bb", "w1_0", "w3_1"):
        values = theta.params[key]
        for _ in range(3):
            where = tuple(int(rng.integers(0, n)) for n in values.shape)
            keep = float(values[where])
            values[where] = keep + 1e-5
            up, _ = shared._step(theta, data, rows, 1.0)
            values[where] = keep - 1e-5
            down, _ = shared._step(theta, data, rows, 1.0)
            values[where] = keep
            numeric = (up - down) / 2e-5
            assert abs(numeric - float(grads[key][where])) <= 1e-4 + 1e-2 * abs(numeric), key


def test_same_data_same_theta() -> None:
    data = _tiny_data()
    one = shared.train(data, TINY)
    two = shared.train(data, TINY)
    assert all(np.array_equal(one.params[k], two.params[k]) for k in one.params)
    assert one.history == two.history


def test_resume_reaches_the_unbroken_bytes(tmp_path: Path) -> None:
    from chess_variant_sandbox.tournament import checkpoint

    data = _tiny_data(positions=80)
    unbroken = shared.train(data, TINY)
    with checkpoint.active(tmp_path / "c", build="test") as campaign:
        saves = campaign.saves({"test": 1})
        with pytest.raises(shared.OutOfTime):
            shared.train(data, TINY, saves=saves, deadline=time.monotonic() - 1.0)
    with checkpoint.active(tmp_path / "c", build="test") as campaign:
        resumed = shared.train(data, TINY, saves=campaign.saves({"test": 1}))
    assert all(np.array_equal(unbroken.params[k], resumed.params[k]) for k in unbroken.params)
    assert unbroken.history == resumed.history


@pytest.mark.parametrize("name", ["dull.json", "dobutsu.json", "tessabi.json"])
def test_materialised_store_is_this_games_net_byte_for_byte(name: str) -> None:
    engine = Engine(load_spec(RULESETS / name))
    descriptors = shared.input_descriptors(engine.spec)
    assert descriptors.shape == (int(engine.core.net_inputs()), shared.DESCRIPTOR_WIDTH)
    assert np.isfinite(descriptors).all()
    theta = shared.initial(replace(TINY, hidden=16, accumulator=16))
    blob = shared.materialise(theta, engine.spec)
    assert shared.materialise(theta, engine.spec) == blob
    net = Net.parse(blob)
    assert net.n_inputs == int(engine.core.net_inputs())


def test_digest_names_the_recipe() -> None:
    assert shared.ProbeRecipe().digest != replace(shared.ProbeRecipe(), seed=1).digest
    assert len(shared.ProbeRecipe().digest) == 12
