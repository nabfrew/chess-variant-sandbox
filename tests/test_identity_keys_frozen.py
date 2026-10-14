"""Every committed spec's `relate.identity_key`, frozen (Q866).

The guard for a change to how `identity_key` is *computed* that must not change what it
*says*, such as cutting `_orientation_played`'s second compile (Q866's conclusion lives in
`docs/reference/engine-and-spec-record.md`, *Things learned the hard way*, and in
`compile.compile_two_readings`).
`_identity_keys.json` was taken on `origin/main` 54dedd9. Each row stores the spec's
mechanics hash beside its key, and a row whose spec has since become another game
(different hash) is skipped rather than failed: an in-place mechanics edit is a new game
on purpose (`CLAUDE.md`), and its key is not this test's to hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.relate import _normal_cached, identity_key

ROOT = Path(__file__).resolve().parent.parent
FROZEN: dict[str, list[str]] = json.loads(
    (Path(__file__).with_name("_identity_keys.json")).read_text(encoding="utf-8")
)


@pytest.mark.slow
def test_every_committed_specs_identity_key_is_unchanged() -> None:
    _normal_cached.cache_clear()
    moved, held = [], 0
    for name, (frozen_hash, frozen_key) in sorted(FROZEN.items()):
        path = ROOT / name
        if not path.exists():
            continue
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec_hash(spec) != frozen_hash:
            continue
        held += 1
        if identity_key(spec) != frozen_key:
            moved.append(name)
    assert held, "no frozen spec is still the same game; this test asks nothing"
    assert not moved, f"identity_key moved on: {moved}"
