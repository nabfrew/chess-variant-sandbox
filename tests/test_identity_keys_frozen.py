"""Every committed spec's `relate.identity_key`, frozen (Q866).

The guard for a change to how `identity_key` is *computed* that must not change what it
*says*, such as cutting `_orientation_played`'s second compile (Q866's conclusion lives in
`docs/reference/engine-and-spec-record.md`, *Things learned the hard way*, and in
`compile.compile_two_readings`).
`_identity_keys.json` was taken on `origin/main` 54dedd9. Each row stores the spec's
mechanics hash beside its key, and a row whose spec has since become another game
(different hash) is skipped rather than failed: an in-place mechanics edit is a new game
on purpose (`CLAUDE.md`), and its key is not this test's to hold.

Q1072 found it red on 72 of 72 held rows, from two changes, while it sat in the slow tier:
- T1 (a53ead54) put `moves[].path` into every move's tuple in `relate._move_form`, its off
  value included, which moved every key. A regression: fixed in the code (the path joins a
  move only where it turns), so every key read before T1 came back unchanged.
- Q1008b (2cd61421) made `relate._build` read every spec through
  `spelling.current_spelling`, on purpose, so a spelling is not a difference. That moved
  the four rows written in an old spelling (makruk's lone promotion rank and torus's,
  omega's and dobutsu's re-spelt rules). Re-frozen at Q1072 (omega eab7a152, dobutsu
  5ef17730, makruk f0370a1c, torus 23eaab8d, each the old key's first eight digits). No
  stored record or store row holds an identity key; every reader compares two keys
  computed in one process.
It takes under a second, so since Q1072 it runs in the fast suite, not the slow tier.
"""

from __future__ import annotations

import json
from pathlib import Path

from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.relate import _normal_cached, identity_key

ROOT = Path(__file__).resolve().parent.parent
FROZEN: dict[str, list[str]] = json.loads(
    (Path(__file__).with_name("_identity_keys.json")).read_text(encoding="utf-8")
)


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
