"""The committed PGO profile was trained on the core's current source (R-L5d).

A profile trained on older source still builds, so a stale one is silent: the functions a
change recompiled lose their profile and compile as without PGO. SR8h (generation 125) left
it stale and the shipped chess read 2.2% slower, atomic 11%, for no change of their own
(`docs/reference/speed-history.md` §4). This fails, naming the fix, whenever `rust/src/`
has moved since `rust/pgo/core.profdata` was trained: the change that moves the core
regenerates the profile as its last step (skill `bench-core` step 7).

Source only: the toolchain half of `regenerate.py --check` is the build backend's (it
declines a profile from another `rustc` or host, `test_pgo_backend.py`). A change under
`rust/` reaches every test (`tests/affected.py`), so a green stamp never vouches for this
one across a core change.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_variant_sandbox import measurement

STAMP = Path(__file__).resolve().parents[1] / "rust" / "pgo" / "core.profdata.json"


def test_the_committed_profile_was_trained_on_this_source():
    now = measurement.rust_source_fingerprint()
    if now is None or not STAMP.exists():
        pytest.skip("no rust/src or no committed profile here")
    trained_on = json.loads(STAMP.read_text(encoding="utf-8")).get("source")
    assert trained_on == now, (
        f"rust/pgo/core.profdata was trained on source {trained_on}, rust/src is now {now}: "
        "regenerate it as the core change's last step (python .claude/scripts/compute_lease.py "
        "acquire --profile heavy ... && uv run python rust/pgo/regenerate.py) and commit "
        "core.profdata with core.profdata.json"
    )
