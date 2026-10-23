"""Canonical serialisation, and a game's name.

**A game's name is its identity** (`spec_hash`, Q1008a): the name it had before Q1008a, read
from the fixed legacy table, or else the hash of its normal form (`normal.normal_form`). So
two spellings of one game are one game, and no game that had a name has changed it.

The name covers *mechanics only* -- ``board``, ``pieces``, ``setup`` and ``rules`` -- and
deliberately excludes ``name`` and ``description``. Two proposals that differ only in flavour
text therefore collide, which is what the design loop wants: re-proposing an
already-evaluated game under a new name is the single most common way a generation wastes a
tournament budget, and a colliding hash catches it before anything is simulated.

``spelling_hash`` is the name as it was computed before Q1008a, the SHA-256 of the mechanics
*as written* (``schema_version`` included). Only the legacy table's generator reads it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MECHANIC_KEYS = ("schema_version", "board", "pieces", "setup", "rules")
HASH_LENGTH = 12


def canonical_json(spec: dict[str, Any], mechanics_only: bool = False) -> str:
    """Serialise ``spec`` so that equal specs always produce byte-identical output."""
    payload = {k: spec[k] for k in MECHANIC_KEYS if k in spec} if mechanics_only else spec
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_text(text: str) -> str:
    """The short hash every name is spelled in."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def spelling_hash(spec: dict[str, Any]) -> str:
    """The hash of the mechanics as written: every game's name before Q1008a."""
    return hash_text(canonical_json(spec, mechanics_only=True))


def spec_hash(spec: dict[str, Any]) -> str:
    """This game's identity: its legacy name if it had one, else its normal-form hash.

    The name runs, measurements, nets, the site and the position keys are all keyed by.
    """
    # Deferred: `normal` reads `schema`, which imports this module.
    from chess_variant_sandbox.spec.normal import identity

    return identity(spec)


def load_spec(path: str | Path) -> dict[str, Any]:
    """Read a spec from disk. Does not validate; call ``validate_spec`` separately."""
    with open(path, encoding="utf-8") as handle:
        spec: dict[str, Any] = json.load(handle)
    return spec
