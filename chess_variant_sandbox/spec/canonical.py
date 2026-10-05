"""Canonical serialisation and content hashing for rule specs.

The hash covers *mechanics only* — ``schema_version``, ``board``, ``pieces``, ``setup``
and ``rules`` — and deliberately excludes ``name`` and ``description``. Two proposals
that differ only in flavour text therefore collide, which is what the design loop
wants: re-proposing an already-evaluated game under a new name is the single most
common way a generation wastes a tournament budget, and a colliding hash catches it
before anything is simulated.
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


def spec_hash(spec: dict[str, Any]) -> str:
    """Return the short content hash identifying this ruleset's mechanics."""
    blob = canonical_json(spec, mechanics_only=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:HASH_LENGTH]


def load_spec(path: str | Path) -> dict[str, Any]:
    """Read a spec from disk. Does not validate; call ``validate_spec`` separately."""
    with open(path, encoding="utf-8") as handle:
        spec: dict[str, Any] = json.load(handle)
    return spec


def save_spec(spec: dict[str, Any], directory: str | Path) -> Path:
    """Write ``spec`` into ``directory`` named by its content hash, and return the path.

    Specs are immutable: if the file already exists its content is by definition
    identical, so the write is skipped rather than repeated.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec_hash(spec)}.json"
    if not path.exists():
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(spec, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    return path
