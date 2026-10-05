"""The building blocks a game uses, read off a spec rather than off gameplay.

`available()` is the schema's own two lists -- the five movement kinds and every rule
key `complexity._RULE_COSTS` charges for -- and `used(spec)` is the subset a given spec
draws on. This lives beside `complexity.py` rather than inside it: that module is one of
`measurement.METRICS_SOURCES`, and a new function there would move the fingerprint every
stored measurement is read against. See `docs/plan-meta-game.md` §2.9, "Where the reader
lives".

A rule key's *active* test is not re-implemented here. `used()` asks
`complexity._count_special_rules` about one key at a time, so it shares the one place
that reading is done and copies none of it. Both functions read `complexity._RULE_COSTS`
at call time rather than caching it, so a test that adds a charged key is picked up
without an import-order dependency.
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox.spec import complexity, schema


def _charged_rule_keys() -> tuple[str, ...]:
    """Rule keys worth teaching a player. Bookkeeping (`max_plies` and friends) is free."""
    return tuple(key for key, cost in complexity._RULE_COSTS.items() if cost > 0)


def available() -> tuple[str, ...]:
    """Every building block a spec could use: movement kinds, then rule keys."""
    return schema.MOVE_KINDS + _charged_rule_keys()


def used(spec: dict[str, Any]) -> frozenset[str]:
    """The subset of `available()` this spec draws on.

    A movement kind is used when some piece declares a move of that kind. A rule key
    is used when `_count_special_rules` would charge for it -- asked with only that key
    present, so a default value or a rule the compiler drops is not counted.
    """
    found: set[str] = set()

    for piece in spec.get("pieces", []):
        for move in piece.get("moves", []):
            kind = move.get("kind")
            if kind in schema.MOVE_KINDS:
                found.add(kind)

    rules = spec.get("rules", {})
    for key in _charged_rule_keys():
        if key not in rules:
            continue
        if complexity._count_special_rules({key: rules[key]}) > 0:
            found.add(key)

    return frozenset(found)
