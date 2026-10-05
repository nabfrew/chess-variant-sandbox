"""Why a game has no Fairy-Stockfish counterpart, in the designer's words.

The translator refuses a ruleset with reasons that open on the part of the game's text
they are about -- ``rules.wrap: Fairy-Stockfish has no board without edges`` -- because
their first readers are the `fairy` command and the tests, which match on those paths.
A visitor never sees the game's text. The pages name a rule by its card's title, so the
same reason reads *the rule 'edges that join': Fairy-Stockfish has no board without
edges.* (`docs/plan-understanding.md` W5b, the guard's *a path into the game's text*).

A reason in, the same reason out, one for one: this changes no condition and drops
nothing. The names are `spec/plain.py`'s, the one table of what each part is called.
"""

from __future__ import annotations

import re

from chess_variant_sandbox.spec.describe import RULE_WORDS
from chess_variant_sandbox.spec.plain import FIELD_WORDS

# A part of the game's text as the translator spells it: `rules.check_limit`,
# `rules.explode_on_capture.by / .of`, `piece.captures`, `move.condition.region`. The
# translator's levels are singular (`piece`, `move`) where the text's lists are plural.
_PATH = re.compile(
    r"\b(rules|board|setup|piece|pieces|move|placement|spec)((?:\.[a-z_]+)+)"
    r"((?: / \.[a-z_]+)*)"
    # A value written beside its part: `rules.repetition_draw "never"`, `is 4`.
    r"(?:(?: is)? (\"[^\"]*\"|'[^']*'|\d+)(?=[:;]|$| beside))?"
)

# Where a level's fields are named in `FIELD_WORDS`, and how its owner is said.
_LEVELS = {
    "piece": ("pieces", "a piece's"),
    "pieces": ("pieces", "a piece's"),
    "move": ("pieces.moves", "a way of moving's"),
    "placement": ("setup.placements", "a row of where they start,"),
    "setup": ("setup", "where they start,"),
    "spec": ("", "the game's"),
}

# A label that reads as a setting rather than a name, said as the part it is.
_NAMED = {
    "board.shape": "the board's shape",
    "board.lattice": "the board's cells",
    "board.width": "the board's width",
    "board.height": "the board's height",
    "setup.orientation": "the second player's side of the board",
}


def _label(dotted: str, key: str) -> str:
    words = FIELD_WORDS.get(dotted)
    return words[0] if words else key.replace("_", " ")


def _named(level: str, keys: list[str], more: list[str]) -> str:
    """One path as the part of the game a person finds it under."""
    if level == "rules":
        head, subs = keys[0], keys[1:]
        title = RULE_WORDS.get(head, (head.replace("_", " "), ""))[0]
        said = f"the rule '{title}'"
        if not subs:
            return said
        stem = ".".join([head, *subs[:-1]])
        labels = [_label(f"rules.{stem}.{field}", field) for field in [subs[-1], *more]]
        return said + ", " + " and ".join(f"'{label}'" for label in labels)
    dotted = f"{level}.{'.'.join(keys)}"
    if dotted in _NAMED:
        return _NAMED[dotted]
    if level == "board":
        return f"the board's '{_label(dotted, keys[-1])}'"
    table, owner = _LEVELS[level]
    field = ".".join(keys)
    label = _label(f"{table}.{field}" if table else field, keys[-1])
    return f"{owner} '{label}'"


def plain_reason(reason: str) -> str:
    """One translator reason with every path in it named the way the pages name it."""

    def name(match: re.Match[str]) -> str:
        keys = match.group(2).lstrip(".").split(".")
        more = re.findall(r"\.([a-z_]+)", match.group(3))
        said = _named(match.group(1), keys, more)
        # A value beside its part reads as a setting: *set to "never"*, *set to 4*.
        return f"{said} set to {match.group(4)}" if match.group(4) else said

    text = _PATH.sub(name, reason)
    text = re.sub(r"\bthe spec does\b", "the rules do", text)
    text = re.sub(r"\bthe spec\b", "the rules", text)
    return text


def plain_reasons(reasons: list[str]) -> list[str]:
    """`plain_reason` over a list, in order and one for one."""
    return [plain_reason(reason) for reason in reasons]
