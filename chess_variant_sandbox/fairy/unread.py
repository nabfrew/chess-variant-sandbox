"""Options a Fairy-Stockfish build does not read, and why each one may go unread.

A build drops an option it does not know **in silence** when it loads a ``VariantPath``.
Q912 met the cost: the installed 2021 release reads the goal squares as
``whiteFlag``/``blackFlag``, never as ``flagRegionWhite``/``flagRegionBlack``, so for
racingkings and kingofthehill it searched a game with no goal and played on one move
from it (Q915). `verify` could not see that, because a goal decides the result and not
the legal moves, and the result check asks `pyffish`, a later build that reads both.

The build will say what it skipped, though: ``<binary> check <file>`` parses the file the
way a ``VariantPath`` is parsed and prints ``Invalid option: <key>`` for each key it does
not know and ``<key> - Invalid value <value> for type <type>`` for a value it cannot take.
`UciEngine.unread_options` asks it, and `reasons` holds every unread key to a reason it may
go unread, written down here with the condition that makes it true. A key with no reason,
or whose condition fails, refuses the translation for that build: refuse rather than
approximate (`.claude/rules/fairy-stockfish.md`).

This file says nothing about `pyffish`, which reads every key the translator writes.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

_INVALID_OPTION = re.compile(r"^Invalid option: (\S+)\s*$")
_INVALID_VALUE = re.compile(r"^(\S+) - Invalid value (.*) for type (.*)$")
_HEADER = ("Fairy-Stockfish", "Parsing variant")


def parse_check(output: str) -> frozenset[str] | None:
    """The keys a ``check`` run reports it did not take, or ``None`` if it did not run.

    A line that is neither the banner, a section header nor one of the two reports is
    kept whole, as ``"?" + line``, so that `reasons` refuses on it rather than reading
    past a warning nobody has seen.
    """
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not any(line.startswith("Parsing variant") for line in lines):
        return None
    found: set[str] = set()
    for line in lines:
        if line.startswith(_HEADER):
            continue
        option = _INVALID_OPTION.match(line)
        value = _INVALID_VALUE.match(line)
        if option:
            found.add(option.group(1))
        elif value:
            found.add(value.group(1))
        else:
            found.add("?" + line)
    return frozenset(found)


def options(config: str) -> dict[str, str]:
    """A one-section config's ``key = value`` lines."""
    written: dict[str, str] = {}
    for line in config.splitlines():
        key, equals, value = line.partition("=")
        if equals and not line.lstrip().startswith(("#", ";", "[")):
            written[key.strip()] = value.strip()
    return written


def _beside(older: str) -> Callable[[dict[str, str]], bool]:
    return lambda written: older in written


def _promotes_by_rank(written: dict[str, str]) -> bool:
    # Either the rank is written beside the region, or nothing promotes at all: with
    # `promotionPieceTypes = -` the build's default rank has nothing to offer.
    return "promotionRank" in written or written.get("promotionPieceTypes") == "-"


def _cleared(written: dict[str, str]) -> bool:
    # The 2021 build's Bitboard reader empties its target before it reads a square, so a
    # `-` it then calls invalid has already cleared the region. Probed by Q1030: chess
    # under `enPassantRegion = -` offers no e5d6 after 1.e4 a6 2.e5 d5, and offers it with
    # the line left out. Only `-` is known to do this; any other unread value refuses.
    return written.get("enPassantRegion") == "-"


def _custom_initial(written: dict[str, str]) -> list[str]:
    """The custom pieces whose notation carries an initial (`i`) move, by letter."""
    return [
        value.partition(":")[0]
        for key, value in written.items()
        if key.startswith("customPiece") and "i" in value.partition(":")[2]
    ]


def _no_custom_initial(written: dict[str, str]) -> bool:
    # The 2021 build reads the pawn slot's double step from `doubleStepRank*`, but gives a
    # custom piece's `i` move from every square: probed by Q1030, the sergeant
    # (`fWfFmifW2`) plays d5d7 and berolina's pawn its diagonal double step anywhere.
    return not _custom_initial(written)


def _spares_its_pawn(written: dict[str, str]) -> bool:
    # The 2021 build's blast spares its pawn slot and nothing else, without the option
    # (probed by Q915: a knight taking beside three pawns and a bishop leaves the pawns).
    return written.get("blastImmuneTypes", "").upper() == written.get("pawn", "p").upper()


def _always(_: dict[str, str]) -> bool:
    return True


# Every key the installed 2021 build (Fairy-Stockfish 14 LB) does not read in some library
# translation, each with the condition under which leaving it unread plays our game, and
# why. Found by `check` over every translating library ruleset (Q915). A new key reaches
# this table only with its reason; widening a condition to pass a check is the mistake
# this file exists to catch.
UNREAD: dict[str, tuple[Callable[[dict[str, str]], bool], str]] = {
    "flagRegionWhite": (_beside("whiteFlag"), "whiteFlag is written beside it (Q915)"),
    "flagRegionBlack": (_beside("blackFlag"), "blackFlag is written beside it (Q915)"),
    "dropRegionWhite": (
        _beside("whiteDropRegion"), "whiteDropRegion is written beside it (Q784)"
    ),
    "dropRegionBlack": (
        _beside("blackDropRegion"), "blackDropRegion is written beside it (Q784)"
    ),
    "promotionRegionWhite": (
        _promotes_by_rank,
        "promotionRank is written beside it (a zone that is not whole ranks is refused), "
        "or nothing promotes",
    ),
    "promotionRegionBlack": (
        _promotes_by_rank,
        "promotionRank is written beside it (a zone that is not whole ranks is refused), "
        "or nothing promotes",
    ),
    "doubleStepRegionWhite": (
        _no_custom_initial,
        "a move rule: doubleStepRank stands beside it where the ranks are one band "
        "(Q783), and `verify` walks the double steps either way; never beside a custom "
        "piece's initial move, which the build gives from every square (Q1030)",
    ),
    "doubleStepRegionBlack": (
        _no_custom_initial,
        "a move rule: doubleStepRank stands beside it where the ranks are one band "
        "(Q783), and `verify` walks the double steps either way; never beside a custom "
        "piece's initial move, which the build gives from every square (Q1030)",
    ),
    "promotionPawnTypes": (
        _always,
        "a move rule: the build promotes its pawn slot only, and `verify` walks the "
        "promotions (Build limits)",
    ),
    "castlingRookPieces": (_always, "a move rule: `verify` walks castling"),
    "castlingRookKingsideFile": (_always, "a move rule: `verify` walks castling"),
    "castlingRookQueensideFile": (_always, "a move rule: `verify` walks castling"),
    "nMoveRuleTypes": (
        _always,
        "only the build's search near the no-progress limit differs (Q411)",
    ),
    "enPassantRegion": (
        _cleared,
        "the build calls `-` an invalid region after it has emptied the region, so no "
        "square takes a capture in passing, which is what `-` says (Q1030)",
    ),
    "blastImmuneTypes": (
        _spares_its_pawn,
        "the build spares its pawn slot from a blast without being told",
    ),
}


def reasons(config: str, unread: Iterable[str]) -> list[str]:
    """Why this build cannot play ``config`` as written, one line per key; empty if it can."""
    written = options(config)
    said: list[str] = []
    for key in sorted(unread):
        if key.startswith("?"):
            said.append(f"the installed Fairy-Stockfish warned about this definition: {key[1:]}")
            continue
        entry = UNREAD.get(key)
        if entry is not None and entry[0](written):
            continue
        if key.startswith("doubleStepRegion") and _custom_initial(written):
            # Named, not "does not read": what refuses is the build's custom-piece `i`.
            sentence = (
                "the installed Fairy-Stockfish gives a piece's first-move-only step "
                f"(piece {', '.join(_custom_initial(written))}) from every square, "
                "so it would play another game"
            )
            if sentence not in said:
                said.append(sentence)
            continue
        said.append(
            f"the installed Fairy-Stockfish does not read {key}"
            + (f" = {written[key]}" if key in written else "")
            + ", so it would play another game"
        )
    return said
