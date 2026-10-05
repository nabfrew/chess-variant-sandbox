"""Speaking to Fairy-Stockfish: the translation, the FEN, and the engine process.

Fairy-Stockfish plays a large fraction of this project's corpus independently, which
makes it the one available answer to a question the harness cannot ask itself: *how good
is the ladder, really?* Every strength number in the index is the roster grading itself,
and a roster that is uniformly weak looks exactly like a roster that is uniformly strong.
An outside opponent is the only thing that can tell those apart.

Three things live under here and the split is by what can be wrong with them:

* `betza.py` and `variant.py` write our rules as theirs. Wrong here is *silent*, so
  nothing they produce is believed.
* `position.py` turns a position into a FEN and a move into text. Wrong here is loud:
  a move that comes back unmatched raises rather than being played.
* `engine.py` runs the process. `verify.py` is the check that makes the first bullet
  safe for the moves: it plays both engines over the same tree and the same random games
  and reports the first position whose legal moves differ. It establishes the same legal
  moves in every position it walks, not the same result. `results.py` checks the result:
  it asks both engines, at the same positions, whether the game is over and for whom.
  It needs `pyffish`, since the binary cannot be asked
  (`docs/reference/fairy-stockfish.md`, Q96 and Q404).

`available()` and `supports()` are what the rest of the project asks. Neither builds an
engine, because both are asked while a page is being drawn.
"""

from __future__ import annotations

from chess_variant_sandbox.fairy.betza import term
from chess_variant_sandbox.fairy.engine import (
    BINARY_ENV,
    FairyEngineError,
    UciEngine,
    available,
    binary_path,
    engine_name,
)
from chess_variant_sandbox.fairy.position import MoveMismatch, fen, move_of, uci_of
from chess_variant_sandbox.fairy.results import ResultCheck, check_results
from chess_variant_sandbox.fairy.variant import Untranslatable, Variant, translate, variant_name
from chess_variant_sandbox.fairy.verify import Verification, verify
from chess_variant_sandbox.fairy.words import plain_reasons

__all__ = [
    "BINARY_ENV",
    "FairyEngineError",
    "MoveMismatch",
    "ResultCheck",
    "UciEngine",
    "Untranslatable",
    "Variant",
    "Verification",
    "available",
    "binary_path",
    "check_results",
    "checked",
    "config_path",
    "engine_name",
    "fen",
    "move_of",
    "plain_reasons",
    "playable",
    "supports",
    "term",
    "translate",
    "uci_of",
    "variant_name",
    "verdict",
    "verify",
    "why_not",
]


# Translating a ruleset is pure and not free -- it compiles the spec -- and the picker
# asks about the same ruleset on every request. Keyed on the mechanics hash, which is
# what the answer actually depends on.
_TRANSLATED: dict[str, Variant | None] = {}


def supports(spec: dict) -> bool:
    """Whether this ruleset can be written as a Fairy-Stockfish variant at all.

    Cheap and pure: it says the translation exists, not that the two engines offer the
    same moves. That second question is `verify`'s, and it is answered against a running
    engine, which is far too expensive to ask while a menu is being built. `verify`
    stops there: it shows the same legal moves in every position it walks, not the same
    result. What protects a game in between is `position.move_of`, which refuses a move
    our engine does not have rather than playing it.
    """
    return translation(spec) is not None


def translation(spec: dict) -> Variant | None:
    """The translated variant for this ruleset, or ``None`` if there is not one."""
    from chess_variant_sandbox.spec.canonical import spec_hash

    try:
        key = spec_hash(spec)
    except (KeyError, TypeError, ValueError):
        return None
    if key not in _TRANSLATED:
        try:
            _TRANSLATED[key] = translate(spec)
        except Exception:  # noqa: BLE001 - a ruleset that will not translate is not an error
            _TRANSLATED[key] = None
    return _TRANSLATED[key]


def why_not(spec: dict) -> list[str]:
    """Why this ruleset has no Fairy-Stockfish counterpart, for a reader.

    Separate from `supports` because the reasons cost a translation attempt to produce
    and nothing that merely draws a menu wants them.
    """
    try:
        translate(spec)
    except Untranslatable as error:
        return list(error.reasons)
    except Exception as error:  # noqa: BLE001
        return [f"{type(error).__name__}: {error}"]
    return []


# ------------------------------------------------------- the variant file on disk


_CONFIG_PATHS: dict[str, str] = {}


def config_path(variant: Variant) -> str:
    """Where this variant's definition is written, for an engine to read.

    ``VariantPath`` takes a path and there is no option that takes the text, so a
    translated ruleset has to become a file. One file per variant for the life of the
    process, removed at exit: an engine is told the path once and reads it when the
    variant is selected, so a temporary that vanished when the caller returned would
    leave the next `setoption` reading nothing -- and a `UCI_Variant` naming a variant
    that failed to load is a segfault on the next position rather than an error.
    """
    import atexit
    import os
    import tempfile

    existing = _CONFIG_PATHS.get(variant.name)
    if existing is not None and os.path.exists(existing):
        return existing
    handle, path = tempfile.mkstemp(prefix=f"{variant.name}-", suffix=".ini")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(variant.config)
    if not _CONFIG_PATHS:
        atexit.register(_clean_configs)
    _CONFIG_PATHS[variant.name] = path
    return path


def _clean_configs() -> None:
    import contextlib
    import os

    for path in _CONFIG_PATHS.values():
        with contextlib.suppress(OSError):
            os.unlink(path)
    _CONFIG_PATHS.clear()


# ------------------------------------------------------------ can it actually play


# One verification per ruleset per process. It costs an engine start and a second or
# two of perft, and the picker asks about the same ruleset on every request.
_PLAYABLE: dict[str, Verification] = {}


def playable(spec: dict, *, depth: int = 2, games: int = 2, plies: int = 40) -> bool:
    """Whether this ruleset translates *and* offers the same legal moves over there.

    That is the same legal moves in every position the check walks, not the same result.
    The win and draw rules are not asked here: `chess-variant-sandbox fairy` checks them against
    `pyffish` (`results.check_results`, Q404), and a game against the opponent is still
    scored by our engine alone (`docs/reference/fairy-stockfish.md`, Q96).

    `supports` is the cheap half and is not enough to offer an opponent: a translation
    that is nearly right produces a legal-looking game under rules nobody wrote, and
    the corpus contains several -- a promotion the engine does not make, a double step
    it makes from the wrong rank, a royal piece it moves like a king. Every one of them
    was found by playing the two engines against each other and none of them by reading
    the configuration.

    Shallower than the committed corpus check on purpose. This runs while a menu is
    being drawn; the deep version is `chess-variant-sandbox fairy`, and what it protects is
    the same thing either way, because `position.move_of` refuses a move our engine
    does not have whatever this said.
    """
    return checked(spec, depth=depth, games=games, plies=plies) is not None


def checked(
    spec: dict, *, depth: int = 2, games: int = 2, plies: int = 40
) -> Verification | None:
    """The verification for this ruleset, or ``None`` if it did not pass."""
    result = verdict(spec, depth=depth, games=games, plies=plies)
    return result if result is not None and result.ok else None


def verdict(
    spec: dict, *, depth: int = 2, games: int = 2, plies: int = 40
) -> Verification | None:
    """What the installed engine said about this ruleset, passed *or failed*.

    ``None`` only where there was nothing to ask: no usable engine, or no translation
    to put to it. Otherwise the memoised `Verification`, whichever way it went -- so the
    picker that offers the opponent (`checked`) and the page that explains its absence
    read one record and cannot disagree. Before this the page had the yes and lost the
    no: a ruleset the installed build could not play looked exactly like a machine with
    no engine on it, and the reason -- the first position the two engines part company
    over, which is the one thing a reader can act on -- was computed and then dropped.
    """
    from chess_variant_sandbox.spec.canonical import spec_hash

    if not available():
        return None
    try:
        key = spec_hash(spec)
    except (KeyError, TypeError, ValueError):
        return None
    result = _PLAYABLE.get(key)
    if result is None:
        variant = translation(spec)
        if variant is None:
            return None
        try:
            with UciEngine() as engine:
                result = verify(
                    spec,
                    engine,
                    config_path(variant),
                    depth=depth,
                    games=games,
                    plies=plies,
                )
        except Exception as error:  # noqa: BLE001 - an engine that will not run is a no
            result = Verification(
                ruleset_hash=key,
                name=str(spec.get("name", "")),
                ok=False,
                reasons=[f"{type(error).__name__}: {error}"],
            )
        _PLAYABLE[key] = result
    return result
