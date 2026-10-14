"""What a library card asks of a game's compiled rules, and nothing else (Q887).

Three answers per ruleset sit on every card: the cheap gate's verdict (`playable`), whether
the rules translate to Fairy-Stockfish (`fairyTranslates`) and the opening position the
card draws (`still`). Each needs the rules compiled and handed to the core, and on a
process's first listing that compile is nine tenths of the payload: 6.7 s of 7.2 s over
the 87-game library on the development machine, about a minute on the box.

So `web/library.py` keeps the three answers on disk beside the tables cache, and reads
them back in the next process instead of compiling. **This module's import closure is
that store's key** (`library.card_answers_key`), with the piece drawings under
`assets/pieces/` and the core's build fingerprint: an edit to any file this module can
reach, a new drawing or a rebuilt core moves the key, and an answer stored under another
key is never read. That is why the answers are worked out *here* rather than in
`library.py` or `designing.py`, whose closures are most of the package and would make
every deploy a cold one; and why everything here takes the engine as a callable from
its caller rather than importing `library.engine_for`.

Keep it that way: a helper an answer calls is imported here, never reached through a
module this one does not import, or an edit to it would leave a stored answer standing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.loop.gates import SMOKE_GAMES
from chess_variant_sandbox.spec.schema import SpecError
from chess_variant_sandbox.viz.heading import directional_types
from chess_variant_sandbox.viz.pieces import icons_for, side_names
from chess_variant_sandbox.viz.walks import walk_payload
from chess_variant_sandbox.viz.wrapped import (
    fold_payload,
    goal_payload,
    portal_payload,
    walls_payload,
    wrap_payload,
    zone_payload,
)

# Compiles one spec, as `library.engine_for` does: `Engine(spec)`, memoised by the caller.
Compile = Callable[[], Engine]


def smoke(spec: dict[str, Any], compile_rules: Compile) -> tuple[bool, tuple[str, ...], bool]:
    """The cheap gate's verdict, played on the engine `compile_rules` returns, and whether
    the clock cut it.

    The card's still board and the translation ask for the same compile, so the three
    share one (the first-visit study's N8: a cold listing compiled every ruleset three
    times). Rules that do not compile go to the gate as text, which says why in the
    designer's words.

    The third value is true when the gate's time budget ran out before its games did
    (`smoke_test`'s refuse-or-whole): that refusal is about how busy the machine was, so
    it is answered for this process and never kept on disk.

    **On every core**, where the gate's default leaves one free. The playouts are one
    seeded stream per game, so the verdict is the same at any thread count, and a card
    asks while a listing waits: on the box's two vCPUs the default is one thread, and the
    gate's 150 games a ruleset were half of its minute's warm-up after a deploy (Q887).
    The core releases the interpreter while it plays, so requests still run beside it.
    """
    # Looked up when called, so a test that stands in for the gate reaches this one.
    from chess_variant_sandbox.loop.gates import smoke_test

    try:
        engine: Engine | None = compile_rules()
    except (SpecError, NotImplementedError):
        engine = None
    result = smoke_test(spec, engine=engine, workers=os.cpu_count() or 1)
    timed_out = result.stats.get("games", SMOKE_GAMES) < SMOKE_GAMES
    return result.passed, tuple(result.reasons), timed_out


def fairy(compile_rules: Compile) -> tuple[bool, tuple[str, ...]]:
    """One translation attempt per ruleset, for both halves of the answer.

    `fairy.supports` and `fairy.why_not` are the same attempt asked twice -- the first
    throws the reasons away and the second recomputes them -- and a listing that called
    both would compile every ruleset the translator refuses a second time to be told
    again why. No reasons *is* support, which is what `why_not` documents returning.

    It is `why_not`'s attempt, handed the rules `compile_rules` compiled rather than the
    text, so it shares the card's one compile with the gate and the still board.
    """
    from chess_variant_sandbox.fairy.variant import Untranslatable, translate

    try:
        translate(compile_rules().spec)
    except Untranslatable as error:
        reasons = tuple(error.reasons)
    except Exception as error:  # noqa: BLE001 - as `why_not`: a refusal, not an error
        reasons = (f"{type(error).__name__}: {error}",)
    else:
        reasons = ()
    return not reasons, reasons


def still(
    spec: dict[str, Any], compile_rules: Compile
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """One opening position per ruleset, with each piece's drawing named by its stem.

    `designing.opening_preview`'s position (`preview` on the initial state), with the
    drawings taken out: the listing sends each once (`library.piece_art`). Returned as
    the shape's JSON and the stem-to-drawing pairs, which is what the library memoises
    and stores.
    """
    engine = compile_rules()
    shown = preview(engine, engine.initial_state(), spec)
    art: dict[str, str] = {}
    named: list[list[dict[str, Any] | None]] = []
    for pair in shown.get("icons") or []:
        row: list[dict[str, Any] | None] = []
        for icon in pair:
            if icon is None:
                row.append(None)
                continue
            art[str(icon["stem"])] = str(icon["href"])
            row.append({"stem": icon["stem"], "rotate": bool(icon.get("rotate"))})
        named.append(row)
    shape = {**shown, "icons": named}
    return json.dumps(shape, sort_keys=True), tuple(sorted(art.items()))


def preview(engine: Any, state: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Enough for the editor to draw an opening position with the page's board.

    The designer's check and the card's still board both draw with this (it moved here
    from `designing.py` with Q887, so that the still board's stored answer is keyed on
    what draws it)."""
    art = icons_for(
        engine.spec.piece_names,
        engine.spec.piece_ids,
        (spec.get("appearance") or {}).get("icons"),
    )
    return {
        "width": engine.spec.width,
        "height": engine.spec.height,
        "lattice": engine.spec.lattice,
        "board": list(state.board),
        # Whose move it is in this position, the state's own: the game page's start
        # position names the side that moves first from it (K1).
        "toMove": int(state.side_to_move),
        **walls_payload(engine.spec),
        "pieceIds": list(engine.spec.piece_ids),
        "pieceNames": list(engine.spec.piece_names),
        "nTypes": engine.spec.n_types,
        "sideStride": engine.spec.side_stride,
        "facingAngles": [list(row) for row in engine.spec.facing_headings()],
        "directional": list(directional_types(engine.spec)),
        "icons": art,
        "sides": list(side_names(art)),
        # The same three compiled rules the play board reads, so the still preview
        # draws the goal ring, the promotion band and the seam rather than a bare
        # grid the draft's own rules are invisible on. Compiled, per player: the
        # spec spells these as square names, a keyword and edge names, and
        # `engine/compile.py` is the one place that means anything by them.
        # What each player holds at ply zero, the way the state carries it: a count
        # per piece type. A drops game's hand is part of its opening position, and a
        # still that drew only the board read as a game with no hand at all.
        "hands": [list(state.hands[0]), list(state.hands[1])],
        "handsUsed": engine.spec.drops is not None,
        "handsFromCaptures": bool(
            engine.spec.drops is not None and engine.spec.drops.from_captures
        ),
        "goals": goal_payload(engine.spec.reach),
        "zones": zone_payload(engine.spec.promotions),
        "wrap": wrap_payload(engine.spec.wrap, engine.spec.copies),
        # Schema 19: each portal end's pair, colour, movers and doors, compiled, so a
        # still draws an end as the play board does (`portalGlyph`). Absent without
        # portals, which keeps every other game's still as it was.
        **({"portals": portals} if (portals := portal_payload(engine.spec)) else {}),
        # Schema 22 (Q843d): where each fold stands, which is no board byte, so a still
        # draws its pool (`fold_payload`). Absent without folds.
        **({"folds": folds} if (folds := fold_payload(engine, state)) else {}),
        # Q852: every wild piece's walk in order, its starts and its jumps
        # (`viz.walks`), so a still draws where each goes. Absent without wild pieces.
        **({"walks": walks} if (walks := walk_payload(spec, engine.spec)) else {}),
    }
