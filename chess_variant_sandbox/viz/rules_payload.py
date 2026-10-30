"""The compiled rules a page draws, assembled once for every page (CP5, PY14).

Three pages draw a game: the play board (`web/play.py`'s snapshot, its ``ruleset`` block),
the offline replay (`viz/html.py`) and the still preview a library card and the designer
draw (`web/card_rules.py`). Each used to pick its own subset of the same compiled rules,
and they drifted: the replay carried no explosion, speed, rest or purse, the still no
explosion. `rules_payload` is the whole of it, and each page takes it whole and adds only
what is its own (a still's hands and position, a game's seats, a replay's frames).

Every key is always present, ``None`` (or an empty list) where the game has no such rule,
so a reader tests a value and never a key. Compiled answers only: `engine/compile.py` is
the one place that means anything by the spec's spelling, and nothing here reads the DSL.
`tests/test_rules_payload.py` holds each page to every key.

`effect_pairs` is the one encoding of a position's effects (schema 30) as ``[[square,
bits], ...]``, for a history entry, a replayed position and a still.
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox.engine.moves import move_gate
from chess_variant_sandbox.spec.schema import PURSE_MAX
from chess_variant_sandbox.viz.heading import directional_types
from chess_variant_sandbox.viz.wrapped import (
    goal_payload,
    portal_payload,
    pounce_kinds_payload,
    walls_payload,
    wrap_payload,
    zone_payload,
)


def rules_payload(engine: Any) -> dict[str, Any]:
    """Every compiled rule a page draws, keyed as the pages' scripts read them."""
    spec = engine.spec
    drops = spec.drops
    return {
        "width": spec.width,
        "height": spec.height,
        # `"hex"` draws each cell as a hexagon on a sheared grid (schema 14).
        "lattice": spec.lattice,
        "pieceIds": list(spec.piece_ids),
        "pieceNames": list(spec.piece_names),
        # A whole army's worth of board values, the headings a facing piece is drawn at,
        # and which pieces may be turned to a heading at all.
        "nTypes": spec.n_types,
        "sideStride": spec.side_stride,
        "facingAngles": [list(row) for row in spec.facing_headings()],
        "directional": list(directional_types(spec)),
        "nFacings": spec.n_facings,
        # Squares nothing may occupy (walls alone) and the cells a hexagon lacks.
        **walls_payload(spec),
        # Generation 40 (N3c): each type's compiled pounce kind, or ``None`` where nothing
        # pounces: the phase's words tell a pounce on an attacker from one on prey by it.
        "pounceKinds": pounce_kinds_payload(spec.pounce),
        # Schema 18e (Q785): each hand slot's other face, or -1; ``None`` where a held
        # piece comes back as itself only.
        "dropFaces": drop_faces_payload(drops),
        # Q1059a: each player's gate squares, ``None`` where a held piece drops anywhere.
        "gates": gates_payload(drops),
        # Whether the game has hands at all, and whether they fill from captures.
        "handsUsed": drops is not None,
        "handsFromCaptures": bool(drops is not None and drops.from_captures),
        # Squares a game ends on when a piece reaches one, per player (``None`` without),
        # and whether it is decided by leaving one rather than standing on it (schema 18).
        "goals": goal_payload(spec.reach),
        "goalLeaves": bool(spec.reach and spec.reach.leaves),
        # Squares a piece promotes on, per player; ``None`` without promotion.
        "zones": zone_payload(spec.promotions),
        # The wrap rule, per player and per axis; ``None`` for a board with edges.
        "wrap": wrap_payload(spec.wrap, spec.copies),
        # Which piece types are royal, one flag per type.
        "royal": [t in spec.royal_types for t in range(spec.n_types)],
        # The blast, which types survive it and which captures fire it; ``None`` without.
        "explosion": explosion_payload(spec),
        # Schema 19: each portal end's pair, colour, movers and doors; ``None`` without.
        "portals": portal_payload(spec),
        # Schema 28 (Q943b): each type's speed; empty in a game without one.
        "speed": list(spec.simultaneous.speed) if spec.simultaneous is not None else [],
        # Schema 29 (Q983b): each type's rest; empty in a game without one.
        "rest": list(spec.rest),
        # Generation 42 (N4c): the duck's type, or ``None`` without `rules.duck`.
        "duck": spec.duck.piece_type if spec.duck is not None else None,
        # TC7: the purse's rule (`market_payload`); ``None`` without a purse.
        "purse": market_payload(engine),
        # What the colouring on the board is for, compiled: ``{mode, win_at}``, or ``None``
        # in a game that claims no ground.
        "territory": (
            {"mode": spec.territory.mode, "win_at": spec.territory.win_at}
            if spec.territory is not None else None
        ),
    }


def explosion_payload(spec: Any) -> dict[str, Any] | None:
    """The blast neighbourhood per square, a flag per type saying which types are immune,
    and which captures fire it (``by``, ``of``, schema 10; ``None`` when every capture
    explodes), or ``None`` without an explosion rule.

    ``by`` and ``of`` have to cross for the reason the blast does: **the page cannot work out
    from two positions which board change was which**, so a rule that narrows *when* the
    blast fires has to reach it or the animation lies.
    """
    ex = spec.explosion
    if ex is None:
        return None
    n_types = spec.n_types

    def selection(types: frozenset[int] | None) -> list[bool] | None:
        return None if types is None else [t in types for t in range(n_types)]

    return {
        "blast": [list(neighbours) for neighbours in ex.blast],
        "immune": [t in ex.immune_types for t in range(n_types)],
        "by": selection(ex.by_types),
        "of": selection(ex.of_types),
    }


def market_payload(engine: Any) -> dict[str, Any] | None:
    """TC7 (`docs/plan-turncoat.md` §5.8): the purse's rule as the page draws it, compiled
    (`PurseRule`), or ``None`` without a purse: the market (each type and its price, in the
    spec's order), the cap, and which types are for hire or bribe at what cost
    (``hire[type]``/``bribe[type]``, 0 where not). The page offers only the moves the engine
    lists; these are the words beside them."""
    purse = engine.spec.purse
    if purse is None:
        return None
    return {
        "start": purse.start,
        "cap": PURSE_MAX,
        "market": [
            {"type": int(piece_type), "price": int(price)}
            for piece_type, price in zip(purse.market, purse.price, strict=True)
        ],
        "hire": list(purse.hire),
        "bribe": list(purse.bribe),
    }


def gate_entry(engine: Any, move: int) -> int | None:
    """Q1059a: the type a gating move brings in from the hand, or ``None``.

    Read off the compiled rule as well as the word, since the tag is another field in a
    game without gating.
    """
    drops = engine.spec.drops
    gate = move_gate(move)
    if drops is None or drops.gates is None or not gate:
        return None
    return int(gate - 1)


def gates_payload(drops: Any) -> list[list[int]] | None:
    """Q1059a: each player's gate squares under ``drops.gating``, or ``None`` without it."""
    if drops is None or drops.gates is None:
        return None
    return [sorted(drops.gates[0]), sorted(drops.gates[1])]


def drop_faces_payload(drops: Any) -> list[int] | None:
    """Each hand slot's other face under ``drops.either_face`` (schema 18e, Q785), or ``None``.

    The compiled pairs, so the page offers a piece in hand as both faces without reading
    the spec's ``promoted_from``.
    """
    if drops is None or not drops.faces:
        return None
    return list(drops.faces)


def effect_pairs(engine: Any, state: Any) -> list[list[int]]:
    """Schema 30: the effects on ``state`` as ``[[square, bits], ...]``, only the squares whose
    piece is under one (1 frozen, 2 disarmed, 4 shielded), the core's answer
    (`Engine.effects`). Empty in a game without a sight."""
    if engine.spec.sight is None:
        return []
    return [[square, bits] for square, bits in enumerate(engine.effects(state).bits) if bits]
