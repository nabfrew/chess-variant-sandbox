"""Which pieces a game can put on the board, and which a player can hold.

Read off the spec. The validator asks the first question, to refuse a rule about a piece
that can never stand. `metrics.components` asks the second, to charge `piece_utilisation`
only for the types a player owns. So a change here can move a stored score, and this file
is in `measurement.METRICS_SOURCES`. It sits apart from `schema.py` so the rest of the
validator stays outside that fingerprint.
"""

from __future__ import annotations

from typing import Any


def _str_set(values: Any) -> set[str]:
    """The strings among ``values``: a malformed entry is refused by its own shape check,
    so the coherence checks that build a set of ids skip it instead of raising on it."""
    return {value for value in values if isinstance(value, str)}


def _in(value: Any, ids: Any) -> bool:
    """``value in ids`` for a value that may not be hashable: only a string can be an id."""
    return isinstance(value, str) and value in ids


def pieces_a_player_can_hold(spec: dict[str, Any]) -> set[str]:
    """Every piece id a player can own at some point in a game, by any route.

    The player-owned half of ``_pieces_that_can_stand``. A wild piece is not a seed, so a
    type only the wild phase places is absent unless a player route reaches it: placed
    for a player, stocked in a hand, a promotion or strike of a held piece, a contact
    that changes a held piece, or a capture that puts a wild piece into a hand.

    Over-approximates in the same direction as the walk it mirrors. A type that stands
    and cannot move is still held, which is the difference from "a type a player moves".
    """
    pieces = spec.get("pieces")
    pieces = pieces if isinstance(pieces, list) else []
    rules = spec.get("rules")
    rules = rules if isinstance(rules, dict) else {}
    declared = {
        piece["id"]
        for piece in pieces
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    }
    standing = _pieces_that_can_stand(spec, pieces, rules, declared)

    held: set[str] = set()
    setup = spec.get("setup")
    placements = setup.get("placements") if isinstance(setup, dict) else None
    for placement in placements if isinstance(placements, list) else []:
        if isinstance(placement, dict) and _in(placement.get("piece"), declared):
            held.add(placement["piece"])
    stock = (rules.get("drops") or {}).get("stock")
    if isinstance(stock, dict):
        held |= declared.intersection(stock)

    entries, links, contact_links, hands_fill, demoted = _type_changes(pieces, rules, declared)

    # A captured piece of any standing type, wild included, enters the captor's hand.
    if hands_fill:
        held |= {demoted.get(pid, pid) for pid in standing}

    while True:
        grown = set(held)
        for entry in entries:
            if entry.get("piece") in held:
                grown |= declared.intersection(_promotion_target_ids(entry))
        for link in links:
            if link.get("from") in held and link.get("to") in declared:
                grown.add(link["to"])
        for link in contact_links:
            if link.get("mover") not in standing or link.get("onto") not in standing:
                continue
            for side in ("mover", "onto"):
                result = link.get(f"{side}_to")
                if link.get(side) in held and isinstance(result, str) and result in declared:
                    grown.add(result)
        if grown == held:
            return held
        held = grown


def _pieces_that_can_stand(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], declared: set[str]
) -> set[str]:
    """Every piece id that can ever appear on the board, by any route.

    The routes are a closed set and ``engine.compile._royalty_can_multiply`` already
    enumerates them for one piece: placed at ply zero, a promotion target, a strike
    chain's ``to``, out of a hand, or a wild piece. Asked here of every piece and closed
    to a fixed point, because a promotion target can itself be what promotes, and what
    comes back out of a hand is what went into it.

    Over-approximates wherever the spec is unclear -- a piece admitted here only makes a
    refusal *less* likely, and the direction that matters is never refusing a piece that
    could turn up.
    """
    reachable: set[str] = set()
    setup = spec.get("setup")
    placements = setup.get("placements") if isinstance(setup, dict) else None
    for placement in placements if isinstance(placements, list) else []:
        if isinstance(placement, dict) and _in(placement.get("piece"), declared):
            reachable.add(placement["piece"])
    npcs = rules.get("npcs")
    for npc in npcs if isinstance(npcs, list) else []:
        if isinstance(npc, dict) and _in(npc.get("piece"), declared):
            reachable.add(npc["piece"])
    # Schema 10 (Q51): a stocked piece is in a hand from ply zero, so it is on the board
    # the moment somebody plays it. The sixth route, and the first one that does not need
    # the piece to have been somewhere else first.
    stock = (rules.get("drops") or {}).get("stock") if isinstance(rules, dict) else None
    if isinstance(stock, dict):
        reachable |= declared.intersection(stock)

    entries, links, contact_links, hands_fill, demoted = _type_changes(pieces, rules, declared)

    while True:
        grown = set(reachable)
        for entry in entries:
            if _in(entry.get("piece"), reachable):
                grown |= declared.intersection(_promotion_target_ids(entry))
        for link in links:
            if _in(link.get("from"), reachable) and _in(link.get("to"), declared):
                grown.add(link["to"])
        for link in contact_links:
            if not _in(link.get("mover"), reachable) or not _in(link.get("onto"), reachable):
                continue
            grown |= declared.intersection(
                {
                    link[role]
                    for role in ("mover_to", "onto_to")
                    if isinstance(link.get(role), str)
                }
            )
        if hands_fill:
            grown |= {demoted[pid] for pid in reachable if pid in demoted}
        if grown == reachable:
            return reachable
        reachable = grown


def _type_changes(
    pieces: list[Any], rules: dict[str, Any], declared: set[str]
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], bool, dict[str, str]
]:
    """The spec's type-changing rules, read once for both reachability walks.

    Promotion entries, strike chain links, contact links, whether captures fill hands,
    and what each piece reverts to in a hand.
    """
    entries = [entry for entry in _promotion_entries(rules) if isinstance(entry, dict)]
    strike = rules.get("strike")
    links: list[dict[str, Any]] = []
    if isinstance(strike, dict) and strike.get("effect", "capture") != "capture":
        immune = strike.get("immune")
        immune_ids = _str_set(immune) if isinstance(immune, list) else set()
        chain = strike.get("chain")
        links = [
            link
            for link in (chain if isinstance(chain, list) else [])
            if isinstance(link, dict) and not _in(link.get("from"), immune_ids)
        ]
    # The seventh route (schema 10, Q19a): a contact puts a piece on the board on both
    # of its columns, and a link fires only when *both* of its pieces can be standing
    # there to meet -- which is the one place the walks are keyed by a pair.
    contacts = rules.get("contacts")
    contact_links: list[dict[str, Any]] = []
    if isinstance(contacts, dict):
        immune = contacts.get("immune")
        immune_ids = _str_set(immune) if isinstance(immune, list) else set()
        contact_links = [
            link
            for link in (contacts["links"] if isinstance(contacts.get("links"), list) else [])
            if isinstance(link, dict) and not _in(link.get("onto"), immune_ids)
        ]
    drops = rules.get("drops")
    hands_fill = isinstance(drops, dict) and bool(drops) and drops.get("from_captures", True)
    # A captured piece enters the captor's hand as its `promoted_from` where it declares
    # one, so a hand is a route back for the *base* piece and not for a new one.
    demoted = {
        piece["id"]: piece["promoted_from"]
        for piece in pieces
        if isinstance(piece, dict)
        and isinstance(piece.get("id"), str)
        and piece.get("promoted_from") in declared
    }
    return entries, links, contact_links, bool(hands_fill), demoted


def _promotion_target_ids(entry: dict[str, Any]) -> set[str]:
    """Everything one promotion rule can turn a piece into, both spellings, both players.

    Defensive rather than ``promotion_targets``: this runs over specs already known to be
    malformed, because ``validate_spec`` collects errors instead of stopping.
    """
    found: set[str] = set()
    for key in ("to", "to_by_player"):
        block = entry.get(key)
        lists = block.values() if isinstance(block, dict) else [block]
        for targets in lists:
            if isinstance(targets, list):
                found |= {target for target in targets if isinstance(target, str)}
    return found


def _promotion_entries(rules: dict[str, Any]) -> list[Any]:
    """The promotion rules as a list, whichever of the two spellings the spec used."""
    promotion = rules.get("promotion")
    if isinstance(promotion, list):
        return promotion
    return [promotion] if promotion else []
