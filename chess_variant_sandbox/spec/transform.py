"""Mechanical edits to a rule spec: spec in, spec out.

These were written to generate the labelled mutant corpus -- take a game that works and
break it in a way whose direction is known in advance -- and they are the whole
mechanical-variation half of a search over rule specs. `metrics/mutants.py` uses them to
make negatives; `loop/operators.py` uses them to make offspring. One implementation,
because a transform that is safe in one of those places is safe in the other, and two
copies would drift.

Two properties every transform here holds to.

**Referential integrity.** Piece ids are cross-referenced from `setup.placements`,
`rules.promotion`, `castling`, `extinction` and `drops`, so removing a piece naively
leaves four rules pointing at nothing. A transform repairs what it breaks, or it is not
a transform, it is a way of generating invalid specs.

**Written against the spec's own vocabulary, never against chess.** "Remove the piece
with the most placements" works on any ruleset; "remove the pawn" works on one.
"""

from __future__ import annotations

import copy
from typing import Any

from chess_variant_sandbox.spec.schema import (
    STOCK_UNLIMITED,
    drop_face_pairs,
    strike_is_live,
    strike_states_nothing,
    strike_survives_anything,
    wrap_axes,
)


def pieces(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return spec.get("pieces", [])


def royal_id(spec: dict[str, Any]) -> str | None:
    for piece in pieces(spec):
        if piece.get("royal"):
            return str(piece["id"])
    return None


def most_numerous(spec: dict[str, Any]) -> str | None:
    """The id placed on the most squares -- the pawn, in anything shaped like chess."""
    counts: dict[str, int] = {}
    for placement in spec.get("setup", {}).get("placements", []):
        counts[placement["piece"]] = counts.get(placement["piece"], 0) + len(
            placement.get("squares", [])
        )
    royal = royal_id(spec)
    if royal is not None:
        counts.pop(royal, None)
    return max(counts, key=lambda k: counts[k]) if counts else None


def strongest(spec: dict[str, Any]) -> str | None:
    candidates = [p for p in pieces(spec) if not p.get("royal")]
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: float(p.get("value", 0)))["id"])


def drop_piece(spec: dict[str, Any], piece_id: str | None) -> dict[str, Any]:
    """Remove a piece type entirely: definition, placements and promotion targets."""
    if piece_id is None:
        return spec
    spec["pieces"] = [p for p in pieces(spec) if p["id"] != piece_id]
    setup = spec.setdefault("setup", {})
    setup["placements"] = [
        p for p in setup.get("placements", []) if p["piece"] != piece_id
    ]
    # Castling names its rook by id, so removing that piece leaves a rule pointing at
    # nothing. The validator catches it, which is the system working -- but a mutant
    # that fails to build is a mutant that tests nothing, so the reference goes too.
    castling = spec.get("rules", {}).get("castling")
    if isinstance(castling, dict) and piece_id in (castling.get("rook"), castling.get("king")):
        spec["rules"].pop("castling", None)

    # **Promotion is a list, and this read it only as an object.** Schema 2 made
    # `rules.promotion` one rule per promoting piece, so every ruleset written since --
    # which is every ruleset with promotion in the corpus, chess included -- took the
    # `isinstance(promotion, dict)` branch and skipped it. Removing a promotion target
    # therefore left `promotion.to` naming a piece that no longer existed, the validator
    # refused the result by name, and `mutate` quietly drew again.
    #
    # So `remove-a-piece` has been a **no-op on most of the corpus**, silently: an
    # operator whose every child is discarded looks exactly like an operator that had
    # nothing to do. Found by `chess_variant_sandbox/fuzz.py` -- not by the fuzzer failing, but by it
    # failing to find something it should have. A wall on a rank a king castles along
    # needs a back rank emptied first, and the only operator that can empty one could
    # not, so seventeen thousand cases never once produced the shape.
    rules = spec.setdefault("rules", {})
    promotion = rules.get("promotion")
    written_as_list = isinstance(promotion, list)
    entries = promotion if written_as_list else [promotion]
    kept: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("piece") == piece_id:
            continue
        targets = entry.get("to")
        if isinstance(targets, list):
            entry["to"] = [t for t in targets if t != piece_id]
            if not entry["to"]:
                continue
        # `to_by_player` is the same relation written per side, and a target removed from
        # one side has to go from both or the two armies promote into different pieces
        # for a reason nobody wrote down.
        by_player = entry.get("to_by_player")
        if isinstance(by_player, dict):
            for side, side_targets in list(by_player.items()):
                if isinstance(side_targets, list):
                    by_player[side] = [t for t in side_targets if t != piece_id]
            if not any(by_player.values()):
                continue
        kept.append(entry)

    if promotion is not None and not kept:
        rules.pop("promotion", None)
        rules.pop("en_passant", None)
    elif promotion is not None:
        rules["promotion"] = kept if written_as_list else kept[0]
        # En passant is defined against *the* promoting piece. If the one it was about
        # is gone, so is the rule; the validator says as much and would refuse the spec.
        if not any(entry.get("piece") for entry in kept):
            rules.pop("en_passant", None)

    _forget_piece(spec, piece_id)
    return spec


def _forget_piece(spec: dict[str, Any], piece_id: str) -> None:
    """Every *other* place a piece id can appear, taken out with the piece.

    The same bug the promotion list had above, six more times (Q85). ``drop_piece``
    cleaned the definition, the placements, castling and promotion, and
    ``loop.operators`` claimed on the strength of that -- *"removing a piece takes its
    placements and its rule references with it"* -- while a capture matrix, an icon, a
    strike chain, a drop region, a wild piece's route and an explosion's immunity all
    went on naming a piece that no longer existed. The validator refused each by name and
    `mutate` drew again in silence, so ``remove-a-piece`` was a **total loss on 25 of the
    53 committed and archived specs** -- antichess, atomic, shogi, xiangqi, every RPS
    version, every hierarchy game, `wolves` -- and the operator looked exactly like one
    that had nothing to do. It is also the only operator that can shrink an army, so the
    army axis of `loop.grid` was half unreachable for free on half the corpus.

    Written as an explicit cascade rather than a walk of every string in the spec,
    because a piece id is one or two characters and a blind walk would eat an icon name
    or a square. The cascade is held complete by measurement instead of by care:
    `tests/test_operators.py` drops **every** piece of **every** committed spec and
    requires the result to validate, so a schema bump that puts a piece id somewhere new
    fails there rather than in a design run.

    A container emptied by the removal takes its rule with it, on the argument
    ``castling`` already makes above: a rule that names nothing is either refused
    outright or is a charge for a rule that can never fire.
    """
    rules = spec.setdefault("rules", {})

    for piece in pieces(spec):
        captures = piece.get("captures")
        if isinstance(captures, list):
            piece["captures"] = [p for p in captures if p != piece_id]
        if piece.get("promoted_from") == piece_id:
            piece.pop("promoted_from", None)

    icons = spec.get("appearance", {}).get("icons")
    if isinstance(icons, dict):
        icons.pop(piece_id, None)

    def _prune_list(holder: dict[str, Any], key: str) -> None:
        """Take the id out of a list of piece ids, and the key with it if it empties."""
        value = holder.get(key)
        if not isinstance(value, list):
            return
        holder[key] = [p for p in value if p != piece_id]
        if not holder[key]:
            holder.pop(key)

    drops = rules.get("drops")
    if isinstance(drops, dict):
        for key in ("no_doubled", "no_mate"):
            _prune_list(drops, key)
        for key in ("region", "stock"):
            mapping = drops.get(key)
            if isinstance(mapping, dict):
                mapping.pop(piece_id, None)
                if not mapping:
                    drops.pop(key)
        # Schema 10 (Q51). A stock is one of the two things that can fill a hand, so
        # dropping the last stocked piece out of a hand that does not also fill from
        # captures leaves a rule the validator refuses by name -- *"nothing else fills a
        # hand, so there is never anything to drop"*. The `castling` argument again: a
        # rule left naming nothing goes with what it named.
        if not drops.get("from_captures", True) and not drops.get("stock"):
            rules.pop("drops", None)
        # Schema 18e (Q785). The last `promoted_from` pair gone leaves a flag that pairs
        # nothing, which the validator refuses by name, so the flag goes with the pair.
        elif drops.get("either_face") is True and not drop_face_pairs(pieces(spec)):
            drops.pop("either_face")

    reach = rules.get("reach")
    if isinstance(reach, dict) and reach.get("piece") == piece_id:
        rules.pop("reach", None)

    extinction = rules.get("extinction")
    if isinstance(extinction, dict) and isinstance(extinction.get("pieces"), list):
        extinction["pieces"] = [p for p in extinction["pieces"] if p != piece_id]
        if not extinction["pieces"]:
            rules.pop("extinction", None)
        elif extinction.get("per_type") is True and len(set(extinction["pieces"])) < 2:
            # Schema 18c: one type counted apart is the count together, which the
            # validator refuses as stating nothing, so the flag goes with the type.
            extinction.pop("per_type")

    npcs = rules.get("npcs")
    if isinstance(npcs, list):
        kept_npcs = [n for n in npcs if not (isinstance(n, dict) and n.get("piece") == piece_id)]
        for npc in kept_npcs:
            if isinstance(npc, dict):
                _prune_list(npc, "immune")
        if kept_npcs:
            rules["npcs"] = kept_npcs
        else:
            # A `scoring` rule counting points has nothing left to count, which is the
            # pair `clear_npcs` already keeps together.
            rules.pop("npcs", None)
            if isinstance(rules.get("scoring"), dict) and rules["scoring"].get("count") == "points":
                rules.pop("scoring", None)

    # Schema 19: a pair whose ends moved like the dropped piece goes with it, and the rule
    # with its last pair. Later pairs renumber, since a pair's number is its place.
    portals = rules.get("portals")
    if isinstance(portals, list):
        kept_pairs = [
            pair for pair in portals
            if not (isinstance(pair, dict) and pair.get("piece") == piece_id)
        ]
        if kept_pairs:
            rules["portals"] = kept_pairs
        else:
            rules.pop("portals", None)

    strike = rules.get("strike")
    if isinstance(strike, dict):
        _prune_list(strike, "immune")
        chain = strike.get("chain")
        if isinstance(chain, list):
            strike["chain"] = [
                link
                for link in chain
                if not (isinstance(link, dict) and piece_id in (link.get("from"), link.get("to")))
            ]
            # `effect` other than "capture" needs somewhere for the victim to go, so a
            # chain with no links left is not a weaker strike rule: it is no rule.
            if not strike["chain"]:
                rules.pop("strike", None)
        # Schema 17 (Q540): the pieces that change sides on a capture. An emptied list is
        # gone, and with it a `"capture"` strike that said nothing else.
        if "strike" in rules and "switches_sides" in strike:
            _prune_list(strike, "switches_sides")
            if not strike_is_live(strike):
                rules.pop("strike", None)

    contacts = rules.get("contacts")
    if isinstance(contacts, dict):
        # The strike's other spelling (schema 10, Q19a), and the cascade is coarser than
        # the chain's above: a link names up to four pieces and has to keep saying what
        # at least one of the two *becomes*, so a link mentioning the dropped piece
        # anywhere goes whole. Pruning the key alone would leave `{mover, onto}` with
        # neither `mover_to` nor `onto_to`, which the validator refuses by name as an
        # ordinary capture written the long way.
        _prune_list(contacts, "immune")
        links = contacts.get("links")
        if isinstance(links, list):
            contacts["links"] = [
                link
                for link in links
                if not (
                    isinstance(link, dict)
                    and piece_id
                    in (
                        link.get("mover"),
                        link.get("onto"),
                        link.get("mover_to"),
                        link.get("onto_to"),
                    )
                )
            ]
            if not contacts["links"]:
                rules.pop("contacts", None)  # a table with no link is no rule

    explode = rules.get("explode_on_capture")
    if isinstance(explode, dict):
        _prune_list(explode, "immune")
        # `by` and `of` are schema 10, each "the types this is about". Pruning the id out
        # and popping the key when the list empties (as `immune` does above) would leave
        # the key absent, which means "everyone" -- so removing the one piece a narrowed
        # blast named would *widen* it into unrestricted atomic chess rather than narrow
        # it further (Q435: `remove-a-piece` taking the queen from `archive/queen-bomb`).
        # A list that named someone and now names nobody is not a wider rule, it is no
        # rule, so the piece going takes the whole blast with it instead of just the key.
        drop_whole_rule = False
        for key in ("by", "of"):
            value = explode.get(key)
            if isinstance(value, list) and piece_id in value:
                narrowed = [p for p in value if p != piece_id]
                if not narrowed:
                    drop_whole_rule = True
                    break
                explode[key] = narrowed
        if drop_whole_rule:
            rules.pop("explode_on_capture", None)

    # A pruned chain or table may keep nobody alive, which `choose` refuses.
    settle_landing(spec)
    # And may then reach nobody at all, its every link left starting at (or landing on)
    # an immune piece: a rule the validator refuses as stating nothing (Q538), so it goes
    # with the last piece it could reach, the way an emptied chain goes above.
    remaining = [str(piece.get("id")) for piece in pieces(spec)]
    for key in ("strike", "contacts"):
        if strike_states_nothing(rules.get(key), key, remaining):
            rules.pop(key, None)


def set_moves_to(spec: dict[str, Any], source_id: str | None) -> dict[str, Any]:
    """Give every non-royal piece the moves of one of them."""
    source = next((p for p in pieces(spec) if p["id"] == source_id), None)
    if source is None:
        return spec
    for piece in pieces(spec):
        if not piece.get("royal"):
            piece["moves"] = copy.deepcopy(source["moves"])
            piece["value"] = source.get("value", 1)
    # Promotion into a piece that moves identically to the promoting one is no longer a
    # rule anyone would notice; removing it keeps the spec honest about what it contains.
    spec.get("rules", {}).pop("promotion", None)
    spec.get("rules", {}).pop("en_passant", None)
    return spec


def all_slides_to_leaps(spec: dict[str, Any]) -> dict[str, Any]:
    """Clip every sliding piece to a single step. Removes long-range power entirely.

    A ``sweep`` is clipped with the slides. It walks a ray as a slide does, and reach
    along the ray is what `probe-no-sliders` takes away. A one-step sweep has no squares
    between origin and landing, so nothing is ever swept and it is exactly a leap of its
    vectors. Mode, symmetry and condition carry over as they do for a slide.

    The pawn's opening double step is a slide, so on chess this takes en passant's
    subject away with it -- ``settle_en_passant`` drops the rule rather than leaving a
    mutant the validator refuses and `probe-no-sliders` therefore measuring nothing.
    """
    for piece in pieces(spec):
        for move in piece.get("moves", []):
            if move.get("kind") in ("slide", "sweep"):
                move["kind"] = "leap"
                move.pop("range", None)
                move.pop("min_range", None)
    return settle_en_passant(spec)


def omnipotent_piece(spec: dict[str, Any]) -> dict[str, Any]:
    """Give the strongest piece every direction at unlimited range, and a leap too.

    One piece that reaches everything collapses the game onto it: nothing else on the
    board is worth moving, and the position is decided by who trades it off.
    """
    target = strongest(spec)
    for piece in pieces(spec):
        if piece["id"] == target:
            piece["moves"] = [
                {"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"},
                {"kind": "leap", "vectors": [[1, 2], [1, 3], [2, 3]], "symmetry": "dihedral8", "mode": "both"},
            ]
            piece["value"] = 40
    return spec


def widen_royal(spec: dict[str, Any], steps: int = 6) -> dict[str, Any]:
    """Let the royal piece run. A king that outruns everything can never be mated."""
    royal = royal_id(spec)
    for piece in pieces(spec):
        if piece["id"] == royal:
            for move in piece.get("moves", []):
                move["kind"] = "slide"
                move["range"] = steps
    castling = spec.get("rules", {}).get("castling")
    if isinstance(castling, dict):
        spec["rules"].pop("castling", None)
    return spec


def resize_board(spec: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Change the board without moving the army, leaving the extra space empty.

    The lattice stays: a hex board resized is a hex board, and dropping the key would
    turn it into a square game with the same vectors. So does ``shape`` (Q281): dropped,
    a hexagon would become its whole rhombus with no walls, a plausible game and the wrong
    one. A size the hexagon cannot take is the validator's to refuse.

    **A promotion zone spelled as ranks moves with the far edge**, by the change in
    height, so it keeps its distance from where the second army stands when
    ``setup.mirror`` places it from the far edge. Every rule is moved, a list's as well as a lone
    one, and a ``ranks`` band as well as a ``rank``; a width change moves nothing, and
    ``"last"`` is written out as the number it now resolves to, as a lone rule's always
    was (the mutant corpus is built through here and the hash reads the spelling). Q102:
    this used to put a lone ``rank`` on the last rank whatever it was, even for a width
    change, and to skip the rest, which refused shogi's band on a shorter board and left a
    list's rank silently off one. A rank moved below zero is left for the validator.

    **Nothing spelled as squares moves**: a placement, a region, a reach goal, a wild
    route, schema 14's promotion ``squares``. A named square means that square, and
    nothing in the spec says whether a goal is the far edge (racing kings) or a place (a
    king of the hill's centre). A shrink that strands one is refused and ``mutate`` draws
    again -- ``loop/operators.py``, *invalid output is discarded, not repaired*.
    """
    board = spec.get("board", {})
    grown = height - int(board.get("height", 8))
    spec["board"] = {"width": width, "height": height}
    for key in ("lattice", "shape"):
        if board.get(key) is not None:
            spec["board"][key] = board[key]
    promotion = spec.get("rules", {}).get("promotion")
    for rule in promotion if isinstance(promotion, list) else [promotion]:
        if not isinstance(rule, dict):
            continue
        if rule.get("rank") == "last":
            rule["rank"] = height - 1
        elif _is_rank(rule.get("rank")):
            rule["rank"] += grown
        if isinstance(rule.get("ranks"), list):
            shifted = [entry + grown if _is_rank(entry) else entry for entry in rule["ranks"]]
            # A band reaching the near edge loses the ranks a shorter board does not have
            # (Q781: `einstein`'s zone is every rank), rather than naming rank -1.
            on_board = [entry for entry in shifted if not (_is_rank(entry) and entry < 0)]
            rule["ranks"] = on_board or shifted
    return spec


def _is_rank(value: Any) -> bool:
    """An integer the validator would read as a rank; ``bool`` is an ``int`` in Python."""
    return isinstance(value, int) and not isinstance(value, bool)


def promotion_rank(spec: dict[str, Any], rank: int) -> dict[str, Any]:
    promotion = spec.get("rules", {}).get("promotion")
    if isinstance(promotion, dict):
        promotion["rank"] = rank
    return spec


def promote_early(spec: dict[str, Any], steps: int = 2) -> dict[str, Any]:
    """Move the promotion rank to a few steps in front of where the pawns start.

    Written the hard way. The first version set the rank to 1, on the reasoning that a
    low rank means a short walk -- but the promoting piece *starts* on rank 1 and moves
    away from it, so the rank was simply unreachable and the mutant was chess with
    promotion silently switched off. It scored 0.804, inside the human band, and looked
    for all the world like a blind spot in the index until its per-piece move counts
    turned out to match chess almost exactly.

    The rank has to be derived from where the piece actually stands, which is what this
    does: whichever rank the promoting piece is placed on, plus a couple of steps.
    """
    promotion = spec.get("rules", {}).get("promotion")
    if not isinstance(promotion, dict):
        return spec
    target = promotion.get("piece")
    height = spec.get("board", {}).get("height", 8)

    ranks = [
        int(square[1:]) - 1
        for placement in spec.get("setup", {}).get("placements", [])
        if placement.get("piece") == target
        for square in placement.get("squares", [])
    ]
    start = max(ranks) if ranks else 1
    promotion["rank"] = min(start + steps, height - 1)
    return spec


def restrict_captures(
    spec: dict[str, Any], piece_id: str | None, allowed: list[str]
) -> dict[str, Any]:
    """Say which enemy types one piece may capture, or hand the permission back.

    ``allowed`` of ``None`` is not accepted, because the difference between "may take
    everything" and "may take nothing" is the whole feature and an ambiguous argument
    would eventually be passed the wrong one. Removing the restriction is
    ``free_captures``.

    A schema-3 edit, so it bumps the declared version. A spec that says 2 and carries a
    capture list is rejected by the validator, which is what the version is for.
    """
    if piece_id is None:
        return spec
    for piece in pieces(spec):
        if piece.get("id") == piece_id:
            piece["captures"] = list(allowed)
            spec["schema_version"] = max(int(spec.get("schema_version", 1)), 3)
            return spec
    return spec


def free_captures(spec: dict[str, Any], piece_id: str | None) -> dict[str, Any]:
    """Let one piece capture anything again.

    The declared version is left where it is: the spec may still hold other schema-3
    pieces, and lowering it on the strength of one edit would produce a spec that says 2
    and validates as 3.
    """
    if piece_id is None:
        return spec
    for piece in pieces(spec):
        if piece.get("id") == piece_id:
            piece.pop("captures", None)
            return spec
    return spec


def strip_rule(spec: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        spec.get("rules", {}).pop(key, None)
    return spec


def set_rule(spec: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    spec.setdefault("rules", {})[key] = value
    return spec


def settle_en_passant(spec: dict[str, Any]) -> dict[str, Any]:
    """Drop ``en_passant`` once the spec can no longer hold it. Two reasons.

    The dangling-reference repair ``drop_piece`` already makes for a promotion whose
    piece is gone, one level down: en passant answers a **two-square** move by the
    promoting piece, so an edit that shortens that move to one square, or turns it into
    a leap, takes the rule with it. Without this the child is refused by
    ``schema._check_inert_rules`` and ``mutate`` draws again in silence -- the class
    ``tests/test_operators.py::TestEveryOperatorOverTheWholeCorpus`` exists to catch,
    where an operator whose every child is thrown away looks exactly like one that had
    nothing to do.

    The second reason is a turn of more than one move, where the window a double step
    opens is still open on the mover's own next move. ``set_turn`` calls this, which is
    why ``_lengthen_the_turn`` still has something to do on the 45 library specs that
    declare en passant instead of being refused on every one of them.

    Asked through ``schema.can_double_step`` and ``schema.turn_is_long``, which are the
    validator's own readings, so a transform can never leave behind a spec the validator
    will refuse for either reason. Called by every edit that can *shorten* a move or
    *lengthen* a turn.
    """
    from chess_variant_sandbox.spec.schema import can_double_step, turn_is_long

    rules = spec.get("rules", {})
    if not rules.get("en_passant"):
        return spec
    if turn_is_long(rules):
        rules.pop("en_passant", None)
        return spec
    promotion = rules.get("promotion")
    entries = promotion if isinstance(promotion, list) else [promotion]
    if len(entries) != 1 or not isinstance(entries[0], dict):
        return spec  # no single promoting piece: a refusal of its own, not this one
    promoter = next(
        (piece for piece in pieces(spec) if piece.get("id") == entries[0].get("piece")),
        None,
    )
    if promoter is not None and not can_double_step(promoter):
        rules.pop("en_passant", None)
    return spec


def capture_is_possible(spec: dict[str, Any]) -> bool:
    """Whether any piece can ever take any other: the validator's own reading.

    ``schema._capture_is_possible`` is the premise of three inert-rule refusals
    (``must_capture``, ``explode_on_capture``, a hand that fills from captures), asked
    here of a whole spec because every caller holds one and none holds the piece-id set.
    An operator that would *add* one of those three rules asks this first, since in a game
    where nothing can be taken the rule it adds is refused on every draw.
    """
    from chess_variant_sandbox.spec.schema import _capture_is_possible

    listed = spec.get("pieces")
    if not isinstance(listed, list):
        return True
    return _capture_is_possible(listed, _declared(listed))


def pieces_that_can_stand(spec: dict[str, Any]) -> set[str]:
    """Every piece id that can ever appear on the board: the validator's own walk.

    ``reachability._pieces_that_can_stand``, which is what refuses a ``reach`` or a
    ``promotion`` about a piece that can never exist (inert-rule case 13).
    """
    from chess_variant_sandbox.spec.reachability import _pieces_that_can_stand

    listed = spec.get("pieces")
    rules = spec.get("rules")
    if not isinstance(listed, list) or not isinstance(rules, dict):
        return set()
    return _pieces_that_can_stand(spec, listed, rules, _declared(listed))


def wild_pieces_pay(spec: dict[str, Any]) -> bool:
    """Whether a ``scoring`` count of ``"points"`` would have anything to count.

    The validator's own reading, asked without parsing its English (Q436).
    ``schema._check_npc_coherence`` refuses ``points`` with no wild piece, or with none
    whose bounty is above zero, and none of its other refusals reads ``rules.scoring``.
    So it refuses more with a ``points`` count than without one exactly when the count
    has nothing to count. A wild piece's bounty defaults to its type's ``value``, and a
    type worth nothing (`antichess`' ``M``) pays nothing.
    """
    from chess_variant_sandbox.spec.schema import _check_npc_coherence

    listed = spec.get("pieces")
    rules = spec.get("rules")
    if not isinstance(listed, list) or not isinstance(rules, dict):
        return False
    counted: list[str] = []
    bare: list[str] = []
    _check_npc_coherence(spec, listed, {**rules, "scoring": {"count": "points"}}, counted)
    _check_npc_coherence(
        spec, listed, {key: value for key, value in rules.items() if key != "scoring"}, bare
    )
    return len(counted) == len(bare)


def _declared(listed: list[Any]) -> set[str]:
    """The declared piece ids, read the way ``schema._check_inert_rules`` reads them."""
    return {
        piece["id"]
        for piece in listed
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    }


def settle_inert_rules(spec: dict[str, Any]) -> list[str]:
    """Drop every rule ``schema._check_inert_rules`` would refuse, and say which went.

    ``settle_en_passant``'s repair, for all twelve clauses at once (Q110). An edit that
    takes away what a rule acts on -- the last capture a ``must_capture`` could insist
    on, the only route by which a promoting piece ever stands, the rook a castling needs
    on the king's rank -- leaves a rule the validator refuses as declaring nothing, and
    ``mutate`` then draws again in silence. The repair is the one ``drop_piece`` makes
    for castling and ``settle_en_passant`` for a double step: **the rule goes with what
    it named.** A rule that can never fire is either refused or charged for nothing, so
    taking it away changes no game.

    One pass rather than a ``settle_*`` per clause, and ``loop.operators.Operator.apply``
    runs it after every edit. Q110's audit found the pairs that reach these clauses
    spread over twelve operators and none of them reachable from the committed
    corpus in one step, so the corpus sweep -- the guard that caught every earlier case
    -- cannot see them. A call per operator would depend on each operator's author
    knowing which clauses their edit can reach, which is how every earlier case arose.

    Each arm asks the validator's own predicate under the validator's own guard, so on
    a spec the validator accepts this is a no-op, and it removes nothing the validator
    would not refuse. Run to a fixed point, because one removal can empty another: a
    promotion rule dropped takes a promotion *target* route with it, and an explosion
    dropped can leave ``check_limit`` beside ``forbid_check`` with nothing to count.

    **And three refusals outside that function (Q436)**, the same shape in other places:
    ``promote_on_entry`` that can never fire (``_check_entry_promotion_coherence``), a
    ``points`` count with no wild piece that pays, and a wild piece's ``immune`` list that
    takes it from nobody. Seven operators walked into the first, from parents one or two
    steps from the corpus. A fourth since Q538: a strike that states more than plain
    capture and compiles to it, or a contact table that does
    (``schema.strike_states_nothing``). ``drop_piece`` settles its own route to it, a
    chain pruned down to its immune links, and this holds any other.

    Returns the rules removed, as ``rules`` keys, ``"promotion:<id>"`` for one promotion
    rule out of several and ``"immune:<id>"`` for a wild type's immunity; empty when
    nothing moved.
    """
    listed = spec.get("pieces")
    rules = spec.get("rules")
    if not isinstance(listed, list) or not isinstance(rules, dict):
        return []
    gone: list[str] = []
    while True:
        dropped = _drop_inert(spec, listed, rules)
        if not dropped:
            return gone
        gone.extend(dropped)


def _drop_inert(spec: dict[str, Any], listed: list[Any], rules: dict[str, Any]) -> list[str]:
    """One round of ``settle_inert_rules``, in ``_check_inert_rules``' own order."""
    from chess_variant_sandbox.spec.reachability import (
        _pieces_that_can_stand,
        _promotion_entries,
    )
    from chess_variant_sandbox.spec.schema import (
        _capture_is_possible,
        _castling_ranks,
        _placed_ranks,
        npc_immune_attackers,
    )

    declared = _declared(listed)
    has_royal = any(isinstance(piece, dict) and piece.get("royal") for piece in listed)
    dropped: list[str] = []

    def drop(key: str) -> None:
        rules.pop(key, None)
        dropped.append(key)

    # Cases 3, 4 and 5: nothing can ever be taken.
    if not _capture_is_possible(listed, declared):
        for key in ("must_capture", "explode_on_capture"):
            if rules.get(key):
                drop(key)
        drops = rules.get("drops")
        if (
            isinstance(drops, dict)
            and drops
            and drops.get("from_captures", True)
            and not drops.get("stock")
        ):
            drop("drops")
    # Case 6: a hand that neither captures nor a stock can fill.
    drops = rules.get("drops")
    if (
        isinstance(drops, dict)
        and drops
        and not drops.get("from_captures", True)
        and not drops.get("stock")
    ):
        drop("drops")
    # Cases 7 to 10: a rule about a royal piece, in a game with none.
    if not has_royal:
        for key in ("bare_royal", "check_limit", "forbid_check"):
            if rules.get(key):
                drop(key)
        if rules.get("royal_facing") == "forbidden":
            drop("royal_facing")
    # Case 11: a check count nothing may advance.
    if (
        rules.get("check_limit")
        and rules.get("forbid_check")
        and not rules.get("explode_on_capture")
        and has_royal
    ):
        drop("check_limit")
    # Case 1: en passant with no double step to answer (and Q341's long turn).
    if rules.get("en_passant"):
        settle_en_passant(spec)
        if not rules.get("en_passant"):
            dropped.append("en_passant")
    # Case 12: castling with no unmoved rook on the king's rank.
    castling = rules.get("castling")
    board = spec.get("board")
    raw_setup = spec.get("setup")
    if (
        isinstance(castling, dict)
        and isinstance(raw_setup, dict)
        and isinstance(raw_setup.get("placements"), list)
        and isinstance(board, dict)
        and isinstance(board.get("width"), int)
        and isinstance(board.get("height"), int)
        and castling.get("king") in declared
        and castling.get("rook") in declared
    ):
        width, height = board["width"], board["height"]
        orientation = raw_setup.get("orientation", "mirror")
        king_ranks = set(_castling_ranks(rules, raw_setup, width, height, orientation))
        rook_ranks = _placed_ranks(raw_setup, castling["rook"], width, height, orientation)
        if not king_ranks & rook_ranks:
            drop("castling")
    # Case 13: a rule about a piece that can never stand on the board.
    standing = _pieces_that_can_stand(spec, listed, rules, declared)
    reach = rules.get("reach")
    if (
        isinstance(reach, dict)
        and reach.get("piece") in declared
        and reach.get("piece") not in standing
    ):
        drop("reach")
    entries = _promotion_entries(rules)
    unborn = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("piece") in declared
        and entry.get("piece") not in standing
    ]
    if unborn:
        dropped.extend(f"promotion:{entry['piece']}" for entry in unborn)
        kept = [entry for entry in entries if not any(entry is gone for gone in unborn)]
        if kept:
            rules["promotion"] = kept if isinstance(rules.get("promotion"), list) else kept[0]
        else:
            # The `drop_piece` repair: a rule defined against the promotion rule cannot
            # outlive it, and the validator refuses each by name.
            rules.pop("promotion", None)
            for key in ("en_passant", "promote_on_entry"):
                if rules.get(key):
                    drop(key)

    # Outside `_check_inert_rules` (Q436): the validator's other *can never fire*
    # refusals that an operator can walk into, each asked through the validator's own
    # function. `docs/reference/bugs-investigations.md`, "Q436", has the sweep.
    #
    # `promote_on_entry`. Every refusal `_check_entry_promotion_coherence` makes is about
    # the flag -- nothing promotes, nothing enters play except by moving, or no promoting
    # piece can enter its zone -- so any refusal from it means the flag goes. The last
    # includes a hand no promoting piece ever reaches (Q483: a stone-only hand), and this
    # arm follows it with no change here because it asks the function rather than a copy.
    if rules.get("promote_on_entry") is True:
        from chess_variant_sandbox.spec.schema import _check_entry_promotion_coherence

        refused: list[str] = []
        _check_entry_promotion_coherence(spec, rules, refused)
        if refused:
            drop("promote_on_entry")
    # A `points` count with no wild piece that pays: both players score nothing, so the
    # clock is a draw either way.
    scoring = rules.get("scoring")
    if isinstance(scoring, dict) and scoring.get("count") == "points" and not wild_pieces_pay(spec):
        drop("scoring")
    # A strike that states more than plain capture and is played as it (Q538,
    # `schema.strike_states_nothing`). `drop_piece` settles the one route Q538 found, a
    # chain pruned down to its immune links; this holds any other edit that empties a
    # strike, and the rule goes with the survivor it named. Under
    # the validator's own guard, a strike whose shape is sound: a malformed one is refused
    # for that instead, and is not this arm's to remove.
    # The contact table is the strike's other spelling and settles the same way.
    # Schema 18 (Q781) first: a keeping link whose mover has lost its promotion (an edit
    # took the promotion, or every target of it) is refused as changing nothing, and goes
    # here; a table it empties is no rule and goes with it.
    if settle_keeping_links(spec):
        if (rules.get("contacts") or {}).get("links"):
            dropped.append("keeping")
        else:
            drop("contacts")
    from chess_variant_sandbox.spec.schema import (
        _validate_contacts_shape,
        _validate_strike_shape,
        promoting_piece_ids,
    )

    piece_ids = [
        piece["id"]
        for piece in listed
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    ]
    for key, shape_of in (("strike", _validate_strike_shape), ("contacts", _validate_contacts_shape)):
        rule = rules.get(key)
        shape: list[str] = []
        if key == "contacts":
            _validate_contacts_shape(rule, piece_ids, shape, promoting_piece_ids(rules))
        else:
            shape_of(rule, piece_ids, shape)
        if not shape and strike_states_nothing(rule, key, piece_ids):
            drop(key)
    # A wild piece's `immune` list that takes it from nobody (`_validate_npc_immune`):
    # every type not named is wild, so the key does nothing. Not asked of a list naming
    # an undeclared id, which the validator refuses for that instead.
    for npc in rules.get("npcs") or ():
        immune = npc.get("immune") if isinstance(npc, dict) else None
        if (
            isinstance(immune, list)
            and all(pid in declared for pid in immune)
            and not npc_immune_attackers(rules, declared, npc)
        ):
            npc.pop("immune")
            key = f"immune:{npc.get('piece')}"
            if key not in dropped:
                dropped.append(key)
    return dropped


def no_captures(spec: dict[str, Any]) -> dict[str, Any]:
    """Make every move non-capturing. Nothing can ever be resolved."""
    for piece in pieces(spec):
        piece["moves"] = [m for m in piece.get("moves", []) if m.get("mode") != "capture"]
        for move in piece["moves"]:
            move["mode"] = "move"
    return strip_rule(spec, "en_passant")


def settle_keeping_links(spec: dict[str, Any]) -> bool:
    """Drop every keeping link whose mover has no promotion left to override (Q781).

    Schema 18's keeping link (``mover_to`` equal to ``mover``) states a rule only beside a
    promotion of the mover's own (``schema.keeping_link_holds``); without one it is the
    plain capture written the long way, and refused as such. An edit that takes the
    promotion away (``never_forced``, or ``drop_piece`` removing its every target) takes
    the links with it, as castling goes with its rook. True when anything went. A table
    left with no link is the caller's to remove.
    """
    from chess_variant_sandbox.spec.schema import (
        is_keeping_link,
        keeping_link_holds,
        promoting_piece_ids,
    )

    rules = spec.get("rules")
    contacts = rules.get("contacts") if isinstance(rules, dict) else None
    links = contacts.get("links") if isinstance(contacts, dict) else None
    if not isinstance(contacts, dict) or not isinstance(links, list):
        return False
    promoting = promoting_piece_ids(rules)
    kept = [
        link for link in links
        if not is_keeping_link(link) or keeping_link_holds(link, promoting)
    ]
    if len(kept) == len(links):
        return False
    contacts["links"] = kept
    return True


def never_forced(spec: dict[str, Any]) -> dict[str, Any]:
    """Remove every reason a game has to make progress.

    Promotion goes, and the no-progress clock is pushed out to the ply limit so it can
    never fire first. What is left is a game that can shuffle for as long as it likes
    and then be declared drawn -- which is the failure a designer named as "the game is
    just going back and forth and back and forth", and the one
    ``progress_irreversibility`` was built to see.

    **The two clocks have different ceilings, and this used to ignore one of them**
    (Q56, Q85). ``no_progress_draw`` stops at 1,000 and ``max_plies`` runs to 10,000, so
    a spec with a long ply limit got a no-progress clock the validator refused by name
    and the mutant was thrown away in silence -- the shape ``drop_piece``'s promotion-list
    bug had, where an operator whose every child is rejected looks exactly like one that
    had nothing to do. No committed ruleset declares ``max_plies`` above 1,000, which is
    what kept it a trap rather than a live loss.

    Both clocks come down together, because clamping only the no-progress one would leave
    it firing *first* on exactly the specs the clamp was for, and a mutant that quietly
    stops being the mutant it is named after is worse than one that is refused. Equal is
    the value this has always written and is deliberately kept: it is legal, it is one
    integer from any ``>=`` refusal anyone might add, and at the tie the engine reads the
    no-progress arm first -- so the record's ``reason`` stays what it has always been.
    """
    from chess_variant_sandbox.spec.schema import CLOCK_BOUNDS

    rules = spec.setdefault("rules", {})
    rules.pop("promotion", None)
    # En passant is defined against the promoting piece, so it cannot outlive promotion
    # -- the validator says so by name and it is right to. Nor can promotion on entry
    # (`archive/shogi-drop-promoted`, Q787, is the first library spec to carry it).
    rules.pop("en_passant", None)
    rules.pop("promote_on_entry", None)
    # Nor can a keeping link (schema 18, Q781): it overrides a promotion, and none is left.
    if settle_keeping_links(spec) and not (rules.get("contacts") or {}).get("links"):
        rules.pop("contacts", None)
    low, high = CLOCK_BOUNDS["no_progress_draw"]
    limit = min(high, max(low, int(rules.get("max_plies", 400))))
    rules["no_progress_draw"] = limit
    if int(rules.get("max_plies", 400)) > limit:
        rules["max_plies"] = limit
    return spec


def captures_come_back(spec: dict[str, Any]) -> dict[str, Any]:
    """Every captured piece returns to its captor's hand, so nothing is ever removed.

    A material lead becomes a loan. It is a real mechanic -- crazyhouse and shogi both
    do it and both are good games -- so what makes this a *mutant* is doing it to an
    army and a board designed around permanent capture, where the effect is that no
    advantage can be banked and the game runs to its clock.
    """
    spec.setdefault("rules", {})["drops"] = {"from_captures": True}
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 2)
    return spec


def confine_all_but(spec: dict[str, Any], *mobile: str | None) -> dict[str, Any]:
    """Pin every piece but the named ones to the squares it starts on.

    A region a piece may never leave is a schema-2 feature meant for palaces and river
    banks; pointed at a piece's own opening squares it produces an army that is present,
    counted, and incapable of doing anything. The failure a designer named as "these
    five pieces are just sitting in the back", taken to its limit.

    At least one piece has to stay mobile or there is no game to measure -- confining
    *everything* produced a position with no legal opening move, which the smoke gate
    rejects for the wrong reason and which says nothing about an idle army.
    """
    keep = {name for name in mobile if name}
    placements: dict[str, list[str]] = {}
    for placement in spec.get("setup", {}).get("placements", []):
        placements.setdefault(str(placement["piece"]), []).extend(placement["squares"])
    for piece in pieces(spec):
        pid = str(piece.get("id"))
        if pid in keep or pid not in placements:
            continue
        piece["region"] = sorted(set(placements[pid]))
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 2)
    return spec


def keep_only(spec: dict[str, Any], keep: set[str]) -> dict[str, Any]:
    for piece in list(pieces(spec)):
        if piece["id"] not in keep:
            drop_piece(spec, str(piece["id"]))
    return spec


def rename_everything(spec: dict[str, Any]) -> dict[str, Any]:
    """Rename every piece. Changes the spec and its hash; changes no game at all."""
    for index, piece in enumerate(pieces(spec)):
        piece["name"] = f"piece {index + 1}"
    return spec


def reorder_pieces(spec: dict[str, Any]) -> dict[str, Any]:
    """Declare the same pieces in the opposite order.

    Piece order fixes the type indices, which fix the board encoding, the Zobrist keys
    and the order moves come out of the generator -- so a reordered spec is the same game
    to a player and a different object to the engine.

    What it actually produced, measured: 33 of 144 games came out with different move
    sequences, and every raw component was identical to twelve decimal places. The games
    that moved were all random-vs-random, which the measurements exclude by design, and
    they kept their length and result. So this verifies something real and slightly
    narrower than intended -- that relabelling and re-indexing a spec cannot reach the
    index at all -- rather than exercising its numerical stability under genuinely
    different competent play. Getting that to move would mean changing something strong
    players respond to, which is the definition of changing the game. Recorded here
    because a null stability of exactly 0.000 should be read for what it is.
    """
    spec["pieces"] = list(reversed(pieces(spec)))
    return spec


def reverse_backrank(spec: dict[str, Any]) -> dict[str, Any]:
    """Mirror the non-royal back rank left-to-right.

    A null by construction on a symmetric setup: the same army on the same rank, laid
    out the other way round. If the index moves on this it is reading the seed, not the
    game.
    """
    setup = spec.get("setup", {})
    width = spec.get("board", {}).get("width", 8)
    files = "abcdefghijklmnop"
    for placement in setup.get("placements", []):
        flipped = []
        for square in placement.get("squares", []):
            file_index = files.index(square[0])
            flipped.append(f"{files[width - 1 - file_index]}{square[1:]}")
        placement["squares"] = sorted(flipped)
    spec.get("rules", {}).pop("castling", None)
    return spec


# --------------------------------------------------------- schema 3 and 4 endings
#
# Stage 13 made the argument for `restrict_captures` and acted on it: "no amount of
# changing how a piece *moves* changes what it may *take*, so without it the free
# operator cannot enter the region the model can now propose into". Then `rules.scoring`,
# `rules.territory` and `rules.turn` arrived with no mechanical edit that could add any
# of them, and two of the grid's six `_ending` bins became reachable only by a model
# call -- which is the *expensive* operator. These are the three that close that.
#
# What each edit *picks* is the part that is not mechanical, and every choice below is a
# rule with a reason rather than a number somebody liked. An operator has no designer:
# `territory.win_at` is 41 on a 9x9 board because a person chose a majority, and the
# stand-in for that person has to be something statable.


def settle_schema_version(spec: dict[str, Any]) -> dict[str, Any]:
    """Stamp the spec with the lowest version it actually validates under, either way.

    An edit that adds a schema-4 rule raises the stamp, and removing the rule again used
    to leave it raised -- so the two halves of an add-or-remove operator could not return
    a candidate to its parent. `schema_version` is part of the mechanics hash, so what
    came back was a *new* hash for an identical game: a tournament spent re-measuring
    something the archive already held, which is precisely the waste
    `test_a_known_game_is_not_rediscovered` exists to prevent.

    **Raising is the other half of the same argument**, and it was missing until the
    designer needed it. The stamp is a claim about which features a spec uses, so a spec
    that declares 1 and uses a schema-6 rule is not a spec with a low version: it is a
    spec whose stamp is wrong, and refusing it tells a person to hand-edit a number they
    should never have had to type. The lowest version a spec validates under is a
    *function of the spec*, so applying it always is what makes two people who built the
    same game get one hash rather than two.

    Asked of the validator rather than answered from a list of features. A hand-kept
    "which rules need which version" table here would be a second copy of the one thing
    `spec/schema.py` is for, and it would go stale on the next bump without failing.

    A spec that validates under no version at all is left exactly as it was: it is
    invalid for some reason that is not the stamp, and moving the number would only
    change which complaint the validator makes.

    Deliberately not applied by `free_captures`, which documents the opposite choice for
    a reason that still holds: it drops the restriction from *one* piece and the spec may
    carry others, so it cannot say anything about the version on its own. This can,
    because it asks about the whole spec.
    """
    from chess_variant_sandbox.spec.schema import SUPPORTED_SCHEMA_VERSIONS, validate_spec

    for version in SUPPORTED_SCHEMA_VERSIONS:
        if not validate_spec({**spec, "schema_version": version}):
            spec["schema_version"] = version
            return spec
    return spec


def set_scoring(spec: dict[str, Any], count: str | None = None) -> dict[str, Any]:
    """Decide the game on a count when the clock runs out, instead of drawing.

    ``count`` defaults to the one countable this spec can actually supply:

    * ``territory`` if the spec already colours squares -- free, and exactly the shape
      version 5 of the RPS sequence took;
    * ``points`` if the spec already has wild pieces to take (schema 9), for the same
      reason: the countable is already there and nothing else reads it. Only when one
      of them pays (`wild_pieces_pay`): a wild piece of a type worth nothing scores
      nobody anything, and the validator refuses the count (Q436);
    * ``pieces`` if every piece is declared at the same value, where "material" and
      "how many pieces" are the same measurement and the second says so honestly;
    * ``material`` otherwise, which is the finer of the two.

    ``region`` is deliberately never chosen. It needs a named set of squares, and an
    operator inventing a region is inventing the interesting half of the rule.
    """
    rules = spec.setdefault("rules", {})
    if count is None:
        if rules.get("territory"):
            count = "territory"
        elif rules.get("npcs") and wild_pieces_pay(spec):
            count = "points"
        else:
            values = {
                piece.get("value") for piece in pieces(spec) if not piece.get("royal")
            }
            count = "pieces" if len(values) <= 1 else "material"
    rules["scoring"] = {"count": count}
    spec["schema_version"] = max(
        int(spec.get("schema_version", 1)), 9 if count == "points" else 3
    )
    return spec


def set_territory(
    spec: dict[str, Any], mode: str = "permanent", win_at: int | None = None
) -> dict[str, Any]:
    """Make moving onto a square claim it, and enough claimed squares win.

    ``win_at`` defaults to a **bare majority of the board**, and the reason is that a
    majority is the only threshold both players cannot hold at once -- so the rule is
    decisive by construction rather than by a number that happened to work. It is also
    what the RPS designer chose: 41 of 81.

    Bounded at both ends. Never more than a majority, because a threshold the board
    cannot supply is a win condition nobody reaches and a generation spent on a
    candidate the gate throws out; and never fewer than three, because a board small
    enough for a majority to be one or two squares would be won on the opening move.

    Reachability is a fair question for ``repaint``, where a claim can be taken back and
    progress is not monotone. It is left legal: `rps-v4` is exactly that game and its
    designer kept playing it, and "this particular army cannot get there" is a question
    the cheap gate already asks better than a fraction can.
    """
    board = spec.get("board", {})
    squares = int(board.get("width", 8)) * int(board.get("height", 8))
    if win_at is None:
        win_at = max(3, min(squares // 2 + 1, squares))
    rules = spec.setdefault("rules", {})
    rules["territory"] = {"mode": mode, "win_at": win_at}
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 4)
    return spec


def set_turn(spec: dict[str, Any], moves: int = 2) -> dict[str, Any]:
    """Give a turn more than one move, with the opening ramp that keeps it fair.

    The ramp is not optional here, and that is the whole decision. Stage 16 measured
    both halves on gardner self-play: one move per turn leaves the first seat at 0.569,
    two moves per turn takes it to **0.931**, and the designer's 1-2-2-2 ramp puts it
    back to 0.597. `first_player_advantage` is a scored component, so an operator that
    emitted the unramped version would be manufacturing candidates the index is
    guaranteed to mark down -- a generation spent to rediscover a measurement already in
    the archive.

    Two moves rather than up to four for the same reason: the ramp that levels a
    two-move turn is the one that was measured. A longer turn wants its own ramp and
    nobody has measured one, and guessing at it here would be inventing data -- which is
    the same rule that keeps `screen` and `turn` out of `rulesets/`.

    `settle_en_passant` takes en passant with it, because a player who moves twice can
    answer their own double step.
    """
    rules = spec.setdefault("rules", {})
    rules["turn"] = {"moves": moves, "ramp": [1]}
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 4)
    return settle_en_passant(spec)


def set_wrap(spec: dict[str, Any], axis: str = "files") -> dict[str, Any]:
    """Join the board's opposite edges on one axis.

    Files rather than ranks by default, and the reason is measured rather than
    aesthetic: nothing shields a back rank from the seam side, so joining the *ranks* of
    an ordinary chess-shaped setup puts the two players' rooks on each other on move
    one. Joining the files does not, because a rook's own rank is already blocked by its
    neighbours. An operator that reached for the degenerate half by default would be
    manufacturing candidates the index is guaranteed to reject.

    Both edges rather than one. The per-player, per-edge spelling exists for a designer
    who means it; picking one of four asymmetric arrangements at random is not a local
    move in spec space, it is a different game with the same name.

    A hexagon joins a pair of sides rather than an axis (Q281): ``n_s``, ``ne_sw`` or
    ``nw_se``. Each shape's keys come from `schema.wrap_axes`, and the other shape's are
    refused here rather than written for the validator to refuse, because `files` on a
    hexagon is not a game at all but a caller asking for the wrong board.
    """
    shape = (spec.get("board") or {}).get("shape")
    keys = [key for key, _ in wrap_axes(shape)]
    if axis not in keys:
        raise ValueError(
            f"wrap axis {axis!r} is not one this board joins; it takes {', '.join(keys)}"
        )
    rules = spec.setdefault("rules", {})
    rules["wrap"] = {axis: "both"}
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 5)
    return spec


def set_obstacle(spec: dict[str, Any], square: str) -> dict[str, Any]:
    """Wall one square off, keeping any wall already there.

    One square at a time, and mirrored, which is the whole of the decision. A wall is
    the cheapest large change in the set -- it takes nothing away and teaches one
    sentence -- but a *block* of walls chosen at random is not a local move in spec
    space, it is a different board with the same name. Mirrored because an unmirrored
    wall hands one player an advantage the index would mark down as a first-seat
    asymmetry, which would be manufacturing candidates that are guaranteed to score
    badly for a reason that is not about the rule.
    """
    rules = spec.setdefault("rules", {})
    obstacles = rules.setdefault("obstacles", {"squares": [], "mirror": True})
    if square not in obstacles["squares"]:
        obstacles["squares"].append(square)
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 6)
    # A wall on castling's rank beyond every rook is schema 18's walled rank (Q781).
    from chess_variant_sandbox.spec.schema import castling_rank_walled_beyond_its_spans

    if castling_rank_walled_beyond_its_spans(spec):
        spec["schema_version"] = max(int(spec["schema_version"]), 18)
    return spec


def set_facing(
    spec: dict[str, Any], piece_id: str, axes: int = 4
) -> dict[str, Any]:
    """Give one piece an orientation it may spend a move turning.

    Named rather than chosen here, because the *choice* is the operator's and the
    constraint is real: a piece whose moves are symmetric under the turn would gain a
    move that does nothing, and the compiler refuses that outright. This function does
    not check it -- validation does, by name -- so a caller that picks badly gets a
    rejected candidate rather than a silently wrong game.
    """
    for piece in spec.get("pieces", []):
        if isinstance(piece, dict) and piece.get("id") == piece_id:
            piece["facing"] = {"axes": axes}
            spec["schema_version"] = max(int(spec.get("schema_version", 1)), 8)
            break
    return spec


def clear_facing(spec: dict[str, Any], piece_id: str) -> dict[str, Any]:
    """Take an orientation back off one piece, and off the squares it starts on.

    The placement is the other half: `setup.placements[].facing` says which way a copy
    points at the opening, and the validator refuses one *"set but the piece has no
    facing to start with"*. Leaving it behind made `turn-to-face` a total loss on both
    committed specs that use the feature -- `chess-oriented-pawns` and
    `archive/weathervane`, every draw refused -- which is the `drop_piece` shape again:
    an operator whose every child is thrown away looks like one that had nothing to do.
    """
    for piece in spec.get("pieces", []):
        if isinstance(piece, dict) and piece.get("id") == piece_id:
            piece.pop("facing", None)
            break
    for placement in spec.get("setup", {}).get("placements", []):
        if isinstance(placement, dict) and placement.get("piece") == piece_id:
            placement.pop("facing", None)
    return spec


def set_npc(
    spec: dict[str, Any],
    piece_id: str,
    route: list[str],
    bounty: float | None = None,
    mirror: bool = True,
    immune: list[str] | None = None,
) -> dict[str, Any]:
    """Loose one wild piece of a declared type on a route (schema 9).

    ``immune`` is schema 10: the only types that may take it, and it is the difference
    between a wild piece the army clears away and one the army has to be arranged around.
    Passing it raises the stamp to 10, because the stamp is a claim about which features
    a spec uses.

    Mirrored by default for the reason `set_obstacle` mirrors: a route has a direction,
    so a single wild piece is an asymmetry the index would mark down as a first-seat
    effect, and an operator manufacturing candidates that are guaranteed to score badly
    for a reason that is not about the rule is manufacturing noise. The bounty is left to
    default to the type's value unless a caller says otherwise, so this adds no number of
    its own. Whether the route is free, off every wall and off every other route is the
    validator's to refuse -- by name -- not this function's to check.
    """
    rules = spec.setdefault("rules", {})
    entry: dict[str, Any] = {"piece": piece_id, "route": list(route)}
    if bounty is not None:
        entry["bounty"] = bounty
    if mirror:
        entry["mirror"] = True
    if immune is not None:
        entry["immune"] = list(immune)
    rules.setdefault("npcs", []).append(entry)
    spec["schema_version"] = max(
        int(spec.get("schema_version", 1)), 10 if immune is not None else 9
    )
    return spec


def clear_npcs(spec: dict[str, Any]) -> dict[str, Any]:
    """Take every wild piece away, and the scoring rule that counted them if there is one.

    A `scoring` rule counting points is a rule with nothing to count once the wild
    pieces are gone, and the validator refuses it by name -- so an operator removing one
    without the other would spend a draw on a candidate that could never be played.
    That is the `set_territory` shape exactly.
    """
    rules = spec.get("rules", {})
    rules.pop("npcs", None)
    if isinstance(rules.get("scoring"), dict) and rules["scoring"].get("count") == "points":
        rules.pop("scoring")
    return spec


def set_promote_on_entry(spec: dict[str, Any]) -> dict[str, Any]:
    """A piece that enters play inside its promotion zone promotes at once (schema 9).

    One boolean, so this adds no number and no choice of its own: what the piece becomes
    is the first target its promotion rule already lists. Whether anything can ever enter
    a zone -- a hand a promoting piece reaches and a drop region that reaches the zone, a
    strike chain that leaves a promoting piece standing -- is the validator's to refuse,
    by name, not this function's to check.
    """
    spec.setdefault("rules", {})["promote_on_entry"] = True
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 9)
    return spec


def set_strike(
    spec: dict[str, Any], effect: str = "demote", position: str = "stay"
) -> dict[str, Any]:
    """Turn capture into transformation, with a chain read off the declared values.

    The chain is derived rather than invented: sort the non-royal pieces by the value
    the spec already declares and point each at the next one down, so the bottom piece
    has no link and is captured outright. That is the one chain a ruleset can be said
    to have asked for, and it means the operator adds no numbers of its own -- the same
    rule that keeps a guessed turn ``ramp`` out of ``set_turn``.

    ``stay`` rather than ``swap`` by default because it is the smaller change: the
    pieces end up where they would have under an ordinary capture minus the attacker's
    advance, whereas a swap teleports a defender behind enemy lines and is a different
    game rather than a modified one.
    """
    ranked = sorted(
        (
            piece
            for piece in spec.get("pieces", [])
            if isinstance(piece, dict) and not piece.get("royal")
        ),
        key=lambda piece: float(piece.get("value", 1)),
        reverse=effect == "demote",
    )
    chain = [
        {"from": ranked[index]["id"], "to": ranked[index + 1]["id"]}
        for index in range(len(ranked) - 1)
    ]
    if not chain:
        return spec
    rules = spec.setdefault("rules", {})
    rules["strike"] = {
        "effect": effect,
        "position": position,
        "chain": chain,
        "immune": [
            piece["id"]
            for piece in spec.get("pieces", [])
            if isinstance(piece, dict) and piece.get("royal")
        ],
    }
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 6)
    return spec


def set_stone(spec: dict[str, Any], piece_id: str, name: str = "stone") -> dict[str, Any]:
    """A piece with no moves, and the hand that is the only way it reaches the board.

    Schema 10 (Q51) in one call, because its two halves are one fact: a moveless piece is
    refused unless ``rules.drops.stock`` puts it in a hand, and a stock is what lets a
    hand hold something that was never captured. Writing either alone is an invalid spec,
    so they are not two transforms.

    **An unlimited supply rather than a count**, and that is the choice this makes rather
    than a number it invents: a stone a player may always place is not a large number of
    stones, `STOCK_UNLIMITED` is the schema's own word for it, and picking 3 or 12 would
    be exactly the "number somebody liked" this module refuses elsewhere. The piece
    declares no ``value`` for the same reason -- readers default it to 1 and a declared 1
    would be a number stated twice.

    ``from_captures: false`` when the hand has to be created, so this stays *one* edit: a
    hand that also filled from captures would be this rule and crazyhouse's at once, and
    the validator's *"nothing else fills a hand"* refusal is answered by the stock.
    """
    if any(str(piece.get("id")) == piece_id for piece in pieces(spec)):
        return spec
    spec.setdefault("pieces", []).append({"id": piece_id, "name": name, "moves": []})
    drops = spec.setdefault("rules", {}).setdefault(
        "drops", {"from_captures": False}
    )
    drops.setdefault("stock", {})[piece_id] = STOCK_UNLIMITED
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 10)
    return spec


def set_explosion(
    spec: dict[str, Any],
    radius: int = 1,
    by: list[str] | None = None,
    of: list[str] | None = None,
) -> dict[str, Any]:
    """A capture that takes its neighbours with it: atomic chess, and who it happens to.

    ``by`` and ``of`` are schema 10 (Q18) and narrow *when* the blast fires rather than
    what it reaches -- ``by`` the capturer types whose captures explode, ``of`` the victim
    types whose being taken does. Both are passed rather than guessed, and both default to
    absent, which is every type: a spec written before them means what it always did.

    ``radius`` 1 because it is atomic's own and the smallest the schema allows, so an
    operator adding one adds the known game rather than a bigger version of it.
    """
    rules = spec.setdefault("rules", {})
    explosion: dict[str, Any] = {"radius": radius}
    if by is not None:
        explosion["by"] = list(by)
    if of is not None:
        explosion["of"] = list(of)
    rules["explode_on_capture"] = explosion
    # An *empty* `by` is still schema 10 -- it is a rule that never fires, which the
    # complexity table charges nothing for and the compiler drops -- so the stamp asks
    # whether the key was passed, not whether it named anything.
    narrowed = by is not None or of is not None
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 10 if narrowed else 2)
    return spec


def set_contacts(
    spec: dict[str, Any], effect: str = "promote", position: str = "advance"
) -> dict[str, Any]:
    """The same rule keyed by *both* pieces: the attacker becomes what it took (schema 10).

    `set_strike` is the sibling and this is written to match it link for link -- the same
    value ranking, the same ``reverse`` on ``effect``, the same "adds no numbers of its
    own" -- because they are two spellings of one mechanism and the validator refuses
    them together. What the second spelling can say and the first cannot is the *mover*
    column: a strike is a map from the victim alone, so "the piece that took it changes
    too" is inexpressible in it however the chain is written. That is the axis this
    exists to let an operator cross, and it is what `distance.contacts_change_the_mover`
    reads.

    One link per adjacent rung of the value ladder rather than one per ordered pair, and
    the reason is the elegance metric rather than taste: the full cross product is 66
    links on `shogi` against a chain's 12, so a table written that way would be a spec
    dominated by one rule and a "local" edit that is anything but. Adjacent rungs give a
    table the same size as the chain it generalises and a rule that is still one
    sentence -- *take the piece one rung above you and you become it*, or one rung below
    under ``demote``.

    Ties are ordered by declaration and linked anyway, which is deliberate: the games
    where every piece is worth the same (`intransitive`, every `rps-*`) are exactly the
    ones a contact table has the most to say about, and a ranking that refused to order
    them would decline there.

    ``advance`` because nothing survives a link here, so it is the ordinary capture's own
    geometry and therefore the smaller change -- `set_strike`'s argument for ``stay``,
    which lands on the other value for the other spelling. The validator refuses
    ``advance`` beside a surviving target, which is why the two defaults differ.
    """
    ranked = sorted(
        (
            piece
            for piece in spec.get("pieces", [])
            if isinstance(piece, dict) and not piece.get("royal")
        ),
        key=lambda piece: float(piece.get("value", 1)),
        reverse=effect == "demote",
    )
    links = [
        {
            "mover": ranked[index]["id"],
            "onto": ranked[index + 1]["id"],
            "mover_to": ranked[index + 1]["id"],
        }
        for index in range(len(ranked) - 1)
    ]
    if not links:
        return spec
    rules = spec.setdefault("rules", {})
    rules["contacts"] = {
        "position": position,
        "links": links,
        # Every royal, and the validator says so by name: a royal that can be turned into
        # something else makes check a question about a piece that is not on the board.
        "immune": [
            piece["id"]
            for piece in spec.get("pieces", [])
            if isinstance(piece, dict) and piece.get("royal")
        ],
    }
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 10)
    return spec


# The landing set an operator writes under ``position: "choose"``: a king's step from the
# struck square. It is the example the Q20 specification uses and the blast's own radius 1,
# and it resolves to at most 8 candidates, far inside `MAX_LANDING_CANDIDATES`.
KING_STEP_LANDING: dict[str, Any] = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}

# The same landing on a hex lattice (schema 14): the six cells that share a side with the
# struck one, which is what a king's step is on a square board -- every cell touching it.
# Written out under ``none``, because the validator refuses every other symmetry on a hex
# board (a quarter turn carries a hex vector off the lattice), so `KING_STEP_LANDING` there
# was a refused child every time (Q484: 15 of them two steps from the corpus, all
# `hex-glinski-promoting`). A hex board has no cell touching only at a corner, so the
# Glinski king's six diagonals are two steps away and are not in it.
HEX_STEP_LANDING: dict[str, Any] = {
    "vectors": [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]],
    "symmetry": "none",
}


def step_landing(spec: dict[str, Any]) -> dict[str, Any]:
    """The landing `set_landing` writes on ``spec``'s lattice: a fresh copy, one step out."""
    hexed = spec.get("board", {}).get("lattice") == "hex"
    return copy.deepcopy(HEX_STEP_LANDING if hexed else KING_STEP_LANDING)


def keeps_a_survivor(spec: dict[str, Any], key: str) -> bool:
    """Whether ``rules[key]`` leaves a struck piece standing: what ``choose`` needs.

    The validator's own question, `schema.strike_survives_anything`, which asks the
    compiler's ``StrikeRule.survives_anything`` (Q271), with the piece ids read off
    ``spec``. This function used to count links instead -- a strike whose effect is not
    ``capture`` with any link, or a table with any ``onto_to`` -- and that count took a
    link from or onto an immune piece for a survivor, although a strike never reaches
    one. So `set_landing` wrote ``choose`` on a rule the validator refuses under it, and
    `settle_landing` kept it there (Q413). Operators never write an immune link, so
    only a spec that arrived carrying one, an LLM proposal, could reach it.

    A wrapper rather than the validator's function at each call site, because every
    caller holds a spec and none holds the piece-id list; this is the one place that
    turns the one into the other.
    """
    piece_ids = list(
        dict.fromkeys(
            piece["id"]
            for piece in pieces(spec)
            if isinstance(piece, dict) and isinstance(piece.get("id"), str)
        )
    )
    return strike_survives_anything(spec.get("rules", {}).get(key), key, piece_ids)


def set_landing(spec: dict[str, Any], key: str) -> dict[str, Any]:
    """Let the attacker place a surviving struck piece a king's step away (schema 12, Q20).

    Written only where ``rules[key]`` keeps a survivor, because ``choose`` with nothing to
    place is refused as ``advance`` written the long way. The step follows the lattice
    (`step_landing`): on a hex board it is the six neighbours, spelled without symmetry.
    """
    rule = spec.get("rules", {}).get(key)
    if not isinstance(rule, dict) or not keeps_a_survivor(spec, key):
        return spec
    rule["position"] = "choose"
    rule["landing"] = step_landing(spec)
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 12)
    return spec


def clear_landing(spec: dict[str, Any], key: str, position: str = "stay") -> dict[str, Any]:
    """Put ``rules[key]`` back under a position that places nothing, and drop its landing."""
    rule = spec.get("rules", {}).get(key)
    if isinstance(rule, dict):
        rule["position"] = position
        rule.pop("landing", None)
    return spec


def settle_landing(spec: dict[str, Any]) -> dict[str, Any]:
    """Drop ``choose`` once nothing is left for it to place.

    `settle_en_passant`'s repair for schema 12. An edit that removes the last link keeping
    a struck piece alive leaves ``choose`` refused as ``advance`` written the long way, so
    the rule becomes exactly that: ``advance``, with no ``landing``.
    """
    rules = spec.get("rules", {})
    for key in ("strike", "contacts"):
        rule = rules.get(key)
        if not isinstance(rule, dict):
            continue
        if rule.get("position") == "choose" and not keeps_a_survivor(spec, key):
            clear_landing(spec, key, "advance")
        elif rule.get("position") != "choose":
            rule.pop("landing", None)
    return spec


def merge_links(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Every own-piece contact link `set_own_link` may add, read off the declared values.

    Ranked as `set_contacts` ranks, non-royal pieces by value with ties in declaration
    order. The cheapest piece moves onto an own piece of some rung and both become the
    piece one rung above it: `archive/merger`'s pawn onto a rook making a queen. The link
    carries ``mover_to`` only, so the target is removed and nothing survives to place.

    Every validator refusal is excluded here rather than repaired after. No rung is royal,
    so no royal is merged away or made; a target in ``contacts.immune`` is skipped; a pair
    that already has an own link is skipped; and ``mover_to`` is never the mover, since it
    sits at least one rung above the cheapest piece.
    """
    ranked = sorted(
        (piece for piece in pieces(spec) if not piece.get("royal")),
        key=lambda piece: float(piece.get("value", 1)),
    )
    contacts = spec.get("rules", {}).get("contacts")
    contacts = contacts if isinstance(contacts, dict) else {}
    declared = contacts.get("immune")
    immune: list[Any] = declared if isinstance(declared, list) else []
    taken = {
        (link.get("mover"), link.get("onto"))
        for link in contacts.get("links") or []
        if isinstance(link, dict) and link.get("own") is True
    }
    if not ranked:
        return []
    mover = ranked[0]["id"]
    return [
        {"mover": mover, "onto": ranked[index]["id"], "own": True, "mover_to": ranked[index + 1]["id"]}
        for index in range(len(ranked) - 1)
        if ranked[index]["id"] not in immune and (mover, ranked[index]["id"]) not in taken
    ]


def set_own_link(spec: dict[str, Any], link: dict[str, Any]) -> dict[str, Any]:
    """Add one own-piece contact link, creating the table where there is none (schema 11).

    A new table takes ``advance``, `set_contacts`' default and the only position that
    needs nothing else, and names every royal immune as the validator requires.
    """
    rules = spec.setdefault("rules", {})
    contacts = rules.get("contacts")
    if isinstance(contacts, dict) and isinstance(contacts.get("links"), list):
        contacts["links"].append(dict(link))
    else:
        rules["contacts"] = {
            "position": "advance",
            "links": [dict(link)],
            "immune": [piece["id"] for piece in pieces(spec) if piece.get("royal")],
        }
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 11)
    return spec


def split_piece(
    spec: dict[str, Any], source_id: str | None, new_id: str, share: float = 0.5
) -> dict[str, Any]:
    """Turn some of one piece's copies into a new piece type of their own.

    The only edit here that makes an army *bigger*, and it exists because measuring the
    operator set found it could only ever make one smaller: nothing could cross the
    grid's army axis upward, so half of that dimension was reachable by a model call and
    nothing else.

    A split rather than an invention, and that is what keeps it local and keeps it
    valid. The new piece starts as a copy of an existing one and takes over a share of
    its squares, so no square is invented, none is left doubly occupied, and the army
    size is unchanged in *pieces* while changing in *kinds* -- which is exactly the axis
    being crossed. What makes it a different piece is left to the operator that calls
    this: on its own a split produces two identical armies with two names, and the
    caller perturbs one of them.

    Does nothing if the source has only one placed copy. Something that stands on one
    square cannot be shared, and taking it over entirely would be a rename.

    A share keeps its placement's ``player`` and ``facing``, one new placement per pair
    the source was written with. A split changes what a piece is, never whose it is or
    which way it points; writing every share into one bare placement handed a written-out
    second army's pieces to the first player (Q330).
    """
    if source_id is None or any(piece.get("id") == new_id for piece in pieces(spec)):
        return spec
    source = next((p for p in pieces(spec) if p.get("id") == source_id), None)
    if source is None or source.get("royal"):
        return spec

    squares = [
        placement
        for placement in spec.get("setup", {}).get("placements", [])
        if placement.get("piece") == source_id and len(placement.get("squares", [])) > 1
    ]
    if not squares:
        return spec

    clone = copy.deepcopy(source)
    clone["id"] = new_id
    clone["name"] = f"{source.get('name', source_id)} variant"
    clone.pop("promoted_from", None)
    spec.setdefault("pieces", []).append(clone)

    # Keyed by what the placement says of its pieces besides where they stand, in the
    # order first met, so a spec with one bare placement gets the one it always did.
    taken: dict[tuple[Any, Any], list[str]] = {}
    for placement in squares:
        held = placement["squares"]
        count = max(1, int(len(held) * share))
        owner = (placement.get("player"), placement.get("facing"))
        taken.setdefault(owner, []).extend(held[-count:])
        placement["squares"] = held[:-count]
    spec["setup"]["placements"] = [
        placement
        for placement in spec["setup"]["placements"]
        if placement.get("squares")
    ]
    for (player, facing), cells in taken.items():
        placed: dict[str, Any] = {"piece": new_id}
        if player is not None:
            placed["player"] = player
        placed["squares"] = sorted(cells)
        if facing is not None:
            placed["facing"] = facing
        spec["setup"]["placements"].append(placed)
    return spec
