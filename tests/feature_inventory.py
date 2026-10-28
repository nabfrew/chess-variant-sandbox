"""What each schema bump added, by name: the feature inventory the tests read (Q1008c).

Until Q1008c these were the validator's version collectors (`spec/schema.py`): a spec stamped
below the bump that added a feature was refused for using it. The stamp is no longer read by
the validator (`docs/reference/schema-reboot-assessment.md` §9), so the gates went; the
inventories stay here, because three kinds of test read them as the list of what each bump
added: the operator reachability sweep (`test_operators.py`), the showcase audit
(`test_operators.py::_EVERY_SCHEMA_FEATURES`, `docs/reference/schema-showcase-coverage.md`)
and each schema's own tests. Nothing in the package imports this module.

Each answers *which of this bump's features does this spec use*, by name, whatever the
spec's stamp. ``speed`` and ``rest`` are named only at a value other than their off value
0 (`schema.has_speeds`, `schema.has_rests`), since Q1008c made ``0`` the key's absence.

A rule added after schema 30 adds its names as a new entry of `LAST_BUMP`'s successor here,
for the tests; it bumps no version and gates nothing (skill `new-move-primitive`).
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox.spec.reachability import _promotion_entries
from chess_variant_sandbox.spec.schema import (
    EXTINCTION_PER_TYPE_OWN,
    REPETITION_NEVER,
    SIMULTANEOUS_CHASE_KEYS,
    WRAP_PAIRS,
    _listed,
    castling_rank_walled_beyond_its_spans,
    duck_goes,
    duck_piece,
    has_rests,
    has_speeds,
    hire_ids,
    keeping_link_holds,
    majority_ids,
    neutral_ids,
    npc_pounce,
    npc_respawn,
    npc_split,
    npc_turns_back,
    piece_bounties,
    piece_prices,
    piece_sights,
    promoting_piece_ids,
    purse_bribes,
    purse_rule,
    strike_is_live,
    turn_is_long,
    uses_sweep,
)

# The last schema version that was a feature bump. The inventory runs from 9 to here.
LAST_BUMP = 51

# Schema 9's two rule keys, which were `schema.V9_RULE_KEYS`.
V9_RULE_KEYS = frozenset({"npcs", "promote_on_entry"})


def _schema_nine_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 9 added, by name.

    One collector rather than one ``if`` per feature, so that the second key schema 9
    carries (queue task Q13) joins here and the message stays one sentence: a spec that
    reaches for two schema-9 features under an older stamp is told both at once.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    used |= {f"rules.{key}" for key in V9_RULE_KEYS.intersection(rules)}
    scoring = rules.get("scoring")
    if isinstance(scoring, dict) and scoring.get("count") == "points":
        used.add('rules.scoring.count "points"')
    return used


def _schema_ten_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 10 added, by name.

    The same collector shape as ``_schema_nine_uses`` and for a stronger reason: schema 10
    carries five features and exactly *one* of them is a top-level key under ``rules``, so
    a ``V10_RULE_KEYS`` frozenset written the schema-4-to-6 way would gate one fifth of the
    bump and wave the rest through under any stamp. Q84 sorted them -- a new value on an
    existing key, three nested keys, one relaxation and one rule key -- and only the last
    is indexable.

    **The set of names this returns is the bump's inventory.** ``describe.py`` has to be
    able to say each of them, ``docs/reference/schema-showcase-coverage.md`` audits against
    them, and each owes a demonstrating ruleset. Everything is spelled as its own ``if`` so
    that a spec reaching for two of them under an older stamp is told about both at once.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    if rules.get("repetition_draw") == REPETITION_NEVER:
        used.add('rules.repetition_draw "never"')
    npcs = rules.get("npcs")
    if isinstance(npcs, list) and any(
        isinstance(npc, dict) and "immune" in npc for npc in npcs
    ):
        used.add("rules.npcs[].immune")
    explosion = rules.get("explode_on_capture")
    if isinstance(explosion, dict):
        used |= {
            f"rules.explode_on_capture.{key}"
            for key in ("by", "of")
            if key in explosion
        }
    drops = rules.get("drops")
    if isinstance(drops, dict) and "stock" in drops:
        used.add("rules.drops.stock")
    # The bump's one indexable feature, and the only one a `V10_RULE_KEYS` frozenset
    # could have caught -- which is why the collector exists rather than the frozenset.
    if "contacts" in rules:
        used.add("rules.contacts")
    # The relaxation, named separately from the key that gates it. A moveless piece and a
    # stock arrive together in a valid spec, but they are two of the bump's five features
    # and the collector's set *is* the inventory -- `describe.py` owes a sentence for each
    # and the showcase audit reads this list, so folding one into the other would hide it.
    pieces = spec.get("pieces")
    if isinstance(pieces, list) and any(
        isinstance(piece, dict) and piece.get("moves") == [] for piece in pieces
    ):
        used.add("a piece with no moves of its own")
    return used


def _schema_eleven_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 11 added, by name.

    One feature: a contact link about the mover's *own* piece (Q19b). It is a nested key,
    so no key set could gate it. It has a version to itself because it moves perft.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    contacts = rules.get("contacts")
    links = contacts.get("links") if isinstance(contacts, dict) else None
    if isinstance(links, list) and any(
        isinstance(link, dict) and "own" in link for link in links
    ):
        used.add("rules.contacts.links[].own")
    return used


def _schema_twelve_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 12 added, by name.

    One feature on two spellings: ``position: "choose"`` and its ``landing`` set (Q20),
    on ``strike`` and on ``contacts``. A new value and a nested key, so no key set could
    gate either.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    for key in ("strike", "contacts"):
        rule = rules.get(key)
        if not isinstance(rule, dict):
            continue
        if rule.get("position") == "choose":
            used.add(f"rules.{key}.position=choose")
        if "landing" in rule:
            used.add(f"rules.{key}.landing")
    return used


def _schema_thirteen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 13 added, by name.

    One feature: the ``sweep`` move kind (Q21). A new value on a move descriptor, gated the
    way ``lame`` is.
    """
    return {"the 'sweep' move kind"} if uses_sweep(spec) else set()


def _schema_fourteen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 14 added, by name.

    Two features (Q229): the ``board.lattice`` declaration and a promotion zone stated as
    ``squares``. Nested keys both, so no key set could gate either. The key is schema 14's
    whatever its value, so ``"square"`` written out is named too; it compiles to the board
    it always was and owes no showcase.
    """
    used: set[str] = set()
    board = spec.get("board")
    if isinstance(board, dict) and "lattice" in board:
        used.add(f"board.lattice={board['lattice']}")
    rules = spec.get("rules")
    if isinstance(rules, dict) and any(
        isinstance(entry, dict) and "squares" in entry for entry in _promotion_entries(rules)
    ):
        used.add("rules.promotion.squares")
    return used


def _schema_fifteen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 15 added, by name.

    One feature (Q452, traditional shogi's B1): ``promotion.on_leaving``, a promotion
    entry that also promotes its piece on a move *out of* its zone. A nested key, so no
    key set could gate it. The key is schema 15's whatever its value, as
    ``board.lattice`` is schema 14's, so ``false`` written out is named too; it compiles
    to the game that was already there and owes no showcase.
    """
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return set()
    used: set[str] = set()
    for entry in _promotion_entries(rules):
        if isinstance(entry, dict) and "on_leaving" in entry:
            used.add(
                "rules.promotion.on_leaving"
                if entry["on_leaving"] is True
                else "rules.promotion.on_leaving=false"
            )
    return used


def _schema_sixteen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 16 added, by name.

    Two features (Q281): ``board.shape`` and a hexagon's three pairs of sides under
    ``rules.wrap``. Nested keys both, so no key set could gate either. The shape is
    named with its value, as ``board.lattice`` is, so ``"grid"`` written out is named
    too; it compiles to the board it always was and owes no showcase. A pair key is
    named whatever its value, because a schema-15 spec may not write even an inert one.
    """
    used: set[str] = set()
    board = spec.get("board")
    if isinstance(board, dict) and "shape" in board:
        used.add(f"board.shape={board['shape']}")
    rules = spec.get("rules")
    wrap = rules.get("wrap") if isinstance(rules, dict) else None
    if isinstance(wrap, dict) and any(
        key in wrap or f"{key}_by_player" in wrap for key in WRAP_PAIRS
    ):
        used.add("rules.wrap.hexagon")
    return used


def _schema_seventeen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 17 added, by name.

    Four features (Q540, *what a capture leaves behind*), all new values or nested keys,
    so no key set could gate any of them:

    * ``rules.strike.position=stay(capture)`` -- rifle: a capture that leaves the
      attacker where it stood. Schema 6 to 16 refused the spelling as stating nothing
      (Q538); the value is old, the combination is the feature.
    * ``rules.strike.position=home`` -- circe: a struck piece reborn on its home square.
    * ``rules.strike.switches_sides`` -- Andernach: the attacker changes sides. Named
      whatever its value, as ``board.lattice`` is, because an older spec may not write
      even an empty one.
    * ``rules.explode_on_capture.radius=0`` -- kamikaze: the blast takes the capturer and
      its victim and nothing around them.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    strike = rules.get("strike")
    if isinstance(strike, dict):
        position = strike.get("position", "advance")
        if position == "stay" and strike.get("effect", "capture") == "capture":
            used.add("rules.strike.position=stay(capture)")
        if position == "home":
            used.add("rules.strike.position=home")
        if "switches_sides" in strike:
            used.add("rules.strike.switches_sides")
    explosion = rules.get("explode_on_capture")
    if isinstance(explosion, dict) and explosion.get("radius") == 0 and not isinstance(
        explosion.get("radius"), bool
    ):
        used.add("rules.explode_on_capture.radius=0")
    return used


def _schema_eighteen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 18 added, by name.

    Schema 18 is built in several tasks (`docs/archive/plan-schema-17-18.md` §9.3, 18a to
    18f), one version for all of them: each later task adds its names here, and until it lands its
    feature is refused by name at every version, so nothing accepted earlier is read
    differently afterwards. 18a (Q781) opens it with two lifted restrictions, so both
    are predicates over the spec rather than keys:

    * ``rules.contacts.mover_to=mover(promotion)`` -- Einstein's keeping link: a contact
      link whose ``mover_to`` is the mover itself, accepted where the mover has a
      promotion of its own (``keeping_link_holds``). The contact overrides the promotion
      (Q539), so a queen that captures stays a queen where its promotion would demote
      it. Schema 10 to 17 refused the spelling as changing nothing, which it still does
      where no promotion could fire.
    * ``rules.castling(walled rank)`` -- Omega's castling: a wall or an absent cell on a
      rank castling starts from, but on no square any castling crosses or lands on
      (``castling_rank_walled_beyond_its_spans``). A walled span stays refused.

    18c (Q783) adds two nested keys, each named whatever its value, as schema 15's
    ``on_leaving`` is, because an older spec may not write even an inert one; ``false``
    is named with its value and compiles to the game that was already there:

    * ``rules.extinction.per_type`` -- Extinction chess: each listed type is counted on
      its own, so a side loses when any one of them runs out, not when all of them
      together do.
    * ``rules.promotion.only_lost`` -- Grand chess: a promotion target is offered only
      while the mover's side has fewer of that type than it started with
      (``_check_only_lost_coherence``).

    18d (Q784) adds one nested key, named whatever its value as 18c's are:

    * ``rules.drops.placement`` -- Placement chess and Sittuyin: while a side holds
      anything in hand its only legal moves are drops, so a stock is played out before
      a piece on the board moves. A royal may start in the stock
      (``royal_starts_in_hand``, ``_check_placement_coherence``).

    18e (Q785) adds one nested key, named whatever its value as 18d's is:

    * ``rules.drops.either_face`` -- Kyoto shogi: a piece in hand may be dropped as
      itself or as its other face, the piece whose ``promoted_from`` it is, so a captured
      dragon (held as a rook) may come back as either (``_check_either_face_coherence``).

    18f (Q786) adds one new value and its key, named whatever its value as 18c's are:

    * ``rules.reach.on=leave`` -- Jeson Mor: the game is decided by the move that takes
      the named piece off a goal square, not by a piece standing on one. The core keeps
      one position field for it, the side that has just left (``Game::goal_left_by``).
      ``rules.reach.on=arrive`` is the rule written out, and compiles to the game that was
      already there.

    18b (Q782) adds two new values on an existing key, each named wherever it is written,
    on a piece's move or on a strike's or contact table's landing:

    * ``symmetry=rotate6`` and ``symmetry=dihedral12`` -- the hexagon's turns, so a hex
      step is written once and turned six ways (or six ways and mirrored). A spelling and
      not a rule: `complexity.expand_symmetry` turns them, the compiler reads its answer,
      and a spec written with them compiles to the tables of its twin that lists every
      vector under ``none``. Accepted only on a hex lattice (`_symmetry_refusals`).
    """
    used: set[str] = set()
    used |= _hex_symmetries_used(spec)
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    extinction = rules.get("extinction")
    if isinstance(extinction, dict) and "per_type" in extinction:
        used.add(
            "rules.extinction.per_type"
            if extinction["per_type"] is not False
            else "rules.extinction.per_type=false"
        )
    for entry in _promotion_entries(rules):
        if isinstance(entry, dict) and "only_lost" in entry:
            used.add(
                "rules.promotion.only_lost"
                if entry["only_lost"] is True
                else "rules.promotion.only_lost=false"
            )
    drops = rules.get("drops")
    if isinstance(drops, dict) and "placement" in drops:
        used.add(
            "rules.drops.placement"
            if drops["placement"] is True
            else "rules.drops.placement=false"
        )
    if isinstance(drops, dict) and "either_face" in drops:
        used.add(
            "rules.drops.either_face"
            if drops["either_face"] is True
            else "rules.drops.either_face=false"
        )
    contacts = rules.get("contacts")
    links = contacts.get("links") if isinstance(contacts, dict) else None
    promoting = promoting_piece_ids(rules)
    if isinstance(links, list) and any(
        keeping_link_holds(link, promoting) for link in links
    ):
        used.add("rules.contacts.mover_to=mover(promotion)")
    if castling_rank_walled_beyond_its_spans(spec):
        used.add("rules.castling(walled rank)")
    reach = rules.get("reach")
    if isinstance(reach, dict) and "on" in reach:
        used.add(f"rules.reach.on={reach['on']}")
    return used


def _schema_nineteen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 19 added, by name (Q771, portals).

    One new top-level rule and its two nested keys, each named whatever its value, as
    schema 15's ``on_leaving`` is, because an older spec may not write even an inert one:

    * ``rules.portals`` -- pairs of portals: a piece whose move ends on an open end comes
      out one king's step beyond the other end, in the direction it travelled
      (`docs/archive/plan-portals.md` §§1-2).
    * ``rules.portals.moved_by`` -- which players may move each end (the free agent, §3).
    * ``rules.portals.passage`` -- per player, which end that player's pieces may go in
      by: both, one way, or neither, when the pair is an obstacle to them.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if not isinstance(rules, dict) or "portals" not in rules:
        return used
    used.add("rules.portals")
    portals = rules.get("portals")
    for pair in portals if isinstance(portals, list) else ():
        if isinstance(pair, dict):
            if "moved_by" in pair:
                used.add("rules.portals.moved_by")
            if "passage" in pair:
                used.add("rules.portals.passage")
    return used


def _schema_twenty_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 20 added, by name (Q807, no straight back).

    * ``rules.portals.no_straight_back`` -- an end of the pair, once moved, may not be moved
      straight back onto the square it left on the very next move: a portal's ko, the answer
      to Ferry's repetition by shuffling (`docs/archive/plan-portals.md` *Ferry's shuffle*). Named
      whatever its value, as schema 19's names are, because an older spec may not write even
      the inert ``false``.
    """
    rules = spec.get("rules")
    portals = rules.get("portals") if isinstance(rules, dict) else None
    for pair in portals if isinstance(portals, list) else ():
        if isinstance(pair, dict) and "no_straight_back" in pair:
            return {"rules.portals.no_straight_back"}
    return set()


def _schema_twenty_one_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 21 added, by name (Q843a, portal maps and filters).

    `docs/archive/plan-portals-2.md` §2. Each named whatever its value, as schema 19's names are,
    because an older spec may not write even an inert one:

    * ``rules.portals.ends`` -- a group written as a list of ends rather than a pair: a
      ring, a funnel or any other map of three to eight ends.
    * ``rules.portals.ends.to`` -- the end a piece that goes in by this one comes out of.
    * ``rules.portals.ends.enter`` -- whose pieces may go in by this end.
    * ``rules.portals.carries`` -- which pieces may go in by an end, in either spelling.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    portals = rules.get("portals") if isinstance(rules, dict) else None
    for group in portals if isinstance(portals, list) else ():
        if not isinstance(group, dict):
            continue
        if "carries" in group:
            used.add("rules.portals.carries")
        if "ends" not in group:
            continue
        used.add("rules.portals.ends")
        for end in _listed(group["ends"]):
            if not isinstance(end, dict):
                continue
            if "to" in end:
                used.add("rules.portals.ends.to")
            if "enter" in end:
                used.add("rules.portals.ends.enter")
            if "carries" in end:
                used.add("rules.portals.carries")
    return used


def _schema_twenty_two_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 22 added, by name: nothing any more.

    Schema 22 (Q843c) added fold portals, ``rules.portals.kind``. Their rules changed at
    schema 24 (Q867b, `docs/archive/plan-portals-2.md` §3R.6), and the first rules were deleted, not
    kept behind the old number (one implementation), so ``kind`` is
    `_schema_twenty_four_uses`' and schema 22 keeps no rule of its own.
    """
    del spec
    return set()


def _schema_twenty_three_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 23 added, by name (Q850, several wild pieces on
    one walk).

    `docs/reference/designer-case-study-wolves.md` §2.2. Named whatever its value, because
    an older spec may not write even an empty one:

    * ``rules.npcs.also_at`` -- squares of the entry's own route where a further wild piece
      of the same kind stands at ply zero, every one of them walking the route in step with
      the piece on the route's first square.
    """
    rules = spec.get("rules")
    npcs = rules.get("npcs") if isinstance(rules, dict) else None
    for npc in npcs if isinstance(npcs, list) else ():
        if isinstance(npc, dict) and "also_at" in npc:
            return {"rules.npcs.also_at"}
    return set()


def _schema_twenty_four_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 24 added, by name (Q867b, folds revised).

    `docs/archive/plan-portals-2.md` §3R. Named whatever its value, as schema 19's names are,
    because an older spec may not write even the default:

    * ``rules.portals.kind`` -- a group's rule set: ``"fold"``, a portal a piece of either
      side may stand in, where a move that ends on it empty may stay or go through and end
      on the end it leads to, taking an enemy piece there as any landing does (absent:
      today's momentum portal). Schema 22's first fold rules are gone with their version.
    """
    rules = spec.get("rules")
    portals = rules.get("portals") if isinstance(rules, dict) else None
    for group in portals if isinstance(portals, list) else ():
        if isinstance(group, dict) and "kind" in group:
            return {"rules.portals.kind"}
    return set()


def _schema_twenty_five_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 25 added, by name (Q869, a fixed facing).

    `docs/archive/plan-library-audit.md` *The rear pawn: a fixed facing*. Named whatever its value,
    because an older spec may not write even the default:

    * ``pieces[].facing.turns`` -- whether a move may turn the piece (absent: it may, as
      every facing piece could from schema 8). ``false`` keeps each piece pointing the way
      it was placed, for the whole game.
    """
    pieces = spec.get("pieces")
    for piece in pieces if isinstance(pieces, list) else ():
        facing = piece.get("facing") if isinstance(piece, dict) else None
        if isinstance(facing, dict) and "turns" in facing:
            return {"pieces[].facing.turns"}
    return set()


def _schema_twenty_six_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 26 added, by name (Q918a, simultaneous moves).

    `docs/reference/rounds.md` §5. Named whatever its value, because an older spec may not
    write even the default:

    * ``rules.simultaneous`` -- the game is played in rounds: both players choose a move on
      the same board, in secret, and both are revealed and resolved together;
    * ``rules.simultaneous.tie`` -- what decides a collision both pieces win or both lose
      (absent: both are eliminated), with its value;
    * ``pieces[].beats`` -- the pieces this one eliminates when the two collide.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if isinstance(rules, dict) and "simultaneous" in rules:
        used.add("rules.simultaneous")
        rule = rules["simultaneous"]
        if isinstance(rule, dict) and "tie" in rule:
            used.add(f"rules.simultaneous.tie {rule['tie']!r}")
    pieces = spec.get("pieces")
    for piece in pieces if isinstance(pieces, list) else ():
        if isinstance(piece, dict) and "beats" in piece:
            used.add("pieces[].beats")
            break
    return used


def _schema_twenty_seven_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 27 added, by name.

    One version, three tasks (`docs/plan-spy.md` §2.2), each adding its names here as it
    lands, as schema 18's six did:

    * ``rules.extinction.per_type.own`` -- Q936, `docs/reference/extinction-any-kind.md`
      §2: each listed kind counted on its own, and only against a side that starts with
      at least one, so two armies of different kinds can each lose by running out of any
      one of their own.

    * ``rules.simultaneous.rest`` -- Q942a, `docs/archive/plan-round-chase.md` §3 R1, the piece
      that made a player's move resting the next round -- was here until schema 29 retired it
      for ``pieces[].rest`` (Q983a, `docs/plan-rest.md` §4.4); only the converter
      reads it now, so this collector does not name it;
    * ``rules.simultaneous.swap`` -- Q942a, R2: two moves that each end on the other's
      origin meet as a collision (``beats`` and the tie decide) instead of passing;
    * ``rules.simultaneous.both`` -- Q942a, X1: an ending that fires for both players in
      one round is settled by ``rules.scoring`` instead of drawn.

    * ``pieces.spy`` -- Q935a, `docs/plan-spy.md` §1, §2: a piece type that decides how the
      game is played: alternating while both players have one, announced rounds (the
      player without one moves first, in the open) once one is lost, blind rounds once
      both are.

    Each name is collected whatever its value, since an older spec may not write even a
    refused one.
    """
    used: set[str] = set()
    pieces = spec.get("pieces")
    if any(isinstance(piece, dict) and "spy" in piece for piece in _listed(pieces)):
        used.add("pieces.spy")
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    extinction = rules.get("extinction")
    if isinstance(extinction, dict) and extinction.get("per_type") == EXTINCTION_PER_TYPE_OWN:
        used.add("rules.extinction.per_type.own")
    simultaneous = rules.get("simultaneous")
    if isinstance(simultaneous, dict):
        for key in SIMULTANEOUS_CHASE_KEYS:
            if key in simultaneous:
                used.add(f"rules.simultaneous.{key}")
    return used


def _schema_twenty_eight_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 28 added, by name (Q943a, `docs/archive/plan-speed.md`).

    * ``pieces.speed`` -- a piece type's speed, a rank (absent: 0, the slowest). When the
      two pieces moving in a round differ in speed, the faster move is made first, as a move
      of an alternating game, and the slower is then made only if its piece still stands
      and the move can still be made on that board (§2.4, §2.5).

    Collected whatever its value, since an older spec may not write even a refused one.
    """
    if has_speeds(spec):
        return {"pieces.speed"}
    return set()


def _schema_twenty_nine_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 29 added, by name (Q983a, `docs/plan-rest.md`).

    * ``pieces.rest`` -- a piece type's rest, 1 to ``REST_MAX`` (absent: 0, it never rests).
      A piece that makes its player's move may not make that player's move during that
      player's next *rest* turns (an alternating turn or a round), unless the rest would
      leave the player no legal move, when it is lifted for that turn (§3).

    Collected whatever its value, since an older spec may not write even a refused one.
    """
    if has_rests(spec):
        return {"pieces.rest"}
    return set()


def _schema_thirty_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 30 added, by name (Q999a, `docs/archive/plan-passive.md`).

    * ``pieces.sight.freeze`` -- a line of sight whose pieces may not move and attack nothing.
    * ``pieces.sight.disarm`` -- a line of sight whose pieces may not capture and attack nothing.
    * ``pieces.sight.shield`` -- a line of sight whose pieces may not be captured (a shielded
      royal is never in check).
    * ``pieces.sight.through`` -- a sight that sees past pieces (an aura) rather than stopping
      at the first one (a ray).

    ``whose`` and ``affects`` are values of a sight's keys, as ``mode`` is of a move's, and
    have no names of their own (§4.1). Collected whatever else the entry says.
    """
    used: set[str] = set()
    for piece in _listed(spec.get("pieces")):
        if not isinstance(piece, dict) or "sight" not in piece:
            continue
        for entry in _listed(piece["sight"]):
            if not isinstance(entry, dict):
                continue
            effect = entry.get("effect")
            if isinstance(effect, str) and effect in _SCHEMA_THIRTY_EFFECTS:
                used.add(f"pieces.sight.{effect}")
            if "through" in entry:
                used.add("pieces.sight.through")
    return used


# Schema 30's three effects; the fourth, `command`, is generation 31's (below).
_SCHEMA_THIRTY_EFFECTS = ("freeze", "disarm", "shield")


def _schema_thirty_one_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 31 added, by name (Q1007c,
    `docs/archive/plan-schema-31.md` §3; numbered by order of landing, bumping no version).

    * ``pieces.sight.command`` -- a line of sight whose pieces their opponent may also move,
      onto empty squares, as that player's move and as their owner would move them.
    * ``pieces.sight.captures_own`` -- a command whose pieces may also be made to take a piece
      of their own side (a turncoat capture).

    Collected whatever else the entry says.
    """
    used: set[str] = set()
    for piece in _listed(spec.get("pieces")):
        if not isinstance(piece, dict) or "sight" not in piece:
            continue
        for entry in _listed(piece["sight"]):
            if not isinstance(entry, dict):
                continue
            if entry.get("effect") == "command":
                used.add("pieces.sight.command")
            if "captures_own" in entry:
                used.add("pieces.sight.captures_own")
    return used


def _schema_thirty_two_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 32 added, by name (Q1007a,
    `docs/archive/plan-schema-31.md` §2; numbered by order of landing, bumping no version).

    * ``pieces.moves.lame.range`` -- a lame move with a ``range`` other than 1: the lame
      rider, jump after jump, each blocked on its leg (a lifted refusal; 1 is the single
      lame leap every earlier file means).
    * ``pieces.moves.lame.min_range`` -- the nearest jump a lame rider may stop on, at a
      value other than 1.
    """
    used: set[str] = set()
    for piece in _listed(spec.get("pieces")):
        moves = piece.get("moves") if isinstance(piece, dict) else None
        for move in _listed(moves):
            if not isinstance(move, dict) or move.get("kind") != "lame":
                continue
            if move.get("range", 1) != 1:
                used.add("pieces.moves.lame.range")
            if move.get("min_range", 1) != 1:
                used.add("pieces.moves.lame.min_range")
    return used


def _schema_thirty_three_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 33 added, by name (I1,
    `docs/archive/plan-pieces-moves.md` §4; numbered by order of landing, bumping no version).

    * ``pieces.moves.empty`` -- a type with ``moves: []`` that the setup places, or that a
      promotion, strike chain or contact can create, or that is a wild type: a piece that
      never moves, standing on the board (a lifted restriction). A stone only a stock puts
      in a hand is schema 10's *a piece with no moves of its own*, not this.
    """
    pieces = [p for p in _listed(spec.get("pieces")) if isinstance(p, dict)]
    still = {p.get("id") for p in pieces if p.get("moves") == []}
    if not still:
        return set()
    rules = spec.get("rules") if isinstance(spec.get("rules"), dict) else {}
    made: set[Any] = set()
    setup = spec.get("setup") if isinstance(spec.get("setup"), dict) else {}
    made |= {p.get("piece") for p in _listed(setup.get("placements")) if isinstance(p, dict)}
    made |= {n.get("piece") for n in _listed(rules.get("npcs")) if isinstance(n, dict)}
    for entry in _promotion_entries(rules):
        if not isinstance(entry, dict):
            continue
        for key in ("to", "to_by_player"):
            block = entry.get(key)
            for targets in block.values() if isinstance(block, dict) else [block]:
                made |= set(_listed(targets))
    strike = rules.get("strike") if isinstance(rules.get("strike"), dict) else {}
    made |= {link.get("to") for link in _listed(strike.get("chain")) if isinstance(link, dict)}
    contacts = rules.get("contacts") if isinstance(rules.get("contacts"), dict) else {}
    for link in _listed(contacts.get("links")):
        if isinstance(link, dict):
            made |= {link.get("mover_to"), link.get("onto_to")}
    # A stock alone is not a route here: the stocked stone is schema 10's.
    return {"pieces.moves.empty"} if still & made else set()



def _schema_thirty_four_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 34 added, by name (T1,
    `docs/archive/plan-pieces-moves.md` §5; numbered by order of landing, bumping no version).

    * ``pieces.moves.slide.path.circle`` -- a slide whose path turns to the next direction
      round at every step (the rose, the ring);
    * ``pieces.moves.slide.path.zigzag`` -- a slide whose path alternates between two
      neighbouring directions (the bolt).

    ``"straight"`` is the off value and names nothing.
    """
    used: set[str] = set()
    for piece in _listed(spec.get("pieces")):
        if not isinstance(piece, dict):
            continue
        for move in _listed(piece.get("moves")):
            if (
                isinstance(move, dict) and move.get("kind") == "slide"
                and move.get("path") in ("circle", "zigzag")
            ):
                used.add(f"pieces.moves.slide.path.{move['path']}")
    return used

def _schema_thirty_five_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 35 added, by name (A1,
    `docs/archive/plan-pieces-moves.md` §6; numbered by order of landing, bumping no version).

    * ``rules.strike.by`` -- only the listed pieces' captures strike (under rifle's
      reading, a piece that shoots: the arrow).
    """
    rules = spec.get("rules")
    strike = rules.get("strike") if isinstance(rules, dict) else None
    if isinstance(strike, dict) and strike.get("by"):
        return {"rules.strike.by"}
    return set()


def _schema_thirty_six_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 36 added, by name (N1, `docs/archive/plan-npc.md`
    §4.1; numbered by order of landing, bumping no version).

    * ``rules.npcs.respawn`` -- a taken wild piece comes back to its start a fixed number of
      phases later (the off value 0 is no rule, so it names nothing).
    """
    rules = spec.get("rules")
    npcs = rules.get("npcs") if isinstance(rules, dict) else None
    for npc in npcs if isinstance(npcs, list) else ():
        if npc_respawn(npc):
            return {"rules.npcs.respawn"}
    return set()


def _schema_thirty_seven_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 37 added, by name (N2, `docs/archive/plan-npc.md`
    §4.2; numbered by order of landing, bumping no version).

    * ``pieces.sight.wild.freeze``, ``.disarm``, ``.shield`` -- a wild type (an `npcs` entry's
      piece) carrying a line of sight with that effect, ``whose`` ``"any"``.
    * ``pieces.sight.shield.beside_npcs`` -- a shield, on any type, in a game with wild
      pieces: the lifted refusal, with its rule (a wild piece never takes a shielded piece).
    """
    rules = spec.get("rules")
    npcs = rules.get("npcs") if isinstance(rules, dict) else None
    if not isinstance(npcs, list) or not npcs:
        return set()
    wild = {npc.get("piece") for npc in npcs if isinstance(npc, dict)}
    used: set[str] = set()
    for piece in _listed(spec.get("pieces")):
        if not isinstance(piece, dict):
            continue
        for entry in _listed(piece.get("sight")):
            if not isinstance(entry, dict):
                continue
            effect = entry.get("effect")
            if piece.get("id") in wild and effect in _SCHEMA_THIRTY_EFFECTS:
                used.add(f"pieces.sight.wild.{effect}")
            if effect == "shield":
                used.add("pieces.sight.shield.beside_npcs")
    return used


def _schema_thirty_eight_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 38 added, by name (N5, `docs/archive/plan-npc.md`
    §4.7; numbered by order of landing, bumping no version).

    * ``rules.npcs.turns_back`` -- a wild piece paces its route as a line and turns round at
      its ends and wherever it is blocked (the off value ``false`` names nothing).
    """
    rules = spec.get("rules")
    npcs = rules.get("npcs") if isinstance(rules, dict) else None
    for npc in npcs if isinstance(npcs, list) else ():
        if npc_turns_back(npc):
            return {"rules.npcs.turns_back"}
    return set()


def _schema_thirty_nine_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 39 added, by name (N3b, `docs/archive/plan-npc.md`
    §4.3; numbered by order of landing, bumping no version).

    * ``rules.npcs.pounce.prey`` -- a wild piece pounces on a piece of the player who has just
      moved that it attacks, by the prey order; check from it is the pass test.
    """
    rules = spec.get("rules")
    npcs = rules.get("npcs") if isinstance(rules, dict) else None
    for npc in npcs if isinstance(npcs, list) else ():
        if npc_pounce(npc) == "prey":
            return {"rules.npcs.pounce.prey"}
    return set()


def _hex_symmetries_used(spec: dict[str, Any]) -> set[str]:
    """Schema 18b's names for the hexagon's turns, wherever a descriptor writes one."""
    written: list[Any] = []
    pieces = spec.get("pieces")
    for piece in pieces if isinstance(pieces, list) else ():
        moves = piece.get("moves") if isinstance(piece, dict) else None
        for move in moves if isinstance(moves, list) else ():
            if isinstance(move, dict):
                written.append(move.get("symmetry"))
    rules = spec.get("rules")
    for key in ("strike", "contacts"):
        rule = rules.get(key) if isinstance(rules, dict) else None
        landing = rule.get("landing") if isinstance(rule, dict) else None
        if isinstance(landing, dict):
            written.append(landing.get("symmetry"))
    return {f"symmetry={name}" for name in ("rotate6", "dihedral12") if name in written}


def uses(version: int, spec: dict[str, Any]) -> set[str]:
    """The names schema ``version`` added that ``spec`` uses (9 to `LAST_BUMP`)."""
    return set(_COLLECTORS[version](spec))


def _schema_forty_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 40 added, by name (N3c, `docs/archive/plan-npc.md`
    §4.4; numbered by order of landing, bumping no version).

    * ``rules.npcs.pounce.attackers`` -- a wild piece pounces on a piece of the player who has
      just moved that attacks it, by the prey order; check from it is the pass test.
    """
    rules = spec.get("rules")
    npcs = rules.get("npcs") if isinstance(rules, dict) else None
    for npc in npcs if isinstance(npcs, list) else ():
        if npc_pounce(npc) == "attackers":
            return {"rules.npcs.pounce.attackers"}
    return set()


def _schema_forty_one_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 41 added, by name (N3d, `docs/archive/plan-npc.md`
    §4.5; numbered by order of landing, bumping no version).

    * ``rules.npcs.split`` -- an ``"attackers"`` pouncer grows a copy of itself on each
      attacker's square while its type stands under the cap, then lunges as before.
    """
    rules = spec.get("rules")
    npcs = rules.get("npcs") if isinstance(rules, dict) else None
    for npc in npcs if isinstance(npcs, list) else ():
        if npc_split(npc):
            return {"rules.npcs.split"}
    return set()


def _schema_forty_two_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 42 added, by name (N4b, `docs/archive/plan-npc.md`
    §4.6; numbered by order of landing, bumping no version).

    * ``rules.duck`` -- every turn ends with the mover moving one neutral, uncapturable piece;
    * ``rules.duck.goes.by_moves`` -- the duck walks its type's moves instead of going anywhere;
    * ``rules.duck.start`` -- the duck starts on a square rather than off the board;
    * ``pieces.sight.duck`` -- the duck's type carries a sight (N2's duck half, §4.2).
    """
    rules = spec.get("rules")
    pid = duck_piece(rules)
    if pid is None:
        return set()
    names = {"rules.duck"}
    if duck_goes(rules) == "by_moves":
        names.add("rules.duck.goes.by_moves")
    if isinstance(rules, dict) and "start" in rules["duck"]:
        names.add("rules.duck.start")
    pieces = spec.get("pieces")
    for piece in pieces if isinstance(pieces, list) else ():
        if isinstance(piece, dict) and piece.get("id") == pid and piece.get("sight"):
            names.add("pieces.sight.duck")
    return names


def _schema_forty_three_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 43 added, by name (TC3, `docs/plan-turncoat.md`
    §4; numbered by order of landing, bumping no version).

    * ``setup.placements.player.nobody`` -- pieces nobody owns stand on the opening board;
    * ``pieces.control.majority`` -- a type nobody owns there is moved by the player with
      strictly more pieces next to it (a tie: nobody);
    * ``pieces.control.majority.geometry`` -- *next to* is a named leap's geometry rather than
      the king's step.

    A ``control`` on a type nobody's placement holds is all the validator admits, so the two
    control names need the placement; ``"stands"`` is the off value and names nothing.
    """
    names: set[str] = set()
    if neutral_ids(spec):
        names.add("setup.placements.player.nobody")
    held = majority_ids(spec)
    if held:
        names.add("pieces.control.majority")
    pieces = spec.get("pieces")
    for piece in pieces if isinstance(pieces, list) else ():
        if (
            isinstance(piece, dict) and piece.get("id") in held
            and isinstance(piece.get("control"), dict)
        ):
            names.add("pieces.control.majority.geometry")
    return names


def _schema_forty_four_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 44 added, by name (TC4, `docs/plan-turncoat.md`
    §5.1; numbered by order of landing, bumping no version).

    * ``pieces.bounty`` -- taking a piece of some type pays its bounty into the taker's points
      (and lifts the old refusal of ``scoring.count: "points"`` without wild pieces).

    ``0`` is the off value and names nothing. The purse TC4 builds has no key yet (TC5's
    ``rules.purse``), so it names nothing either.
    """
    return {"pieces.bounty"} if piece_bounties(spec) else set()


def _schema_forty_five_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 45 added, by name (TC5, `docs/plan-turncoat.md`
    §5.4; numbered by order of landing, bumping no version).

    * ``rules.purse`` -- coins each player holds, filled by bounties, and a market to spend
      them in (lifting TC4's refusal of a type's bounty without a points count);
    * ``pieces.price`` -- what a market type costs (a purse needs one on each, so the two
      come together until TC6 prices pieces on the board);
    * ``rules.purse.region`` -- where a bought piece is placed, named rather than the first
      rank.

    ``0`` is a price's off value and names nothing.
    """
    names: set[str] = set()
    purse = purse_rule(spec)
    if purse is not None:
        names.add("rules.purse")
        if "region" in purse:
            names.add("rules.purse.region")
    if piece_prices(spec):
        names.add("pieces.price")
    return names


def _schema_forty_six_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 46 added, by name (TC6, `docs/plan-turncoat.md`
    §5.5; numbered by order of landing, bumping no version).

    * ``pieces.price.hire`` -- a price on a type placed for nobody, under a purse: a player
      next to one may hire it (lifting TC5's refusal of a price no market charges, and of a
      purse with no market);
    * ``rules.purse.bribe`` -- an enemy piece of a priced type next to one of yours may be
      bought at twice its price.

    ``bribe: false`` is the off value and names nothing.
    """
    names: set[str] = set()
    if hire_ids(spec):
        names.add("pieces.price.hire")
    if purse_bribes(spec):
        names.add("rules.purse.bribe")
    return names


def _schema_forty_seven_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 47 added, by name (SR1a,
    `docs/plan-schema-next.md` §1 rank 1; numbered by order of landing, bumping no version).

    * ``pieces.moves.legs`` -- a lame jump's blocking squares stated (Janggi's elephant; any
      lame on a hex lattice, lifting X1);
    * ``pieces.moves.legs.leap`` -- a leap's legs, where a wall stops it under
      ``obstacles.blocks_leaps`` (on a hex lattice, lifting X8).

    The key has no off value: absent is the computed leg, and nothing written is inert.
    """
    names: set[str] = set()
    lists = [piece.get("moves") for piece in spec.get("pieces") or [] if isinstance(piece, dict)]
    portals = (spec.get("rules") or {}).get("portals")
    lists += [g.get("moves") for g in portals or [] if isinstance(g, dict)] if isinstance(
        portals, list) else []
    for moves in lists:
        for move in moves if isinstance(moves, list) else ():
            if isinstance(move, dict) and "legs" in move:
                names.add("pieces.moves.legs.leap" if move.get("kind") == "leap"
                          else "pieces.moves.legs")
    return names


def _schema_forty_eight_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 48 added, by name (SR1b,
    `docs/plan-schema-next.md` §1 rank 2; numbered by order of landing, bumping no version).

    * ``rules.explode_on_capture.blast`` -- a blast's shape stated as vectors from the
      capture square, in place of ``radius``'s box (a cross; a file; a hex ring).

    The key has no off value: absent is the radius's box, and nothing written is inert.
    """
    explosion = (spec.get("rules") or {}).get("explode_on_capture")
    if isinstance(explosion, dict) and "blast" in explosion:
        return {"rules.explode_on_capture.blast"}
    return set()


def _schema_forty_nine_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 49 added, by name (SR1c,
    `docs/plan-schema-next.md` §1 rank 5; numbered by order of landing, bumping no version).

    * ``pieces.moves.slide.path.hex`` -- a turning path (``circle`` or ``zigzag``) on a hex
      lattice, lifting PA2 (the hex rose: the hex knight's twelve, turned one step round each
      jump).

    A lifted refusal, not a key: the path's own names are generation 34's.
    """
    if (spec.get("board") or {}).get("lattice") != "hex":
        return set()
    for piece in _listed(spec.get("pieces")):
        for move in _listed(piece.get("moves") if isinstance(piece, dict) else None):
            if (
                isinstance(move, dict) and move.get("kind") == "slide"
                and move.get("path") in ("circle", "zigzag")
            ):
                return {"pieces.moves.slide.path.hex"}
    return set()


def _schema_fifty_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 50 added, by name (SR2,
    `docs/plan-schema-next.md` §3.2; numbered by order of landing, bumping no version). Nine
    lifted refusals, each a pair the probe found the core already plays, and no key:

    * ``pieces.control.majority.turn`` -- majority control in a turn longer than one move
      (L5): the holder is read at each move's start;
    * ``rules.purse.turn`` -- a purse in a turn longer than one move (L6): each buy is a move;
    * ``rules.strike.drops`` -- a shot beside a hand (T2): its victim goes to the hand;
    * ``rules.contacts.drops`` -- removing links beside a hand (T5): the target goes to it;
    * ``rules.promotion.only_lost.drops`` -- ``only_lost`` beside a hand that fills from
      captures (H1): a piece in hand is lost;
    * ``rules.promotion.only_lost.home`` -- ``only_lost`` beside a circe strike (H2): a
      reborn piece is not;
    * ``rules.drops.either_face.explode`` -- ``either_face`` beside a blast (H6): the blast's
      victims enter no hand;
    * ``rules.duck.sweep`` -- a player's sweep beside the duck (W9): the duck blocks it;
    * ``pieces.moves.slide.path.pounce`` -- a turning path on a wild type that pounces on its
      prey (N20), which reads its own capture moves.
    """
    rules = spec.get("rules") if isinstance(spec.get("rules"), dict) else {}
    assert isinstance(rules, dict)
    found: set[str] = set()
    long_turn = turn_is_long(rules)
    if long_turn and majority_ids(spec):
        found.add("pieces.control.majority.turn")
    if long_turn and purse_rule(spec) is not None:
        found.add("rules.purse.turn")
    drops = rules.get("drops")
    hand = isinstance(drops, dict) and bool(drops)
    if hand and strike_is_live(rules.get("strike")):
        found.add("rules.strike.drops")
    contacts = rules.get("contacts")
    if hand and isinstance(contacts, dict) and contacts.get("links"):
        found.add("rules.contacts.drops")
    if any(isinstance(e, dict) and e.get("only_lost") is True for e in _promotion_entries(rules)):
        if hand and isinstance(drops, dict) and drops.get("from_captures", True) is not False:
            found.add("rules.promotion.only_lost.drops")
        strike = rules.get("strike")
        if isinstance(strike, dict) and strike.get("position") == "home":
            found.add("rules.promotion.only_lost.home")
    if (
        isinstance(drops, dict) and drops.get("either_face") is True
        and rules.get("explode_on_capture")
    ):
        found.add("rules.drops.either_face.explode")
    if duck_piece(rules) is not None and uses_sweep(spec):
        found.add("rules.duck.sweep")
    preying = {
        npc.get("piece") for npc in _listed(rules.get("npcs")) if npc_pounce(npc) == "prey"
    }
    for piece in _listed(spec.get("pieces")):
        if not isinstance(piece, dict) or piece.get("id") not in preying:
            continue
        for move in _listed(piece.get("moves")):
            if isinstance(move, dict) and move.get("path") in ("circle", "zigzag"):
                found.add("pieces.moves.slide.path.pounce")
    return found


def _schema_fifty_one_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that generation 51 added, by name (SR2b,
    `docs/plan-schema-next.md` §3.2; numbered by order of landing, bumping no version). Two
    lifted refusals, each a pair the probe found the core already plays, and no key:

    * ``pieces.sight.command.strike`` -- a command beside a live strike (V5): a commanded
      move lands only on empty squares, so the strike never meets it;
    * ``pieces.sight.command.contacts`` -- a command beside contacts (V6), as V5.

    Both stay refused beside ``captures_own`` (SR4h), so neither names a turncoat command.
    """
    rules = spec.get("rules") if isinstance(spec.get("rules"), dict) else {}
    assert isinstance(rules, dict)
    found: set[str] = set()
    sights = [sight for entries in piece_sights(spec).values() for sight in entries]
    if not any(sight.effect == "command" for sight in sights):
        return found
    if strike_is_live(rules.get("strike")):
        found.add("pieces.sight.command.strike")
    contacts = rules.get("contacts")
    if isinstance(contacts, dict) and contacts.get("links"):
        found.add("pieces.sight.command.contacts")
    return found


_COLLECTORS = {
    9: _schema_nine_uses,
    10: _schema_ten_uses,
    11: _schema_eleven_uses,
    12: _schema_twelve_uses,
    13: _schema_thirteen_uses,
    14: _schema_fourteen_uses,
    15: _schema_fifteen_uses,
    16: _schema_sixteen_uses,
    17: _schema_seventeen_uses,
    18: _schema_eighteen_uses,
    19: _schema_nineteen_uses,
    20: _schema_twenty_uses,
    21: _schema_twenty_one_uses,
    22: _schema_twenty_two_uses,
    23: _schema_twenty_three_uses,
    24: _schema_twenty_four_uses,
    25: _schema_twenty_five_uses,
    26: _schema_twenty_six_uses,
    27: _schema_twenty_seven_uses,
    28: _schema_twenty_eight_uses,
    29: _schema_twenty_nine_uses,
    30: _schema_thirty_uses,
    31: _schema_thirty_one_uses,
    32: _schema_thirty_two_uses,
    33: _schema_thirty_three_uses,
    34: _schema_thirty_four_uses,
    35: _schema_thirty_five_uses,
    36: _schema_thirty_six_uses,
    37: _schema_thirty_seven_uses,
    38: _schema_thirty_eight_uses,
    39: _schema_thirty_nine_uses,
    40: _schema_forty_uses,
    41: _schema_forty_one_uses,
    42: _schema_forty_two_uses,
    43: _schema_forty_three_uses,
    44: _schema_forty_four_uses,
    45: _schema_forty_five_uses,
    46: _schema_forty_six_uses,
    47: _schema_forty_seven_uses,
    48: _schema_forty_eight_uses,
    49: _schema_forty_nine_uses,
    50: _schema_fifty_uses,
    51: _schema_fifty_one_uses,
}
