"""Structural complexity of a rule spec: how much a new player has to be told.

It is a fact about the rules rather than a reading of the score. The game page shows it
as *complexity*, the library sorts by it, ``chess-variant-sandbox validate`` prints it, and
``special_rules`` is an axis of the behaviour grid. Under ``v10`` it is also the
denominator of rule yield, the elegance family, which the user took out of the score
from ``v11`` on.

Everything here is counted off the spec itself, never off gameplay. That separation
is the whole reason rules are declarative data: a game cannot smuggle complexity past
this module into procedural code, so "how much rule is there?" has an honest answer.

Raw counts only. Normalisation against the calibration rulesets happens in
``chess_variant_sandbox.metrics`` so that the scale lives with the weights that use it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from chess_variant_sandbox.spec.canonical import canonical_json


@dataclass(frozen=True)
class SpecComplexity:
    """Raw structural counts for one spec."""

    piece_types: int
    move_descriptors: int
    distinct_directions: int
    conditional_moves: int
    # How many pieces restrict what they may capture (schema 3). Zero for every ruleset
    # written before the capture matrix existed, which is why adding this moved no
    # measurement and no hash.
    capture_restrictions: int
    facing_pieces: int
    special_rules: float
    spec_bytes: int
    description_words: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def rule_mass(self) -> float:
        """A single scalar standing in for "how much game is there to learn".

        Weighted towards the things a human actually has to hold in their head: each
        piece is a concept, each conditional or special rule is an exception, and
        exceptions cost far more than another direction on an existing piece.
        """
        return (
            3.0 * self.piece_types
            + 1.0 * self.move_descriptors
            + 0.25 * self.distinct_directions
            + 4.0 * self.conditional_moves
            + 2.0 * self.capture_restrictions
            + 2.0 * self.facing_pieces
            + 6.0 * self.special_rules
        )


# Every ``rules`` key that a player has to be *told about*, and what it costs.
#
# This used to be five hardcoded ``if`` statements, which meant every rule added to the
# schema after them was free: a spec could take on drops, regions or a bare-royal win
# and measure exactly as complex as one without. Nothing failed -- the spec validated,
# the engine played it, and the elegance term quietly rewarded it. That is the same
# shape as "a feature the validator accepts is a promise the engine has to keep": an
# uncounted feature makes one game read simpler than another that states the same
# rule. Under ``v10``, where rule mass is a *denominator* in ``rule_yield``, it also
# inflated the ratio.
#
# So the table is exhaustive over ``schema._RULES_KEYS`` and a test enforces that.
# Adding a rule to the schema without deciding what it costs is now a failure, not an
# omission.
#
# Costs are in "exceptions a new player must learn". A flag that changes one ending is
# one; a subsystem with its own placement, timing and legality rules is more.
_RULE_COSTS: dict[str, float] = {
    # Bookkeeping. Every game needs a way to stop; none of these is a rule anyone
    # learns, so they cost nothing.
    "max_plies": 0.0,
    "repetition_draw": 0.0,
    "no_progress_draw": 0.0,
    # Endings. Costed only when they differ from the default, handled below.
    "win": 1.0,
    "stalemate": 1.0,
    "checkmate": 1.0,
    # Named subsystems, each a paragraph in a rulebook. Schema 15's ``on_leaving`` is not
    # costed on top, on the grounds `screen_range` and a turn `ramp` are not: "a move out
    # of the zone promotes too" is a clause of the promotion sentence rather than a rule
    # of its own, and a player learns it with the zone. Its own feature in `distance.py`
    # is what keeps two specs differing only by it from reading as one game.
    "promotion": 1.0,
    "en_passant": 1.0,
    "castling": 1.0,
    # Schema 2. Drops and regions are whole subsystems -- where you may place, which
    # squares a piece may occupy -- and cost accordingly. The rest are single
    # exceptions.
    "drops": 2.0,
    "reach": 1.0,
    "extinction": 1.0,
    "bare_royal": 1.0,
    "check_limit": 1.0,
    "must_capture": 1.0,
    "forbid_check": 1.0,
    "royal_facing": 1.0,
    "explode_on_capture": 1.5,
    "forbid_immobility": 1.0,
    # Schema 3. "Whoever has more of this when the clock runs out wins" is one sentence
    # for material or a headcount; naming a region to hold is a second, because the
    # region has to be described as well as counted.
    "scoring": 1.5,
    # Schema 4. A board that remembers who has been where is not one exception, it is a
    # second thing going on at once -- the designer whose sequence prompted it rejected
    # the version that had both, saying "it felt like I was playing three different
    # games at the same time". Costed accordingly.
    "territory": 2.5,
    # Schema 4. "You make two moves per turn" is one sentence, and one of the shortest
    # in the schema -- rule mass measures what a player has to be *told*, not how hard
    # the game then is to play, and a longer turn is the extreme case of that gap. The
    # opening ramp is not costed on top, on the same grounds as `screen` and
    # `screen_range`: it parameterises a rule already counted rather than adding an
    # exception to it.
    "turn": 1.0,
    # Schema 5. One sentence to state -- "a piece that leaves one edge comes back on the
    # opposite one" -- and unlike a longer turn it does not stop there: the sentence
    # changes what every line on the board means, so a player has to re-read the whole
    # position rather than learn an exception to it. Costed with `explode_on_capture`
    # and `scoring` rather than with the single flags. An asymmetric seam is a second
    # sentence and is not costed twice, on the same grounds as `screen` and a turn
    # `ramp`: it parameterises a rule already counted.
    "wrap": 1.5,
    # Schema 6. One sentence -- "these squares are walls" -- and it is a fact about the
    # board a player reads off the diagram rather than an exception they have to
    # remember mid-game, which is what puts it at the bottom of this table beside the
    # single flags. ``blocks_leaps`` is not costed on top: it parameterises a rule
    # already counted, on the same grounds as ``screen_range`` and a turn ``ramp``.
    "obstacles": 1.0,
    # Schema 6. Three sentences, not one: what a strike does, the chain a player has to
    # learn piece by piece, and where the two pieces end up afterwards. That is a
    # paragraph, and it lands nearer ``drops`` than ``reach`` -- and unlike a wall it
    # cannot be read off the board, because the whole vocabulary of exchange changes
    # without anything looking different.
    "strike": 2.0,
    # Schema 10 (Q19a). A contact table is a strike with the attacker's column filled in,
    # and it is charged half a point more than one for exactly that: the three sentences
    # a strike costs are still owed, and the fourth is that *which* transformation
    # happens depends on who took what -- so a player learns the table by pairs rather
    # than by pieces, which is the same jump `captures` makes over a single flag. Not
    # costed on top of `strike`, because the validator refuses the two together: a spec
    # pays 2.0 or 2.5, never both.
    "contacts": 2.5,
    # Schema 9. A wild piece is literally "a second thing going on at once", which is the
    # `territory` argument, and a player has to be told four things about it -- what it
    # is, where it walks, what it eats and what taking it is worth -- plus one more that
    # changes how they read every position: a royal standing where it steps next is in
    # check. Costed with `territory` rather than with `strike`, because a strike changes
    # what a move does and a wild piece adds an actor.
    #
    # Schema 10's `npcs[].immune` is **not** costed on top, on the same grounds as
    # `screen`, a turn `ramp` and `blocks_leaps`: it parameterises a rule already counted
    # rather than adding an exception to it, and "only a knight may take it" is part of
    # the same sentence as "it eats what it walks into". The shorter spelling is the point
    # of the feature and it is not this table's doing -- what it removes is five per-piece
    # `captures` lists, each of which `measure_complexity` counts as a
    # `capture_restriction` and each of which is real rule text a player has to read.
    #
    # Schema 23's `npcs[].also_at` is **not** costed either, on `immune`'s grounds and on
    # `mirror`'s: a second wolf on the patrol is one more actor of a kind the rule already
    # counts, as a mirrored entry's second wolf is, and what a player is told about it is
    # the clause "and another starts on h5", plus the queue rule (one waits behind
    # another), which is the waiting rule they were already told.
    "npcs": 2.5,
    # Schema 9. One sentence -- "a pawn dropped or struck onto the far rank is a queen at
    # once" -- and an exception rather than a parameter of `promotion`: it changes what a
    # drop and a strike *do* in one part of the board, and a player who has not been told
    # will drop a pawn to block and find a queen there. Costed with `en_passant` and the
    # single flags. `false` is the absence of the rule and costs nothing, on the grounds
    # `"en_passant": false` costs nothing.
    "promote_on_entry": 1.0,
    # Schema 19 (Q771). A portal pair is a new actor on the board in the `npcs` sense -- a
    # square that sends a piece somewhere else, with a direction rule a player has to be
    # taught (it comes out one step beyond, the way it was going) and a set of blocked
    # exits -- so it is costed as `npcs` is. Each further pair, a stated mover and a
    # stated one-way or closed passage are charged on top by `_portal_extras`.
    "portals": 2.5,
    # Schema 26 (Q918a). Both players move at once: the largest change to what a turn is
    # the language has, and a player has to be taught four things -- the round, that only
    # the same square is a collision and who survives one, that a capture whose victim ran
    # lands or is void, and the void bar -- so it costs more than `territory` and `npcs`.
    # The tie rule is one clause of the same sentence and costs nothing more; `beats` is
    # costed per piece, as `captures` is (`capture_restrictions`).
    "simultaneous": 3.0,
}

# Schema 19: what `rules.portals` costs beyond its first pair, and why. A further pair is
# one more place on the board a player has to read the numbers of (0.5 each); someone
# moving an end is a second kind of move, the portal's own (1.0, once); a passage that is
# one way or closed is an exception per player about which end is a door (1.0, once).
# Defaults cost nothing: `moved_by: []` and `both_ways` are the rule as first stated.
# Schema 20 (Q807): a bar on moving a portal straight back is one exception to remember
# about the portal's own move, a ko (0.5, once); `false` is the rule's absence.
_PORTAL_PAIR = 0.5
_PORTAL_MOVED = 1.0
_PORTAL_PASSAGE = 1.0
_PORTAL_NO_BACK = 0.5
# Schema 21 (Q843a, `docs/archive/plan-portals-2.md` §6): a map written with `ends` is a second reading
# of where a portal leads -- not "the other one" but "the one its arrow names" -- learnt
# once (1.0), and each end past a pair's two is one more place to read (0.25 each); a
# filter on which pieces may go in is one exception per end a player has to remember,
# charged once (1.0) as a passage is. A pair, the rule as first stated, costs as before.
_PORTAL_MAP = 1.0
_PORTAL_END = 0.25
_PORTAL_FILTER = 1.0
# Schema 22 (Q843c, `docs/archive/plan-portals-2.md` §6), revised at schema 24 (Q867b, §3R): a fold
# group is the second rule set -- a portal a piece may stand in, where a move that ends on
# an empty one may stay or come out on the end it leads to -- read per group, as the plan
# charges it: each fold group is one more place where a move does not end where the board
# says (1.0 each; the revision left the weight as it was).
_PORTAL_FOLD = 1.0


# Schema 27 (Q942a, `docs/archive/plan-round-chase.md` §3): each chase switch is one more sentence
# a player is taught about the round -- a swap is a collision, a double ending goes to the
# score -- so each is charged as one exception on top of `simultaneous`, and only at the
# value that turns it on (the validator refuses the others). The third, the round's rest,
# became schema 29's `pieces[].rest` (`_rest_extras`) at the same charge.
_SIMULTANEOUS_SWITCH = 1.0
_SIMULTANEOUS_SWITCHES = {"swap": "collision", "both": "score"}

# Schema 29 (Q983a, `docs/plan-rest.md` §5.7): the rest is one rule a player is taught, charged
# once at the switch's cost it replaced, plus a little for each distinct rest beyond the first
# (a fact about pieces a player has to remember), so a uniform `"rest": 1` costs exactly what
# `rules.simultaneous.rest` did. Read off the spelling, as the spy and the speeds are.
_REST = _SIMULTANEOUS_SWITCH
_REST_PER_VALUE = 0.25


def _rest_extras(pieces: Any) -> float:
    """What a game's rests cost: the rule once and each distinct rest beyond the first."""
    values = {
        piece["rest"] for piece in (pieces if isinstance(pieces, list) else [])
        if isinstance(piece, dict) and isinstance(piece.get("rest"), int)
        and not isinstance(piece.get("rest"), bool) and piece["rest"] > 0
    }
    return _REST + _REST_PER_VALUE * (len(values) - 1) if values else 0.0


# Schema 27 (Q935a, `docs/plan-spy.md` §2.6): a spy is a piece flag that changes the turn
# structure, so it is charged as a rule, not as a piece property: a player has to be taught
# three phases, the reply turn and the announced round, on top of the round `simultaneous`
# already costs. Once for the game however many types are flagged, read off the spelling as
# the switches are (this file is in `measurement.METRICS_SOURCES`, `schema.py` is not).
_SPY = 2.0


def _spy_extras(pieces: Any) -> float:
    """What a spy game's spy costs, beside `rules.simultaneous`' own charge."""
    flagged = any(
        isinstance(piece, dict) and piece.get("spy") is True
        for piece in (pieces if isinstance(pieces, list) else [])
    )
    return _SPY if flagged else 0.0


# Schema 28 (Q943a, `docs/plan-speed.md` §4.3): speed is one more rule a player is taught
# about the round -- unequal speeds move in order, the slower re-read -- charged once for the
# game, plus a little for each type that declares a speed, since each is a fact about a piece
# a player has to remember (the hierarchy is the design). Read off the spelling, as the spy is.
_SPEED = 1.0
_SPEED_PER_TYPE = 0.25


def _speed_extras(pieces: Any) -> float:
    """What a game's speeds cost: the rule once and each declared speed."""
    declared = sum(
        1 for piece in (pieces if isinstance(pieces, list) else [])
        if isinstance(piece, dict) and isinstance(piece.get("speed"), int)
        and not isinstance(piece.get("speed"), bool) and piece["speed"] > 0
    )
    return _SPEED + _SPEED_PER_TYPE * declared if declared else 0.0


# Schema 30 (Q999a, `docs/plan-passive.md` §4.3): each effect a game's sights use is one rule a
# player is taught (what a frozen, disarmed or shielded piece may do, and how check reads it),
# charged once for the game; each sight entry beyond the first of an effect is a fact about a
# piece a player has to remember; an aura (`through`) is one more sentence, once for the game.
# Read off the spelling, as the rests and the speeds are (this file is in
# `measurement.METRICS_SOURCES`, `schema.py` and its `piece_sights` are not).
_SIGHT_EFFECT = 1.0
_SIGHT_PER_ENTRY = 0.25
_SIGHT_AURA = 0.25


def _sight_extras(pieces: Any) -> float:
    """What a game's sights cost: each distinct effect, each further entry of an effect, and
    an aura once."""
    per_effect: dict[str, int] = {}
    aura = False
    for piece in pieces if isinstance(pieces, list) else []:
        sight = piece.get("sight") if isinstance(piece, dict) else None
        for entry in sight if isinstance(sight, list) else []:
            if not isinstance(entry, dict) or not isinstance(entry.get("effect"), str):
                continue
            per_effect[entry["effect"]] = per_effect.get(entry["effect"], 0) + 1
            aura |= entry.get("through") is True
    return (
        _SIGHT_EFFECT * len(per_effect)
        + _SIGHT_PER_ENTRY * sum(count - 1 for count in per_effect.values())
        + (_SIGHT_AURA if aura else 0.0)
    )


def _simultaneous_extras(rule: Any) -> float:
    """The part of `rules.simultaneous`' charge `_RULE_COSTS` cannot say (schema 27).

    Read off the spelling here, as `_portal_extras` is, since this file is in
    `measurement.METRICS_SOURCES` and `schema.py` is not."""
    if not isinstance(rule, dict):
        return 0.0
    return _SIMULTANEOUS_SWITCH * sum(
        1 for key, on in _SIMULTANEOUS_SWITCHES.items() if key in rule and rule[key] == on
    )


def _portal_extras(portals: Any) -> float:
    """The part of `rules.portals`' charge `_RULE_COSTS` cannot say: see `_PORTAL_PAIR`.

    Read off the spelling here rather than through `schema.portal_groups`: this file is in
    `measurement.METRICS_SOURCES` and `schema.py` is not, so a score must not depend on it
    (`tests/test_measurement.py`). The charges are the same either way.
    """
    if not isinstance(portals, list) or not portals:
        return 0.0
    groups = [group for group in portals if isinstance(group, dict)]
    total = _PORTAL_PAIR * max(len(groups) - 1, 0)
    pairs = [group for group in groups if "ends" not in group]
    maps = [
        [end for end in group["ends"] if isinstance(end, dict)]
        for group in groups if isinstance(group.get("ends"), list)
    ]
    if any(
        any((pair.get("moved_by") or {}).get(end) for end in ("orange", "blue"))
        for pair in pairs
        if isinstance(pair.get("moved_by"), dict)
    ) or any(end.get("moved_by") for ends in maps for end in ends):
        total += _PORTAL_MOVED
    if any(
        word != "both_ways"
        for pair in pairs
        if isinstance(pair.get("passage"), dict)
        for word in pair["passage"].values()
    ) or any("enter" in end for ends in maps for end in ends):
        total += _PORTAL_PASSAGE
    if any(group.get("no_straight_back") is True for group in groups):
        total += _PORTAL_NO_BACK
    if maps:
        total += _PORTAL_MAP + _PORTAL_END * sum(max(len(ends) - 2, 0) for ends in maps)
    if any("carries" in pair for pair in pairs) or any(
        "carries" in end for ends in maps for end in ends
    ):
        total += _PORTAL_FILTER
    total += _PORTAL_FOLD * sum(1 for group in groups if group.get("kind") == "fold")
    return total

# Keys whose cost applies only when they depart from the engine's default. Stating
# "checkmate loses" is not a rule anyone has to learn.
_RULE_DEFAULTS: dict[str, Any] = {
    "win": "checkmate",
    "stalemate": "draw",
    "checkmate": "loss",
    "royal_facing": "allowed",
}

# Every key `rules.wrap` states a seam under: a grid's two axes and, since schema 16
# (Q281), a hexagon's three pairs of sides. `schema.wrap_axes` is the source; this file
# spells them out rather than importing it, because `complexity.py` is in the `metrics`
# fingerprint and `schema.py` is not, so a score must not read through it.
# `tests/test_complexity.py` holds the two lists together.
_WRAP_KEYS = ("files", "ranks", "n_s", "ne_sw", "nw_se")

# Keys whose presence is not enough: the *value* decides whether there is a rule here at
# all. `_RULE_DEFAULTS` cannot express this because it compares for equality, and a turn
# rule has more than one way to say "one move, like every other game" -- `{"moves": 1}`
# and `{"moves": 1, "ramp": [1, 1]}` are both the game that was already there, and the
# compiler drops both.
_RULE_ACTIVE: dict[str, Any] = {
    # Schema 26: present as an object is the rule, and `{}` is its whole default spelling,
    # which the truthiness test below would read as absent.
    "simultaneous": lambda value: isinstance(value, dict),
    "turn": lambda value: isinstance(value, dict)
    and (
        int(value.get("moves", 1)) > 1
        or any(int(length) > 1 for length in value.get("ramp") or ())
    ),
    # Same shape as `turn`, and for the same reason: a board with edges has several
    # spellings -- absent, `{"files": "none"}`, `{"ranks_by_player": {"0": "none",
    # "1": "none"}}` -- and the compiler drops all of them. Over both shapes' keys
    # (`_WRAP_KEYS`): a valid spec states only its own shape's, so the union is exact.
    "wrap": lambda value: isinstance(value, dict)
    and any(
        edge not in (None, "none")
        for key in _WRAP_KEYS
        for edge in (
            [value.get(key)]
            + list((value.get(f"{key}_by_player") or {}).values())
        )
    ),
    # Same shape again. ``{"effect": "capture"}`` is the game that was already there --
    # a taken piece is removed -- and so is a chain whose every ``from`` is immune,
    # because a strike never reaches an immune piece. The compiler drops both
    # (``StrikeRule.survives_anything``). What makes a strike a rule is one link that
    # leaves a struck piece standing.
    #
    # Schema 17 (Q540) makes two `"capture"` positions rules -- `stay` (rifle) and `home`
    # (circe) -- and `switches_sides` (Andernach) a rule under any effect. Each costs the
    # strike's one charge: it is one sentence about what a capture leaves behind.
    "strike": lambda value: isinstance(value, dict)
    and (
        (
            value.get("effect", "capture") != "capture"
            and any(
                link.get("from") not in (value.get("immune") or ())
                for link in value.get("chain") or ()
            )
        )
        or (
            value.get("effect", "capture") == "capture"
            and value.get("position") in ("stay", "home")
        )
        or bool(value.get("switches_sides"))
    ),
    # Schema 10 (Q19a), the same shape one spelling over. A table with no link is no rule
    # (`docs/reference/bugs-investigations.md` §*Ruled on in advance, for Q86*), and
    # neither is one whose every target is immune, because a contact reaches an immune piece only through `own`. The
    # compiler drops both (``StrikeRule.survives_anything`` or ``changes_mover``). What
    # makes a table a rule is one link that reaches its target and changes a piece.
    "contacts": lambda value: isinstance(value, dict)
    and any(
        (link.get("own") is True or link.get("onto") not in (value.get("immune") or ()))
        and ("onto_to" in link or "mover_to" in link)
        for link in value.get("links") or ()
    ),
    # Schema 10 (Q18), and Q56 ruled on it before the key existed
    # (`docs/reference/bugs-investigations.md`). `by` is the capturer types whose captures explode and `of` the victim types whose being taken
    # does; an *empty* list of either is a rule no capture can fire -- the `captures: []`
    # shape one level up -- and the compiler drops the whole explosion for it. Without
    # this clause `{"by": []}` would be charged 1.5 for a no-op, and a wrong charge
    # fails silently: one game reads as two complexities. Under `v10`, where `rule_mass`
    # is the denominator of `rule_yield`, it moved the ratio the index rewarded.
    #
    # A list naming *every* type is the other degenerate case and is charged in full,
    # correctly: that is atomic's explosion written the long way and it is a real rule.
    # ``{}`` and ``false`` stay free, as they were: the compiler returns no rule for
    # either, so the clause has to keep the truthiness test it replaces rather than
    # only adding to it.
    "explode_on_capture": lambda value: isinstance(value, dict)
    and bool(value)
    and all(value.get(key) is None or bool(value.get(key)) for key in ("by", "of")),
}


def _count_special_rules(rules: dict[str, Any]) -> float:
    """Weighted count of the exceptions a player has to be taught.

    Falsy values do not count: ``"en_passant": false`` is the absence of a rule, and
    a spec that spells out a default should measure the same as one that omits it.
    """
    total = 0.0
    for key, cost in _RULE_COSTS.items():
        if key not in rules:
            continue
        value = rules[key]
        if key in _RULE_DEFAULTS:
            if value != _RULE_DEFAULTS[key]:
                total += cost
            continue
        if key in _RULE_ACTIVE:
            if _RULE_ACTIVE[key](value):
                total += cost
            continue
        if value:
            total += cost
    total += _portal_extras(rules.get("portals"))
    total += _simultaneous_extras(rules.get("simultaneous"))
    return total


def measure_complexity(spec: dict[str, Any]) -> SpecComplexity:
    """Count the structural features of a validated spec."""
    pieces = spec.get("pieces", [])
    rules = spec.get("rules", {})

    move_descriptors = 0
    conditional_moves = 0
    capture_restrictions = 0
    facing_pieces = 0
    directions: set[tuple[int, int]] = set()

    for piece in pieces:
        # A piece that names what it may take is a line in the rules table. Counted per
        # piece rather than per pair: rock-paper-scissors is three lines, not nine, and
        # a matrix that grew quadratically would make any cyclic design read as a heavy
        # rulebook (and, under `v10`, unaffordable on elegance) before it was played.
        #
        # Costed at 2.0 -- between a direction and a conditional move. A hop's `screen`
        # field is deliberately *not* costed separately, for the same reason `range` and
        # `screen_range` are not: it parameterises a primitive already counted as a move
        # descriptor, rather than adding an exception on top of one.
        if isinstance(piece.get("captures"), list):
            capture_restrictions += 1
        # Schema 26: a piece's collision table is a second such line, beside `captures`.
        if isinstance(piece.get("beats"), list):
            capture_restrictions += 1
        # A facing is one more line in the rules table for this piece, on exactly the
        # argument `captures` is costed by: an exception a player has to be taught about
        # *that piece*, and one sentence however many orientations it turns out to have.
        # Its own field rather than a second meaning for `capture_restrictions`, and
        # deliberately not in `_RULE_COSTS`, which is a table over `rules` -- this is a
        # piece property, counted where `captures` and `region` are counted.
        #
        # Zero for every ruleset written before schema 8, so `rule_mass` -- under `v10`
        # the denominator of `rule_yield` -- does not move for any of them.
        if isinstance(piece.get("facing"), dict):
            facing_pieces += 1
        for move in piece.get("moves", []):
            move_descriptors += 1
            if move.get("condition"):
                conditional_moves += 1
            symmetry = move.get("symmetry", "none")
            for vector in move.get("vectors", []):
                directions.update(expand_symmetry(tuple(vector), symmetry))

    special_rules = (
        _count_special_rules(rules) + _spy_extras(pieces) + _speed_extras(pieces)
        + _rest_extras(pieces) + _sight_extras(pieces)
    )

    # Schema 12: a `landing` set is charged as a descriptor is, per set and per expanded
    # direction, so a wide throw costs rule mass. Its directions are counted apart from the
    # pieces', because a throw is not a move a piece already has. A landing beside a rule
    # the compiler drops throws nothing and costs nothing.
    landing_directions: set[tuple[int, int]] = set()
    for key in ("strike", "contacts"):
        rule = rules.get(key)
        landing = rule.get("landing") if _RULE_ACTIVE[key](rule) else None
        if isinstance(landing, dict):
            move_descriptors += 1
            for vector in landing.get("vectors", []):
                landing_directions |= expand_symmetry(
                    tuple(vector), landing.get("symmetry", "none")
                )

    return SpecComplexity(
        piece_types=len(pieces),
        move_descriptors=move_descriptors,
        distinct_directions=len(directions) + len(landing_directions),
        conditional_moves=conditional_moves,
        capture_restrictions=capture_restrictions,
        facing_pieces=facing_pieces,
        special_rules=special_rules,
        spec_bytes=len(canonical_json(spec, mechanics_only=True).encode("utf-8")),
        description_words=len(str(spec.get("description", "")).split()),
    )


def expand_symmetry(vector: tuple[int, int], symmetry: str) -> set[tuple[int, int]]:
    """Expand one base vector into every direction its symmetry group implies.

    Vectors are written in the first player's orientation: ``+y`` is forward. The
    second player's directions are produced by the engine at compile time, not here,
    so that ``distinct_directions`` counts concepts rather than doubling them.

    ``rotate6`` and ``dihedral12`` (schema 18b) are the hexagon's, read in the hex
    lattice's axial coordinates (file ``q``, rank ``r``): a turn by 60 degrees is
    ``(q, r) -> (-r, q + r)``, which carries the six neighbours round in order, and the
    mirror is ``(q, r) -> (r, q)``. Any one mirror generates the same twelve, so the
    choice of axis changes nothing. Both sets are closed under the half turn, which is
    the hex board's second-player orientation, so the engine's flip leaves them as they
    are.
    """
    x, y = vector
    if symmetry == "none":
        return {(x, y)}
    if symmetry == "lateral":
        return {(x, y), (-x, y)}
    if symmetry == "rotate4":
        return {(x, y), (-y, x), (-x, -y), (y, -x)}
    if symmetry == "dihedral8":
        rotations = {(x, y), (-y, x), (-x, -y), (y, -x)}
        return rotations | {(a, -b) for a, b in rotations}
    if symmetry in ("rotate6", "dihedral12"):
        turns = {(x, y), (-y, x + y), (-x - y, x), (-x, -y), (y, -x - y), (x + y, -x)}
        if symmetry == "rotate6":
            return turns
        return turns | {(b, a) for a, b in turns}
    raise ValueError(f"unknown symmetry {symmetry!r}")
