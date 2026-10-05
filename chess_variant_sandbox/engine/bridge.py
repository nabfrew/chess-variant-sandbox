"""Flatten a ``CompiledSpec`` into the arrays the Rust core reads.

The rule DSL is interpreted in exactly one place -- ``compile.py`` -- and this module
is the only thing that crosses the language boundary with the result. That split is
deliberate. Reimplementing the compiler in Rust would double the surface on which the
two sides could disagree about what a rule *means*; as it stands the only thing
duplicated is how a table is walked, and perft tests that directly.

Everything here is a flat list of integers rather than the nested tuples the compiler
produces. Two reasons: the conversion is one bulk extraction per array instead of a
Python object per entry, and the layout is already the one the generator wants, so the
core does no rearranging at construction time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from chess_variant_sandbox.engine.compile import CompiledSpec

# A stated ending result, as the integer the core carries: a loss for the side the rule
# describes, a draw, or a win for them. ``ENDING_RESULTS`` promises all three, so all
# three have to survive the crossing -- a boolean here turned "draw" into whichever of
# the other two it was not, silently, in three different rules.
_ENDING_RESULT = {"loss": -1, "draw": 0, "win": 1}

# What a scoring rule counts, as the integer the core carries.
_SCORING_COUNT = {"material": 0, "pieces": 1, "region": 2, "territory": 3, "points": 4}

# Where the two pieces end up after a strike, as the integer the core carries. The
# *effect* has no number: ``demote`` and ``promote`` differ only in which way the chain
# points, and the compiler has already resolved the chain into ``strike_to``.
_STRIKE_POSITION = {"stay": 0, "advance": 1, "swap": 2, "choose": 3}

# Whether ``rust/src/game.rs`` can put a struck piece somewhere other than off the board.
# Set false in the commit that taught the compiler the rule and true in the one that
# taught the core, so the intermediate state raised instead of playing plain capture
# under a spec that asked for something else. Kept as a named constant rather than
# deleted with the refusal because it is the honest record of that ordering.
_CORE_KNOWS_STRIKE = True

# Whether ``rust/src/game.rs`` can read a facing out of a piece value, walk the frame it
# names, and generate the move that changes it. Same shape and same purpose as the line
# above: false in the commit that taught the compiler the rule, true in the one that
# taught the core, so the intermediate state raises rather than playing a game in which
# every piece faces forward whatever its board byte says.
#
# It matters more here than it did for the strike. A strike that the core ignored would
# have played plain capture, which is a different game and a visible one. A facing the
# core ignored is *silent*: the board bytes are legal, the tables are the right length,
# and a piece would simply read frame 0 of a table that has eight -- moving the way it
# was drawn pointing and not the way it is pointing.
_CORE_KNOWS_FACING = True

# Whether ``rust/src/game.rs`` can hold a piece neither player owns: a third owner's
# occupancy set and value rows, the phase inside ``make`` that walks every wild piece
# along its route, the undo log that puts them back, the third block of attack sources
# the check filter has to walk, and the tally a bounty is paid into. Same shape and same
# purpose as the two lines above: false in the commit that taught the compiler the rule
# (queue task Q9a), true in the one that teaches the core (Q9b), so the intermediate
# state raises rather than plays.
#
# It matters more here than for either of those. A strike the core ignored played plain
# capture, which is a different game and a visible one. A facing it ignored played every
# piece forward, which is silent. A wild piece it did not know would be a byte it has no
# row for -- `value_owner` indexed past its end, an occupancy set that does not exist --
# and whatever that reads as is a plausible game with a piece on the board that nothing
# can see.
#
# Schema 9's second key rides the same flag. ``rules.promote_on_entry`` (Q13) compiles
# to a table the core has to read at every place a piece arrives on a square without a
# move of its own -- the drop branch of ``make``, ``strike_writes``, the immobility
# filter's drop arm, ``is_forbidden_drop_mate`` and the legality filter's own drop arm;
# `docs/archive/plan-schema-6-9.md` C6 names four of those five -- and until it does, a
# dropped or struck pawn on the far rank would stand there under a spec that said it
# should be a queen.
# That is the strike's failure mode (a visible, different game) rather than the wild
# piece's, but it is a plausible wrong game all the same.
#
# **Lifted once, in Q9b, for the whole of schema 9** -- both keys, in the commit that
# taught the core the phase and ``Tables::arriving_value``. Ten holds, zero plausible
# wrong games played, and three of the ten retired by teaching the core the thing they
# guarded, which is what a guard is for.
_CORE_KNOWS_NPCS = True

# Schema 10 (Q18): whether ``game.rs`` reads ``explosion_by`` / ``explosion_of``.
#
# The eleventh hold, and the same shape as the ten before it. The compiler can already
# say "only a queen's captures explode" and the core still explodes on every capture, so
# a spec that narrows the blast would play a *plausible wrong game* -- visibly atomic,
# quietly not the rules anybody wrote. Refused by name until the core walks the two
# tables. Absent ``by`` and ``of`` compile to ``None`` and are not refused, so every
# ruleset written before schema 10 -- ``atomic.json`` above all -- goes through here
# exactly as it did.
#
# **Lifted in the same task, one commit later**, when ``ExplosionRule::fires`` landed and
# the two tables reached ``make`` and the legality filter. Eleven holds, zero plausible
# wrong games played, four of the eleven retired by teaching the core the thing they
# guarded. The constant stays as the honest record of the ordering.
_CORE_KNOWS_SELECTIVE_EXPLOSION = True

# Schema 10 (Q19a): the contact table, ``rules.contacts``. The mover's column is the new
# half -- a meeting can turn the *attacker* into something else, and under ``advance`` it
# can do that with nothing left standing, which is a strike write where schema 6 had
# none. A core that read only ``strike_to`` would play the target's half of the rule and
# silently drop the attacker's, which is the failure mode this file exists to refuse:
# a plausible wrong game rather than a visible refusal.
#
# **Lifted in the task that added it**, once ``strike_writes`` took the mover's type and
# ``tables.rs`` widened the compact array on the way in. Twelve holds, zero plausible
# wrong games played, five of the twelve retired by teaching the core the thing they
# guarded. The constant stays as the honest record of the ordering.
_CORE_KNOWS_CONTACTS = True

# Schema 15 (Q452): ``promotion.on_leaving``, a piece that promotes on a move *out of* its
# zone. The compiler resolves it into ``promo_on_leaving``, and a core that never read it
# would play the arrival-only promotion it always had -- shogi's rules as written before
# Q450, and a plausible wrong game rather than a refusal. Lifted in the task that added
# it, once both of the generator's promotion sites (`push_arrival` and the hop's empty
# square) asked the origin too. The constant stays as the honest record of the ordering.
_CORE_KNOWS_PROMOTION_ON_LEAVING = True


def _flatten_entries(
    table: tuple[tuple[tuple[tuple[Any, ...], ...], ...], ...],
    n_frames: int,
    n_types: int,
    n_squares: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """`[frame][type][square] -> entries` becomes four parallel arrays.

    ``index`` holds one offset per slot plus a terminator, so a slot's entries are
    ``refs[index[slot] : index[slot + 1]]`` with no length arithmetic in the hot loop.

    The first index is a *frame* rather than a player -- see ``compile.frame_index``.
    They are the same thing today; the loop is over ``n_frames`` so that they can stop
    being the same thing without this function changing.
    """
    index = [0]
    refs: list[int] = []
    squares: list[int] = []
    modes: list[int] = []

    for frame in range(n_frames):
        for piece_type in range(n_types):
            for square in range(n_squares):
                for entry in table[frame][piece_type][square]:
                    # Leaps and slides carry three fields; a hop carries two more, how
                    # far past its screen it may land and whose piece that screen has to
                    # be. Both travel rather than being dropped: without the first the
                    # core cannot tell a grasshopper from a cannon, and without the
                    # second it cannot tell a piece that vaults its own army from one
                    # that jumps anything. Either difference is a different game rather
                    # than a failing assertion.
                    targets, mode_masks, requires_unmoved = entry[0], entry[1], entry[2]
                    screen_range = entry[3] if len(entry) > 3 else 0
                    screen_owner = entry[4] if len(entry) > 4 else 0
                    refs.extend(
                        (
                            len(squares),
                            len(targets),
                            int(requires_unmoved),
                            int(screen_range),
                            int(screen_owner),
                        )
                    )
                    squares.extend(targets)
                    modes.extend(mode_masks)
                index.append(len(refs) // 5)
    return index, refs, squares, modes


def build_tables(spec: CompiledSpec) -> dict[str, Any]:
    """Everything the core needs, keyed by name."""
    n_types = spec.n_types
    n_squares = spec.n_squares
    n_frames = spec.n_frames

    leap_index, leap_refs, leap_squares, leap_modes = _flatten_entries(
        spec.leaps, n_frames, n_types, n_squares
    )
    slide_index, slide_refs, slide_squares, slide_modes = _flatten_entries(
        spec.slides, n_frames, n_types, n_squares
    )
    hop_index, hop_refs, hop_squares, hop_modes = _flatten_entries(
        spec.hops, n_frames, n_types, n_squares
    )
    # Schema 13. Empty unless something sweeps, so no earlier fingerprint moves.
    has_sweep = spec.has_sweep
    sweep_index: list[int] = []
    sweep_refs: list[int] = []
    sweep_squares: list[int] = []
    sweep_modes: list[int] = []
    if has_sweep:
        sweep_index, sweep_refs, sweep_squares, sweep_modes = _flatten_entries(
            spec.sweeps, n_frames, n_types, n_squares
        )

    attack_index = [0]
    attack_srcs: list[int] = []
    attack_paths: list[int] = []
    attack_sweep: list[int] = []
    attack_path_modes: list[int] = []
    # Over however many owners the compiler built blocks for: two, or three once the
    # spec has wild pieces (schema 9), whose block the core walks with the same loop.
    for player in range(len(spec.attack_sources)):
        for square in range(n_squares):
            for entry in spec.attack_sources[player][square]:
                (
                    origin, piece_value, prefix, window, screens, unmoved, screen_owner,
                    sweep, path_modes,
                ) = entry
                path = tuple(prefix) + tuple(window)
                if has_sweep:
                    attack_sweep.append(sweep)
                    attack_path_modes.extend(path_modes if sweep else (0,) * len(path))
                # ``prefix_len`` is the seventh field and the whole reason a bounded
                # screen range is expressible: the path arrives concatenated, and without
                # a marker for where the must-be-empty prefix ends the core would read a
                # grasshopper's line as a cannon's.
                attack_srcs.extend(
                    (
                        origin,
                        piece_value,
                        screens,
                        int(unmoved),
                        len(attack_paths),
                        len(path),
                        len(prefix),
                        screen_owner,
                    )
                )
                attack_paths.extend(path)
            attack_index.append(len(attack_srcs) // 8)

    # The capture matrix, one byte per (attacker, victim) pair. Flat rather than nested
    # for the same reason everything else here is: the core wants one index, not two
    # pointer hops, on a path that runs per capture test.
    capture_allowed = [
        int(allowed) for row in spec.capture_allowed for allowed in row
    ]

    dedup = [
        int(square in spec.dedup_squares[frame][piece_type])
        for frame in range(n_frames)
        for piece_type in range(n_types)
        for square in range(n_squares)
    ]

    castling = spec.castling
    # En passant is defined against the promoting piece: the piece that advances,
    # cannot retreat, and is therefore the only one a double step can run past.
    # With multi-promotion, find the ep piece from any promotion rule that exists.
    ep_type: int = -1
    if spec.en_passant and spec.promotions:
        ep_type = spec.promotions[0].piece_type

    # Which destinations open an en passant window, and the square the mover stepped
    # over to reach each. CSR over [frame][origin], because the core reads it once per
    # quiet move of one piece type and never wants a second pointer hop. It replaced
    # arithmetic in the core -- halving the move's delta -- which is right only on a
    # board whose edges are edges.
    ep_pass_flat: list[int] = []
    ep_pass_index: list[int] = [0]
    for frame in range(n_frames):
        for origin in range(n_squares):
            for target, passed in spec.ep_passes[frame][origin]:
                ep_pass_flat.extend((target, passed))
            ep_pass_index.append(len(ep_pass_flat) // 2)

    # Which squares can see which down a file, and what has to be empty for them to.
    # CSR over [square], because the core reads one square's entries per check test and
    # never wants a second pointer hop. It replaced arithmetic in the core -- walking a
    # file by `+= width` -- which is right only on a board whose edges are edges and
    # whose empty squares are the only empty squares. Empty for every ruleset that does
    # not declare `royal_facing: forbidden`, so nothing else's bytes move.
    facing_index: list[int] = [0]
    facing_entries: list[int] = []
    facing_paths: list[int] = []
    for lines in spec.facing_lines:
        for other, path in lines:
            facing_entries.extend((other, len(facing_paths), len(path)))
            facing_paths.extend(path)
        facing_index.append(len(facing_entries) // 3)
    if not spec.facing_lines:
        facing_index = []

    # --- Promotion tables (schema 2: per-piece, per-player, optional, zones) ---
    # promo_squares: flat [player][piece_type * n_squares + square] -> 0/1
    promo_squares: list[int] = []
    for player in (0, 1):
        row = [0] * (n_types * n_squares)
        for rule in spec.promotions:
            for square in rule.squares[player]:
                row[rule.piece_type * n_squares + square] = 1
        promo_squares.extend(row)

    # promo_targets_flat + promo_targets_index: CSR-style [player * n_types + pt]
    promo_targets_flat: list[int] = []
    promo_targets_index: list[int] = [0]
    for player in (0, 1):
        targets_by_type: list[tuple[int, ...]] = [()] * n_types
        for rule in spec.promotions:
            targets_by_type[rule.piece_type] = rule.targets[player]
        for pt in range(n_types):
            promo_targets_flat.extend(targets_by_type[pt])
            promo_targets_index.append(len(promo_targets_flat))

    # promo_optional: per piece type
    promo_optional = [0] * n_types
    for rule in spec.promotions:
        promo_optional[rule.piece_type] = int(not rule.mandatory)

    # Which piece types have a promotion rule
    promo_piece_types = sorted({rule.piece_type for rule in spec.promotions})
    # Schema 15: which of them also promote on a move out of their zone. The compiler has
    # already dropped a flag that cannot fire, so this is empty for every game before it.
    promo_on_leaving = sorted({rule.piece_type for rule in spec.promotions if rule.on_leaving})

    # --- Explosion ---
    explosion = spec.explosion
    explosion_blast: list[int] = []
    explosion_blast_index: list[int] = [0]
    explosion_immune: list[int] = []
    # Schema 10 (Q18): whose capture explodes, and what has to be taken for it to.
    # **Empty means "any", not "none"**, and that is what keeps `atomic.json` still. A
    # full-length vector of ones would be a new boundary field at a non-inert value, and
    # `canonical_tables` would then move the fingerprint of every ruleset with an
    # explosion rule -- which is exactly the corpus-wide replay `_INERT_DEFAULTS` exists
    # to remove. The compiler has already normalised "names every type" to `None`.
    explosion_by: list[int] = []
    explosion_of: list[int] = []
    if explosion is not None:
        explosion_immune = [
            int(pt in explosion.immune_types) for pt in range(n_types)
        ]
        if explosion.by_types is not None:
            explosion_by = [int(pt in explosion.by_types) for pt in range(n_types)]
        if explosion.of_types is not None:
            explosion_of = [int(pt in explosion.of_types) for pt in range(n_types)]
        for square in range(n_squares):
            explosion_blast.extend(explosion.blast[square])
            explosion_blast_index.append(len(explosion_blast))

    # --- Drops ---
    drops = spec.drops
    drops_regions: list[int] = []
    drops_no_doubled: list[int] = []
    drops_no_mate: list[int] = []
    if drops is not None:
        for player in (0, 1):
            for pt in range(n_types):
                region = drops.regions[player][pt]
                if region is None:
                    drops_regions.extend([-1] * n_squares)
                else:
                    drops_regions.extend(int(sq in region) for sq in range(n_squares))
        drops_no_doubled = [int(pt in drops.no_doubled) for pt in range(n_types)]
        drops_no_mate = [int(pt in drops.no_mate) for pt in range(n_types)]

    # --- Reach ---
    reach = spec.reach
    reach_squares: list[int] = []
    if reach is not None:
        for player in (0, 1):
            reach_squares.extend(int(sq in reach.squares[player]) for sq in range(n_squares))

    # How far each piece is from a winning square, from every square, on an empty board.
    # Compiled by `compile.py` because it is a fact about the rules; flattened here for
    # the same reason everything else is, and read once per evaluated node.
    goal_distance = [
        distance
        for player in spec.goal_distance
        for per_type in player
        for distance in per_type
    ]
    # Q708 (F1): the same, from each goal square alone, [player][goal][type][square] in
    # the goals' ascending square order. Empty unless a priced race gives a side more
    # than one goal square, so no other ruleset's tables move.
    goal_distance_each = [
        distance
        for player in spec.goal_distance_each
        for per_goal in player
        for per_type in per_goal
        for distance in per_type
    ]

    # --- Turn length ---
    # Sent as a table of turn lengths plus the length everything past it settles to, so
    # the core reads an index rather than reimplementing the ramp.
    turn = spec.turn

    # --- Territory ---
    territory = spec.territory

    # --- Scoring ---
    # The declared piece values rather than the evaluator's derived ones. A rule has to
    # be objective: the evaluator is an agent's opinion about what a piece is worth, and
    # two agents holding different opinions must not disagree about who won.
    scoring = spec.scoring
    scoring_squares: list[int] = []
    if scoring is not None and scoring.squares is not None:
        for player in (0, 1):
            scoring_squares.extend(
                int(sq in scoring.squares[player]) for sq in range(n_squares)
            )

    # --- Strike ---
    # Three fields, not five, and the two that are missing are the point. The chain, the
    # ``immune`` list and the choice of ``demote`` or ``promote`` are all resolved by the
    # compiler into one array -- ``strike_to[victim] `` is the type the victim becomes,
    # or -1 for it leaving the board -- so an effect flag and an immunity mask crossing
    # beside it would be two more places the same rule is stated and could be stated
    # differently. What a rule *means* is interpreted once, and this is what that costs
    # when it is done properly: the boundary carries the answer, not the question.
    #
    # Schema 10 (Q19a) makes it three *or* five, and which one it is says something. The
    # compiled rule is always the general contact table -- (mover, target) -> (what the
    # mover becomes, what the target becomes) -- and `compact_to` asks whether that table
    # can be said keyed by the target alone. Where it can, the schema-6 array crosses and
    # the two contact tables stay empty; where it cannot, the array stays empty and the
    # two tables cross. So a ruleset written before schema 10 sends exactly the bytes it
    # always did, and the core is handed one shape either way because `tables.rs` widens
    # the compact form on the way in. The inert default is what keeps every earlier
    # fingerprint still, the way `npc_next` and `entry_promotion` do.
    strike = spec.strike
    compact = strike.compact_to if strike else None
    strike_to = list(compact) if compact is not None else []
    contact_onto_to = [] if compact is not None or not strike else list(strike.onto_to)
    contact_mover_to = [] if compact is not None or not strike else list(strike.mover_to)
    # Schema 11 (Q19b): the friendly-target tables, empty unless a link says `own`.
    contact_own_onto_to = list(strike.own_onto_to) if strike else []
    contact_own_mover_to = list(strike.own_mover_to) if strike else []
    # Schema 12 (Q20): the landing list per (player, square), flat with an index, the
    # explosion's shape. Empty unless the position is `choose`.
    strike_landing: list[int] = []
    strike_landing_index: list[int] = []
    if strike is not None and strike.landing:
        strike_landing_index.append(0)
        for per_square in strike.landing:
            for candidates in per_square:
                strike_landing.extend(candidates)
                strike_landing_index.append(len(strike_landing))

    # --- Facing (schema 8) ---
    # Two arrays, both per piece type, and both empty for a spec in which nothing turns
    # -- which is the inert default that keeps every earlier ruleset's fingerprint still.
    # `n_frames` already says how many readings the movement tables hold; what the core
    # cannot get from it is *which* of them one piece may be in, and which one it starts
    # in. A piece with two axes on a four-facing board turns by two indices at a time,
    # and the core divides `n_facings` by `facing_axes` once to find that out.
    facing_axes = list(spec.facing_axes) if spec.has_facing else []
    facing_initial = list(spec.facing_initial) if spec.has_facing else []

    # --- Wild pieces (schema 9) ---
    # Two board-sized tables and a flag, and that is the whole rule at the boundary: the
    # square each wild piece steps to next, and what taking the one that walks each
    # square is worth. Which wild piece stands where is on the board; where it may
    # threaten is the third block of attack sources above; which types it may eat is the
    # capture matrix it already shares with every player's piece. Empty when nothing is
    # wild, so no earlier ruleset's fingerprint moves.
    npcs = spec.npcs
    npc_next = list(npcs.next) if npcs is not None else []
    npc_bounty = [float(worth) for worth in npcs.bounty] if npcs is not None else []

    # --- Promotion on entering play (schema 9) ---
    # One table, `[player][piece_type][square]` flattened in that order, holding the
    # type a piece that *enters play* on that square becomes, or -1. The compiler has
    # already chosen the target and already decided that `mandatory` does not apply, so
    # what crosses is the answer and not the rule -- the strike's `strike_to` shape.
    # Empty when the flag is off, so no earlier ruleset's fingerprint moves.
    entering = spec.entry_promotion
    entry_promotion = (
        [
            becomes
            for per_type in entering.becomes
            for row in per_type
            for becomes in row
        ]
        if entering is not None
        else []
    )

    # --- Extinction ---
    extinction = spec.extinction

    # --- Immobile squares ---
    immobile_squares = [
        int(square in spec.immobile_squares[frame][piece_type])
        for frame in range(n_frames)
        for piece_type in range(n_types)
        for square in range(n_squares)
    ]

    # --- Bare royal ---
    # Stated from the point of view of the side reduced to royalty alone: shatranj's
    # rule is a loss for them. "off" is carried separately from the result because
    # "draw" is one of the three answers ``ENDING_RESULTS`` promises and cannot double
    # as the off switch -- which is exactly the bug a boolean produced here.
    bare_royal_result = _ENDING_RESULT[spec.bare_royal] if spec.bare_royal else 0

    return {
        "width": spec.width,
        "height": spec.height,
        "n_squares": n_squares,
        "n_types": n_types,
        # How many readings of the movement vectors the tables hold. Two today, one per
        # player. Sits in `_INERT_DEFAULTS` at 2, so adding it moved no ruleset's tables
        # fingerprint -- which is item 2's canonicalisation doing exactly the job it was
        # built for, on the first field to arrive after it.
        "n_frames": n_frames,
        "royal_types": list(spec.royal_types),
        "leap_index": leap_index,
        "leap_refs": leap_refs,
        "leap_squares": leap_squares,
        "leap_modes": leap_modes,
        "slide_index": slide_index,
        "slide_refs": slide_refs,
        "slide_squares": slide_squares,
        "slide_modes": slide_modes,
        "hop_index": hop_index,
        "hop_refs": hop_refs,
        "hop_squares": hop_squares,
        "hop_modes": hop_modes,
        "sweep_index": sweep_index,
        "sweep_refs": sweep_refs,
        "sweep_squares": sweep_squares,
        "sweep_modes": sweep_modes,
        "has_sweep": has_sweep,
        "attack_index": attack_index,
        "attack_srcs": attack_srcs,
        "attack_paths": attack_paths,
        "attack_sweep": attack_sweep,
        "attack_path_modes": attack_path_modes,
        "dedup": dedup,
        "capture_allowed": capture_allowed,
        "has_capture_matrix": spec.has_capture_matrix,
        "has_hop_attacks": spec.has_hop_attacks,
        "initial_board": spec.initial_board,
        "initial_unmoved": spec.initial_unmoved.to_bytes(32, "little"),
        "win": spec.win,
        "stalemate": spec.stalemate,
        "checkmate_result": spec.checkmate,
        "max_plies": spec.max_plies,
        "repetition_draw": spec.repetition_draw,
        "no_progress_draw": spec.no_progress_draw,
        # Which types can never return to a square they left, so that a move by one is
        # progress and clears both clocks above. Derived from the movement tables by
        # `compile._one_way_types`; sent as the list of types, which `type_flags` turns
        # back into a flag per type, exactly as `promo_piece_types` is. It used to be
        # `ep_type` doing this job under an en passant name -- see Q113.
        "one_way_types": sorted(spec.one_way_types),
        "ep_type": ep_type,
        "ep_pass_flat": ep_pass_flat,
        "ep_pass_index": ep_pass_index,
        # Promotion tables
        "promo_squares": promo_squares,
        "promo_targets_flat": promo_targets_flat,
        "promo_targets_index": promo_targets_index,
        "promo_optional": promo_optional,
        "promo_piece_types": promo_piece_types,
        "promo_on_leaving": promo_on_leaving,
        # Castling
        "castle_king": castling.king_type if castling else -1,
        "castle_rook": castling.rook_type if castling else -1,
        "castle_steps": castling.king_steps if castling else 0,
        "castle_king_files": (
            list(castling.king_files) if castling and castling.king_files else [-1, -1]
        ),
        "castle_rook_files": (
            list(castling.rook_files) if castling and castling.rook_files else [-1, -1]
        ),
        # Schema 2 runtime features
        "multi_royal": spec.multi_royal,
        "demotions": list(spec.demotions),
        "must_capture": spec.must_capture,
        "forbid_check": spec.forbid_check,
        "forbid_immobility": spec.forbid_immobility,
        "royal_facing_forbidden": spec.royal_facing_forbidden,
        "facing_index": facing_index,
        "facing_entries": facing_entries,
        "facing_paths": facing_paths,
        "check_limit": spec.check_limit,
        "has_bare_royal": spec.bare_royal != "",
        "bare_royal": bare_royal_result,
        "immobile_squares": immobile_squares,
        # Explosion
        "has_explosion": explosion is not None,
        "explosion_immune": explosion_immune,
        "explosion_blast": explosion_blast,
        "explosion_blast_index": explosion_blast_index,
        "explosion_by": explosion_by,
        "explosion_of": explosion_of,
        # Drops
        "has_drops": drops is not None,
        "drops_from_captures": drops.from_captures if drops else False,
        "drops_regions": drops_regions,
        "drops_no_doubled": drops_no_doubled,
        "drops_no_mate": drops_no_mate,
        # Schema 10 (Q51). Empty when nothing is stocked, which is every ruleset written
        # before it -- crazyhouse, shogi and dobutsu all keep the fingerprints they had.
        "drops_stock": list(drops.stock) if drops else [],
        "drops_unlimited": [int(flag) for flag in drops.unlimited] if drops else [],
        # Reach
        "has_reach": reach is not None,
        "reach_piece_type": reach.piece_type if reach else -1,
        "reach_squares": reach_squares,
        "reach_result": _ENDING_RESULT[reach.result] if reach else 1,
        "reach_safe": reach.safe if reach else False,
        "goal_distance": goal_distance,
        "goal_diameter": spec.goal_diameter,
        "goal_distance_each": goal_distance_each,
        # Extinction
        "has_turn": turn is not None,
        "turn_moves": turn.moves if turn else 1,
        "turn_ramp": list(turn.lengths) if turn else [],
        "has_territory": territory is not None,
        "territory_repaints": territory.mode == "repaint" if territory else False,
        "territory_win_at": territory.win_at if territory else 0,
        "has_scoring": scoring is not None,
        "scoring_count": _SCORING_COUNT.get(scoring.count, 0) if scoring else 0,
        "scoring_squares": scoring_squares,
        "piece_values": [float(v) for v in spec.piece_values],
        # Strike
        # Facing
        "facing_axes": facing_axes,
        "facing_initial": facing_initial,
        "has_strike": strike is not None,
        "strike_position": _STRIKE_POSITION[strike.position] if strike else 1,
        "strike_to": strike_to,
        "contact_onto_to": contact_onto_to,
        "contact_mover_to": contact_mover_to,
        "contact_own_onto_to": contact_own_onto_to,
        "contact_own_mover_to": contact_own_mover_to,
        "strike_landing": strike_landing,
        "strike_landing_index": strike_landing_index,
        # Wild pieces
        "has_npcs": npcs is not None,
        "npc_next": npc_next,
        "npc_bounty": npc_bounty,
        # Promotion on entering play
        "entry_promotion": entry_promotion,
        "has_extinction": extinction is not None,
        "extinction_types": list(extinction.piece_types) if extinction else [],
        "extinction_any_piece": extinction.any_piece if extinction else False,
        "extinction_min_count": extinction.min_count if extinction else 0,
        "extinction_result": _ENDING_RESULT[extinction.result] if extinction else -1,
        # Zobrist
        "zobrist_pieces": [
            value for row in spec.zobrist_pieces for value in row
        ],
        "zobrist_side": spec.zobrist_side,
        "zobrist_ep": list(spec.zobrist_ep),
        "zobrist_colour": [value for row in spec.zobrist_colour for value in row],
        "zobrist_turn": list(spec.zobrist_turn),
    }


# Keys whose value is the same for every ruleset that does not reach for the rule they
# belong to, and what that value is. Dropped before a ruleset's tables are fingerprinted,
# which is the whole reason the table exists: `measurement.tables_fingerprint` asks *what
# the compiler produced for this game*, and a new key added here at its inert default
# would otherwise change that answer for all forty-seven of them at once and mark every
# stored measurement stale for a rule none of them uses.
#
# This is `complexity._RULE_ACTIVE` one layer down. That table decides whether a player
# has to be *taught* a rule; this one decides whether the core is *told* one. Both answer
# "is this rule here at all", and both exist because a default spelled out loud has to
# compile to the game that was already there.
#
# Only a *constant* default belongs here. Several keys are an array of zeros as long as
# the board -- `promo_squares`, `immobile_squares`, `ep_pass_index` -- and those stay in
# the fingerprint: their default is not one value, so dropping them would need a rule
# about lengths rather than about values, and they are already in every stored basis.
_INERT_DEFAULTS: dict[str, Any] = {
    "n_frames": 2,
    "facing_axes": [],
    "facing_initial": [],
    "has_capture_matrix": False,
    "has_hop_attacks": False,
    "ep_type": -1,
    # Empty is "every piece can come back", which is a real reading and the one every
    # ruleset whose pieces all turn or retrace has -- so such a spec's bytes do not move
    # for this field existing.
    "one_way_types": [],
    "ep_pass_flat": [],
    "promo_targets_flat": [],
    "promo_piece_types": [],
    # Schema 15 (Q452). Empty unless a promoting piece can leave its zone under the flag.
    "promo_on_leaving": [],
    "castle_king": -1,
    "castle_rook": -1,
    "castle_steps": 0,
    "castle_king_files": [-1, -1],
    "castle_rook_files": [-1, -1],
    "multi_royal": False,
    "must_capture": False,
    "forbid_check": False,
    "forbid_immobility": False,
    "royal_facing_forbidden": False,
    "facing_index": [],
    "facing_entries": [],
    "facing_paths": [],
    "check_limit": 0,
    "has_bare_royal": False,
    "bare_royal": 0,
    "has_explosion": False,
    "explosion_immune": [],
    "explosion_blast": [],
    "explosion_blast_index": [0],
    # Schema 10 (Q18). Empty is "every capturer" and "every victim", so `atomic.json`
    # -- and every other ruleset with an explosion -- keeps the fingerprint it had.
    "explosion_by": [],
    "explosion_of": [],
    "has_drops": False,
    "drops_from_captures": False,
    "drops_regions": [],
    "drops_no_doubled": [],
    "drops_no_mate": [],
    # Schema 10 (Q51). Empty is "nothing is held before a capture", which is every
    # ruleset with a hand up to now.
    "drops_stock": [],
    "drops_unlimited": [],
    "has_reach": False,
    "reach_piece_type": -1,
    "reach_squares": [],
    "reach_result": 1,
    "reach_safe": False,
    "goal_distance": [],
    "goal_diameter": 1,
    "goal_distance_each": [],
    "has_turn": False,
    "turn_moves": 1,
    "turn_ramp": [],
    "has_territory": False,
    "territory_repaints": False,
    "territory_win_at": 0,
    "has_scoring": False,
    "scoring_count": 0,
    "scoring_squares": [],
    "has_strike": False,
    "strike_position": 1,
    "strike_to": [],
    "contact_onto_to": [],
    "contact_mover_to": [],
    # Schema 11 (Q19b). Empty unless a link says `own`, so no earlier fingerprint moves.
    "contact_own_onto_to": [],
    "contact_own_mover_to": [],
    # Schema 12 (Q20). Empty unless the position is `choose`.
    "strike_landing": [],
    "strike_landing_index": [],
    # Schema 13 (Q21). Empty unless a piece sweeps.
    "sweep_index": [],
    "sweep_refs": [],
    "sweep_squares": [],
    "sweep_modes": [],
    "has_sweep": False,
    "attack_sweep": [],
    "attack_path_modes": [],
    "has_npcs": False,
    "npc_next": [],
    "npc_bounty": [],
    "entry_promotion": [],
    "has_extinction": False,
    "extinction_types": [],
    "extinction_any_piece": False,
    "extinction_min_count": 0,
    "extinction_result": -1,
}

# Keys that are part of what every game means, whatever rules it declares: the board's
# shape, the movement and attack tables, the opening array, the ending clocks, the
# declared values and the Zobrist draw. None of these has an "absent" reading, so none is
# ever dropped.
#
# The two sets must together cover `build_tables` exactly, and `tests/test_measurement.py`
# asserts it. That is what makes a new boundary field a named test failure rather than a
# silent corpus-wide staleness: whoever adds one has to say which kind it is.
_ALWAYS_MEANINGFUL = frozenset({
    "width", "height", "n_squares", "n_types", "royal_types",
    "leap_index", "leap_refs", "leap_squares", "leap_modes",
    "slide_index", "slide_refs", "slide_squares", "slide_modes",
    "hop_index", "hop_refs", "hop_squares", "hop_modes",
    "attack_index", "attack_srcs", "attack_paths",
    "dedup", "capture_allowed",
    "initial_board", "initial_unmoved",
    "win", "stalemate", "checkmate_result",
    "max_plies", "repetition_draw", "no_progress_draw",
    "ep_pass_index",
    "promo_squares", "promo_targets_index", "promo_optional",
    "demotions", "immobile_squares", "piece_values",
    "zobrist_pieces", "zobrist_side", "zobrist_ep", "zobrist_colour", "zobrist_turn",
})


def canonical_tables(spec: CompiledSpec) -> dict[str, Any]:
    """`build_tables` with every key sitting at an inert default dropped.

    The canonical form of *what this ruleset compiles to*, which is the quantity a stored
    measurement actually rests on. The architecture already claims a game is a function
    of these arrays plus the code that walks them; this is that claim taken literally, so
    that an edit to `compile.py` which changes no ruleset's tables invalidates nothing.

    Canonicalising is not cosmetic and it is not an optimisation. Without it the first
    new boundary field -- at `False`, at `[]`, at `-1` -- would move every fingerprint in
    the corpus and hand back exactly the replay this is built to remove.
    """
    tables = build_tables(spec)
    return {
        key: value
        for key, value in tables.items()
        if key not in _INERT_DEFAULTS or value != _INERT_DEFAULTS[key]
    }


def rust_supports(spec: CompiledSpec) -> str | None:
    """``None`` if the Rust core can play this ruleset, else why it cannot.

    Read the direction carefully: this asks what the *core* can express, not what version
    the spec declares. The answer is now always yes. The core implements every schema-2
    runtime feature -- drops, explosions, reach, extinction, compulsory capture, forbidden
    checks, forbidden immobility, royal facing, check limits, bare royal, multi-royal,
    checkmate-as-win, demotions, multiple promotion rules, optional promotion, promotion
    zones, per-player targets, castling by destination file -- and, since the attack
    sources gained a ``prefix_len``, bounded screen ranges as well. That last one was the
    only remaining gap and the grasshopper was the only ruleset in it.

    The function is kept rather than deleted with the check it guarded, because it is the
    honest place to say *no* if a future primitive lands in the compiler before it lands
    in the core. Being wrong in the permissive direction here is silent and produces a
    plausible game under rules nobody wrote, so a new primitive adds a check here in the
    same change that adds the row to ``MOVE_KINDS``.

    Schema 3 passed through here on the way in, and that is the intended use: the check
    for a capture matrix was written and returning a refusal before ``tables.rs`` knew
    the word, so the intermediate state raised rather than playing a game in which every
    piece could take everything. It came out when the core could answer, and the
    score-comparison ending in the same bump went through the same door, and so did
    schema 4's per-square territory, and schema 4's multi-move turn. Four features, four
    times the guard held an intermediate state shut rather than letting it play a
    plausible wrong game.

    **Schema 5 needed almost nothing.** A wrapped board resolves into the per-square
    tables and the core walks them without knowing what a file is. It reached the core
    in exactly the places that computed a square from arithmetic instead of reading a
    table, and there were three. En passant's midpoint became a compiled table, because
    refusing it would have cost a real chess rule on every wrapped board. Castling is
    stated as a span of one rank and does not wrap, which the prose says out loud. And
    ``royal_facing`` was refused, twice -- for a seam in schema 5 and for a wall in
    schema 6.

    **Schema 6's obstacles cross nothing.** ``rules.obstacles`` has no entry in
    ``build_tables`` at all, because a wall is *static*: ``_compile_square`` answers
    every question it raises once, and by the time anything crosses this module the
    walls are gone -- a ray that met one is simply a shorter ray, a leap onto one was
    never emitted.

    ``rules.strike`` is the other half and the opposite case. Movegen is untouched -- a
    strike is generated as a capture, blocked like a capture, permitted by the same
    matrix -- so there is nothing to say about *generating* one. But ``make`` has to put
    a struck piece somewhere, and that is five fields across the boundary and a refusal
    held here until ``game.rs`` could answer. The sixth time this guard has kept an
    intermediate state shut rather than letting a plausible wrong game play.

    **Both ``royal_facing`` refusals are gone, and nothing replaced them.** They were the
    same defect twice -- ``royals_face`` worked a line out by walking a file with ``+=
    width``, which is only right on a board whose edges are edges and whose empty squares
    are the only empty squares. It is a compiled table now, ``facing_lines``, built the
    way ``ep_passes`` was built in schema 5 and for the same reason. So the rule works on
    a cylinder and behind a wall instead of being held shut on both, and xiangqi stops
    being the one ruleset that cannot be crossed with anything.

    **Schema 9's two holds came off together, in Q9b.** ``rules.npcs`` compiles to a
    third owner's byte on the opening board, a third block of attack sources and two
    board-sized tables; ``rules.promote_on_entry`` compiles to ``entry_promotion``, a
    table the core reads wherever a piece enters play without a move of its own. Neither
    meant anything to a core built to hold two owners and to leave a dropped pawn
    standing, and both were refused by name until ``game.rs`` had the phase, the undo
    log, the third occupancy set, the tally and ``Tables::arriving_value``. Ten holds so
    far, zero plausible wrong games played, and three of the ten retired by teaching the
    core the thing they guarded -- which is what a guard is for. The constants stay as
    the honest record of the ordering.
    """
    if spec.npcs is not None and not _CORE_KNOWS_NPCS:
        return (
            "rules.npcs: the compiler places a piece neither player owns and the core "
            "holds two owners, so it has no row, no occupancy set and no phase for it"
        )
    if spec.entry_promotion is not None and not _CORE_KNOWS_NPCS:
        return (
            "rules.promote_on_entry: the compiler resolves what a dropped or struck piece "
            "becomes inside its promotion zone and the core still leaves it standing"
        )
    if any(rule.on_leaving for rule in spec.promotions) and not (
        _CORE_KNOWS_PROMOTION_ON_LEAVING
    ):
        return (
            "rules.promotion.on_leaving: the compiler says which pieces promote on a move "
            "out of their zone and the core still promotes only on arriving"
        )
    if spec.has_facing and not _CORE_KNOWS_FACING:
        return (
            "piece.facing: the compiler builds a frame per orientation and the core "
            "still reads the player's frame, so every piece would move as though it "
            "faced forward"
        )
    if spec.strike is not None and not _CORE_KNOWS_STRIKE:
        return (
            "rules.strike: the compiler resolves the chain but the core still removes a "
            "struck piece rather than transforming it"
        )
    if (
        spec.strike is not None
        and spec.strike.changes_mover
        and not _CORE_KNOWS_CONTACTS
    ):
        return (
            "rules.contacts: the compiler resolves what the *attacker* becomes and the "
            "core still leaves it as it was"
        )
    if spec.strike is not None and spec.strike.landing:
        from chess_variant_sandbox.spec.schema import MAX_LANDING_CANDIDATES

        widest = max(len(candidates) for per in spec.strike.landing for candidates in per)
        if widest > MAX_LANDING_CANDIDATES:
            return (
                f"rules.{spec.strike.spelling}.landing: one square resolves to {widest} "
                f"landing squares, and the move word holds at most "
                f"{MAX_LANDING_CANDIDATES}"
            )
    if (
        spec.explosion is not None
        and not spec.explosion.fires_on_every_capture
        and not _CORE_KNOWS_SELECTIVE_EXPLOSION
    ):
        return (
            "rules.explode_on_capture.by / .of: the compiler narrows which captures "
            "explode and the core still explodes on every one of them"
        )
    return None


# ---------------------------------------------------------------------------------------
# Is the extension on this machine the one this package was written against?
#
# ``rust_supports`` above asks what the core *can express*. This asks a different question
# with the same failure mode: whether the compiled extension actually on disk is the one
# that goes with the Python beside it. A person hits this by pulling a change to
# ``rust/src/`` and not rebuilding -- a plain ``uv sync`` does not notice that ``rust/``
# changed -- or by checking out an older branch over a newer build.
#
# It has happened. Schema 9 grew the packed position by one field: the bounty tally joined
# the group in slot 11, in the same commit on both sides of the boundary. A tree with the
# new Python and a core built before it answered every game page with
# ``IndexError: tuple index out of range`` from ``engine._unpack``, which names neither the
# build nor the fix. ``tournament/play.py`` already had this answer for the *agent*
# boundary, wrapping the conversion error with the rebuild command; this is the same answer
# for the *position* boundary, asked once at construction rather than found by whichever
# consumer indexed a tuple first.
#
# Two things cross this boundary, so there are two checks. ``core_boundary_mismatch`` probes
# the *position* and needs no declaration from either side. ``core_table_mismatch`` compares
# the *tables* against the vocabulary the core reports, because a table the core never reads
# is invisible from here -- it cannot be probed, only asked about.

# The two commands that rebuild the extension. ``CLAUDE.md`` carries the same pair.
REBUILD_THE_CORE = (
    "    uv run maturin develop --release --manifest-path rust/Cargo.toml\n"
    "or, for the installed package:\n"
    "    uv sync --extra dev --reinstall-package chess-variant-sandbox"
)


def _shape(value: Any) -> Any:
    """A packed position with its leaves thrown away: nesting and lengths, nothing else.

    Sequences are ``(length, contents)`` and everything else is ``None``. ``list`` and
    ``tuple`` count as the same thing on purpose -- the core hands back a ``list`` where
    ``_pack`` writes a ``tuple`` in two slots, and that difference is PyO3's, not a
    disagreement about the boundary.
    """
    if isinstance(value, (tuple, list)):
        return len(value), tuple(_shape(item) for item in value)
    return None


def _where(path: tuple[int, ...]) -> str:
    inner = [f"field {index}" for index in reversed(path[1:])]
    return " of ".join([*inner, f"slot {path[0]}"])


def _first_disagreement(mine: Any, theirs: Any, path: tuple[int, ...] = ()) -> str | None:
    """Where two packed shapes first differ, in words, or ``None`` if they agree."""
    if mine == theirs:
        return None
    if mine is None or theirs is None:
        return f"{_where(path)} is a group on one side of the boundary and a value on the other"
    if mine[0] != theirs[0]:
        return f"{_where(path)} carries {theirs[0]} fields where this package writes {mine[0]}"
    for index, (left, right) in enumerate(zip(mine[1], theirs[1], strict=True)):
        found = _first_disagreement(left, right, (*path, index))
        if found is not None:
            return found
    return None  # pragma: no cover - unreachable: unequal shapes differ somewhere


def _stale_core(subject: str, detail: str) -> str:
    """The refusal a stale extension gets, whichever half of the boundary found it."""
    return (
        f"the loaded `chess_variant_sandbox._core` does not speak this package's {subject}: {detail}.\n\n"
        "That is almost certainly a stale build of the extension: `rust/src/` moved and "
        "`chess_variant_sandbox._core` was compiled on the other side of the move. A plain `uv sync` "
        "does not notice that `rust/` changed. Rebuild with:\n"
        f"{REBUILD_THE_CORE}"
    )


def _listed(keys: Iterable[str]) -> str:
    return ", ".join(f"`{key}`" for key in sorted(keys))


def core_boundary_mismatch(
    core: Any,
    read: Callable[[Any], Any],
    write: Callable[[Any], Any],
) -> str | None:
    """``None`` if this core and this package agree on a position, else why they do not.

    **What it compares, and why not the fingerprint.** ``_core.BUILD_FINGERPRINT`` hashes
    every byte of ``rust/src/`` and so moves on every edit to the core, including the many
    that change nothing a Python caller can see. ``measurement`` wants exactly that -- a
    game played by a different core is a different measurement whatever moved -- and
    ``agents/valuations.py`` stamps it for the same reason. Refusing to *run* on any
    difference would be a much worse trade: it would make a hard error out of every
    developer's first command after a ``git pull``, for a build that plays identically, and
    the cure would be more disruptive than the disease. So this asks the narrower question
    that actually broke: not *which* core is loaded, but whether it and this package are
    describing the same position.

    A version constant the core reported would answer it too, and was rejected on two
    counts. It has to be bumped by hand, and this repo has the receipts on what that is
    worth: ``_core.pyi`` -- the boundary's other hand-maintained declaration -- "sat two
    fields stale through the whole of schema 4 while mypy reported success". And a number
    only says the two sides *think* they agree; it cannot notice a field that moved without
    anybody remembering the constant.

    So the check is a probe of the thing itself, and it holds no declaration of its own:
    it asks the core for its opening position, hands it to ``read``, hands the result back
    to ``write``, and compares the shape that came out with the shape that went in. Both
    directions of drift fall out of that one comparison. A core built *before* a field was
    added cannot be read at all -- schema 9's ``IndexError``, which is caught here and
    named. A core built *after* one can be read, but the field is silently dropped, and the
    position that comes back out is a field short. Neither needs a constant to notice,
    which is the point: the check cannot go stale, because ``_pack`` and ``_unpack`` *are*
    what it tests, rather than a third copy of what they say.

    ``read`` and ``write`` are passed in rather than imported for the same reason. They
    live in ``engine/engine.py``, which imports this module, so reaching for them here
    would be a cycle -- but it would also let the check drift from its subject, and a
    boundary check that tests a copy of the boundary is worth nothing.

    **What it does not catch.** A core that ignores a *table* says nothing about the
    position, so the round trip is clean and this returns ``None``. That is the other half
    of the same question and ``core_table_mismatch`` below asks it, off the vocabulary the
    core now reports.
    """
    packed = core.initial_state()
    try:
        state = read(packed)
    except (IndexError, TypeError, ValueError) as error:
        return _stale_core(
            "position boundary",
            f"reading its opening position raised {type(error).__name__}: {error}",
        )
    disagreement = _first_disagreement(_shape(write(state)), _shape(packed))
    if disagreement is not None:
        return _stale_core("position boundary", disagreement)
    return None


def core_table_mismatch(sent: Iterable[str], read: Iterable[str]) -> str | None:
    """``None`` if the core reads exactly the keys this package sends, else why not.

    ``sent`` is the ``build_tables`` package -- a dict iterates as its keys -- and ``read``
    is ``_core.TABLE_KEYS``, which is ``rust/src/tables.rs``'s own registry of every key
    its readers are spelled out of. A key can only be read there by naming a registered
    constant, so the vocabulary cannot fall behind the readers and this is a comparison
    against what the core *actually* does rather than against a declaration about it.

    **Why a comparison and not just ``field``.** ``tables.rs::field`` raises for a key that
    is missing, but only for the branches a given ruleset reaches -- ``explosion_immune``
    is read only behind ``has_explosion`` -- and it ignores an extra key entirely. So a
    schema bump that adds a table and leaves the packed position alone passes every other
    check on this boundary while the core plays the game it already knew. Both directions
    fall out of one set comparison, taken before the tables cross.
    """
    package = set(sent)
    vocabulary = set(read)
    unread = package - vocabulary
    absent = vocabulary - package
    if not unread and not absent:
        return None
    detail = []
    if unread:
        detail.append(f"it never reads {_listed(unread)}, which this package sends")
    if absent:
        detail.append(f"it reads {_listed(absent)}, which this package does not send")
    counts = f"{len(package)} keys sent against the {len(vocabulary)} the core reads"
    return _stale_core("tables", f"{'; '.join(detail)} -- {counts}")
