"""How far apart two rulesets are, as a number.

The content hash gives identity and never neighbourhood: two specs either are the same
game or are simply different, with nothing in between. Three things want the in-between.
Diversity maintenance needs to know whether an archive holds two games or one game
twice. A behaviour grid needs axes, and the cheap axes are spec-derived. And any future
crossover needs to know which pairs are close enough to be worth recombining.

The features are counts the spec already carries -- board shape, how many kinds of piece
there are, what those pieces can do, which optional rules are present -- normalised so
that no single one dominates, and compared as a mean absolute difference. That makes the
result a number between 0 and 1 with an honest meaning: *the share of the measured
description these two games differ on*.

Two limits, stated because a distance that is quietly wrong is worse than none:

**It measures the description, not the play.** Two specs a move apart can play utterly
differently -- moving a promotion rank two ranks in front of the pawns is a tiny edit and
a different game -- and this cannot see that. Behavioural distance needs a tournament and
is a different measurement.

**It is symmetric and unweighted.** No feature is more important than another here,
because nothing has measured which features matter and a weighting invented without that
would be a preference presented as arithmetic.
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox.spec.complexity import expand_symmetry, measure_complexity
from chess_variant_sandbox.spec.schema import (
    PURSE_MAX,
    REPETITION_NEVER,
    REST_MAX,
    SIGHT_EFFECTS,
    STOCK_UNLIMITED,
    WRAP_PAIRS,
    absent_squares,
    duck_goes,
    duck_piece,
    facing_turns,
    hire_ids,
    majority_ids,
    move_push,
    neutral_ids,
    npc_pounce,
    npc_respawn,
    npc_split,
    npc_turns_back,
    obstacle_squares,
    piece_bounties,
    piece_prices,
    piece_rests,
    piece_sights,
    piece_speeds,
    piece_strengths,
    piece_weights,
    portal_groups,
    portal_map_shape,
    portal_stand_in_shape,
    promotion_squares,
    purse_bribes,
    purse_pays_loser,
    purse_rule,
    push_lists,
    push_rule,
    pushable_ids,
    pushing_ids,
    reach_pieces,
    simultaneous_switch,
    spy_ids,
    strike_changes_anything,
    strikers,
    sweep_length,
    takes_royal_ids,
    wrap_stated,
)
from chess_variant_sandbox.spec.transform import keeps_a_survivor

# Each feature is divided by the scale on which it varies across real games, so that a
# board-area difference of one square and a piece-count difference of one type do not
# arrive at the comparison in different units. The scales come from the committed corpus:
# boards run 12 to 120 squares, armies 1 to 9 piece types, rule mass 12 to 100.
_SCALES: dict[str, float] = {
    "area": 100.0,
    "aspect": 1.0,
    "piece_types": 8.0,
    "army": 40.0,
    "rule_mass": 90.0,
    "special_rules": 6.0,
    "vectors_per_piece": 8.0,
    "slide_share": 1.0,
    "leap_share": 1.0,
    "hop_share": 1.0,
    # Schema 13 (Q21). A line that removes what it crosses is a different piece from one
    # that stops at it, so a slide turned sweep must not measure zero from its parent.
    "sweep_share": 1.0,
    # Q1063. A sweep that clears only its first steps and slides on past them; without this
    # it and the sweep that clears its whole line read as one game.
    "short_sweep_share": 1.0,
    # Q1007a (`docs/archive/plan-schema-31.md` §2.5). A lame move with a `range` other than 1 rides,
    # jump after jump; without this a lame rider and the single lame leap it was written from
    # read as one game (every other count is equal), and the loop would skip the tournament.
    "lame_rider_share": 1.0,
    # SR1a (`docs/plan-schema-next.md` §1 rank 1). A jump whose legs are stated is blocked
    # elsewhere than the computed leg (Janggi's elephant beside the (2, 3) lame); without
    # this the two would differ only by the rule charge in `special_rules`.
    "legs_share": 1.0,
    # T1 (`docs/archive/plan-pieces-moves.md` §5.5). A slide whose path turns (a circle or a zigzag)
    # reaches other squares from the same directions; without this the ring queen and the
    # queen would differ only by the rule charge in `special_rules`.
    "turning_share": 1.0,
    "royal_types": 2.0,
    "royal_step": 2.0,
    "royal_slides": 1.0,
    "move_orthogonal_share": 1.0,
    "capture_orthogonal_share": 1.0,
    "mean_step": 3.0,
    "has_promotion": 1.0,
    "has_drops": 1.0,
    # Schema 10 (Q51), and two rather than one for the reason the strike has three. A
    # hand that fills only from captures is a game about recycling material; a hand that
    # starts full is a game about placement, and one that never empties is a game where
    # a move is always available and stalemate is unreachable. Those are different games
    # on the same board, and `has_drops` alone would call crazyhouse and a placement game
    # the same distance from chess.
    "has_stock": 1.0,
    "stock_is_endless": 1.0,
    # Schema 18d (Q784). Whether a held piece *must* be played: the whole difference
    # between a stock spent when it suits and a placement phase, and not costed in
    # `complexity` (as 18c's flags are not), so it needs a feature or the pair reads as
    # one game.
    "drops_placement": 1.0,
    # Schema 18e (Q785). Whether a held piece may come back as its other face: not costed
    # in `complexity` (as 18c's and 18d's flags are not), so it needs a feature.
    "drops_either_face": 1.0,
    # Q1059a. Whether a held piece enters only by gating: the difference between a hand
    # played anywhere and Seirawan's, not otherwise visible to the features above.
    "drops_gating": 1.0,
    # The relaxation gets a feature too, because it is a feature of the *game* and not
    # only of the schema: a piece that never moves is a wall a player builds, and a spec
    # differing from another in nothing else is a different game.
    "has_a_moveless_piece": 1.0,
    "has_castling": 1.0,
    "win_is_capture": 1.0,
    # Each way of ending a game is its own feature rather than a count of them. King of
    # the hill and three-check are otherwise identical descriptions -- both are chess
    # plus one extra ending -- and they are not the same game.
    "wins_by_arrival": 1.0,
    "wins_by_check_count": 1.0,
    "wins_by_extinction": 1.0,
    "wins_by_bare_royal": 1.0,
    # Q1069a: not costed in `complexity` (a clock-like draw), so it needs a feature, or two
    # specs differing only by it read as one game.
    "draws_insufficient_material": 1.0,
    # Schema 3 and 4. All three separate a *kind* of game rather than a size of one, so
    # each is worth as much as a win condition. `repaints_territory` earns its own entry
    # rather than folding into `has_territory` for the reason the test below exists:
    # versions 2 and 4 of the RPS corpus differ in nothing else, and without it they
    # measure identical -- which would let the loop treat one as a rediscovery of the
    # other and skip a tournament it should have run.
    "restricts_captures": 1.0,
    "has_territory": 1.0,
    "repaints_territory": 1.0,
    "must_capture": 1.0,
    # Schema 5, and the same lesson a third time. A joined edge separates a kind of game
    # rather than a size of one, and the three flags are needed rather than one because
    # `cylinder`, `torus` and `onewaycylinder` differ in *nothing else* -- same board,
    # same army, same rules, one field apart. Folding them into a single `has_wrap`
    # would let the loop treat one as a rediscovery of another and skip a tournament it
    # should have run, which is exactly what `repaints_territory` exists to prevent.
    "wraps_files": 1.0,
    "wraps_ranks": 1.0,
    "wraps_asymmetrically": 1.0,
    # Q281: a hexagon's three pairs of sides, one feature each for the reason above. No
    # feature says `shape`: a hexagon compiles to its rhombus cut by walls, so the native
    # board and its walled twin differ by `has_obstacles` alone.
    "wraps_n_s": 1.0,
    "wraps_ne_sw": 1.0,
    "wraps_nw_se": 1.0,
    # Not schema 5, and found by it. `test_no_two_committed_rulesets_measure_identical`
    # reported `atomic` and `cylinder` identical, and only half of that was the missing
    # wrap: an exploding capture had no feature of its own either, so any ruleset that is
    # chess plus one rule of the same cost collided with it.
    "explodes": 1.0,
    # Schema 10 (Q18). `explodes` alone cannot separate atomic from a game where only a
    # queen's capture explodes, and those are different games on the same board with the
    # same army -- the `repaints_territory` lesson a sixth time. Read off the same
    # normalisation the compiler uses, so an empty `by` reads as *no* explosion, which is
    # what it compiles to.
    "explodes_selectively": 1.0,
    # Schema 6, and the same lesson a fourth time. `hierarchy-fall` and `hierarchy-rise`
    # are the sharpest case this project has: same board, same army, the chain pointing
    # the other way. Without `strike_promotes` they measure identical, and the loop
    # would treat one as a rediscovery of the other and skip the tournament that is the
    # entire point of running both. `swaps_on_strike` is there for the same reason one
    # level down -- where the two pieces end up is a different game, not a different
    # size of one.
    "has_obstacles": 1.0,
    "strikes": 1.0,
    "strike_promotes": 1.0,
    "swaps_on_strike": 1.0,
    # Schema 10 (Q19a), and two rather than one for the reason the strike has three. A
    # contact table sets `strikes` -- it *is* a strike, generally -- and these separate a
    # game where only the taken piece changes from one where the taker changes too, and
    # one whose outcomes depend on the pair from one that is a strike written the long
    # way. `hierarchy-fall` re-spelled as a contact table has to measure zero from its
    # original, and a table that promotes the attacker has to measure a long way from
    # both; without these two it would measure zero from both.
    "contacts_by_pair": 1.0,
    "contacts_change_the_mover": 1.0,
    # Schema 11 (Q19b). A contact onto a friendly piece is a move that no other rule
    # generates, so a table with one is a different game from the same table without.
    "contacts_meet_own": 1.0,
    # Schema 18 (Q781). A link that keeps its mover overrides the mover's promotion on
    # that capture (Einstein's queen), and reads nothing else: `contacts_change_the_mover`
    # is already set by any `mover_to`, so without this a table and the same table plus
    # its keeping links measure zero apart and the loop skips the tournament.
    "contacts_keep_the_mover": 1.0,
    # Schema 12 (Q20). Where the survivor goes is a different game, the `swaps_on_strike`
    # argument, and the attacker choosing it is a third answer beside stay and swap.
    "strike_places_the_survivor": 1.0,
    # Schema 17 (Q540), one each, the `swaps_on_strike` argument again: a capturer that
    # never moves (rifle), a victim reborn at home (circe) and a capturer that changes
    # sides (Andernach) are each a different game from chess and from one another.
    "strike_stays_put": 1.0,
    "strike_rebirths": 1.0,
    "capturer_switches_sides": 1.0,
    # A1 (`docs/archive/plan-pieces-moves.md` §6.5): the share of the capturing types whose captures
    # strike, 1.0 without `by`. Without it rifle and the archers (rifle given to the bishops
    # alone) would differ only by nothing, and read as distance zero.
    "strikers_share": 1.0,
    # Schema 17 (Q540): a blast of radius 0 (kamikaze) is not atomic.
    "explodes_alone": 1.0,
    # SR1b (`docs/plan-schema-next.md` §1 rank 2): a blast of a stated shape (a cross, a
    # file) is not atomic's box; without this the two would read as distance zero.
    "explodes_shaped": 1.0,
    # Two, for the same reason the strike has three. A rule with a cost but no feature
    # collides with every other rule of the same cost, and two specs then report distance
    # zero -- which makes the loop skip a tournament it should have run. `turns_freely`
    # separates a piece that can point four ways from one that can only reverse, and
    # those are as different from each other as a demotion chain is from its reverse.
    "has_facing": 1.0,
    "turns_freely": 1.0,
    # Schema 25 (Q869): a facing no move turns. Without it a rear guard placed facing
    # backwards reads as the same game as one whose pawns may turn round, and they are not.
    "facing_fixed": 1.0,
    # Schema 9, and three for the reason the strike has three. A wild piece that walks
    # and blocks is a moving wall; one that eats is a hunter; one worth points is an
    # objective. Those are different games on the same board with the same army, and
    # `boar` against `boar-tame` -- the schema-9 showcase and its control -- differ in
    # nothing but the second flag.
    "has_npcs": 1.0,
    "npcs_bite": 1.0,
    "npcs_pay": 1.0,
    # Schema 10, and a fourth for the reason the other three are three. A wild piece
    # anything can take is a nuisance to be cleared; one only a knight can take is an
    # obstacle the army has to be arranged around, and the two are different games on
    # the same board with the same route. Without a feature of its own `wolves` and its
    # schema-10 re-spelling would be the only pair that *should* measure identical,
    # while a wolves-with-immunity against a wolves-without would report distance zero
    # and the loop would skip the tournament that tells them apart.
    "npcs_are_hard_to_kill": 1.0,
    # Schema 9's second key. One flag, one feature, and it is owed for the usual reason:
    # `hierarchy-fall` with it and without it are the same board, the same army and the
    # same chain, differing in whether a knight demoted on the far rank is a pawn that
    # never moves again or a queen. Without a feature of its own the two would report
    # distance zero and the loop would skip the tournament that tells them apart.
    "promotes_on_entry": 1.0,
    # Schema 10, and it closes a gap that predates it. `distance.py` has never had a
    # clock feature at all, so two specs differing only in `repetition_draw` have always
    # reported distance zero -- harmless while the difference was 3 against 4, and no
    # longer harmless once the value can say the rule does not exist. A game where
    # repeating a position is never a draw is a different *kind* of game -- a fortress
    # is a fortress rather than a half point, and a losing side cannot shuffle -- so it
    # is worth as much as a win condition, and only the sentinel is a feature. The other
    # clock values stay out: 3 against 4 repetitions is a size of a game, not a kind.
    "no_repetition_draw": 1.0,
    # Schema 14 (Q229). `hex_lattice` keeps `nearest` from pairing a hex game with a square
    # game whose counts happen to match. `promotion_zone_share` is the first player's zone
    # over the board's cells, so a legan-shaped zone, a far edge and a rank measure apart
    # where `has_promotion` alone calls them one game.
    "hex_lattice": 1.0,
    "promotion_zone_share": 1.0,
    # Schema 15 (Q452). Shogi promoting on the way out of its zone and shogi promoting
    # only on the way in are the same board, army and zone, one flag apart, and the rule
    # is not costed in `complexity` -- so without a feature of its own the two would
    # report distance zero and the loop would skip the tournament that tells them apart.
    "promotes_on_leaving": 1.0,
    # Schema 18c (Q783). Two flags, each the whole difference between two games that
    # share a board, an army and every costed rule (neither is costed in `complexity`, as
    # `on_leaving` is not), so each needs a feature of its own or the loop reads the pair
    # as one game.
    "extinction_per_type": 1.0,
    "promotes_only_lost": 1.0,
    # Schema 18f (Q786). Whether the game is decided by leaving a goal square rather than
    # standing on one: not costed in `complexity` (as 18c to 18e's flags are not), so it
    # needs a feature or a spec and its arrival twin read as one game.
    "wins_by_leaving": 1.0,
    # SR3a (`docs/plan-schema-next.md` §1 rank 4). Whether a goal names a choice of runners
    # (`reach.piece` a list of two or more): not costed in `complexity`, a clause of the
    # goal's own sentence, so it needs a feature or a race and its one-runner twin read as
    # one game.
    "reach_runners": 1.0,
    # Schema 19 (Q771). How many portal pairs, as a share of the four a spec may have, so
    # one pair and two read as near and none as far; whether anyone moves an end; whether
    # any passage is one way or closed. The last two are charged in `complexity` only once
    # however many pairs state them, so each needs a feature or a spec and its twin with
    # every end fixed and every door open read as one game.
    "portal_pairs": 1.0,
    "portals_move": 1.0,
    "portals_one_way": 1.0,
    # Schema 20 (Q807). Whether any pair bars moving a portal straight back: charged once
    # in `complexity`, so Ferry and its twin with the bar need this to read as two games.
    "portals_no_back": 1.0,
    # Schema 21 (Q843a, `docs/archive/plan-portals-2.md` §6). The map's shape beside the pair (a ring, a
    # funnel, any other map) and whether any end filters which pieces go in: each charged
    # once in `complexity`, so a ring and its twin as pairs, or a filtered pair and the
    # open one, need these to read as two games.
    "portals_ring": 1.0,
    "portals_funnel": 1.0,
    "portals_other_map": 1.0,
    "portals_filtered": 1.0,
    # Schema 22 (Q843c). The share of groups that are folds: charged per fold group in
    # `complexity`, so a fold pair and the momentum pair on the same squares read as two
    # games.
    "portals_fold": 1.0,
    # Schema 23 (Q850). Whether any route carries more than one wild piece (`also_at`): the
    # tables are the one-piece route's, so without it a patrol and its twin with a second
    # wolf read as one game and the loop would skip the tournament.
    "npcs_share_a_walk": 1.0,
    # Generation 36 (N1). Whether a taken wild piece comes back (`npcs[].respawn`): the tables
    # are the gone-for-good route's, so without it a hunt and its returning twin read as one
    # game and the loop would skip the tournament.
    "npcs_come_back": 1.0,
    # Generation 37 (N2). Whether a wild piece carries a line of sight: its type's sight is the
    # only change, so without it a walk and its seeing twin read apart only by the sight
    # features, which count a player's sights alike.
    "npcs_see": 1.0,
    # Generation 38 (N5). Whether a wild piece paces a line and turns back (`npcs[].turns_back`):
    # the route's squares are the cycle's, so without it a patrol and its pacing twin read as
    # one game and the loop would skip the tournament.
    "npcs_turn_back": 1.0,
    # Generation 39 (N3b). Whether a wild piece pounces on its prey (`npcs[].pounce`): the
    # routes are the same, so without it a patrol and its hunting twin read as one game.
    "npcs_pounce": 1.0,
    # Generation 40 (N3c). Whether a wild piece pounces on its attackers rather than its prey
    # (`npcs[].pounce: "attackers"`): without it a hunter and its guarding twin read as one game.
    "npcs_pounce_attackers": 1.0,
    # Generation 41 (N3d). Whether a wild piece splits (`npcs[].split`): without it a guard and
    # its hydra twin read as one game.
    "npcs_split": 1.0,
    # Generation 42 (N4b). Whether every turn ends with a duck move (`rules.duck`), and whether
    # the duck goes by its type's moves rather than anywhere: without them chess and Duck chess
    # differ only in one moveless piece type, and a classic duck and its walking twin read as
    # one game.
    "has_duck": 1.0,
    "duck_by_moves": 1.0,
    # Q1011: the nineteen features added from schema 26 on, computed in `features` and left
    # out of this table, so `distance` never read them: a queen that freezes and one that
    # disarms measured 0.0 apart. `test_every_feature_has_a_scale` holds the two lists
    # together. Schema 26 (Q918a) and 27 (Q942a): the round, how its tie resolves, the
    # collision table's share, the two chase switches. Every one a flag or a share.
    "simultaneous": 1.0,
    "simultaneous_initiative": 1.0,
    "simultaneous_impossible": 1.0,
    "beats_share": 1.0,
    "simultaneous_swap_collides": 1.0,
    "simultaneous_both_score": 1.0,
    # Schema 27 (Q935a): a count of spy types, on the scale of the other small counts
    # (`royal_types`): one spy and two read half a scale apart, none and two a whole one.
    "spy_types": 2.0,
    # Schema 28 (Q943a): the number of ranks of speed is a count (the same small scale; a
    # three-rank game against none is past it and capped), the three readings of each type's
    # speed over the top rank are shares of 1.
    "speed_ranks": 2.0,
    "speed_royal": 1.0,
    "speed_by_value": 1.0,
    "speed_by_reach": 1.0,
    # Schema 29 (Q983a): the rests are already divided by `REST_MAX`.
    "rest_mean": 1.0,
    "rest_max": 1.0,
    # Schema 30 (Q999a) and generation 31 (Q1007c): the share of types with a sight of each
    # effect, the share of sights that are auras and the turncoat share (half weight, so
    # already within 0-0.5).
    "sight_freeze_share": 1.0,
    "sight_disarm_share": 1.0,
    "sight_shield_share": 1.0,
    "sight_command_share": 1.0,
    "sight_aura_share": 1.0,
    "sight_turncoat_share": 1.0,
    # TC3 (`docs/plan-turncoat.md` §5.8): the share of declared types placed for nobody, the
    # share moved by the majority, and whether a majority type names its own geometry, so a
    # game and its neutral twin, or a standing and a held neutral, are not at distance zero.
    "nobody_share": 1.0,
    "majority_share": 1.0,
    "majority_geometry": 1.0,
    # Q1077 (`docs/plan-turncoat.md` §4.7): the share of declared types whose held pieces take
    # royals, so a held neutral that checks and one that does not are not at distance zero.
    "majority_takes_royal": 1.0,
    # TC4 (`docs/plan-turncoat.md` §5.8): the share of declared types carrying a bounty, so a
    # game and its bounty twin are not at distance zero.
    "bounty_share": 1.0,
    # TC5 (`docs/plan-turncoat.md` §5.8): the purse as a flag, its opening coins (a share of
    # `PURSE_MAX`), whether it names a region, and the share of declared types with a price,
    # so a game and its purse twin, or two purses that differ only in one, are not at
    # distance zero.
    "has_purse": 1.0,
    "purse_start": 1.0,
    "purse_region": 1.0,
    "price_share": 1.0,
    # TC6 (`docs/plan-turncoat.md` §5.8): hires (a priced piece placed for nobody) and bribes,
    # each a flag, so a purse game and its hiring or bribing twin are not at distance zero.
    "has_hire": 1.0,
    "has_bribe": 1.0,
    # TC9c (§5.10): a bounty paid to the player who lost the piece, a flag, so a purse game
    # and its twin paying the loser are not at distance zero.
    "bounty_to_loser": 1.0,
    # Q1074b5 (`docs/plan-push.md` §4): the push, so a game and its pushing twin, or two push
    # games apart in one knob, are not at distance zero. The share of move entries that push;
    # over those entries, whose pieces they push (enemy 0, any 0.5, own 1), the share that also
    # capture, and the way the pushed piece goes (the move 0, its rank 0.5, its file 1); over
    # the pushers, the share that name whom they push and the mean share of the declared types
    # they may push; the chain (one 0, row 0.5, weight 1) and an edge that blocks; and under
    # `"weight"` the mean weight and the pushers' mean strength, on the weight's own scale
    # (a weight is 1 to 9). Under `"one"` and `"row"` a weight is the list's shorthand, read
    # through the lists, so the two spellings of one game measure alike.
    "push_share": 1.0,
    "push_whose": 1.0,
    "push_captures": 1.0,
    "push_along": 1.0,
    "push_lists": 1.0,
    "push_reach": 1.0,
    "push_chain": 1.0,
    "push_edge_blocks": 1.0,
    # Q1074bo: b6's bar on pushing straight back and b7's crushing walls, a flag each.
    "push_no_back": 1.0,
    "push_walls_crush": 1.0,
    "push_weight": 9.0,
    "push_strength": 9.0,
}


def features(spec: dict[str, Any]) -> dict[str, float]:
    """The measured description of one ruleset, as a flat vector.

    A portal game is read with its group moves as the stand-in piece it declared before
    spelling 31 (`schema.portal_stand_in_shape`), so no feature moves with the re-spelling.
    """
    spec = portal_stand_in_shape(spec)
    board = spec.get("board", {})
    width = float(board.get("width", 8))
    height = float(board.get("height", 8))
    pieces = spec.get("pieces", []) or []
    rules = spec.get("rules", {}) or {}
    complexity = measure_complexity(spec)

    kinds: dict[str, int] = {"slide": 0, "leap": 0, "hop": 0, "sweep": 0}
    vectors = 0
    steps: list[float] = []
    # Counts of vectors that step along a rank or file, split by what the move is *for*.
    # Without this split, chess and berolina measure identical: berolina swaps the pawn's
    # two powers, so it has the same one orthogonal and one diagonal vector as chess and
    # differs only in which of them captures. That is a different game, and a distance
    # that calls it zero is not a distance.
    orthogonal = {"move": 0, "capture": 0}
    counted = {"move": 0, "capture": 0}
    for piece in pieces:
        for move in piece.get("moves", []) or []:
            kinds[str(move.get("kind", "slide"))] = kinds.get(str(move.get("kind")), 0) + 1
            mode = str(move.get("mode", "both"))
            for vector in move.get("vectors", []) or []:
                vectors += 1
                steps.append(float(max(abs(v) for v in vector)) if vector else 0.0)
                if mode in ("move", "capture"):
                    counted[mode] += 1
                    if 0 in vector:
                        orthogonal[mode] += 1
    total_moves = max(1, sum(kinds.values()))

    # The royal piece's own geometry. Knightmate is chess with the royal piece moving
    # like a knight and the knights moving like kings: every count above is identical,
    # and what actually changed is what has to be trapped.
    royal = next((piece for piece in pieces if piece.get("royal")), None)
    royal_vectors = [
        vector
        for move in (royal or {}).get("moves", []) or []
        for vector in move.get("vectors", []) or []
    ]
    royal_step = (
        sum(max(abs(v) for v in vector) for vector in royal_vectors) / len(royal_vectors)
        if royal_vectors
        else 0.0
    )
    royal_slides = 1.0 if any(
        move.get("kind") == "slide" and move.get("range", 0) != 1
        for move in (royal or {}).get("moves", []) or []
    ) else 0.0

    army = sum(
        len(placement.get("squares", []) or [])
        for placement in spec.get("setup", {}).get("placements", []) or []
    )

    return {
        "area": width * height,
        # Shape, not size: a 12x8 board and an 8x12 board are the same shape reversed,
        # and this keeps that distinct from a square one.
        "aspect": width / height if height else 1.0,
        "piece_types": float(len(pieces)),
        "army": float(army),
        "rule_mass": float(complexity.rule_mass),
        "special_rules": float(complexity.special_rules),
        "vectors_per_piece": vectors / max(1, len(pieces)),
        "slide_share": kinds["slide"] / total_moves,
        "leap_share": kinds["leap"] / total_moves,
        "hop_share": kinds["hop"] / total_moves,
        "sweep_share": kinds["sweep"] / total_moves,
        "short_sweep_share": sum(
            1 for piece in pieces for move in piece.get("moves", []) or []
            if sweep_length(move)
        ) / total_moves,
        "legs_share": sum(
            1 for piece in pieces for move in piece.get("moves", []) or []
            if isinstance(move, dict) and "legs" in move
        ) / total_moves,
        "lame_rider_share": sum(
            1 for piece in pieces for move in piece.get("moves", []) or []
            if move.get("kind") == "lame" and move.get("range", 1) != 1
        ) / total_moves,
        "turning_share": sum(
            1 for piece in pieces for move in piece.get("moves", []) or []
            if move.get("kind") == "slide" and move.get("path") in ("circle", "zigzag")
        ) / total_moves,
        "royal_types": float(sum(1 for piece in pieces if piece.get("royal"))),
        "royal_step": royal_step,
        "royal_slides": royal_slides,
        "move_orthogonal_share": orthogonal["move"] / counted["move"] if counted["move"] else 0.5,
        "capture_orthogonal_share": (
            orthogonal["capture"] / counted["capture"] if counted["capture"] else 0.5
        ),
        "mean_step": sum(steps) / len(steps) if steps else 0.0,
        "has_promotion": 1.0 if rules.get("promotion") else 0.0,
        "has_drops": 1.0 if rules.get("drops") else 0.0,
        **_stock_features(rules.get("drops")),
        "has_a_moveless_piece": 1.0 if any(
            isinstance(piece, dict) and not (piece.get("moves") or [])
            for piece in pieces
        ) else 0.0,
        "has_castling": 1.0 if rules.get("castling") else 0.0,
        "win_is_capture": 1.0 if rules.get("win", "checkmate") != "checkmate" else 0.0,
        "wins_by_arrival": 1.0 if rules.get("reach") else 0.0,
        "wins_by_check_count": 1.0 if rules.get("check_limit") else 0.0,
        "wins_by_extinction": 1.0 if rules.get("extinction") else 0.0,
        "wins_by_bare_royal": 1.0 if rules.get("bare_royal") else 0.0,
        # Read off the key as written rather than `schema.insufficient_material_live`, so an
        # inert rule elsewhere (a strike or contact table that changes nothing) measures as no
        # rule here as it does in every other feature.
        "draws_insufficient_material": 1.0 if rules.get("insufficient_material") is True else 0.0,
        "restricts_captures": 1.0 if any(
            isinstance(piece.get("captures"), list) for piece in spec.get("pieces", [])
        ) else 0.0,
        "has_territory": 1.0 if rules.get("territory") else 0.0,
        "repaints_territory": 1.0 if (rules.get("territory") or {}).get(
            "mode"
        ) == "repaint" else 0.0,
        "must_capture": 1.0 if rules.get("must_capture") else 0.0,
        **_wrap_features(rules.get("wrap"), board.get("shape")),
        **_explosion_features(rules.get("explode_on_capture")),
        "has_obstacles": 1.0 if (rules.get("obstacles") or {}).get("squares") else 0.0,
        **_strike_features(spec),
        **_contact_features(pieces, rules.get("contacts")),
        **_facing_features(pieces),
        **_npc_features(pieces, rules.get("npcs")),
        "has_duck": 1.0 if duck_piece(rules) is not None else 0.0,
        "duck_by_moves": (
            1.0 if duck_piece(rules) is not None and duck_goes(rules) == "by_moves" else 0.0
        ),
        "promotes_on_entry": 1.0 if rules.get("promote_on_entry") else 0.0,
        "no_repetition_draw": (
            1.0 if rules.get("repetition_draw") == REPETITION_NEVER else 0.0
        ),
        "hex_lattice": 1.0 if board.get("lattice") == "hex" else 0.0,
        "promotion_zone_share": _promotion_zone_share(spec),
        "promotes_on_leaving": _promotes_on_leaving(rules),
        # Schema 27 (Q936): "own" half way, so it is neither `true`'s game nor the summed
        # count's at distance zero.
        "extinction_per_type": (
            {True: 1.0, "own": 0.5}.get(rules["extinction"].get("per_type"), 0.0)
            if isinstance(rules.get("extinction"), dict)
            else 0.0
        ),
        "promotes_only_lost": _promotes_only_lost(rules),
        "wins_by_leaving": (
            1.0
            if isinstance(rules.get("reach"), dict) and rules["reach"].get("on") == "leave"
            else 0.0
        ),
        "reach_runners": 1.0 if len(reach_pieces(rules.get("reach"))) > 1 else 0.0,
        **_portal_features(rules.get("portals")),
        **_simultaneous_features(spec),
        **_rest_features(spec),
        **_sight_features(spec),
        **_neutral_features(spec),
        **_bounty_features(spec),
        **_purse_features(spec),
        **_push_features(spec, total_moves),
    }


# Q1074b5: each push knob's values, in the order a feature reads them from 0 to 1.
_PUSH_WHOSE = {"enemy": 0.0, "any": 0.5, "own": 1.0}
_PUSH_ALONG = {"move": 0.0, "rank": 0.5, "file": 1.0}
_PUSH_CHAIN = {"one": 0.0, "row": 0.5, "weight": 1.0}


def _push_features(spec: dict[str, Any], total_moves: int) -> dict[str, float]:
    """Q1074b5: the push, read through `schema.move_push`, `push_rule` and `push_lists`, the
    readers the validator and the compiler share; every feature 0 in a game without one."""
    rule = push_rule(spec)
    entries = [
        push for piece in spec.get("pieces") or [] if isinstance(piece, dict)
        for move in piece.get("moves") or []
        if (push := move_push(move)) is not None
    ]
    if rule is None or not entries:
        return {name: 0.0 for name in _PUSH_FEATURES}
    declared = [
        piece["id"] for piece in spec.get("pieces") or []
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    ]
    lists = push_lists(spec)
    pushable = pushable_ids(spec)
    reach = [
        len(pushable if listed is None else listed) / len(declared) if declared else 0.0
        for listed in lists.values()
    ]
    weighed = rule["chain"] == "weight"
    weights = piece_weights(spec) if weighed else {}
    strengths = piece_strengths(spec) if weighed else {}
    pushers = set(pushing_ids(spec))
    pusher_strengths = [value for pid, value in strengths.items() if pid in pushers]
    return {
        "push_share": len(entries) / total_moves,
        "push_whose": sum(_PUSH_WHOSE.get(e["whose"], 0.0) for e in entries) / len(entries),
        "push_captures": sum(1.0 for e in entries if e["captures"] is True) / len(entries),
        "push_along": sum(_PUSH_ALONG.get(e["along"], 0.0) for e in entries) / len(entries),
        "push_lists": (
            sum(1.0 for listed in lists.values() if listed is not None) / len(lists)
            if lists else 0.0
        ),
        "push_reach": sum(reach) / len(reach) if reach else 0.0,
        "push_chain": _PUSH_CHAIN.get(rule["chain"], 0.0),
        "push_edge_blocks": 1.0 if rule["edge"] == "blocks" else 0.0,
        "push_no_back": 1.0 if rule["no_push_back"] is True else 0.0,
        "push_walls_crush": 1.0 if rule["walls"] == "crush" else 0.0,
        "push_weight": sum(weights.values()) / len(weights) if weights else 0.0,
        "push_strength": (
            sum(pusher_strengths) / len(pusher_strengths) if pusher_strengths else 0.0
        ),
    }


_PUSH_FEATURES = (
    "push_share", "push_whose", "push_captures", "push_along", "push_lists", "push_reach",
    "push_chain", "push_edge_blocks", "push_no_back", "push_walls_crush", "push_weight",
    "push_strength",
)


def _purse_features(spec: dict[str, Any]) -> dict[str, float]:
    """TC5: the purse, its start, its region and the priced share of the declared types."""
    purse = purse_rule(spec)
    declared = [
        piece for piece in spec.get("pieces") or []
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    ]
    start = purse.get("start", 0) if purse is not None else 0
    return {
        "has_purse": 1.0 if purse is not None else 0.0,
        "purse_start": (
            start / PURSE_MAX if isinstance(start, int) and not isinstance(start, bool) else 0.0
        ),
        "purse_region": 1.0 if purse is not None and "region" in purse else 0.0,
        "price_share": len(piece_prices(spec)) / len(declared) if declared else 0.0,
        "has_hire": 1.0 if hire_ids(spec) else 0.0,
        "has_bribe": 1.0 if purse_bribes(spec) else 0.0,
        "bounty_to_loser": 1.0 if purse_pays_loser(spec) else 0.0,
    }


def _bounty_features(spec: dict[str, Any]) -> dict[str, float]:
    """TC4: the share of declared types whose taking pays a bounty."""
    declared = [
        piece for piece in spec.get("pieces") or []
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    ]
    paying = piece_bounties(spec)
    return {"bounty_share": len(paying) / len(declared) if declared else 0.0}


def _neutral_features(spec: dict[str, Any]) -> dict[str, float]:
    """TC3: the share of declared types some placement puts in nobody's block, the share of
    them moved by the majority, and whether a majority type names what next to means."""
    declared = [
        piece for piece in spec.get("pieces") or []
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    ]
    placed = neutral_ids(spec)
    held = majority_ids(spec)
    return {
        "nobody_share": len(placed) / len(declared) if declared else 0.0,
        "majority_share": len(held) / len(declared) if declared else 0.0,
        "majority_geometry": float(any(
            piece["id"] in held and isinstance(piece.get("control"), dict) for piece in declared
        )),
        # Q1077 (`docs/plan-turncoat.md` §4.7): the share of them whose held pieces take royals.
        "majority_takes_royal": (
            len(takes_royal_ids(spec)) / len(declared) if declared else 0.0
        ),
    }


def _sight_features(spec: dict[str, Any]) -> dict[str, float]:
    """Schema 30 (Q999a, `docs/archive/plan-passive.md` §4.3): for each effect, the share of declared
    piece types with a line of sight of it, and the share of sight entries that are auras, so
    a spec differing only by a sight, its effect or its blocking is not at distance zero by
    its features, beside the special rules `complexity` charges for it."""
    declared = [
        piece for piece in spec.get("pieces") or []
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    ]
    sights = piece_sights(spec)
    entries = [sight for read in sights.values() for sight in read]
    out = {
        f"sight_{effect}_share": (
            sum(1 for read in sights.values() if any(s.effect == effect for s in read))
            / len(declared) if declared else 0.0
        )
        for effect in SIGHT_EFFECTS
    }
    out["sight_aura_share"] = (
        sum(1 for sight in entries if sight.through) / len(entries) if entries else 0.0
    )
    # Q1007c (`docs/archive/plan-schema-31.md` §3.8): a command's `captures_own` at half the weight of
    # an effect (`sight_command_share` above), so a turncoat command is near its plain twin.
    out["sight_turncoat_share"] = (
        0.5 * sum(1 for read in sights.values() if any(s.captures_own for s in read))
        / len(declared) if declared else 0.0
    )
    return out


def _rest_features(spec: dict[str, Any]) -> dict[str, float]:
    """Schema 29 (Q983a, `docs/plan-rest.md` §5.7): the mean rest over the declared types and
    the largest, each over ``REST_MAX``, in any game, so a spec differing only by its rests
    is not at distance zero and a heavier rest reads farther than a lighter one. They replace
    schema 27's ``simultaneous_rest`` (the round's switch, `"rest": 1` on every piece now)."""
    rests = list(piece_rests(spec).values())
    if not rests or not any(rests):
        return {"rest_mean": 0.0, "rest_max": 0.0}
    return {
        "rest_mean": sum(rests) / len(rests) / REST_MAX,
        "rest_max": max(rests) / REST_MAX,
    }


def _simultaneous_features(spec: dict[str, Any]) -> dict[str, float]:
    """Schema 26 (Q918a): the round itself, its tie rule, and the share of pieces whose
    collision table (`beats`) is their own, so a spec differing only by any of them is not
    at distance zero; and schema 27's chase switches (Q942a), one feature each (the third, the
    round's rest, is schema 29's ``pieces[].rest``, `_rest_features`)."""
    rules: dict[str, Any] = spec["rules"] if isinstance(spec.get("rules"), dict) else {}
    rule = rules.get("simultaneous")
    tie = rule.get("tie") if isinstance(rule, dict) else None
    pieces = [piece for piece in spec.get("pieces") or [] if isinstance(piece, dict)]
    beats = sum(1 for piece in pieces if isinstance(piece.get("beats"), list))
    return {
        "simultaneous": 1.0 if isinstance(rule, dict) else 0.0,
        "simultaneous_initiative": 1.0 if tie == "initiative" else 0.0,
        "simultaneous_impossible": 1.0 if tie == "impossible" else 0.0,
        "beats_share": beats / len(pieces) if pieces else 0.0,
        "simultaneous_swap_collides": 1.0 if simultaneous_switch(rules, "swap") else 0.0,
        "simultaneous_both_score": 1.0 if simultaneous_switch(rules, "both") else 0.0,
        # Schema 27 (Q935a): the spy, so a spy game is never at distance zero from its
        # round twin; the count of spy types, so two spies a side read apart from one.
        "spy_types": float(len(spy_ids(spec))),
        **_speed_features(pieces),
    }


def _speed_features(pieces: list[dict[str, Any]]) -> dict[str, float]:
    """Schema 28 (Q943a, `docs/archive/plan-speed.md` §4.3): how many ranks of speed, and each type's
    speed over the top rank read three ways that need no piece id -- by the royal types, by
    declared value and by how many directions a type moves in -- so a game whose king is the
    fast piece and one whose knight is read apart, as do two games with different ranks."""
    speeds = piece_speeds(pieces)
    top = max(speeds.values(), default=0)
    if top == 0:
        return {"speed_ranks": 0.0, "speed_royal": 0.0, "speed_by_value": 0.0, "speed_by_reach": 0.0}

    def weighted(weight: Any) -> float:
        rows = [(weight(piece), speeds.get(str(piece.get("id")), 0) / top) for piece in pieces]
        total = sum(w for w, _ in rows)
        return sum(w * share for w, share in rows) / total if total else 0.0

    def reach(piece: dict[str, Any]) -> float:
        return float(sum(
            len(expand_symmetry(tuple(vector), move.get("symmetry", "none")))
            for move in piece.get("moves") or [] if isinstance(move, dict)
            for vector in move.get("vectors") or []
        ))

    def value(piece: dict[str, Any]) -> float:
        worth = piece.get("value")
        return float(worth) if isinstance(worth, int | float) and not isinstance(worth, bool) else 0.0

    return {
        "speed_ranks": float(top),
        "speed_royal": weighted(lambda piece: 1.0 if piece.get("royal") is True else 0.0),
        "speed_by_value": weighted(value),
        "speed_by_reach": weighted(reach),
    }


def _portal_features(portals: Any) -> dict[str, float]:
    """Schema 19: the group count as a share of four, and the per-group modifiers (schema
    20's bar on going straight back the third; schema 21's map shapes and filters), read
    through `portal_groups`."""
    groups = portal_groups({"portals": portals})
    ends = [end for group in groups for end in group["ends"]]
    shapes = {portal_map_shape(group) for group in groups}
    return {
        "portal_pairs": len(groups) / 4,
        "portals_move": 1.0 if any(end["moved_by"] for end in ends) else 0.0,
        "portals_one_way": 1.0 if any(
            end["enter"] != {0, 1} for end in ends if end["to"] is not None
        ) else 0.0,
        "portals_no_back": (
            1.0 if any(group["no_straight_back"] for group in groups) else 0.0
        ),
        "portals_ring": 1.0 if "ring" in shapes else 0.0,
        "portals_funnel": 1.0 if "funnel" in shapes else 0.0,
        "portals_other_map": 1.0 if "other" in shapes else 0.0,
        "portals_filtered": 1.0 if any(end["carries"] is not None for end in ends) else 0.0,
        "portals_fold": (
            sum(1 for group in groups if group["kind"] == "fold") / len(groups) if groups else 0.0
        ),
    }


def _promotes_only_lost(rules: dict[str, Any]) -> float:
    """Whether any promotion entry offers its targets only once lost (schema 18c)."""
    promotion = rules.get("promotion")
    entries = promotion if isinstance(promotion, list) else [promotion]
    return 1.0 if any(
        isinstance(entry, dict) and entry.get("only_lost") is True for entry in entries
    ) else 0.0


def _promotes_on_leaving(rules: dict[str, Any]) -> float:
    """Whether any promotion entry also promotes on a move out of its zone (schema 15)."""
    promotion = rules.get("promotion")
    entries = promotion if isinstance(promotion, list) else [promotion]
    return 1.0 if any(
        isinstance(entry, dict) and entry.get("on_leaving") is True for entry in entries
    ) else 0.0


def _promotion_zone_share(spec: dict[str, Any]) -> float:
    """The first player's promotion zone over the board's cells, walls and absent cells
    excluded from both sides, so a hexagon and its rhombus cut by walls read alike and
    the share stays within 0-1 (Q923: a walled zone square was counted, and one free
    cell under a two-square zone read 2.0)."""
    board = spec.get("board", {})
    width, height = int(board.get("width", 8)), int(board.get("height", 8))
    rules = spec.get("rules", {}) or {}
    orientation = (spec.get("setup") or {}).get("orientation", "mirror")
    promotion = rules.get("promotion")
    entries = promotion if isinstance(promotion, list) else [promotion]
    zone: set[int] = set()
    for entry in entries:
        if isinstance(entry, dict):
            zone |= promotion_squares(entry, width, height, orientation)[0]
    off = obstacle_squares(rules, width, height, orientation) | absent_squares(board)
    cells = width * height - len(off)
    return len(zone - off) / cells if cells > 0 else 0.0


def _npc_features(pieces: list[Any], npcs: Any) -> dict[str, float]:
    """Whether anything is wild, whether it eats, and whether taking it pays.

    What a wild piece eats is its type's `captures` list: absent is anything, an empty
    list is nothing. Its bounty defaults to its type's value, which is the compiler's
    reading too.
    """
    absent = {
        "has_npcs": 0.0, "npcs_bite": 0.0, "npcs_pay": 0.0, "npcs_are_hard_to_kill": 0.0,
        "npcs_share_a_walk": 0.0, "npcs_come_back": 0.0, "npcs_see": 0.0,
        "npcs_turn_back": 0.0, "npcs_pounce": 0.0, "npcs_pounce_attackers": 0.0,
        "npcs_split": 0.0,
    }
    if not isinstance(npcs, list) or not npcs:
        return absent
    by_id = {p.get("id"): p for p in pieces if isinstance(p, dict)}
    bites = pays = immune = shared = returns = sees = paces = hunts = guards = splits = False
    for npc in npcs:
        if not isinstance(npc, dict):
            continue
        piece = by_id.get(npc.get("piece"), {})
        captures = piece.get("captures")
        if captures is None or captures:
            bites = True
        bounty = npc.get("bounty", piece.get("value", 1))
        if isinstance(bounty, (int, float)) and not isinstance(bounty, bool) and bounty > 0:
            pays = True
        if isinstance(npc.get("immune"), list):
            immune = True
        if npc.get("also_at"):
            shared = True
        if npc_respawn(npc):
            returns = True
        if npc_turns_back(npc):
            paces = True
        if npc_pounce(npc) is not None:
            hunts = True
        if npc_pounce(npc) == "attackers":
            guards = True
        if npc_split(npc):
            splits = True
        if (by_id.get(npc.get("piece")) or {}).get("sight"):
            sees = True
    return {
        "has_npcs": 1.0,
        "npcs_bite": 1.0 if bites else 0.0,
        "npcs_pay": 1.0 if pays else 0.0,
        "npcs_are_hard_to_kill": 1.0 if immune else 0.0,
        "npcs_share_a_walk": 1.0 if shared else 0.0,
        "npcs_come_back": 1.0 if returns else 0.0,
        "npcs_see": 1.0 if sees else 0.0,
        "npcs_turn_back": 1.0 if paces else 0.0,
        "npcs_pounce": 1.0 if hunts else 0.0,
        "npcs_pounce_attackers": 1.0 if guards else 0.0,
        "npcs_split": 1.0 if splits else 0.0,
    }


def _facing_features(pieces: list[Any]) -> dict[str, float]:
    """Whether anything turns, how many ways it can point, and whether a facing is fixed."""
    facing = [
        piece
        for piece in pieces
        if isinstance(piece, dict) and isinstance(piece.get("facing"), dict)
    ]
    if not facing:
        return {"has_facing": 0.0, "turns_freely": 0.0, "facing_fixed": 0.0}
    axes = [piece["facing"].get("axes") for piece in facing]
    return {
        "has_facing": 1.0,
        "turns_freely": 1.0 if max(a for a in axes if isinstance(a, int)) > 2 else 0.0,
        "facing_fixed": 1.0 if any(not facing_turns(piece) for piece in facing) else 0.0,
    }


def _strike_features(spec: dict[str, Any]) -> dict[str, float]:
    """Whether a capture transforms, which way the chain points, and where they end up.

    A strike that keeps nothing alive is plain capture written the long way, and reads
    as zero here for the same reason the compiler drops it. That includes a chain whose
    every link starts at an immune piece, since a strike never reaches one. So whether
    anything survives is not read off the spec here but asked of the compiler, through
    the validator's `schema.strike_survives_anything` (`transform.keeps_a_survivor` hands
    it the piece ids). This function used to ask only for a non-``capture`` effect and a
    chain, which read such a chain as a strike: a spec that compiles to chess's own
    tables measured 0.018 to 0.036 from chess (Q414, the gap Q95 closed in
    `complexity`).
    """
    absent = {
        "strikes": 0.0,
        "strike_promotes": 0.0,
        "swaps_on_strike": 0.0,
        "strike_places_the_survivor": 0.0,
        "strike_stays_put": 0.0,
        "strike_rebirths": 0.0,
        "capturer_switches_sides": 0.0,
        "strikers_share": 0.0,
    }
    strike = (spec.get("rules") or {}).get("strike")
    if not isinstance(strike, dict):
        return absent
    effect = strike.get("effect", "capture")
    position = strike.get("position", "advance")
    # Schema 17 (Q540): a capture that keeps its capturer still, or one that sends it
    # over, is a rule with no survivor.
    stays_put = effect == "capture" and position == "stay"
    switches = bool(strike.get("switches_sides"))
    if not (keeps_a_survivor(spec, "strike") or stays_put or switches):
        return absent
    return {
        "strikes": 1.0,
        "strike_promotes": 1.0 if effect == "promote" else 0.0,
        "swaps_on_strike": 1.0 if position == "swap" else 0.0,
        "strike_places_the_survivor": 1.0 if position == "choose" else 0.0,
        "strike_stays_put": 1.0 if stays_put else 0.0,
        "strike_rebirths": 1.0 if position == "home" else 0.0,
        "capturer_switches_sides": 1.0 if switches else 0.0,
        "strikers_share": _strikers_share(spec, strike),
    }


def _strikers_share(spec: dict[str, Any], strike: dict[str, Any]) -> float:
    """A1: the listed types over every type whose captures the strike reaches."""
    by = strike.get("by")
    pieces = spec.get("pieces")
    if not isinstance(by, list) or not by or not isinstance(pieces, list):
        return 1.0
    firing = strikers(pieces, strike)
    if not firing:
        return 1.0
    return len(set(firing) & set(by)) / len(firing)


def _contact_features(pieces: list[Any], contacts: Any) -> dict[str, float]:
    """The contact table on the strike's three axes, plus the two that are its own.

    A contact table *is* a strike -- the validator refuses the two spellings together and
    the compiler puts them in one slot -- so it sets the same three features, and a
    ``hierarchy-fall`` re-spelled as a table measures zero from the original it re-spells.
    The two of its own are the axes only the table can move along: whether the *mover*
    changes, and whether what the target becomes depends on who took it.

    ``strike_promotes`` is read off the declared values rather than off an ``effect``
    key, because a table that names both new types has already said which way it points
    and a second declaration would be a second chance to disagree. A table that raises a
    value anywhere reads as promoting, which is the ``effect: "promote"`` reading: an
    ascending chain may also descend somewhere and is still an ascending game.

    **Only the links the compiler keeps are read** (Q417). A link onto an immune target
    is dropped, since a contact never reaches one, so a table whose every link lands on
    one compiles to no rule and reads as zero here, as a strike chain of only immune
    links does in `_strike_features`. Each link is asked of the compiler through the
    validator's `schema.strike_changes_anything`, whose answer for a table is the
    compiler's whole one: kept when something survives *or* only the attacker changes.
    `strike_survives_anything` would answer only the first half, and would drop a table
    whose links say ``mover_to`` alone. A table is kept exactly when one of its links is,
    because each link fills its own cell, so this is the table's answer too. It used to
    read every link, and a dropped one could set ``contacts_change_the_mover`` and
    ``strike_promotes`` beside live links. No committed or archived spec carries one.
    """
    absent = {
        "contacts_by_pair": 0.0,
        "contacts_change_the_mover": 0.0,
        "contacts_meet_own": 0.0,
        "contacts_keep_the_mover": 0.0,
    }
    if not isinstance(contacts, dict):
        return absent
    piece_ids = list(
        dict.fromkeys(
            piece["id"]
            for piece in pieces
            if isinstance(piece, dict) and isinstance(piece.get("id"), str)
        )
    )
    links = [
        link
        for link in (contacts.get("links") or [])
        if isinstance(link, dict)
        and strike_changes_anything({**contacts, "links": [link]}, "contacts", piece_ids)
    ]
    if not links:
        return absent

    values = {
        piece["id"]: float(piece.get("value", 1.0))
        for piece in pieces
        if isinstance(piece, dict) and "id" in piece
    }

    def raises(before: Any, after: Any) -> bool:
        return after in values and before in values and values[after] > values[before]

    promotes = any(
        raises(link.get("mover"), link.get("mover_to"))
        or raises(link.get("onto"), link.get("onto_to"))
        for link in links
    )
    # What the target becomes read two ways for one target is the vampire axis: the same
    # piece taken by two different pieces leaves two different things behind, which no
    # strike chain can say.
    by_target: dict[tuple[str, bool], set[str]] = {}
    for link in links:
        target = (str(link.get("onto")), link.get("own") is True)
        by_target.setdefault(target, set()).add(str(link.get("onto_to")))
    return {
        "strikes": 1.0,
        "strike_promotes": 1.0 if promotes else 0.0,
        "swaps_on_strike": 1.0 if contacts.get("position") == "swap" else 0.0,
        "strike_places_the_survivor": 1.0 if contacts.get("position") == "choose" else 0.0,
        "contacts_by_pair": 1.0 if any(len(s) > 1 for s in by_target.values()) else 0.0,
        "contacts_change_the_mover": (
            1.0 if any("mover_to" in link for link in links) else 0.0
        ),
        "contacts_meet_own": 1.0 if any(link.get("own") is True for link in links) else 0.0,
        "contacts_keep_the_mover": (
            1.0 if any(link.get("mover_to") == link.get("mover") for link in links) else 0.0
        ),
        # A1: the strike's `strikers_share` read off the table, the capturing types that
        # move in some live link over every capturing type, so a strike re-spelt as a
        # table measures as the strike does.
        "strikers_share": _movers_share(pieces, {str(link.get("mover")) for link in links}),
    }


def _movers_share(pieces: list[Any], movers: set[str]) -> float:
    capturing = {
        str(piece["id"])
        for piece in pieces
        if isinstance(piece, dict) and "id" in piece and piece.get("captures") != []
        and any(
            isinstance(move, dict) and move.get("mode", "both") != "move"
            for move in piece.get("moves") or []
        )
    }
    if not capturing:
        return 1.0
    return len(capturing & movers) / len(capturing)


def _stock_features(drops: Any) -> dict[str, float]:
    """Whether a hand starts full, and whether it can ever empty (schema 10, Q51)."""
    placement = (
        1.0 if isinstance(drops, dict) and drops.get("placement") is True else 0.0
    )
    either = (
        1.0 if isinstance(drops, dict) and drops.get("either_face") is True else 0.0
    )
    gating = 1.0 if isinstance(drops, dict) and isinstance(drops.get("gating"), dict) else 0.0
    absent = {
        "has_stock": 0.0,
        "stock_is_endless": 0.0,
        "drops_placement": placement,
        "drops_either_face": either,
        "drops_gating": gating,
    }
    stock = drops.get("stock") if isinstance(drops, dict) else None
    if not isinstance(stock, dict) or not stock:
        return absent
    return {
        "has_stock": 1.0,
        "stock_is_endless": 1.0 if any(
            held == STOCK_UNLIMITED for held in stock.values()
        ) else 0.0,
        "drops_placement": placement,
        "drops_either_face": either,
        "drops_gating": gating,
    }


def _explosion_features(explosion: Any) -> dict[str, float]:
    """Whether a capture explodes, and whether only some captures do.

    An empty `by` or `of` is a rule no capture can fire, so it reads as no explosion at
    all -- the same answer `engine.compile._compile_explosion` gives, and the reason this
    is a function rather than the truthiness test it replaces.
    """
    absent = {
        "explodes": 0.0, "explodes_selectively": 0.0, "explodes_alone": 0.0,
        "explodes_shaped": 0.0,
    }
    if not isinstance(explosion, dict) or not explosion:
        return absent
    narrowed = False
    for key in ("by", "of"):
        listed = explosion.get(key)
        if not isinstance(listed, list):
            continue
        if not listed:
            return absent
        narrowed = True
    return {
        "explodes": 1.0,
        "explodes_selectively": 1.0 if narrowed else 0.0,
        "explodes_alone": 1.0 if explosion.get("radius", 1) == 0 else 0.0,
        "explodes_shaped": 1.0 if "blast" in explosion else 0.0,
    }


def _wrap_features(wrap: Any, shape: Any = None) -> dict[str, float]:
    """Which axes or pairs of sides are joined, and whether the players cross them alike."""
    joined = {
        key: [str(edge) for edge in edges]
        for key, edges in wrap_stated(wrap, shape).items()
    }
    return {
        **{
            f"wraps_{key}": 1.0 if any(e != "none" for e in joined.get(key, ())) else 0.0
            for key in ("files", "ranks", *WRAP_PAIRS)
        },
        "wraps_asymmetrically": 1.0 if any(
            edges[0] != edges[1] for edges in joined.values()
        ) else 0.0,
    }


def distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    """How far apart two rulesets are: 0.0 for the same description, 1.0 for nothing
    in common. Symmetric, and zero only when every measured feature agrees."""
    left, right = features(first), features(second)
    total = 0.0
    for name, scale in _SCALES.items():
        gap = abs(left.get(name, 0.0) - right.get(name, 0.0)) / scale
        total += min(gap, 1.0)
    return round(total / len(_SCALES), 4)


def nearest(
    spec: dict[str, Any], others: dict[str, dict[str, Any]]
) -> list[tuple[str, float]]:
    """``others`` sorted by how close they are to ``spec``, closest first."""
    return sorted(
        ((name, distance(spec, other)) for name, other in others.items()),
        key=lambda pair: pair[1],
    )
