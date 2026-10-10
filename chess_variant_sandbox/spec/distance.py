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

from chess_variant_sandbox.spec.complexity import measure_complexity
from chess_variant_sandbox.spec.schema import (
    REPETITION_NEVER,
    STOCK_UNLIMITED,
    WRAP_PAIRS,
    absent_squares,
    obstacle_squares,
    promotion_squares,
    strike_changes_anything,
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
    # Schema 17 (Q540): a blast of radius 0 (kamikaze) is not atomic.
    "explodes_alone": 1.0,
    # Two, for the same reason the strike has three. A rule with a cost but no feature
    # collides with every other rule of the same cost, and two specs then report distance
    # zero -- which makes the loop skip a tournament it should have run. `turns_freely`
    # separates a piece that can point four ways from one that can only reverse, and
    # those are as different from each other as a demotion chain is from its reverse.
    "has_facing": 1.0,
    "turns_freely": 1.0,
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
}


def features(spec: dict[str, Any]) -> dict[str, float]:
    """The measured description of one ruleset, as a flat vector."""
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
        "promotes_on_entry": 1.0 if rules.get("promote_on_entry") else 0.0,
        "no_repetition_draw": (
            1.0 if rules.get("repetition_draw") == REPETITION_NEVER else 0.0
        ),
        "hex_lattice": 1.0 if board.get("lattice") == "hex" else 0.0,
        "promotion_zone_share": _promotion_zone_share(spec),
        "promotes_on_leaving": _promotes_on_leaving(rules),
        "extinction_per_type": (
            1.0
            if isinstance(rules.get("extinction"), dict)
            and rules["extinction"].get("per_type") is True
            else 0.0
        ),
        "promotes_only_lost": _promotes_only_lost(rules),
        "wins_by_leaving": (
            1.0
            if isinstance(rules.get("reach"), dict) and rules["reach"].get("on") == "leave"
            else 0.0
        ),
        **_portal_features(rules.get("portals")),
    }


def _portal_features(portals: Any) -> dict[str, float]:
    """Schema 19: the pair count as a share of four, and the per-pair modifiers (schema
    20's bar on going straight back the third)."""
    pairs = [pair for pair in portals if isinstance(pair, dict)] if isinstance(
        portals, list
    ) else []
    moved = any(
        isinstance(pair.get("moved_by"), dict)
        and any(pair["moved_by"].get(end) for end in ("orange", "blue"))
        for pair in pairs
    )
    one_way = any(
        isinstance(pair.get("passage"), dict)
        and any(word != "both_ways" for word in pair["passage"].values())
        for pair in pairs
    )
    return {
        "portal_pairs": len(pairs) / 4,
        "portals_move": 1.0 if moved else 0.0,
        "portals_one_way": 1.0 if one_way else 0.0,
        "portals_no_back": (
            1.0 if any(pair.get("no_straight_back") is True for pair in pairs) else 0.0
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
    excluded, so a hexagon and its rhombus cut by walls read alike."""
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
    return len(zone) / cells if cells > 0 else 0.0


def _npc_features(pieces: list[Any], npcs: Any) -> dict[str, float]:
    """Whether anything is wild, whether it eats, and whether taking it pays.

    What a wild piece eats is its type's `captures` list: absent is anything, an empty
    list is nothing. Its bounty defaults to its type's value, which is the compiler's
    reading too.
    """
    absent = {
        "has_npcs": 0.0, "npcs_bite": 0.0, "npcs_pay": 0.0, "npcs_are_hard_to_kill": 0.0
    }
    if not isinstance(npcs, list) or not npcs:
        return absent
    by_id = {p.get("id"): p for p in pieces if isinstance(p, dict)}
    bites = pays = immune = False
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
    return {
        "has_npcs": 1.0,
        "npcs_bite": 1.0 if bites else 0.0,
        "npcs_pay": 1.0 if pays else 0.0,
        "npcs_are_hard_to_kill": 1.0 if immune else 0.0,
    }


def _facing_features(pieces: list[Any]) -> dict[str, float]:
    """Whether anything turns, and how many ways it can point."""
    axes = [
        piece["facing"].get("axes")
        for piece in pieces
        if isinstance(piece, dict) and isinstance(piece.get("facing"), dict)
    ]
    if not axes:
        return {"has_facing": 0.0, "turns_freely": 0.0}
    return {
        "has_facing": 1.0,
        "turns_freely": 1.0 if max(a for a in axes if isinstance(a, int)) > 2 else 0.0,
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
    }


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
    }


def _stock_features(drops: Any) -> dict[str, float]:
    """Whether a hand starts full, and whether it can ever empty (schema 10, Q51)."""
    placement = (
        1.0 if isinstance(drops, dict) and drops.get("placement") is True else 0.0
    )
    either = (
        1.0 if isinstance(drops, dict) and drops.get("either_face") is True else 0.0
    )
    absent = {
        "has_stock": 0.0,
        "stock_is_endless": 0.0,
        "drops_placement": placement,
        "drops_either_face": either,
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
    }


def _explosion_features(explosion: Any) -> dict[str, float]:
    """Whether a capture explodes, and whether only some captures do.

    An empty `by` or `of` is a rule no capture can fire, so it reads as no explosion at
    all -- the same answer `engine.compile._compile_explosion` gives, and the reason this
    is a function rather than the truthiness test it replaces.
    """
    absent = {"explodes": 0.0, "explodes_selectively": 0.0, "explodes_alone": 0.0}
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
