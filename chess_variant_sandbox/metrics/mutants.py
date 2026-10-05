"""A corpus of deliberately perturbed rulesets, for calibrating the index against.

The index had twelve observations to be fitted against, ten of which carried the same
label. Sixty-odd free parameters -- fifteen curves with four points each, plus their
weights -- cannot be honestly set from that, and the consequence showed up as soon as
anyone looked: a third of the weight sat on components with no variance across the
whole reference corpus, and nothing in the pipeline would have said so.

The missing ingredient was never analysis. It was **negatives**. A scale that only ever
sees good games has no way to learn where "not a game" begins, so its zero ends get
guessed, and a guessed zero end is what produces a component that reads 1.00 for
everything including the controls.

So: take a game that works and break it, in ways whose direction is known in advance.
Each mutant carries an ``expectation``:

``worse``
    Something a designer would recognise as a defect. The index must score it below
    every human variant. When one of these lands inside the human band, the index is
    blind to that defect and the miss is specific enough to fix.

``same``
    A change that alters the spec, and therefore its hash, without altering the game in
    any way a player would notice. The index must *not* separate these from their base.
    A scale that moves on a renaming is reading noise, and this is the only check that
    catches it.

``probe``
    Genuinely uncertain, and held out of the fit. These are the interesting ones -- no
    castling, no en passant -- and they are recorded so that a future re-anchoring can
    be checked against them rather than tuned to them.

Nothing here is hand-designed. Every mutant is a mechanical transform of a committed
ruleset, so the corpus costs no design effort, no LLM calls, and regenerates from
scratch whenever the base rulesets change.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.spec.schema import format_square, parse_square
from chess_variant_sandbox.spec.transform import (
    all_slides_to_leaps,
    captures_come_back,
    confine_all_but,
    drop_piece,
    keep_only,
    most_numerous,
    never_forced,
    no_captures,
    omnipotent_piece,
    pieces,
    promote_early,
    promotion_rank,
    rename_everything,
    reorder_pieces,
    resize_board,
    reverse_backrank,
    royal_id,
    set_moves_to,
    set_rule,
    strip_rule,
    strongest,
    widen_royal,
)

EXPECTATIONS = ("worse", "same", "probe")


def flatten_army(spec: dict[str, Any]) -> dict[str, Any]:
    """Every non-royal piece moves like the strongest, and nothing unreachable is left.

    ``set_moves_to`` is the transform; the second half is bookkeeping that only bites
    on a base whose army is graded *through promotion*. Removing promotion leaves shogi
    declaring six promoted piece types that nothing places and nothing can promote into
    any more, and a spec still carrying declarations the game can no longer reach -- rule
    mass for nothing, charged by the elegance family under ``v10`` -- would fall for a
    reason that is a fact about the transform rather than about the game. So the unreachable declarations go with the rule that reached them,
    icons included. The icon sweep below is belt and braces since Q85 -- ``drop_piece``
    prunes ``appearance.icons`` itself now, along with the other five references it used
    to leave dangling -- and it is kept because it is idempotent and because this
    function, not ``drop_piece``, is what decides which pieces are unreachable.

    On chess and gardner every declared piece is placed, so this is exactly
    ``set_moves_to(strongest)`` there and both committed runs stay valid, byte for byte.
    """
    set_moves_to(spec, strongest(spec))
    placed = {
        str(placement["piece"])
        for placement in spec.get("setup", {}).get("placements", [])
    }
    for piece in list(pieces(spec)):
        if str(piece["id"]) not in placed and not piece.get("royal"):
            drop_piece(spec, str(piece["id"]))
    icons = spec.get("appearance", {}).get("icons")
    if isinstance(icons, dict):
        live = {str(piece["id"]) for piece in pieces(spec)}
        spec["appearance"]["icons"] = {
            key: value for key, value in icons.items() if key in live
        }
    return spec


# The idle piece's id. No base the wanderer is built on may already use it.
WANDERER_ID = "W"


def idle_wanderer(spec: dict[str, Any]) -> dict[str, Any]:
    """Give each side one extra piece that walks a file nothing else may enter.

    The board gains a file on each side and the army shifts one file right, so the
    centre the evaluator measures from stays where it was. Every existing piece is
    confined to the original squares. A region stops a piece *standing* on a square,
    and the new edge files lie beyond every ray, so no line, leap or capture of the
    original game changes. The wanderer stands on the left file, moves one step up or
    down, never captures and is worth nothing. The right file stays empty and only
    keeps the centre in place.

    Written for a base whose only square names are its placements. Anything else that
    names a square would need shifting too, so it is refused rather than left wrong.
    """
    board = spec["board"]
    width, height = int(board["width"]), int(board["height"])
    rules = spec.get("rules", {})
    refused = (
        "reach", "obstacles", "scoring", "territory", "npcs", "drops", "wrap",
        "promote_on_entry",
    )
    for key in refused:
        if key in rules:
            raise ValueError(f"idle_wanderer cannot shift rules.{key}")
    if any(piece.get("id") == WANDERER_ID for piece in pieces(spec)):
        raise ValueError(f"idle_wanderer needs the piece id {WANDERER_ID!r} free")

    def shifted(name: str) -> str:
        coords = parse_square(name, width, height)
        if coords is None:
            raise ValueError(f"idle_wanderer cannot parse square {name!r}")
        return format_square(coords[0] + 1, coords[1])

    for placement in spec.get("setup", {}).get("placements", []):
        placement["squares"] = [shifted(str(square)) for square in placement["squares"]]
    original = [
        format_square(file + 1, rank) for rank in range(height) for file in range(width)
    ]
    for piece in pieces(spec):
        region = piece.get("region")
        for move in piece.get("moves", []):
            if (move.get("condition") or {}).get("region"):
                raise ValueError("idle_wanderer cannot shift a move condition's region")
        piece["region"] = (
            [shifted(str(square)) for square in region] if region else original
        )

    spec["board"] = {**board, "width": width + 2}
    spec["pieces"].append({
        "id": WANDERER_ID,
        "name": "wanderer",
        "value": 0,
        "moves": [{"kind": "leap", "vectors": [[0, 1], [0, -1]], "mode": "move"}],
    })
    spec["setup"]["placements"].append({"piece": WANDERER_ID, "squares": ["a1"]})
    spec["schema_version"] = max(int(spec.get("schema_version", 1)), 2)
    return spec


@dataclass(frozen=True)
class Mutant:
    """One perturbation, with the reason it is expected to land where it does."""

    name: str
    base: str
    expectation: str
    rationale: str
    transform: Callable[[dict[str, Any]], dict[str, Any]]

    def build(self, base_spec: dict[str, Any]) -> dict[str, Any]:
        spec = self.transform(copy.deepcopy(base_spec))
        spec["name"] = self.name
        spec["description"] = f"Mutant of {self.base}: {self.rationale}"
        return spec


# ------------------------------------------------------------------- the corpus

MUTANTS: tuple[Mutant, ...] = (
    # --- known bad: nothing to choose between ---------------------------------
    Mutant(
        "mutant-uniform-army", "chess", "worse",
        "every piece moves like the strongest one, so no piece means anything different",
        flatten_army,
    ),
    Mutant(
        "mutant-all-pawns", "chess", "worse",
        "every piece moves like the most numerous one, which barely moves at all",
        lambda s: set_moves_to(s, most_numerous(s)),
    ),
    # Reclassified from "worse" after the first calibration run, and the reason is worth
    # keeping. Both were labelled broken on the assumption that removing long-range play,
    # or concentrating it in one piece, must ruin a game. The index scored them 0.74 and
    # 0.70 -- inside the human band -- and on inspection the index is right and the label
    # was wrong. A game of single-step pieces is roughly what shatranj is, and shatranj
    # is in the corpus at 0.83. A game with one dominant compound piece is roughly what
    # an amazon variant is. Neither is degenerate; both are simply different.
    #
    # This is what a labelled corpus is for. It tests the index against the labeller as
    # much as the other way round, and tuning the curves until the index agreed with
    # these two would have been fitting the index to a mistake.
    Mutant(
        "probe-one-piece-rules", "chess", "probe",
        "one piece reaches every square, which concentrates the game onto it",
        omnipotent_piece,
    ),
    Mutant(
        "probe-no-sliders", "chess", "probe",
        "every sliding piece is clipped to a single step, which is close to what a "
        "shatranj-shaped game already does",
        all_slides_to_leaps,
    ),
    # --- known bad: nothing can be resolved ------------------------------------
    Mutant(
        "mutant-no-captures", "chess", "worse",
        "no move may capture, so no advantage can ever be converted",
        no_captures,
    ),
    # Also reclassified. This one scored 0.86, above every game in the corpus, which was
    # the single most alarming number in the first run. It is still not evidence of a
    # blind spot: a king with extended movement is a real variant family, the games it
    # produces are decisive, and search still pays in them. The smoke gate does reject
    # it -- 95% of *random* games stall, because a runaway king is very hard to corner
    # by accident -- and that disagreement between the cheap gate and the full index is
    # itself the interesting finding, recorded rather than tuned away.
    Mutant(
        "probe-running-king", "chess", "probe",
        "the royal piece slides six squares, so it can outrun most attacks",
        lambda s: widen_royal(s, 6),
    ),
    Mutant(
        "mutant-bare-kings", "chess", "worse",
        "nothing on the board but the royal pieces",
        lambda s: keep_only(s, {royal_id(s) or ""}),
    ),
    Mutant(
        "mutant-royal-and-pawns", "chess", "worse",
        "the royal piece and the most numerous piece, with nothing else",
        lambda s: keep_only(s, {royal_id(s) or "", most_numerous(s) or ""}),
    ),
    # --- known bad: the shape of the board is wrong ----------------------------
    Mutant(
        "mutant-vast-board", "chess", "worse",
        "the same army on a 16x16 board, mostly empty space",
        lambda s: resize_board(s, 16, 16),
    ),
    Mutant(
        "mutant-cramped-board", "chess", "worse",
        "the same army crushed onto a board barely wider than the back rank",
        lambda s: resize_board(s, 8, 4),
    ),
    # --- known bad: an ending that rewards the wrong thing ----------------------
    Mutant(
        "mutant-instant-promotion", "chess", "worse",
        "promotion two ranks in front of the pawns, so the strongest piece is available "
        "within a couple of moves and the pawn structure never becomes a structure",
        promote_early,
    ),
    # Labelled "worse" on the theory that rewarding self-immobilisation is a perverse
    # incentive. Measured, it is a rule that never fires: across 144 games the outcome
    # breakdown and the per-piece move counts are byte-identical to chess, because
    # stalemate simply does not arise between these agents. The index scored it 0.811
    # against chess's 0.819, and under ``v10`` the entire difference was ``rule_yield``
    # charging it for a rule that buys nothing, which is what elegance-as-yield was for.
    # With rule yield out of the score (from ``v11``) nothing in the index charges it,
    # and it stays one game with a second spelling. A probe, and a good demonstration
    # that "this rule is perverse" and "this rule matters" are separate claims.
    Mutant(
        "probe-stalemate-wins", "chess", "probe",
        "running out of moves wins the game, which never comes up at this roster's "
        "strength and so costs only the rule it adds",
        lambda s: set_rule(s, "stalemate", "win"),
    ),
    # --- the same, on a second base, so nothing is an artefact of one ruleset ---
    # --- known bad: aimed at failure modes the corpus did not contain -------------
    #
    # Added for the v5 campaign, and aimed rather than drawn: the ten "worse" mutants
    # above all break a game by flattening its *pieces*, so the corpus had no example of
    # a game whose pieces are fine and whose *process* is broken. Every candidate
    # component from stage 12 measures a process, and a corpus with no such negatives
    # cannot tell whether one of them discriminates.
    #
    # Written before any weight was decided, which is the only order that makes them
    # evidence rather than decoration.
    # Two of the four were labelled "worse" and measured healthy, and the labels were
    # wrong. This is the second time a batch of confident labels has lost an argument
    # with the index -- the first cost four mutants their "worse" label and is recorded
    # above -- and the correction is the same: reclassify, write down why, and do not
    # touch a curve.
    #
    # `mutant-endless-shuffle` was built on the hypothesis that removing promotion and
    # slackening the no-progress clock would make chess shuffle. Against searching
    # players it does not: stall rate 0.087, decisive rate 0.862, 197 Elo per doubling.
    # It is chess without promotion, which is roughly shatranj, which scores 0.813. The
    # failure it was aimed at -- a designer's "back and forth and back and forth" -- is a
    # judgement made by a *person* over many sittings, and CLAUDE.md already says agents
    # do not get bored. The mutant does not exhibit the thing it was built to exhibit.
    Mutant(
        "probe-endless-shuffle", "chess", "probe",
        "no promotion and a no-progress clock pushed out to the ply limit -- built to "
        "shuffle, and measured decisive in 86% of games between competent players",
        never_forced,
    ),
    # `mutant-everything-returns` was built on the hypothesis that handing captures back
    # makes a lead a loan and stops games resolving. Measured: stall rate 0.000, decisive
    # rate 1.000, mean length 47 plies against chess's 77. Returning captures makes a
    # game *sharper*, which is why crazyhouse and shogi are good games rather than
    # despite it, and crazyhouse scores 0.814 in this same corpus.
    Mutant(
        "probe-everything-returns", "chess", "probe",
        "every captured piece returns to its captor's hand -- built to stall, and "
        "measured the most decisive ruleset in the corpus",
        captures_come_back,
    ),
    Mutant(
        # These two kept their label: both land below the human floor, so the index
        # agrees with them. `mutant-idle-army` is caught by `piece_utilisation` (0.560
        # against chess's 0.956) rather than by `instance_participation` (0.666 against
        # 0.743), which is the reverse of what stage 12 predicted and is recorded in the
        # report rather than fixed.
        "mutant-idle-army", "chess", "worse",
        "the minor pieces and rooks are pinned to the squares they start on, so a third "
        "of the army is present, counted, and incapable of doing anything, while the "
        "game itself still works",
        lambda s: confine_all_but(s, most_numerous(s), royal_id(s), strongest(s)),
    ),
    Mutant(
        "mutant-gardner-shuffle", "gardner", "worse",
        "nothing forces progress, on the smallest serious board",
        never_forced,
    ),
    Mutant(
        "mutant-gardner-uniform", "gardner", "worse",
        "every piece moves like the strongest one, on the smallest serious board",
        flatten_army,
    ),
    Mutant(
        "mutant-gardner-no-captures", "gardner", "worse",
        "no move may capture, on the smallest serious board",
        no_captures,
    ),
    Mutant(
        "probe-gardner-running-king", "gardner", "probe",
        "the royal piece slides four squares on a five-square board",
        lambda s: widen_royal(s, 4),
    ),
    # --- four more bases, so the negatives are not two games in a trenchcoat -----
    #
    # Every "worse" mutant above descends from chess or from gardner, which is chess on
    # a 5x5 board. That makes the negatives load-bearing twice over on one family of
    # geometries: sign accuracy is a count over them, and ``anchor_curves`` places every
    # ``zero_at`` on the *median of their raws*, so "where not-a-game begins" has been
    # answered by two rulesets that share an army, a board topology and a win condition.
    #
    # These four extend the two transforms with the best evidence behind them -- the
    # only two that kept their `worse` label on both existing bases -- to four bases
    # that share none of that: drops and a graded army (shogi), a board with a river and
    # a palace and a piece that captures by jumping (xiangqi), a board with no edges
    # (torus chess), and a game won by claiming ground rather than by mate
    # (rps territory).
    #
    # **The transform is the claim, and the base is the variable.** Three of the four
    # are the same flattening that took chess to 0.623 and gardner to a gate, so what is
    # being tested is whether "no piece means anything different" is a defect of games
    # or a defect of chess. The fourth cannot be: ``flatten_army`` on an RPS game is a
    # *provable null* -- rock, paper and scissors already move identically and the whole
    # asymmetry lives in ``captures``, so the transform returns the base spec with the
    # same content hash. That is the case `.claude/rules/metrics-and-calibration.md`
    # names when it refuses a move-set-diversity component, arriving here as an identity
    # rather than as an argument, and it is why the territory base takes the other
    # transform instead of being dropped.
    #
    # Written, argued and committed **before any of the four was played** (Q49). A label
    # chosen after seeing a score is not a label, and the git history is the evidence.
    Mutant(
        "mutant-shogi-uniform", "shogi", "worse",
        "every piece moves like the dragon and the promotion ladder goes with it, so an "
        "army graded across eight ranks of piece -- which is most of what the design is "
        "-- becomes one piece repeated, still handed back and forth by the drop rule",
        flatten_army,
    ),
    Mutant(
        "mutant-xiangqi-uniform", "xiangqi", "worse",
        "every piece moves like the chariot, so the cannon's screen, the horse's block "
        "and the elephant's river stop existing and the palace walls are the only thing "
        "left telling two pieces apart",
        flatten_army,
    ),
    Mutant(
        "mutant-torus-uniform", "torus chess", "worse",
        "every piece moves like the queen on a board with no edges, where a slider's "
        "reach wraps and nothing is ever out of range of anything",
        flatten_army,
    ),
    Mutant(
        "mutant-territory-no-captures", "rps territory", "worse",
        "no move may capture, so the cyclic capture relation that carries the whole "
        "asymmetry never fires and two armies race for ground on a board where they "
        "cannot touch each other -- the designer's own complaint that the players stop "
        "interacting, taken to its limit",
        no_captures,
    ),
    # --- nulls: the spec changes, the game does not ----------------------------
    Mutant(
        "null-renamed", "chess", "same",
        "every piece renamed and nothing else touched",
        rename_everything,
    ),
    # Was labelled a null and is not one. Reflecting chess's back rank exchanges the
    # king and queen -- R N B Q K B N R becomes R N B K Q B N R -- which is a different
    # opening game, not the same game written backwards, and the transform drops castling
    # on top of that. A mirror image would have been a genuine null; this is a variant,
    # so it is recorded as a probe and the null role goes to piece reordering, which is
    # provably gameplay-neutral.
    Mutant(
        "probe-mirrored-setup", "chess", "probe",
        "the back rank laid out left-to-right, which swaps the king and queen, with "
        "castling removed because it no longer refers to the same squares",
        reverse_backrank,
    ),
    Mutant(
        "null-reordered-pieces", "chess", "same",
        "the same pieces declared in the opposite order, which changes every internal "
        "index and therefore which games get played, but changes no rule",
        reorder_pieces,
    ),
    Mutant(
        "null-reordered-gardner", "gardner", "same",
        "the same pieces declared in the opposite order",
        reorder_pieces,
    ),
    Mutant(
        "null-renamed-gardner", "gardner", "same",
        "every piece renamed and nothing else touched",
        rename_everything,
    ),
    # The deciding fixture of `docs/archive/plan-variety.md` §3.6, committed with its label
    # before any game of it is played. The game is chess plus a piece per side that can
    # only shuffle on a file nobody else may enter, so the games differ from chess's
    # only where the idle piece wanders.
    #
    # Declared direction: `same` on the whole index. For the variety family the plan
    # fixes what a reading means. If `position_divergence` (3.2) rises above chess by
    # more than twice its noise, shuffle room is being scored as variety and
    # `event_divergence` (3.3), which ignores reversible plies, replaces it.
    #
    # Not byte-identical play, and not meant to be. A spare reversible move removes
    # zugzwang and stalemate, the extra piece enters `advance`'s average, and the regions
    # lengthen the spec, so under `v10` elegance reads a longer rulebook for the same game.
    Mutant(
        "null-idle-wanderer", "chess", "same",
        "each side gains a piece that steps up and down a file no other piece may "
        "enter, never captures and is worth nothing, so the game is chess with room "
        "to shuffle",
        idle_wanderer,
    ),
    # --- probes: held out of the fit, recorded for the argument ----------------
    Mutant(
        "probe-no-castling", "chess", "probe",
        "castling removed",
        lambda s: strip_rule(s, "castling"),
    ),
    Mutant(
        "probe-no-en-passant", "chess", "probe",
        "en passant removed",
        lambda s: strip_rule(s, "en_passant"),
    ),
    Mutant(
        "probe-no-promotion", "chess", "probe",
        "promotion removed, which is most of what a shatranj-shaped game does without",
        lambda s: strip_rule(s, "promotion", "en_passant"),
    ),
    Mutant(
        "probe-midboard-promotion", "chess", "probe",
        "promotion on the sixth rank rather than the eighth",
        lambda s: promotion_rank(s, 5),
    ),
    Mutant(
        "probe-wide-board", "chess", "probe",
        "the same army on a board two files wider",
        lambda s: resize_board(s, 10, 8),
    ),
    Mutant(
        "probe-no-royal-check", "chess", "probe",
        "the royal piece is captured rather than mated",
        lambda s: set_rule(s, "win", "royal_capture"),
    ),
)


def build_mutants(
    bases: dict[str, dict[str, Any]], only: tuple[str, ...] = ()
) -> list[tuple[Mutant, dict[str, Any]]]:
    """Instantiate every mutant whose base ruleset was supplied.

    Returns pairs so the expectation travels with the spec: a mutant score is
    meaningless without the direction it was supposed to move in.
    """
    built: list[tuple[Mutant, dict[str, Any]]] = []
    for mutant in MUTANTS:
        if only and mutant.expectation not in only:
            continue
        base = bases.get(mutant.base)
        if base is None:
            continue
        built.append((mutant, mutant.build(base)))
    return built
