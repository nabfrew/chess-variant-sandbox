"""A rule spec, written out as a Fairy-Stockfish variant definition.

This is the translation the whole integration rests on, and its governing rule is the
one `bridge.rust_supports` already states for the other engine this project speaks to:
**being wrong in the permissive direction is silent.** A ruleset Fairy-Stockfish cannot
express is refused here by name; a ruleset it can express *nearly* is the failure this
module exists to not have, because a nearly-right variant plays a legal-looking game
under rules nobody wrote, and the opponent that results is neither ours nor theirs.

So nothing here is trusted. `translate` produces a candidate and `verify.py` plays it:
move-for-move agreement between the two engines over the opening tree and over whole
random games, on the ruleset's own board. Only a candidate that survives that is offered
as an opponent. **That check, not this file, is what makes the translation's moves safe**
-- this file is allowed to be optimistic about them, and the corpus report in
`docs/reference/fairy-stockfish.md` records which rulesets its optimism turned out to be
justified on.

**The results are another matter.** `verify` establishes the same legal moves in every
position it walks, not the same result. The win and draw rules this file writes
(`checkmateValue`, `stalemateValue`, the extinction and flag options, `checkCounting`,
`nMoveRule`, `nFoldRule`, and the royalty model `win` picks) are checked by
`results.check_results`, which asks both engines who won at the same positions (Q404).
A result rule that crosses only approximately gets its key in `Variant.approximates`
beside its note, so that check expects the difference, in the one shape the note
describes. The record (Q96) names the keys, and the value readings it found wrong are
refused by name in `_royal_mode` (Q403). Which moves reset the no-progress clock is
written out, and what cannot be written is noted (`_progress_lines`, Q411).

## What crosses, and what cannot

Fairy-Stockfish is a chess-variant engine with a fixed idea of what a board is, so the
schema's later additions are exactly what it has no words for:

| ours | theirs |
|---|---|
| `board.shape: "hexagon"` | nothing -- its boards are rectangles |
| `wrap` | nothing -- a board with no edges is not a shape it has |
| `obstacles` beside a hop, or stopping a leap over it | a wall there is a screen, and a leap passes over it |
| `strike`, `contacts` | nothing -- a capture removes, and it never changes the taker |
| `territory`, `scoring` | nothing -- no square changes hands |
| `turn` | nothing -- a side moves once |
| `piece.captures` | nothing -- what may be taken is not a piece property |
| `piece.facing` | nothing -- a notation describes one piece, a facing makes it several; a fixed one (`turns: false`) would be one piece per facing, and the translator writes one per type |
| a board over 12x10 | a compile-time ceiling |
| several promoting pieces, one able to leave its zone (shogi) | a promotion on the way out too |

Each is refused with the reason, and the reasons are collected rather than raised one at
a time, for the same reason `SpecError` collects them: a caller told one thing at a time
learns one thing at a time.

**And so is a rule this file has never heard of.** A refusal is only safe if the list of
things to refuse is exhaustive; a rule with no branch anywhere is not refused, it is
*dropped*, and the definition then describes a different game while claiming to describe
this one. `_RULES_TRANSLATED` and `_RULES_REFUSED` are exhaustive over the schema's own
`rules` keys, `_RULE_OBJECT_KEYS` says the same thing one level down, `_KEYS_BELOW_RULES`
says it for every other level of the spec (the board, the setup, a piece, a move), and
anything outside them is refused by name. Widening the DSL therefore costs a line here,
which is the point: it is a line of thought rather than a line of code.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from chess_variant_sandbox.engine.compile import (
    NO_REPETITION_DRAW,
    CompiledSpec,
    PromotionRule,
    _lame_path,
    frame_index,
)
from chess_variant_sandbox.fairy import betza
from chess_variant_sandbox.spec.complexity import expand_symmetry
from chess_variant_sandbox.spec.legs import stated_legs
from chess_variant_sandbox.spec.schema import (
    DEFAULT_RANGE,
    SpecError,
    blast_directions,
    format_square,
    parse_square,
    path_of,
)

# The one blast Fairy-Stockfish has (`blastOnCapture`): the eight squares round the capture.
_KING_STEP = frozenset((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0))

# Fairy-Stockfish's large-board build. Beyond this the engine returns nonsense rather
# than an error, which is why the ceiling is checked here rather than discovered later.
MAX_FILES = 12
MAX_RANKS = 10

# How many pieces can be given a definition: twenty-five custom slots plus the king.
# `spec.schema.MAX_PIECE_TYPES` is 24, so this has never bound -- it is checked anyway,
# because the day it does bind the symptom is a silently missing piece.
MAX_CUSTOM = 25

# The letters a piece may be given, in the order they are handed out. Familiar first, so
# that a translated chess reads as chess in a FEN and a bug is visible to a person.
_PREFERRED = "PNBRQKACDEFGHIJLMOSTUVWXYZ"

# Letters that mean something else somewhere in the pipeline and are never handed out.
# `k` is the king's by convention and taken explicitly; the rest are FEN punctuation in
# letter form or would collide with the castling field.
_RESERVED = frozenset("k")


class Untranslatable(Exception):
    """Raised when a ruleset has no Fairy-Stockfish counterpart.

    Carries every reason, not just the first. A ruleset that wraps *and* scores on
    territory should say both, so that a reader knows the answer is not one small
    change away.
    """

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


@dataclass(frozen=True)
class Variant:
    """One ruleset as Fairy-Stockfish sees it."""

    name: str
    config: str
    # Our piece index -> the letter Fairy-Stockfish knows it by, uppercase (first
    # player). The second player's is the lowercase of the same letter.
    chars: tuple[str, ...]
    # Our piece index of the royal piece, and how Fairy-Stockfish is told about it.
    #
    # `king` is the piece in the engine's *king slot*, and it is `None` far more often
    # than "this ruleset has no royalty": there are three ways to be royal over there
    # and only one of them uses that slot. `royalty` says which -- see `_royal_mode`.
    king: int | None
    # Whether a position has to carry a pocket, a check count, or castling rights, so
    # that `position.py` need not re-derive the same answers from the config text.
    drops: bool
    check_counting: bool
    castling: bool
    # Q1059a: whether the castling field also carries the gate files (`seirawanGating`).
    gating: bool = False
    # Which of the three royalty models this ruleset uses: "king", "pseudo", "extinct"
    # or "none". See `_royal_mode`.
    royalty: str = "none"
    # How a promotion is spelled in a move: "piece" names the new piece and "plus"
    # writes `+`. The same distinction `_promotion_lines` makes, carried rather than
    # guessed, because a move spelled the other way is not recognised at all.
    promotion_suffix: str = "piece"
    # What the translation approximated rather than reproduced. Never empty on a
    # ruleset that needed approximating, and shown wherever the opponent is offered.
    notes: tuple[str, ...] = field(default_factory=tuple)
    # The result rules among those notes, by `rules` key: the endings the two engines
    # are known to reach differently. Written beside the note it stands for, so that
    # `results.check_results` expects the difference rather than reporting it (Q404),
    # and only in the shape that note describes (`results.admits`, Q411).
    approximates: frozenset[str] = frozenset()
    # The letter of the piece whose mating drop `drops.no_mate` forbids, or "". Ours
    # leaves such a drop out of the legal moves, and theirs (``shogiPawnDropMateIllegal``)
    # lists it as a move that loses. `verify` accepts that one extra move of theirs, and
    # only where their own engine says the drop leaves them mated (Q453).
    drop_mate: str = ""


def variant_name(ruleset_hash: str) -> str:
    """What to call this ruleset over the wire.

    Keyed on the mechanics hash rather than on the spec's name, because the name is
    outside the hash: two rulesets called "chess" with different rules must not be able
    to load over one another in an engine that keeps its variants in a map.
    """
    return f"ng{ruleset_hash}"


def translate(spec: dict[str, Any] | CompiledSpec) -> Variant:
    """Write ``spec`` as a Fairy-Stockfish variant, or say why it cannot be."""
    if isinstance(spec, CompiledSpec):
        if not spec.source:
            raise Untranslatable(
                ["the compiled ruleset carries no source; it predates CompiledSpec.source"]
            )
        source = json.loads(spec.source)
        compiled = spec
    else:
        from chess_variant_sandbox.engine.compile import compile_spec

        source = spec
        compiled = compile_spec(spec)

    reasons: list[str] = []
    notes: list[str] = []
    approximates: set[str] = set()
    rules = source.get("rules", {})

    _check_board(compiled, reasons)
    _check_unsupported_rules(compiled, rules, reasons)
    _check_walls(source, compiled, reasons)
    _check_unconsidered_keys(source, reasons)

    chars = _assign_chars(compiled, reasons)
    royalty = _royal_mode(compiled, reasons)
    king = compiled.royal_types[0] if royalty == "king" else None
    slots = piece_slots(source, compiled, king)
    pieces = _piece_lines(source, compiled, chars, slots, reasons)
    if not reasons:
        _check_skipped_squares(source, compiled, reasons)
    if reasons:
        raise Untranslatable(reasons)

    lines: list[str] = [f"[{variant_name(compiled.hash)}]"]
    lines.append(f"maxRank = {compiled.height}")
    lines.append(f"maxFile = {compiled.width}")
    lines.extend(pieces)
    lines.extend(_region_lines(source, compiled, slots))
    lines.append(f"startFen = {start_fen(compiled, chars, king)}")
    lines.extend(
        _rule_lines(source, compiled, chars, slots, royalty, reasons, notes, approximates)
    )
    if reasons:
        raise Untranslatable(reasons)

    return Variant(
        name=variant_name(compiled.hash),
        config="\n".join(lines) + "\n",
        chars=tuple(chars),
        king=king,
        drops=compiled.drops is not None,
        check_counting=bool(compiled.check_limit),
        castling=compiled.castling is not None,
        gating=_gating(compiled),
        royalty=royalty,
        promotion_suffix=_promotion_suffix(compiled),
        notes=tuple(notes),
        approximates=frozenset(approximates),
        drop_mate=_drop_mate_char(compiled, chars, slots),
    )


# ------------------------------------------------------------------ what cannot cross


def _check_board(compiled: CompiledSpec, reasons: list[str]) -> None:
    if compiled.width > MAX_FILES or compiled.height > MAX_RANKS:
        reasons.append(
            f"the board is {compiled.width}x{compiled.height}; Fairy-Stockfish's large-board "
            f"build stops at {MAX_FILES}x{MAX_RANKS}"
        )
    if compiled.n_types > MAX_CUSTOM + 1:
        reasons.append(
            f"{compiled.n_types} piece types; Fairy-Stockfish has {MAX_CUSTOM} custom slots "
            "and a king"
        )


# Every key under `rules` that reaches Fairy-Stockfish, and what became of each. The two
# sets are exhaustive over `spec.schema._RULES_KEYS`, and `tests/test_fairy.py` says so.
#
# Exhaustive is the whole point, and it is not tidiness. Every branch in
# `_check_unsupported_rules` is a rule somebody thought about; a rule nobody thought
# about takes no branch at all, so it is neither translated nor refused but **dropped**,
# and the definition handed to the engine then states a different game from the spec.
#
# That is worse than most wrong translations, because `verify.py` cannot see it.
# Verification establishes the same legal moves in every position it walks, not the same
# result. So a dropped rule that only decides who wins (`checkmate`, `stalemate`,
# `max_plies`, a draw clock) leaves both engines offering exactly the same moves in every
# position while they play for different results, and the check that exists to hold the
# two readings together reports success. They agree falsely. A rule that does move a
# piece is caught by verification, but only where the walk happens to reach it, which is
# a matter of luck rather than of design.
#
# A *listed* result rule that is translated wrongly is just as invisible to `verify`, and
# no list helps with that. This guard stops the drop. `results.check_results` checks the
# reading, where a walk reaches the ending (`docs/reference/fairy-stockfish.md`, Q96 and
# Q404).
#
# So the next rule the DSL grows has to be put in one of these lists before a ruleset
# using it is offered to Fairy-Stockfish at all.

# Written into the variant definition, or into a `note` where the two only nearly agree.
# A rule here may still be refused for a *value* it cannot carry -- `check_limit` crosses
# as `checkCounting` with the count in the FEN, and is refused beside a royal outside
# the king slot (`_royal_mode`); `obstacles` crosses as `*` walls in the FEN, and is
# refused beside a hop or where `blocks_leaps` stops a leap (`_check_walls`) -- because
# being in this list says the translator reads the rule, not that every game it can
# express survives.
_RULES_TRANSLATED = frozenset(
    {
        "castling",
        "check_limit",
        "checkmate",
        "drops",
        # Generation 42 (N4c, `docs/archive/plan-npc.md` §4.6): the classic duck, `wallingRule = duck`.
        "duck",
        "en_passant",
        "explode_on_capture",
        "extinction",
        "forbid_check",
        "forbid_immobility",
        "max_plies",
        "must_capture",
        "no_progress_draw",
        "obstacles",
        "promotion",
        "reach",
        "repetition_draw",
        "royal_facing",
        "stalemate",
        "win",
    }
)

# Refused below by name, with the reason.
_RULES_REFUSED = frozenset(
    {
        "bare_royal",
        "npcs",
        "portals",
        "promote_on_entry",
        # TC5 (`docs/plan-turncoat.md` §5.7): read off `compiled.purse`.
        "purse",
        "scoring",
        "simultaneous",
        "strike",
        "contacts",
        "territory",
        "turn",
        "wrap",
    }
)

# Rules Fairy-Stockfish knowingly need not be told about, each with the reason it makes
# no difference to a game it could play.
#
# Empty, and the emptiness is the finding rather than an oversight: every rule the schema
# has so far either crosses or is refused. A rule belongs here only when leaving it out
# provably changes no game -- never because nobody has looked, which is the state this
# whole guard exists to make impossible.
_RULES_IRRELEVANT: dict[str, str] = {}

# The same hole one level down. `rules.explode_on_capture` crosses, so a key added
# *inside* it would be dropped exactly as a new top-level rule would -- and this is not
# hypothetical, it is the shape the next schema version is already planned in.
#
# Only the rules that cross need a table. One that is refused whole is guarded by the
# compiler's own answer instead: `compiled.strike is None` is the compiler saying the
# rule does nothing, and a new key that made an inert rule active would change that
# answer and trip the refusal below.
_RULE_OBJECT_KEYS: dict[str, frozenset[str]] = {
    "castling": frozenset({"king", "king_file", "king_steps", "rook", "rook_file"}),
    # `stock` is schema 10: a finite one is the startFen's pocket and an unlimited slot is
    # refused (`_check_unsupported_rules`). `placement` is schema 18 (Q784) and is written
    # as `mustDrop` by `_drop_lines`; `either_face` is schema 18 (Q785), `dropPromoted`
    # (`_drop_face_lines`).
    # `gating` is Q1059a: `gating` and `seirawanGating`, with the gate files in the FEN's
    # castling field (`_drop_lines`, `position._castling_field`).
    "drops": frozenset(
        {
            "from_captures", "no_doubled", "no_mate", "region", "stock", "placement",
            "either_face", "gating",
        }
    ),
    # `by` and `of` are schema 10 and are listed here so the catch-all stops applying to
    # them -- which, as this table's own comment says, is what stops the silence rather
    # than the whole job. The branch that decides what they mean is in
    # `_check_unsupported_rules`, and it refuses them.
    # `blast` is generation 48 (SR1b): a stated shape crosses only where it is the king's
    # step, the one blast Fairy-Stockfish has; `_check_unsupported_rules` refuses the rest.
    "explode_on_capture": frozenset({"immune", "radius", "by", "of", "blast"}),
    # Schema 6 (Q703). `squares` and `mirror` are read off the compiled walls, which
    # `position.py` writes as `*`; `blocks_leaps` is asked of the compiler by `_check_walls`.
    "obstacles": frozenset({"blocks_leaps", "mirror", "squares"}),
    # `per_type` is schema 18 (Q783): read off the compiled rule by `_ending_lines`, which
    # writes it as the per-type count `extinctionPieceTypes` always was over there.
    "extinction": frozenset({"min_count", "pieces", "per_type", "result"}),
    # `squares` is schema 14. A shaped zone crosses as square names in `promotionRegion*`.
    # `on_leaving` is schema 15 (Q452): read off the compiled entry by
    # `_check_leaving_the_zone`, which passes it on the map and refuses it on the list.
    # `only_lost` is schema 18 (Q783): read off the compiled entry by `_promotion_lines`,
    # which writes it as `promotionLimit` from the setup's counts.
    "promotion": frozenset(
        {
            "mandatory", "on_leaving", "only_lost", "piece", "rank", "ranks", "squares",
            "to", "to_by_player",
        }
    ),
    # `on` is schema 18 (Q786): read off the compiled rule by `_ending_lines`, which refuses
    # `leave` by name. Over there `flagMove` gives the other side one more move after a
    # flag is reached, which is not leaving (`general-fairy-fork.md`).
    "reach": frozenset({"mirror", "on", "piece", "result", "safe", "squares"}),
    # Generation 42 (N4c): `piece` is the wall `position.py` writes as `*`; `start` and `goes`
    # are read off the compiled `DuckRule` by `_check_duck`, which refuses all but the classic.
    "duck": frozenset({"goes", "piece", "start"}),
}


def _check_unsupported_rules(
    compiled: CompiledSpec, rules: dict[str, Any], reasons: list[str]
) -> None:
    """Every rule with no counterpart, named individually.

    Read off the *compiled* spec rather than off the JSON wherever the compiler already
    decides whether a rule is really there -- `{"effect": "capture"}` is a strike rule
    that compiles to nothing, and refusing it would refuse a game that is not different.

    Ends in a catch-all, because a list of refusals is only safe when it is exhaustive.
    """
    if compiled.wrap is not None:
        reasons.append("rules.wrap: Fairy-Stockfish has no board without edges")
    if compiled.absent:
        # Schema 16 (Q281a). A hexagon's corners are compiled into `obstacles` as well as
        # `absent` (`compile._obstacles_with_absent`), so without this branch the twin of a
        # walled hex board would be refused as one with walls, for a rule it does not have.
        # Read off the compiled cells, so `"shape": "grid"` written out refuses nothing.
        reasons.append("board.shape: Fairy-Stockfish has no hexagonal board")
    if compiled.strike is not None:
        # Schema 6 and schema 10 (Q19a) land in the same compiled slot, so one branch
        # refuses both -- but it names the key the spec actually wrote, because an error
        # pointing at `rules.strike` for a file that says `rules.contacts` sends a reader
        # looking for a rule that is not there. Read off the compiled spec like every
        # other refusal here, so `{"effect": "capture"}` still translates.
        reasons.append(
            f"rules.{compiled.strike.spelling}: over there a capture removes the piece "
            "and nothing else"
        )
    if compiled.neutrals is not None:
        # TC3 (`docs/plan-turncoat.md` §5.7): read off the compiled rule, since the key guard
        # does not see `setup.*` or `pieces[].*`; a `control` the validator admits always has a
        # nobody's placement beside it, so one branch names both.
        reasons.append(
            "setup.placements.player: a piece nobody owns has no counterpart in "
            "Fairy-Stockfish, where every piece is a side's"
        )
        if compiled.neutrals.has_majority:
            reasons.append(
                "pieces.control: a piece moved by whichever side has more pieces next to it "
                "has no counterpart there"
            )
    if compiled.bounty:
        # TC4 (`docs/plan-turncoat.md` §5.7): read off the compiled table, since the key guard
        # does not see `pieces[].*`; empty for every type without one.
        reasons.append(
            "pieces.bounty: a capture there pays nothing into a tally; a pocket is a count of "
            "pieces, not a price"
        )
    if compiled.purse is not None:
        # TC5 (`docs/plan-turncoat.md` §5.7): its gating and pockets are not a purse; a pocket
        # is a count of pieces, never a price. `pieces[].price` is read only beside it.
        reasons.append(
            "rules.purse: currency, and a market that sells pieces for it, have no "
            "counterpart in Fairy-Stockfish"
        )
    if compiled.territory is not None:
        reasons.append("rules.territory: no square changes hands in Fairy-Stockfish")
    if compiled.scoring is not None:
        reasons.append("rules.scoring: a clock there runs out to a draw, never to a count")
    if compiled.duck is not None:
        # Generation 42 (N4c, `docs/archive/plan-npc.md` §4.6): asked before `rules.turn`, since a
        # duck game's compiled turn is two moves, which is the duck's half and crosses.
        _check_duck(compiled, reasons)
    elif compiled.turn is not None:
        reasons.append("rules.turn: a side moves once per turn in Fairy-Stockfish")
    if _capture_matrix_beside_duck(compiled):
        reasons.append(
            "piece.captures: what a piece may take is not a property a piece can have there"
        )
    if compiled.multi_royal:
        reasons.append("more than one royal piece: there is one king slot")
    if compiled.bare_royal:
        reasons.append("rules.bare_royal: no counterpart in the current variant options")
    # `check_limit` other than three crosses since Q703: the count is the FEN's
    # (`position.fen` writes the checks remaining, `5+5` for five), as the built-in
    # `5check` has it, so nothing in the variant names the number.
    explosion = compiled.explosion
    stated = rules.get("explode_on_capture", {}).get("blast") if explosion is not None else None
    if stated is not None:
        if blast_directions(stated) != _KING_STEP:
            reasons.append(
                "rules.explode_on_capture.blast other than the king's step: the blast there "
                "is the eight adjacent squares"
            )
    elif explosion is not None and rules.get("explode_on_capture", {}).get("radius", 1) != 1:
        reasons.append("rules.explode_on_capture.radius other than 1: the blast there is adjacent")
    if explosion is not None and not explosion.fires_on_every_capture:
        # Schema 10 (Q18). `blastOnCapture` is a switch and not a table: over there every
        # capture explodes, and `blastImmuneTypes` narrows only what the blast *reaches*,
        # never who sets it off or what has to be taken. Read off the compiled spec, so a
        # `by` naming every type -- atomic written the long way -- refuses nothing, and an
        # empty `by` refuses nothing either because it compiles to no explosion at all.
        reasons.append(
            "rules.explode_on_capture.by / .of: blastOnCapture is a switch there, so "
            "every capture explodes and no option says which ones do"
        )
    if compiled.orientation not in ("mirror", "rotate"):
        reasons.append(f"setup.orientation {compiled.orientation!r}")
    if compiled.n_facings > 1:
        # Schema 8. A piece that has a facing reads its movement out of a *frame*, and
        # turning is a move -- there is nothing over there that a Betza notation could
        # say it with, because a notation describes one piece and this is a piece that
        # is several depending on which way it is pointing.
        if any(compiled.turns(t) for t in range(compiled.n_types)):
            reasons.append(
                "piece.facing: a piece that turns has no counterpart; a Betza notation "
                "describes one piece and a facing makes it several"
            )
        else:
            # Schema 25 (Q869). A facing no move turns could be written as one Fairy
            # piece per facing, but one type held as two would part the hand (a captured
            # piece comes back in its initial facing) and every promotion rule from the
            # spec's; refused rather than half-translated. Every committed game that uses
            # it wraps, which is refused beside it anyway.
            reasons.append(
                "piece.facing.turns: a facing no move turns would be a separate "
                "Fairy-Stockfish piece for each way it points, and each kind of piece is "
                "written as one"
            )
    if compiled.npcs is not None:
        # Schema 9. A wild piece belongs to neither side and moves when nobody is to
        # move. Fairy-Stockfish has two colours and a side to move, and nothing in a
        # variant definition acts between turns.
        reasons.append(
            "rules.npcs: a piece neither player owns has no counterpart; nothing in "
            "Fairy-Stockfish moves between turns"
        )
    if compiled.portals is not None:
        # Schema 19 (Q771). Fairy-Stockfish has no square that sends a piece somewhere
        # else, and no option approximates one (`docs/archive/plan-portals.md` §5); its walling
        # rules place a wall, and a wall is not a door. Read off the compiled spec, so an
        # empty or absent rule refuses nothing; schema 21's maps and filters live under
        # it. `tests/portal_oracle.py` is the second implementation the core's transits
        # are held against instead.
        reasons.append(
            "rules.portals: a square that sends a piece out of another has no counterpart; "
            "no variant option moves a piece anywhere but where its move ends"
        )
    if compiled.spy is not None:
        # Schema 27 (Q935a, `docs/plan-spy.md` §3.8). A spy game always declares rounds,
        # refused just below; named apart so the reason says which rule asked for them. The
        # alternating phase is checked through the alternating twin, as a round game's is.
        reasons.append(
            "pieces[].spy: a game whose turns become rounds when a spy is lost has no "
            "counterpart; Fairy-Stockfish always has a side to move"
        )
    if compiled.simultaneous is not None and compiled.simultaneous.speed:
        # Schema 28 (Q943a, `docs/archive/plan-speed.md` §4.6). Speed orders a round's two moves, so
        # it lives only beside the rounds refused just below; named apart so the reason says
        # which rule asked for them. Read off the compiled table, empty without a speed.
        # Speed changes no legal move at a round's start, so the twin check still holds.
        reasons.append(
            "pieces[].speed: a faster move made before a slower one in the same round has no "
            "counterpart; Fairy-Stockfish has no round"
        )
    if compiled.rest:
        # Schema 29 (Q983a, `docs/plan-rest.md` §5.5). Fairy-Stockfish has no per-piece
        # cooldown. Read off the compiled table, empty without a rest. The rest only filters
        # the generator, so the generator itself is still checked through the twin with every
        # rest removed, at positions where nothing rests; the filter is the rest oracle's
        # (`tests/rest_oracle.py`).
        reasons.append(
            "pieces[].rest: a piece that may not move for its player's next turns after it "
            "moves has no counterpart; Fairy-Stockfish has no per-piece wait"
        )
    if compiled.sight is not None:
        # Schema 30 (Q999a, `docs/archive/plan-passive.md` §4.5). Fairy-Stockfish has no piece whose
        # line of sight freezes, disarms or shields another (Q999a read the installed build:
        # pyffish 0.0.90's 121 built-in variants hold none of Madrasi's shape, so a
        # translation of S1 is not a task). Read off the compiled rule,
        # `None` without a sight. The effects only filter the generator and the attack test,
        # so the generator itself is still checked through the twin with every sight
        # removed; the effects are the sight oracle's (`tests/sight_oracle.py`). Q1007c's
        # `command` (`docs/archive/plan-schema-31.md` §3.8) is a sight too, refused here with the rest:
        # Fairy-Stockfish has no move of the opponent's pieces, and the commanded moves (the
        # one part of a command the twin does not see) are the sight oracle's as well.
        reasons.append(
            "pieces[].sight: a piece whose line of sight freezes, disarms, shields or commands "
            "the pieces it sees has no counterpart; Fairy-Stockfish has no passive effect and "
            "no move of the opponent's pieces"
        )
    if compiled.simultaneous is not None:
        # Schema 26 (Q918a). Fairy-Stockfish has a side to move and no round in which both
        # players move at once (`docs/archive/plan-simultaneous.md` §6.5). The per-side generator is
        # still checked: the alternating twin, the same file without `simultaneous` and
        # `beats`, offers each player exactly a round's moves, and `verify` reads the twin.
        # `tests/simultaneous_oracle.py` holds the resolution instead.
        reasons.append(
            "rules.simultaneous: both players moving at once has no counterpart; "
            "Fairy-Stockfish always has a side to move"
        )
    drops = compiled.drops
    # SR2 (`docs/plan-schema-next.md` §2.5 H1, H6): two pairs the validator lifted, whose
    # reading over there was not checked, refused by name. `promotionLimit` beside a hand
    # that fills from captures: whether it counts a piece in hand was not probed. A blast
    # beside `dropPromoted`: what a blast's victims leave in a hand was not probed.
    if drops is not None and drops.from_captures and any(
        rule.only_lost for rule in compiled.promotions
    ):
        reasons.append(
            "rules.promotion.only_lost beside rules.drops filling from captures: whether "
            "promotionLimit counts a piece in hand was not checked"
        )
    if drops is not None and compiled.explosion is not None and any(
        face >= 0 for face in drops.faces
    ):
        reasons.append(
            "rules.drops.either_face beside rules.explode_on_capture: what a blast leaves in a "
            "hand under dropPromoted was not checked"
        )
    if drops is not None and drops.unlimited:
        # Schema 10 (Q51). No variant option makes a slot inexhaustible over there:
        # `pocketSize` bounds a hand, and a hand counts down. A *finite* stock is the
        # startFen's pocket (`compiled_initial_state`), which Q784 lifted from this
        # refusal: the FEN's hand seeds one, as `seirawan`'s `[HEhe]` does
        # (`reference/general-fairy-fork.md`, probed with `[XXxx]`). Read off the
        # compiled rule, so a spec without an unlimited slot refuses nothing.
        reasons.append(
            'rules.drops.stock "unlimited": a hand there counts down, and no option makes '
            "a slot inexhaustible"
        )
    # A piece with no moves (schema 10's relaxation, which the stock gates) crosses since
    # Q703 as an empty notation, `customPieceN = x:`: the engine's own `immobile` piece is
    # `from_betza("")`, and a probe on `pyffish` 0.0.90 dropped one and found it made no
    # moves (`reference/general-fairy-fork.md` §1.3).
    #
    # `repetition_draw: "never"` crosses since Q703 as `nFoldRule = 0` (`_rule_lines`),
    # which switches the rule off over there (`position.cpp`, `if (n_fold_rule())`). It is
    # the sentinel `NO_REPETITION_DRAW` that `_rule_lines` must never write as a count: a
    # widened value under a listed key passes the key guard, and the two engines would
    # play different games while `verify` compared move lists that agreed (Q86).
    if compiled.entry_promotion is not None:
        # Schema 9's second key. Over there a dropped piece is never promoted and a
        # capture removes what it takes, so there is no event for the rule to fire on
        # and no variant option that spells it. Read off the compiled spec, so
        # `"promote_on_entry": false` -- the game that was already there -- refuses
        # nothing, the `{"effect": "capture"}` shape.
        reasons.append(
            "rules.promote_on_entry: a piece that arrives without moving never promotes "
            "in Fairy-Stockfish; no variant option promotes on a drop"
        )

    _check_unconsidered_rules(rules, reasons)


def _check_duck(compiled: CompiledSpec, reasons: list[str]) -> None:
    """The duck crosses in its classic form only (generation 42, N4c, §4.6).

    `wallingRule = duck` there is a wall every move sets down anew on any empty square but
    the one it stands on, starting off the board, which is our `goes: "anywhere"` with no
    `start`. Its letter is `*`, a wall's, so it has no sight, and nothing over there moves
    between turns. Each other form is refused by name, read off the compiled `DuckRule`.
    """
    duck = compiled.duck
    assert duck is not None
    if duck.by_moves:
        reasons.append(
            'rules.duck.goes "by_moves": the duck there goes to any empty square, never by '
            "a piece's moves"
        )
    if duck.start >= 0:
        reasons.append(
            "rules.duck.start: the duck there starts off the board, and the first move "
            "puts it down"
        )
    if compiled.sight is not None and compiled.sight.entries[duck.piece_type]:
        reasons.append("rules.duck with a sight: the duck there is a wall, which sees nothing")
    if compiled.npcs is not None:
        reasons.append(
            "rules.duck beside rules.npcs: nothing there moves between turns, so the wild "
            "phase after the duck's half has no counterpart"
        )


def _capture_matrix_beside_duck(compiled: CompiledSpec) -> bool:
    """Whether the capture matrix restricts anything the duck does not (generation 42).

    The compiler clears the duck's row and column (nothing takes it, it takes nothing),
    which over there is what a wall is; any other cleared cell is `piece.captures`.
    """
    if compiled.duck is None:
        return compiled.has_capture_matrix
    duck = compiled.duck.piece_type
    return any(
        not allowed
        for taker, row in enumerate(compiled.capture_allowed)
        for taken, allowed in enumerate(row)
        if duck not in (taker, taken)
    )


def _check_unconsidered_rules(rules: dict[str, Any], reasons: list[str]) -> None:
    """Rules this translator has never been through, at either level.

    Refused rather than dropped, and refused *without deciding what they mean*: this
    says only that nobody has held the two rule systems up against each other for this
    key, which is exactly what makes dropping it unsafe. Whoever adds the rule to the
    schema decides, in one of the three lists above.

    Keyed on the key's *presence*, which everything else here is deliberately not --
    the named refusals read the compiled spec so that an inert declaration
    (`{"effect": "capture"}`, `"promote_on_entry": false`) refuses nothing. That is
    consistent rather than an exception to it: presence is all there is to go on for a
    rule the compiler's own fields have no name for, and the moment somebody lists the
    key this stops applying to it and the compiled reading takes over. Listing a key is
    therefore not the whole job; it is what stops the silence while the branch is
    written.
    """
    for key in sorted(set(rules) - _RULES_TRANSLATED - _RULES_REFUSED - set(_RULES_IRRELEVANT)):
        reasons.append(
            f"rules.{key}: this translator has never been through it, so it would be "
            "dropped rather than refused and the variant would state a different game "
            "than the spec does"
        )

    for key, known in _RULE_OBJECT_KEYS.items():
        value = rules.get(key)
        entries = value if isinstance(value, list) else [value]
        unknown: set[str] = set()
        for entry in entries:
            if isinstance(entry, dict):
                unknown |= set(entry) - known
        for sub in sorted(unknown):
            reasons.append(
                f"rules.{key}.{sub}: rules.{key} does cross, and this part of it has "
                "never been through this translator, so it would be dropped inside a "
                "rule that otherwise looks translated"
            )


# Every key the schema allows outside `rules`, level by level, and what becomes of it
# (Q97). The guard above stopped at `rules`, and a key added under `setup`, in a piece or
# in a move descriptor was dropped exactly as a new rule was before Q94. Two of these
# levels `verify` barely sees: the opening array reaches the other engine only as the
# `startFen` written here, and a hand only as the pocket `position.py` writes from our
# state, so it never applies a setup key or a demotion of its own in any position it is
# asked about.
#
# Each key says one of three things, with the reason beside it:
#
# ``translated``  written into the definition or the FEN, or refused by name for a
#                 value it cannot carry;
# ``refused``     refused by name wherever it changes the game, read off the compiled
#                 spec so that an inert declaration refuses nothing;
# ``irrelevant``  changes no game Fairy-Stockfish is asked to play, for the reason given.
#
# There is no fourth kind for a movement key left to `verify`, and none is needed. The
# one movement key the notation cannot spell, `min_range`, is asked of the compiler
# (`_check_skipped_squares`) rather than hoped of the walk, which sees a dropped
# movement key only in a position that happens to use it.
#
# `piece.facing` holds an object and is not given a table. It is refused whole, so the
# compiler's own answer (`compiled.n_facings`) guards its keys, exactly as
# `_RULE_OBJECT_KEYS` says of a rule that is refused whole. `tests/test_fairy.py` holds
# every table to the schema's own key set for its level.
_TRANSLATED = "translated"
_REFUSED = "refused"
_IRRELEVANT = "irrelevant"

_KEYS_BELOW_RULES: dict[str, dict[str, tuple[str, str]]] = {
    "spec": {
        "schema_version": (
            _IRRELEVANT,
            "says which keys the validator admits, and each of those is listed here; "
            "the compiler reads nothing from it",
        ),
        "name": (_IRRELEVANT, "outside the mechanics hash"),
        "description": (_IRRELEVANT, "outside the mechanics hash"),
        "appearance": (_IRRELEVANT, "outside the mechanics hash: how a piece is drawn"),
        "provenance": (_IRRELEVANT, "outside the mechanics hash: where the rules came from"),
        "board": (_TRANSLATED, "its keys are listed below"),
        "pieces": (_TRANSLATED, "a piece's keys are listed below"),
        "setup": (_TRANSLATED, "its keys are listed below"),
        "rules": (_TRANSLATED, "its keys are in the three lists above"),
    },
    "board": {
        "width": (_TRANSLATED, "maxFile, inside the ceiling `_check_board` refuses past"),
        "height": (_TRANSLATED, "maxRank, the same"),
        "lattice": (
            _IRRELEVANT,
            "no table reads it: a hex spec's vectors are already steps on a width by "
            "height grid, and the tables and the notation are both written from them",
        ),
        "shape": (
            _REFUSED,
            "board.shape, read off the compiled absent cells, so a hexagon is refused "
            "and a grid written out refuses nothing",
        ),
    },
    "setup": {
        "placements": (_TRANSLATED, "startFen, written from the compiled opening board"),
        "mirror": (_TRANSLATED, "the same: it arrives only through the compiled board"),
        "orientation": (
            _TRANSLATED,
            "rotate is Fairy-Stockfish's own; mirror is refused for any piece it moves "
            "differently (`_mirror_matches_rotation`), and any other value by name",
        ),
    },
    "placement": {
        "piece": (_TRANSLATED, "through the compiled opening board"),
        "player": (
            _TRANSLATED,
            'through the compiled opening board; TC3 refuses "nobody" off the '
            "compiled rule (compiled.neutrals)",
        ),
        "squares": (_TRANSLATED, "through the compiled opening board"),
        "facing": (
            _REFUSED,
            "the compiler reads it only for a piece that has a facing, and such a piece "
            "is refused as piece.facing",
        ),
    },
    "piece": {
        "id": (_TRANSLATED, "the letter the piece is written with (`_assign_chars`)"),
        "name": (_IRRELEVANT, "a display name; nothing that decides a move or a result reads it"),
        "royal": (
            _TRANSLATED,
            "`_royal_mode`: the king slot, pseudo-royalty or an extinction rule, and "
            "the shapes none of those fits are refused by name",
        ),
        "value": (
            _IRRELEVANT,
            "a rule reads it only through rules.scoring's material count, a wild "
            "piece's bounty and a strike's change of material, and all three are "
            "refused whole; otherwise it is our agents' evaluation, and Fairy-Stockfish "
            "has its own",
        ),
        "moves": (_TRANSLATED, "one Betza term per descriptor; a descriptor's keys are below"),
        "region": (_TRANSLATED, "mobilityRegion (`_region_lines`)"),
        "promoted_from": (
            _TRANSLATED,
            "as the pocket and as a mark on the board: every position Fairy-Stockfish "
            "is asked about carries our hand, and a piece whose type demotes is written "
            "promoted (`board_letters`), because that engine demotes by history and "
            "ours by type. Its own promotions demote as ours do where each target's "
            "promoted_from is the piece that promotes into it, or with dropLoop where "
            "nothing demotes; any other shape is noted (`_demotion_lines`; record, "
            "Q97 and Q406)",
        ),
        "captures": (
            _REFUSED,
            "piece.captures, read off the compiled matrix so a list that permits "
            "every capture refuses nothing",
        ),
        "facing": (_REFUSED, "piece.facing, read off the compiled frame count"),
        "beats": (
            _REFUSED,
            "only beside rules.simultaneous, which is refused whole: Fairy-Stockfish has "
            "no collision for it to decide",
        ),
        "spy": (
            _REFUSED,
            "schema 27: a spy game alternates and then plays rounds, and Fairy-Stockfish has "
            "no round; refused with rules.simultaneous, which a spy game must declare, and "
            "read off the compiled rule (compiled.spy)",
        ),
        "speed": (
            _REFUSED,
            "schema 28: only beside rules.simultaneous, which is refused whole: Fairy-Stockfish "
            "has no round to order, and speed changes no legal move at a round's start, so the "
            "per-side generator is still checked through the alternating twin",
        ),
        "rest": (
            _REFUSED,
            "schema 29: a piece that waits its player's next turns after it moves, read off "
            "the compiled table (compiled.rest); Fairy-Stockfish has no per-piece wait, and "
            "the generator is checked through the twin with every rest removed",
        ),
        "control": (
            _REFUSED,
            "TC3: how a piece nobody owns is moved, read off the compiled rule "
            "(compiled.neutrals); Fairy-Stockfish has no piece that belongs to nobody",
        ),
        "bounty": (
            _REFUSED,
            "TC4: what taking a piece pays, read off the compiled table (compiled.bounty); "
            "Fairy-Stockfish has no tally a capture pays into",
        ),
        "price": (
            _REFUSED,
            "TC5: what a piece costs from the market (TC6: to hire or bribe), read only beside "
            "rules.purse, which is "
            "refused off the compiled rule (compiled.purse); a pocket is a count, not a price",
        ),
        "sight": (
            _REFUSED,
            "schema 30: lines of sight and their effects, read off the compiled rule "
            "(compiled.sight); Fairy-Stockfish has no passive effect, and the generator is "
            "checked through the twin with every sight removed. A sight entry's own keys are "
            "a level below the piece and are never translated, since the piece-level refusal "
            "already names the whole sight",
        ),
    },
    "move": {
        "kind": (
            _TRANSLATED,
            "leap, slide, hop and lame are Betza atoms; a sweep and an unknown kind are "
            "refused by name",
        ),
        "vectors": (_TRANSLATED, "the directions the Betza term is written from"),
        "symmetry": (_TRANSLATED, "the same directions, expanded first"),
        "range": (_TRANSLATED, "the term's distance"),
        "min_range": (
            _TRANSLATED,
            "left out of the notation, which always rides from the first square, and "
            "refused by name unless the compiler says leaving it out changes no table "
            "(`_check_skipped_squares`)",
        ),
        "mode": (_TRANSLATED, "the m and c modifiers"),
        "condition": (_TRANSLATED, "its keys are listed below"),
        "screen_range": (_TRANSLATED, "0 is p and 1 is g; any other is refused by name"),
        "sweep_range": (
            _REFUSED,
            "Q1063: only on a sweep, which `_descriptor_term` refuses by name whatever its "
            "lengths (Betza has no piece that captures what it passes over)",
        ),
        "path": (
            _REFUSED,
            "T1: a circle or a zigzag, by name in `_descriptor_term`, since pyffish reads "
            "XBetza's q and z as the plain piece; \"straight\", the off value, refuses nothing",
        ),
        "screen": (
            _TRANSLATED,
            "any crosses; a hop that chooses whose piece it jumps is refused by name",
        ),
        "legs": (
            _TRANSLATED,
            "SR1a: on a lame, the n modifier where every direction's legs are the squares "
            "`betza.lame_path` says Fairy-Stockfish's lame leaper is blocked on (Janggi's "
            "elephant is nZ), refused by name otherwise; on a leap, a wall's stop under "
            "blocks_leaps, which `_check_walls` asks of the compiler",
        ),
    },
    "move.condition": {
        "from_rank": (_TRANSLATED, "the i modifier and doubleStepRegion (`_pawn_lines`)"),
        "first_move": (
            _REFUSED,
            "by name where true: the unmoved mask it reads has no FEN field for a piece "
            "that does not castle",
        ),
        "region": (
            _REFUSED,
            "by name, except the xiangqi soldier's band of whole ranks, which is the "
            "engine's own soldier (`soldier_rank`)",
        ),
    },
}


def _check_unconsidered_keys(source: dict[str, Any], reasons: list[str]) -> None:
    """Keys outside `rules` this translator has never been through, refused by name.

    `_check_unconsidered_rules` over every other level of the spec, keyed on presence for
    the same reason. One reason per key and level, not per piece: a key nobody has
    considered is one decision however many pieces carry it.
    """
    unknown: set[tuple[str, str]] = set()

    def look(level: str, entry: Any) -> None:
        if isinstance(entry, dict):
            unknown.update((level, key) for key in set(entry) - set(_KEYS_BELOW_RULES[level]))

    look("spec", source)
    look("board", source.get("board"))
    setup = source.get("setup")
    look("setup", setup)
    for placement in setup.get("placements", []) if isinstance(setup, dict) else []:
        look("placement", placement)
    for piece in source.get("pieces", []):
        look("piece", piece)
        for move in piece.get("moves", []) if isinstance(piece, dict) else []:
            look("move", move)
            if isinstance(move, dict):
                look("move.condition", move.get("condition"))

    for level, key in sorted(unknown):
        reasons.append(
            f"{level}.{key}: this translator has never been through it, so it would be "
            "dropped rather than refused and the variant would state a different game "
            "than the spec does"
        )


def _check_walls(
    source: dict[str, Any], compiled: CompiledSpec, reasons: list[str]
) -> None:
    """Refuse the walls a `*` in the FEN does not stand for (Q703).

    Over there a `*` is a square nothing stands on and nothing takes. A slider stops
    before it, a leap cannot land on it, a lame leap whose leg is on it is blocked, and
    no piece is dropped there, which is what a compiled obstacle is here (probed on
    `pyffish` 0.0.90; `reference/general-fairy-fork.md` §1.3). Two readings part:

    * **A hop.** Their wall is an occupied square, so a cannon may use one as its screen.
      Ours ends the hop's line at a wall (`compile._compile_square_full`: *a wall is not a
      screen*). Refused wherever a piece hops, since which hops meet a wall is the
      compiler's geometry and not this file's.
    * **`blocks_leaps`.** Their leap passes over a wall. Ours stops a leap whose
      `_lame_path` square is a wall. Asked of the compiler, as `_check_skipped_squares`
      asks it of `min_range`: the spec compiled without the key is the game the walls
      describe there, and only equal tables translate. Quarry's knights are stopped, so
      it stays refused; a `blocks_leaps` that stops nothing refuses nothing.

    Castling across a wall is the schema's to refuse (`castling_rank_walled_beyond_its_spans`),
    and a blast leaves a wall standing in both engines. What else a wall touches is
    `verify`'s question, as for every translation.
    """
    if not compiled.obstacles - compiled.absent:
        return
    hoppers = sorted(
        str(piece.get("id"))
        for piece in source["pieces"]
        if any(move.get("kind") == "hop" for move in piece["moves"])
    )
    if hoppers:
        reasons.append(
            f"rules.obstacles beside a hop ({', '.join(hoppers)}): a wall there is a screen "
            "a hop may jump, and a wall here ends the hop's line"
        )
    walls = source.get("rules", {}).get("obstacles") or {}
    if not walls.get("blocks_leaps"):
        return
    from chess_variant_sandbox.engine.compile import compile_spec

    vaulted = copy.deepcopy(source)
    vaulted["rules"]["obstacles"].pop("blocks_leaps")
    try:
        same = _rule_tables(compile_spec(vaulted)) == _rule_tables(compiled)
    except SpecError:
        same = False
    if not same:
        reasons.append(
            "rules.obstacles.blocks_leaps: a leap there passes over a wall, and here a wall "
            "stops a leap this game makes"
        )


def _check_skipped_squares(
    source: dict[str, Any], compiled: CompiledSpec, reasons: list[str]
) -> None:
    """Refuse a `min_range` whose notation would add moves this game does not have.

    Betza has no ride that starts past its first square, so `_descriptor_term` writes a
    descriptor as if `min_range` were not there, and the piece over there may also stop
    on the squares the move skips. That is the same piece exactly when another of its
    moves already reaches those squares. It is so for every pawn's double step in the
    corpus, where the single step beside it covers the square the double step passes.

    The compiler is asked, not this file, because the compiler is the one place a move
    descriptor means anything. The spec is compiled again with every `min_range` left
    out, which is the game the notation describes, and the tables are compared. Only the
    fields a hash seeds are left out of the comparison, since the hash is all that leaving
    out a key changes about them. Equal tables are enough and not required: tables can
    differ and still generate the same moves, and such a ruleset is refused. The compiler
    already folds the likeliest case, a single step written as a leap beside a double step
    written as a slide, so that one translates. A refusal is loud and names the key; the
    permissive mistake would be silent.

    Asked last, and only of a ruleset nothing else has refused. It costs a second compile,
    and on a ruleset already refused it would add a reason about a notation that will not
    be written.
    """
    pieces = source["pieces"]
    if all(move.get("min_range", 1) == 1 for piece in pieces for move in piece["moves"]):
        return
    from chess_variant_sandbox.engine.compile import compile_spec

    widened = copy.deepcopy(source)
    for piece in widened["pieces"]:
        for move in piece["moves"]:
            move.pop("min_range", None)
    try:
        same = _rule_tables(compile_spec(widened)) == _rule_tables(compiled)
    except SpecError:
        same = False
    if not same:
        reasons.append(
            "move.min_range: Betza has no ride that starts past its first square, so the "
            "piece there may also stop on the squares this move skips, and the compiler "
            "does not find another move of the same piece that reaches them"
        )


def _rule_tables(compiled: CompiledSpec) -> tuple[Any, ...]:
    """Every compiled field that the rules decide, leaving out what the hash seeds."""
    return tuple(
        getattr(compiled, name)
        for name in (item.name for item in dataclasses.fields(compiled))
        if name not in ("hash", "source") and not name.startswith("zobrist_")
    )


# ------------------------------------------------------------------------ the pieces


def _assign_chars(compiled: CompiledSpec, reasons: list[str]) -> list[str]:
    """One uppercase letter per piece type, stable for a given ruleset.

    A piece keeps its own id where that is a single letter and nothing has taken it, so
    a translated position is readable beside one of ours. Everything else takes the next
    free letter in a fixed order, which makes the assignment a pure function of the
    ruleset and therefore reproducible from the spec alone.
    """
    chars: list[str] = ["" for _ in compiled.piece_ids]
    taken: set[str] = set()
    for index, pid in enumerate(compiled.piece_ids):
        wanted = pid[0].upper() if pid else ""
        if len(pid) == 1 and wanted.isalpha() and wanted not in taken:
            chars[index] = wanted
            taken.add(wanted)
    for index, char in enumerate(chars):
        if char:
            continue
        for candidate in _PREFERRED:
            if candidate not in taken and candidate.lower() not in _RESERVED:
                chars[index] = candidate
                taken.add(candidate)
                break
        else:
            reasons.append(f"no letter left for piece {compiled.piece_ids[index]!r}")
            chars[index] = "?"
    return chars


def _royal_mode(compiled: CompiledSpec, reasons: list[str]) -> str:
    """Which of Fairy-Stockfish's three royalty models this ruleset needs.

    There is more than one because a king over there is not a single idea:

    ``king``
        The engine's king slot: the ordinary check filter, checkmate, castling. What
        every plain chess variant wants.
    ``pseudo``
        An ordinary piece plus ``extinctionPseudoRoyal``. This is what Fairy-Stockfish
        does for its own atomic, and the reason is exact: under a blast rule "does this
        move leave my king attacked" and "could the attacker survive taking it" stop
        being the same question, and only the second one is the rule. Translating atomic
        with a real king produces a game where a rook may step onto an open file in
        front of the enemy queen -- a legal-looking game under rules nobody wrote, and
        the first thing `verify.py` caught.
    ``extinct``
        An ordinary piece under an extinction rule and no check filter at all, which is
        what ``win: royal_capture`` says: the game ends when the piece is taken, and
        nothing stops you leaving it where it can be taken.

    ``none`` is the fourth answer and the right one for annihilation.

    **Two shapes have no model here, and both are refused by name** (Q96 found them,
    Q403 refuses them; each was probed with `pyffish`). Both decide only the result, and
    `verify` compares moves, so neither is a thing it could be trusted to catch.

    * ``win: "none"`` beside a royal piece. The core does not end that game when the
      royal is lost (`Tables::royal_loss_ends_it`), and every model above does: the king
      slot and ``extinctionValue`` both make losing it a loss. Writing the royal as an
      ordinary piece instead would be faithful only where the flag means nothing, and
      under ``"none"`` the flag exists to be read by ``check_limit``, ``forbid_check``
      and ``royal_facing`` -- rules Fairy-Stockfish ties to the king slot.
    * ``check_limit`` beside any model but ``king``. Fairy-Stockfish counts checks only
      against its king slot: a check on an ``extinct`` or ``pseudo`` royal leaves the
      FEN's counter where it was, so ours ends the game at the third check and theirs
      never does. No other model fits, since ``royal_capture`` has no check filter and a
      blast rule's royal is pseudo-royal precisely so that it is not in the king slot.

    The mode is still returned, the same one as before, so the refusal adds its own
    reason and nothing downstream objects to anything else.
    """
    if not compiled.royal_types:
        return "none"
    if len(compiled.royal_types) > 1:
        reasons.append("more than one royal piece type; there is one king slot")
        return "none"
    if compiled.win == "checkmate":
        mode = "pseudo" if compiled.explosion is not None else "king"
    else:
        mode = "extinct"

    if compiled.win == "none":
        reasons.append(
            'rules.win "none" beside a royal piece: losing it does not end this game, and '
            "every way Fairy-Stockfish has of holding a royal (the king slot or an "
            "extinction rule) makes losing it a loss"
        )
    if compiled.check_limit and mode != "king":
        cause = (
            "rules.explode_on_capture, which makes the royal pseudo-royal"
            if mode == "pseudo"
            else f'rules.win "{compiled.win}", which leaves the royal free to stand in check'
        )
        reasons.append(
            f"rules.check_limit beside {cause}: Fairy-Stockfish counts checks only "
            "against its king slot, so a royal outside it is never counted as checked "
            "and the limit is never reached there"
        )
    return mode


def _promotion_suffix(compiled: CompiledSpec) -> str:
    """How a promotion is spelled in a move under this translation.

    One promoting piece names its target, because the choice is a list and a move has
    to say which was taken. Several promoting pieces make the relation a map with one
    target each, there is nothing left to choose, and the move is written with a bare
    ``+`` -- which is shogi's notation and the engine's.
    """
    return "plus" if len(compiled.promotions) > 1 else "piece"


def board_letters(compiled: CompiledSpec, chars: Sequence[str]) -> tuple[str, ...]:
    """How each piece is written on a FEN's board: its letter, marked where it demotes.

    The two engines decide differently what a captured piece becomes in a hand. Ours
    decides by type: the piece goes in as its `promoted_from`, and the compiled spec
    holds that as `demotions`. Fairy-Stockfish decides by history: it demotes a piece
    only when the piece was promoted there, or a FEN says it was. So a piece whose type
    demotes is written as promoted, which re-expresses a fact our state already holds
    (Q406). The spelling follows the shape `_promotion_lines` wrote the promotion in:

    * ``+P`` where it wrote the `promotedPieceType` map, which is shogi's shape. The
      engine reads it as whatever the map turns P into, remembered as a P.
    * ``H~`` where it wrote one promoting piece and a list of targets, which is
      dobutsu's shape. The engine reads it as a promoted H and demotes it to its one
      promoting piece.

    A demoting type that neither spelling reaches keeps its bare letter, and
    `_demotion_lines` puts a note on the translation. Nothing is marked unless
    captures go to a hand, because that is the only place the mark is read.
    """
    letters = list(chars)
    if not _captures_to_hand(compiled):
        return tuple(letters)
    for piece, demoted in enumerate(compiled.demotions):
        if demoted != piece:
            letters[piece] = _promoted_letter(compiled, chars, piece, demoted) or chars[piece]
    return tuple(letters)


def _captures_to_hand(compiled: CompiledSpec) -> bool:
    return compiled.drops is not None and compiled.drops.from_captures


def _promoted_letter(
    compiled: CompiledSpec, chars: Sequence[str], piece: int, demoted: int
) -> str | None:
    """``piece`` written so that Fairy-Stockfish demotes it to ``demoted``, if it can be.

    Probed with `pyffish` 0.0.90 (`tests/test_fairy.py`). In the map shape ``+X`` stands
    for the piece the map promotes X into, so it is right only where the map promotes
    ``demoted`` into ``piece``. In the list shape ``~`` marks the piece promoted, and a
    promoted piece demotes to the engine's main promotion pawn type, which is the one
    promoting piece. Where there is no promotion at all there is no such type, and
    ``~`` would demote to nothing.
    """
    rules = compiled.promotions
    if len(rules) > 1:
        for rule in rules:
            becomes = {t for player in (0, 1) for t in rule.targets[player]}
            if rule.piece_type == demoted and becomes == {piece}:
                return f"+{chars[demoted]}"
        return None
    if len(rules) == 1 and rules[0].piece_type == demoted:
        return f"{chars[piece]}~"
    return None


def _demotion_lines(
    compiled: CompiledSpec, chars: list[str], notes: list[str]
) -> list[str]:
    """Make Fairy-Stockfish's search put a captured piece in the hand as ours does.

    `board_letters` covers a piece that is already promoted in a position we hand over.
    What is left is a piece the engine promotes inside its own search. It remembers the
    piece that promoted and demotes the captured piece to it, and we demote to the
    target's `promoted_from`. The two agree where every target's `promoted_from` is the
    piece that promotes into it, which holds in shogi and dobutsu.

    Where no piece demotes at all, as in crazyhouse, ``dropLoop`` tells the engine to
    demote nothing, and the two agree again: a queen promoted and then lost goes into
    the hand as a queen in both. Any other shape cannot be told, so each piece that
    differs gets a note rather than a refusal. The opponent is still handed our pocket
    in every position it is asked about, so every move it plays is legal here, and only
    its search expects the wrong piece in hand.
    """
    if not _captures_to_hand(compiled):
        return []
    rules = compiled.promotions
    demotions = compiled.demotions
    if all(demoted == piece for piece, demoted in enumerate(demotions)):
        return ["dropLoop = true"] if rules else []

    ids = compiled.piece_ids
    letters = board_letters(compiled, chars)
    for piece, demoted in enumerate(demotions):
        if demoted != piece and letters[piece] == chars[piece]:
            notes.append(
                f"a captured {ids[piece]!r} goes into the hand as {ids[demoted]!r} here "
                f"and as {ids[piece]!r} there, which cannot be told it is promoted; the "
                "opponent's search expects the wrong piece in hand"
            )
    # A turn back (Q785, `_flip_pairs`) is `pieceDemotion` there, which leaves the plain
    # face that goes to a hand as ours does, so it needs no note.
    flips = _flip_pairs(compiled)
    for rule in rules:
        for target in sorted({t for player in (0, 1) for t in rule.targets[player]}):
            if flips.get(target) == rule.piece_type:
                continue
            if demotions[target] != rule.piece_type:
                notes.append(
                    f"a {ids[target]!r} promoted from {ids[rule.piece_type]!r} goes into "
                    f"the hand as {ids[demotions[target]]!r} here and as "
                    f"{ids[rule.piece_type]!r} there; the opponent's search expects the "
                    "wrong piece in hand"
                )
    return []


# Fairy-Stockfish's own pieces, by the Betza notation it defines each of them with.
#
# Copied from ``PieceMap::init`` in ``piece.cpp``. A piece of ours whose notation comes
# out equal to one of these is written into *that* slot instead of a custom one, and the
# reason is not tidiness:
#
# * **Castling only works with the built-in rook.** Naming a custom piece in
#   ``castlingRookPieces`` is accepted and then ignored: the FEN parser scans the
#   castling rank for a piece of the built-in rook type, finds nothing, and walks off
#   the board. The symptom is an access violation on the first `position` command with
#   ``KQkq`` in it -- so a translated chess played fine right up until it was asked
#   about its own opening position.
# * **A built-in piece has tuned values and a custom one does not**, so an opponent
#   made of custom pieces is a weaker opponent for no reason.
# * **The pawn and the soldier carry rules no notation can express** -- drop
#   restrictions, blast immunity, the river -- which is why those two have their own
#   detectors above rather than appearing in this table.
#
# ``soldier``, ``janggiElephant`` and ``immobile`` are deliberately absent: the first
# two mean more than their notation says, and the third is the empty string, which
# every piece with no writable move would match.
BUILTIN_PIECES: tuple[tuple[str, str], ...] = (
    ("knight", "N"),
    ("bishop", "B"),
    ("rook", "R"),
    ("queen", "RB"),
    ("fers", "F"),
    ("alfil", "A"),
    ("fersAlfil", "FA"),
    ("silver", "FfW"),
    ("aiwok", "RNF"),
    ("bers", "RF"),
    ("archbishop", "BN"),
    ("chancellor", "RN"),
    ("amazon", "RBN"),
    ("knibis", "mNcB"),
    ("biskni", "mBcN"),
    ("kniroo", "mNcR"),
    ("rookni", "mRcN"),
    ("shogiPawn", "fW"),
    ("lance", "fR"),
    ("shogiKnight", "fN"),
    ("gold", "WfF"),
    ("dragonHorse", "BW"),
    ("clobber", "cW"),
    ("breakthrough", "fmWfF"),
    ("cannon", "mRcpR"),
    ("janggiCannon", "pR"),
    ("horse", "nN"),
    ("elephant", "nA"),
    ("wazir", "W"),
    ("commoner", "WF"),
    ("centaur", "WFN"),
)

# The one notation the king slot understands without being told, so that a royal piece
# that moves like a king is written `king = k` rather than `king = k:WF`.
_KING_NOTATION = "WF"


def piece_slots(
    source: dict[str, Any], compiled: CompiledSpec, king: int | None
) -> dict[int, str]:
    """Our piece index -> the config key Fairy-Stockfish knows that piece by.

    ``king``, ``pawn``, or ``customPiece3``. Derived once in `translate` and passed
    down, because four things read it and they have to agree: the definition line,
    ``mobilityRegion<Colour><Piece>`` (whose key is built from the capitalised form of
    this name), the drop rules asking whether a piece landed in the pawn slot, and
    castling asking whether the rook landed in the rook slot. Four independent
    derivations would be four chances to give one piece a definition and another piece
    its region.

    **The pawn slot is not a shortcut.** Fairy-Stockfish's pawn is a hard-coded piece
    rather than a Betza one, and two rules that matter here are written against that
    type by name and not against the configurable ``pawnTypes``: a pawn may not be
    dropped on its own first rank or in the promotion zone, and a pawn survives an
    atomic blast. A crazyhouse pawn written as a custom piece is therefore droppable
    onto the eighth rank, which is what `verify.py` found. A piece goes in the slot
    only when it moves exactly as that hard-coded pawn does -- see `_is_chess_pawn`.
    """
    slots: dict[int, str] = {}
    taken: set[str] = set()
    slot = 0
    duck = compiled.duck.piece_type if compiled.duck is not None else None
    for index, piece in enumerate(source["pieces"]):
        if index == duck:
            # Generation 42 (N4c): the duck is a wall there (`wallingRule = duck`, written
            # `*`), so it takes no slot and has no definition line.
            continue
        if index == king:
            slots[index] = "king"
            continue
        found = _builtin_slot(piece, compiled, taken)
        if found is not None:
            slots[index] = found
            taken.add(found)
            continue
        slot += 1
        slots[index] = f"customPiece{slot}"
    return slots


def _builtin_slot(
    piece: dict[str, Any], compiled: CompiledSpec, taken: set[str]
) -> str | None:
    """Which of Fairy-Stockfish's own pieces this one is, if it is one of them.

    A slot is used at most once. Two of our pieces that move identically are two
    pieces, and the second one has to be a custom piece so that the two keep separate
    letters, separate hands and separate promotion rules.
    """
    if "pawn" not in taken and _is_chess_pawn(piece, compiled) and _pawn_slot_is_safe(
        piece, compiled
    ):
        return "pawn"
    if "soldier" not in taken and soldier_rank(piece, compiled) is not None:
        return "soldier"
    notation = _plain_notation(piece, compiled)
    if notation is None:
        return None
    for name, spelling in BUILTIN_PIECES:
        if spelling == notation and name not in taken:
            return name
    return None


def _pawn_slot_is_safe(piece: dict[str, Any], compiled: CompiledSpec) -> bool:
    """Whether this piece can be the engine's own pawn without leaving the board.

    Fairy-Stockfish's pawn is the one piece whose move generation is hand-written
    rather than read off a notation, and it assumes what chess guarantees: **a pawn is
    never standing on the rank it would promote from.** Mandatory promotion makes that
    true in chess and it is not true here. A pawn with nowhere to promote to, or one
    the opening array puts on its own promotion rank, gets pushed one square further --
    off the board, into a square the engine will then offer as a legal move.

    Both cases are in the corpus. `dull` is a board of pawns with no promotion rule at
    all, and `sardines` fills every rank including the eighth. Neither needs a double
    step or an en passant window, so the custom-piece spelling costs them nothing, and
    a custom piece is masked to the board properly.
    """
    index = compiled.piece_ids.index(piece["id"])
    promotion = next((r for r in compiled.promotions if r.piece_type == index), None)
    if promotion is None:
        return False
    for player in (0, 1):
        value = compiled.encode_piece(player, index)
        for square, occupant in enumerate(compiled.initial_board):
            if occupant == value and square in promotion.squares[player]:
                return False
    return True


def _plain_notation(piece: dict[str, Any], compiled: CompiledSpec) -> str | None:
    """This piece's Betza, with nothing ruleset-specific attached.

    The en passant flag is deliberately not applied: it says something about the
    *rules* rather than about the movement, and a piece that captures in passing is
    still a pawn-shaped piece for the purpose of matching a slot. It is also never
    reached here, because a piece that captures in passing is a promoting piece and
    those match `pawn` first.
    """
    terms: list[str] = []
    for descriptor in piece["moves"]:
        term = _descriptor_term(
            descriptor, orientation=compiled.orientation, where="", reasons=[]
        )
        if term is None:
            return None
        terms.append(term)
    return "".join(terms)


# What a chess pawn's move descriptors come to once written as Betza: one forward step
# that does not capture, two forward diagonal captures, and optionally a double step
# available from one rank. Compared as a set, because the order descriptors are written
# in is not part of the piece.
_PAWN_TERMS = ({"mfW", "cfF"}, {"mfW", "mifW2", "cfF"})


def soldier_rank(piece: dict[str, Any], compiled: CompiledSpec) -> int | None:
    """The rank at which this piece starts moving sideways, if it is a soldier.

    A piece that steps forward everywhere and sideways only past a line is the one
    shape in the schema that a Betza string cannot hold, because Betza describes a
    piece and this is a piece that is two pieces depending on where it stands. It is
    also the xiangqi soldier, so Fairy-Stockfish has it built in: the ``soldier`` type
    plus ``soldierPromotionRank``, which is the rank from which the sideways step
    appears. Recognising it is worth a special case because refusing it costs the whole
    of xiangqi -- the one ruleset in the corpus that most needs an outside opinion,
    since it is the board this project's own ladder sees least deeply.

    Returns the one-based rank, or ``None`` if the piece is anything else. Deliberately
    strict: a piece that widens by a *different* move, or over a region that is not a
    band of whole ranks, is not this piece and gets no slot.
    """
    moves = piece["moves"]
    if len(moves) != 2:
        return None
    forward = [m for m in moves if not (m.get("condition") or {}).get("region")]
    sideways = [m for m in moves if (m.get("condition") or {}).get("region")]
    if len(forward) != 1 or len(sideways) != 1:
        return None

    terms = {
        _descriptor_term(
            dict(descriptor, condition=None),
            orientation=compiled.orientation,
            where="",
            reasons=[],
        )
        for descriptor in moves
    }
    if terms != {"fW", "sW"}:
        return None

    region = sideways[0]["condition"]["region"]
    ranks = set()
    for name in region:
        coords = parse_square(name, compiled.width, compiled.height)
        if coords is None:
            return None
        ranks.add(coords[1])
    # A band of whole ranks running to the far edge, and nothing else: that is what
    # "past the river" means and what `zone_bb` gives the engine.
    lowest = min(ranks)
    if ranks != set(range(lowest, compiled.height)):
        return None
    if len(region) != len(ranks) * compiled.width:
        return None
    return lowest + 1


def _is_chess_pawn(piece: dict[str, Any], compiled: CompiledSpec) -> bool:
    """Whether this piece is exactly the pawn Fairy-Stockfish has built in.

    Decided on the generated notation rather than on the descriptors, so that two
    spellings of the same pawn -- a `slide` of range 1 and a `leap`, say -- give the
    same answer. Anything else stays a custom piece: berolina's pawn moves diagonally
    and captures straight ahead, and putting it in this slot would replace it with an
    ordinary pawn and lose the entire point of the ruleset.
    """
    collected: set[str] = set()
    for descriptor in piece["moves"]:
        term = _descriptor_term(
            descriptor, orientation=compiled.orientation, where="", reasons=[]
        )
        if term is None:
            return False
        collected.add(term)
    return collected in _PAWN_TERMS


def _piece_lines(
    source: dict[str, Any],
    compiled: CompiledSpec,
    chars: list[str],
    slots: dict[int, str],
    reasons: list[str],
) -> list[str]:
    """A definition line per piece, in whichever slot the piece belongs to."""
    lines: list[str] = []
    orientation = compiled.orientation
    en_passant_piece = _en_passant_piece(compiled)

    custom = sum(1 for name in slots.values() if name.startswith("customPiece"))
    if custom > MAX_CUSTOM:
        reasons.append(
            f"{custom} pieces need a custom slot and there are {MAX_CUSTOM}"
        )
        return lines

    for index, piece in enumerate(source["pieces"]):
        if index not in slots:
            continue  # the duck (generation 42): a wall, with no definition line
        notation = _betza_for(
            piece,
            compiled=compiled,
            orientation=orientation,
            en_passant=index == en_passant_piece,
            reasons=reasons,
        )
        if notation is None:
            continue
        name = slots[index]
        if name == "soldier":
            lines.append(f"soldier = {chars[index].lower()}")
            lines.append(f"soldierPromotionRank = {soldier_rank(piece, compiled)}")
        elif name == "king":
            # A royal that moves like a king takes the slot's own movement; anything
            # else is a custom royal, which is what a notation after the char means
            # in this one slot.
            suffix = "" if notation == _KING_NOTATION else f":{notation}"
            lines.append(f"king = {chars[index].lower()}{suffix}")
        elif name.startswith("customPiece"):
            lines.append(f"{name} = {chars[index].lower()}:{notation}")
        else:
            # A built-in slot carries its own movement. Writing a notation after the
            # char would be a second statement of a piece the engine already has, and
            # the slots do not all read it the same way.
            lines.append(f"{name} = {chars[index].lower()}")
    return lines


def _region_lines(
    source: dict[str, Any], compiled: CompiledSpec, slots: dict[int, str]
) -> list[str]:
    """Where a confined piece is allowed to stand.

    ``piece.region`` is a palace or a river bank -- squares a piece may not leave -- and
    over there it is ``mobilityRegion``, which is a per-piece, per-colour board mask.
    The two say the same thing because both are about where a piece may *stop*: a ray
    out of a palace is not blocked by the wall of it, it simply may not end outside, and
    that is what a mobility mask means as well.

    The mask is written per player from the compiled tables, not mirrored here, because
    the compiler has already applied the ruleset's own orientation to it.
    """
    lines: list[str] = []
    for index, piece in enumerate(source["pieces"]):
        squares = piece.get("region")
        if not squares:
            continue
        name = slots[index]
        capitalised = name[0].upper() + name[1:]
        for player, label in ((0, "White"), (1, "Black")):
            allowed = _region_squares(squares, compiled, player)
            written = " ".join(format_square(sq % compiled.width, sq // compiled.width) for sq in sorted(allowed))
            lines.append(f"mobilityRegion{label}{capitalised} = {written or '-'}")
    return lines


def _region_squares(squares: list[str], compiled: CompiledSpec, player: int) -> set[int]:
    """A named square set as the second player sees it, under this orientation."""
    found: set[int] = set()
    for name in squares:
        coords = parse_square(name, compiled.width, compiled.height)
        if coords is None:
            continue
        file, rank = coords
        if player == 1:
            rank = compiled.height - 1 - rank
            if compiled.orientation == "rotate":
                file = compiled.width - 1 - file
        found.add(rank * compiled.width + file)
    return found


def _en_passant_piece(compiled: CompiledSpec) -> int | None:
    """Which piece captures in passing, where the rule is on.

    The schema defines en passant against *the* promoting piece and the validator
    refuses the rule beside more than one promotion rule, so there is at most one
    answer and it is that piece.
    """
    if not compiled.en_passant or not compiled.promotions:
        return None
    return compiled.promotions[0].piece_type


def _betza_for(
    piece: dict[str, Any],
    *,
    compiled: CompiledSpec,
    orientation: str,
    en_passant: bool,
    reasons: list[str],
) -> str | None:
    """One piece's whole move list as Betza, or ``None`` after recording why not."""
    pid = piece.get("id")
    if soldier_rank(piece, compiled) is not None:
        # The slot carries this piece's whole movement, including the part no notation
        # can hold. Writing one anyway would be a second, wrong statement of it.
        return ""

    terms: list[str] = []
    for position, descriptor in enumerate(piece["moves"]):
        where = f"piece {pid!r} move {position}"
        term = _descriptor_term(descriptor, orientation=orientation, where=where, reasons=reasons)
        if term is None:
            return None
        terms.append(term)
    notation = "".join(terms)
    if en_passant:
        notation += "e"
    return notation


def _descriptor_term(
    descriptor: dict[str, Any],
    *,
    orientation: str,
    where: str,
    reasons: list[str],
) -> str | None:
    """One move descriptor as one or more Betza terms."""
    kind = descriptor["kind"]
    # T1 (`docs/archive/plan-pieces-moves.md` §5.4): refused before any Betza is written. pyffish
    # 0.0.90 loads XBetza's `q` (circular), `z` (crooked) and `o` without an error and plays
    # the plain piece: `qN` is the knight, `zB` the bishop, so a translation would be a
    # second, wrong game. `path_of` reads "straight", the off value, as no path.
    path = path_of(descriptor)
    if path != "straight":
        reasons.append(
            f"{where}: a {path} path turns as it goes, and Fairy-Stockfish reads XBetza's "
            "circular and crooked modifiers as the plain piece"
        )
        return None
    directions: set[tuple[int, int]] = set()
    symmetry = descriptor.get("symmetry", "none")
    for vector in descriptor["vectors"]:
        directions |= expand_symmetry((vector[0], vector[1]), symmetry)

    if orientation == "mirror" and not _mirror_matches_rotation(directions):
        reasons.append(
            f"{where}: the second player's army is derived by reflecting the rank, and "
            "Fairy-Stockfish derives it by rotating the board; this piece moves "
            "differently under the two"
        )
        return None

    condition = descriptor.get("condition") or {}
    if "region" in condition:
        reasons.append(f"{where}: a move condition on a region has no counterpart")
        return None
    if condition.get("first_move"):
        reasons.append(
            f"{where}: a first-move condition is carried by our `unmoved` mask, which a "
            "FEN cannot express for a piece that does not castle"
        )
        return None
    initial = "from_rank" in condition

    rng = descriptor.get("range", DEFAULT_RANGE[kind])
    mode = descriptor.get("mode", "both")

    if kind == "hop":
        if descriptor.get("screen", "any") != "any":
            reasons.append(f"{where}: a hop that chooses whose piece it jumps has no counterpart")
            return None
        screen_range = descriptor.get("screen_range", 0)
        if screen_range not in (0, 1):
            reasons.append(
                f"{where}: a hop may land anywhere past its screen or immediately past it, "
                f"not exactly {screen_range} squares past"
            )
            return None
        term = betza.term(
            directions, distance=0, mode=mode, hop="g" if screen_range == 1 else "p"
        )
    elif kind == "lame":
        # Q1007a: the range is passed, not assumed. A lame rider (`range` other than 1) was
        # exported as the lame *leaper*, a different game, silently, while the key was
        # refused; `betza._atom_text` refuses any lame distance but 1, and the reason says
        # why. Fairy-Stockfish has no lame rider whose leg is ours to rely on (`docs/
        # reference/engine-and-spec-record.md` *Generation 32*).
        if rng != 1:
            reasons.append(
                f"{where}: a lame rider (a lame move with range {rng}) is blocked on every "
                "jump's leg, and Fairy-Stockfish has no rider of that kind we can rely on"
            )
            return None
        # SR1a: Fairy-Stockfish's `n` is blocked on the squares `betza.lame_path` walks, so
        # a lame crosses only where every direction's legs, stated or computed, are those.
        # Its `nZ` is Janggi's elephant, two legs: ours with them stated, and only that.
        if "legs" in descriptor:
            ours, _ = stated_legs(descriptor["vectors"], descriptor["legs"], symmetry)
        else:
            ours = {d: (_lame_path(*d),) for d in directions}
        parted = sorted(
            d for d in directions if set(ours[d]) != set(betza.lame_path(*d))
        )
        if parted:
            wanted = ", ".join(
                f"{list(d)} on {[list(leg) for leg in ours[d]]}" for d in parted[:2]
            )
            theirs = ", ".join(
                f"{[list(leg) for leg in betza.lame_path(*d)]}" for d in parted[:2]
            )
            reasons.append(
                f"{where}: a lame leap blocked {wanted} (its legs), where Fairy-Stockfish's "
                f"lame leaper is blocked on {theirs}; its nZ is Janggi's elephant, which "
                "a (2, 3) lame with legs [[0, 1], [1, 2]] states"
            )
            return None
        term = betza.term(directions, distance=rng, mode=mode, lame=True)
    elif kind == "leap":
        term = betza.term(directions, distance=rng, mode=mode, initial=initial)
    elif kind == "slide":
        # A ride that starts past the first square is a Betza the notation cannot spell:
        # a rider always covers every square up to its limit. Every such descriptor in
        # the corpus is a pawn's double step, which has a single-step sibling that makes
        # the union right -- and `_check_skipped_squares` asks the compiler whether it
        # does, rather than leaving it to where the walk happens to go (Q97).
        term = betza.term(directions, distance=rng, mode=mode, initial=initial)
    elif kind == "sweep":
        reasons.append(f"{where}: a sweep captures what it passes over, and Betza has no such piece")
        return None
    else:
        reasons.append(f"{where}: unknown move kind {kind!r}")
        return None

    unlettered = sorted(
        {tuple(sorted((abs(dx), abs(dy)), reverse=True)) for dx, dy in directions}
        - set(betza.LEAPERS)
    )
    if term is None and unlettered:
        # Q1030: the (3,4) antelope. Fairy-Stockfish's leaper atoms stop at (3,3), and
        # both builds read a bracketed `(3,4)` as a piece with no moves, without a word
        # (probed on the 2021 binary and pyffish 0.0.90), so no spelling exists to try.
        leaps = ", ".join(f"({b},{a})" for a, b in unlettered)
        reasons.append(
            f"{where}: Fairy-Stockfish has no letter for a {leaps} leap "
            "(its leapers stop at (3,3)), so this piece cannot be written there"
        )
    elif term is None:
        reasons.append(
            f"{where}: no Betza notation for a {kind} along "
            f"{sorted(directions)} at range {rng}"
        )
    return term


def _mirror_matches_rotation(directions: set[tuple[int, int]]) -> bool:
    """Whether reflecting this move set and rotating it give the same piece.

    Our second player's vectors are ``(dx, -dy)`` under `mirror` and ``(-dx, -dy)``
    under `rotate`; Fairy-Stockfish only ever negates a whole direction, so it is always
    the second. The two agree exactly when the set is already closed under negating the
    file -- which every symmetry but `none` guarantees, and which a bare asymmetric
    vector does not.
    """
    return all((-dx, dy) in directions for dx, dy in directions)


# ------------------------------------------------------------------- the opening array


def start_fen(compiled: CompiledSpec, chars: list[str], king: int | None) -> str:
    """The opening position, in the FEN Fairy-Stockfish expects.

    The same fields a mid-game position carries, and for the same reason: an opening
    array written without its castling rights is a game whose first move has already
    lost them, and nothing downstream would say so.
    """
    from chess_variant_sandbox.fairy.position import fen

    return fen(
        compiled,
        compiled_initial_state(compiled),
        chars,
        king,
        drops=compiled.drops is not None,
        check_limit=compiled.check_limit,
        castling=compiled.castling is not None,
        gating=_gating(compiled),
    )


def compiled_initial_state(compiled: CompiledSpec) -> Any:
    """The opening position as a `GameState`, without building an `Engine`.

    Translating a ruleset must not need a compiled core: `supports()` is asked while a
    page is being drawn, and building an engine there would put a Rust compile-and-load
    on the path of a menu.
    """
    from chess_variant_sandbox.engine.state import GameState

    # A finite stock is each side's opening hand (schema 10), written into the startFen's
    # pocket; the hand is otherwise empty, as the core's opening state is.
    stock = compiled.drops.stock if compiled.drops is not None else ()
    return GameState(
        board=compiled.initial_board,
        side_to_move=0,
        unmoved=compiled.initial_unmoved,
        colour=bytes(compiled.n_squares) if compiled.territory is not None else b"",
        hands=(tuple(stock), tuple(stock)) if stock else ((), ()),
    )


# ------------------------------------------------------------------------- the rules


def _rule_lines(
    source: dict[str, Any],
    compiled: CompiledSpec,
    chars: list[str],
    slots: dict[int, str],
    royalty: str,
    reasons: list[str],
    notes: list[str],
    approximates: set[str],
) -> list[str]:
    """Everything that is not a piece or a square.

    A result rule that crosses only approximately gets a note and its key in
    ``approximates``, together, so the result check can tell a known difference from a
    new one without reading the note's words.
    """
    lines: list[str] = []
    rules = source.get("rules", {})

    lines.extend(_ending_lines(compiled, chars, royalty, reasons, notes, approximates))
    lines.extend(_promotion_lines(compiled, chars, reasons))
    lines.extend(_pawn_lines(source, compiled, chars))
    lines.extend(_castling_lines(compiled, chars, royalty, slots, reasons))
    lines.extend(_drop_lines(compiled, chars, slots, reasons))
    lines.extend(_demotion_lines(compiled, chars, notes))

    if compiled.duck is not None:
        # Generation 42 (N4c, `docs/archive/plan-npc.md` §4.6): every move ends by setting the wall
        # down on any empty square but its own (`_check_duck` refused every other duck).
        lines.append("wallingRule = duck")
    if compiled.must_capture:
        lines.append("mustCapture = true")
    if compiled.forbid_check:
        lines.append("checking = false")
    if compiled.royal_facing_forbidden:
        lines.append("flyingGeneral = true")
    if compiled.forbid_immobility:
        lines.append("immobilityIllegal = true")
    if compiled.explosion is not None:
        lines.append("blastOnCapture = true")
        immune = "".join(sorted(chars[t] for t in compiled.explosion.immune_types))
        # The blast there spares pawns unconditionally, so a ruleset whose immune list
        # is empty is not the game it describes. `blastImmuneTypes` is the whole answer
        # and it is written out even when empty, because the default is not "none".
        lines.append(f"blastImmuneTypes = {immune if immune else '-'}")
    if compiled.check_limit:
        lines.append("checkCounting = true")

    # The clocks. Ours count plies and theirs count moves, and the halving is exact
    # only for an even count -- an odd `no_progress_draw` is rounded up, which gives
    # the opponent one ply more rope than our engine does and can only ever make it
    # play on in a position our engine has already drawn.
    if compiled.no_progress_draw:
        lines.append(f"nMoveRule = {(compiled.no_progress_draw + 1) // 2}")
        if compiled.no_progress_draw % 2:
            notes.append(
                f"the no-progress clock is {compiled.no_progress_draw} plies here and is "
                "counted in whole moves there, so it is rounded up"
            )
            approximates.add("no_progress_draw")
    else:
        lines.append("nMoveRule = 0")
    if compiled.repetition_draw == NO_REPETITION_DRAW:
        # `"never"` (Q703). `nFoldRule = 0` switches the rule off there, so no position
        # ends by repetition in either engine, and the result check asks theirs at a
        # position repeated past three folds (`results._Asker.repetition`). Their search
        # is another matter: its cycle detection (`has_game_cycle`) does not read the
        # option and still scores a returning line as a draw. That is how their engine
        # plays the game, not which game it plays, so it is a note and not a key in
        # `approximates`: no ending of theirs differs from ours for `admits` to expect.
        lines.append("nFoldRule = 0")
        notes.append(
            "a position may repeat without a draw in both engines, but the opponent's "
            "search still scores a line that returns to a position as a draw"
        )
    elif compiled.repetition_draw:
        lines.append(f"nFoldRule = {compiled.repetition_draw}")

    if compiled.max_plies:
        notes.append(
            f"the {compiled.max_plies}-ply limit is ours alone; the opponent is not told "
            "the game has a length"
        )
        approximates.add("max_plies")
    lines.extend(_progress_lines(compiled, chars, slots, royalty, notes, approximates))
    if rules.get("checkmate", "loss") == "win":
        lines.append("checkmateValue = win")
    return lines


def _progress_lines(
    compiled: CompiledSpec,
    chars: list[str],
    slots: dict[int, str],
    royalty: str,
    notes: list[str],
    approximates: set[str],
) -> list[str]:
    """Which moves reset the no-progress clock, and the two ways that still differ (Q411).

    Ours resets on a change of material, a promotion, and any move or drop of a one-way
    piece (`game.rs`'s `progress`, reading `compiled.one_way_types`). Theirs resets on a
    capture, a promotion, a move of a piece in ``nMoveRuleTypes``, and any drop. The
    default ``nMoveRuleTypes`` is its own pawn slot, which is our one-way set in chess
    and not in shogi, where the pawn, lance and knight are one-way and none of them is
    that slot. So the one-way set is written out wherever it is not the default. The
    released 2021 build does not know the option and ignores it (record, Q411).

    Two differences are left, and there is no option for either, so each is a note and
    a key in ``approximates``. The result check admits each in its own shape only
    (`results.admits`).

    * **A drop of a piece that is not one-way** resets theirs and not ours. Their clock
      then runs longer than ours, so the key is ``no_progress_draw``, as for an odd
      count.
    * **A checkmate on the ply the clock runs out** is a mate in both where the royal is
      in their king slot: theirs lets a player claim the draw only when not mated, and
      since Q412 ours looks for mate before the clocks. A pseudo-royal is never in check
      there, so their claim comes first and the mate is a draw (probed in atomic). The
      key is ``checkmate``, since what differs is when a mate counts. Until Q412 it was
      the other way round: the 18 translations with a king slot carried the note, the
      mate being a draw in ours, and atomic agreed.
    """
    if not compiled.no_progress_draw:
        return []
    lines: list[str] = []
    one_way = set(compiled.one_way_types)
    if one_way != {piece for piece, slot in slots.items() if slot == "pawn"}:
        written = "".join(sorted(chars[piece] for piece in one_way))
        lines.append(f"nMoveRuleTypes = {written or '-'}")

    ids = compiled.piece_ids
    # Q1059a: a gate is not a drop, and resets neither clock (probed on `pyffish` 0.0.90).
    gating = compiled.drops is not None and compiled.drops.gates is not None
    dropped = [] if gating else sorted(
        repr(ids[piece]) for piece in _types_in_hand(compiled) - one_way
    )
    if dropped:
        listed = dropped[-1]
        if len(dropped) > 1:
            listed = f"{', '.join(dropped[:-1])} or {listed}"
        notes.append(
            f"dropping {listed} resets the no-progress clock there and not here, so the "
            "opponent's search may count on a drop to put off a draw our engine declares"
        )
        approximates.add("no_progress_draw")
    if royalty == "pseudo" and not compiled.forbid_check:
        notes.append(
            "a checkmate given on the ply the no-progress clock runs out is a mate here "
            "and a draw there, where the royal is never in check, so the opponent's "
            "search may count on a draw our engine scores as a mate"
        )
        approximates.add("checkmate")
    return lines


def _types_in_hand(compiled: CompiledSpec) -> set[int]:
    """The piece types a hand can hold: each capturable type, as it enters the hand.

    A hand fills from captures, and from a finite stock (Q784), which the startFen's
    pocket seeds and which may hold a royal under the placement phase. A royal is never
    taken into one, because taking it ends the game or is not a move at all.
    """
    drops = compiled.drops
    stocked = (
        {piece for piece, held in enumerate(drops.stock) if held}
        if drops is not None
        else set()
    )
    if not _captures_to_hand(compiled):
        return stocked
    royal = set(compiled.royal_types)
    return stocked | {
        compiled.demotions[piece] for piece in range(compiled.n_types) if piece not in royal
    }


def _ending_lines(
    compiled: CompiledSpec,
    chars: list[str],
    royalty: str,
    reasons: list[str],
    notes: list[str],
    approximates: set[str],
) -> list[str]:
    """Stalemate, running out, and winning by arrival.

    There is one extinction rule over there and up to two here -- the win condition may
    imply one and `rules.extinction` may state another -- so a ruleset that wants both
    is refused by name rather than played with whichever happened to be written last.
    """
    lines = [f"stalemateValue = {compiled.stalemate}"]

    implied: list[str] = []
    if royalty in ("pseudo", "extinct"):
        royal = "".join(sorted(chars[t] for t in compiled.royal_types))
        implied = ["extinctionValue = loss", f"extinctionPieceTypes = {royal}"]
        if royalty == "pseudo":
            implied.append("extinctionPseudoRoyal = true")
    elif compiled.win == "annihilation":
        implied = ["extinctionValue = loss", "extinctionPieceTypes = *"]

    rule = compiled.extinction
    if rule is not None:
        if implied:
            reasons.append(
                "the win condition and rules.extinction both end the game by running "
                "out of a piece, and there is one extinction rule over there"
            )
        # Over there each type in `extinctionPieceTypes` is counted on its own (the built-in
        # `extinction` variant), which is schema 18's `per_type`. Our summed count over
        # two or more types has no option, and writing the list would play the per-type
        # game, so it is refused (Q783 found the old translation wrote it). One type and
        # `*` read the same either way.
        if not rule.any_piece and len(set(rule.piece_types)) > 1 and not rule.per_type:
            reasons.append(
                "rules.extinction counts its listed types together, and "
                "extinctionPieceTypes counts each on its own; say per_type to mean that"
            )
        # Schema 27 (Q936): `extinctionPieceTypes` is one list for both colours, and a side
        # holding none of a listed type has run out of it, so `"own"` there is a game over
        # before the first move (`docs/reference/extinction-any-kind.md` §3.3). No setting
        # counts a type against one colour only.
        if rule.own:
            reasons.append(
                'rules.extinction.per_type "own": extinctionPieceTypes is one list for both '
                "colours, so a type one side never had is run out for it from the start"
            )
        # Their count includes the hand (`count_with_hand`) and ours is the board, so a
        # type gone to a hand is run out here and not there.
        if rule.per_type and compiled.drops is not None:
            reasons.append(
                "rules.extinction.per_type beside rules.drops: a type held in hand counts "
                "over there and not here"
            )
        types = "*" if rule.any_piece else "".join(sorted(chars[t] for t in rule.piece_types))
        implied = [
            f"extinctionValue = {rule.result}",
            f"extinctionPieceTypes = {types}",
            f"extinctionPieceCount = {max(rule.min_count - 1, 0)}",
        ]
    lines.extend(implied)

    reach = compiled.reach
    if reach is not None and reach.leaves:
        # Schema 18 (Q786). Jeson Mor over there is a flag on e5 plus `flagMove`, which
        # gives the other side one more move after the flag is reached and then calls it
        # won: the knight never has to step off, and the win comes a ply early. That is
        # another rule, so it is refused rather than written as the nearest one.
        reasons.append(
            'rules.reach.on "leave": the game is decided by the move off a goal square, '
            "and Fairy-Stockfish has no such rule (its flagMove gives the other side one "
            "more move after a flag is reached, which is not leaving)"
        )
    elif reach is not None:
        if reach.result != "win":
            notes.append(
                f"rules.reach ends the game as a {reach.result} for the arriving player; "
                "over there arriving always wins, so the opponent plays for the wrong "
                "result"
            )
            approximates.add("reach")
        lines.append(f"flagPiece = {chars[reach.piece_type]}")
        for player, label in ((0, "White"), (1, "Black")):
            squares = " ".join(
                format_square(square % compiled.width, square // compiled.width)
                for square in sorted(reach.squares[player])
            )
            lines.append(f"flagRegion{label} = {squares if squares else '-'}")
            # The installed 2021 build reads only the older spelling and ignored the line
            # above in silence, so it searched these games with no goal at all: one move
            # from it, it played on (Q912, Q915). Both are written, as `whiteDropRegion`
            # is beside `dropRegion*` (Q784); `pyffish` reads either.
            lines.append(f"{label.lower()}Flag = {squares if squares else '-'}")
        if reach.safe:
            # No older spelling: the 2021 build has no `flagPieceSafe`, and
            # `unread.reasons` refuses a translation that writes it there (Q915).
            lines.append("flagPieceSafe = true")
    return lines


def _promotion_limit(
    compiled: CompiledSpec, rule: PromotionRule, chars: list[str], reasons: list[str]
) -> list[str]:
    """Schema 18's ``only_lost`` as ``promotionLimit`` (Grand chess's option, Q783).

    Over there a target is offered while the mover has fewer of it than the limit, which
    is ours with the limit set to the opening count. One limit serves both colours, so
    the two armies must start with the same number of every target. A limit of 0 means
    *no limit* over there, and the validator refuses a target a side starts with none of,
    so every count written here is at least 1.
    """
    targets = sorted({t for player in (0, 1) for t in rule.targets[player]})
    uneven = [t for t in targets if rule.opening[0][t] != rule.opening[1][t]]
    if uneven:
        reasons.append(
            "rules.promotion.only_lost with armies that start with different numbers of "
            + ", ".join(repr(compiled.piece_ids[t]) for t in uneven)
            + "; promotionLimit is one count for both colours"
        )
        return []
    pairs = " ".join(f"{chars[t].lower()}:{rule.opening[0][t]}" for t in targets)
    return [f"promotionLimit = {pairs}"]


def _promotion_lines(
    compiled: CompiledSpec, chars: list[str], reasons: list[str]
) -> list[str]:
    """Promotion, in whichever of the two shapes the ruleset needs.

    One promoting piece is the simple shape: a zone and a list of targets. Several is
    shogi's, where each piece has one target and they differ, and the option that says
    so is a *map* -- which is why the two are not one code path with a longer list.
    """
    rules = compiled.promotions
    if not rules:
        # The *regions* have to be cleared too, and forgetting them is a rule nobody
        # wrote: they default to the far rank on each side, mandatory promotion is on
        # by default, and a piece in the pawn slot with nothing to promote into can
        # then not reach its last rank at all. `dull` is a board of pawns and that is
        # exactly what happened to it -- three legal moves out of five, and no perft
        # shallow enough to show it because the position is seven plies in.
        return [
            "promotionPieceTypes = -",
            "promotionPawnTypes = -",
            "promotionRegionWhite = -",
            "promotionRegionBlack = -",
        ]

    movers = "".join(sorted(chars[rule.piece_type] for rule in rules))
    lines = [f"promotionPawnTypes = {movers}"]
    # A zone of whole ranks is written as `*n` rank tokens. A zone that is a shape (schema
    # 14) is refused: `verify` found the installed engine reads a square list in
    # `promotionRegion*` and does not promote there, so writing one plays another game.
    zones: dict[int, list[int]] = {}
    for player, label in ((0, "White"), (1, "Black")):
        zone = frozenset().union(*(rule.squares[player] for rule in rules))
        if any(rule.squares[player] != zone for rule in rules):
            reasons.append("two promoting pieces with different promotion zones")
            return lines
        ranks = sorted({square // compiled.width for square in zone})
        if len(zone) != len(ranks) * compiled.width:
            reasons.append(
                "a promotion zone that is not whole ranks; Fairy-Stockfish does not "
                "promote on a square list in promotionRegion"
            )
            return lines
        zones[player] = ranks
        lines.append(f"promotionRegion{label} = " + " ".join(f"*{rank + 1}" for rank in ranks))

    # `promotionRank` says the same thing in one number, and both are written because
    # the two spellings belong to different engine versions. The released engine knows
    # only this one and ignores the regions in silence -- which does not fail, it just
    # stops promoting, so a translated makruk pushed a pawn onto the promotion rank and
    # left it a pawn. Newer builds read `promotionRank` first and let the regions
    # override it, so saying both is safe rather than merely compatible.
    #
    # It can only be said when the zone is a band running to the far edge on both
    # sides, because that is all one rank number can mean. A zone that is not -- a
    # promotion rank in the middle of the board with ordinary squares beyond it -- is
    # left to the regions alone, and `verify` refuses it where they are not understood.
    if zones[0] and zones[1] and min(zones[0]) == compiled.height - 1 - max(zones[1]):
        banded = zones[0] == list(range(min(zones[0]), compiled.height)) and zones[1] == list(
            range(0, max(zones[1]) + 1)
        )
        # A zone that is a single rank in the middle of the board says the same thing as
        # the band above it *when promotion is mandatory*, because then nothing can ever
        # be standing beyond it. Makruk is that case -- promotion at the sixth rank of
        # eight -- and without this it is the one ruleset the older spelling cannot
        # reach. Where promotion is optional the zone has to be exact, and shogi's
        # happens to be a band anyway.
        if banded or all(rule.mandatory for rule in rules):
            lines.append(f"promotionRank = {min(zones[0]) + 1}")

    if len(rules) == 1:
        rule = rules[0]
        _check_leaving_the_zone(compiled, reasons, listed=True)
        shared = [chars[t] for t in rule.targets[0]]
        if [chars[t] for t in rule.targets[1]] == shared:
            # One key rather than two, and this is compatibility rather than brevity:
            # `promotionPieceTypesWhite` and `...Black` are newer than the released
            # engine, which ignores them in silence and then generates a promotion as
            # an ordinary move -- so a translated makruk pushed a pawn onto the
            # promotion rank and left it a pawn. Only a ruleset whose two sides really
            # do promote into different armies needs the per-colour form, and on a
            # build that does not know it `verify` refuses the ruleset rather than
            # playing it wrong.
            lines.append(f"promotionPieceTypes = {''.join(shared) if shared else '-'}")
        else:
            for player, label in ((0, "White"), (1, "Black")):
                targets = "".join(chars[t] for t in rule.targets[player])
                lines.append(f"promotionPieceTypes{label} = {targets if targets else '-'}")
        lines.append(f"mandatoryPawnPromotion = {'true' if rule.mandatory else 'false'}")
        if rule.only_lost:
            lines.extend(_promotion_limit(compiled, rule, chars, reasons))
        return lines

    # Several promoting pieces: each must have exactly one target, which is what makes
    # the relation a map. Shogi is the ruleset this is for and the one that shows why a
    # shared target list would be wrong -- there a lance becomes a gold and a rook a
    # dragon, and a shared list would let either become either. The map also promotes
    # a piece leaving its zone, which ours does only under `on_leaving`, and that is
    # what refuses shogi as committed.
    _check_leaving_the_zone(compiled, reasons, listed=False)
    if any(rule.only_lost for rule in rules):
        reasons.append(
            "rules.promotion.only_lost beside another promoting piece; promotionLimit "
            "is read for the pawn types' list, not for the map"
        )
        return lines
    flips = _flip_pairs(compiled)
    if flips:
        return lines + _flip_lines(compiled, chars, flips, zones, reasons)
    pairs: list[str] = []
    for rule in rules:
        becomes = {t for player in (0, 1) for t in rule.targets[player]}
        if len(becomes) != 1:
            reasons.append(
                f"piece {compiled.piece_ids[rule.piece_type]!r} promotes to "
                f"{len(becomes)} pieces beside another promoting piece; there each one "
                "would need its own list and the option is a map"
            )
            return lines
        pairs.append(f"{chars[rule.piece_type]}:{chars[becomes.pop()]}")
    lines.append("promotedPieceType = " + " ".join(pairs))
    lines.append("promotionPieceTypes = -")
    mandatory = all(rule.mandatory for rule in rules)
    if mandatory != any(rule.mandatory for rule in rules):
        reasons.append("some promotions are mandatory and some are not; there it is one flag")
    lines.append(f"mandatoryPiecePromotion = {'true' if mandatory else 'false'}")
    lines.append("mandatoryPawnPromotion = false")
    return lines


def _flip_pairs(compiled: CompiledSpec) -> dict[int, int]:
    """Kyoto shogi's turn-over, read off the compiled promotions (Q785): base to face.

    A pair is two promotion rules that undo each other, ``a -> b`` and ``b -> a``, where
    ``b``'s ``promoted_from`` is ``a`` (so ``b`` goes to a hand as ``a``). Fairy-Stockfish
    says that as the map's ``a:b`` beside ``pieceDemotion``, which turns a promoted piece
    back when it moves. Empty unless every promotion rule belongs to such a pair, which
    leaves every earlier translation as it was: a both-ways map with no ``promoted_from``
    (Q533's audit spec) is still written as a map of eight entries.
    """
    rules = compiled.promotions
    if len(rules) < 2:
        return {}
    single: dict[int, int] = {}
    for rule in rules:
        becomes = {t for player in (0, 1) for t in rule.targets[player]}
        if len(becomes) != 1 or rule.piece_type in single:
            return {}
        single[rule.piece_type] = becomes.pop()
    flips = {
        base: face
        for base, face in single.items()
        if single.get(face) == base and face != base and compiled.demotions[face] == base
    }
    if set(flips) | set(flips.values()) != set(single):
        return {}
    return flips


def _flip_lines(
    compiled: CompiledSpec,
    chars: list[str],
    flips: dict[int, int],
    zones: dict[int, list[int]],
    reasons: list[str],
) -> list[str]:
    """The map's half of a turn-over (Q785): ``a:b`` per pair and ``pieceDemotion``.

    ``pieceDemotion`` turns a promoted piece back on any move it makes, so it says ours
    exactly where both directions are mandatory over the whole board, which is Kyoto
    shogi's every move. A pair confined to a zone, or optional, is refused rather than
    widened.
    """
    whole = list(range(compiled.height))
    if any(zones[player] != whole for player in (0, 1)):
        reasons.append(
            "a piece turns back and forth only inside a zone; pieceDemotion turns a "
            "promoted piece back on every move it makes"
        )
        return []
    if not all(rule.mandatory for rule in compiled.promotions):
        reasons.append(
            "a piece may decline to turn over; pieceDemotion turns a promoted piece back "
            "on every move it makes"
        )
        return []
    return [
        "promotedPieceType = "
        + " ".join(f"{chars[base]}:{chars[face]}" for base, face in sorted(flips.items())),
        "promotionPieceTypes = -",
        "pieceDemotion = true",
        "mandatoryPiecePromotion = true",
        "mandatoryPawnPromotion = false",
    ]


def _check_leaving_the_zone(
    compiled: CompiledSpec, reasons: list[str], *, listed: bool
) -> None:
    """Refuse a promoting piece that can move out of its zone where the two rules differ.

    Two shapes are written, and they promote a piece leaving its zone differently.

    **The map** (``promotedPieceType``). Over there a piece in it may promote on a move
    that *starts* in its zone as well as on one that ends there. That is real shogi's
    rule, and `movegen.cpp`'s ``generate_moves`` narrows the promotions to the zone only
    when the origin is outside it. Ours does the same only where the entry says
    ``on_leaving`` (schema 15, Q452). Without it ours promotes on arriving and only
    then (`promotes_on` reads the destination), so a silver, bishop or rook standing
    unpromoted in its zone has a promotion there for every move out of it and none
    here. With ``mandatoryPiecePromotion`` the two differ on the same moves the other
    way round: there every move from the zone must promote, and here a move out of it
    cannot. No option confines theirs to arriving. `piecePromotionOnCapture` and
    `promotionLimit` narrow something else. So a map piece that can leave passes only
    with the flag (Q453).

    **The list** (``promotionPawnTypes``). A piece there promotes on arriving only, so
    it matches ours without the flag and differs with it, and no option makes it
    promote on leaving. So a list piece that can leave passes only *without* the flag.

    Found by a rating run on shogi (Q432), which stopped on ``e3d4+`` with
    `MoveMismatch`. `verify` had passed it, because a random walk of 60 plies from the
    opening seldom leaves a piece unpromoted in its zone (record, Q438).

    The compiled tables are asked, not the descriptors: a piece can leave where one of
    its entries from a square of its zone lands outside the zone. Whether an unpromoted
    piece can ever *stand* there is not asked. In shogi a declined promotion or a drop
    puts it there, and a ruleset where nothing can is refused as well.
    """
    leaving: set[int] = set()
    for rule in compiled.promotions:
        if _on_leaving(rule) != listed:
            continue
        piece = rule.piece_type
        for player in (0, 1):
            zone = rule.squares[player]
            for facing in range(compiled.n_facings):
                frame = frame_index(player, facing, compiled.n_facings)
                for origin in zone:
                    # The squares each entry reaches: a leap's targets, a ray's squares.
                    reached: list[tuple[int, ...]] = [
                        entry[0]
                        for table in (compiled.leaps, compiled.slides, compiled.sweeps)
                        if table
                        for entry in table[frame][piece][origin]
                    ]
                    reached += [entry[0] for entry in compiled.hops[frame][piece][origin]]
                    if any(target not in zone for squares in reached for target in squares):
                        leaving.add(piece)
    if not leaving:
        return
    names = sorted(repr(compiled.piece_ids[piece]) for piece in leaving)
    named = names[-1] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    if listed:
        reasons.append(
            f"rules.promotion.on_leaving: {named} can move out of the promotion zone, and "
            "Fairy-Stockfish promotes a piece in promotionPawnTypes only on arriving; "
            "ours promotes on leaving too, and no option says that for the list"
        )
        return
    reasons.append(
        f"rules.promotion: {named} can move out of the promotion zone, and "
        "Fairy-Stockfish promotes a piece in promotedPieceType on a move that starts in "
        "the zone as well as on one that ends there; ours promotes only on arriving, and "
        "no option confines theirs to that"
    )


def _on_leaving(rule: PromotionRule) -> bool:
    """Whether a compiled promotion entry also promotes on a move out of its zone.

    `PromotionRule.on_leaving`, schema 15 (Q452, Q450's B1). The compiler decides what the
    flag means, and drops it where the piece cannot leave its zone, so a flag that never
    fires refuses nothing here; this only reads its answer.
    """
    return rule.on_leaving


def _pawn_lines(
    source: dict[str, Any], compiled: CompiledSpec, chars: list[str]
) -> list[str]:
    """The double step and the en passant window that follows it.

    Both are one board mask each in Fairy-Stockfish and a `from_rank` condition here,
    and the mask is *shared by every piece* -- there is one "initial move" region per
    colour. Two pieces whose initial moves start from different ranks would therefore
    need two masks, so the union is written and `verify.py` refuses the result. That is
    the deliberate shape: be optimistic, then be checked.
    """
    lines: list[str] = []
    ranks: dict[int, set[int]] = {0: set(), 1: set()}
    for piece in source["pieces"]:
        for descriptor in piece["moves"]:
            condition = descriptor.get("condition") or {}
            if "from_rank" not in condition:
                continue
            rank = int(condition["from_rank"])
            ranks[0].add(rank)
            ranks[1].add(compiled.height - 1 - rank)

    if ranks[0]:
        lines.append("doubleStep = true")
        for player, label in ((0, "White"), (1, "Black")):
            spans = " ".join(f"*{rank + 1}" for rank in sorted(ranks[player]))
            lines.append(f"doubleStepRegion{label} = {spans}")
        # The released (2021) engine ignores the regions and reads two rank numbers,
        # counted from each side's own end, exactly as `promotionRank` stands beside
        # `promotionRegion*` (Q783: Grand chess's pawns double-step from the third rank,
        # and without these the installed binary had them double-step from the second).
        # Both are said only where the ranks are one band, which is all two numbers mean;
        # `ranks[1]` is the reflection, so the band is the same counted from either end.
        band = sorted(ranks[0])
        if band == list(range(band[0], band[-1] + 1)):
            lines.append(f"doubleStepRankMin = {band[0] + 1}")
            lines.append(f"doubleStepRank = {band[-1] + 1}")
    else:
        # `doubleStep` and not only the regions. The built-in pawn slot brings a double
        # step with it -- the option is on by default and its region defaults to the
        # second rank -- so a ruleset whose pawn simply steps once acquires one, and
        # every shatranj-shaped game in the corpus opened with moves nobody wrote. An
        # empty region is not enough to turn it off.
        lines.append("doubleStep = false")
        lines.append("doubleStepRegionWhite = -")
        lines.append("doubleStepRegionBlack = -")

    if not compiled.en_passant:
        lines.append("enPassantRegion = -")
    return lines


def _castling_lines(
    compiled: CompiledSpec,
    chars: list[str],
    royalty: str,
    slots: dict[int, str],
    reasons: list[str],
) -> list[str]:
    """Castling, worked out from where the opening array puts the two pieces.

    Both players are looked at and *neither* is required to have a king. Horde is the
    reason: one side is an army of pawns with no king at all, and a translation that
    demanded one refused the whole ruleset over a rule that only ever applies to the
    other player.
    """
    rule = compiled.castling
    if rule is None:
        return ["castling = false"]
    if royalty == "none" or rule.king_type not in compiled.royal_types:
        reasons.append("castling with a piece that is not this ruleset's royalty")
        return []

    board = compiled.initial_board
    ranks: set[int] = set()
    files: set[int] = set()
    rook_files: set[tuple[int, int]] = set()
    for player in (0, 1):
        kings = _squares_of(compiled, board, player, rule.king_type)
        if not kings:
            continue
        if len(kings) != 1:
            reasons.append("castling needs at most one king per side in the opening array")
            return []
        king_square = kings[0]
        rank = king_square // compiled.width
        king_file = king_square % compiled.width
        # Both sides castle along their own back rank, and `castlingRank` counts from
        # each player's own end, so the second player's rank is folded onto the first's
        # before the two are compared.
        ranks.add(rank if player == 0 else compiled.height - 1 - rank)
        files.add(king_file)
        on_rank = sorted(
            square % compiled.width
            for square in _squares_of(compiled, board, player, rule.rook_type)
            if square // compiled.width == rank
        )
        if len(on_rank) != 2 or not (on_rank[0] < king_file < on_rank[1]):
            reasons.append(
                "castling needs one rook either side of the king on its own rank; there "
                "the two rooks are named by file"
            )
            return []
        rook_files.add((on_rank[0], on_rank[1]))

    if not ranks:
        reasons.append("castling is declared and no player has a king to castle with")
        return []
    if len(ranks) > 1 or len(files) > 1 or len(rook_files) > 1:
        reasons.append(
            "the two sides castle from different squares; there the rank, the king file "
            "and the two rook files are one setting each"
        )
        return []

    rank = ranks.pop()
    king_file = files.pop()
    queenside_rook, kingside_rook = rook_files.pop()

    if rule.king_files is not None:
        queenside, kingside = rule.king_files
    else:
        queenside = king_file - rule.king_steps
        kingside = king_file + rule.king_steps
    if not (0 <= queenside < compiled.width and 0 <= kingside < compiled.width):
        reasons.append("castling walks the king off the board")
        return []

    if slots.get(rule.rook_type) != "rook":
        # A custom piece named in `castlingRookPieces` is accepted and then ignored:
        # the FEN parser scans the castling rank for the *built-in* rook, finds none,
        # and reads off the end of the board. The crash lands on the first `position`
        # command carrying castling rights, which is the opening array -- so this is
        # refused here rather than discovered there.
        reasons.append(
            "castling with a rook Fairy-Stockfish does not recognise as its own rook; "
            "only a piece that slides orthogonally without limit can castle there"
        )
        return []

    lines = [
        "castling = true",
        f"castlingRank = {rank + 1}",
        f"castlingKingFile = {_file_name(king_file)}",
        f"castlingKingPiece = {chars[rule.king_type]}",
        f"castlingRookPieces = {chars[rule.rook_type]}",
        f"castlingRookQueensideFile = {_file_name(queenside_rook)}",
        f"castlingRookKingsideFile = {_file_name(kingside_rook)}",
        f"castlingQueensideFile = {_file_name(queenside)}",
        f"castlingKingsideFile = {_file_name(kingside)}",
    ]
    if rule.rook_files is not None and tuple(rule.rook_files) != (queenside + 1, kingside - 1):
        # Where the rook lands is fixed by the king's destination over there, so a rule
        # that states it separately can only be honoured when the two already agree.
        reasons.append(
            "castling puts the rook somewhere other than beside the king, which is the "
            "only place it can land there"
        )
    return lines


def _drop_lines(
    compiled: CompiledSpec,
    chars: list[str],
    slots: dict[int, str],
    reasons: list[str],
) -> list[str]:
    """Pieces coming back from a hand.

    Fairy-Stockfish has one drop region per colour and it applies to every piece, so a
    ruleset that restricts one piece and not the others cannot be written that way. It
    has a *second* mechanism for exactly that case, though, and it is the one crazyhouse
    needs: a piece named in ``pawnTypes`` may not be dropped on its own first rank or in
    the promotion zone. Where a per-piece region is precisely those two exclusions, this
    says so and writes no region at all; anything else is refused rather than widened.
    """
    rule = compiled.drops
    if rule is None:
        return ["pieceDrops = false"]
    if rule.gates is not None:
        return _gating_lines(compiled, rule, reasons)
    lines = [
        "pieceDrops = true",
        f"capturesToHand = {'true' if rule.from_captures else 'false'}",
    ]
    if rule.placement:
        # Schema 18 (Q784): while a side holds anything in hand it must drop, which is
        # what `mustDrop` says over there (`placement`, `sittuyin`).
        lines.append("mustDrop = true")
    if rule.faces:
        lines.extend(_drop_face_lines(compiled, rule, reasons))
    if rule.no_doubled:
        if len(rule.no_doubled) > 1:
            reasons.append("more than one piece may not be doubled on a file; there it is one")
        else:
            lines.append(f"dropNoDoubled = {chars[next(iter(rule.no_doubled))]}")
    if rule.no_mate:
        _check_drop_mate(compiled, rule.no_mate, slots, reasons)
        lines.append("shogiPawnDropMateIllegal = true")

    restricted = {
        piece
        for player in (0, 1)
        for piece, region in enumerate(rule.regions[player])
        if region is not None
    }
    if not restricted:
        return lines

    pawnish = {
        piece
        for piece in restricted
        if slots[piece] == "pawn" and _is_pawn_drop_region(compiled, rule, piece)
    }
    # The types a hand can ever hold. Under a hand that fills from captures that is every
    # type, as it always was; a stock-only hand (Q784) holds only what it stocks, so a
    # region on any other type, or none, is never read on either side.
    needed = (
        set(range(compiled.n_types))
        if rule.from_captures
        else {piece for piece, held in enumerate(rule.stock) if held}
    )
    rest = (restricted - pawnish) & needed
    if not rest:
        # Nothing to write: the pawn slot already carries both exclusions, and saying
        # them again through `dropRegion` would apply them to every other piece too.
        return lines

    rook_line = _sittuyin_rook_drop(compiled, rule, rest, slots)
    if rook_line:
        rest = rest - {piece for piece in rest if slots[piece] == "rook"}
        lines.append(rook_line)
    shared = {rule.regions[0][piece] for piece in rest}
    if len(shared) != 1 or (needed - pawnish) - restricted:
        reasons.append(
            "the drop regions differ between pieces in a way that is neither a shared "
            "mask, the pawn rule nor Sittuyin's rook rule; there a region applies to "
            "every piece"
        )
        return lines
    for player, label in ((0, "White"), (1, "Black")):
        region = rule.regions[player][next(iter(rest))] or frozenset()
        squares = " ".join(format_square(sq % compiled.width, sq // compiled.width) for sq in sorted(region))
        lines.append(f"dropRegion{label} = {squares if squares else '-'}")
        # The installed 2021 build knows only the older spelling and ignored the line
        # above in silence, dropping anywhere (found by Q784's `placement-chess`, the
        # first translation with a drop region to reach a walk); both are written, as
        # `doubleStepRankMin` is beside `doubleStepRegion*` (Q783).
        lines.append(f"{label.lower()}DropRegion = {squares if squares else '-'}")
    return lines


def _gating(compiled: CompiledSpec) -> bool:
    """Q1059a: whether the hand enters by gating (``drops.gating``), read off the compiled rule."""
    return compiled.drops is not None and compiled.drops.gates is not None


def _gating_lines(compiled: CompiledSpec, rule: Any, reasons: list[str]) -> list[str]:
    """``drops.gating`` (Q1059a) is ``gating`` with ``seirawanGating``, where it says ours.

    Over there the gates are squares of each side's castling rank, named by file in the
    FEN's castling field (``position._castling_field``), and a ``K`` or ``Q`` there gates
    the king's square and that rook's as well. So a gate off a side's first rank cannot be
    said, and under castling a gate set that leaves out the castling king's or a castling
    rook's opening square would be widened by the ``K`` and ``Q``; both are refused. The
    rest of the hand (no capture hand, no drop) is the validator's already.
    """
    width, height = compiled.width, compiled.height
    for player, rank in ((0, 0), (1, height - 1)):
        off = sorted(square for square in rule.gates[player] if square // width != rank)
        if off:
            reasons.append(
                "rules.drops.gating names a square off a side's first rank "
                f"({', '.join(format_square(sq % width, sq // width) for sq in off)}); "
                "Fairy-Stockfish gates only on the castling rank"
            )
            return []
    castling = compiled.castling
    if castling is not None:
        for square, value in enumerate(compiled.initial_board):
            if not value or not compiled.initial_unmoved >> square & 1:
                continue
            player = compiled.owner_of(value)
            if player in (0, 1) and compiled.type_of(value) in (
                castling.king_type, castling.rook_type,
            ) and square not in rule.gates[player]:
                reasons.append(
                    "rules.drops.gating leaves out a castling piece's opening square, which "
                    "Fairy-Stockfish's K and Q gate whatever the field says"
                )
                return []
    return [
        "pieceDrops = false",
        "capturesToHand = false",
        "gating = true",
        "seirawanGating = true",
    ]


def _drop_face_lines(compiled: CompiledSpec, rule: Any, reasons: list[str]) -> list[str]:
    """``drops.either_face`` (schema 18, Q785) is ``dropPromoted``, where it says ours.

    Over there a piece in hand may also be dropped as what the ``promotedPieceType`` map
    promotes it into, from the same drop region. Ours drops it as the piece whose
    ``promoted_from`` it is, also from its slot's region. So the two agree where the map
    and our pairs name the same faces; under the turn-over (`_flip_pairs`) the map holds
    the bases only. ``no_doubled`` and ``no_mate`` read the face that stands here, and
    which face they read there was not checked, so a pair either names is refused.
    """
    pairs = {slot: face for slot, face in enumerate(rule.faces) if face >= 0}
    mapped: dict[int, int] = {}
    if len(compiled.promotions) > 1:
        for promotion in compiled.promotions:
            becomes = {t for player in (0, 1) for t in promotion.targets[player]}
            if len(becomes) == 1:
                mapped[promotion.piece_type] = next(iter(becomes))
    flips = _flip_pairs(compiled)
    if (flips or mapped) != pairs:
        reasons.append(
            "drops.either_face: a piece's other face is not what the promotedPieceType "
            "map promotes it into, and dropPromoted drops a piece as its map promotion"
        )
        return []
    touched = set(pairs) | set(pairs.values())
    if (rule.no_doubled | rule.no_mate) & touched:
        reasons.append(
            "drops.either_face beside drops.no_doubled or drops.no_mate on a paired "
            "piece; which face those read over there is not checked"
        )
        return []
    return ["dropPromoted = true"]


def _sittuyin_rook_drop(
    compiled: CompiledSpec, rule: Any, rest: set[int], slots: dict[int, str]
) -> str:
    """``sittuyinRookDrop = true`` where it says our regions exactly (Q784), else "".

    Over there one region applies to every piece, and Sittuyin's one exception is an
    option: a piece in the rook slot drops only on its side's first rank, inside that
    region. So it is ours exactly when one piece in the rook slot has a region of its
    own, that region is every other droppable piece's shared region cut to the first
    rank on both sides, and nothing else differs.
    """
    rooks = [piece for piece in rest if slots.get(piece) == "rook"]
    others = rest - set(rooks)
    if len(rooks) != 1 or not others:
        return ""
    shared = {(rule.regions[0][piece], rule.regions[1][piece]) for piece in others}
    if len(shared) != 1:
        return ""
    (region,) = shared
    if all(rule.regions[player][rooks[0]] == region[player] for player in (0, 1)):
        return ""  # the rook shares the region, and `dropRegion` says it already
    width = compiled.width
    first_rank = (
        frozenset(range(width)),
        frozenset(range(compiled.n_squares - width, compiled.n_squares)),
    )
    for player in (0, 1):
        if region[player] is None:
            return ""
        if rule.regions[player][rooks[0]] != region[player] & first_rank[player]:
            return ""
    return "sittuyinRookDrop = true"


def _check_drop_mate(
    compiled: CompiledSpec, no_mate: frozenset[int], slots: dict[int, str], reasons: list[str]
) -> None:
    """Refuse a `drops.no_mate` that ``shogiPawnDropMateIllegal`` cannot say.

    The option reads only its shogi-pawn slot, whatever piece ``no_mate`` names, so it
    says our rule exactly when ``no_mate`` names the one piece in that slot and nothing
    else. Anything else is refused.

    Where it does say it, the two engines still list the drop differently. Ours leaves a
    drop that mates out of the legal moves: *uchifuzume* is an illegal move. Theirs
    keeps it and scores the mate it gives as a win for the side that is mated
    (``Position::checkmate_value``). Their search never plays it while it has another
    move, and where it has no other, ours has no move and the game is already over, so
    no game differs. Q438 refused it for the move lists alone, found in 7 of 40 random
    games of 400 plies. Q450 settled that the rule is traditional and the difference is
    one of spelling, so Q453 lifted the refusal and `verify` accepts that one extra move
    of theirs (`Variant.drop_mate`).
    """
    if _drop_mate_piece(compiled, slots) is not None:
        return
    ids = sorted(repr(compiled.piece_ids[piece]) for piece in no_mate)
    reasons.append(
        f"drops.no_mate: {' and '.join(ids)} is not exactly the piece in the shogiPawn "
        "slot, and shogiPawnDropMateIllegal reads that slot only"
    )


def _drop_mate_piece(compiled: CompiledSpec, slots: dict[int, str]) -> int | None:
    """The one piece `drops.no_mate` names, where it is the piece in the shogiPawn slot."""
    rule = compiled.drops
    if rule is None or len(rule.no_mate) != 1:
        return None
    piece = next(iter(rule.no_mate))
    return piece if slots.get(piece) == "shogiPawn" else None


def _drop_mate_char(compiled: CompiledSpec, chars: list[str], slots: dict[int, str]) -> str:
    """`Variant.drop_mate`: the letter of that piece, or "" where there is none."""
    piece = _drop_mate_piece(compiled, slots)
    return "" if piece is None else chars[piece]


def _is_pawn_drop_region(compiled: CompiledSpec, rule: Any, piece: int) -> bool:
    """Whether this piece's drop region is exactly what ``pawnTypes`` already means.

    Every square except the player's own first rank and the zone it would promote in --
    which is what a crazyhouse pawn is restricted to, and what Fairy-Stockfish applies
    to a pawn type by default.
    """
    promotion = next((p for p in compiled.promotions if p.piece_type == piece), None)
    if promotion is None:
        return False
    for player in (0, 1):
        region = rule.regions[player][piece]
        if region is None:
            return False
        first_rank = 0 if player == 0 else compiled.height - 1
        excluded = {
            square
            for square in range(compiled.n_squares)
            if square // compiled.width == first_rank
            or square in promotion.squares[player]
        }
        if set(region) != set(range(compiled.n_squares)) - excluded:
            return False
    return True


# ------------------------------------------------------------------------ small parts


def _squares_of(compiled: CompiledSpec, board: bytes, player: int, piece: int) -> list[int]:
    wanted = compiled.encode_piece(player, piece)
    return [square for square, value in enumerate(board) if value == wanted]



def _file_name(file: int) -> str:
    return format_square(file, 0)[0]


__all__ = [
    "MAX_FILES",
    "MAX_RANKS",
    "Untranslatable",
    "Variant",
    "parse_square",
    "translate",
    "variant_name",
]
