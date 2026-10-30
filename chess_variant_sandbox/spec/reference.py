"""The rule language as one Markdown file a person can write a game from (`RULES.md`).

`rule_spec_schema()` is the source of every key, type, bound, allowed value and most of
the sentences, and `validate_spec` is the source of every refusal shown: the reference
walks the one and calls the other, so a key added to the language appears here the day
it is added, and a refusal reads exactly as the validator words it. What neither holds
lives below, beside the walk that uses it:

- ``NOTES``: a sentence for each key the schema leaves undescribed, because its name was
  enough for the model the schema was first written for (``board.width``).
- ``DEFAULTS``: what an omitted optional key means. The validator and the compiler read
  each with ``.get(key, default)``; this table states those literals once more, in words.
- The opening, the coordinates, the keys that do not change the game, and the two worked
  examples, which ``tests/test_rules_reference.py`` validates out of the committed file.

``chess-variant-sandbox rules-reference --out RULES.md`` writes the file, and the same
test fails when the committed copy differs from what this module renders. The file is
exported to the public repository (`export_public.ALLOW`), so nothing here may name the
site's internals or anything that stays private.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from chess_variant_sandbox.spec.describe import RULE_WORDS
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import (
    _APPEARANCE_KEYS,
    _PROVENANCE_KEYS,
    DEFAULT_RANGE,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    validate_spec,
)

COMMAND = "uv run chess-variant-sandbox rules-reference --out RULES.md"

PUBLISHED = "/rules.md"
"""Where a reader finds the file this module writes: the running server's own copy.

Both servers answer this path with `render_reference()` of the code they run (Q905;
`web/server.py`'s handler, which the site inherits), so the reference a reader opens is
always the one for the validator that will read their file. The public repository's
`RULES.md` is a snapshot from the last export and can lag behind it. The one spelling of
the link (Q830b): the designer sends it to the page with its kit (`GET /api/design`), and
the route is this string too.
"""

PAGE = "/rules"
"""Where a reader finds the same text as a web page (G8 of `docs/plan-growth.md`): both
servers draw it from `render_reference()` too (`web/page/rules_page.py`), with a link to
every heading, so the page and `PUBLISHED` cannot say two different things."""

TITLE = f"# The rule language, schema version {SCHEMA_VERSION}"
"""The reference's first line. It names the version, so any copy says what it is for;
`deploy/deploy.sh`'s prove step reads it off the live `/rules.md` (`deploy/rules.py`)."""

LOAD_CONTROL = "load a rule file"
"""The designer's button that reads a file somebody wrote into the draft (Q830b). Named
here because *Trying a game* tells a reader to press it: the page draws the label it is
sent with the kit, so the step and the button cannot come to say two different things."""

PASTE_CONTROL = "the game as text"
"""The designer's box a file's text can be pasted into instead, named by *Trying a game*
for the same reason as `LOAD_CONTROL`."""

# Paths are dotted from the top of a game file, `[]` for every item of a list and `{}` for
# every value of a map keyed by piece id: `pieces[].moves[].range`.
NOTES: dict[str, str] = {
    "schema_version": (
        "Which spelling of this language the file is written in. Any version from 1 up is "
        "accepted and none limits what the file may use; the converter writes the current "
        "one."
    ),
    "name": "The game's name, shown wherever the game is listed. It must not be blank.",
    "board": "The size and shape of the board.",
    "board.width": "How many files (columns) the board has.",
    "board.height": "How many ranks (rows) the board has.",
    "pieces": (
        "Every kind of piece in the game, each with its own one- or two-character id. "
        "The order does not matter."
    ),
    "pieces[].id": (
        "A short id, one or two characters, unique in the game, that every other part of "
        "the file uses to name this piece: `P`, `N`, `Q2`."
    ),
    "pieces[].name": "The piece's name, as players read it.",
    "pieces[].value": (
        "What the piece is worth, roughly in pawns. Computer players use it as a first "
        "guess at how much a piece matters, and `rules.scoring` with count 'material' "
        "adds it up. A royal piece is usually 0."
    ),
    "pieces[].moves[].vectors": (
        "The steps this move is made of, each `[dx, dy]` (see *Coordinates*). The "
        "symmetry turns each one into its images."
    ),
    "pieces[].moves[].condition": (
        "Restrictions on when this move is available. Every condition stated must hold."
    ),
    "rules.portals[].moves[].vectors": (
        "The steps this move is made of, each `[dx, dy]`, as on a piece's move."
    ),
    "rules.portals[].moves[].condition": (
        "Restrictions on when this move is available: only `first_move` (a portal has no "
        "side, so no rank or region), and not on a fold."
    ),
    "setup": "Where every piece stands when the game begins.",
    "setup.placements": "The starting squares, one entry per piece kind and player.",
    "setup.placements[].piece": "The id of the piece placed.",
    "setup.placements[].player": (
        "Whose pieces these are: 0 for the first player, who moves first, 1 for the "
        "second, or `\"nobody\"` for pieces neither player owns (they block, either player "
        "may take them, and `pieces[].control` says who may move them). With `setup.mirror` "
        "on, write only the first player's (and nobody's, whose squares are mirrored too, "
        "still nobody's) and leave this out for the first player's."
    ),
    "rules": (
        "Everything about the game beyond the pieces and the board: how it is won, and "
        "each extra rule. Every key is optional; an empty object is the defaults below."
    ),
    "rules.stalemate": "The result for the player who has no legal move but is not in check.",
    "rules.max_plies": (
        "How many moves the game may last, counting each player's move separately, "
        "before it stops as a draw (or is decided by `rules.scoring`)."
    ),
    "rules.no_progress_draw": (
        "The game is drawn after this many moves in a row, counting both players', with no "
        "capture and no promotion (chess's fifty-move rule is 100)."
    ),
    "rules.promotion.piece": "The id of the piece that promotes.",
    "rules.promotion.to": "The ids it may promote to; the player chooses among them.",
    "rules.promotion.to_by_player": (
        "Instead of `to`, a separate list of promotion targets for each player."
    ),
    "rules.castling.king": "The id of the piece that castles, usually the royal piece.",
    "rules.castling.rook": "The id of the piece it castles with.",
    "rules.castling.king_steps": "How many squares the castling piece moves.",
    "rules.castling.king_file": (
        "Instead of `king_steps`, the file letter the castling piece lands on, on each side."
    ),
    "rules.castling.king_file.kingside": "The file it lands on when castling toward the higher files.",
    "rules.castling.king_file.queenside": "The file it lands on when castling toward the lower files.",
    "rules.castling.rook_file": "The file letter the partner piece lands on, on each side.",
    "rules.castling.rook_file.kingside": "The file the partner lands on when castling toward the higher files.",
    "rules.castling.rook_file.queenside": "The file the partner lands on when castling toward the lower files.",
    "rules.extinction.min_count": (
        "A player with fewer than this many of the listed pieces gets the result. Each "
        "player must start with at least this many (of each listed type, under per_type; "
        "of each listed type they start with any of, under per_type 'own'), "
        "or the game would be over before the first move."
    ),
    "rules.reach.result": "The result for the player whose piece reaches the squares.",
    "rules.contacts.landing.vectors": "The landing steps, each `[dx, dy]`.",
    "rules.strike.landing.vectors": "The landing steps, each `[dx, dy]`.",
    "rules.portals[].moved_by": (
        "Who may spend a turn moving each end of the pair, with the pair's `moves`. Leave "
        "it out and the portals never move."
    ),
    "rules.portals[].passage": (
        "Which end each player's pieces may go in by. Leave it out and both ends work "
        "both ways for both players."
    ),
}

# What an optional key means when it is left out, in the words the reference prints.
DEFAULTS: dict[str, str] = {
    "schema_version": f"{SCHEMA_VERSION}, the version this reference describes",
    "description": "empty",
    "rules": "`{}`, every rule at its default",
    "board.lattice": "`\"square\"`",
    "board.shape": "`\"grid\"`",
    "pieces[].name": "the piece's id",
    "pieces[].value": "`1`",
    "pieces[].royal": "`false`",
    "pieces[].captures": "the piece may capture any piece",
    "pieces[].moves[].symmetry": "`\"none\"`",
    "pieces[].moves[].range": ", ".join(
        f"`{distance}` for a {kind}" for kind, distance in DEFAULT_RANGE.items()
    ),
    "pieces[].moves[].min_range": "`1`",
    "pieces[].moves[].mode": "`\"both\"`",
    "pieces[].moves[].screen": "`\"any\"`",
    "pieces[].moves[].screen_range": "`0`",
    "pieces[].moves[].sweep_range": "the move's `range`: the sweep clears every square it moves over",
    "pieces[].moves[].path": "`\"straight\"`, every step the same direction",
    "pieces[].moves[].legs": "absent: a lame is blocked on one computed square, a step along "
    "its longer axis or an even diagonal's midpoint",
    "pieces[].moves[].condition.first_move": "`false`",
    "rules.portals[].moves[].symmetry": "`\"none\"`",
    "rules.portals[].moves[].mode": "`\"move\"`, the only mode a portal's move may have",
    "rules.portals[].moves[].condition.first_move": "`false`",
    "pieces[].facing.initial": "`\"forward\"`",
    "pieces[].sight[].symmetry": "`\"none\"`",
    "pieces[].sight[].range": "`0`, to the edge of the board",
    "pieces[].sight[].min_range": "`1`",
    "pieces[].sight[].affects": "every type `whose` names",
    "pieces[].sight[].through": "absent, a ray stopped by the first piece it sees",
    "pieces[].facing.turns": "`true`",
    "setup.mirror": "`true`",
    "setup.orientation": "`\"mirror\"`",
    "setup.placements[].player": "`0`",
    "rules.win": "`\"checkmate\"`",
    "rules.stalemate": "`\"draw\"`",
    "rules.checkmate": "`\"loss\"`",
    "rules.en_passant": "`false`",
    "rules.must_capture": "`false`",
    "rules.forbid_check": "`false`",
    "rules.forbid_immobility": "`false`",
    "rules.max_plies": "`400`",
    "rules.repetition_draw": "`3`",
    "rules.no_progress_draw": "`100`",
    "rules.promote_on_entry": "`false`",
    "rules.promotion.rank": "`\"last\"`",
    "rules.promotion.mandatory": "`true`",
    "rules.promotion.on_leaving": "`false`",
    "rules.promotion.only_lost": "`false`",
    "rules.castling.king_steps": "`2`",
    "rules.obstacles.mirror": "`false`",
    "rules.obstacles.blocks_leaps": "`false`",
    "rules.strike.effect": "`\"capture\"`",
    "rules.strike.position": "`\"advance\"`",
    "rules.explode_on_capture.radius": "`1`",
    "rules.explode_on_capture.blast": "absent: every square within radius of the capture square, "
    "diagonals included",
    "rules.explode_on_capture.blast.symmetry": "`\"none\"`",
    "rules.reach.mirror": "`true`",
    "rules.reach.safe": "`false`",
    "rules.reach.result": "`\"win\"`",
    "rules.reach.on": "`\"arrive\"`",
    "rules.extinction.min_count": "`1`",
    "rules.extinction.result": "`\"loss\"`",
    "rules.extinction.per_type": "`false`",
    "rules.drops.from_captures": "`true`",
    "rules.territory.win_at": "`0`",
    "rules.scoring.count": "`\"material\"`",
    "rules.royal_facing": "`\"allowed\"`",
    "rules.wrap.files": "`\"none\"`",
    "rules.wrap.ranks": "`\"none\"`",
    "rules.wrap.n_s": "`\"none\"`",
    "rules.wrap.ne_sw": "`\"none\"`",
    "rules.wrap.nw_se": "`\"none\"`",
    "rules.strike.landing.symmetry": "`\"none\"`",
    "rules.contacts.position": "`\"advance\"`",
    "rules.contacts.landing.symmetry": "`\"none\"`",
    "rules.contacts.links[].own": "`false`",
    "rules.npcs[].bounty": "the piece's value",
    "rules.npcs[].mirror": "`false`",
    "rules.drops.placement": "`false`",
    "rules.drops.either_face": "`false`",
    "rules.portals[].moved_by.orange": "`[]`, nobody",
    "rules.portals[].moved_by.blue": "`[]`, nobody",
    "rules.portals[].passage.0": "`\"both_ways\"`",
    "rules.portals[].passage.1": "`\"both_ways\"`",
    "rules.portals[].no_straight_back": "`false`",
}

# Where the reference's type is wider than the schema's: the schema handed to a model offers
# only the current version, and the validator accepts every earlier stamp as well.
_TYPES: dict[str, str] = {
    "schema_version": (
        f"integer from {min(SUPPORTED_SCHEMA_VERSIONS)} to {max(SUPPORTED_SCHEMA_VERSIONS)}"
    ),
}

# The sections the reference is cut into, in order, by the path of the object each opens.
# A key that opens a section is listed in its parent as a pointer, not drawn out again.
_SECTIONS = (
    ("", "The top level"),
    ("board", "`board`"),
    ("pieces[]", "`pieces`: one entry per kind of piece"),
    ("pieces[].moves[]", "`pieces[].moves`: one entry per way of moving"),
    ("setup", "`setup`"),
    ("rules", "`rules`"),
)

_INTRO = f"""\
{TITLE}

This is the whole language a game is written in, at schema version {SCHEMA_VERSION}. A game
is one JSON file. It says how big the board is, what pieces there are and how each moves,
where they start, and how the game is won; nothing about a game lives anywhere else. A
person can write one by hand, or give this file to an AI and ask it for one.

**Which version.** A game written from this reference should declare
`"schema_version": {SCHEMA_VERSION}`. The number names the file's spelling, not what it may
use: a new rule does not change it, and a file in an older spelling is read through the
converter (`python -m chess_variant_sandbox.spec.spelling FILE`). The site serves the
reference it runs at `{PUBLISHED}`; when that copy names a later version than this one,
write from that copy.

<!-- Generated from the schema by `{COMMAND}`. Do not edit by hand. -->

## Trying a game

1. Write the file. Start from one of the two worked examples at the end, or from any game
   in `rulesets/`, and change it.
2. Check it. `uv run chess-variant-sandbox validate my-game.json` lists every problem at
   once, in the words shown under *How a wrong file is refused*, or says the game is valid
   and how many moves the first player has.
3. Play it. In the designer, on the website or on your own machine
   (`uv run chess-variant-sandbox serve --open`), press *{LOAD_CONTROL}* and pick the file,
   or paste its text into *{PASTE_CONTROL}*. A file with a mistake is refused in the same
   words as `validate`, and the game you were editing is left as it was; a good one becomes
   your draft, to see on the board, play against the computer and save.
"""

_READING = """\
## Coordinates and directions

- **Squares** are written as in chess: a file letter then a rank number, `a1` in the
  first player's bottom-left corner. Files run `a`, `b`, `c`, ... left to right and ranks
  `1`, `2`, `3`, ... bottom to top, as the first player sees the board.
- **A step** is a vector `[dx, dy]`, written from the moving player's own side: `+dy` is
  forward, toward the other player, and `+dx` is to the mover's right. The second
  player's moves are the same vectors turned to face the other way, so a piece is written
  once for both players.
- **A ray** repeats its step: a slide with `[[0, 1]]` and range 0 goes straight forward
  as far as nothing blocks it.
- **Symmetry** saves listing every direction: `[1, 0]` turned by `rotate4` is the four
  rook directions, and `[1, 2]` by `dihedral8` is the knight's eight jumps.
- **A hex board** (`board.lattice: "hex"`) is a width-by-height rhombus in axial
  coordinates: file `q`, rank `r`, and the six neighbours `[1, 0]`, `[-1, 0]`, `[0, 1]`,
  `[0, -1]`, `[1, -1]`, `[-1, 1]`. Squares are still named `a1`, `b1`, ...
- **Players** are numbered 0 (the first to move, at the bottom) and 1.

## How to read the reference

Each key gives its type and allowed values, whether it is required, what it means when it
is left out, and what it does in play. A key that is not listed is refused: the language
never ignores a key it does not know, so a misspelt rule is an error rather than a rule
that silently does nothing.
"""

# What the language can make, family by family (Q994). A model given this file and nothing
# else wrote five games that were one idea, steering away from the rules it could not check
# (`docs/reference/rules-md-blind-trial.md` §2.4): the reference below is ordered as the
# schema is, and says what each key does but never what kind of game a family makes. Every
# rule key, every piece feature beyond the basic four, the board's two shape keys and the
# five movement kinds are placed in exactly one family; `tests/test_rules_reference.py`
# holds the table to the schema, so a key added to the language fails until it is placed.
FAMILIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "How a game is won",
        "The object need not be checkmate. A royal piece can simply be taken, or there can "
        "be none; a game can be a race to some squares, a fight to run the other side out of "
        "a kind of piece, a count of checks given, ground claimed by moving onto it, or a "
        "score read when time runs out. Changing the object changes what every piece is for.",
        ("rules.win", "rules.checkmate", "rules.stalemate", "pieces[].royal",
         "rules.bare_royal", "rules.check_limit", "rules.reach", "rules.extinction",
         "rules.territory", "rules.scoring"),
    ),
    (
        "How pieces move",
        "Besides chess's slides and leaps: a hop vaults one screening piece, as the xiangqi "
        "cannon captures or a grasshopper lands; a lame leap can be blocked on its way; a "
        "sweep clears every piece along its line, or only along its first few steps before "
        "it slides on. A leap can be ridden, jump after jump in "
        "one direction: as a slide over the jump it is stopped only where it would land "
        "(the nightrider), and as a lame move with a range each jump is also stopped by a "
        "piece on the one square it passes, the square a lame leap is blocked on. A slide's "
        "path may turn as it goes: round in a circle, each step the next direction round "
        "(the rose, on the knight's eight), or in a zigzag between two neighbouring "
        "directions (the crooked bishop). A lame jump may state the squares it is blocked "
        "on (Janggi's elephant, blocked on two; any lame on a hex board). A piece "
        "can be kept to one part of the board, or point one way and spend a move turning.",
        ("slide", "leap", "hop", "lame", "sweep", "pieces[].region", "pieces[].facing"),
    ),
    (
        "What a capture does",
        "A piece may be able to take only certain others, so armies can beat one another in "
        "a circle. A taken piece can become another piece, be sent home, change places with "
        "its taker, blow up with everything round it, or go into the captor's hand to be "
        "dropped back into play. A piece can shoot, taking what it captures without moving "
        "onto its square; a player can start with pieces in hand; captures can be "
        "compulsory; and taking a piece can pay its bounty into points counted when time "
        "runs out. A move can push instead of taking: the piece it lands on goes one step "
        "on, or off the board at the edge (push chess), and each piece may push only the "
        "kinds it names, or those no heavier than itself, or a row no heavier in all than "
        "its strength.",
        ("pieces[].captures", "rules.strike", "rules.contacts", "rules.explode_on_capture",
         "rules.drops", "pieces[].promoted_from", "rules.promote_on_entry",
         "rules.must_capture", "pieces[].bounty", "rules.push", "pieces[].pushes",
         "pieces[].weight", "pieces[].strength"),
    ),
    (
        "The board",
        "Hexagons as well as squares; walls that give the board any outline or cut it in "
        "two; edges that join, so the board is a cylinder or a torus; portals that carry a "
        "piece across the board, which the players may be allowed to move.",
        ("board.lattice", "board.shape", "rules.obstacles", "rules.wrap", "rules.portals"),
    ),
    (
        "Who moves, and when",
        "Several moves a turn; or both players choosing at once, neither seeing the other's "
        "move, with rules for what wins a collision, which pieces are fast enough to move "
        "first, and spies whose loss takes a player's secrecy away; and pieces that must rest "
        "for some of their player's turns after they move, by turns or in rounds.",
        ("rules.turn", "rules.simultaneous", "pieces[].beats", "pieces[].speed",
         "pieces[].spy", "pieces[].rest"),
    ),
    (
        "What pieces do to each other without moving",
        "A piece can have lines of sight, written like its moves, that put the pieces they see "
        "under an effect while they stand there: frozen in place, unable to capture, "
        "impossible to take, or open to the opponent too, who may move them as their own "
        "(and, if the sight says so, make them take their own side). A sight can be stopped "
        "by the first piece in its way or see past them all, and can touch the enemy's "
        "pieces, its own side's, or both, of every kind or only some. Check reads the effects, "
        "so freezing a checker, or moving it away by command, answers its check.",
        ("pieces[].sight",),
    ),
    (
        "Pieces nobody owns",
        "Wild pieces that walk a fixed route, one step after every move, eat what they may, "
        "and pay a bounty to whoever takes them; the duck, which every turn ends by "
        "moving to an empty square, where nothing may take it; and neutral pieces placed for "
        "nobody, which stand and block until taken, or are moved by whichever player has "
        "more pieces next to them.",
        ("rules.npcs", "rules.duck", "pieces[].control"),
    ),
    (
        "Coins and a market",
        "Each player can hold a purse of coins, filled by the bounties on the pieces they "
        "take (or, where the game says so, on the pieces they lose), and spend a whole move buying a piece from a market onto their own back rank "
        "(or any squares the game names), hiring a priced piece nobody owns that stands next "
        "to one of theirs, or bribing an enemy one away at twice its price. Every exchange "
        "is then priced twice, in material and in coins, and every turn asks whether to "
        "raise a piece or move one.",
        ("rules.purse", "pieces[].price"),
    ),
    (
        "Chess's own rules",
        "None is assumed: promotion, en passant and castling, each chess's answer to a "
        "problem chess has; and limits on giving check, on the royal pieces facing each "
        "other down an open file (xiangqi's rule), and on moves that strand a piece.",
        ("rules.promotion", "rules.en_passant", "rules.castling", "rules.forbid_check",
         "rules.royal_facing", "rules.forbid_immobility"),
    ),
    (
        "When a game stops",
        "How long a game may run, and when a repetition or a stall draws it; `rules.scoring` "
        "can decide such an ending instead. A game that is chess underneath may also be "
        "drawn the moment too little material is left for anyone to mate.",
        ("rules.max_plies", "rules.repetition_draw", "rules.no_progress_draw",
         "rules.insufficient_material"),
    ),
)

# The piece keys every game uses, which the map above does not place.
_BASIC_PIECE_KEYS = frozenset({"id", "name", "value", "moves"})


def _render_families() -> str:
    lines = [
        "## What the language can make",
        "",
        "The reference below goes key by key, in the order a file is written. These are the",
        "same keys by what they do to play. Every family is checked the same way: a file that",
        "asks for two rules the language cannot play together, or for a rule that could never",
        "fire, is refused with a line saying why, so trying an unfamiliar rule costs a round of",
        "`validate`.",
        "",
    ]
    for title, words, keys in FAMILIES:
        named = ", ".join(f"`{key}`" if "." in key else f"`\"{key}\"`" for key in keys)
        lines.append(f"- **{title}.** {words} ({named})")
    return "\n".join(lines) + "\n"


_PLAY_WELL = """\
## A valid game is not yet a good one

`validate` says whether a file is a game; it cannot say whether the game is worth playing.
When a game is measured, computer players of several strengths play it against one another
many times, and it is ruled out as not viable when the stronger players mostly draw or most
games never finish, when many games are over within a few moves, when one side wins far
more often than the other, or when thinking longer does not make a player stronger. Some
things the rules allow make a poor game even so, and each is easy to miss without a board:

- **A capture on the first move.** Set out the starting position and look at every first
  move of both players. A hopping piece behind its own pawn, on a line that runs to
  something of the other side's, captures at once; so does a slider on an open line. The
  opening is then a forced exchange that every game begins with.
- **A rule used up early.** The rule a game is built around should still matter in the
  middle game. A wild piece that can be killed at once, a goal reached in a few moves, or a
  hazard every piece can simply stay away from leaves behind the game it was meant to
  change.
- **Rules carried over from chess.** Castling, en passant and the pawn's double step solve
  problems chess has. Keep one when the new game has that problem.
- **A first player who is simply ahead.** Whoever moves first is a move ahead, and a race,
  a goal or an early capture can turn that move into a win.

The `description` is part of the design: say in it what decision the game asks a player to
keep making, and what a game of it tends to look like.
"""

_REFUSALS_INTRO = """\
## How a wrong file is refused

`validate` checks a file in two passes and reports every problem it finds, not just the
first. The first pass is the shape described above: unknown keys, wrong types, values
outside their list or range, missing required keys. The second is whether the parts
agree: every piece id named anywhere is a declared piece, every square is on the board,
every move can be made on this board, a rule that needs another has it (en passant needs
promotion), two rules that cannot be played together are not, and a rule that could never
fire is refused rather than kept. A rule written at the value that means no rule (`"rest":
0`, `"tie": "both"`) is not refused: it is the game without it.

Each line below is what `validate` prints for one mistake made to the first worked
example.
"""

_OTHER_KEYS = f"""\
## Keys that do not change the game

Two more top-level objects are accepted. Neither changes how the game plays, so a file
written from this reference needs neither.

- **`appearance`**: how the pieces are drawn ({", ".join(f"`{key}`" for key in sorted(_APPEARANCE_KEYS))}),
  mapping a piece id to the name of a drawing, or to a pair of names, one per player.
  Without it every piece is drawn from its name, or as its letter.
- **`provenance`**: where the game came from, as free text
  ({", ".join(f"`{key}`" for key in sorted(_PROVENANCE_KEYS))}). `added` is a day written
  `YYYY-MM-DD` and `parents` a list of game identifiers.
"""

_MINICHESS: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "name": "Five by five",
    "description": (
        "Chess on a five by five board: one rook, knight, bishop, queen and king a side "
        "behind five pawns. Pawns step one square and promote on the far rank; no castling, "
        "no en passant."
    ),
    "board": {"width": 5, "height": 5},
    "pieces": [
        {
            "id": "P", "name": "pawn", "value": 1,
            "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
                {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
            ],
        },
        {
            "id": "N", "name": "knight", "value": 3,
            "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}],
        },
        {
            "id": "B", "name": "bishop", "value": 3,
            "moves": [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}],
        },
        {
            "id": "R", "name": "rook", "value": 5,
            "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}],
        },
        {
            "id": "Q", "name": "queen", "value": 9,
            "moves": [{"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
        },
        {
            "id": "K", "name": "king", "value": 0, "royal": True,
            "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
        },
    ],
    "setup": {
        "mirror": True,
        "placements": [
            {"piece": "P", "squares": ["a2", "b2", "c2", "d2", "e2"]},
            {"piece": "R", "squares": ["a1"]},
            {"piece": "N", "squares": ["b1"]},
            {"piece": "B", "squares": ["c1"]},
            {"piece": "Q", "squares": ["d1"]},
            {"piece": "K", "squares": ["e1"]},
        ],
    },
    "rules": {
        "win": "checkmate",
        "stalemate": "draw",
        "promotion": {"piece": "P", "rank": "last", "to": ["Q", "R", "B", "N"]},
        "max_plies": 200,
    },
}

_MINICHESS_NOTES = """\
- The pawn has two ways of moving: a one-square slide forward that may not capture
  (`"mode": "move"`), and a diagonal one-square leap that may only capture, mirrored left
  and right by `lateral`.
- The other pieces leave `mode` out, so they move and capture alike. A slide with no
  `range` runs to the edge of the board; a leap is one jump.
- `setup.mirror` is on, so only the first player's army is written: the second player's
  stands on the mirrored squares (`a4` to `e4`, and `a5` to `e5`).
- `rules` states only what differs from the defaults or is worth saying: checkmate wins,
  stalemate draws, a pawn promotes on the far rank, and the game stops as a draw after 200
  moves.
"""

_PORTALS = copy.deepcopy(_MINICHESS)
_PORTALS.update({
    "name": "Five by five with moving portals",
    "description": (
        "The five by five game with a pair of portals on the middle rank, on b3 and d3. A "
        "piece whose move ends on one comes out one step beyond the other, in the direction "
        "it was going. Either player may spend a turn moving either portal one king's step "
        "to an empty square, but not straight back on the very next move."
    ),
})
_PORTALS["rules"]["portals"] = [
    {
        "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
        "orange": "b3",
        "blue": "d3",
        "moved_by": {"orange": [0, 1], "blue": [0, 1]},
        "no_straight_back": True,
    },
]

_PORTALS_NOTES = """\
- The portal pair is a rule, `rules.portals`, and its ends are not pieces: nothing in
  `pieces` or the setup is a portal.
- The pair's `moves` say how an end moves when a player moves it, written as a piece's
  moves are: here one king's step. A portal only steps onto an empty square, so each move
  leaves `mode` out (it means `"move"` on a portal).
- `moved_by` lets both players (`[0, 1]`) move both ends; leave it out for portals that
  never move, and then leave `moves` out too.
- `no_straight_back` stops two players moving the same portal back and forth for ever.
- With no `passage`, both ends work both ways for both players. A pawn pushed from `b2`
  onto `b3` comes out on `d4`; a rook sliding up the d-file onto `d3` comes out on `b4`.
"""

EXAMPLES: tuple[tuple[str, dict[str, Any], str], ...] = (
    ("A small chess-like game", _MINICHESS, _MINICHESS_NOTES),
    ("Portals that the players move", _PORTALS, _PORTALS_NOTES),
)


def _mistakes() -> list[tuple[str, dict[str, Any]]]:
    """Broken copies of the first example, each with what was done to it."""

    def changed(edit: Any) -> dict[str, Any]:
        spec = copy.deepcopy(_MINICHESS)
        edit(spec)
        return spec

    return [
        ("A key the language does not have, on the knight: `\"stamina\": 2`.",
         changed(lambda s: s["pieces"][1].update(stamina=2))),
        ("A move kind that does not exist: `\"kind\": \"teleport\"`.",
         changed(lambda s: s["pieces"][2]["moves"][0].update(kind="teleport"))),
        ("A square off the board, `f1`, in the setup.",
         changed(lambda s: s["setup"]["placements"][1].update(squares=["f1"]))),
        ("A promotion to a piece that is not declared: `\"to\": [\"Q\", \"A\"]`.",
         changed(lambda s: s["rules"]["promotion"].update(to=["Q", "A"]))),
        ("En passant, with the promotion rule it needs taken away.",
         changed(lambda s: (s["rules"].pop("promotion"), s["rules"].update(en_passant=True)))),
    ]


def render_reference() -> str:
    """The whole of `RULES.md`."""
    schema = rule_spec_schema(strict=True)
    parts = [_INTRO, _render_families(), _PLAY_WELL, _READING]
    starts = {path for path, _ in _SECTIONS}
    for path, title in _SECTIONS:
        node = _node_at(schema, path)
        parts.append(f"## {title}\n")
        own = _text(path, node) if path else ""
        if own:
            parts.append(own + "\n")
        if path == "rules":
            parts.append(_render_rules(node, starts))
        else:
            parts.append("\n".join(_bullets(node, path, starts, depth=0)) + "\n")
    parts.append(_OTHER_KEYS)
    parts.append(_render_refusals())
    parts.append("## Worked examples\n")
    for number, (title, spec, notes) in enumerate(EXAMPLES, start=1):
        parts.append(f"### {number}. {title}\n")
        parts.append("```json\n" + _pretty(spec) + "\n```\n")
        parts.append(notes)
    return "\n".join(part.rstrip("\n") + "\n" for part in parts)


def schema_paths(node: dict[str, Any] | None = None, prefix: str = "") -> list[str]:
    """Every key path in the schema, in the reference's spelling."""
    if node is None:
        node = rule_spec_schema(strict=True)
    out: list[str] = []
    for key, child in _children(node):
        path = _join(prefix, key)
        out.append(path)
        out.extend(schema_paths(child, path) if _is_object(child) else [])
        inner = _inner(child)
        if inner is not None:
            out.extend(schema_paths(inner[1], path + inner[0]))
    return out


def described(path: str, node: dict[str, Any]) -> bool:
    """Whether the reference has words for this key."""
    return bool(_text(path, node))


# --- the walk ---------------------------------------------------------------------------


def _node_at(schema: dict[str, Any], path: str) -> dict[str, Any]:
    node = schema
    for part in path.split(".") if path else []:
        node = node["properties"][part.removesuffix("[]")]
        if part.endswith("[]"):
            node = node["items"]
    return node


def _children(node: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """An object's keys; a promotion's object-or-list reads as the object."""
    obj = _as_object(node)
    return list(obj["properties"].items()) if obj is not None else []


def _as_object(node: dict[str, Any]) -> dict[str, Any] | None:
    if "properties" in node:
        return node
    for branch in node.get("anyOf", []):
        if "properties" in branch:
            return branch
    return None


def _is_object(node: dict[str, Any]) -> bool:
    return _as_object(node) is not None


def _inner(node: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The object inside a list of objects, or inside a map's values."""
    items = node.get("items")
    if isinstance(items, dict) and "properties" in items:
        return "[]", items
    values = node.get("additionalProperties")
    if isinstance(values, dict) and "properties" in values:
        return "{}", values
    return None


def _join(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _text(path: str, node: dict[str, Any]) -> str:
    if node.get("description"):
        return str(node["description"])
    if path in NOTES:
        return NOTES[path]
    head, _, last = path.rpartition(".")
    if head.startswith("rules") and head.count(".") == 0 and last in RULE_WORDS:
        return RULE_WORDS[last][1]
    if last.endswith("_by_player"):
        base = last.removesuffix("_by_player")
        return f"Instead of `{base}`, a separate value for each player, keyed `\"0\"` and `\"1\"`."
    if last in ("0", "1") and head.endswith(("_by_player", "passage")):
        return "For the first player." if last == "0" else "For the second player."
    return ""


def _bullets(node: dict[str, Any], prefix: str, starts: set[str], depth: int) -> list[str]:
    obj = _as_object(node)
    if obj is None:
        return []
    required = set(obj.get("required", []))
    lines: list[str] = []
    pad = "  " * depth
    for key, child in obj["properties"].items():
        path = _join(prefix, key)
        inner = _inner(child)
        opens = path in starts or (inner is not None and path + inner[0] in starts)
        need = key in required and path not in DEFAULTS
        status = "Required" if need else "Optional"
        if not need and path in DEFAULTS:
            status += f"; default {DEFAULTS[path]}"
        text = _text(path, child)
        line = f"{pad}- **`{key}`**: {_TYPES.get(path) or _type(child)}. {status}."
        if text:
            line += f" {text}"
        if opens:
            target = path + (inner[0] if inner is not None and path + inner[0] in starts else "")
            line += f" See *{dict(_SECTIONS)[target].replace('`', '')}* below."
        lines.append(line)
        if opens:
            continue
        if _is_object(child):
            lines.extend(_bullets(child, path, starts, depth + 1))
        elif inner is not None:
            lines.extend(_bullets(inner[1], path + inner[0], starts, depth + 1))
    return lines


def _render_rules(rules: dict[str, Any], starts: set[str]) -> str:
    out = []
    for key, child in rules["properties"].items():
        path = f"rules.{key}"
        title = RULE_WORDS.get(key, (None, ""))[0]
        heading = f"### `{path}`" + (f": {title}" if title else "")
        line = f"*{_type(child)}*."
        if path in DEFAULTS:
            line += f" Default {DEFAULTS[path]}."
        else:
            line += " Leave it out for no such rule."
        text = _text(path, child)
        body = [heading, "", line + (f" {text}" if text else "")]
        inner = _inner(child)
        nested = _bullets(child, path, starts, 0) if _is_object(child) else (
            _bullets(inner[1], path + inner[0], starts, 0) if inner is not None else []
        )
        if nested:
            body += ["", "Its keys:" if _is_object(child) else "Each entry's keys:", ""]
            body += nested
        out.append("\n".join(body) + "\n")
    return "\n".join(out)


def _type(node: dict[str, Any]) -> str:
    if "anyOf" in node:
        return " or ".join(_type(branch) for branch in node["anyOf"])
    if "const" in node:
        return f"`{json.dumps(node['const'])}`"
    if "enum" in node:
        return "one of " + ", ".join(f"`{json.dumps(value)}`" for value in node["enum"])
    kind = node.get("type")
    if isinstance(kind, list):
        return " or ".join(kind)
    if kind == "null":
        return "`null`"
    if kind == "array" and node.get("minItems") == node.get("maxItems") == 2:
        bounds = _type(node.get("items", {})).removeprefix("integer ")
        return f"`[dx, dy]` pairs (each number {bounds})"
    if kind == "array":
        items = node.get("items", {})
        inner = "object" if "properties" in items else _type(items)
        if inner.startswith("one of"):
            word = "list, each " + inner
        elif inner.startswith("`") or "," not in inner:
            word = "list of " + (inner if inner.startswith("`") else _plural(inner))
        else:
            word = f"list of ({inner})"
        return word + _count(node.get("minItems"), node.get("maxItems"), "entries")
    if kind == "object":
        values = node.get("additionalProperties")
        if isinstance(values, dict):
            return f"object from piece id to {_type(values)}"
        return "object"
    if kind in ("integer", "number"):
        word = "integer" if kind == "integer" else "number"
        low, high = node.get("minimum"), node.get("maximum")
        if low is not None and high is not None:
            return f"{word} from {low} to {high}"
        if low is not None:
            return f"{word}, at least {low}"
        return word
    if kind == "string":
        return "string" + _count(node.get("minLength"), node.get("maxLength"), "characters")
    return str(kind)


def _plural(word: str) -> str:
    head, _, rest = word.partition(" ")
    return head + "s" + (" " + rest if rest else "")


def _count(low: int | None, high: int | None, unit: str) -> str:
    if low is not None and high is not None:
        return f", {low} to {high} {unit}" if low != high else f", exactly {low} {unit}"
    if low:
        return f", at least {low}" + (f" {unit}" if low > 1 else "")
    if high is not None:
        return f", at most {high} {unit}"
    return ""


def _render_refusals() -> str:
    out = [_REFUSALS_INTRO]
    for what, spec in _mistakes():
        errors = validate_spec(spec)
        if not errors:
            raise AssertionError(f"the reference's mistake is accepted: {what}")
        out.append(f"- {what}\n\n  ```text\n" + "".join(
            f"  {error}\n" for error in errors
        ) + "  ```\n")
    return "\n".join(out)


def _pretty(value: Any, indent: int = 0) -> str:
    """JSON as a person writes it: short lists and objects on one line."""
    flat = json.dumps(value)
    if not isinstance(value, (dict, list)) or len(flat) + indent <= 88:
        return flat
    pad = " " * (indent + 2)
    if isinstance(value, list):
        rows = [pad + _pretty(item, indent + 2) for item in value]
        return "[\n" + ",\n".join(rows) + "\n" + " " * indent + "]"
    rows = [
        f"{pad}{json.dumps(key)}: {_pretty(item, indent + 2)}" for key, item in value.items()
    ]
    return "{\n" + ",\n".join(rows) + "\n" + " " * indent + "}"
