"""JSON Schema for the rule DSL, generated from the validator's own constants.

The schema handed to the model and the schema enforced by ``spec.schema`` are the same
thing by construction. Writing the model's contract out by hand would work exactly
until someone adds a movement primitive and updates only one of the two -- at which
point the model starts proposing rules that validate in its head and are rejected
here, and every rejection looks like the model being careless.

The failure is quieter than that in the other direction, and schema 4 produced it:
``rules.turn`` landed in the validator and not here, and because ``strict`` writes
``additionalProperties: false`` the model could not have proposed a longer turn even
by accident. A built primitive no proposal can reach is a primitive that does not
exist as far as the design loop is concerned, and nothing failed. Parity with the
validator's own key sets is asserted in ``tests/test_spec.py`` now rather than
promised in this docstring.

Two keys are withheld on purpose and the test names them: ``appearance`` is how a
piece is drawn, and ``provenance`` names a human author that a model shown one has
every reason to carry over into a game they did not write.
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox.spec.schema import (
    BARE_ROYAL_HOLDER,
    BLAST_STRIKE_VALUES,
    BOUNTY_MAX,
    CONTACT_POSITIONS,
    ENDING_RESULTS,
    FACING_AXES,
    FACING_NAMES,
    HOP_DISPLACE,
    INCOME_PER_PIECE_MAX,
    LATTICES,
    MAX_BOARD_DIM,
    MAX_LEGS,
    MAX_MOVE_DESCRIPTORS,
    MAX_MOVES_PER_TURN,
    MAX_PIECE_TYPES,
    MAX_STOCK,
    MAX_VECTOR_COMPONENT,
    MODES,
    MOVE_KINDS,
    ORIENTATIONS,
    PATHS,
    PORTAL_PASSAGES,
    PRICE_MAX,
    PURSE_MAX,
    PUSH_ALONG,
    PUSH_CHAINS,
    PUSH_EDGES,
    PUSH_WALLS,
    PUSH_WHOSE,
    REACH_BOTH,
    REACH_COUNT_MAX,
    REACH_ON,
    REPETITION_NEVER,
    REST_MAX,
    SCHEMA_VERSION,
    SCORING_COUNTS,
    SCREEN_OWNERS,
    SHAPES,
    SIGHT_EFFECTS,
    SIGHT_MAX,
    SIGHT_WHOSE,
    STALEMATE_RESULTS,
    STOCK_UNLIMITED,
    STRENGTH_MAX,
    STRIKE_EFFECTS,
    STRIKE_POSITIONS,
    SWEEP_EFFECTS,
    SYMMETRIES,
    TERRITORY_MODES,
    UPKEEP_MAX,
    UPKEEP_PROMOTIONS,
    WEIGHT_MAX,
    WIN_CONDITIONS,
    WRAP_FILE_EDGES,
    WRAP_PAIRS,
    WRAP_RANK_EDGES,
)


def rule_spec_schema(strict: bool = True) -> dict[str, Any]:
    """The full rule-spec schema, suitable for ``output_config.format``."""
    vector = {
        "type": "array",
        "items": {
            "type": "integer",
            "minimum": -MAX_VECTOR_COMPONENT,
            "maximum": MAX_VECTOR_COMPONENT,
        },
        "minItems": 2,
        "maxItems": 2,
        "description": "[dx, dy] written from the first player's view; +y is forward.",
    }

    # Schema 12 (Q20): where `position: "choose"` may place a survivor. Shared by `strike`
    # and `contacts`, which compile to one rule.
    landing = _object(
        {
            "vectors": {"type": "array", "items": vector, "minItems": 1},
            "symmetry": {
                "type": "string",
                "enum": list(SYMMETRIES),
                "description": (
                    "How the vectors are expanded, as on a move, and refused off its "
                    "lattice as there."
                ),
            },
        },
        required=["vectors"],
        strict=strict,
    )
    landing_description = (
        "Only with position 'choose', and required there: the squares the attacker may "
        "place a surviving piece on, as [dx, dy] steps from where it stood, forward being "
        "the attacker's. [[1, 0], [1, 1]] with rotate4 is a king's step. A piece with no "
        "empty landing square cannot be taken."
    )

    move = _object(
        {
            "kind": {
                "type": "string",
                "enum": list(MOVE_KINDS),
                "description": (
                    "slide: step repeatedly, blocked by pieces. Over a vector longer than "
                    "one square it is a rider, repeating that jump and stopped only by a "
                    "piece on a square it would land on (a slide over [1, 2] by dihedral8 is "
                    "the nightrider). "
                    "leap: jump straight to the target, ignoring anything between. "
                    "hop: jump exactly one screening piece, then continue. "
                    "lame: like leap, but blocked by a piece on the way, as the "
                    "xiangqi horse is: the square a jump passes is one step along its "
                    "longer axis, or the midpoint of an equal diagonal ([1, 2] passes "
                    "[0, 1], [2, 0] passes [1, 0], [2, 2] passes [1, 1]). With a range it "
                    "is a lame rider, jump after jump, each blocked on that square too. A "
                    "one-square vector has no square to be blocked on and is refused: use a "
                    "leap or a range-1 slide. Its legs may be stated instead (legs). "
                    "sweep: jump to any square along the line, removing every piece "
                    "passed over, friendly included; a piece it may not capture blocks, a "
                    "shielded piece (pieces[].sight effect shield) among them, "
                    "and a sweep that could pass over the enemy royal is check. Beside a "
                    "hand that fills from captures (rules.drops) every swept piece but the "
                    "mover's own goes to the mover's hand, as a taken piece does; a swept "
                    "friendly piece goes to none. Each swept piece but the mover's own "
                    "pays its bounty as a capture of it does (a wild piece its route's, to "
                    "the mover). With a "
                    "sweep_range shorter than its range it clears only that far, and rides "
                    "on past it as a slide. Beside rules.strike the strike acts on the "
                    "capture on its landing square alone, or under effects path on every "
                    "swept piece too (a strike placed choose is refused beside a sweep); "
                    "beside rules.contacts the same, each swept piece meeting the mover by "
                    "its link (a contact placed choose is refused beside a sweep). On a "
                    "wrapped board (rules.wrap) a sweep reaching one square two ways round "
                    "is two moves, each sweeping its own way, and one where both sweep the "
                    "same pieces. Beside rules.portals a momentum end stops a sweep, and a "
                    "landing on an empty fold may go through to the fold it leads to, "
                    "sweeping only the pieces before the fold. A commanded sweeper (a "
                    "command sight) sweeps as its owner's would but removes only what the "
                    "command lets it take: without captures_own any piece ends its line; "
                    "with it, its own side's pieces its captures allow, never a royal or a "
                    "shielded piece, and any other piece ends the line."
                ),
            },
            "vectors": {"type": "array", "items": vector, "minItems": 1},
            "symmetry": {
                "type": "string",
                "enum": list(SYMMETRIES),
                "description": (
                    "How the base vectors are expanded. none: as written. "
                    "lateral: left-right mirror. rotate4: four right-angle rotations. "
                    "dihedral8: all eight images, so [1,2] gives the knight. "
                    "On a hex lattice only none, rotate6 (six 60-degree turns, so [1,0] "
                    "gives the six neighbours) and dihedral12 (the six turns and their "
                    "mirrors, so [1,2] gives the hex knight's twelve); the square's three "
                    "turns are refused there, and the hex two on a square board."
                ),
            },
            "range": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_BOARD_DIM,
                "description": (
                    "Maximum distance of a slide or a hop; 0 means unlimited. On a lame "
                    "move, how many jumps it may make (default 1, a single lame leap; 0 "
                    "unlimited, a lame rider). Not for a leap, which is one jump per vector "
                    "and refuses it: to reach further, list more vectors, or ride it."
                ),
            },
            "min_range": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_BOARD_DIM,
                "description": (
                    "Minimum distance of a slide or a hop, or the nearest jump a lame "
                    "rider may stop on (nearer landings still block). Set it equal to range "
                    "for a move that must travel exactly that far, such as a pawn's opening "
                    "double step. Refused on a leap, like range."
                ),
            },
            "mode": {
                "type": "string",
                "enum": list(MODES),
                "description": "Whether this move may occupy an empty square, capture, or both.",
            },
            "legs": {
                "type": "array",
                "items": {"type": "array", "items": vector, "minItems": 1,
                          "maxItems": MAX_LEGS},
                "minItems": 1,
                "description": (
                    "For a lame (or a leap under obstacles.blocks_leaps), the squares that "
                    "block each jump, stated: one list per base vector, each square a step "
                    "from the mover, turned with its vector by the symmetry. A lame is "
                    "blocked by a piece or a wall on any of them; a leap only by a wall. "
                    "Absent, a lame is blocked on the one computed square. Janggi's "
                    "elephant: [[2, 3]] by dihedral8, legs [[[0, 1], [1, 2]]]; a horse "
                    "blocked on the diagonal: [[1, 2]], legs [[[1, 1]]]. Each square is "
                    "shorter than its jump, none twice; the symmetry's turns must agree on "
                    "each direction's squares. Required on a lame on a hex lattice, and on "
                    "a leap longer than a step under blocks_leaps on one; a leap's legs "
                    "need blocks_leaps."
                ),
            },
            "path": {
                "type": "string",
                "enum": list(PATHS),
                "description": (
                    "For a slide (or a lame rider), how its path runs. straight (the "
                    "default): every step the same direction. circle: each step turns to the next of the "
                    "move's directions round, always the same way (either way), until it "
                    "would land on a square it has been on; the knight's eight, [[1, 2]] "
                    "by dihedral8, is the rose, round an octagon of up to seven knight "
                    "steps, and the queen's, [[1, 0], [1, 1]] by rotate4, is a ring of "
                    "one-square steps. zigzag: the first direction alternates with a "
                    "neighbouring one (either), so [[1, 1]] by rotate4 runs NE, SE, NE, "
                    "... along a rank or a file on its colour, Betza's crooked bishop. "
                    "Any start, both senses. Each step's landing square is looked at, as a "
                    "slide's: empty, it may stop and go on; occupied, the path ends there, "
                    "taking it if it may. A square reached two ways is one move. range "
                    "counts steps. A circle needs directions all the way round (no gap of "
                    "half a turn); a zigzag two less than half a turn apart. On a hex "
                    "lattice the directions go round as they lie in the plane: the hex "
                    "knight's twelve, [[1, 2]] by dihedral12, is the hex rose. On a lame "
                    "with a range other than 1 (a lame rider), each jump is also blocked "
                    "by a piece on its own direction's legs, read from the square it jumps "
                    "from (stated legs turn with the direction), and a blocked leg ends that "
                    "path. Refused on a leap, hop or sweep, at range 1 (a single lame leap "
                    "included), beside portals, on the piece en passant is about, and on a "
                    "wild piece, unless it pounces on its prey (which reads its own capture "
                    "moves)."
                ),
            },
            "screen_range": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_BOARD_DIM,
                "description": (
                    "For hops only, how far past its screen the piece may land. 0 allows "
                    "the whole rest of the line, which is the cannon, and 1 means the "
                    "piece must land immediately beyond, which is the grasshopper."
                ),
            },
            "sweep_range": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_BOARD_DIM,
                "description": (
                    "For sweeps only, how far along its line the sweep clears; range is how "
                    "far it moves. Over its first sweep_range steps it removes every piece "
                    "it passes and may land on any of those squares, as a sweep does; past "
                    "them it moves on as a slide up to range, stopped by any piece and "
                    "taking an enemy piece only where it lands (so a sweep of 3 and range 0 "
                    "clears three squares and then slides on without limit). Omitted, it "
                    "equals range: the sweep clears every square it moves over. It may not "
                    "exceed range (0 means unlimited, so only beside an unlimited range)."
                ),
            },
            # SR6a (`docs/plan-schema-next.md` §2.6): the sweep switch.
            "effects": {
                "type": "string",
                "enum": list(SWEEP_EFFECTS),
                "description": (
                    "For sweeps only, which of the sweep's removals rules.strike, "
                    "rules.contacts or rules.explode_on_capture acts on. "
                    "landing (the default): the capture on its landing square alone, as on "
                    "any capture; the swept pieces are removed. path: every swept piece is "
                    "struck too, in path order after the landing: each follows its chain "
                    "link where it stands (or goes home under position home), with no link "
                    "it is removed; the strike's position governs only the landing's piece, "
                    "and strike.by must name the sweeper. Beside contacts each swept piece "
                    "meets the mover by its link where it stands (onto_to; a friendly one "
                    "by an own link; no link, removed), and the mover's mover_to applies "
                    "once: the landing's link's if it names one, else the last swept "
                    "meeting's that does. Beside explode_on_capture, landing makes one "
                    "blast, centred on the landing square, when the sweep removed a piece "
                    "whose capture explodes (swept or taken on landing); path makes one on "
                    "each swept square whose piece's capture explodes, and one on the "
                    "landing if its capture does, all cleared together, the sweeper "
                    "destroyed once. Every sweep of one piece says the "
                    "same; path needs a strike that strikes what the piece sweeps, a "
                    "contact link whose mover is the piece, or a blast its captures set off."
                ),
            },
            # TS4 (`docs/plan-tessabi.md` §4): a hop over a run, and displacement.
            "run": {
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "integer", "minimum": 2, "maximum": MAX_BOARD_DIM},
                ],
                "description": (
                    "For a hop of mode 'move' only. true: the hop jumps a whole run of pieces "
                    "-- one or more occupied squares in a row along its line, starting on the "
                    "square next to the mover -- and lands on the first square past it, which "
                    "must be empty (Halma's or a leapfrog's jump). A number caps the run's "
                    "length (2 or more). Every piece in the run must be one screen allows; a "
                    "wall or the edge at the run's end is no move. Not beside screen_range "
                    "or push. false (the default) is the plain hop over one screen."
                ),
            },
            "displace": {
                "type": "string",
                "enum": list(HOP_DISPLACE),
                "description": (
                    "For a hop of mode 'move' only. back: every piece the hop jumps moves one "
                    "square back along the line, toward where the hopper came from; the "
                    "nearest takes the hopper's old square. Nothing leaves the board and "
                    "nothing is captured, and it moves pieces nobody owns too. A displaced "
                    "piece does not promote; one displaced onto its own goal counts there. On "
                    "a plain one-screen hop it is a leapfrog exchange. Not beside a royal "
                    "piece, territory, wrap, portals, en passant, rounds, sweeps, wild pieces, "
                    "the duck, sights or rest. none (the default) moves only the hopper."
                ),
            },
            # SR3c (`docs/plan-schema-next.md` §1 rank 7): an owner word or a list of ids.
            "screen": {
                "anyOf": [
                    {"type": "string", "enum": list(SCREEN_OWNERS)},
                    {"type": "array", "items": {"type": "string"}, "minItems": 1},
                ],
                "description": (
                    "For hops only, which piece may serve as the screen. 'any' is the "
                    "xiangqi cannon and the default, 'own' is a piece that vaults its "
                    "own army, and 'enemy' is one that can only jump an opponent. A list "
                    "of piece ids names the types that may screen, of either side: "
                    "Janggi's cannon lists every piece but the cannon, so it may not "
                    "jump another cannon."
                ),
            },
            # Q1074b2 (`docs/plan-push.md` §1.2, §1.3).
            "push": {
                "anyOf": [
                    {"type": "boolean"},
                    _object(
                        {
                            "whose": {
                                "type": "string",
                                "enum": list(PUSH_WHOSE),
                                "description": (
                                    "Whose pieces this move pushes: 'enemy' (the default), "
                                    "'own' or 'any'. Nobody's pieces, wild pieces and the "
                                    "duck are never pushed; they block."
                                ),
                            },
                            "captures": {
                                "type": "boolean",
                                "description": (
                                    "false (the default): landing on an enemy piece it may "
                                    "push, the move pushes and never captures. true: both "
                                    "are offered, a capture and a push. Only on a move that "
                                    "can capture and pushes enemies."
                                ),
                            },
                            "along": {
                                "type": "string",
                                "enum": list(PUSH_ALONG),
                                "description": (
                                    "Which way the pushed piece goes: 'move' (the default), "
                                    "the move's own step (a leap's vector over its "
                                    "components' common divisor: a knight pushes a further "
                                    "knight jump); 'rank' or 'file', that component of the "
                                    "step (a knight's 'rank' push is one square sideways). "
                                    "Not 'rank'/'file' on a hex lattice."
                                ),
                            },
                        },
                        required=[],
                        strict=strict,
                    ),
                ],
                "description": (
                    "This move pushes: landing on a piece it may push, the piece goes one "
                    "step further along the push direction, if the square there is empty, "
                    "or off the board at the edge (removed). A slide pushes only at the "
                    "first piece it reaches. Which types it may push is the piece's "
                    "pushes list. true is every default. On a move with mode 'move' it "
                    "is a piece that pushes and never captures. Not on a sweep, a turning "
                    "path, or beside portals, strike, contacts, explode_on_capture, "
                    "simultaneous, en passant's pushing captures, sights, rest, reach, "
                    "territory or a purse."
                ),
            },
            "condition": _object(
                {
                    "from_rank": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Only from this rank, counted from the mover's own back rank.",
                    },
                    "first_move": {
                        "type": "boolean",
                        "description": "Only if this piece has never moved.",
                    },
                    "region": _squares(
                        "Only from these squares. Written for the first player; the "
                        "second player's region is the reflection."
                    ),
                },
                required=[],
                strict=strict,
            ),
        },
        required=["kind", "vectors"],
        strict=strict,
    )
    # SR3c: a portal's moves are a piece's but for a screen named by type, which names
    # pieces and is refused there; its screen is an owner word alone.
    portal_move = {
        **move,
        "properties": {
            **move["properties"],
            "screen": {**move["properties"]["screen"]["anyOf"][0], "description": (
                "For hops only, whose piece may serve as the screen: 'any' (the default), "
                "'own' or 'enemy'."
            )},
        },
    }

    # Schema 30 (Q999a, `docs/archive/plan-passive.md` §2): one line of sight, written with a slide's
    # geometry keys (the same vector, symmetries and bounds as a move's) and its effect.
    sight_line = _object(
        {
            "vectors": {
                "type": "array",
                "items": vector,
                "minItems": 1,
                "description": (
                    "The directions it looks in, as a slide's vectors, from the piece's own "
                    "view (turned for the second player, and with the piece's facing)."
                ),
            },
            "symmetry": {
                "type": "string",
                "enum": list(SYMMETRIES),
                "description": (
                    "How the vectors are expanded, exactly as on a move: [[1, 0], [1, 1]] by "
                    "rotate4 is the eight lines of a queen, [[1, 2]] by dihedral8 with range 1 "
                    "the knight's eight squares."
                ),
            },
            "range": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_BOARD_DIM,
                "description": (
                    "How many steps along each line it sees, as a slide's range: 0 is to the "
                    "edge of the board, 1 the adjacent squares."
                ),
            },
            "min_range": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_BOARD_DIM,
                "description": (
                    "The nearest step it sees, as a slide's min_range. Nearer squares are "
                    "not seen, but a piece on one still stops a ray."
                ),
            },
            "effect": {
                "type": "string",
                "enum": list(SIGHT_EFFECTS),
                "description": (
                    "What happens to a piece in this sight, for as long as it stands there. "
                    "freeze: it may make no move of any kind (no step, capture, turn, castling, "
                    "strike or promotion), and it attacks nothing, so it gives no check. "
                    "disarm: it may move but may not capture (no capture, en passant, strike, "
                    "contact, or sweep over a piece), and it attacks nothing, so it gives no "
                    "check. shield: nothing may capture it or take it in passing, and a "
                    "shielded royal is never in check. command: its opponent may also move it, "
                    "as that player's move and as its owner would move it, onto an empty "
                    "square only (never taking the mover's own pieces); it never castles, "
                    "takes or opens en passant, or promotes, and a royal is never commanded. "
                    "A frozen or shielded piece still stands, blocks and counts."
                ),
            },
            "whose": {
                "type": "string",
                "enum": list(SIGHT_WHOSE),
                "description": (
                    "Whose pieces it touches, counted from this piece's owner: 'enemy', the "
                    "other player's; 'own', its own player's; 'any', both. There is no "
                    "default, because the natural one differs by effect."
                ),
            },
            "affects": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": (
                    "The piece ids it touches; omit it for every type. It may name this "
                    "piece's own type (Madrasi: a queen freezes the enemy queen). Never an "
                    "empty list. Omitted, it touches no wild piece; it may name a wild type "
                    "(or the duck, for a freeze) under whose 'any', never for a command."
                ),
            },
            "through": {
                "type": "boolean",
                "const": True,
                "description": (
                    "true makes it an aura that sees past every piece, up to range; omit it "
                    "for a ray, which sees the first piece on each line and stops there. A "
                    "wall or the board's edge ends both."
                ),
            },
            "captures_own": {
                "type": "boolean",
                "const": True,
                "description": (
                    "Only beside effect 'command': true lets a commanded piece also be made to "
                    "take a piece of its own side that its captures list allows (never a "
                    "royal or a shielded piece, and not while it is disarmed); omit it for a "
                    "command that moves its pieces onto empty squares only. The piece taken "
                    "goes to the commanding player's hand where captures fill one, and beside "
                    "rules.strike or rules.contacts it is simply removed: nothing is struck, "
                    "no link is read and no side changes."
                ),
            },
        },
        required=["vectors", "effect", "whose"],
        strict=strict,
    )

    piece = _object(
        {
            "id": {"type": "string", "minLength": 1, "maxLength": 2},
            "name": {"type": "string"},
            "royal": {
                "type": "boolean",
                "description": "Losing every royal piece loses the game.",
            },
            "value": {"type": "number", "minimum": 0, "maximum": 100},
            # No minItems (Q925, I1): `moves: []` is a piece that never moves, on any piece;
            # the validator refuses the rules such a piece cannot act on, which a JSON Schema
            # bound cannot say.
            "moves": {
                "type": "array",
                "items": move,
                "maxItems": MAX_MOVE_DESCRIPTORS,
                "description": (
                    "The ways this piece may move, each one a separate entry; the piece "
                    "may make any one of them on its turn. An empty list is a piece that "
                    "never moves: it stands where it is put, blocks, and can be taken, and "
                    "gives no check. It may not promote by moving, castle, be the piece a "
                    "goal or a capture rule names, or leave a player nothing to move."
                ),
            },
            "region": _squares(
                "The only squares this piece may ever occupy, such as a palace or one "
                "bank of a river. Written for the first player and reflected."
            ),
            "promoted_from": {
                "type": "string",
                "description": (
                    "The piece this one was promoted from. It only matters where "
                    "captures go to hand, because a promoted piece returns as its "
                    "modest original."
                ),
            },
            "captures": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The piece ids this one is allowed to capture. Omit it and the "
                    "piece may capture anything, which is what every ordinary game "
                    "means. State it and you get a dominance relation. Give rock "
                    "['S'], scissors ['P'] and paper ['R'] and no piece can take the "
                    "one that takes it. An empty list is a piece that never captures."
                ),
            },
            "beats": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Only in a simultaneous game (rules.simultaneous): the piece "
                    "ids this one eliminates when the two collide, both moving onto the same "
                    "square in one round. Omit it and it is the piece's captures, or every "
                    "piece where captures is omitted too. If "
                    "exactly one of the two beats the other, that one stands on the square "
                    "and the other is eliminated; if both do or neither does, it is a tie "
                    "(rules.simultaneous.tie). An empty list loses every collision it does "
                    "not tie."
                ),
            },
            "spy": {
                "type": "boolean",
                "const": True,
                "description": (
                    "This piece is a spy, and the game is played in three phases "
                    "by who still has one on the board. While both players have a spy, they "
                    "take turns as usual. When one player loses their last spy, they first "
                    "play one ordinary reply move if it is their turn; then the game is played "
                    "in rounds (rules.simultaneous, which a spy game must declare) in which the "
                    "player without a spy announces their move first and the other chooses "
                    "knowing it, both resolved together. Once neither has a spy, both choose in "
                    "secret. A spy can be gained on the board, as the player's own spy type: "
                    "by a promotion into it, a strike chain into it (the struck piece's owner "
                    "gains it), a strike.switches_sides naming it (a spy that changes sides is "
                    "lost by its old side and gained by its new one in that move) or a contact "
                    "link into it (gained by whoever owns the piece after the link), or by a "
                    "drop: a spy in a hand (in drops.stock, or one taken under "
                    "drops.from_captures, which its side loses at the capture and which is the "
                    "taker's) counts for nobody until it is dropped, and dropping it gains it, "
                    "a drop in a round colliding as any drop does, or by a buy from the "
                    "market (rules.purse.market), gained when the bought spy stands after its "
                    "move or round (a buy in a round collides as any arrival, its price paid "
                    "whether or not it stands); the phase "
                    "is read afresh after every move and round, both ways: "
                    "a second spy is one more to lose; when a round leaves both players with a "
                    "spy they take turns again, the player who did not gain moving first (the "
                    "first player if both did); a gain in secret rounds starts announced ones, "
                    "the other player announcing. Both players must start with a spy on the "
                    "board, it may not be royal, and something of the other "
                    "player's must be able to remove it. A spy nobody owns (placed for nobody, "
                    "or a nobody's piece a strike or a contact turns into one) counts for "
                    "neither player: taken, it is gone, or under drops.from_captures it is the "
                    "taker's to drop, as the taker's own spy type. Leave it out for an ordinary piece."
                ),
            },
            "speed": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Only in a game played in rounds (rules.simultaneous): this "
                    "piece's speed, a rank; omit it for 0, the slowest. A "
                    "higher speed moves first. When the two pieces moving in a round have the "
                    "same speed the round is resolved as always (collisions, beats, the tie). "
                    "When they differ, the faster move is made first, exactly as chosen, as a "
                    "move of an alternating game, and if it ends the game the round ends "
                    "there. Then the slower move is made if its piece still stands where it "
                    "was and the move is still one that piece could make on the new board (a "
                    "move onto a square the faster piece now holds is a capture of it if the "
                    "piece may capture there); otherwise it is void and the piece stays, or "
                    "lost if the piece was taken or changed. So a slower piece cannot flee a "
                    "faster strike, and unequal speeds never collide or swap. Only the order "
                    "is a rule: the speeds in use, 0 counted for every piece that names none, "
                    "must be 0, 1, 2, ... with none skipped, and some piece of one player "
                    "must differ in speed from some piece of the other. Never on a wild "
                    "piece."
                ),
            },
            "rest": {
                "type": "integer",
                "minimum": 1,
                "maximum": REST_MAX,
                "description": (
                    "How many of its player's turns this piece rests after it "
                    "makes its player's move, 1 to 4; omit it for 0 (it never rests). "
                    "In any game, played by turns or in rounds (a round is a "
                    "turn of both players): a piece that moved, turned in place, was dropped, "
                    "or struck without leaving its square may not make its player's move "
                    "during that player's next rest turns, counted on the owner's own turns "
                    "only, as the type it is after the move (a pawn promoted to a queen rests "
                    "as the queen). A resting piece still stands: it attacks, blocks, gives "
                    "check (which must be answered) and can be captured; captured, moved off "
                    "its square or changed by the other player, it rests no more. When every "
                    "move a player has is a resting piece's, the rest is lifted for that "
                    "turn, so the rest never leaves a player without a move. Beside a "
                    "command sight a commanded move rests the piece as its owner's move "
                    "would, counted on its owner's turns, and while a piece rests neither "
                    "player may move it (the lift reads each player's own moves, commands "
                    "included). In a turn of several moves (rules.turn moves or ramp) a piece "
                    "that moved earlier in the turn may not move again in that turn, and then "
                    "rests its rest turns after it (the lift still applies). Never on a wild "
                    "piece."
                ),
            },
            "sight": {
                "type": "array",
                "items": sight_line,
                "minItems": 1,
                "maxItems": SIGHT_MAX,
                "description": (
                    f"This piece's lines of sight, 1 to {SIGHT_MAX}; omit it for a "
                    "piece that sees nothing. Each line is "
                    "written like a slide (vectors, symmetry, range, min_range) and puts the "
                    "pieces it sees under an effect for as long as they stand there, read "
                    "from the board as it is, with no memory. A ray (no through) sees each "
                    "line up to and including its first piece; an aura (through: true) sees "
                    "past pieces. Attack reads the effects: a frozen or disarmed piece attacks "
                    "nothing and gives no check, a shielded piece is attacked by nothing and a "
                    "shielded royal is never in check, so a check may be answered by freezing "
                    "or disarming the checker or shielding the royal, and a move that frees an "
                    "attacker of its own royal is illegal. A sight works whatever is done to "
                    "its own piece (two pieces may freeze each other), and two sights of one "
                    "effect are one. There is no lift: a player whom a freeze leaves with no "
                    "legal move is checkmated if in check, and otherwise rules.stalemate "
                    "decides. In rounds the effects decide what is offered, never how a "
                    "collision resolves. On a wild piece (rules.npcs) a sight must have whose "
                    "'any' (a wild piece has no side), may not command, and must read the "
                    "same for both players (a symmetry such as rotate4 or dihedral8); it "
                    "freezes, disarms or shields both players' pieces alike. A wild piece "
                    "never takes a shielded piece: its step onto one waits, and it comes back "
                    "only onto a start no shield covers. A piece placed for nobody sees as a "
                    "wild piece does, under the same three limits. A sight whose affects names "
                    "a wild type, under whose 'any', binds those wild pieces too, read when "
                    "they start to move: a frozen one neither walks nor pounces, a disarmed "
                    "one walks only onto empty squares and never pounces, a shielded one "
                    "cannot be taken, and a frozen or disarmed one threatens nothing; the duck "
                    "may be named only by a freeze, and a frozen duck passes. Beside "
                    "rules.explode_on_capture a "
                    "shielded piece survives a blast as a piece of an immune type does (a "
                    "shielded royal next to a capture is not lost), the shield read on the "
                    "board the capture leaves, before anything goes; a frozen or disarmed "
                    "piece makes no capture, so sets off no blast. Beside rules.portals a "
                    "momentum end stops a line, ray or aura, as a wall does (it sees nothing "
                    "at or past the end); a fold is floor to it, and a line reaching an empty "
                    "fold that lets the seeing piece in also sees the square of the end that "
                    "fold leads to, and the piece there, but nothing past it; an attack "
                    "through either kind reads the effects as any attack does. A shielded "
                    "piece ends a sweep's line, as any piece the sweep may not take does, "
                    "its own side's included. A sight that can never bind (no "
                    "type it affects is ever held by the side it affects, a freeze on types "
                    "with no move, a disarm on types that capture nothing, a shield on types "
                    "nothing can capture, a command on types with no move) is refused. A "
                    "command never touches a royal (an affects naming one is refused) and is "
                    "refused beside rules.simultaneous and a sweep move. Beside "
                    "pieces[].rest a commanded move rests the piece on its owner's turns, "
                    "and a resting piece is moved by neither player. "
                    "Beside rules.portals a commanded piece goes through a portal as its "
                    "owner's piece would (the owner's passage and carries, its exit under the "
                    "owner's wrap), landing at the exit only where a commanded piece may land. "
                    "A commanded piece arrives for its owner: on its owner's "
                    "rules.reach goal it wins for its owner (leaving one, under 'leave'), and "
                    "on rules.territory it claims for its owner. Beside rules.strike or rules.contacts "
                    "a commanded capture of its own side (captures_own) is an ordinary "
                    "removal: nothing is struck, no link is read, no side changes."
                ),
            },
            "control": {
                "anyOf": [
                    {"type": "string", "enum": ["stands", "majority", "shared"]},
                    _object(
                        {
                            "majority": _object(
                                {
                                    "vectors": {
                                        "type": "array", "items": vector, "minItems": 1,
                                        "description": (
                                            "The offsets, from the piece, of the squares "
                                            "that count as next to it, written as a leap's."
                                        ),
                                    },
                                    "symmetry": {
                                        "type": "string", "enum": list(SYMMETRIES),
                                        "description": (
                                            "How the offsets are expanded, as a move's; it "
                                            "must cover both players' views."
                                        ),
                                    },
                                },
                                required=["vectors"],
                                strict=strict,
                                description=(
                                    "Moved by the majority, counting the squares these "
                                    "offsets reach instead of the king's step."
                                ),
                            ),
                        },
                        required=["majority"],
                        strict=strict,
                    ),
                ],
                "description": (
                    "How pieces of this type move while they belong to nobody (a placement "
                    'with "player": "nobody"). "stands", the default: nobody moves them; they '
                    'block and either player may take them. "majority": the player with '
                    "strictly more of their own pieces next to one (any of their pieces, "
                    "royals included) may move it as their move, in nobody's frame, onto an "
                    "empty square or taking a non-royal piece of the other player; a tie, "
                    "none against none included, means nobody moves it. Moving it does not "
                    "make it yours. Next to means the king's step (six steps on a hex "
                    'lattice), or the geometry {"majority": {"vectors": [...], "symmetry": '
                    '...}} names. A held piece attacks nothing and gives no check, unless the '
                    "type says takes_royal. A majority "
                    "type must read the same for both players (no facing, region, condition, "
                    "promotion, owner-named screen, sweep, or a move whose symmetry does not "
                    "cover both), and some placement must put it in nobody's block, or a "
                    "strike or contact must make a piece nobody owns into it. Never on "
                    "a royal or a wild piece. Beside rules.territory a held move claims the "
                    "square it lands on for the player who moved it; beside rules.reach a "
                    "held arrival is read by the goal's own rule (piece, on, safe) as if the "
                    "mover owned the piece, so onto the mover's goal it wins for the mover "
                    "and onto the other's, or standing there afterwards, decides nothing. "
                    "Beside rules.explode_on_capture a held piece's blast is the "
                    "mover's: one that would destroy the mover's royal is no move, and one "
                    "that destroys the other player's wins for the mover. In a turn of "
                    "several moves the holder is read afresh at the start of each move; in "
                    "rounds (rules.simultaneous) it is read on the board the round begins on, "
                    "and the held move is the holder's move of the round. "
                    '"shared": either player may move it as their move, whoever stands next '
                    "to it, in nobody's frame by its type's moves, onto an empty square or "
                    "taking a non-royal piece of the other player its captures allow (never "
                    "the mover's own). Moving it does not make it yours. A shared type reads "
                    "the same for both players as a majority type must, and is refused beside "
                    "takes_royal, rules.simultaneous, a price, and on a portal end; it arrives "
                    "on a goal and claims territory as a held piece does."
                ),
            },
            # Q1077 (`docs/plan-turncoat.md` §4.7).
            "takes_royal": {
                "type": "boolean",
                "description": (
                    'Only on a type with control "majority" placed for nobody, in a game with '
                    "a royal. true: a held piece of this type may also take the royal of the "
                    "player who does not hold it, so it attacks that royal: it gives check, it "
                    "can mate, and a king may not stand where a held piece could take it. "
                    "Since who holds it is counted from the board, a check by it can also be "
                    "answered by tying the count beside it, and a move that hands the majority "
                    "to the opponent may expose one's own king. false, the default: a held "
                    "piece attacks nothing. Refused beside a line of sight (pieces[].sight)."
                ),
            },
            "bounty": {
                "type": "integer",
                "minimum": 1,
                "maximum": BOUNTY_MAX,
                "description": (
                    f"What taking a piece of this type pays, 1 to {BOUNTY_MAX}; omit it for "
                    "nothing. Paid into the points tally of the player whose move took it, "
                    "whoever owned it (a piece nobody owns too), never for a player's own "
                    "piece; a wild piece's taking pays no type bounty, and a blast pays the "
                    "mover for every piece it destroys as taking it would. Needs "
                    'rules.scoring.count "points", which compares the tallies at the clock, '
                    "or rules.purse, whose coins it also pays (capped at "
                    f"{PURSE_MAX}). Never on a royal, a wild type (its rules.npcs entry "
                    "carries its bounty) or the duck."
                ),
            },
            # Q1074b2 (`docs/plan-push.md` §1.4, §2).
            "pushes": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The piece types this piece's pushing moves may push, read as captures "
                    "is: omitted, every type (never a wild type or the duck). A circle "
                    "(A pushes B, B pushes C, C pushes A) is three lists. Only on a piece "
                    "with a pushing move, and never empty."
                ),
            },
            "weight": {
                "type": "integer",
                "minimum": 1,
                "maximum": WEIGHT_MAX,
                "description": (
                    f"1 to {WEIGHT_MAX}: shorthand for a pushes list, every type whose "
                    "weight is at most this one's (types without a weight are never pushed "
                    "by it). Not beside a pushes list on the same piece, and only where "
                    "some move pushes. Under rules.push.chain 'weight' it is instead the "
                    "piece's load when pushed (and its strength when it pushes, unless it "
                    "states a strength): a line moves only if the pusher's strength is at "
                    "least the line's summed weight."
                ),
            },
            # Q1074b4s (§2's correction): strength apart from weight.
            "strength": {
                "type": "integer",
                "minimum": 1,
                "maximum": STRENGTH_MAX,
                "description": (
                    f"1 to {STRENGTH_MAX}, only under rules.push.chain 'weight' and only on a "
                    "piece that pushes: how much it can push. A line moves only if the "
                    "pusher's strength is at least the sum of the weights in it. Omitted, a "
                    "piece's strength is its weight; a light, strong piece states a small "
                    "weight and a large strength. A pusher nothing may push needs only a "
                    "strength; one that may be pushed needs a weight too."
                ),
            },
            "price": {
                "type": "integer",
                "minimum": 1,
                "maximum": PRICE_MAX,
                "description": (
                    f"What one piece of this type costs, 1 to {PRICE_MAX}; omit it and the "
                    "type is not for sale. Read by the market (rules.purse.market), by a "
                    "hire (a piece of the type placed for nobody may be hired at its price "
                    "by a player with a piece next to it) and by a bribe "
                    "(rules.purse.bribe: an enemy piece of the type next to one of yours, "
                    "at twice the price). Only where one of those reads it; never on a "
                    "royal or a wild type. A price is only what buying costs: a priced piece "
                    "taken pays its bounty, never its price, and one gained (a switch, a "
                    "drop) moves no coins. On a spy only a market can charge it, since a spy "
                    "game plays in rounds, where hiring and bribing are refused."
                ),
            },
            "upkeep": {
                "type": "integer",
                "minimum": 1,
                "maximum": UPKEEP_MAX,
                "description": (
                    f"What a move of this type costs per row, 1 to {UPKEEP_MAX}; omit it and "
                    "the type moves for nothing. Under rules.purse.upkeep a move pays this "
                    "times the rows from its owner's first row to where the piece ends (0 on "
                    "the first row), from the purse, before the move; a move the purse cannot "
                    "pay for may not be made. Needs rules.purse.upkeep; never on a wild type."
                ),
            },
            "facing": _object(
                {
                    "axes": {
                        "type": "integer",
                        "enum": list(FACING_AXES),
                        "description": (
                            "How many directions this piece can point: 2 for forward "
                            "and backward, 4 for the quarter turns."
                        ),
                    },
                    "initial": {
                        "type": "string",
                        "enum": list(FACING_NAMES),
                        "description": (
                            "Which way it points at the start, relative to its own "
                            "forward. It must be one the piece can turn back to, so a "
                            "piece with 2 axes has only forward and backward."
                        ),
                    },
                    "turns": {
                        "type": "boolean",
                        "description": (
                            "Whether a move may turn it. False is a facing "
                            "no move turns: each piece points the way its placement says "
                            "for the whole game, and one that enters play later points "
                            "its initial way. A pawn placed facing backward stays a pawn "
                            "going backward. Some placement must then point it a second "
                            "way, or the facing would change nothing."
                        ),
                    },
                },
                required=["axes"],
                strict=strict,
                description=(
                    "This piece has an orientation, and (unless turns is false) may spend "
                    "a whole move turning to any other one instead of moving. Its move vectors are read "
                    "rotated, so a pawn facing right advances east. Give it only to a "
                    "piece whose moves are *not* symmetric under the turn -- a rook "
                    "reads the same rays whichever way it faces, so turning one would "
                    "be a move that does nothing, and the validator refuses that. A "
                    "royal piece may not have one."
                ),
            ),
        },
        required=["id", "moves"],
        strict=strict,
    )

    placement = _object(
        {
            "piece": {"type": "string"},
            "player": {
                "anyOf": [
                    {"type": "integer", "enum": [0, 1]},
                    {"type": "string", "enum": ["nobody"]},
                ],
                "description": (
                    'Whose pieces these are: 0, 1, or "nobody" for pieces neither player '
                    "owns (under a mirrored setup their squares are mirrored too, still "
                    "nobody's). A piece nobody owns blocks and may be taken by either player; "
                    "its type's control says whether anyone may move it. Never a royal or a "
                    "wild type. A spy placed for nobody gives nobody secrecy (pieces.spy). In rounds (rules.simultaneous) one stands, and two moves "
                    "onto its square collide as any two do, by pieces.beats and the tie, the "
                    "winner taking it too (two shots at it meet the same way); a held one "
                    "(control 'majority') is moved by the player who holds it when the round "
                    "begins, as that player's move of the round, colliding as any move does and "
                    "claiming or reaching for that player (tie 'impossible' is refused beside "
                    "it: one held piece can meet another of its type). Beside rules.npcs "
                    "a wild piece eats one its type may capture, paying nobody; the duck never "
                    "lands on one. Beside rules.portals a held piece goes through a portal as "
                    "its holder's piece would, and a fold may move under one. Beside "
                    "rules.strike or rules.contacts one struck or linked becomes what the "
                    "chain or link says and stays nobody's (so the type it becomes may not be "
                    "one nobody may own), and a held piece's capture strikes, or meets its "
                    "target, as its holder's piece of the type would, staying nobody's and "
                    "never changing sides."
                ),
            },
            "facing": {
                "type": "string",
                "enum": list(FACING_NAMES),
                "description": (
                    "Which way these pieces start pointing, for a piece that has a "
                    "facing. Relative to their own player's forward, so a mirrored "
                    "setup gives both armies the same orientations. Omit it and they "
                    "start in the piece's own initial facing."
                ),
            },
            "squares": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Algebraic squares such as a1 or e4.",
            },
        },
        required=["piece", "squares"],
        strict=strict,
    )

    promotion_rule = _object(
        {
            "piece": {"type": "string"},
            "rank": {
                "type": ["string", "integer"],
                "description": "'last', or a rank index counted from the first player's side.",
            },
            "ranks": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "minItems": 1,
                "description": "A promotion zone several ranks deep, instead of 'rank'.",
            },
            "squares": _squares(
                "A promotion zone as named squares, for a zone that is a shape, instead of "
                "'rank' or 'ranks'. The only spelling on a hex lattice."
            ),
            "mandatory": {
                "type": "boolean",
                "description": "False lets a piece decline to promote, as in shogi.",
            },
            # Schema 15 (Q452): what `compile.PromotionRule.on_leaving` reads.
            "on_leaving": {
                "type": "boolean",
                "description": (
                    "True also promotes the piece on a move that starts in the zone, "
                    "wherever it lands, as in shogi: a silver standing in the zone may "
                    "promote as it steps back out. False, the default, promotes only on "
                    "a move that ends in the zone."
                ),
            },
            # Schema 18 (Q783): what `compile.PromotionRule.only_lost` reads.
            "only_lost": {
                "type": "boolean",
                "description": (
                    "True offers a target only while the mover's side has fewer of that "
                    "type on the board than it started with, as in Grand chess: a pawn "
                    "promotes only into a piece its side has lost. Every target must be in "
                    "both armies' setup. It reads the board: a piece in a hand is lost, and "
                    "one a circe strike puts back on its home square is not. Not beside "
                    "promote_on_entry."
                ),
            },
            "to": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "to_by_player": _object(
                {
                    "0": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "1": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                required=["0", "1"],
                strict=strict,
            ),
        },
        required=["piece"],
        strict=strict,
    )

    files = _object(
        {"kingside": {"type": "string"}, "queenside": {"type": "string"}},
        required=["kingside", "queenside"],
        strict=strict,
    )

    rules = _object(
        {
            "win": {
                "type": "string",
                "enum": list(WIN_CONDITIONS),
                "description": (
                    "checkmate: trap the royal piece. royal_capture: take it, with no "
                    "obligation to answer a threat. annihilation: clear the board of an "
                    "army. none: the game is decided only by the endings stated below."
                ),
            },
            "stalemate": {"type": "string", "enum": list(STALEMATE_RESULTS)},
            "checkmate": {
                "type": "string",
                "enum": ["loss", "win"],
                "description": (
                    "The result for the player who is checkmated. 'win' inverts the "
                    "object of the game, as losers chess does."
                ),
            },
            "promotion": {
                "anyOf": [
                    promotion_rule,
                    {"type": "array", "items": promotion_rule, "minItems": 1},
                ],
                "description": "One rule, or one per promoting piece as shogi needs.",
            },
            "en_passant": {
                "type": "boolean",
                "description": (
                    "Requires promotion; applies to the promoting piece. In a turn of several "
                    "moves the window is the opponent's: it is opened by the "
                    "mover's last double step of the turn and open for the opponent's first "
                    "move only."
                ),
            },
            "castling": _object(
                {
                    "king": {"type": "string"},
                    "rook": {"type": "string"},
                    "king_steps": {"type": "integer", "minimum": 1, "maximum": 4},
                    "king_file": files,
                    "rook_file": files,
                },
                required=["king", "rook"],
                strict=strict,
            ),
            "must_capture": {
                "type": "boolean",
                "description": "A capture on offer is the only kind of move there is.",
            },
            "forbid_check": {
                "type": "boolean",
                "description": "Giving check is not allowed at all.",
            },
            "forbid_immobility": {
                "type": "boolean",
                "description": (
                    "A move may not leave its own piece somewhere it could never move "
                    "from again."
                ),
            },
            # H2 (`docs/plan-hunger.md` §3): what `compile.IncomeRule` reads.
            "income": {
                "anyOf": [
                    {"type": "boolean"},
                    _object(
                        {
                            "per_piece": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": INCOME_PER_PIECE_MAX,
                                "description": (
                                    f"Coins each piece in the region earns a turn, 1 (the "
                                    f"default) to {INCOME_PER_PIECE_MAX}."
                                ),
                            },
                            "region": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "The squares where a piece earns, written for the first "
                                    "player and reflected for the second. Leave it out for "
                                    "each player's first rank. Algebraic squares such as a1 "
                                    "or e4."
                                ),
                            },
                        },
                        required=[],
                        strict=strict,
                    ),
                ],
                "description": (
                    "H2: at the start of each of a player's turns, the first player's first "
                    "turn included, the bank pays them per_piece coins (1 by default) for each "
                    "of their pieces, the royal too, standing in the region (by default their "
                    "first rank), into the purse, up to its cap. Counted on the board as the "
                    "turn starts, so a piece taken in the region earns nothing that turn. true "
                    "is a coin per piece on the first rank; false the game without it. Needs "
                    "rules.purse; refused beside scoring.count \"points\" and rounds."
                ),
            },
            # Q1069a: what `compile.InsufficientMaterialRule` reads.
            "pass": {
                "type": "boolean",
                "description": (
                    "H3: the side to move may pass, making no move at all: nothing on the "
                    "board changes and the turn goes to the opponent. Offered wherever making "
                    "no move is legal, so never while in check where the game has check. It "
                    "costs nothing, counts toward the no-progress clock and can repeat a "
                    "position, and with it no position is stalemate (rules.stalemate must be "
                    "left at its default). false is the game without it. Refused beside "
                    "rounds, the duck, a turn of more than one move or of parts, "
                    "must_capture, a placement phase, wild pieces, portals and a push."
                ),
            },
            "insufficient_material": {
                "type": "boolean",
                "description": (
                    "Draw by insufficient material (FIDE's dead positions): the game is "
                    "drawn as soon as nothing is left but the royals, a royal and one lone "
                    "knight or bishop against a royal, or bishops all on squares of one "
                    "colour. It states a rule only where the game is chess once its other "
                    "pieces are gone, and elsewhere does nothing: an 8x8 board, one royal moving as a king, won by checkmate, "
                    "stalemate a draw, and only chess's other rules (promotion, en passant, "
                    "castling, the clocks). Pieces count by how they move: a knight is a "
                    "type that jumps (1, 2) every way, a bishop one that rides the four "
                    "diagonals; any other type on the board keeps the game alive."
                ),
            },
            "royal_facing": {
                "type": "string",
                "enum": ["allowed", "forbidden"],
                "description": "Whether the two royal pieces may see each other down an open file.",
            },
            "bare_royal": {
                "type": "string",
                "enum": list(ENDING_RESULTS),
                "description": (
                    f"The result for {BARE_ROYAL_HOLDER}, so 'loss' is shatranj's rule."
                ),
            },
            "check_limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "Delivering this many checks wins.",
            },
            "territory": _object(
                {
                    "mode": {
                        "type": "string",
                        "enum": list(TERRITORY_MODES),
                        "description": (
                            "Moving onto a square claims it. 'permanent' means a claim "
                            "can never be taken back, which makes every claim a one-way "
                            "step; 'repaint' lets the other player claim it later."
                        ),
                    },
                    "win_at": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Claiming this many squares wins outright. 0 leaves the "
                            "claiming to whatever else reads it -- rules.scoring with "
                            "count 'territory' compares ground held when time runs out."
                        ),
                    },
                },
                required=["mode"],
                strict=strict,
            ),
            "turn": _object(
                {
                    "moves": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_MOVES_PER_TURN,
                        "description": (
                            "How many moves a turn is. 1 is the ordinary game. This is "
                            "the one rule a search feels rather than reads. The "
                            "branching factor becomes the legal move count raised to "
                            "this, so it is capped rather than left open."
                        ),
                    },
                    "ramp": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_MOVES_PER_TURN,
                        },
                        "minItems": 1,
                        "maxItems": 8,
                        "description": (
                            "The lengths of the opening turns, in order, before "
                            "'moves' takes over. With two moves a turn whoever starts "
                            "is a whole move ahead forever; 'moves': 2 with 'ramp': "
                            "[1] gives the first turn one move and levels it."
                        ),
                    },
                    "parts": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_MOVES_PER_TURN,
                        "items": _object(
                            {
                                "pieces": {
                                    "anyOf": [
                                        {"type": "string", "enum": ["own", "shared"]},
                                        {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "minItems": 1,
                                        },
                                    ],
                                    "description": (
                                        "Which pieces this part of the turn moves: 'own' "
                                        "(the mover's pieces and any nobody's piece they "
                                        "hold), 'shared' (the pieces either player moves, "
                                        "control 'shared'), or a list of piece ids."
                                    ),
                                },
                                "optional": {
                                    "type": "boolean",
                                    "description": (
                                        "The mover may keep this part instead of moving."
                                    ),
                                },
                            },
                            required=["pieces"],
                            strict=strict,
                        ),
                        "description": (
                            "A turn written as parts, in place of 'moves': one move per "
                            "part, each moving only the pieces its part names (Neutron: "
                            "the shared piece, then your own). A move is offered only if "
                            "the turn can still be finished after it."
                        ),
                    },
                    "order": {
                        "type": "string",
                        "enum": ["fixed", "any"],
                        "description": (
                            "'fixed' fills the parts in the order written; 'any' lets each "
                            "move fill whichever unfilled part admits its piece (the parts "
                            "must then admit no piece in common)."
                        ),
                    },
                },
                required=[],
                strict=strict,
            ),
            "obstacles": _object(
                {
                    "squares": _squares(
                        "Squares nothing may occupy. A ray stops at one and a piece "
                        "may not land on one."
                    ),
                    "mirror": {
                        "type": "boolean",
                        "description": (
                            "Also wall the mirror image of every square listed. A wall "
                            "belongs to the board rather than to a player, so this is "
                            "shorthand for writing a symmetric wall out -- not a way to "
                            "give one side a wall the other does not have. Defaults to "
                            "false, unlike setup.mirror, because a square list here is "
                            "usually meant absolutely."
                        ),
                    },
                    "blocks_leaps": {
                        "type": "boolean",
                        "description": (
                            "Whether a wall also stops a jumping piece. False, the "
                            "default, lets a knight vault a wall; true stops it. The "
                            "square a jump is stopped at is the one the xiangqi horse "
                            "is blocked at, one step along the longer axis, or the "
                            "leap's own stated legs. On a hex lattice every leap longer "
                            "than a step must state its legs."
                        ),
                    },
                },
                required=["squares"],
                strict=strict,
            ),
            "strike": _object(
                {
                    "effect": {
                        "type": "string",
                        "enum": list(STRIKE_EFFECTS),
                        "description": (
                            "What happens to a piece that is taken. 'capture' removes "
                            "it, which is the ordinary game unless the position is "
                            "'stay' or 'home'. 'demote' and 'promote' turn it into "
                            "another piece instead, following the chain."
                        ),
                    },
                    "chain": {
                        "type": "array",
                        "items": _object(
                            {
                                "from": {
                                    "type": "string",
                                    "description": "The piece that was taken.",
                                },
                                "to": {
                                    "type": "string",
                                    "description": "What it becomes instead.",
                                },
                            },
                            required=["from", "to"],
                            strict=strict,
                        ),
                        "minItems": 1,
                        "description": (
                            "Where a struck piece goes, one link per piece. A piece "
                            "with no link is removed when it is taken, as usual, so "
                            "the bottom of a demotion chain is simply left out. A "
                            "demotion chain may not cycle; a promotion chain may."
                        ),
                    },
                    "position": {
                        "type": "string",
                        "enum": list(STRIKE_POSITIONS),
                        "description": (
                            "Where the two pieces end up. 'advance' is the ordinary "
                            "capture -- the attacker takes the square -- and may only "
                            "be used when nothing survives the chain, because there "
                            "would be nowhere to put a survivor. 'stay' leaves the "
                            "attacker where it was and transforms the struck piece "
                            "where it stands; under effect 'capture' the taken piece "
                            "is removed and the attacker never moves (rifle chess). "
                            "'swap' exchanges the two. 'choose' advances the attacker "
                            "and lets it place the struck piece on a square of "
                            "'landing'. 'home' advances the attacker and puts the "
                            "struck piece -- itself under 'capture', what the chain "
                            "makes it otherwise -- back on its home square, one of "
                            "the squares its side's setup gives its type. A type with "
                            "at most one home on each colour goes to the one on the "
                            "capture square's colour, so a rook, knight or bishop "
                            "keeps its square colour; any other type, such as a pawn "
                            "with a home on every file, goes to the one on the "
                            "capture's file, else the one on the capture square's "
                            "colour; failing both, the nearest. A hex board has no "
                            "two colours, so there the file decides. If that square "
                            "is taken, or the type has no home, it is removed "
                            "(orthodox circe)."
                        ),
                    },
                    "switches_sides": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": (
                            "Pieces that change sides when they capture: the capturer "
                            "becomes the other player's piece where it lands "
                            "(Andernach chess). A piece that faces keeps pointing the way "
                            "it pointed, so it may face its new side's own camp. A royal "
                            "piece may not be listed."
                        ),
                    },
                    "by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": (
                            "The only capturing pieces whose captures strike; every other "
                            "piece captures as usual, moving onto the square it took. "
                            "Leave it out and every piece strikes. With effect 'capture' "
                            "and position 'stay' it is a piece that shoots: a capture by "
                            "a listed piece removes its target and the piece stays where "
                            "it stood (an archer). Each listed piece must capture "
                            "something the strike reaches, and the list may not name every "
                            "piece that does. Beside switches_sides the two lists are read "
                            "apart: by says whose captures strike, switches_sides who "
                            "changes sides, so a switcher outside by captures as usual and "
                            "changes sides on the square it took, and one in both strikes "
                            "and changes sides where the strike leaves it. A strike that "
                            "only switches sides reaches nothing, so by beside it is "
                            "refused."
                        ),
                    },
                    "landing": {**landing, "description": landing_description},
                    "immune": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Pieces a strike cannot reach; taking one removes it as "
                            "usual. Every royal piece must be listed here, because a "
                            "king that turns into something else makes check a "
                            "question about a piece that is not on the board yet. A "
                            "wild piece's type is immune without being listed, "
                            "and no strike list may name one."
                        ),
                    },
                },
                required=[],
                strict=strict,
            ),
            "contacts": _object(
                {
                    "links": {
                        "type": "array",
                        "items": _object(
                            {
                                "mover": {
                                    "type": "string",
                                    "description": "The piece that is doing the taking.",
                                },
                                "onto": {
                                    "type": "string",
                                    "description": "The piece it takes.",
                                },
                                "mover_to": {
                                    "type": "string",
                                    "description": (
                                        "What the mover becomes. Leave it out and the "
                                        "mover is unchanged, which is what an ordinary "
                                        "capture does."
                                    ),
                                },
                                "onto_to": {
                                    "type": "string",
                                    "description": (
                                        "What the piece it took becomes. Leave it out "
                                        "and that piece is removed, as usual."
                                    ),
                                },
                                "own": {
                                    "type": "boolean",
                                    "const": True,
                                    "description": (
                                        "Set true and the link is about the mover's own "
                                        "piece: the mover may move onto a friendly 'onto' "
                                        "along its capture moves, and the link says what "
                                        "both become. Leave it out for an enemy target. "
                                        "Beside rules.must_capture moving onto your own "
                                        "piece is no capture: never compulsory, and not "
                                        "allowed while a capture is on offer."
                                    ),
                                },
                            },
                            required=["mover", "onto"],
                            strict=strict,
                        ),
                        "minItems": 1,
                        "description": (
                            "What two pieces that meet both become, one link per pair. "
                            "A pair with no link is an ordinary capture, so only the "
                            "meetings that matter need saying. A link must change at "
                            "least one of the two."
                        ),
                    },
                    "position": {
                        "type": "string",
                        "enum": list(CONTACT_POSITIONS),
                        "description": (
                            "Where the two pieces end up. 'advance' is the ordinary "
                            "capture -- the mover takes the square -- and may only be "
                            "used when no link leaves the target standing, because "
                            "there would be nowhere to put it. 'stay' leaves the mover "
                            "where it was and transforms the target where it stands. "
                            "'swap' exchanges the two. 'choose' advances the mover and "
                            "lets it place a surviving target on a square of 'landing'."
                        ),
                    },
                    "landing": {**landing, "description": landing_description},
                    "immune": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Pieces no contact reaches; taking one removes it as usual "
                            "and leaves the mover alone. Every royal piece must be "
                            "listed here, because a king that turns into something else "
                            "makes check a question about a piece that is not on the "
                            "board yet. A wild piece's type is immune without "
                            "being listed, and no link may name one."
                        ),
                    },
                },
                required=["links"],
                strict=strict,
            ),
            "wrap": _object(
                {
                    "files": {
                        "type": "string",
                        "enum": list(WRAP_FILE_EDGES),
                        "description": (
                            "Which side edges a piece may run off and continue from the "
                            "opposite one. 'both' makes the board a cylinder. Edges are "
                            "named from the standard diagram, with the first player at "
                            "the bottom and file a on the left. Not on a hexagon, "
                            "whose sides are joined by n_s, ne_sw and nw_se."
                        ),
                    },
                    "ranks": {
                        "type": "string",
                        "enum": list(WRAP_RANK_EDGES),
                        "description": (
                            "The same for the top and bottom edges. Both axes together "
                            "make a torus, which has no corner to trap a king in."
                        ),
                    },
                    "files_by_player": _object(
                        {
                            "0": {"type": "string", "enum": list(WRAP_FILE_EDGES)},
                            "1": {"type": "string", "enum": list(WRAP_FILE_EDGES)},
                        },
                        required=["0", "1"],
                        strict=strict,
                    ),
                    "ranks_by_player": _object(
                        {
                            "0": {"type": "string", "enum": list(WRAP_RANK_EDGES)},
                            "1": {"type": "string", "enum": list(WRAP_RANK_EDGES)},
                        },
                        required=["0", "1"],
                        strict=strict,
                    ),
                    **_hexagon_wrap(strict),
                },
                required=[],
                strict=strict,
            ),
            "npcs": {
                "type": "array",
                "items": _object(
                    {
                        "piece": {
                            "type": "string",
                            "description": (
                                "A declared, non-royal piece id. The wild piece has that "
                                "type's name, drawing and value, and eats what that type "
                                "may capture; its moves are what it does in a player's "
                                "hands, not how it walks."
                            ),
                        },
                        "route": _squares(
                            "The squares it walks, in order and cyclically, starting on "
                            "the first, one step after every move (in a turn of several "
                            "moves, rules.turn, after each of them). Distinct squares, on "
                            "no other wild piece's route, with no piece of the opening "
                            "array on the first. One square is a wild piece that never "
                            "moves."
                        ),
                        "also_at": _squares(
                            "More wild pieces of the same kind on this same route: the "
                            "squares, other than the route's first, where each further one "
                            "starts, listed in route order and with no piece of the opening "
                            "array on any. Every one walks the route in step with the "
                            "first; one that finds another in its way waits behind it. "
                            "Leave the key out for one wild piece on the route."
                        ),
                        "respawn": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 16,
                            "description": (
                                "A taken wild piece of this entry comes back this many "
                                "phases after it was taken (a phase follows every move, so "
                                "8 is four moves of each player; one a round in rounds), "
                                "onto the first of its starts -- the route's first square, "
                                "then also_at in route order -- that is empty or holds a "
                                "player's non-royal piece its type may take, which it eats. "
                                "While every start is held it comes back at the first phase "
                                "one is free. One clock per entry and per mirrored half; it "
                                "starts again while the entry is still short. Leave it out "
                                "(or write 0) for a wild piece that, once taken, is gone."
                            ),
                        },
                        "turns_back": {
                            "type": "boolean",
                            "description": (
                                "true walks the route as a line instead of a cycle: first to "
                                "last and back, never from the last square to the first. "
                                "The piece faces forward at the start and turns round at "
                                "each end. Where the square ahead holds another wild piece "
                                "or a piece its type may not take, it turns round and steps "
                                "the other way, and blocked both ways it waits, turned; so "
                                "blocking it steers it. Not on a one-square route, not "
                                "beside also_at, and only in a game without check (win "
                                "royal_capture, annihilation or none; no forbid_check; no "
                                "reach.safe). Leave it out (or write false) for a cycle."
                            ),
                        },
                        "pounce": {
                            "type": "string",
                            "enum": ["prey", "attackers"],
                            "description": (
                                "\"prey\": after each move, before the wild pieces step, this "
                                "piece takes a piece of the player who has just moved that it "
                                "attacks by its own type's capture moves, moving onto it; with "
                                "several, a royal first, then the higher value, then the "
                                "nearer, then the square nearest the mover's first rank and "
                                "furthest to the mover's left. A royal it attacks is in check. "
                                "\"attackers\": instead it takes a piece of the player who has "
                                "just moved that attacks it (one that could take it, and that "
                                "its type may take; a frozen or disarmed piece attacks "
                                "nothing), chosen in the same order; a royal that attacks it "
                                "is in check, so nothing walks up to it, and it falls to a "
                                "piece that already attacks it when its owner moves. "
                                "Off its route it stands until it pounces again, and back on "
                                "it it walks on. Not in rounds, not beside respawn, "
                                "turns_back or forbid_check; its type's moves must read the "
                                "same for both players (no facing, region, move condition or "
                                "owner-named hop screen), and every entry of one type must "
                                "agree on pounce and bounty."
                            ),
                        },
                        "split": {
                            "type": "integer",
                            "minimum": 2,
                            "maximum": 8,
                            "description": (
                                "Only beside pounce \"attackers\": the most wild pieces of "
                                "this type that may stand at once. When it has attackers to "
                                "take, a copy of it grows on each attacker's square in the "
                                "pounce order, eating it, while fewer than this many stand "
                                "(counted as the phase goes, so two of the type share the "
                                "room); past the room it moves onto the next as a plain "
                                "pouncer does, and the rest are spared. A copy is the same "
                                "wild piece in every way and acts from the next phase. Not "
                                "below the type's starts, not beside respawn or drops that "
                                "fill a hand from captures; every entry of one type must agree."
                            ),
                        },
                        "bounty": {
                            "type": "number",
                            "minimum": 0,
                            "description": (
                                "What a player gains by taking it (or by a capture whose "
                                "blast destroys it), read by rules.scoring "
                                "with count 'points'. Defaults to the piece's value."
                            ),
                        },
                        "mirror": {
                            "type": "boolean",
                            "description": (
                                "Also place a second wild piece on the mirror image of "
                                "the route, walking it in the same order. A route has a "
                                "direction, so one wild piece is an asymmetry and a "
                                "mirrored pair is not. Defaults to false."
                            ),
                        },
                        "immune": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "The only piece ids that may take this wild piece; it is "
                                "immune to every other. Leave the key out and anything "
                                "may take it. Another wild piece is not bound by it. It "
                                "holds only while the piece is wild: one taken into a "
                                "hand and dropped (rules.drops), or placed for a player, "
                                "is captured as any piece of its type."
                            ),
                        },
                    },
                    required=["piece", "route"],
                    strict=strict,
                ),
                "minItems": 1,
                "description": (
                    "Wild pieces are pieces neither player owns. After every move each one "
                    "steps to the next square of its route and eats whatever stands there "
                    "that its type may capture, or waits if it may not. A royal piece "
                    "standing where a wild piece steps next is in check from it when that "
                    "type may capture the royal. Taking a wild piece is worth its bounty. "
                    "A wild piece's type may carry a line of sight (pieces[].sight, whose "
                    "'any'), freezing, disarming or shielding both players' pieces alike, and "
                    "a wild piece never takes a shielded piece: it waits in front of one, so "
                    "a shielded royal on its next square is not in check from it. "
                    "Where captures explode, a wild piece's eat never does, "
                    "and a wild piece a player's blast destroys pays its bounty to the "
                    "mover, as taking it would."
                ),
            },
            "portals": {
                "type": "array",
                "items": _object(
                    {
                        "moves": {
                            "type": "array",
                            "items": portal_move,
                            "maxItems": MAX_MOVE_DESCRIPTORS,
                            "description": (
                                "How an end of this group moves when a player moves it "
                                "(moved_by), written exactly as a piece's moves are. A "
                                "portal only ever steps onto an empty square, so each move "
                                "leaves mode out (it means 'move' here) or says 'move'; "
                                "none sweeps, none has a from_rank or region condition, and "
                                "each reads the same for both players (a king's step is "
                                "kind 'leap', vectors [[1, 0], [1, 1]], symmetry "
                                "'rotate4'). Required when anybody may move an end; omit "
                                "it when nobody does."
                            ),
                        },
                        "orange": {
                            "type": "string",
                            "description": (
                                "Where the pair's first end starts: an empty square, not a "
                                "wall. Algebraic, such as c4. The keys name the pair's two "
                                "ends, this one first; both are drawn in the pair's colour."
                            ),
                        },
                        "blue": {
                            "type": "string",
                            "description": "Where the pair's second end starts, likewise.",
                        },
                        "moved_by": _object(
                            {
                                end: {
                                    "type": "array",
                                    "items": {"type": "integer", "enum": [0, 1]},
                                    "description": (
                                        f"Which players may move the pair's {which} end "
                                        f"(`{end}` in the file), as a turn: [] for nobody "
                                        "(the default), [0], [1], or [0, 1] for either."
                                    ),
                                }
                                for end, which in (("orange", "first"), ("blue", "second"))
                            },
                            required=[],
                            strict=strict,
                        ),
                        "passage": _object(
                            {
                                player: {
                                    "type": "string",
                                    "enum": list(PORTAL_PASSAGES),
                                    "description": (
                                        f"Which end player {player}'s pieces may go in by: "
                                        "'both_ways' (the default), 'orange_to_blue' (in at "
                                        "the first end only), 'blue_to_orange' (in at the "
                                        "second end only), or 'closed', when both ends are "
                                        "an uncapturable obstacle to that player."
                                    ),
                                }
                                for player in ("0", "1")
                            },
                            required=[],
                            strict=strict,
                        ),
                        "carries": _object(
                            {
                                end: {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        f"The piece ids that may go in by the "
                                        f"pair's {which} end; every other piece meets it as "
                                        "an uncapturable obstacle. Absent: every piece. "
                                        "Never empty, never every piece that can move."
                                    ),
                                }
                                for end, which in (("orange", "first"), ("blue", "second"))
                            },
                            required=[],
                            strict=strict,
                            description=(
                                "Which pieces may go in by each end of a pair, "
                                "keyed by the end; an end not listed lets every piece in."
                            ),
                        ),
                        "ends": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 8,
                            "items": _object(
                                {
                                    "at": {
                                        "type": "string",
                                        "description": (
                                            "Where this end starts: an empty square, not a "
                                            "wall. Algebraic, such as c4."
                                        ),
                                    },
                                    "to": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 7,
                                        "description": (
                                            "The index in this group's ends of the end a "
                                            "piece that goes in here comes out of: never "
                                            "this end itself. Absent: a way out only, which "
                                            "nobody goes in by and pieces only come out of."
                                        ),
                                    },
                                    "enter": {
                                        "type": "array",
                                        "items": {"type": "integer", "enum": [0, 1]},
                                        "description": (
                                            "Which players' pieces may go in by this end: "
                                            "[0], [1] or [0, 1] (the default). Only beside "
                                            "'to'."
                                        ),
                                    },
                                    "moved_by": {
                                        "type": "array",
                                        "items": {"type": "integer", "enum": [0, 1]},
                                        "description": (
                                            "Which players may move this end, as a turn: [] "
                                            "for nobody (the default), [0], [1] or [0, 1]."
                                        ),
                                    },
                                    "carries": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": (
                                            "The piece ids that may go in by this end; "
                                            "absent, every piece. Only beside 'to'."
                                        ),
                                    },
                                },
                                required=["at"],
                                strict=strict,
                            ),
                            "description": (
                                "Three to eight ends, each leading to another "
                                "of the group or nowhere: a ring (each to the next), a "
                                "funnel (several to one way out), or any other map. Every "
                                "end must lead somewhere or be led to. Two ends are a "
                                "pair: write them as orange and blue instead."
                            ),
                        },
                        "no_straight_back": _NO_STRAIGHT_BACK,
                        "kind": {
                            "type": "string",
                            "enum": ["fold"],
                            "description": (
                                "'fold' makes the group fold portals; absent, they "
                                "are momentum portals (the default). A fold is "
                                "not a piece: its square may hold a piece of either side, "
                                "which is captured there as anywhere, and may start under a "
                                "piece. A move that ends on an empty fold that lets the piece "
                                "in -- a leap, slide, hop or pawn's step -- may stay there, or "
                                "go through and end on the end this one leads to instead, as "
                                "a landing there: onto an empty square if the step could end "
                                "on one, onto an enemy piece it may take (captured) if the "
                                "step could capture, never onto its own piece or its own "
                                "square, promoting by that square. One fold per move. A move "
                                "that passes over a fold, or ends on one a piece stands in, is "
                                "not changed by it. A fold that does not let a piece in, or "
                                "leads nowhere, is plain floor to it. A fold moved by a player "
                                "may land on a piece of either side (never on another portal), "
                                "which then stands in it; moving it never moves that piece. "
                                "Castling, drops and en passant never go through a fold, and a "
                                "move through one opens no en passant."
                            ),
                        },
                    },
                    required=[],
                    strict=strict,
                    description=(
                        "One group: a pair, written with orange and blue (and moved_by, "
                        "passage and carries keyed by them), or a map, written with ends "
                        "alone. Never both. Its moves are how its ends move."
                    ),
                ),
                "minItems": 1,
                "maxItems": 4,
                "description": (
                    "Groups of portals, numbered by their place in the list (at most four "
                    "groups and eight ends in all). A group is a pair (orange and blue, "
                    "each leading to the other) or a map of ends (each leading to one "
                    "other end of its group, or nowhere). A piece whose move ends on a "
                    "portal end that lets it in does not stop there: it comes out of the "
                    "end that one leads to, one king's step beyond it in the direction it "
                    "was travelling (right-and-up by any amounts comes out diagonally up "
                    "and right; on a hex lattice along the nearest of the twelve hex king's "
                    "steps, so a hex line continues and the hex knight comes out "
                    "diagonally), and its move ends there -- capturing, promoting or winning "
                    "there as usual. A blocked exit (off the board, a wall, its own piece, a "
                    "piece it may not take, another portal, or its own square) is no move. "
                    "An end that does not let a piece in is an uncapturable obstacle to it. "
                    "A portal is never captured. A group with kind 'fold' works "
                    "differently (see kind). Beside strike or contacts the square a move "
                    "comes out on (a fold's far square) is the capture square, read as "
                    "anywhere: a strike that stays leaves the mover where it set out, a swap "
                    "puts the struck piece on that square, home and choose move the mover to "
                    "the exit. Beside explode_on_capture the blast centres on that square; "
                    "a portal end in it stands, and a fold in it loses the piece standing in "
                    "it and stays. Beside territory a move through a portal claims the "
                    "square it ends on (the exit, or a fold's far square), never a portal "
                    "end's, and moving a portal claims nothing. A piece with a region comes "
                    "out only on a square of its region: an exit outside it is blocked, no "
                    "move, as a wall is (and, as for any landing, it goes in only by an end "
                    "standing in its region). Beside a sweep a momentum end stops it, and a "
                    "landing on an empty fold may go through, sweeping only before the fold. "
                    "Not usable beside npcs."
                ),
            },
            "purse": _object(
                {
                    "start": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": PURSE_MAX,
                        "description": (
                            f"Coins each player starts with, 0 to {PURSE_MAX}. A purse holds "
                            f"at most {PURSE_MAX}; every bounty (pieces[].bounty, "
                            "rules.npcs[].bounty) is paid into it as well, and what passes "
                            "the cap is lost."
                        ),
                    },
                    "market": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "The piece types a player may buy, each with a pieces[].price. "
                            "Buying is a whole move: the price leaves the buyer's purse and "
                            "the piece is placed on an empty square of the region, entering "
                            "play as a dropped piece does (its first facing, promoted if it "
                            "enters its zone under promote_on_entry). Supply is unlimited. "
                            "Never a royal, a wild type, the duck or a type with no moves. A "
                            "spy for sale is a spy gained when it stands on the board "
                            "(pieces.spy), only as a spy type both players start with. Optional: leave it out where coins buy only hires or "
                            "bribes."
                        ),
                    },
                    "bribe": {
                        "type": "boolean",
                        "description": (
                            "true: a player may also buy an enemy piece of a priced type "
                            "standing next to one of theirs, at twice its price, as a whole "
                            "move; it becomes theirs where it stands, keeping the way it "
                            "points. false (the default) never."
                        ),
                    },
                    "bounty_to": {
                        "type": "string",
                        "enum": ["taker", "loser"],
                        "description": (
                            "Who a capture's bounty (pieces[].bounty) is paid to, into the "
                            "purse and the points alike. \"taker\" (the default): the player "
                            "whose move took the piece. \"loser\": the player who owned it, "
                            "so coins run against the lead and a side that loses a piece can "
                            "buy one back; a piece nobody owns then pays nobody. \"loser\" "
                            "needs a pieces[].bounty and is refused beside a wild piece's "
                            "bounty. Keep a type's bounty under what the coins buy back, or "
                            "losing it is no loss."
                        ),
                    },
                    "upkeep": {
                        "anyOf": [
                            {"type": "boolean"},
                            _object(
                                {
                                    "promotion": {
                                        "type": "string",
                                        "enum": list(UPKEEP_PROMOTIONS),
                                        "description": (
                                            "What a promotion pays for: \"both\" (the "
                                            "default) the moving piece's arrival and the "
                                            "promoted type's on the same square, \"new\" "
                                            "only the promoted type's."
                                        ),
                                    },
                                },
                                required=[],
                                strict=strict,
                            ),
                        ],
                        "description": (
                            "true: every move costs coins, its piece's pieces[].upkeep times "
                            "the rows from its owner's first row to the square the piece ends "
                            "on (0 on the first row), paid from the purse before the move (a "
                            "capture's bounty comes after); a move the purse cannot pay for "
                            "may not be made, and attack still ignores coins. A castle pays "
                            "for the castling piece only, a buy its price and its square, a "
                            "hire nothing. Needs a type with an upkeep. Refused beside rounds, "
                            "a long turn, portals, wild pieces, a command, a held or shared "
                            "type, drops, a hex board, contacts and switches_sides. false "
                            "(the default) never."
                        ),
                    },
                    "region": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Where a bought piece may be placed, written for the first player "
                            "and reflected for the second; leave it out for each player's "
                            "first rank. Each market type needs a square in it it may stand "
                            "on (not a wall, inside its own region). Algebraic squares such "
                            "as a1 or e4."
                        ),
                    },
                },
                required=["start"],
                strict=strict,
                description=(
                    "Coins, and what to spend them on: each player holds a purse, filled "
                    "by bounties and spent on buying pieces from a market, hiring a priced "
                    "piece nobody owns that stands next to one of theirs (its price), or "
                    "under bribe buying an enemy one (twice the price). In rounds "
                    "(rules.simultaneous) a buy is an arrival with the round's moves: two "
                    "buys onto one square, or a buy and a move, collide by pieces.beats and "
                    "the tie, each price paid whether or not the bought piece survives; a "
                    "purse that can hire or bribe is refused there. Beside explode_on_capture "
                    "a blast pays the mover "
                    "the bounty of every piece it destroys as taking it would (nothing for "
                    "the mover's own), up to the cap; under drops.placement a "
                    "held hand is placed first. In a turn of several moves each buy is a "
                    "move, paid from the purse as the move before left it. Beside portals a "
                    "piece is never bought onto a portal's square, but may be bought into an "
                    "empty fold, where it stands."
                ),
            ),
            # Q1074b2 (`docs/plan-push.md` §1.2, §2): the core plays the single push, and
            # since Q1074b4 rows and weights, since Q1074b6 no push back (K5), since Q1074b7
            # the crush (W1) and rows on a wrapped board (H4).
            "push": _object(
                {
                    "chain": {
                        "type": "string",
                        "enum": list(PUSH_CHAINS),
                        "description": (
                            "'one' (the default): a push moves one piece, so a piece "
                            "standing behind the one landed on blocks the push. 'row': a "
                            "push moves the whole line, the piece landed on and every piece "
                            "behind it up to the first empty square, each one step (the "
                            "last falls off at the edge, taken); the push is allowed only "
                            "if the pusher may push every piece in the line. 'weight': as "
                            "'row', and only if the pusher's strength (pieces[].strength, "
                            "else its weight) is at least the sum of the weights in the "
                            "line; every piece that may be pushed then needs a weight, every "
                            "pusher a strength or a weight, and a pushes list still applies. "
                            "On a wrapped board a line that comes round to the piece landed "
                            "on fills its ring and cannot move, and one that comes round to "
                            "the pusher's own square ends there."
                        ),
                    },
                    "edge": {
                        "type": "string",
                        "enum": list(PUSH_EDGES),
                        "description": (
                            "'removes' (the default): a piece pushed off the board (or into "
                            "a hexagon's missing corner) is removed, counted as taken by the "
                            "pusher; never the mover's own royal. 'blocks': a push that "
                            "would leave the board is not offered. A wall blocks unless "
                            "walls is 'crush'."
                        ),
                    },
                    "walls": {
                        "type": "string",
                        "enum": list(PUSH_WALLS),
                        "description": (
                            "'block' (the default): a push that would press a piece into a "
                            "wall (rules.obstacles) is not offered. 'crush': the piece "
                            "pressed against the wall is removed instead, counted as taken by "
                            "the pusher as one pushed off the board is; never the mover's "
                            "own royal. Needs a wall on the board."
                        ),
                    },
                    "no_push_back": {
                        "type": "boolean",
                        "description": (
                            "true: a piece just pushed from one square to the next, and not "
                            "removed, may not be carried back onto the square it was pushed "
                            "from by a push on the very next move, whether it pushes back "
                            "itself or another piece pushes a row it stands in (no shoving "
                            "straight back); any other move is allowed, and the bar lasts one "
                            "move. false (the default): no such bar. Needs a move that pushes "
                            "enemy pieces (or a turn of several moves)."
                        ),
                    },
                },
                required=[],
                strict=strict,
                description=(
                    "How pushes behave, for a game in which some move pushes "
                    "(pieces[].moves[].push); leave it out for the defaults. A pushed piece "
                    "is not moved by its owner: it does not promote, and it loses any "
                    "first-move right (a double step, castling). A push threat on a royal "
                    "is check, as a capture threat is."
                ),
            ),
            "duck": _object(
                {
                    "piece": {
                        "type": "string",
                        "description": (
                            "The duck's type: a declared, non-royal piece that no player "
                            "owns and nothing else names. Every turn is the mover's move and "
                            "then the mover's duck move: the same player moves the duck, "
                            "which may not stay where it is. It stands, blocks lines and "
                            "screens a hop; nothing may capture it and it captures nothing. "
                            "Only beside win 'royal_capture', 'annihilation' or 'none' (no "
                            "check rule), never in rounds or beside turn, and its type never "
                            "sweeps; a player's sweep stops short of it, as of any piece it may "
                            "not capture. A blast beside it leaves it standing, and no strike "
                            "or contact reaches it. Beside portals it "
                            "blocks a portal exit or a fold end it stands on, never goes "
                            "through one, never lands on a portal end and may stop in an "
                            "empty fold, which never moves it. max_plies must be even (each "
                            "half is a ply)."
                        ),
                    },
                    "start": {
                        "type": "string",
                        "description": (
                            "Where the duck starts: an empty square. Leave it out and it "
                            "starts off the board, and the first player's first duck move "
                            "puts it on any empty square (Duck chess's opening)."
                        ),
                    },
                    "goes": {
                        "type": "string",
                        "enum": ["anywhere", "by_moves"],
                        "description": (
                            "'anywhere' (the default, Duck chess): to any empty square but "
                            "its own; its type is then written moves: []. 'by_moves': by "
                            "its type's moves, which must read the same for both players and "
                            "land on empty squares. With nowhere to go the duck passes."
                        ),
                    },
                },
                required=["piece"],
                strict=strict,
            ),
            "simultaneous": _object(
                {
                    "tie": {
                        "type": "string",
                        "enum": ["initiative", "impossible"],
                        "description": (
                            "What decides a collision that is a tie, the two pieces each "
                            "beating the other or neither beating the other. Leave it out "
                            "and both are eliminated, leaving the square empty. "
                            "'initiative': one player holds the initiative, the "
                            "second player at the start; a tie goes to the holder's piece, "
                            "and the initiative then passes to the other player. "
                            "'impossible': a promise, checked, that no tie can ever occur, "
                            "because for every piece each player can have, exactly one of "
                            "the two beats the other (the armies use different pieces, "
                            "such as a circle of rock, paper and scissors)."
                        ),
                    },
                    "swap": {
                        "type": "string",
                        "enum": ["collision"],
                        "description": (
                            "'collision': two moves that each end on the "
                            "other's starting square meet as a collision, decided by "
                            "pieces.beats and the tie rule, instead of the two pieces "
                            "passing each other; a winner ends on its destination and a "
                            "loser is removed. Leave it out and swapping pieces pass. "
                            "Refused where no piece of one player can take a piece of the "
                            "other that can take it back."
                        ),
                    },
                    "both": {
                        "type": "string",
                        "enum": ["score"],
                        "description": (
                            "'score': an ending both players reach in the same "
                            "round (both royals taken, both armies gone, both on a goal) "
                            "goes to the player ahead under rules.scoring, and is a draw "
                            "only when the scores are equal. Leave it out and such a round "
                            "is a draw. Requires rules.scoring."
                        ),
                    },
                },
                required=[],
                strict=strict,
                description=(
                    "Both players move at once. Each round, each player chooses "
                    "one of their own legal moves on the same board without seeing the "
                    "other's; both are revealed and made together. Only two pieces moving "
                    "onto the same square collide, and pieces.beats decides who survives. "
                    "A piece that moves onto a square its target has just left lands "
                    "there if its move could land on an empty square, and otherwise "
                    "stays where it was (a capture-only move is void). Two such void "
                    "moves at each other two rounds in a row bar that pair of moves for "
                    "both players until either square changes. There is no check: "
                    "rules.win must be 'royal_capture', 'annihilation' or 'none', and an "
                    "ending reached by both players in the same round is a draw. A round "
                    "counts as two moves on the clocks. Not usable beside castling, en "
                    "passant, a sweep, check rules, a turn of more than one move, a "
                    "portal a player moves, a strike or contact position 'choose', or "
                    "drops.no_mate. A capture under a strike placed 'stay' is a shot: "
                    "it never moves the shooter, so it never collides, and it is made "
                    "before the round's moves, so the piece it strikes does not make its "
                    "own. Two switches work "
                    "against a royal chased round after round, swap and both; "
                    "pieces[].rest (\"rest\": 1 on every piece) is a third remedy."
                ),
            ),
            "promote_on_entry": {
                "type": "boolean",
                "description": (
                    "A piece that enters play inside its own promotion zone -- dropped "
                    "there, or left standing there by a strike that turned it into a "
                    "promoting piece -- promotes at once, to the first piece its promotion "
                    "rule lists, even where promotion is otherwise optional. Needs "
                    "rules.promotion and a way in: rules.drops, or a strike whose chain "
                    "leaves a piece standing. Refused where no promoting piece could ever "
                    "enter its zone, such as a pawn whose drop region stops short of it. "
                    "False, the default, leaves such a piece standing where it arrived."
                ),
            },
            "scoring": _object(
                {
                    "count": {
                        "type": "string",
                        "enum": list(SCORING_COUNTS),
                        "description": (
                            "What to compare when the ply limit or the no-progress "
                            "clock runs out, instead of drawing. 'material' sums the "
                            "declared piece values, 'pieces' counts them, 'region' "
                            "counts your own pieces standing in the squares below, "
                            "'territory' counts ground claimed and needs rules.territory, "
                            "'points' counts bounties collected and needs a bounty, on a wild "
                            "piece (rules.npcs) or a piece type (pieces[].bounty). "
                            "Higher wins; equal is still a draw."
                        ),
                    },
                    "region": _squares(
                        "The ground being held, for count 'region'. Written for the "
                        "first player and reflected, so 'hold the centre' is symmetric "
                        "and 'hold their half' is not."
                    ),
                },
                required=["count"],
                strict=strict,
            ),
            "reach": _object(
                {
                    # SR3a (`docs/plan-schema-next.md` §1 rank 4): one id or a list.
                    "piece": {
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                        ],
                        "description": (
                            "Omit for any piece. A list: any listed piece arriving "
                            "decides it, so a side chooses its runner."
                        ),
                    },
                    "squares": _squares("Where arriving decides the game."),
                    "mirror": {
                        "anyOf": [
                            {"type": "boolean"},
                            {"type": "string", "enum": ["rotate"]},
                        ],
                        "description": (
                            "true aims each side at the other's end by reflecting "
                            "the rank; 'rotate' reflects both rank and file (180° "
                            "rotation); false sends both to the same squares."
                        ),
                    },
                    "safe": {
                        "type": "boolean",
                        "description": "The arriving piece must not be capturable where it lands.",
                    },
                    "result": {"type": "string", "enum": list(ENDING_RESULTS)},
                    "on": {
                        "type": "string",
                        "enum": list(REACH_ON),
                        "description": (
                            "'arrive' (the default): a piece standing on a "
                            "goal square decides the game. 'leave': the move that takes "
                            "the piece off a goal square decides it (Jeson Mor), and "
                            "standing there decides nothing. Not with safe."
                        ),
                    },
                    # TS1 (`docs/plan-tessabi.md` §1): scoring goals.
                    "removes": {
                        "type": "boolean",
                        "description": (
                            "true: a listed piece standing on its own side's goal at the end "
                            "of a move is taken off the board and banked for its owner (not "
                            "a capture). Not with safe, on 'leave', a royal runner, check, "
                            "extinction, bare_royal or rounds."
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": REACH_COUNT_MAX,
                        "description": (
                            "How many banked plus standing listed pieces a side needs on its "
                            "goal to reach it (1, the default: the first arrival)."
                        ),
                    },
                    "both": {
                        "type": "string",
                        "enum": list(REACH_BOTH),
                        "description": (
                            "When both sides reach their goal on one move: 'first' (the "
                            "default, player 0's), 'draw', or 'mover' (the side that moved)."
                        ),
                    },
                },
                required=["squares"],
                strict=strict,
            ),
            "extinction": _object(
                {
                    "pieces": {
                        "anyOf": [
                            {"type": "string", "enum": ["any"]},
                            {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        ],
                        "description": "'any' counts the whole army.",
                    },
                    "min_count": {"type": "integer", "minimum": 1, "maximum": 64},
                    "result": {
                        "type": "string",
                        "enum": list(ENDING_RESULTS),
                        "description": "The result for the player who falls below the count.",
                    },
                    # Schema 18 (Q783): what `compile.ExtinctionRule.per_type` reads;
                    # schema 27 (Q936) adds "own" (`ExtinctionRule.own`), which reads the
                    # hand beside drops since SR4d (H11).
                    "per_type": {
                        "anyOf": [
                            {"type": "boolean"},
                            {"type": "string", "enum": ["own"]},
                        ],
                        "description": (
                            "True counts each listed type on its own, so a side loses when "
                            "any one of them falls below min_count, as in Extinction chess. "
                            "'own' counts each listed type on its own too, but "
                            "only against a player who starts with it, so two armies of "
                            "different kinds each lose by running out of any one of their "
                            "own; list both armies' kinds. Under 'own' a piece in hand "
                            "counts (rules.drops): a player starts with what its stock holds "
                            "as well as what stands on the board, and has not run out of a "
                            "kind while one is held. False, the default, counts the "
                            "listed types together. Needs two or more types listed."
                        ),
                    },
                },
                required=["pieces"],
                strict=strict,
            ),
            "drops": _object(
                {
                    "from_captures": {
                        "type": "boolean",
                        "description": "Captured pieces change sides and may be played back.",
                    },
                    "region": {
                        "type": "object",
                        "description": "Per piece id, the squares it may be dropped on.",
                        "additionalProperties": _squares("Where this piece may be dropped."),
                    },
                    "no_doubled": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "These may not be dropped onto a file already holding one of yours.",
                    },
                    "no_mate": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "These may not deliver checkmate on the drop itself.",
                    },
                    "stock": {
                        "type": "object",
                        "description": (
                            "Per piece id, how many each player holds before anything "
                            "has been captured, as a count of 1 to "
                            f"{MAX_STOCK} or \"{STOCK_UNLIMITED}\" for a supply that "
                            "never runs out. A stocked piece may have an empty 'moves' "
                            "list, a stone that is placed and never moves."
                        ),
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "integer", "minimum": 1, "maximum": MAX_STOCK},
                                {"const": STOCK_UNLIMITED},
                            ]
                        },
                    },
                    # SR3b: what `compile.DropRule.stock_by_player` reads.
                    "stock_by_player": _object(
                        {
                            player: {
                                "type": "object",
                                "additionalProperties": {
                                    "anyOf": [
                                        {"type": "integer", "minimum": 1, "maximum": MAX_STOCK},
                                        {"const": STOCK_UNLIMITED},
                                    ]
                                },
                            }
                            for player in ("0", "1")
                        },
                        required=["0", "1"],
                        strict=strict,
                        description=(
                            "Each player's own stock, in place of 'stock': '0' the first "
                            "player's, '1' the second's, each as 'stock' is ({} for "
                            "nothing). A handicap: the second player starts with two "
                            "knights in hand, {\"0\": {}, \"1\": {\"N\": 2}}. Not beside "
                            "'stock'; not both empty. A type 'unlimited' for one player is "
                            "'unlimited' for the other too, or absent there."
                        ),
                    ),
                    # Schema 18 (Q784): what `compile.DropRule.placement` reads.
                    "placement": {
                        "type": "boolean",
                        "description": (
                            "True is a placement phase: while a player holds any piece in "
                            "hand, a drop is their only legal move, so a stock is played "
                            "out before anything on the board moves (Placement chess, "
                            "Sittuyin). Only then may a royal start in the stock, once, "
                            "and not also placed."
                        ),
                    },
                    # Schema 18 (Q785): what `compile.DropRule.faces` reads.
                    "either_face": {
                        "type": "boolean",
                        "description": (
                            "True lets a piece in hand be dropped as itself or as its "
                            "other face, the piece whose promoted_from it is (Kyoto "
                            "shogi). Needs at least one promoted_from pair a hand can "
                            "hold; each base may have one face, and no face may itself be "
                            "a base, stocked, or royal. Beside explode_on_capture only the "
                            "piece the capture took goes to the hand; the blast's victims "
                            "enter none."
                        ),
                    },
                    # Q1059a: what `compile.DropRule.gates` reads.
                    "gating": _object(
                        {
                            "ranks": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 0},
                                "minItems": 1,
                                "description": "Rows, counted from each player's own side.",
                            },
                            "squares": _squares(
                                "The first player's gate squares, reflected for the second."
                            ),
                        },
                        required=[],
                        strict=strict,
                        description=(
                            "Gating (Seirawan chess): a piece in hand enters only as part of "
                            "a move, onto the square a piece of its side vacates by leaving "
                            "its starting square for the first time, where that square is in "
                            "these rows (ranks) or squares; a castle may gate on the king's "
                            "or the rook's square. Never by a drop otherwise. Needs a stock "
                            "and from_captures false; refused beside region, no_doubled, "
                            "no_mate, placement and either_face."
                        ),
                    ),
                },
                required=[],
                strict=strict,
            ),
            "explode_on_capture": _object(
                {
                    "radius": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 3,
                        "description": (
                            "How far the blast reaches around the capture square; 1 is "
                            "atomic chess. 0 destroys only the capturer and the piece "
                            "it took (kamikaze chess), and then 'immune' may not be set. "
                            "Not with blast, which states the shape instead."
                        ),
                    },
                    # Generation 48 (SR1b): the blast's shape stated, in place of `radius`.
                    "blast": {
                        **_object(
                            {
                                "vectors": {
                                    "type": "array",
                                    "items": vector,
                                    "minItems": 1,
                                    "description": (
                                        "The steps from the capture square the blast "
                                        "takes, each turned by the symmetry."
                                    ),
                                },
                                "symmetry": {
                                    "type": "string",
                                    "enum": list(SYMMETRIES),
                                    "description": (
                                        "How the vectors are expanded, as on a move, and "
                                        "refused off its lattice as there."
                                    ),
                                },
                            },
                            required=["vectors"],
                            strict=strict,
                        ),
                        "description": (
                            "Instead of radius: the squares the blast takes, as [dx, dy] "
                            "steps from the capture square. [[1, 0]] with rotate4 is a "
                            "cross that spares the diagonals; [[0, 1], [0, 2], [0, -1], "
                            "[0, -2]] two squares each way along the file. The blast is "
                            "the board's, one shape round every square, so it must read "
                            "the same for both players (its mirror image is itself). Not "
                            "with radius."
                        ),
                    },
                    "immune": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Pieces that survive standing next to a capture.",
                    },
                    "by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "The only capturing pieces whose captures explode. Leave it "
                            "out and every capture does. An empty list is no rule at all."
                        ),
                    },
                    "of": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "The only captured pieces whose being taken explodes. Leave "
                            "it out and taking anything does. An empty list is no rule "
                            "at all."
                        ),
                    },
                    # SR5a (T1): which of an exploding capture's pieces rules.strike reaches.
                    "strikes": {
                        "type": "string",
                        "enum": list(BLAST_STRIKE_VALUES),
                        "description": (
                            "Beside rules.strike, which pieces of a capture that explodes "
                            "the strike reaches, after the blast has cleared: 'none' (the "
                            "default; the capture strikes nothing, it explodes), 'target' "
                            "(the piece taken is struck: it follows its chain link and "
                            "stands on the capture square, or goes home under position "
                            "'home'), 'blast' (each piece the blast destroys that is not "
                            "immune to the strike is struck where it stood, or sent home), "
                            "or 'both'. A piece with no link is removed; the capturer is "
                            "always destroyed and never changes sides. Other than 'none' "
                            "it needs rules.strike.position 'stay' or 'home' (not effect "
                            "'capture' with 'stay', which only removes) and no "
                            "rules.contacts. A capture that explodes makes no contact."
                        ),
                    },
                },
                required=[],
                strict=strict,
            ),
            "max_plies": {"type": "integer", "minimum": 2, "maximum": 10000},
            # Schema 10: the one clock that may say it does not exist. `"never"` rather
            # than a number, because no number means "no rule" here -- see
            # `schema.REPETITION_NEVER`.
            "repetition_draw": {
                "anyOf": [
                    {"type": "integer", "minimum": 2, "maximum": 10},
                    {"const": REPETITION_NEVER},
                ],
                "description": (
                    "How many repetitions of a position draw the game, or \"never\" for "
                    "a game with no draw by repetition at all."
                ),
            },
            "no_progress_draw": {"type": "integer", "minimum": 2, "maximum": 1000},
        },
        required=["win"],
        strict=strict,
    )

    return _object(
        {
            "schema_version": {"type": "integer", "enum": [SCHEMA_VERSION]},
            "name": {"type": "string", "minLength": 1},
            "description": {
                "type": "string",
                "description": "How the game plays and what makes it worth playing, in a few sentences.",
            },
            "board": _object(
                {
                    "width": {"type": "integer", "minimum": 2, "maximum": MAX_BOARD_DIM},
                    "height": {"type": "integer", "minimum": 2, "maximum": MAX_BOARD_DIM},
                    "lattice": {
                        "type": "string",
                        "enum": list(LATTICES),
                        "description": (
                            "'square', the default, or 'hex'. A hex board is a width by "
                            "height rhombus of hexagons in axial coordinates, file q and "
                            "rank r, whose six neighbours are [1,0] [0,1] [-1,1] [-1,0] "
                            "[0,-1] [1,-1]; cut the shape you want out of it with "
                            "rules.obstacles, or set shape 'hexagon' for a regular hexagon."
                        ),
                    },
                    "shape": {
                        "type": "string",
                        "enum": list(SHAPES),
                        "description": (
                            "'grid', the default, is every cell of the width by height "
                            "frame. 'hexagon' needs lattice 'hex' and an odd width equal "
                            "to the height (11 is six cells a side): the rhombus's corners "
                            "are not cells at all, no rule may name them, and rules.wrap "
                            "joins the hexagon's sides with n_s, ne_sw and nw_se."
                        ),
                    },
                },
                required=["width", "height"],
                strict=strict,
            ),
            "pieces": {
                "type": "array",
                "items": piece,
                "minItems": 1,
                "maxItems": MAX_PIECE_TYPES,
            },
            "setup": _object(
                {
                    "mirror": {
                        "type": "boolean",
                        "description": (
                            "When true, list only the first player's army; the second "
                            "player's is the reflection across the middle of the board."
                        ),
                    },
                    "orientation": {
                        "type": "string",
                        "enum": list(ORIENTATIONS),
                        "description": (
                            "How the second player's army and move directions are "
                            "derived. mirror reflects the rank and leaves the file, as "
                            "chess does; rotate turns the whole board 180 degrees."
                        ),
                    },
                    "placements": {"type": "array", "items": placement, "minItems": 1},
                },
                required=["placements"],
                strict=strict,
            ),
            "rules": rules,
        },
        required=["schema_version", "name", "description", "board", "pieces", "setup", "rules"],
        strict=strict,
    )


# What each pair's sides are called on the page, in `WRAP_PAIRS` order.
_PAIR_WORDS = {
    "n_s": "top ('n') and bottom ('s')",
    "ne_sw": "upper-right ('ne') and lower-left ('sw')",
    "nw_se": "upper-left ('nw') and lower-right ('se')",
}


def _hexagon_wrap(strict: bool) -> dict[str, Any]:
    """A hexagon's three pairs of sides and their per-player twins (schema 16)."""
    out: dict[str, Any] = {}
    for key, pair in WRAP_PAIRS.items():
        values = ["none", pair.low, pair.high, "both"]
        out[key] = {
            "type": "string",
            "enum": values,
            "description": (
                f"On a hexagon only: whether its {_PAIR_WORDS[key]} sides are joined. "
                "'both' lets a piece leave by either and come back in by the other, "
                "straight across; a side's name lets it leave by that side only. Sides "
                "are named from the standard diagram, the first player at the bottom."
            ),
        }
        out[f"{key}_by_player"] = _object(
            {
                "0": {"type": "string", "enum": values},
                "1": {"type": "string", "enum": values},
            },
            required=["0", "1"],
            strict=strict,
        )
    return out


def _squares(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "description": f"{description} Algebraic squares such as a1 or e4.",
    }


_NO_STRAIGHT_BACK: dict[str, Any] = {
    "type": "boolean",
    "description": (
        "When true, a portal of this group that has just been moved may not be "
        "moved straight back onto the square it came from on the very next move; it may go "
        "anywhere else, and back on the move after. Stops two players undoing each other's "
        "portal moves for ever. Needs an end that whoever moves next may move too (in "
        "moved_by for both players, or a turn longer than one move), and group moves "
        "with a move back; a bar no move can meet is refused. Default false."
    ),
}


def _object(
    properties: dict[str, Any],
    required: list[str],
    strict: bool = True,
    description: str | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    if description is not None:
        schema["description"] = description
    if strict:
        schema["additionalProperties"] = False
    return schema
