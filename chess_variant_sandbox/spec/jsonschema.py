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
    CONTACT_POSITIONS,
    ENDING_RESULTS,
    FACING_AXES,
    FACING_NAMES,
    LATTICES,
    MAX_BOARD_DIM,
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
    STRIKE_EFFECTS,
    STRIKE_POSITIONS,
    SYMMETRIES,
    TERRITORY_MODES,
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
                    "leap or a range-1 slide. "
                    "sweep: jump to any square along the line, removing every piece "
                    "passed over, friendly included; a piece it may not capture blocks, "
                    "and a sweep that could pass over the enemy royal is check."
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
            "path": {
                "type": "string",
                "enum": list(PATHS),
                "description": (
                    "For a slide only, how its path runs. straight (the default): every "
                    "step the same direction. circle: each step turns to the next of the "
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
                    "half a turn); a zigzag two less than half a turn apart. Refused on "
                    "any other kind, at range 1, on a hex lattice, beside portals, on the "
                    "piece en passant is about, and on a wild piece."
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
            "screen": {
                "type": "string",
                "enum": list(SCREEN_OWNERS),
                "description": (
                    "For hops only, whose piece may serve as the screen. 'any' is the "
                    "xiangqi cannon and the default, 'own' is a piece that vaults its "
                    "own army, and 'enemy' is one that can only jump an opponent."
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
                    "empty list or a wild piece."
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
                    "command that moves its pieces onto empty squares only. Refused beside "
                    "drops that fill a hand from captures."
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
                    "secret. Both players must start with a spy on the board, nothing may "
                    "bring one back (no promotion to it, no stock or hand holding it, no "
                    "drops.from_captures, no strike.switches_sides), it may not be royal, and "
                    "something of the other player's must be able to remove it. Leave it out "
                    "for an ordinary piece."
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
                    "turn, so the rest never leaves a player without a move. Never on a "
                    "wild piece, and not beside a turn of more than one move "
                    "(rules.turn moves or ramp)."
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
                    "only onto a start no shield covers. Refused beside "
                    "rules.portals and rules.explode_on_capture, a shield is refused beside a "
                    "sweep move, and a sight that can never bind (no "
                    "type it affects is ever held by the side it affects, a freeze on types "
                    "with no move, a disarm on types that capture nothing, a shield on types "
                    "nothing can capture, a command on types with no move) is refused. A "
                    "command never touches a royal (an affects naming one is refused) and is "
                    "refused beside rules.simultaneous, rules.reach, rules.territory, "
                    "rules.strike, rules.contacts, pieces[].rest and a sweep move."
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
            "player": {"type": "integer", "enum": [0, 1]},
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
                    "both armies' setup. Not beside drops that fill from captures, a circe "
                    "strike or promote_on_entry."
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
                "description": "Requires promotion; applies to the promoting piece.",
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
                },
                required=["moves"],
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
                            "is blocked at, one step along the longer axis."
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
                            "(Andernach chess). A royal piece may not be listed."
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
                            "something the strike reaches, the list may not name every "
                            "piece that does, and it may not stand beside switches_sides."
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
                            "question about a piece that is not on the board yet."
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
                                        "both become. Leave it out for an enemy target."
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
                            "board yet."
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
                            "the first. Distinct squares, on no other wild piece's route, "
                            "with no piece of the opening array on the first. One square "
                            "is a wild piece that never moves."
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
                                "What a player gains by taking it, read by rules.scoring "
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
                                "may take it. Another wild piece is not bound by it. Not "
                                "usable beside rules.drops, because capture permission is "
                                "a fact about a type rather than about who owns it and "
                                "the immunity would follow the piece out of a hand."
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
                    "a shielded royal on its next square is not in check from it."
                ),
            },
            "portals": {
                "type": "array",
                "items": _object(
                    {
                        "moves": {
                            "type": "array",
                            "items": move,
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
                    "and right), and its move ends there -- capturing, promoting or winning "
                    "there as usual. A blocked exit (off the board, a wall, its own piece, a "
                    "piece it may not take, another portal, or its own square) is no move. "
                    "An end that does not let a piece in is an uncapturable obstacle to it. "
                    "A portal is never captured. A group with kind 'fold' works "
                    "differently (see kind). Not usable on a hex lattice or beside "
                    "npcs, strike, contacts, explode_on_capture, territory, a sweep, or a "
                    "piece region."
                ),
            },
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
                            "check rule), never in rounds or beside turn, portals, a blast, "
                            "a strike, contacts or a sweep; max_plies must be even (each "
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
                    "portal a player moves, a strike or contact position 'choose', a "
                    "strike position 'stay', or drops.no_mate. Two switches work "
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
                            "'points' counts bounties collected and needs rules.npcs. "
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
                    "piece": {"type": "string", "description": "Omit for any piece."},
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
                    # schema 27 (Q936) adds "own" (`ExtinctionRule.own`).
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
                            "own; list both armies' kinds. False, the default, counts the "
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
                            "a base, stocked, or royal."
                        ),
                    },
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
                            "it took (kamikaze chess), and then 'immune' may not be set."
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
