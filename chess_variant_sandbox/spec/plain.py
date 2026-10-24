"""The rules' complaints in the designer's words (`docs/archive/plan-understanding.md` W5b, D40-D50).

`validate_spec` names what is wrong by where it is in the game's text --
``setup.placements[0].squares entry 'z9' is not a square on this board`` -- because its
first readers write that text: the proposing LLM repairs a draft from those paths, the
command line prints them and the tests match them. A person in the designer never sees
the text. They see a piece called *knight* with rows of *ways of moving*, a section called
*Where they start* and a card per rule, so the same problem reads *Where they start, row
1: z9 isn't a square on this board.*

This module is that translation and nothing else. It changes no condition: a problem in,
the same problem out, one for one, in other words. The validator's own sentences stay the
stable form everyone else matches on, and the designer's two servers translate at the
point they show a problem (`plain_problems`).

`FIELD_WORDS` is also the one table of what each part of the rules is called on the rule
cards (`web/cards.py`, D40): the label a field carries, and, where the schema's description
is written for the LLM (*rules.scoring with count 'points'*, *False, the default*), the
sentence a visitor reads under it instead.
"""

from __future__ import annotations

import re
from typing import Any

from chess_variant_sandbox.spec.describe import RULE_WORDS
from chess_variant_sandbox.spec.schema import (
    SIGHT_EFFECTS,
    is_simultaneous,
    piece_sights,
)

# Dotted from the top of the game's text, with no list indices: `pieces.moves.range` is
# every way of moving's *how far*. A label is what the field is called on the page; the
# second item, when there is one, replaces the schema's description under it.
FIELD_WORDS: dict[str, tuple[str, str | None]] = {
    # The parts the hand-written form draws, in the form's own words.
    "name": ("the game's name", None),
    "board": ("the board", None),
    "board.width": ("board width", None),
    "board.height": ("board height", None),
    "board.lattice": ("the cells are", None),
    "board.shape": ("the board is", None),
    "pieces": ("the pieces", None),
    "pieces.id": ("letter", None),
    "pieces.name": ("name", None),
    "pieces.value": ("worth", None),
    "pieces.royal": ("royal", None),
    "pieces.moves": ("ways of moving", None),
    "pieces.moves.kind": ("how it travels", None),
    "pieces.moves.symmetry": ("turned", None),
    "pieces.moves.vectors": ("direction", None),
    "pieces.moves.mode": ("and it may", None),
    "pieces.moves.range": ("how far", None),
    "pieces.moves.min_range": ("no nearer than", None),
    "pieces.moves.condition.from_rank": ("only from rank", None),
    "setup": ("where they start", None),
    "setup.placements": ("where they start", None),
    "setup.placements.piece": ("piece", None),
    "setup.placements.player": ("whose", None),
    "setup.placements.squares": ("squares", None),
    "setup.mirror": ("mirror for the other player", None),
    "setup.orientation": ("the second player's side of the board", None),
    "rules": ("the rules", None),
    # A piece's and a move's generated fields.
    "pieces.region": ("only on these squares", None),
    "pieces.promoted_from": ("promoted from", None),
    "pieces.captures": (
        "may take only",
        "The pieces this one may take. Leave it empty and it may take anything, as in every "
        "ordinary game. Set it and you get a ring of who beats whom: give rock scissors, "
        "scissors paper and paper rock, and no piece can take the one that takes it.",
    ),
    "pieces.facing": (
        "points a way",
        "This piece points in a direction, and unless it may not turn, may spend a whole "
        "move turning instead of moving. Its moves turn with it, so a pawn pointing right "
        "advances to the right. Give it only to a piece whose moves change when it turns: a rook moves the same "
        "whichever way it points, so turning one would be a move that does nothing, and "
        "that is refused. A royal piece may not have one.",
    ),
    "pieces.facing.axes": ("how many ways it can point", None),
    "pieces.facing.initial": ("which way it points at the start", None),
    "pieces.facing.turns": (
        "may turn",
        "Whether this piece may spend a move turning. Switch it off and each one points the "
        "way it was placed for the whole game, as a rear guard of pawns placed facing "
        "backwards does; one that comes into play later points the way it starts. Some "
        "must then be placed pointing another way, or the pointing would change nothing.",
    ),
    "pieces.moves.screen_range": ("how far past the piece it jumps", None),
    "pieces.moves.screen": (
        "whose piece it jumps",
        "For a hop only: whose piece it may jump over. 'any' is the xiangqi cannon, and the "
        "usual choice; 'own' jumps only its own side's pieces, and 'enemy' only the other "
        "side's.",
    ),
    "pieces.moves.condition": ("only when", None),
    "pieces.moves.condition.first_move": ("only on its first move", None),
    "pieces.moves.condition.region": ("only from these squares", None),
    "setup.placements.facing": (
        "pointing",
        "Which way these pieces start pointing, for a piece that points a way. Counted from "
        "their own player's forward, so a mirrored start gives both armies the same "
        "directions. Leave it empty and they start the way the piece itself says.",
    ),
    # The rule cards' fields.
    "rules.win": (
        "how the game is won",
        "checkmate: trap the royal piece. royal capture: take it, with no obligation to "
        "answer a threat. annihilation: clear the board of an army. none: the game is "
        "decided only by the endings stated below.",
    ),
    "rules.checkmate": (
        "what being checkmated does",
        "The result for the player who is checkmated. 'win' turns the game inside out, as "
        "losers' chess does.",
    ),
    "rules.bare_royal": (
        "a player left with royalty alone",
        "The result for a player left with only their royal pieces: 'loss' is shatranj's "
        "rule.",
    ),
    "rules.promotion.piece": ("the piece that promotes", None),
    "rules.promotion.rank": (
        "on row",
        "'last', or a row counted from the first player's side.",
    ),
    "rules.promotion.ranks": ("on rows", "A promotion zone several rows deep."),
    "rules.promotion.squares": (
        "on squares",
        "A promotion zone of named squares, for a zone that is a shape. The only way to "
        "give one on hexagons.",
    ),
    "rules.promotion.mandatory": (
        "must promote",
        "No lets a piece decline to promote, as in shogi.",
    ),
    "rules.promotion.on_leaving": (
        "also on leaving the zone",
        "Yes also promotes a piece on a move that starts in the zone, wherever it lands, as "
        "in shogi: a silver standing in the zone may promote as it steps back out. No, the "
        "usual choice, promotes only on a move that ends in the zone.",
    ),
    "rules.promotion.only_lost": (
        "only into what was lost",
        "Yes offers a piece only while its side has fewer of it on the board than it "
        "started with, as in Grand chess: a pawn promotes only into a piece its side has "
        "lost. No, the usual choice, offers every piece listed.",
    ),
    "rules.promotion.to": ("it may become", None),
    "rules.promotion.to_by_player": ("it may become, for each player", None),
    "rules.castling.king": ("the piece that castles", None),
    "rules.castling.rook": ("the piece it castles with", None),
    "rules.castling.king_steps": ("how far it steps", None),
    "rules.castling.king_file": ("where it lands", None),
    "rules.castling.king_file.kingside": ("on the king's side", None),
    "rules.castling.king_file.queenside": ("on the queen's side", None),
    "rules.castling.rook_file": ("where the other piece lands", None),
    "rules.castling.rook_file.kingside": ("on the king's side", None),
    "rules.castling.rook_file.queenside": ("on the queen's side", None),
    "rules.territory.mode": (
        "how a claim works",
        "Moving onto a square claims it. 'permanent' means a claim can never be taken back; "
        "'repaint' lets the other player claim it later.",
    ),
    "rules.territory.win_at": (
        "claiming this many wins",
        "Claiming this many squares wins outright. 0 leaves the claiming to another rule, "
        "such as 'who wins when a clock runs out' comparing the ground each side holds.",
    ),
    "rules.turn.moves": ("moves a turn", None),
    "rules.turn.ramp": (
        "the opening turns",
        "How many moves each of the first turns has, in order, before 'moves a turn' takes "
        "over. With two moves a turn whoever starts is a whole move ahead forever; two "
        "moves a turn with an opening turn of one levels it.",
    ),
    "rules.obstacles.squares": ("walled squares", None),
    "rules.obstacles.mirror": (
        "mirrored",
        "Also wall the mirror image of every square listed: a quick way to write a "
        "symmetric wall. A wall belongs to the board, not to a player. No, the usual choice, "
        "walls only the squares listed.",
    ),
    "rules.obstacles.blocks_leaps": (
        "stops jumps too",
        "Whether a wall also stops a jumping piece. No, the usual choice, lets a knight "
        "jump a wall; yes stops it, on the square the xiangqi horse is blocked on, one step "
        "along the longer side of its jump.",
    ),
    "rules.strike.effect": (
        "what happens to it",
        "What happens to a piece that is taken. 'capture' removes it, as in the ordinary "
        "game. 'demote' and 'promote' turn it into another piece instead, following the "
        "list of what each becomes.",
    ),
    "rules.strike.chain": ("what each piece becomes", None),
    "rules.strike.chain.from": ("the piece taken", None),
    "rules.strike.chain.to": ("becomes", None),
    "rules.strike.position": (
        "where they end up",
        "Where the two pieces end up. 'advance' is the ordinary capture, the taker moving "
        "onto the square, and works only when nothing taken survives, since there would be "
        "nowhere to put it. 'stay' leaves the taker where it was and changes the other "
        "where it stands; with 'capture' the piece taken is simply gone and the taker never "
        "moves. 'swap' exchanges the two. 'choose' moves the taker in and lets it put the "
        "piece it changed on one of the landing squares. 'home' moves the taker in and puts "
        "the piece taken back on its starting square, or removes it if that is occupied.",
    ),
    "rules.strike.landing": (
        "where it may put the piece",
        "Only with 'choose', and needed there: the squares the taker may put a surviving "
        "piece on, as steps from where it stood, forward being the taker's. A piece with "
        "no empty landing square cannot be taken.",
    ),
    "rules.strike.landing.vectors": ("steps", None),
    "rules.strike.landing.symmetry": ("turned", "Which turns of each step it also gets, as on a way of moving."),
    "rules.strike.immune": ("pieces it cannot reach", None),
    "rules.strike.switches_sides": (
        "pieces that change sides",
        "A piece listed here that takes something becomes the other player's piece on the "
        "square it took. A king may not be listed.",
    ),
    "rules.contacts.links": ("what each pair becomes", None),
    "rules.contacts.links.mover": ("the piece taking", None),
    "rules.contacts.links.onto": ("the piece taken", None),
    "rules.contacts.links.mover_to": ("the taker becomes", None),
    "rules.contacts.links.onto_to": ("the piece taken becomes", None),
    "rules.contacts.links.own": (
        "onto its own side",
        "Yes and the pair is about the mover's own piece: it may move onto a friendly "
        "piece along the moves it takes with, and the pair says what both become. Leave it "
        "empty for a piece of the other side.",
    ),
    "rules.contacts.position": (
        "where they end up",
        "Where the two pieces end up. 'advance' is the ordinary capture, the mover taking "
        "the square, and works only when no pair leaves the other piece standing, since "
        "there would be nowhere to put it. 'stay' leaves the mover where it was and changes "
        "the other where it stands. 'swap' exchanges the two. 'choose' moves the mover in "
        "and lets it put the other piece on one of the landing squares.",
    ),
    "rules.contacts.landing": (
        "where it may put the piece",
        "Only with 'choose', and needed there: the squares the mover may put the other "
        "piece on, as steps from where it stood, forward being the mover's. A piece with "
        "no empty landing square cannot be taken.",
    ),
    "rules.contacts.landing.vectors": ("steps", None),
    "rules.contacts.landing.symmetry": ("turned", "Which turns of each step it also gets, as on a way of moving."),
    "rules.contacts.immune": ("pieces it cannot reach", None),
    "rules.wrap.files": (
        "the left and right edges",
        "Whether a piece may run off a side edge and come back in on the other. 'both' "
        "makes the board a cylinder. The edges are named as the board is drawn, the first "
        "player at the bottom.",
    ),
    "rules.wrap.ranks": (
        "the top and bottom edges",
        "The same for the top and bottom edges. Both together make a ring-shaped board "
        "with no corners.",
    ),
    "rules.wrap.files_by_player": ("the left and right edges, by player", None),
    "rules.wrap.ranks_by_player": ("the top and bottom edges, by player", None),
    "rules.wrap.n_s": (
        "top and bottom",
        "Whether the hexagon's top and bottom sides are joined. 'both' lets a piece leave "
        "by either and come back in by the other, straight across; one side lets it leave "
        "by that side only.",
    ),
    "rules.wrap.n_s_by_player": ("top and bottom, by player", None),
    "rules.wrap.ne_sw_by_player": ("upper right and lower left, by player", None),
    "rules.wrap.nw_se_by_player": ("lower right and upper left, by player", None),
    "rules.wrap.ne_sw": (
        "upper right and lower left",
        "Whether the hexagon's upper-right and lower-left sides are joined. 'both' lets a "
        "piece leave by either and come back in by the other, straight across; one side "
        "lets it leave by that side only.",
    ),
    "rules.wrap.nw_se": (
        "upper left and lower right",
        "Whether the hexagon's upper-left and lower-right sides are joined. 'both' lets a "
        "piece leave by either and come back in by the other, straight across; one side "
        "lets it leave by that side only.",
    ),
    "rules.npcs.piece": (
        "which piece",
        "One of the game's pieces, not a royal one. The wild piece has its name, drawing "
        "and worth, and takes what that piece may take; how that piece moves for a player "
        "is not how the wild one walks.",
    ),
    "rules.npcs.route": (
        "its walk",
        "The squares it walks, in order and round again, starting on the first. Different "
        "squares, on no other wild piece's walk (for a second one on this walk, use 'more "
        "on this walk'), and none of the starting pieces on the first. One square is a "
        "wild piece that never moves.",
    ),
    "rules.npcs.also_at": (
        "more on this walk",
        "More wild pieces of the same kind on this same walk: the squares, other than the "
        "walk's first, where each further one starts, in the walk's order. They walk it "
        "in step, and one that finds another in its way waits behind it. Leave it empty "
        "for one wild piece on the walk.",
    ),
    "rules.npcs.bounty": (
        "worth taking",
        "What a player gains by taking it, counted when 'who wins when a clock runs out' "
        "compares points. Its piece's worth when left empty.",
    ),
    "rules.npcs.mirror": (
        "and a mirrored one",
        "Also put a second wild piece on the mirror image of the walk, walking it in the "
        "same order. A walk has a direction, so one wild piece makes the game lopsided and "
        "a mirrored pair does not. No, the usual choice, is one wild piece.",
    ),
    "rules.npcs.immune": (
        "only taken by",
        "The only pieces that may take this wild piece; no other may. Leave it empty and "
        "anything may take it. Another wild piece is not bound by it. Not with pieces "
        "played back onto the board, because the piece would carry the protection out of "
        "a player's hand.",
    ),
    "rules.portals": (
        "portals",
        "Pairs of portals, numbered by their place in the list. A piece whose move ends on "
        "a portal open to it comes out of the other portal of the pair, one step beyond it "
        "in the direction it was going, and its move ends there: it captures, promotes or "
        "wins there as on any square. If that square is off the board, a wall, its own "
        "piece, a piece it may not take, another portal or the square it started from, "
        "there is no such move. A portal is never captured.",
    ),
    "rules.portals.moves": (
        "moves",
        "How a portal of this group moves when a player moves it, written as a piece's "
        "moves are. Every one goes only onto an empty square and is the same seen from "
        "either side. Leave them out when nobody moves the group.",
    ),
    # Q838c: a pair's two ends are its *first* and *second*, never a colour. The file's keys
    # and its one-way passage values keep their names, since renaming them would rehash
    # every portal game; the words go from the page.
    "rules.portals.orange": (
        "first end on",
        "Where the pair's first end starts: an empty square, not a wall.",
    ),
    "rules.portals.blue": (
        "second end on",
        "Where the pair's second end starts: an empty square, not a wall.",
    ),
    "rules.portals.moved_by": ("who may move each end", None),
    "rules.portals.moved_by.orange": (
        "the first end",
        "Which players may spend a turn moving the pair's first end: nobody, the usual "
        "choice, either one of them, or both.",
    ),
    "rules.portals.moved_by.blue": (
        "the second end",
        "Which players may spend a turn moving the pair's second end: nobody, the usual "
        "choice, either one of them, or both.",
    ),
    "rules.portals.passage": ("whose pieces may go in at which end", None),
    "rules.portals.passage.0": (
        "the first player's pieces",
        "both ways: in at either end, the usual choice. One way: in at the pair's first "
        "end only, or only at its second. closed: neither, and both ends are solid blocks "
        "to these pieces.",
    ),
    "rules.portals.passage.1": (
        "the second player's pieces",
        "both ways: in at either end, the usual choice. One way: in at the pair's first "
        "end only, or only at its second. closed: neither, and both ends are solid blocks "
        "to these pieces.",
    ),
    "rules.portals.no_straight_back": (
        "no moving a portal straight back",
        "When on, a portal that has just been moved may not be moved straight back where it "
        "came from on the very next move: it can go anywhere else, and back the move after. "
        "It stops two players undoing each other's portal moves until the game is drawn.",
    ),
    # Schema 21 (Q843a): which pieces may go in, and a map of ends in place of a pair.
    "rules.portals.carries": (
        "which pieces may go in",
        "Which pieces may go in at each end of the pair; an end with no list lets every "
        "piece in.",
    ),
    "rules.portals.carries.orange": (
        "at the first end",
        "The only pieces that may go in at the pair's first end; every other piece meets it "
        "as a solid block. Leave it out and every piece may.",
    ),
    "rules.portals.carries.blue": (
        "at the second end",
        "The only pieces that may go in at the pair's second end; every other piece meets "
        "it as a solid block. Leave it out and every piece may.",
    ),
    "rules.portals.ends": (
        "portals and where each leads",
        "Three to eight portals, each leading to another of them or nowhere: a ring, where "
        "each leads to the next, or a funnel, where several lead to one way out. Two "
        "portals that lead to each other are a pair.",
    ),
    "rules.portals.ends.at": ("on", "Where this portal starts: an empty square, not a wall."),
    "rules.portals.ends.to": (
        "leads to",
        "The portal a piece that goes in here comes out beside, counted from 0 in this "
        "list. Leave it out for a way out only, which nobody goes in by.",
    ),
    "rules.portals.ends.enter": (
        "whose pieces may go in",
        "Which players' pieces may go in here: both, the usual choice, or one of them.",
    ),
    "rules.portals.ends.moved_by": (
        "who may move it",
        "Which players may spend a turn moving this portal: nobody, the usual choice, "
        "either one of them, or both.",
    ),
    # Schema 22 (Q843c): the second rule set.
    # Schema 26 (Q918a): simultaneous moves.
    "rules.simultaneous": (
        "playing in rounds: both players move at once",
        "Each round both players choose a move without seeing the other's, and both are made "
        "together. Only two pieces moving onto the same square collide, and who each piece "
        "beats decides who stays. A piece whose target ran lands where it stood if its move "
        "could end on an empty square, and otherwise stays where it was. There is no check, "
        "so the game is won by taking the royal piece, by clearing an army, or by its other "
        "endings, and an ending both players reach in one round is a draw.",
    ),
    "rules.simultaneous.tie": (
        "when both beat or neither does",
        "Leave it empty and both pieces go. The initiative: one player holds it, the second "
        "at the start; a tie goes to that player's piece, and the initiative passes on. "
        "Impossible: a promise, checked, that no two pieces that can meet ever tie, because "
        "the two armies are different pieces in a ring of who beats whom.",
    ),
    # Schema 27 (Q942a): the switches against a royal chased round after round (the third,
    # the round's rest, is schema 29's `pieces.rest`).
    "rules.simultaneous.swap": (
        "two pieces swapping squares",
        "Collision: two pieces that each move onto the other's square meet as if they had "
        "moved onto the same one, and who each beats decides. Left empty, they pass.",
    ),
    "rules.simultaneous.both": (
        "an ending both reach at once",
        "Score: the player ahead by the scoring rule wins it, and equal scores draw. Left "
        "empty, it is a draw.",
    ),
    "pieces.beats": (
        "in a collision beats",
        "Only when both players move at once: the pieces this one knocks out when both move "
        "onto the same square. Leave it empty and it is the pieces it may take. If one of the "
        "two beats the other it stays; if both do or neither does, it is a tie.",
    ),
    # Schema 27 (Q935a): the spy.
    "pieces.spy": (
        "a spy",
        "While both players have a spy, they take turns. When one loses theirs, they make one "
        "ordinary reply move if it is their turn, and then each round the player without a spy "
        "moves first, in the open, and the other answers knowing it. Once neither has one, "
        "both choose in secret. A spy game must be played in rounds, and a lost spy never "
        "comes back.",
    ),
    # Schema 29 (Q983a): rest, a piece property in any game.
    "pieces.rest": (
        "rests after it moves",
        "How many of its player's turns this piece sits out after it moves, 1 to 4. A resting "
        "piece still stands: it attacks, blocks, gives check and can be taken; it just may not "
        "move, unless nothing else can. In a game in rounds a round is a turn. Leave it empty "
        "and the piece never rests.",
    ),
    # Schema 30 (Q999a, `docs/archive/plan-passive.md` §4.3): lines of sight, a piece's own card.
    "pieces.sight": (
        "what it sees",
        "Lines along which this piece puts what it sees under an effect, for as long as it "
        "stands there: up to four, each drawn like a way of moving. Leave it empty and the "
        "piece sees nothing.",
    ),
    "pieces.sight.vectors": (
        "direction",
        "The directions it looks in, from the piece's own side, as a way of moving's are.",
    ),
    "pieces.sight.symmetry": (
        "turned",
        "How each direction is turned into more, as on a way of moving: all four turns of "
        "one step straight ahead and one diagonal are the eight lines a queen moves along.",
    ),
    "pieces.sight.range": (
        "how far",
        "How many squares along each line it sees; 0 is as far as the board goes.",
    ),
    "pieces.sight.min_range": (
        "no nearer than",
        "The nearest square it sees. A piece on a nearer square is not seen, but still stops "
        "the line.",
    ),
    "pieces.sight.effect": (
        "what it does",
        "Freeze: the piece cannot move, and threatens nothing, so it gives no check. Disarm: "
        "it may move but cannot capture, and threatens nothing. Shield: nothing may capture "
        "it, and a shielded royal is never in check. Command: the other player may also move "
        "it, as its owner would, onto an empty square; a royal is never commanded, and a frozen "
        "piece is moved by nobody.",
    ),
    "pieces.sight.whose": (
        "to whose pieces",
        "The other player's, its own player's, or both, counted from this piece's owner.",
    ),
    "pieces.sight.affects": (
        "only these pieces",
        "The only kinds of piece it touches. Leave it empty for every kind.",
    ),
    "pieces.sight.through": (
        "sees through pieces",
        "Yes: it sees past every piece on its lines. Left empty, each line stops at the "
        "first piece, which is seen.",
    ),
    # Generation 31 (Q1007c, `docs/plan-schema-31.md` §3): a command's turncoat capture, a box
    # beside *what it does* shown only while the effect is a command (Q1007d; worded by Q1013).
    "pieces.sight.captures_own": (
        "may make them take their own side",
        "Only beside a command. Yes: the other player may also make a commanded piece take a "
        "piece of its own side that it may capture, never a royal or a shielded one, and not "
        "while it is disarmed. Left empty, a commanded piece moves only onto empty squares.",
    ),
    # The four effects' names (the fifth name a sight carries, `pieces.sight.through`, is
    # the field above, by the same words).
    "pieces.sight.freeze": ("freezes what it sees", None),
    "pieces.sight.disarm": ("disarms what it sees", None),
    "pieces.sight.shield": ("shields what it sees", None),
    "pieces.sight.command": ("commands what it sees", None),
    # Schema 28 (Q943a): speed.
    "pieces.speed": (
        "speed",
        "Only when both players move at once: a higher speed moves first. When the two "
        "moving pieces differ in speed, the faster move is made first and the slower one "
        "after it, on the board the faster left, if it still can be; otherwise it is lost or "
        "stays put. Equal speeds move together. Leave it empty for the slowest, 0.",
    ),
    "rules.portals.kind": (
        "how it works",
        "Fold portals, where momentum portals are the usual kind: a piece may stand in a "
        "fold, and a move that ends on an empty fold may stay there or come out on the fold "
        "it leads to, taking a piece there as any move does.",
    ),
    "rules.portals.ends.carries": (
        "which pieces may go in",
        "The only pieces that may go in here; every other piece meets it as a solid block. "
        "Leave it out and every piece may.",
    ),
    "rules.promote_on_entry": (
        "promoting on arrival",
        "A piece that comes into play inside its own promotion zone, played back there or "
        "left standing there by a capture that turned it into a promoting piece, promotes "
        "at once, to the first piece its promotion lists, even where promotion is "
        "otherwise a choice. Needs promotion and a way in: pieces played back onto the "
        "board, or a capture that leaves a piece standing. Refused where no promoting "
        "piece could ever come into its zone. No, the usual choice, leaves such a piece "
        "as it arrived.",
    ),
    "rules.scoring.count": (
        "what is compared",
        "What to compare when the turn limit or the no-progress rule ends the game, "
        "instead of calling it a draw. 'material' adds up the pieces' worth, 'pieces' "
        "counts them, 'region' counts your pieces standing on the squares below, "
        "'territory' counts ground claimed and needs claiming ground, 'points' counts what "
        "taking wild pieces earned and needs wild pieces. Higher wins; equal is still a draw.",
    ),
    "rules.scoring.region": (
        "the squares held",
        "The ground being held, for 'region'. Written for the first player and mirrored, so "
        "holding the centre is the same for both and holding the other side's half is not.",
    ),
    "rules.reach.piece": ("which piece", "Leave it empty for any piece."),
    "rules.reach.squares": ("the squares", None),
    "rules.reach.mirror": (
        "for the other player",
        "Yes aims each side at the other's end, mirrored; 'rotate' turns the squares round "
        "the board's centre instead; no sends both players to the same squares.",
    ),
    "rules.reach.safe": ("only where it cannot be taken", None),
    "rules.reach.result": ("the result", None),
    "rules.reach.on": (
        "arriving or leaving",
        "'arrive', the usual choice, decides the game when the piece stands on one of the "
        "squares; 'leave' decides it when the piece moves off one, so it must get there and "
        "then step away, as in Jeson Mor.",
    ),
    "rules.extinction.pieces": ("which pieces", "'any' counts the whole army."),
    "rules.extinction.min_count": ("fewer than", None),
    "rules.extinction.result": ("the result", None),
    "rules.extinction.per_type": (
        "each kind on its own",
        "Yes counts each listed piece separately, so a player loses when any one kind "
        "runs out, as in Extinction chess. 'own' does the same, but counts a kind only "
        "against a player who starts with it, for two armies made of different pieces. "
        "No, the usual choice, counts them together.",
    ),
    "rules.drops.from_captures": ("captured pieces", None),
    "rules.drops.region": ("where each may be played", "For each piece, the squares it may be played back on."),
    "rules.drops.no_doubled": ("not onto a column holding one", None),
    "rules.drops.no_mate": ("not checkmating as it is played", None),
    "rules.drops.placement": (
        "place everything first",
        "Yes means a player who holds any piece must play one onto the board instead of "
        "moving, so the pieces in hand are all placed before the game proper starts, as in "
        "Placement chess. A king may then start in hand. No, the usual choice, lets a held "
        "piece wait.",
    ),
    "rules.drops.either_face": (
        "either face",
        "Yes lets a piece in hand be played as itself or as the piece that reverts to it "
        "when captured, so a captured dragon, held as a rook, may come back as either, as "
        "in Kyoto shogi. No, the usual choice, plays it back as itself.",
    ),
    "rules.drops.stock": (
        "held at the start",
        "For each piece, how many each player holds before anything is taken: a count of 1 "
        "to 32, or 'unlimited' for a supply that never runs out. A held piece is the only "
        "one that may have no way of moving.",
    ),
    "rules.explode_on_capture.radius": ("how far it reaches", None),
    "rules.explode_on_capture.immune": ("pieces that survive it", None),
    "rules.explode_on_capture.by": ("only when taken by", None),
    "rules.explode_on_capture.of": ("only when taking", None),
}

# The rule cards themselves, by the titles their cards already carry.
for _key, (_title, _) in RULE_WORDS.items():
    FIELD_WORDS.setdefault(f"rules.{_key}", (_title, None))

# A switch between two ways of writing one field, by the kind of thing each holds (D40).
KIND_WORDS = {
    "integer": "a number",
    "number": "a number",
    "string": "a word",
    "boolean": "yes or no",
}


# A sentence for each value a menu showed bare (Q995c, fix 9, F7): *initiative*, *impossible*,
# *own*, *collision* and *score* were the schema's spellings, explained only in the paragraph
# under the menu. Each option now has a label that reads as a choice, and the sentence the
# win menu's options carry. Keyed by the field's dotted path and the value. Built in
# `web/cards.py` by Q995c, moved here by Q995e so the command line and the model read them too.
OPTION_WORDS: dict[tuple[str, Any], tuple[str, str]] = {
    ("rules.simultaneous.tie", "initiative"): (
        "the player with the initiative",
        "One player holds the initiative, the second player at the start. A tie goes to "
        "that player's piece, and the initiative passes to the other.",
    ),
    ("rules.simultaneous.tie", "impossible"): (
        "it can't happen here",
        "A promise, checked with the rest of the game, that no two pieces that can meet ever "
        "tie, because the two armies are different pieces in a ring of who beats whom.",
    ),
    ("rules.simultaneous.swap", "collision"): (
        "they meet, as on one square",
        "Two pieces that each move onto the other's square meet as if they had moved onto "
        "the same one, and who each beats decides.",
    ),
    ("rules.simultaneous.both", "score"): (
        "the scores decide it",
        "The player ahead by the rule 'who wins when a clock runs out' wins it, and equal "
        "scores draw.",
    ),
    ("rules.extinction.per_type", True): (
        "yes",
        "Each listed piece is counted on its own, so a player loses when any one kind runs "
        "out, as in Extinction chess.",
    ),
    ("rules.extinction.per_type", False): (
        "no",
        "The listed pieces are counted together, the usual choice.",
    ),
    ("rules.extinction.per_type", "own"): (
        "yes, for the kinds a player starts with",
        "Each listed piece is counted on its own, but only against a player who starts with "
        "it, for two armies made of different pieces.",
    ),
}


def field_words(path: str) -> tuple[str, str | None] | None:
    """`FIELD_WORDS` for a dotted path that may carry list marks (`[]`, `[3]`)."""
    return FIELD_WORDS.get(re.sub(r"\[\d*\]", "", path))


# ------------------------------------------------------------------ problems, translated

# A place in the game's text: a top-level part, then `.key` or `[n]` steps. Only taken
# where it is plainly a path (at the start of a problem, or holding a `.` or `[`), so the
# word *name* in a sentence is never read as the game's name.
_PATH = re.compile(
    r"\b(?:spec|name|board|pieces|setup|rules|appearance|provenance)"
    r"(?:\[\d*\]|\['[^']*'\]|\.[A-Za-z_0-9]+)*"
)

# The validator's own vocabulary, in a visitor's (§3.2). Applied after the paths.
_WORDS: tuple[tuple[str, str], ...] = (
    (r"\bmust be a non-empty string\b", "can't be empty"),
    (r"\bmust be a boolean\b", "must be yes or no"),
    (r"\bmust be a non-negative integer\b", "must be a whole number, 0 or more"),
    (r"\bmust be a positive integer\b", "must be a whole number, 1 or more"),
    (r"\bmust be an integer\b", "must be a whole number"),
    (r"\bmust be a number of 0 or more\b", "must be a number, 0 or more"),
    (r"\bmust be a non-empty array of algebraic squares\b", "needs at least one square"),
    (r"\bmust be a non-empty array of piece ids\b", "needs at least one piece"),
    (r"\bmust be a non-empty array of \[dx, ?dy\] pairs\b", "needs at least one direction"),
    (r"\bmust be a non-empty array of ranks\b", "needs at least one row"),
    (r"\bmust be a non-empty array\b", "needs at least one"),
    (r"\bone of \(([^()]*)\)", r"one of \1"),
    (r"\ba non-empty array of algebraic squares\b", "at least one square"),
    (r"\ba non-empty array of piece ids\b", "at least one piece"),
    (r"\ba non-empty array of \[dx, ?dy\] pairs\b", "at least one direction"),
    (r"\ba non-empty array\b", "at least one entry"),
    (r"\ban array of piece ids\b", "a list of pieces"),
    (r"\[dx, ?dy\] integer pairs", "pairs of whole numbers, sideways then forward"),
    (r"\[dx, ?dy\] steps", "steps, sideways then forward"),
    (r"\bmust be an object\b", "is written wrongly"),
    (r"\bis not a declared piece id\b", "isn't one of the game's pieces"),
    (r"\bis not a declared piece\b", "isn't one of the game's pieces"),
    (r"\bmust be a piece id\b", "must be one of the game's pieces"),
    (r"\bpiece ids\b", "pieces"),
    (r"\bpiece id\b", "piece"),
    (r"\bis not a square on this board\b", "isn't a square on this board"),
    (r"\bis not a rank on this board\b", "isn't a row on this board"),
    (r"\bis duplicated\b", "is used by more than one piece"),
    (r"\bvectors\b", "directions"),
    (r"\bvector\b", "direction"),
    (r"\bplies\b", "turns"),
    (r"\bply\b", "turn"),
    (r"\bthe validator\b", "the designer"),
    (r"\bmust be 0 or 1\b", "must be the first player (0) or the second (1)"),
    (r"\bare not cells of this board\b", "aren't cells of this board"),
    (r", got\b", ", not"),
    # Q1013: the one sight field the validator names bare, outside a path (a turncoat capture
    # beside a capture hand, or one nothing could make), by its box's label; the validator's
    # gloss of the key beside it says the label again, so it goes.
    (r"\bcaptures_own\b(?: \([^()]*\))?", "a command's 'may make them take their own side'"),
)

# *The spec* is *the rules* to a visitor, and the rules are plural where the spec was
# singular, so the verb after it, and a pronoun or possessive that points back at it, turns
# with it (LP0: *the rules states no ending of its own* was the designer's sentence).
_THE_SPEC = re.compile(
    r"\bthe spec(?P<own>'s)?(?: (?P<self>itself)\b)?"
    r"(?: (?P<adverb>already|also|alone|never|now|only|still|then))?"
    r"(?: (?P<word>[a-z]+)\b)?"
)
_PLURAL_VERBS = {"is": "are", "was": "were", "has": "have", "does": "do", "goes": "go",
                 "doesn't": "don't",
                 "isn't": "aren't", "wasn't": "weren't", "hasn't": "haven't"}
# Words ending in *s* that follow a subject without being its verb.
_NOT_VERBS = frozenset({"as", "its", "this", "thus", "plus", "unless", "us", "less"})


def _plural_verb(word: str) -> str:
    """A third-person singular verb as the plural's: *states* → *state*, *carries* →
    *carry*, *passes* → *pass*, *is* → *are*. Anything else, unchanged."""
    if word in _PLURAL_VERBS:
        return _PLURAL_VERBS[word]
    if word in _NOT_VERBS or not word.endswith("s") or word.endswith("ss") or len(word) < 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if re.search(r"(?:ss|sh|ch|x|z)es$", word):
        return word[:-2]
    return word[:-1]


def the_rules(text: str) -> str:
    """*The spec* said as *the rules*, with what agrees with it made plural: *the spec
    states* is *the rules state*, *the spec's* is *the rules'*, *the spec itself* is *the
    rules themselves*. A sentence's meaning is untouched; only its grammar follows."""

    def said(match: re.Match[str]) -> str:
        out = "the rules'" if match["own"] else "the rules"
        if match["self"]:
            out += " themselves"
        if match["adverb"]:
            out += " " + match["adverb"]
        if match["word"]:
            # After a possessive the next word is the thing owned, never a verb.
            out += " " + (match["word"] if match["own"] else _plural_verb(match["word"]))
        return out

    text = _THE_SPEC.sub(said, text)
    # *of its own* a few words on in the same clause points back at the rules too.
    return re.sub(r"\b(the rules (?:[a-z']+ ){1,4}?of )its own\b", r"\1their own", text)


# The two clocks, counted in turns (§3.2: each player's move is one turn).
_CLOCK_PATHS = frozenset({"rules.max_plies", "rules.no_progress_draw"})

# In a game played in rounds the clocks still count each player's move, so a round is two
# (Q995e; the rule's own sentence for rounds is `describe.ROUND_WORDS`).
ROUND_CLOCK_NOTE = " (a round is two)"

# Whole problems a person meets most, said the way the audit asked (D48-D50).
_PHRASED: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^name must be a non-empty string$"), "Give the game a name."),
    (re.compile(r"^spec is missing required key 'name'$"), "Give the game a name."),
    (re.compile(r"^no royal piece appears in setup\.placements$"),
     "No royal piece is on the board at the start."),
    (re.compile(r"^spec must be a JSON object$"), "These rules couldn't be read."),
)


# Q995d (fix 7 of `docs/reference/designer-coverage.md`): what the designer says about rest
# across the whole army. A rest is a piece's own (schema 29), and every game that rests
# today rests every piece the same, so the designer sets every piece at once beside *add a
# piece* and keeps each card's field for the exceptions. The page fills these in: `{rest}`
# is a count in `units` (a turn, or a round in a game played in rounds), `{names}` a list of
# pieces joined with `and`. The server's sentences about a choice are here too
# (`designing.rest_every_piece`).
REST_WORDS: dict[str, Any] = {
    # The control, read as one sentence: *every piece rests [1 turn] after it moves*.
    "label": "every piece rests",
    "after": "after it moves",
    "never": "never",
    # The menu while the pieces differ: a line, not a choice.
    "differ": "each its own",
    "units": {"turns": ["turn", "turns"], "rounds": ["round", "rounds"]},
    "and": "and",
    # The card's head, beside a piece's speed.
    "head": "rests {rest}",
    # The army, said under the control.
    "every": "Every piece rests {rest} after it moves: it sits that out before it moves again.",
    "none": {
        "turns": "No piece rests: any piece may move on each of its player's turns.",
        "rounds": "No piece rests: any piece may move in every round.",
    },
    "each": "Each piece rests after it moves: {groups}.",
    "group": "{rest} for the {names}",
    "groupNever": "never for the {names}",
    "override": "A piece's own rest is on its card, under 'how it takes its turns'.",
    # What a choice did, or why it was not made (the server's).
    "cleared": "No piece rests now.",
    "set": "Every piece rests {rest} now.",
    "setSome": "Every piece that can rest now rests {rest}.",
    "left": "Left without a rest: {reason}",
    "refused": "'{choice}' can't be chosen for this game yet, so nothing was changed.",
}


def _where(path: str, spec: Any) -> tuple[str, str | None]:
    """A path as the place a person finds it, and the field there by its label.

    ``pieces[1].moves[0].range`` is (*Piece 2 (knight), way of moving 1*, *how far*);
    ``setup.placements[0].squares`` is (*Where they start, row 1*, *squares*);
    ``rules.max_plies`` is (*The rule 'the longest a game may run'*, None).
    """
    steps = re.findall(r"\['[^']*'\]|[A-Za-z_0-9]+|\[\d*\]", path)
    head, rest = steps[0], steps[1:]

    def index(step: str) -> int | None:
        inner = step[1:-1]
        if inner.startswith("'"):
            return None
        return int(inner) if inner.isdigit() else None

    if head == "spec":
        return "The game", None
    if head == "pieces" and rest and index(rest[0]) is not None:
        number = index(rest[0])
        assert number is not None
        pieces = spec.get("pieces") if isinstance(spec, dict) else None
        piece = pieces[number] if isinstance(pieces, list) and number < len(pieces) else None
        called = piece.get("name") if isinstance(piece, dict) else None
        place = f"Piece {number + 1}" + (f" ({called})" if isinstance(called, str)
                                         and called.strip() else "")
        rest = rest[1:]
        dotted = "pieces"
        if len(rest) >= 2 and rest[0] == "moves" and index(rest[1]) is not None:
            moved = index(rest[1])
            assert moved is not None
            place += f", way of moving {moved + 1}"
            rest = rest[2:]
            dotted = "pieces.moves"
        elif len(rest) >= 2 and rest[0] == "sight" and index(rest[1]) is not None:
            # Schema 30 (Q999a): one line of sight, numbered as a way of moving is.
            seen = index(rest[1])
            assert seen is not None
            place += f", line of sight {seen + 1}"
            rest = rest[2:]
            dotted = "pieces.sight"
            if not rest:
                return place, None
        return place, _label(dotted, rest)
    if head == "pieces" and rest and (rest[0] == "[]" or not rest[0].startswith("[")):
        # A field of the whole army (`pieces[].speed`, `pieces.speed`): the pieces that
        # carry it, or every piece when none does yet.
        keys = [step for step in rest if not step.startswith("[")]
        if len(keys) == 1:
            carriers = _carrying(spec, keys[0])
            place = _pieces_called(spec, carriers) if carriers else "Every piece"
            return place, _label("pieces", rest)
    if head == "setup" and len(rest) >= 2 and rest[0] == "placements" \
            and index(rest[1]) is not None:
        row = index(rest[1])
        assert row is not None
        return f"Where they start, row {row + 1}", _label("setup.placements", rest[2:])
    if head == "rules" and not rest:
        return "The rules", None
    if head == "rules":
        key = rest[0]
        title = RULE_WORDS.get(key, (key.replace("_", " "), ""))[0]
        place = f"The rule '{title}'"
        dotted = f"rules.{key}"
        tail: list[str] = []
        keyed = ""
        for step in rest[1:]:
            number = index(step) if step.startswith("[") else None
            if step.startswith("['"):
                # One piece's entry in a per-piece table: *for P*.
                keyed = f", for {step[2:-2]}"
                continue
            if step.startswith("["):
                if number is not None and tail == []:
                    # One of several: *promotion 2*, *wild piece 1*.
                    place += f", number {number + 1}"
                elif number is not None:
                    tail.append(f"{number + 1}")
                continue
            dotted += f".{step}"
            tail.append(step)
        if not tail:
            return place + keyed, None
        words = FIELD_WORDS.get(dotted)
        label = words[0] if words else " ".join(t.replace("_", " ") for t in tail)
        return place, label + keyed
    if head == "setup":
        if not rest or rest == ["placements"]:
            return "Where they start", None
        return "Where they start", _label("setup", rest)
    if head == "board":
        return "The board", _label("board", rest) if rest else None
    if head == "name":
        return "The game's name", None
    if head == "appearance":
        return "The pieces' drawings", None
    return "Where the game comes from", None


def _label(dotted: str, rest: list[str]) -> str | None:
    keys = [step for step in rest if not step.startswith("[")]
    if not keys:
        return None
    words = FIELD_WORDS.get(dotted + "." + ".".join(keys))
    return words[0] if words else " ".join(key.replace("_", " ") for key in keys)


def _place(path: str, spec: Any) -> str:
    where, field = _where(path, spec)
    return f"{where}, '{field}'" if field else where


# ------------------------------------------------------------------ a field of the whole army

# The validator checks some piece fields across the army at once and names them with no
# index (`pieces[].speed`). `_where` read a path
# into `pieces` only by its index, so these five sentences, about the newest fields, opened
# *Where the game comes from* and named neither the piece nor the field (Q995a F3). Now the
# pieces that carry the field are named from the draft (built in `web/designing.py` by
# Q995b, moved here by Q995e so the command line and the model's repairs read it too).
_ARMY_PATH = re.compile(
    r"\bpieces(?:\[\])?\.(?P<key>[a-z_]+)\b"
    # Schema 30 (Q999a): a sight's name carries its effect, or `through`, one step down.
    r"(?:\.(?P<sub>" + "|".join((*SIGHT_EFFECTS, "through")) + r")\b)?"
)

# The three a newcomer meets most: a field of a game in rounds set on a game that is not
# in one. Each says where rounds are chosen.
_ROUNDS_ONLY = (
    (re.compile(r"^pieces\[\]\.speed orders the two moves of a round, and this game has none"),
     "speed", "a speed only matters in a game played in rounds", "clear {its} speed"),
    (re.compile(r"^pieces\[\]\.spy: a spy decides when a game plays in rounds"),
     "spy", "a spy only works in a game played in rounds", "take {its} spy away"),
    (re.compile(r"^pieces\[\d+\]\.beats says who wins a collision, and only matters"),
     "beats", "who a piece beats in a collision only matters in a game played in rounds",
     "clear who {it} beats"),
)
_CHOOSE_ROUNDS = "Choose 'in rounds' under 'how the players take turns', or"


def _carrying(spec: Any, key: str, sub: str | None = None) -> list[int]:
    """The pieces that carry a field; for a sight's name (``sub``, schema 30), only those with
    a line of sight of that effect, or one that sees through pieces."""
    pieces = spec.get("pieces") if isinstance(spec, dict) else None
    if not isinstance(pieces, list):
        return []
    carrying = [index for index, piece in enumerate(pieces)
                if isinstance(piece, dict) and key in piece]
    if sub is None:
        return carrying
    sights = piece_sights(spec)
    return [
        index for index in carrying
        if any(
            (sight.through if sub == "through" else sight.effect == sub)
            for sight in sights.get(pieces[index].get("id"), ())
        )
    ]


def _pieces_called(spec: Any, indices: list[int]) -> str:
    """*Piece 2 (knight)*, *Pieces 2 (knight) and 3 (bishop)*: the place `spec.plain`
    gives one piece, for one or several."""
    pieces = spec["pieces"]

    def one(index: int) -> str:
        name = pieces[index].get("name")
        return f"{index + 1}" + (f" ({name})" if isinstance(name, str) and name.strip() else "")

    named = [one(index) for index in indices]
    if len(named) == 1:
        return f"Piece {named[0]}"
    return f"Pieces {', '.join(named[:-1])} and {named[-1]}"


def _army_problem(problem: str, spec: Any) -> str | None:
    """A problem about a field of the army, with the pieces that carry it named."""
    beats = re.match(r"^pieces\[(\d+)\]\.beats", problem)
    for pattern, key, says, way_out in _ROUNDS_ONLY:
        if not pattern.match(problem):
            continue
        carriers = [int(beats[1])] if key == "beats" and beats else _carrying(spec, key)
        if not carriers or not all(index < len(spec["pieces"]) for index in carriers):
            break
        several = len(carriers) > 1
        out = way_out.format(its="their" if several else "its", it="they" if several else "it")
        return f"{_pieces_called(spec, carriers)}: {says}. {_CHOOSE_ROUNDS} {out}."
    if _ARMY_PATH.search(problem) is None:
        return None
    plural: list[tuple[str, str]] = []

    def name(found: re.Match[str]) -> str:
        key, sub = found["key"], found["sub"]
        field = f"{key}.{sub}" if sub else key
        words = field_words(f"pieces.{field}")
        label = words[0] if words else key.replace("_", " ")
        # The path the sentence is about, at its start, is the pieces that carry it; one
        # inside a reason is the field.
        subject = found.start() == 0
        carriers = _carrying(spec, key, sub) if subject else []
        if not carriers:
            return f"every piece's '{label}'" if subject else f"'{label}'"
        if len(carriers) > 1:
            plural.append((_pieces_called(spec, carriers[:1]), _pieces_called(spec, carriers)))
        return f"pieces[{carriers[0]}].{field}"

    said = _translate(_ARMY_PATH.sub(name, problem), spec)
    for alone, every in plural:
        said = said.replace(alone, every).replace(alone.lower(), every.lower())
    return said


def plain_problem(problem: str, spec: Any = None) -> str:
    """One validator problem, in the designer's words. Never drops or adds a problem."""
    # A whole sentence said its own way goes first: the old rest switch's names a piece
    # field, which the army's sentences would otherwise take (Q995d).
    for pattern, said in _PHRASED:
        if pattern.match(problem):
            return said
    army = _army_problem(problem, spec)
    return army if army is not None else _translate(problem, spec)


def _translate(problem: str, spec: Any) -> str:
    for pattern, said in _PHRASED:
        if pattern.match(problem):
            return said
    text = problem
    # Two shapes worth their own sentence (D48, D50).
    square = re.match(r"^(\S+) entry '([^']*)' is not a square on this board$", text)
    if square:
        # A row of *Where they start* is nothing but squares, so the row is the place.
        place = (_where(square.group(1), spec)[0]
                 if re.fullmatch(r"setup\.placements\[\d+\]\.squares", square.group(1))
                 else _place(square.group(1), spec))
        return f"{place}: {square.group(2)} isn't a square on this board."
    leap = re.match(r"^(\S+)\.(range|min_range) means nothing on a leap, "
                    r"which is one jump per vector; to reach further, list more vectors, or "
                    r"ride it with a 'slide' over the vector or a 'lame' with a range$", text)
    if leap:
        where, _ = _where(leap.group(1), spec)
        label = FIELD_WORDS[f"pieces.moves.{leap.group(2)}"][0]
        return (f"{where}: '{label}' does nothing on a leap. "
                "To reach further, add more directions, or let it jump again.")
    lame_step = re.match(
        r"^(\S+)\.vectors\[\d+\] \[-?\d+, -?\d+\] is a one-square 'lame' leap, "
        r"which has no square to be blocked on; use an ordinary 'leap' or a 'slide' of "
        r"range 1 instead$", text)
    if lame_step:
        where, _ = _where(lame_step.group(1), spec)
        return (f"{where}: a one-square blockable leap has no square to be blocked on. "
                "Make it an ordinary leap, or a slide of range 1, instead.")
    taken = re.match(r"^(\S+) '([^']*)' is already occupied by '([^']*)'$", text)
    if taken:
        where, _ = _where(taken.group(1), spec)
        return f"{where}: {taken.group(2)} already has a piece on it ({taken.group(3)})."
    stranger = re.match(r"^(\S+) (?:entry )?'([^']*)' is not a declared piece(?: id)?$", text)
    if stranger:
        # A row of *Where they start* names one piece, so the row is the place.
        place = (_where(stranger.group(1), spec)[0]
                 if re.fullmatch(r"setup\.placements\[\d+\]\.piece", stranger.group(1))
                 else _place(stranger.group(1), spec))
        return f"{place}: '{stranger.group(2)}' isn't one of the game's pieces."
    missing = re.match(r"^(\S+) is missing required key '([^']*)'$", text)
    if missing:
        where, _ = _where(missing.group(1), spec)
        dotted = re.sub(r"\[\d*\]", "", missing.group(1)) + "." + missing.group(2)
        if missing.group(1) == "spec":
            dotted = missing.group(2)
        words = FIELD_WORDS.get(dotted)
        label = words[0] if words else missing.group(2).replace("_", " ")
        return f"{where}: '{label}' is missing."
    unknown = re.match(r"^(\S+) has unknown keys: (.*)$", text)
    if unknown:
        where, _ = _where(unknown.group(1), spec)
        return f"{where}: these aren't parts of a game: {unknown.group(2)}."

    return _generic(text, spec)


def _generic(text: str, spec: Any) -> str:
    """Any other problem: the path it opens with is its place, and any other path inside
    it is named the same way."""
    opening = _PATH.match(text)
    place: str | None = None
    field: str | None = None
    body = text
    if opening and text[opening.end():opening.end() + 1] in (" ", ""):
        place, field = _where(opening.group(0), spec)
        body = text[opening.end():].strip()

    def inner(match: re.Match[str]) -> str:
        token = match.group(0)
        if "." not in token and "[" not in token:
            return token
        named_at, named = _where(token, spec)
        named_at = f"{named_at[0].lower()}{named_at[1:]}"
        return f"{named_at}, '{named}'" if named else named_at

    body = _PATH.sub(inner, body)
    for pattern, said in _WORDS:
        body = re.sub(pattern, said, body)
    body = the_rules(body)
    # A menu's values as the menu spells them: *royal capture*, not *royal_capture*.
    body = re.sub(r"'([a-z0-9]+(?:_[a-z0-9]+)+)'",
                  lambda m: "'" + m.group(1).replace("_", " ") + "'", body)
    if opening and opening.group(0) in _CLOCK_PATHS:
        body = re.sub(r"\b(between \d+ and \d+)",
                      r"\1 turns" + (ROUND_CLOCK_NOTE if is_simultaneous(spec) else ""), body)
    if place is None:
        text = body
    elif re.match(r"needs\b|stands\b", body):
        # *Piece 1, way of moving 1 needs at least one direction.*
        body = re.sub(r"^stands a piece", "a piece stands", body)
        text = f"{place} {body}" if body.startswith("needs") else f"{place}: {body}"
    elif field and re.match(r"['\"\d]|entry ", body):
        # *on row 9 isn't a row on this board*: the label reads into its value.
        text = f"{place}: {field} {body.removeprefix('entry ')}"
    elif field:
        text = f"{place}: '{field}' {body}"
    elif re.match(r"(?:is|must|has|needs|beside|cannot|can't|isn't)\b|['\"]", body):
        text = f"{place} {body}"
    else:
        text = f"{place}: {body}"
    text = " ".join(text.split())
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "?", "!")) else text + "."


def plain_problems(problems: list[str], spec: Any = None) -> list[str]:
    """`plain_problem` over a list, in order and one for one."""
    return [plain_problem(problem, spec) for problem in problems]
