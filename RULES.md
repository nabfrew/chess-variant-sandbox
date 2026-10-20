# The rule language, schema version 29

This is the whole language a game is written in, at schema version 29. A game
is one JSON file. It says how big the board is, what pieces there are and how each moves,
where they start, and how the game is won; nothing about a game lives anywhere else. A
person can write one by hand, or give this file to an AI and ask it for one.

**Which version.** A game written from this reference should declare
`"schema_version": 29`. A site or checkout running a later version still
accepts it, because every version ever cut stays accepted, and a file may use only what
the version it declares had. The site serves the reference for the version it runs at
`/rules.md`; when that copy names a later version than this one, write from that copy.

<!-- Generated from the schema by `uv run chess-variant-sandbox rules-reference --out RULES.md`. Do not edit by hand. -->

## Trying a game

1. Write the file. Start from one of the two worked examples at the end, or from any game
   in `rulesets/`, and change it.
2. Check it. `uv run chess-variant-sandbox validate my-game.json` lists every problem at
   once, in the words shown under *How a wrong file is refused*, or says the game is valid
   and how many moves the first player has.
3. Play it. In the designer, on the website or on your own machine
   (`uv run chess-variant-sandbox serve --open`), press *load a rule file* and pick the file,
   or paste its text into *the game as text*. A file with a mistake is refused in the same
   words as `validate`, and the game you were editing is left as it was; a good one becomes
   your draft, to see on the board, play against the computer and save.

## What the language can make

The reference below goes key by key, in the order a file is written. These are the
same keys by what they do to play. Every family is checked the same way: a file that
asks for two rules the language cannot play together, or for a rule that could never
fire, is refused with a line saying why, so trying an unfamiliar rule costs a round of
`validate`.

- **How a game is won.** The object need not be checkmate. A royal piece can simply be taken, or there can be none; a game can be a race to some squares, a fight to run the other side out of a kind of piece, a count of checks given, ground claimed by moving onto it, or a score read when time runs out. Changing the object changes what every piece is for. (`rules.win`, `rules.checkmate`, `rules.stalemate`, `pieces[].royal`, `rules.bare_royal`, `rules.check_limit`, `rules.reach`, `rules.extinction`, `rules.territory`, `rules.scoring`)
- **How pieces move.** Besides chess's slides and leaps: a hop vaults one screening piece, as the xiangqi cannon captures or a grasshopper lands; a lame leap can be blocked on its way; a sweep clears every piece along its line. A piece can be kept to one part of the board, or point one way and spend a move turning. (`"slide"`, `"leap"`, `"hop"`, `"lame"`, `"sweep"`, `pieces[].region`, `pieces[].facing`)
- **What a capture does.** A piece may be able to take only certain others, so armies can beat one another in a circle. A taken piece can become another piece, be sent home, change places with its taker, blow up with everything round it, or go into the captor's hand to be dropped back into play; a player can start with pieces in hand; captures can be compulsory. (`pieces[].captures`, `rules.strike`, `rules.contacts`, `rules.explode_on_capture`, `rules.drops`, `pieces[].promoted_from`, `rules.promote_on_entry`, `rules.must_capture`)
- **The board.** Hexagons as well as squares; walls that give the board any outline or cut it in two; edges that join, so the board is a cylinder or a torus; portals that carry a piece across the board, which the players may be allowed to move. (`board.lattice`, `board.shape`, `rules.obstacles`, `rules.wrap`, `rules.portals`)
- **Who moves, and when.** Several moves a turn; or both players choosing at once, neither seeing the other's move, with rules for what wins a collision, which pieces are fast enough to move first, and spies whose loss takes a player's secrecy away; and pieces that must rest for some of their player's turns after they move, by turns or in rounds. (`rules.turn`, `rules.simultaneous`, `pieces[].beats`, `pieces[].speed`, `pieces[].spy`, `pieces[].rest`)
- **Pieces nobody owns.** Wild pieces that walk a fixed route, one step after every move, eat what they may, and pay a bounty to whoever takes them. (`rules.npcs`)
- **Chess's own rules.** None is assumed: promotion, en passant and castling, each chess's answer to a problem chess has; and limits on giving check, on the royal pieces facing each other down an open file (xiangqi's rule), and on moves that strand a piece. (`rules.promotion`, `rules.en_passant`, `rules.castling`, `rules.forbid_check`, `rules.royal_facing`, `rules.forbid_immobility`)
- **When a game stops.** How long a game may run, and when a repetition or a stall draws it; `rules.scoring` can decide such an ending instead. (`rules.max_plies`, `rules.repetition_draw`, `rules.no_progress_draw`)

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

## The top level

- **`schema_version`**: integer from 1 to 29. Optional; default 29, the version this reference describes. Which version of this language the file is written in. A file may declare any version from 1 up, but may use only what that version had; declaring the version this reference describes makes every key below available.
- **`name`**: string, at least 1. Required. The game's name, shown wherever the game is listed. It must not be blank.
- **`description`**: string. Optional; default empty. How the game plays and what makes it worth playing, in a few sentences.
- **`board`**: object. Required. The size and shape of the board. See *board* below.
- **`pieces`**: list of objects, 1 to 24 entries. Required. Every kind of piece in the game, each with its own one- or two-character id. The order does not matter. See *pieces: one entry per kind of piece* below.
- **`setup`**: object. Required. Where every piece stands when the game begins. See *setup* below.
- **`rules`**: object. Optional; default `{}`, every rule at its default. Everything about the game beyond the pieces and the board: how it is won, and each extra rule. Every key is optional; an empty object is the defaults below. See *rules* below.

## `board`

The size and shape of the board.

- **`width`**: integer from 2 to 16. Required. How many files (columns) the board has.
- **`height`**: integer from 2 to 16. Required. How many ranks (rows) the board has.
- **`lattice`**: one of `"square"`, `"hex"`. Optional; default `"square"`. 'square', the default, or 'hex'. A hex board is a width by height rhombus of hexagons in axial coordinates, file q and rank r, whose six neighbours are [1,0] [0,1] [-1,1] [-1,0] [0,-1] [1,-1]; cut the shape you want out of it with rules.obstacles, or set shape 'hexagon' for a regular hexagon.
- **`shape`**: one of `"grid"`, `"hexagon"`. Optional; default `"grid"`. 'grid', the default, is every cell of the width by height frame. 'hexagon' needs lattice 'hex' and an odd width equal to the height (11 is six cells a side): the rhombus's corners are not cells at all, no rule may name them, and rules.wrap joins the hexagon's sides with n_s, ne_sw and nw_se.

## `pieces`: one entry per kind of piece

- **`id`**: string, 1 to 2 characters. Required. A short id, one or two characters, unique in the game, that every other part of the file uses to name this piece: `P`, `N`, `Q2`.
- **`name`**: string. Optional; default the piece's id. The piece's name, as players read it.
- **`royal`**: boolean. Optional; default `false`. Losing every royal piece loses the game.
- **`value`**: number from 0 to 100. Optional; default `1`. What the piece is worth, roughly in pawns. Computer players use it as a first guess at how much a piece matters, and `rules.scoring` with count 'material' adds it up. A royal piece is usually 0.
- **`moves`**: list of objects, at most 12 entries. Required. The ways this piece may move, each one a separate entry; the piece may make any one of them on its turn. A piece may have none only when it starts in a player's hand (rules.drops.stock) or is a portal nobody moves. See *pieces[].moves: one entry per way of moving* below.
- **`region`**: list of strings, at least 1. Optional. The only squares this piece may ever occupy, such as a palace or one bank of a river. Written for the first player and reflected. Algebraic squares such as a1 or e4.
- **`promoted_from`**: string. Optional. The piece this one was promoted from. It only matters where captures go to hand, because a promoted piece returns as its modest original.
- **`captures`**: list of strings. Optional; default the piece may capture any piece. The piece ids this one is allowed to capture. Omit it and the piece may capture anything, which is what every ordinary game means. State it and you get a dominance relation. Give rock ['S'], scissors ['P'] and paper ['R'] and no piece can take the one that takes it. An empty list is a piece that never captures.
- **`beats`**: list of strings. Optional. Schema 26. Only in a simultaneous game (rules.simultaneous): the piece ids this one eliminates when the two collide, both moving onto the same square in one round. Omit it and it is the piece's captures, or every piece where captures is omitted too; never write that list out. If exactly one of the two beats the other, that one stands on the square and the other is eliminated; if both do or neither does, it is a tie (rules.simultaneous.tie). An empty list loses every collision it does not tie. Never on a portal.
- **`spy`**: boolean. Optional. Schema 27. This piece is a spy, and the game is played in three phases by who still has one on the board. While both players have a spy, they take turns as usual. When one player loses their last spy, they first play one ordinary reply move if it is their turn; then the game is played in rounds (rules.simultaneous, which a spy game must declare) in which the player without a spy announces their move first and the other chooses knowing it, both resolved together. Once neither has a spy, both choose in secret. Both players must start with a spy on the board, nothing may bring one back (no promotion to it, no stock or hand holding it, no drops.from_captures, no strike.switches_sides), it may not be royal, and something of the other player's must be able to remove it. Leave it out for an ordinary piece; never write false.
- **`speed`**: integer, at least 1. Optional. Schema 28. Only in a game played in rounds (rules.simultaneous): this piece's speed, a rank; omit it for 0, the slowest, and never write 0. A higher speed moves first. When the two pieces moving in a round have the same speed the round is resolved as always (collisions, beats, the tie). When they differ, the faster move is made first, exactly as chosen, as a move of an alternating game, and if it ends the game the round ends there. Then the slower move is made if its piece still stands where it was and the move is still one that piece could make on the new board (a move onto a square the faster piece now holds is a capture of it if the piece may capture there); otherwise it is void and the piece stays, or lost if the piece was taken or changed. So a slower piece cannot flee a faster strike, and unequal speeds never collide or swap. Only the order is a rule: the speeds in use, 0 counted for every piece that names none, must be 0, 1, 2, ... with none skipped, and some piece of one player must differ in speed from some piece of the other. Never on a portal or a wild piece.
- **`rest`**: integer from 1 to 4. Optional. Schema 29. How many of its player's turns this piece rests after it makes its player's move, 1 to 4; omit it for 0 (it never rests), and never write 0. In any game, played by turns or in rounds (a round is a turn of both players): a piece that moved, turned in place, was dropped, or struck without leaving its square may not make its player's move during that player's next rest turns, counted on the owner's own turns only, as the type it is after the move (a pawn promoted to a queen rests as the queen). A resting piece still stands: it attacks, blocks, gives check (which must be answered) and can be captured; captured, moved off its square or changed by the other player, it rests no more. When every move a player has is a resting piece's, the rest is lifted for that turn, so the rest never leaves a player without a move. Never on a portal or a wild piece, and not beside a turn of more than one move (rules.turn moves or ramp). Replaces schema 27's rules.simultaneous.rest, which is "rest": 1 on every piece of a game in rounds.
- **`facing`**: object. Optional. This piece has an orientation, and (unless turns is false) may spend a whole move turning to any other one instead of moving. Its move vectors are read rotated, so a pawn facing right advances east. Give it only to a piece whose moves are *not* symmetric under the turn -- a rook reads the same rays whichever way it faces, so turning one would be a move that does nothing, and the validator refuses that. A royal piece may not have one.
  - **`axes`**: one of `2`, `4`. Required. How many directions this piece can point: 2 for forward and backward, 4 for the quarter turns.
  - **`initial`**: one of `"forward"`, `"right"`, `"backward"`, `"left"`. Optional; default `"forward"`. Which way it points at the start, relative to its own forward. It must be one the piece can turn back to, so a piece with 2 axes has only forward and backward.
  - **`turns`**: boolean. Optional; default `true`. Whether a move may turn it. False is a facing no move turns: each piece points the way its placement says for the whole game, and one that enters play later points its initial way. A pawn placed facing backward stays a pawn going backward. Some placement must then point it a second way, or the facing would change nothing.

## `pieces[].moves`: one entry per way of moving

- **`kind`**: one of `"slide"`, `"leap"`, `"hop"`, `"lame"`, `"sweep"`. Required. slide: step repeatedly, blocked by pieces. leap: jump straight to the target, ignoring anything between. hop: jump exactly one screening piece, then continue. lame: like leap, but blocked by a piece on the way, as the xiangqi horse is. A one-square vector has no square to be blocked on and is refused: use a leap or a range-1 slide. sweep: jump to any square along the line, removing every piece passed over, friendly included; a piece it may not capture blocks, and a sweep that could pass over the enemy royal is check.
- **`vectors`**: list of `[dx, dy]` pairs (each number from -8 to 8), at least 1. Required. The steps this move is made of, each `[dx, dy]` (see *Coordinates*). The symmetry turns each one into its images.
- **`symmetry`**: one of `"none"`, `"lateral"`, `"rotate4"`, `"dihedral8"`, `"rotate6"`, `"dihedral12"`. Optional; default `"none"`. How the base vectors are expanded. none: as written. lateral: left-right mirror. rotate4: four right-angle rotations. dihedral8: all eight images, so [1,2] gives the knight. On a hex lattice only none, rotate6 (six 60-degree turns, so [1,0] gives the six neighbours) and dihedral12 (the six turns and their mirrors, so [1,2] gives the hex knight's twelve); the square's three turns are refused there, and the hex two on a square board.
- **`range`**: integer from 0 to 16. Optional; default `0` for a slide, `0` for a hop, `1` for a leap, `1` for a lame, `0` for a sweep. Maximum distance of a slide or a hop; 0 means unlimited. Not for a leap or a lame leap, which are one jump per vector and refuse it. To reach further, list more vectors.
- **`min_range`**: integer from 1 to 16. Optional; default `1`. Minimum distance of a slide or a hop. Set it equal to range for a move that must travel exactly that far, such as a pawn's opening double step. Refused on a leap or a lame leap, like range.
- **`mode`**: one of `"move"`, `"capture"`, `"both"`. Optional; default `"both"`. Whether this move may occupy an empty square, capture, or both.
- **`screen_range`**: integer from 0 to 16. Optional; default `0`. For hops only, how far past its screen the piece may land. 0 allows the whole rest of the line, which is the cannon, and 1 means the piece must land immediately beyond, which is the grasshopper.
- **`screen`**: one of `"any"`, `"own"`, `"enemy"`. Optional; default `"any"`. For hops only, whose piece may serve as the screen. 'any' is the xiangqi cannon and the default, 'own' is a piece that vaults its own army, and 'enemy' is one that can only jump an opponent.
- **`condition`**: object. Optional. Restrictions on when this move is available. Every condition stated must hold.
  - **`from_rank`**: integer, at least 0. Optional. Only from this rank, counted from the mover's own back rank.
  - **`first_move`**: boolean. Optional; default `false`. Only if this piece has never moved.
  - **`region`**: list of strings, at least 1. Optional. Only from these squares. Written for the first player; the second player's region is the reflection. Algebraic squares such as a1 or e4.

## `setup`

Where every piece stands when the game begins.

- **`mirror`**: boolean. Optional; default `true`. When true, list only the first player's army; the second player's is the reflection across the middle of the board.
- **`orientation`**: one of `"mirror"`, `"rotate"`. Optional; default `"mirror"`. How the second player's army and move directions are derived. mirror reflects the rank and leaves the file, as chess does; rotate turns the whole board 180 degrees.
- **`placements`**: list of objects, at least 1. Required. The starting squares, one entry per piece kind and player.
  - **`piece`**: string. Required. The id of the piece placed.
  - **`player`**: one of `0`, `1`. Optional; default `0`. Whose pieces these are: 0 for the first player, who moves first, 1 for the second. With `setup.mirror` on, write only the first player's and leave this out.
  - **`facing`**: one of `"forward"`, `"right"`, `"backward"`, `"left"`. Optional. Which way these pieces start pointing, for a piece that has a facing. Relative to their own player's forward, so a mirrored setup gives both armies the same orientations. Omit it and they start in the piece's own initial facing.
  - **`squares`**: list of strings, at least 1. Required. Algebraic squares such as a1 or e4.

## `rules`

Everything about the game beyond the pieces and the board: how it is won, and each extra rule. Every key is optional; an empty object is the defaults below.

### `rules.win`: how the game is won

*one of `"checkmate"`, `"royal_capture"`, `"annihilation"`, `"none"`*. Default `"checkmate"`. checkmate: trap the royal piece. royal_capture: take it, with no obligation to answer a threat. annihilation: clear the board of an army. none: the game is decided only by the endings stated below.

### `rules.stalemate`: a player with no legal move

*one of `"draw"`, `"loss"`, `"win"`*. Default `"draw"`. The result for the player who has no legal move but is not in check.

### `rules.checkmate`: what being checkmated does

*one of `"loss"`, `"win"`*. Default `"loss"`. The result for the player who is checkmated. 'win' inverts the object of the game, as losers chess does.

### `rules.promotion`: promotion

*object or list of objects, at least 1*. Leave it out for no such rule. One rule, or one per promoting piece as shogi needs.

Its keys:

- **`piece`**: string. Required. The id of the piece that promotes.
- **`rank`**: string or integer. Optional; default `"last"`. 'last', or a rank index counted from the first player's side.
- **`ranks`**: list of (integer, at least 0), at least 1. Optional. A promotion zone several ranks deep, instead of 'rank'.
- **`squares`**: list of strings, at least 1. Optional. A promotion zone as named squares, for a zone that is a shape, instead of 'rank' or 'ranks'. The only spelling on a hex lattice. Algebraic squares such as a1 or e4.
- **`mandatory`**: boolean. Optional; default `true`. False lets a piece decline to promote, as in shogi.
- **`on_leaving`**: boolean. Optional; default `false`. True also promotes the piece on a move that starts in the zone, wherever it lands, as in shogi: a silver standing in the zone may promote as it steps back out. False, the default, promotes only on a move that ends in the zone.
- **`only_lost`**: boolean. Optional; default `false`. True offers a target only while the mover's side has fewer of that type on the board than it started with, as in Grand chess: a pawn promotes only into a piece its side has lost. Every target must be in both armies' setup. Not beside drops that fill from captures, a circe strike or promote_on_entry.
- **`to`**: list of strings, at least 1. Optional. The ids it may promote to; the player chooses among them.
- **`to_by_player`**: object. Optional. Instead of `to`, a separate list of promotion targets for each player.
  - **`0`**: list of strings, at least 1. Required. For the first player.
  - **`1`**: list of strings, at least 1. Required. For the second player.

### `rules.en_passant`: taking a piece in passing

*boolean*. Default `false`. Requires promotion; applies to the promoting piece.

### `rules.castling`: castling

*object*. Leave it out for no such rule. The royal piece and a piece that has not moved make one move together, the royal stepping toward it and that piece jumping to the square the royal crossed.

Its keys:

- **`king`**: string. Required. The id of the piece that castles, usually the royal piece.
- **`rook`**: string. Required. The id of the piece it castles with.
- **`king_steps`**: integer from 1 to 4. Optional; default `2`. How many squares the castling piece moves.
- **`king_file`**: object. Optional. Instead of `king_steps`, the file letter the castling piece lands on, on each side.
  - **`kingside`**: string. Required. The file it lands on when castling toward the higher files.
  - **`queenside`**: string. Required. The file it lands on when castling toward the lower files.
- **`rook_file`**: object. Optional. The file letter the partner piece lands on, on each side.
  - **`kingside`**: string. Required. The file the partner lands on when castling toward the higher files.
  - **`queenside`**: string. Required. The file the partner lands on when castling toward the lower files.

### `rules.must_capture`: a capture must be taken

*boolean*. Default `false`. A capture on offer is the only kind of move there is.

### `rules.forbid_check`: check is not allowed

*boolean*. Default `false`. Giving check is not allowed at all.

### `rules.forbid_immobility`: no move that strands a piece

*boolean*. Default `false`. A move may not leave its own piece somewhere it could never move from again.

### `rules.royal_facing`: royal pieces facing each other

*one of `"allowed"`, `"forbidden"`*. Default `"allowed"`. Whether the two royal pieces may see each other down an open file.

### `rules.bare_royal`: a player left with royalty alone

*one of `"win"`, `"loss"`, `"draw"`*. Leave it out for no such rule. The result for the player reduced to royalty alone, so 'loss' is shatranj's rule.

### `rules.check_limit`: winning by giving checks

*integer from 1 to 20*. Leave it out for no such rule. Delivering this many checks wins.

### `rules.territory`: claiming ground

*object*. Leave it out for no such rule. Moving onto a square claims it, and claimed ground can win the game outright or settle it when a clock runs out.

Its keys:

- **`mode`**: one of `"permanent"`, `"repaint"`. Required. Moving onto a square claims it. 'permanent' means a claim can never be taken back, which makes every claim a one-way step; 'repaint' lets the other player claim it later.
- **`win_at`**: integer, at least 0. Optional; default `0`. Claiming this many squares wins outright. 0 leaves the claiming to whatever else reads it -- rules.scoring with count 'territory' compares ground held when time runs out.

### `rules.turn`: more than one move a turn

*object*. Leave it out for no such rule. A turn is this many moves rather than one, and the opening turns can be given their own lengths.

Its keys:

- **`moves`**: integer from 1 to 4. Required. How many moves a turn is. 1 is the ordinary game. This is the one rule a search feels rather than reads. The branching factor becomes the legal move count raised to this, so it is capped rather than left open.
- **`ramp`**: list of integers from 1 to 4, 1 to 8 entries. Optional. The lengths of the opening turns, in order, before 'moves' takes over. With two moves a turn whoever starts is a whole move ahead forever; 'moves': 2 with 'ramp': [1] gives the first turn one move and levels it.

### `rules.obstacles`: walls on the board

*object*. Leave it out for no such rule. Squares nothing may stand on. A sliding line stops at one, and a jump vaults it unless the rule says otherwise. This is how a board is given a shape other than a rectangle.

Its keys:

- **`squares`**: list of strings, at least 1. Required. Squares nothing may occupy. A ray stops at one and a piece may not land on one. Algebraic squares such as a1 or e4.
- **`mirror`**: boolean. Optional; default `false`. Also wall the mirror image of every square listed. A wall belongs to the board rather than to a player, so this is shorthand for writing a symmetric wall out -- not a way to give one side a wall the other does not have. Defaults to false, unlike setup.mirror, because a square list here is usually meant absolutely.
- **`blocks_leaps`**: boolean. Optional; default `false`. Whether a wall also stops a jumping piece. False, the default, lets a knight vault a wall; true stops it. The square a jump is stopped at is the one the xiangqi horse is blocked at, one step along the longer axis.

### `rules.strike`: what a taken piece becomes

*object*. Leave it out for no such rule. A piece that is taken turns into another piece instead of leaving the board, following a chain of what becomes what.

Its keys:

- **`effect`**: one of `"capture"`, `"demote"`, `"promote"`. Optional; default `"capture"`. What happens to a piece that is taken. 'capture' removes it, which is the ordinary game unless the position is 'stay' or 'home'. 'demote' and 'promote' turn it into another piece instead, following the chain.
- **`chain`**: list of objects, at least 1. Optional. Where a struck piece goes, one link per piece. A piece with no link is removed when it is taken, as usual, so the bottom of a demotion chain is simply left out. A demotion chain may not cycle; a promotion chain may.
  - **`from`**: string. Required. The piece that was taken.
  - **`to`**: string. Required. What it becomes instead.
- **`position`**: one of `"stay"`, `"advance"`, `"swap"`, `"choose"`, `"home"`. Optional; default `"advance"`. Where the two pieces end up. 'advance' is the ordinary capture -- the attacker takes the square -- and may only be used when nothing survives the chain, because there would be nowhere to put a survivor. 'stay' leaves the attacker where it was and transforms the struck piece where it stands; under effect 'capture' the taken piece is removed and the attacker never moves (rifle chess). 'swap' exchanges the two. 'choose' advances the attacker and lets it place the struck piece on a square of 'landing'. 'home' advances the attacker and puts the struck piece -- itself under 'capture', what the chain makes it otherwise -- back on its home square, one of the squares its side's setup gives its type. A type with at most one home on each colour goes to the one on the capture square's colour, so a rook, knight or bishop keeps its square colour; any other type, such as a pawn with a home on every file, goes to the one on the capture's file, else the one on the capture square's colour; failing both, the nearest. A hex board has no two colours, so there the file decides. If that square is taken, or the type has no home, it is removed (orthodox circe).
- **`switches_sides`**: list of strings, at least 1. Optional. Pieces that change sides when they capture: the capturer becomes the other player's piece where it lands (Andernach chess). A royal piece may not be listed.
- **`landing`**: object. Optional. Only with position 'choose', and required there: the squares the attacker may place a surviving piece on, as [dx, dy] steps from where it stood, forward being the attacker's. [[1, 0], [1, 1]] with rotate4 is a king's step. A piece with no empty landing square cannot be taken.
  - **`vectors`**: list of `[dx, dy]` pairs (each number from -8 to 8), at least 1. Required. The landing steps, each `[dx, dy]`.
  - **`symmetry`**: one of `"none"`, `"lateral"`, `"rotate4"`, `"dihedral8"`, `"rotate6"`, `"dihedral12"`. Optional; default `"none"`. How the vectors are expanded, as on a move, and refused off its lattice as there.
- **`immune`**: list of strings. Optional. Pieces a strike cannot reach; taking one removes it as usual. Every royal piece must be listed here, because a king that turns into something else makes check a question about a piece that is not on the board yet.

### `rules.contacts`: what two pieces become when they meet

*object*. Leave it out for no such rule. One link per pair says what each of two pieces becomes when one moves onto the other, and where the two of them end up.

Its keys:

- **`links`**: list of objects, at least 1. Required. What two pieces that meet both become, one link per pair. A pair with no link is an ordinary capture, so only the meetings that matter need saying. A link must change at least one of the two.
  - **`mover`**: string. Required. The piece that is doing the taking.
  - **`onto`**: string. Required. The piece it takes.
  - **`mover_to`**: string. Optional. What the mover becomes. Leave it out and the mover is unchanged, which is what an ordinary capture does.
  - **`onto_to`**: string. Optional. What the piece it took becomes. Leave it out and that piece is removed, as usual.
  - **`own`**: boolean. Optional; default `false`. Set true and the link is about the mover's own piece: the mover may move onto a friendly 'onto' along its capture moves, and the link says what both become. Leave it out for an enemy target.
- **`position`**: one of `"stay"`, `"advance"`, `"swap"`, `"choose"`. Optional; default `"advance"`. Where the two pieces end up. 'advance' is the ordinary capture -- the mover takes the square -- and may only be used when no link leaves the target standing, because there would be nowhere to put it. 'stay' leaves the mover where it was and transforms the target where it stands. 'swap' exchanges the two. 'choose' advances the mover and lets it place a surviving target on a square of 'landing'.
- **`landing`**: object. Optional. Only with position 'choose', and required there: the squares the attacker may place a surviving piece on, as [dx, dy] steps from where it stood, forward being the attacker's. [[1, 0], [1, 1]] with rotate4 is a king's step. A piece with no empty landing square cannot be taken.
  - **`vectors`**: list of `[dx, dy]` pairs (each number from -8 to 8), at least 1. Required. The landing steps, each `[dx, dy]`.
  - **`symmetry`**: one of `"none"`, `"lateral"`, `"rotate4"`, `"dihedral8"`, `"rotate6"`, `"dihedral12"`. Optional; default `"none"`. How the vectors are expanded, as on a move, and refused off its lattice as there.
- **`immune`**: list of strings. Optional. Pieces no contact reaches; taking one removes it as usual and leaves the mover alone. Every royal piece must be listed here, because a king that turns into something else makes check a question about a piece that is not on the board yet.

### `rules.wrap`: edges that join

*object*. Leave it out for no such rule. A piece that runs off one edge continues from the opposite one, which makes the board a cylinder or a torus.

Its keys:

- **`files`**: one of `"none"`, `"left"`, `"right"`, `"both"`. Optional; default `"none"`. Which side edges a piece may run off and continue from the opposite one. 'both' makes the board a cylinder. Edges are named from the standard diagram, with the first player at the bottom and file a on the left. Not on a hexagon, whose sides are joined by n_s, ne_sw and nw_se.
- **`ranks`**: one of `"none"`, `"bottom"`, `"top"`, `"both"`. Optional; default `"none"`. The same for the top and bottom edges. Both axes together make a torus, which has no corner to trap a king in.
- **`files_by_player`**: object. Optional. Instead of `files`, a separate value for each player, keyed `"0"` and `"1"`.
  - **`0`**: one of `"none"`, `"left"`, `"right"`, `"both"`. Required. For the first player.
  - **`1`**: one of `"none"`, `"left"`, `"right"`, `"both"`. Required. For the second player.
- **`ranks_by_player`**: object. Optional. Instead of `ranks`, a separate value for each player, keyed `"0"` and `"1"`.
  - **`0`**: one of `"none"`, `"bottom"`, `"top"`, `"both"`. Required. For the first player.
  - **`1`**: one of `"none"`, `"bottom"`, `"top"`, `"both"`. Required. For the second player.
- **`n_s`**: one of `"none"`, `"s"`, `"n"`, `"both"`. Optional; default `"none"`. On a hexagon only: whether its top ('n') and bottom ('s') sides are joined. 'both' lets a piece leave by either and come back in by the other, straight across; a side's name lets it leave by that side only. Sides are named from the standard diagram, the first player at the bottom.
- **`n_s_by_player`**: object. Optional. Instead of `n_s`, a separate value for each player, keyed `"0"` and `"1"`.
  - **`0`**: one of `"none"`, `"s"`, `"n"`, `"both"`. Required. For the first player.
  - **`1`**: one of `"none"`, `"s"`, `"n"`, `"both"`. Required. For the second player.
- **`ne_sw`**: one of `"none"`, `"sw"`, `"ne"`, `"both"`. Optional; default `"none"`. On a hexagon only: whether its upper-right ('ne') and lower-left ('sw') sides are joined. 'both' lets a piece leave by either and come back in by the other, straight across; a side's name lets it leave by that side only. Sides are named from the standard diagram, the first player at the bottom.
- **`ne_sw_by_player`**: object. Optional. Instead of `ne_sw`, a separate value for each player, keyed `"0"` and `"1"`.
  - **`0`**: one of `"none"`, `"sw"`, `"ne"`, `"both"`. Required. For the first player.
  - **`1`**: one of `"none"`, `"sw"`, `"ne"`, `"both"`. Required. For the second player.
- **`nw_se`**: one of `"none"`, `"nw"`, `"se"`, `"both"`. Optional; default `"none"`. On a hexagon only: whether its upper-left ('nw') and lower-right ('se') sides are joined. 'both' lets a piece leave by either and come back in by the other, straight across; a side's name lets it leave by that side only. Sides are named from the standard diagram, the first player at the bottom.
- **`nw_se_by_player`**: object. Optional. Instead of `nw_se`, a separate value for each player, keyed `"0"` and `"1"`.
  - **`0`**: one of `"none"`, `"nw"`, `"se"`, `"both"`. Required. For the first player.
  - **`1`**: one of `"none"`, `"nw"`, `"se"`, `"both"`. Required. For the second player.

### `rules.npcs`: wild pieces

*list of objects, at least 1*. Leave it out for no such rule. Wild pieces are pieces neither player owns. After every move each one steps to the next square of its route and eats whatever stands there that its type may capture, or waits if it may not. A royal piece standing where a wild piece steps next is in check from it when that type may capture the royal. Taking a wild piece is worth its bounty.

Each entry's keys:

- **`piece`**: string. Required. A declared, non-royal piece id. The wild piece has that type's name, drawing and value, and eats what that type may capture; its moves are what it does in a player's hands, not how it walks.
- **`route`**: list of strings, at least 1. Required. The squares it walks, in order and cyclically, starting on the first. Distinct squares, on no other wild piece's route, with no piece of the opening array on the first. One square is a wild piece that never moves. Algebraic squares such as a1 or e4.
- **`also_at`**: list of strings, at least 1. Optional. More wild pieces of the same kind on this same route: the squares, other than the route's first, where each further one starts, listed in route order and with no piece of the opening array on any. Every one walks the route in step with the first; one that finds another in its way waits behind it. Leave the key out for one wild piece on the route. Algebraic squares such as a1 or e4.
- **`bounty`**: number, at least 0. Optional; default the piece's value. What a player gains by taking it, read by rules.scoring with count 'points'. Defaults to the piece's value.
- **`mirror`**: boolean. Optional; default `false`. Also place a second wild piece on the mirror image of the route, walking it in the same order. A route has a direction, so one wild piece is an asymmetry and a mirrored pair is not. Defaults to false.
- **`immune`**: list of strings. Optional. The only piece ids that may take this wild piece; it is immune to every other. Leave the key out and anything may take it. Another wild piece is not bound by it. Not usable beside rules.drops, because capture permission is a fact about a type rather than about who owns it and the immunity would follow the piece out of a hand.

### `rules.portals`: portals

*list of objects, 1 to 4 entries*. Leave it out for no such rule. Groups of portals, numbered by their place in the list (at most four groups and eight ends in all). A group is a pair (orange and blue, each leading to the other) or a map of ends (each leading to one other end of its group, or nowhere). A piece whose move ends on a portal end that lets it in does not stop there: it comes out of the end that one leads to, one king's step beyond it in the direction it was travelling (right-and-up by any amounts comes out diagonally up and right), and its move ends there -- capturing, promoting or winning there as usual. A blocked exit (off the board, a wall, its own piece, a piece it may not take, another portal, or its own square) is no move. An end that does not let a piece in is an uncapturable obstacle to it. A portal is never captured. A group with kind 'fold' works differently (see kind). Not usable on a hex lattice or beside npcs, strike, contacts, explode_on_capture, territory, a sweep, or a piece region.

Each entry's keys:

- **`piece`**: string. Required. A declared piece id whose moves the group's ends use when a player moves one. It must be worth 0, not royal, not placed by setup.placements and named by no other rule; every move it has must say mode 'move' and read the same for both players. It may have no moves when nobody moves the group.
- **`orange`**: string. Optional. Where the pair's first end starts: an empty square, not a wall. Algebraic, such as c4. The keys name the pair's two ends, this one first; both are drawn in the pair's colour.
- **`blue`**: string. Optional. Where the pair's second end starts, likewise.
- **`moved_by`**: object. Optional. Who may spend a turn moving each end of the pair, with the moves of the pair's `piece`. Leave it out and the portals never move.
  - **`orange`**: list, each one of `0`, `1`. Optional; default `[]`, nobody. Which players may move the pair's first end (`orange` in the file), as a turn: [] for nobody (the default), [0], [1], or [0, 1] for either.
  - **`blue`**: list, each one of `0`, `1`. Optional; default `[]`, nobody. Which players may move the pair's second end (`blue` in the file), as a turn: [] for nobody (the default), [0], [1], or [0, 1] for either.
- **`passage`**: object. Optional. Which end each player's pieces may go in by. Leave it out and both ends work both ways for both players.
  - **`0`**: one of `"both_ways"`, `"orange_to_blue"`, `"blue_to_orange"`, `"closed"`. Optional; default `"both_ways"`. Which end player 0's pieces may go in by: 'both_ways' (the default), 'orange_to_blue' (in at the first end only), 'blue_to_orange' (in at the second end only), or 'closed', when both ends are an uncapturable obstacle to that player.
  - **`1`**: one of `"both_ways"`, `"orange_to_blue"`, `"blue_to_orange"`, `"closed"`. Optional; default `"both_ways"`. Which end player 1's pieces may go in by: 'both_ways' (the default), 'orange_to_blue' (in at the first end only), 'blue_to_orange' (in at the second end only), or 'closed', when both ends are an uncapturable obstacle to that player.
- **`carries`**: object. Optional. Schema 21. Which pieces may go in by each end of a pair, keyed by the end; an end not listed lets every piece in.
  - **`orange`**: list of strings. Optional. Schema 21. The piece ids that may go in by the pair's first end; every other piece meets it as an uncapturable obstacle. Absent: every piece. Never empty, never every piece that can move, never a portal.
  - **`blue`**: list of strings. Optional. Schema 21. The piece ids that may go in by the pair's second end; every other piece meets it as an uncapturable obstacle. Absent: every piece. Never empty, never every piece that can move, never a portal.
- **`ends`**: list of objects, 3 to 8 entries. Optional. Schema 21. Three to eight ends, each leading to another of the group or nowhere: a ring (each to the next), a funnel (several to one way out), or any other map. Every end must lead somewhere or be led to. Two ends are a pair: write them as orange and blue instead.
  - **`at`**: string. Required. Where this end starts: an empty square, not a wall. Algebraic, such as c4.
  - **`to`**: integer from 0 to 7. Optional. The index in this group's ends of the end a piece that goes in here comes out of: never this end itself. Absent: a way out only, which nobody goes in by and pieces only come out of.
  - **`enter`**: list, each one of `0`, `1`. Optional. Which players' pieces may go in by this end: [0], [1] or [0, 1] (the default). Only beside 'to'.
  - **`moved_by`**: list, each one of `0`, `1`. Optional. Which players may move this end, as a turn: [] for nobody (the default), [0], [1] or [0, 1].
  - **`carries`**: list of strings. Optional. The piece ids that may go in by this end; absent, every piece. Only beside 'to'.
- **`no_straight_back`**: boolean. Optional; default `false`. Schema 20. When true, a portal of this group that has just been moved may not be moved straight back onto the square it came from on the very next move; it may go anywhere else, and back on the move after. Stops two players undoing each other's portal moves for ever. Needs an end that whoever moves next may move too (in moved_by for both players, or a turn longer than one move), and a portal piece with a move back; a bar no move can meet is refused. Default false.
- **`kind`**: one of `"fold"`. Optional. Schema 24. 'fold' makes the group fold portals; absent, they are momentum portals (the default, never written). A fold is not a piece: its square may hold a piece of either side, which is captured there as anywhere, and may start under a piece. A move that ends on an empty fold that lets the piece in -- a leap, slide, hop or pawn's step -- may stay there, or go through and end on the end this one leads to instead, as a landing there: onto an empty square if the step could end on one, onto an enemy piece it may take (captured) if the step could capture, never onto its own piece or its own square, promoting by that square. One fold per move. A move that passes over a fold, or ends on one a piece stands in, is not changed by it. A fold that does not let a piece in, or leads nowhere, is plain floor to it. A fold moved by a player may land on a piece of either side (never on another portal), which then stands in it; moving it never moves that piece. Castling, drops and en passant never go through a fold, and a move through one opens no en passant.

### `rules.simultaneous`: both players moving at once

*object*. Leave it out for no such rule. Schema 26. Both players move at once. Each round, each player chooses one of their own legal moves on the same board without seeing the other's; both are revealed and made together. Only two pieces moving onto the same square collide, and pieces.beats decides who survives. A piece that moves onto a square its target has just left lands there if its move could land on an empty square, and otherwise stays where it was (a capture-only move is void). Two such void moves at each other two rounds in a row bar that pair of moves for both players until either square changes. There is no check: rules.win must be 'royal_capture', 'annihilation' or 'none', and an ending reached by both players in the same round is a draw. A round counts as two moves on the clocks. Not usable beside castling, en passant, a sweep, check rules, a turn of more than one move, a portal a player moves, a strike or contact position 'choose', a strike position 'stay', or drops.no_mate. Schema 27 adds two switches against a royal chased round after round, swap and both; schema 29's pieces[].rest ("rest": 1 on every piece) is the rest that was the third.

Its keys:

- **`tie`**: one of `"initiative"`, `"impossible"`. Optional. What decides a collision that is a tie, the two pieces each beating the other or neither beating the other. Leave it out and both are eliminated, leaving the square empty; never write that out. 'initiative': one player holds the initiative, the second player at the start; a tie goes to the holder's piece, and the initiative then passes to the other player. 'impossible': a promise, checked, that no tie can ever occur, because for every piece each player can have, exactly one of the two beats the other (the armies use different pieces, such as a circle of rock, paper and scissors).
- **`swap`**: one of `"collision"`. Optional. Schema 27. 'collision': two moves that each end on the other's starting square meet as a collision, decided by pieces.beats and the tie rule, instead of the two pieces passing each other; a winner ends on its destination and a loser is removed. Leave it out and swapping pieces pass. Refused where no piece of one player can take a piece of the other that can take it back.
- **`both`**: one of `"score"`. Optional. Schema 27. 'score': an ending both players reach in the same round (both royals taken, both armies gone, both on a goal) goes to the player ahead under rules.scoring, and is a draw only when the scores are equal. Leave it out and such a round is a draw. Requires rules.scoring.

### `rules.promote_on_entry`: promoting on arrival

*boolean*. Default `false`. A piece that enters play inside its own promotion zone -- dropped there, or left standing there by a strike that turned it into a promoting piece -- promotes at once, to the first piece its promotion rule lists, even where promotion is otherwise optional. Needs rules.promotion and a way in: rules.drops, or a strike whose chain leaves a piece standing. Refused where no promoting piece could ever enter its zone, such as a pawn whose drop region stops short of it. False, the default, leaves such a piece standing where it arrived.

### `rules.scoring`: who wins when a clock runs out

*object*. Leave it out for no such rule. A clock that runs out settles the game by comparing something -- material, pieces, ground held -- instead of drawing it.

Its keys:

- **`count`**: one of `"material"`, `"pieces"`, `"region"`, `"territory"`, `"points"`. Optional; default `"material"`. What to compare when the ply limit or the no-progress clock runs out, instead of drawing. 'material' sums the declared piece values, 'pieces' counts them, 'region' counts your own pieces standing in the squares below, 'territory' counts ground claimed and needs rules.territory, 'points' counts bounties collected and needs rules.npcs. Higher wins; equal is still a draw.
- **`region`**: list of strings, at least 1. Optional. The ground being held, for count 'region'. Written for the first player and reflected, so 'hold the centre' is symmetric and 'hold their half' is not. Algebraic squares such as a1 or e4.

### `rules.reach`: winning by arriving

*object*. Leave it out for no such rule. Arriving on one of these squares decides the game there and then; or, where the rule says so, moving off one does, and standing there decides nothing.

Its keys:

- **`piece`**: string. Optional. Omit for any piece.
- **`squares`**: list of strings, at least 1. Required. Where arriving decides the game. Algebraic squares such as a1 or e4.
- **`mirror`**: boolean or one of `"rotate"`. Optional; default `true`. true aims each side at the other's end by reflecting the rank; 'rotate' reflects both rank and file (180° rotation); false sends both to the same squares.
- **`safe`**: boolean. Optional; default `false`. The arriving piece must not be capturable where it lands.
- **`result`**: one of `"win"`, `"loss"`, `"draw"`. Optional; default `"win"`. The result for the player whose piece reaches the squares.
- **`on`**: one of `"arrive"`, `"leave"`. Optional; default `"arrive"`. Schema 18. 'arrive' (the default): a piece standing on a goal square decides the game. 'leave': the move that takes the piece off a goal square decides it (Jeson Mor), and standing there decides nothing. Not with safe.

### `rules.extinction`: losing by running out of a piece

*object*. Leave it out for no such rule. A player whose whole army, or whose count of some piece, falls below a number gets this result.

Its keys:

- **`pieces`**: one of `"any"` or list of strings, at least 1. Required. 'any' counts the whole army.
- **`min_count`**: integer from 1 to 64. Optional; default `1`. A player with fewer than this many of the listed pieces gets the result. Each player must start with at least this many (of each listed type, under per_type; of each listed type they start with any of, under per_type 'own'), or the game would be over before the first move.
- **`result`**: one of `"win"`, `"loss"`, `"draw"`. Optional; default `"loss"`. The result for the player who falls below the count.
- **`per_type`**: boolean or one of `"own"`. Optional; default `false`. True counts each listed type on its own, so a side loses when any one of them falls below min_count, as in Extinction chess. 'own' (schema 27) counts each listed type on its own too, but only against a player who starts with it, so two armies of different kinds each lose by running out of any one of their own; list both armies' kinds. False, the default, counts the listed types together. Needs two or more types listed.

### `rules.drops`: pieces played back onto the board

*object*. Leave it out for no such rule. A player may put a piece from their hand onto an empty square instead of moving one. The hand is filled by what they capture, by a stock they start with, or by both.

Its keys:

- **`from_captures`**: boolean. Optional; default `true`. Captured pieces change sides and may be played back.
- **`region`**: object from piece id to list of strings, at least 1. Optional. Per piece id, the squares it may be dropped on.
- **`no_doubled`**: list of strings. Optional. These may not be dropped onto a file already holding one of yours.
- **`no_mate`**: list of strings. Optional. These may not deliver checkmate on the drop itself.
- **`stock`**: object from piece id to integer from 1 to 32 or `"unlimited"`. Optional. Per piece id, how many each player holds before anything has been captured, as a count of 1 to 32 or "unlimited" for a supply that never runs out. A stocked piece is the only one that may have an empty 'moves' list.
- **`placement`**: boolean. Optional; default `false`. True is a placement phase: while a player holds any piece in hand, a drop is their only legal move, so a stock is played out before anything on the board moves (Placement chess, Sittuyin). Only then may a royal start in the stock, once, and not also placed.
- **`either_face`**: boolean. Optional; default `false`. True lets a piece in hand be dropped as itself or as its other face, the piece whose promoted_from it is (Kyoto shogi). Needs at least one promoted_from pair a hand can hold; each base may have one face, and no face may itself be a base, stocked, or royal.

### `rules.explode_on_capture`: captures that explode

*object*. Leave it out for no such rule. A capture destroys the capturing piece and everything standing around the square it happened on.

Its keys:

- **`radius`**: integer from 0 to 3. Optional; default `1`. How far the blast reaches around the capture square; 1 is atomic chess. 0 destroys only the capturer and the piece it took (kamikaze chess), and then 'immune' may not be set.
- **`immune`**: list of strings. Optional. Pieces that survive standing next to a capture.
- **`by`**: list of strings. Optional. The only capturing pieces whose captures explode. Leave it out and every capture does. An empty list is no rule at all.
- **`of`**: list of strings. Optional. The only captured pieces whose being taken explodes. Leave it out and taking anything does. An empty list is no rule at all.

### `rules.max_plies`: the longest a game may run

*integer from 2 to 10000*. Default `400`. How many moves the game may last, counting each player's move separately, before it stops as a draw (or is decided by `rules.scoring`).

### `rules.repetition_draw`: drawing by repetition

*integer from 2 to 10 or `"never"`*. Default `3`. How many repetitions of a position draw the game, or "never" for a game with no draw by repetition at all.

### `rules.no_progress_draw`: how long a game may stall

*integer from 2 to 1000*. Default `100`. The game is drawn after this many moves in a row, counting both players', with no capture and no promotion (chess's fifty-move rule is 100).

## Keys that do not change the game

Two more top-level objects are accepted. Neither changes how the game plays, so a file
written from this reference needs neither.

- **`appearance`**: how the pieces are drawn (`icons`),
  mapping a piece id to the name of a drawing, or to a pair of names, one per player.
  Without it every piece is drawn from its name, or as its letter.
- **`provenance`**: where the game came from, as free text
  (`added`, `author`, `exhibit`, `history`, `origin`, `parents`, `preview`, `source`, `theme`, `year`). `added` is a day written
  `YYYY-MM-DD` and `parents` a list of game identifiers.

## How a wrong file is refused

`validate` checks a file in two passes and reports every problem it finds, not just the
first. The first pass is the shape described above: unknown keys, wrong types, values
outside their list or range, missing required keys. The second is whether the parts
agree: every piece id named anywhere is a declared piece, every square is on the board,
every move can be made on this board, a rule that needs another has it (en passant needs
promotion), two rules that cannot be played together are not, and a rule that could never
fire is refused rather than kept. A file that uses a key newer than the `schema_version`
it declares is refused with the version to declare.

Each line below is what `validate` prints for one mistake made to the first worked
example.

- A key the language does not have, on the knight: `"stamina": 2`.

  ```text
  pieces[1] has unknown keys: stamina
  ```

- A move kind that does not exist: `"kind": "teleport"`.

  ```text
  pieces[2].moves[0].kind must be one of ('slide', 'leap', 'hop', 'lame', 'sweep'), got 'teleport'
  ```

- A square off the board, `f1`, in the setup.

  ```text
  setup.placements[1].squares entry 'f1' is not a square on this board
  ```

- A promotion to a piece that is not declared: `"to": ["Q", "A"]`.

  ```text
  rules.promotion.to entry 'A' is not a declared piece
  ```

- En passant, with the promotion rule it needs taken away.

  ```text
  rules.en_passant requires rules.promotion: en passant is defined against the promoting piece, so there must be one
  ```

- The portals of the second example under `"schema_version": 18`.

  ```text
  the spec uses rules.portals, rules.portals.moved_by, added in schema 19; declare "schema_version": 19 to use it
  the spec uses rules.portals.no_straight_back, added in schema 20; declare "schema_version": 20 to use it
  ```

## Worked examples

### 1. A small chess-like game

```json
{
  "schema_version": 29,
  "name": "Five by five",
  "description": "Chess on a five by five board: one rook, knight, bishop, queen and king a side behind five pawns. Pawns step one square and promote on the far rank; no castling, no en passant.",
  "board": {"width": 5, "height": 5},
  "pieces": [
    {
      "id": "P",
      "name": "pawn",
      "value": 1,
      "moves": [
        {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"}
      ]
    },
    {
      "id": "N",
      "name": "knight",
      "value": 3,
      "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]
    },
    {
      "id": "B",
      "name": "bishop",
      "value": 3,
      "moves": [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}]
    },
    {
      "id": "R",
      "name": "rook",
      "value": 5,
      "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]
    },
    {
      "id": "Q",
      "name": "queen",
      "value": 9,
      "moves": [{"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
    },
    {
      "id": "K",
      "name": "king",
      "value": 0,
      "royal": true,
      "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
    }
  ],
  "setup": {
    "mirror": true,
    "placements": [
      {"piece": "P", "squares": ["a2", "b2", "c2", "d2", "e2"]},
      {"piece": "R", "squares": ["a1"]},
      {"piece": "N", "squares": ["b1"]},
      {"piece": "B", "squares": ["c1"]},
      {"piece": "Q", "squares": ["d1"]},
      {"piece": "K", "squares": ["e1"]}
    ]
  },
  "rules": {
    "win": "checkmate",
    "stalemate": "draw",
    "promotion": {"piece": "P", "rank": "last", "to": ["Q", "R", "B", "N"]},
    "max_plies": 200
  }
}
```

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

### 2. Portals that the players move

```json
{
  "schema_version": 29,
  "name": "Five by five with moving portals",
  "description": "The five by five game with a pair of portals on the middle rank, on b3 and d3. A piece whose move ends on one comes out one step beyond the other, in the direction it was going. Either player may spend a turn moving either portal one king's step to an empty square, but not straight back on the very next move.",
  "board": {"width": 5, "height": 5},
  "pieces": [
    {
      "id": "P",
      "name": "pawn",
      "value": 1,
      "moves": [
        {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"}
      ]
    },
    {
      "id": "N",
      "name": "knight",
      "value": 3,
      "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]
    },
    {
      "id": "B",
      "name": "bishop",
      "value": 3,
      "moves": [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}]
    },
    {
      "id": "R",
      "name": "rook",
      "value": 5,
      "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]
    },
    {
      "id": "Q",
      "name": "queen",
      "value": 9,
      "moves": [{"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
    },
    {
      "id": "K",
      "name": "king",
      "value": 0,
      "royal": true,
      "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
    },
    {
      "id": "O",
      "name": "portal",
      "value": 0,
      "moves": [
        {
          "kind": "leap",
          "vectors": [[1, 0], [1, 1]],
          "symmetry": "rotate4",
          "mode": "move"
        }
      ]
    }
  ],
  "setup": {
    "mirror": true,
    "placements": [
      {"piece": "P", "squares": ["a2", "b2", "c2", "d2", "e2"]},
      {"piece": "R", "squares": ["a1"]},
      {"piece": "N", "squares": ["b1"]},
      {"piece": "B", "squares": ["c1"]},
      {"piece": "Q", "squares": ["d1"]},
      {"piece": "K", "squares": ["e1"]}
    ]
  },
  "rules": {
    "win": "checkmate",
    "stalemate": "draw",
    "promotion": {"piece": "P", "rank": "last", "to": ["Q", "R", "B", "N"]},
    "max_plies": 200,
    "portals": [
      {
        "piece": "O",
        "orange": "b3",
        "blue": "d3",
        "moved_by": {"orange": [0, 1], "blue": [0, 1]},
        "no_straight_back": true
      }
    ]
  }
}
```

- The portal pair is a rule, `rules.portals`, and its ends are not pieces in the setup.
  `O` is declared only to say how an end moves when a player moves it: one king's step,
  onto an empty square (`"mode": "move"` is required of a portal's moves), worth 0 and not
  royal.
- `moved_by` lets both players (`[0, 1]`) move both ends; leave it out for portals that
  never move, and `O` may then have no moves at all.
- `no_straight_back` stops two players moving the same portal back and forth for ever.
- With no `passage`, both ends work both ways for both players. A pawn pushed from `b2`
  onto `b3` comes out on `d4`; a rook sliding up the d-file onto `d3` comes out on `b4`.
