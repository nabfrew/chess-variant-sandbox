"""Asking the model for a ruleset.

What the proposer is shown is the design decision here, and it is a short list:

* the DSL schema, generated from the validator so the two cannot disagree
* a design brief written in the language of games, not of metrics
* the parent ruleset, when this is a mutation
* prose observations about how the parent played -- never scores, never weights

What it is never shown is the index: not the formula, not the component names, not the
numbers. See ``feedback.py`` for why that line is drawn where it is.

``SCHEMA_PREAMBLE`` is the fourth silent surface a new rule has to reach, after the
engine, ``spec/describe.py`` and ``spec/jsonschema.py`` -- and since the rule schema
stopped being a *response* schema it is no longer the weakest of the four. Nothing on the
wire constrains a reply any more; the schema is prose in this prompt, and this is the
layer of it that says what a feature is *for*. A list headed "movement primitives" that
names three of the four is read as the four.

It went two bumps without noticing. ``lame`` had never been in it, and ``rules.wrap`` --
five committed rulesets, one of them the highest-scoring game in the corpus -- was absent
entirely: 16 of the 20 rule keys ``complexity._RULE_COSTS`` charges a player for were
named nowhere the model would read them. ``tests/test_loop.py`` asserts against that same
table, the one ``describe.py`` is held to and for the same reason: rule mass already
decides what counts as something a player has to be taught, so it is the honest answer to
"what must the proposer be told about".

The brief states design *goals* and is deliberately opinionated, but an opinion is not
the same as a restriction, and it used to hold one that the rest of the project does not:
"asymmetry between the two players' armies is not allowed". The DSL states asymmetry in
five places, ``horde`` is a scored band member, and ``onewaycylinder`` -- the two players
crossing the same seam in opposite directions -- outscores chess. A brief that forbids
what the corpus proves works is narrowing the search for nothing; what asymmetry actually
costs is fairness-by-construction, and saying *that* is the sentence that belongs there.

Validation failures get exactly one corrective retry, carrying the specific errors. A
second failure is recorded as a rejected candidate rather than retried again -- if the
model cannot satisfy the schema with the errors in front of it, asking a third time
mostly buys a third bill.

The prompt carries no version number, so its behaviour changes are recorded here. On
2026-09-13 (Q166) the `bare_royal` sentence was reversed to match the core: the value is
the result for the side reduced to royalty alone. Before that, the preamble and the
schema description told the proposer the opposite, so a proposal that wrote `bare_royal`
meant the inverse of what it played.

Also on 2026-09-13 (Q183), ten schema descriptions and the `describe_rules` sentences the
proposer reads were reworded for style. Colon-hinged sentences became two sentences or a
joined clause. The enum "value: meaning." descriptions kept their shape, because
`web/cards.py` splits them per option. No rule's meaning changed.

On 2026-09-24 (Q400) the brief stopped asking for few rules. The user took rule yield
out of the score with no limit on rule mass in its place (2026-09-23,
`docs/archive/plan-variety.md` §5.4, A2), and the brief's *Elegance* bullet ("few pieces, few
exceptions, rules that fit in a paragraph", "special cases are expensive and most
proposals should have none") was that family in prose. It is now *Purpose*: every piece
and rule should change what a player thinks about, which the remaining score still reads
through how much each piece is used. The preamble's subsystems paragraph no longer says
each rule "has to earn" its sentence. The ruin list keeps "a special rule to patch a
problem that a change to a piece would fix better": that is about which fix is better,
not how many rules there are.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from chess_variant_sandbox.llm.client import PROPOSAL_MODEL, Claude, cached_system, constrainable
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import BARE_ROYAL_HOLDER, validate_spec
from chess_variant_sandbox.spec.spelling import current_spelling

DESIGN_BRIEF = """\
You are designing two-player abstract strategy games in the family that includes \
chess, shogi and xiangqi: a rectangular board, pieces with fixed movement rules, \
perfect information, no randomness.

You are not trying to reproduce chess. You are trying to find a game that is worth \
playing on its own terms. The qualities that matter:

- **Depth.** A player who thinks harder should reliably win. The best move should be \
  findable but not obvious.
- **Purpose.** Every piece and every rule should change what a player has to think \
  about. One that a player could ignore and still play the same game is detail, not \
  design.
- **Tension.** Positions should offer real trade-offs -- attacking costs something, \
  defending costs something, and the choice is genuinely hard.
- **Variety.** Two strong players should not play the same game twice. If one opening \
  is clearly best, the game is smaller than it looks.
- **Balance.** Moving first should be an advantage, not a decision.

Things that reliably ruin a design, from experience:

- Pieces so strong that the game is decided by whoever attacks first.
- Pieces so weak that positions lock up and neither side can make progress.
- A board so large that the armies never meet, or so small that everything is forced.
- Movement that is powerful in every direction at once: it removes the geometry that \
  makes positions differ from each other.
- Adding a special rule to patch a problem that a change to a piece would fix better.

**The two sides do not have to be given the same game.** A mirrored setup is the \
default and the cheap way to be fair, and most designs should use it. But different \
armies, different objectives, or a board that behaves differently for each player are \
real designs and this project has measured good ones. What an asymmetric game gives up \
is fairness for free: a mirrored game is even by construction, and an uneven one has to \
be made even on purpose. So the question to answer before proposing one is what each \
side is being paid for what it gives up -- thirty-six pawns against a full army, a \
short route to the goal against a stronger piece to travel it with. If you cannot say \
what the bargain is, mirror it.
"""

SCHEMA_PREAMBLE = """\
Rules are expressed as data in the schema below and interpreted by an engine. There \
is no way to write procedural code, and nothing outside this schema exists. If a rule \
you want cannot be expressed here, design a different rule -- do not approximate it \
and describe the difference in prose, because the prose is not executed.

Movement primitives, and what they are for:

- **slide**: repeat a step until blocked. Rooks, bishops, queens. `range` caps the \
  distance (0 is unlimited); `min_range` sets a floor, so `range: 2, min_range: 2` is \
  "exactly two squares, and the path must be clear". A slide over a jump longer than   one square is a rider: `[[1,2]]` by `dihedral8` is the nightrider, jump after jump,   stopped only by a piece on a square it would land on.
  A slide's `path` may turn: `"circle"` takes the next direction round at every step, \
  either way, until it would come back on itself (`[[1,2]]` by `dihedral8` is the rose, \
  round an octagon of knight steps; `[[1,0],[1,1]]` by `rotate4` a ring of king steps), \
  and `"zigzag"` alternates its first direction with a neighbour (`[[1,1]]` by `rotate4` \
  is the crooked bishop, along a rank or file on its colour). A piece on a square the \
  path lands on ends it; a square reached two ways is one move, so a rose can check \
  along two octagons at once and one block answers only one. On a hex board the \
  directions go round as they lie (`[[1,2]]` by `dihedral12` is the hex rose, the hex \
  knight's twelve). Slides only, never range 1, and not beside portals, on the en \
  passant piece or a wild piece (but for one that pounces on its prey, which reads \
  its capture moves).
- **leap**: jump straight to the target square, ignoring everything in between. \
  Knights and kings. Blockers are irrelevant. It is one jump per vector and takes no \
  `range`: to reach further, list more vectors -- `[[1,1],[2,2],[3,3]]` is a diagonal \
  leaper three deep.
- **lame**: a leap a piece on the way can block, as the xiangqi horse and elephant \
  are. It reaches like a leap and is stopped like a slide, which is how a piece with a \
  knight's reach can still be weak enough to be worth playing around. A one-square   vector has nothing to be blocked on and is refused: use a `leap` or a range-1 `slide`.   With a `range` (counted in jumps; 0 unlimited) it is a lame rider: jump after jump,   each also stopped by a piece on the one square the jump passes (one step along its   longer axis), so a `[2,0]` lame rider is a rook that cannot jump its own pawns.
  `legs` states which squares block each jump instead, one list per vector, turned   with it by the symmetry: Janggi's elephant is `[[2,3]]` by `dihedral8` with   `"legs": [[[0,1],[1,2]]]`, blocked on both. A lame on a hex lattice must state them;   so must a hex leap longer than a step under `obstacles.blocks_leaps`, where a leap's   legs are where a wall stops it.
- **hop**: jump exactly one screening piece, then continue. The xiangqi cannon. \
  Powerful and unusual: it makes a crowded board more dangerous, not less.
- **sweep**: travel a line like a slide, but jump to any square on it and remove \
  every piece passed over, your own included. A piece it may not capture blocks it, \
  and a sweep that could pass over the enemy royal is check. Leaving home through your \
  own pawn costs the pawn, which is what keeps it from being a longer rook. A promoting \
  piece may not sweep, and a sweep cannot share a game with `strike`, `contacts`, \
  `explode_on_capture`, `drops`, `npcs` or `wrap`. `sweep_range` (no more than its \
  `range`, which it equals when left out) makes it clear only its first steps: \
  `"sweep_range": 3` with no `range` sweeps three squares and then slides on as far as \
  the line goes, stopped by the first piece past them.

`symmetry` expands your base vectors so you do not have to list them: `[[1,2]]` with \
`dihedral8` is the knight's eight moves; `[[1,0]]` with `rotate4` is the four \
orthogonals; `[[1,1]]` with `lateral` gives the two forward diagonals a pawn captures \
along. Vectors are always written from the first player's view, with +y forward; the \
engine mirrors them for the second player. Those three are the square's; a hex board \
has its own two: `rotate6` turns a vector six ways by 60 degrees, so \
`[[1,0]]` is the six neighbours and `[[1,1]]` the six diagonals, and `dihedral12` adds \
the mirror images, so `[[1,2]]` is the hex knight's twelve. Each lattice refuses the \
other's.

`mode` separates moving from capturing, which is what makes a pawn a pawn.

A piece also carries three things that are not movement. `royal` marks the piece whose loss the game may be built around. `value` is what the piece is declared to be worth, and it is read rather than decorative: it is what `rules.scoring` counts under `"material"`. `promoted_from` names what a promoted piece reverts to when it is captured, which only matters where captures go to hand.

`captures` is a different question from `mode`, and it is the one most likely to be overlooked. `mode` says whether a move may take *anything*; `captures` says *which kinds*. Omit it and a piece may take any enemy piece, which is what an ordinary game means. State it and you get a dominance relation the board cannot otherwise express: give rock `["S"]`, scissors `["P"]` and paper `["R"]` and every piece beats exactly one and loses to exactly one, so no piece is stronger than another and material stops being the thing you count. Three identical movers with a cycle between them is a real design and not a trick, and it is unreachable with movement rules alone. An empty list is a piece that never captures at all.

A hop may also say whose piece it jumps: `screen` is `"any"` by default, which is the cannon, `"own"` for a piece that vaults its own army, `"enemy"` for one that can only leap an opponent.

`condition` says *when* a descriptor may be used. `from_rank` counts from the mover's own back rank and is how a pawn's double step is stated; `first_move` gives any piece a move it has only once; `region` names the squares it may be used from. `piece.region` is the other half of the same idea and is stronger -- the squares that piece may ever stand on at all, which is the xiangqi palace and the two banks of its river. A region is the cheapest way to make one piece's power local instead of general.

`rules.wrap` takes the edges off the board: a piece that runs off one side comes back on the other. `files` joins the left and right edges, `ranks` the top and bottom, each `"none"`, `"left"`/`"right"` (or `"bottom"`/`"top"`) for a seam that may be crossed one way only, or `"both"`. Joining the files makes a cylinder; joining both axes makes a torus with no corner to trap a royal piece in. This is not a cosmetic change to the geometry -- an edge is what a mating net presses a king against and what a rook's line stops at, so joining one changes what every line on the board is worth, and a rule that fits on one line makes a genuinely different game rather than a variation on this one.

**The two players need not be given the same game, and five keys say so.** `setup.mirror: false` means the two armies are stated separately rather than one being reflected, and each placement then names its `player`, 0 or 1: that is how one side gets thirty-six pawns and no royal piece while the other gets an ordinary army. `promotion.to_by_player` gives the two sides different pieces to promote into. `wrap.files_by_player` and `wrap.ranks_by_player` give them different seams, so one player may cross an edge the other cannot -- and wrap edges are named absolutely, from the diagram with the first player at the bottom and file `a` on the left, so `{"0": "left", "1": "right"}` really is two different permissions rather than the same one said twice. `reach.mirror: false` sends both players at the same squares instead of at each other's end. Symmetry is still the default everywhere and still the right answer most of the time; these exist so that an uneven game can be *stated*, not so that every game should be one.

`setup.orientation` is a different question and applies however the armies are stated: `"mirror"` reflects the rank and leaves the file, as chess does, so the two armies face their counterparts down the file they started on; `"rotate"` turns the whole board 180 degrees, which is what a game whose pieces advance diagonally needs, because a rank-only flip would leave the two armies moving in unrelated directions.

**How a game is won.** `rules.win` is `"checkmate"`, `"royal_capture"` (take the royal piece outright, with no obligation to answer a threat), `"annihilation"` (clear the board of an army, and then nothing may be royal), or `"none"`, which means the only endings are the ones you state below. `stalemate` says what happens to a player with no move -- `"draw"`, `"loss"` or `"win"` -- and `checkmate: "win"` inverts the object of the game, which is what losers chess is. On top of that, and combinable with it:

- `reach`: arriving on a named square decides the game. `safe` demands the arriving piece not be capturable where it lands, which is the difference between a race and a race worth running. `"on": "leave"` decides it instead by the move that takes the piece *off* a named square, so standing there decides nothing and the piece must survive a turn on it before it can win by stepping away (Jeson Mor's knights and its centre square); it does not go with `safe`.
- `extinction`: running out of a named piece, or of `"any"` piece at all, decides it. `result` says for whom, so it states both the horde's loss and antichess's win. Listed types are counted together unless `per_type: true`, which counts each on its own, so running out of any one of them decides it (Extinction chess: `{"pieces": ["K", "Q", "R", "B", "N", "P"], "per_type": true}`, with a non-royal king). `per_type: "own"` counts each listed type on its own but only against a player who starts with it, for armies of different kinds: list both armies' kinds and each player loses by running out of any one of their own.
- `check_limit`: delivering that many checks wins. `bare_royal` states the result for """ + BARE_ROYAL_HOLDER + """, so `"loss"` is shatranj's rule.
- `must_capture` makes a capture on offer the only kind of move there is, and `forbid_check` makes giving check illegal. Both are restrictions on moving that end up defining what the game is about.

**Named subsystems, each a paragraph a player has to be taught.** `promotion` (a `rank`, `ranks` for a band several deep, or `squares` for a zone that is a shape, `mandatory: false` to let a piece decline, `to` for the choices, and `on_leaving: true` to promote a piece on a move that *starts* in its zone as well as on one that ends there -- shogi's rule, where a silver that declined on the way in may still promote on the way back out; it only matters for a piece that can move out of its zone, and with `mandatory` every such move must promote; and `only_lost: true` to offer a target only while the mover's side has fewer of it than it started with -- Grand chess's rule, a pawn promotes only into a piece already lost, every target must be in both armies' setup, and it reads the board, so a piece in a hand counts as lost and one a circe strike puts back home does not; it is refused beside `promote_on_entry`); `en_passant`, which is defined against the promoting piece and needs one; `castling` by `king_steps` or by `king_file`, and which alone does not follow a wrapped board's shape; `drops`, which brings captured pieces back into play in their captor's hands -- and whose `stock` gives each player pieces in hand *before* anything has been captured, either a count or `"unlimited"` for a supply that never runs out; a stocked piece with an empty `moves` list is a stone that is placed, blocks, can be taken, and never moves again; and `placement: true` makes it a placement phase -- while a player holds anything in hand, a drop is their only move, so the stock is played out (each piece onto its `region`) before anything on the board moves, as in Placement chess or Sittuyin, and only then may a royal start in the stock (once, and not also placed); and `either_face: true` lets a held piece come back as itself or as its other face, the piece whose `promoted_from` it is -- Kyoto shogi's rule, where a captured rook (held as the pawn it turns from) may be dropped as a pawn or a rook; each base may have one face, and no face may be stocked, royal or itself a base, and beside `explode_on_capture` only the piece a capture took reaches a hand (the blast's victims reach none); `drops.gating` (`{"ranks": [0]}` or `{"squares": [...]}`) is Seirawan chess's way in: a held piece never drops, and enters only as part of a move, onto the square a piece of its side leaves on its first move from where it started, on those rows or squares (a castle may gate on the king's square or the rook's), with `"from_captures": false` and a `stock` (`{"H": 1, "E": 1}`), and none of `region`, `no_doubled`, `no_mate`, `placement`, `either_face`; `explode_on_capture`, which takes the neighbourhood with it -- `immune` spares bystanders, and `by` and `of` say *which* captures explode at all (`"by": ["Q"]` is a game where only the queen is a bomb, `"of": ["P"]` one where only taking a pawn sets one off); leave both out and every capture explodes, as in atomic; and `blast` states the blast's shape instead of a `radius`, as steps from the capture square turned as a move's are (`"blast": {"vectors": [[1, 0]], "symmetry": "rotate4"}` is a cross that spares the diagonals), which must read the same for both players; `royal_facing`, the flying general; `forbid_immobility`, which stops a move stranding its own piece where it could never move again. Each of these is a sentence someone has to read before playing, and each should change how the game is played once they have.

`max_plies`, `repetition_draw` and `no_progress_draw` are the clocks. They are bookkeeping rather than rules and every game needs them, but what they do when they fire is a real decision:

`repetition_draw` alone may be `"never"` instead of a number, which removes the draw by repetition from the game entirely. That is not a long clock and it is not bookkeeping: a fortress becomes a fortress rather than a half point, a losing side cannot shuffle to safety, and a player who could once repeat their way out of a lost position now has to find something else. Use it when the game is *meant* to be decided -- a race, a game with a scoring count, a hierarchy where a stand-off should not be a way out -- and leave the number alone otherwise. Do not reach for it to hide a game that draws too often; that is a fault in the pieces.

`rules.scoring` decides a game that runs out of clock instead of drawing it. Set `count` to `"material"`, `"pieces"`, `"region"` or `"territory"` and whoever has more when the ply limit or the no-progress clock fires wins; equal is still a draw. `"region"` takes a list of squares and counts your own pieces standing on them, which is "hold the centre"; `"territory"` counts ground actually claimed and needs `rules.territory`. This is the Go ending rather than the chess one, and it changes what a game is about: a small advantage is enough on its own, rather than needing to be converted into a mate first.

`rules.territory` gives the board a memory: moving onto a square claims it for you. `mode` is `"permanent"`, where a claim can never be taken back, or `"repaint"`, where the other player can claim it later -- and those are different games, because only the second one lets ground be fought over twice. `win_at` ends the game the moment someone holds that many squares; leave it out and the colouring is scenery unless `rules.scoring` reads it. This is the one rule here that makes the pieces a means rather than the object.

`rules.obstacles` walls squares off: nothing may stand on one, and no line runs through one. `squares` lists them, `mirror` adds their reflections so a symmetric wall need not be spelled twice, and `blocks_leaps` decides whether a knight may vault a wall or is stopped by it -- two different games, and neither is the obvious default. A wall costs the engine nothing and changes the board a great deal: it is the cheapest way to make one part of the board mean something different from another, and unlike a seam it can be put exactly where the lines matter. Do not wall a square any piece starts on, and do not wall one a `reach`, `scoring` or `drops` rule names. Under `castling`, do not wall a square between a king and a rook it castles with, or one either lands on; a wall elsewhere on that rank is fine (Omega chess's corner cells).

`board.lattice: "hex"` says the board is a grid of hexagons. Write it as a width by height rhombus in axial coordinates and cut the shape you want with `obstacles`; a cell's six neighbours are `[1,0] [0,1] [-1,1] [-1,0] [0,-1] [1,-1]`, its six diagonals `[1,1] [-1,-1] [2,-1] [-2,1] [1,-2] [-1,2]`, and a rook is the first set as a slide, a bishop the second, a knight the twelve cells at distance three off both. Turn a vector with `symmetry: "rotate6"` or `"dihedral12"`, or list every vector under `"none"`, because the square symmetries do not fit a hexagon; set `setup.orientation: "rotate"` and `reach.mirror: "rotate"`, because the rank reflection is not a hex symmetry; give a piece a `facing` only with two axes; and state a promotion zone as `promotion.squares`, since a hexagon's far edge is a zigzag rather than a rank. `promotion.squares` works on a square board too, for a zone that is a shape. Hex is not a bigger chess. With 91 cells and thirty-seven moves a position, the game is slower and drawier at strength, which a promotion zone is there to fix.

`board.shape: "hexagon"` makes a hex board a regular hexagon. Set `width` and `height` equal and odd (11 is six cells a side), and the rhombus's corners are not cells at all, so no `obstacles` are needed to cut them. A hexagon's `wrap` joins its three pairs of opposite sides independently: `n_s` (top and bottom), `ne_sw` (upper right and lower left) and `nw_se` (upper left and lower right). Each is `"none"`, `"both"`, or the one side a piece may leave by (`"n"` or `"s"`, `"ne"` or `"sw"`, `"nw"` or `"se"`), for both players, or per player with `n_s_by_player` and the like. A piece that leaves by one side comes back in by the opposite one, straight across. Join a pair whose sides the armies do not stand on, or they meet across it on move one. With every side joined, each line through a cell runs through all the others, so an unbounded slider sees the whole board.

`rules.strike` changes what a capture *does*. With `effect: "demote"` or `"promote"` a taken piece is not removed but turned into another piece, following `chain` -- a list of `{"from": ..., "to": ...}` links. A piece with no link is captured outright, so the bottom of a chain is simply left out. `position` says where the two end up: `"stay"` leaves the attacker where it was and transforms the struck piece in place, `"swap"` exchanges the two squares, and `"advance"` is the ordinary capture and may only be used when nothing survives. `"choose"` is a fourth answer: the attacker takes the square and places the survivor itself, on an empty square of `landing` read from where the survivor stood -- `{"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}` is a king's step -- and a piece with no empty landing square cannot be taken, so a wall of pieces protects what stands behind it. Every royal piece must be listed in `immune`, because a king that turns into something else makes check a question about a piece that is not there yet. This is a deeper change than it looks: it does not alter a single legal move, only what happens when one is played, so material stops being conserved by trades and the whole vocabulary of a good exchange has to be relearned. A descending chain must terminate; an ascending one may cycle, and a game where taking a piece hands the defender a better one is a real design rather than a trick. A strike can also say what a *plain* capture leaves behind, with `effect: "capture"` and no chain: `"position": "stay"` removes the taken piece and leaves the capturer where it stood (rifle chess: nothing ever arrives on the square a capture was made on, so nothing can be recaptured there), and `"position": "home"` puts the taken piece back on its home square -- for a type with at most one setup square of each colour (a rook, knight or bishop), the one of the capture square's colour; for any other type (a pawn), the one on the capture's file, or failing that of the capture square's colour -- and removes it only if that square is occupied (circe; a reborn piece stands as if it had never moved). `switches_sides`, a list of non-royal pieces, makes a capturer change sides on the square it took (Andernach chess), with any effect. `by`, a list of piece ids, gives the strike to those pieces alone, as `explode_on_capture.by` does for a blast: every other piece captures as usual. `{"effect": "capture", "position": "stay", "immune": ["K"], "by": ["B"]}` makes the bishops archers, whose captures remove the target while the bishop stays where it stood, so a guard protects nothing against them; a shooter that should walk and shoot is a `"mode": "move"` descriptor beside a `"mode": "capture"` one. Each listed piece must capture something the strike reaches, the list may not name every piece that does, and it may not sit beside `switches_sides`. A shot (`"effect": "capture"`, `"position": "stay"`, no `switches_sides`) may sit beside `drops`, its victim going to the shooter's hand as any capture's does; every other strike may not, since a struck piece cannot both stand on the board and sit in a hand. And `explode_on_capture` with `"radius": 0` destroys only the capturer and its victim (kamikaze chess: every capture is a trade, and a king cannot capture at all).

`rules.contacts` is the same idea said in full, and the two may not both appear: where `strike` says what the *taken* piece becomes, a contact table says what **both** pieces become and says it per pair. `links` is a list of `{"mover": ..., "onto": ..., "mover_to": ..., "onto_to": ...}`; leave `mover_to` out and the mover is unchanged, leave `onto_to` out and the piece it took is removed, and a pair with no link is an ordinary capture. Beside `drops` every link must leave `onto_to` out: the removed target goes to the mover's hand. `position` and `immune` mean what they mean for a strike, except that `advance` is allowed as long as no link leaves a target standing -- which is where "a pawn that takes a rook becomes a knight" belongs. A link may carry `"own": true`, which makes it about the mover's *own* piece: the mover may move onto a friendly `onto` along its capture moves, and the link says what both become -- under `"advance"`, `{"mover": "P", "onto": "R", "own": true, "mover_to": "Q"}` merges a pawn and its own rook into a queen. An own link may not target an immune piece, and may not sit beside `must_capture`. A link's `mover_to` wins over a promotion the same move makes, so a link may name the mover itself where the mover has a promotion of its own: `{"mover": "Q", "onto": "R", "mover_to": "Q"}` beside a promotion that demotes a queen on every move says a queen that captures stays a queen (Einstein chess); without such a promotion it changes nothing and is refused. Reach for it when the interesting thing about a capture is *who made it*: a piece that takes the movement of whatever it last took, or a promotion earned by capturing rather than by reaching a rank. A table of one or two links is a rule; a table with a link for every pair is a chore, and `strike` says that much better.

`moves: []` is a piece that never moves, on the board from the setup or anywhere else: it blocks, it can be taken, it gives no check, and a royal one (Stratego's flag) can never step out of trouble, so under `royal_capture` taking it wins and under `checkmate` a check on it must be met by taking or blocking. It cannot promote by moving, castle, be a goal's or a capture rule's piece, or carry a speed; a player whose every piece has no moves and who has nothing to drop is refused. `explode_on_capture` with `"radius": 0, "of": [its id]` makes it a one-shot bomb that takes its taker with it.

`piece.facing` gives one piece an orientation and lets it spend a whole move turning. `axes` is 2 (forward and backward) or 4 (the quarter turns), and `initial` says which way it starts; the piece reads its own move vectors rotated, so a pawn facing right advances east. Give it only to a piece whose moves are *not* symmetric under the turn -- a rook reads the same rays whichever way it points, so turning one would be a move that does nothing, which is a pass, and the validator refuses it. A royal piece may not have one. It is the only property that lets a directional piece change its mind, and the cost is that every piece with one adds up to three legal moves per position, so a whole army of them is a branching factor rather than a game. `"turns": false` is a facing no move turns: each piece points the way its placement's `facing` says for the whole game (one entering later points its `initial` way) and adds no moves, which is how you write a pawn that marches backwards -- an ordinary pawn placed `"facing": "backward"`, never a second piece with its vectors reversed. Place some of them another way, or the facing changes nothing and is refused.

`rules.turn` makes a turn longer than one move: `moves` is how many, and `ramp` lists the lengths of the opening turns before that takes over. Be careful with it -- with two moves a turn whoever starts is a whole move ahead forever, and `"moves": 2, "ramp": [1]` is the fix. It is also the only rule that multiplies the branching factor rather than changing what a move is. Majority control and `rules.purse` play in it move by move (who holds a neutral is read afresh before each move, and each buy is a move paid from the purse as the last left it). Two rules may not sit beside a turn of more than one move: `npcs`, because a wild piece walks between turns and a legality filter would have to know whether the move it is testing ends the turn, and `en_passant`, because the window a double step opens is still open on the mover's own next move and a player could take their own pawn in passing.

`rules.promote_on_entry` closes a gap two of the rules above open. With `drops` a pawn can be put straight onto the far rank, and with a `strike` a knight can be demoted into a pawn standing there, and by default a piece that arrives in its promotion zone without moving there simply stands where it arrived -- a chess pawn on the far rank never moves again. Set it to `true` and a piece that enters play inside its promotion zone promotes at once, to the first piece its promotion rule lists, whether or not promotion is otherwise optional. It needs `promotion` and a way in -- `drops`, or a strike whose chain leaves a piece standing -- and it is refused where no promoting piece could ever reach its zone: crazyhouse's `drops.region` keeps the pawn off the far rank, so there it would be a rule about nothing. It is one sentence, and it turns a drop or a strike in the last rank into a promotion, which is a real change to what those two rules are worth.

`rules.npcs` puts a piece on the board that neither player owns. Each entry is one wild piece: `piece` names a declared, non-royal type, and `route` lists the squares it walks, starting on the first and stepping to the next one after every move, round and round; one square is a wild piece that never moves. It eats whatever stands on the square it steps to that its type's `captures` allows -- omit `captures` and it eats anything, give it `[]` and it is a moving wall -- and a piece it eats is gone. A royal piece standing where a wild piece steps next is in check from it when that type may capture the royal (so not a wild piece whose `captures` list leaves the royal out, nor one with `captures: []`). Taking a wild piece is worth its `bounty` (default its `value`), and `rules.scoring` with count `"points"` decides the clock on bounties collected. Routes may not share a square; to put several wild pieces of one kind on one route, give that entry `also_at`, the squares other than the first where each further one starts, in route order -- they walk it in step, and one that finds another in its way waits behind it. `mirror: true` places a second wild piece on the reflected route -- a route has a direction, so one wild piece is an asymmetry and a mirrored pair is not. `immune` names the only piece ids that may take it, and it is immune to every other -- "only a knight may take a wolf" is `"immune": ["N"]`, one line rather than a `captures` list on every other piece; leave the key out and anything may take it, and do not use it beside `drops`, because what may take a piece is a fact about the type rather than about who owns it and the immunity would follow it out of a hand. `respawn` (1 to 16) makes a taken wild piece come back: that many phases after it was taken (a phase follows every move, so 8 is four moves of each player; one a round in rounds) it returns to the first of its entry's starts (the route's first square, then `also_at`) that is empty or holds a player's non-royal piece its type may take, which it eats; while every start holds a royal, a piece it may not take or another wild piece, it comes back at the first phase one is free. One clock per entry and per mirrored half, and it starts again while the entry is still short -- the cure for a hunt that clears the board and leaves its points rule nothing to count; leave it out for a wild piece that, once taken, is gone. `turns_back: true` walks the route as a line rather than a cycle: first to last and back, never from the last square to the first; the piece faces forward at the start and turns round at each end, and where the square ahead holds another wild piece or a piece its type may not take it turns round and steps the other way (blocked both ways, it waits, turned), so whoever blocks it steers it. Only in a game without check (`win` `royal_capture`, `annihilation` or `none`, no `forbid_check`, no `reach.safe`), never on a one-square route or beside `also_at`. `pounce: "prey"` makes it a hunter: after each move, before any wild piece steps, it takes a piece of the player who has just moved that it attacks by its own type's capture moves, moving onto it (a royal first, then the most valuable, then the nearest, then the square nearest the mover's first rank and furthest to the mover's left), and it does not step that turn; so a piece left where it reaches is lost when its owner moves, and a royal it attacks is in check. Off its route it stands until it pounces again; back on it, it walks on. `pounce: "attackers"` is the hunter turned guard: instead it takes a piece of the player who has just moved that attacks it -- one that could capture it if it were that piece's turn and that its type may capture (a frozen or disarmed piece attacks nothing), in the same order -- so no piece may end a move attacking it, a royal that attacks it is in check, and it falls only to a piece that already attacks it when its owner comes to move: a line the other player's move opened, or a piece it stepped next to on its walk. `split` (2 to 8, only beside `pounce: "attackers"`) makes such a guard a hydra: the most pieces of its type that may stand at once. While fewer stand, it stays where it is and a copy of it grows on the square of each piece that attacks it, in the same order, eating it; past that room it moves onto the next as a plain guard does, and the rest are spared. A copy is the same wild piece in every way and acts from the next move; the count is the type's, counted as the phase goes, so two hydras share the room. Not below the type's starting count, not beside `respawn` or `drops` that fill a hand from captures, and every entry of a type must agree on it. Its type's moves must read the same for both players (no facing, region, move condition or owner-named hop screen), every entry of one type must agree on `pounce`, `split` and `bounty`, and it is refused in rounds and beside `respawn`, `turns_back` or `forbid_check`. This is a third actor rather than a rule about moves: it changes what a formation is worth and what check means without adding a legal move to anybody's list, and it is the one thing in the schema that acts when nobody is choosing.

`rules.duck` ends every turn with a second move by the same player: `{"piece": "D"}` names a declared, non-royal type nobody owns and nothing else names (not in the setup, a hand, a promotion, a capture list, a goal or `npcs`), and after each move the mover puts that piece -- the duck -- on any empty square but the one it stands on. It starts off the board unless `start` names an empty square, and the first player's first duck move places it. `"goes": "by_moves"` makes it walk its type's moves instead (moves that read the same for both players, onto empty squares); under the default `"anywhere"` write its type `moves: []`. With nowhere to go it stays, and the turn ends. The duck stands, blocks lines and screens a hop; nothing captures it and it captures nothing, so give its type no `captures`. A move that takes a royal ends the game with no duck move. There is no check beside it -- a move's legality would depend on the duck move after it -- so use `win` `"royal_capture"`, `"annihilation"` or `"none"`, with no `forbid_check` or `check_limit`; never beside `turn`, `simultaneous`, `portals`, `explode_on_capture`, `strike` or `contacts`, its type never sweeps (a player's sweep stops short of the duck, which nothing may capture), and `max_plies` must be even (each half is a ply). Its type may carry a `sight` with `whose: "any"`, so a duck can shield or freeze whatever stands round it wherever its players put it; beside `npcs` a wild piece waits at the duck, and the wild pieces step once a turn, after the duck move. Duck chess is chess won by taking the king with this rule: every threat can be walled off for one move, and every move asks where the opponent's duck will go.

`rules.portals` puts pairs of portals on the board, at most four, numbered by their place in the list. A piece whose move ends on a portal does not stop there: it comes out of the other portal of its pair, one king's step beyond it in the direction it was travelling -- the *sign* of its move, so a knight's (1, 2) and a bishop's (3, 3) both come out diagonally up and to the right -- and its move ends there, capturing, promoting or winning as on any square. If that square is off the board, a wall, its own piece, a piece it may not take, another portal or its own starting square, there is no such move; where a piece can reach one portal from two directions (round a wrapped board), each exit is its own move. A pawn's push comes out only onto an empty square and its diagonal only onto a piece it may take. Each pair is `{"orange": "c4", "blue": "f5"}` on two empty squares. A group's `moves` are how its ends move when a player moves one, written exactly as a piece's moves (`{"moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}], "orange": "c4", "blue": "f5"}` is a king's step): a portal only steps onto an empty square, so leave each move's `mode` out (it means `"move"` there), and each must read the same for both players. Leave `moves` out when nobody moves the group; a portal is not a piece, so never declare one in `pieces`. `moved_by` says per end who may spend a turn moving it (`{"orange": [0, 1]}` is either player, the default `[]` nobody); `passage` says per player which end their pieces may go in by (`{"0": "orange_to_blue", "1": "blue_to_orange"}`, or `"closed"`, when both ends are a solid block to that player, or the default `"both_ways"`). `"no_straight_back": true` bars moving a portal of the pair straight back onto the square it came from on the very next move -- without it two players can undo each other's portal moves until the game is drawn by repetition. The bar is refused where no move could ever meet it: it binds only the very next move, so either both players must be able to move an end (a `moved_by` listing both players on an end) or a turn must be two moves, and the group's moves must include a move back (a move in the opposite direction of another of them, as the `symmetry` of a king's step gives). A portal is never captured and nothing moves onto one. Not on a hex lattice, and not beside `npcs`, `strike`, `contacts`, `explode_on_capture`, `territory`, a sweep or a piece with a `region`. A door changes what distance means: a piece beside a portal is a move from the far side of the board, so the interesting games put the doors where both sides have to watch them. A group may instead be a map of three to eight ends, `{"ends": [{"at": "b3", "to": 1}, {"at": "e4", "to": 2}, {"at": "c6", "to": 0}]}` (a ring: each end leads to the next) -- each end leads to the end of its own group that its `to` indexes, or, with no `to`, is a way out only that nobody goes in by (several ends leading to one way out is a funnel); `enter` per end says whose pieces may go in (default both players) and `moved_by` per end who may move it. Two ends are always a pair: write them with `orange` and `blue` (and `passage` for one way), never with `ends`. `carries` says which pieces may go in by an end -- `{"orange": ["N", "B"]}` on a pair, a list on an end of a map -- and every other piece meets that end as a solid block; never empty and never every piece. At most four groups and eight ends in all. `"kind": "fold"` on a group makes its portals folds (leave `kind` out for the momentum portals above). A fold is not a piece: a piece of either side may stand in it, and is captured there as anywhere, and a fold may start under a piece in the setup. A move that ends on an empty fold that lets the piece in -- a leap, a slide, a hop, a pawn's push or its diagonal -- is two moves: stay on that square, or go through and end on the square of the fold it leads to, judged as a landing there (a push only onto an empty square, a capture step only onto an enemy piece it may take, never onto its own piece, promoting by that square). A move goes through one fold at most, and a move that only passes over a fold, or ends on one a piece stands in, is not changed by it. So a piece next to a fold threatens whatever stands on the far end. A fold a player moves (`moved_by`, the group's `moves`, no `first_move` condition) may land on a piece of either side, never on another portal; moving a fold never moves the piece in it. Castling, drops and en passant never go through a fold, and a move through one opens no en passant. Momentum and fold groups may share a game.

`setup.placements[].player: "nobody"` puts pieces on the board that neither player owns -- neutral pieces: `{"piece": "C", "player": "nobody", "squares": ["c4", "f4"]}`, mirrored with the rest of the setup (the images are nobody's too). They are in neither army, so no ending, material count or royal rule counts them; they block, and either player may capture one its `captures` allows (into the taker's hand under `drops`). By default they stand still. `pieces[].control: "majority"` on their type lets the player with strictly more of their own pieces next to one (a king's step away, six steps on a hex board; any of their pieces counts, royals included) move it as that player's move, the way its type moves, onto an empty square or capturing a non-royal piece of the other player -- never one of the mover's own, nothing nobody owns, never a royal; on a tie, none against none included, nobody may move it. Who holds it is read from the board at every move, with no memory, and moving it leaves it nobody's, so the next move may hand it to the other side. A held piece attacks nothing and gives no check (moving it may uncover a check, and a move of it that leaves the mover's royal in check is illegal). `{"majority": {"vectors": [[1, 0]], "symmetry": "rotate4"}}` names what next to means instead (here the four orthogonal neighbours). A majority type must read the same for both players (no facing, region, move condition, owner-named hop screen, sweep or promotion), never royal, a spy, resting or seeing. Neutral pieces are refused beside `rules.npcs`, `rules.duck`, `rules.portals`, `rules.simultaneous`, `rules.strike` and `rules.contacts`, and majority control beside `rules.territory`, `rules.reach` and `rules.explode_on_capture`. Neutrals in the centre make a scrum: every move near them asks whether to crowd one, break the opponent's crowd, or take it off the board.

`pieces[].bounty` (a whole number, 1 to 20) is what taking a piece of that type pays the player whose move took it, into the points tally that `rules.scoring` with `"count": "points"` compares when the clock runs out, or into the purse below; a bounty needs one of the two. Any type but a royal, a wild type (its `rules.npcs` entry carries its bounty) or the duck; a piece nobody owns pays its taker too. Taking your own piece pays nothing, and a piece removed by a blast or a wild piece's step pays nobody. Bounties turn a long game into a race for points: price the pieces you want hunted, and set `max_plies` so the race ends.

`rules.purse` gives each player coins and a market to spend them in: `"purse": {"start": 2, "market": ["P", "N"]}`, with `pieces[].price` (a whole number, 1 to 40) on every type the market sells. Each player starts with `start` coins (0 to 99); every bounty they are paid (above) goes into their purse too, which holds at most 99 (the rest is lost). Instead of moving, a player may spend a whole move buying one piece of a market type they can afford: the price leaves their purse and the piece is placed on an empty square of their first rank, or of `region` (squares written for the first player and reflected for the second), entering play as a dropped piece does. Supply is unlimited and a purchase cannot be undone, so it resets the repetition and no-progress clocks. A market type may not be royal, a spy, a wild type, the duck or a type with no moves, and needs a square in the region it may stand on. Coins also hire: a piece of a priced type placed for nobody (`"player": "nobody"`) may be bought by a player with one of their own pieces next to it (the king's step, or its `control` geometry), for its price, as a whole move; it becomes theirs where it stands. `"bribe": true` lets a player buy an enemy piece of a priced type next to one of theirs the same way, for twice its price, keeping the way it points; whoever paid last owns it. `market` is optional when coins can hire or bribe; a purse with nothing to spend on, and a price nothing charges, are refused, as is a price on a royal, a spy or a wild type. Not beside `rules.simultaneous`, portals or `explode_on_capture`; under `drops.placement` a held hand is placed before anything is bought. A bounty below the price is what stops the side ahead snowballing: a capture then pays for less than it took. Every turn asks whether to raise a piece or move one.

`rules.simultaneous` makes both players move at once: `"simultaneous": {}`. Each round each player chooses one of their own legal moves on the same board without seeing the other's, and both are made together, so a game becomes a sequence of guesses. Only two pieces moving onto the same square collide; crossing paths and two pieces swapping squares do not. A piece moving onto a square its target has just left lands there if that move could land on an empty square (mode `both`), and otherwise (mode `capture`, a pawn's diagonal, a cannon's hop) is void and stays put; two void moves at each other two rounds in a row bar that pair until either square changes. `pieces[].beats` (only beside the rule) lists what a piece eliminates in a collision -- leave it out and it is the piece's `captures`, so in an ordinary army every collision is a tie -- and is where a simultaneous game's character lives: a pawn with `"beats": []` stops being a suicide blocker, a royal that beats nothing cannot force a drawn meeting. A tie (both beat or neither) eliminates both unless `"tie": "initiative"` (the holder wins it and the token passes; the second player starts with it) or `"tie": "impossible"` (a promise the validator checks: the two armies use different pieces and every pair has exactly one winner, a circle of rock, paper and scissors). There is no check: `rules.win` must be `royal_capture`, `annihilation` or `none`, and an ending both reach in one round is a draw (unless `both`, below). Not beside castling, en passant, a sweep, `forbid_check`, `check_limit`, `royal_facing: "forbidden"`, a turn of more than one move, a portal a player moves, a strike or contact `position: "choose"`, a strike `position: "stay"`, an own contact link with `position: "swap"`, `drops.no_mate`, or a royal in a placement stock. Such a game is measured like any other, and the loop plays it with round players: the screen and the full roster put `matrix:nodes=16N` wherever an alternating game would have `minimax:nodes=N`, sixteen times the budget, since a round player below about a round's whole matrix of move pairs plays greedy's move. Two switches inside `rules.simultaneous` work against a royal chased round after round (an attacked royal must move every round, and so must a piece it attacks back, so the two can lock a game into a dance that ends in repetition), and the third remedy is a rest (`"rest": 1` on every piece, below: nothing runs or strikes two rounds running and every attack gives a round's warning): `"swap": "collision"` (two pieces each moving onto the other's square meet as a collision, decided by `beats` and the tie, instead of passing; refused where no piece of one player can take one of the other that takes it back), and `"both": "score"` (an ending both players reach in one round goes to the player ahead under `rules.scoring`, which it requires, and is a draw only on equal scores). Leave each out for its default.

`pieces[].spy: true` makes a piece type a spy, and who still has one decides how the game is played. While both players have a spy on the board they take turns as usual. When one player loses their last spy (captured, collided, blown up, promoted away), they first play one ordinary reply move if it is their turn, so a spy's taker can be retaken; from then on the game is played in rounds in which the player without a spy announces their move first and the other chooses knowing it, both resolved together as a round of `rules.simultaneous`; once neither has a spy, both choose in secret. A spy game must declare `rules.simultaneous` (so every refusal above applies to the whole game, including its turn-taking phase), both players must start with a spy on the board, a spy may not be royal, something of the other player's must be able to remove it, and nothing may bring one back: no promotion to it, no strike chain or contact link into it, no stock holding it, no `drops.from_captures`, no `strike.switches_sides` naming it. A spy is best weak (a step that captures nothing): the information it buys is the point, and guarding it should cost something. Without a rest on its pieces (`"rest": 1`) an announced round favours the spy's holder heavily, since the side that sees the move can always step a threatened piece away and the announcer can never recapture; the rest restores the recapture, and it rests a piece in the turn-taking phase too. Only random players play a spy game so far, so the loop's screen cannot yet test one.

`pieces[].speed` gives a piece type a speed in a game played in rounds: an integer of at least 1, a rank (leave it out for 0, the slowest). Speed is where a round game's tempo lives. When the two pieces moving in a round have the same speed, the round is resolved as above. When they differ, the faster move is made first, as an ordinary alternating move, and if it ends the game the round ends there; then the slower move is made if its piece still stands where it was and the move can still be made on that board (onto a square the faster piece now holds it is a capture if the piece may capture there), and otherwise it is void (the piece stays) or lost (the piece was taken or changed). So a slower piece cannot flee a faster strike, a faster piece that ends next to a slower one it threatens takes nothing of its own back, and pieces of different speeds never collide or swap. Only the order is a rule: the speeds in use, 0 counted for every type that names none, must be 0, 1, 2, ... with none skipped, and some piece of one player must differ in speed from some piece of the other. A mirrored army keeps the collisions (every type meets its own mirror at its own speed); two armies whose types never share a speed have none at all, and then `beats`, `tie` and `swap` are refused as inert. Price speed against value: a royal slower than most enemy pieces is caught by any faster attacker it cannot answer with a faster piece of its own, which makes a short game.

`pieces[].rest` makes a piece type rest after it moves, in any game, by turns or in rounds: an integer from 1 to 4 (leave it out for 0, never resting). A piece that makes its player's move (moves, turns in place, is dropped, or strikes without leaving its square) may not make that player's move during that player's next `rest` turns, counted on its own player's turns only (a round is a turn of both), as the type it is after the move (a pawn promoted to a queen rests as the queen). A resting piece still stands: it attacks, blocks and can be captured, and it gives check, which must still be answered, so in a checkmate game a resting queen keeps its grip; in a `royal_capture` game or in rounds a resting attacker threatens nothing for its rest. When every move a player has is a resting piece's, the rest is lifted for that turn, so the rest never leaves a player without a move and never mates on its own. Captured, moved or changed by the other player, a piece rests no more. Never on a wild piece, and not beside a turn of more than one move (`rules.turn` moves or ramp). Rest is how to make a strong piece cost tempo: a queen that rests 2 moves one turn in three.

`pieces[].sight` gives a piece type lines of sight: a list of 1 to 4 entries, each with a slide's geometry keys (`vectors`, `symmetry`, `range` with 0 to the edge and the default, `min_range`) and three of its own: `effect` (required: `"freeze"`, `"disarm"`, `"shield"` or `"command"`), `whose` (required, counted from the sight's owner: `"enemy"`, `"own"` or `"any"`) and `affects` (optional, a non-empty list of piece ids; leave it out for every type, and it may name the piece's own type). A sight is a ray, which sees each line up to and including the first piece on it, unless it says `"through": true`, which makes it an aura that sees past every piece up to its range; a wall or the edge ends both, and a one-square sight (`"range": 1`) or a knight-shaped one (`[[1, 2]]` by `dihedral8`, range 1) is the same either way. Every piece of the side `whose` names, of a type `affects` names, standing on a square a sight sees, is under its effect for as long as it stands there, read from the board as it is with no memory: a frozen piece may make no move of any kind (no step, capture, turn in place, castling as king or as partner, strike or promotion); a disarmed piece may move but may not capture (no capture, en passant, strike, contact, or sweep over a piece); a shielded piece cannot be captured or taken in passing. Attack reads the effects: a frozen or disarmed piece attacks nothing and gives no check, and a shielded piece is attacked by nothing, so a shielded royal is never in check. A check may therefore be answered by freezing or disarming the checker (moving, dropping or promoting a piece whose sight reaches it) or by shielding the royal, and a move that unfreezes an attacker of the mover's own royal (moving its freezer away, stepping into the ray) is illegal. A sight works whatever effect lies on its own piece, so two pieces can freeze each other and only a third can free them; two sights of one effect are one effect. There is no lift: a player whom freezes leave with no legal move has none, so the game's `rules.stalemate` decides, or checkmate if in check; a total freeze is a weapon, and `"stalemate": "loss"` makes it a win. In rounds the effects of the board the round starts on decide what each player is offered, and the round resolves as always (a disarmed piece still wins a collision by `beats`). On a wild piece (a `rules.npcs` type) a sight must say `whose: "any"`, may not command, and must read the same for both players (a symmetry such as `rotate4`): a wild aura freezes, disarms or shields both players' pieces alike; a wild piece never takes a shielded piece (its step onto one waits, and a shielded royal on its next square is not in check). Refused beside `rules.portals` and `rules.explode_on_capture`; a `shield` is refused beside a `sweep` move; and a sight that can never bind is refused (nothing it affects is ever held by the side it affects, a freeze on types with no move, a disarm on types that can capture nothing, a shield on types nothing can capture). Madrasi chess is every type freezing its enemy twin with a ray equal to its capture geometry; an immobiliser is a range-1 freeze on every enemy type. A `"command"` sight hands the affected pieces to their opponent as well: that player may also move each commanded piece, as its own move, the way the piece's owner would move it (its owner's forward, its first-move rights), onto an empty square only, never taking the moving player's own pieces; with `"captures_own": true` (only beside `command`) the commanded piece may also be made to take a piece of its own side that its `captures` list allows, never a royal and never a shielded piece, and not while disarmed. A royal is never commanded (an `affects` naming one is refused; without `affects` royals are skipped), so check keeps its one meaning: a check may be answered by commanding the checker away or a piece into the line, a commanded move may give check, and one that leaves the mover's own royal attacked is illegal. A commanded piece never castles, takes or opens en passant, drops or promotes (a move that would have to promote is not offered), and it counts as having moved; a frozen piece is moved by nobody. `command` is refused beside `rules.simultaneous`, `rules.reach`, `rules.territory`, `pieces[].rest` and a `sweep` move. Beside `rules.strike` or `rules.contacts` a command is allowed, since its moves land only on empty squares and neither rule touches them, but its `captures_own` is refused there, as beside drops that fill a hand from captures.
"""


@dataclass
class Proposal:
    """One model response, whether or not it was usable."""

    spec: dict[str, Any] | None
    errors: list[str] = field(default_factory=list)
    attempts: int = 1
    parent_hash: str | None = None

    @property
    def accepted(self) -> bool:
        return self.spec is not None

    @property
    def hash(self) -> str | None:
        return spec_hash(self.spec) if self.spec else None


def propose_ruleset(
    client: Claude | None = None,
    parent: dict[str, Any] | None = None,
    feedback: str = "",
    avoid: list[str] | None = None,
    goal: str = "",
    failures: str = "",
) -> Proposal:
    """Ask for a new ruleset, optionally as a revision of ``parent``.

    ``failures`` is prose about revisions of this same parent that were already tried
    and did not work. It costs nothing -- the archive already holds it -- and it is the
    difference between a model that rediscovers a dead end and one that does not.
    """
    client = client or Claude(model=PROPOSAL_MODEL, effort="high")
    schema = rule_spec_schema()

    stable = "\n\n".join(
        [DESIGN_BRIEF, SCHEMA_PREAMBLE, "The schema:", json.dumps(schema, indent=2)]
    )
    system = cached_system(client.model, stable)
    user = _build_task(parent, feedback, avoid or [], goal, failures)

    # A rule spec cannot be a *response* schema -- 67 of its properties are optional
    # against a grammar-compiler limit of 24, because a game states the rules it has and
    # nothing about the rest. `client.constrainable` is the measurement, asked rather
    # than assumed so that a schema which one day fits gets its guarantee back without
    # anyone remembering to come here.
    #
    # The schema is in the system prompt above either way, which is what makes this a
    # change of where the contract is stated rather than whether there is one. What is
    # given up is the guarantee that a reply *parses* against it; what is not given up is
    # any protection, because `validate_spec` below is the thing that ever protected the
    # pipeline.
    response_schema = schema if constrainable(schema) else None

    errors: list[str] = []
    for attempt in (1, 2):
        candidate = client.structured(system=system, user=user, schema=response_schema)
        # In the current spelling, so a rule the model wrote at its off value is tidied
        # away rather than refused, and what is kept is what the library would keep (Q1008c).
        if isinstance(candidate, dict):
            candidate = current_spelling(candidate)
        errors = validate_spec(candidate)
        if not errors:
            return Proposal(
                spec=candidate,
                attempts=attempt,
                parent_hash=spec_hash(parent) if parent else None,
            )
        if attempt == 1:
            user = (
                f"{user}\n\nYour previous attempt was rejected by the validator:\n"
                + "\n".join(f"- {error}" for error in errors)
                + "\n\nReturn a corrected ruleset. Fix every listed problem; do not "
                "introduce new pieces or rules while correcting."
            )

    return Proposal(
        spec=None,
        errors=errors,
        attempts=2,
        parent_hash=spec_hash(parent) if parent else None,
    )


def _build_task(
    parent: dict[str, Any] | None,
    feedback: str,
    avoid: list[str],
    goal: str,
    failures: str = "",
) -> str:
    parts: list[str] = []

    if parent is None:
        parts.append(
            "Design a new game. Start from the qualities in the brief rather than from "
            "chess: reaching for a chess variant is the easy move and usually the "
            "weaker one."
        )
    else:
        parts.append(
            "Revise the game below. You may change pieces, board, setup and rules "
            "freely -- a revision that only tweaks a number rarely changes how a game "
            "plays. Prefer removing something over adding something."
        )
        parts.append("\nThe current game:\n")
        parts.append(describe_spec(parent))
        parts.append("\nAs data:\n")
        # ``provenance`` is withheld along with the prose: it names a human author, and a
        # model shown one has every reason to carry it over into a game they did not
        # write. It is not in the response schema either, so a proposal cannot invent one.
        withheld = {"description", "provenance"}
        parts.append(
            json.dumps({k: parent[k] for k in parent if k not in withheld}, indent=2)
        )

    if feedback:
        parts.append(
            "\nHow it actually played when simulated. These are observations from real "
            "games, not opinions:\n"
        )
        parts.append(feedback)

    if failures:
        parts.append(
            "\nAlready tried from here, and what went wrong. These are outcomes rather "
            "than opinions: do not repeat them, and do not treat a whole direction as "
            "closed because one attempt at it failed.\n"
        )
        parts.append(failures)

    if goal:
        parts.append(f"\nFocus for this revision: {goal}")

    if avoid:
        parts.append(
            "\nThe following mechanic fingerprints have already been evaluated. A "
            "proposal that reduces to one of them is wasted work:\n"
            + "\n".join(f"- {digest}" for digest in avoid[:40])
        )

    parts.append(
        # The shape of the reply is asked for here because it can no longer be enforced
        # on the wire: a rule spec is too big for the grammar compiler, so this sentence
        # is the whole of the format contract. `client.json_object` recovers a fenced or
        # prefaced object anyway, and `validate_spec` is what actually decides.
        "\nReturn one complete ruleset as a single JSON object matching the schema, and "
        "nothing else -- no prose around it and no code fence. In `description`, say in "
        "two or three sentences what the game is about and what decision the player "
        "faces -- not a list of the rules, which are already written above it."
    )
    return "\n".join(parts)
