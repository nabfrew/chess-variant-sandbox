# Changelog

What this project can do now that it could not before, in plain language, newest first.
Every entry is something you would notice — at the command line, on the web page, or in the
numbers. How the work was organised is not in here. Pay particular attention to the style section of CLAUDE.md

The site's *What's new* is this file, so write an entry to a visitor: what they would notice, in their words, with no
command, no path in the repository and no *you decided*. An entry that only a maintainer would notice (a command, a
folder, a deploy, a decision taken) begins with the line `*For maintainers.*`, alone as its first paragraph; the local
application shows it as written and the site skips it (`chess_variant_sandbox/web/changelog.py`, `MAINTAINERS`).

---

## 2026-10-21

### Three games that show resting pieces

Three new previews show how long-resting pieces change a game: Gardner where the queen and rooks tire, breather where
pieces rest by weight, and chess where the queen sits out a turn after it moves.

### Two new games, designed by a model from the rules alone

*The golem well* and *kraken ford* join the library. A model wrote each from the published rules and nothing else; both
measured well over four runs.

### The watering hole is back

The watering hole now starts its animals one row back from the hole, which gives the guards time against the stronger
computer player, and it is a full library game again. The version with speed is a preview for now.

### Faster return visits

Coming back to the site now downloads about a fifth of what it did: the page's styles and scripts are kept by your browser
until they change.

### Follow what's new

This page is now readable without scripts, and two feeds let a feed reader follow it: one for what changed, one for
new games.

### Share a proven position

Every proven position has a page of its own with a picture of the position, so a link to it shows the board. A button
copies the link. The page says whose move it is and what to find, never the move.

### The rule language on a page of its own

The description of the rule language is now a readable page, with a link on every heading so you can point someone at
exactly the part they need. The landing page and the designer link to it.

### Set how long pieces rest in the designer

One choice beside adding a piece sets how long every piece waits after it moves, and each piece's card can set its own.
The designer says in words what each piece rests.

### The designer counts in rounds

In a game played in rounds, the designer now counts the longest a game may run and the opening choices in rounds, and the
rule itself is called playing in rounds.

### Every rule easy to find in the designer

Each piece's card now says how it takes its turns: its speed, what it beats and whether it is a spy in a game played in
rounds, and how long it rests in any game. You can find any rule by typing a word such as rounds, atomic or walls.

### Proven positions for more games, kept up to date

When a game joins the library, its proven positions are now found and proved by themselves. Makruk offers its first,
and the rest of the library is being worked through.

### Badges for proven positions

Win every proven position on a game against the hardest computer player and you keep a badge for it. Do that on 5, 10
and 25 games for three more. They stay yours even when more positions are proved later.

### A page for each game's proven positions

Each game's proven positions now have a page of their own: the board beside the list, ticks on the ones you have solved,
how much of the game you have done, and previous and next, so you can work through them without leaving. Your profile
shows how much of all of them you have solved.

### Rest for each kind of piece, and a stronger opponent in rounds

How long a piece must wait after it moves is now set for each kind of piece, in any game, whether players take turns or
move at once. The hardest computer player in games played in rounds is much stronger. Every score was measured again
for these changes: some games moved, both watering holes now end too quickly to count as viable, and courier rose.

## 2026-10-20

### Make a game played in rounds in one step

The designer now opens with how the players take turns: one after the other, in rounds, or more than one move a turn.
Choosing rounds makes every change the game needs, lists them, and lets you undo them. When a rule can't go in a game,
the designer gives every reason at once, in plain words, beside the rule it concerns.

### The rules document helps you design, not only write

The published description of the rule language now opens with a map of what each family of rules is for, and adds a
section on how a game that passes every check can still play badly. It also says exactly when a wild piece gives check.

### Fewer, better proven positions

The proven positions no longer include the ones that are too easy, where the defender's moves change nothing; the list
says how many were left out and why. A position stays only if it has exactly one winning move and the quick computer
players don't find it: 21 across the seven games, 3 on *Intransitive*. A game where every position was too easy says so.

### Scores kept from a sample say so

When the engine changes in a way that should not move any game, a score can now be confirmed by replaying a sample of its
games instead of all of them. Such a score is marked as kept from a sample.

### A newer computer player on Intransitive and Gardner

The learned computer player on *Intransitive* and on Gardner is a newer one, trained on its predecessors' games. Gardner's
score moved up with it.

### Links to the site show a preview

Paste a link to the site, or to any game on it, into a chat or a post and it now shows a card with the page's
title, a short description and a picture. A game's card shows that game's own starting position.

### Asking for an invitation reaches a person

When you ask for an invitation, the person who runs the site now gets an email about it straight away, so
your request no longer waits to be noticed. The privacy notice says so.

### Circe mortal diagonals

A new game to try: circe chess where a captured queen or bishop is gone for good, while a captured knight, rook or pawn
is still reborn on its home square. The endless capture loops of plain circe stop, and games end in checkmate; it scores
well above circe on the project's measure of how interesting a game is.

## 2026-10-19

### A new map of the games

The map of games is redrawn from scratch. Games now sit near the games they share their rarer rules with, families are
found and named automatically from what marks them out, each game lists the five most like it and what they share, and
the unusual games are listed on their own. Adding a game no longer reshuffles the map. A game's own page lists the same
games like it as the map does, in the same order.

### The computer varies its play from game to game

Every game on the site used to start from the same random seed, so in a given position the computer always answered
the same way. Each game now starts from its own seed, and a finished game still replays exactly.

### The watering hole, easier to read

The white elephant, camel and wazir were drawn see-through, so the board showed through them; they are filled white now.
Both watering hole games, and the round versions of Gardner's minichess, now say first how you win.

### Every game has its own address

Each game now lives at its own web address, and the page carries the game's name, description, score and nearby games
even before any script runs. A link to a game shared anywhere now names that game, and older links still open the right
one. Search engines are now welcome: the site lists its games for them, and every page says what it is.

### The board alone, full screen, on a phone

On a phone or tablet, the play page now has a *full screen* button: the board alone fills the screen, upright or on its
side, with the side to move and the result still in view. Turning the phone keeps the game where it was, and the back
gesture returns to the page.

### Proven positions are back on their games' pages

Since 16 October the site had quietly lost track of some of its stored files: the proven positions, the ladders of
computer players for a few games made here, and designs members saved in that time. They are all restored. The filter
for games with proven positions now finds the seven that have them. Six of the project's own early designs are now listed
in the library, each with a sentence on why.

### Find the games with proven positions

The games list can now be narrowed to games that have proven positions to solve: open *narrow it down* and choose
*has some to solve*. Seven games have them today. *The watering hole by speed* has also joined the library as a full
game, with a ladder of computer players to play against.

### Speed: in games played in rounds, faster pieces move first

In a game where both players move at once, pieces can now have a speed. When two moves meet in a round, the faster
piece's move happens first, and the slower move may find its target already gone, or its own piece already taken. The
page shows the order: the faster move plays, then the slower one. Three new games show it. *The watering hole by
speed*, where cat, rat, dog and elephant race to the water and arriving first matters as much as who eats whom, measured
as a real game, with every game decided at the hole. Two versions of *gardner in rounds*, one with a swift king and one
with swift knights, are offered as previews: both turned out drawish, and in the knights' version strong players trade
the two fast knights on the first move.

### Proven positions: endgames the computer has proved are won

Seven games' pages now have a *Proven positions* section: endgames taken from the computer's own games, each one proved
won for the side to move by a deep search. If you are signed in, you can play one out against the hardest computer
player, and the game's own ending says whether you converted it. Where only one move wins, anyone can try to find that
move, and the answer comes from the stored proof; a move the search could not settle is said to be unsettled, never
wrong. There is no timer and nothing to keep up: a miss is just a game you lost, and the positions you have solved show
on your own page, for you alone. About eight hundred positions are on offer across the seven games.

### One fewer preview

*Gardner in rounds resting* leaves the previews. It turned out to be *breather* with two of its rules taken away, and it
played the same way, so it offered nothing *breather* doesn't. Its page and games are kept. The one preview left is
*gardner with spies resting*.

## 2026-10-18

### Circe now follows the real Circe rule

In *circe*, a captured piece comes back to life on one of its starting squares. A captured rook, knight or bishop now
returns to the starting square of the same colour as the square it was taken on, as in the published game, so a bishop
always keeps its colour. Before, it went back to the starting square on the same file, which differed for captures on
six of the eight files. The game was measured again under the corrected rule and now counts as a playable game, where
before it fell just short.

### Every game says what it is about

Sixty-six games have new descriptions. Most used to be a list of rules, which the page already prints beside them,
or explained a game by comparing it with another game in the library that a visitor had no reason to know. Each one
now says what the game is exploring and the choice it keeps putting to you: in a game where a capture upgrades the
piece it strikes, whose gift is each exchange; in a game with a door either side may move, whether to move the army
or the road. Terms from other variants and from shogi are explained where they appear, and where a game is a
published or traditional one, it says so. No game plays any differently. While checking every rule the old texts
stated, a few turned out to be loosely put (a pawn that can turn may face any way in one move, not only a quarter
turn; two games with one-way edges are not exactly fair), and the new descriptions say what the games really do.

### The Gardner game where both players move at once is now called *breather*

Its old name and description explained it by contrast with an earlier experiment most people never saw. Now the
description says what the game is: both players pick a move in secret and move together, a piece that has just moved
must sit out the next round, and you win by taking the king. Nothing about how it plays has changed, and its score and
computer opponents are the same. Several other games' descriptions were reworded the same way, so each one reads on
its own.

### Fewer previews, and the ones left are worth a look

Six preview games are leaving the Previews tab. Each was made to show off a new rule, and each ended up mostly drawn or
leaning heavily to whoever moved first, so they made poor games. Two stay: *gardner in rounds resting*, which plays evenly
and is decided most of the time, and *gardner with spies resting*, the one game that shows the spy.

## 2026-10-17

### Riders and leapers joins the library

*Riders and leapers* moves from the previews into the library, with a score and computer opponents of its own. It is a
game with no king and two different armies: one side has rooks, bishops and guards, the other champions, knights and
elephants, and a side loses the moment any one of its own three kinds of piece is gone, which no other game in the library
asks. A pawn that
reaches the far rank can bring back a kind that is about to run out, so every trade is a question of which kind it leaves
short. It is played on a six by six board and its games are short and sharp.

### Games with spies against another person

A game with spies can now be played against another person, each of you moving when you like. While both spies
stand you take turns, as in any game. Once a spy is lost, its player replies with one move, and then each round the
player without a spy announces a move first, which both of you see on the board, and the spy's holder answers it
before both moves are made at once. Once both spies are lost, each of you seals a move in secret, as in the other
games played in rounds. If you switched on move emails, each one says what is asked of you: your reply, your
announcement, your answer to the move announced, or your sealed move.

### Stronger computer opponents in games played in rounds

In a game where both players move at once, *easy*, *medium* and *hard* now weigh sixteen times as many rounds
before each move as they did, so each of them looks past the round in front of it. Before, *easy* played exactly like *greedy*, and *hard* came out
weaker than *medium* on both such games in the library. The strongest still answers in well under a second, the
three are rated afresh on each game, and without an account you can still play every one of them up to *hard*.

### How often the computer played the same game again

Each game's page now says how many of the games behind its score were different games. On a few
small boards the computer players play some games over again, move for move, and where that is
enough to move the score, the page says the score rests on repeated games and gives the figure
counted once. The test that rules out a game won by one side now counts each different game once
too, so *Gardner with spies* and *Gardner with spies resting* are no longer ruled out for it:
their strongest players won nearly every game from the first side, but those games were a handful
played over and over. No score changed.

### Computer players guard their spy

In a game with spies, the computer players now count holding the only spy as worth a piece, so they guard their own, hunt
the other side's and no longer trade it for a pawn. Every game without a spy plays exactly as before.

### Two games played in rounds join the library

*The watering hole* and *Gardner in rounds against the chase* move from the previews into the library, with scores and
computer opponents of their own. They are the first games in the library where both players move at once.

### Two previews with spies

*Gardner with spies* and *Gardner with spies resting* put a spy on each side of Gardner's small board. While a side's spy
stands, the other side announces its move before both move at once. Both are previews for practice: in the first, the
spies are traded away early and the rest is a draw; in the second, where a piece that moved must rest, the spy matters,
but the first player still wins too often for the game to be scored as fair.

### Games with a spy can be scored

A game with a spy can now be given a score like any other game: the computer players play it through every phase, and
the measurements read each turn and each round by what the players knew when they chose, including a round in which one
player saw the other's move first. A spy game's score also says how often a spy was lost, how early, how, and how long
each phase of the game lasted. The automatic game designer can now try spy games too. No game in the library has a spy
yet.

### Games with a spy can be played on the board

A game with a spy can now be played against the computer, or by two people at one screen, from its first move to its end.
While both spies stand, the players take turns. Once one is lost, play goes to rounds: the player without a spy shows
their move first, drawn on the board as a solid arrow, and the other answers knowing it, then both moves are made at
once. Once both are lost, both players choose in secret. The line above the board says which of these the game is in.
No game in the library has a spy yet; one you build with a spy can be tried on the designer's board. Games between two
people at their own screens do not take a spy yet.

### Four new previews: a royal chase in rounds, and its cures

In Gardner's small board played in rounds, a king could be chased round the board for the whole game. Four new previews
try the cures. When a piece that moved must rest the next round, the chase all but ends and most games are decided; with
all three new rules together, *Gardner in rounds against the chase*, nearly every game is decided and its score is read
as a good game.

### Resting pieces and collisions are drawn on the board

In a game where a piece that moved must rest the next round, a resting piece now wears a pause sign. When two pieces try to
swap squares and collide, a burst is drawn between their squares, and the replay shows them meeting half-way. When both
royal pieces fall in the same round, a sentence says who won and why. The computer players also stop walking a resting king
onto a square an enemy piece can still reach next round.

### New rules for games where both players move at once, and for losing a kind of piece

The rule language has four new choices. In a game where both players move at once, a piece that moved can be made to
rest for the next round; two pieces that try to swap squares can be made to collide, so both are lost; and when both
players reach an ending in the same round, the one ahead on the score can be made the winner instead of calling it a draw.
These answer a royal piece being chased round the board for the whole game. Separately, a game whose two armies are made
of different pieces can now make a side lose as soon as any one of its own kinds runs out.

A spy piece, which sees what the other side plans, is in the rule language too, but no computer player or board plays it
yet; those come next.

## 2026-10-16

### Stronger computer players measure the games where both players move at once

In a game where both players move at once, the weakest computer players used to search so little that they made the same
move as the simplest player every round, so the ladder of opponents looked flatter than it is. Every rung now searches
sixteen times as far. The watering hole's score rose from 0.50 to 0.61, because its players now separate clearly. Gardner in
rounds shows no score for now: its games end in a draw so often that two of its four measurements cannot be read.

### Standoffs in a round game have their own colours

In a game where both players move at once, the board marks two kinds of standoff: a round in which nothing happened, and
a pair of moves barred from being repeated. Both were drawn in the same dark green as a move you have picked but not
yet sent, so the three were easy to confuse. A standoff after an empty round is now an umber dashed link, a barred pair a
blue barred link, and your own picked move keeps its green, in light and dark themes.

## 2026-10-15

### Play a game in rounds against a friend, a round at a time

A game where both players move at once can now be played against another member by correspondence, like any other game between two people. Each round you choose your move and seal it; your opponent sees only that your move is in, never which, until theirs is in too, and then both are made at once. A sealed move is final, so the board lets you change your mind until you press *Seal my move*. If you switched on move emails, you get one when your opponent's move is in and one telling you how each round went. Each round has five days: whoever has not sealed by then loses, and if neither of you has, the game is drawn.

### Previews: four games where both players move at once

The games page has a new tab, *Previews*, with four games played in rounds: three versions of Gardner's five by five minichess, and the watering hole, a new game on a seven by seven board where elephants, cats, dogs and rats race for the water hole in the middle and every animal eats exactly one kind of the other side's. The watering hole is the one we think plays best. Each carries a *Preview* label: a new kind of game, still being tested. You can play any of them against the computer, at five strengths, or watch two computer players. For now every game on a preview is practice: previews have no leaderboard, are kept out of the library's rankings and comparisons, and their scores are a first reading, not yet trusted.

### Games where both players move at once

A game can now be played in rounds: both players choose a move in secret, the two moves are revealed together, and then both are made. When two pieces land on the same square, the game's own table decides which survives, or whether both are lost. The play page shows both moves arriving together, explains each clash, and marks a capture that has been barred for repeating. The computer opponent chooses its move without seeing yours, and two people can also play at one screen. You can try it from the designer by turning on *both players move at once*, or play the four previews below.

### The rules guide matches what the site accepts

The rules guide and the designer no longer say a piece may move in at most eight directions, or that every piece needs a move of its own. Neither was ever enforced when a game was checked, and some games in the library already go beyond both. A piece that only ever enters the board from the hand may now be written with no moves at all.

## 2026-10-14

### The learned player on the leaderboard

On Intransitive, racing kings, gardner, king of the hill and rps-v2, the learned player is no longer an experimental practice opponent: you can play it for rating, and it has its own place on the game's one leaderboard, beside people and the other computer players. On Intransitive and racing kings it is now the strongest player there. Earlier games against it stay as practice.

### Five games measured by their learned players

Where a game has a trained computer player that has proved itself, that player now does the measuring in place of the ordinary one. Five games were measured again this way. Racing kings changed most: with players that understand it, it now passes as a proper game where before it was ruled out. The other four moved only a little. Each score keeps its earlier value in its history.

## 2026-10-13

### The rules guide, current and versioned

The designer's link to the rule language now opens the site's own copy, which always matches the rules the site checks, and its
first line says which version of the language it describes. A game written from it, by hand or with your own AI, declares that
version. The public copy of the code is now refreshed with every update to the site, so it stays in step too.

### Stronger learned opponents

Racing kings now has a learned computer opponent, trained on the game itself, and its second generation is much stronger
still: it wins about nineteen games in twenty against the ordinary hardest opponent. In rps-v2 the learned opponent has been replaced by a newer generation that beats the old one, and wins
nearly two games in three against the ordinary hard opponent. Both are offered as the experimental learned player.

### A plainer note when a score rests on fewer runs

When the computer players could not be told apart by strength on one of a game's measuring runs, that run gives no score,
so the whole score rests on the others. The note under such a score now says so ("the score rests on 3 runs alone")
instead of "parts of the score".

### A quicker start for the library

When the site restarts, including straight after an update, the library and the front page now appear in a few seconds
rather than about a minute.

### Rated opponents for the new games

Pinwheel, Pitchfork, Sluice, Crease, Pawnproof Wormhole, rear guard torus chess and torus crazyhouse now have rated
computer opponents, so each one has a leaderboard. Every game in the library now does. In Perpetual, Fairy-Stockfish has joined the
leaderboard too.

### Quarry's walls, the same on both sides

In Quarry, a wall beside a piece used to stop its diagonal step on one side but not on the other. A king or pawn
next to a wall can now step diagonally past it either way. Quarry's score barely moved.

### Play Perpetual against Fairy-Stockfish

Fairy-Stockfish can now be your opponent in Perpetual, the chess variant with no draw by repetition. And a new game
whose portal rule against going straight back could never take effect is now turned away, with a sentence saying why; games
already saved with one still load and play as before.

### Three new portal games

*Sluice* is chess with a funnel of portals for each side, several ways in leading to one way out. *Crease* is Capablanca chess with two pairs of folds that the players can move, and a piece that lands on a fold may stay or
come out on the far side. *Pawnproof wormhole* is Wormhole with pawns kept out of the tunnel, which takes away the first player's
head start.

### Fairer scores for portal and turning games

Two things the scores counted were wrong. A portal was being counted as one of a player's pieces that never moved,
which made portal games look as if their armies sat idle, and a piece turning on the spot was being counted as a capture on
offer. Every game has been measured again. Seven portal games score higher (carousel from 0.771 to 0.804), and no game's
verdict changed.

### Pitchfork

A new game: chess with eight wolves circling the centre, and only a pawn can bring one down. The wolves walk in
pairs, nose to tail, so a pawn that takes the leading wolf is usually eaten by the one behind it. It replaces *wolves immune*,
which played the same as the plain wolves game.

### Pawns that face backwards

A piece can now be set facing one way for good, so that no move turns it round. In rear guard torus chess and
torus crazyhouse the rear pawns are now ordinary pawns placed facing backwards, and every pawn is drawn pointing the way it
walks, for both players. When you design a game, the pawn's card has a setting for whether it may turn.

### Pinwheel, and clearer names

A new hexagonal game, Pinwheel, replaces four variants of Glinski's hexagonal chess. Its knights are one-handed:
each reaches only six of the twelve cells a hexagonal knight could, all turning the same way, and the board's two empty flanks are joined, one way for each player.
Nineteen games have new names that say what they are, such as gift chess, wolf hunt, steady ferry and glinski hexagonal
chess.

### Folds, revised

In games with fold portals, a piece that ends its move on a fold now chooses: stay there, or come out at the
linked fold, capturing anything standing there. Pieces no longer reach out of a fold they stand in, and a sliding piece no
longer carries on through one.

### Folds on the board

The far end of a fold no longer shows a ghost piece. Staying on a fold or going through it is an ordinary dot
to tap. And if you have a piece picked up and tap a portal you could also move, the page now asks whether you mean to move the
piece or the portal, instead of making the move.

### A tidier library

The library had several games that were near copies of each other, differing in a detail that barely changed
how they play. Fourteen of them are gone, leaving 87 games, and 29 games have new descriptions that say what their
difference does on the board.

## 2026-10-12

### Offer a draw

Playing the computer, you can now offer a draw once the game is under way. The computer answers at once: it asks its
strongest player how the position stands and accepts only if neither side is clearly ahead. Between two people, the other
player can accept or decline. A draw by agreement counts as a draw on the leaderboard.

### A saved design can be given a description later

Saving a design again with a new name or description now keeps the new words, and the page says so, instead of answering
that it was already saved. A published game keeps the words it was published with.

Under *games near this one*, a piece the two games call by different names is now named both ways, so the sentence no
longer names a piece the other game doesn't have.

On a wide screen the map of games lists its communities once, with folding in each community's own panel, so the map
starts higher. On a phone, the designer's bar for a game that plays is a single row of buttons.

### The chancellor and the archbishop look like themselves

The chancellor, which moves as a rook or a knight, was easy to mistake for a knight. It is now drawn as a rook's tower with
a knight's head rising from its battlements, and the archbishop, which moves as a bishop or a knight, as a bishop with a
small knight's head.

### Two more versions of Divide

*Divide*'s door was tried two other ways: open in both directions, and fixed in place. Both are in the library beside the
original, with their scores and rated computer players. Neither plays better: strong players still mostly draw, and a much
deeper search did not change that, so the walled board itself is what holds the game level. The original stays *Divide*.

### Two wolves on one patrol

A wild piece's walk can now carry more than one piece: in the designer's wild pieces card, *more on this walk* adds another
start on the same route, and the pieces on one walk step in turn, the front one first. The library has a new game made this
way, *wolves patrol*: two wolves circle the middle of the board, one starting from each end of their loop. It has a leaderboard and rated computer players, like every library game.

### The designer explains wild pieces better

Picking a piece from the tray that is also a wild piece in your game now says so, and points to the wild pieces card where
its walk is set. Two walks that cross are refused in one sentence naming both, with the way to put a second piece on the
same walk. The wild pieces card has one short introduction, and *add one* starts with the piece you added last.

A wild piece's walk is now drawn on the board: coloured arrows show the way it goes, a jump between squares that do not
touch is dashed and named under the board, and several pieces on one walk share one path. The designer draws it too, even
while a draft is not yet a game.

Picking a walk in the designer is quicker: the board stays in view, each square shows as you click it, and a whole row or
column joins on from the end nearer the square you picked last, so a loop takes a few presses. A new *+ wild piece* button
under the pieces puts a wild piece straight onto the board and starts its walk there.

## 2026-10-11

### A design too big to save says so

Saving or loading a design over the 64 KB limit now shows the site's message saying so, instead of a request that fails
without a reason.

### Portals in rings, funnels and folds

A rule file can now link portals in more ways than pairs: a ring that passes a piece on round a loop, several portals
leading into one, and portals only some kinds of piece may use. A portal's description says by square where it leads and
which pieces may go in. There is also a second kind, the *fold*: a portal a piece can stand in, move out of from either
end, and slide straight through. Every existing game plays exactly as before.

On the board, each portal now has a small beak on its rim pointing to where a piece comes out, and a *portal map* button
draws the arrows between them. A portal only some pieces may use has dashed arms and dims when you pick a piece it turns
away. The designer's *+ portals* places a pair, a ring or a funnel, with a *leads to* choice for each end and a choice of
which pieces may go in. On a phone, a portal's arms are easier to tell apart.

A fold looks like a bright pool lying in the board: a piece standing in it shows faintly at the far end too, and a piece
passing through sends a ripple across. The designer asks how each group of portals works, by momentum or as a fold, and
its preview shows the moves that go through a fold.

### Portals easier to read

A first reading of the new portals showed one place people went wrong: a portal nobody can move looked as if it stood on
a token. Such a portal is now drawn as a hole in the board, a portal nobody may enter has no arms and stays still, and the
legend's (i) teaches the look in two rows: what a portal stands on (who may move it), and its arms (whose pieces may go in).

### A member's game is rated as soon as it is published

Publish a game and, within a few minutes, its five standard computer players are rated against each other: rated games
and the game's leaderboard are open from the start, and the stronger computer players join once they have been rated too.
A game too slow to rate quickly says so, and waits to be rated by the project's own machine.

### Portals, redrawn

Portals have a new look. Each pair has its own colour, the same at both ends, so you can see which portal leads where;
games with several pairs show each in a different colour, with its number. Inside each portal, slowly turning arms show whose
pieces may go in, and a portal either side may move stands on a white or black token, while one nobody can move is sunk into
the board. Moves and messages now name a portal by its number and square, the legend's (i) shows four examples, and the
designer builds pairs the same way: click a square for the first end, then another for the second.

### Write a game by hand, or with your own AI

The rule language every game is written in is now described in full, with worked examples, in a page in the project's
public code. The designer links to it and has a *load a rule file* button: write a game as a file, load it, and it becomes
your draft. A file with a mistake is refused with the same explanation the designer gives, and your current draft is left as
it was.

### Pieces travel through portals

A piece moving through a portal now makes the trip on the board: it slides to the portal, spins down into it, and spins
up out of the other end with a springy landing. Stepping forward through a finished game shows it too. If your device asks
for reduced motion, the move is a short fade instead.

## 2026-10-10

### Ask for an invitation

Accounts are still by invitation, but you no longer need to know a member to get one. The sign-in page has a *Request an
invitation* button: leave your email address and, if you like, a note on how you found the site, and an invitation is sent
if the request is granted.

### Each game has a leaderboard

A game's page now shows one leaderboard: its computer players and everyone who has finished a rated game on it, strongest
first. A number still settling over someone's first games carries a question mark, your own row is marked, and a game nobody
has played yet says so plainly.

### Portal games say when only a portal can move

In check on a portal game where your only way out is to move a portal, tapping one of your pieces now says so and names
the portals to tap, instead of only saying that piece has no legal move.

### Portal moves: choosing the way is readable again

When a piece can reach a square two ways, directly or through a portal, the box asking which way now shows the piece and
names each route in ordinary text. Before, the route's words were squeezed into the piece's picture and spilled across the
box.

### Every game in the library has rated computer opponents

The 29 newest games in the library had no rated computer players, so they offered only casual play against the computer. All of
them now do: every game in the library can be played for a rating against computer players of known strength. The computer players on
every game were rated again the same evening by today's players, each game's table centred on the same five standard
computer players.

### New games stay marked as new, and the games list sorts by date

The front page's *new in the library* shelf now follows the day each game joined, so a game added tomorrow appears there with no
one updating a list, and it lets go of a game after two weeks. Every game now carries the day it was added, and the games list can
be ordered newest first, each card saying its day.

### Smaller fixes from watching people's first visits

A resign or cancel button now waits fifteen seconds for its second press, so there is time to read *really resign?*. On a phone,
tapping a tag on a game's card says what the tag means instead of opening the game; the cards on the front page open from anywhere
on them; a list with no scored games opens in name order rather than *by score*; and the games list is ready about a third sooner
after the site restarts. In the designer, picking a piece no longer makes the board jump, *teach a move* names its squares, and a
tall board in a game between two people now fits on a phone's first screen.

### Ferry without the shuffle

In Ferry, two in five games ended in a repetition, nearly all of them by a player moving a portal straight back to where it had just
been. A twin game, Ferry no straight back, forbids exactly that, the way Go forbids retaking a ko at once: repetitions fall to fewer
than one game in ten, and it scores higher than Ferry. In the designer, a portal pair's card has a *Not straight back* tick for this.

### Make your own portal games

The designer has a portal row in its tray: put down a pair, move either end, renumber or remove it, and choose who may move each
end. Each pair has a card that says in plain sentences what it does, and when you teach a piece a move the board shows where it would
come out of a portal before you try it. To start from one that works, open Divide, Ferry or Carousel and choose *make your own from this*.

### Portals, and seven games that use them

A board can now hold pairs of portals: a piece that steps into one comes out of the other and carries on in the same direction,
and a square it can reach two ways asks which way you meant. Seven new games in the library use them. In Divide a wall splits the
board and the portals are the only way across; in Ferry either player may move the portals themselves; in Carousel some doors open
for one side only. Four small games each try one idea on a little board, and two of those are marked not viable: there, the
player who moves first gets through the door first and almost always wins.

### Hex moves you write once and turn

In the designer, a piece's move on a hexagonal board can now be written once and turned all six ways, or turned and mirrored, instead
of listing every direction by hand. Two new archive games are Gliński's hexagonal chess written this way; they play exactly like it.

## 2026-10-09

### Jeson Mor, where you win by leaving the centre

A game can now be won by moving a piece off a square. In Jeson Mor, each side has nine knights, and the first to land a knight
on the centre square and then step off it again wins. It is in the library with a score of 0.70, below the bar
the library sets for a good game.

### Kyoto shogi, where a piece in hand can come back as either of its faces

Kyoto shogi is in the library with a score. Its pieces turn over every time they move, and a captured piece can be put back on the
board as either of its two faces: the board page now shows one button for each face when you hold such a piece.

### Two games that start with pieces in hand: Placement chess and Sittuyin

In Placement chess and Sittuyin (Burmese chess), the pieces other than the pawns begin off the board, the king among them, and the
game opens with each side placing them one at a time. Both are in the library with scores.

### Two more games: Extinction chess and Grand chess

Extinction chess, where you lose when any one kind of your pieces is gone, and Grand chess, where a pawn may only promote to a
piece you have already lost, now play by their real rules and are in the library with scores.

### Two new games: Einstein chess and Omega chess

The rule language can now say two things it could not: a piece that captures may stay what it is instead of changing (Einstein
chess, where a queen that takes stays a queen), and castling may happen on a row that has walls beyond the rooks (Omega chess).
Both games are in the library with scores.

### The map's settings and the designer's checkboxes are back in place

On a wide screen the map's settings fit on one line again, so the drawing starts on the first screen instead of below it. In the
designer, the *royal* box sits beside its word again. On a phone, a ticked line under *Publish* that wraps now lines up under its
own words.

### Scores for the three newest games

Pawn Knockdown, Shogi Backward Drops and Shogi Drop Promoted now have scores, each the average of four measurements.

### A game whose computer players could be told apart only some of the time now says so

Each library game is measured four times. Where the stronger computer players beat the weaker ones clearly in only some of those
measurements, the game's page now says in how many, instead of calling them steadily stronger. How far the score is trusted does
not change.

### The designer explains a draft that isn't a game yet

When you ask the designer what a piece can do in a draft that is not yet a complete game, it now says what is missing in a plain
sentence instead of failing with a server error.

### Plainer words when measuring stops, and the board's tabs first on a phone

When measuring a game in your browser stops before it finishes, the page now says so in plain words instead of showing a programming error. Game pages no longer show the technical names another chess program uses for pieces and rules, and on a phone the board page's moves, rules and pieces tabs now come before the players' lines instead of under them.

### New games and Intransitive on the front page; the games page goes straight to the list

The front page now shows the games added to the library most recently, each with the day it was added and its score, or a plain *not measured* where the computer players have not measured it yet. Below them, *Intransitive*, the rock, paper, scissors strategy game, has a place of its own, beside today's game and this month's game, which used to sit at the top of the games page. The games page now opens straight on the search and the filters, with every game in the list, instead of repeating the front page's picks above a folded *All games*.

### The game builder and the map in the site's look

The game builder and the map now look like the rest of the site. In the builder, each piece is its own sheet of paper, the one you are editing is edged in green, and the rules are rows separated by thin lines instead of boxes inside a box. On the map, the settings above the drawing sit on one sheet. On a phone that sheet folds to a single line, so the map is the first thing you see, and one tap opens the settings.

### Where the computer players stand among people

On a game where at least five people have a settled rating, the line about its strongest computer player now says, under *more*, where each computer player people have played sits among them: about which rating a person would need to play like it, and how many of the people with a settled rating it is above. It comes from rated games people played against it, and it changes no game's score.

### Placing a captured piece, right on the board

In games where taking a piece throws it onto a square of your choosing, the choice of square now appears in a box over the board, as choosing a new piece for a pawn does: each square a large button named as on the board's edge, with *Look at the board* and Cancel. When your pawn takes on the far side, the box first asks what the pawn becomes, with each piece drawn and named, then where the piece you took goes. Before, both were a row of small buttons under the board that named the new piece by a single letter.

## 2026-10-08

### The wolves show both of their next steps

In games with pieces that belong to nobody and walk on their own, like the wolves, each one takes two steps before it is your move again: one after your move and one after your opponent's. The board now shows both. The dotted arrow is where it goes after the next move, and a fainter dotted arrow carries on from there to where it goes after the move after that. The line under the board says which is which, in a game against the computer, in a game against another person and when you step back through a game.

### Finding other players

Members now have *People* at the top of every page. It opens a list of every member, A to Z, with a search box: type a few letters of a name, or of a game somebody made, and press a name to open that member's page. A member's page shows the games they made, newest first, the games they played lately, their milestones and badges, and says plainly what there is nothing of yet. *Challenge* on their page now offers the games they made first, and starts a correspondence game when they accept. Names elsewhere on the site, on a game's page and on your own page, now open that person's page too.

### Choosing what your pawn becomes, right on the board

When your pawn reaches the far side, the board now darkens and a box over it asks *Choose what your pawn becomes*, with each piece drawn large; tap one, press *Look at the board* to see the whole position first (the move waits until you choose), or press Cancel. Before, the choice was a row of small buttons under the board that was easy to miss.

## 2026-10-05

### Members' games on the front page show their own pieces

The small boards under *Made by members* on the front page showed every piece as a letter in a circle the first time you opened the page. They now show each game's own pieces, the same drawings as on the games page and the game's own page, including pieces a member drew themselves.

### A bigger board in a game against another person

In a game against another member, a tall board (torus chess crazyhouse, say) used to sit at half the width it had room
for, with *Send your move* and *Change my move* squeezed under it, *Change my move* cut to a few letters. On a wide
screen those buttons now stand beside the board when that leaves the board more room, so a tall board is about half
as wide again. On a phone they take the whole width and wrap their words rather than cut them off. The board no longer
shrinks by a few pixels the first time you pick a move.

### Publishing a game no longer waits for the site's check

When you build a game, the button that measures it on your computer is now right beside *Publish*, under the line that
says a score is needed, so there is no scrolling away and back. Once your computer has finished measuring, *Publish*
works straight away: press it and the page says publishing will be complete once the site has checked the score. You
can leave then. The game goes on trial by itself when the check passes, a few minutes later. If the check finds a
problem, nothing is published and the page says why.

### Games made by members, on the front page

The front page now has a shelf of its own for games members have designed and published here, straight under the
opening board: the four published most recently, each with its opening position, the name of the member who made it
and a *Play* button, and a link to the rest when there are more. Drafts are never on it. Until somebody publishes a
game, the shelf says it is empty and invites you to build one. The opening board also now says which of its games is
today's game.

### Changing your mind about a move before you send it

In a game against another person, a move you have picked but not sent can now be changed as easily as it was picked.
Tap another square the same piece can reach and the move goes there instead; tap a different piece of yours to pick
that one; tap the picked piece or the square it would land on to take the move back. Beside *Send your move* there is
now a *Change my move* button that does the same (at a keyboard, so does the Escape key), and the words above and under the board say the move can be changed
until you send it. Nothing reaches the other player until you press *Send your move*.

### A move you have picked but not sent is plain to see

In a game against another person, the move you pick now shows on the board itself: the piece stands faded on the
square it would go to, with a dashed arrow and dashed outlines, while the other player's last move keeps its usual
yellow shading. Under the board a large *Send your move* button sits in a dashed box that says the other player has
not seen your move yet; nothing reaches them until you press it, and *put it back* lets you choose again. If you leave
the page first, the browser asks before you go, and the move is still waiting on the board when you come back on the
same device. On your list of games waiting on your move, such a game says *Not sent yet*.

### Inviting a friend to play, from the game itself

Under *Play a person* on a game's page there is now a place for a friend who isn't a member yet. If the person who
runs the site has given you invitations, type your friend's email address there and they get an invitation from you;
once they have joined, challenge them on the same page. If you have none, it says so and who to ask. The button at the
top right, which used to look like just your name, now reads *your page* and looks like a button you can press.

### A score is an estimate that keeps improving

Every score on the site is now treated as the best estimate we can make today, not a final verdict. When the computer
players get stronger or the way a game is measured improves, the games are measured again on their own, one at a time,
and each score moves to its new value. While a game waits its turn it keeps its last score, marked *provisional*. The
yardstick a score is read against no longer shifts when a game joins or leaves the reference set, so a score moves
because something about its own game was learned, not because the list around it changed.

## 2026-10-04

### Games you make can be published

A game you design can now go on trial, where others can find it, play it and vote on it. Measure it on your own
computer from the designer: the site then checks that every game follows the rules, replays a sample of them exactly,
and once that spot check passes, a few minutes later, *Publish* opens. Until then the game stays a draft that anybody
with its link can open. A score whose replayed games come out differently is not kept, and the account cannot publish
until the person who runs the site has looked at it.

### The board holds still

On a phone, the board no longer jumps up and down while you play. The line above it, which says whose move it is,
*in check*, *thinking* or the result, now keeps the same room whatever it says. A move that could not be made is
explained in a message at the foot of the screen rather than above the board.

### The front page for everyone, and the games at their own address

The front page that says what this is now greets everybody, signed in or not. The games have an address of their own,
`/game`: *Games* at the top of every page and *games* at the foot go there, and so does signing in or accepting an
invitation. The name at the top goes back to the front page. Links you already have, to a game, to a game in progress
or from an email, still open what they named. On the front page, a member sees *Your page* where a visitor sees
*Sign in*.

### A way to get in touch

The front page and the foot of every page now say how to reach the person who makes and runs the site: a link to
their GitHub page. The terms name the same place for asking to take something down.

### The terms, and what publishing a game gives away

The site now has terms, linked at the foot of every page. Games made here are for everybody: when you publish one,
you give its rules, name and description to the public domain (CC0 1.0), so anyone may copy, change, share or sell
it, and the site keeps nothing and claims no credit. *Publish* shows that sentence beside a box to tick first. Each
point on the terms page is in plain words, with the legal wording beside it.

### A front page that says what this is

The site now opens on a page of its own before the games: a quiet board showing some of the games in turn, then what
the site is for, how games are built from a palette of rules, playing the computer and other people, how the computer
checks that a game works, and what the score is: a best estimate that keeps improving, not a final judgement. *What
goes into the score* opens to show every ingredient, why it counts, how it is worked out and how much it weighs.
*See the games* is one tap away, and members go straight to the games; the page is always at *about* in the footer.

### No password box before the site

The site no longer asks for a shared password in a browser box before it opens. Anybody with its address can look
around and play practice games against the computer players up to *hard*; *hardest* is for people with an account,
and the opponent list says *sign in to play* beside it. An invitation's link now goes straight to choosing a handle
and a password.

### Invite somebody

If you have invitations to send, your profile now has *Invite somebody*: how many you have left, and a box for the
address. The person gets the usual invitation email, naming you.

### The designer keeps out of its own way

When a change makes your game *not a game yet*, the note at the foot of a phone's screen is now one line with the first
thing to fix, and the full list is one tap away, so it no longer covers the setting you are fixing. On a computer, the
mouse wheel over the right-hand column scrolls the page again. *Start from* a template or a copy of a library game now
sits above the board; each way of moving you type is read back in words under it; and the verdict says when it was
last checked.

### A gentle first game, with the rules at hand

The first time you start a game on a game's page, it seats the gentlest computer player on that game (marked
*gentlest* among the opponents), as a practice game, and the board opens with the rules beside it; *a harder one* is a
tap away. While you play, tapping one of your pieces shows how it moves in a line under the board. On a game far from
chess (no royal piece, won by reaching a square, or with wild pieces) the page also offers to watch two computer
players first.

### Each piece on a board of its own

Under *The pieces* on a game's page, each piece now has a small board showing it alone, with a dot on every square it
can move to and a ring where it can take. The pieces that move like nothing in chess come first, with their boards
open; a piece that moves exactly like one of chess's says so (*moves like a chess knight*) and keeps its board folded.

### A game's page shows where the pieces start

A game's page now opens on its board: the start position drawn large at the head of the play column, before you
start a game, with the squares its rules single out (a goal, a promotion band, a seam) marked and named in a line
under it, and which side moves first. The list of where each piece starts is still there, folded under *The rules*.

### Challenge a game's maker at their own game

If you designed a game that is out on trial or further, you can now switch on, on your own page, being listed on
your games as taking challengers. Other members then see you first under *Play a person* on each of your games, as
*made this game and takes challengers*, with a button to challenge you; you accept or decline as with any challenge,
and you can switch it off at any time. It is off unless you switch it on. A win over a game's maker at their own game,
in a rated game, earns a new badge, *Beat the maker*, once per game for each winner. A challenge carries no message.

### A page for each member

Every member now has a page of their own that other signed-in members can open from the players' list, from a name in
*Play a person* or from a seat in a game between two people: their mark and handle, when they joined, how they finish
their games against people, what they have finished lately, the games they designed that are out on trial or further,
their milestones and badges, and who they play most. A button on it challenges them to a game. Visitors cannot see
these pages, and a draft never shows. Your own page has a new box to show the games you gave five stars; it is off
unless you tick it.

### Tag a game when you rate it

Beside the stars, a vote can now carry up to three of eight tags: *I'd play it again*, *a clever rule*, *first
player has it easy*, *lots of draws*, *goes on too long*, *over too fast*, *hard to learn* and *feels new*. Everybody
sees how many votes chose each tag, never who, and the person who designed the game sees the same counts under
*Your games*. There is nowhere to write words with a vote, so nothing anybody types reaches a designer.

### Every member has a little invader of their own

Each person here now has a mark: a small pixel creature in the family of the site's invader rook, with its own crest,
eyes and legs in one of eight inks (claret, cobalt, violet, brass, walnut, petrol, olive and plum) that read on the
paper in light and dark alike. It is made from your account, so it is the same every time and nobody has to
upload anything. It stands beside your name in the corner and on your page, beside each person on the players' list,
in a game between two people, in *Play a person*, on a challenge sent to you, and beside your opponent in the games
waiting on you.

### The site's own pages in the club room's look

The players, your own page, the maintainer's page, the privacy notice, the credits and *What's new* now match the
rest of the site: tables and counts sit on sheets of the same paper as the cards, a page to read keeps a comfortable
width with each section under a rule, *What's new* sets each day's entries on one sheet with the date beside it, and
every page's footer is signed with the invader rook. An address with no page behind it now shows a proper page that
says so, with the way back to the games, instead of a line of code.

### Every game shows its board before you open it

Each game in the list now shows its starting position beside its name, on its own wood, and the card is calmer: where
the game stands and its score are the two marks, and the rest is a quiet line under them. The games to start with and
this month's game show their boards too. A game's page puts the play panel first, with the computer players as cards
you tap (the name first, how strong it is under it), and folds the rest of the game under headings, so on a phone the
page is a list of what there is to read rather than a long scroll.

### How far you have got against each computer player

Signed in, a rated game's page now shows every one of its computer players in a row, weakest first, each marked with
your best result against it: *won*, *drew* or *not yet*. A row of *won* is the whole ladder.

### Whether people have beaten the strongest computer player

Each game's page now says, under its computer players, how the strongest one has stood up to people so far: *not
enough games yet*, *held so far* once twenty rated games by five different people have not beaten it, or *beaten by
people* once somebody has. Under *more* are the games and people it counts. It is read from people's own games, and it
never changes the game's score or how far to trust it.

### On hexagons, the designer no longer offers turns that don't fit

On a hexagonal board, a way of moving's *turned* menu now offers only *as written*: the other turns are the square
board's, which a hexagonal game refuses, so every direction is listed by hand.

### This month's game

The front page now names one game a month for everybody, so there are people to play it with by correspondence.
October's is *reckoning*, a six-by-six game the computer invented here. It changes on the first of each month, taking
the games to start with in turn. Playing it earns nothing extra; it is the game most people will be looking for a
partner in.

### The front page says what this place is

Under the day's game, three lines now say what the site is: people invent chess-like games here and anyone can play
them; the computer plays each one hundreds of times and scores how interesting it is; play this one now, or build your
own. Below them are nine games to start with, each with the one sentence that says how it differs from chess. The
whole library, with its search, filters and order, is one click further down, under *All games*.

### Intransitive waits longer before a draw

In *Intransitive*, a game now ends in a draw after 200 turns in all without a capture, 100 by each player, where it
used to end after 100 in all. That is how the program on the designer's own site counts it. Long games of careful
manoeuvring are no longer cut short. To the site this is a new version of the game, so it was measured again; the
old score is kept in its history. The computer player that taught itself plays the new version too, after being
tested on it again: against the standard computer player that looks at as many positions, it still takes about
nine points in ten, as before.

## 2026-10-03

### A game a day on the front page

The front page now opens on one game, a different one each day and the same for everybody: cylinder chess, three-check,
quarry, los alamos and crazyhouse in turn, each chess with one rule changed. It shows the board and that one rule in a
sentence, with *Play the computer* to start a practice game against the gentlest computer player, and a harder one a tap
away.

### A score that is out of date says so

When a game is being measured again, its page now shows its last score marked *provisional*, with the day it was worked out,
instead of a blank. A provisional score shows no trust word and never counts in a comparison.

### How far to trust a score

Every score now says how far to trust it, in one word: *early estimate* when only the site's own computer players
have measured the game, *checked* when something else agrees with them, and *questioned* when something has found more
in the game than they see. The word sits beside the score on each card, and a game's page says it under the number
with one sentence of why. Behind *more* is each piece of evidence: how the computer players order by strength, what an
engine written elsewhere made of the game, what a computer player that taught itself found, how many people have
played the strongest computer player and whether anyone has beaten it, and any earlier score, dated. People beating the
strongest computer player can move the word; they never move the score.

### Scores no longer depend on which reference games are chosen

The ends of the scoring scale now come from what each measurement means, not from the best and worst of a few reference
games. Every game's score moved once (most rose by about a tenth), and the order of the library shuffled a little.

### Play the learned player on Intransitive

You can now play the learned player on *Intransitive*: a computer player that taught itself the game by playing
against itself, and takes about nine points in ten from the usual hard player when both think as long. It is experimental and not
rated, so a game against it is practice.

### Each game's card is its own sheet of paper

Each card in the library now has its own slightly different paper.

### What is wrong with a draft, in plain words

When a draft isn't a game yet, the designer now says what is wrong by the names you see on the page (*Where they
start, row 1: z9 isn't a square on this board*, *Piece 2 (knight), way of moving 1: 'how far' does nothing on a
leap*) instead of a path into the game's text, the quick check of whether a game works speaks of sides and turns,
and every field on a rule card has a plain name and a plain sentence under it.

### The game designer in plain words

Building a game, the editor now says what each step costs and when it happens in everyday words (*saved as your
draft*, *happens as you edit*, *decided by the person who runs the site*), its refusals say what to do instead of
naming a file format or a fingerprint, and measuring in your browser talks about your computer's cores rather than
workers.

### The library and the map in plain words

The library's cards, filters and sort menu, and the map's panels, now say what they mean without the site's internal
jargon: *ruled out* instead of *cannot be won*, *games played by people*, *beaten by 3 people*, *not played here yet*,
the parts of the score by their plain names, a map that talks of *likeness* and *the slider*, and an (i) on the
Baseline tag and on the line that explains the card's words.

### Every page says when the site is updating

Your profile, the players, suggestions, a challenge link, signing in and joining now say in a plain sentence that the
site is updating or busy, as the board does, instead of a browser's error, and a page that only reads tries once more
before it says so.

## 2026-10-02

### See where the wolves are going

In a game with pieces nobody owns, like the wolves, the board now draws a faint dotted arrow from each one to the square
it will step to after the next move, and stepping back through the game shows where they were heading then.

### When the site is updating, the board says so

A move made while the site was being updated or was busy used to show a browser's error, such as *The string did not
match the expected pattern*, until you reloaded. The board now says in a plain sentence that the site is updating or
busy, puts the position back as it is, and you can make the move again in a moment without reloading.

### A new look

The site has a new look: warm paper and baize green, a serif for the headings, a clearer header with a small invader-rook
mark (also your browser tab's icon), and a switch for light, dark or your device's setting at the foot of every page.
On a phone the sections fold behind a menu button. The pages say the same things; they now read more easily.

On a game's page, whose move it is shows as a coloured dot and *in check* as a red label above the board, the square
names sit outside the squares, a phone gets a row of small icons with words, and a finished game shows who won across the
board, with a button for a rematch and one to step back through the game.

Every board is now wood: each square is its own piece of grained maple or darker wood, in a walnut frame, and the
same board looks the same every time you open it.

### Play another person

Two members can now play each other, each moving when they like. A game's page lists members who said they are open to a
game of it, you can challenge someone by username or send a friend a link, and a finished game offers a rematch. A move
you choose waits for *Send*, so a slip can be taken back. If the player to move hasn't moved in five days, they lose,
and every game says how long is left to move. Your page lists your games, the one closest to a loss on time first. You
can switch on one email each time it becomes your move, off unless you ask.

### Make your own from this

Every game's page has *make your own from this* right under its name, on a phone too, and the designer's *start again
from* menu lists every game in the library. Your copy gets a new name and keeps a note of the game it came from; the
original is never changed.

### The designer on a hexagon

Choosing a hexagonal board in the designer now turns your game into a playable hexagon game, and the board always shows
your draft's own cells instead of disappearing. On a computer a hexagon board is about one and a half times wider, and a
phone is told once that building a game is easier on a computer.

### Plainer sentences, and a clearer first visit

A new member's page says what to do next instead of showing zeros; the players page and the monthly round appear once
they have something to show; when the strongest computer player is busy, the opponent list says so before you pick it;
and the sentences that refuse something say what is true and what to do. A game's page no longer shows the project's
own history notes.

### The site's own pages in plain words

The badges, the milestones on your profile, the players page, the monthly round and the messages you get when a game
can't start now say *rated game* and *practice*, *the strongest computer player* and *the person who runs the site*,
and name a computer player instead of printing its settings. Every badge still earns exactly what it did.

### A game's page in plain words

A game's page now gives each computer player one rating, the one earned on this site, and explains its score, the
tests a game must pass and why a run was set aside in plain words; *practice* replaces *sandbox*, and the rules count
turns rather than half-moves.

### Opponents with names, and your own rating

The opponent menu now names each computer player first (*hard*, rated on this site, or how far above or below you it
is when you're signed in), with what it does in plain words underneath, and the move list counts in the same turns.
Your profile shows your own rating on each game instead of only how far you are from the top.

### The explanations speak plainly

Every (i) on the site, and the panel that says how a game's score is worked out, now use plain words: the score's (i)
says what the score is first, with more under *more*, and in-house terms are gone from them.

### Clearer starts, shareable drafts

The start button now says what it starts, rated or practice and against which opponent, and remembers a rated choice. A
practice game can be resigned. A designer's draft has a copy-link button and is credited to its designer by name, and
the sign-in and join pages say that accounts are by invitation and that a game played without one isn't kept.

### Game pages stop talking to the developer

Visitors no longer see setup instructions for a chess engine on a game page, a long code where a game's name should be,
or the designer's notes about command lines and folders. The editor's trial game now says it isn't recorded, instead of
calling a saved game unsaved.

### Check and the end of a game are on the board

When a king is in check, its square is marked and the line above the board says so; the move list carries + for check
and # for mate, and when a game ends the result appears right above the board. Tapping a piece that has no legal move
now says why. The take-back, flip and resign buttons sit directly under a board that is sized to fit the screen, and
the promotion zone is a labelled hatch instead of rings that were easy to misread.

## 2026-10-01

### Every game re-scored against today's computer players

The computer players got better at seeing threats at the end of their search on 29 September, so every game in the
library was played again and re-scored. Most scores barely moved; the typical change is under a hundredth. Racing kings
no longer counts as a game the computer players can measure fairly, while gardner, tunnel chess (vertical) and bastion
now do. The site carries the new scores, along with the hexagonal boards and the designer's recent fixes.

## 2026-09-29

### Badge rules say what they mean

The rules for the badges earned by building on another game no longer say the new game must be listed; a draft that
passes the site's check has always counted, and now the wording says so.

### Milestones for game designers

A designer's profile now marks, with dates, their first kept draft, the first to pass the site's check, the first
played by somebody else and the first another player built on. A designed game's page shows whether it passes the
check, how many people have tried it, and who has built on it.

### The designer's rules panel says how each piece moves

The designer's plain-English rules now include a sentence for how every piece moves, the same ones a game's own
page uses. A game you watch in the designer stops thinking while its board is scrolled out of view, and a strike's
chain card now starts from settings that work.

### Clearer publishing and shortlist wording

The panel for publishing a draft no longer says the same thing twice, and a game's own designer is now told plainly
that its reading needs refreshing, rather than being told only its designer can do it.

### The designer is easier on a first visit

The two ways of playing a draft now say which is which, and the board you play on in place is wider on a phone. A
piece you add says it is not on the board yet until you place it, and pressing a design step a second time now offers
a new change instead of quietly undoing the last one.

### Nominate a game from the designer

A designer can now put their own game forward for the shortlist from the designer itself, where it shows how far
the game has come, instead of only from the game's page.

### A hexagon's one-way seam shown past the board's edge

On a hexagonal board whose every pair of sides is joined, a one-way seam now carries on through the surrounding
copies of the board, as it already did on a rectangle, instead of stopping at the board's outline.

### Small fixes on the pages a new game climbs through

A shortlisted game's page now gives the voting round's closing date once rather than twice, and the row showing a
game's level fits on one line on a phone.

### A check on whether the computer plays a game well enough to score it

Each game can now be checked for whether its computer players are strong enough for its score to mean much. A game
that fails keeps its score, marked with a sentence saying why, and is left out of the games that set the scale. The
check has not been run over the library yet.

### A monthly round decides which games join the library

Members can vote yes or no on shortlisted games on a monthly round page. The maintainer then confirms or turns each one
down with a published reason. A game turned down waits 60 days before it can be put forward again.

### Quiet games rest, and the measuring games say why they were chosen

A game on trial that nobody else has finished in 60 days, after 90 days on trial, leaves the listing as *Resting*. Its
link still works, and its designer can publish it again once. A game moved into the set every score is measured
against now shows when and why on its page and in What's new.

### How close a game is to the shortlist

A game on trial now shows how near it is to the shortlist: days on trial, how many other people have finished a game
on it, and how many votes count. Its designer gets a Nominate button that works once the bar is met, and the star panel
says whether your own vote counts. No game can reach the shortlist until scores measured in the browser can be attached.

### Every game says where it stands

Each game now shows one plain word for where it is on the way into the library: Draft, On trial, Shortlisted, Library
or Baseline. The word is on its card, its page and the designer, the library has its own tab for games on trial, and
your profile lists your games by where they stand, with the next step for each.

### Games you finish are kept, move by move

Every game finished on the site, practice games included, is now kept with its whole move list, even if you undo or
start again afterwards. Nothing about who played is added. This lets a win against the computer be replayed and
studied, to make the computer player stronger.

### A copy of an existing game is named, not published

If a draft's rules play exactly like a game already in the library or on trial, only spelled differently, the designer
now says which game it copies, and publishing it is refused.

### Report a game

A signed-in player can report a game as spam, offensive or a copy of another, from a small link under its rules. A
report hides nothing by itself; the maintainer reviews it.

### A computer player can now learn a game by playing it

A new training command has the computer play a game against itself, learn from those games which positions are good,
and keep what it learned only if it then beats the standard computer player. Nothing on the site plays this way yet:
the first real training runs come next.

### Saving a game no longer puts it on show

A game you save in the designer now stays a private draft, reachable only by its link. Publishing becomes its own
step, with a button in the designer that opens once the site can check a score measured in your browser. Games that
saving had already listed go back to drafts, their links still work, and your profile says why.

## 2026-09-28

### A score measured in your browser is checked by the site

When a signed-in designer measures their own saved game in the browser, the site now checks it and shows its own number,
labelled as measured on your machine and not yet confirmed. This reaches the site with the next update.

### The computer players think about a third faster

Games between computer players, and the measuring that rests on them, now run about 30% faster, with every move
exactly as before.

### You can design a hexagonal game

The designer can now turn a hex board into a hexagon, choose how many cells it has on a side, and join each pair of
opposite sides so pieces cross from one to the other. A move across a joined side is offered as the short step.
On a square board whose edges join, a move taught across the join is offered as the short step too. When you change a
board's shape or size, walls, pieces and promotion squares that fall off it are removed, and the designer says which.

### Hexagonal boards look like hexagons

A game on a hexagonal board is now drawn as one: 91 cells with no corner squares, on the play board, the replay and the
designer's preview. Where a game joins two opposite sides, faded copies of the board show what lies across, and a line
along the side shows who may cross it and which way.

## 2026-09-27

### The designer says what you changed

A game you copy and change in the designer now lists, under *What you changed*, each piece move and rule that
differs from the game it started from, in the rules' own sentences.

## 2026-09-26

### The game designer explains itself better

Each starting template now arrives with its own name and a sentence about the game. A rule you pick is no longer
cleared by the page redrawing a moment later, which made the first press of *add it* do nothing. On a laptop-width
window, the board no longer slides over the form as you scroll.

### Every game can be played for a rating

Until now most games in the library could only be played in the sandbox, because the site had no rated computer
opponents for them. Every one of the library's 70 games now has its ladder of rated opponents, so any of them can be
played in leaderboard mode.

### Shogi plays by the traditional rules

A piece may now promote on a move out of the far three ranks as well as into them, as in traditional shogi. That
makes the game the outside chess engine plays, so shogi can be checked against it and played against it again. The
one difference left is that a perpetual check is a draw here rather than a loss for the side giving it. Shogi was
measured afresh and reads 0.835.

### A stronger opponent in *Intransitive*

The computer players used to count one step towards the winning corner as worth about six pieces, so they gave away
material to race. They now weigh the race sensibly and also know that a piece nothing can capture is bound to win.
The hardest opponent beats its old self about three games in four. Only *Intransitive*, *three stones* and the
rock-paper-scissors test games play differently. Their scores moved with it, and *RPS repaint* now counts as a
proper game. Every other game's score is unchanged.

### A game's page opens faster

Opening a game's page, or starting a new game against the computer, now reads only that game's results
instead of the whole library, so it comes back in about a fifth of the time. What the page shows is unchanged.

### The map groups games by how alike they are

The map of games now draws a line between two games when their rules are close, thicker the closer they
are, and colours the families of related games it finds. It opens with 90 lines and 13 families, each drawn
as its own knot. A slider shows more or fewer lines. Hovering a game lights it and its neighbours, a close-up
shows one game and the games around it, and the search box centres the game it finds. The old lines, one
rule apart, are still there behind a switch. Each family of three or more games is outlined in its colour and
labelled with its name, and a family can be folded down to a single dot.

## 2026-09-25

### Three games get their own-values opponent back

On crazyhouse, xiangqi and los alamos, the computer opponent that plays with piece values searched
for that game is offered again. The values were searched afresh under the rules for counting pieces
held in hand, and crazyhouse's table changed. Chess, makruk, shatranj, capablanca and shogi keep the
values their rules imply, because the search found nothing better. No game's score moved.

### Tunnel chess horizontal leaves the library

Its rules put pawn promotion on a row past the edge of the board, so no pawn could ever promote.
It has been retired, since other wraparound versions of chess cover the same ground. Its page and
past games still open. A new design that names a promotion row past its board is now refused with
a sentence saying so.

### A built opponent can't quietly copy another

Building an opponent that is only a scaled copy of a plan the game already offers is now refused,
naming both. Before, it was saved and then played as the existing one. If an opponent built before
this change clashes that way, only it is left out, and the strength panel says which one and why.

### Removing a piece no longer widens a narrow blast

In the designer's one-step changes, removing the one piece a blast rule was limited to used to
leave a blast that went off on every capture, turning the game into atomic chess. The rule now goes
with the piece.

### Tournaments with the outside engine run faster

*For maintainers.*

At the command line, a tournament that includes the outside chess engine now plays every game
between the project's own players together in the fast engine, and only the outside engine's games
one at a time. The same 16-game chess tournament went from 14 to 11 seconds, with the same games.

### A design too slow to test is refused, not judged on part of its games

Before a design is accepted, a quick check plays a batch of random games on it within a time
limit. If only some finished in time, the check used to judge the design on those alone, so a
faster or slower computer could reach a different verdict. Now a design whose test games cannot
all finish in time is refused, and the message says how many finished.

### The decisions score says which piece values it judges by

The explanation beside *decisions that matter* now says it always judges moves by the piece values
the rules imply, the same way on every game. That holds even on a game whose computer players
use values fitted to that game. The number stays comparable from one game to the next, and the
explanation says what that costs.

### A design that was played but not scored says so

A design candidate whose games were played but that got no score used to read *not measured* on
its card, as if nobody had played it. It now reads *no score* and names the run that played it.
A candidate that never got as far as a game still reads *not measured*.

### The check against the outside engine plays longer games

The command that checks which games the outside chess engine plays by the same rules now plays
each test game for up to 400 moves instead of 60. Some rules only come into play late, such as
promotion or a checkmating drop. The short games reached the first disagreement in only a
handful of cases, and the longer ones catch nearly all of them.

### Shogi no longer offers the outside engine as an opponent

The outside chess engine plays shogi by the traditional rules, which let a piece promote as it
leaves the far zone and treat a checkmating pawn drop differently from this project's shogi. Games
against it could stop with an error part-way through. It is no longer offered on shogi, and the game
page says which two rules differ.

### Every score moves: the new scoring version

The library is now scored under a new version of the score. It no longer gives a game credit for
having few rules, and it judges variety by how many different positions a game's games pass
through, rather than by how the rules are written. Most games rise a little; small, light games
such as gardner, dobutsu and los alamos fall, and shogi climbs from the lower half into the top
twenty. No game's verdict changed. On the set of known good and bad games the project checks
itself against, the new version does not tell them apart any better than the old one (slightly
worse, in fact): it was adopted to stop rewarding short rulebooks, not as a sharper judge. Scores
from the two versions are never ranked together.

### Rating the opponents says how long it takes, and takes less

The button that works out how hard each computer opponent is on a game used to promise *a minute
or so*, and on large boards with the outside chess engine among the opponents it ran for seven
minutes or more. It is now about twice as fast on those games, says honestly how long to expect
(under a minute on a small board, up to ten on a large one with the outside engine), and counts
its progress while it runs.

### The move list says what the wolves did

In games with pieces that move by themselves after each turn, such as the wolves games, the move
list now has a line under each move saying what they did, for example *and the wolf on g5 takes
the bishop on f4*. The saved replay of a game steps through the move and the wild pieces' turn
separately.

### Intransitive's page states its rules

The published game *intransitive* used to describe itself only by pointing at its designer's
video. Its description now states the rules the video gives, including that there is no draw by
repetition, and says which details the video leaves open. The game itself did not change.

### A one-step change that empties a rule says so

When a change in the designer's *and now try* panel, or in the automatic designer, leaves a rule
with nothing to act on (a capture rule when nothing can be taken, castling with no rook to castle
with), the change now drops that rule and says so in its sentence, instead of the result being
quietly thrown away.

### The stones game says why its number is not a score

The stones game is kept in the library on purpose, as an example of a game that cannot really be
played. Wherever its number appears (its card, its page, a nearby game's list and the map) it now
says so, so nobody takes the number for a score.

### Make the game in between, from your own draft

A draft's own page now names the library game nearest to it and offers *make the game in
between*, which opens the designer on a game halfway between the two. On the site only the draft's
owner sees the button on an unlisted draft; anyone can use it on a listed one.

### A reloaded game keeps how far the computer looked

On the site, the note beside each computer move saying how many moves ahead it looked now survives
reloading the page. Games played before this change show no such notes.

### The rules text no longer says an untouchable piece can be taken

A piece that never moves used to be described as staying *until it is taken*, even in games where
nothing can take it, such as the stones games. It now says it stays for the rest of the game. The
optional hint that checks a design's description gets the same correction, so it no longer flags
those games by mistake. And four games whose rules mention a scored clock now say checkmate is
looked for first.

### The hand-written opponents follow each game's own rules first

The practice opponents that play by a hand-written plan now pick the plans that fit a game's own
rules before the general ones: games won by capturing a particular piece get a player that goes
for captures, race games a racer, drop games one that keeps pieces in hand. On 25 games the list of
these opponents changes; on chess, for example, one that pushes pawns to promote replaces one that
holds the centre.

### A broken opponent is left out and explained, instead of failing

On crazyhouse, xiangqi and losalamos, the opponent that plays with the game's own searched piece
values was built for an older version of the rules and could not play: picking it, or asking how
hard each opponent is, failed with a message blaming the wrong thing. It is no longer offered, and
the game's page says in one sentence why and how it gets rebuilt.

### Checkmate on the last move before a draw clock now wins

A checkmate given on the very move that runs out a game's no-progress count or move limit used
to be scored as the clock's result, usually a draw. It now wins, as in the official rules of
chess. A stalemate on that move still ends by the clock. No game already in the library ended
that way.

### The computer players now value the pieces they hold in hand

On crazyhouse, shogi, dobutsu and the other games where captured pieces change sides and can be
dropped back, the built-in computer players now count a captured piece in hand as fully their
own. They used to count it as worth nothing until it was dropped, so they gave pieces away too
easily there. Scores for those games will move when the library is next re-measured.

## 2026-09-24

### The program's own name is now Chess variant sandbox too

*For maintainers.*

The program you run at the command line is now called *chess-variant-sandbox*, matching the
name the site took on the 23rd; the old *newgame* command no longer exists. Nothing about the
games or their scores changed with the name.

### A one-step change no longer copies the game or swaps pieces between sides

The automatic designer now throws away a change that only re-spells the game it started from,
instead of spending a tournament on it; about one change in fifty was such a copy. And *add a
piece*, there and in the designer's *and now try* panel, no longer hands the second player's new
pieces to the first on a game whose armies are written out side by side.

### The board says how hard the computer thinks

Under each computer player the board now says how much it considers before a move, for example
*looks at 6,400 positions before each move*, and beside each of its moves how many moves ahead
it looked. On the public site, a move played before the page was reloaded shows no depth.

### A one-off tournament keeps to its own folder

*For maintainers.*

A tournament run from the command line without naming where to save it now goes into a folder of
its own beside the library's, so it can no longer land on, or overwrite, the games a library
score was measured from. The command's help says where its games go.

### The library opens at once after a restart

The first visit to the library after the app starts no longer waits several seconds while the
games are checked; that work now happens while the app starts up.

### Every score says how many runs it rests on

Badges, the design page's measurement lines, the lines comparing two related games, and the
upkeep panel now say whether a score is the mean of four runs or a single run. Where the runs
behind a score can't be read, the page shows no number rather than a bare one.

### A watch respects the strongest players' ration

Starting a game between two of the strongest computer players is now refused when only one of
their seats is free, with a sentence saying how many are free. Before, such a game could slip
one over the ration. The ration and the daily game limit also hold when two games start at
the same moment. A game against the strongest player that sat idle for ten minutes, or
was reopened after it ended, waits for a free place like a new game, and says so.

### What the ± on a score means, now measured

The note explaining a score's ± range used to say nobody had checked it by playing more games.
Someone has now: the whole library was played again from four fresh starting points, and the
new average landed inside the stated range about five times in six, as it should.

### The score panel counts what the score is made of

A game's page now says how many things its score measures by reading them from the scoring
version the score was taken under. It used to say sixteen, a count from an older version;
today's scores are made of thirteen.

## 2026-09-23

### Game names in one style

Every game in the library now has its name in the same lowercase style, so *Cannonade*,
*Marchfield*, *Reckoning* and *Intransitive* read *cannonade*, *marchfield*, *reckoning* and
*intransitive*, and *chess-sweeping-bishops* reads *chess sweeping bishops*.

### The project is now called Chess variant sandbox

The site, its pages, the invitation email and the picture credits now carry the name *Chess
variant sandbox*, where they used to say *newgame*. Games the computer designed are signed
with the new name too. When this reaches the site everyone is signed out once, so sign in
again.

### A game's page says how its neighbours differ

On a game's page, games the same number of rules away are now listed most alike first, and
each one says in words what kind of thing differs: for example *no rules, 6 mechanisms, 1
piece* between chess and berolina chess. A game nothing is close to, such as shogi, now lists
the four games most like it overall (crazyhouse, courier, shatranj and makruk), and the map
draws its faint line to the first of them.

### Repeated failed sign-ins now have to wait

After ten failed sign-in attempts from one place within a quarter of an hour, the site asks you
to wait a few minutes before trying again. This makes guessing someone's password impractical.

## 2026-09-19

### The explorer is now the map

The library's map is now the steerable drawing that was offered beside it as the new
explorer. You can drag it, fold a family of similar games into one dot, size the dots by
score or by links, tidy the layout and reset it. The old fixed map is gone, and a link kept
to the explorer still opens the same view.

### A new way to explore the library map

The map now has a *new explorer* link, a second drawing of the same seventy games you can
drag, scroll and pinch, with the same lines, colours and detail panels the map already has.
It now also has the map's chips, its search box, its five score filters, names on every dot,
a legend and full keyboard control, plus a *hide the rest* switch the original map does not
have. Whole families of similar games, and the nineteen games far from everything else, can
now be folded into one labelled dot and opened again, and dots can be dragged and tidied.
A dot's size can now be switched between how many games it links to, how it scored, what
people rated it out of five, or how many games have been finished on it, and the explorer can
be narrowed to unbeaten, untried or unplayed games the same way the library page can.

### One-sidedness is judged more fairly, and shown even when it isn't disqualifying

A game used to be refused for favouring one side only if every single decided game went the
same way. Now a game is refused only if it goes one side's way at least two times in three,
which is fairer to a small sample, and *weathervane half-turn* is back in the rankings under
the new reading. A game that leans toward one side without crossing that line now says so
next to its score, instead of the lean going unmentioned.

### Every library score's margin is now confirmed honest

A published score's plus-or-minus range was tested on four fresh, never-before-played game
seeds rather than ones already on file, and it held: a range that claims to cover the true
score about five out of six times actually did, on games nobody had looked at when the range
was written down.

### Play your draft without leaving the designer

The game designer's board has a fourth view, *play it*. A draft that passes its checks can be
played right there on the large board, against the computer, by hand for both sides, or
watched as the computer plays itself. Changing the draft ends the game and says so. Nothing
is saved and no game played this way counts for anything. The *and now try* panel moved up
beside the board, so on a phone it is near the top of the page. The older *play it* button
at the foot still opens the full board page, which has the move list the inline board lacks.

### And now try, inside the designer

The game designer has an *and now try* panel beside the verdict on your draft. It lists the
changes the design loop knows how to make to the game you are editing, each in a sentence,
and one press makes the change, says what it did, and leaves a game you can play. Undo takes
it back. It works on a draft you have never saved, and nothing is kept until you save. On a
phone the panel is at the very bottom of the page for now.

### The designer speaks English, and lends you the library's pieces

In the game designer every rule card now has a title and an opening sentence in plain words,
where it used to show the rule language's own keys. The *add a rule* menu only offers rules
that can go into the game as it stands, and says why for the ones it holds back. A palette
lists every distinct piece in the library, with its moves in sentences and the games that
use it, and you can lift one into your draft with its moves attached. A move along a slanted
line, the kind a nightrider makes, now has a sentence of its own in teach-a-move. A game
with pieces held in hand shows the hand beside the board. Still rough: a piece you add
yourself still has a row of seven controls for each move, and a piece lifted from the
palette arrives without being placed on the board.

### A game you have saved and not published is yours alone

Since the neighbours list arrived on game pages two days ago, a library game's page could
name a saved, unpublished game that was within three rules of it, and say which rules
differed, to anybody looking. That is closed. A saved game that is not published now appears
in no other game's neighbours and nowhere on the map, for anybody but the person who made it
and the person who runs the site. A game taken off the site now leaves the map as well as
the library.

### The designer is built around one big board

The game designer now has one large board in the middle of the page. You set the opening
position up on it, picking pieces from a tray beside it. You teach a piece a move by clicking
the square it should reach and choosing from plain sentences, such as *one square
forward-right* on a hex board, where before you typed numbers. A third view shows where a
piece can actually go from the opening position, with its own army in the way. It draws hex
boards, boards with walls and every other board in the library. The rule cards below the
board still use the rule language's own words, which is the next stage. A game with pieces
held in hand does not show the hand on this board yet.

### A suggestion box

The site has a *suggestions* link in the footer. Anybody signed in can post an idea, either
for a game or a rule, or for the site itself, and sees their own ideas listed under the form.
For now only the person who wrote a suggestion and the person who runs the site can read it.
A post can be up to 2000 characters, and a person can post ten a day.

### The designer's buttons stay in reach on a phone

On a phone the designer's verdict on your draft, and its save, play and watch buttons, now
stay at the bottom of the screen. They used to sit at the end of the editor, about nine
screens down. When a draft fails, the list of what is wrong scrolls inside the bar, so the
buttons never move. On a computer nothing changes.

### Make a new game from the map

On the library's map, clicking a game now offers *and now try*. It lists every change the
design loop knows how to make to that game, each in a sentence, and opens the one you pick in
the editor as a playable draft. Chess has 31. A dashed line between two games that differ
only in their declared rules offers *make the game in between*, which builds the game with
one rule from each side. Ten of the 55 dashed lines can do that, and the others say in a
sentence why they cannot. Nothing is kept until you save, and saving still needs an account.

### A page for everybody who plays

The site has a players page, linked from the footer, for anybody signed in. It opens with
what nobody has claimed yet, which is the rated games nobody has played and the ones where
nobody has beaten the strongest computer player. Under that it lists everybody who has
finished a rated game, beside how many games they have tried, won, and beaten the strongest
computer player on. Nobody appears until they have finished a game, and an empty site says so in a sentence.

### The map says how to get from one game to its neighbour

Clicking a line between two games that differ by one rule now also says what a designer
would do to turn one into the other, in the words of the step that does it. From chess to
king of the hill it reads *the game is now also won by walking the K to the centre*. 23 of
the 56 one-rule lines have such a step. The rest join games that differ by a rule no single
design step changes.

### The map shows scores, and what a rule did to them

On the library's map each game's dot is now filled by what the game scored, from the lowest
score in the library to the highest, with a key beside the map. A game the checks ruled out
carries a ring inside its dot, and a game with no score is hollow. Clicking a line between
two games says what the differing rule did to the score, in the same words the game's own
page uses, and names the three measurements that moved most. The map can also light up just
the scored, unscored or out-of-date games, or the games matching a name. Games that do not
match fade and stay where they are, so no line is left hanging.

## 2026-09-18

### This page

The site and the local app have a *what's new* link in the footer. It shows this changelog
day by day, newest first, to everybody, signed in or not. Older entries were written for the
person who runs the project, so some of them name commands and files a visitor cannot reach.
Entries from here on are written for both readers.

### A map of the whole library

The library has a map view. Every game is a dot, and two games that differ by one rule are
joined by a line coloured by the kind of rule that differs. A dashed line marks a pair two or
three rules apart with nothing between them, which is a game nobody has made yet. Arrows show
which game was made from which. Chess sits in the middle of a star of 22 one-rule neighbours,
and nineteen games with nothing within three rules sit in a row along the bottom. Clicking a
game shows its score as its card does and lists every line touching it, nearest first, with
the rule that differs in words. Clicking a line says what it claims. It works on a phone.
The library's filters do not narrow the map yet, and a find box stands in for them.

### The site shows the four-run scores

The site was deployed this evening with the four-run measurements copied over first, so the
three entries below are live. Every score is the mean of four runs
with its halved margin. Every card and game page names its run count and says how a game's
runs split. Rps race and weathervane half-turn have left the rankings. For the rest of today
the site is deployed after each change you would notice.

### A game cannot combine en passant with a turn of several moves

A game that gave a player two moves a turn and also had en passant let that player capture
their own pawn in passing, and crashed the engine outright if it had a strike rule as well.
No game in the library did either, so no score was wrong. Validation now refuses the pair by
name. The design loop's *lengthen the turn* step drops en passant from the game it makes and
says so.

### Every library score is the mean of four runs

All 71 games now rest on four runs where they rested on one, so the margin of error beside
each score is about half what it was. A game counts as accepted only when a clear majority of
its runs accept it. Under that rule 59 games are accepted and 12 refused. Rps race leaves the
rankings, because three of its four runs refused it, and weathervane half-turn leaves on a
two-two tie. Dull and hierarchy chess show no score where they showed a placeholder. Most
scores moved by about a hundredth; three-check moved most, from 0.80 to 0.74. The site
showed the one-run scores until the evening of the same day.

### Measuring a game on four seeds from the command line

*For maintainers.*

`chess-variant-sandbox measure --seeds four` measures a game on four seeds and pays for the fitted opponent
once. `chess-variant-sandbox measure --adopt <campaign root>` takes in runs that were already played
elsewhere, after replaying a sample of each to check it. `chess-variant-sandbox test` is now the four-run
measurement, and `promote` refuses a game measured on one seed and tells you to run
`chess-variant-sandbox test` first. The library itself is still on one run until the four-seed campaign
finishes.

### Every score says how many runs it rests on

The library's cards and a game's page now name the run count beside each score, so a number
from one run cannot pass for an average of four. Today every game reads *1 run*. When the
four-run library arrives, a game's page will list its four scores and say which claim the
margin of error is making. A game whose runs disagreed will say how they split, where before
it would have shown one verdict as if the runs agreed. A game too few of whose runs could be
read shows *no score*. This reached the site the same evening.

### The site runs everything below

The site was deployed at the end of the day, so every entry under this date that said *with the
next deploy* is live: the neighbours list on a game's page, carry-on games, a retired game's
page and its delete button, Gardner's minichess refused, xiangqi's new score, and the wider
margins of error.

### The groundwork for scores averaged over four runs

*For maintainers.*

You decided that a library game's score should be the average of four runs, and that a game is
accepted only when a clear majority of its runs accept it. The first step is in: a game's
score now lives in a file that can hold one run or four, and every published number is what it
was. The one visible change is hierarchy chess, which showed 0.0 and now shows no score,
because the computer players could not read the game and 0.0 was never a measurement. The
averaged scores themselves arrive after two more steps and a 25-minute run. When they do, rps
race will leave the rankings and `dull` will show no score.

### The first pieces of measuring a game in the browser

*For maintainers.*

A check this morning showed the game engine runs unchanged in a browser, at about 1.6 times
the server's time per thread, and plays the server's exact games. Two pieces of the build
followed. A measurement can now be cut into independent units and gathered back to the same
files, which is what lets a browser spread it over several workers. And one command,
`deploy/pyodide/build.sh`, builds the browser package from a git ref in about a minute, a
13.6 MB download a visitor would fetch once. Nothing on the site uses it yet. You decided how
a browser-made score is trusted: the server checks every game is legal, replays a random
sample, and repeats the whole measurement only when the designer asks for an official score.

### A game's page lists the games a rule or two away

Every game's page now lists its nearest neighbours in the library, nearest first. Each line
says what differs in words, which kind of rule it is (board, pieces, setup, captures, how the
game ends), and what the change did to the score, or *no clear change* where the difference
is inside the noise. Links are found automatically by comparing the rules, ignoring how a
game happens to be written down. Chess has 22 games one rule away. A game with nothing
within three rules shows its nearest four. This is the first stage of a map of the whole
library; the map itself is two stages on. On the site it arrives with the next deploy.

### The margin of error beside a score is wider, and says what it means

The whole library was played three more times, and a score turned out to move by about 0.025
from one run to the next, more than the margin printed beside it allowed. The margin was
short by a standard small-sample correction, which is now applied, so every margin is about
15 percent wider and matches what replays show. No score changed. The library page now says
that the margin describes the run it came from and is not a forecast of a replay. Fewer pairs
of games count as clearly separated on the leaderboard as a result. Still open is the larger
repair you chose, averaging four runs for every library game, which is being designed.

### A game you walked away from can be found again

A rated game you left unfinished is offered back as a *carry on* entry on that game's page and
on your profile, with the opponent, your seat and the moves played. Reloading a game's address,
or opening it from a bookmark, draws the board where it used to drop you in the library. On the
site this arrives with the next deploy.

### Xiangqi was measured again, and its score rose

A fix to how the rules are compiled changed the order one xiangqi piece's moves are listed
in, so xiangqi alone was played again. Its score went from about 0.73 to about 0.78, which is
further than its stated margin of error allows for a change that small. The margin describes
one set of games and not what a second set would give, and a task is open to measure how much
a score moves between runs. Every other game's score is unchanged.

### A retired game keeps its page, and can be deleted

A game that has left the library has a working page again. It shows the rules, the score
and the games played on it, with a *retired* label where the board was. The person who made
the game can delete its rules for good from a button on that page, confirmed once, and the
site's keepers can too. Games played on it, ratings, votes and badges all stay and name it *deleted game*, so
nobody's statistics change. A saved draft cannot be deleted this way yet.

### A game decided by who moves first is caught more often

The check for a game that one seat wins regardless of play now reads the three strongest
computer players as well as the top two. It was tried on three fresh runs of the whole
library first and caught Gardner's minichess every time, which the narrower check missed.
Gardner's now shows as refused and keeps its number, and nothing else in the library moved.
Eleven of the 71 games are refused. The site shows this after the next deploy.

### Badges are visible

Badges you have earned now appear as dated lines on your profile, with the rule under each,
and a game's badges appear on its page with the figures they were earned under. A page at
`/badges` lists every badge and what earns it, the same for everybody, so a rule is readable
before it is met. No page counts badges or ranks people by them. The last earnable badge,
*And another thing* (lost, asked for a rematch, won it), is recorded too. On main, and on the
site at the next deploy.

### The board says when a game counts

A rated game on the site now says so on the board, hides the take-back it would refuse,
and lets you resign it. A rematch of a rated game is rated too, where it had quietly become
a practice game. A watched game names who is playing, and the game page has a watch button.
The replay bar no longer offers *pause* while a person holds a seat, and one line under the
board names the marks it draws. Resigning used to leave the board playing on; it now ends
the game. On main, and on the site at the next deploy.

### Watching two computer players play works on the site again

The site refused a game with no person in it, which was a shortcut from the day a game
became a database row rather than a decision. A visitor can again set both seats to a
computer player and watch a whole game, and every board has a *watch it play itself*
button. A watched game is nobody's game. It moves no rating, earns no badge and counts for
nothing, and it is rationed like two strong games. On main, and on the site at the next
deploy.

### The profile shows what you have done and what is still open

A signed-in profile now lists the dated things you have done on the site, with each one's
rule beneath it, and what is still open for you, grouped by kind of game. The table says how
many of each game's computer players you have beaten and how far your rating is from the
top. On main, and on the site at the next deploy.

### Piece moves in words

A game's page now says how each piece moves in words a player would use, such as *one
square forward, not to take* or *any distance along all four diagonals*, for every kind of
move the rule language has, hexagonal boards and turning pieces included. The rule as
written is one tap away behind the (i). On main, and on the site at the next deploy.

### Cards and game pages say what a score settles

Under the score ordering, a game's card and its page now say how many measured games it
clearly beats, how many clearly beat it, and how many are too close to call, instead of
implying a rank the score's margin cannot support. On most games *too close to call* is the
largest of the three. The band is absent when a score is out of date. On main, and on the
site at the next deploy.

### The site's own pages look like the rest of the site

The sign-in, profile, credits and privacy pages now share the app's header
and stylesheet. A game named on the profile opens that game instead of the library. The
profile lists what you have played first, then the rest by name. The credits read as a list of artists rather than a list of files. On main, and on
the site at the next deploy.

### A game's page names its firsts

A game's page now says who played it first, who the highest-rated person on it is, and
everyone who has beaten its strongest computer player, in order, with the date and the side
they played. Each line has an explanation behind its (i), and a line is absent when there is
nothing to say. On main, and on the site at the next deploy.

### Rate a game with five stars

A signed-in member can now rate a game one to five stars on its page, under a panel called
*What players make of it*, and sees their own vote beside everybody's average with its
count. A visitor who is not signed in sees the average and an invitation to sign in. The
library's cards and its star-rating order use the same number. On a phone the game's name
now sits above the play panel. On main, and on the site at the next deploy.

### Errors appear where you are looking

An error now lands under the button you pressed, on a line of its own, or pinned to the
bottom of the screen with a dismiss when no button raised it. It no longer sits at the top
of a page you have scrolled past. On main, and on the site at the next deploy.

### Players and designers earn badges

A game that five people have played and rated at four stars or better earns *Crowd pleaser*,
and *Proved the clanker wrong* on top if its own score is low. Rating a game after finishing
three games on it earns *Speaks from experience*, which never reads the stars you gave.

A player now earns *Well read* at half, three quarters and all of the rule building blocks
that the rated games between them use, counted over the games they have finished.


Saving a game on the site now records the designer badges. They are for using a quarter, a
half or three quarters of the rule building blocks across your listed games, for having both
beaten a game's strongest computer player and designed a game that passes the check, and
for the three levels of building on somebody else's game. A game built from a library game
earns nothing at any level. No page draws them yet. On main, and on the site at the next
deploy.

### The library opens on the best games, and a game page fits a phone

The site's library now opens on the best-scoring games rather than alphabetically. A game
with an out-of-date score sorts last beside the unmeasured, because the site shows no
number for it. The library says *Loading the library…* while it warms instead of showing a
blank page, every word on a card has an explanation, the order-by menu leads with the four
orders a player uses, and every page change starts at the top. Tapping the (i) beside an
out-of-date score now says why. On a phone a game's page opens on the play panel. On main,
and on the site at the next deploy.

### The site records play badges

Finishing a rated game on the site now records the badges it earned. They are for beating
each named computer player on more and more games, a win over somebody rated far above you,
a draw with a game's strongest computer player, a win on one game of every kind, wins by four
different endings, and a game that ran to the move limit. A win by the other side leaving
earns nothing, and practice games count for nothing. The privacy notice says the site keeps
them. No page draws them yet. On main, and on the site at the next deploy.

### A game's page shows both ratings for a computer player

Each computer player on a game's page now carries two numbers under two headings, *in the
tournament* and *on the site*. The first is the fixed scale the computer players were placed
on, centred on zero. The second is the scale people are rated on, and it moves as people beat
the computer. The panel about beating the computer uses the same two labels, so +802 and
2302 read as one opponent on two scales. On main, and on the site at the next deploy.

### The designer has no dead ends

A half-finished edit no longer takes the board away. The last valid board stays, dimmed,
with the errors beside it, and it can be repaired by clicking on it. Changing a piece's
letter now carries to every place that names the piece. Ctrl+Z and Ctrl+Y undo and redo, and
a reload keeps the draft. A new rule card arrives filled in with pieces from your own game,
so castling names your royal piece and a corner piece. When random games run out of clock,
the designer says which counter ran out and what it is set to. Seventeen rule and template
pairs still refuse, each for a reason about the game, such as a rule about a royal piece on
a game that has none. On main, and on the site at the next deploy.

## 2026-09-17

### Three fresh runs of the whole library, read for one-sided games

*For maintainers.*

All 71 games were played three more times with different random draws, to test a wider
check for games decided by who moves first. It caught Gardner's small chess every time, and
queen bomb, which had escaped narrowly, reads as fair. Two surprises came with it. Rps race
is swept by its first player in all three fresh runs, though its stored run hid that. And
weathervane half-turn, refused today, is not refused in any fresh run. No score or verdict
has changed yet. Whether to adopt the wider check, and what to do about the two surprises,
is a question in `docs/decisions.md`.

### A game that leaves the library leaves the site too

When a game is taken out of the library, the site now stops listing it and starts no new
games on it. Games people already played on it stay on their profiles, marked *retired*, and
the game returns by itself if its rules come back. This also closes a hole: a retired game
could still have been played by anyone who knew its address. Whether a retired game's own
page should keep working is still to be decided.

### "Watch it play itself" plays a real game

The designer's watch used the weakest searching computer player. On a draft where every
piece can turn, that player could not see one move ahead and spent half its moves turning
pieces, which looked like a broken game. The watch now uses the *hard* player, which costs
a few hundredths of a second a move, and the board says who is playing and how many legal
moves the position offers. Still open: the computer players' built-in judgement puts no
value on which way a piece faces, so even strong players turn pieces more than they should.
On main, and on the site at the next deploy.

### The square-grid hex game is retired

The older hexagonal chess, which laid a hexagon out on a square board with walled-off
corners, is out of the library. Hex glinski promoting, on a true hexagonal board, is the
hexagonal game now. The library holds 71 games and all 71 scores read current. The old
game's measurements are kept as history. The site still lists it until the next deploy.

### The designer says what a save did

After a save, the designer now says beside the save button whether the draft was kept, how
far the playability check got, and whether the game was listed and why. A refused save shows
its reason in the same place. This is on main and reaches the site at the next deploy.

### The site is up to date

The site was deployed this evening. It now shows a current score for all 72 games, where it
had shown every score as out of date since 14 September. It also has what was waiting on
main. Boards can be hexagonal. People can be invited to the site by email. A
signed-in person can save a draft from the designer, and the draft is listed only if it
passes the playability check. The game page and the designer no longer draw buttons the
site refuses.

### Two more games are refused as decided by who moves first

The replay's predictions are graded and written up. The check that refuses a game when one
seat wins nearly every game between the strongest computer players now refuses three games,
where it had refused one. Tunnel chess vertical and the sweeping-bishops chess are won by
the first player 16 games to none. Weathervane half-turn is won by the second player 14
games to 1. A wider version of the same check would also refuse Gardner's small chess. It was seen on the
same games it was first noticed on, so fresh games are played before it is adopted.

### Saving a draft cannot hang the site

The check that runs when someone saves a draft now has a 15 second limit. A draft whose
rules are too slow to check is refused with a sentence that says to try a smaller board or
a lower move limit, and nothing half-saved is left behind. Ordinary games check in well
under two seconds. That limit was the last thing the next deploy waited on.

### Every game's score is current again

All 72 games in the library have a current score, taken on today's engine and rules. That
includes the new hex chess example and three games that had never been scored. Run locally,
the library shows them. The site still shows the old out-of-date scores until the next
deploy, which waits on a time limit for the check that runs when someone saves a draft.

## 2026-09-15

### The designer writes sweeping pieces and hexagonal boards

The designer's move menu now offers pieces that sweep, and its board form can make the cells
hexagons, so neither needs hand-written JSON. On the site, the designer no longer shows
buttons the site refuses (measuring, nominating, uploading an icon, saving an opponent); a
sentence says why instead. The site gets this with the next deploy.

### A game's page on the site stops offering what it cannot do

On the site, a game's page no longer shows buttons that answer *no such endpoint*, such as
*work them out again* or *measure it again*. A short sentence stands where each one was, saying
the site does not run tournaments yet. The score panel now says in one plain sentence whether
a score is current, and why not when it is not. The opponent menu lists you first, then the
seven difficulty levels, then everything else. Run locally, the page still shows every button.
This arrives with the next deploy.

### Saving a draft on the site comes back

A signed-in person will be able to save a draft on the site again, check that it plays, and
follow its progress. A visitor is asked to sign in. A saved draft that passes the playability
check appears in the library; one that fails stays unlisted and opens only by its link. Either
can be played for practice, and the site's keepers can take one down. The long
measurement, nominating a game, and re-rating a game's computer players stay off on the site.
This arrives with the next deploy, once the playability check has a time limit.

### Invite people from the Maintenance page

*For maintainers.*

An admin can now invite somebody by typing their email address into the site's Maintenance
page. The page lists who was invited, by whom, when, and whether they joined. An address that
already has an account is refused, and the refusal names the account. Typing an address again
before that person has joined sends a fresh link, and the earlier link still works. The
`chess-variant-sandbox site invite` command does the same thing. It is not on the site yet; the next deploy
comes after the replay.

### Hexagonal boards

A game can now say its board is hexagonal. Its rules text then calls the board hexagonal,
and its boards are drawn as hexagons. The checker refuses the spellings that used to produce
a plausible but wrong hex game without complaint. A promotion zone can now be any set of
squares rather than whole ranks, which suits a hexagon's far edge and also a shaped zone on a
square board. A new example hex chess game uses that zone; it has no score yet, and the
replay gives it one. The designer's board form has no hex
option yet, so a hex draft has to be written by hand.

### Scores no longer look current when they are not

A stored score now reads out of date when any of the code behind it changes. Before today,
changes in nine places could move a score and leave it looking current: the game loop, the
tournament schedule, how played games are read back, parts of the engine the analysis reads,
and the complexity count. Every score reads out of date until the replay anyway, so nothing
changes on screen yet.

## 2026-09-14

### The site runs today's work

*For maintainers.*

Everything above is now live on the site, along with the previous days' work that had not
been deployed: usage counting, the designer's measuring flow, and schemas 11 to 13. Game
scores still read *out of date* until the whole library is replayed, and the site will be
deployed again when that is done. Saving a draft on the site is still switched off; turning
it back on is queued. The site's usage counter was checked on the server after the deploy
and is recording, including its own share of computer time and memory.

### The library can show games you have not tried, and games nobody has played

The library page has two new filters beside *unbeaten*. One shows the games you have not
played yet, and it needs you to be signed in. The other shows the games nobody has played
at all. Each also puts a small mark on the matching cards.

### You can type into a piece's fields in the designer

Clicking into a piece's letter, name, worth, icon or move settings used to redraw the whole
form, so whatever you typed was lost. The click now just highlights the piece, and the
fields keep what you type. Nothing on the designer page is covered by another element at
a narrow window or on a phone.

### A game whose goal square sits behind a wall no longer passes the rules check

A game can say that reaching a square wins, and for the second player that square is the
first player's reflected. The rules check and the engine reflected it differently, so the
check could pass a game whose goal for one player sat on a wall and could never be reached.
The design loop was producing exactly such variants. Both now reflect the same way, the
loop no longer walls off a goal, and no existing game changes.

### A game the first player always wins is kept out of the rankings

A new check reads the games between the two strongest computer players in every
measurement and refuses a game the first mover wins nearly every time, the way a game that
never ends is refused. Tunnel chess (vertical) drops out of the rankings on it. Gardner's
minichess, the other known case, is not caught from the stored games alone. Rather than pay
for extra games in every measurement, the coming replay will test a cheaper reading that
may catch it. The design loop also no longer tells the proposer anything about fairness
from the old first-player figure, which was too noisy to grade any game.

### The designer's skirmish template is a game again

The first template the designer offers, a skirmish of soldiers, could not be won by any of
the computer players. Stepping into range only lost the soldier, so both sides shuffled
sideways until the game was called a draw. Captures in the template are now compulsory, so
the lines meet and the trades that follow decide it. Every check on a fresh draft now reads
green before you change a rule. The playability check itself was not loosened.

### A sweeping piece is measured on the moves it has

Two of the score's ingredients read a game's geometry straight from its rules, and until
now they did not know about the new sweeping move. A game whose pieces sweep was scored as
if those pieces could not move at all, which made it look like a different game. They now
read sweeps like any other move. No existing game is affected, and the example game with
sweeping bishops will score right when the library is next replayed.

Score ranges will be worked out by re-drawing whole pairings of players from the next
scoring version, which makes them wider and more honest.

### A piece can sweep along its line

A game's rules can now give a piece a sweeping move. It jumps to any square along its lines
and removes every piece it passes over, its own included. A king standing on a sweeper's
line is in check, just as it would be from a rook. A piece that can never be captured stops
the sweep. The new example game is chess with sweeping bishops, where white has a mate in
one from the start, so expect it to score poorly. Every existing game plays as before, and
the whole library's scores wait on the replay.

### The designer's buttons stay clickable on a long game

On a game with a long rules form, the board panel in the designer stayed in view while you
scrolled, but it slid over the *Which game this is* panel, so its buttons could not be
clicked. The whole right-hand column now stays in view instead, with its own scrollbar,
and nothing hides under the board.

### "Show the stuck game" shares show me's limit on the site

On the site, the designer's *show the stuck game* button now needs an account, like *show
me*, and the two buttons share one limit of 50 presses an hour. On your own machine nothing
changes. The message you see at the limit now names both buttons.

The site's hourly usage counts are now saved every five minutes, together with a reading of
how much computer time, memory and disk the server has left, so a restart no longer loses
them. As before, no count names a person.

### A deploy no longer leaves the library slow for the first visitor

After a deploy that copies new results to the server, the first request for the game
library used to take about 20 seconds, and the deploy's own final check gave up waiting.
The deploy now opens the library once on the server before it finishes, so the check and
the first visitor both find it ready. The deploy guide also gained a short list of things
to check after every deploy.

### Hexagonal chess is in the archive

A version of Glinski's hexagonal chess, played on a 91-cell board, is now in the archive.
It needed no new rules, and it scores close to ordinary chess, so the computer players and
the index handle a hex board as well as a square one. It has no promotion yet, because the
rules have no way to mark a hexagon's far edge, and that is probably why strong players
draw it more often than weak ones.

### A low depth score is weaker evidence than a high one

The index's depth score measures how much a game rewards deeper thinking, by pitting
computer players of rising strength against each other. A match one step stronger than the
strongest of them, run on every committed game, shows that a high depth score mostly holds
up. A low one can mean the computer players are too weak to see what the game is about.
King of the hill and racing kings are the clearest cases, and intransitive is among them.
Read a low depth score with that in mind.

A second match, one step lower, checked four games whose high depth scores fell away. Two
of them, sardines on a torus and crazyhouse on a torus, really are deep, and the rated
players simply top out. The other two are weaker than their scores say. Tunnel-vertical in
particular is a forced win for whoever moves first, once a computer player looks a little
way ahead, even though it scores near the top on depth. Whether the index should catch
games like that is being looked into.

That look found a second one: Gardner's minichess is also won by whoever moves first once
the players think hard, and both games pass every check today. It also found that the
index's "first-player advantage" number is too noisy to trust for any single game, because
it rests on too few games. Whether to reject, mark down or simply note games like these is
still to be decided.

### Games with neutral wild pieces are no longer marked down for them

A game with neutral pieces that no player controls, like the wolves games, used to lose
points for "idle" piece types that no player could ever move. It no longer does. The
wolves games gain a little under a tenth on that part of the score, and no other game
moves. The new figures show up after the library's next replay.

### A capturing piece can throw what it strikes

A game's rules can now let a capture push the struck piece onto a nearby empty square of
the attacker's choosing, rather than removing it. The play page asks where to place it, and
the move list names the square. The new example game is a version of the hierarchy game
with this rule. Every existing game plays as before. One rare case also got stricter: in a
game whose captures leave the struck piece standing, an en passant capture that promoted a
piece giving check could previously be allowed when it should not have been.

### A piece can merge with one of its own

A game's rules can now let a piece move onto one of its own pieces, and say what the two
become. The new example game is chess where a pawn moving onto its own rook becomes a
queen. Kings can never be merged away. Every existing game plays exactly as before. The new
rule has no score yet, and the whole library's scores are waiting on a replay that runs
after the next two rule additions, so they read as out of date until then.

### The wolves game is fair to both players

The wolves game looked like it favoured the first player. Five versions with the wolves
starting in different places show no first-player edge in any of them. The earlier figure
came from too few games to judge.

## 2026-09-13

### The designer takes a draft all the way to a score

The game designer now shows each step from a draft to a measured score, with what each
step costs. From a saved draft you can queue the full measurement and nominate the game for
the library. The page draws the family of drafts you have saved as a tree. All of this works
on your own machine. On the public site these steps stay off until the project decides which
to offer.

### The site counts its own use, without counting anybody

The site can now keep hourly totals of what gets used: requests per page and action, the
computer time they take, and how many different signed-in people made them. Visitors
without an account count only as activity. No total names a person, stores an address or
sets a cookie, and the privacy page says so. The totals are held in memory for now and
start being saved once the five-minute sampler lands, which should happen before more
people are invited.