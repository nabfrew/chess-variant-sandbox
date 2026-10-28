"""The badges of `docs/archive/plan-meta-game.md` §2.9, as boxes E3 and E5 to E9 built them.

Those six boxes and their *done when* are in `docs/archive/plan-meta-game.md`; a bare §
below is the live plan's.

Seven of them are readings over leaderboard rows the site already writes: U2 *beat easy /
medium / hard / hardest on n games*, N2 *Giant-killer*, N3 *Held the top*, N4 *The grand
tour*, N7 *Every which way*, N8 *The long way round* and N9 *And another thing*. Five more
read `games` rows: U5 *The whole toolbox*, N6 *Both hats*, and *Family tree*'s three
levels (N12). U4 *Well read* reads both, since it counts the building blocks of the games
a person has played.
Three read `votes` as well: N13 *Speaks from experience*, N1 *Crowd pleaser* and U1
*Proved the clanker wrong*.

`site/awards.py` knows how to write a row and `site/milestones.py` how to read a live
milestone. This module is the third thing, the table of what each badge is and the reader
that says which of them a person has now earned. The reader is pure, in
`milestones.read_from`'s idiom -- rows and tables go in, awards come out, nothing opens a
connection -- and `grant_for_match` and `grant_for_save` are the two functions that do
hold a connection, so `matches._finish` and the save route can each write the badges of
their event inside that event's own transaction (`grant_for_vote` and `grant_for_proof`
are two more of those).

## What a row has to be before any badge sees it

A finished leaderboard match with a result, both seats held by somebody, no `source`, and
nobody having left. Sandbox rows and the local application's rows are dropped at the door
as they are in `milestones`, and so are resignations and the idle sweep's forfeits, which
§2.9 leaves out of N7 and which no other badge here should be earnable by either. Every
figure a badge quotes is read off a row. Nothing is worked out (`CLAUDE.md` rule 3).

## What a `games` row has to be before a designer badge sees it

*Passes the screen* is §2.9's phrase for `origin = 'designed'` with `screen_passed`,
which is what `drafts.keep` records from the playability check. It used to be read off
`visibility = 'public'`, when a pass listed the game; since Q503a a save lists nothing
(`0016_drafts.sql` copied the old reading into the column, so no badge moved). So a
draft counts: `plan-game-pipeline.md` §5 (Q503a) has *passes the screen* badges read
`screen_passed`, and `docs/archive/plan-meta-game.md` has D2 keep its meaning, *a draft that passed*.
The rule sentences said *and is listed* from the days a pass listed the game, and Q543
took the words out rather than add a check, which would have made every designer badge
unearnable until publishing opens and contradicted rows already written. A killed
game (`games.revoked`) counts for nothing at any level, since the kill switch overrides
the check. A designed game is told from a library game by `origin` and never by its
owner, because `chess-variant-sandbox site load` gives every library game the maintainer's account.

## The six over rows

- **U2** is four series, one per named rung of the picker, with steps at 1, 5, 10 and 25
  different games. A win over a stronger named rung fills the weaker series on that game
  too, so nobody replays *easy* on 25 games after beating *hardest* there.
- **N2** *Giant-killer* is a win over a computer player whose rating going into the match
  was `GIANT_GAP` above the person's, with the person's own rating established at the
  time. Both figures are the row's, so a table that moves afterwards cannot make or
  unmake one.
- **N3** *Held the top* is a draw against the top of a game's table, read off the `top`
  bit the match was started with.
- **N4** *The grand tour* is a win over the top on one game of every kind the rated games
  cover, the same bins M6's *one of every kind* walks.
- **N7** *Every which way* is wins by `ENDINGS` different endings.
- **N8** *The long way round* is a finished game that ran to the move limit.
- **N9** *And another thing* is a loss, the rematch the board's control started on it, and
  a win in that rematch. `matches.rematch_of` is the link, and only the control writes it,
  so the badge reads the person's own click and never a guess about two matches in a row.

## The one over both

- **U4** *Well read* is a series over the building blocks the games a person has finished
  draw on between them, against the blocks the rated games use between them at that
  moment. It takes the games off the person's rows and the blocks off `games.spec`
  through `spec/rules_used.used`, which is the one reading of what a spec draws on.

## The five over designs

- **U5** *The whole toolbox* is a series over the building blocks a person's passing
  designs draw on between them, counted once each however many games use them, with steps
  at a quarter, a half and three quarters of every block there is. What a spec uses comes
  from `spec/rules_used.used` and from nothing else here.
- **N6** *Both hats* is one person who has beaten some game's top and owns a passing
  design. Either event can be the second one, so both hooks ask for it.
- **Ancestor**, **Descendant** and **Family tree** are N12's three levels. Every link
  crosses between two people, the library is not a person, and the top level takes three.
  §2.9 reads all four of those narrowly on purpose, because widening a narrow reading
  later is a pass over rows that still hold their dates, while narrowing a wide one would
  mean removing kept rows.

## The one over people

- **E11** *Beat the maker* (N11) is a win over a designed game's owner, on that game, in a
  rated game between two people (`archive/plan-return.md` §3 C4, Q606). It sits on the game with
  the winner as holder, once per game for each winner. The owner is read off `games` by
  `origin`, as `designer_of` reads it, so a library game earns nothing.

## The one over ratings

- **N13** *Speaks from experience* is a series over the games a person has rated after
  finishing three of them there. It reads who rated which game and when, and never the
  star, the tags or the note -- `Rated` cannot hold a star, which is a firmer statement of
  that rule than a line somebody has to remember. §3 says plainly that the badge cannot
  pass test 6 whole, so the design narrows what a rating can buy: the value is not read,
  one rating per game counts, play comes first, and nothing on the page prompts for it.

## The two on the game

- **N1** *Crowd pleaser* is `CROWD_PEOPLE` different people with a finished game on it and
  a mean of `CROWD_STARS` or better from as many voters. **U1** *Proved the clanker wrong*
  is N1 and a current measurement whose headline and whole range sit under
  `CLANKER_SCORE`. The score comes from `awards.score_evidence` and from nowhere else, so
  a stale, missing or never-measured game grants no U1 and waits.
- Both are dated by the later of the two counts landing, and both keep the figures that
  completed them. A vote that drops the mean afterwards removes nothing, and the walk over
  the ratings in date order is what makes that true of a badge nothing had yet written.
- On a designed game the owner's own games and vote are left out and the owner is the
  `holder`; on a library game nobody is left out and the `holder` is null, because
  `chess-variant-sandbox site load` gives every library game the maintainer's account.

## The two over proven positions

- **Every proven position** and **Game after game** (Q993e, `docs/archive/plan-proven.md` §4) read
  `proven_marks` against the positions a game's record offers on the day. The first is a
  win from every one of them, each played out against hardest (`converted`; a `found`
  mark never counts, since the page shows the proved move to anybody who asks), once per
  game with the person as holder. The second is a series over how many games a person
  holds the first on, at `PROVEN_STEPS`, read off the kept rows, so a re-proof that adds
  positions takes nothing away. The site grants both when a try ends in the person's
  win, in that finish's transaction (`grant_for_proof`). A visitor has no marks, so earns
  neither. These two are the only badges a practice game can earn: a try from a proven
  position is practice by its nature, with no take-back.

## How often each is earned

§2.9 fixes this for N8 and N9, *once per person*, for *Ancestor*, and for the two upper
levels, *once per game*. It leaves the rest open, so the shape of the badge decides. A
badge about one game sits on that game -- N2 and N3 as M2 and M3 do, *Descendant* and
*Family tree* on the game that was built on with its owner as `holder`, and N1 and U1 on
the game they are about. A badge about the library or about a person's whole record is
held once, with `game` null and what it counted in its evidence. U2, U4, N4, N7, N9, U5,
N6, N13 and *Ancestor* are the second kind. E11 is the first kind, on the game, and so is
*Every proven position*; *Game after game* is the second.

## What is stored

§2.9's rule for a badge earned by play: the evidence holds the ids of the matches that
completed it, and the `rule` holds the thresholds in force, so a row still explains itself
after a constant here moves. A rules badge stores *n of N*, the schema version and the
names counted. Neither field ever holds a handle.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from chess_variant_sandbox.loop import grid
from chess_variant_sandbox.site import awards, drafts, milestones, quota, rating
from chess_variant_sandbox.site import matches as match_rows
from chess_variant_sandbox.spec import rules_used, schema
from chess_variant_sandbox.web.human import RESIGNED
from chess_variant_sandbox.web.opponents import ladder

if TYPE_CHECKING:
    from chess_variant_sandbox.site.proven import Record

# ------------------------------------------------------------------ the kinds and steps

BEAT = ("beat-easy", "beat-medium", "beat-hard", "beat-hardest")
GIANT_KILLER = "giant-killer"
HELD_THE_TOP = "held-the-top"
GRAND_TOUR = "grand-tour"
EVERY_WHICH_WAY = "every-which-way"
LONG_WAY_ROUND = "long-way-round"
AND_ANOTHER_THING = "and-another-thing"
WELL_READ = "well-read"
WHOLE_TOOLBOX = "whole-toolbox"
BOTH_HATS = "both-hats"
ANCESTOR = "ancestor"
DESCENDANT = "descendant"
FAMILY_TREE = "family-tree"
SPEAKS_FROM_EXPERIENCE = "speaks-from-experience"
CROWD_PLEASER = "crowd-pleaser"
CLANKER = "proved-the-clanker-wrong"
BEAT_THE_MAKER = "beat-the-maker"
EVERY_PROVEN = "every-proven-position"
GAME_AFTER_GAME = "game-after-game"

NAMED_PLAYERS: tuple[str, ...] = ("easy", "medium", "hard", "hardest")
"""The rungs U2 is a series over, weakest first.

The names alone. Which computer player each one *is* -- the agent string with its node
budget, which is a player's identity -- comes from `opponents.ladder()`, so a budget that
moves in the picker moves here with it and is never written down twice.
"""

GAME_STEPS: tuple[int, ...] = (1, 5, 10, 25)
"""U2's steps, in different games. The user's answer of 2026-09-17."""

PROVEN_STEPS: tuple[int, ...] = GAME_STEPS[1:]
"""*Game after game*'s steps, in different games finished (`docs/archive/plan-proven.md` §4,
written before any person's marks were read): U2's ladder without its 1, which *Every
proven position* already is on each game, so a reader learns one ladder. 5 is reachable
with the seven games that offer positions today, 10 and 25 once the refresher has proved
the library's (Q993d), and nothing above 25, past which the series would reward doing a
lot (test 6)."""

PROVEN_WAYS: tuple[str, ...] = ("converted",)
"""The marks *Every proven position* reads: a try played out against hardest and won
(`proven.CONVERTED`). *Find the move* is not one of them, because the page shows the
proved move to anybody who asks, and a kept badge cannot rest on a mark handed out on
request (`docs/archive/plan-proven.md` §4). Spelled here rather than imported, so that the table
reads no record module at import; `tests/test_site_proven_badges.py` holds it to
`proven.CONVERTED`."""

GIANT_GAP = 300.0
"""N2's gap in rating points, over which the opponent counts as far above."""

ENDINGS = 4
"""N7's k, the different endings a person's wins have to cover."""

MOVE_LIMIT: tuple[str, ...] = ("ply_limit", "scored")
"""The two endings N8 reads as *ran to the move limit*.

`rust/src/game.rs` writes both: a game that reached the limit undecided, and one a
scoring rule decided when it got there. The engine keeps them apart because one is a draw
and the other a result, and the badge is about the length rather than the verdict.
"""

READING_STEPS: tuple[int, ...] = (50, 75, 100)
"""U4's steps, as a share of the building blocks the rated games use between them. The
user's answer of 2026-09-17.

A share of what is in play rather than of every block the schema offers, because §2.9
asks for the blocks *some rated game uses*: a block no game on the site uses is one
nobody could read about, and the top step has to be reachable by playing.
"""

TOOLBOX_STEPS: tuple[int, ...] = (25, 50, 75)
"""U5's steps, as a share of every building block there is. The user's answer of
2026-09-17. Against today's 30 blocks the three steps stand at 8, 15 and 23.

A share rather than a count, because `rules_used.available()` grows with each schema
version. A granted row keeps the count it was earned under in its own `rule`, so a later
schema cannot make an old row read as unearned.
"""

PLAYED_BEFORE_RATING = 3
"""N13's three finished games on a game before a rating there counts.

§2.9 chose it from 1, 3 and 5. One game can be a ten-move loss and five would turn the
badge into one for playing, so three. The user has not seen the number.
"""

RATING_STEPS: tuple[int, ...] = GAME_STEPS
"""N13's steps, in different games rated this way.

U2's ladder on purpose, so that a reader learns one ladder rather than two.
"""

CROWD_PEOPLE = 5
"""N1 and U1's count, for both the people who played and the people who rated.

The user's answer of 2026-09-17, chosen from 5, 10 and 20. One number covers both counts
because §2.9 names the same five for each.
"""

CROWD_STARS = 4.0
"""N1 and U1's mean, over `CROWD_PEOPLE` ratings or more. The user's answer of
2026-09-17, chosen from 4.0 and 4.5."""

CLANKER_SCORE = 0.65
"""U1's mark, which the headline and both ends of its range have to sit under. The user's
answer of 2026-09-17, chosen from 0.65 and 0.75.

*Under*, so a range that reaches the mark earns nothing. A game whose measurement has no
range at all earns nothing either: the badge's whole point is that the index was low, and
a headline with no ends does not say so.
"""


def _needed(share: int, total: int) -> int:
    """How many building blocks a step asks for, rounded up.

    25% of 30 is 7.5, and a badge cannot be earned on half a rule, so the step lands at
    the eighth.
    """
    return -(-share * total // 100)


@dataclass(frozen=True)
class Badge:
    """One badge, as the table holds it.

    `name` is what it is called, `sentence` what a holder's line says with the step filled
    in, and `rule` the one sentence naming what it reads, which the plain list of badges
    shows to everybody whether or not they hold it (§3 test 7). `steps` are the series'
    tiers and are empty for a badge with none. `thresholds` is what a granted row copies
    into its own `rule`, so the row stays readable after a constant moves.
    """

    kind: str
    name: str
    sentence: str
    rule: str
    steps: tuple[int, ...] = ()
    thresholds: dict[str, Any] = field(default_factory=dict)


def _rungs() -> tuple[tuple[str, str, str], ...]:
    """The four named rungs, weakest first, as (name, what it does, agent string).

    Read off `opponents.ladder()`, each rung's name (*easy*) and plain words (*looks at
    400 positions before each move*). A rung the picker renamed drops out here, and
    `tests/test_badges.py` is what refuses that rather than a check at import: a missing
    rung is a badge that quietly stops being earnable, which a test should say out loud.
    """
    found: dict[str, tuple[str, str]] = {
        entry.name: (entry.about, entry.agent) for entry in ladder()
    }
    return tuple(
        (name, found[name][0], found[name][1]) for name in NAMED_PLAYERS if name in found
    )


def _series_badges() -> dict[str, Badge]:
    table: dict[str, Badge] = {}
    for index, (name, gloss, _agent) in enumerate(_rungs()):
        table[BEAT[index]] = Badge(
            kind=BEAT[index],
            name=f"Beat {name}",
            sentence=f"A win over {name} on {{tier}} different {{games}}.",
            rule=(
                f"A win over the computer player called {name}, which {gloss}, on that "
                f"many different games. A win over a stronger named player counts on "
                f"that game too."
            ),
            steps=GAME_STEPS,
            thresholds={"steps": list(GAME_STEPS)},
        )
    return table


BADGES: dict[str, Badge] = {
    **_series_badges(),
    GIANT_KILLER: Badge(
        kind=GIANT_KILLER,
        name="Giant-killer",
        sentence="A win over a computer player rated far above you.",
        rule=(
            f"A win over a computer player whose rating going into the game was at "
            f"least {GIANT_GAP:.0f} points above yours, once your own rating had settled."
        ),
    ),
    HELD_THE_TOP: Badge(
        kind=HELD_THE_TOP,
        name="Held the top",
        sentence="A draw against this game's strongest computer player.",
        rule=(
            "A draw against this game's strongest computer player, as it stood when the "
            "game started."
        ),
    ),
    GRAND_TOUR: Badge(
        kind=GRAND_TOUR,
        name="The grand tour",
        sentence="A win over the strongest computer player on one game of every kind.",
        rule=(
            "A win over the strongest computer player on one game of each kind the rated "
            "games cover, for every question a kind answers, such as how the game is won "
            "or how big the board is. The kinds are the ones the library held when it "
            "was earned."
        ),
    ),
    EVERY_WHICH_WAY: Badge(
        kind=EVERY_WHICH_WAY,
        name="Every which way",
        sentence="Wins by four different endings.",
        rule=(
            f"Wins that between them ended in {ENDINGS} different ways. A game somebody "
            f"resigned, or one resigned after {quota.IDLE_SAID}, is not an ending."
        ),
    ),
    LONG_WAY_ROUND: Badge(
        kind=LONG_WAY_ROUND,
        name="The long way round",
        sentence="A game that ran all the way to the move limit.",
        rule="A finished rated game that reached the move limit, once.",
    ),
    AND_ANOTHER_THING: Badge(
        kind=AND_ANOTHER_THING,
        name="And another thing",
        sentence="You lost a game, started another from the same board, and won it.",
        rule=(
            "A rated game you lost, a game of the same rules started again from "
            "that board rather than from anywhere else, and a win in it. Once."
        ),
    ),
    WELL_READ: Badge(
        kind=WELL_READ,
        name="Well read",
        sentence="The games you have played between them use {tier}% of the building blocks.",
        rule=(
            "Rated games you have finished, using between them that share of the "
            "building blocks the rated games draw on, which is the five ways a piece can "
            "move and the special rules. A building block counts once however many games "
            "use it."
        ),
        steps=READING_STEPS,
        thresholds={"shares": list(READING_STEPS)},
    ),
    WHOLE_TOOLBOX: Badge(
        kind=WHOLE_TOOLBOX,
        name="The whole toolbox",
        sentence="Your designed games between them use {tier}% of the building blocks.",
        rule=(
            "Games you designed that passed the playability check, drafts included, using "
            "between them that share of every building block the rules offer, which is "
            "the five ways a piece can move and the special rules. A building block "
            "counts once however many of your games use it."
        ),
        steps=TOOLBOX_STEPS,
        thresholds={"shares": list(TOOLBOX_STEPS)},
    ),
    BOTH_HATS: Badge(
        kind=BOTH_HATS,
        name="Both hats",
        sentence="You have beaten a game's strongest computer player and designed a game.",
        rule=(
            "A win over some game's strongest computer player, and a game of your own "
            "that passed the playability check, even as a draft. Whichever of the two "
            "comes second earns it."
        ),
    ),
    ANCESTOR: Badge(
        kind=ANCESTOR,
        name="Ancestor",
        sentence="You built a new game from somebody else's design.",
        rule=(
            "A game of yours that passed the playability check, even as a draft, built from "
            "a game another person designed. Building on your own game, or on a game "
            "that came with the site, earns nothing."
        ),
    ),
    DESCENDANT: Badge(
        kind=DESCENDANT,
        name="Descendant",
        sentence="Somebody else built a new game from this one.",
        rule=(
            "Another person saved a game built from this game of yours, and theirs "
            "passed the playability check. It counts from the save, while it is still a "
            "draft, and does not wait for it to be published."
        ),
    ),
    FAMILY_TREE: Badge(
        kind=FAMILY_TREE,
        name="Family tree",
        sentence="A game built from this one was built on again by a third person.",
        rule=(
            "Somebody built a game from this game of yours, a third person then built a "
            "game from theirs, and both of those passed the playability check, drafts "
            "included."
        ),
    ),
    SPEAKS_FROM_EXPERIENCE: Badge(
        kind=SPEAKS_FROM_EXPERIENCE,
        name="Speaks from experience",
        sentence="You have rated {tier} different {games} you had played.",
        rule=(
            f"A rating you gave a game after finishing {PLAYED_BEFORE_RATING} rated games "
            f"on it, on that many different games. The stars you gave are not read, "
            f"and rating a game again earns nothing."
        ),
        steps=RATING_STEPS,
        thresholds={"steps": list(RATING_STEPS), "played": PLAYED_BEFORE_RATING},
    ),
    CROWD_PLEASER: Badge(
        kind=CROWD_PLEASER,
        name="Crowd pleaser",
        sentence="People chose this game and rated it highly.",
        rule=(
            f"{CROWD_PEOPLE} different people have finished a rated game on it, and "
            f"{CROWD_PEOPLE} or more rated it, at a mean of {CROWD_STARS:.1f} stars or "
            f"better. On a game somebody here designed, their own games and their own "
            f"rating are left out."
        ),
        thresholds={"people": CROWD_PEOPLE, "stars": CROWD_STARS},
    ),
    CLANKER: Badge(
        kind=CLANKER,
        name="Proved the clanker wrong",
        sentence="People chose this game and rated it highly while its score was low.",
        rule=(
            f"What Crowd pleaser reads, and a score under {CLANKER_SCORE:.2f}, its whole "
            f"likely range included. The score is quoted as it stood on the day, "
            f"with its run count and scoring rules version, and is compared with "
            f"nothing since."
        ),
        thresholds={
            "people": CROWD_PEOPLE,
            "stars": CROWD_STARS,
            "score": CLANKER_SCORE,
        },
    ),
    BEAT_THE_MAKER: Badge(
        kind=BEAT_THE_MAKER,
        name="Beat the maker",
        sentence="A win over the person who designed this game, at their own game.",
        rule=(
            "A win in a rated game between two people, over the member who designed that "
            "game, on the game itself. Once per game for each winner. A practice game, a "
            "game they resigned or left, and a game on a game that came with the site "
            "earn nothing."
        ),
    ),
    EVERY_PROVEN: Badge(
        kind=EVERY_PROVEN,
        name="Every proven position",
        sentence=(
            "Every proven position this game offered, each one played out against hardest "
            "and won."
        ),
        rule=(
            "A win from every proven position a game offers, each one played out from the "
            "position against hardest, the strongest computer player. Finding the move "
            "does not count toward it, because the page shows the proved move to anybody "
            "who asks. Once per game, and positions the game offers later never take it "
            "away."
        ),
        thresholds={"ways": list(PROVEN_WAYS)},
    ),
    GAME_AFTER_GAME: Badge(
        kind=GAME_AFTER_GAME,
        name="Game after game",
        sentence="Every proven position, on {tier} different {games}.",
        rule=(
            "Every proven position won, on that many different games. A game counts from "
            "the day it was finished, whatever positions it offers since."
        ),
        steps=PROVEN_STEPS,
        thresholds={"steps": list(PROVEN_STEPS)},
    ),
}
"""Every badge, by kind. E10 adds its own here."""

KINDS: tuple[str, ...] = tuple(BADGES)


def sentence_for(award: awards.Award) -> str:
    """What a held award's line says, with its step filled in.

    Unknown kinds answer their own kind rather than raising. A page drawing a person's
    awards is not the place to find out that a row was written by a box this one has not
    seen.
    """
    badge = BADGES.get(award.kind)
    if badge is None:
        return award.kind
    if award.tier is None:
        return badge.sentence
    return badge.sentence.format(
        tier=award.tier, games="game" if award.tier == 1 else "games"
    )


def name_for(award: awards.Award) -> str:
    """What a held award is called. An unknown kind answers its own kind."""
    badge = BADGES.get(award.kind)
    return award.kind if badge is None else badge.name


def rule_for(award: awards.Award) -> str:
    """The one sentence naming what this badge reads, empty for an unknown kind.

    The table's own text and never a page's. A rule drawn beside a held badge and the
    same rule in the plain list of every badge come from this one string, so the two
    cannot come to say different things (§3, test 7).
    """
    badge = BADGES.get(award.kind)
    return "" if badge is None else badge.rule


# ---------------------------------------------------- what a held badge's line quotes


def line(award: awards.Award) -> dict[str, Any]:
    """One held badge, as the profile and the game's page draw it.

    Both pages read this, so a badge reads the same wherever it is drawn. The name, the
    sentence and the rule are the table's; the date and the figures are the row's. There
    is no count and no total here, because no page counts badges (§2.1's first rule).
    """
    return {
        "kind": award.kind,
        "tier": award.tier,
        "name": name_for(award),
        "sentence": sentence_for(award),
        "rule": rule_for(award),
        "quoted": quoted(award),
        "when": None if award.earned is None else award.earned.isoformat(),
        "game": award.game,
    }


def on_the_game(held: Iterable[awards.Award], owner: int | None) -> list[awards.Award]:
    """Of the awards filed under a game, the ones the game itself holds.

    `awards.held(game=...)` answers every row under the hash, and those are two
    different things. N2 and N3 sit on the game with the *player* as holder, the way M2
    and M3 do, and those belong on that person's profile: their sentences are written to
    whoever is reading -- *a win over a computer player rated far above you* -- and a
    page that drew one per player would be counting the people who had done it, which
    §3's sixth test refuses.

    What is left is the badges whose subject is the game. A library game's have no holder
    at all, and a designed game's sit on its owner, which is the shape `game_due_from`
    and `_family` write. `owner` is that owner, and nothing on a library game.
    """
    return [
        award
        for award in held
        if award.holder is None or (owner is not None and award.holder == owner)
    ]


def designer_of(connection: Any, game: str) -> int | None:
    """Who designed this game, or nothing at all for a library game.

    `chess-variant-sandbox site load` gives every library game the maintainer's account, so `origin`
    is what tells a designed game from a library one and a null owner never is.
    """
    with connection.cursor() as cursor:
        cursor.execute("select owner, origin from games where hash = %s", (game,))
        found = cursor.fetchone()
    if found is None or found[0] is None or str(found[1]) != drafts.DESIGNED:
        return None
    return int(found[0])


def quoted(award: awards.Award) -> str:
    """The figures the row kept, as the clause a dated line quotes them in.

    §2.9's dated quotation, minus the date, which the line carries anyway. Every figure
    is read off `evidence` as the grant wrote it, so a vote cast since, a re-anchor or a
    schema that grew moves none of it. A score is quoted with its whole range, the runs it
    was the mean of and the weights version it was taken under, which is what keeps it
    out of every ordering: it says what stood on the day and is compared with nothing.

    Every clause is a noun phrase, so a page can put *on that day* in front of the lot
    and read as one sentence. A badge whose evidence quotes no figure answers an empty
    string, and a page draws the sentence alone.
    """
    evidence = award.evidence or {}
    score = evidence.get("score")
    said = [
        _scored_as(score) if isinstance(score, dict) else "",
        _stars_as(evidence),
        _players_as(evidence),
        _blocks_as(evidence),
        _positions_as(evidence),
    ]
    kept = [clause for clause in said if clause]
    if len(kept) < 2:
        return kept[0] if kept else ""
    return ", ".join(kept[:-1]) + " and " + kept[-1]


def _figure(value: Any, places: int) -> str | None:
    """One number as the sentence spells it, or nothing for a field that holds none."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return f"{float(value):.{places}f}"


def _whole(value: Any) -> int | None:
    """One counted thing, or nothing for a field that holds no count."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _scored_as(score: dict[str, Any]) -> str:
    """What the index said, as the row quoted it, and how many runs it rested on.

    A library game's score is the mean of four runs and a visitor's is one, and the two
    are not the same claim, so the count is part of the quotation rather than a detail
    beside it. It is read off the basis the row kept (`ScoreEvidence.basis`), and a row
    whose basis names no seeds quotes no score at all: `CLAUDE.md`'s reader that cannot
    match a score's basis shows no number, and a figure with no count is a figure that
    does not say which claim it is.
    """
    headline = _figure(score.get("headline"), 2)
    runs = _runs_as(score.get("basis"))
    if headline is None or not runs:
        return ""
    low, high = _figure(score.get("ci_low"), 2), _figure(score.get("ci_high"), 2)
    weights = str(score.get("weights") or "")
    # S10 of `docs/archive/plan-understanding.md` §3.3, in the site's words
    # (`docs/reference/plain-words.md` §2): *score*, *likely*, *the average of four runs*,
    # *scoring rules version 11*. All of it inside
    # the one bracket, because `quoted` joins its clauses with commas and a score's own
    # commas would run into the stars' clause after it.
    version = weights[1:] if weights[:1] == "v" and weights[1:].isdigit() else weights
    said = [f"likely {low} to {high}"] if low is not None and high is not None else []
    said.append(runs + (f", scoring rules version {version}" if weights else ""))
    return f"a score of {headline} ({'; '.join(said)})"


_RUN_WORDS: tuple[str, ...] = (
    "no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty",
)
"""The page's `RUN_WORDS` (`web/static/page/core.js`), so a badge and a card spell a count alike."""


def _runs_as(basis: Any) -> str:
    """`one run` or `the average of four runs`, off a kept basis, in the site's words (S10).

    Two spellings of the seeds, one meaning, as `measurement.Basis.from_dict` reads them:
    `seeds` is a list, and `seed` is what a basis written before a game could have more
    than one run recorded, which is one run. Anything else answers an empty string, and
    `_scored_as` then quotes no score.
    """
    if not isinstance(basis, dict):
        return ""
    seeds = basis.get("seeds")
    if seeds is None and _whole(basis.get("seed")) is not None:
        seeds = [basis["seed"]]
    if not isinstance(seeds, list) or not seeds:
        return ""
    if any(_whole(seed) is None for seed in seeds):
        return ""
    count = len(seeds)
    if count == 1:
        return "one run"
    word = _RUN_WORDS[count] if count < len(_RUN_WORDS) else str(count)
    return f"the average of {word} runs"


def _stars_as(evidence: dict[str, Any]) -> str:
    """The mean and how many gave it, for the two badges about stars."""
    mean = _figure(evidence.get("mean"), 1)
    voters = _whole(evidence.get("voters"))
    if mean is None or voters is None:
        return ""
    return f"{mean} stars from {voters} {'person' if voters == 1 else 'people'}"


def _players_as(evidence: dict[str, Any]) -> str:
    """How many people had played the game, which is the other half of N1's count."""
    people = _whole(evidence.get("people"))
    if people is None:
        return ""
    return f"{people} {'person' if people == 1 else 'people'} who had played it"


def _blocks_as(evidence: dict[str, Any]) -> str:
    """The *n of N* a rules badge stores, which a later schema never moves."""
    used, total = _whole(evidence.get("used")), _whole(evidence.get("available"))
    if used is None or total is None:
        return ""
    return f"{used} of {total} building blocks"


def _positions_as(evidence: dict[str, Any]) -> str:
    """How many proven positions the game offered when *Every proven position* was won.

    The row's figure, never the record's today: a re-proof that adds positions moves the
    game's own line (*converted n of m*), and this clause still says what was finished.
    """
    count = _whole(evidence.get("positions"))
    if count is None:
        return ""
    return f"{count} proven position{'' if count == 1 else 's'} on the game"


# --------------------------------------------------------------------------- the rows


@dataclass(frozen=True)
class _Played:
    """One counted match from this person's side of the board.

    Every field is the row's own. `mine` and `theirs` are the two ratings going in and
    `deviation` this person's, all three written at the finish, which is why the hook
    runs after `rating.finish`. `rematch` is the match this one was started from, which
    only the board's own control writes.
    """

    when: datetime
    match: str
    game: str
    seat: int
    opponent: rating.PlayerKey
    score: float
    reason: str
    top: bool
    mine: float | None
    theirs: float | None
    deviation: float | None
    rematch: str | None = None


@dataclass(frozen=True)
class Design:
    """One `games` row, as the designer badges read it.

    `blocks` is what `rules_used.used` found in the spec, carried on the row so that the
    reader stays pure and no spec is walked twice. U5 reads it on a person's own designs
    and U4 on the games anybody has played, which are mostly the library's.
    """

    game: str
    owner: int | None
    parent: str | None
    origin: str
    screen_passed: bool
    created: datetime
    revoked: bool = False
    blocks: frozenset[str] = frozenset()

    @property
    def designed(self) -> bool:
        """A game a person built here, still standing."""
        return self.origin == drafts.DESIGNED and not self.revoked

    @property
    def passes(self) -> bool:
        """§2.9's *passes the screen*, which `drafts.keep` records from the check."""
        return self.designed and self.screen_passed


@dataclass(frozen=True)
class Rated:
    """One rating as N13 reads it -- who rated which game, and when.

    No star, no tags, no note. §2.9 forbids the badge those three, and a type that cannot
    carry a star says so more firmly than a rule a later reader has to keep. `rated` is
    the `votes.updated` column, which moves when a person votes again, and that is how a
    rating given before the third game comes to count once it is given again.
    """

    voter: int
    game: str
    rated: datetime


@dataclass(frozen=True)
class Given:
    """One rating as N1 and U1 read it -- the value, who gave it, and when.

    The mean is what those two test, so this one does carry the star. It is a type of its
    own rather than `Rated` with a field added, because the point of `Rated` is that N13
    has no star to read.
    """

    voter: int
    stars: int
    rated: datetime


def _counted(rows: Iterable[match_rows.Row]) -> tuple[match_rows.Row, ...]:
    """The rows a badge may read, in finish order with `id` breaking a tie.

    `milestones._counted`'s filter and one more clause. A resignation and the sweep's
    forfeit are dropped: §2.9 leaves them out of N7 by name, and a badge nobody could
    earn by playing should not be earnable by somebody else's leaving either.
    """
    kept = [
        (row.finished, row.id, row)
        for row in rows
        if row.mode == match_rows.LEADERBOARD
        and row.source is None
        and row.finished is not None
        and row.result is not None
        and row.seat0 is not None
        and row.seat1 is not None
        and not row.left
        and row.reason != RESIGNED
    ]
    kept.sort(key=lambda found: (found[0], found[1]))
    return tuple(row for _, _, row in kept)


def _mine(user: int, rows: Iterable[match_rows.Row]) -> list[_Played]:
    """This person's side of every counted row, oldest first."""
    me = rating.person(user)
    found: list[_Played] = []
    for row in rows:
        seat = row.seat_of(me)
        if seat < 0 or row.finished is None or row.result is None:
            continue
        score = row.score_for(seat)
        if score is None:
            continue
        found.append(
            _Played(
                when=row.finished,
                match=row.id,
                game=row.game,
                seat=seat,
                opponent=row.player(1 - seat),
                score=score,
                reason=row.reason,
                top=row.top,
                mine=row.rating_before(seat),
                theirs=row.rating_before(1 - seat),
                deviation=row.deviation_before(seat),
                rematch=row.rematch_of,
            )
        )
    return sorted(found, key=_order)


def _order(played: _Played) -> tuple[datetime, str]:
    return (played.when, played.match)


def _wins(played: Iterable[_Played]) -> list[_Played]:
    return [each for each in played if each.score == 1.0]


def _earliest(played: Iterable[_Played]) -> dict[str, _Played]:
    """The first of these matches on each game."""
    found: dict[str, _Played] = {}
    for each in played:
        found.setdefault(each.game, each)
    return found


# ------------------------------------------------------------------------- the reader


def due_from(
    user: int,
    rows: Iterable[match_rows.Row],
    tables: milestones.Tables,
    cells: milestones.Cells,
    held: Iterable[awards.Award] = (),
    games: Iterable[Design] = (),
    ratings: Iterable[Rated] = (),
) -> tuple[awards.Award, ...]:
    """Every badge this person has earned and does not hold, oldest first.

    `rows` is this person's leaderboard rows -- unlike `milestones.read_from`, no badge
    here reads anybody else's -- `tables` every game's rating table, `cells` each rated
    game's `grid.descriptor`, `held` what the store already has for them, `games` every
    `games` row with the building blocks its spec draws on, and `ratings` this person's
    own votes without their stars. The designer badges need the whole of `games` rather
    than this person's own rows, because *Descendant* is earned by what somebody else
    saved, and U4 reads the library's rows for the games this person has played on.

    N1 and U1 are not here. They sit on a game rather than on a person, so they are
    `game_due_from`'s.

    Each award is dated by the row that completed it, so a badge a match from last month
    completed arrives with this month's finish carrying last month's date. Handing back
    what is already held would be harmless, since `awards.grant` writes once whatever it
    is asked, but a caller that wants to say *you have just earned this* reads the
    grant's answer, and a reader that kept re-offering old badges would make that useless.
    """
    played = _mine(user, _counted(rows))
    designed = _by_date(games)
    found: list[awards.Award] = []
    found += _beat_the_ladder(user, played)
    found += _giant_killer(user, played)
    found += _held_the_top(user, played)
    found += _grand_tour(user, played, tables, cells)
    found += _every_which_way(user, played)
    found += _long_way_round(user, played)
    found += _and_another_thing(user, played)
    found += _well_read(user, played, tables, designed)
    found += _whole_toolbox(user, designed)
    found += _both_hats(user, played, designed)
    found += _family(user, designed)
    found += _speaks_from_experience(user, played, ratings)
    found += _beat_the_maker(user, played, designed)
    already = {award.key for award in held}
    return tuple(
        award
        for award in sorted(found, key=_when_earned)
        if award.key not in already
    )


def _when_earned(award: awards.Award) -> tuple[float, str, int]:
    when = award.earned.timestamp() if award.earned is not None else 0.0
    return (when, award.kind, award.tier or 0)


def _award(
    kind: str,
    user: int,
    completed: _Played,
    evidence: dict[str, Any],
    game: str | None = None,
    tier: int | None = None,
    rule: dict[str, Any] | None = None,
) -> awards.Award:
    """One award, dated by the match that completed it and stamped with its thresholds."""
    return _earned(kind, user, completed.when, evidence, game=game, tier=tier, rule=rule)


def _earned(
    kind: str,
    user: int,
    when: datetime | None,
    evidence: dict[str, Any],
    game: str | None = None,
    tier: int | None = None,
    rule: dict[str, Any] | None = None,
) -> awards.Award:
    """One award, dated by whatever completed it and stamped with its thresholds."""
    thresholds = dict(BADGES[kind].thresholds)
    thresholds.update(rule or {})
    return awards.Award(
        kind=kind,
        tier=tier,
        holder=user,
        game=game,
        earned=when,
        evidence=evidence,
        rule=thresholds,
    )


def _beat_the_ladder(user: int, played: Sequence[_Played]) -> list[awards.Award]:
    """U2. Four series over different games, a stronger win filling the weaker series.

    The filling is why this is one pass over the rungs rather than four readings: a win
    over `hardest` on a game is a win over `hard`, `medium` and `easy` on that game, and
    the weaker series takes its date from that same win.
    """
    rungs = _rungs()
    rank = {agent: index for index, (_name, _gloss, agent) in enumerate(rungs)}
    filled: list[dict[str, _Played]] = [{} for _ in rungs]
    for win in _wins(played):
        beaten = rank.get(win.opponent.agent or "")
        if beaten is None:
            continue
        for index in range(beaten + 1):
            filled[index].setdefault(win.game, win)

    found: list[awards.Award] = []
    for index, (_name, _gloss, agent) in enumerate(rungs):
        games = sorted(filled[index].values(), key=_order)
        for step in GAME_STEPS:
            if len(games) < step:
                break
            reached = games[:step]
            found.append(
                _award(
                    BEAT[index],
                    user,
                    reached[-1],
                    tier=step,
                    evidence={
                        "player": agent,
                        "games": [each.game for each in reached],
                        "matches": [each.match for each in reached],
                    },
                )
            )
    return found


def _beat_the_maker(
    user: int, played: Sequence[_Played], games: Sequence[Design]
) -> list[awards.Award]:
    """E11 *Beat the maker* (N11). A counted win over a designed game's owner, on it.

    Once per game for each winner: the award sits on the game with the winner as holder,
    dated by the first such win there. The opponent is a person and the game's owner,
    read off `games` by `origin` (`designer_of`'s rule), so a library game, which the
    maintainer's account owns, earns nothing, and neither does a killed game. A practice
    game never reaches `_counted`, nor does a resignation or a game somebody left.
    """
    owners = {design.game: design.owner for design in games if design.designed}
    beaten = [
        win
        for win in _wins(played)
        if win.opponent.user is not None
        and win.opponent.user != user
        and owners.get(win.game) == win.opponent.user
    ]
    return [
        _award(
            BEAT_THE_MAKER,
            user,
            win,
            game=win.game,
            evidence={"match": win.match, "seat": win.seat},
        )
        for win in _earliest(beaten).values()
    ]


def _giant_killer(user: int, played: Sequence[_Played]) -> list[awards.Award]:
    """N2. A win over somebody rated far above, with this person's rating established.

    Both ratings and the deviation are the row's own, written at the finish, so a table
    that moves afterwards neither grants this nor takes it away.
    """
    qualifying = [
        win
        for win in _wins(played)
        if win.mine is not None
        and win.theirs is not None
        and win.deviation is not None
        and win.deviation < rating.ESTABLISHED_DEVIATION
        and win.theirs - win.mine >= GIANT_GAP
    ]
    return [
        _award(
            GIANT_KILLER,
            user,
            win,
            game=win.game,
            evidence={
                "match": win.match,
                "opponent": win.opponent.agent,
                "mine": win.mine,
                "theirs": win.theirs,
                "gap": round((win.theirs or 0.0) - (win.mine or 0.0), 1),
                "deviation": win.deviation,
            },
            rule={"gap": GIANT_GAP, "established": rating.ESTABLISHED_DEVIATION},
        )
        for win in _earliest(qualifying).values()
    ]


def _held_the_top(user: int, played: Sequence[_Played]) -> list[awards.Award]:
    """N3. A draw against the top of a game's table, as the table stood at the start."""
    drawn = [each for each in played if each.top and each.score == 0.5]
    return [
        _award(
            HELD_THE_TOP,
            user,
            each,
            game=each.game,
            evidence={"match": each.match, "opponent": each.opponent.agent},
        )
        for each in _earliest(drawn).values()
    ]


def _grand_tour(
    user: int,
    played: Sequence[_Played],
    tables: milestones.Tables,
    cells: milestones.Cells,
) -> list[awards.Award]:
    """N4. A win over the top on one game of every kind the rated games cover.

    The kinds are `milestones._kinds`, which is M6's *one of every kind* with wins in
    place of tries. Called rather than restated, so the badge and the milestone cannot
    come to disagree about what a kind is.
    """
    kinds = milestones._kinds(milestones._rated(tables), cells)
    if not kinds:
        return []
    beaten = _earliest(each for each in _wins(played) if each.top)
    covering: list[tuple[tuple[int, int], _Played]] = []
    for bin_key, games in kinds.items():
        reached = [beaten[game] for game in games if game in beaten]
        if not reached:
            return []
        covering.append((bin_key, min(reached, key=_order)))
    completed = max((each for _key, each in covering), key=_order)
    return [
        _award(
            GRAND_TOUR,
            user,
            completed,
            evidence={
                "games": sorted({each.game for _key, each in covering}),
                "matches": sorted({each.match for _key, each in covering}),
            },
            rule={"bins": [_bin_name(key) for key, _each in covering]},
        )
    ]


def _bin_name(key: tuple[int, int]) -> str:
    """One bin of one grid axis, as a name a sentence can carry."""
    axis = grid.AXES[key[0]]
    return f"{axis.name}: {axis.bins[key[1]]}"


def _every_which_way(user: int, played: Sequence[_Played]) -> list[awards.Award]:
    """N7. Wins by `ENDINGS` different endings, dated by the one that completed the set."""
    seen: dict[str, _Played] = {}
    for win in _wins(played):
        if not win.reason:
            continue
        seen.setdefault(win.reason, win)
        if len(seen) == ENDINGS:
            return [
                _award(
                    EVERY_WHICH_WAY,
                    user,
                    win,
                    evidence={
                        "reasons": sorted(seen),
                        "matches": [each.match for each in seen.values()],
                    },
                    rule={"endings": ENDINGS},
                )
            ]
    return []


def _long_way_round(user: int, played: Sequence[_Played]) -> list[awards.Award]:
    """N8. A finished game that ran to the move limit, once per person."""
    for each in played:
        if each.reason in MOVE_LIMIT:
            return [
                _award(
                    LONG_WAY_ROUND,
                    user,
                    each,
                    evidence={
                        "match": each.match,
                        "game": each.game,
                        "reason": each.reason,
                    },
                    rule={"reasons": list(MOVE_LIMIT)},
                )
            ]
    return []


def _and_another_thing(user: int, played: Sequence[_Played]) -> list[awards.Award]:
    """N9. A loss, the game started again from that board, and a win in it.

    `matches.rematch_of` is the whole of *started again from that board*, and only the
    board's own control writes it, so nothing here has to guess from two matches in a
    row. `matches._followed` already refuses a link to another game or another person's
    match; the same-game test below is this reader's own, because the reader is handed
    rows rather than a store and says what it reads.

    A resignation is not the loss, since `_counted` drops one at the door for every badge
    here. Once per person, dated by the win.
    """
    lost = {each.match: each for each in played if each.score == 0.0}
    for win in _wins(played):
        beaten = lost.get(win.rematch or "")
        if beaten is None or beaten.game != win.game:
            continue
        return [
            _award(
                AND_ANOTHER_THING,
                user,
                win,
                evidence={
                    "game": win.game,
                    "lost": beaten.match,
                    "won": win.match,
                },
            )
        ]
    return []


# ----------------------------------------------------------------- U4, the reading badge


def _in_play(tables: milestones.Tables, games: Sequence[Design]) -> frozenset[str]:
    """The building blocks the rated games use between them, as they stand now.

    U4's denominator. `milestones._rated` decides what a rated game is, so the badge and
    the milestones cannot come to disagree about it, and a block no rated game uses is
    outside the count -- nobody could read about it by playing.
    """
    blocks = {design.game: design.blocks for design in games}
    found: set[str] = set()
    for game in milestones._rated(tables):
        found |= blocks.get(game, frozenset())
    return frozenset(found)


def _well_read(
    user: int,
    played: Sequence[_Played],
    tables: milestones.Tables,
    games: Sequence[Design],
) -> list[awards.Award]:
    """U4. The blocks a person's finished games draw on between them.

    One game counts once however many times it was played, and one block once however
    many games use it, so a step is dated by the first finish on the game that reached
    it. The row keeps its own *n of N* and the names, because a game added to the library
    moves N.
    """
    available = _in_play(tables, games)
    if not available:
        return []
    blocks = {design.game: design.blocks for design in games}
    steps = [(share, _needed(share, len(available))) for share in READING_STEPS]
    seen: set[str] = set()
    found: list[awards.Award] = []
    reached: set[int] = set()
    counted: list[_Played] = []
    for first in sorted(_earliest(played).values(), key=_order):
        seen |= blocks.get(first.game, frozenset()) & available
        counted.append(first)
        for share, needed in steps:
            if share in reached or len(seen) < needed:
                continue
            reached.add(share)
            found.append(
                _award(
                    WELL_READ,
                    user,
                    first,
                    tier=share,
                    evidence={
                        "used": len(seen),
                        "available": len(available),
                        "schema": schema.SCHEMA_VERSION,
                        "names": sorted(seen),
                        "games": [each.game for each in counted],
                        "matches": [each.match for each in counted],
                    },
                    rule={"share": share, "needed": needed, "available": len(available)},
                )
            )
    return found


# --------------------------------------------------------------- the designer badges


def _by_date(games: Iterable[Design]) -> tuple[Design, ...]:
    """Every game row, oldest first, with the hash breaking a tie."""
    return tuple(sorted(games, key=lambda design: (design.created, design.game)))


def _mine_designed(user: int, games: Sequence[Design]) -> list[Design]:
    """This person's passing designs, oldest first."""
    return [design for design in games if design.owner == user and design.passes]


def _whole_toolbox(user: int, games: Sequence[Design]) -> list[awards.Award]:
    """U5. The building blocks a person's passing designs draw on between them.

    A block counts once however many games use it (§3 test 6, so thirty near-copies earn
    no more than one game), and a step is dated by the save that first reached it. The
    row keeps *n of N* and the names, because N grows with the schema.
    """
    total = len(rules_used.available())
    steps = [(share, _needed(share, total)) for share in TOOLBOX_STEPS]
    seen: set[str] = set()
    found: list[awards.Award] = []
    reached: set[int] = set()
    counted: list[str] = []
    for design in _mine_designed(user, games):
        seen |= design.blocks
        counted.append(design.game)
        for share, needed in steps:
            if share in reached or len(seen) < needed:
                continue
            reached.add(share)
            found.append(
                _earned(
                    WHOLE_TOOLBOX,
                    user,
                    design.created,
                    tier=share,
                    evidence={
                        "used": len(seen),
                        "available": total,
                        "schema": schema.SCHEMA_VERSION,
                        "names": sorted(seen),
                        "games": list(counted),
                    },
                    rule={"share": share, "needed": needed, "available": total},
                )
            )
    return found


def _both_hats(
    user: int, played: Sequence[_Played], games: Sequence[Design]
) -> list[awards.Award]:
    """N6. A win over some game's top, and a passing design, in either order."""
    beaten = [win for win in _wins(played) if win.top]
    designed = _mine_designed(user, games)
    if not beaten or not designed:
        return []
    win, design = beaten[0], designed[0]
    return [
        _earned(
            BOTH_HATS,
            user,
            max(win.when, design.created),
            evidence={
                "match": win.match,
                "game": win.game,
                "design": design.game,
            },
        )
    ]


def _family(user: int, games: Sequence[Design]) -> list[awards.Award]:
    """N12's three levels, all of them off the same rows.

    §2.9's narrow reading throughout. *Somebody else's game* is another person's designed
    game, so a game built from a library game earns nothing; the new game must pass the
    screen, while the game built from need only be a design still standing, since an
    unlisted game opens by its link and building on one is still building on it; every
    link crosses between two people; and the top level takes three of them.
    """
    by_hash = {design.game: design for design in games}
    children: dict[str, list[Design]] = {}
    for design in games:
        if design.parent:
            children.setdefault(design.parent, []).append(design)

    def built_on(design: Design) -> Design | None:
        """The design of another person's this one was built from, if it was."""
        parent = by_hash.get(design.parent or "")
        if parent is None or not parent.designed:
            return None
        if parent.owner is None or parent.owner == design.owner:
            return None
        return parent

    def built_by_others(design: Design, besides: tuple[int | None, ...]) -> list[Design]:
        """The passing games other people built from this one, oldest first."""
        return [
            child
            for child in children.get(design.game, ())
            if child.passes and child.owner is not None and child.owner not in besides
        ]

    found: list[awards.Award] = []
    for design in games:
        if design.owner != user or not design.passes:
            continue
        parent = built_on(design)
        if parent is not None:
            found.append(
                _earned(
                    ANCESTOR,
                    user,
                    design.created,
                    evidence={"game": design.game, "parent": parent.game},
                )
            )
            break

    for design in games:
        if design.owner != user or not design.designed:
            continue
        built = built_by_others(design, (user,))
        if not built:
            continue
        found.append(
            _earned(
                DESCENDANT,
                user,
                built[0].created,
                game=design.game,
                evidence={"game": design.game, "child": built[0].game},
            )
        )
        third = [
            (grandchild, child)
            for child in built
            for grandchild in built_by_others(child, (user, child.owner))
        ]
        if third:
            grandchild, child = min(third, key=lambda pair: (pair[0].created, pair[0].game))
            found.append(
                _earned(
                    FAMILY_TREE,
                    user,
                    grandchild.created,
                    game=design.game,
                    evidence={
                        "game": design.game,
                        "child": child.game,
                        "grandchild": grandchild.game,
                    },
                )
            )
    return found


# ------------------------------------------------------------- N13, the voting badge


def _speaks_from_experience(
    user: int, played: Sequence[_Played], ratings: Iterable[Rated]
) -> list[awards.Award]:
    """N13. A series over the games this person rated after playing them.

    A rating counts when it was given after `PLAYED_BEFORE_RATING` finished games of
    theirs on that game. `votes` keeps one row per person per game and replaces it on a
    re-vote, so the series counts games rather than clicks, and a rating given too early
    starts counting from the moment it is given again. The value is not here to read.

    The step is dated by the rating that reached it, and the three matches under each
    rating travel with the row so that the badge can say what it was earned on.
    """
    mine: dict[str, list[_Played]] = {}
    for each in played:
        mine.setdefault(each.game, []).append(each)

    counted: dict[str, tuple[Rated, list[_Played]]] = {}
    for rated in sorted(ratings, key=lambda each: (each.rated, each.game)):
        if rated.voter != user:
            continue
        before = [each for each in mine.get(rated.game, ()) if each.when < rated.rated]
        if len(before) < PLAYED_BEFORE_RATING:
            continue
        counted.setdefault(rated.game, (rated, before[:PLAYED_BEFORE_RATING]))
    qualifying = sorted(counted.values(), key=lambda pair: (pair[0].rated, pair[0].game))

    found: list[awards.Award] = []
    for step in RATING_STEPS:
        if len(qualifying) < step:
            break
        reached = qualifying[:step]
        found.append(
            _earned(
                SPEAKS_FROM_EXPERIENCE,
                user,
                reached[-1][0].rated,
                tier=step,
                evidence={
                    "games": [rated.game for rated, _played in reached],
                    "ratings": [
                        {
                            "game": rated.game,
                            "rated": rated.rated.isoformat(),
                            "matches": [each.match for each in behind],
                        }
                        for rated, behind in reached
                    ],
                },
            )
        )
    return found


# --------------------------------------------------- N1 and U1, the badges on the game


def _people_on(
    rows: Iterable[match_rows.Row], besides: int | None
) -> list[tuple[int, datetime]]:
    """Each person with a counted game on this game, by their first such finish.

    Distinct people and never matches (§3 test 6), so playing a game two hundred times
    counts once. `besides` is the owner of a designed game, whose own games §2.9 leaves
    out so that a designer cannot play their game into being.
    """
    first: dict[int, datetime] = {}
    for row in _counted(rows):
        for key in (row.seat0, row.seat1):
            if key is None or key.is_computer or key.user is None:
                continue
            if besides is not None and int(key.user) == besides:
                continue
            if row.finished is not None:
                first.setdefault(int(key.user), row.finished)
    return sorted(first.items(), key=lambda pair: (pair[1], pair[0]))


def _loved(given: Iterable[Given], besides: int | None) -> tuple[datetime, int, float] | None:
    """When the ratings first stood at `CROWD_PEOPLE` with a mean of `CROWD_STARS`.

    The walk is over the ratings in date order, and the earliest prefix that clears both
    marks is the one that completed the badge. Reading only the mean as it stands today
    would let a later low rating unearn a badge nothing had yet written, which §2.9
    refuses: a vote that drops the mean removes nothing, whether the row exists or not.

    `votes` keeps no history, so this is a reconstruction from the latest answer each
    person gave. It is the whole of what the store holds, and the figures it returns are
    what the row quotes.
    """
    counted = sorted(
        (each for each in given if besides is None or each.voter != besides),
        key=lambda each: (each.rated, each.voter),
    )
    total = 0
    for index, each in enumerate(counted, start=1):
        total += int(each.stars)
        if index < CROWD_PEOPLE:
            continue
        mean = total / index
        if mean >= CROWD_STARS:
            return (each.rated, index, mean)
    return None


def game_due_from(
    game: str,
    design: Design | None,
    rows: Iterable[match_rows.Row],
    given: Iterable[Given],
    score: awards.ScoreEvidence | None = None,
    held: Iterable[awards.Award] = (),
) -> tuple[awards.Award, ...]:
    """N1 and U1 on this game, as far as they are earned and not held.

    `design` is the game's own row, which says whether a person designed it and who; a
    library game's row has the maintainer as owner and is told apart by `origin`, never by
    the account. `rows` is every leaderboard match on the game, `given` every rating of
    it, and `score` what `awards.score_evidence` answered -- which is `None` for a stale,
    missing or never-measured game, and is the only thing U1 may quote.

    Both badges sit on the game. The holder is the owner of a designed game, so that the
    line reaches their profile too, and nobody on a library game.
    """
    if design is not None and design.revoked:
        return ()
    designed = design is not None and design.designed
    owner = design.owner if (designed and design is not None) else None
    holder = owner if designed else None

    people = _people_on(rows, owner)
    if len(people) < CROWD_PEOPLE:
        return ()
    loved = _loved(given, owner)
    if loved is None:
        return ()
    rated_when, voters, mean = loved
    when = max(people[CROWD_PEOPLE - 1][1], rated_when)
    evidence: dict[str, Any] = {
        "game": game,
        "people": len(people),
        "voters": voters,
        "mean": round(mean, 3),
    }

    found = [
        awards.Award(
            kind=CROWD_PLEASER,
            holder=holder,
            game=game,
            earned=when,
            evidence=evidence,
            rule=dict(BADGES[CROWD_PLEASER].thresholds),
        )
    ]
    if score is not None and _scored_low(score):
        found.append(
            awards.Award(
                kind=CLANKER,
                holder=holder,
                game=game,
                earned=when,
                evidence={**evidence, "score": score.as_dict()},
                rule=dict(BADGES[CLANKER].thresholds),
            )
        )
    already = {award.key for award in held}
    return tuple(award for award in found if award.key not in already)


def _scored_low(score: awards.ScoreEvidence | None) -> bool:
    """Whether U1's index test passes on what `awards.score_evidence` answered.

    No score is not a low score. A stale, missing or never-measured game answers `None`
    there and so grants nothing, which is `CLAUDE.md`'s rule that a reader which cannot
    match a score's basis shows no number, in the one form a kept badge can take it.
    """
    if score is None or not score.ranged:
        return False
    ends = (score.headline, score.ci_low, score.ci_high)
    return all(end is not None and end < CLANKER_SCORE for end in ends)


# ------------------------------------------- the two over proven positions (Q993e)


@dataclass(frozen=True)
class Conversion:
    """One `converted` mark (`proven_marks`), as the proven-position badges read it.

    `when` is the mark's own `marked`, and `match` the try that wrote it, null where its
    row went. A `found` mark is never one of these: `conversions` selects only
    `PROVEN_WAYS`.
    """

    game: str
    position: str
    when: datetime
    match: str | None = None


def proven_due_from(
    user: int,
    conversions: Iterable[Conversion],
    records: Mapping[str, Record | None],
    held: Iterable[awards.Award] = (),
) -> tuple[awards.Award, ...]:
    """*Every proven position* and *Game after game* for this person, earned and not held.

    `conversions` is this person's `converted` marks, `records` each marked game's record
    as the site loads it now (`proven.load`: the positions it offers, the too-easy ones
    left out), and `held` what the store has for them.

    *Every proven position* on a game is due when every position the record offers has a
    conversion. It is dated by the last of those marks, so a game finished before this
    badge existed is dated by the try that finished it, and the row keeps how many
    positions the game offered, the record's proof date and the tries.

    *Game after game* counts the games the person **holds** *Every proven position* on,
    the kept rows and the ones this call adds, and never the live marks. A re-proof that
    adds positions to a finished game cannot un-finish it, for either badge: nothing here
    removes a row, and the count reads rows.
    """
    by_game: dict[str, dict[str, Conversion]] = {}
    for each in sorted(conversions, key=lambda c: (c.when, c.game, c.position)):
        by_game.setdefault(each.game, {}).setdefault(each.position, each)

    found: list[awards.Award] = []
    for game in sorted(by_game):
        record = records.get(game)
        if record is None or not record.positions:
            continue
        marks = by_game[game]
        offered = [position.id for position in record.positions]
        if any(position not in marks for position in offered):
            continue
        used = [marks[position] for position in offered]
        last = max(used, key=lambda c: (c.when, c.position))
        found.append(
            _earned(
                EVERY_PROVEN,
                user,
                last.when,
                game=game,
                evidence={
                    "game": game,
                    "positions": len(offered),
                    "proved": record.made,
                    "matches": [c.match for c in used if c.match],
                },
            )
        )

    finished: dict[str, awards.Award] = {}
    for award in [*held, *found]:
        if award.kind != EVERY_PROVEN or award.holder != user or not award.game:
            continue
        earlier = finished.get(award.game)
        if earlier is None or _when_earned(award) < _when_earned(earlier):
            finished[award.game] = award
    order = sorted(finished.values(), key=lambda a: (_when_earned(a), a.game or ""))
    for step in PROVEN_STEPS:
        if len(order) < step:
            break
        reached = order[:step]
        found.append(
            _earned(
                GAME_AFTER_GAME,
                user,
                reached[-1].earned,
                tier=step,
                evidence={"games": [award.game for award in reached]},
            )
        )
    already = {award.key for award in held}
    return tuple(
        award for award in sorted(found, key=_when_earned) if award.key not in already
    )


def conversions(connection: Any, person: int) -> tuple[Conversion, ...]:
    """This person's `converted` marks, oldest first. A `found` mark is not selected."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, position, marked, match from proven_marks"
            " where person = %s and way = any (%s) order by marked, game, position",
            (int(person), list(PROVEN_WAYS)),
        )
        return tuple(
            Conversion(
                game=str(row[0]),
                position=str(row[1]),
                when=row[2],
                match=None if row[3] is None else str(row[3]),
            )
            for row in cursor.fetchall()
        )


def grant_for_proof(
    connection: Any, person: int, records: Callable[[str], Record | None]
) -> list[awards.Award]:
    """Write the proven-position badges this person is due, and return what landed.

    What the site calls when a try from a proven position ends in the person's win, in
    the finish's own transaction, after `proven.converted` has written the mark
    (`server._session_action`). `records` loads a game's record as the site offers it;
    the store has no record, so the caller that knows where they are on disk hands it
    over. Every game the person has converted on is read, so a game finished before
    this badge existed is granted at their next win, dated by its own last mark.
    """
    mine = conversions(connection, person)
    games = sorted({each.game for each in mine})
    return awards.grant(
        connection,
        proven_due_from(
            person,
            mine,
            {game: records(game) for game in games},
            awards.held(connection, holder=person),
        ),
    )


# ------------------------------------------------------------------------- the writer


def grant_for_match(connection: Any, match_id: str) -> list[awards.Award]:
    """Write the badges this finish completed for the people in it, and return them.

    What `matches._finish` calls, inside the finish's own transaction and after
    `rating.finish` has written the rating columns, which N2 reads. Nothing commits.

    The whole of a person's record is re-read rather than only this match, because a
    badge completed before its box landed is due at their next finish, dated by the
    older match. `awards.grant` writes once, so the pass costs nothing the second time.

    The game's own badges go with it, because this finish can be the fifth person's and
    so the second half of N1 or U1.
    """
    found = match_rows.row(connection, match_id)
    if found.mode != match_rows.LEADERBOARD or found.source is not None:
        return []
    seated = [
        seat.user
        for seat in (found.seat0, found.seat1)
        if seat is not None and not seat.is_computer and seat.user is not None
    ]
    if not seated:
        return []
    return _grant_for(connection, seated) + grant_for_games(connection, [found.game])


def grant_for_vote(connection: Any, user: int, game: str) -> list[awards.Award]:
    """Write the badges this rating completed, and return them.

    What `/api/vote` calls inside the vote's own transaction, so the rating and whatever
    it completed land together or neither does. Two things can be completed: N13 for the
    person who rated, and N1 or U1 on the game they rated, since this can be its fifth
    rating.
    """
    return _grant_for(connection, [user]) + grant_for_games(connection, [game])


def grant_for_games(connection: Any, games: Sequence[str]) -> list[awards.Award]:
    """Write N1 and U1 where they are due on these games, and return what landed.

    The third caller is `chess-variant-sandbox site load`, after `load_measurements` has written what
    each measurement now reads as: a game that had its five people and its five ratings
    while its score was stale earns U1 on the load that finds the score current.

    The ratings for the whole set are read in one query and a game nobody has rated is
    dropped before any other read, so a load over the library costs two queries plus a
    few for the handful of games that could possibly qualify.
    """
    wanted = [game for game in dict.fromkeys(games) if game]
    if not wanted:
        return []
    ratings = given_by_game(connection, wanted)
    rows = {design.game: design for design in designs(connection)}
    written: list[awards.Award] = []
    for game in wanted:
        votes_on = ratings.get(game, ())
        if len(votes_on) < CROWD_PEOPLE:
            continue
        written += awards.grant(
            connection,
            game_due_from(
                game,
                rows.get(game),
                match_rows.leaderboard_matches(connection, game=game),
                votes_on,
                awards.score_evidence(connection, game),
                awards.held(connection, game=game),
            ),
        )
    return written


def grant_for_save(connection: Any, user: int, game: str) -> list[awards.Award]:
    """Write the badges this saved design completed, and return them.

    What the save route calls after `drafts.keep`, inside the save's own transaction.
    One save can complete a badge for three people at once -- the saver earns *Ancestor*,
    the game they built from earns its owner *Descendant*, and the game behind that one
    earns its owner *Family tree* -- so the pass runs for each of them.
    """
    rows = designs(connection)
    return _grant_for(connection, _in_the_line(user, game, rows), rows)


def _in_the_line(user: int, game: str, rows: Sequence[Design]) -> list[int]:
    """The saver, and the owners of the two games above this one.

    Nobody further up can be completed by this save, because the top level is where
    N12 stops.
    """
    by_hash = {design.game: design for design in rows}
    people = [user]
    found = by_hash.get(game)
    for _step in range(2):
        found = None if found is None else by_hash.get(found.parent or "")
        if found is not None and found.owner is not None and found.owner not in people:
            people.append(found.owner)
    return people


def _grant_for(
    connection: Any, people: Sequence[int], rows: Sequence[Design] | None = None
) -> list[awards.Award]:
    """Grant what each of these people is due, off one read of the shared tables."""
    tables = rating.load_many(connection)
    grid_cells = cells(connection, milestones._rated(tables))
    games = designs(connection) if rows is None else rows
    written: list[awards.Award] = []
    for person in people:
        written += awards.grant(
            connection, _due_for(connection, person, tables, grid_cells, games)
        )
    return written


def due_for(connection: Any, user: int) -> tuple[awards.Award, ...]:
    """This person's outstanding badges, read off the store."""
    tables = rating.load_many(connection)
    return _due_for(
        connection,
        user,
        tables,
        cells(connection, milestones._rated(tables)),
        designs(connection),
    )


def _due_for(
    connection: Any,
    user: int,
    tables: milestones.Tables,
    grid_cells: milestones.Cells,
    games: Sequence[Design],
) -> tuple[awards.Award, ...]:
    return due_from(
        user,
        match_rows.leaderboard_matches(connection, user=user),
        tables,
        grid_cells,
        awards.held(connection, holder=user),
        games,
        ratings(connection, user),
    )


def ratings(connection: Any, voter: int) -> tuple[Rated, ...]:
    """What this person has rated and when, for N13. No star is selected.

    The three columns the badge is allowed, which is why this is a query of its own
    rather than a call to `votes.of`: a reader that never fetches `stars` cannot be made
    to read one by a later edit.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, updated from votes where voter = %s order by updated, game",
            (int(voter),),
        )
        return tuple(
            Rated(voter=int(voter), game=str(row[0]), rated=row[1])
            for row in cursor.fetchall()
        )


def given_by_game(connection: Any, games: Sequence[str]) -> dict[str, tuple[Given, ...]]:
    """Every rating of each of these games, oldest first, for N1 and U1.

    One query for the whole set, because `chess-variant-sandbox site load` asks about the library at
    once. These two badges do test the value, so the star is selected here; N13's reader
    is the one that must not see it.
    """
    if not games:
        return {}
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, voter, stars, updated from votes where game = any (%s)"
            " order by game, updated, voter",
            (list(games),),
        )
        rows = cursor.fetchall()
    found: dict[str, list[Given]] = {}
    for game, voter, stars, updated in rows:
        found.setdefault(str(game), []).append(
            Given(voter=int(voter), stars=int(stars), rated=updated)
        )
    return {game: tuple(each) for game, each in found.items()}


def designs(connection: Any) -> tuple[Design, ...]:
    """Every `games` row the designer badges read, oldest first.

    The whole table rather than one person's rows, because *Descendant* and *Family tree*
    are earned by what somebody else saved, and a library game has to be here too so that
    a game built from one can be told apart from a game built from a person's design.

    Every row carries its building blocks, the library's included: U4 counts the blocks
    of the games a person has played and of the rated games, which are library rows.
    `rules_used.used` is the one reading of what a spec draws on.

    A deleted game is left out. Its rules are gone (`0010_deleted.sql`), so there are no
    blocks to count and nothing for a design to descend from; the badges already granted
    on it are rows of their own and stay.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash, owner, parent, origin, screen_passed, revoked, created, spec"
            " from games where spec is not null order by created, hash"
        )
        rows = cursor.fetchall()
    return tuple(
        Design(
            game=str(row[0]),
            owner=None if row[1] is None else int(row[1]),
            parent=None if row[2] is None else str(row[2]),
            origin=str(row[3]),
            screen_passed=bool(row[4]),
            created=row[6],
            revoked=row[5] is not None,
            blocks=rules_used.used(dict(row[7])),
        )
        for row in rows
    )


def cells(connection: Any, games: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Each of these games' grid cell, worked out from the spec the store holds.

    `grid.descriptor` reads the declared rules and nothing else, so this is a dictionary
    walk rather than a measurement, and the site keeps no cell column to fall out of step
    with the spec beside it.

    A game whose rules were deleted reads as an **empty** cell rather than as a missing
    one. There are no rules to take a cell off, so it covers no kind and is in nobody's
    *still open* list; an absent entry would mean something else entirely, which is a
    reading that lost track of a game (`milestones._cell`). Its rating table and the
    games played on it stay, so it is still a rated game.
    """
    if not games:
        return {}
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash, spec from games where hash = any (%s)", (list(games),)
        )
        return {
            str(found[0]): () if found[1] is None else grid.descriptor(dict(found[1]))
            for found in cursor.fetchall()
        }
