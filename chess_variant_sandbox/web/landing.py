"""The landing page: what everybody meets at `/`, the games being at `/game` (Q718, Q723).

`docs/archive/plan-understanding.md` §1 and §2 say what the site is and what its score means, in a
visitor's words; this module puts them on one page that scrolls, under a quiet board drawn
by the page's own still-board code (`web/static/page/landing.js`). The page is markup made
here, once, at import, and carried inside the single page as a `<template>`: it needs no
request to show, and `landing.js` only clones it and starts the board.

**The score's ingredients are read from the scoring, not restated.** Each one's weight, its
quality, the ends of its scale and the tests that rule a game out come from the current
weights file through `scoring.load_weights` and `scoring.FLOOR`, and how many computer players
measure a game and how many games a run holds come from `corpus`. What this module adds is
prose: what each ingredient asks, why it counts, and how it is worked out, in the words
`docs/reference/plain-words.md` §2 settles (*turns*, *the computer players*, *ruled out*).
A weights version that scores something this module has no words for fails at import
(`ingredients`), so the page cannot quietly describe a scoring rule that no longer exists,
and `tests/test_landing.py` holds the page's numbers to the weights file read on its own.

Display only: nothing here reaches a score, and nothing in `measurement.METRICS_SOURCES`
imports it.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.corpus import CANONICAL_AGENTS, CANONICAL_GAMES
from chess_variant_sandbox.metrics.components import QUICK_KILL_PLIES
from chess_variant_sandbox.metrics.scoring import DEFAULT_WEIGHTS, FLOOR, Weights, load_weights
from chess_variant_sandbox.spec.reference import PAGE as RULES_PAGE

# The contact line (Q720): the user's words, 2026-10-04, *"contact ... Refer to my github"*.
# A profile page, never an email address, so the page hands out nothing a robot could
# harvest and sets nothing; `site/terms.py` names the same address for a take-down, and
# `web/static/page/core.js` links it from the site's footer (`tests/test_landing.py` holds
# all three equal).
CONTACT_ID = "landingContact"
CONTACT = "https://github.com/nabfrew"
CONTACT_SAID = "github.com/nabfrew"
CONTACT_LINE = (
    "The site is made and run by one person. To get in touch, find them on GitHub at "
    f'<a href="{CONTACT}" rel="noopener">{CONTACT_SAID}</a>.'
)

# `site/quota.py`'s `CORRESPONDENCE_IDLE_SAID`, said again here because the site imports the
# page and so the page cannot import the site; `tests/test_landing.py` holds the two equal.
CORRESPONDENCE_IDLE_SAID = "five days without a move"


@dataclass(frozen=True)
class Ingredient:
    """One thing the score measures, said for a visitor."""

    title: str
    # What it asks of a game, in one sentence.
    asks: str
    # Why it belongs in the score.
    why: str
    # How the computer works it out.
    how: str
    # How a raw number is written on the page: `share` as a percentage, `number` as itself,
    # `seat` as the first player's share of the points (`first_player_advantage` only).
    unit: str = "number"
    # The word after a number of this kind (*moves*, *rating points*), if any.
    noun: str = ""


# The six qualities, in the order the page lists them, and what each asks.
QUALITIES: dict[str, tuple[str, str]] = {
    "depth": ("Depth", "Does thinking pay?"),
    "decisions": ("Decisions", "Is there something to think about on most turns?"),
    "pacing": ("Pacing", "Are the choices a good size, and do games vary in length?"),
    "drama": ("Drama", "Can a game turn round, and does a lead mean something?"),
    "variety": ("Variety", "Do two games go to different places?"),
    "balance": ("Balance", "Is it fair to both sides, and do games finish?"),
}

INGREDIENTS: dict[str, Ingredient] = {
    "skill_gradient": Ingredient(
        title="Does thinking longer win",
        asks="How much stronger a computer player gets each time it is allowed to think "
             "twice as long.",
        why="If more thought buys nothing, there is nothing to be good at. This is the "
            "biggest single ingredient, because skill paying is what makes a strategy game.",
        how="The computer players that look ahead differ only in how many positions they "
            "may look at before each move. Each gets a rating from its results, and a "
            "straight line through the ratings, against how many times the thinking "
            "doubled, gives the rating points one doubling is worth. Chess measures about "
            "120.",
        noun="rating points a doubling",
    ),
    "decision_criticality": Ingredient(
        title="Decisions that matter",
        asks="On a typical turn, how many of the moves on offer would cost you something.",
        why="Near none, every move is as good as any other and nothing needs thinking "
            "about; near all, there is one right move and the game is a puzzle. Of "
            "everything measured, it tells a broken game from a real one best.",
        how="In positions taken from the games played, every legal move is scored the same "
            "way on every game, with the piece values the rules imply. This is the share of "
            "moves that come out clearly worse than the best one: worse by more than the "
            "cheapest piece is worth.",
        unit="share",
    ),
    "piece_utilisation": Ingredient(
        title="Use of the army",
        asks="Whether every kind of piece gets used, or some are scenery.",
        why="A piece nobody moves is a rule that adds nothing to the game.",
        how="Counts how many moves each kind of piece made, and measures how evenly they "
            "are spread, from 0 (one kind makes every move) to 1 (every kind moves as often "
            "as the others), so a game with three kinds of piece compares with one with "
            "nine.",
    ),
    "instance_participation": Ingredient(
        title="Pieces that move",
        asks="Whether the pieces on the board at the start ever move at all.",
        why="Use of the army looks at kinds of piece and cannot see five of one kind "
            "sitting at the back while a sixth does all the work. A designer testing a new "
            "game named that failure, so it is counted.",
        how="The share of pieces on the board at the start that move at least once in a "
            "game, over the games played. When few move the game is dull, and when every "
            "single one must, the game may be a slog.",
        unit="share",
    ),
    "mean_branching": Ingredient(
        title="Choices per turn",
        asks="How many legal moves a player typically has.",
        why="Too few and there is nothing to decide; too many and a position is soup "
            "nobody can read.",
        how="The number of legal moves, averaged over every turn of every game played.",
        noun="moves",
    ),
    "ply_spread": Ingredient(
        title="Length variety",
        asks="Whether games vary in length or always run about the same.",
        why="A game that always ends on the same turn is running a script.",
        how="How far game lengths spread around their average, divided by the average, so "
            "a long game and a short one compare fairly: 0 is every game the same length.",
    ),
    "blunder_recoverability": Ingredient(
        title="Recovery from a mistake",
        asks="Whether one bad move ends the game.",
        why="If a single slip is always fatal, the game is a memory test; if nothing you "
            "do matters, it is not a game. It is noisy, but it tells broken games from "
            "real ones better than almost anything else here.",
        how="A bad move is forced into a real position from a game played, and the game is "
            "played on from there. This is the share of those trials the side that slipped "
            "still saved.",
        unit="share",
    ),
    "lead_conversion": Ingredient(
        title="Converting a lead",
        asks="Whether being ahead is worth something.",
        why="If a lead means nothing, the middle of a game is noise; if the first real lead "
            "always wins, the rest of the game is a formality.",
        how="Among the won games where one side was clearly ahead in material part of the "
            "way through, the share that side went on to win. When a run has too few such "
            "games, or the computer players cannot judge who is ahead in this game, it is "
            "left out rather than scored as zero.",
        unit="share",
    ),
    "position_divergence": Ingredient(
        title="Games that go different places",
        asks="Whether each game goes somewhere the others did not.",
        why="If the same game keeps being replayed, there is little to explore.",
        how="For each game where neither side moves at random, the positions it "
            "passed through are compared with the most similar other game. This is the "
            "share of positions the two did not have in common, averaged over the games.",
        unit="share",
    ),
    "stall_rate": Ingredient(
        title="Games that never finish",
        asks="How often a game is stopped by the turn limit or by a long run of moves "
             "with nothing taken, instead of reaching an ending its rules name.",
        why="A game that does not end is not a contest. Too many and the game is ruled "
            "out altogether.",
        how="The share of games which are stopped this way, among those where neither side "
            "is the computer player that moves at random. A game drawn by repeating a position has reached an ending, "
            "so it does not count here.",
        unit="share",
    ),
    "first_player_advantage": Ingredient(
        title="First-player advantage",
        asks="How much of an edge moving first is worth.",
        why="Both sides should be playing the same game. It counts for little, because it "
            "is the least certain thing measured: games differ on it by not much more than "
            "chance alone moves it.",
        how="Only games where a computer player meets an exact copy of itself are read, so "
            "any difference comes from moving first. The first player's share of the points "
            "is compared with an even half.",
        unit="seat",
    ),
}

# The tests that rule a game out, by what they read, in a visitor's words. `{n}` is the
# threshold, written as the page writes that kind of number.
RULE_OUTS: dict[str, tuple[str, str]] = {
    "search_advantage": (
        "A computer player that looks ahead must take at least {n} of the points from one "
        "that moves at random.", "share"),
    "stall_rate_worst_half": (
        "No more than {n} of games may be stopped by the turn limit or by a long run of "
        "moves with nothing taken.", "share"),
    "mean_branching": (
        "A player must have at least {n} legal moves on a typical turn.", "number"),
    "decisive_rate_worst_half": (
        "At least {n} of games must have a winner, rather than nearly all being drawn.",
        "share"),
    "mean_plies_worst_half": (
        "Games must last at least {n} turns on average.", "number"),
    "skill_gradient": (
        "Thinking twice as long must be worth at least {n} rating points.", "number"),
    "quick_kill_rate_worst_half": (
        "No more than {n} of games may be won inside " + str(QUICK_KILL_PLIES) + " turns, "
        "which would mean an opening trap decides them.", "share"),
}

# Said once under the list: what *the half that does worse* means (`HOW_IT_WORKS`' second
# entry, `metrics/glossary.py`).
WORSE_HALF = (
    "Where a test is about how games end, it is read twice, on the games of the weaker "
    "computer players and on those of the stronger ones, and the half that does worse is "
    "the one that counts: one set of rules can be an opening trap for weak players and a "
    "dead shuffle for strong ones, and averaged together each failing would hide the "
    "other. One more test asks whether the strongest computer players win from one side "
    "nearly every time, whichever of them sits there: then the side decides the game, "
    "not the play.")


def _number(value: float) -> str:
    """A number as the page writes it: to three places under 10 (a curve's end such as 0.735
    must not read as 0.7), to a tenth from 10, whole where it is whole, with a thousands
    comma and no trailing zeros."""
    rounded = round(value, 3 if abs(value) < 10 else 1)
    if rounded == int(rounded):
        return f"{int(rounded):,}"
    return f"{rounded:,.3f}".rstrip("0")


def _percent(value: float) -> str:
    return f"{_number(value * 100)}%"


def written(value: float, unit: str, noun: str = "") -> str:
    """One raw number of an ingredient's kind, as the page writes it."""
    if unit == "share":
        return _percent(value)
    if unit == "seat":
        return f"{_percent(0.5 + value / 2)} of the points"
    return f"{_number(value)} {noun}".strip()


def share(weight: float, weights: Weights) -> str:
    """A weight as its share of the whole, to a tenth of a percent."""
    return f"{weight / sum(weights.weights.values()) * 100:.1f}%"


def scale(name: str, weights: Weights) -> str:
    """Where an ingredient earns full marks and where none, from its curve's ends."""
    entry = INGREDIENTS[name]
    curve = weights.curves[name]
    points = curve.points

    def at(key: str) -> str:
        return written(points[key], entry.unit, entry.noun)

    # *Or less* and *or more* are left off an end a share cannot pass (0% and 100%).
    def below(key: str) -> str:
        edge = entry.unit == "share" and points[key] <= 0
        return at(key) + ("" if edge else " or less")

    def above(key: str) -> str:
        edge = entry.unit == "share" and points[key] >= 1
        return at(key) + ("" if edge else " or more")

    if curve.kind == "lower_better":
        return (f"Full marks at {at('good_at')}, falling in a straight line to nothing at "
                f"{above('zero_at')}.")
    if curve.kind == "higher_better":
        return (f"Nothing at {below('zero_at')}, rising in a straight line to full "
                f"marks at {above('good_at')}.")
    return (f"Best in the middle: nothing at {below('zero_low')}, full marks from "
            f"{at('good_low')} to {at('good_high')}, and nothing again at "
            f"{above('zero_high')}.")


class LandingWordsMissing(Exception):
    """The weights score something the landing page has no words for, or the reverse."""


def ingredients(version: str = DEFAULT_WEIGHTS) -> list[dict[str, Any]]:
    """The score's ingredients under `version`, quality by quality, each with its words, its
    weight's share of the whole and the ends of its scale, all read from the weights."""
    weights = load_weights(version)
    scored, worded = set(weights.weights), set(INGREDIENTS)
    families = set(weights.families.values())
    if scored != worded or families != set(QUALITIES):
        raise LandingWordsMissing(
            f"weights {weights.version} score {sorted(scored ^ worded) or 'the same things'}"
            f" and group them as {sorted(families)}; the landing page has words for "
            f"{sorted(worded)} in {sorted(QUALITIES)}")
    found = []
    for family, (title, asks) in QUALITIES.items():
        names = sorted((name for name in weights.weights if weights.families[name] == family),
                       key=lambda name: -weights.weights[name])
        found.append({
            "family": family,
            "title": title,
            "asks": asks,
            "share": share(weights.family_weights()[family], weights),
            "ingredients": [{
                "name": name,
                "title": INGREDIENTS[name].title,
                "asks": INGREDIENTS[name].asks,
                "why": INGREDIENTS[name].why,
                "how": INGREDIENTS[name].how,
                "share": share(weights.weights[name], weights),
                "scale": scale(name, weights),
                "optional": name in weights.optional,
            } for name in names],
        })
    return found


def rule_outs(version: str = DEFAULT_WEIGHTS) -> list[str]:
    """The tests that rule a game out under `version`, each with its threshold."""
    weights = load_weights(version)
    found = []
    for gate in weights.gates:
        if gate.component not in RULE_OUTS:
            raise LandingWordsMissing(f"no words for the test on {gate.component}")
        words, unit = RULE_OUTS[gate.component]
        bound = gate.minimum if gate.minimum is not None else gate.maximum
        assert bound is not None
        found.append(words.format(n=written(float(bound), unit)))
    return found


def _players() -> tuple[int, list[int], int]:
    """How many computer players measure a game, the positions the ones that look ahead may
    look at, and how many games one run holds: `CANONICAL_GAMES` over each ordered pairing,
    a player's pairing with itself included."""
    budgets = sorted(int(found[1]) for agent in CANONICAL_AGENTS
                     if (found := re.search(r"nodes=(\d+)", agent)))
    count = len(CANONICAL_AGENTS)
    return count, budgets, count * count * CANONICAL_GAMES


_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
          8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def _say(n: int) -> str:
    return _WORDS.get(n, str(n))


def _e(text: str) -> str:
    return html.escape(text, quote=True)


def _section(key: str, heading: str, body: str, needs: str = "") -> str:
    """One scrolling section. `needs` names what the server must have for it to show
    (`landing.js` hides it otherwise): `accounts` for the site's people."""
    marked = f' data-needs="{needs}"' if needs else ""
    return (f'<section class="landingPart" id="landing-{key}"{marked}>'
            f"<h3>{_e(heading)}</h3>{body}</section>")


# Q741: *Made by members*, the games members designed and published, straight under the
# opening board. The site sends them (`site/showcase.py`, `madeHere` on `/api/library`:
# newest published first, at most four); `landing.js` fills the shelf. The shelf's height
# is fixed (`style.py`'s `.membersShelf`), so filling it moves nothing below it: it holds
# its place empty while the listing loads, then shows the games, or, before anybody has
# published, the invitation below in the same space. The local application has no members
# and no Publish, so the section goes with `data-needs="accounts"`.
MEMBERS_HEADING = "Made by members"
MEMBERS_LEAD = "Games members designed here and published, the newest first."
MEMBERS_EMPTY = (
    "<b>The shelf is empty so far.</b> No member has published a game here yet; the "
    "first one goes here.")
MEMBERS_ALL = "All {n} on the games page"


def _members_section() -> str:
    head = (
        '<div class="membersHead">'
        f"<h3>{_e(MEMBERS_HEADING)}</h3>"
        # `{n}` is filled in by `landing.js` when there are more than the shelf shows.
        '<a href="/game" id="membersAll" class="small" hidden>'
        + _e(MEMBERS_ALL).replace("{n}", '<span id="membersCount"></span>')
        + "</a></div>")
    empty = (
        '<div class="membersEmpty" id="membersEmpty" hidden>'
        f"<p>{MEMBERS_EMPTY}</p>"
        '<a class="button" href="#/design">Build your own</a></div>')
    return (
        '<section class="landingPart landingMembers" id="landing-members"'
        ' data-needs="accounts">'
        f'{head}<p class="muted">{_e(MEMBERS_LEAD)}</p>'
        '<div class="membersShelf" id="membersShelf" aria-busy="true">'
        f"{empty}</div></section>")


# Q788 (the user, 2026-10-09: *"I'd like the new games highlighted, as well as
# intransitive"*): under the opening board, the library's newest games, then the spotlight
# (*Intransitive*) beside this month's game. The games, their sentences and the day each
# joined are the server's (`web/launch.py`'s `new_cards`, `SPOTLIGHT`, `month_game`, sent
# as `newGames`, `spotlight` and `monthGame` on `/api/library`); `landing.js` fills the
# places below. Each card's score is the library card's own pill (`scoreTag`), so a game
# nobody has measured yet says *not measured* and never shows a number, and a stale one is
# marked provisional. The shelf holds one height whatever it holds, as *Made by members*
# does. `/game` repeats none of it: it opens on the search and the filters.
#
# Q826: the shelf is derived from the day each game joined (`launch.NEW_WINDOW_DAYS`).
# When nothing joined inside the window, the server sends the most recently added marked
# not `fresh`, and `landing.js` puts the `data-older` words in place of the heading and
# the lead, so a game a month old is never called new.
NEW_HEADING = "New in the library"
NEW_LEAD = (
    "The games added in the last two weeks, newest first, each one playable now. A game the "
    "computer players have not measured yet says so, rather than show a score.")
# The shelf holds `launch.NEW_SHOWN`, and a day can add more than that (fourteen games
# joined on 9 and 10 October 2026), so its link opens the games page ordered by *date
# added*, where every one of them is, newest first (`landing.js`).
NEW_ALL = "All the games, newest first"
NEW_OLDER_HEADING = "Recently added to the library"
NEW_OLDER_LEAD = (
    "Nothing has joined the library in the last two weeks, so these are the games added "
    "most recently, newest first, each one playable now.")
NEW_ADDED = "added {day}"
SPOTLIGHT_LABEL = "In the spotlight"
MONTH_LABEL = "This month's game"


def _new_section() -> str:
    return (
        '<section class="landingPart landingShelf" id="landing-new">'
        '<div class="membersHead">'
        f'<h3 data-older="{_e(NEW_OLDER_HEADING)}">{_e(NEW_HEADING)}</h3>'
        f'<a href="/game" class="small" data-newest>{_e(NEW_ALL)}</a></div>'
        f'<p class="muted" data-older="{_e(NEW_OLDER_LEAD)}">{_e(NEW_LEAD)}</p>'
        f'<div class="newShelf" id="newShelf" aria-busy="true" data-added="{_e(NEW_ADDED)}">'
        "</div></section>")


TODAY_LABEL = "Today's game"
TODAY_NOTE = (
    "You move first, against <b data-opponent></b>. A practice game: undo and restart are "
    "on, and it counts for nothing.")


def _pick(key: str, label: str, ways: str, extra: str = "") -> str:
    """One panel of the picks row: a board, its label, and places `landing.js` fills. The
    panel holds its place while the listing loads, and goes when the server sent no game."""
    return (
        f'<div class="pick panel" id="{key}" aria-busy="true">'
        f'<div class="pickBoard" id="{key}Board"></div>'
        '<div class="pickText">'
        f'<div class="muted small">{_e(label)}<span data-when></span></div>'
        '<h4><a data-name></a></h4>'
        "<p data-sentence></p>"
        f"{extra}"
        f'<div class="row pickWays">{ways}</div>'
        "</div></div>")


def _picks_section() -> str:
    """Q788: the day's game (Q564's first step, moved here from `/game`), the spotlight and
    this month's game (Q603's, moved here too), side by side at 1120 px and one above the
    other on a phone."""
    today = _pick(
        "landingToday", TODAY_LABEL,
        '<button type="button" class="primary" data-play>Play the computer</button>'
        '<button type="button" data-harder hidden>A harder one: <span></span></button>',
        f'<p class="small muted" data-note>{TODAY_NOTE}</p>')
    spotlight = _pick(
        "landingSpotlight", SPOTLIGHT_LABEL,
        '<a class="button primary" data-go>Play</a><span data-score></span>',
        '<p class="small" data-why></p>')
    month = _pick(
        "landingMonth", MONTH_LABEL,
        '<a class="button primary" data-go>Play</a>',
        '<p class="small" data-why></p>'
        '<p class="small" data-needs="member"><a data-find>Find someone to play it '
        "with</a>.</p>"
        '<p class="small" data-needs="visitor"><a href="/signin">Sign in</a> to play it '
        "with someone.</p>"
        '<p class="small muted" data-how></p>')
    return (
        '<section class="landingPart landingPicks" id="landing-picks">'
        + spotlight + today + month + "</section>")


def _ingredients_html(version: str) -> str:
    weights = load_weights(version)
    qualities = ingredients(version)
    parts = []
    for quality in qualities:
        rows = "".join(
            '<li class="ingredient" data-ingredient="{name}">'
            '<div class="ingredientHead"><b>{title}</b>'
            '<span class="ingredientWeight">weight {share}</span></div>'
            '<p>{asks}</p>'
            '<p><span class="muted">Why it counts.</span> {why}</p>'
            '<p><span class="muted">How it is worked out.</span> {how}</p>'
            '<p><span class="muted">Its scale.</span> {scale}</p></li>'.format(
                name=_e(item["name"]), title=_e(item["title"]), share=_e(item["share"]),
                asks=_e(item["asks"]), why=_e(item["why"]), how=_e(item["how"]),
                scale=_e(item["scale"]))
            for item in quality["ingredients"])
        parts.append(
            f'<div class="quality" data-quality="{_e(quality["family"])}">'
            f'<h4>{_e(quality["title"])} <span class="muted">· {_e(quality["share"])} of the '
            f'weight</span></h4><p class="muted">{_e(quality["asks"])}</p>'
            f'<ul class="ingredients">{rows}</ul></div>')
    number = re.sub(r"^v", "", weights.version)
    tests = "".join(f"<li>{_e(line)}</li>" for line in rule_outs(version))
    floor = f"{FLOOR:g}"
    return (
        f'<details class="landingMore" id="landingIngredients">'
        f"<summary>What goes into the score, why, and how each part is worked out</summary>"
        f'<div class="landingMoreBody">'
        f"<p>These are the scoring rules in force now, version {_e(number)}. Each of the "
        f"{_say(len(weights.weights))} ingredients below is measured on the games the "
        f"computer players played, and turned into marks from 0 to 1 on its own scale. "
        f"Ingredients that answer the same question are averaged, the heavier ones counting "
        f"for more, which gives {_say(len(qualities))} qualities, each out of 1. The "
        f"qualities are then multiplied together, each raised to its weight, so one quality "
        f"near zero pulls the whole score down instead of being outvoted: a game with nothing "
        f"in it can look simple, varied and dramatic, and averaging let a deliberately dull "
        f"test game score 0.64. No quality counts as lower than {_e(floor)}, so a game that "
        f"fails one still differs from a game that fails three.</p>"
        f"<p>A weight is the share of the whole it carries; the weights of the ingredients "
        f"add up to the weight of their quality.</p>"
        f'{"".join(parts)}'
        f'<div class="quality" data-quality="rule-outs"><h4>What rules a game out</h4>'
        f"<p>Some failings are not marked down but rule a game out: its score still shows, "
        f"and means nothing. A game is <i>viable</i> when it passes all of these.</p>"
        f'<ul class="ruleOuts">{tests}</ul><p class="muted">{_e(WORSE_HALF)}</p></div>'
        f"</div></details>")


# The page's headline and its first paragraph: what the site is, in its first readable words.
# The site draws both outside the template too (`drawn_intro`, G4 of `docs/plan-growth.md`),
# and `web/page/front.py` makes the front page's description of them.
HEADLINE = "Invent a chess-like game, play it straight away, and see how it measures up."
LEAD = "A vibe-coded playground for designing chess-like strategy games."

# The two ways in that every reader is offered, before `landing.js` adds the reader's own.
_WAYS = (
    '<a class="button primary" href="/game">See the games</a>'
    '<a class="button" href="#/design">Build your own</a>'
)


def drawn_intro() -> str:
    """The headline, the first paragraph and the two ways in, outside the `<template>` (G4).

    The site puts this in `#view` at `/`, so a reader without script, and a crawler, reads
    what the site is. It is the start of the hero's own markup, so when `landing.js` clones
    the template over it the headline lands where it already stood and a person sees the
    landing they always saw; only the board and the reader's own way in arrive with it.
    """
    return (
        '<div class="landing"><section class="landingHero"><div class="landingIntro">'
        f'<h2 class="landingTitle">{_e(HEADLINE)}</h2>'
        f'<p class="landingLead">{_e(LEAD)}</p>'
        f'<div class="row landingWays">{_WAYS}</div>'
        "</div></section></div>"
    )


def html_page(version: str = DEFAULT_WEIGHTS) -> str:
    """The landing page's markup: the opening board's place, then the sections in order."""
    count, budgets, games = _players()
    looking = len(budgets)
    budget_words = (f"{budgets[0]:,} to {budgets[-1]:,}" if len(budgets) > 1
                    else f"{budgets[0]:,}") if budgets else ""
    hero = (
        '<section class="landingHero">'
        '<div class="landingIntro">'
        f'<h2 class="landingTitle">{_e(HEADLINE)}</h2>'
        f'<p class="landingLead">{_e(LEAD)}</p>'
        '<div class="row landingWays">'
        + _WAYS +
        # Q723: everybody meets this page, so the third way in is the reader's own: the
        # door for a visitor to the site, a member's page for a member, and on the local
        # application neither (`landing.js` removes what does not apply).
        '<a class="button" href="/signin" data-needs="visitor">Sign in</a>'
        '<a class="button" href="/profile" data-needs="member">Your page</a>'
        # Q760: and for a member, the way to everybody else.
        '<a class="button" href="/players" data-needs="member">Find players</a></div>'
        '<p class="muted small landingScroll">What it is, and how the score works, below.</p>'
        "</div>"
        '<figure class="landingShow" id="landingShow">'
        '<div class="landingBoard" id="landingBoard" aria-hidden="true"></div>'
        # Q741: the day's game is named as the day's (`landing.js`).
        '<figcaption class="small" id="landingCaption" data-today="Today&#39;s game">'
        "Some of the games here, each a different mix of rules.</figcaption>"
        '<div class="row landingShowControls">'
        '<button type="button" class="quiet" id="landingPause" hidden>pause</button>'
        '<button type="button" class="quiet" id="landingNext" hidden>another game</button>'
        "</div></figure></section>")
    parts = [
        _new_section(),
        _picks_section(),
        _members_section(),
        _section("intent", "What this is", (
            "<p>A <b>vibe-coded playground for designing chess-like strategy games</b>. "
            "Vibe-coded means it was built by describing it, in plain language, to an AI that "
            "wrote the code, so expect a rough edge here and there. If it had been hand "
            "crafted, it would have been immaculate, obviously.</p>"
            "<p>It is a place to invent games, play them, and find the ones worth playing. "
            "Chess is here too, as a familiar point of comparison.</p>")),
        _section("palette", "Games are built from a palette of rules", (
            "<p>Every game here is a mix of rules taken from one palette: the size and shape "
            "of the board (squares or hexagons, and edges that can join round like a "
            "cylinder), the pieces and how each one moves and takes, where they start, what "
            "happens to a piece that is taken (it can come back into play from your hand), "
            "promotion, and how a game is won: by checkmate, by checking three times, by "
            "reaching a square, and more.</p>"
            "<p>The designer puts the palette in front of you as cards and boards, with no "
            "programming. Change a rule and the board answers at once: where each piece can "
            "go, and whether the rules still make a game.</p>"
            # G8: the palette written down, the page a person reads and an AI is given.
            f'<p>The whole palette is written down as <a href="{RULES_PAGE}">the rule '
            "language</a>: a game is one file in it, which you can write by hand or ask an AI "
            "to write, then load into the designer to see it on the board.</p>")),
        _section("play", "Play any of them now, against the computer", (
            "<p>Every game is playable in the browser straight away, against computer "
            "players from gentle to strong. No account is needed: a practice game has undo "
            "and restart, and counts for nothing. A game you are designing can be played the "
            "same way, before it is saved.</p>")),
        _section("people", "Play other people, by correspondence", (
            "<p>Members can challenge each other to a game by correspondence: each of you "
            "moves when it suits you, the site keeps the board, and it can send an email "
            "when it is your move. A game is lost by the player to move after "
            f"{CORRESPONDENCE_IDLE_SAID}. "
            "With an account your games count: you get a rating on each game, and beating "
            "the strongest computer player on a game puts your name on it. Joining is by "
            "invitation for now.</p>"
            # Q760: a member is shown the way to find somebody to play.
            '<p data-needs="member"><a href="/players">Find a member</a> by name, open '
            "their page to see the games they made and played, and challenge them from "
            "there.</p>"), needs="accounts"),
        _section("viable", "The computer checks that a game works", (
            f"<p>Before a game is scored, {_say(count)} computer players play it: one that "
            f"moves at random, one that takes whatever it can, and {_say(looking)} that look "
            f"ahead, allowed to look at {budget_words} positions before each move. Each "
            f"meets every other, and a copy of itself, from both sides: {games:,} games "
            "a run. That is one run; a game in the library is measured four times, on "
            "four different sets of games.</p>"
            "<p>A game is <b>viable</b> when it passes the basic tests of a working game: "
            "games finish, they are not almost all drawn or won by an opening trap, a "
            "player who thinks beats one who moves at random, thinking longer pays, and "
            "neither side wins just by moving first. A game that fails one is ruled out, "
            "however well it does elsewhere. The tests and their numbers are under the score "
            "below.</p>")),
        _section("score", "The score: a best estimate, not a judgement", (
            "<p>Every measured game gets a <b>score out of 1</b> for how interesting it "
            "looks when the computer plays it. A game scores well when the stronger player "
            "usually wins, so skill pays; there are real choices on most turns; games don't "
            "stall or end in a few moves; a game can turn round; no two games go the same "
            "way; and the first player has no big edge.</p>"
            "<p><b>It is a preliminary, evolving measure: the best estimate available of "
            "giving a game a single value, and not to be treated as a definitive "
            "judgement.</b> The computer players are strong on games like chess and weaker "
            "on games won by a long plan, where they can miss what a person sees. So each "
            "score says what it rests on (<i>0.810 · 4 runs</i>, with a ± for how far "
            "chance alone could move it), each game says how far to trust its computer "
            "players, and most games are too close to put in order: the score tells a "
            "working game from a broken one, not first from second.</p>"
            "<p>Scores are measured again as the computer players get stronger and as the "
            "scoring improves. Until a game is measured again, its old score stays on show, "
            "marked <i>provisional</i>, and it is never compared with current ones. If you "
            "beat the strongest computer player on a game easily, that is news about the "
            "score too.</p>"
            + _ingredients_html(version))),
        _section("more", "And more", (
            '<ul class="landingFeatures">'
            '<li><a href="/game">The library</a>: every game, with a search, filters by the '
            "kind of game, and sorting by any ingredient of the score.</li>"
            '<li><a href="#/map">The map</a>: the games laid out by how alike they are, in '
            "communities of games more like each other than like the rest.</li>"
            '<li><a href="#/design">The designer</a>: start from a blank board, a template '
            "or any game in the library, and save and publish what you make.</li>"
            f'<li><a href="{RULES_PAGE}">The rule language</a>: every rule a game can use, '
            "written down, with what makes a valid game and what makes a good one.</li>"
            '<li data-needs="round"><a href="#/round">The monthly round</a>: members put games forward and vote '
            "on them.</li>"
            '<li data-needs="accounts"><a href="/badges">Badges</a> for things done, and a '
            "page for each member with their games and ratings.</li>"
            "<li>A computer player that learned a game from its own play, offered on a game or "
            "two as an experiment.</li>"
            '<li><a href="#/whats-new">What\'s new</a>: every change, newest first.</li>'
            "</ul>")),
    ]
    return (
        '<div class="landing" id="landing">'
        + hero + "".join(parts)
        + f'<div class="landingFoot" id="{CONTACT_ID}"><p>{CONTACT_LINE}</p></div>'
        + "</div>")


# The page's markup, made once: the weights file does not change under a running server.
HTML = html_page()
