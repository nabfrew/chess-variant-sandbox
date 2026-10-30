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
from urllib.parse import quote

from chess_variant_sandbox.corpus import CANONICAL_AGENTS, CANONICAL_GAMES
from chess_variant_sandbox.metrics.components import QUICK_KILL_PLIES
from chess_variant_sandbox.metrics.scoring import DEFAULT_WEIGHTS, FLOOR, Weights, load_weights
from chess_variant_sandbox.spec import reference
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


# ------------------------------------------------------------------ the palette, as words

# LP2 (`docs/archive/plan-landing.md` §3.2, direction B's *Your idea is in here somewhere*): every
# family of rules the language offers, one row each, and in each row the rules as words a
# visitor might use for an idea. Each word is a rule the designer has a control for (`key`,
# and inside its card `field` and the option `value` it means), says the designer's own name
# for that control (`card`), and links to the section of `/rules` that writes it down.
# `tests/test_landing_wall.py` holds the table to the language: every family of `RULES.md`'s
# *What the language can make* (`reference.FAMILIES`) has a row or a reason in
# `NOT_ON_THE_WALL`; every word's key is in a family its row shows; every `card` is the
# designer's (`web/cards.py`'s card model, or the hand-drawn control's words in the
# designer's script); every link lands on a heading of `/rules`. `web/stories.py` reads a
# story's family and its colour from here. The colours are the map's per-kind hues (the
# user's choice E1, §4), one per row, set on the row's class by `web/page/style.py`.


@dataclass(frozen=True)
class Word:
    """One rule on the wall."""

    said: str
    # The rule key as `reference.FAMILIES` spells it: `rules.reach`, `pieces[].region`, a
    # kind of move (`slide`), `board.lattice`.
    key: str
    # The designer's words for the control that sets it: a rule card's label, a piece
    # field's, or the words of a control the designer draws by hand (a kind of move, a
    # sight's effect, the board's cells).
    card: str
    # The control inside the key's card the word means, dotted, where it is one of them.
    field: str = ""
    # The option of that control the word means, where it is one.
    value: str | bool | None = None


@dataclass(frozen=True)
class Block:
    """One family of rules, a row of the wall."""

    # Lower case, as a story names its family (`web/stories.py`).
    family: str
    # The class that carries the row's colour, and the token it reads (choice E1).
    hue: str
    colour: str
    # The families of `RULES.md`'s *What the language can make* this row's words are drawn
    # from (`reference.FAMILIES`' titles).
    shows: tuple[str, ...]
    words: tuple[Word, ...]

    @property
    def title(self) -> str:
        return self.family[:1].upper() + self.family[1:]


BLOCKS: tuple[Block, ...] = (
    Block("how a game is won", "k-ending", "kind-ending", ("How a game is won",), (
        Word("checkmate", "rules.win", "how the game is won", value="checkmate"),
        Word("take the king", "rules.win", "how the game is won", value="royal_capture"),
        Word("take every piece", "rules.win", "how the game is won", value="annihilation"),
        Word("reach a square", "rules.reach", "winning by arriving"),
        Word("run them out of a kind", "rules.extinction", "losing by running out of a piece"),
        Word("three checks", "rules.check_limit", "winning by giving checks"),
        Word("claim ground", "rules.territory", "claiming ground"),
        Word("a score when time runs out", "rules.scoring", "who wins when a clock runs out"),
    )),
    Block("how pieces move", "k-move", "kind-pieces", ("How pieces move",), (
        Word("slide", "slide", "slides"),
        Word("leap", "leap", "leaps"),
        Word("hop over a screen", "hop", "hops over"),
        Word("leap, but blockable", "lame", "blockable leap"),
        Word("sweep a line clear", "sweep", "sweeps"),
        Word("ride, jump after jump", "leap", "jumps again"),
        Word("keep to a region", "pieces[].region", "only on these squares"),
        Word("face a way, and turn", "pieces[].facing", "points a way"),
    )),
    Block("what a capture does", "k-capture", "kind-capture", ("What a capture does",), (
        Word("take only some kinds", "pieces[].captures", "may take only"),
        Word("become another piece", "rules.strike", "what a taken piece becomes",
             field="chain"),
        Word("go home", "rules.strike", "what a taken piece becomes", field="position",
             value="home"),
        Word("change places", "rules.strike", "what a taken piece becomes", field="position",
             value="swap"),
        Word("change sides", "rules.strike", "what a taken piece becomes",
             field="switches_sides"),
        Word("explode", "rules.explode_on_capture", "captures that explode"),
        Word("must be taken", "rules.must_capture", "a capture must be taken"),
    )),
    Block("what a piece does by looking", "k-sight", "kind-sight",
          ("What pieces do to each other without moving",), (
        Word("freeze", "pieces[].sight", "freezes it", field="effect", value="freeze"),
        Word("disarm", "pieces[].sight", "disarms it", field="effect", value="disarm"),
        Word("shield", "pieces[].sight", "shields it", field="effect", value="shield"),
        Word("let the opponent move it", "pieces[].sight", "commands it", field="effect",
             value="command"),
    )),
    Block("the board", "k-board", "kind-board", ("The board",), (
        Word("hexagons", "board.lattice", "hexagons", value="hex"),
        Word("walls", "rules.obstacles", "walls on the board"),
        Word("edges that join", "rules.wrap", "edges that join"),
        Word("portals", "rules.portals", "portals"),
        Word("portals the players move", "rules.portals", "portals", field="moved_by"),
    )),
    Block("who moves, and when", "k-turn", "kind-turn", ("Who moves, and when",), (
        Word("several moves a turn", "rules.turn", "more than one move a turn"),
        Word("both at once, in secret", "rules.simultaneous",
             "playing in rounds: both players move at once"),
        Word("collisions", "pieces[].beats", "in a collision beats"),
        Word("speeds", "pieces[].speed", "speed"),
        Word("spies", "pieces[].spy", "a spy"),
        Word("pieces that rest", "pieces[].rest", "rests after it moves"),
    )),
    Block("pieces nobody owns", "k-wild", "kind-wild", ("Pieces nobody owns",), (
        Word("walk a route", "rules.npcs", "wild pieces", field="route"),
        Word("eat what they meet", "rules.npcs", "wild pieces", field="piece"),
        Word("a bounty on their heads", "rules.npcs", "wild pieces", field="bounty"),
        Word("only some may take them", "rules.npcs", "wild pieces", field="immune"),
    )),
    # `RULES.md` puts a hand under *what a capture does*, since a taken piece is what goes
    # into it; the wall gives it a row of its own, because a stone placed from a hand is an
    # idea of its own (the stones story's family).
    Block("pieces in hand", "k-hand", "kind-hand", ("What a capture does",), (
        Word("drop what you took", "rules.drops", "pieces played back onto the board",
             field="from_captures", value=True),
        Word("start with pieces to place", "rules.drops", "pieces played back onto the board",
             field="stock"),
        Word("place them all first", "rules.drops", "pieces played back onto the board",
             field="placement", value=True),
        Word("no checkmate by a drop", "rules.drops", "pieces played back onto the board",
             field="no_mate"),
    )),
    Block("chess's own, if you want them", "k-chess", "muted",
          ("Chess's own rules", "When a game stops"), (
        Word("promotion", "rules.promotion", "promotion"),
        Word("en passant", "rules.en_passant", "taking a piece in passing"),
        Word("castling", "rules.castling", "castling"),
        Word("no check allowed", "rules.forbid_check", "check is not allowed"),
        Word("kings may not face", "rules.royal_facing", "royal pieces facing each other"),
        Word("a turn limit", "rules.max_plies", "the longest a game may run"),
        Word("repetition", "rules.repetition_draw", "drawing by repetition"),
        Word("a stall", "rules.no_progress_draw", "how long a game may stall"),
    )),
)

# A family of `RULES.md`'s that has no row, by its title, with the reason. Empty: every
# family is on the wall.
NOT_ON_THE_WALL: dict[str, str] = {
    # TC5 (`docs/plan-turncoat.md` §5.8): the designer's purse card and the page's market tray
    # are TC7's; the row goes on the wall with them, and this entry goes.
    "Coins and a market": "the designer has no purse card yet (TC7, docs/plan-turncoat.md §5.8)",
}

WALL_HEADING = "Your idea is in here somewhere"
WALL_LEAD = (
    "The whole palette, by what each part does to play. Every word is a rule the designer "
    "has a control for, and links to where <a href=\"{rules}\">the rule language</a> writes "
    "it down.")
WALL_BAND = "Your idea next."

# LP7 (`docs/archive/plan-landing.md` §5): each word, and each story's card, is a door into the
# designer at its rule, `#/design?rule=<key>`; `web/designing.doors` says what each key opens.
DOOR_SAID = "start a game with this rule"


def door_link(key: str) -> str:
    """The designer's address at one rule key."""
    return "#/design?rule=" + quote(key, safe=".")


def section_key(key: str) -> str:
    """The rule key whose section of `/rules` writes `key` down: a rule's own section, the
    board's, a piece's, or a way of moving's."""
    if key.startswith("rules."):
        return key
    if key.startswith("board."):
        return "board"
    if key.startswith("pieces[]."):
        return "pieces"
    return "pieces[].moves"


_HEADING = re.compile(r"^#{1,6} +`([^`]+)`(?::.*)?$")


def _slug(words: str, taken: dict[str, int]) -> str:
    """GitHub's id for a heading, as `web/page/rules_page.py`'s `slug` makes it (which this
    module cannot import: the page package imports this one). `tests/test_landing_wall.py`
    holds every link here to the ids that page draws."""
    base = re.sub(r"[^\w\- ]", "", words.strip().lower()).replace(" ", "-")
    seen = taken.get(base, 0)
    taken[base] = seen + 1
    return base if seen == 0 else f"{base}-{seen}"


def rule_sections(markdown: str | None = None) -> dict[str, str]:
    """Each rule key with a section of its own in the reference, and that section's id."""
    taken: dict[str, int] = {}
    found: dict[str, str] = {}
    fenced = False
    for line in (markdown if markdown is not None else reference.render_reference()).splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        heading = None if fenced else re.match(r"^#{1,6} +(.+?)\s*$", line)
        if heading is None:
            continue
        words = heading.group(1).replace("`", "").replace("**", "").replace("*", "")
        ident = _slug(words, taken)
        keyed = _HEADING.match(line.strip())
        if keyed:
            found.setdefault(keyed.group(1), ident)
    return found


def word_link(word: Word, sections: dict[str, str]) -> str:
    """Where a word links: its section of `/rules`."""
    return f"{RULES_PAGE}#{sections[section_key(word.key)]}"


def _wall_html() -> str:
    sections = rule_sections()
    rows = []
    for block in BLOCKS:
        # The door is the key alone (`data-door`); `landing.js` draws it beside the word as
        # `door_link`, so the page every view carries holds one attribute a word, not a link.
        words = "".join(
            f'<a class="wallWord" href="{_e(word_link(word, sections))}" '
            f'title="{_e("in the designer: " + word.card)}" data-door="{_e(word.key)}">'
            f'{_e(word.said)}</a>'
            for word in block.words)
        rows.append(
            f'<div class="wallRow {block.hue}" data-wall="{_e(block.family)}">'
            f'<dt><span class="swatch" aria-hidden="true"></span>{_e(block.title)}</dt>'
            f"<dd>{words}</dd></div>")
    return (
        f'<p class="muted">{WALL_LEAD.format(rules=RULES_PAGE)}</p>'
        f'<dl class="wall">{"".join(rows)}</dl>'
        f'<div class="wallBand"><p>{_e(WALL_BAND)}</p>'
        '<div class="row"><a class="button primary" href="#/design">Start building a game</a>'
        '<a class="button" href="/game">See the games</a></div></div>')


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


def _section(key: str, heading: str, body: str, needs: str = "", classes: str = "") -> str:
    """One scrolling section. `needs` names what the server must have for it to show
    (`landing.js` hides it otherwise): `accounts` for the site's people."""
    marked = f' data-needs="{needs}"' if needs else ""
    more = f" {classes}" if classes else ""
    return (f'<section class="landingPart{more}" id="landing-{key}"{marked}>'
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
# does. Since LP3 `/game` opens on the same shelves as one band (`shelves_html`).
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


# LP3 (`docs/archive/plan-landing.md` §4.2): *How the score works*, the score's long explanation as
# a view of its own, `#/score` on both servers. It carries the landing's two sections as
# they were, *The computer checks that a game works* and *The score*, with the ingredients,
# the tests that rule a game out and the provisional and re-certified marks, under the ids
# they had on the landing (`SCORE_ANCHORS`), so a link made to one of them still lands: the
# router sends a bare `#landing-score` here and scrolls to it (`landing.js`'s `showScore`).
# It is linked from the score pills' (i) (`core.js`'s `SCORE_LINK`), from the band at the
# top of `/game`, from every footer and from the landing.
SCORE_ROUTE = "#/score"
SCORE_TITLE = "How the score works"
SCORE_LEAD = (
    "How a game here is checked, and then scored: who plays it, what rules it out, what goes "
    "into the number, and how far to trust it.")
SCORE_ANCHORS = ("landing-viable", "landing-score", "landingIngredients")
SCORE_POINTER = (
    "Before a game is scored, computer players of several strengths play it to check that it "
    "works; then it gets a score out of 1 for how interesting it looks when they play it. The "
    f'score is a best estimate, not a judgement. <a href="{SCORE_ROUTE}">{SCORE_TITLE}</a> has '
    "the tests, every ingredient, and what each score rests on.")


def _viable_section() -> str:
    count, budgets, games = _players()
    looking = len(budgets)
    budget_words = (f"{budgets[0]:,} to {budgets[-1]:,}" if len(budgets) > 1
                    else f"{budgets[0]:,}") if budgets else ""
    return _section("viable", "The computer checks that a game works", (
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
        "below.</p>"))


def _score_section(version: str) -> str:
    return _section("score", "The score: a best estimate, not a judgement", (
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
        "marked <i>provisional</i>, and it is never compared with current ones. A score "
        "the site kept after playing only a sample of its games again says <i>from a "
        "sample</i>. If you beat the strongest computer player on a game easily, that is "
        "news about the score too.</p>"
        + _ingredients_html(version)))


def score_html(version: str = DEFAULT_WEIGHTS) -> str:
    """*How the score works*: the view's markup, carried as `<template id="scoreView">`."""
    return (
        '<div class="landing scoreView" id="howScore">'
        f'<h2 class="landingTitle">{_e(SCORE_TITLE)}</h2>'
        f'<p class="landingLead">{_e(SCORE_LEAD)}</p>'
        + _viable_section() + _score_section(version)
        + '<p class="muted"><a href="/game">See the games</a>, each with its score and what '
        "it rests on.</p></div>")


# LP3 (§4.2): the five shelves, *today's game*, *this month's game*, *in the spotlight*,
# *new in the library* and *made by members*, as one band at the top of `/game`, above the
# search (`browse.js` clones `<template id="shelvesBand">` into the library). The parts are
# the landing's own markup, so `landing.js`'s `fillShelves` fills either; the landing keeps
# them too until LP4 replaces it. The band's last line is its way to the score's account.
SHELVES_SCORE = (
    f'Every score here is a best estimate. <a href="{SCORE_ROUTE}">{SCORE_TITLE}</a>.')


def shelves_html() -> str:
    return (
        '<div class="landing shelves" id="shelves">'
        + _picks_section() + _new_section() + _members_section()
        + f'<p class="muted small shelvesScore">{SHELVES_SCORE}</p></div>')


# ------------------------------------------------------------------ the hero (LP4)

# LP4 (`docs/archive/plan-landing.md` §3.2, direction B): the page opens on a question, *Wouldn't it
# be cool if…*, whose ending turns over every few seconds (`landing.js`), each ending
# underlined in its family's colour beside the game that idea became, drawn from the
# listing's still. The story's own idea leads (`web/stories.py`, the hero's clause), and these
# follow, leaving out the story's family. Each game is a listed library game
# (`tests/test_landing_story.py` holds it), found in the listing by its name.
QUESTION = "Wouldn't it be cool if…"


@dataclass(frozen=True)
class Idea:
    """One ending of the hero's question, and the library game it became."""

    family: str  # a row of `BLOCKS`
    clause: str
    chip: str
    game: str  # the game's name in the library


HERO_IDEAS: tuple[Idea, ...] = (
    Idea("the board", "the left and right edges were joined?", "the board: edges that join",
         "cylinder chess"),
    Idea("pieces nobody owns",
         "two golems nobody owns walked the middle, eating what stood in their way?",
         "pieces nobody owns", "the golem well"),
    Idea("the board", "there were doors across the board that either player could move?",
         "portals the players move", "ferry"),
    Idea("how a game is won", "walking your king to the centre won the game?",
         "winning by arriving", "king of the hill"),
    Idea("who moves, and when", "both players moved at once, in secret?",
         "both players move at once", "breather"),
)

LEAD = (
    "Every one of those is a game here, built from one palette of rules. Put your idea "
    "together the same way: the site plays it at once, tells you how it plays, and gives you "
    "an opponent.")
NO_ACCOUNT = "Free to build and to play online, in your browser, with no account needed."
START = "Start building a game"
PLAY_TODAY = "Play today's game"

# What a search result and a link preview say of the front page after the question with the
# first story's ending (`web/page/front.py`): what the site does, in the lead's words. Fixed
# text, so the page's description is the same whichever stories hold.
DESCRIPTION_TAIL = (
    "Build one from a palette of rules and play it free online in your browser.")


def question(clause: str) -> str:
    """The question with one ending, as a sentence."""
    return f"Wouldn't it be cool if {clause}"

_FAMILY_HUE = {block.family: block.hue for block in BLOCKS}


def _ideas(story: dict[str, Any] | None) -> list[dict[str, str]]:
    """The hero's endings in order: the story's own first, then the others, leaving out its
    family (§3.2)."""
    ideas = []
    if story is not None:
        hero = story["hero"]
        ideas.append({"hue": story["hue"], "clause": hero["clause"], "chip": hero["chip"],
                      "game": hero["game"], "at": "" if hero["at"] is None else str(hero["at"]),
                      "story": story["key"], "example": "1" if story["example"] else ""})
    family = story["family"] if story is not None else None
    for idea in HERO_IDEAS:
        if idea.family != family:
            ideas.append({"hue": _FAMILY_HUE[idea.family], "clause": idea.clause,
                          "chip": idea.chip, "game": idea.game, "at": "", "story": "",
                          "example": ""})
    return ideas


# A constructed story's game is no library game (§2.1): the hero names it as what it is.
EXAMPLE_CAPTION = "an example made for this page"

STORY_SLOTS = 8
"""How many stories the pool may hold (LP6): the stylesheet has a rule per place in the pool
for the drawn headline's ending (`style.py`), and a test holds the pool within it."""


def headline(story: dict[str, Any] | None) -> str:
    """The question with its leading ending, as one line of text."""
    ideas = _ideas(story)
    return question(ideas[0]["clause"]) if ideas else QUESTION


# LP9 (`docs/reference/landing-cold-read.md` §3): a story's play doors start a practice game of
# its fixed game at once, the visitor in the seat the story is about (`landing.js`
# `wirePlayDoors`). Without the script each is a link to the game's page with that seat chosen
# (`?seat=<side>`, honoured by `detail.js`). In the hero the story's door stands beside
# *Play today's game*, and at 375 in its place, so a visitor who wants to move first meets the
# story's own game rather than leaving for another.
PLAY_STORY = "Play {name}"
PLAY_THIS = "Play this game"


def seat_link(digest: str, side: str) -> str:
    """The fixed game's page with the story's seat chosen for the visitor."""
    return f"/game/{digest}?seat={quote(side)}"


def play_door(story: dict[str, Any] | None, words: str, classes: str = "button") -> str:
    """A door that starts a practice game of the story's fixed game in the story's seat, or
    nothing for no story or a constructed one (its game is no library game)."""
    if story is None or story.get("example"):
        return ""
    fixed = story["fixed"]
    seat = story.get("seat") or {"index": 0, "side": "white"}
    digest = str(fixed["hash"])
    # LP11: a story may name the computer player its doors seat (`Story.opponent`), as it
    # names the seat; without one, `landing.js` asks the game's page for its gentlest.
    against = story.get("opponent")
    agent = f' data-agent="{_e(against["agent"])}"' if against else ""
    return (f'<a class="{classes}" href="{_e(seat_link(digest, seat["side"]))}" '
            f'data-play="{_e(digest)}" data-seat="{int(seat["index"])}"{agent}>'
            f'{_e(words.format(name=fixed["name"]))}</a>')


def _ways(story: dict[str, Any] | None = None) -> str:
    door = play_door(story, PLAY_STORY, "button storyWay")
    today = "button todayWay" if door else "button"
    return (f'<a class="button primary" href="#/design">{_e(START)}</a>' + door
            + f'<a class="{today}" href="/game" data-today>{_e(PLAY_TODAY)}</a>')


# Q1056: the landing's switch, a quiet line under the hero's words: which story of how many,
# and a step either way through the pool (`landing.js` `wireStorySwitch`, `story.js`
# `landingStoryTo`), without loading the page again and without keeping the choice.
STORY_SWITCH_LABEL = "which story"
STORY_SWITCH_SAID = "story {at} of {count}"
STORY_PREVIOUS = "previous story"
STORY_NEXT = "next story"


def story_switch(at: int, count: int) -> str:
    """The switch for the story at place `at` (from 0) of a pool of `count`, or nothing when
    there is no other story to switch to."""
    if count < 2:
        return ""
    return (
        f'<nav class="storySwitch small muted" aria-label="{_e(STORY_SWITCH_LABEL)}">'
        f'<button type="button" class="quiet" data-story-step="-1" aria-label="{_e(STORY_PREVIOUS)}"'
        f' title="{_e(STORY_PREVIOUS)}">‹</button>'
        f'<span>{_e(STORY_SWITCH_SAID.format(at=at + 1, count=count))}</span>'
        f'<button type="button" class="quiet" data-story-step="1" aria-label="{_e(STORY_NEXT)}"'
        f' title="{_e(STORY_NEXT)}">›</button></nav>')


def _hero(story: dict[str, Any] | None, switch: str = "") -> str:
    ideas = _ideas(story)
    clauses = "".join(
        f'<span class="clause {idea["hue"]}{" on" if n == 0 else ""}">{_e(idea["clause"])}</span>'
        for n, idea in enumerate(ideas))
    states = "".join(
        f'<div class="heroState {idea["hue"]}{" on" if n == 0 else ""}" data-game="{_e(idea["game"])}"'
        + (f' data-at="{idea["at"]}"' if idea["at"] else "")
        # LP11: the story's own ending names its story, at a turn or at the opening, so its
        # board can start the story's game (`landing.js` `wireHeroBoard`).
        + (f' data-story="{_e(idea["story"])}"' if idea["story"] else "")
        + '><div class="heroBoard" data-board></div>'
        f'<figcaption class="small"><span class="chip familyChip"><span class="swatch" '
        f'aria-hidden="true"></span>{_e(idea["chip"])}</span> '
        + (f'<b>{_e(idea["game"])}</b> <span class="muted">{_e(EXAMPLE_CAPTION)}</span>'
           if idea["example"] else f'<a href="/game" data-name>{_e(idea["game"])}</a>')
        + "</figcaption></div>"
        for n, idea in enumerate(ideas))
    dots = "".join(
        f'<button type="button" class="heroDot{" on" if n == 0 else ""}" '
        f'aria-label="{_e(idea["chip"])}"></button>' for n, idea in enumerate(ideas))
    return (
        '<section class="landingHero heroB">'
        '<div class="landingIntro">'
        f'<h2 class="landingTitle"><span class="wcbi">{_e(QUESTION)}</span> '
        f'<span class="clauses" id="heroClauses" aria-live="polite">{clauses}</span></h2>'
        f'<p class="landingLead">{_e(LEAD)}</p>'
        f'<div class="row landingWays">{_ways(story)}</div>'
        f'<p class="muted small">{_e(NO_ACCOUNT)}</p>'
        + switch
        + "</div>"
        '<figure class="heroShow" id="heroShow">'
        f'<div class="heroStage" id="heroStage">{states}</div>'
        f'<div class="row heroDots" role="group" aria-label="which idea">{dots}'
        '<button type="button" class="quiet" id="heroPause">pause</button></div>'
        "</figure></section>")


def drawn_intro(story: dict[str, Any] | None = None) -> str:
    """The question, its lead and the two ways in, outside the `<template>` (G4).

    The site puts this in `#view` at `/`, so a reader without script, and a crawler, reads
    what the site is: the question with the leading story's ending, written whole. It is the
    start of the hero's own markup, so when `landing.js` clones the template over it the
    headline lands where it already stood.

    LP6: with no `story` named, every holding story's ending is drawn, in the pool's order,
    and the stylesheet shows the one `story.js` drew (`html[data-story-at]`, the first
    without the script, `STORY_SLOTS`), so the headline does not change when the template
    lands."""
    pool = [story] if story is not None else list(_CURRENT["stories"])
    if pool:
        clause = '<span class="clauses drawnClauses">' + "".join(
            f'<span class="clause {one["hue"]} on" data-story="{_e(one["key"])}">'
            f'{_e(one["hero"]["clause"])}</span>' for one in pool) + "</span>"
    else:
        ideas = _ideas(None)
        clause = (f'<span class="clauses"><span class="clause {ideas[0]["hue"]} on">'
                  f'{_e(ideas[0]["clause"])}</span></span>') if ideas else ""
    return (
        '<div class="landing"><section class="landingHero heroB"><div class="landingIntro">'
        f'<h2 class="landingTitle"><span class="wcbi">{_e(QUESTION)}</span> {clause}</h2>'
        f'<p class="landingLead">{_e(LEAD)}</p>'
        f'<div class="row landingWays">{_ways()}</div>'
        f'<p class="muted small">{_e(NO_ACCOUNT)}</p>'
        "</div></section></div>"
    )


# ------------------------------------------------------------------ the journey (LP4)

# §2.1: one idea, start to finish, in six steps, written from a story `web/stories.py`
# rendered: every sentence and number is the record's or its games' own (rule 3, and the
# provisional rule: a story with a number that is not current is not in the pool, so never
# drawn). The games' frames are not here: `landing.js` asks `/api/stories` for them and
# plays them on the steps' boards; the verdicts stay hidden (`pending`) until each game ends.
JOURNEY_TITLE = "One idea, start to finish"
STEP_NAMES = ("an idea, as a rule", "what goes wrong", "the one change", "what it changes",
              "play it against the computer", "challenge a friend")
LADDER_SAID = "Computer opponents, from the gentlest to the hardest."
PLAY_THIS_SAID = "You play {side}, against {against}, in a practice game."
GENTLEST = "the gentlest"
NAMED_OPPONENT = "the computer player {name}"


def play_this_said(story: dict[str, Any]) -> str:
    seat = story.get("seat") or {"side": "white"}
    named = story.get("opponent")
    against = NAMED_OPPONENT.format(name=named["name"]) if named else GENTLEST
    return PLAY_THIS_SAID.format(side=_capital(str(seat["side"])), against=against)


CHANGE_NOTE = "Every save is its own game with its own answer, and undo is one key."
CHALLENGE_SAID = "Send a link. Move when it suits you."
CHALLENGE_LINE = (
    "A challenge link is for one game, taken up by one person. The site keeps the board and "
    "can email whoever is to move; a game is lost by the player to move after "
    f"{CORRESPONDENCE_IDLE_SAID}.")
CHALLENGE_ASK = "Members, by invitation for now."


def _linked(text: str, names: dict[str, str]) -> str:
    """`text` escaped, each game name in `names` made a link to the game's own page."""
    said = _e(text)
    for name, digest in names.items():
        if name:
            said = said.replace(_e(name), f'<a href="/game/{_e(digest)}">{_e(name)}</a>', 1)
    return said


def _verdict(side: dict[str, Any], good: bool, quote: str = "") -> str:
    """A game's answer, in the site's words with its basis, or a constructed one's example
    line and the designer's quick answers."""
    if side.get("example"):
        quick = side.get("quick") or {}
        said = "".join(f"<li>{_e(line)}</li>" for line in quick.get("said") or [])
        return (f'<p class="small exampleNote">{_e(side["name"])}: {_e(side["line"])}</p>'
                f'<ul class="small quickSaid">{said}</ul>'
                f'<p class="muted small">{_e(_capital(str(quick.get("when", ""))))}</p>')
    name = f'<a href="/game/{_e(str(side["hash"]))}">{_e(side["name"])}</a>'
    words = _failings(list(side.get("words") or []))
    mark = ("verdict-word-good", "it stays in") if good else ("verdict-word-bad", "ruled out")
    out = (f'<p class="small">The site\'s answer for {name}, measured in minutes: '
           f'<b class="{mark[0]}">{mark[1]}</b>.'
           + (f' <span class="muted">{_e(words)}.</span>' if words and not good else "")
           + f'</p><p class="muted small"><span class="mono" data-line>{_e(side["line"])}</span> '
           f'· <a href="{SCORE_ROUTE}">{_e(SCORE_TITLE.lower())}</a></p>')
    if good and quote:
        out += f'<p class="small muted"><q>{_e(quote)}</q></p>'
    if good and side.get("lean"):
        out += f'<p class="small muted">The site adds: {_e(side["lean"])}.</p>'
    return out


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _failings(words: list[str]) -> str:
    """The site's words for the tests a game failed, one a run (`run 1 of 4: ...`): each test
    in the order first met, the first run's words whole, and where every run failed that
    test, said once more for all of them (LP5: a game can fail several tests in every run)."""
    if not words:
        return ""
    runs = {said.split(": ", 1)[0] for said in words if said.startswith("run ")}
    groups: dict[str, list[str]] = {}
    for said in words:
        groups.setdefault(said.split(": ", 1)[-1].split(",")[0], []).append(said)
    out: list[str] = []
    for group in groups.values():
        if len(runs) > 1 and len(group) == len(runs):
            out.append(f"{group[0]}; the same in all {len(group)} runs")
        else:
            out += group
    return "; ".join(out)


def _quick(side: dict[str, Any]) -> str:
    """The designer's quick answers on the failed game (a library story's step 2)."""
    quick = side.get("quick")
    if not quick or side.get("example"):
        return ""
    lines = "".join(
        f'<p class="out small{"" if said.startswith("✗") else " good"}">{_e(said)}</p>'
        for said in quick.get("said") or [])
    when = str(quick.get("when", ""))
    return (f'<p class="muted small">{_e(when[:1].upper() + when[1:])}:</p>{lines}'
            '<p class="muted small">A first answer in a second; a fuller one in minutes.</p>')


def _bars(failed: dict[str, Any], fixed: dict[str, Any]) -> str:
    """The six qualities, the failed game's faint and the fixed one's solid on one bar each;
    a constructed story has none (§2.1). The bars' lengths are set by `landing.js` from
    `data-v`, never by a style attribute."""
    if failed.get("example") or fixed.get("example"):
        return ""
    rows = []
    for name in _BAR_ORDER:
        a, b = failed["qualities"].get(name), fixed["qualities"].get(name)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        rows.append(
            f'<span>{_e(name)}</span><span class="pairBar"><b data-v="{a:.4f}"></b>'
            f'<i data-v="{b:.4f}"></i></span><span class="mono small">{a:.2f} → {b:.2f}</span>')
    if not rows:
        return ""
    return (f'<div class="pairBars" aria-label="{_e(failed["name"])} faint, {_e(fixed["name"])} '
            f'solid">{"".join(rows)}</div>'
            f'<p class="small muted pairKey"><span class="key faint"></span>{_e(failed["name"])} '
            f'<span class="key"></span>{_e(fixed["name"])}</p>')


def _play(which: str, beat: dict[str, Any], verdict: str) -> str:
    return (
        f'<p class="muted">{_e(beat["lead"])}</p>'
        f'<div class="beatBoard" data-board="{which}"></div>'
        f'<div class="beatTray row" data-tray="{which}"></div>'
        f'<p class="small mono playWords" data-words="{which}"></p>'
        f'<div class="row playCtl" data-ctl="{which}">'
        '<button type="button" class="quiet" data-step="back" aria-label="one step back">‹</button>'
        '<button type="button" class="quiet" data-step="play">play</button>'
        '<button type="button" class="quiet" data-step="next" aria-label="one step on">›</button>'
        "</div>"
        f'<div class="verdictBlock pending" data-answer="{which}">'
        f'<p class="small"><b>{_e(beat["end"])}</b></p>{verdict}'
        f'<p class="muted small">The board is {_e(beat["source"])}.</p></div>')


def _step(n: int, body: str, classes: str = "") -> str:
    more = f" {classes}" if classes else ""
    return (f'<li class="beat{more}" data-beat="{n}">'
            f'<p class="beatNo mono">{n} · {_e(STEP_NAMES[n - 1])}</p>{body}</li>')


_BAR_ORDER = ("depth", "drama", "decisions", "pacing", "variety", "balance")


OTHERS_SAID = "More ideas, start to finish:"


def _others(story: dict[str, Any], others: list[dict[str, Any]]) -> str:
    """The journey's foot: a chip per other holding story, each opening the page on it."""
    chips = "".join(
        f'<a class="chip familyChip {other["hue"]}" href="/?story={_e(other["key"])}" '
        f'title="{_e(other["family"])}"><span class="swatch" aria-hidden="true"></span>'
        f'{_e(other["title"])}{", an example" if other["example"] else ""}</a>'
        for other in others if other["key"] != story["key"])
    if not chips:
        return ""
    return (f'<nav class="otherStories" aria-label="other stories"><p class="muted small">'
            f'{_e(OTHERS_SAID)}</p><div class="row">{chips}</div></nav>')


def story_html(story: dict[str, Any], others: list[dict[str, Any]] | None = None) -> str:
    """One story's journey, the six steps of §2.1, from `stories.render`'s record, and a
    chip for each of the `others`."""
    failed, fixed, card = story["failed"], story["fixed"], story["card"]
    names = {failed["name"]: str(failed["hash"])} if not story["example"] else {}
    hue = story["hue"]
    before = card.get("before") or ""
    lines = story["change"]["lines"]
    diff = "  ".join(
        (f'<del>{_e(line["del"])}</del>' if line["del"] else "")
        + (" " if line["del"] and line["ins"] else "")
        + (f'<ins>{_e(line["ins"])}</ins>' if line["ins"] else "") for line in lines)
    steps = [
        _step(1, f'<h3>{_e(story["idea"])}</h3><p class="muted">{_linked(story["note"], names)}</p>'
                 f'<div class="ruleCard {hue}"><span class="swatch" aria-hidden="true"></span>'
                 f'<div><b>{_e(card["title"])}</b><span class="small muted">{_e(card["lead"])} '
                 f'{_e(card["was"])}{_e(card["tail"])}</span></div></div>'
                 f'<p class="storyDoor small"><a href="{_e(door_link(card["door"]))}">'
                 f'{_e(DOOR_SAID)}</a></p>'
                 + (f'<pre class="ruleCode"><code>{_e(before)}</code></pre>' if before else "")),
        _step(2, f'<h3 class="say bad">{_e(story["wrong"]["say"])}</h3>'
                 + _play("wrong", story["wrong"], _quick(failed) + _verdict(failed, False))),
        _step(3, f'<h3 class="say change">{_e(story["change"]["say"])}</h3>'
                 f'<div class="ruleCard edit {hue}"><span class="swatch" aria-hidden="true"></span>'
                 f'<div><b>{_e(card["title"])}</b><span class="small">{_e(card["lead"])} '
                 f'<del>{_e(card["was"])}</del> <ins>{_e(card["now"])}</ins>{_e(card["tail"])}'
                 f'</span></div></div><pre class="ruleCode diff"><code>{diff}</code></pre>'
                 f'<p class="muted small">{_e(CHANGE_NOTE)}</p>', "change"),
        _step(4, f'<h3 class="say good">{_e(story["right"]["say"])}</h3>'
                 + _play("right", story["right"],
                         _verdict(fixed, True, story.get("quote") or "")
                         + _bars(failed, fixed))),
        _step(5, f'<h3>{_e(LADDER_SAID)}</h3>'
                 # An example is no library game, so it has no page to read a ladder from.
                 + ('' if story["example"] else
                    f'<div class="ladder" data-ladder="{_e(str(fixed["hash"]))}"></div>')
                 + '<p class="small muted">'
                 + ("Every draft can be played at once, an example like this one too. "
                    if story["example"] else
                    f'Each one on <a href="/game/{_e(str(fixed["hash"]))}">{_e(fixed["name"])}</a>, '
                    "as its page offers them. ")
                 + "A practice game has undo and restart and needs no account; a rated game "
                 "needs one.</p>"
                 # LP9: one plain way in, in the story's seat, against the gentlest or the story's own
                 # opponent (LP11).
                 + (f'<div class="row playThis">{play_door(story, PLAY_THIS)}'
                    f'<span class="small muted">{_e(play_this_said(story))}</span></div>'
                    if not story["example"] else "")),
        _step(6, f'<h3>{_e(CHALLENGE_SAID)}</h3>'
                 '<p class="challengeLink small"><span class="muted">the site\'s address</span> '
                 '<span class="mono">/c/…</span></p>'
                 f'<p class="small">{_e(CHALLENGE_LINE)}</p>'
                 f'<p class="small muted" data-needs="accounts">{_e(CHALLENGE_ASK)} '
                 '<a href="/request-invitation">Ask for one</a>.</p>'),
    ]
    return (
        f'<section class="journey {hue}" id="journey" data-story="{_e(story["key"])}">'
        f'<h2 class="journeyTitle">{_e(JOURNEY_TITLE)}</h2>'
        '<div class="journeyGrid">'
        '<figure class="pin" aria-hidden="true"><div class="pinStatus mono small" id="pinWords">'
        '</div><div class="pinBoard" id="pinBoard"></div><div class="pinTray row" id="pinTray">'
        '</div><div class="pinVerdict mono small" id="pinVerdict"></div></figure>'
        f'<ol class="beats" id="beats">{"".join(steps)}</ol></div>'
        + _others(story, others or []) + '</section>')


def html_page(version: str = DEFAULT_WEIGHTS,
              stories: list[dict[str, Any]] | None = None) -> str:
    """The landing page's markup: for every holding story, in the pool's order, its hero and
    its journey as one part (`storyPart`, LP5), then the wall of words and the contact line.
    `landing.js` keeps one part when it clones the template (`?story=<key>`, else the first;
    LP6 draws one at random) and drops the others, so each part's ids stand once on the page.
    With no story holding, the journey is left out and the hero still turns."""
    count = len(stories or [])
    parts = "".join(
        f'<div class="storyPart" data-story="{_e(story["key"])}">'
        + _hero(story, story_switch(at, count)) + story_html(story, stories) + "</div>"
        for at, story in enumerate(stories or [])) or _hero(None)
    pool = " ".join(story["key"] for story in stories or [])
    return (
        f'<div class="landing landingB" id="landing" data-stories="{_e(pool)}">'
        + parts
        + _section("palette", WALL_HEADING, _wall_html(), classes="landingWall")
        + f'<div class="landingFoot" id="{CONTACT_ID}"><p>{CONTACT_LINE}</p></div>'
        + "</div>")


# The page's markup, drawn from the stories the server last rendered (`use`): the page package
# draws `current_html()` into every page's `<template id="landingPage">`. Before any server
# has rendered its stories (a test's page, a server not yet warm) the landing has no journey.
_CURRENT: dict[str, Any] = {"stories": [], "html": ""}


def use(stories: list[dict[str, Any]]) -> str:
    """Draw the landing from `stories` (`stories.render`'s holding ones), and keep it as the
    one every page carries from now on."""
    _CURRENT["stories"] = list(stories)
    _CURRENT["html"] = html_page(stories=list(stories))
    return _CURRENT["html"]


def current_html() -> str:
    if not _CURRENT["html"]:
        _CURRENT["html"] = html_page(stories=_CURRENT["stories"])
    return _CURRENT["html"]


def current_stories() -> list[dict[str, Any]]:
    return list(_CURRENT["stories"])


# The page's markup, made once: the weights file does not change under a running server.
SCORE_HTML = score_html()
SHELVES_HTML = shelves_html()
