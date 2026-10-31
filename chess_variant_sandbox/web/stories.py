"""The landing's stories as data (`docs/archive/plan-landing.md` §2.1, §2.2, box LP1).

Each story is an idea that fails and the one change that mends it, told in three beats a
visitor can follow without reading a number: **what goes wrong** (a game of the idea played
until the failure shows on the board), **the one change** (the designer's card with words
struck and replaced, and the rule language's diff), **what it changes** (a game of the
changed rule played until that shows too), then the verdict.

`STORIES` is the record, in §2.1's order, and it holds **no measured number**: every score,
interval, run count, quality and verdict sentence is read at render from the game's own
`runs/<hash>/canonical.json` through `library.Game` and `sitescore.worded`, with its basis.
The sentences that name a fact of a recorded game (a turn, a square, a count, who won) are
templates (`{turns}`, `{moment_to}`, `{a}` ...) filled from that game's replay, so a sentence
cannot drift from the board it describes. The numbers left in a record are its parameters:
the seed and line that name a recorded game, the turns it shows, its pace and its moment, and
the change itself (a rule value).

Two kinds of story. A **library** story is a real pair of library games and keeps its real
numbers. A **constructed** story (the user, 2026-10-24) is a pair made for the page: it says
it is an example, carries no score and no basis, and quotes only the designer's own quick
answers, computed here on its specs (`designing.check`, `candidacy.screen_draft`). Its specs
and played games are under `web/stories/`.

**What the board shows is the engine's** (rule 3). Every game is replayed move by move through
`Engine` here; each frame carries the squares that changed and what the move *did*, read off
the engine: `from`/`to` (`decode_move`), a portal's two ends (`viz.wrapped.portal_transit`), a
blast's centre, area (the compiled `explosion.blast` table) and the squares it cleared (the two
positions), a drop's square, the claims that changed and both counts, the hands, and whether
it is a key frame (§2.2). A library game's moves are copied from its run's `games.jsonl` into
`web/stories/recorded.json` by `web/stories/copy_records.py`, because the site is never sent a
`runs/<hash>-<seed>/` game directory (`deploy/copy-runs.sh`); `tests/test_stories.py` holds the
copy to the record wherever the run is on disk.

**A story that stops reading as told leaves the pool** (§4.1). A library story *holds* when
both games are listed library games, their mechanics differ by exactly the declared change,
the failed game's estimate is current and ruled out, the fixed one's current and staying in,
and both recorded games replay to the ending their sentences claim; a constructed story when
both specs validate, differ by exactly the change and their games replay to their endings. A
provisional number never enters the pool. `render` returns the holding stories and names each
dropped one with why; nothing draws a dropped story.

    uv run python -m chess_variant_sandbox.web.stories [--runs RUNS]

prints the four stories as `render` reads them.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_DROP, MOVE_PROMOTION, decode_move
from chess_variant_sandbox.spec.canonical import MECHANIC_KEYS, load_spec, spec_hash
from chess_variant_sandbox.viz.wrapped import (
    portal_transit,
    push_transit,
    square_name,
    sweep_transit,
)
from chess_variant_sandbox.web.landing import BLOCKS

HERE = Path(__file__).resolve().parent / "stories"
EXAMPLES_FILE = HERE / "examples.json"
RECORDED_FILE = HERE / "recorded.json"

LIBRARY = "library"
CONSTRUCTED = "constructed"

# The six qualities, in the order the story's paired bars list them (§2.1).
QUALITIES = ("depth", "drama", "decisions", "pacing", "variety", "balance")

# The palette's families and their colour classes: the wall of words' rows (LP2,
# `web/landing.py`'s `BLOCKS`), the one list a story's family must be one of.
FAMILIES: dict[str, str] = {block.family: block.hue for block in BLOCKS}

ABSENT = object()  # a key the change adds or removes

# Small counts as a sentence spells them, as `sitescore` does.
_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve")


def _word(count: int) -> str:
    return _WORDS[count] if 0 <= count < len(_WORDS) else str(count)


class StoryError(Exception):
    """Why a story does not hold, in a sentence."""


# ------------------------------------------------------------------ the record


@dataclass(frozen=True)
class Played:
    """Where one beat's game is, and how the page plays it (§2.2)."""

    hash: str
    # The ending the beat's sentences claim, the engine's own words: (winner, reason),
    # winner 0 for the first player, 1 for the second, None for a draw.
    ending: tuple[int | None, str]
    # A library game: its run's seed and its line (0-based) in that run's games.jsonl.
    seed: int | None = None
    line: int | None = None
    # A constructed game: its key in `web/stories/examples.json`.
    example: str | None = None
    # Which turns the page shows: None for every one, else inclusive ranges; a jump between
    # two ranges is a cut (§2.2, the queen bomb's turns 14 to 62).
    show: tuple[tuple[int, int], ...] | None = None
    # Key frames under reduced motion: "events" (each portal and blast), "drops" (each
    # stone placed), or every n-th turn; the moment and the last are always key.
    keys: str | int = "events"
    # Milliseconds a plain turn takes on the page.
    pace: int = 400
    # The turn the rule acts and the page opens on under reduced motion; None for the last.
    moment: int | None = None

    @property
    def record_key(self) -> str:
        """The game's key in `recorded.json` (library) or `examples.json` (constructed)."""
        if self.example is not None:
            return self.example
        return f"{self.hash}-{self.seed}/{self.line}"


@dataclass(frozen=True)
class Beat:
    """*What goes wrong* or *what it changes*: one sentence, and a game that shows it."""

    say: str  # the beat's heading
    lead: str  # what to watch
    end: str  # said when the game ends
    source: str  # which game this is
    game: Played


@dataclass(frozen=True)
class Card:
    """The designer's card for the rule, with the words the change strikes and adds."""

    title: str
    lead: str
    was: str
    now: str
    tail: str
    # Where the rule sits in a spec, so its line of the language is read off the spec.
    rule: tuple[str | int, ...]
    # The rule's key as the wall of words spells it (`landing.Word.key`): the card's door
    # into the designer, *start a game with this rule* (LP7, `#/design?rule=<key>`).
    door: str


@dataclass(frozen=True)
class Change:
    """One edit at one path of the mechanics; ABSENT where the key is added or removed."""

    path: tuple[str | int, ...]
    was: Any = ABSENT
    now: Any = ABSENT


@dataclass(frozen=True)
class Story:
    key: str
    title: str
    family: str
    kind: str  # LIBRARY or CONSTRUCTED
    # The hero's clause (*wouldn't it be cool if...*), its chip, and the turn of the fixed
    # game its board shows (None: the opening).
    clause: str
    chip: str
    idea: str
    note: str
    card: Card
    tweak: str  # the one change, in one sentence
    changes: tuple[Change, ...]
    wrong: Beat
    right: Beat
    hero_at: int | None = None
    # Words of the fixed game's own description the story quotes, if any (without the
    # description's punctuation after them).
    quote: str = ""
    # A constructed story's two spec files under `web/stories/`.
    files: tuple[str, str] | None = None
    # The seat the story is about, 0 the first player and 1 the second (LP9): its play doors
    # seat the visitor there (`?seat=<side>` on the fixed game's page), so the door story's
    # visitor plays Black, whose door it now is.
    seat: int = 0
    # The computer player its play doors seat against, by its agent string (LP11,
    # `landing-cold-read.md` §6.2 item 4), as `seat` names the visitor's side; empty for the
    # gentlest the fixed game's page offers (LP9's rule, the default). Blast's gentlest falls
    # to a three-move queen trap, which contradicts its own last beat, *the game goes on*.
    opponent: str = ""


STORIES: tuple[Story, ...] = (
    Story(
        key="stones", title="stones", family="pieces in hand", kind=LIBRARY,
        clause="you could place a stone instead of moving?",
        chip="pieces in hand: stones",
        idea=("Wouldn't it be cool if either player could place a stone on any empty square "
              "instead of moving?"),
        note=("A stone never moves, is never captured and never leaves: a wall that blocks "
              "lines for both sides. That is {failed}, a game in the library."),
        card=Card(title="pieces played back onto the board", lead="each player holds a stone ×",
                  was="as many as they like", now="three", tail="", rule=("rules", "drops"),
                  door="rules.drops"),
        tweak="Three stones each, not as many as they like.",
        changes=(Change(("rules", "drops", "stock", "X"), "unlimited", 3),),
        wrong=Beat(
            say=("With as many stones as they like, both players wall the board up until "
                 "nobody can move."),
            lead="The two strongest computer players play it. Watch the empty squares.",
            end=("Turn {turns}: every square is filled and the player to move has no move. "
                 "A draw."),
            source=("a recorded game between the two strongest computer players, {turns} "
                    "turns (seed {seed}, game {game} of {of})"),
            game=Played(hash="93b02b591c87", seed=0, line=222, ending=(None, "stalemate"),
                        keys=12, pace=110)),
        right=Beat(
            say=("Three stones are a choice of where to spend them. Once they are down it is "
                 "chess again, and somebody wins."),
            lead="The same two players, three stones each. Watch the stones in hand.",
            end=("Turn {turns}: checkmate, with all six stones on the board. {winner} "
                 "wins."),
            source=("a recorded game between the two strongest computer players, {turns} "
                    "turns (seed {seed}, game {game} of {of})"),
            game=Played(hash="03aa7c3f2f18", seed=1000, line=252, ending=(1, "checkmate"),
                        keys="drops", pace=120)),
        quote="The limit is what makes the idea work",
    ),
    Story(
        key="door", title="a door", family="the board", kind=LIBRARY,
        clause=("a door through the middle of the board let your pieces strike the enemy "
                "camp?"),
        chip="portals",
        idea=("Wouldn't it be cool if a door through the middle of the board let either "
              "player's pieces step in on one side and out on the other?"),
        note=("On Gardner's five-by-five minichess, a pair of portals on {ends}, open to both "
              "sides: a piece that ends its move on one end comes out one step beyond the "
              "other, capturing there as usual. That is {failed}, a game in the library."),
        card=Card(title="portals", lead="a pair on {ends}, open to", was="both players",
                  now="Black only", tail="", rule=("rules", "portals"),
                  door="rules.portals"),
        tweak="Shut the door to the player who moves first.",
        changes=(Change(("rules", "portals", 0, "passage"), now={"0": "closed"}),),
        wrong=Beat(
            say="The first player always gets through the door first, and the first player wins.",
            lead=("The hardest computer player plays it against itself. Watch the first "
                  "move."),
            end=("Turn {turns}: checkmate. White's pawn was through the door on turn "
                 "{moment} and a queen by turn {first_promotion}."),
            source=("a recorded game between the hardest computer player and itself, {turns} "
                    "turns (seed {seed}, game {game} of {of})"),
            game=Played(hash="3732e55621b1", seed=0, line=252, ending=(0, "checkmate"),
                        keys="events", pace=420, moment=1)),
        right=Beat(
            say=("Now the door is Black's answer to moving second, and the first move no "
                 "longer decides the game."),
            lead=("The same player against itself. White's pieces meet the door as a wall; "
                  "watch Black's rook."),
            end=("Turn {moment}: Black's rook goes in on {portal_in} and comes out on "
                 "{moment_to}. Turn {turns}: checkmate. {winner} wins."),
            source=("a recorded game between the hardest computer player and itself, {turns} "
                    "turns (seed {seed}, game {game} of {of})"),
            game=Played(hash="58afebca2286", seed=0, line=252, ending=(1, "checkmate"),
                        keys="events", pace=380, moment=28)),
        quote="Shut to White, the race is Black's, and the door proves worth more than the tempo",
        seat=1,
    ),
    Story(
        key="blast", title="a blast", family="what a capture does", kind=LIBRARY,
        clause="the queen went off like a bomb when she struck?",
        chip="captures that explode",
        idea=("Wouldn't it be cool if taking a queen set off an explosion that destroyed "
              "whatever took her, and everything but a pawn around her?"),
        note=("It is atomic chess's explosion, kept for one capture alone, and a king caught "
              "in the blast is lost, and with it the game. That is {failed}, a game in the "
              "library."),
        card=Card(title="captures that explode", lead="when a queen", was="is taken",
                  now="takes", tail="; pawns are immune",
                  rule=("rules", "explode_on_capture"), door="rules.explode_on_capture"),
        tweak="The queen explodes when she takes, not when she is taken.",
        changes=(Change(("rules", "explode_on_capture", "of"), was=["Q"]),
                 Change(("rules", "explode_on_capture", "by"), now=["Q"])),
        wrong=Beat(
            say=("Each king starts beside its queen, so the first queen taken blows up a king, "
                 "and the game is over in a few moves."),
            lead="The hardest computer player against a weak one. Watch {moment_to}.",
            end=("Turn {moment}: White's queen takes Black's on {moment_to}. The blast takes "
                 "the king beside her, and the game."),
            source=("a recorded game between the hardest computer player and a weak one, "
                    "{turns} turns (seed {seed}, game {game} of {of})"),
            game=Played(hash="328cab4db5cc", seed=0, line=232, ending=(0, "royal_destroyed"),
                        keys="events", pace=650, moment=9)),
        right=Beat(
            say=("Now the queen is a bomb her owner aims: she blows a hole where she strikes "
                 "and goes with it, and the game goes on."),
            lead="The hardest computer player against itself. Watch {moment_to}.",
            end=("Turn {moment}: White's queen takes the pawn on {moment_to} and goes off, "
                 "and a rook, a knight and a bishop go with her; the pawns beside her stand. "
                 "The game goes on to turn {turns}: checkmate, {winner} wins."),
            source=("a recorded game between the hardest computer player and itself, {turns} "
                    "turns (seed {seed}, game {game} of {of})"),
            game=Played(hash="313c913a5b3b", seed=1000, line=253, ending=(0, "checkmate"),
                        show=((1, 13), (63, 63)), keys="events", pace=520, moment=13)),
        hero_at=13,
        quote="So the queen becomes a one-shot weapon",
        opponent="minimax:nodes=400",
    ),
    Story(
        key="sweep", title="a sweep", family="how pieces move", kind=LIBRARY,
        clause="your bishop swept its diagonal clear, taking everything it passed over?",
        chip="pieces that sweep",
        idea=("Wouldn't it be cool if a bishop swept down its diagonal, removing every piece "
              "it passed over, its own included?"),
        note=("Nothing stops it: it sweeps any distance, through anything in its way. That is "
              "{failed}, a game in the library."),
        card=Card(title="sweep a line clear", lead="the bishop sweeps", was="any distance",
                  now="up to three squares", tail="", rule=("pieces", 2, "moves"),
                  door="sweep"),
        tweak="Three squares at most, not any distance.",
        changes=(Change(("pieces", 2, "moves", 0, "range"), now=3),),
        wrong=Beat(
            say=("The bishop sweeps any distance, through anything, so White's first move is "
                 "checkmate."),
            lead=("The hardest computer player plays it against itself. Watch the first "
                  "move."),
            end=("Turn {moment}: White's bishop sweeps {moment_swept} off the board and lands "
                 "on {moment_to}. Nothing can block a sweep, and the king has no square: "
                 "checkmate, {winner} wins."),
            source=("a recorded game between the hardest computer player and itself, {turns} "
                    "turn{turns_s} (seed {seed}, game {game} of {of})"),
            game=Played(hash="01cdfe2e679b", seed=0, line=252, ending=(0, "checkmate"),
                        keys="events", pace=600, moment=1)),
        right=Beat(
            say=("Three squares is a door that closes: the sweep stays, but a sweeping check "
                 "has to be set up, and can be seen coming."),
            lead="The same player against itself. Watch the bishops.",
            end=("Turn {moment}: Black's bishop sweeps {moment_swept} in one move and takes on "
                 "{moment_to}. Turn {turns}: checkmate, {winner} wins."),
            source=("a recorded game between the hardest computer player and itself, {turns} "
                    "turn{turns_s} (seed {seed}, game {game} of {of})"),
            game=Played(hash="ceec60c5fbbc", seed=3000, line=255, ending=(0, "checkmate"),
                        keys="events", pace=360, moment=24)),
        quote="a shield worth keeping, or a door worth burning",
    ),
)

# Constructed examples out of the pool (LPD, 2026-10-26: *a sweep*, a measured pair, took
# ground's place). Kept, with their files under `web/stories/`, so the constructed kind stays
# tested; `render` draws only `STORIES`.
GROUND = Story(
    key="ground", title="ground", family="how a game is won", kind=CONSTRUCTED,
    clause=("every square your knight landed on became yours, and {goal} of them won?"),
    chip="claiming ground",
    idea=("Wouldn't it be cool if every square your knight landed on became yours, and "
          "holding {goal} of the {squares} won?"),
    note=("Five knights a side on a seven-by-seven board; taking every enemy knight wins "
          "too. An example made for this page, not a library game: {failed}."),
    card=Card(title="claiming ground", lead="a claimed square", was="can be painted back",
              now="is kept for good", tail="; {goal} of the {squares} wins",
              rule=("rules", "territory"), door="rules.territory"),
    tweak="A claim is for good.",
    changes=(Change(("rules", "territory", "mode"), "repaint", "permanent"),),
    wrong=Beat(
        say=("Painting over the other side's square is worth two and an empty one only "
             "one, so nobody claims empty ground and nobody reaches {goal}."),
        lead="Two computer players ({nodes} positions each) play it. Watch the count.",
        end=("The count never gets past {best}, short of {goal}: what one side paints, the "
             "other paints back. Turn {turns}: {a} squares to {b}, and the moves run out: "
             "a draw."),
        source=("a game played for this page by two {nodes}-position computer players "
                "(seed {seed}); {typical} played ran out of moves like this one"),
        game=Played(hash="0247e54a9305", example="ground-repaint",
                    ending=(None, "ply_limit"), keys=20, pace=70)),
    right=Beat(
        say=("Every claim is now a step towards {goal} that cannot be undone, so the empty "
             "squares run out and somebody gets there."),
        lead="The same two players. Watch the count.",
        end="Turn {turns}: {winner} claims its {goal}th square and wins, {a} to {b}.",
        source=("a game played for this page by the same two players (seed {seed}); "
                "{typical} played finished by the count"),
        game=Played(hash="f7eafa39e370", example="ground-for-good",
                    ending=(0, "coloured"), keys=10, pace=130)),
    hero_at=44,
    files=("ground-repaint.json", "ground-for-good.json"),
)

EXAMPLES: tuple[Story, ...] = (GROUND,)

BY_KEY = {story.key: story for story in (*STORIES, *EXAMPLES)}


# ------------------------------------------------------------------ the change


def mechanics(spec: dict[str, Any]) -> dict[str, Any]:
    """The part of a spec its hash names (`canonical.MECHANIC_KEYS`)."""
    return {key: copy.deepcopy(spec[key]) for key in MECHANIC_KEYS if key in spec}


def apply_changes(spec: dict[str, Any], changes: tuple[Change, ...]) -> dict[str, Any]:
    """The failed spec's mechanics with the story's declared change made.

    Refuses a change whose `was` is not what the spec holds, so a declared change that no
    longer describes the failed game cannot be applied over it."""
    edited = mechanics(spec)
    for change in changes:
        node: Any = edited
        for step in change.path[:-1]:
            node = node[step]
        last = change.path[-1]
        held = node.get(last, ABSENT) if isinstance(node, dict) else node[last]
        if held is not change.was and held != change.was:
            raise StoryError(f"the change at {'.'.join(map(str, change.path))} does not "
                             "start from what the failed game holds")
        if change.now is ABSENT:
            del node[last]
        else:
            node[last] = copy.deepcopy(change.now)
    return edited


def differ_by_change(failed: dict[str, Any], fixed: dict[str, Any],
                     changes: tuple[Change, ...]) -> bool:
    """Whether the two specs' mechanics differ by exactly the declared change."""
    try:
        edited = apply_changes(failed, changes)
    except (StoryError, KeyError, IndexError, TypeError):
        return False
    return json.dumps(edited, sort_keys=True) == json.dumps(mechanics(fixed), sort_keys=True)


def _language(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def change_lines(changes: tuple[Change, ...]) -> list[dict[str, str]]:
    """The change as the rule language's diff, one line per edit: `del` and `ins` as
    `"key": value`, read off the declared change (never typed as text)."""
    lines = []
    for change in changes:
        key = _language(str(change.path[-1]))
        lines.append({
            "del": "" if change.was is ABSENT else f"{key}: {_language(change.was)}",
            "ins": "" if change.now is ABSENT else f"{key}: {_language(change.now)}",
        })
    return lines


def rule_line(spec: dict[str, Any], path: tuple[str | int, ...]) -> str | None:
    """The rule's line of the language as `spec` writes it, or None where it has none."""
    node: Any = spec
    for step in path:
        held = ((isinstance(node, dict) and step in node)
                or (isinstance(node, list) and isinstance(step, int) and 0 <= step < len(node)))
        if not held:
            return None
        node = node[step]
    return f"{_language(str(path[-1]))}: {_language(node)}"


# ------------------------------------------------------------------ the games


@lru_cache(maxsize=1)
def _recorded() -> dict[str, Any]:
    return dict(json.loads(RECORDED_FILE.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def _examples() -> dict[str, Any]:
    return dict(json.loads(EXAMPLES_FILE.read_text(encoding="utf-8")))


def game_record(played: Played) -> dict[str, Any]:
    """The recorded game a beat plays: its players, moves and ending as stored."""
    store = _examples() if played.example is not None else _recorded()
    record = store.get(played.record_key)
    if record is None:
        raise StoryError(f"no recorded game {played.record_key}")
    if record.get("hash") != played.hash:
        raise StoryError(f"the recorded game {played.record_key} is another game's")
    return dict(record)


def _shown(played: Played, plies: int) -> list[int]:
    if played.show is None:
        return list(range(1, plies + 1))
    return [n for first, last in played.show for n in range(first, last + 1) if n <= plies]


def _claims(colour: bytes) -> dict[int, int]:
    return {square: owner for square, owner in enumerate(colour) if owner}


def _did(engine: Engine, state: Any, after: Any, move: int) -> dict[str, Any]:
    """What the move did, read off the engine and the two positions it produced."""
    c = engine.spec
    origin, target, kind, promotion = decode_move(move)
    item: dict[str, Any] = {"to": target}
    if kind == MOVE_DROP:
        item["drop"] = target
    else:
        item["from"] = origin
    # A drop's move carries the dropped type where a promotion carries the new one, so a
    # drop is never a promotion.
    if kind != MOVE_DROP and (kind == MOVE_PROMOTION or promotion):
        item["promotion"] = True
    trip = portal_transit(engine, state, move)
    if trip:
        item["portal"] = [trip["portal"], trip["portalOut"]]
    # The squares the sweep cleared besides its landing square, in path order: the play
    # board's own list (`viz.wrapped.sweep_transit`), so the story cuts the same pieces down.
    cleared = sweep_transit(engine, state, move)
    if cleared:
        item["sweep"] = cleared["swept"]
    # Q1074bo: a push's line, the play board's own (`viz.wrapped.push_transit`), so the story
    # slides the line, and drops the piece thrown off or crushes the one pressed on a wall,
    # rather than drawing the piece landed on as taken.
    shoved = push_transit(engine, state, move)
    if shoved:
        item["push"] = shoved
    explosion = c.explosion
    if (explosion is not None and kind != MOVE_DROP and state.board[target]
            and not after.board[target]):
        item["blast"] = {
            "at": target,
            "area": [target, *explosion.blast[target]],
            "cleared": [i for i in range(len(state.board))
                        if state.board[i] and not after.board[i] and i != origin],
        }
    return item


def _push_removes(item: dict[str, Any]) -> bool:
    """Whether the move pushed a piece off the board or crushed it on a wall (Q1074bo): the
    push's event, as a sweep's is; a push that only shoves is a plain turn."""
    push = item.get("push")
    return bool(push) and push["pushed"][-1][1] is None


def _key(played: Played, n: int, last: int, item: dict[str, Any]) -> bool:
    keys = played.keys
    return bool(
        n == last
        or n == (played.moment or last)
        or (keys == "events" and ("portal" in item or "blast" in item or "sweep" in item
                                  or _push_removes(item)))
        or (keys == "drops" and "drop" in item)
        or (isinstance(keys, int) and n % keys == 0))


def _ends(engine: Engine, state: Any) -> list[int]:
    """Where the portal ends stand at the start, the compiled end types found on the board."""
    c = engine.spec
    if c.portals is None:
        return []
    wanted = {c.encode_piece(WILD, t) for t in c.portals.end_types}
    return [i for i, v in enumerate(state.board) if v in wanted]


def _nodes(agents: list[str]) -> str:
    for agent in agents:
        _, _, budget = str(agent).partition("nodes=")
        if budget.isdigit():
            return f"{int(budget):,}"
    return ""


def replay(spec: dict[str, Any], played: Played) -> dict[str, Any]:
    """One beat's game replayed through `Engine`: its frames and the facts its sentences use.

    Every move is checked legal where it is played and the last position is asked for its
    ending, so a recorded game the engine no longer plays the same way is refused rather
    than drawn."""
    from chess_variant_sandbox.web import card_rules

    record = game_record(played)
    engine = Engine(spec)
    c = engine.spec
    state = engine.initial_state()
    shape, art = card_rules.still(spec, lambda: engine)
    start = json.loads(shape)
    if c.drops is not None:
        start["unlimited"] = list(c.drops.unlimited)
    if state.colour:
        start["claims"] = sorted(_claims(state.colour).items())
        start["counts"] = [state.colour.count(1), state.colour.count(2)]
    ends = _ends(engine, state)
    moves = [int(m) for m in record["moves"]]
    wanted = _shown(played, len(moves))
    want = set(wanted)
    last = wanted[-1] if wanted else 0
    shown_board = list(state.board)
    owned = _claims(state.colour) if state.colour else {}
    frames: list[dict[str, Any]] = []
    previous = 0
    best = 0
    for n, move in enumerate(moves, 1):
        if move not in engine.legal_moves(state):
            raise StoryError(f"turn {n} of {played.record_key} is not a legal move")
        text = engine.describe_move(state, move)
        after = engine.apply(state, move)
        if after.colour:
            best = max(best, after.colour.count(1), after.colour.count(2))
        if n in want:
            item: dict[str, Any] = {"n": n, "move": text, **_did(engine, state, after, move)}
            now = list(after.board)
            item["set"] = [[i, b] for i, (a, b) in enumerate(zip(shown_board, now, strict=True))
                           if a != b]
            shown_board = now
            if after.colour:
                held = _claims(after.colour)
                item["claims"] = [[sq, o] for sq, o in sorted(held.items())
                                  if owned.get(sq) != o]
                item["claims"] += [[sq, 0] for sq in sorted(set(owned) - set(held))]
                item["counts"] = [after.colour.count(1), after.colour.count(2)]
                owned = held
            if c.drops is not None:
                item["hands"] = [list(after.hands[0]), list(after.hands[1])]
            if previous and previous != n - 1:
                item["cut"] = True
            item["key"] = _key(played, n, last, item)
            frames.append(item)
            previous = n
        state = after
    outcome = engine.terminal(state)
    if outcome is None:
        raise StoryError(f"{played.record_key} does not end where its record stops")
    if (outcome.winner, outcome.reason) != (record.get("winner"), record.get("reason")):
        raise StoryError(f"{played.record_key} ends otherwise than its record says")
    if (outcome.winner, outcome.reason) != played.ending:
        raise StoryError(f"{played.record_key} ends otherwise than its story says")
    sides = start.get("sides") or ["white", "black"]
    moment = played.moment or last
    at = next((f for f in frames if f["n"] == moment), {})
    promoted = next((f["n"] for f in frames if f.get("promotion")), None)
    width = c.width
    played_games = record.get("played") or []
    alike = sum(1 for g in played_games if g.get("reason") == record.get("reason"))
    final = frames[-1] if frames else {}
    facts: dict[str, Any] = {
        "turns": len(moves),
        "seed": record.get("seed", played.seed),
        "game": (played.line + 1) if played.line is not None else None,
        "of": record.get("of"),
        "moment": moment,
        "moment_to": square_name(at["to"], width) if "to" in at else "",
        "portal_in": square_name(at["portal"][0], width) if "portal" in at else "",
        "moment_swept": " and ".join(square_name(i, width) for i in at.get("sweep") or []),
        "turns_s": "" if len(moves) == 1 else "s",
        "first_promotion": promoted,
        "winner": (str(sides[outcome.winner]).capitalize() if outcome.winner is not None
                   else ""),
        "a": (final.get("counts") or [None, None])[0],
        "b": (final.get("counts") or [None, None])[1],
        "best": best,
        "nodes": _nodes(list(record.get("agents") or [])),
        "typical": ((f"all {_word(len(played_games))}" if alike == len(played_games)
                     else f"{_word(alike)} of the {_word(len(played_games))}")
                    if played_games else ""),
    }
    return {
        "key": played.record_key,
        "hash": played.hash,
        "agents": list(record.get("agents") or []),
        "seed": facts["seed"],
        "line": played.line,
        "of": record.get("of"),
        "plies": len(moves),
        "outcome": {"winner": outcome.winner, "reason": outcome.reason},
        "pace": played.pace,
        "moment": moment,
        "start": start,
        "ends": ends,
        "frames": frames,
        "pieceArt": dict(art),
        "facts": facts,
    }


# A game replayed once per spec and record in a process: the server reads the stories again
# whenever a measurement moves, and the games themselves do not.
_REPLAYED: dict[tuple[str, Played], dict[str, Any]] = {}


def replayed(spec: dict[str, Any], played: Played) -> dict[str, Any]:
    """`replay`, kept by the spec's hash and the whole record of the game (its ending
    included), handed out as a copy."""
    key = (spec_hash(spec), played)
    if key not in _REPLAYED:
        _REPLAYED[key] = replay(spec, played)
    return copy.deepcopy(_REPLAYED[key])


# ------------------------------------------------------------------ what the site says


class Roots(Protocol):
    """What `render` reads: an application's three directories (`web.server.App`)."""

    runs_dir: Path
    ruleset_dir: Path
    archive_dir: Path


def library_paths(app: Roots) -> dict[str, tuple[Path, str]]:
    """Every listed library game's spec by hash, committed first, previews left out."""
    from chess_variant_sandbox.web import library

    found: dict[str, tuple[Path, str]] = {}
    for directory, source in ((app.ruleset_dir, "committed"), (app.archive_dir, "archived")):
        for path in sorted(Path(directory).glob("*.json")):
            try:
                spec = load_spec(path)
            except (OSError, ValueError):
                continue
            if source == "archived" and library.preview_sentence(spec) is not None:
                continue
            found.setdefault(spec_hash(spec), (path, source))
    return found


def read_score(app: Roots, digest: str, spec: dict[str, Any], path: Path,
               source: str) -> dict[str, Any] | None:
    """A library game's estimate as its own page reads it (`library.Game.score`): the
    stored `canonical.json` joined with its index file, its status and its basis."""
    from chess_variant_sandbox.web import library

    game = library.Game(hash=digest, name=str(spec.get("name", path.stem)),
                        description=str(spec.get("description", "")), path=path,
                        source=source, spec=spec)
    return library.reread_one(game, Path(app.runs_dir)).score


def verdict(score: dict[str, Any] | None, name: str) -> dict[str, Any]:
    """A library game's verdict and numbers, each with its basis, in the site's words."""
    from chess_variant_sandbox.web import sitescore

    if not score:
        raise StoryError(f"{name} has no estimate to show")
    worded = sitescore.worded(score) or {}
    seeds = list(worded.get("seeds") or [])
    families = worded.get("families") or {}
    index = worded.get("index")
    sd = worded.get("sd")
    certified = worded.get("certified")
    viable = bool(worded.get("viable"))
    # The third verdict (plan-teacher §10.16) is no verdict either way: never *ruled out*.
    unmeasured = worded.get("measured") is False
    said = "not measured" if unmeasured else "stays in" if viable else "ruled out"
    revision = str(worded.get("revision") or "")
    found = (revision[1:] if revision.startswith("v") and revision[1:].isdigit() else revision)
    basis = [f"{len(seeds)} run{'s' if len(seeds) != 1 else ''}",
             f"scoring rules version {found}"]
    if certified:
        basis.append("re-certified from a sample")
    number = f"{index:.3f}" if isinstance(index, (int, float)) else ""
    if number and isinstance(sd, (int, float)):
        number += f" ±{sd:.3f}"
    return {
        "name": name,
        "hash": worded.get("ruleset_hash") or worded.get("hash"),
        "status": worded.get("status"),
        "viable": viable,
        "measured": not unmeasured,
        "verdict": said,
        "index": index,
        "sd": sd,
        "runs": len(seeds),
        "revision": worded.get("revision"),
        "measuredAt": worded.get("measured_at"),
        "certified": certified,
        "qualities": {q: families.get(q) for q in QUALITIES},
        "words": list((worded.get("site") or {}).get("violations") or []),
        "split": ((worded.get("verdict") or {}).get("site") or {}).get("split", ""),
        "lean": ((worded.get("seat") or {}).get("site") or {}).get("lean", ""),
        "line": " · ".join([said, number, *basis]) if number else said,
    }


@lru_cache(maxsize=8)
def _quick(blob: str) -> dict[str, Any]:
    """The designer's two quick answers on a spec, cached by the spec itself."""
    from chess_variant_sandbox.web import candidacy, designing

    raw = json.loads(blob)
    checked = designing.check(raw)
    screened = candidacy.screen_draft(raw)
    stats = screened.get("stats") or {}
    games = int(stats.get("games") or 0)
    decisive = int(stats.get("decisive") or 0)
    opening = checked.get("openingWords") or f"{checked.get('openingMoves')} opening moves"
    first = ("It plays and reaches an ending, with " + str(opening) + "."
             if checked.get("playable") else
             "It plays, but it does not finish: " + " ".join(checked.get("playableReasons") or []))
    second = ("The eight-game check finds a winner in "
              + (f"all {games}" if decisive == games else f"{decisive} of its {games}")
              + " games." if screened.get("passed") else
              f"✗ {_screen_name()}: " + " ".join(screened.get("reasons") or []))
    return {
        "hash": checked.get("hash"),
        "playable": checked.get("playable"),
        "reasons": list(checked.get("playableReasons") or []),
        "opening": opening,
        "screen": {"passed": screened.get("passed"), "reasons": list(screened.get("reasons") or []),
                   "games": games, "decisive": decisive},
        "said": [first, second],
        "when": "the designer, as you edit, under a second; computed for this example",
    }


def _screen_name() -> str:
    """The designer's name for the eight-game check (`candidacy`'s rung), which a failed
    check is said under, with a ✗, as the designer shows it."""
    from chess_variant_sandbox.web import candidacy

    return next(rung["name"] for rung in candidacy._local_rungs(0, "") if rung["key"] == "screen")


def quick_answers(spec: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(_quick(json.dumps(spec, sort_keys=True)))


# A library game's quick answers are the designer's on its own rules, said as the designer
# says them while a draft is edited (LP4: stones' step 2 quotes them).
LIBRARY_QUICK_WHEN = "the designer, as you edit, under a second"


EXAMPLE_LINE = "an example · not measured"


def _example(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "example": True, "hash": spec_hash(spec), "line": EXAMPLE_LINE,
            "quick": quick_answers(spec)}


def _fill(text: str, facts: dict[str, Any]) -> str:
    return text.format_map(facts)


def _specs(story: Story, app: Roots, listed: dict[str, tuple[Path, str]]
           ) -> tuple[dict[str, Any], dict[str, Any], list[tuple[Path, str]]]:
    if story.kind == CONSTRUCTED:
        assert story.files is not None
        found = []
        for file, played in zip(story.files, (story.wrong.game, story.right.game), strict=True):
            path = HERE / file
            spec = load_spec(path)
            if spec_hash(spec) != played.hash:
                raise StoryError(f"{file} is not the game its story names")
            if played.hash in listed:
                raise StoryError(f"{file} is a library game, not an example")
            found.append((spec, path))
        return found[0][0], found[1][0], [(p, CONSTRUCTED) for _, p in found]
    specs = []
    for played in (story.wrong.game, story.right.game):
        if played.hash not in listed:
            raise StoryError(f"{played.hash} is not a listed library game")
        path, source = listed[played.hash]
        specs.append((load_spec(path), path, source))
    return specs[0][0], specs[1][0], [(p, s) for _, p, s in specs]


def _validates(spec: dict[str, Any], name: str) -> None:
    from chess_variant_sandbox.spec.schema import validate_spec

    if validate_spec(spec):
        raise StoryError(f"{name} does not validate")


def render_story(story: Story, app: Roots,
                 listed: dict[str, tuple[Path, str]] | None = None) -> dict[str, Any]:
    """One story joined with its games' frames and its numbers; raises `StoryError` where
    it does not hold."""
    listed = library_paths(app) if listed is None else listed
    failed_spec, fixed_spec, where = _specs(story, app, listed)
    failed_name = str(failed_spec.get("name", ""))
    fixed_name = str(fixed_spec.get("name", ""))
    if story.kind == CONSTRUCTED:
        _validates(failed_spec, failed_name)
        _validates(fixed_spec, fixed_name)
    if not differ_by_change(failed_spec, fixed_spec, story.changes):
        raise StoryError(f"{failed_name} and {fixed_name} differ by more than the change")
    wrong = replayed(failed_spec, story.wrong.game)
    right = replayed(fixed_spec, story.right.game)
    if story.kind == LIBRARY:
        (failed_path, failed_source), (fixed_path, fixed_source) = where
        before = verdict(read_score(app, story.wrong.game.hash, failed_spec, failed_path,
                                    failed_source), failed_name)
        after = verdict(read_score(app, story.right.game.hash, fixed_spec, fixed_path,
                                   fixed_source), fixed_name)
        for said, name in ((before, failed_name), (after, fixed_name)):
            if said["status"] != "current":
                raise StoryError(f"{name}'s estimate is {said['status']}, not current")
        if before["viable"] or not before["measured"]:
            raise StoryError(f"{failed_name} is no longer ruled out")
        if not after["viable"]:
            raise StoryError(f"{fixed_name} no longer stays in")
        before["quick"] = {**quick_answers(failed_spec), "when": LIBRARY_QUICK_WHEN}
    else:
        before = _example(failed_name, failed_spec)
        after = _example(fixed_name, fixed_spec)
    engine = Engine(failed_spec)
    c = engine.spec
    shared = {
        "failed": failed_name,
        "fixed": fixed_name,
        "goal": c.territory.win_at if c.territory is not None else "",
        "squares": c.width * c.height,
        "ends": " and ".join(square_name(i, c.width) for i in wrong["ends"]),
    }

    def said(beat: Beat, game: dict[str, Any]) -> dict[str, Any]:
        facts = {**shared, **game["facts"]}
        return {"say": _fill(beat.say, facts), "lead": _fill(beat.lead, facts),
                "end": _fill(beat.end, facts), "source": _fill(beat.source, facts),
                "game": game}

    art = {**wrong.pop("pieceArt"), **right.pop("pieceArt")}
    sides = right["start"].get("sides") or ["white", "black"]
    return {
        "key": story.key,
        "title": story.title,
        "family": story.family,
        "hue": FAMILIES[story.family],
        "kind": story.kind,
        "example": story.kind == CONSTRUCTED,
        "hero": {"clause": _fill(story.clause, shared), "chip": story.chip,
                 "game": fixed_name, "at": story.hero_at},
        "idea": _fill(story.idea, shared),
        "note": _fill(story.note, shared),
        "card": {"title": story.card.title, "lead": _fill(story.card.lead, shared),
                 "was": story.card.was, "now": story.card.now,
                 "tail": _fill(story.card.tail, shared),
                 "before": rule_line(failed_spec, story.card.rule),
                 "after": rule_line(fixed_spec, story.card.rule),
                 "door": story.card.door},
        "wrong": said(story.wrong, wrong),
        "change": {"say": story.tweak, "lines": change_lines(story.changes),
                   "edits": [{"path": list(ch.path),
                              **({} if ch.was is ABSENT else {"was": ch.was}),
                              **({} if ch.now is ABSENT else {"now": ch.now})}
                             for ch in story.changes]},
        "right": said(story.right, right),
        "failed": before,
        "fixed": after,
        # Quoted only while the fixed game's own description still says it.
        "quote": story.quote if story.quote in str(fixed_spec.get("description", "")) else "",
        # Whose seat the story is about, by its place and by the side's name on the board.
        "seat": {"index": story.seat, "side": str(sides[story.seat])},
        # Whom its play doors seat against: a rung of the ladder every board offers, by its
        # agent and its name, or None for the gentlest the game's page offers.
        "opponent": door_opponent(story),
        "pieceArt": art,
    }


def door_opponent(story: Story) -> dict[str, str] | None:
    """The rung of the alternating ladder a story's doors seat against, or None for the
    gentlest; a story naming a player the ladder does not hold is refused."""
    if not story.opponent:
        return None
    from chess_variant_sandbox.web.opponents import HUMAN, RUNGS

    for agent, name, _tail in RUNGS:
        if agent == story.opponent and agent != HUMAN:
            return {"agent": agent, "name": name}
    raise StoryError(f"{story.key}'s door opponent {story.opponent} is no rung of the ladder")


def inputs(app: Roots) -> tuple[tuple[str, int], ...]:
    """What `render` reads that can change under a running server: every file in each
    library story's games' `runs/<hash>/`, with its time of change."""
    found: list[tuple[str, int]] = []
    for story in STORIES:
        for played in (story.wrong.game, story.right.game):
            folder = Path(app.runs_dir) / played.hash
            if folder.is_dir():
                found += [(str(path), path.stat().st_mtime_ns)
                          for path in sorted(folder.glob("*.json"))]
    return tuple(found)


def render(app: Roots) -> dict[str, Any]:
    """Every story that holds, in `STORIES`' order, and every one that does not, with why.

    `pool` is the keys of the holding stories (§4.1's `data-stories`); a dropped story is
    in `dropped` alone and is never drawn."""
    listed = library_paths(app)
    stories: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    for story in STORIES:
        try:
            stories.append(render_story(story, app, listed))
        except (StoryError, OSError, ValueError, KeyError) as error:
            dropped.append({"key": story.key, "why": str(error)})
    return {"pool": [s["key"] for s in stories], "stories": stories, "dropped": dropped}


# ------------------------------------------------------------------ as text


def _qualities(failed: dict[str, Any], fixed: dict[str, Any]) -> str:
    def number(value: Any) -> str:
        return f"{value:.2f}" if isinstance(value, (int, float)) else "-"

    return ", ".join(f"{q} {number(failed['qualities'].get(q))} → "
                     f"{number(fixed['qualities'].get(q))}" for q in QUALITIES)


def text(story: dict[str, Any]) -> str:
    """One rendered story as plain lines: what a crawler would read of it, and what the
    command prints."""
    kind = "an example made for this page" if story["example"] else "a library pair"
    lines = [
        f"{story['title']} ({story['family']}; {kind})",
        f"  hero: Wouldn't it be cool if {story['hero']['clause']} ({story['hero']['game']})",
        f"  {story['idea']}",
        f"  {story['note']}",
        f"  ✗ {story['wrong']['say']}",
        f"    {story['wrong']['lead']}",
        f"    {story['wrong']['end']}",
        f"    ({story['wrong']['source']})",
        f"    {story['failed']['name']}: {story['failed']['line']}",
    ]
    lines += [f"      {words}" for words in story["failed"].get("words") or []]
    lines += [f"      {said}" for said in (story["failed"].get("quick") or {}).get("said") or []]
    card = story["card"]
    lines += [
        f"  → {story['change']['say']}",
        f"    {card['title']}: {card['lead']} ~~{card['was']}~~ {card['now']}{card['tail']}",
    ]
    lines += [f"    - {line['del']}" for line in story["change"]["lines"] if line["del"]]
    lines += [f"    + {line['ins']}" for line in story["change"]["lines"] if line["ins"]]
    lines += [
        f"  ✓ {story['right']['say']}",
        f"    {story['right']['lead']}",
        f"    {story['right']['end']}",
        f"    ({story['right']['source']})",
        f"    {story['fixed']['name']}: {story['fixed']['line']}",
    ]
    if story["fixed"].get("lean"):
        lines.append(f"      {story['fixed']['lean']}")
    lines += [f"      {said}" for said in (story["fixed"].get("quick") or {}).get("said") or []]
    if not story["example"]:
        lines.append(f"    qualities: {_qualities(story['failed'], story['fixed'])}")
    if story["quote"]:
        lines.append(f"    “{story['quote']}”")
    return "\n".join(lines)


@dataclass
class _Roots:
    runs_dir: Path
    ruleset_dir: Path
    archive_dir: Path


def main(argv: list[str] | None = None) -> None:
    from chess_variant_sandbox.web import library

    parser = argparse.ArgumentParser(description="Print the landing's stories as render reads them.")
    parser.add_argument("--runs", type=Path, default=library.RUNS_DIR)
    args = parser.parse_args(argv)
    # The stories' marks (✗ → ✓) on a console whose own encoding cannot print them.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")
    rendered = render(_Roots(args.runs, library.RULESET_DIR, library.ARCHIVE_DIR))
    print("pool:", " ".join(rendered["pool"]) or "(empty)")
    for story in rendered["stories"]:
        print()
        print(text(story))
    for gone in rendered["dropped"]:
        print(f"\ndropped {gone['key']}: {gone['why']}")


if __name__ == "__main__":
    main()
