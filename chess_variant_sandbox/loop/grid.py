"""The behaviour grid: what kind of game a ruleset is, before anything plays it.

Given a hard viability floor, a flat top, and an interval wider than the gaps it is being
asked to rank, a ranked leaderboard is the wrong shape for this search. "Discover a good
chess-like game" wants an archive of *different* good games, not one local optimum
polished for eighty generations -- so the archive becomes a grid of cells, each holding
the best game found of that kind, and the index becomes a viability gate plus an in-cell
tie-break rather than a global ranking.

**Every axis here is spec-derivable, and that is the point.** Ordinary MAP-Elites mutates
blindly and hopes to land in an empty cell. A model can be *told* the cell -- "a board of
about thirty-six squares, pieces played from hand, won by arrival rather than by mate" --
which is a much better pairing than either half suggests alone, and it only works for
descriptors known *before* a tournament. Mean branching and mean plies are the tempting
measured axes and they are excluded deliberately: membership would be known only after
the games, and a candidate could migrate cells between two evaluations without changing
at all.

**The grid is sized for the budget, not for the descriptor space.** At model-proposal
rates a long run is hundreds of evaluations, not tens of thousands, so these 432 cells
will be sparsely filled with one elite each. That makes this a novelty archive more than
a converged grid, which is fine -- provided it is called one.

Validated against the committed rulesets before it was used for anything, and the
validation changed it. Chess, shogi and xiangqi land in three different cells, which is
the check the plan named; but the first four axes also put *eight* of the twenty-two in
a single cell -- the whole 8x8, six-piece, mate-ending neighbourhood, which is most of
the region a search over chess-like games actually explores. The fifth axis, how many
special rules a game carries, splits that into five and three.

The sixth arrived with schema 3, and it is the one axis that separates a *kind* of game
rather than a size of one: whether what a piece may capture depends on what it is
looking at. Under an ordinary capture rule material is a total order and trades,
sacrifices and counting all follow from that; under a dominance relation none of it
transfers. Every ruleset written before schema 3 sits in bin 0, so the axis doubled the
grid without moving anything already in it -- which is what an inert feature looks like
from up here.

The ending axis grew a fifth bin in the same bump, for a game decided on a count when
the clock runs out. It sits ahead of arrival in the order because a scoring rule changes
what the *whole game* is about rather than adding a second way to finish it: a small
advantage simply wins, instead of needing to be converted into an event first. A sixth
bin arrived with schema 4's territory, ahead of even that one: a game won by claiming
ground is not a chess-shaped game with an ending bolted on, it is a game where the
pieces are a means rather than the object.

The seventh arrived with schema 5 and clears the same bar the sixth does: whether the
board has edges at all. An edge is what a mating net presses a king against, what a
rook's line ends at, and what makes a corner worth less than the centre, and joining two
of them transfers none of it. It was not optional -- five wrapped rulesets landed in the
chess cell and took it from two to seven, past the very threshold the fifth axis exists
to hold. The eighth arrived with schema 6 on the same test, and ``_taking`` carries its
argument: whether taking a piece removes it from the game or turns it into another one.
1,728 cells now, and nothing already in the grid has moved for any of them.

**A facing rule gets no axis yet, and the reason is the test rather than the argument.**
Schema 8's ``piece.facing`` -- a piece that reads its vectors through an orientation it
may spend a move turning -- clears the bar the argument sets, and clears it easily.
``archive/weathervane`` is ``rulesets/sardines.json`` with directional pawns and nothing
else changed, and it quadruples ``quick_kill_rate``: each king in sardines is walled in by its
own pawns, and those same pawns on those same squares are not the same wall once they
point wherever the position rolled them. A rule that dismantles a static formation
without moving anything is a kind of game rather than a size of one.

What is missing is the test every axis here was actually *added* under, which ``_taking``
states outright -- the operational one, not the strength of the argument alone. That test
is a batch of rulesets collapsing into one cell: ``edges`` was five wrapped boards taking
a cell to seven of thirty-three, ``taking`` was schema 6's showcase set landing together
-- and that set was weighed over ``archive/`` as well, because none of it was ever
committed. Weigh a facing the same way and there are two specs, ``chess-oriented-pawns``
and ``archive/weathervane``. The grid does file each with the game it was made from, but
they land in *different* cells and neither cell is a collapse: six and five. One game is
not a batch, and a ninth axis is 3,456 cells against a run of a few hundred evaluations.
**Add the axis when a second committed facing ruleset lands.** That is the batch; the
argument is banked here and does not need deciding again.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.spec.schema import strike_changes_anything, wrap_stated

# --------------------------------------------------------------------------- axes


def _area(spec: dict[str, Any]) -> int:
    board = spec.get("board", {})
    area = int(board.get("width", 8)) * int(board.get("height", 8))
    if area < 40:
        return 0
    return 1 if area <= 80 else 2


def _piece_types(spec: dict[str, Any]) -> int:
    count = len(spec.get("pieces", []) or [])
    if count <= 3:
        return 0
    return 1 if count <= 6 else 2


def _hand(spec: dict[str, Any]) -> int:
    """Whether captured material comes back. It changes what a trade means, which is
    about as deep a structural difference as this vocabulary can express."""
    return 1 if (spec.get("rules", {}) or {}).get("drops") else 0


def _ending(spec: dict[str, Any]) -> int:
    """How the game is won, taking the most distinctive ending a ruleset declares.

    Order matters here: king of the hill has both a reach condition and checkmate, and
    what makes it *king of the hill* is the arrival. A ruleset is filed under the ending
    that distinguishes it, not under the one it inherited.
    """
    rules = spec.get("rules", {}) or {}
    if rules.get("territory", {}).get("win_at"):
        # Ahead of everything, including scoring. A game won by claiming ground is not
        # a chess-shaped game with an extra ending bolted on -- the pieces are a means
        # to colouring the board rather than the thing being fought over, and every
        # strategic word changes meaning with them.
        return 5
    if rules.get("scoring"):
        # Ahead of arrival, and ahead of everything else it can be combined with. A game
        # decided by *having more of something when time runs out* is a different kind
        # of game from one decided by an event, whatever else it also says: it is the Go
        # ending rather than the chess one, and a small advantage simply wins instead of
        # needing to be converted into an event first.
        return 4
    if rules.get("reach"):
        return 2
    if rules.get("check_limit") or rules.get("extinction") or rules.get("bare_royal") \
            or rules.get("must_capture") or rules.get("win") == "none":
        return 3
    return 0 if rules.get("win", "checkmate") == "checkmate" else 1


def _capture(spec: dict[str, Any]) -> int:
    """Whether what a piece may capture depends on what it is looking at.

    A genuinely different kind of game rather than a variation on one, which is the bar
    every axis here has to clear. Under an ordinary capture rule material is a total
    order and the whole strategic vocabulary follows -- trades, sacrifices, counting.
    Under a dominance relation none of that transfers: a piece can be worthless against
    one army and decisive against another, and the same position is worth different
    amounts to the two players.

    Two bins rather than three. "Cyclic" versus "hierarchical" is a tempting third, but
    the grid is sized for the budget rather than for the descriptor space, and 432 cells
    filled from a few hundred evaluations is already sparse.
    """
    pieces = spec.get("pieces", []) or []
    return 1 if any(isinstance(p.get("captures"), list) for p in pieces) else 0


def _exceptions(spec: dict[str, Any]) -> int:
    """How many special rules the game carries beyond how its pieces move.

    Castling, en passant, promotion, drops, explosions, regions: each is a sentence a
    new player has to be told. A facing is equally a sentence and is *not* among them --
    ``special_rules`` counts ``rules``, and a facing is a piece property counted apart in
    ``complexity.facing_pieces`` -- which is why ``chess-oriented-pawns`` is filed with
    chess. See the facing note in the module docstring.

    Two bins, because the split that matters is between a game whose rules are its
    pieces and one that needs a paragraph of exceptions -- and because it is the axis
    that pulls the crowded chess-shaped cell apart. Without it, eight of the twenty-two
    committed rulesets share one cell, which is most of the region a search over
    chess-like games actually explores.
    """
    from chess_variant_sandbox.spec.complexity import measure_complexity

    return 0 if measure_complexity(spec).special_rules <= 2 else 1


def _edges(spec: dict[str, Any]) -> int:
    """Whether the board has edges at all.

    The second axis to clear the bar the `capture` axis set -- a different *kind* of
    game rather than a variation on one. An edge is what a mating net presses a king
    against, what a rook's line ends at, and what makes a corner worth less than the
    centre; join two of them and none of that transfers. It is also the reason this axis
    had to exist rather than be folded into the rule count: five wrapped rulesets landed
    in the chess cell and took it from two to seven, past the threshold the fifth axis
    was added to hold.

    Two bins rather than four, for the reason `capture` gives: "one axis joined" versus
    "both" is a tempting third and fourth, and 864 cells filled from a few hundred
    evaluations is already sparse.
    """
    wrap = (spec.get("rules", {}) or {}).get("wrap")
    shape = (spec.get("board", {}) or {}).get("shape")
    stated = wrap_stated(wrap, shape)
    return 1 if any(edge != "none" for edges in stated.values() for edge in edges) else 0


def _taking(spec: dict[str, Any]) -> int:
    """Whether taking a piece removes it from the game or turns it into another one.

    The third axis to clear the bar the `capture` axis set, and by the same argument
    that one makes. Under an ordinary capture rule material is conserved downward by
    trades: a piece taken is a piece gone, counting works, and the whole vocabulary of
    exchange -- a winning exchange, a sacrifice, being a piece up -- rests on it. Under
    a chain none of that transfers. A queen taken becomes a rook, so a trade that looks
    winning may not be; under an ascending chain taking a piece *hands the defender a
    better one*, and the sign of an exchange can flip outright.

    Added on the operational test the `edges` axis was added under, not on the strength
    of the argument alone: schema 6's three showcase rulesets all landed in the chess
    cell and took it from five to eight, past the same threshold the fifth and seventh
    axes exist to hold. With this axis the pair moves out and the cell settles at six.
    1,728 cells now, and nothing already in the grid has moved -- `parse` pads a label
    written before an axis existed, and a spec written before schema 6 removes what it
    takes.

    Obstacles deliberately get no axis. A wall is a fact about the board's shape, which
    the `board` axis is already a coarse reading of, and it changes where the lines run
    rather than what a move means.

    **Both spellings, since schema 10 (Q108).** ``rules.contacts`` is the same mechanism
    with the attacker's column filled in: the validator refuses the two together, the
    compiler resolves both into one ``StrikeRule``, and `distance` gives the table the
    strike's own three features plus two of its own. Reading only ``strike`` filed every
    contact-table game under *removes* -- so a `hierarchy-fall` and the same game
    re-spelled as a table landed in different cells, and every child of
    `become-what-you-take` looked to the grid like an intra-cell edit. No committed or
    archived spec uses `contacts`, so nothing already in the grid moves.

    An explosion still reads as *removes*, and that is the same argument rather than an
    omission: a blast takes more pieces off, it does not turn one into another, and
    counting still works on what is left. `atomic` is filed where it was.

    **Transforms is what the compiler keeps** (Q417). Either spelling is asked of the
    compiler through the validator's `schema.strike_changes_anything`, which is its whole
    answer: a strike is kept when something survives, a table when something survives
    *or* only the attacker changes. This used to read any non-``capture`` chain and any
    table with links as transforming, so a chain whose every link starts at an immune
    piece, or a table whose every link lands on one, was filed apart from the chess it
    compiles to. A strike never reaches an immune piece, and the compiler drops both as
    plain capture written the long way. No committed or archived spec carries either, so
    nothing already in the grid moves.
    """
    rules = spec.get("rules", {}) or {}
    piece_ids = list(
        dict.fromkeys(
            piece["id"]
            for piece in spec.get("pieces", []) or []
            if isinstance(piece, dict) and isinstance(piece.get("id"), str)
        )
    )
    kept = any(
        strike_changes_anything(rules.get(key), key, piece_ids) for key in ("contacts", "strike")
    )
    return 1 if kept else 0


@dataclass(frozen=True)
class Axis:
    name: str
    bins: tuple[str, ...]
    of: Callable[[dict[str, Any]], int]


AXES: tuple[Axis, ...] = (
    Axis("board", ("small", "medium", "large"), _area),
    Axis("army", ("spare", "middling", "rich"), _piece_types),
    Axis("hand", ("no drops", "drops"), _hand),
    Axis(
        "ending", ("mate", "capture", "arrival", "other", "score", "ground"), _ending
    ),
    Axis("rules", ("plain", "ornamented"), _exceptions),
    Axis("capture", ("anything", "restricted"), _capture),
    Axis("edges", ("edges", "joined"), _edges),
    Axis("taking", ("removes", "transforms"), _taking),
)

CELLS = 1
for _axis in AXES:
    CELLS *= len(_axis.bins)


def descriptor(spec: dict[str, Any]) -> tuple[int, ...]:
    """Which cell this ruleset belongs to. Exact, free, and known before it is played."""
    return tuple(axis.of(spec) for axis in AXES)


def label(cell: Iterable[int]) -> str:
    """A cell as a short string, for archives, filenames and leaderboards."""
    return "-".join(str(value) for value in cell)


def parse(text: str) -> tuple[int, ...]:
    """A cell label back into a cell, settling a label written before an axis existed.

    Axes are appended, and every ruleset written before an axis existed sits in its bin
    0 -- that is what made each addition inert. A label stored by an earlier run is
    therefore not stale data, it is a *prefix*, and padding it with zeroes is the
    migration rather than a guess: a spec written before schema 5 has a board with edges.
    Without this, a design run resumed across a bump dies in ``empty_neighbours`` with an
    index error rather than carrying on.
    """
    values = tuple(int(part) for part in text.split("-") if part != "")
    if len(values) >= len(AXES):
        return values[: len(AXES)]
    return values + (0,) * (len(AXES) - len(values))


# What each bin means, in the language of games. One table, two readers: `describe_cell`
# joins it into the sentence a proposal is aimed with, and `describe_axes` hands it to an
# interface one axis at a time. They were separate strings until an interface wanted the
# second shape, and a cell rendered as `1-1-0-0-1-0` is not a thing anybody can read.
_BIN_PROSE: dict[str, dict[str, str]] = {
    "board": {
        "small": "under forty squares",
        "medium": "forty to eighty squares",
        "large": "more than eighty squares",
    },
    "army": {
        "spare": "at most three kinds of piece",
        "middling": "four to six kinds of piece",
        "rich": "seven or more kinds of piece",
    },
    "hand": {
        "drops": "captured pieces returning to play from hand",
        "no drops": "captured pieces gone for good",
    },
    "ending": {
        "mate": "won by trapping a royal piece",
        "capture": "won by capturing a royal piece outright",
        "arrival": "won by getting a piece to a particular place",
        "other": "won by something other than mate, capture or arrival -- a count of "
                 "checks, running an opponent out of a piece, or a condition of your own",
        "score": "decided on a count when time runs out -- material, pieces, or ground "
                 "held -- so that being a little ahead is enough to win",
        "ground": "won by claiming enough of the board, so that the pieces are a means "
                  "of colouring squares rather than the thing being fought over",
    },
    "rules": {
        "plain": "no special rules to speak of -- the pieces are the rules",
        "ornamented": "room for two or three special rules on top of how the "
                      "pieces move",
    },
    "edges": {
        "edges": "a board with edges, where a corner is a place a king can be trapped",
        "joined": "a board whose opposite edges are the same edge, so that lines run "
                  "off one side and back on the other and there is no corner to trap "
                  "anything in",
    },
    "taking": {
        "removes": "a piece that is taken leaves the game, so material is conserved by "
                   "trades and counting it means something",
        "transforms": "a piece that is taken becomes another piece instead of leaving, "
                      "so a trade can change what both sides have rather than only how "
                      "much, and an exchange that looks winning may not be",
    },
    "capture": {
        "anything": "any piece may take any enemy piece",
        "restricted": "pieces that may only capture particular kinds of enemy piece, "
                      "so that no piece is simply stronger than another",
    },
}

# What each axis is asking. A reader looking at "ending: arrival" needs to be told that
# the column is "how it is won" before the bin means anything; the bin's own prose above
# answers the question, and this one states it.
_AXIS_QUESTION: dict[str, str] = {
    "board": "how big the board is",
    "army": "how many kinds of piece there are",
    "hand": "whether captured pieces come back",
    "ending": "how the game is won",
    "taking": "what happens to a piece that is taken",
    "rules": "how much there is beyond how the pieces move",
    "capture": "whether what a piece may take depends on what it is",
    "edges": "whether the board has edges",
}


def bins_of(cell: Iterable[int]) -> tuple[str, ...]:
    """The cell as its bin names -- the short readable form of ``label``."""
    return tuple(axis.bins[value] for axis, value in zip(AXES, cell, strict=True))


def describe_axes(cell: Iterable[int]) -> list[dict[str, str]]:
    """The cell one axis at a time: what is being asked, and this game's answer.

    The same seven facts ``describe_cell`` puts in a sentence, in the shape an interface
    needs to lay them out beside each other. A sentence is right for a model, which
    reads it once and is aimed by it; a table is right for a person comparing two games.
    """
    return [
        {
            "axis": axis.name,
            "asks": _AXIS_QUESTION[axis.name],
            "bin": axis.bins[value],
            "means": _BIN_PROSE[axis.name][axis.bins[value]],
        }
        for axis, value in zip(AXES, cell, strict=True)
    ]


def axis_vocabulary() -> list[dict[str, Any]]:
    """Every axis and every bin it has, with the prose for each.

    ``describe_axes`` says where one game sits; this says what the space is. An
    interface needs both -- a cell means nothing without the legend, and the legend is
    the same six questions for every ruleset, so it travels once rather than per game.
    """
    return [
        {
            "axis": axis.name,
            "asks": _AXIS_QUESTION[axis.name],
            "bins": [
                {"bin": name, "means": _BIN_PROSE[axis.name][name]} for name in axis.bins
            ],
        }
        for axis in AXES
    ]


def describe_cell(cell: Iterable[int]) -> str:
    """The cell in the language of games, for telling a model where to aim.

    This is the half of MAP-Elites that an LLM operator makes possible: an ordinary
    mutation lands where it lands, and a model can be asked for a specific kind of game.
    """
    means = [entry["means"] for entry in describe_axes(cell)]
    # Built from however many axes there are rather than unpacked into a fixed seven.
    # The eighth axis broke the fixed form, and the ninth would have broken it again --
    # loudly here and in the library, which is better than quietly, but this is one of
    # the places `AXES` is genuinely allowed to grow without asking permission.
    #
    # The connective belongs to the sentence, not to the bin: a table cell reading
    # "and room for two or three special rules" is a table cell with a stray word in it.
    body = ", ".join(means[1:-1])
    return f"a board of {means[0]}, {body}, and {means[-1]}"


# --------------------------------------------------------------------------- grid


@dataclass
class Grid:
    """The occupied cells and what holds each one.

    Deliberately thin: the archive is still the record of what happened, and this is a
    view over it. Rebuilding it from entries is cheap and means there is no second copy
    of the truth to go stale.
    """

    elites: dict[tuple[int, ...], Any]

    @property
    def filled(self) -> int:
        return len(self.elites)

    @property
    def coverage(self) -> float:
        return self.filled / CELLS

    def elite(self, cell: Iterable[int]) -> Any | None:
        return self.elites.get(tuple(cell))

    def empty_neighbours(self, cell: Iterable[int]) -> list[tuple[int, ...]]:
        """Cells one step away on a single axis that nothing has reached yet.

        One step, because a neighbour two changes away is a different game rather than a
        variation on this one, and the point of naming a target is that a revision of
        *this* ruleset could plausibly land there.
        """
        values = tuple(cell)
        found: list[tuple[int, ...]] = []
        for index, axis in enumerate(AXES):
            for value in range(len(axis.bins)):
                if value == values[index]:
                    continue
                candidate = values[:index] + (value,) + values[index + 1 :]
                if candidate not in self.elites:
                    found.append(candidate)
        return found

    def summary(self) -> str:
        lines = [f"{self.filled} of {CELLS} cells filled ({self.coverage:.0%})"]
        for cell, entry in sorted(self.elites.items()):
            name = getattr(entry, "name", "?")
            # The elite's number as the board shows it (`Entry.scored_as`): with the runs
            # it rests on, or *no score* for an entry that never recorded them.
            scored_as = getattr(entry, "scored_as", None)
            lines.append(
                f"  {label(cell):9s} {name}"
                + (f"  {scored_as()}" if callable(scored_as) else "")
            )
        return "\n".join(lines)


def build(entries: Iterable[Any], prefer: Any | None = None) -> Grid:
    """One elite per cell: the best viable entry of each kind.

    Three tests in order, and the order is the whole design. The index decides when it
    can -- when one entry's interval clears the other's. Where it cannot, the judge's
    pairwise ordering decides, if it has seen enough of both to have an opinion that
    separates. Where neither can, the higher point estimate holds the cell, which is an
    arbitrary choice made *locally*: the loser stays in the archive, both are games of
    the same kind, and the consequence is which of two similar games gets revised rather
    than which direction the whole search takes.
    """
    elites: dict[tuple[int, ...], Any] = {}
    for entry in entries:
        cell = getattr(entry, "cell", None)
        if not cell or not getattr(entry, "viable", False):
            continue
        key = parse(cell) if isinstance(cell, str) else tuple(cell)
        current = elites.get(key)
        if current is None or _wins_cell(entry, current, prefer):
            elites[key] = entry
    return Grid(elites=elites)


def _wins_cell(challenger: Any, holder: Any, prefer: Any | None) -> bool:
    if challenger.beats(holder):
        return True
    if holder.beats(challenger):
        return False
    if prefer is not None:
        if prefer.beats(challenger.hash, holder.hash):
            return True
        if prefer.beats(holder.hash, challenger.hash):
            return False
    return (challenger.index or 0.0) > (holder.index or 0.0)
