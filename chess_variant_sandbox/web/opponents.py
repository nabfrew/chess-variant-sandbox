"""Who you can play in one ruleset, and how strong each of them is in it.

**The roster is a function of the ruleset, and it did not used to be.** The ladder --
random, greedy, a row of node budgets -- is the same player on every board, which is the
whole reason the index can be a scale. A rule of thumb is not: `agents/features.py`
normalises by board size, so one digest is not one player on two boards, and the plans
committed for a ruleset are the plans that read *its* declared rules. A module-level
tuple could express the first and never the second.

**Ratings here are a separate table, on purpose, and the separation is load-bearing.**
`runs/<hash>-<seed>/tournament.json` holds the rated ladder's Elo for a ruleset and is
read by the index; heuristic players are kept out of it by construction, because
`rating_spread` reads the whole table and no name filter can reach that. So the on-demand
run below writes `runs/opponents/<hash>.json`, which nothing in `metrics/` looks at. Its
games are kept beside it since L3 (`docs/plan-leaderboard.md` §2.2), as
`runs/opponents/<hash>.games.jsonl` -- the seats, the seed and the winner, never a run
directory's `games.jsonl` -- so a newly offered player can be added to a ladder by playing
its own games alone; nothing under `metrics/` reads those either.

That leaves two tables of Elo for one ruleset, which is a real hazard and is handled the
way this project handles two weights versions: they are never mixed in one column. The
stored table is labelled as the rated ladder's, the on-demand one as the picker's, and
each says how many games it rests on.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents.fairy_stockfish import DEFAULT_NODES as FAIRY_STOCKFISH_NODES
from chess_variant_sandbox.agents.fitted import STORE_DIR as FIT_DIR
from chess_variant_sandbox.agents.heuristics import (
    STORE_DIR,
    Heuristic,
    HeuristicError,
    HeuristicSet,
    other_spelling,
    register_heuristic,
    store_path,
)
from chess_variant_sandbox.agents.nets import STORE_DIR as NET_DIR
from chess_variant_sandbox.agents.valuations import SEARCH as SEARCH_PROCEDURE
from chess_variant_sandbox.agents.valuations import STORE_DIR as VALUE_DIR
from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament

# Who everybody is measured against. A round robin over the whole picker is the obvious
# design and costs four times as much where it hurts: `minimax:nodes=200000` is ~0.2s a
# move, and in a seventeen-player round robin it plays sixty-four games rather than
# twenty. So the expensive players are measured against a fixed cheap spine instead, and
# the spine's cost is fixed by its own length rather than by how many rules of thumb a
# ruleset happens to have.
#
# It has to span the range, which is the part that is easy to get wrong. With a spine
# topping out at 1,600 nodes every strong player beats all of it every time, and a
# player with a perfect score has no determined rating at all -- they would come back
# tied at whatever the fit's ceiling is. 6,400 nodes is cheap (three to eight
# milliseconds a move) and loses often enough to everything above it to place them.
SPINE: tuple[str, ...] = (
    "random",
    "greedy",
    "minimax:nodes=400",
    "minimax:nodes=1600",
    "minimax:nodes=6400",
)

HUMAN = "human"

# Node budgets offered as opponents, with what each one is. Every entry is a rated
# player in the ladder the index reads, which is the point of offering these and not,
# say, a depth -- and adding one changes no existing agent, which is why the top of this
# list could move without touching a calibration number.
#
# It used to stop at 6,400 nodes, which is three to eight milliseconds a move. That is
# the strongest thing the interface offered and it is nowhere near the strongest thing
# the search can do in a time a person will wait: on the real `rps-v2` opening, 200,000
# nodes reaches two plies deeper in a fifth of a second, and on `rps-v1` it reaches
# depth 7 in 178ms. Two to three extra plies for a fifth of a second was the single
# cheapest improvement available to the complaint that started stage 17.
#
# The labels exist because "minimax:nodes=6400" tells a person nothing about how hard
# the thing will be, and the top of the list being labelled "hardest" is a claim the
# measurements in `docs/reference/stage-17-measurements.md` back up.
#
# Each rung is its agent string, its name and what it does in plain words. **The name
# leads and is the whole of what a sentence calls it** (*medium could not move*); the
# words are `play.strength`'s for the budget it plays at, the same words the seat panel
# prints under a seat, so the picker and the board name one player one way (F7 of
# `docs/reference/first-visit-study.md`; `docs/plan-understanding.md` §3.3, P2-P5). The
# agent string is the player's identity and never changes here: only what it is called.
RUNGS: tuple[tuple[str, str, str], ...] = (
    (HUMAN, "you", ""),
    ("random", "random", ""),
    ("greedy", "greedy", ""),
    ("minimax:nodes=400", "easy", ""),
    ("minimax:nodes=1600", "medium", ""),
    ("minimax:nodes=6400", "hard", ""),
    ("minimax:nodes=200000", "hardest", ", so it is the slowest"),
    ("mcts:nodes=1600", "tree search", ""),
)

# Between a name and its plain words in a label. A sentence wants the name alone, and a
# reader of a label takes the name off before this (`Opponent.name`, which every caller
# reads rather than cutting a label).
SEPARATOR = " · "


def labelled(name: str, about: str) -> str:
    """A label: the name, then what it does, or the name alone where nothing is said."""
    return f"{name}{SEPARATOR}{about}" if about else name


def _rung_about(agent: str, tail: str) -> str:
    from chess_variant_sandbox.web.play import strength

    words = strength(agent)
    return words + tail if words else ""


# (agent, label), weakest first: the picker's rows as a person reads them.
LADDER: tuple[tuple[str, str], ...] = tuple(
    (agent, labelled(name, _rung_about(agent, tail))) for agent, name, tail in RUNGS
)

# What the picker heads the rungs with: the order is the ladder's own (P1).
LADDER_GROUP = "computer players, weakest first"

# What a rule of thumb searches when you play it.
#
# The same budget as the "hard" tier, deliberately. The heuristic *measurement* holds an
# objective fixed and varies depth, because that is the only way to ask whether a game
# has anything to think about. Playing one asks the other question -- what is it like to
# face somebody with a different idea of what matters -- and the honest form of that is
# to hold depth fixed and vary the objective, so "hard" and "hard, playing 'hunt the
# king'" differ in exactly one thing.
HEURISTIC_NODES = 6_400

# Where the stores live, relative to the heuristics root, what to call them, and whether
# their first entry is the built-in evaluator's own vector. Stage 23's model sets are
# kept in a sub-store rather than merged into the main one, because `heuristic_headroom`
# is a median over a set and demonstrably moves with it -- see
# `docs/reference/stage-23-proposer.md`.
#
# The third flag is not decoration. A *generated* set opens with the built-in anchor by
# convention, so offering index 0 would be offering `minimax:nodes=6400` twice under two
# names -- and a set somebody built by hand has no anchor, so skipping its first entry
# would silently drop the first opponent they made.
#
# The written and the proposed plans share one heading on purpose (P7): to somebody
# choosing an opponent they are the same thing, a player with a strategy, and `store`
# still says which file each came from.
SOURCES: tuple[tuple[str, str, bool], ...] = (
    ("", "with a strategy", True),
    ("model", "with a strategy", True),
    # Built in the browser, by a person, while playing. Kept out of the main store for
    # the reason `model/` is: `heuristic_headroom` is a median over a *set*, so what is
    # in the set moves a committed measurement.
    ("custom", "with a strategy you built", False),
)


@dataclass(frozen=True)
class Opponent:
    """One entry in the picker."""

    agent: str
    label: str
    group: str
    plan: str = ""
    # Which store this came out of: "ladder" for the rungs every board shares, "written"
    # for the committed plans, "model" for the proposed ones, "custom" for what somebody
    # built in the browser, and "fairy", "fitted" or "valuations".
    # `group` is a *label* and gets reworded; anything that needs to know where an
    # opponent came from asks this, so rewording a menu heading cannot empty a list.
    store: str = ""
    # What a sentence calls it (*medium*, *Fairy-Stockfish*), and what it does in plain
    # words (*looks at 1,600 positions before each move*). The label is the two joined;
    # an opponent with nothing to add beyond its name has an empty `about` and its whole
    # label as its name.
    name: str = ""
    about: str = ""
    # A player no game against is ever rated (Q683): the learned player. It is kept out
    # of the on-demand rating run (`agents_of`), so it never reaches the picker's table
    # and from there the site's; the site refuses a rated game against it by its agent
    # string (`never_rated`); and the page sets the game to practice when it is chosen.
    practice_only: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.label)

    def as_dict(self) -> dict[str, Any]:
        return {"agent": self.agent, "label": self.label, "group": self.group,
                "plan": self.plan, "store": self.store, "name": self.name,
                "about": self.about, "practiceOnly": self.practice_only}


def ladder() -> list[Opponent]:
    """The roster that is the same on every board."""
    return [
        Opponent(
            agent=agent,
            label=labelled(name, about),
            group=LADDER_GROUP,
            store="ladder",
            name=name,
            about=about,
        )
        for agent, name, about in (
            (agent, name, _rung_about(agent, tail)) for agent, name, tail in RUNGS
        )
    ]


def name_of(agent: str, roster: list[Opponent] | None = None) -> str:
    """What a sentence calls this player: its name off `roster` (else the ladder every
    board shares), and *the computer* for one neither names -- never its agent string.
    """
    for entry in [*(roster or []), *ladder()]:
        if entry.agent == agent:
            return entry.name
    return "the computer"


def table_name(agent: str) -> str:
    """What a rating table's row calls a player no roster entry names (G17).

    The ladder that scored a game plays its own players -- the built-in objective fitted to
    the corpus, at six budgets -- which are not the picker's rungs, so a table of them named
    each one by its agent string, or, once that was refused, all of them *the computer*.
    A searching player is named by what it does, *the 6,400-position player*; one the
    ladder names keeps that name, and anything else is `name_of`'s.
    """
    from chess_variant_sandbox.agents.base import parse_agent

    for entry in ladder():
        if entry.agent == agent:
            return entry.name
    try:
        config = parse_agent(agent)
    except ValueError:
        return name_of(agent)
    nodes = config.options.get("nodes")
    if config.kind == "minimax" and isinstance(nodes, (int, float)):
        return f"the {int(nodes):,}-position player"
    if config.kind == "mcts" and isinstance(nodes, (int, float)):
        return f"the {int(nodes):,}-game tree search"
    if config.kind in ("random", "greedy"):
        return config.kind
    return name_of(agent)


def for_game(
    ruleset_hash: str,
    root: str | Path = STORE_DIR,
    fits: str | Path = FIT_DIR,
    valuations: str | Path = VALUE_DIR,
    spec: dict[str, Any] | None = None,
    nets: str | Path = NET_DIR,
) -> list[Opponent]:
    """The ladder, plus every stated plan, fitted objective and searched valuation here,
    and the learned player where an accepted net for this ruleset is stored (Q683).

    Registers each set as a side effect, which is not incidental: ``resolve`` refuses an
    unknown digest rather than defaulting, so an agent string naming a heuristic cannot
    be *built* until its store has been loaded. Listing an opponent and being able to
    play it therefore have to be the same call.

    Index 0 of a *generated* set is skipped. It is the built-in evaluator's own vector,
    so offering it beside ``minimax:nodes=6400`` would be offering the same player twice
    under two names -- see ``SOURCES`` for why that is per-store rather than universal.

    When ``spec`` is provided and the ruleset is supported by Fairy-Stockfish, a
    Fairy-Stockfish opponent is appended -- but only if the binary is installed.

    An entry that would name a second player under a digest already held is left out,
    and only that entry (Q446); `_heuristic_stores` says why, and `refused_stores` puts
    its name on the page.
    """
    roster = ladder()
    roster.extend(fairy_stockfish_for(spec))
    roster.extend(fitted_for(ruleset_hash, fits))
    roster.extend(valued_for(ruleset_hash, valuations))
    roster.extend(learned_for(ruleset_hash, nets, spec))
    for store in _heuristic_stores(ruleset_hash, root):
        for index, heuristic in enumerate(store.heuristics):
            if (store.anchored and index == 0) or index in store.clashes:
                continue
            roster.append(
                Opponent(
                    agent=f"minimax:h={heuristic.digest},nodes={HEURISTIC_NODES}",
                    label=heuristic.name,
                    group=store.group,
                    plan=heuristic.plan,
                    store=store.subdir or "written",
                )
            )
    return roster


@dataclass(frozen=True)
class _HeuristicStore:
    """One of this game's heuristic stores, read whole, as the picker registered it."""

    subdir: str
    group: str
    anchored: bool
    path: Path
    heuristics: tuple[Heuristic, ...]
    # Position in the file -> the player that already holds that entry's digest.
    clashes: dict[int, Heuristic] = field(default_factory=dict)


def _heuristic_stores(ruleset_hash: str, root: str | Path) -> list[_HeuristicStore]:
    """This game's stores in `SOURCES` order, every entry registered on its own.

    **A clash leaves out the entry, never its store** (Q446). A digest names one raw
    vector (Q433), so an entry spelling a held digest with other numbers cannot be
    registered, and `load_set` refuses its whole store for it, all or nothing. A
    `custom/` store built before Q433 can hold such an entry, and through `load_set`
    one old opponent took every opponent built for the game out of the picker with it.
    So the store is read without registering, and each entry is registered or refused by
    itself. Whoever holds the digest first keeps it: what this process registered
    already, then the stores before it in `SOURCES` order, then the entries before it in
    its own file.

    Not in `load_set`, deliberately. A set shorter than its file would move the index-0
    anchor that this and `measure_heuristics` read by position, and `forget_opponent`,
    which saves the set it loaded, would delete the skipped entry from disk. Here the
    whole file is kept, positions and all, and the clash is only not offered.

    A store that will not load at all -- another vocabulary version, a file edited by
    hand -- is still refused whole, as it always was: here that means the ruleset simply
    has no rules of thumb from it, which is what an unmeasured one looks like.
    """
    stores: list[_HeuristicStore] = []
    for subdir, group, anchored in SOURCES:
        path = store_path(ruleset_hash, Path(root) / subdir if subdir else Path(root))
        if not path.exists():
            continue
        try:
            found = HeuristicSet.load(path)
        except (HeuristicError, OSError, ValueError):
            continue
        clashes: dict[int, Heuristic] = {}
        for index, heuristic in enumerate(found.heuristics):
            held = other_spelling(heuristic)
            if held is None:
                register_heuristic(heuristic)
            else:
                clashes[index] = held
        stores.append(
            _HeuristicStore(subdir, group, anchored, path, found.heuristics, clashes)
        )
    return stores


def fairy_stockfish_for(spec: dict[str, Any] | None) -> list[Opponent]:
    """A Fairy-Stockfish opponent for this ruleset, where one is available.

    Returns an empty list when the engine is not installed or this ruleset has no
    counterpart over there. That is what keeps every caller unconditional: the roster
    appends the result without asking, and an empty list is a no-op.

    **Offered only where the two engines have been played against each other**, not
    merely where a translation exists. `chess_variant_sandbox/fairy/variant.py` says why at length:
    a translation that is nearly right is a legal-looking game under rules nobody
    wrote, and the difference is not visible in the configuration. The check is
    memoised per ruleset, so the second visit to a game is free.

    The label carries the node budget for the same reason the ladder's labels do, and
    the plan carries whatever the translation had to approximate -- a player is
    entitled to know that the opponent it is about to face is not being told the game
    has a ply limit.
    """
    from chess_variant_sandbox import fairy

    if spec is None:
        return []
    verified = fairy.verdict(spec)
    if verified is None or not verified.ok:
        return []
    agent = f"fairystockfish:nodes={FAIRY_STOCKFISH_NODES}"
    about = _rung_about(agent, "")
    return [
        Opponent(
            agent=agent,
            label=labelled("Fairy-Stockfish", about),
            group="written by other people",
            plan=_fairy_plan(verified),
            store="fairy",
            name="Fairy-Stockfish",
            about=about,
        )
    ]


def fairy_check(spec: dict[str, Any] | None) -> dict[str, Any]:
    """What the installed engine said about this ruleset, for the page to repeat.

    The same record `fairy_stockfish_for` decides on -- `fairy.verdict`, memoised per
    ruleset -- so the picker and the sentence under it cannot disagree. Three states,
    and they are different news to somebody wondering where the opponent went:

    * ``engine`` is ``None``: no usable Fairy-Stockfish on this machine. Nothing was
      asked, and installing one is the whole repair.
    * ``ok`` is ``True``: the two engines agreed, and the opponent is in the picker.
    * ``ok`` is ``False``: there is an engine, it was asked, and it plays a different
      game. ``reasons`` are the verification's own sentences and ``fen`` the first
      position the two part company over, which is the one thing a reader can act on.
      Six committed rulesets sit here on the 2021 release --
      ``docs/reference/fairy-stockfish.md`` names them and what the build lacks.

    ``ok`` stays ``None`` where there is no translation at all; the library's
    ``fairyReasons`` already carries that answer, and this does not restate it.
    """
    from chess_variant_sandbox import fairy

    check: dict[str, Any] = {
        "engine": None, "ok": None, "reasons": [], "fen": "", "ours": [], "theirs": [],
        "depth": 0, "positions": 0,
    }
    if spec is None or not fairy.available():
        return check
    check["engine"] = fairy.engine_name() or "Fairy-Stockfish"
    if not fairy.supports(spec):
        return check
    result = fairy.verdict(spec)
    if result is None:
        return check
    check.update(
        ok=result.ok,
        reasons=fairy.plain_reasons(list(result.reasons)),
        fen=result.fen,
        ours=list(result.ours),
        theirs=list(result.theirs),
        depth=result.depth,
        positions=result.positions,
    )
    return check


def _fairy_plan(verified: Any) -> str:
    """What this opponent is, and what the translation did not carry across.

    In a visitor's words (P12): what it is and why it is here. The notes are the
    verification's own and name rules; they stay, because a player is entitled to know
    the opponent is not being told part of the game.
    """
    text = (
        "A chess-variant engine written by other people: the one computer player here "
        "that was written somewhere else."
    )
    if verified.notes:
        text += " Where its game differs from this one: " + "; ".join(verified.notes) + "."
    return text


# What a fitted opponent searches. The same budget as the "hard" tier and as a stated
# plan, for the same reason: choosing between the three has to vary the objective and
# nothing else, or the comparison is about how long each one thought.
FITTED_NODES = 6_400

# What the picker calls the two stored objectives, before the acceptance score. Named
# once because `refused_stores` has to call an absent opponent by the name it would have
# had, or the sentence explaining its absence names something nobody can find.
FITTED_LABEL = "taught by this game"
VALUED_LABEL = "own piece values"


def _against(nodes: int, ordinary: bool = False) -> str:
    """The rung a stored objective was measured against: the ladder's player at its
    budget, by name (*hard*, or with `ordinary` *the ordinary hard player*), else the
    ordinary player at that budget."""
    for agent, name, _tail in RUNGS:
        if agent == f"minimax:nodes={nodes}":
            return f"the ordinary {name} player" if ordinary else name
    return f"the ordinary player at {nodes:,} positions"


# Each term of the feature vocabulary as a visitor reads it after *more* or *less*, for
# the sentence a fitted opponent's plan is (P10). `agents/features.GLOSSES` defines each
# term for the model that writes a vector; these name it in a clause. A test holds the
# two key sets together.
TERM_WORDS: dict[str, str] = {
    "material": "material",
    "prey_weight": "weight on what its pieces can take right now",
    "mobility": "freedom to move",
    "advance": "advance up the board",
    "centrality": "hold on the middle of the board",
    "royal_exposure": "pressure on the royal pieces",
    "territory_margin": "ground held",
    "score_margin": "lead on the count",
    "score_urgency": "lead on the count as time runs out",
    "check_margin": "checks given",
    "extinction_margin": "of the kind of piece that must not run out",
    "hand_material": "pieces in hand",
    "promotion_occupancy": "pieces standing ready to promote",
    "reach_progress": "progress towards the winning squares",
    "facing_alignment": "pieces facing forward",
}


def fitted_for(ruleset_hash: str, root: str | Path = FIT_DIR) -> list[Opponent]:
    """The fitted objective for this ruleset, where there is one that was accepted.

    **A refused fit is not offered.** ``FittedObjective.weights`` is the empty tuple when
    the fit lost its acceptance match, and an agent naming it therefore plays the
    built-in objective -- so listing it would be listing ``minimax:nodes=6400`` a second
    time under a better name. That is exactly why index 0 of a heuristic set is skipped,
    and it matters more here: a refusal is the *evidence* the fit did not work, and an
    interface that offered it anyway would be hiding a negative result behind a label.

    The label carries the acceptance score, because "fitted" on its own is a claim and
    that number is what backs it: 0.93 on racing kings means the fitted objective took
    93% of the points against the built-in one at the same budget.
    """
    from chess_variant_sandbox.agents.fitted import FitError, load_fit

    try:
        fit = load_fit(ruleset_hash, root=root)
    except (FitError, OSError, ValueError):
        # Not offered, and not silently: `refused_stores` says why on the page.
        return []
    if fit is None or not fit.accepted:
        return []
    return [
        Opponent(
            agent=f"minimax:fit={fit.procedure},nodes={FITTED_NODES}",
            label=(
                f"{FITTED_LABEL} ({fit.accept_score * 100:.0f}% of the points against "
                f"{_against(FITTED_NODES)})"
            ),
            group="taught by this game's own games",
            plan=_fitted_plan(fit),
            store="fitted",
        )
    ]


def _fitted_plan(fit: Any) -> str:
    """What the fitted vector turned out to care about, as a sentence.

    A weight vector nobody wrote still has to be *stateable* -- that is the whole reason
    `Heuristic.plan` exists -- and the honest statement of a fit is its largest terms and
    their signs. Three of them, because a fourteen-term list is not a sentence, and the
    tail is near zero on every accepted fit in the store.
    """
    from chess_variant_sandbox.agents.features import VOCABULARY

    ranked = sorted(
        ((abs(weight), index) for index, weight in enumerate(fit.vector)), reverse=True
    )
    named = []
    for size, index in ranked[:3]:
        if size < 0.05:
            break
        term = TERM_WORDS.get(VOCABULARY[index], VOCABULARY[index].replace("_", " "))
        named.append(f"{'more' if fit.vector[index] > 0 else 'less'} {term}")
    beat = (
        f"beat {_against(FITTED_NODES, ordinary=True)} over {fit.accept_games} games, "
        "which is why it is offered."
    )
    if not named:
        return f"It learned what to aim for from games of this one, and {beat}"
    return (
        "It learned what to aim for from games of this one: "
        + ", ".join(named)
        + f". It {beat}"
    )


# What a searched-valuation opponent searches: **the budget its own search was run at**,
# read off the procedure rather than restated, so the two cannot drift.
#
# This is the one opponent in the picker whose budget is not a free choice, and stage 31
# measured why. A searched table is a player at one depth and not a better price list: on
# xiangqi it takes 0.74 of the points off the derived table at 6,400 nodes and **0.46** at
# 400, and losalamos goes from 0.61 at 6,400 to 0.37 at 25,600 -- worse than the table it
# replaced, by three and a half standard errors. Offering one at another rung would be
# offering a player nobody measured, and sometimes a worse one under a better name.
# `docs/reference/piece-values.md` §3 has the curves.
VALUED_NODES = SEARCH_PROCEDURE.nodes


def valued_for(ruleset_hash: str, root: str | Path = VALUE_DIR) -> list[Opponent]:
    """The searched piece values for this ruleset, where a search was accepted.

    **This is where stage 31 ships, and where it deliberately stops.** A searched
    valuation is a real improvement in play -- on xiangqi it takes 78% of the points off
    the derived table at the same budget -- and it is *not* on the rated roster, because
    every skill component reads `CANONICAL_AGENTS` to ask whether thinking harder wins,
    and a rung that also priced the pieces differently would beat the rung below it for
    two reasons at once. Costing nothing and buying the play experience is the whole
    trade; see `docs/reference/piece-values.md`.

    **A refused search is not offered**, for the reason a refused fit is not: its
    `values` resolve to the empty tuple, so an agent naming it plays the ordinary agent,
    and listing it would list `minimax:nodes=6400` a second time under a better name.
    """
    from chess_variant_sandbox.agents.valuations import ValuationError, load_valuation

    try:
        # Q142, Q430. The weight vector a searched table carries is `builtin_vector` *as
        # it was when the search ran*, and a store searched before a vocabulary bump
        # holds one of the old length. The core refuses it, and it is right to: read as
        # this vocabulary it would be a different player under an unchanged name.
        # Offered, it failed the first game anybody started against it, and failed the
        # whole on-demand rating with it. The loader now refuses it, so it lands here,
        # and `refused_stores` puts the reason on the page.
        valuation = load_valuation(ruleset_hash, root=root)
    except (ValuationError, OSError, ValueError):
        return []
    if valuation is None or not valuation.accepted:
        return []
    return [
        Opponent(
            agent=f"minimax:values={valuation.procedure},nodes={VALUED_NODES}",
            label=(
                f"{VALUED_LABEL} ({valuation.accept_score * 100:.0f}% of the points "
                f"against {_against(VALUED_NODES)})"
            ),
            group="with this game's own piece values",
            plan=_valued_plan(valuation),
            store="valuations",
        )
    ]


def _valued_plan(valuation: Any) -> str:
    """What the search decided, as a sentence about the pieces that moved.

    A table nobody wrote still has to be stateable, exactly as a fitted vector does, and
    the honest statement of this one is short by construction: coordinate ascent keeps a
    step only when it clears two measured standard errors, so an accepted valuation
    usually moves one or two pieces and says so in a clause.
    """
    moved = [
        f"{piece} at {ratio:.2f}×"
        for piece, ratio in zip(valuation.piece_ids, valuation.ratios, strict=True)
        if abs(ratio - 1.0) > 1e-9
    ]
    rung = _against(VALUED_NODES)
    tuned = (
        f"The values were tuned by trial games, and it took "
        f"{valuation.accept_score * 100:.0f}% of the points against {rung} over "
        f"{valuation.accept_games} games, which is why it is offered."
    )
    if not moved:  # pragma: no cover - an accepted search always moved something
        return f"Plays like {rung} with its own piece values. {tuned}"
    return (
        f"Plays like {rung} but values " + ", ".join(moved)
        + f" of what the rules imply. {tuned}"
    )


# ------------------------------------------------------------ the learned player (Q683)

# What the picker calls the learned player, and the heading it sits under. One player per
# game at most, so the name says which kind of thing it is rather than which net.
LEARNED_NAME = "the learned player (experimental)"
LEARNED_GROUP = "that taught itself this game"


def never_rated(agent: str) -> bool:
    """Whether a game against this player can never be rated: one that plays a learned net.

    Read off the agent string, so it holds wherever a seat is checked -- the site's
    `matches.start` and its rating seeds -- without a roster in hand. A net player's name
    does not say which net sat down (`agents/nets.py`), so no number it earned could be
    filed under it honestly, and the user asked for it as an experiment to try (Q683).
    """
    from chess_variant_sandbox.agents.base import parse_agent

    try:
        config = parse_agent(agent)
    except ValueError:
        return False
    return config.options.get("net") is not None


@dataclass(frozen=True)
class _Learned:
    """An accepted net's record, as much of it as the picker reads."""

    procedure: str
    agent: str
    parent: str
    score: float
    games: int
    nodes: int
    # The plan's verdict on this generation (`plan-teacher.md` §9.6's R5-1, Q706): True
    # for a kept one, False for one the plan read and did not keep (an accepted child
    # that failed G-P, G-N, G-C or a guard), None where no verdict was written -- a net
    # outside any generation line, such as a guard game's strongest net. The trainer
    # cannot write it, since keeping needs gates it does not play; the `kept` field in
    # the store's JSON is written when the plan records the verdict.
    kept: bool | None = None


def _learned_record(path: Path, ruleset_hash: str) -> _Learned | None:
    """The accepted net a store's JSON describes, or ``None`` if it is not one to offer.

    Offered only at the budget its acceptance duel was played at, read off the record's
    own ``acceptance.candidate``: a net is a player at one budget the way a searched
    valuation is, and offering it at another would be offering a player nobody measured.
    """
    from chess_variant_sandbox.agents.base import parse_agent
    from chess_variant_sandbox.agents.nets import ACCEPTED, normalise_procedure

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("verdict") != ACCEPTED:
        return None
    if data.get("ruleset_hash") != ruleset_hash:
        return None
    acceptance = data.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("accepted") is not True:
        return None
    procedure = str(data.get("procedure", ""))
    agent = str(acceptance.get("candidate", ""))
    try:
        config = parse_agent(agent)
    except ValueError:
        return None
    nodes = config.options.get("nodes")
    named = config.options.get("net")
    if (config.kind != "minimax" or not isinstance(nodes, int) or named is None
            or normalise_procedure(named) != procedure):
        return None
    lineage = data.get("lineage")
    if not isinstance(lineage, dict):
        lineage = {}
    kept = data.get("kept")
    return _Learned(
        procedure=procedure,
        agent=agent,
        parent=str(lineage.get("parent") or ""),
        score=float(acceptance.get("score", 0.0)),
        games=int(acceptance.get("games", 0)),
        nodes=nodes,
        kept=kept if isinstance(kept, bool) else None,
    )


def _kept(found: list[_Learned]) -> _Learned | None:
    """The net the picker offers: the newest *kept* generation, not the newest accepted.

    By `plan-teacher.md` R5-6 a child that passes the acceptance duels is stored as
    accepted whatever the rest of the gates read, and only *kept* makes it a parent and
    the line's player (Q706: T5-g2's generation 2 was accepted and not kept). So:

    - a net the plan did not keep (``kept: false``) is never offered;
    - where any net here is kept, the offer is among the kept ones only, so an accepted
      child stored beside its kept parent, with or without a verdict written, does not
      replace it;
    - where none is (a game no generation line runs on), the newest accepted net is
      offered, as before.

    Among those, the newest is the one no other candidate was trained from; two with no
    child go to the higher acceptance score, then the procedure, so the answer never
    depends on the order files are listed in.
    """
    found = [entry for entry in found if entry.kept is not False]
    kept = [entry for entry in found if entry.kept]
    return _newest(kept or found)


def _newest(found: list[_Learned]) -> _Learned | None:
    """The net no other one here was trained from: the latest generation of those given."""
    parents = {entry.parent for entry in found}
    leaves = [entry for entry in found if entry.procedure not in parents]
    if not leaves:
        return None
    return max(leaves, key=lambda entry: (entry.score, entry.procedure))


def learned_for(
    ruleset_hash: str, root: str | Path = NET_DIR, spec: dict[str, Any] | None = None
) -> list[Opponent]:
    """The learned player for this ruleset, where an accepted net is stored and sound.

    Derived from the stores alone (`nets/<hash>-<procedure>.json`, verdict *accepted*),
    so it appears on exactly the games a net was accepted on and names none of them. Of
    several, the newest kept generation (`_kept`, Q706). **A
    net that is not there is not offered**, and nothing else changes: a missing ``.net``,
    weights that do not hash to the recorded SHA-256 (`nets.load_net` refuses both), a
    store that will not parse, or one whose input layout is not this ruleset's. A ``net=``
    player with no usable net refuses to play (`nets.require_net`, Q709), so offering one
    would offer a game that cannot start; before Q709 it played the built-in objective
    under the learned player's name, *hard* a second time under a better one.

    Loading it registers it, which is what the game started from this entry then plays
    (`nets.net_bytes` reads the registry first) -- listing a player and being able to
    play it are one call, as for a rule of thumb.

    It is **practice only**: never rated, never in the on-demand rating run, and the plan
    says so in the sentence a person reads under the menu.
    """
    from chess_variant_sandbox.agents.nets import NetError, load_net

    records = [
        found
        for path in sorted(Path(root).glob(f"{ruleset_hash}-*.json"))
        if (found := _learned_record(path, ruleset_hash)) is not None
    ]
    chosen = _kept(records)
    if chosen is None:
        return []
    try:
        stored = load_net(ruleset_hash, chosen.procedure, root=root)
    except (NetError, OSError, ValueError):
        return []
    if stored is None or not stored.playing or not _fits(stored.weights, spec):
        return []
    about = _rung_about(chosen.agent, "")
    return [
        Opponent(
            agent=chosen.agent,
            label=labelled(LEARNED_NAME, about),
            group=LEARNED_GROUP,
            plan=_learned_plan(chosen),
            store="learned",
            name=LEARNED_NAME,
            about=about,
            practice_only=True,
        )
    ]


def _fits(weights: bytes, spec: dict[str, Any] | None) -> bool:
    """Whether a net's store parses and reads as many inputs as this ruleset's layout.

    The core refuses a net of another layout when the first move is asked for; asking it
    here keeps that player out of the menu instead. Without a spec only the parse is
    checked, and the start of a game is where a mismatch would then surface.
    """
    from struct import error as StructError

    from chess_variant_sandbox.agents.nets import Layout, NetError

    try:
        net = _parsed(weights)
    except (NetError, StructError):
        return False
    if spec is None:
        return True
    try:
        from chess_variant_sandbox.web.library import engine_for

        layout = Layout.of(engine_for(spec).spec)
    except Exception:  # noqa: BLE001 - a ruleset that will not compile offers no net
        return False
    return net.n_inputs == layout.n_inputs


def _parsed(weights: bytes) -> Any:
    """A store's parse, kept per weights digest: a game page lists its roster per visit."""
    import hashlib

    from chess_variant_sandbox.agents.nets import Net

    digest = hashlib.sha256(weights).hexdigest()
    held = _PARSED.get(digest)
    if held is None:
        held = _PARSED[digest] = Net.parse(weights)
    return held


_PARSED: dict[str, Any] = {}


def _learned_plan(learned: _Learned) -> str:
    """What the learned player is, why it is here, and that a game against it is practice.

    In a visitor's words: no net, no procedure, no node count beyond the strength line
    above it. The share is the acceptance duel's, the number that is the reason it is
    offered, as the fitted and valued players' plans give theirs.
    """
    rung = _against(learned.nodes, ordinary=True)
    return (
        "It taught itself this game by playing against itself, and it took "
        f"{learned.score * 100:.0f}% of the points from {rung} over "
        f"{learned.games} games, both looking at the same number of positions. It is "
        "not rated, so a game against it is practice: it counts towards nobody's rating "
        "and no game's score."
    )


# ------------------------------------------------ stored objectives this build refuses


def _foreign_length(weights: Any) -> str:
    """Why a stored weight vector is not one this build's vocabulary can read, or "".

    Empty is the built-in objective and always readable. Anything else must be exactly
    one weight per `VOCABULARY` term, which is the check `AgentSpec::from_parts` makes on
    the far side of the boundary; asking it here is what lets the page name the store
    rather than the core name the agent kind.
    """
    from chess_variant_sandbox.agents.features import VOCABULARY

    if not weights or len(weights) == len(VOCABULARY):
        return ""
    return (
        f"carries {len(weights)} weights where this build's feature vocabulary has "
        f"{len(VOCABULARY)} terms"
    )


def _stored_refusal(path: Path) -> bool:
    """Whether a store this build will not load says it was refused anyway.

    Read off the raw file because the loader has already said no. Anything unreadable, or
    a file that does not say, is not known to be a refusal, so the page still names it.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("accepted") is False


def _ruleset_argument(path: str | Path | None, ruleset_hash: str) -> str:
    """The ruleset as the repair command should name it: relative where it can be."""
    if path is None:
        return f"<the ruleset file for {ruleset_hash}>"
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def refused_stores(
    ruleset_hash: str,
    ruleset_path: str | Path | None = None,
    fits: str | Path = FIT_DIR,
    valuations: str | Path = VALUE_DIR,
    heuristics: str | Path = STORE_DIR,
) -> list[str]:
    """Why a stored objective for this ruleset is kept out of the picker, and the repair.

    The sentences of `refusals`, which says what this one does."""
    return [said for _, said in refusals(ruleset_hash, ruleset_path, fits, valuations,
                                         heuristics)]


def unavailable(name: str) -> str:
    """The site's sentence for one player `refusals` names (G19): no file, no command."""
    return f"One computer player, {name}, is unavailable on this game for now."


def refusals(
    ruleset_hash: str,
    ruleset_path: str | Path | None = None,
    fits: str | Path = FIT_DIR,
    valuations: str | Path = VALUE_DIR,
    heuristics: str | Path = STORE_DIR,
) -> list[tuple[str, str]]:
    """Each stored player left out of the picker, by name, and the maintainer's sentence:
    why it is kept out of the picker, and the repair.

    **A stale store is refused, never played as the built-in objective** (Q142). The
    other way to meet one -- fall back to `minimax:nodes=6400` and say so -- was open, and
    it is the failure this project ranks worst: an agent string that names a fitted or
    searched player while the built-in one plays under it. The on-demand rating would
    then file that player's Elo under the stored player's name, where it outlives the
    note that explained it. So the store is left out of the roster (and with it out of
    every game and every rating run, which all build from the roster), and this is the
    sentence the page shows instead: what is stale, and the command that repairs it.

    One sentence per store, and only for a store that was *accepted*: a refused fit or
    search was never offered, stale or not, and saying it is "not offered" would be news
    about nothing. Three states are named. A searched valuation whose weight vector
    predates a vocabulary bump, which is the one the core met on crazyhouse. A current
    fit store that will not load. And an accepted fit filed only under a procedure this
    build no longer runs, which `load_fit` never looks for and which would otherwise
    vanish from the picker without a word.

    Before them, one sentence per heuristic the picker left out of a store it offers the
    rest of, because it spells a held digest with other numbers (Q446): a `custom/` store
    built before Q433 can hold one. It names the entry, the player holding the digest,
    and the file it stays in. There is no command to name: taking it out of that file is
    the repair, and until then `build_opponent` refuses to add to the store. The stores
    are read by `_heuristic_stores`, registering and all, exactly as `for_game` reads
    them, so the sentence and the picker cannot disagree about who was left out.
    """
    from chess_variant_sandbox.agents.fitted import FitError, load_fit, procedure_digest
    from chess_variant_sandbox.agents.fitted import store_path as fit_path
    from chess_variant_sandbox.agents.valuations import ValuationError, load_valuation
    from chess_variant_sandbox.agents.valuations import store_path as value_path

    ruleset = _ruleset_argument(ruleset_path, ruleset_hash)
    said: list[tuple[str, str]] = [
        (store.heuristics[index].name,
         f"The opponent '{store.heuristics[index].name}' is not offered because it has "
        f"the balance of '{held.name}' with other exact numbers, and the engine plays the "
        "exact numbers, so one name would call two players; it is still stored in "
        f"`{store.path.as_posix()}`, and taking it out of that file is the repair.")
        for store in _heuristic_stores(ruleset_hash, heuristics)
        for index, held in sorted(store.clashes.items())
        # An anchor is never offered, so its absence is news about nothing.
        if not (store.anchored and index == 0)
    ]

    def refuse(label: str, what: str, command: str, repair: str) -> None:
        said.append((
            label,
            f"The opponent '{label}' is not offered because {what}; "
            f"`chess-variant-sandbox {command} {ruleset} --force` {repair}.",
        ))

    here = value_path(ruleset_hash, SEARCH_PROCEDURE.digest, valuations).as_posix()
    try:
        # A store that loads is one this build can play: `Valuation.from_dict` refuses a
        # weight vector of another vocabulary's length (Q430), so that is the one stale
        # state a searched table has, and it arrives here as an error.
        load_valuation(ruleset_hash, root=valuations)
    except (ValuationError, OSError, ValueError) as error:
        # Refused whatever its verdict, so whether it was ever offered is read off the
        # file itself: a refused search was not, and its staleness is not news.
        if not _stored_refusal(Path(here)):
            refuse(VALUED_LABEL, f"the search stored in {here} cannot be read ({error})",
                   "values", "searches it again")

    current = procedure_digest()
    here = fit_path(ruleset_hash, current, fits).as_posix()
    try:
        fit = load_fit(ruleset_hash, root=fits)
    except (FitError, OSError, ValueError) as error:
        refuse(FITTED_LABEL, f"the fit stored in {here} cannot be read ({error})",
               "fit", "refits it")
        return said
    if fit is not None:
        return said
    for path in sorted(Path(fits).glob(f"{ruleset_hash}-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not data.get("accepted"):
            continue
        procedure = str(data.get("procedure", path.stem.rsplit("-", 1)[-1]))
        if procedure == current:
            continue
        length = _foreign_length(data.get("weights") or ())
        refuse(
            FITTED_LABEL,
            f"the only accepted fit stored for it, {path.as_posix()}, was made by "
            f"procedure {procedure}, which this build no longer runs"
            + (f", and {length}" if length else ""),
            "fit",
            f"refits it under procedure {current}",
        )
        break
    return said


def agents_of(roster: list[Opponent]) -> list[str]:
    """The agent strings in a roster to rate, minus the seat a person sits in.

    Minus a practice-only player too (Q683): what this feeds is the on-demand rating run,
    and a table that held the learned player would seed it on the site as a rated one.
    """
    return [
        entry.agent for entry in roster
        if entry.agent != HUMAN and not entry.practice_only and not never_rated(entry.agent)
    ]


# ------------------------------------------------------- rating the whole picker


def stored_path(ruleset_hash: str, runs_dir: str | Path) -> Path:
    """Where the picker's own rating table lives.

    Named `opponents.json`, and that matters rather than being a preference: the library
    finds the rated ladder's table by globbing for `tournament.json`, and a second file
    by that name anywhere under `runs/` would be picked up as one. Nothing under
    `metrics/` looks at this path at all.
    """
    return Path(runs_dir) / "opponents" / f"{ruleset_hash}.json"


# ------------------------------------------------- the ladder's stamp and its games (L3)

# `docs/plan-leaderboard.md` §2.2. A ladder written since L3 says what it rated, on what
# build and when, and keeps its games beside it, so a newly offered player can be added by
# playing its games alone (a *top-up*) and a reader can tell a ladder whose players have
# since changed. A file without `format` is one from before, read as before: its standings
# seed the site, and it can only be rated again whole.
LADDER_FORMAT = 2

# The changes the maintainer marks as changing how the computer players play, by the date
# each landed on main. A ladder rated before the latest of them has *old players*
# (`deploy/ladders_check.py`), and `deploy/ladders.py --rerate` rates it again. **Not every
# core rebuild:** a ladder is a seed that people's games correct, not a score, so it carries
# the build fingerprint (`built`) for the record and is never called stale by it alone;
# re-rating the library on each fingerprint would cost 30 to 60 slot-hours a time for
# numbers people move anyway. A change goes here when it changes what the picker's players
# do on the board -- a search repair turned on by default, a new evaluation term -- and the
# entry is a dated sentence, so the report can say which change a ladder predates.
PLAYER_CHANGES: tuple[tuple[str, str], ...] = (
    ("2026-09-29", "Q502i turned threat-aware quiescence on for every minimax player"),
)


def games_path(ruleset_hash: str, runs_dir: str | Path) -> Path:
    """Where a picker ladder's own games are kept, beside its table.

    Never `tournament.json` or `games.jsonl` under a run directory, for `stored_path`'s
    reason: nothing under `metrics/` reads these, and nothing that globs for a measured
    corpus can find them here.
    """
    return Path(runs_dir) / "opponents" / f"{ruleset_hash}.games.jsonl"


def _today() -> str:
    """The date a ladder is stamped with, UTC. A seam for the tests."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).date().isoformat()


def _built() -> str:
    from chess_variant_sandbox.measurement import built_fingerprint

    return built_fingerprint()


def ladder_date(table: dict[str, Any] | None, path: str | Path | None = None) -> str | None:
    """When a ladder's games were played: its stamp, else its file's modification date.

    The stamp's `rated` is the date of the whole rating; a top-up keeps it, because the
    older games are still in the fit. A ladder from before L3 has no stamp, and its file's
    date on this machine is the honest stand-in (a copy elsewhere may have lost it).
    """
    if isinstance(table, dict) and isinstance(table.get("rated"), str):
        return str(table["rated"])
    if path is None:
        return None
    from datetime import datetime, timezone

    try:
        stamp = Path(path).stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(stamp, tz=timezone.utc).date().isoformat()


def changed_since(date: str | None) -> tuple[tuple[str, str], ...]:
    """The marked player changes a ladder played on `date` predates, oldest first.

    A ladder with no date at all predates every one of them: nothing says otherwise.
    """
    return tuple(change for change in PLAYER_CHANGES if date is None or date < change[0])


def _game_line(record: Any) -> dict[str, Any]:
    """What a kept game holds: what the fit reads and what reproduces it, not its moves.

    `tournament.ratings.rate` reads the two seats and the winner; the seed and the seats
    are the game's reproduction triple with the ruleset's hash, which the file name holds.
    So the file stays a few kilobytes a game and travels with `runs/opponents/` cheaply.
    """
    return {
        "seed": int(record.seed),
        "agents": list(record.agents),
        "winner": record.winner,
        "reason": str(record.reason),
        "plies": int(record.plies),
    }


def read_games(ruleset_hash: str, runs_dir: str | Path) -> list[Any] | None:
    """A ladder's kept games as records the fit reads, or ``None`` when none are kept.

    A line that will not parse makes the whole file unreadable rather than shorter: a fit
    over some of a ladder's games would be a different ladder under the same name.
    """
    from chess_variant_sandbox.tournament.record import GameRecord

    path = games_path(ruleset_hash, runs_dir)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    records = []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            records.append(GameRecord(
                ruleset_hash=ruleset_hash,
                ruleset_name="",
                seed=int(data["seed"]),
                agents=(str(data["agents"][0]), str(data["agents"][1])),
                moves=[],
                winner=None if data["winner"] is None else int(data["winner"]),
                reason=str(data.get("reason", "")),
                plies=int(data.get("plies", 0)),
            ))
        except (ValueError, KeyError, IndexError, TypeError):
            return None
    return records


def _write_ladder(
    ruleset_hash: str, runs_dir: str | Path, table: dict[str, Any], records: list[Any]
) -> Path:
    """Write the games, then the table that names them: a table is never ahead of them."""
    path = stored_path(ruleset_hash, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = games_path(ruleset_hash, runs_dir)
    kept.write_text(
        "".join(json.dumps(_game_line(record), separators=(",", ":")) + "\n"
                for record in records),
        encoding="utf-8",
    )
    path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    return path


def move_aside(ruleset_hash: str, runs_dir: str | Path, date: str | None = None) -> Path | None:
    """Move a game's ladder and its games to `runs/opponents/before-<date>/`, never deleting.

    `--rerate`'s first step (`plan-launch-set.md` §4.3, `plan-leaderboard.md` §2.2). A
    second re-rating on one day goes to `before-<date>.2/` and so on, so nothing is ever
    written over. Returns the directory, or ``None`` when there was no ladder to move.
    """
    path = stored_path(ruleset_hash, runs_dir)
    if not path.exists():
        return None
    date = date or _today()
    base = path.parent / f"before-{date}"
    aside, count = base, 1
    while (aside / path.name).exists():
        count += 1
        aside = path.parent / f"before-{date}.{count}"
    aside.mkdir(parents=True, exist_ok=True)
    kept = games_path(ruleset_hash, runs_dir)
    if kept.exists():
        kept.replace(aside / kept.name)
    path.replace(aside / path.name)
    return aside


def restore(ruleset_hash: str, runs_dir: str | Path, aside: Path) -> None:
    """Put a moved-aside ladder back, for a re-rating that failed: a game never loses one.

    Only where nothing has taken its place, which a failed rating never writes.
    """
    path = stored_path(ruleset_hash, runs_dir)
    kept = games_path(ruleset_hash, runs_dir)
    for moved, home in ((aside / kept.name, kept), (aside / path.name, path)):
        if moved.exists() and not home.exists():
            moved.replace(home)
    with contextlib.suppress(OSError):
        aside.rmdir()  # only when empty, which is when it held nothing else


class LadderError(ValueError):
    """A top-up asked of a ladder that cannot take one: rate it again whole."""


@dataclass(frozen=True)
class _SpineConfig(TournamentConfig):
    """A schedule in which challengers meet the spine and never each other.

    `run_tournament` asks the config how many games a pairing gets, which is the seam
    this needs and the reason no change to `tournament/` was required. Returning zero
    drops a pairing entirely, because the scheduler loops `range(games_for(...))`.

    `fresh`, for a top-up, narrows that to the pairings a ladder lacks: a newly offered
    player against each spine player, each seat, and nothing the kept games already hold.
    """

    spine: tuple[str, ...] = ()
    fresh: tuple[str, ...] = ()

    def games_for(self, first: str, second: str) -> int:
        if self.fresh:
            if (first in self.spine and second in self.fresh) or (
                second in self.spine and first in self.fresh
            ):
                return self.games_per_pairing
            return 0
        if first in self.spine or second in self.spine:
            return self.games_per_pairing
        return 0

    def as_dict(self) -> dict[str, Any]:
        found = {**super().as_dict(), "spine": list(self.spine)}
        if self.fresh:
            found["fresh"] = list(self.fresh)
        return found


@dataclass(frozen=True)
class _Plan:
    """What one rating run will play, decided before any game starts."""

    config: _SpineConfig
    entries: tuple[str, ...]
    # For a top-up: the ladder it extends and that ladder's games.
    held: dict[str, Any] | None = None
    kept: tuple[Any, ...] = ()


def _plan(
    entries: list[str],
    games: int,
    seed: int,
    held: dict[str, Any] | None = None,
    kept: list[Any] | None = None,
) -> _Plan:
    """The schedule: the whole picker, or for a top-up only the newcomers' spine games.

    A top-up's seeds start one past the highest seed in the kept games, so a ladder's
    games never share a seed and the whole schedule is fixed before a thread starts. It
    plays at the ladder's own games per pairing, and against the ladder's own spine.
    """
    if held is None:
        spine = tuple(agent for agent in SPINE if agent in entries)
        config = _SpineConfig(
            agents=tuple(entries),
            games_per_pairing=games,
            seed=seed,
            include_self_play=False,
            collect_series=False,
            spine=spine,
        )
        return _Plan(config, tuple(entries))

    if held.get("format") != LADDER_FORMAT or kept is None:
        raise LadderError(
            "this ladder keeps no games (it predates the stamp), so a top-up has nothing "
            "to fit beside; rate it again whole with --rerate"
        )
    if len(kept) != int(held.get("games", -1)):
        raise LadderError(
            f"this ladder says {held.get('games')} games and {len(kept)} are kept beside "
            "it; rate it again whole with --rerate"
        )
    rows = {str(row["name"]) for row in held.get("standings") or ()}
    fresh = tuple(agent for agent in entries if agent not in rows)
    spine = tuple(str(agent) for agent in held.get("spine") or ())
    if any(agent in SPINE for agent in fresh) or not spine:
        raise LadderError(
            "a spine player is new to this ladder, so every other player would need its "
            "games again; rate it again whole with --rerate"
        )
    config = _SpineConfig(
        agents=(*spine, *fresh),
        games_per_pairing=int(held.get("games_per_pairing", games)),
        seed=1 + max((int(record.seed) for record in kept), default=seed - 1),
        include_self_play=False,
        collect_series=False,
        spine=spine,
        fresh=fresh,
    )
    return _Plan(config, (*[str(a) for a in held.get("roster") or ()], *fresh),
                 held, tuple(kept))


@dataclass
class RatingRun:
    """One on-demand rating of a ruleset's picker, in progress or finished."""

    ruleset_hash: str
    name: str
    total: int
    played: int = 0
    state: str = "running"  # running | finished | failed
    error: str = ""
    table: dict[str, Any] | None = None
    started: float = field(default_factory=time.monotonic)
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=60))

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.ruleset_hash,
            "name": self.name,
            "state": self.state,
            "error": self.error,
            # `played` is 0 until the core's games come back, which they do in one
            # piece: the core plays them on its threads and has nothing to report until
            # they are all done. `run_tournament` then calls back once for them and once
            # for each of Fairy-Stockfish's games, where there are any, as the Python loop
            # plays them (Q446). Elapsed seconds is what this can honestly offer before.
            "played": self.played,
            "total": self.total,
            "seconds": round(time.monotonic() - self.started, 1),
            "table": self.table,
            "log": list(self.lines),
        }


class Ratings:
    """On-demand rating runs, one per ruleset, on a thread.

    Same shape as `evolving.Runs` and for the same reason: this is minutes of tournament
    inside a process that also has to keep serving the page, so it cannot happen in a
    request. A finished table is written to disk and re-read on the next visit, so the
    cost is paid once per ruleset.
    """

    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._runs: dict[str, RatingRun] = {}
        self._lock = threading.Lock()

    def stored(self, ruleset_hash: str) -> dict[str, Any] | None:
        """A table from an earlier run, if there is one."""
        path = stored_path(ruleset_hash, self.runs_dir)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def get(self, ruleset_hash: str) -> RatingRun | None:
        with self._lock:
            return self._runs.get(ruleset_hash)

    def start(
        self,
        spec: dict[str, Any],
        ruleset_hash: str,
        name: str,
        roster: list[Opponent],
        games: int = 2,
        seed: int = 0,
        top_up: bool = False,
    ) -> RatingRun:
        """Begin rating this ruleset's picker, or return the run already doing it.

        `top_up` extends the ladder on disk instead of replacing it (L3): each player the
        picker offers and the ladder has no row for plays the ladder's spine, each seat,
        and the fit is redone over the kept games and the new ones together. A ladder that
        cannot take one (no kept games, a spine player new to it) is a run that failed at
        once, with the reason, and nothing written.
        """
        with self._lock:
            existing = self._runs.get(ruleset_hash)
            if existing is not None and existing.state == "running":
                return existing
            entries = agents_of(roster)
            try:
                held = self.stored(ruleset_hash) if top_up else None
                if top_up and held is None:
                    raise LadderError("there is no ladder here to top up; rate it whole")
                plan = _plan(entries, games, seed, held,
                             read_games(ruleset_hash, self.runs_dir) if top_up else None)
            except LadderError as error:
                run = RatingRun(ruleset_hash=ruleset_hash, name=name, total=0,
                                state="failed", error=str(error))
                self._runs[ruleset_hash] = run
                return run
            from chess_variant_sandbox.tournament.runner import schedule

            run = RatingRun(
                ruleset_hash=ruleset_hash,
                name=name,
                # A top-up with nobody new plays nothing (`_play` returns the ladder).
                total=(0 if plan.held is not None and not plan.config.fresh
                       else len(schedule(plan.config))),
            )
            self._runs[ruleset_hash] = run

        thread = threading.Thread(
            target=self._play,
            args=(run, spec, roster, plan),
            daemon=True,
        )
        thread.start()
        return run

    def _play(
        self,
        run: RatingRun,
        spec: dict[str, Any],
        roster: list[Opponent],
        plan: _Plan,
    ) -> None:
        from chess_variant_sandbox.tournament.ratings import rate

        try:
            config = plan.config
            if plan.held is not None and not config.fresh:
                # Nothing offered is missing from the ladder: a top-up with no games.
                run.table = plan.held
                run.lines.append("every offered player has a row; nothing to add")
                run.state = "finished"
                return
            run.lines.append(
                f"{len(config.fresh)} new opponents against the spine, {run.total} games"
                if plan.held is not None else
                f"{len(plan.entries)} opponents, {run.total} games"
            )
            # One schedule over the whole picker. The runner sends the games whose seats
            # both have a native form to the core, on its threads, and plays only the
            # ones with Fairy-Stockfish in the Python loop (Q439), so this no longer
            # splits the roster itself as Q432 had to. It tells `played` as games come
            # back, which the page's count reads while the outside engine's games run.
            def count(played: int) -> None:
                run.played = played

            result = run_tournament(spec, config, on_progress=count)
            run.played = len(result.records)
            records = [*plan.kept, *result.records]
            # The fit over every game the ladder holds. For a whole run that is exactly
            # the runner's own table; for a top-up it is the old games and the new.
            table = rate(records) if plan.held is not None else result.ratings
            labels = {entry.agent: entry for entry in roster}
            before = {
                str(row["name"]): row for row in (plan.held or {}).get("standings") or ()
            }

            def said(name: str, key: str) -> str:
                # The live roster's words, else what the ladder said when it rated the
                # player (one no longer offered keeps its row and its name), else its
                # agent string for the label.
                entry = labels.get(name)
                if entry is not None:
                    return str(getattr(entry, key))
                return str(before.get(name, {}).get(key, name if key == "label" else ""))

            stamp: dict[str, Any] = {
                "format": LADDER_FORMAT,
                "roster": list(plan.entries),
                "spine": list(config.spine),
                "rated": _today(),
                "built": _built(),
                "games_file": games_path(run.ruleset_hash, self.runs_dir).name,
                "top_ups": [],
            }
            if plan.held is not None:
                # A top-up keeps the ladder's own stamp -- its older games are still in
                # the fit, so its date and build are still what most of it rests on --
                # and adds a line saying who joined, when and on what build.
                stamp.update(
                    rated=plan.held.get("rated"),
                    built=plan.held.get("built"),
                    top_ups=[
                        *(plan.held.get("top_ups") or ()),
                        {"players": list(config.fresh), "games": len(result.records),
                         "seed": config.seed, "rated": _today(), "built": _built()},
                    ],
                )
            run.table = {
                "ruleset_hash": run.ruleset_hash,
                "games_per_pairing": config.games_per_pairing,
                "seed": (plan.held or {}).get("seed", config.seed),
                "games": len(records),
                **stamp,
                "standings": [
                    {
                        **standing,
                        "label": said(standing["name"], "label"),
                        "group": said(standing["name"], "group"),
                        "plan": said(standing["name"], "plan"),
                    }
                    for standing in table.as_dict()["standings"]
                ],
            }
            path = _write_ladder(run.ruleset_hash, self.runs_dir, run.table, records)
            run.lines.append(f"rated {len(run.table['standings'])} opponents -> {path}")
            run.state = "finished"
        except Exception as error:  # noqa: BLE001 - a thread must report, not vanish
            run.state = "failed"
            run.error = f"{type(error).__name__}: {error}"
            run.lines.append(traceback.format_exc().strip().splitlines()[-1])
