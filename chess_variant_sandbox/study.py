"""The study record: a game's stored computer games as an opening tree and six example
games (`docs/plan-website.md`, *The study page*, R-W4b1).

`runs/study/<hash>.json`, beside `runs/puzzles/` and `runs/lessons/`, because the site is
never sent a `runs/<hash>-<seed>/` directory. Built from the measurement's records through
the reader the lesson shares (`stored_games`).

**Rule 3 holds by construction.** Python groups the records by the moves they already
hold, counts them, and reads the winner each record already states. The engine walks the
stored moves (`Engine.steps`), words each (`describe_move`) and, in `check`, answers
whether each stored step is still legal (`legal_moves`, `legal_moves_for`). Nothing here
lists a move to choose from, picks a move, values a position or decides an outcome, and no
search runs. Nothing here names a game, a piece or a rule: the module reads the compiled
spec and the records.
"""

from __future__ import annotations

import datetime
import json
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

from chess_variant_sandbox import stored_games
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.stored_games import Stored

FORMAT = 1
"""The record's shape. A record in an older format is owed again."""

STUDY_VERSION = 1
"""What this module reads into a record: the tree's rules, the six example rules, the
pool. A change to any of them is a new version, and every record is owed again."""

STUDY_DEPTH = 24
"""How many steps deep the tree goes (fixed by the plan from the size alone)."""

STUDY_FLOOR = 4
"""The fewest games that make a step a branch: one pairing's games in one run."""

MIN_POOL = 32
"""Under this many pool games there is no tree, only the reason and the examples."""

DRAW_SHARE = 0.1
"""The share of the pool that must be drawn for a draw to be an example."""

TURNED = 1.0
"""How far behind, in the record's material units, a winner must have been to have
turned the game round."""

DIRECTORY = "study"

SOURCE_LIBRARY = "library records"
SOURCE_SITE = "the site's measurement"
SOURCE_SPOT_CHECKED = "measured on its designer's machine and spot-checked"

LINE, FIRST_WON, SECOND_WON, TURNED_ROUND, QUICKEST, DRAW = (
    "line", "first_won", "second_won", "turned", "quickest", "draw")
RULES = (LINE, FIRST_WON, SECOND_WON, TURNED_ROUND, QUICKEST, DRAW)
"""The example rules, in the order they pick (a game is taken once)."""

NO_RECORDS = ("No computer games are stored for this game yet. They are played when it "
              "is measured.")


def record_path(runs_dir: Path | str, digest: str) -> Path:
    return Path(runs_dir) / DIRECTORY / f"{digest}.json"


# --------------------------------------------------------------------------- players


def budget(agent: str) -> int | None:
    """The node budget a seat plays at, read as `web/play._budget` reads it: the budget it
    names, else its kind's default off the core's parameters; None for a player that does
    not search (a random mover, a one-move greedy player) or that cannot be read."""
    from chess_variant_sandbox.agents.base import AgentConfig, native_spec, parse_agent

    try:
        config = parse_agent(agent)
    except ValueError:
        return None
    named = config.options.get("nodes")
    if isinstance(named, (int, float)) and not isinstance(named, bool):
        return int(named) if named > 0 else None
    try:
        native = native_spec(AgentConfig(config.kind))
    except (ValueError, KeyError):
        return None
    return int(native[1]) if native is not None and native[1] > 0 else None


def searches(agent: str) -> bool:
    """Whether a seat chose its moves by searching: its configuration names a node
    budget. (A record's `searches` count cannot say: a game over in one move has none.)"""
    return budget(agent) is not None


def learned(agent: str) -> bool:
    """Whether a seat searched with an evaluation trained on this game (a net)."""
    from chess_variant_sandbox.agents.base import parse_agent

    try:
        return parse_agent(agent).options.get("net") is not None
    except ValueError:
        return False


def player_name(agent: str) -> tuple[str, bool]:
    """What the site calls this player (`opponents.name_of`), and whether it named it; a
    player it does not name gets its plain description."""
    from chess_variant_sandbox.web.opponents import name_of

    named = name_of(agent)
    if named != "the computer":
        return named, True
    nodes = budget(agent)
    if nodes is None:
        return named, False
    tail = ", using an evaluation trained on this game" if learned(agent) else ""
    return f"a computer player that looks at {nodes:,} positions a move{tail}", False


# ------------------------------------------------------------------------------ tree


@dataclass
class _Node:
    """One step of the tree while it is counted: the position it was chosen on, its moves
    (one for a turn, two in seat order for a round), and what the games through it did."""

    state: GameState | None
    step: tuple[int, ...]
    games: int = 0
    results: list[int] = field(default_factory=lambda: [0, 0, 0])
    ended: int = 0
    chosen: Counter[tuple[int, str]] = field(default_factory=Counter)
    children: dict[tuple[int, ...], _Node] = field(default_factory=dict)


def _result_slot(winner: int | None) -> int:
    """Where a game's ending is counted: first player won, drew, second player won."""
    return 0 if winner == 0 else 2 if winner == 1 else 1


def _walk(engine: Engine, moves: list[int], depth: int) -> tuple[list[tuple[GameState, tuple[int, ...]]], bool]:
    """A record's first ``depth`` steps as the engine walks them, and whether the record
    ends inside them (it has no step past the last one returned)."""
    walked = list(islice(engine.steps(engine.initial_state(), moves), depth + 1))
    ended = len(walked) <= depth
    return walked[:depth], ended


def _count(engine: Engine, pool: list[Stored]) -> tuple[_Node, dict[tuple[int, int], list[tuple[int, ...]]]]:
    """Every pool game counted into the tree by its moves, and each game's steps as the
    tree read them (for the line most often played)."""
    root = _Node(None, ())
    lines: dict[tuple[int, int], list[tuple[int, ...]]] = {}
    for game in pool:
        walked, ended = _walk(engine, game.moves, STUDY_DEPTH)
        slot = _result_slot(game.winner)
        agents = game.agents
        root.games += 1
        root.results[slot] += 1
        if ended and not walked:
            root.ended += 1
        node = root
        for index, (state, step) in enumerate(walked):
            child = node.children.get(step)
            if child is None:
                child = node.children[step] = _Node(state, step)
            child.games += 1
            child.results[slot] += 1
            if len(step) == 2:
                child.chosen[(0, agents[0])] += 1
                child.chosen[(1, agents[1])] += 1
            else:
                seat = state.side_to_move
                child.chosen[(seat, agents[seat])] += 1
            if ended and index == len(walked) - 1:
                child.ended += 1
            node = child
        lines[game.order] = [step for _state, step in walked]
    return root, lines


def _ordered(node: _Node) -> list[_Node]:
    """A step's next steps, by how often they were played, then by their moves; never by
    how they went."""
    return sorted(node.children.values(), key=lambda child: (-child.games, child.step))


def _row(engine: Engine, node: _Node, agent_index: dict[str, int], depth: int) -> dict[str, Any]:
    """One kept step as the record holds it, with its kept next steps below it."""
    assert node.state is not None
    row: dict[str, Any] = {
        "moves": list(node.step),
        "seat": None if len(node.step) == 2 else node.state.side_to_move,
        "words": [engine.describe_move(node.state, move) for move in node.step],
    }
    row.update(_counts(node, agent_index))
    row.update(_below(engine, node, agent_index, depth))
    return row


def _counts(node: _Node, agent_index: dict[str, int]) -> dict[str, Any]:
    out: dict[str, Any] = {"games": node.games, "results": list(node.results), "ended": node.ended}
    if node.chosen:
        out["players"] = sorted([seat, agent_index[agent], count]
                                for (seat, agent), count in node.chosen.items())
    return out


def _below(engine: Engine, node: _Node, agent_index: dict[str, int], depth: int) -> dict[str, Any]:
    if depth >= STUDY_DEPTH:
        return {"next": []}
    kept, other_moves, other_games = [], 0, 0
    for child in _ordered(node):
        if child.games >= STUDY_FLOOR:
            kept.append(_row(engine, child, agent_index, depth + 1))
        else:
            other_moves += 1
            other_games += child.games
    out: dict[str, Any] = {"next": kept}
    if other_moves:
        out["other"] = {"moves": other_moves, "games": other_games}
    return out


def tree_steps(tree: dict[str, Any] | None) -> int:
    """How many steps a stored tree keeps (the root is not a step)."""
    if not tree:
        return 0
    return sum(1 + tree_steps(child) for child in tree.get("next") or [])


def _top_line(root: _Node) -> list[tuple[int, ...]]:
    """The tree's most-played move at each step, followed down from the opening."""
    line, node = [], root
    while node.children:
        node = _ordered(node)[0]
        line.append(node.step)
    return line


# -------------------------------------------------------------------------- examples


def _deficit(game: Stored) -> float:
    """How far the winner was behind at worst, on the record's material series (signed
    from the first player's side: `tournament/play.py` records `material(state, 0)`)."""
    series = [float(value) for value in game.record.get("material") or []]
    if game.winner is None or not series:
        return 0.0
    return max(0.0, -min(series)) if game.winner == 0 else max(0.0, max(series))


def _plies(game: Stored) -> int:
    return len(game.moves)


def _nearest(games: list[Stored], target: float) -> Stored | None:
    if not games:
        return None
    return min(games, key=lambda game: (abs(_plies(game) - target), game.order))


def examples(
    pool: list[Stored], strongest: set[str], lines: dict[tuple[int, int], list[tuple[int, ...]]],
    top: list[tuple[int, ...]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """The six rules over the pool, in order, a game taken once and a game identical to
    one already taken skipped: the examples, and for each rule that found none, why."""
    taken: set[tuple[int, int]] = set()
    seen: set[tuple[int, ...]] = set()
    chosen: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    total = len(pool)

    def free(games: Iterable[Stored]) -> list[Stored]:
        return [game for game in games
                if game.order not in taken and tuple(game.moves) not in seen]

    def take(rule: str, game: Stored | None, why: str, **extra: Any) -> None:
        if game is None:
            missing.append({"rule": rule, "reason": why})
            return
        taken.add(game.order)
        seen.add(tuple(game.moves))
        chosen.append({"rule": rule, "seed": game.seed, "place": game.place,
                       "players": list(game.agents), "winner": game.winner,
                       "reason": game.record.get("reason"), "plies": _plies(game),
                       "moves": game.moves, **extra})

    best = [game for game in pool if set(game.agents) <= strongest]
    identical = "every game it could pick is identical to one already shown"

    # 1. The line most often played: of the two strongest's games, the one that follows
    # the tree's most-played move longest.
    def follows(game: Stored) -> int:
        steps = lines.get(game.order, [])
        n = 0
        while n < len(steps) and n < len(top) and steps[n] == top[n]:
            n += 1
        return n

    candidates = free(best) or free(pool)
    pick = min(candidates, key=lambda game: (-follows(game), game.order)) if candidates else None
    take(LINE, pick, identical, follows=follows(pick) if pick else 0)

    # 2, 3. A win for each player, nearest the median length of the pool's decisive games.
    decisive = [game for game in pool if game.winner is not None]
    middle = statistics.median(_plies(game) for game in decisive) if decisive else 0.0
    for rule, seat, who in ((FIRST_WON, 0, "first"), (SECOND_WON, 1, "second")):
        wins_best = [game for game in best if game.winner == seat]
        wins_pool = [game for game in pool if game.winner == seat]
        pick = _nearest(free(wins_best), middle) or _nearest(free(wins_pool), middle)
        why = (f"the {who} player won none of these {total} games" if not wins_pool
               else identical)
        take(rule, pick, why)

    # 4. A game that turned round: the winner furthest behind, if more than one unit.
    turned = [game for game in free(decisive) if _deficit(game) > TURNED]
    pick = min(turned, key=lambda game: (-_deficit(game), game.order)) if turned else None
    take(TURNED_ROUND, pick,
         "no winner was ever behind by more than one unit on the computer players' own "
         "count of pieces" if not any(_deficit(g) > TURNED for g in decisive) else identical,
         behind=round(_deficit(pick), 4) if pick else 0.0)

    # 5. The quickest finish: the shortest decisive game.
    quick = free(decisive)
    pick = min(quick, key=lambda game: (_plies(game), game.order)) if quick else None
    take(QUICKEST, pick, "none of these games was won" if not decisive else identical)

    # 6. A draw, where at least a tenth of the pool was drawn.
    drawn = [game for game in pool if game.winner is None]
    if drawn and len(drawn) >= DRAW_SHARE * total:
        middle_drawn = statistics.median(_plies(game) for game in drawn)
        take(DRAW, _nearest(free(drawn), middle_drawn), identical)
    else:
        missing.append({"rule": DRAW, "reason":
                        f"{len(drawn)} of these {total} games were drawn, under a tenth"})
    return chosen, missing


# ---------------------------------------------------------------------------- record


def basis_now(spec: dict[str, Any], runs_dir: Path | str, seeds: Iterable[int],
              source: str = SOURCE_LIBRARY) -> dict[str, Any]:
    """What a record stands on: this module's version, what the game compiles to, the
    run seeds, each games file's digest, the day they were played and where they came
    from. Not the weights: a new weights version does not stale a record."""
    from chess_variant_sandbox.measurement import tables_fingerprint
    from chess_variant_sandbox.spec.canonical import spec_hash

    seeds = tuple(seeds)
    digest = spec_hash(spec)
    return {
        "study_version": STUDY_VERSION,
        "tables": tables_fingerprint(spec),
        "seeds": list(seeds),
        "files": stored_games.digests(runs_dir, digest, seeds),
        "played_on": stored_games.played_on(runs_dir, digest, seeds),
        "source": source,
        "engine": _played_by(runs_dir, digest),
    }


def _played_by(runs_dir: Path | str, digest: str) -> str | None:
    """The core and game loop that played the measurement (its stamp's `engine`), so a
    reader can say the games were played before a later change to the engine
    (`puzzles.stale_basis`'s rule). Not part of what makes a record owed."""
    try:
        stored = json.loads((Path(runs_dir) / digest / "canonical.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    basis = stored.get("basis") if isinstance(stored, dict) else None
    found = basis.get("engine") if isinstance(basis, dict) else None
    return str(found) if found else None


def build(
    spec: dict[str, Any], runs_dir: Path | str, seeds: Iterable[int],
    source: str = SOURCE_LIBRARY, today: datetime.date | None = None,
) -> dict[str, Any]:
    """One game's study record from the stored games of its runs at ``seeds``."""
    from chess_variant_sandbox.spec.canonical import spec_hash

    seeds = tuple(seeds)
    digest = spec_hash(spec)
    record: dict[str, Any] = {
        "format": FORMAT,
        "ruleset_hash": digest,
        "made": (today or datetime.date.today()).isoformat(),
        "basis": basis_now(spec, runs_dir, seeds, source),
    }
    absent = stored_games.missing(runs_dir, digest, seeds)
    if len(absent) == len(seeds):
        record["none"] = NO_RECORDS
        return record
    if absent:
        record["none"] = (f"Its measurement's runs are not all on disk (seeds "
                          f"{', '.join(str(seed) for seed in absent)}).")
        return record
    games = list(stored_games.read(runs_dir, digest, seeds))
    agents = sorted({agent for game in games for agent in game.agents},
                    key=lambda agent: (budget(agent) or 0, agent))
    agent_index = {agent: index for index, agent in enumerate(agents)}
    players = []
    for agent in agents:
        name, named = player_name(agent)
        players.append({"agent": agent, "budget": budget(agent), "searches": searches(agent),
                        "learned": learned(agent), "name": name, "named": named})
    pool = [game for game in games if searches(game.agents[0]) and searches(game.agents[1])]
    record.update({"players": players, "games": len(games), "pool": len(pool),
                   "left_out": len(games) - len(pool)})
    engine = Engine(spec)
    root, lines = _count(engine, pool)
    if len(pool) >= MIN_POOL:
        tree: dict[str, Any] = {"games": root.games, "results": list(root.results),
                                "ended": root.ended}
        tree.update(_below(engine, root, agent_index, 0))
        record["tree"] = tree
    else:
        record["tree"] = None
        record["no_tree"] = (f"Only {len(pool)} of these games had a searching player in "
                             f"both seats, under {MIN_POOL}: too few for a tree.")
    budgets = sorted({b for b in (budget(agent) for agent in agents) if b is not None})
    strongest = {agent for agent in agents if budget(agent) in budgets[-2:]}
    chosen, missing = examples(pool, strongest, lines, _top_line(root))
    record["examples"] = chosen
    record["missing_examples"] = missing
    return record


def write(path: Path, data: dict[str, Any]) -> None:
    """Write a record compactly, through a temporary file, so a reader never sees half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ----------------------------------------------------------------------- owed, check


def owes(data: Any, spec: dict[str, Any], runs_dir: Path | str, seeds: Iterable[int]) -> str | None:
    """Why a game's study record is owed again, or None where its basis still matches:
    not this game's, an older format or study version, another compile of its rules,
    other seeds, or record files that moved (a game measured again, or measured at last)."""
    from chess_variant_sandbox.measurement import tables_fingerprint
    from chess_variant_sandbox.spec.canonical import spec_hash

    seeds = tuple(seeds)
    digest = spec_hash(spec)
    if not isinstance(data, dict) or data.get("ruleset_hash") != digest:
        return "the record cannot be read as this game's"
    written = data.get("format")
    if not isinstance(written, int) or isinstance(written, bool) or written < FORMAT:
        return f"its record is format {written}, and the code writes {FORMAT}"
    found = data.get("basis")
    basis: dict[str, Any] = found if isinstance(found, dict) else {}
    version = basis.get("study_version")
    if not isinstance(version, int) or version < STUDY_VERSION:
        return f"it was made by study version {version}, and the code's is {STUDY_VERSION}"
    if basis.get("tables") != tables_fingerprint(spec):
        return "its rules compile to different tables than it was made on"
    if list(basis.get("seeds") or []) != list(seeds):
        return f"it was made from seeds {basis.get('seeds')}, not {list(seeds)}"
    if (basis.get("files") or {}) != stored_games.digests(runs_dir, digest, seeds):
        return "its games files have changed since it was made"
    return None


class _Illegal(Exception):
    pass


def _legal(engine: Engine, state: GameState, step: tuple[int, ...]) -> bool:
    """Whether the engine still offers this stored step here: a turn's move among the
    side to move's moves, a round's two among each seat's. The engine lists; this compares."""
    if engine.is_round(state):
        return (len(step) == 2 and step[0] in engine.legal_moves_for(state, 0)
                and step[1] in engine.legal_moves_for(state, 1))
    return len(step) == 1 and step[0] in engine.legal_moves(state)


def _after(engine: Engine, state: GameState, step: tuple[int, ...]) -> GameState:
    if not _legal(engine, state, step):
        raise _Illegal
    return engine.apply_round(state, *step) if len(step) == 2 else engine.apply(state, step[0])


def _check_tree(engine: Engine, row: dict[str, Any], state: GameState) -> int:
    count = 0
    for child in row.get("next") or []:
        after = _after(engine, state, tuple(child["moves"]))
        count += 1 + _check_tree(engine, child, after)
    return count


def _check_game(engine: Engine, moves: list[int]) -> None:
    state, index = engine.initial_state(), 0
    while index < len(moves):
        width = 2 if engine.is_round(state) else 1
        step = tuple(moves[index:index + width])
        state = _after(engine, state, step)
        index += width


def check(engine: Engine, data: dict[str, Any]) -> str | None:
    """Why a record no longer replays, or None: every kept tree step and every example's
    every move is asked of the engine as still legal where the record says it was played.
    The loader's check; it never mends a record."""
    if "none" in data:
        return None
    try:
        if data.get("tree"):
            _check_tree(engine, data["tree"], engine.initial_state())
    except _Illegal:
        return "a step of its tree is no longer legal"
    for example in data.get("examples") or []:
        try:
            _check_game(engine, list(example.get("moves") or []))
        except _Illegal:
            return f"its {example.get('rule')} example no longer replays"
    return None


# ---------------------------------------------------------------------------- survey


MAKE = "make"


def library(root: Path | str = ".") -> list[tuple[str, str, dict[str, Any]]]:
    """Every Library and Baseline game as `(hash, name, spec)`, `rulesets/` then
    `archive/`, each hash once: `puzzles.library`, with an archived preview kept. A
    preview is played and shown on its own page, and its measurement's games are stored
    like any other's, so it has a study (the plan's 140); proven positions leave it out."""
    from chess_variant_sandbox.spec.canonical import load_spec, spec_hash

    found: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for directory in ("rulesets", "archive"):
        folder = Path(root) / directory
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
            try:
                spec = load_spec(path)
            except (OSError, ValueError):
                continue
            if isinstance(spec, dict):
                found.setdefault(spec_hash(spec), (spec_hash(spec), str(spec.get("name") or path.stem), spec))
    return list(found.values())


@dataclass(frozen=True)
class Standing:
    """Where one game's study record stands: `needs` is `make` where it is owed (with
    why), `""` where it holds (with what it holds)."""

    hash: str
    name: str
    needs: str
    reason: str

    @property
    def owed(self) -> bool:
        return bool(self.needs)


def summary(data: dict[str, Any]) -> str:
    """What a current record holds, in a line."""
    if "none" in data:
        return f"current: no study ({data['none']})"
    kept = tree_steps(data.get("tree"))
    tree = f"a tree of {kept} steps" if data.get("tree") else "no tree"
    return (f"current: {data.get('pool')} of {data.get('games')} games in the pool, {tree}, "
            f"{len(data.get('examples') or [])} examples")


def standing(
    digest: str, name: str, spec: dict[str, Any], out_dir: Path | str,
    games_dir: Path | str, seeds: Iterable[int], replay: bool = True,
) -> Standing:
    """One game's `Standing`: owed where it has no record, its basis no longer matches
    (`owes`), or (with ``replay``) a stored step no longer replays (`check`)."""
    seeds = tuple(seeds)
    data = load(record_path(out_dir, digest))
    if data is None:
        return Standing(digest, name, MAKE, "no record")
    why = owes(data, spec, games_dir, seeds)
    if why is None and replay:
        why = check(Engine(spec), data)
    if why is not None:
        return Standing(digest, name, MAKE, why)
    return Standing(digest, name, "", summary(data))


def make(
    spec: dict[str, Any], out_dir: Path | str, games_dir: Path | str, seeds: Iterable[int],
    source: str = SOURCE_LIBRARY,
) -> dict[str, Any]:
    """Build one game's record from ``games_dir``'s runs and write it under ``out_dir``."""
    from chess_variant_sandbox.spec.canonical import spec_hash

    data = build(spec, games_dir, seeds, source)
    write(record_path(out_dir, spec_hash(spec)), data)
    return data
