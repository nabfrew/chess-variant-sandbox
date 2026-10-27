"""The map's families: who is near whom, which games form a family, and what each shares.

`docs/archive/plan-map.md` §4, M2. The relation is `spec/likeness.py`'s and nothing here
re-weighs it; this module turns it into the three things the map shows besides its
picture.

- **Neighbours.** Two games are joined when each is among the other's `MUTUAL` nearest by
  likeness (the *mutual nearest* graph), the line weighing their likeness. Every game is
  judged against its own surroundings, so a dense corner of the library and a sparse one
  both make families without a cut that suits only one of them.
- **Families** are the Louvain communities of that graph (`web/communities.py`,
  resolution 1). A game in no family of two or more is *unusual*, ranked by how far its
  nearest game is.
- **Memory.** A map remembers the families it last published (`runs/graph/map.json`,
  stamped `MAP_VERSION`), and the next map is *settled* from them
  (`communities.settle`, M2b): **a game stays in its family while a line still joins it
  to another member**; a game no line holds -- new, unusual, or cut off -- is placed
  afresh by Louvain's local move, and one the last map placed joins a family only
  through two of its lines. So a weight that wavers, or a line that comes and goes, moves
  nobody, a new game lands beside its relatives, and a family keeps its id while it keeps
  most of its members. Every family stays connected. A cold run -- every order, the best
  modularity -- happens only when there is no previous map or its version differs.

**The name** of a family is the word that best marks it out, in words. A word *marks a
family out* when the harmonic mean of two shares -- of the family that has it, and of
the games having it that are in the family -- is over half (`MARKS`) and most of the
family has it; a word every member
has is preferred to one most have, then the one that marks it out most, then a rule
before a setting, a piece, a move and the board. A draw clock never names a family (how
long a game may run is not what it is). Said as: a rule in `relate.RULE_WORDS`' words, a
setting in its rule's (after its value, when the setting is the rule's own one word); a
piece by what the games call it, or, when it is a piece everybody has made different, by
what was made different; a move by its piece; a board by its lattice, size, joined edges,
or every square filled. A family no word marks out is named *near* its centre game, the
member most like the rest. Two families with one name each add their centre's.

**A pure function of what it is given** -- the library and the previous map. The games
are sorted by id before anything runs, so the order they arrive in moves nothing;
settling visits them in that order, and the cold run is `communities.partition`'s,
deterministic already. No constant here names a game or is read off a library: the
parameters are `MUTUAL`, Louvain's resolution, the two lines that let a placed game join
a family, and the half in `MARKS`.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from chess_variant_sandbox.spec import likeness as likeness_module
from chess_variant_sandbox.spec.likeness import (
    ARMY,
    IDENTITY,
    LATTICE,
    MECHANISM,
    RULE,
    SETTING,
    SIZE,
    WRAP,
    Likeness,
    likeness_of_forms,
)
from chess_variant_sandbox.spec.relate import RULE_KINDS, RULE_WORDS, NormalForm, mechanisms
from chess_variant_sandbox.web import communities

# Bumped when a change here would put a game in another family for a library and a
# previous map that did not change. A stored map of another version is not settled from:
# the next map is drawn cold, which is the one deliberate re-draw.
MAP_VERSION = "m2"

# How many nearest each game offers when the mutual graph is drawn. The plan's.
MUTUAL = 5

# A word names a family only when the harmonic mean of the share of the family that has
# it and the share of its holders in the family is over this.
MARKS = 0.5

# The kinds of word in the order a name prefers them, when two mark a family out alike.
KINDS = (RULE, SETTING, IDENTITY, MECHANISM, LATTICE, WRAP, SIZE, ARMY)

# What a family's id is spelt as: this, then a number no family of this map's line has
# held before.
FAMILY = "f"

# The memory's file, under `runs/`.
MAP_FILE = Path("graph") / "map.json"


# ------------------------------------------------------------------------ the result


@dataclass(frozen=True)
class Family:
    """One family: its id, its members (ids, sorted), its centre and its name."""

    id: str
    members: tuple[str, ...]
    centre: str
    word: str | None
    name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "word": self.word,
            "centre": self.centre,
            "members": list(self.members),
            "size": len(self.members),
        }


@dataclass(frozen=True)
class FamilyMap:
    """What one build found: families, the unusual, the mutual lines, and the memory."""

    families: tuple[Family, ...]
    # Each game's family id, `None` for an unusual game.
    family_of: dict[str, str | None]
    # Every unusual game with the distance to its nearest game, the furthest first.
    unusual: tuple[tuple[str, float], ...]
    # The mutual-nearest lines, each pair sorted, by likeness.
    lines: dict[tuple[str, str], float]
    # The next family number this map's line has not used.
    next_number: int
    # Whether this map was settled from a previous one (`False`: drawn cold).
    warm: bool
    version: str = MAP_VERSION
    likeness_version: str = likeness_module.LIKENESS_VERSION
    neighbours: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Each game's place in the picture (`web/mapping.py`, M3), and the layout's version;
    # empty and `None` for a map built without its picture.
    positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    layout: str | None = None

    def memory(self) -> Memory:
        return Memory(
            family_of=dict(self.family_of),
            next_number=self.next_number,
            positions=dict(self.positions),
            layout=self.layout,
            likeness=self.likeness_version,
        )

    def as_dict(self) -> dict[str, Any]:
        """The memory's file: what the next build settles from, and what M3 adds to."""
        games: dict[str, dict[str, Any]] = {}
        for game in sorted(self.family_of):
            row: dict[str, Any] = {"family": self.family_of[game]}
            if game in self.positions:
                row["position"] = list(self.positions[game])
            games[game] = row
        held: dict[str, Any] = {
            "version": self.version,
            "likeness": self.likeness_version,
            "next": self.next_number,
            "families": {family.id: family.as_dict() for family in self.families},
            "games": games,
        }
        if self.layout is not None:
            held["layout"] = self.layout
        return held


@dataclass(frozen=True)
class Memory:
    """What a map remembers of the last one: each game's family and place, the next number."""

    family_of: Mapping[str, str | None]
    next_number: int = 1
    positions: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    # The version of the layout that drew `positions` (`mapping.LAYOUT_VERSION`), and of
    # the relation it was drawn over (`likeness.LIKENESS_VERSION`).
    layout: str | None = None
    likeness: str | None = None


# ------------------------------------------------------------------------- the graph


def mutual_lines(relation: Likeness, ids: Sequence[str], k: int = MUTUAL) -> dict[
    tuple[str, str], float
]:
    """The mutual `k`-nearest graph: a line where each game is among the other's nearest.

    Weighed by likeness; a pair that shares nothing rare (likeness 0) is no line.
    """
    near = [set(relation.nearest(i, k)) for i in range(len(ids))]
    lines: dict[tuple[str, str], float] = {}
    for i, mine in enumerate(near):
        for j in sorted(mine):
            if j > i and i in near[j]:
                weight = relation.similarity(i, j)
                if weight > 0:
                    pair = (ids[i], ids[j]) if ids[i] <= ids[j] else (ids[j], ids[i])
                    lines[pair] = weight
    return dict(sorted(lines.items()))


def settle(
    ids: Sequence[str],
    lines: Mapping[tuple[str, str], float],
    previous: Mapping[str, str | None] | None,
) -> list[list[str]]:
    """The communities of these lines: cold without `previous`, else settled from it.

    Every community is connected, and a game with no line is alone. Settling is
    `communities.settle`, with every game `previous` names (in a family or unusual) known.
    """
    nodes = sorted(ids)
    if previous is None:
        found = communities.partition(lines)
    else:
        held: dict[str, list[str]] = defaultdict(list)
        for node in nodes:
            family = previous.get(node)
            if family is not None:
                held[family].append(node)
        start = [held[family] for family in sorted(held)]
        known = [node for node in nodes if node in previous]
        found = communities.settle(nodes, lines, start, known)
    return _whole(found, nodes)


def _whole(found: Sequence[Sequence[str]], nodes: Sequence[str]) -> list[list[str]]:
    """These communities of two or more, then every other node alone, in one order."""
    kept = [sorted(part) for part in found if len(part) >= 2]
    placed = {node for part in kept for node in part}
    kept += [[node] for node in nodes if node not in placed]
    return sorted(kept, key=lambda part: (-len(part), part))


# ------------------------------------------------------------------------- the build


def build(
    ids: Sequence[str],
    forms: Sequence[NormalForm],
    names: Sequence[str],
    previous: Memory | None = None,
) -> FamilyMap:
    """The families of these games, settled from `previous` when there is one.

    `ids` name the games (a hash), `forms` are their normal forms and `names` what a
    reader calls them, all in one order; which order moves nothing.
    """
    if not len(ids) == len(forms) == len(names):
        raise ValueError("ids, forms and names must be one list each")
    if len(set(ids)) != len(ids):
        raise ValueError("a game is listed twice")
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    ids = [ids[i] for i in order]
    forms = [forms[i] for i in order]
    names = [names[i] for i in order]
    at = {game: i for i, game in enumerate(ids)}
    relation = likeness_of_forms(forms)
    lines = mutual_lines(relation, ids)
    parts = settle(ids, lines, None if previous is None else previous.family_of)

    grouped = [part for part in parts if len(part) >= 2]
    numbers, next_number = _numbered(grouped, previous)
    centres = [_centre(relation, [at[game] for game in part], ids) for part in grouped]
    holders = Counter(word for bag in relation.words for word in bag)
    words = [_word(relation, holders, [at[game] for game in part]) for part in grouped]
    said = [
        None if word is None else words_for(word, [forms[at[game]] for game in part])
        for word, part in zip(words, grouped, strict=True)
    ]
    alike = Counter(name for name in said if name)
    families: list[Family] = []
    for part, number, centre, word, name in zip(
        grouped, numbers, centres, words, said, strict=True
    ):
        centre_name = names[at[centre]]
        shown = (
            f"near {centre_name}" if not name
            else f"{name} ({centre_name})" if alike[name] > 1
            else name
        )
        families.append(
            Family(
                id=f"{FAMILY}{number}",
                members=tuple(part),
                centre=centre,
                word=word,
                name=shown,
            )
        )

    family_of: dict[str, str | None] = {game: None for game in ids}
    for family in families:
        for game in family.members:
            family_of[game] = family.id
    unusual = sorted(
        (
            (game, relation.distance(at[game], relation.nearest(at[game], 1)[0]))
            for game in ids
            if family_of[game] is None and len(ids) > 1
        ),
        key=lambda row: (-row[1], row[0]),
    )
    neighbours = {
        game: tuple(ids[j] for j in relation.nearest(at[game], MUTUAL)) for game in ids
    }
    return FamilyMap(
        families=tuple(families),
        family_of=family_of,
        unusual=tuple(unusual),
        lines=lines,
        next_number=next_number,
        warm=previous is not None,
        neighbours=neighbours,
    )


def _numbered(
    parts: list[list[str]], previous: Memory | None
) -> tuple[list[int], int]:
    """Each family's number: the previous family most of its members were in, else new.

    A previous family's number goes to the community holding most of its members (more
    members first, then the community listed first, then the lower number), once. A
    community no previous family claims gets the next number nobody has held.
    """
    found: list[int | None] = [None] * len(parts)
    next_number = 1 if previous is None else max(1, previous.next_number)
    if previous is not None:
        claims: list[tuple[int, int, int]] = []
        for index, part in enumerate(parts):
            held = Counter(
                _number(previous.family_of.get(game)) for game in part
            )
            for number, count in held.items():
                if number is not None:
                    claims.append((-count, index, number))
        taken: set[int] = set()
        for _count, index, number in sorted(claims):
            if found[index] is None and number not in taken:
                found[index] = number
                taken.add(number)
                next_number = max(next_number, number + 1)
    numbers: list[int] = []
    for number in found:
        if number is None:
            number = next_number
            next_number += 1
        numbers.append(number)
    return numbers, next_number


def _number(family: str | None) -> int | None:
    if family is None or not family.startswith(FAMILY):
        return None
    tail = family.removeprefix(FAMILY)
    return int(tail) if tail.isdigit() else None


def _centre(relation: Likeness, members: list[int], ids: Sequence[str]) -> str:
    """The member most like the rest of its family, the lesser id breaking a tie."""
    best = min(
        members,
        key=lambda i: (
            -math.fsum(relation.similarity(i, j) for j in members if j != i),
            ids[i],
        ),
    )
    return ids[best]


def _word(relation: Likeness, holders: Mapping[str, int], members: list[int]) -> str | None:
    """The word that best marks these games out from the library, or `None`.

    A word's mark is the harmonic mean of the share of the members that have it and the
    share of the games having it that are members: `2 * inside / (members + holders)`.
    Only a word most members have, with a mark over `MARKS`, names; a word every member
    has comes first, then the higher mark, then the kind `KINDS` lists first, then the
    word. A draw clock never names.
    """
    inside = Counter(word for i in members for word in relation.words[i])
    size = len(members)
    found: list[tuple[bool, float, int, str]] = []
    for word, count in inside.items():
        mark = 2 * count / (size + holders[word])
        if mark > MARKS and count > MARKS * size and not _clock(word):
            kind = word.partition(":")[0] + ":"
            rank = KINDS.index(kind) if kind in KINDS else len(KINDS)
            found.append((count < size, -mark, rank, word))
    if not found:
        return None
    *_, best = min(found)
    return best


def _clock(word: str) -> bool:
    """Whether a word is a rule of the draw clock's (`relate.RULE_KINDS`)."""
    kind, _, body = word.partition(":")
    head = body.split("=", 1)[0]
    for mark in (".", "["):
        head = head.split(mark, 1)[0]
    return kind + ":" in (RULE, SETTING) and RULE_KINDS.get(head) == "clock"


# ------------------------------------------------------------------------- the words


def shared_words(
    relation: Likeness, members: Sequence[int], forms: Sequence[NormalForm], few: int
) -> list[str]:
    """The rare things these games all have, said in words, the rarest first, at most `few`.

    `members` index `relation` and `forms` alike. A word every game has weighs nothing and
    says nothing about kinship, so it is not said; nor is a draw clock, which names no
    family either. Two words said alike are said once.
    """
    said: list[str] = []
    for word in relation.shared(members):
        if len(said) >= few:
            break
        if relation.weights[word] <= 0 or _clock(word):
            continue
        phrase = words_for(word, [forms[i] for i in members])
        if phrase not in said:
            said.append(phrase)
    return said


def words_for(word: str, forms: Sequence[NormalForm]) -> str:
    """One of `likeness.words`' words, said for a reader, from the games that share it."""
    kind, _, body = word.partition(":")
    kind += ":"
    if kind == RULE:
        return rule_words(body)
    if kind == SETTING:
        path, _, value = body.partition("=")
        head = path
        for mark in (".", "["):
            head = head.split(mark, 1)[0]
        said = rule_words(head)
        # The rule's own one setting, when it is a word: *annihilation (how the game is won)*.
        if path == head and len(value) > 2 and value[0] == value[-1] == "'":
            said = f"{value[1:-1].replace('_', ' ')} ({said})"
        return said
    if kind == IDENTITY:
        made = _made_different(word, forms)
        if made:
            return rule_words(f"piece.{made}")
        name = _called(word, forms)
        return f"the {name}" if name else "a piece they all have"
    if kind == MECHANISM:
        name = _called(word, forms)
        return f"the {name}'s move" if name else "a move they all have"
    if kind == LATTICE:
        return f"a {body} board"
    if kind == SIZE:
        width, _, height = body.partition("x")
        return f"a {width} by {height} board"
    if kind == WRAP:
        joined = next(
            (form.wrap for form in forms
             if form.wrap is not None and WRAP + likeness_module._canon(form.wrap) == word),
            None,
        )
        return RULE_WORDS["wrap"] if joined is None else _wrap_words(joined)
    if kind == ARMY:
        holding = [form for form in forms if f"{ARMY}{len(form.cells)}" == word]
        if holding and all(len(form.cells) == _squares(form) for form in holding):
            return "every square filled at the start"
        return f"{body} pieces at the start"
    return word


def _wrap_words(wrap: Sequence[Sequence[str]]) -> str:
    """A board's joined edges in words: on every axis or one, both ways or one."""
    joined = [pair for pair in wrap if any(edge != "none" for edge in pair)]
    said = (
        "a board whose opposite edges all join" if len(joined) == len(wrap)
        else "a board with two opposite edges joined"
    )
    if any(edge not in ("none", "both") for pair in joined for edge in pair):
        said += ", one way"
    return said


def _squares(form: NormalForm) -> int:
    """How many squares a board has: its frame less its walls and absent cells."""
    walled = form.walls[0] if form.walls else ()
    return form.size[0] * form.size[1] - len(walled)


def _made_different(word: str, forms: Sequence[NormalForm]) -> str | None:
    """The first field a piece word's piece states beyond how it moves, or `None`.

    A piece everybody has (a rook) made different (it may not take the wolf) is what was
    made different, not *the rook*.
    """
    for form in forms:
        identities = likeness_module._identities(form)
        for pid in sorted(form.pieces):
            if IDENTITY + likeness_module._canon(identities[pid]) == word:
                extra = identities[pid][-1]
                return str(extra[0][0]) if extra else None
    return None


def rule_words(key: str) -> str:
    """One of `relate.rules_involved`'s keys in `relate`'s words for it."""
    if key == "board.wrap":
        return RULE_WORDS["wrap"]
    if key == "board.walls":
        return RULE_WORDS["obstacles"]
    if key == "piece.captures":
        return "who may take whom"
    if key == "piece.spy":
        return "a spy whose loss turns the game into rounds"
    if key == "piece.speed":
        return "pieces of different speeds, the faster moving first in a round"
    if key == "piece.rest":
        return "pieces that rest for some turns after they move"
    if key == "piece.bounty":
        return "pieces whose taking pays a bounty in points"
    if key == "piece.price":
        return "pieces a player may buy from a market, hire or bribe"
    if key == "piece.sight":
        return "pieces that freeze, disarm or shield what they see"
    if key.startswith("board.lattice."):
        return f"a {key.removeprefix('board.lattice.')} board"
    return RULE_WORDS.get(key) or f"the {key.replace('_', ' ').replace('.', ' ')} rule"


def _called(word: str, forms: Sequence[NormalForm]) -> str | None:
    """What the games sharing this piece or movement word most often call its piece."""
    said: Counter[str] = Counter()
    for form in forms:
        named = dict(form.names)
        for pid in sorted(form.pieces):
            if word in _piece_words(form, pid):
                said[str(named.get(pid, pid)).strip().lower()] += 1
    if not said:
        return None
    return min(said, key=lambda name: (-said[name], name))


def _piece_words(form: NormalForm, pid: str) -> frozenset[str]:
    """The identity and movement words one piece alone gives its game."""
    alone = replace(form, forms={pid: form.forms[pid]})
    found = {MECHANISM + likeness_module._canon(m) for m in mechanisms(alone)}
    found.add(IDENTITY + likeness_module._canon(likeness_module._identities(form)[pid]))
    return frozenset(found)


# ------------------------------------------------------------------------ the memory


def read_memory(path: Path) -> Memory | None:
    """The memory a map file holds, or `None` when there is none or it is another version."""
    try:
        held = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(held, dict) or held.get("version") != MAP_VERSION:
        return None
    games = held.get("games")
    if not isinstance(games, dict):
        return None
    family_of: dict[str, str | None] = {}
    positions: dict[str, tuple[float, float]] = {}
    for game, row in games.items():
        family = row.get("family") if isinstance(row, dict) else None
        family_of[str(game)] = family if isinstance(family, str) else None
        place = row.get("position") if isinstance(row, dict) else None
        if (
            isinstance(place, list)
            and len(place) == 2
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in place)
        ):
            positions[str(game)] = (float(place[0]), float(place[1]))
    number = held.get("next")
    layout = held.get("layout")
    relation = held.get("likeness")
    return Memory(
        family_of=family_of,
        next_number=number if isinstance(number, int) else 1,
        positions=positions,
        layout=layout if isinstance(layout, str) and positions else None,
        likeness=relation if isinstance(relation, str) else None,
    )


def write_memory(path: Path, found: FamilyMap) -> bool:
    """Write this map as the next build's memory, whole or not at all; whether it wrote.

    A map the file already holds, byte for byte, is not written again, so a build over an
    unchanged library and memory leaves the file as it was.
    """
    text = json.dumps(found.as_dict(), indent=1, sort_keys=True) + "\n"
    try:
        if path.read_text(encoding="utf-8") == text:
            return False
    except (OSError, ValueError):
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".tmp")
    scratch.write_text(text, encoding="utf-8", newline="\n")
    scratch.replace(path)
    return True
