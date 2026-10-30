"""Q660: a board with computer-moved pieces shows where each one goes next.

The user, 2026-10-02: *"the npc games need some visual indicator of their coming path."*
The only pieces the rules move are schema 9's wild pieces (`rules.npcs`), and each walks
a fixed route: nothing about where one goes is drawn at random. What the rules cannot know
in advance is the coming move, which may step into a wild piece's way, clear it, or take
the piece. So the board draws the phase *as the position now stands*, which is a question
for the core (`engine.coming.coming_phase`), never worked out on the page or in Python.

Every test here is on `PATROL`, a five-by-five game written for it, not on chess: two
kings, two rooks and two guards, and one ferret walking a six-square loop through the
middle that may take a rook and nothing else.
"""

from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Any

import node_session
import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.coming import coming_phase, coming_phases
from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.web import designing, library
from chess_variant_sandbox.web.play import Sessions

ROOT = Path(__file__).resolve().parents[1]

_KING = [{"kind": "leap", "mode": "both", "symmetry": "rotate4", "vectors": [[1, 0], [1, 1]]}]
_WAZIR = [{"kind": "leap", "mode": "both", "symmetry": "rotate4", "vectors": [[1, 0]]}]

PATROL: dict[str, Any] = {
    "schema_version": 9,
    "name": "patrol",
    "description": "a test game: kings, rooks and guards, and a ferret walking a loop",
    "board": {"width": 5, "height": 5},
    "pieces": [
        {"id": "K", "name": "king", "value": 0, "royal": True, "moves": _KING},
        {"id": "R", "name": "rook", "value": 5, "moves": [
            {"kind": "slide", "mode": "both", "symmetry": "rotate4", "vectors": [[1, 0]]}]},
        {"id": "G", "name": "guard", "value": 2, "moves": _WAZIR},
        {"id": "F", "name": "ferret", "value": 1, "captures": ["R"], "moves": _WAZIR},
    ],
    "setup": {"mirror": True, "placements": [
        {"piece": "K", "squares": ["e1"]}, {"piece": "R", "squares": ["a1"]},
        {"piece": "G", "squares": ["c1"]}]},
    "rules": {
        "win": "checkmate", "stalemate": "draw", "max_plies": 80,
        "npcs": [{"piece": "F", "route": ["b3", "c3", "d3", "d2", "c2", "b2"]}],
    },
}


def _square(name: str) -> int:
    return (int(name[1:]) - 1) * 5 + ord(name[0]) - ord("a")


def _steps(engine: Engine, state: Any) -> list[tuple[int, int, int]]:
    return [(s.origin, s.target, s.eaten) for s in coming_phase(engine, state)]


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(PATROL)


@pytest.fixture(scope="module")
def patrol_game() -> library.Game:
    return library.Game(
        hash=spec_hash(PATROL), name="patrol", description=PATROL["description"],
        path=Path("rulesets/patrol.json"), source="designed", spec=copy.deepcopy(PATROL),
    )


def _walk(engine: Engine, seed: int, plies: int) -> list[Any]:
    """A seeded game's positions, opening first: a pure function of the seed."""
    rng = random.Random(seed)
    state = engine.initial_state()
    seen = [state]
    for _ in range(plies):
        legal = engine.legal_moves(state)
        if not legal or engine.terminal(state, legal) is not None:
            break
        state = engine.apply(state, rng.choice(legal))
        seen.append(state)
    return seen


class TestTheCoreAnswers:
    def test_the_spec_is_a_game(self) -> None:
        assert validate_spec(PATROL) == []

    def test_the_opening_ferret_steps_to_the_next_square_of_its_loop(self, engine: Engine) -> None:
        assert _steps(engine, engine.initial_state()) == [(_square("b3"), _square("c3"), 0)]

    def test_the_preview_is_the_phase_any_move_out_of_its_way_runs(self, engine: Engine) -> None:
        """The defining property. Over a seeded game, every legal move that touches no
        square of a coming step is followed by exactly the phase the preview said, by the
        core's own log of it (`apply_with_phase`). A wild piece's way is the square it
        stands on and the next square of its route, read off the compiled table: a test's
        oracle, since a move onto that square, or off it, is the move that changes what
        the phase does."""
        assert engine.spec.npcs is not None
        route_next = engine.spec.npcs.next
        checked = 0
        for state in _walk(engine, seed=7, plies=40):
            coming = coming_phase(engine, state)
            near = set()
            for square, value in enumerate(state.board):
                if value and engine.spec.owner_of(value) == WILD:
                    near |= {square, route_next[square]}
            for move in engine.legal_moves(state):
                origin, target, _, _ = decode_move(move)
                if origin in near or target in near:
                    continue
                assert engine.apply_with_phase(state, move)[1] == coming
                checked += 1
        assert checked > 100, checked

    def test_it_waits_in_front_of_a_piece_its_type_may_not_take(self, engine: Engine) -> None:
        state = engine.make_state({
            "e1": ("K", 0), "e5": ("K", 1), "c3": ("G", 0), "b3": ("F", WILD)})
        assert _steps(engine, state) == []
        # And the core agrees once a move is made that leaves the guard standing.
        move = engine.find_move(state, "Ke1-d1")
        assert engine.apply_with_phase(state, move)[1] == ()

    def test_it_says_what_it_would_eat(self, engine: Engine) -> None:
        state = engine.make_state({
            "e1": ("K", 0), "e5": ("K", 1), "c3": ("R", 1), "b3": ("F", WILD)})
        ((origin, target, eaten),) = _steps(engine, state)
        assert (origin, target) == (_square("b3"), _square("c3"))
        assert eaten == state.board[_square("c3")] != 0

    def test_asking_moves_nothing(self, engine: Engine) -> None:
        for state in _walk(engine, seed=3, plies=12):
            before = (state, engine.legal_moves(state), engine._compute_hash(state))
            coming_phase(engine, state)
            assert (state, engine.legal_moves(state), engine._compute_hash(state)) == before

    def test_the_answer_is_a_function_of_the_position(self) -> None:
        """Determinism: two engines, the same seeded game, the same answers, asked twice."""
        one, two = Engine(PATROL), Engine(copy.deepcopy(PATROL))
        first = [_steps(one, state) for state in _walk(one, seed=11, plies=30)]
        again = [_steps(two, state) for state in _walk(two, seed=11, plies=30)]
        assert first == again
        assert first == [_steps(one, state) for state in _walk(one, seed=11, plies=30)]
        assert any(first), "a walk on which nothing was coming tests nothing"

    def test_a_game_without_wild_pieces_has_nothing_coming(self) -> None:
        dull = Engine(load_spec(ROOT / "rulesets" / "dull.json"))
        for state in _walk(dull, seed=1, plies=6):
            assert coming_phase(dull, state) == ()


class TestTheSessionCarriesIt:
    def test_every_position_carries_its_own_answer(self, patrol_game: library.Game) -> None:
        session = Sessions().start(patrol_game, ("random", "random"), seed=5)
        engine = session.engine
        snapshot = session.snapshot()
        assert snapshot["coming"] == [s.as_dict() for s in coming_phase(engine, session.state)]
        for _ in range(8):
            snapshot = session.play_agent()
        assert snapshot["coming"] == [s.as_dict() for s in coming_phase(engine, session.state)]
        # Each move carries what was coming from the position it left, which is what a
        # position stepped back to draws.
        for entry, state in zip(session.history, session.states[1:] + [session.state],
                                strict=True):
            assert entry["coming"] == [s.as_dict() for s in coming_phase(engine, state)]

    def test_a_replay_carries_it_from_the_opening(self, patrol_game: library.Game) -> None:
        session = Sessions().start(patrol_game, ("random", "random"), seed=5)
        for _ in range(4):
            session.play_agent()
        timeline = designing.timeline(session)
        assert timeline[0]["coming"] == [{"from": _square("b3"), "to": _square("c3"), "eaten": 0}]
        assert [row["coming"] for row in timeline[1:]] == [e["coming"] for e in session.history]

    def test_a_finished_game_has_nothing_coming(self, engine: Engine,
                                                patrol_game: library.Game) -> None:
        session = Sessions().start(patrol_game, ("human", "human"))
        # Black king boxed in the corner by two rooks, the ferret far away on d2.
        session.state = engine.make_state({
            "e1": ("K", 0), "a5": ("K", 1), "b1": ("R", 0), "b3": ("R", 0),
            "d2": ("F", WILD)}, side_to_move=0)
        session.play(engine.find_move(session.state, "Rb1-a1"))
        snapshot = session.snapshot()
        assert snapshot["outcome"] is not None
        assert snapshot["coming"] == []


# ------------------------------------------------------------------ the page draws it


def _wrapped(expression: str) -> Any:
    from chess_variant_sandbox.viz.wrapped import WRAPPED

    done = node_session.run(WRAPPED + "\nconsole.log(JSON.stringify(" + expression + "));")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


_POINTS = re.compile(r'x1="([-\d.]+)" y1="([-\d.]+)"\s+x2="([-\d.]+)" y2="([-\d.]+)"')
_TIP = re.compile(r'd="M[-\d.]+ [-\d.]+L([-\d.]+) ([-\d.]+)L')


class TestTheArrow:
    def test_one_arrow_per_step_clear_of_the_piece_and_the_dot(self) -> None:
        """From b3 to c3 (straight) and from b2 to c3 (diagonal), on a 5x5 board of cell
        20: each arrow starts outside the wild piece's disc and its tip stops short of
        the middle of the square it goes to, where a legal-move dot is drawn."""
        html = _wrapped("""
          comingHtml([{from: 11, to: 12, eaten: 0}, {from: 6, to: 12, eaten: 0}], 20, 5, 5,
                     flatView(5, 5, false), null)
        """)
        assert html.count('data-mark="coming"') == 2
        assert 'data-coming="2"' in html and "var(--coming)" in html
        centre = {11: (30.0, 50.0), 12: (50.0, 50.0), 6: (30.0, 70.0)}
        for (x1, y1, _, _), (tx, ty), (start, end) in zip(
            [tuple(map(float, found)) for found in _POINTS.findall(html)],
            [tuple(map(float, found)) for found in _TIP.findall(html)],
            [(11, 12), (6, 12)], strict=True,
        ):
            fx, fy = centre[start]
            ex, ey = centre[end]
            assert ((x1 - fx) ** 2 + (y1 - fy) ** 2) ** 0.5 >= 0.4 * 20 - 1e-6
            # The tip is the arrow's nearest point to the middle, and a legal-move dot
            # reaches r .13 plus half its .05 stroke from it.
            assert ((tx - ex) ** 2 + (ty - ey) ** 2) ** 0.5 >= 0.19 * 20 > 0.155 * 20

    def test_nothing_coming_draws_nothing(self) -> None:
        assert _wrapped("[comingHtml([], 20, 5, 5, null, null), comingHtml(null, 20, 5, 5, null, null)]") == ["", ""]

    def test_the_legend_key_is_the_arrow(self) -> None:
        key = _wrapped("markKey('coming')")
        assert 'data-mark="coming"' in key and "var(--coming)" in key
        row = _wrapped("MARK_KINDS.coming")
        assert row["shape"] == "arrow" and row["name"] and row["label"]
        # The name is written on the page, not the wood, so it has a colour of its own
        # (`coming-ink`, held to AA by `tokens.FLOORS`), and the legend uses it.
        assert row["ink"] == "var(--coming-ink)"
        from page_text import BOARD

        assert "color:${look.ink || look.colour}" in BOARD


def _timeline(snapshots: list[dict]) -> list[dict]:
    """The page's own timeline built from the snapshots a real session handed out, in node:
    each beat's kind and what it says is coming."""
    from page_text import PAGE

    source = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert source is not None
    runner = f"""
      for (const snapshot of {json.dumps(snapshots)}) {{ SESSION = snapshot; record(snapshot); }}
      console.log(JSON.stringify(TIMELINE.map(e => ({{ kind: e.kind, plies: e.plies,
        steps: e.steps, coming: e.coming, comingThen: e.comingThen }}))));
    """
    from test_web import _ANIMATION_STUB

    done = node_session.run(_ANIMATION_STUB + source.group(1) + "\n" + runner)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


class TestSteppingBackShowsWhatWasComing:
    def test_each_beat_carries_what_was_coming_from_it(self, patrol_game: library.Game) -> None:
        """The opening beat says where the ferret goes first; the beat between a move and
        the phase says the phase about to run (the core's log of it); a ply's last beat
        says what the core answered for the position the ply left."""
        session = Sessions().start(patrol_game, ("random", "random"), seed=9)
        snapshots = [json.loads(json.dumps(session.snapshot()))]
        for _ in range(5):
            snapshots.append(json.loads(json.dumps(session.play_agent())))
        for snapshot in snapshots:
            snapshot["moves"] = []
            snapshot["ruleset"]["icons"] = []
        beats = _timeline(snapshots)

        assert beats[0]["kind"] == "start" and beats[0]["coming"] == snapshots[0]["coming"]
        phased = 0
        for i, beat in enumerate(beats[1:], start=1):
            move = session.history[beat["plies"] - 1]
            last = i + 1 == len(beats) or beats[i + 1]["plies"] != beat["plies"]
            if last:
                assert beat["coming"] == move["coming"], i
            else:
                assert beats[i + 1]["kind"] == "phase"
                assert beat["coming"] == move["phase"] == beats[i + 1]["steps"], i
                phased += 1
        assert phased >= 3, "the ferret walks after every move"


# ------------------------------------------------------------------ Q764: the step after


def _near(engine: Engine, state: Any) -> set[int]:
    """A wild piece's way on ``state`` over two phases: the square it stands on and its
    route's next two squares, read off the compiled table (a test's oracle, as in the
    property above)."""
    assert engine.spec.npcs is not None
    route_next = engine.spec.npcs.next
    near: set[int] = set()
    for square, value in enumerate(state.board):
        if value and engine.spec.owner_of(value) == WILD:
            near |= {square, route_next[square], route_next[route_next[square]]}
    return near


class TestTheStepAfter:
    """The user, 2026-10-08: a wild piece moves twice before your next move -- after you,
    then after your opponent -- so the board shows both. The second is the core's too."""

    def test_two_phases_the_first_being_the_one_phase(self, engine: Engine) -> None:
        for state in _walk(engine, seed=5, plies=20):
            phases = coming_phases(engine, state)
            assert len(phases) == 2
            assert phases[0] == coming_phase(engine, state)

    def test_the_opening_ferret_steps_twice_along_its_loop(self, engine: Engine) -> None:
        first, then = coming_phases(engine, engine.initial_state())
        assert [(s.origin, s.target) for s in first] == [(_square("b3"), _square("c3"))]
        assert [(s.origin, s.target) for s in then] == [(_square("c3"), _square("d3"))]

    def test_the_second_is_the_phase_after_two_moves_out_of_its_way(self, engine: Engine) -> None:
        """The defining property: over a seeded game, a move and a reply that touch none of
        a wild piece's next two squares are followed by exactly the two phases the
        preview said, by the core's own log of each (`apply_with_phase`)."""
        checked = 0
        for state in _walk(engine, seed=13, plies=30):
            first, then = coming_phases(engine, state)
            near = _near(engine, state)
            clear = [m for m in engine.legal_moves(state)
                     if not {decode_move(m)[0], decode_move(m)[1]} & near]
            for move in clear[:6]:
                after, phase = engine.apply_with_phase(state, move)
                assert phase == first
                if engine.terminal(after) is not None:
                    continue
                for reply in engine.legal_moves(after)[:6]:
                    origin, target, _, _ = decode_move(reply)
                    if origin in near or target in near:
                        continue
                    assert engine.apply_with_phase(after, reply)[1] == then
                    checked += 1
        assert checked > 50, checked

    def test_it_says_what_the_second_step_would_eat(self, engine: Engine) -> None:
        state = engine.make_state({
            "e1": ("K", 0), "e5": ("K", 1), "d3": ("R", 1), "b3": ("F", WILD)})
        first, then = coming_phases(engine, state)
        assert [(s.origin, s.target, s.eaten) for s in first] == [
            (_square("b3"), _square("c3"), 0)]
        ((origin, target, eaten),) = [(s.origin, s.target, s.eaten) for s in then]
        assert (origin, target) == (_square("c3"), _square("d3"))
        assert eaten == state.board[_square("d3")] != 0

    def test_asking_moves_nothing(self, engine: Engine) -> None:
        for state in _walk(engine, seed=3, plies=12):
            before = (state, engine.legal_moves(state), engine._compute_hash(state))
            coming_phases(engine, state)
            assert (state, engine.legal_moves(state), engine._compute_hash(state)) == before

    def test_a_game_without_wild_pieces_has_two_empty_phases(self) -> None:
        dull = Engine(load_spec(ROOT / "rulesets" / "dull.json"))
        assert coming_phases(dull, dull.initial_state()) == ((), ())

    def test_every_position_carries_both(self, patrol_game: library.Game) -> None:
        session = Sessions().start(patrol_game, ("random", "random"), seed=5)
        engine = session.engine

        def said(state: Any) -> dict[str, Any]:
            first, then = coming_phases(engine, state)
            return {"coming": [s.as_dict() for s in first],
                    "comingThen": [s.as_dict() for s in then]}

        snapshot = session.snapshot()
        assert {k: snapshot[k] for k in ("coming", "comingThen")} == said(session.state)
        assert snapshot["comingThen"], "the opening ferret walks twice"
        for _ in range(6):
            snapshot = session.play_agent()
        assert {k: snapshot[k] for k in ("coming", "comingThen")} == said(session.state)
        for entry, state in zip(session.history, session.states[1:] + [session.state],
                                strict=True):
            assert {k: entry[k] for k in ("coming", "comingThen")} == said(state)
        timeline = designing.timeline(session)
        assert [row["comingThen"] for row in timeline[1:]] == [
            e["comingThen"] for e in session.history]

    def test_a_finished_game_has_nothing_coming_after_either(
        self, engine: Engine, patrol_game: library.Game
    ) -> None:
        session = Sessions().start(patrol_game, ("human", "human"))
        session.state = engine.make_state({
            "e1": ("K", 0), "a5": ("K", 1), "b1": ("R", 0), "b3": ("R", 0),
            "d2": ("F", WILD)}, side_to_move=0)
        session.play(engine.find_move(session.state, "Rb1-a1"))
        snapshot = session.snapshot()
        assert snapshot["outcome"] is not None
        assert snapshot["coming"] == snapshot["comingThen"] == []


_WIDTH = re.compile(r'data-mark="(\w+)"[^>]*?stroke-width="([\d.]+)"', re.S)


class TestTheSecondArrow:
    def test_it_is_drawn_fainter_and_under_the_first(self) -> None:
        """b3 to c3, then c3 to d3, on a 5x5 board of cell 20: the second arrow is its own
        kind, in the same violet at the same opacity (so it keeps the first's 3 : 1 on both
        woods, `test_style`), and fainter by its thinner dots, set further apart."""
        html = _wrapped("""
          comingHtml([{from: 11, to: 12, eaten: 0}], 20, 5, 5, flatView(5, 5, false), null,
                     [{from: 12, to: 13, eaten: 0}])
        """)
        assert 'data-coming="1"' in html and 'data-coming-then="1"' in html
        widths = dict(_WIDTH.findall(html))
        assert float(widths["comingThen"]) < float(widths["coming"])
        assert html.index('data-mark="comingThen"') < html.index('data-mark="coming"')
        assert html.count('opacity="0.85"') == 2 and html.count("var(--coming)") == 2

    def test_a_step_and_the_step_straight_back_are_side_by_side(self) -> None:
        """Two arrows on one line: a wild piece on a two-square route steps b3 to c3 and
        then back. Each is set to its own right, so neither is drawn over the other; a step
        with no way back drawn stays on the line between the squares' middles."""
        html = _wrapped("""
          comingHtml([{from: 11, to: 12, eaten: 0}], 20, 5, 5, flatView(5, 5, false), null,
                     [{from: 12, to: 11, eaten: 0}])
        """)
        (_, y1a, _, y2a), (_, y1b, _, y2b) = [
            tuple(map(float, p)) for p in _POINTS.findall(html)]
        assert y1a == y2a and y1b == y2b, "both stay level"
        assert abs(y1a - y1b) >= 2 * 0.12 * 20, "and apart by more than a dot's width"
        alone = _wrapped("""
          comingHtml([{from: 11, to: 12, eaten: 0}], 20, 5, 5, flatView(5, 5, false), null,
                     [{from: 12, to: 13, eaten: 0}])
        """)
        assert {float(p[1]) for p in _POINTS.findall(alone)} == {50.0}

    def test_only_the_second_is_drawn_when_the_first_is_empty(self) -> None:
        html = _wrapped("""
          comingHtml([], 20, 5, 5, flatView(5, 5, false), null, [{from: 12, to: 13, eaten: 0}])
        """)
        assert html.count('data-mark="comingThen"') == 1 and 'data-mark="coming"' not in html

    def test_the_legend_names_it_in_plain_words(self) -> None:
        row = _wrapped("MARK_KINDS.comingThen")
        assert row["faint"] is True and row["colour"] == "var(--coming)"
        assert row["name"] == "faint dotted arrow"
        assert row["label"] == "where it steps after the move after that"
        key = _wrapped("markKey('comingThen')")
        assert 'data-mark="comingThen"' in key
        from page_text import BOARD

        assert "...(comingThen && comingThen.length ? ['comingThen'] : [])" in BOARD


class TestSteppingBackShowsBoth:
    def test_each_beat_carries_the_step_after_too(self, patrol_game: library.Game) -> None:
        """The opening beat and a ply's last beat carry the core's second answer for that
        position; the beat between a move and its phase carries, after the phase about to
        run, the first answer for the position the ply left."""
        session = Sessions().start(patrol_game, ("random", "random"), seed=9)
        snapshots = [json.loads(json.dumps(session.snapshot()))]
        for _ in range(5):
            snapshots.append(json.loads(json.dumps(session.play_agent())))
        for snapshot in snapshots:
            snapshot["moves"] = []
            snapshot["ruleset"]["icons"] = []
        beats = _timeline(snapshots)

        assert beats[0]["comingThen"] == snapshots[0]["comingThen"] != []
        for i, beat in enumerate(beats[1:], start=1):
            move = session.history[beat["plies"] - 1]
            last = i + 1 == len(beats) or beats[i + 1]["plies"] != beat["plies"]
            assert beat["comingThen"] == (move["comingThen"] if last else move["coming"]), i


# ------------------------------------------------------------------ Q1070: every fixed path


def _path_games() -> list[Path]:
    """Every library file (`rulesets/`, `archive/`) with a piece on a fixed path, read off
    the files rather than pinned, so a game added later is held too."""
    found = []
    for folder in ("rulesets", "archive"):
        for path in sorted((ROOT / folder).glob("*.json")):
            spec = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(spec.get("rules"), dict) and spec["rules"].get("npcs"):
                found.append(path)
    return found


PATH_GAMES = _path_games()


class TestEveryFixedPathTelegraphsTwoSteps:
    """The user, 2026-10-30: *"All npcs with pre-determined paths should telegraph their path
    two steps ahead the way wolf circuit does."* Pitchfork's arrows were in the data all
    along; the page drew the pieces that had just moved over them (below). Here: every
    library game with a piece on a fixed path hands the page both phases, at the opening and
    after moves, as the core answers them, with something to draw in each."""

    def test_the_library_has_the_games_the_user_named(self) -> None:
        names = {json.loads(p.read_text(encoding="utf-8"))["name"] for p in PATH_GAMES}
        assert {"pitchfork", "wolf circuit"} <= names, names

    @pytest.mark.parametrize("path", PATH_GAMES, ids=lambda p: p.stem)
    def test_both_phases_are_on_every_snapshot(self, path: Path) -> None:
        spec = load_spec(path)
        game = library.Game(hash=spec_hash(spec), name=spec["name"],
                            description=spec.get("description", ""), path=path,
                            source="archive", spec=spec)
        session = Sessions().start(game, ("random", "random"), seed=3)
        for ply in range(3):
            snapshot = session.snapshot() if ply == 0 else session.play_agent()
            if snapshot["outcome"] is not None:
                break
            first, then = coming_phases(session.engine, session.state)
            assert snapshot["coming"] == [s.as_dict() for s in first], ply
            assert snapshot["comingThen"] == [s.as_dict() for s in then], ply
            walks = [s for s in snapshot["coming"] if s["from"] != s["to"]]
            after = [s for s in snapshot["comingThen"] if s["from"] != s["to"]]
            assert walks and after, f"{path.stem} at ply {ply}: nothing to draw"


class TestTheArrowsStayOnTop:
    def test_the_arrows_are_lifted_over_the_pieces_that_moved(self) -> None:
        """`animateChange` lifts every travelling piece to the top of the board so it passes
        over the stationary ones; where wild pieces walk in a convoy (pitchfork) each arrow's
        head lies on a piece that has just moved, and the arrows vanished under them. Once
        the animation is set up, the arrows' group is lifted over them again (Q1070)."""
        from page_text import BOARD

        lift = "if (arrows && beats) svg.appendChild(arrows);"
        assert BOARD.count(lift) == 1
        assert BOARD.index("animateChange(svg, previous") < BOARD.index(lift)
        assert BOARD.index(lift) < BOARD.index("arrows.animate([{ opacity: 0 }, { opacity: 1 }]")
