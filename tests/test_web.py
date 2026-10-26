"""The local application: the library it shows and the games it lets you play.

The rule these tests hold down is the one the whole web package lives inside: it may
hold a position and ask the engine questions, and it may not answer them. So they check
what the API reports, whether the opponent it seats is the *rated* one, and whether the
two moves a person cannot express by clicking two squares -- a promotion and a drop --
reach the page as choices rather than as guesses.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import replace
from http.client import HTTPConnection
from pathlib import Path
from urllib.request import urlopen

import node_session
import pytest

from chess_variant_sandbox import corpus
from chess_variant_sandbox.agents import build_agent, parse_agent
from chess_variant_sandbox.agents.features import VOCABULARY
from chess_variant_sandbox.agents.heuristics import Heuristic, HeuristicSet, store_path
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.tournament.play import choose_move
from chess_variant_sandbox.viz.pieces import icons_for
from chess_variant_sandbox.web import families, graphing, keeping, library
from chess_variant_sandbox.web import server as server_module
from chess_variant_sandbox.web.play import Sessions
from chess_variant_sandbox.web.server import App, PortInUse, Server, handler_for, serve, warm

# `serve_forever` looks for its stop flag every half second by default, so each test
# that stopped a server waited out that half second in `shutdown()`. 20 ms stops it at once.
SERVE_FAST = {"poll_interval": 0.02}

RULESETS = library.RULESET_DIR

# A vector these arms build an opponent out of, and one that is not a player at all.
# Derived from the vocabulary rather than written out, because a positional list one
# entry short does not fail loudly -- every term after the gap has shifted, and the arm
# goes on running while testing something else.
_ZEROS = [0.0] * len(VOCABULARY)
_MATERIAL_ONLY = [1.0] + _ZEROS[1:]


def _weights(**terms: float) -> list[float]:
    weights = [0.0] * len(VOCABULARY)
    for term, weight in terms.items():
        weights[VOCABULARY.index(term)] = weight
    return weights


@pytest.fixture(scope="module")
def games(warm_library: None, tmp_path_factory: pytest.TempPathFactory):
    """The committed library, listed once first (`warm_library`): a card's `summary`
    runs the smoke gate on its game, 4 s a test over every card on a cold worker (Q749),
    and the listing pays it once per worker for every test that reads a card.

    **Over an empty runs directory of its own, never the tree's `runs/`** (Q995f). That
    one holds whatever its developer saved from a served page, and `discover` lists a
    saved draft beside the committed games under the draft's own name: a copy of torus
    chess saved as *chess* made `{game.name: ...}["chess"]` the draft's card, and
    `fairyTranslates` read False on chess. A test that means the tree's own `runs/`
    discovers it itself (`test_the_stored_ladder_table_contains_no_heuristic`).
    """
    return library.discover(RULESETS, tmp_path_factory.mktemp("no-runs"))


@pytest.fixture
def cold_library(monkeypatch, tmp_path):
    """The library's four compile memos, empty for this test and warm again after it.

    A test that asks what a cold card costs clears them, and clearing the module's own
    caches left every later test on the worker to list a cold library: 11 s on one core,
    paid again by whichever test listed next, after `warm_library` had paid it once (Q749).
    Each memo is swapped for a fresh one of the same size, so the test's `cache_clear`
    empties only its own, and `monkeypatch` puts the warm ones back. So is the store the
    answers outlive a process in (`library.CardAnswers`, Q887), for an empty one of its
    own, so a cleared memo is answered by compiling and never by an earlier test's save.
    """
    from functools import lru_cache

    for name in ("_smoke_cached", "_fairy_cached", "_engine_cached", "_still_cached"):
        warm = getattr(library, name)
        monkeypatch.setattr(library, name, lru_cache(**warm.cache_parameters())(warm.__wrapped__))
    monkeypatch.setattr(library, "_CARDS", library.CardAnswers(tmp_path / "library-cards.json"))


@pytest.fixture(scope="module")
def chess_game(games):
    return next(game for game in games if game.name == "chess")


@pytest.fixture(scope="module")
def horde(games):
    """The asymmetric one: thirty-six pawns against an ordinary army."""
    return next(game for game in games if game.name == "horde")


@pytest.fixture(scope="module")
def boar_game(chess_spec):
    """A three-owner ruleset, built here because no committed one declares ``npcs``.

    Chess plus two boars walking a loop, which is `docs/archive/plan-schema-6-9.md`
    stage C's showcase shape. Constructed rather than discovered on purpose: the page
    a ruleset with wild
    pieces gets is the thing under test, and until the showcase is committed the only
    way to ask for one is to write one.
    """
    import copy

    spec = copy.deepcopy(chess_spec)
    spec["schema_version"] = 9
    spec["name"] = "boar"
    spec["pieces"].append(
        {
            "id": "W",
            "name": "boar",
            "value": 3,
            "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}],
        }
    )
    spec["rules"]["npcs"] = [
        {"piece": "W", "route": ["c3", "c4", "d4", "d3"], "bounty": 3, "mirror": True}
    ]
    spec["rules"]["scoring"] = {"count": "points"}
    return library.Game(
        hash=spec_hash(spec),
        name="boar",
        description="chess, with two boars walking a loop",
        path=Path("rulesets/boar.json"),
        source="designed",
        spec=spec,
    )


class TestLibrary:
    def test_every_committed_ruleset_is_listed(self, games) -> None:
        names = {game.name for game in games}
        for expected in ("chess", "shogi", "xiangqi", "dull"):
            assert expected in names
        # And nothing else, whatever the tree's `runs/` holds: a draft saved there under a
        # committed game's name would be read as that game's card by name (Q995f).
        assert {game.source for game in games} == {"committed"}
        assert len(names) == len(games)

    def test_a_card_says_enough_to_choose_from(self, chess_game) -> None:
        card = chess_game.summary()
        assert card["board"] == "8x8"
        assert card["pieceTypes"] == 6
        assert card["hash"] == spec_hash(chess_game.spec)

    def test_a_card_says_whether_the_game_can_be_won_at_all(self, games) -> None:
        """The index cannot say this and the cheap gate can.

        ``viable`` reads a score, so it is absent for anything unscored and it never
        carries a reason -- and "scored badly" and "cannot be won" are not the same news
        to somebody choosing what to spend twenty minutes on. `rps-hierarchy-chess` is
        unwinnable as its designer first stated it, the gate catches it *by name*, and
        the library used to show nothing at all.
        """
        by_name = {game.name: game.summary() for game in games}
        hierarchy = next(
            card for name, card in by_name.items() if "hierarchy" in name.lower()
        )
        assert hierarchy["playable"] is False
        assert any("may be out of reach" in reason for reason in hierarchy["playableReasons"])

        chess = next(card for name, card in by_name.items() if name == "chess")
        assert chess["playable"] is True
        assert chess["playableReasons"] == []

    def test_a_card_says_whether_an_outside_engine_has_a_counterpart(self, games) -> None:
        """The half of the Fairy-Stockfish question a listing can honestly ask.

        `supports` is a pure translation check; `playable` plays the two engines against
        each other and is what actually offers the opponent. A card carries the first,
        under a key named after it, because the second costs an engine start per ruleset
        and needs a binary that is discovered rather than shipped -- so a card reporting
        it would answer "no" for the whole library on a machine that has no engine, which
        is a statement about a PATH and not about a game.
        """
        by_name = {game.name: game.summary() for game in games}

        chess = by_name["chess"]
        assert chess["fairyTranslates"] is True
        assert chess["fairyReasons"] == []

        # Refused for a fact about the two rule systems rather than about a build: there
        # is no board without edges over there, whatever version is installed.
        torus = by_name["torus chess"]
        assert torus["fairyTranslates"] is False
        # The rule by its card's title, never by its path (Q681).
        assert any("edges that join" in reason for reason in torus["fairyReasons"])

    def test_the_translation_and_the_still_board_share_one_compile(
        self, games, monkeypatch, cold_library
    ) -> None:
        """First-visit study N8: a cold listing compiled every ruleset three times (the
        cheap gate, the still board, the translation). The translation now reads the
        rules `engine_for` compiled, which the still board asks for next, and answers what
        `fairy.why_not` answers from the text."""
        import json

        from chess_variant_sandbox import fairy
        from chess_variant_sandbox.engine import compile as compiling
        from chess_variant_sandbox.engine import engine

        by_name = {game.name: game for game in games}
        compiled: list[dict] = []
        real = compiling.compile_spec

        def counted(spec, *args, **kwargs):  # type: ignore[no-untyped-def]
            compiled.append(spec)
            return real(spec, *args, **kwargs)

        monkeypatch.setattr(compiling, "compile_spec", counted)
        monkeypatch.setattr(engine, "compile_spec", counted)
        for name in ("chess", "torus chess"):
            spec = by_name[name].spec
            blob = json.dumps(spec, sort_keys=True)
            library._fairy_cached.cache_clear()
            library._engine_cached.cache_clear()
            library._still_cached.cache_clear()
            compiled.clear()
            _, reasons = library._fairy_cached(blob)
            library.still(by_name[name])
            # The translator's own proof that a key it leaves out is inert compiles a
            # widened copy (`_check_skipped_squares`); the rules themselves, once.
            assert compiled.count(spec) == 1, name
            monkeypatch.setattr(compiling, "compile_spec", real)
            assert reasons == tuple(fairy.why_not(spec)), name
            monkeypatch.setattr(compiling, "compile_spec", counted)

    def test_a_card_compiles_its_rules_once(self, games, monkeypatch, cold_library) -> None:
        """First-visit study N8, the last third (Q823b): the cheap gate played its own
        compile of the rules. It now plays `engine_for`'s, which the translation and the
        still board read too, so a cold card compiles its rules once, and the gate's
        answer is the one it gives the text."""
        import json

        from chess_variant_sandbox.engine import compile as compiling
        from chess_variant_sandbox.engine import engine
        from chess_variant_sandbox.loop.gates import smoke_test

        by_name = {game.name: game for game in games}
        compiled: list[dict] = []
        real = compiling.compile_spec

        def counted(spec, *args, **kwargs):  # type: ignore[no-untyped-def]
            compiled.append(spec)
            return real(spec, *args, **kwargs)

        # A pass, a game the gate refuses by name, and one on a torus.
        for name in ("chess", "hierarchy chess", "torus chess"):
            game = by_name[name]
            for cache in (library._smoke_cached, library._fairy_cached,
                          library._engine_cached, library._still_cached):
                cache.cache_clear()
            compiled.clear()
            monkeypatch.setattr(compiling, "compile_spec", counted)
            monkeypatch.setattr(engine, "compile_spec", counted)
            card = game.summary()
            library.still(game)
            monkeypatch.setattr(compiling, "compile_spec", real)
            monkeypatch.setattr(engine, "compile_spec", real)
            # The translator's proof that a skipped key is inert compiles a widened copy.
            assert compiled.count(game.spec) == 1, name
            alone = smoke_test(json.loads(json.dumps(game.spec)))
            assert card["playable"] is alone.passed, name
            assert card["playableReasons"] == alone.reasons, name
        assert not by_name["hierarchy chess"].summary()["playable"]

    def test_rules_that_do_not_compile_are_refused_in_the_gates_words(
        self, cold_library
    ) -> None:
        """The listing has no engine for rules that do not compile, so the gate gets the
        text and says why, as it did when it compiled every spec itself."""
        import json

        from chess_variant_sandbox.loop.gates import smoke_test

        broken = {"name": "broken", "schema_version": 1, "board": {"width": 0}}
        library._smoke_cached.cache_clear()
        passed, reasons = library._smoke_cached(json.dumps(broken, sort_keys=True))
        alone = smoke_test(broken)
        assert (passed, list(reasons)) == (alone.passed, alone.reasons)
        assert not passed and reasons[0].startswith("the rules don't fit together")

    @pytest.mark.slow
    def test_the_gate_answers_the_same_on_a_shared_compile_over_the_corpus(
        self, cold_library
    ) -> None:
        """Every library spec, `rulesets/` and `archive/`: the gate on `engine_for`'s
        compile gives the verdict, reasons and numbers it gives compiling the text."""
        import json

        from chess_variant_sandbox.loop.gates import smoke_test

        everything = library.discover(RULESETS, library.RUNS_DIR, corpus.ARCHIVE_DIR)
        assert any(game.spec.get("name") == "chess" for game in everything)
        for game in everything:
            spec = json.loads(json.dumps(game.spec))
            alone = smoke_test(spec).as_dict()
            shared = smoke_test(spec, engine=library.engine_for(spec)).as_dict()
            assert shared == alone, game.name
            library._smoke_cached.cache_clear()
            card = library._playability(game.spec)
            assert (card["playable"], card["playableReasons"]) == (
                alone["passed"], alone["reasons"]), game.name

    def test_the_card_never_claims_the_engine_can_play_it(self, chess_game) -> None:
        """A translation that is nearly right is worse than none, so the key name matters.

        The listing may say the rules translate and must not say a game can be played:
        four things the released build cannot do are facts about a download rather than
        about a ruleset, and `web/opponents.fairy_stockfish_for` -- which verifies against
        the engine that will actually play -- is the only thing allowed to conclude that
        one will.
        """
        card = chess_game.summary()
        assert {key for key in card if key.startswith("fairy")} == {
            "fairyTranslates", "fairyReasons",
        }

    def test_the_detail_view_joins_the_four_sources(self, chess_game) -> None:
        detail = library.detail(chess_game)
        assert detail["rules"].startswith("chess - a two-player game")
        assert [piece["id"] for piece in detail["pieces"]] == list("PNBRQK")
        assert detail["provenance"]["origin"]
        # A score is read from disk, never computed here. Absent is absent.
        if detail["score"] is not None:
            assert detail["score"]["ruleset_hash"] == chess_game.hash

    def test_a_card_says_which_kind_of_game_it_is(self, chess_game, games) -> None:
        """The app was built before the grid and knew nothing about cells.

        The cell is spec-derived, so it is available for every game in the library
        whether or not anything has ever played it -- which is the same property that
        lets a design run aim a proposal at an empty neighbour.
        """
        from chess_variant_sandbox.loop.grid import descriptor, label

        assert chess_game.summary()["cell"] == label(descriptor(chess_game.spec))
        assert library.detail(chess_game)["cellDescription"].startswith("a board of")
        # No game may be filed nowhere: every axis has a bin for every spec.
        assert all(game.summary()["cell"] for game in games)

    def test_coverage_counts_kinds_of_game_and_not_games(self, games) -> None:
        """A library is not an elite archive, and this is not `Grid`.

        `loop.grid.build` keeps one *viable* entry per cell; the library holds
        everything, including the games the gates threw out. Same axes, same labels,
        no ranking -- so the check is that every game is counted exactly once and the
        cell count is a count of cells.
        """
        from chess_variant_sandbox.loop.grid import CELLS

        report = library.coverage(games)
        assert report["cells"] == CELLS
        assert report["filled"] == len(report["occupied"]) <= CELLS
        assert sum(len(cell["games"]) for cell in report["occupied"]) == len(games)
        assert all(cell["description"] for cell in report["occupied"])

    def test_a_verdict_is_prose_and_is_absent_until_one_is_written(self, tmp_path) -> None:
        """The half of `llm/judge.py` that survived, and the half that did not.

        A verdict reaches the page as text; there is no number in it to slip back into
        a ranking. A game nobody has judged carries `None`, which the page renders as
        nothing at all -- "not judged" and "judged and found wanting" are different
        answers and must not look alike.
        """
        spec = load_spec(RULESETS / "gardner.json")
        spec["name"] = "unjudged"
        (tmp_path / "rulesets").mkdir()
        (tmp_path / "rulesets" / "x.json").write_text(json.dumps(spec), encoding="utf-8")
        run = tmp_path / "runs" / "somewhere"
        run.mkdir(parents=True)

        assert library.discover(tmp_path / "rulesets", tmp_path / "runs")[0].verdict is None

        (run / "verdict.json").write_text(json.dumps({
            "ruleset_hash": spec_hash(spec),
            "verdict": "A small game that knows what it is.",
            "strengths": ["short"], "weaknesses": [], "suspected_degeneracy": "",
            "would_play_again": True,
        }), encoding="utf-8")

        found = library.discover(tmp_path / "rulesets", tmp_path / "runs")[0]
        assert found.verdict is not None
        assert found.verdict["verdict"].startswith("A small game")
        assert not any(isinstance(value, (int, float)) and not isinstance(value, bool)
                       for value in found.verdict.values())

    def test_an_unscored_ruleset_reports_no_score_rather_than_zero(self, tmp_path) -> None:
        spec = load_spec(RULESETS / "gardner.json")
        spec["name"] = "not scored anywhere"
        (tmp_path / "rulesets").mkdir()
        (tmp_path / "rulesets" / "x.json").write_text(json.dumps(spec), encoding="utf-8")
        (tmp_path / "runs").mkdir()

        found = library.discover(tmp_path / "rulesets", tmp_path / "runs")
        assert [game.summary()["index"] for game in found] == [None]


def _variants(root: Path, edits: dict[str, dict]) -> list[library.Game]:
    """`gardner` as `base`, and one committed game per entry, made by the edit it names.

    An edit's `rules` update the base's rules and its `values` move pieces' prices. A small
    library of cheap games, so the map these tests build over it costs a fraction of a
    second rather than the whole library's compile.
    """
    base = load_spec(RULESETS / "gardner.json")
    base["name"] = "base"
    (root / "rulesets").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    written = {"base": base}
    for name, edit in edits.items():
        made = json.loads(json.dumps(base))
        made["name"] = name
        made["rules"].update(edit.get("rules", {}))
        for piece in made["pieces"]:
            if piece["id"] in edit.get("values", {}):
                piece["value"] = edit["values"][piece["id"]]
        written[name] = made
    for name, spec in written.items():
        (root / "rulesets" / f"{name}.json").write_text(json.dumps(spec), encoding="utf-8")
    return library.discover(root / "rulesets", root / "runs")


# Seven games around `base`: a price, a setting, a new rule, two rules at once, and one
# that changes a great deal, so the most alike and the least alike are both real.
VARIANTS = {
    "a-price": {"values": {"Q": 8}},
    "a-setting": {"rules": {"promotion": {"piece": "P", "rank": "last", "to": ["Q"]}}},
    "a-new-rule": {"rules": {"stalemate": "loss"}},
    "two-rules": {"rules": {"stalemate": "loss", "max_plies": 120}},
    "a-clock": {"rules": {"max_plies": 90}},
    "far": {"rules": {"stalemate": "win", "max_plies": 60,
                      "promotion": {"piece": "P", "rank": "last", "to": ["N"]}},
            "values": {"Q": 4, "R": 2, "N": 9}},
}


@pytest.fixture(scope="module")
def variants(tmp_path_factory: pytest.TempPathFactory) -> list[library.Game]:
    """`VARIANTS`' library, and one runs directory for the module, so its map is built once."""
    return _variants(tmp_path_factory.mktemp("variants"), VARIANTS)


def _runs(games: list[library.Game]) -> Path:
    """The runs directory beside these games' rulesets, where their map remembers itself."""
    return Path(games[0].path).parent.parent / "runs"


class TestAGamesNeighbours:
    """M5 of `docs/archive/plan-map.md`: a game's page lists the games most like it, by the map.

    `library.neighbours` asks `graphing.near` which games and in what order -- the map's
    own games, ranked by the map's likeness -- and adds the reading of each pair: what both
    have, what differs (`spec/relate.py`'s count and sentences, tested there) and the gap
    between two scores where both can honestly be read. So what is held here is that the
    page's list *is* the map's, a game off the map is ranked by the same relation, nothing
    orders by the count of rules any more, and the gap is read and never computed.
    """

    def _two_games(self, tmp_path: Path, **rules) -> list[library.Game]:
        """A library of two, one rule apart, with nothing measured."""
        spec = load_spec(RULESETS / "gardner.json")
        spec["name"] = "first"
        other = json.loads(json.dumps(spec))
        other["name"] = "second"
        other["rules"].update(rules)
        (tmp_path / "rulesets").mkdir(exist_ok=True)
        for stem, written in (("a", spec), ("b", other)):
            (tmp_path / "rulesets" / f"{stem}.json").write_text(
                json.dumps(written), encoding="utf-8"
            )
        (tmp_path / "runs").mkdir(exist_ok=True)
        return library.discover(tmp_path / "rulesets", tmp_path / "runs")

    def test_a_detail_view_carries_no_links_of_its_own(self, tmp_path) -> None:
        """The games near a game are the map's, which the server adds (`App.near`)."""
        found = self._two_games(tmp_path, stalemate="loss")
        assert "links" not in library.detail(found[0], found)

    def test_one_changed_rule_is_one_link_that_says_which_rule(self, tmp_path) -> None:
        found = self._two_games(tmp_path, stalemate="loss")
        near = library.neighbours(found[0], found, tmp_path / "runs")
        links = near["links"]
        assert [link["name"] for link in links] == ["second"]
        assert links[0]["hash"] == found[1].hash
        assert links[0]["units"] == 1
        assert links[0]["kind"] == "ending"
        assert links[0]["kinds"] == ["ending"]
        # The sentence says what the rule is about, not which schema key it is: "the
        # stalemate rule" is a link nobody can act on.
        assert links[0]["sentence"] == "what a player with no move gets"
        assert links[0]["differences"][0]["unit"] == "rules.stalemate"
        # The map's reading: how alike -- a library of two shares nothing rare, since what
        # both games have weighs nothing -- and whether a line joins them on the map.
        assert links[0]["likeness"] == 0.0
        assert links[0]["shares"] == [] and links[0]["line"] is False
        assert "distance" not in links[0]
        assert near["onMap"] is True and near["open"] == families.MUTUAL

    def test_a_game_is_never_its_own_neighbour(self, variants) -> None:
        for game in variants:
            links = library.neighbours(game, variants, _runs(variants))["links"]
            assert game.hash not in {link["hash"] for link in links}, game.name
            assert len(links) == len(variants) - 1, "a small library lists every other game"

    def test_the_most_alike_come_first(self, variants) -> None:
        for game in variants:
            alike = [
                link["likeness"]
                for link in library.neighbours(game, variants, _runs(variants))["links"]
            ]
            assert alike == sorted(alike, reverse=True), game.name

    def test_the_page_opens_on_the_map_s_own_games_like_this_one(self, variants) -> None:
        """One meaning of near: the page's first `MUTUAL` are the map panel's `like`, in its
        order, with its likeness, its words and its lines; and the page's community is the
        node's."""
        drawn = graphing.build(variants, _runs(variants))
        for node in drawn.nodes:
            game = next(one for one in variants if one.hash == node.hash)
            near = library.neighbours(game, variants, _runs(variants))
            opened = near["links"][: near["open"]]
            assert [
                {key: link[key] for key in ("hash", "likeness", "shares", "line")}
                for link in opened
            ] == [dict(row, shares=list(row["shares"])) for row in node.like], game.name
            community = near["community"]
            assert (community or {}).get("id") == node.community, game.name

    def test_the_order_is_the_map_s_relation_and_nothing_else(self, variants) -> None:
        """`base`'s list is the map's likeness order (`Likeness.nearest`), whatever the
        counts of rules say: the order the page used to have, fewest rules apart first,
        orders nothing now."""
        base = next(game for game in variants if game.name == "base")
        links = library.neighbours(base, variants, _runs(variants))["links"]
        drawn = graphing._drawn(variants, _runs(variants))
        assert [link["hash"] for link in links] == [
            drawn.ids[other]
            for other in drawn.relation.nearest(drawn.at[base.hash], len(variants))
        ]

    def test_a_game_off_the_map_is_ranked_against_the_map_s_games(
        self, variants, tmp_path
    ) -> None:
        """A draft is not on the map, and its page still lists the library's games most
        like it, by the same relation with itself counted in -- and the map is unchanged."""
        before = graphing.build(variants, _runs(variants))
        [draft] = [
            game for game in _variants(tmp_path, {"a-draft": {"values": {"Q": 7}}})
            if game.name == "a-draft"
        ]
        near = library.neighbours(draft, variants, _runs(variants))
        assert (near["onMap"], near["community"]) == (False, None)
        hashes = [link["hash"] for link in near["links"]]
        assert set(hashes) <= {game.hash for game in variants}
        assert draft.hash not in hashes
        assert not any(link["line"] for link in near["links"])
        alike = [link["likeness"] for link in near["links"]]
        assert alike == sorted(alike, reverse=True)
        # A price moved: the base it was moved from and the game that moved the same
        # price elsewhere are the most like it.
        assert {link["name"] for link in near["links"][:2]} == {"base", "a-price"}
        after = graphing.build(variants, _runs(variants))
        assert [node.like for node in after.nodes] == [node.like for node in before.nodes]

    def test_the_game_least_like_the_rest_still_has_its_list(self, variants) -> None:
        """There is no far case: every game has the games most like it, said as alike as
        they are, and no other ranking takes over past some count of rules."""
        far = next(game for game in variants if game.name == "far")
        links = library.neighbours(far, variants, _runs(variants))["links"]
        assert len(links) == len(variants) - 1
        assert all("distance" not in link for link in links)

    def test_the_list_is_cut_at_the_page_s_twelve(self, tmp_path) -> None:
        many = {f"clock-{plies}": {"rules": {"max_plies": plies}} for plies in range(60, 200, 10)}
        games = _variants(tmp_path, many)
        assert len(games) > library.MOST_LINKS + 1
        near = library.neighbours(games[0], games, tmp_path / "runs")
        assert len(near["links"]) == library.MOST_LINKS

    def test_a_preview_has_no_neighbours(self, variants) -> None:
        preview = replace(variants[0], source=library.PREVIEW)
        near = library.neighbours(preview, variants, _runs(variants))
        assert near["links"] == [] and near["onMap"] is False

    def test_an_unmeasured_pair_shows_no_gap(self, tmp_path) -> None:
        """A difference of two numbers one of which may not be shown is not a number."""
        found = self._two_games(tmp_path, stalemate="loss")
        [link] = library.neighbours(found[0], found, tmp_path / "runs")["links"]
        assert link["gap"] is None

    def test_a_gap_is_read_from_the_two_measurements_and_never_computed(
        self, tmp_path
    ) -> None:
        """Both ends are the two games' own stored numbers, and the gap is their difference.

        Two games one rule apart, each handed a current score (the memo `Game.score`
        fills, so no measurement has to be played).
        """
        first, second = self._two_games(tmp_path, stalemate="loss")
        first._score_cache = {"status": "current", "index": 0.5, "sd": 0.01,
                              "weights_version": "v12", "seeds": [0, 1000, 2000, 3000]}
        second._score_cache = {"status": "current", "index": 0.62, "sd": 0.02,
                               "weights_version": "v12", "seeds": [0]}
        [link] = library.neighbours(first, [first, second], tmp_path / "runs")["links"]
        gap = link["gap"]
        assert (gap["index"], gap["otherIndex"]) == (0.5, 0.62)
        assert gap["gap"] == pytest.approx(0.12)
        assert gap["spread"] == pytest.approx((0.01**2 + 0.02**2) ** 0.5)
        assert gap["clear"] is True
        assert (gap["seeds"], gap["otherSeeds"]) == ([0, 1000, 2000, 3000], [0])
        assert gap["weights"] == "v12"

    def test_each_end_of_a_gap_says_how_many_runs_it_rests_on(self) -> None:
        """Q339. A library game's four runs and a candidate's one can be the two ends of
        one link, so each number travels with its own count -- and an end with no count
        is an end with no number to print, so there is no gap."""
        from types import SimpleNamespace

        def card(index: float, seeds: list[int]) -> SimpleNamespace:
            return SimpleNamespace(score={"index": index, "sd": 0.01,
                                          "weights_version": "v10", "seeds": seeds})

        gap = library._gap(card(0.5, [0, 1000, 2000, 3000]), card(0.7, [0]))
        assert gap is not None
        assert gap["seeds"] == [0, 1000, 2000, 3000] and gap["otherSeeds"] == [0]
        assert library._gap(card(0.5, [0]), card(0.7, [])) is None
        assert library._gap(card(0.5, []), card(0.7, [0])) is None

    def test_the_gap_line_prints_each_number_with_its_own_count(self) -> None:
        gap = {"index": 0.5, "otherIndex": 0.7, "seeds": [0, 1, 2, 3], "otherSeeds": [0],
               "gap": 0.2, "spread": 0.01, "clear": True, "weights": "v10"}
        said = _in_page("", f"gapLine({json.dumps(gap)})")
        assert "0.70 (1 run) against" in said and "0.50 (4 runs), so it scores" in said
        # Without both counts it is no comparison at all, rather than two bare numbers.
        for missing in ("seeds", "otherSeeds"):
            bare = _in_page("", f"gapLine({json.dumps({**gap, missing: []})})")
            assert "No score to compare" in bare and "0.70" not in bare, missing

    def test_the_map_turns_the_counts_round_with_the_numbers(self) -> None:
        """Seen from the link's other end, each count stays with its own number."""
        link = {"source": "a", "target": "b", "gap": {
            "index": 0.5, "otherIndex": 0.7, "seeds": [0, 1, 2, 3], "otherSeeds": [0],
            "gap": 0.2, "spread": 0.01, "clear": True, "weights": "v10", "moved": []}}
        said = _in_page("", f"gapLine(mapGapFrom({json.dumps(link)}, 'b'))")
        assert "0.50 (4 runs) against" in said and "0.70 (1 run), so it scores" in said

    def test_the_page_draws_the_panel_and_the_way_into_a_neighbour(self) -> None:
        from page_text import PAGE

        assert "Games like this one" in PAGE and "Games near this one" not in PAGE
        assert "linksHtml(game)" in PAGE
        # The name is a button that opens that game, which is the point of the panel.
        assert "data-link=" in PAGE
        assert "go('#/game/' + button.dataset.link)" in PAGE
        # No clear change is said in words rather than shown as a difference that is not
        # one: two thirds of one-rule pairs move the index by less than its own noise.
        assert "no clear change" in PAGE


def _in_the_page(setup: str, shown: str) -> str:
    """Run the page's own script in node, then `setup`, and print what `shown` returns.

    The script's top-level `let`s are in scope for what is appended, so a test can hand
    the map a payload of its own and ask a panel what it draws.
    """
    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                                 set: () => true });
      global.document = { getElementById: () => el, querySelector: () => null,
                          querySelectorAll: () => [], addEventListener: () => {},
                          createElement: () => el, body: el };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, scrollTo: () => {}, innerWidth: 1200 };
      global.location = { pathname: '/game', hash: '' };
      global.fetch = () => new Promise(() => {});
    """
    done = node_session.run(
        stub + script.group(1) + "\n" + setup + "\nconsole.log(" + shown + ");"
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _alike(index: float, words: str) -> dict:
    """A link's `similarity` as the server sends it. The page reads no count."""
    return {"index": index, "counts": {}, "words": words}


class TestTheSimilarityReadingOnThePage:
    """`docs/archive/plan-game-graph.md` §9.5's T3, after M5 of `docs/archive/plan-map.md`.

    The similarity index once ordered a game's neighbours among games as many rules away.
    Near now has one meaning, the map's likeness (`TestAGamesNeighbours`), and the index's
    counts are left with what they were always also for: saying *what differs*, by kind --
    *no rules, 6 mechanisms, 1 piece* -- beside each game like this one. They order nothing.

    `tests/test_relate.py` holds what the counts are and what the index weighs them at.
    What is held here is that the server hands the words to the page and the page does no
    arithmetic on them.
    """

    def test_each_link_says_what_differs_by_kind_in_relate_s_words(self, variants) -> None:
        from chess_variant_sandbox.spec import relate

        base = next(game for game in variants if game.name == "base")
        by_hash = {game.hash: game for game in variants}
        links = library.neighbours(base, variants, _runs(variants))["links"]
        readings = [relate.similarity(base.spec, by_hash[link["hash"]].spec) for link in links]
        assert [link["similarity"] for link in links] == [
            {**one.as_dict(), "site": library.likeness_words(one.counts)} for one in readings
        ]
        # They order nothing: the list is the map's, most alike first.
        alike = [link["likeness"] for link in links]
        assert alike == sorted(alike, reverse=True)

    # ------------------------------------------------------------------ on the map

    def test_the_map_lists_the_games_like_this_one_in_the_server_s_order(self) -> None:
        """M4: a game's panel lists its nearest as `like` sends them, nearest first.

        Replaces the rule-step lines' order (count, then index, then features), which left
        the map with those lines. A game a line joins holds that line up; any other holds
        the game up. The order, the shares and which are joined are the server's.
        """
        names = ["hub", "zulu", "alpha", "beta"]
        nodes = [{"hash": name, "name": name, "names": [name], "like": []} for name in names]
        nodes[0]["like"] = [
            {"hash": "zulu", "likeness": 0.8, "shares": ["the camel"], "line": True},
            {"hash": "alpha", "likeness": 0.5, "shares": [], "line": False},
            {"hash": "beta", "likeness": 0.4, "shares": ["a hex board"], "line": True},
        ]
        lines = [
            {"source": "beta", "target": "hub", "likeness": 0.4, "shares": [], "gap": None},
            {"source": "hub", "target": "zulu", "likeness": 0.8, "shares": [], "gap": None},
        ]
        setup = (
            f"MAP_DATA = {json.dumps({'nodes': nodes, 'neighbours': lines})};"
            "MAP_AT = Object.fromEntries(MAP_DATA.nodes.map(node => [node.hash, node]));"
        )
        drawn = _in_the_page(setup, "mapLikeHtml(MAP_AT.hub)")
        assert re.findall(r"<b>(\w+)</b>", drawn) == ["zulu", "alpha", "beta"]
        assert 'data-line-row="1"' in drawn and 'data-line-row="0"' in drawn
        assert 'data-node-row="alpha"' in drawn
        assert "both have the camel" in drawn
    def test_a_line_s_panel_prints_the_server_s_words(self) -> None:
        """How alike, what both have and what differs, printed as they came, with the (i)."""
        nodes = [{"hash": name, "name": name, "names": [name]} for name in ("one", "two")]
        line = {"source": "one", "target": "two", "likeness": 0.42, "units": 1,
                "shares": ["the camel", "a hex board"], "kinds": ["piece"],
                "differences": [{"kind": "pieces", "sentence": "the camel leaps further"}],
                "sentence": "the camel leaps further", "gap": None}
        setup = (
            f"MAP_DATA = {json.dumps({'nodes': nodes, 'neighbours': [line], 'counts': {}, 'kinds': [], 'mutual': 5})};"
            "MAP_AT = Object.fromEntries(MAP_DATA.nodes.map(node => [node.hash, node]));"
        )
        drawn = _in_the_page(setup, "mapLinePanel(MAP_DATA.neighbours[0])")
        assert "the camel; a hex board" in drawn
        assert "the camel leaps further" in drawn and "one rule apart" in drawn
        assert 'data-gloss="alike"' in drawn
    def test_the_page_opens_on_the_panel_s_few_and_folds_the_rest(self) -> None:
        """The game's page prints the server's list as it came: the first `open` rows, the
        rest folded as less alike, each with how alike, what both have and what differs."""
        links = [
            {"hash": f"h{at}", "name": f"game{at}", "units": units, "likeness": alike,
             "shares": ["the camel"] if at == 0 else [], "line": at < 2,
             "kinds": [], "differences": [], "gap": None,
             "similarity": _alike(392, "no rules, 6 mechanisms, 1 piece")}
            for at, (units, alike) in enumerate([(3, 0.8), (1, 0.6), (0, 0.5), (23, 0.1)])
        ]
        game = {"links": links, "open": 2, "onMap": True,
                "community": {"id": "f1", "name": "the camel"}}
        shown = _in_the_page("", f"linksHtml({json.dumps(game)})")
        assert "Games like this one" in shown and "most alike first" in shown
        assert "0.800 alike, a line on the map" in shown
        assert "0.100 alike" in shown and "0.100 alike, a line" not in shown
        assert "both have the camel" in shown
        assert "3 rules apart" in shown and "1 rule apart" in shown
        assert "the same rules, written two ways" in shown
        assert "by kind: no rules, 6 mechanisms, 1 piece" in shown
        assert "2 more, less alike" in shown
        assert shown.index("game0") < shown.index("game1") < shown.index("game2")
        assert "in the community <b>the camel</b>" in shown
        # No other ranking takes over past a count of rules.
        assert not re.search(r"differs in \d+ rules?", shown)
        assert "fifty-five features" not in shown and "faint line" not in shown

    def test_the_page_says_where_the_map_puts_the_game(self) -> None:
        link = {"hash": "h", "name": "near", "units": 1, "likeness": 0.5, "shares": [],
                "line": False, "kinds": [], "differences": [], "gap": None}
        unusual = _in_the_page("", "linksHtml(" + json.dumps(
            {"links": [link], "open": 5, "onMap": True, "community": None}) + ")")
        assert "unusual: in no community" in unusual
        off = _in_the_page("", "linksHtml(" + json.dumps(
            {"links": [link], "open": 5, "onMap": False, "community": None}) + ")")
        assert "not on the map" in off

    # ------------------------------------------------------------ no arithmetic

    def test_the_page_reads_the_index_and_never_the_counts(self) -> None:
        """The counts and their weighing are the server's; the page ranks and prints."""
        from page_text import PAGE

        assert "similarity.counts" not in PAGE
        assert "similarity.words" in PAGE
        for count in ("rules", "settings", "mechanisms", "pieces", "squares", "books"):
            assert f"counts.{count}" not in PAGE, count

    def test_the_gloss_says_the_weighing_relate_holds(self) -> None:
        """The (i) names a ratio and a half, so it is held to the two constants.

        `docs/archive/plan-game-graph.md` §9.3: the user chose eight to one on 2026-09-23.
        If either constant moves, this fails and names the sentence to rewrite.
        """
        from page_text import CORE, DETAIL, GRAPH

        from chess_variant_sandbox.spec import relate

        entry = CORE.split("  differs: [", 1)[1].split("],", 1)[0]
        assert relate.SIMILARITY_RATIO == 8, "the differs gloss says eight"
        assert "eight squares count as one piece" in entry
        assert relate.SETTING_WEIGHT == 0.5, "the differs gloss says half"
        assert "set differently counts half" in entry
        # Beside the sentence that heads a game's list, which is ordered by the map's
        # likeness and carries its (i), `alike`, as the map's panel does (M5).
        assert "info('differs')" in DETAIL and "info('alike')" in DETAIL
        assert "info('alike')" in GRAPH and "info('differs')" not in GRAPH
        assert "info('likeness')" not in CORE + DETAIL + GRAPH


class TestTheMap:
    """G3 to G13 of `docs/archive/plan-game-graph.md`: the second view, from `/api/graph`.

    The page is JavaScript inside a Python string and the suite runs no browser, so these
    read the string -- this package's idiom. What they hold down is the line §8.2 draws
    between the server and the page. **The server says what is on the map and what is
    true of it; the page decides where a dot sits and how it looks.** Every count, every
    sentence about what two games differ in, every score and the picture the map opens on
    arrive in the payload. After that a position is a reader's own -- a drag, or a
    *reset* -- and three functions in the whole page may write one.

    The drawing is `page/explorer.js`'s and `TestTheExplorer` below holds its mechanics.
    What is read here is `page/graph.js`, which is the map in words: the legend, the
    panels, and the readings of the payload both modules share.
    """

    @staticmethod
    def _body(name: str) -> str:
        """One function's source, from its declaration to the next one's."""
        from page_text import GRAPH

        start = GRAPH.index(f"function {name}(")
        rest = GRAPH.find("\nfunction ", start + 1)
        return GRAPH[start:rest if rest > 0 else len(GRAPH)]

    def test_the_library_has_two_views_and_one_control_that_swaps_them(self) -> None:
        from page_text import BROWSE, EXPLORER, GRAPH, PAGE

        assert GRAPH in PAGE
        # The route, so the map is a place somebody can link to and come back to. One
        # view answers it, and `#/map/explore` -- where the explorer was built -- lands
        # on the same one rather than 404ing a link somebody kept.
        assert "parts[0] === 'map') return showExplorer()" in PAGE
        assert "showMap" not in PAGE
        # One switch, drawn by the map's module and called from both views.
        assert PAGE.count("viewSwitchHtml") == 3
        assert "viewSwitchHtml('list')" in BROWSE
        assert "viewSwitchHtml('map')" in EXPLORER

    def test_the_words_beside_the_map_ask_the_server_nothing_about_the_map(self) -> None:
        """One call draws the map, and it is the drawing's (`TestTheExplorer`).

        What this module asks for is a draft: the remix presses (G6) end in the editor
        holding one, and neither works out what it is. `/api/design/<hash>` is the route
        *make one like this* already asks, and nothing it answers with moves a node, a
        line or a count.
        """
        from page_text import GRAPH

        assert set(re.findall(r"api\('(/api/[^']*)'", GRAPH)) == {"/api/design/"}

    def test_the_page_works_out_no_geometry_and_no_rule(self) -> None:
        """The opening picture is the server's, and so is every reading of what differs.

        No trigonometry, because placing a node is `web/graphing.py`'s; no distance
        between two games, because that is `spec/distance.py`'s and arrives on the link;
        no root, log or power, because a dot's place between a size key's two ends is the
        server's number (G11); no spec anywhere, because what two games differ in is
        `spec/relate.py`'s.
        """
        from page_text import EXPLORER, GRAPH

        for computed in ("Math.cos(", "Math.sin(", "Math.atan2(", "Math.sqrt(",
                         "Math.log(", "Math.pow(", "differences.length",
                         "node.x =", "node.y ="):
            assert computed not in GRAPH + EXPLORER, computed
        # Nothing here reaches for a rule set or for the comparison over two of them.
        for named in ("game.spec", "relate(", "compile(", "rules."):
            assert named not in GRAPH + EXPLORER, named
        # The words beside the map hold no coordinate at all: `x` and `y` reach the page
        # on the payload and go straight to the drawing.
        assert "node.x" not in GRAPH and "node.y" not in GRAPH

    def test_three_functions_may_move_a_dot(self) -> None:
        """§8.2's line, as a list of the functions allowed to cross it.

        The server hands over one picture and the page owns every position after it. A
        position is written when the elements are built, when a reader resets, and when a
        ruled-out dot's ring follows the dot it marks -- a drag is the graph library's own
        and passes through no function of ours. There was a fourth, *tidy*, until Q477.
        """
        from page_text import EXPLORER, GRAPH

        # Nothing in the map's words touches a position.
        assert "position" not in GRAPH and "layout" not in GRAPH
        movers = set()
        for part in EXPLORER.split("\nfunction ")[1:]:
            name = part[: part.index("(")]
            body = part[part.index("(") :]
            if ".position(" in body or ".layout({" in body or "position: explorer" in body:
                movers.add(name)
        assert movers == {
            "explorerElements",     # the server's picture, written once
            "explorerReset",        # back to the server's picture
            "explorerRingsFollow",  # the ring, after whatever moved the dot it marks
        }

    def test_no_kind_of_rule_is_named_here(self) -> None:
        """The legend's words are the payload's.

        `relate.KINDS` are schema names -- *wild*, *hand*, *valuation* -- and a second
        place naming a rule is a second place to name it differently. The page prints
        `kind.words`, which `graphing.KIND_WORDS` decides.
        """
        from page_text import GRAPH

        from chess_variant_sandbox.spec import relate

        assert "mapKindWords(difference.kind)" in GRAPH
        assert "MAP_DATA.kinds.find" in GRAPH and "found.words" in GRAPH
        for kind in relate.KINDS:
            assert f"'{kind}'" not in GRAPH, kind

    def test_a_node_says_its_score_the_way_a_card_does(self) -> None:
        """One reading of a measurement, drawn by the library's own functions.

        `scoreTag` is `page/upkeep.js`'s and carries the run count, the verdict and the
        three states -- current, stale, unscored. A number formatted here in its own way
        would be a second opinion about the same measurement.
        """
        from page_text import GRAPH

        assert "scoreTag({ ...score, staleness: node.staleness })" in GRAPH
        assert "splitSentences(score.runVerdict, score.runs)" in GRAPH
        assert "runsPhrase(score.seeds)" in GRAPH
        assert "toFixed(" not in GRAPH

    def test_a_node_opens_the_game_and_a_link_says_what_differs(self) -> None:
        from page_text import GRAPH, PAGE

        assert "go('#/game/' + open.dataset.open)" in PAGE
        # The words on a line are the payload's: what differs, rule by rule, and what both
        # have.
        assert "esc(difference.sentence)" in GRAPH
        assert "esc(line.shares.join('; '))" in GRAPH

    def test_the_two_kinds_of_line_are_told_apart_and_each_has_a_swatch(self) -> None:
        """A line of likeness and an arrow for *made from*, a swatch each in the legend.

        M4 retired the rule-step lines -- one rule apart, a gap, a nearest relative -- and
        their swatches with them; what is left is two claims a reader must still tell
        apart without a click.
        """
        from page_text import GRAPH

        assert "mapLineSwatch('line')" in GRAPH and "mapLineSwatch('lineage')" in GRAPH
        assert "a line: each is" in GRAPH and "<span>made from <span" in GRAPH
        for gone in ("mapLineSwatch('gap')", "mapLineSwatch('nearest')",
                     "a gap: two or three rules", "nearest relative"):
            assert gone not in GRAPH, gone
    def test_the_first_call_says_what_it_is_waiting_for(self) -> None:
        """Five seconds cold, instant afterwards, so the wait says which it is."""
        from page_text import PAGE

        assert "pending('Drawing the map" in PAGE
        assert "every game is read and placed" in PAGE

    # ------------------------------------------------- G4: the score, and narrowing it

    def test_a_dot_is_filled_by_its_score_over_a_scale_the_payload_gave_it(self) -> None:
        """The two ends of the ramp are the payload's, and so is which games are on it.

        A fill over nought to one would paint a library whose scores sit in the top third
        one colour, and a page that worked out its own ends would be deciding what a
        colour means. Both ends arrive in `scale`; the page places a number between them.
        """
        from page_text import GRAPH

        assert "MAP_DATA.scale" in GRAPH
        fill = self._body("mapScoreFill")
        assert "scale.low" in fill and "scale.high" in fill
        # Two scores under different weights never share a leaderboard, so they never
        # share a ramp: that game is hollow, which is the same dot an unscored one gets.
        assert "score.weights !== scale.weights" in fill
        assert "return ''" in fill

    def test_a_score_the_library_withholds_is_a_hollow_dot_here_too(self) -> None:
        """`Game.score` is already `null` for a stale measurement -- one policy, not two.

        Nothing on this page asks whether a measurement is current; it asks whether there
        is a number, which is the question the library already answered.
        """
        from page_text import EXPLORER

        assert "score.index === null" in self._body("mapScoreFill")
        # An empty fill draws the hollow dot rather than a colour of the page's own
        # invention, because the drawing falls back to the panel a dot sits on.
        assert "explorerColour(fill || 'var(--panel)')" in EXPLORER

    def test_the_operator_sentence_and_the_game_in_between_left_with_their_lines(self) -> None:
        """M4: the rule-step lines retired, and with them the two things only they carried.

        The operator's own sentence on a one-rule line (a deploy-time sweep, `graphing`'s
        `_sweep`) and *make the game in between* on a gap. A game's own *and now try*
        stays: it asks the designer, live, which small changes apply.
        """
        from page_text import EXPLORER, GRAPH

        for gone in ("mapReachHtml", "reach.words", "mapBetweenHtml", "mapMakeBetween",
                     "data-between", "link.between"):
            assert gone not in GRAPH + EXPLORER, gone
        assert "mapAskSteps(" in EXPLORER and "function mapAskSteps(hash)" in GRAPH
    def test_the_gap_on_a_line_is_the_game_page_own_sentence(self) -> None:
        """One pair must not read `clear` in one place and `no clear change` in another.

        `gapLine` is `page/detail.js`'s and the map calls it rather than writing a second
        sentence about the same two numbers -- which is also why no number is formatted
        here (`toFixed` appears nowhere in this module).
        """
        from page_text import GRAPH

        assert "gapLine(line.gap)" in GRAPH
        assert "gapLine(mapGapFrom(line, node.hash))" in GRAPH
        assert "toFixed(" not in GRAPH
        # Read from the other end the two numbers change places. Nothing else moves, and
        # nothing here recomputes a difference.
        flipped = self._body("mapGapFrom")
        assert "index: gap.otherIndex" in flipped and "gap: -gap.gap" in flipped

    def test_which_measurements_moved_is_the_payload_ranking_and_the_glossary_words(
        self,
    ) -> None:
        """The page neither ranks the measurements nor names them.

        Which three moved the index most is `graphing`'s, over the scoring pass's own
        contributions; what each one is called is `metrics/glossary.py`'s, carried in the
        payload. A page that did either would be a second opinion about the score.
        """
        moved = self._body("mapMovedHtml")
        assert "gap.moved" in moved
        for computed in ("sort(", "Math.abs", "- part."):
            assert computed not in moved, computed
        assert "MAP_DATA.components" in self._body("mapComponentWords")

    def test_a_filter_lights_and_never_takes_a_game_off_the_map(self) -> None:
        """The plan's §4 asked for the listing's filters and G3 left the question open.

        A card that goes from a list takes nothing with it; a node that went would leave
        its lines hanging off nothing. So the map fades what does not match, and says so
        -- on a list the same control would have taken them away. Which games a filter
        lights is decided here, over the five states of the listing's own score menu,
        because those are the ones a node can answer out of the payload.
        """
        from page_text import GRAPH

        found = self._body("mapFound")
        for taken in ("display", "remove(", "hidden"):
            assert taken not in found, taken
        matches = self._body("mapMatches")
        assert "node.staleness" in matches and "score.viable" in matches
        assert "faded rather than gone" in GRAPH
        # And no control of its own: the menu is written where the shell is.
        assert 'id="mapShow"' not in GRAPH


class TestTheMapDrawsLikeness:
    """`docs/archive/plan-map.md` M4 (after `docs/archive/plan-graph-map.md` S3 and S4).

    The payload's `neighbours` are the map's lines -- two games each among the other's
    nearest -- every one drawn, as wide as their `likeness`, in their community's colour.
    There is no cut and no slider: which pairs are near enough is each game's own nearest.
    Hover, a local view and the find box look at part of the map; none of them moves a dot
    or works out a neighbour.
    """

    @staticmethod
    def _body(name: str) -> str:
        from page_text import EXPLORER, GRAPH

        source = EXPLORER if f"function {name}(" in EXPLORER else GRAPH
        start = source.index(f"function {name}(")
        rest = source.find("\nfunction ", start + 1)
        return source[start:rest if rest > 0 else len(source)]

    def test_every_line_the_payload_sends_is_drawn_and_no_cut_is_compared(self) -> None:
        """M4 retired the slider (a cut over `levels`, read on 69 games): each line the
        server sends is an element, and nothing on the page compares a level or a cut."""
        from page_text import PAGE

        elements = self._body("explorerElements")
        assert "mapLines().forEach((line, index)" in elements
        assert "kind: 'line'" in elements
        for gone in ("MAP_CUT", "mapLevels", "mapSimilar", "row.level", "exploreCut",
                     "MAP_DATA.levels", "MAP_DATA.similar", "MAP_DATA.links"):
            assert gone not in PAGE, gone
    def test_the_map_opens_with_the_arrows_off_and_one_switch_brings_them(self) -> None:
        """The rule-step switch left with its lines (M4); the arrows keep a switch of their
        own, off at opening and again at *reset*."""
        assert "MAP_ARROWS = false;" in self._body("showExplorer")
        assert "MAP_ARROWS = false;" in self._body("explorerReset")
        assert "return !MAP_ARROWS;" in self._body("mapArrowHidden")
        assert 'id="exploreArrows"' in self._body("explorerLinesHtml")
        assert "explorerSetArrows(arrows.checked)" in self._body("wireExplorer")
        for gone in ("MAP_STEPS", "exploreSteps", "mapLinkHidden"):
            assert gone not in self._body("explorerLinesHtml") + self._body("wireExplorer")
    def test_a_line_is_as_wide_as_its_likeness_in_its_communitys_colour(self) -> None:
        from page_text import EXPLORER

        assert "likeness: line.likeness" in self._body("explorerElements")
        assert "const EXPLORER_WIDTH = 'mapData(likeness," in EXPLORER
        assert "{ selector: 'edge.line', style: { 'width': EXPLORER_WIDTH } }" in EXPLORER
        assert "explorerColour(mapLineColour(line))" in EXPLORER
        colour = self._body("mapLineColour")
        assert "from.community === to.community" in colour
        assert "'var(--muted)'" in colour
    def test_communities_of_three_or_more_take_eight_colours_in_payload_order(self) -> None:
        """A pair stays neutral, and a ninth community never gets a generated hue."""
        from chess_variant_sandbox.web.page.style import STYLE

        sizes = [13, 6, 2, 5, 3, 3, 3, 3, 3, 3, 2]
        communities = [
            {"id": f"c{at}", "size": size, "members": [], "names": [], "name": f"c{at}"}
            for at, size in enumerate(sizes)
        ]
        shown = _in_the_page(
            f"MAP_DATA = {json.dumps({'communities': communities, 'nodes': []})};",
            "JSON.stringify(MAP_DATA.communities.map(one => mapCommunityColour(one.id)))"
            " + '|' + mapCommunityColour(null)",
        )
        colours, none = shown.strip().split("|")
        assert json.loads(colours) == [
            "var(--community-1)", "var(--community-2)", "",
            "var(--community-3)", "var(--community-4)", "var(--community-5)",
            "var(--community-6)", "var(--community-7)", "var(--community-8)", "", "",
        ]
        assert none == ""
        # Eight colours in each theme, and the dot's fill follows the switch. Dark is
        # written twice: once for a system in dark, once for a viewer who picked it.
        from chess_variant_sandbox.web.page import tokens

        for slot in range(1, 9):
            assert STYLE.count(f"--community-{slot}:") == 3, slot
            assert f"community-{slot}" in tokens.LIGHT and f"community-{slot}" in tokens.DARK
        fill = self._body("explorerFill")
        assert "mapScoreFill(node)" in fill and "mapCommunityColour(node.community)" in fill

    def test_a_community_chip_lights_its_members_and_the_lines_between_them(self) -> None:
        nodes = [{"hash": name, "name": name, "names": [name]} for name in "abcd"]
        lines = [
            {"source": "a", "target": "b", "likeness": 0.9},
            {"source": "b", "target": "c", "likeness": 0.5},
            {"source": "a", "target": "c", "likeness": 0.3},
            {"source": "c", "target": "d", "likeness": 0.7},
        ]
        community = {"id": "k", "members": ["a", "b", "c"], "names": ["a", "b", "c"],
                     "size": 3, "centre": "b", "name": "k"}
        payload = {"nodes": nodes, "lineage": [], "neighbours": lines,
                   "communities": [community], "counts": {}}
        shown = _in_the_page(
            f"MAP_DATA = {json.dumps(payload)}; MAP_PICK = {{ kind: 'community', at: 0 }};",
            "JSON.stringify([...mapLit().lines].sort())",
        )
        # The line out to `d` is not the community's.
        assert json.loads(shown) == [0, 1, 2]
    def test_hover_lights_the_drawn_neighbours_and_gives_way_to_a_pick(self) -> None:
        from page_text import EXPLORER

        assert "EXPLORER.on('mouseover', 'node'" in EXPLORER
        assert "EXPLORER.on('mouseout', 'node'" in EXPLORER
        assert ".not('.off, .away')" in self._body("explorerDrawnLines")
        hover = self._body("explorerHover")
        assert "if (!EXPLORER || MAP_PICK" in hover
        assert "addClass('hushed')" in hover
        assert "explorerHushOff();" in self._body("explorerPick")
        # A fade is a colour, as every fade on the map is.
        assert "{ selector: '.faded, .hushed', style: {" in EXPLORER

    def test_the_local_view_steps_out_along_the_drawn_lines_and_has_a_way_back(
        self,
    ) -> None:
        from page_text import GRAPH

        local = self._body("explorerPaintLocal")
        assert "explorerDrawnLines(kept).connectedNodes()" in local
        assert "EXPLORER_LOCAL.steps" in local
        assert "position" not in local
        assert "data-local-leave" in self._body("explorerLocalSay")
        assert "explorerLocal(null, 0)" in self._body("explorerLetGo")
        pressed = self._body("explorerKey")
        assert "event.key === 'l'" in pressed and "explorerStepOut()" in pressed
        assert "explorerLetGo()" in pressed
        assert "data-local=" in GRAPH

    def test_the_find_box_centres_and_holds_its_first_match(self) -> None:
        held = self._body("explorerHoldFirst")
        assert "addClass('held')" in held and "center: { eles: dot }" in held
        assert "explorerHoldFirst();" in self._body("wireExplorer")

    def test_names_are_painted_by_priority_community_centres_first(self) -> None:
        order = self._body("mapLabelOrder")
        assert "community.centre" in order
        nodes = [
            {"hash": name, "name": name, "names": [name], "degree": degree}
            for name, degree in (("busy", 9), ("hub", 1), ("small", 0), ("quiet", 2))
        ]
        communities = [
            {"id": "one", "centre": "hub", "size": 5},
            {"id": "two", "centre": "small", "size": 2},
        ]
        shown = _in_the_page(
            f"MAP_DATA = {json.dumps({'nodes': nodes, 'communities': communities})};",
            "JSON.stringify(mapLabelOrder(new Set(['quiet']))"
            ".map(at => MAP_DATA.nodes[at].name))",
        )
        assert json.loads(shown) == ["quiet", "hub", "small", "busy"]


class TestTheOutlines:
    """`docs/archive/plan-graph-map.md` S5: a smooth outline round each community.

    The outline is the vendored bubble sets' drawing; what the page decides is what each
    one holds -- the community's games on screen and the drawn likeness lines between
    them -- and when it is drawn again. These run the page's own functions in node over a
    headless drawing by the vendored graph library itself, with the outline library
    replaced by a recorder, because the suite runs no browser.
    """

    @staticmethod
    def _body(name: str) -> str:
        from page_text import EXPLORER

        start = EXPLORER.index(f"function {name}(")
        rest = EXPLORER.find("\nfunction ", start + 1)
        return EXPLORER[start:rest if rest > 0 else len(EXPLORER)]

    # Two communities of three, one pair, and one game in none, joined by similarity
    # lines, drawn headless: `a b c` are one community, `d e f` another.
    SETUP = """
      // Read into this context rather than `require`d into node's own: the library
      // tells a plain object by its prototype, and one from another realm is not one.
      const held = { exports: {} };
      new Function('module', 'exports', require('fs').readFileSync(%(library)s, 'utf8'))(
        held, held.exports);
      const cytoscape = held.exports;
      global.requestAnimationFrame = () => 0;
      const fake = () => ({ style: {}, childNodes: [], setAttribute() {}, remove() {},
        removeAttribute() {}, appendChild(kid) { this.childNodes.push(kid); },
        classList: { toggle() {}, add() {} }, textContent: '' });
      document.createElementNS = () => fake();
      document.createTextNode = said => ({ textContent: said });
      const names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'lone'];
      MAP_DATA = {
        nodes: names.map(name => ({ hash: name, name: name, names: [name] })),
        communities: [
          { id: 'one', name: 'the first community', size: 3, members: ['a', 'b', 'c'] },
          { id: 'two', name: 'the second', size: 3, members: ['d', 'e', 'f'] },
          { id: 'pair', name: 'a pair', size: 2, members: ['g', 'h'] },
        ],
        lineage: [], neighbours: [],
      };
      const pairs = [['a', 'b'], ['b', 'c'], ['a', 'c'], ['d', 'e'], ['e', 'f'], ['c', 'd'],
                     ['g', 'h']];
      EXPLORER = cytoscape({ headless: true, elements: names.map((name, at) => ({
          group: 'nodes', data: { id: 'n' + at, at: at, kind: 'node', name: name },
          position: { x: at * 10, y: 0 } }))
        .concat(pairs.map(([from, to], at) => ({ group: 'edges', classes: 'line',
          data: { id: 's' + at, kind: 'line', source: 'n' + names.indexOf(from),
                  target: 'n' + names.indexOf(to) } }))) });
      const drawn = [];
      EXPLORER_BUBBLES = {
        layer: { node: { appendChild() {} } },
        addPath: (nodes, lines, avoid, options) => {
          const path = { nodes: nodes.map(n => n.data('name')).sort(),
                         lines: lines.map(e => e.id()).sort(),
                         avoid: avoid.map(n => n.data('name')).sort(),
                         fill: options.style.fill, node: { style: {} }, gone: false };
          drawn.push(path);
          return path;
        },
        removePath: path => { path.gone = true; },
      };
      EXPLORER_HULLS = true;
      const now = () => drawn.filter(path => !path.gone);
    """

    def _run(self, steps: str) -> dict:
        from chess_variant_sandbox.web import server

        library = json.dumps(str(server.VENDOR_DIR / "cytoscape-3.34.3.min.js"))
        shown = _in_the_page(self.SETUP % {"library": library} + steps, "JSON.stringify(said)")
        return json.loads(shown)

    def test_an_outline_holds_its_community_and_the_lines_between_its_games(self) -> None:
        said = self._run("""
          explorerPaintHulls();
          const said = { first: now(), labels: [...EXPLORER_OUTLINES.values()].map(held =>
            held.label.childNodes[0].childNodes.map(kid => kid.textContent).join('')) };
        """)
        first = {tuple(path["nodes"]): path for path in said["first"]}
        # One outline per community of three or more; the pair and the loner have none.
        assert set(first) == {("a", "b", "c"), ("d", "e", "f")}
        one = first[("a", "b", "c")]
        # The lines inside it, and not the line out to `d`, which is between communities.
        assert one["lines"] == ["s0", "s1", "s2"]
        # It keeps clear of every other game on screen.
        assert one["avoid"] == ["d", "e", "f", "g", "h", "lone"]
        # In its community's colour, and under the server's name, led by that colour.
        assert one["fill"] == "var(--community-1)"
        assert first[("d", "e", "f")]["fill"] == "var(--community-2)"
        assert said["labels"][0] == "\u25cf the first community"

    def test_the_outlines_are_drawn_again_when_a_line_the_local_view_or_a_fold_changes(
        self,
    ) -> None:
        """Only an outline whose holding changed is drawn again; the rest are left."""
        said = self._run("""
          explorerPaintHulls();
          const count = () => drawn.length;
          const said = {};
          // The same picture again draws nothing.
          explorerPaintHulls();
          said.again = count();
          // A line inside the first community is taken off the drawing.
          EXPLORER.getElementById('s2').addClass('off');
          explorerPaintHulls();
          said.cut = [count(), now().map(path => path.lines)];
          // The local view sets `f` aside: the second outline holds two games.
          EXPLORER.getElementById('n5').addClass('aside');
          explorerPaintHulls();
          said.local = [count(), now().map(path => path.nodes)];
          // And sets `e` aside too: one game left is no outline.
          EXPLORER.getElementById('n4').addClass('aside');
          explorerPaintHulls();
          said.alone = now().map(path => path.nodes);
          EXPLORER.nodes().removeClass('aside');
          // A folded community is its dot, and has no outline.
          EXPLORER_FOLDED.add('one');
          explorerPaintHulls();
          said.folded = now().map(path => path.nodes);
          // The switch off takes every outline away.
          EXPLORER_FOLDED = new Set();
          EXPLORER_HULLS = false;
          explorerPaintHulls();
          said.off = [now().length, EXPLORER_OUTLINES.size];
        """)
        assert said["again"] == 2
        assert said["cut"][0] == 3
        assert sorted(said["cut"][1]) == [["s0", "s1"], ["s3", "s4"]]
        assert said["local"][0] == 4
        assert sorted(said["local"][1]) == [["a", "b", "c"], ["d", "e"]]
        assert said["alone"] == [["a", "b", "c"]]
        assert said["folded"] == [["d", "e", "f"]]
        assert said["off"] == [0, 0]

    def test_every_change_to_what_is_drawn_reaches_the_outlines(self) -> None:
        """The arrows' switch, a fold, *hide the rest* and the local view all
        repaint through `explorerPaintLit`, which ends in the local view's paint, which
        ends in the outlines'."""
        assert "explorerPaintHulls();" in self._body("explorerPaintLocal")
        assert "explorerPaintLocal();" in self._body("explorerPaintLit")
        for caller in ("explorerSetArrows", "explorerFoldDone", "explorerLocal"):
            assert "explorerPaintLit();" in self._body(caller), caller
        # A pick or a pointer changes what is lit, not what is drawn: a colour only.
        assert "explorerHullsFade();" in self._body("explorerHover")
        assert "explorerHullsFade();" in self._body("explorerHushOff")
        # A drag redraws an outline by itself; its name follows.
        assert "attributeFilter: ['d']" in self._body("explorerWatchHulls")

    def test_the_switch_loads_the_outlines_and_says_so_when_it_cannot(self) -> None:
        from page_text import EXPLORER

        from chess_variant_sandbox.web.page.style import STYLE

        assert 'id="exploreHulls"' in self._body("explorerLinesHtml")
        assert "outline the communities" in EXPLORER
        switch = self._body("explorerSetHulls")
        assert "await explorerHullLibrary()" in switch
        assert "did not load them" in switch
        assert "explorerSetHulls(EXPLORER_HULLS);" in self._body("showExplorer")
        assert "explorerSetHulls(EXPLORER_HULLS_OPEN);" in self._body("explorerReset")
        # A name is kept only where it fits and covers no game, as a dot's name is.
        names = self._body("explorerHullNames")
        assert "mapOverlaps(other, box)" in names and "renderedBoundingBox" in names
        assert "explorerHullNames(frame)" in self._body("explorerPaintNames")
        assert ".mapHullName {" in STYLE and "pointer-events: none" in STYLE

    def test_a_community_drawn_in_two_places_is_outlined_in_each(self) -> None:
        """M4: the server says where a community is drawn (`parts`), and each place gets an
        outline of its own in the community's one colour, rather than one outline stretched
        across the map between them.

        This replaces Q477's *no two outlines can meet on the server's picture at any cut*,
        a promise of the layout by community (`COMPONENT_GAP` between knots) that M3's
        picture does not make: it draws games by likeness, so two communities alike can sit
        side by side. What holds instead is the first test here -- an outline keeps clear
        of every drawn game that is not its own.
        """
        said = self._run("""
          MAP_DATA.communities[0].parts = [['a', 'b'], ['c']];
          MAP_DATA.communities[1].parts = [['d', 'e', 'f']];
          explorerPaintHulls();
          const said = { ids: [...EXPLORER_OUTLINES.keys()].sort(),
                         drawn: now().map(path => [path.nodes, path.fill]) };
        """)
        # `c` alone is one game, which is no outline; `a b` is, under the community's colour.
        assert said["ids"] == ["one#0", "two#0"]
        assert sorted(said["drawn"]) == [
            [["a", "b"], "var(--community-1)"], [["d", "e", "f"], "var(--community-2)"],
        ]
        both = self._run("""
          MAP_DATA.communities[0].parts = [['a', 'b'], ['c', 'd']];
          MAP_DATA.communities[0].members = ['a', 'b', 'c', 'd'];
          MAP_DATA.communities[1].members = ['e', 'f'];
          MAP_DATA.communities[1].size = 2;
          explorerPaintHulls();
          const said = now().map(path => [path.nodes, path.fill]);
        """)
        assert sorted(both) == [
            [["a", "b"], "var(--community-1)"], [["c", "d"], "var(--community-1)"],
        ]


SAMPLE_CHANGELOG = """# Changelog

What this project can do now, newest first. Instructions to whoever writes an entry.

---

## 2026-09-18

### A map of the whole library

The library has a map view. It works on a phone.

### Measuring a game from the command line

`chess-variant-sandbox measure --seeds four` measures a game on four seeds. The site shows the
one-run scores until the next deploy, which *waits on your word*.

## 2026-09-15

### Hexagonal boards

A game can now say its board is hexagonal.
"""


def _news_drawn(payload: dict) -> dict:
    """Run `paintNews` in node over one payload and report the tree it built.

    Every element here refuses `innerHTML` and there is no way to make a tag but
    `createElement`, so a run whose text is markup can only arrive as a text node. That
    is what makes the safety claim a test rather than a promise: the assertion is about
    the tree, not about a string somebody escaped.

    The page's own boot runs at load, so `fetch` never resolves -- `identify()` waits
    for `/api/me` for ever and no view is drawn behind this one.
    """
    import json as _json
    import re

    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      const loose = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                                    set: () => true });
      function __el(tag) {
        const node = {
          tag: tag, className: '', children: [], _text: '',
          get textContent() { return node._text; },
          set textContent(v) { node._text = String(v); node.children = []; },
          set innerHTML(v) { throw new Error('this element must not be given markup'); },
          append(...kids) { kids.forEach(kid => node.children.push(kid)); },
        };
        return node;
      }
      global.document = {
        createElement: tag => __el(tag),
        createTextNode: said => ({ tag: '#text', _text: String(said) }),
        getElementById: () => loose, querySelector: () => null,
        querySelectorAll: () => [], addEventListener: () => {},
        body: loose,
      };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
      global.location = { pathname: '/game', hash: '' };
      global.fetch = () => new Promise(() => {});
    """
    runner = f"""
      const dump = node => node.tag === '#text'
        ? {{ tag: '#text', text: node._text }}
        : {{ tag: node.tag, className: node.className, text: node._text,
             children: (node.children || []).map(dump) }};
      const where = __el('div');
      paintNews(where, {_json.dumps(payload)});
      console.log(JSON.stringify(dump(where)));
    """
    done = node_session.run(stub + script.group(1) + "\n" + runner)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _texts(tree: dict) -> list[str]:
    """Every string in a drawn tree, whatever element it sits in."""
    said = [tree["text"]] if tree.get("text") else []
    for child in tree.get("children", []):
        said += _texts(child)
    return said


class TestTheExplorer:
    """`docs/archive/plan-game-graph.md` §8: the map at `#/map`, drawn by the graph library.

    Two things are held down here. The **vendored file** is the registry's bytes and
    nothing else -- the manifest names its digest, the route serves what the manifest
    names and refuses every other name, and the file is never edited, so the digest is
    what notices if it ever is. And the drawing **places and never computes**, which is
    the line `TestTheMap` above draws, read over `page/explorer.js` in its own terms.
    """

    # The digest `docs/archive/plan-game-graph.md` §8.1 names, read from the registry's
    # tarball on 2026-09-19. It is written here as well as in the manifest so that an
    # edited manifest and an edited file cannot agree with each other.
    CYTOSCAPE = "5f3b5b529546d5af1fc5628590af033b74511a5b6f789f5f4682845863228b91"

    # The outlines' two files (`docs/archive/plan-graph-map.md` S5), the registry's UMD
    # builds read from their tarballs on 2026-09-26: file -> (bytes, sha256, the notice
    # its licence file carries).
    OUTLINES = {
        "cytoscape-layers-3.1.0.min.js": (
            16025, "5c70894cfdd70c0024126d208c7060e3dd095e51797ec274062363b30b0e5b8e",
            "Copyright (c) 2020 Samuel Gratzl",
        ),
        "cytoscape-bubblesets-4.1.0.min.js": (
            27841, "2c6309b1cf14c1f0a63e4d7bd459d0b93e1aec9100305c920332338edc0e792e",
            "Copyright (c) 2021-2022 Samuel Gratzl",
        ),
    }

    @staticmethod
    def _vendor():
        from chess_variant_sandbox.web import server

        return server.VENDOR_DIR, server.vendored()

    def test_the_vendored_file_is_the_one_the_plan_named(self) -> None:
        """Byte for byte the registry's, with its licence and its row beside it."""
        import hashlib

        where, manifest = self._vendor()
        entry = manifest["cytoscape-3.34.3.min.js"]
        body = (where / entry["file"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == entry["sha256"] == self.CYTOSCAPE
        assert len(body) == entry["bytes"] == 435503
        assert entry["licence"] == "MIT"
        assert entry["source"].startswith("https://registry.npmjs.org/")
        licence = (where / entry["licenceFile"]).read_text(encoding="utf-8")
        assert "Cytoscape Consortium" in licence and "MIT" in licence + entry["licence"]
        # The header of the file itself carries the same notice, so the two travel
        # together even where only the one file is read.
        assert entry["copyright"] in body[:400].decode("utf-8")

    @pytest.mark.parametrize("name", sorted(OUTLINES))
    def test_the_outlines_files_are_pinned_by_version_and_digest(self, name: str) -> None:
        """S5's two files, pinned the way the graph library is: version in the name, the
        registry's bytes, the digest written here and in the manifest, and the licence
        beside them -- theirs, and those of the three packages the bubble-sets file
        carries inside it, each named in the row's `bundled`."""
        import hashlib

        where, manifest = self._vendor()
        entry = manifest[name]
        size, digest, notice = self.OUTLINES[name]
        body = (where / name).read_bytes()
        assert hashlib.sha256(body).hexdigest() == entry["sha256"] == digest
        assert len(body) == entry["bytes"] == size
        assert entry["version"] in name
        assert entry["source"] == (
            f"https://registry.npmjs.org/{entry['name']}/-/"
            f"{entry['name']}-{entry['version']}.tgz"
        )
        assert entry["path"] == "package/build/index.umd.min.js"
        assert entry["licence"] == "MIT" and entry["copyright"] == notice
        licence = (where / entry["licenceFile"]).read_text(encoding="utf-8")
        assert notice in licence and "MIT License" in licence
        for inside in entry.get("bundled", []):
            said = (where / inside["licenceFile"]).read_text(encoding="utf-8")
            assert said.strip(), inside["name"]
            assert inside["source"].startswith("https://registry.npmjs.org/")
        # A UMD file that sets a global and asks for nothing but what comes before it.
        head = body[:600].decode("utf-8")
        assert '"function"==typeof define&&define.amd' in head

    def test_the_bubble_sets_file_names_what_it_carries(self) -> None:
        """The file bundles bubblesets-js, lodash.throttle and tslib; the row says so and
        each has its own licence file, the one of them not MIT (tslib's 0BSD) included."""
        where, manifest = self._vendor()
        entry = manifest["cytoscape-bubblesets-4.1.0.min.js"]
        carried = {inside["name"]: inside for inside in entry["bundled"]}
        assert set(carried) == {"bubblesets-js", "lodash.throttle", "tslib"}
        assert carried["tslib"]["licence"] == "0BSD"
        assert "Microsoft" in (where / carried["tslib"]["licenceFile"]).read_text(
            encoding="utf-8"
        )
        assert "Samuel Gratzl" in (
            where / carried["bubblesets-js"]["licenceFile"]
        ).read_text(encoding="utf-8")
        assert "jQuery Foundation" in (
            where / carried["lodash.throttle"]["licenceFile"]
        ).read_text(encoding="utf-8")

    def test_the_file_reaches_a_reader_through_the_installed_package(self) -> None:
        """The box installs the package and reads the file out of it, not out of a
        checkout, so the directory is found from the module rather than from the tree."""
        from pathlib import Path

        import chess_variant_sandbox.web

        where, manifest = self._vendor()
        assert where == Path(chess_variant_sandbox.web.__file__).resolve().parent / "static" / "vendor"
        for name in manifest:
            assert (where / name).is_file()

    def test_the_route_answers_the_manifests_bytes_and_nothing_else(self) -> None:
        import hashlib
        from urllib.error import HTTPError

        where, manifest = self._vendor()
        app = App(RULESETS, library.RUNS_DIR)
        server = Server(("127.0.0.1", 0), handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            for name, (_size, digest, _notice) in self.OUTLINES.items():
                with urlopen(base + "/static/vendor/" + name) as response:
                    assert hashlib.sha256(response.read()).hexdigest() == digest
                    assert "immutable" in response.headers["Cache-Control"]
            with urlopen(base + "/static/vendor/cytoscape-3.34.3.min.js") as response:
                body = response.read()
                assert response.status == 200
                assert response.headers["Content-Type"] == "text/javascript"
                # A year and immutable, because the version is in the name: a new version
                # is a new request rather than new bytes under the old name.
                assert response.headers["Cache-Control"] == (
                    "public, max-age=31536000, immutable"
                )
            assert hashlib.sha256(body).hexdigest() == self.CYTOSCAPE
            assert body == (where / "cytoscape-3.34.3.min.js").read_bytes()
            # Every other name is a 404, including the manifest itself and a path that
            # tries to walk out of the directory.
            for asked in (
                "/static/vendor/manifest.json",
                "/static/vendor/cytoscape.min.js",
                "/static/vendor/../server.py",
                "/static/vendor/",
            ):
                with pytest.raises(HTTPError) as raised:
                    urlopen(base + asked)
                assert raised.value.code == 404, asked
        finally:
            server.shutdown()
            server.server_close()

    def test_the_explorer_is_the_only_thing_that_asks_for_a_vendored_file(self) -> None:
        """One way of asking, used when the explorer opens and never before.

        The library, the board and the designer request nothing under `/static/vendor/`,
        so a person who never opens the explorer never pays the 136 KB. The one other path
        under `/static/` is B3's measuring worker, which is not vendored and is started
        only when somebody presses *measure it here* in the designer (B5, Q456). The
        outlines' two files are asked for after the graph library, in order, the first
        time the outlines are on (S5). The typefaces under `/static/fonts/` are the
        shell's, not vendored code, and every page asks for them (V1), as it does for the
        shell's own script, `/static/shell.js` (V2), and the landing's draw after it,
        `/static/story.js` (LP6).
        """
        from page_text import EXPLORER, MEASURE, PAGE

        # The head's scripts under their versioned names (G12, `/static/shell.<digest>.js`).
        others = (PAGE.count("/static/") - PAGE.count("/static/fonts/")
                  - PAGE.count("/static/shell.") - PAGE.count("/static/story."))
        assert PAGE.count("/static/vendor/") == 1 and others == 2
        assert MEASURE.count("'/static/measure-worker.js'") == 1
        assert "const EXPLORER_VENDOR = '/static/vendor/';" in EXPLORER
        assert "const EXPLORER_SRC = 'cytoscape-3.34.3.min.js';" in EXPLORER
        assert (
            "const EXPLORER_HULL_SRC = ['cytoscape-layers-3.1.0.min.js',"
            " 'cytoscape-bubblesets-4.1.0.min.js'];" in EXPLORER
        )
        # Every file the page names is one the manifest serves.
        _where, manifest = self._vendor()
        for named in re.findall(r"'([\w.-]+\.min\.js)'", EXPLORER):
            assert named in manifest, named
        # Asked for inside the view rather than in the page's markup, which is what makes
        # it a cost only the explorer's readers pay.
        assert "<script" not in EXPLORER
        assert "tag.src = EXPLORER_VENDOR + file" in EXPLORER
        # The second outline file reads the first's global as it loads, so it is asked
        # for only once the first has arrived.
        loads = self._body("explorerHullLibrary")
        assert "explorerScript(EXPLORER_HULL_SRC[0])" in loads
        assert ".then(first => first && explorerScript(EXPLORER_HULL_SRC[1]))" in loads

    def test_the_explorer_draws_from_one_payload_and_one_named_overlay(self) -> None:
        """The drawing is `/api/graph` and nothing else.

        G12 adds the one further reading the payload itself names, and the route it asks
        for is a constant rather than a path the page builds -- so the explorer reaches
        two endpoints, both written out here, and no third can appear without this test
        saying so.
        """
        from page_text import EXPLORER

        assert EXPLORER.count("await api(") == 2
        assert "api('/api/graph')" in EXPLORER
        assert "const EXPLORER_OVERLAY_PATH = '/api/graph/people';" in EXPLORER
        assert re.findall(r"api\('(/api/[^']*)'", EXPLORER) == ["/api/graph"]
        assert re.findall(r"api\(EXPLORER_OVERLAY_PATH\)", EXPLORER) == [
            "api(EXPLORER_OVERLAY_PATH)"
        ]

    def test_the_explorer_places_and_never_computes(self) -> None:
        """G8 inherits the map's rule exactly (`docs/archive/plan-game-graph.md` §8.2).

        The coordinates are the server's; what this module decides is how many model
        pixels one layout unit is. No trigonometry, no root, no logarithm, nothing that
        formats a number, and no reading of a rule set -- which games are near which and
        what differs between them arrive decided.
        """
        from page_text import EXPLORER

        for computed in ("Math.cos(", "Math.sin(", "Math.atan2(", "Math.sqrt(",
                         "Math.log(", "Math.pow(", "toFixed", "differences.length",
                         "node.x =", "node.y ="):
            assert computed not in EXPLORER, computed
        for named in ("game.spec", "relate(", "compile(", "rules."):
            assert named not in EXPLORER, named
        # The server's picture, taken as given: the library is told to place rather than
        # to lay out, and one function turns a payload coordinate into a model one.
        assert "name: 'preset'" in EXPLORER
        # One function turns a payload coordinate into a model one -- and a group's place
        # is a coordinate like any other, so a folded dot goes through the same one. The
        # fourth is how wide a dot is drawn, which is presentation.
        # The outlines' reach is in layout units too, and is presentation like a width.
        reach = EXPLORER[EXPLORER.index("const EXPLORER_HULL_REACH = {"):]
        reach = reach[: reach.index("};")]
        assert reach.count("* EXPLORER_UNIT") == 5
        assert EXPLORER.replace(reach, "").count("* EXPLORER_UNIT") == 4
        assert "function explorerPlace(node)" in EXPLORER
        assert "explorerPlace(group)" in EXPLORER

    def test_no_kind_of_rule_is_named_in_the_explorer(self) -> None:
        """A colour comes from the payload's kind through the map's own function."""
        from page_text import EXPLORER

        from chess_variant_sandbox.spec import relate

        assert "mapKindColour" not in EXPLORER
        for kind in relate.KINDS:
            assert f"'{kind}'" not in EXPLORER, kind

    def test_the_four_kinds_of_line_the_fill_and_the_ring(self) -> None:
        """What the map draws, drawn again in the library's own stylesheet."""
        from page_text import EXPLORER

        # A line as wide as its likeness, and grey with the arrowhead at the middle. The
        # dashed gap and the dotted nearest relative left with the rule steps (M4).
        assert "{ selector: 'edge.line', style: { 'width': EXPLORER_WIDTH } }" in EXPLORER
        assert "'line-style'" not in EXPLORER
        assert "'mid-target-arrow-shape': 'triangle'" in EXPLORER
        assert "explorerColour('var(--faint)')" in EXPLORER
        # The fill is the map's ramp, resolved: nothing here decides what a score looks
        # like, and a game with no score is the hollow dot the map already draws.
        assert "mapScoreFill(node)" in EXPLORER
        assert "explorerColour(fill || 'var(--panel)')" in EXPLORER
        # The ring, in `--bad`, at half the dot's width so it sits inside the radius the
        # layout reserved.
        assert "node.score.viable === false" in EXPLORER
        assert "across / 2" in EXPLORER
        assert "'border-color': explorerColour('var(--bad)')" in EXPLORER
        # A fade is a colour and never an opacity, which is the map's rule. The one
        # opacity in the stylesheet is the ring's own fill, which is a hole rather than
        # a fade: it lets the dot it marks show through.
        assert re.findall(r"'[a-z-]*opacity'", EXPLORER) == ["'background-opacity'"]

    def test_a_tap_opens_the_maps_own_panels(self) -> None:
        """The panel functions are `page/graph.js`'s, called over the state they read."""
        from page_text import EXPLORER

        assert "EXPLORER.on('tap'" in EXPLORER
        assert "MAP_PICK = pick" in EXPLORER and "MAP_ON = mapLit()" in EXPLORER
        assert "paintMapPanel()" in EXPLORER
        # No second opinion about what a panel says: nothing here draws one.
        for drawn in ("mapNodePanel", "mapLinkPanel", "mapArrowPanel", "mapNothingPanel"):
            assert f"function {drawn}" not in EXPLORER

    def test_pan_and_zoom_are_the_librarys_own(self) -> None:
        from page_text import EXPLORER

        # Wheel, drag and pinch are on by default, and so is dragging one dot now that
        # G10's *reset* can undo it and the ring follows the dot it marks.
        assert "autoungrabify" not in EXPLORER
        assert "userZoom: false" not in EXPLORER and "userPanning: false" not in EXPLORER
        assert "EXPLORER.fit(" in EXPLORER and "EXPLORER.zoom({" in EXPLORER

    def test_a_blocked_script_says_so_and_offers_the_list(self) -> None:
        """The explorer is the map now, so the way out of it is the other view.

        A sentence offering the map would be a link back to the page the reader is
        already on and cannot see.
        """
        from page_text import EXPLORER

        assert "tag.onerror" in EXPLORER
        assert "explorerSayBlocked" in EXPLORER
        assert 'href="#/"' in EXPLORER and 'href="#/map"' not in EXPLORER

    def test_the_explorer_is_the_map(self) -> None:
        """G13: one route, one drawing, and no second map to keep in step.

        `#/map/explore` is the address the explorer was built at, and a link somebody
        kept lands on the same view rather than on nothing.
        """
        from page_text import CORE, PAGE

        assert "parts[0] === 'map') return showExplorer().catch(fail);" in CORE
        assert "'explore'" not in CORE
        assert PAGE.count("showExplorer()") == 2
        # Nothing links to the old address any more, and nothing goes back to a map.
        assert 'href="#/map/explore"' not in PAGE
        assert 'data-back="map"' not in PAGE

    # ----------------------------------- G9: everything the map does, in the explorer

    @staticmethod
    def _body(name: str) -> str:
        """One function's source, from its declaration to the next one's."""
        from page_text import EXPLORER

        start = EXPLORER.index(f"function {name}(")
        rest = EXPLORER.find("\nfunction ", start + 1)
        return EXPLORER[start:rest if rest > 0 else len(EXPLORER)]

    def test_the_legend_the_chips_and_the_find_box_are_the_maps_own(self) -> None:
        """G9 is parity, and parity means calling the map's functions, not copying them.

        The legend -- what a line means, which rule a colour is, the families, and both
        ends of the score ramp printed as numbers -- is `mapLegendHtml`'s, and which
        games a filter lights is `mapFound`'s over `mapMatches`. A second reading of
        either would be a second opinion about one library.
        """
        from page_text import EXPLORER

        assert "${mapLegendHtml()}" in EXPLORER
        assert "mapFound()" in EXPLORER
        assert "mapArrowHidden()" in EXPLORER
        assert "mapSayLit(found)" in EXPLORER
        # The same five states, from the listing's own score menu, and one of each
        # control: they are written here because the shell is, not read from the map.
        assert EXPLORER.count('id="mapShow"') == 1
        assert EXPLORER.count('id="mapFind"') == 1
        for state in ('value="yes"', 'value="viable"', 'value="stale"', 'value="no"'):
            assert state in EXPLORER, state
        # Nothing here draws a legend, a panel or a swatch of its own.
        for drawn in ("mapLegendHtml", "mapScaleHtml", "mapLineSwatch"):
            assert f"function {drawn}" not in EXPLORER

    def test_the_arrows_switch_moves_no_dot(self) -> None:
        """The plan's acceptance for this stage, and the map's own test of it.

        The arrows' switch reaches the arrows and nothing else. Every position arrived
        with the payload and was written once; the paint the switch calls selects only
        edges. (The kind chips this held until M4 left with the rule-step lines.)
        """
        hiding = self._body("explorerPaintLines")
        for reached in ("nodes(", "position", "getElementById('n"):
            assert reached not in hiding, reached
        assert "EXPLORER.edges()" in hiding
        assert "explorerPaintLines();" in self._body("explorerSetArrows")
    def test_a_filter_lights_and_a_line_with_an_unlit_end_fades(self) -> None:
        """A filter lights; it never hides. A line between two lit games keeps its colour.

        The fade is a class that paints in `--faint`, never an opacity, so a faded
        hairline is still exactly a hairline. The legend's counts are the payload's and
        no filter touches them, which is why how many games are lit is said separately.
        """
        from page_text import EXPLORER, GRAPH

        lit = self._body("explorerPaintLit")
        assert "lit(line.source) && lit(line.target)" in lit
        assert "toggleClass('faded'" in lit
        # Nothing in the lighting pass takes a game away except the one control that
        # says it does, and that control is a reader's own press. Twice, because a folded
        # dot answers for its games and goes when none of them is left.
        assert lit.count("toggleClass('hidden'") == 2
        assert lit.count("EXPLORER_ONLY && !shown") == 2
        assert "faded rather than gone" in GRAPH
        # The one sentence that has to differ, because by then they are gone.
        assert "The rest are" in self._body("explorerSayLit")
        assert "hidden, and so is every line with an end among them" in EXPLORER

    def test_hide_the_rest_is_the_only_thing_that_takes_a_game_off_the_drawing(
        self,
    ) -> None:
        """Four classes carry `display: none` and no other style does.

        A chip, the cut or the rule-step switch hiding a line, *hide the rest*, a fold
        -- which takes away the games it folded and puts one dot in their place -- and
        the local view (S4), which a reader asks for by name and leaves by a button. A
        line with an end that is not displayed is not drawn, which is the library's own
        rule and why hiding a game says nothing about its lines.
        """
        from page_text import EXPLORER

        assert EXPLORER.count("'display':") == 1
        # An outline's name the frame has no room for is not drawn -- an SVG attribute on
        # the name, as a dot's name is simply not painted -- and takes no game with it.
        assert EXPLORER.count("setAttribute('display', 'none')") == 1
        assert "setAttribute('display', 'none')" in self._body("explorerHullNames")
        assert (
            "{ selector: '.off, .hidden, .away, .aside', style: { 'display': 'none' } }"
            in EXPLORER
        )
        # `aside` is written in one place, the local view's own paint.
        assert EXPLORER.count("toggleClass('aside'") == 1
        assert "toggleClass('aside'" in self._body("explorerPaintLocal")
        assert "EXPLORER_ONLY = only.checked" in EXPLORER

    def test_every_game_is_a_tab_stop_and_the_keys_move_the_map(self) -> None:
        """A drawing on a canvas has no Tab stops, so the page supplies them.

        One button per game, in the payload's order, each saying its name and its score
        in the words a card uses. Enter picks the game, centres it and moves the focus
        to *open this game*; the arrows, the zoom keys and 0 go through the library's
        own `panBy`, `zoom` and `fit`.
        """
        from chess_variant_sandbox.web.page.style import STYLE

        keys = self._body("explorerKeysHtml")
        assert "MAP_DATA.nodes.map((node, index)" in keys
        assert "mapScoreWords(node)" in keys
        assert "data-game=" in keys
        pressed = self._body("explorerKey")
        for key in ("'ArrowLeft'", "'ArrowRight'", "'ArrowUp'", "'ArrowDown'",
                    "'Enter'", "'Escape'", "'0'"):
            assert key in pressed, key
        assert "EXPLORER.panBy(" in pressed and "EXPLORER.fit(" in pressed
        opened = self._body("explorerOpenGame")
        assert "explorerCentre(index)" in opened
        assert "#mapPanel [data-open]" in opened and "open.focus()" in opened
        # Clipped rather than hidden, because `display: none` would take them out of the
        # Tab order, which is the one thing they are for.
        assert ".exploreKeys button { position: absolute" in STYLE
        assert "clip-path: inset(50%)" in STYLE
        assert ".exploreKeys button:focus { clip-path: none; }" in STYLE

    def test_the_names_are_the_maps_own_greedy_pass(self) -> None:
        """G3 found the name to be the scarce thing at seventy dots, not the space.

        So the order is the map's -- what is held up first, then the busiest games --
        and a name whose box lands on one already painted is left off. The size is a
        constant on screen, which is what makes *does this fit* a question about the
        view rather than about the zoom. Nothing here moves a dot: what it writes is a
        class and the size that class draws at.
        """
        from page_text import EXPLORER

        names = self._body("explorerPaintNames")
        assert "mapLabelOrder(first)" in names and "mapOverlaps(other, box)" in names
        assert "EXPLORER_TEXT_PX" in names and "EXPLORER_TEXT_WIDTH" in names
        assert "position" not in names.replace("renderedPosition", "")
        assert "addClass('named')" in names
        assert "'label': 'data(name)'" in EXPLORER
        # A pan, a zoom or a resize arrives many times a second and one pass per frame
        # is enough.
        assert "EXPLORER.on('zoom pan resize', explorerNamesSoon)" in EXPLORER
        assert "requestAnimationFrame" in self._body("explorerNamesSoon")

    def test_a_theme_change_repaints(self) -> None:
        """A resolved colour is the colour of the theme that was on when it was read.

        Cytoscape draws on a canvas and cannot read a custom property, so the page hands
        it `rgb()` and hands it over again when the theme changes: the colours from the
        payload once more and the stylesheet afresh. Positions are left alone, because
        they are not that function's to move.
        """
        from page_text import EXPLORER

        assert "matchMedia('(prefers-color-scheme: dark)')" in EXPLORER
        repaint = self._body("explorerRepaintTheme")
        assert "explorerElements()" in repaint
        assert "EXPLORER.style(explorerStyle())" in repaint
        for moved in ("position", "layout", "fit("):
            assert moved not in repaint, moved

    # ------------------------------------------------- G10: the groups that fold

    def test_a_group_folds_into_one_dot_and_opens_again(self) -> None:
        """The payload's `groups`, which are a partition: one place or none.

        Folding hides the games where they stand and shows the group's own dot, which
        the payload placed. Nothing moves, which is what makes opening the group put
        every game back where it was.
        """
        from page_text import EXPLORER

        assert "mapGroups()" in EXPLORER
        folded = self._body("explorerFoldPaint")
        assert "toggleClass('away'" in folded
        # The group's dot is the payload's place, written once with every other.
        assert "id: 'g' + index" in EXPLORER
        assert "position: explorerPlace(group)" in EXPLORER
        # Folding and opening are one press, and nothing in it moves a dot.
        toggled = self._body("explorerFoldToggle")
        assert "EXPLORER_FOLDED.add(group.id)" in toggled
        assert "EXPLORER_FOLDED.delete(group.id)" in toggled
        assert "position" not in toggled and "layout" not in toggled

    def test_a_folded_dot_carries_its_games_lines(self) -> None:
        """Not gone, and not nineteen lines drawn to one point.

        Every line a member had to a game outside is drawn once from the folded dot with
        a count on it, and two folded groups share one line. A line between two games of
        the same folded group is gone, which is what folding them into one dot says.
        """
        from page_text import EXPLORER

        lines = self._body("explorerGroupLines")
        assert "if (!folded || first === second) return;" in lines
        assert "held.count += 1" in lines
        assert "kind: 'grouped'" in lines
        # Grey, whatever it stands for: a line of likeness has no kind of rule.
        assert "colour: explorerColour('var(--muted)')" in lines
        assert "'label': 'data(count)'" in EXPLORER
        # An arrow the switch has hidden is not counted into the total, so the switch and
        # a fold read together.
        assert "if (mapArrowHidden())" in lines

    def test_a_folded_dot_opens_a_panel_for_a_group(self) -> None:
        """A group is not a game, so it does not get the game's panel.

        `mapGroupPanel` is `page/graph.js`'s, like every other panel, and its members are
        buttons: pressing one opens the group and picks that game, because a dot nobody
        can see cannot be held up.
        """
        from page_text import EXPLORER, GRAPH

        assert "function mapGroupPanel(group)" in GRAPH
        assert "mapGroupPanel(mapGroupAt(MAP_PICK.at))" in GRAPH
        assert "MAP_PICK.kind === 'group'" in GRAPH
        assert "function mapGroupPanel" not in EXPLORER
        assert "explorerPick({ kind: 'group', at: held.at })" in EXPLORER
        assert "explorerUnfold(node.dataset.nodeRow)" in EXPLORER
        assert "data-open-group" in GRAPH and "data-open-group" in EXPLORER

    def test_the_folds_are_the_communities_of_three_or_more(self) -> None:
        """S5: a community folds (from its panel since Q801), and nothing reads `groups`.

        The payload's communities are a partition, so they fold as the groups did; a pair
        is left alone, as it is left without a colour. The far games' fold went with the
        rule steps' groups, and their faint lines stay off when the rule steps come on.
        """
        from page_text import GRAPH, PAGE

        for gone in ("MAP_DATA.groups", "MAP_DATA.clusters", "data-family", "mapFamilyPanel"):
            assert gone not in PAGE, gone
        communities = [
            {"id": f"c{at}", "size": size, "members": [], "names": [], "name": f"c{at}"}
            for at, size in enumerate([13, 2, 5, 3, 2])
        ]
        shown = _in_the_page(
            f"MAP_DATA = {json.dumps({'communities': communities, 'nodes': []})};",
            "JSON.stringify(mapGroups().map(one => one.id))",
        )
        assert json.loads(shown) == ["c0", "c2", "c3"]
        assert "open this community" in GRAPH
        opened = self._body("showExplorer")
        assert "MAP_ARROWS = false;" in opened
        assert "MAP_ARROWS = false;" in self._body("explorerReset")
        assert "MAP_OFF" not in PAGE

    def test_nothing_on_the_page_lays_the_map_out_and_reset_undoes_everything(self) -> None:
        """The one control over the picture as a whole, and no layout beside it.

        *Reset* puts every dot back on the server's coordinate, opens every group and
        gives the opening lines back. *Tidy*, the library's own force layout over what was
        on screen, packed the communities together until their outlines crossed, and Q477
        retired it: the server's picture is already laid out by community, so the page
        places and never lays out.
        """
        from page_text import EXPLORER

        reset = self._body("explorerReset")
        assert "EXPLORER_FOLDED = new Set()" in reset
        assert "position(explorerPlace(node))" in reset
        # Every outline is worked out again, including one a dragged dot had dented, and
        # after the outline library's own redraws of the move rather than before them.
        assert "explorerHullsAfresh();" in reset
        afresh = self._body("explorerHullsAfresh")
        assert "EXPLORER_BUBBLES.update(true)" in afresh
        assert "EXPLORER_HULL_SETTLE" in afresh
        assert 'id="exploreReset"' in EXPLORER
        # The control, its handler and the library's force layout are gone.
        assert "exploreTidy" not in EXPLORER and "function explorerTidy" not in EXPLORER
        assert ">tidy</button>" not in EXPLORER
        assert ".layout(" not in EXPLORER and "'cose'" not in EXPLORER
        assert EXPLORER.count("layout: { name: 'preset'") == 1

    def test_a_dragged_dot_takes_its_ring_with_it(self) -> None:
        """The ring is a second node at the dot's place, so something has to move it.

        One handler, on the one thing every mover ends in: a position. A drag and a
        *reset* both go through it, and the ring itself is never dragged.
        """
        from page_text import EXPLORER

        follow = self._body("explorerRingsFollow")
        assert "EXPLORER.on('position', 'node'" in follow
        assert "getElementById('r' + dot.data('at'))" in follow
        assert "ring.position({ x: dot.position('x'), y: dot.position('y') })" in follow
        assert "grabbable: false" in EXPLORER
        assert "explorerRingsFollow();" in self._body("explorerDraw")

    def test_the_sentence_says_how_many_groups_are_folded(self) -> None:
        """The line that says how many games are lit says what a fold did as well."""
        said = self._body("explorerSayLit")
        assert "EXPLORER_FOLDED.size" in said
        assert "communities are" in said and "are drawn as" in said

    # ------------------------------------------------------ G11: size by data

    def test_the_size_menu_is_the_payloads_and_the_page_names_no_key(self) -> None:
        """The menu is `sizes`, in the payload's order, and its labels are the server's.

        A key the page knew the name of would be a key the page could size by on its
        own, which is the whole thing this stage moves to the server. So the menu is
        built by walking the payload and the chosen key is only ever read back out of
        it -- the page never spells out *links* or *score*.
        """
        from page_text import EXPLORER

        menu = self._body("explorerSizeHtml")
        assert "explorerSizes()" in menu and "size.label" in menu
        assert 'id="exploreSize"' in menu
        assert "(MAP_DATA && MAP_DATA.sizes)" in self._body("explorerSizes")
        # The one it opens on is the payload's first, which is the size the map draws.
        assert "EXPLORER_SIZE = explorerSizes().length" in EXPLORER
        for spelt in ("'links'", "'score'", "node.degree"):
            assert spelt not in EXPLORER, spelt

    def test_a_dot_is_the_payloads_number_mapped_to_a_radius(self) -> None:
        """The page places and never computes, which holds for a size too.

        `node.size` is the dot's place between the key's two ends and the server worked
        it out; what is decided here is that the smallest dot is half the largest,
        which is G3's range and is presentation. A dot with no value under the key --
        an unscored game under *score* -- draws at the smallest size.
        """
        from page_text import EXPLORER

        value = self._body("explorerSizeValue")
        assert "const held = (node.size || {});" in value
        assert "EXPLORER_SIZE in held" in value
        assert "at === null || at === undefined ? 0 : at" in value
        across = self._body("explorerAcross")
        assert "EXPLORER_SMALLEST" in across and "EXPLORER_LARGEST" in across
        assert "const EXPLORER_SMALLEST = 0.5;" in EXPLORER
        assert "const EXPLORER_LARGEST = 1.0;" in EXPLORER
        # G3's own rule is the map's and stays there; nothing here reads a link count.
        assert "mapNodeRadius" not in EXPLORER

    def test_switching_the_size_moves_no_dot(self) -> None:
        """The map's rule for a chip, and it is the same rule for the size menu.

        What the press writes is a width, on every dot and on the ring that marks one.
        Nothing in the pass touches a position, runs a layout or asks for a fit.
        """
        from page_text import EXPLORER

        sized = self._body("explorerPaintSizes")
        assert "data('across', across)" in sized
        assert "data('across', across / 2)" in sized
        for moved in ("position", "layout", "fit(", "explorerPlace"):
            assert moved not in sized, moved
        assert "explorerPaintSizes();" in EXPLORER
        # A folded group's dot is not sized by the key: it stands for several games and
        # carries a name that has to be read at the whole-map view.
        assert "getElementById('g'" not in sized

    def test_the_legend_says_what_the_size_means_and_both_its_ends(self) -> None:
        """The score ramp prints both of its ends because a colour read against an
        imagined scale is read wrong, and a size is read the same way.

        Every word and both numbers are the server's, so the page formats nothing --
        which is what `lowWords` and `highWords` are for.
        """
        from page_text import EXPLORER

        said = self._body("explorerSizeSay")
        assert "size.words" in said
        assert "size.lowWords" in said and "size.highWords" in said
        assert "The smallest dot is" in said and "the largest" in said
        assert "toFixed" not in EXPLORER and "fmt(" not in said
        # Drawn when the explorer opens and again on every change of the menu.
        assert "explorerSizeSay();" in self._body("wireExplorer")

    def test_the_server_decides_what_a_size_says_and_where_a_dot_sits_on_it(
        self, tmp_path: Path
    ) -> None:
        """The payload's half of this stage, read from the module that builds it.

        `tests/test_graphing.py` holds the menu and every dot's place on it; what this
        one holds is that the page has a payload to read them out of.
        """
        from chess_variant_sandbox.web import graphing

        graph = graphing.build([], tmp_path)
        assert [size["key"] for size in graph.as_dict()["sizes"]] == ["neighbours"]
        # Every entry carries the four things the legend needs, and the two ends are
        # real values rather than nought and one.
        assert set(graphing._sizes([], None)[0]) == {
            "key", "label", "words", "low", "high", "lowWords", "highWords",
        }

    # ------------------------------------------- G12: the site's own numbers, overlaid

    def test_the_overlay_is_asked_for_only_where_the_payload_names_it(self) -> None:
        """`overlays` is a constant per server, so `chess-variant-sandbox serve` asks for nothing.

        A checkout has no store and the two people's keys cannot be answered there. The
        page therefore does not know whether the overlay exists -- it reads the payload
        and asks, or does not -- and a failed request leaves the map drawn without it
        rather than breaking the page.
        """
        from page_text import EXPLORER

        asked = self._body("explorerPeopleOverlay")
        assert "(MAP_DATA && MAP_DATA.overlays) || []" in asked
        assert "indexOf(EXPLORER_OVERLAY) < 0) return null" in asked
        assert "catch (error) {" in asked and "return null" in asked
        assert "EXPLORER_PEOPLE = await explorerPeopleOverlay();" in EXPLORER

    def test_the_overlay_keys_onto_the_nodes_the_payload_drew(self) -> None:
        """One coordinate system, and it is the base payload's.

        The overlay is read by the node hash the map already drew a dot for, so nothing
        here matches a game up by name, by file or by anything else -- which is also
        what makes a twin one record rather than two.
        """
        from page_text import EXPLORER

        at = self._body("explorerPeopleAt")
        assert "(EXPLORER_PEOPLE && EXPLORER_PEOPLE.games) || {}" in at
        assert "return games[hash] || null" in at
        assert "explorerPeopleAt(node.hash)" in EXPLORER

    def test_the_overlays_sizes_join_the_payloads_in_one_menu(self) -> None:
        """G11's mechanism, extended rather than duplicated.

        The overlay carries the same `sizes` shape the payload does, so the menu, the
        legend's sentence and a dot's radius are the ones G11 built. The page still
        names no key: the two new ones reach it in the server's words like the others.
        """
        from page_text import EXPLORER

        sizes = self._body("explorerSizes")
        assert "(MAP_DATA && MAP_DATA.sizes) || []" in sizes
        assert "(EXPLORER_PEOPLE && EXPLORER_PEOPLE.sizes) || []" in sizes
        for spelt in ("'stars'", "'played'", "'links'", "'score'"):
            assert spelt not in EXPLORER, spelt

    def test_a_filter_lights_and_never_hides(self) -> None:
        """G4's rule, and the three filters are read under it.

        A filter narrows what the find box and the score menu left lit, and what it
        narrows is the same set those two produce -- so a game a filter drops is faded,
        its lines are where they were, and the legend's counts have not moved. *Hide the
        rest* is still the one control that takes anything away.
        """
        found = self._body("explorerFound")
        assert "const found = mapFound();" in found
        assert "if (!EXPLORER_KEEP) return found;" in found
        assert "explorerKeeps(node.hash)" in found
        for hidden in ("hidden", "remove(", "EXPLORER_ONLY"):
            assert hidden not in found, hidden
        # The lighting pass and the find box's Enter both read the narrowed set.
        assert "const found = explorerFound();" in self._body("explorerPaintLit")
        assert "const found = explorerFound();" in self._body("explorerGoToFirstFound")

    def test_a_filter_is_a_key_the_server_named_on_a_record_it_wrote(self) -> None:
        """The page knows no filter by name and works out nothing about a game.

        What the overlay sends is a list of filters, each with the key its answer is
        under, and a record per game carrying those keys. So the page asks whether one
        boolean is true -- and a game the overlay has no record for answers no filter,
        which is the *absent is not zero* an unvoted game already lives by.
        """
        from page_text import EXPLORER

        keeps = self._body("explorerKeeps")
        assert "record[one.key] === true" in keeps
        assert "!!record &&" in keeps
        for spelt in ("'unbeaten'", "'untried'", "'unplayed'", "humanGames"):
            assert spelt not in EXPLORER, spelt
        menu = self._body("explorerKeepHtml")
        assert "explorerFilters()" in menu and "one.label" in menu
        assert 'id="exploreKeep"' in menu
        # Offered only where there is an overlay to answer it, so a checkout's explorer
        # has no filter control at all rather than one that lights nothing.
        assert "if (!filters.length) return '';" in menu

    def test_the_panel_prints_the_votes_beside_the_stars(self) -> None:
        """`docs/reference/no-dark-patterns.md` test 7, on the one mark that needs it.

        A mean of one vote draws as large a dot as a mean of twenty, so the count goes
        beside the number. The line is appended under the map's own panel rather than
        written into it, which is what keeps `#/map` printing exactly what it printed.
        """
        from page_text import EXPLORER

        painted = self._body("explorerPaintPanel")
        assert "paintMapPanel();" in painted
        assert "MAP_PICK.kind !== 'node'" in painted
        assert "panel.appendChild(line)" in painted
        assert "line.textContent = said" in painted
        said = self._body("explorerPeopleWords")
        assert "record.votes === 1 ? 'one vote' : record.votes + ' votes'" in said
        assert "Nobody has scored this out of five yet." in said
        assert "Not played here yet." in said
        # The server rounded both numbers; the page prints them.
        assert "toFixed" not in EXPLORER and "fmt(" not in said

    def test_a_checkouts_payload_offers_no_overlay(self, tmp_path: Path) -> None:
        """`chess-variant-sandbox serve` has no store, so the payload's list is empty.

        It is the caller's constant rather than anything the library says about itself,
        which is what keeps one payload one answer for everybody.
        """
        from chess_variant_sandbox.web import graphing

        assert graphing.build([], tmp_path).as_dict()["overlays"] == []
        overlaid = graphing.build([], tmp_path, overlays=["people"])
        assert overlaid.as_dict()["overlays"] == ["people"]
        # A key added to or taken from the payload is a cache under a new shape.
        assert graphing.GRAPH_VERSION == "g15"


class TestWhatsNew:
    """`docs/changelog.md` as a page: *what's new*, newest first, on both servers.

    Two claims, and they are separate. The file is turned into **blocks and inline runs
    and never into markup** (`web/changelog.py`), so what the page receives is strings
    with labels; and the page builds an element per piece and sets its `textContent`, so
    the only way a string reaches the document is as text. Neither half needs an
    escaping step, which is the point -- `esc` is a call somebody can forget and a text
    node is not.

    The wiring is the map's: a route in the router, one call to one endpoint, and a link
    in the footer that both servers draw.
    """

    def test_the_file_becomes_days_of_entries_in_the_order_it_was_written(self) -> None:
        from chess_variant_sandbox.web import changelog

        days = changelog.parse(SAMPLE_CHANGELOG)
        assert [day["date"] for day in days] == ["2026-09-18", "2026-09-15"]
        # The date a reader reads, decided where the date is parsed rather than in the
        # browser, so the page prints a string and formats nothing.
        assert days[0]["words"] == "18 September 2026"
        assert [entry["heading"] for entry in days[0]["entries"]] == [
            "A map of the whole library",
            "Measuring a game from the command line",
        ]
        assert len(days[1]["entries"]) == 1

    def test_the_files_own_header_is_not_shown(self) -> None:
        """The lines above the first `---` are instructions to whoever writes an entry."""
        from chess_variant_sandbox.web import changelog

        said = " ".join(
            run["s"]
            for day in changelog.parse(SAMPLE_CHANGELOG)
            for entry in day["entries"]
            for block in entry["blocks"]
            for run in block.get("runs", [])
        )
        assert "Instructions to whoever writes an entry" not in said
        assert "Changelog" not in said

    def test_a_paragraph_is_emphasis_inline_code_and_text(self) -> None:
        """The three kinds of run the file uses, and the line breaks it is wrapped at
        are not three paragraphs."""
        from chess_variant_sandbox.web import changelog

        entry = changelog.parse(SAMPLE_CHANGELOG)[0]["entries"][1]
        (paragraph,) = entry["blocks"]
        assert [run["t"] for run in paragraph["runs"]] == ["code", "text", "em", "text"]
        assert paragraph["runs"][0]["s"] == "chess-variant-sandbox measure --seeds four"
        assert paragraph["runs"][2]["s"] == "waits on your word"
        # Wrapped across two lines in the file and one sentence on the page.
        assert "scores until the next deploy" in paragraph["runs"][1]["s"]

    def test_an_entry_marked_for_maintainers_is_skipped_on_the_site(self) -> None:
        """W8 (Q615, `docs/archive/plan-understanding.md` §3.3 C5): an entry whose first paragraph
        is `*For maintainers.*` is parsed as the maintainer's; the local application keeps
        it as written and `for_visitors` drops it, and a day left empty with it."""
        from chess_variant_sandbox.web import changelog

        text = ("---\n## 2026-09-18\n\n### For a visitor\n\nA map.\n\n"
                "### A command\n\n*For maintainers.*\n\n`measure --seeds four`.\n\n"
                "## 2026-09-15\n\n### Only a deploy\n\n*For maintainers.*\n\nDeployed.\n\n"
                "## 2026-09-14\n\n### Said later\n\nA map.\n\n*For maintainers.*\n")
        payload = {"days": changelog.parse(text), "note": ""}
        local = [[entry.get("maintainers", False) for entry in day["entries"]]
                 for day in payload["days"]]
        # Only the first paragraph marks: a line later in an entry is a sentence.
        assert local == [[False, True], [True], [False]]
        marked = payload["days"][0]["entries"][1]["blocks"]
        assert marked[0]["runs"] == [{"t": "em", "s": "For maintainers."}]
        site = changelog.for_visitors(payload)
        assert [day["date"] for day in site["days"]] == ["2026-09-18", "2026-09-14"]
        assert [entry["heading"] for entry in site["days"][0]["entries"]] == ["For a visitor"]
        only = "---\n## 2026-09-15\n\n### Only a deploy\n\n*For maintainers.*\n\nDeployed.\n"
        assert changelog.for_visitors({"days": changelog.parse(only), "note": ""}) == {
            "days": [], "note": "The changelog holds no entries yet."}

    def test_the_changelogs_maintainer_entries_are_marked(self) -> None:
        """The real file: the entries C5 names that only a maintainer would notice are
        marked, so the site does not serve their commands and paths."""
        from chess_variant_sandbox.web import changelog

        site = changelog.for_visitors(changelog.payload())
        shown = {entry["heading"] for day in site["days"] for entry in day["entries"]}
        for heading in ("Measuring a game on four seeds from the command line",
                        "Invite people from the Maintenance page",
                        "The groundwork for scores averaged over four runs"):
            assert heading not in shown, heading
        said = " ".join(run["s"] for day in site["days"] for entry in day["entries"]
                        for block in entry["blocks"] for run in block.get("runs", []))
        for phrase in ("measure --seeds four", "site invite", "You decided"):
            assert phrase not in said, phrase
        assert len(shown) > 50

    def test_the_visitors_entries_hold_no_maintainers_phrase(self) -> None:
        """Q681 (C5's leftovers): the seven visitor entries that kept one sentence of ours
        are said in a visitor's words: no path in the repository, no command, no deploy,
        no admin page, no *beat the top* and no *game that counts*."""
        import site_words

        from chess_variant_sandbox.web import changelog

        site = changelog.for_visitors(changelog.payload())
        entries = {entry["heading"]: entry for day in site["days"] for entry in day["entries"]}
        for heading in ("Every library score is the mean of four runs",
                        "A game that leaves the library leaves the site too",
                        "A low depth score is weaker evidence than a high one",
                        "The designer takes a draft all the way to a score",
                        "\"Show the stuck game\" shares show me's limit on the site",
                        "The project is now called Chess variant sandbox",
                        "A page for everybody who plays"):
            runs = [run for block in entries[heading]["blocks"]
                    for run in block.get("runs", [])]
            said = " ".join(run["s"] for run in runs)
            assert not [run for run in runs if run["t"] == "code"], heading
            for phrase in ("docs/", "decisions.md", "deploy", "admin", "you decide"):
                assert phrase not in said.lower(), (heading, phrase)
            assert not site_words.hits(said), (heading, site_words.hits(said))

    def test_no_visitors_entry_names_the_admin_panel(self) -> None:
        """Q728: the admin panel and its account are the maintainer's. A visitor entry
        never names them; one that only they would notice is marked for maintainers."""
        from chess_variant_sandbox.web import changelog

        site = changelog.for_visitors(changelog.payload())
        for day in site["days"]:
            for entry in day["entries"]:
                said = " ".join([entry["heading"]] + [
                    run["s"] for block in entry["blocks"] for run in block.get("runs", [])])
                for phrase in ("admin", "maintenance"):
                    assert phrase not in said.lower(), (entry["heading"], phrase)

    def test_a_line_it_does_not_understand_is_printed_as_itself(self) -> None:
        """A list item keeps its dash and a link its brackets.

        The safe direction. An unsupported construct reads as what it says rather than
        as whatever a parser guessed at, and widening the parser is a case in
        `web/changelog.py` rather than a special case in the page.
        """
        from chess_variant_sandbox.web import changelog

        text = "---\n## 2026-09-18\n\n### One\n\n- a list item\n\n#### deeper\n\n[a](b)\n"
        blocks = changelog.parse(text)[0]["entries"][0]["blocks"]
        said = [run["s"] for block in blocks for run in block.get("runs", [])]
        assert said == ["- a list item", "#### deeper", "[a](b)"]

    def test_nothing_in_the_file_can_become_markup(self) -> None:
        """An entry holding a script tag, and one holding a stray `<`.

        Neither is escaped anywhere, because neither is ever markup: the parser emits a
        labelled string and the page puts it in a text node.
        """
        from chess_variant_sandbox.web import changelog

        text = (
            "---\n## 2026-09-18\n\n"
            "### <script>alert(1)</script>\n\n"
            "A game with <script>alert(2)</script> in it, and `<b>` besides.\n\n"
            "### A board 8 < 9 squares wide\n\nRefused when a < b.\n"
        )
        days = changelog.parse(text)
        entries = days[0]["entries"]
        assert entries[0]["heading"] == "<script>alert(1)</script>"
        assert [run["t"] for run in entries[0]["blocks"][0]["runs"]] == [
            "text", "code", "text"
        ]
        assert "<script>alert(2)</script>" in entries[0]["blocks"][0]["runs"][0]["s"]
        assert entries[0]["blocks"][0]["runs"][1]["s"] == "<b>"
        assert "8 < 9" in entries[1]["heading"]

        drawn = _news_drawn({"days": days, "note": ""})
        # Every tag in the drawn tree is one this module made; the file named none of
        # them. A `<script>` in the file is a string inside an `h4` and a `p`.
        tags = _tags(drawn)
        assert tags == {"div", "section", "h3", "article", "h4", "p", "code", "#text"}
        assert "<script>alert(1)</script>" in _texts(drawn)
        assert "A game with <script>alert(2)</script> in it, and " in _texts(drawn)

    def test_a_missing_file_is_a_sentence_and_not_an_error(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.web import changelog

        payload = changelog.payload(tmp_path / "nothing.md")
        assert payload["days"] == []
        assert payload["note"] and payload["note"].endswith(".")
        # And the page shows that sentence rather than an empty page.
        drawn = _news_drawn(payload)
        assert _texts(drawn) == [payload["note"]]

    def test_an_empty_file_says_so(self, tmp_path: Path) -> None:
        empty = tmp_path / "changelog.md"
        empty.write_text("# Changelog\n\nThe note to the writer.\n\n---\n", encoding="utf-8")
        from chess_variant_sandbox.web import changelog

        payload = changelog.payload(empty)
        assert payload["days"] == [] and "no entries" in payload["note"]

    def test_the_repositorys_own_changelog_reads_cleanly(self) -> None:
        """The file this ships for. Every day is a date, every entry is headed, and no
        emphasis or backtick is left over as text."""
        from chess_variant_sandbox.web import changelog

        payload = changelog.payload()
        assert payload["note"] == "" and len(payload["days"]) >= 3
        for day in payload["days"]:
            assert day["date"] and day["words"] != day["date"]
            assert day["entries"]
            for entry in day["entries"]:
                assert entry["heading"] and entry["blocks"]
        text = "".join(
            run["s"]
            for day in payload["days"]
            for entry in day["entries"]
            for block in entry["blocks"]
            for run in block.get("runs", [])
            if run["t"] == "text"
        )
        assert "`" not in text and "*" not in text

    def test_the_file_the_page_reads_is_one_the_deploy_ships(self) -> None:
        """`deploy/bootstrap.sh` unpacks `git archive REF` into `/opt/chess-variant-sandbox`, which is
        the tracked files and nothing more, and the unit's working directory is that
        directory. So the page is empty on the box unless the file is tracked."""
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "docs/changelog.md"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True,
        )
        assert tracked.returncode == 0, tracked.stderr
        from chess_variant_sandbox.web import changelog

        assert str(changelog.CHANGELOG) in ("docs/changelog.md", "docs\\changelog.md")

    def test_the_local_application_serves_it(self, tmp_path: Path) -> None:
        app = App(RULESETS, tmp_path)
        payload = app.get("/api/changelog")
        assert payload["note"] == "" and payload["days"]

    def test_the_site_classifies_it_as_a_route_anybody_may_read(self) -> None:
        """The changelog holds nothing private, so it is `/api/graph`'s neighbour: served
        to a visitor, with no sign-in sentence on it."""
        from chess_variant_sandbox.site import routes

        found = routes.find("GET", "/api/changelog")
        assert found is not None and found in routes.ALLOWED
        assert found.visitor == ""

    def test_the_design_steps_route_names_the_field_its_handler_reads_back(self) -> None:
        """Q542: `back` is read by the handler, so the route's own description says so."""
        from chess_variant_sandbox.site import routes

        found = routes.find("POST", "/api/design/steps")
        assert found is not None
        assert "`back`" in found.note and "undo" in found.note

    def test_the_page_is_a_route_that_asks_for_it_once(self) -> None:
        from page_text import PAGE, WHATSNEW

        assert WHATSNEW in PAGE
        assert "parts[0] === 'whats-new'" in PAGE and "showWhatsNew()" in PAGE
        assert WHATSNEW.count("await api(") == 1
        assert "api('/api/changelog')" in WHATSNEW

    def test_the_page_reads_no_markdown_of_its_own(self) -> None:
        """What a line of the file means is decided in one place, and it is Python."""
        from page_text import WHATSNEW

        for parsing in ("split('\\n')", "'##'", "'###'", "replace(/", "match(/"):
            assert parsing not in WHATSNEW, parsing
        # And nothing here writes markup into the changelog's own elements.
        assert "innerHTML" not in WHATSNEW.split("function paintNews(")[1]

    def test_it_lands_and_draws_what_the_server_sent(self) -> None:
        from chess_variant_sandbox.web import changelog

        payload = changelog.payload()
        seen = _page_on(
            False, "#/whats-new", _game_payload("abc123def456", site=False),
            answers={"/api/changelog": payload}, landed='id="newsBody"',
        )
        assert seen["errors"] == []
        assert "GET /api/changelog" in seen["fetched"]
        assert "What&#39;s new" in seen["html"] or "What's new" in seen["html"]

    def test_both_servers_put_it_in_the_footer(self) -> None:
        """The one thing in the footer that needs no store, so it is on both.

        Credits and the privacy notice are the site's pages and are drawn only where
        they exist; the changelog is a file in the tree, so the local application has it
        too.
        """
        for accounts in ("true", "false"):
            drawn = _header_after(
                "() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve("
                "{ signedIn: false, handle: null, admin: false, accounts: "
                + accounts + " }) })"
            )
            # The games and the landing page first (Q718, Q723), on both servers, then the
            # changelog.
            # LP3: then *How the score works*.
            assert drawn["foot"][:7] == [{"text": "games", "href": "/game"}, " · ",
                                         {"text": "about", "href": "/"}, " · ",
                                         {"text": "the score", "href": "#/score"}, " · ",
                                         {"text": "what's new", "href": "#/whats-new"}]

    def test_the_sites_own_pages_carry_the_same_link(self) -> None:
        """A person reading the privacy notice is on a page the application never drew,
        so its footer is written out in `site/pages.py` and has to say the same three
        things."""
        from chess_variant_sandbox.site import pages

        # Since G11 the footer names the page the server draws, `/whats-new`, which a
        # crawler can read; the application's own footer keeps the fragment it alone serves.
        assert '<a href="/whats-new">what\'s new</a>' in pages.privacy()
        assert '<a href="/whats-new">what\'s new</a>' in pages.credits()


def _tags(tree: dict) -> set[str]:
    """Every element name in a drawn tree."""
    found = {tree["tag"]}
    for child in tree.get("children", []):
        found |= _tags(child)
    return found


def _settle(*roots: Path) -> None:
    """Backdate every file under these roots, out of the listing cache's settle window.

    The cache refuses to keep a listing whose newest file was written moments ago,
    because a filesystem whose mtime is coarser than its clock cannot yet prove that
    file has not moved again since. A test that wants the cached path has to look like a
    corpus that has been sitting there, which is what every real one looks like.
    """
    old = time.time_ns() - 60_000_000_000
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file():
                os.utime(path, ns=(old, old))


def _one_run(index: float, sd: float, viable: bool = True) -> list[dict]:
    """The `runs` list of a measurement taken on one seed.

    A fixture without one is a file in the single-run form, which never yields a number
    -- so every fixture standing in for a real measurement carries its run.
    """
    return _runs_at((0,), index, sd, viable)


def _runs_at(
    seeds: tuple[int, ...], index: float, sd: float, viable: bool = True
) -> list[dict]:
    """The `runs` list of a measurement taken on these seeds, each scoring the same.

    Four of them is what a fixture standing in for a *promotable* game needs:
    `corpus.promotion_blocker` asks for `LIBRARY_SEEDS` whatever the library currently
    expects, because promotion is a claim about the corpus.
    """
    return [{"seed": seed, "index": index, "sd": sd, "viable": viable,
             "violations": [] if viable else ["quick_kill_rate=0.9 above maximum 0.5"],
             "unreadable": [], "origin": "played"}
            for seed in seeds]


class TestTheListingIsCachedOnWhatItRead:
    """`discover` is cached, and the key is every file behind the answer.

    The listing is the most expensive thing this application does -- several thousand
    file opens under `runs/` -- and it runs on every request on purpose, because a
    listing that cannot see a measurement finish is the bug in
    `archive/bugs-resolved.md`. So the cache has to be invisible, and what makes it so is
    the key: the files themselves, never the directories holding them. A re-score
    rewrites `canonical.json` inside a run directory whose own mtime does not move, so a
    key made of directory mtimes would have put that bug back behind the fix.
    """

    @staticmethod
    def _corpus(tmp_path: Path) -> tuple[Path, Path, str]:
        spec = load_spec(RULESETS / "gardner.json")
        spec["name"] = "the one game"
        rulesets = tmp_path / "rulesets"
        rulesets.mkdir()
        (rulesets / "x.json").write_text(json.dumps(spec), encoding="utf-8")
        runs = tmp_path / "runs"
        runs.mkdir()
        return rulesets, runs, spec_hash(spec)

    @staticmethod
    def _score(rulesets: Path, runs: Path, digest: str, index: float) -> Path:
        """What a measurement of this game leaves behind, current as of now."""
        basis = corpus.current_basis(digest, spec=load_spec(rulesets / "x.json"))
        out = corpus.measurement_dir(digest, runs)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "canonical.json"
        path.write_text(json.dumps({
            "hash": digest, "name": "the one game", "index": index, "sd": 0.02,
            "viable": True, "violations": [], "weights_version": basis.weights_version,
            "families": {}, "standings": [], "games": 4, "basis": basis.as_dict(),
            "seeds": [0], "totals_used": 1, "runs": _one_run(index, 0.02),
        }), encoding="utf-8")
        return path

    def test_a_rescore_rewriting_one_file_in_place_is_seen_by_the_next_listing(
        self, tmp_path
    ) -> None:
        """The trap, named and held down.

        A re-score writes a new `canonical.json` into a run directory that already
        exists. The file's mtime moves; the directory's does not, and the second
        assertion here is that measurement rather than a claim about it. So the listing
        has to be keyed on the file.
        """
        rulesets, runs, digest = self._corpus(tmp_path)
        self._score(rulesets, runs, digest, 0.5)
        _settle(rulesets, runs)
        assert library.discover(rulesets, runs)[0].summary()["index"] == 0.5

        directory = corpus.measurement_dir(digest, runs)
        unmoved = directory.stat().st_mtime_ns
        self._score(rulesets, runs, digest, 0.61)
        assert directory.stat().st_mtime_ns == unmoved

        assert library.discover(rulesets, runs)[0].summary()["index"] == 0.61

    def test_an_unmoved_corpus_is_not_read_a_second_time(self, tmp_path) -> None:
        """The cache, doing its job, observed where it is observable.

        Two listings over a tree nothing has touched hand back the same `Game` objects,
        which they cannot do unless the second one read nothing. The list around them is
        a copy, so a caller that sorts what it was given does not sort what the next
        caller gets.
        """
        rulesets, runs, digest = self._corpus(tmp_path)
        self._score(rulesets, runs, digest, 0.5)
        _settle(rulesets, runs)

        first = library.discover(rulesets, runs)
        second = library.discover(rulesets, runs)
        assert second[0] is first[0]
        assert second is not first

    def test_the_working_under_a_score_is_seen_when_only_it_is_rewritten(
        self, tmp_path
    ) -> None:
        """`chess-variant-sandbox score <run>` writes the index file and nothing else.

        The headline lives in `canonical.json` and the components under it in
        `index-<weights>.json` beside it, so a score run from a terminal against a live
        server moves the second file alone. The key stamps it for that reason, and only
        where a measurement sits -- a design run's hundreds are read by nothing here.
        """
        rulesets, runs, digest = self._corpus(tmp_path)
        canonical = self._score(rulesets, runs, digest, 0.5)
        version = json.loads(canonical.read_text(encoding="utf-8"))["weights_version"]
        working = canonical.parent / f"index-{version}.json"
        working.write_text(json.dumps({"scores": {"depth": 0.1}}), encoding="utf-8")
        _settle(rulesets, runs)
        assert library.discover(rulesets, runs)[0].summary()["components"] == {"depth": 0.1}

        working.write_text(json.dumps({"scores": {"depth": 0.9}}), encoding="utf-8")
        assert library.discover(rulesets, runs)[0].summary()["components"] == {"depth": 0.9}

    def test_a_ruleset_written_while_the_process_runs_appears(self, tmp_path) -> None:
        """The other half of the freshness promise: a design run writes specs in here."""
        rulesets, runs, _ = self._corpus(tmp_path)
        _settle(rulesets, runs)
        assert len(library.discover(rulesets, runs)) == 1

        other = load_spec(RULESETS / "almost.json")
        other["name"] = "a second game"
        (rulesets / "y.json").write_text(json.dumps(other), encoding="utf-8")
        assert len(library.discover(rulesets, runs)) == 2

    def test_a_name_edited_in_place_is_seen(self, tmp_path) -> None:
        """A spec file rewritten under its own name, which is a thing that happens.

        `rulesets/` filenames are human names, not hashes; the key is the mechanics hash,
        which `name` does not move, so editing `name` in place changes neither the key nor
        the file's location -- only its content. The same reason the key stamps files
        rather than the directories they sit in.
        """
        rulesets, runs, _ = self._corpus(tmp_path)
        _settle(rulesets, runs)
        assert library.discover(rulesets, runs)[0].name == "the one game"

        spec = load_spec(rulesets / "x.json")
        spec["name"] = "the same game, renamed"
        (rulesets / "x.json").write_text(json.dumps(spec), encoding="utf-8")
        assert library.discover(rulesets, runs)[0].name == "the same game, renamed"


class TestWarmingPaysForTheFirstListing:
    """`warm()` also runs the smoke gate and the Fairy-Stockfish translation, off the
    request path, so a person's browser is not the first thing to pay for them (Q418).
    The cost itself is Q415's, measured in `docs/reference/fast-suite-review.md`:
    `_playability` and `_fairy` (`web/library.py`) are memoised per spec's content, and
    before Q418 `warm()` only read the corpus (`App.refresh`), never touched either.
    """

    @staticmethod
    def _corpus(root: Path, sources: tuple[str, ...]) -> tuple[Path, Path]:
        """A handful of real rulesets under a name nothing has used before.

        Real specs, so the gate has real games to play and the translator real
        mechanics to carry across -- and a random name on each, so the content-keyed
        memo in `web/library.py` has never seen this exact spec, whatever this test's
        process has already listed. `name` moving does not move a spec's mechanics hash
        (`CLAUDE.md`), but it does move the JSON blob `_smoke_cached`/`_fairy_cached`
        key on, which is the point: two calls of this helper are two corpora neither
        cache has met, so timing one never warms the other by accident.
        """
        rulesets = root / "rulesets"
        rulesets.mkdir(parents=True)
        runs = root / "runs"
        runs.mkdir()
        nonce = uuid.uuid4().hex
        for index, source in enumerate(sources):
            spec = load_spec(RULESETS / source)
            spec["name"] = f"{spec['name']} ({nonce}-{index})"
            (rulesets / f"{index}.json").write_text(json.dumps(spec), encoding="utf-8")
        _settle(rulesets, runs)
        return rulesets, runs

    def test_a_listing_after_warm_finishes_is_much_faster_than_a_cold_one(
        self, tmp_path: Path
    ) -> None:
        """The regression itself. Two separate, never-before-seen corpora rather than one
        cache cleared in place between a cold and a warm call: clearing `library`'s
        process-wide memo would also cost the next test that lists the real corpus --
        exactly the flake Q415 fixed by warming it once per process.
        """
        sources = ("chess.json", "gardner.json", "almost.json")
        cold_rulesets, cold_runs = self._corpus(tmp_path / "cold", sources)
        cold_app = App(cold_rulesets, cold_runs, archive_dir=tmp_path / "cold-archive")
        started = time.monotonic()
        cold_app.get("/api/library")
        cold_elapsed = time.monotonic() - started

        warm_rulesets, warm_runs = self._corpus(tmp_path / "warm", sources)
        warm_app = App(warm_rulesets, warm_runs, archive_dir=tmp_path / "warm-archive")
        thread = warm(warm_app)
        thread.join(30.0)
        assert not thread.is_alive(), "warm() did not finish inside the test's own budget"

        started = time.monotonic()
        warm_app.get("/api/library")
        warm_elapsed = time.monotonic() - started

        assert warm_elapsed < cold_elapsed / 3, (cold_elapsed, warm_elapsed)

    def test_a_request_during_warming_waits_and_shares_the_work(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A listing that lands while `warm()` is still running gets a correct answer,
        and the smoke gate is not paid twice for the same spec.

        The gate is slowed down and counted rather than timed: waiting on a clock to
        land a request mid-warm would be exactly the flaky race Q415's own bug was, so
        this pins the interleaving with an `Event` instead -- the request is not sent
        until `warm()` is provably inside the gate for the first spec, and what proves
        the work was shared rather than duplicated is that the gate ran once per spec,
        not twice.
        """
        from chess_variant_sandbox.loop import gates

        nonce = uuid.uuid4().hex
        rulesets = tmp_path / "rulesets"
        rulesets.mkdir()
        runs = tmp_path / "runs"
        runs.mkdir()
        names = []
        for index, source in enumerate(("gardner.json", "almost.json")):
            spec = load_spec(RULESETS / source)
            spec["name"] = f"{spec['name']} ({nonce}-{index})"
            names.append(spec["name"])
            (rulesets / f"{index}.json").write_text(json.dumps(spec), encoding="utf-8")
        _settle(rulesets, runs)

        app = App(rulesets, runs, archive_dir=tmp_path / "archive")

        calls: list[str] = []
        started_first = threading.Event()
        real_smoke_test = gates.smoke_test

        def slow_smoke_test(spec, *args, **kwargs):
            calls.append(spec["name"])
            started_first.set()
            time.sleep(0.15)
            return real_smoke_test(spec, *args, **kwargs)

        monkeypatch.setattr(gates, "smoke_test", slow_smoke_test)

        thread = warm(app)
        assert started_first.wait(5.0), "warm() never reached the smoke gate"

        during = app.get("/api/library")
        thread.join(10.0)
        assert not thread.is_alive(), "warm() did not finish inside the test's own budget"

        # One call per spec: `warm()` and the request racing it shared the gate rather
        # than each paying for it.
        assert sorted(calls) == sorted(names), calls
        assert {game["name"] for game in during["games"]} == set(names)
        for game in during["games"]:
            assert game["playable"] is True

        # And the answer a request mid-warm gets is the real one, not a partial or a
        # placeholder: a listing taken once every cache is settled agrees with it.
        assert during == app.get("/api/library")


class TestTheIndexBand:
    """Where a game's score sits against the rest, as three counts and never a place.

    `docs/archive/plan-meta-game.md` §2.6 T8. The interval on a score is wider than most of the
    gaps between neighbours, so a rank number would be noise dressed as a ranking. The
    band names the games this one is clearly above, the ones clearly above it, and the
    ones the measurement cannot separate it from -- and says nothing at all where the
    measurement is stale, which is the corpus rule and not a page's choice.
    """

    # Four games with intervals chosen so every pair is unambiguous. A is clear of
    # everything, D is under everything, and B and C overlap each other and nothing else.
    SCORES = {
        "a": (0.90, 0.88, 0.92),
        "b": (0.70, 0.68, 0.72),
        "c": (0.69, 0.67, 0.71),
        "d": (0.30, 0.28, 0.32),
    }
    SPECS = {"a": "gardner.json", "b": "almost.json", "c": "losalamos.json",
             "d": "dobutsu.json"}

    @classmethod
    def _corpus(cls, tmp_path: Path, stale: str = "") -> tuple[Path, Path, dict[str, str]]:
        """Four scored games, optionally with one of them measured under other weights.

        Real specs copied out of `rulesets/`, so the hashes are real and distinct and
        `corpus.current_basis` has something to compile.
        """
        rulesets = tmp_path / "rulesets"
        rulesets.mkdir()
        runs = tmp_path / "runs"
        runs.mkdir()
        digests: dict[str, str] = {}
        for key, source in cls.SPECS.items():
            spec = load_spec(RULESETS / source)
            spec["name"] = f"game {key}"
            path = rulesets / f"{key}.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            digests[key] = spec_hash(spec)
            total, low, high = cls.SCORES[key]
            cls._score(spec, digests[key], runs, total, low, high, stale=key == stale)
        return rulesets, runs, digests

    @staticmethod
    def _score(
        spec: dict, digest: str, runs: Path, total: float,
        low: float, high: float, stale: bool = False,
    ) -> None:
        """What one measurement leaves behind: the headline, and the interval beside it.

        The interval lives in `index-<weights>.json` and never in `canonical.json`,
        which is why the band cannot be computed off the store's headline alone.
        """
        basis = corpus.current_basis(digest, spec=spec)
        stamp = basis.as_dict()
        if stale:
            stamp["weights_version"] = "v0"
        out = corpus.measurement_dir(digest, runs)
        out.mkdir(parents=True, exist_ok=True)
        (out / "canonical.json").write_text(json.dumps({
            "hash": digest, "name": spec["name"], "index": total, "sd": 0.01,
            "viable": True, "violations": [], "weights_version": basis.weights_version,
            "families": {}, "standings": [], "games": 4, "basis": stamp,
            "seeds": [0], "totals_used": 1, "runs": _one_run(total, 0.01),
        }), encoding="utf-8")
        (out / f"index-{basis.weights_version}.json").write_text(json.dumps({
            "total": total, "scores": {}, "families": {},
            "interval": {"resamples": 200, "mean": total, "sd": 0.01,
                         "ci_low": low, "ci_high": high, "deterministic_pairings": 0.0},
        }), encoding="utf-8")

    def _bands(self, rulesets: Path, runs: Path, digests: dict[str, str]) -> dict:
        found = library.bands(library.discover(rulesets, runs))
        return {key: found.get(digest) for key, digest in digests.items()}

    def test_the_three_counts_are_pairwise_over_the_measured_library(self, tmp_path) -> None:
        rulesets, runs, digests = self._corpus(tmp_path)
        bands = self._bands(rulesets, runs, digests)

        # A is clear of all three; D is under all three; B and C overlap each other and
        # are apart from the other two.
        assert bands["a"] == {"clears": 3, "clearedBy": 0, "close": 0, "of": 3}
        assert bands["b"] == {"clears": 1, "clearedBy": 1, "close": 1, "of": 3}
        assert bands["c"] == {"clears": 1, "clearedBy": 1, "close": 1, "of": 3}
        assert bands["d"] == {"clears": 0, "clearedBy": 3, "close": 0, "of": 3}

    def test_the_counts_are_the_separation_rule_and_not_a_comparison_of_headlines(
        self, tmp_path
    ) -> None:
        """B outscores C by 0.01 and the band refuses to say so.

        The whole point of the band. A rank would put B above C on a difference a third
        the width of either interval; `IndexResult.separated_from` puts them in the same
        group, and it is the one separation rule in the project.
        """
        rulesets, runs, digests = self._corpus(tmp_path)
        bands = self._bands(rulesets, runs, digests)
        assert bands["b"]["close"] == 1 and bands["c"]["close"] == 1

    def test_a_stale_measurement_has_no_band_and_is_in_nobody_elses(self, tmp_path) -> None:
        """Absent, never a zero and never a last place.

        And it leaves the comparison too: the other three are compared against each
        other and not against a provisional number (EV2), which a reader is shown marked
        and which never enters a band.
        """
        rulesets, runs, digests = self._corpus(tmp_path, stale="d")
        found = library.discover(rulesets, runs)
        stale = next(game for game in found if game.hash == digests["d"])
        assert stale.measurement is not None and stale.stale
        assert stale.score is not None and stale.score["status"] == "provisional"
        assert stale.current_score is None

        bands = self._bands(rulesets, runs, digests)
        assert bands["d"] is None
        assert bands["a"] == {"clears": 2, "clearedBy": 0, "close": 0, "of": 2}
        assert bands["b"]["of"] == 2

    def test_an_unmeasured_game_has_no_band_either(self, tmp_path) -> None:
        rulesets, runs, digests = self._corpus(tmp_path)
        spec = load_spec(RULESETS / "dull.json")
        spec["name"] = "never measured"
        (rulesets / "e.json").write_text(json.dumps(spec), encoding="utf-8")
        found = library.bands(library.discover(rulesets, runs))
        assert spec_hash(spec) not in found
        assert found[digests["a"]]["of"] == 3

    def test_the_pass_is_paid_once_per_measurement_rather_than_once_per_listing(
        self, tmp_path
    ) -> None:
        """71 games is 71 x 70 comparisons, so the cheap thing has to stay correct.

        The memo is keyed on every figure a comparison reads, so an unchanged corpus is
        a hit and a re-scored game is not -- which is the only way a cached ranking can
        be trusted when one measurement moves.
        """
        rulesets, runs, digests = self._corpus(tmp_path)
        games = library.discover(rulesets, runs)
        library._bands.cache_clear()
        assert library.bands(games)[digests["a"]]["clears"] == 3
        before = library._bands.cache_info().misses
        library.bands(games)
        assert library._bands.cache_info().misses == before, "an unmoved corpus recomputed"

        # D re-scored above everything. The listing is re-read, the key moves with it,
        # and the answer is not the one in the memo.
        self._score(
            load_spec(rulesets / "d.json"), digests["d"], runs, 0.99, 0.97, 0.995
        )
        moved = library.bands(library.discover(rulesets, runs))
        assert moved[digests["d"]] == {"clears": 3, "clearedBy": 0, "close": 0, "of": 3}
        assert moved[digests["a"]]["clearedBy"] == 1

    # Slow (Q749): four cold cards, renamed so no memo holds them, 3.2 s on one core and
    # 3.0 s in the gate; the band's own arithmetic is the tests around it.
    @pytest.mark.slow
    def test_the_card_and_the_game_page_say_the_same_three_counts(self, tmp_path) -> None:
        """One function, two callers, and the server is both of them.

        A page that worked the counts out for itself could get a different answer on the
        card from the one on the page, because a card is drawn over a filtered listing
        and a page is drawn over one game.
        """
        rulesets, runs, digests = self._corpus(tmp_path)
        app = App(rulesets, runs, archive_dir=tmp_path / "archive")
        cards = {card["hash"]: card["band"] for card in app.get("/api/library")["games"]}
        assert cards[digests["b"]] == {"clears": 1, "clearedBy": 1, "close": 1, "of": 3}
        for key, digest in digests.items():
            assert app.get(f"/api/game/{digest}")["band"] == cards[digest], key

    def test_every_count_carries_its_own_sentence(self, tmp_path) -> None:
        """Test 7 of the plan's seven: no number on the page without its rule beside it."""
        from page_text import PAGE

        for key in ("bandClears", "bandClearedBy", "bandClose"):
            assert f"  {key}: [" in PAGE, key
            assert f"info('{key}')" in PAGE, key
        # The card and the score panel reach the one drawing function, so neither can
        # word the band differently from the other.
        assert "bandHtml(game.band)" in PAGE and "bandHtml(band)" in PAGE

    def test_a_move_sentence_keeps_its_rule_behind_an_i(self) -> None:
        """UX A7: the piece list says the move, and the vectors are one tap away.

        The page never writes either of them -- both arrive from `spec/describe.py`, so
        the sentence and the rule it was written from cannot drift apart.
        """
        from page_text import PAGE

        assert "  moveRule: [" in PAGE
        # On the local application; the site's (i) carries no raw rule (W1, A12).
        assert "info('moveRule', ACCOUNTS ? '' : rules[at])" in PAGE
        # Drawn only where the piece is being read, not where it is a click target.
        assert "index === undefined && rules[at] && rules[at] !== line" in PAGE


class TestWhatEveryNumberRestsOn:
    """`docs/archive/plan-four-seeds.md` §2.4, §4, §5 and §7: how many runs a number came from.

    A run is one tournament from one seed. A game the library keeps is measured from four
    seeds and scored as the mean of the four; a game somebody has just measured is one
    run. The two are the same shape and not the same claim -- a one-run interval is about
    twice as wide -- so the count travels with every number, and a game whose four runs
    disagreed about it says how they split.

    The payload is checked against a measurement `corpus._store` wrote, never a
    hand-written file. The drawing is checked by `in` over the page source, which is this
    package's idiom: the page is JavaScript inside Python strings and the suite runs no
    browser.
    """

    @staticmethod
    def _reading(seed: int, total: float | None, viable: bool = True,
                 violations: tuple[str, ...] = ()):
        """One run as the combiner reads it, scored or withheld."""
        from chess_variant_sandbox.metrics.seeds import RunReading

        scored = total is not None
        return RunReading(
            seed=seed,
            scored=scored,
            total=total,
            sd=0.02 if scored else None,
            viable=viable,
            violations=violations,
            scores={"depth": 0.5},
            raw={"depth": 1.0},
            contributions={"depth": 0.5} if scored else {},
            families={"depth": 0.5},
            interval=None if not scored else {
                "mean": total, "sd": 0.02, "ci_low": total - 0.04,
                "ci_high": total + 0.04, "deterministic_pairings": 0.0,
                "stratum_correction": 1.0, "resamples": 200,
            },
            standings=({"name": "random", "rating": -300.0, "games": 64},),
            games=256,
        )

    def _measured(self, tmp_path: Path, readings) -> dict:
        """One committed game measured over those runs, as a card."""
        from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS

        spec = load_spec(RULESETS / "gardner.json")
        spec["name"] = "the measured one"
        rulesets, runs = tmp_path / "rulesets", tmp_path / "runs"
        rulesets.mkdir(exist_ok=True)
        runs.mkdir(exist_ok=True)
        path = rulesets / "x.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        digest = spec_hash(spec)
        corpus._store(
            corpus.Candidate(
                hash=digest, name=spec["name"], spec=spec, path=path, stage="committed"
            ),
            runs, list(readings), DEFAULT_WEIGHTS, lambda line: None,
        )
        found = library.discover(rulesets, runs)[0]
        assert not found.stale, found.staleness
        return found.summary()

    def _four(self, tmp_path: Path, accepting: int) -> dict:
        """Four runs of which this many accept the game, at four different totals."""
        totals = [0.70, 0.72, 0.74, 0.76]
        return self._measured(tmp_path, [
            self._reading(
                seed, total, viable=at < accepting,
                violations=() if at < accepting else ("almost every game is drawn",),
            )
            for at, (seed, total) in enumerate(
                zip([0, 1000, 2000, 3000], totals, strict=True)
            )
        ])

    def test_a_measurement_taken_once_says_it_is_one_run(
        self, tmp_path, monkeypatch
    ) -> None:
        # A committed game owes four runs, so the reader is pointed at one here. What is
        # under test is how a one-run measurement labels itself -- which is what a
        # visitor's own design gets -- and not what the library expects of it.
        monkeypatch.setattr(corpus, "EXPECTED_LIBRARY_SEEDS", corpus.ONE_SEED)
        card = self._measured(tmp_path, [self._reading(0, 0.7)])
        assert card["seeds"] == [0]
        assert [run["seed"] for run in card["runs"]] == [0]
        assert card["totalsUsed"] == 1
        # Nothing to say about a split: there is no vote in one run, and every rule
        # returns that run's own verdict.
        assert card["runVerdict"]["split"] == ""
        assert card["index"] == pytest.approx(0.7)

    def test_four_agreeing_runs_carry_four_totals_and_no_split(self, tmp_path) -> None:
        card = self._four(tmp_path, accepting=4)
        assert card["seeds"] == [0, 1000, 2000, 3000]
        assert [run["index"] for run in card["runs"]] == [0.70, 0.72, 0.74, 0.76]
        assert card["totalsUsed"] == 4
        assert card["viable"] is True
        assert card["runVerdict"]["accept"] == 4 and card["runVerdict"]["split"] == ""
        # The mean of the four totals, never one score over their games pooled.
        assert card["index"] == pytest.approx(0.73)

    def test_three_of_four_accepting_keeps_it_in_and_says_which_refused(
        self, tmp_path
    ) -> None:
        """The case that reads as agreement unless the page says otherwise.

        A viable game carries no violations at all, so the split is the only thing on the
        page that can report the run that refused it -- and which run is the first
        question a reader of a three-to-one has.
        """
        card = self._four(tmp_path, accepting=3)
        assert card["viable"] is True
        assert card["runVerdict"]["split"] == (
            "Three of its four runs accept it and one refuses it, so it stays in."
        )
        refusing = [run for run in card["runs"] if not run["viable"]]
        assert [run["seed"] for run in refusing] == [3000]
        assert refusing[0]["violations"] == ["almost every game is drawn"]

    def test_a_tie_refuses_and_the_sentence_says_that_is_the_rule(self, tmp_path) -> None:
        """The user's answer, 2026-09-18: an explicit majority accepts, a tie refuses.

        The rule the card names carries the seat's answer of 2026-09-19 beside it, and
        the basis stamps that name -- so a file made under the older rule reads stale
        rather than being taken for one of these.
        """
        card = self._four(tmp_path, accepting=2)
        assert card["viable"] is False
        assert card["runVerdict"]["rule"] == "vote-tie-refuses+pooled-seat"
        assert card["runVerdict"]["split"] == (
            "Two of its four runs accept it and two refuse it. A tie refuses, so it is"
            " ruled out."
        )

    def test_one_of_four_accepting_is_ruled_out_and_names_every_refusal(
        self, tmp_path
    ) -> None:
        card = self._four(tmp_path, accepting=1)
        assert card["viable"] is False
        assert card["runVerdict"]["split"].startswith(
            "One of its four runs accepts it and three refuse it"
        )
        # A ruled-out game carries the split and each refusing run's own words, by seed.
        score = library.discover(
            tmp_path / "rulesets", tmp_path / "runs"
        )[0].score
        assert score["violations"][0] == card["runVerdict"]["split"]
        assert "run 1000: almost every game is drawn" in score["violations"]

    def test_too_few_readable_runs_leave_no_number_rather_than_a_low_one(
        self, tmp_path
    ) -> None:
        """§5. A mean needs three of four, and a placeholder is never averaged."""
        card = self._measured(tmp_path, [
            self._reading(0, 0.70),
            self._reading(1000, 0.72),
            self._reading(2000, None, viable=False, violations=("no ladder",)),
            self._reading(3000, None, viable=False, violations=("no ladder",)),
        ])
        assert card["index"] is None
        assert card["totalsUsed"] == 2
        # Measured all the same, and the count says so -- "nobody measured this" and
        # "measured, and it could not be scored" are different news.
        assert card["seeds"] == [0, 1000, 2000, 3000]

    def test_the_card_and_the_game_page_say_the_same_run_count(self, tmp_path) -> None:
        """One join, two callers. `detail` spreads `summary`, so it cannot disagree."""
        self._four(tmp_path, accepting=3)
        app = App(tmp_path / "rulesets", tmp_path / "runs",
                  archive_dir=tmp_path / "archive")
        card = app.get("/api/library")["games"][0]
        page = app.get("/api/game/" + card["hash"])
        for key in ("seeds", "runs", "runVerdict", "totalsUsed"):
            assert page[key] == card[key], key
        # The measurement's own verdict never lands in the key the model's written
        # opinion holds, which is what `runVerdict` is named for.
        assert page["verdict"] is None

    def test_the_ladder_says_how_many_runs_it_was_fitted_over(self, tmp_path) -> None:
        """A four-run ladder is the per-agent mean of four fits, not one fit."""
        self._four(tmp_path, accepting=4)
        game = library.discover(tmp_path / "rulesets", tmp_path / "runs")[0]
        assert game.ratings["seeds"] == [0, 1000, 2000, 3000]
        assert game.ratings["games"] == 1024

    def test_no_number_is_drawn_without_the_count_beside_it(self) -> None:
        """Every place on the page that prints a score, and what says what it rests on.

        A list rather than a scan, because the failure this guards against is a *new*
        place printing a number -- and the repair is to add the place here with the call
        that labels it, so the two are decided together.
        """
        from page_text import PAGE

        for where, said in (
            # the pill on a card
            ("fmt(game.index)", "runsShort(game.seeds)"),
            # the headline on a game's page, and the line under it
            ("scoreHtml(game.score, game.scoreGuide, game.band, game.trust)",
             "runsHtml(score)"),
            # a component's number, while the list is ordered by it
            ("value.toFixed(2)", "runsPhrase(game.seeds)"),
            # the game the draft in the editor came from
            ("parent.index.toFixed(2)", "runsPhrase(parent.seeds)"),
            # each game a sweep has finished, off the item's own seeds (Q339)
            ("fmt(i.index)", "runsShort(i.seeds)"),
            # the measurement rung and each game in a draft's lineage (Q339)
            ("Number(m.index).toFixed(2)", "runsPhrase(m.seeds)"),
            # both ends of a gap, on a game's page and on the map (Q339)
            ("gap.otherIndex.toFixed(2)", "runsShort(gap.otherSeeds)"),
            ("gap.index.toFixed(2)", "runsShort(gap.seeds)"),
            # a design run's own rough number on a card (Q416)
            ("fmt(game.designIndex)", "runsShort(game.designSeeds)"),
        ):
            assert where in PAGE, where
            assert said in PAGE, said
        # The sweep used to borrow its count from the survey's row for the same game,
        # which a game the survey had no row for printed without.
        assert "ranAs(" not in PAGE

    def test_the_split_reaches_the_card_and_the_page_through_one_function(self) -> None:
        """Two drawings of one sentence would be two sentences, eventually."""
        from page_text import PAGE

        assert "function splitSentences(verdict, runs)" in PAGE
        assert PAGE.count("splitSentences(") == 5  # the one definition and four callers
        assert "splitSentences(game.runVerdict, game.runs)" in PAGE
        assert "splitSentences(score.verdict, score.runs)" in PAGE
        # The fourth is the map's node panel, over the same fields a card carries.
        assert "splitSentences(score.runVerdict, score.runs)" in PAGE
        # The page never writes the sentence. Both halves of it -- the split and each
        # refusing run's own words -- arrive from `metrics/seeds.py`, so the words a
        # reader sees are the ones the command line prints.
        for phrase in ("of its four runs", "A tie refuses", "so it stays in"):
            assert phrase not in PAGE, phrase

    def test_the_score_gloss_quotes_no_score(self) -> None:
        """§7. It quoted 0.84 for chess and 0.34 for the dull control, under weights
        nobody runs now, and the dull control has since stopped having a number at all."""
        from page_text import CORE

        entry = CORE.split("  index: [", 1)[1].split("],", 1)[0]
        assert not re.search(r"\d\.\d", entry), entry
        # And it says which of the two claims the +/- beside a number is making.
        assert "mean of four" in entry and "one run" in entry


class TestProvenance:
    def test_an_evolved_game_does_not_inherit_its_parent_author(self, tmp_path) -> None:
        """A mechanical edit of gardner is not Martin Gardner's game.

        The operator deep-copies the parent, and the parent's provenance block names a
        human author -- so without stripping it, every game a design run invented arrived
        in the library claiming to be by whoever wrote the ruleset it was edited from.
        A fabricated attribution travels with the file forever; no provenance at all is
        strictly better.
        """
        import random

        from chess_variant_sandbox.loop.operators import mutate

        parent = load_spec(RULESETS / "gardner.json")
        assert parent["provenance"]["author"] == "Martin Gardner"

        child = mutate(parent, random.Random(0))
        assert child is not None
        assert "provenance" not in child.spec
        # And the parent is untouched, as always.
        assert parent["provenance"]["author"] == "Martin Gardner"

    def test_a_design_runs_score_is_shown_but_never_as_the_corpus_score(self, tmp_path) -> None:
        """A run's own number is visible, in its own key, and does not become `index`.

        It used to. The library globbed every `index*.json` under `runs/` and put
        whatever it found in `index`, which is the field a person reads as "how does
        this compare with chess". A design run scores on four agents at three games a
        pairing because a generation has to fit in the time it has; the corpus roster is
        eight agents at four. Those are two quantities, and the whole project turns on
        not mixing them -- so the run's number travels as `designIndex`, attributed to
        the run that produced it, and `index` stays empty until somebody measures it.
        """
        from chess_variant_sandbox.loop.archive import Archive, Entry

        spec = load_spec(RULESETS / "gardner.json")
        digest = spec_hash(spec)
        runs = tmp_path / "runs"
        archive = Archive(runs / "design")
        archive.record(
            Entry(hash=digest, name="a candidate", generation=1, parent=None,
                  status="evaluated", index=0.61, viable=True, seeds=[0]),
            spec=spec,
        )

        found = library.discover(tmp_path / "rulesets", runs)
        card = next(g for g in found if g.hash == digest).summary()
        assert card["designIndex"] == 0.61
        assert card["designRun"] == "design"
        # And the runs it rests on, which the card says beside it (Q416).
        assert card["designSeeds"] == [0]
        assert card["index"] is None
        assert card["staleness"] is None

    def test_a_design_entry_written_before_its_seeds_were_kept_carries_none(
        self, tmp_path
    ) -> None:
        """No count is invented for it: the key is empty and the card shows no number."""
        from chess_variant_sandbox.loop.archive import Archive, Entry

        spec = load_spec(RULESETS / "gardner.json")
        digest = spec_hash(spec)
        Archive(tmp_path / "runs" / "design").record(
            Entry(hash=digest, name="a candidate", generation=1, parent=None,
                  status="evaluated", index=0.61, viable=True),
            spec=spec,
        )
        found = library.discover(tmp_path / "rulesets", tmp_path / "runs")
        card = next(g for g in found if g.hash == digest).summary()
        assert card["designIndex"] == 0.61 and card["designSeeds"] == []

    def test_the_rough_pill_carries_its_run_count(self) -> None:
        game = {"index": None, "seeds": None, "staleness": None, "designIndex": 0.61,
                "designRun": "design", "designSeeds": [0]}
        drawn = _in_page("", f"scoreTag({json.dumps(game)})")
        assert '0.610 rough <span class="muted">1 run</span>' in drawn
        assert "as the one run" in drawn

    def test_a_rough_number_with_no_count_is_not_drawn(self) -> None:
        """An archive entry from before Q416: *no score*, never a count guessed for it."""
        for seeds in ([], None):
            game = {"index": None, "seeds": None, "staleness": None, "designIndex": 0.61,
                    "designRun": "design", "designSeeds": seeds}
            drawn = _in_page("", f"scoreTag({json.dumps(game)})")
            assert ">no score</span>" in drawn, seeds
            assert "0.61" not in drawn and "rough" not in drawn, seeds

    def test_a_design_run_that_could_not_score_it_says_no_score(self) -> None:
        """Measured and unscored is *no score*; *not measured* is for a game nobody played.

        Since C3 the loop archives a candidate whose result is a placeholder with no index
        (`orchestrator.measure`), so the run that played it is known and its number is
        null. The pill read *not measured* for it (Q440).
        """
        game = {"index": None, "seeds": None, "staleness": None, "designIndex": None,
                "designRun": "design", "designSeeds": [0]}
        drawn = _in_page("", f"scoreTag({json.dumps(game)})")
        assert ">no score</span>" in drawn
        assert "not measured" not in drawn
        assert "design run that invented it (design), as the one run" in drawn
        # A game nobody has played still says so.
        nobody = {"index": None, "seeds": None, "staleness": None, "designIndex": None,
                  "designRun": None, "designSeeds": None}
        assert ">not measured</span>" in _in_page("", f"scoreTag({json.dumps(nobody)})")

    def test_an_evaluated_design_with_no_index_names_the_run_that_played_it(
        self, tmp_path
    ) -> None:
        """The payload half of Q440's pill (Q443): the run and its seeds, the number null.

        `library.discover` kept a design run's score only where the entry had an index,
        so a candidate the run played and could not score carried no `designRun`, and
        its card read *not measured* as if nobody had played it.
        """
        from chess_variant_sandbox.loop.archive import Archive, Entry

        spec = load_spec(RULESETS / "gardner.json")
        digest = spec_hash(spec)
        Archive(tmp_path / "runs" / "design").record(
            Entry(hash=digest, name="a candidate", generation=1, parent=None,
                  status="evaluated", index=None, viable=False, seeds=[0]),
            spec=spec,
        )
        found = library.discover(tmp_path / "rulesets", tmp_path / "runs")
        card = next(g for g in found if g.hash == digest).summary()
        assert card["designIndex"] is None
        assert card["designRun"] == "design" and card["designSeeds"] == [0]

    def test_the_three_design_cards_read_as_three_different_things(self, tmp_path) -> None:
        """From the archive to the pill: scored, played and unscored, never played (Q443).

        An entry that never reached a tournament -- gated out here -- carries no design
        run, so it still reads *not measured*: keeping the run for an unscored entry must
        not make every candidate in the archive look played. And the null never becomes a
        number: the corpus `index` stays empty and no spread is carried for it.
        """
        from chess_variant_sandbox.loop.archive import Archive, Entry

        specs = {name: load_spec(RULESETS / f"{name}.json")
                 for name in ("gardner", "losalamos", "dull")}
        digests = {name: spec_hash(spec) for name, spec in specs.items()}
        archive = Archive(tmp_path / "runs" / "design")
        for name, status, index in (("gardner", "evaluated", 0.61),
                                    ("losalamos", "evaluated", None),
                                    ("dull", "rejected", None)):
            archive.record(
                Entry(hash=digests[name], name=name, generation=1, parent=None,
                      status=status, index=index, index_sd=0.02,
                      viable=index is not None, seeds=[0] if status == "evaluated" else []),
                spec=specs[name],
            )
        found = {g.hash: g for g in library.discover(tmp_path / "rulesets", tmp_path / "runs")}
        cards = {name: found[digest].summary() for name, digest in digests.items()}

        assert found[digests["losalamos"]].design_score == {
            "index": None, "status": "provisional", "revision": None,
            "sd": None, "viable": False, "run": "design", "seeds": [0],
        }
        assert found[digests["dull"]].design_score is None
        for card in cards.values():
            assert card["index"] is None, card["name"]

        def pill(name: str) -> str:
            return _in_page("", f"scoreTag({json.dumps(cards[name])})")

        assert '0.610 rough <span class="muted">1 run</span>' in pill("gardner")
        unscored = pill("losalamos")
        assert ">no score</span>" in unscored and "(design), as the one run" in unscored
        for said in ("not measured", "0.00", "None", "null", "undefined", "NaN"):
            assert said not in unscored, said
        assert ">not measured</span>" in pill("dull")


def _kept(source: str, said: object = "Kept on purpose as a test of the label.") -> library.Game:
    """A small real game whose provenance carries `exhibit`, filed under `source`."""
    spec = load_spec(RULESETS / "gardner.json")
    spec["provenance"] = {**(spec.get("provenance") or {}), "exhibit": said}
    return library.Game(hash=spec_hash(spec), name="kept", description="", source=source,
                        path=Path("kept.json"), spec=spec)


def _card(digest: str, name: str, index: float, exhibit: str | None) -> dict:
    """A card as `/api/library` sends it, scored, with or without the label."""
    return {
        "hash": digest, "name": name, "description": "a game", "source": "archived",
        "board": "8x8", "pieceTypes": 2, "cellBins": [], "families": {}, "components": {},
        "index": index, "viable": False, "weights": "v10", "seeds": [0, 1, 2, 3],
        "runs": None, "runVerdict": None, "seat": None, "staleness": {"current": True},
        "playable": True, "fairyTranslates": False, "exhibit": exhibit,
    }


class TestADeliberateExample:
    """Q141: a game kept on purpose for what it shows says so wherever its number is.

    The user's decision of 2026-09-09 keeps `archive/stones` exactly as it is, as a
    deliberate example of a game that cannot really be played, and obliges a label
    wherever its number is shown, so the number is not read as a score. The label is
    data -- the spec's own `provenance.exhibit`, outside the mechanics hash -- so these
    hold the pages to the field and never to a name, and never to the number, which the
    next replay of the library moves.
    """

    SAID = "Kept on purpose as a test of the label."

    def test_the_stones_file_says_it_in_words_and_with_no_number(self) -> None:
        """The sentence the pages print is the file's, and every kept file is read alike.

        `tests/test_schema_nine.py::FROZEN` holds the file's mechanics hash, which is
        what says adding the sentence moved no game and no measurement.
        """
        stones = load_spec(library.ARCHIVE_DIR / "stones.json")
        said = stones["provenance"]["exhibit"]
        assert said.startswith("Kept on purpose as an example of a game that cannot really")
        assert "not a score" in said
        assert not re.search(r"\d", said), "a number in the label goes stale at a replay"

        carried = {}
        for folder, source in ((RULESETS, "committed"), (library.ARCHIVE_DIR, "archived")):
            for path in sorted(folder.glob("*.json")):
                spec = load_spec(path)
                game = library.Game(hash=spec_hash(spec), name=str(spec["name"]),
                                    description="", path=path, source=source, spec=spec)
                own = (spec.get("provenance") or {}).get("exhibit")
                assert library.exhibit(game) == own, path
                if own:
                    carried[path.name] = own
        assert carried["stones.json"] == said

    def test_only_a_ruleset_the_repository_keeps_may_say_it(self) -> None:
        """A game calling *itself* a deliberate example would be excusing its own number."""
        for source in library.EXHIBIT_SOURCES:
            assert library.exhibit(_kept(source)) == self.SAID, source
        for source in ("designed", "evolved", "draft"):
            assert library.exhibit(_kept(source)) is None, source
        assert library.exhibit(_kept("archived", "   ")) is None
        assert library.exhibit(_kept("archived", f"  {self.SAID}\n")) == self.SAID

    def test_the_schema_takes_a_sentence_and_nothing_else(self) -> None:
        from chess_variant_sandbox.spec import validate_spec

        assert validate_spec(_kept("archived").spec) == []
        assert "provenance.exhibit must be a string" in validate_spec(_kept("archived", 3).spec)
        assert any("provenance.exhibit" in problem
                   for problem in validate_spec(_kept("archived", " ").spec))

    def test_the_card_the_game_and_its_neighbour_carry_it(self, tmp_path) -> None:
        """One field on the listing, the game and a link, from the file through the server."""
        rulesets, runs = tmp_path / "rulesets", tmp_path / "runs"
        rulesets.mkdir()
        runs.mkdir()
        archive = tmp_path / "archive"
        archive.mkdir()
        digests = {}
        for key, source, folder in (("kept", "gardner.json", archive),
                                    ("plain", "dobutsu.json", rulesets)):
            spec = load_spec(RULESETS / source)
            spec["name"] = f"game {key}"
            if key == "kept":
                spec["provenance"] = {"exhibit": self.SAID}
            (folder / f"{key}.json").write_text(json.dumps(spec), encoding="utf-8")
            digests[key] = spec_hash(spec)
            TestTheIndexBand._score(spec, digests[key], runs, 0.2, 0.18, 0.22)

        app = App(rulesets, runs, archive_dir=archive)
        cards = {card["hash"]: card for card in app.get("/api/library")["games"]}
        assert cards[digests["kept"]]["exhibit"] == self.SAID
        assert cards[digests["kept"]]["index"] == 0.2, "the label hides no number"
        assert cards[digests["plain"]]["exhibit"] is None
        assert app.get(f"/api/game/{digests['kept']}")["exhibit"] == self.SAID
        links = app.get(f"/api/game/{digests['plain']}")["links"]
        assert [link["exhibit"] for link in links] == [self.SAID]

    @pytest.mark.parametrize("accounts", [False, True], ids=["local", "site"])
    def test_the_library_draws_it_on_the_card(self, accounts: bool) -> None:
        """On both servers, and the site opens the list ordered by score, best first.

        That ordering is the site's leaderboard of games, and the kept game sits at its
        foot under its own number -- which is where the label matters most.
        """
        listing = {
            "games": [_card("aaaaaaaaaaaa", "a good game", 0.8, None),
                      _card("bbbbbbbbbbbb", "a kept game", 0.17, self.SAID)],
            "opponents": [], "coverage": {"axes": []}, "metrics": None,
            **({"site": {"signedIn": True, "admin": False, "sorts": [], "filters": []}}
               if accounts else {}),
        }
        drawn = _page_on(accounts, "#/", _game_payload("cccccccccccc", site=accounts),
                         answers={"/api/library": listing}, landed='class="panel card"')
        # The cards are drawn into `#results`, under the view's own markup.
        html = drawn["every"]
        assert html.count('class="exhibit"') == 1
        kept = html[html.index('data-hash="bbbbbbbbbbbb"'):]
        assert f'<div class="exhibit">{self.SAID}</div>' in kept
        assert "0.17" in kept.split('class="exhibit"')[0], "the label sits under the number"
        if accounts:
            assert html.index("a good game") < html.index("a kept game")

    @pytest.mark.parametrize("accounts", [False, True], ids=["local", "site"])
    def test_the_game_page_draws_it_over_the_score_and_on_a_neighbour(
        self, accounts: bool
    ) -> None:
        game = _game_payload("dddddddddddd", site=accounts)
        game["exhibit"] = self.SAID
        game["links"] = [{
            "hash": "eeeeeeeeeeee", "name": "its neighbour", "units": 1, "kinds": ["rules"],
            "likeness": 0.5, "shares": [], "line": True,
            "differences": [], "similarity": None,
            "gap": None, "exhibit": "Kept on purpose as the neighbour.",
        }]
        html = _page_on(accounts, "#/game/dddddddddddd", game)["html"]
        scored = html.index("How it scored")
        assert html.index(f'<div class="exhibit">{self.SAID}</div>') > scored
        assert html.index(self.SAID) < html.index('id="keepState"')
        assert '<div class="exhibit">Kept on purpose as the neighbour.</div>' in html

    def test_a_game_with_no_sentence_draws_no_label(self) -> None:
        html = _page_on(False, "#/game/ffffffffffff",
                        _game_payload("ffffffffffff", site=False))["html"]
        assert "How it scored" in html
        assert 'class="exhibit"' not in html

    def test_the_map_draws_it_in_the_panel_and_says_it_in_the_list(self) -> None:
        nodes = [
            {"hash": "kept", "name": "kept", "names": ["kept"], "degree": 0, "far": False,
             "board": "8x8", "score": {"index": 0.17, "viable": False, "seeds": [0, 1, 2, 3],
                                       "weights": "v10"},
             "staleness": {"current": True}, "exhibit": self.SAID},
            {"hash": "plain", "name": "plain", "names": ["plain"], "degree": 0,
             "far": False, "board": "8x8", "score": None, "staleness": None,
             "exhibit": None},
        ]
        setup = (
            f"MAP_DATA = {json.dumps({'nodes': nodes, 'links': [], 'clusters': []})};"
            "MAP_AT = Object.fromEntries(MAP_DATA.nodes.map(node => [node.hash, node]));"
        )
        panel = _in_the_page(setup, "mapNodePanel(MAP_AT.kept)")
        assert f'<div class="exhibit">{self.SAID}</div>' in panel
        assert 'class="exhibit"' not in _in_the_page(setup, "mapNodePanel(MAP_AT.plain)")
        keys = _in_the_page(setup, "explorerKeysHtml()")
        assert f"ruled out. {self.SAID}</button>" in keys
        assert "plain, no score yet</button>" in keys

    def test_the_page_names_no_game_and_writes_no_label(self) -> None:
        """The label reaches the page only as the field: nothing matches `stones`."""
        from page_text import PAGE

        said = load_spec(library.ARCHIVE_DIR / "stones.json")["provenance"]["exhibit"]
        assert "stones" not in PAGE.lower()
        assert said not in PAGE and "93b02b591c87" not in PAGE
        for reader in ("exhibitHtml(game.exhibit)", "exhibitHtml(link.exhibit)",
                       "exhibitHtml(node.exhibit)"):
            assert reader in PAGE, reader


class TestPlaying:
    def test_a_person_and_an_agent_alternate(self, chess_game) -> None:
        session = Sessions().start(chess_game, ("human", "minimax:nodes=200"))
        snapshot = session.snapshot()
        assert len(snapshot["moves"]) == 20
        assert snapshot["awaitingAgent"] is False

        opening = next(m for m in snapshot["moves"] if m["text"].endswith("e4"))
        after = session.play(opening["move"])
        assert after["awaitingAgent"] is True
        assert after["history"][-1]["text"] == "Pe2-e4"

        replied = session.play_agent()
        assert len(replied["history"]) == 2
        assert replied["sideToMove"] == 0
        assert replied["awaitingAgent"] is False

    def test_the_opponent_is_the_agent_the_index_rated(self, chess_game) -> None:
        """The reason the seat goes through ``choose_move``.

        An opponent assembled from ``agents/*.py`` would be a different implementation
        from the one whose rating the index reads, so playing a game to find out what a
        ruleset is like and then reading its score would exercise two different players.
        """
        session = Sessions().start(chess_game, ("greedy", "human"), seed=5)
        engine = Engine(chess_game.spec)
        expected = choose_move(engine, "greedy", session.state, seed=5 + session.state.ply)

        after = session.play_agent()
        assert after["history"][-1]["text"] == engine.describe_move(
            engine.initial_state(), expected
        )

    def test_an_illegal_move_is_refused(self, chess_game) -> None:
        session = Sessions().start(chess_game, ("human", "human"))
        with pytest.raises(ValueError):
            session.play(0xFFFF)

    def test_taking_back_returns_to_a_position_you_were_asked_about(self, chess_game) -> None:
        """One ply is the wrong amount when an agent sits opposite.

        Undoing your own move and leaving the reply on the board leaves a position the
        player was never in, and it is nobody's turn that they chose.
        """
        session = Sessions().start(chess_game, ("human", "minimax:nodes=200"))
        session.play(session.snapshot()["moves"][0]["move"])
        session.play_agent()
        assert len(session.history) == 2

        back = session.undo()
        assert back["history"] == []
        assert back["sideToMove"] == 0
        assert back["seats"][back["sideToMove"]] == "human"

    def test_a_promotion_reaches_the_page_as_a_choice(self) -> None:
        """Several moves share one square pair, and which one is a question for the player."""
        spec = {
            "schema_version": 1,
            "name": "promotion probe",
            "board": {"width": 4, "height": 4},
            "pieces": [
                {"id": "K", "royal": True, "moves": [
                    {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
                {"id": "P", "value": 1, "moves": [
                    {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"}]},
                {"id": "R", "value": 5, "moves": [
                    {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
            ],
            "setup": {"mirror": True, "placements": [
                {"piece": "K", "squares": ["a1"]}, {"piece": "P", "squares": ["c3"]}]},
            "rules": {
                "win": "royal_capture",
                # Rank indices are 0-based here; "last" is the far edge either way.
                "promotion": {"piece": "P", "rank": "last", "to": ["R", "K"]},
                "max_plies": 60,
                "no_progress_draw": 30,
            },
        }
        game = library.Game(
            hash=spec_hash(spec), name=str(spec["name"]), description="",
            # A spec that exists only in this test has no file, and nothing needs one.
            path=Path("<in-memory>"), source="committed", spec=spec,
        )
        session = Sessions().start(game, ("human", "human"))
        promotions = [m for m in session.snapshot()["moves"] if m["kind"] == "promotion"]

        assert len(promotions) >= 2
        # The same two squares, more than once: that is what the picker is for.
        assert len({(m["from"], m["to"]) for m in promotions}) < len(promotions)
        assert all(m["promotion"] is not None for m in promotions)

    def test_a_piece_in_hand_reaches_the_page_as_a_drop(self) -> None:
        """A drop has no origin square, so it cannot be expressed by clicking two."""
        spec = load_spec(RULESETS / "crazyhouse.json")
        engine = Engine(spec)
        game = library.Game(
            hash=spec_hash(spec), name="crazyhouse", description="",
            path=RULESETS / "crazyhouse.json", source="committed", spec=spec,
        )
        session = Sessions().start(game, ("human", "human"))

        # Play until somebody has captured something, then look at what is on offer.
        # A hand is a count per piece type, so "holding something" is a positive sum and
        # not a non-empty tuple -- an empty-looking hand is a row of zeroes.
        for ply in range(80):
            if any(sum(hand) for hand in session.state.hands):
                break
            move = choose_move(engine, "greedy", session.state, seed=ply)
            assert move is not None
            session.play(move)
        assert any(sum(hand) for hand in session.state.hands), (
            "no capture happened in 80 plies; the probe proved nothing"
        )

        snapshot = session.snapshot()
        assert sum(sum(hand) for hand in snapshot["hands"]) > 0
        if sum(snapshot["hands"][snapshot["sideToMove"]]):
            drops = [m for m in snapshot["moves"] if m["kind"] == "drop"]
            assert drops, "a piece in hand produced no drop"
            assert all(m["from"] is None for m in drops)


class TestServer:
    def test_the_page_and_the_api_answer(self, warm_library: None) -> None:
        """Warmed first, as `test_one_connection_serves_several_requests` is: what a cold
        listing costs (11 s on a worker's one core, Q749) is not what this test claims."""
        app = App(RULESETS, library.RUNS_DIR)
        server = Server(("127.0.0.1", 0), handler_for(app))
        assert server.server_port, "--port 0 must still be given a real port"
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/") as response:
                assert b"<title>Chess variant sandbox</title>" in response.read()
            with urlopen(base + "/api/library") as response:
                payload = json.load(response)
            agents = [entry["agent"] for entry in payload["opponents"]]
            assert payload["games"] and "human" in agents
            # Every offered opponent is a real agent config, and every one carries a
            # label -- the roster is what a person picks from, and a raw node budget is
            # not a description of how hard something will be.
            for entry in payload["opponents"]:
                assert entry["label"] and entry["label"] != entry["agent"]
                if entry["agent"] != "human":
                    parse_agent(entry["agent"])
        finally:
            server.shutdown()
            server.server_close()

    def test_a_game_carries_the_engines_verdict_beside_its_roster(self, chess_game, warm_map) -> None:
        """`fairyCheck` is what decided whether the picker offers Fairy-Stockfish.

        Same call, same memo, so the sentence under the picker cannot contradict the
        picker. Without an engine both say nothing was asked; with one, the opponent is
        in the roster exactly when the verdict is ok.
        """
        app = App(RULESETS, library.RUNS_DIR)
        payload = app.get(f"/api/game/{chess_game.hash}")
        check = payload["fairyCheck"]
        assert set(check) >= {"engine", "ok", "reasons", "fen", "ours", "theirs"}
        offered = any(entry["store"] == "fairy" for entry in payload["opponents"])
        assert offered == (check["ok"] is True)
        if check["engine"] is None:
            assert check["ok"] is None and check["reasons"] == []

    def test_api_me_answers_nobody_signed_in_rather_than_404(self) -> None:
        """Q199. This application has no store, so `/api/me` is a constant -- 200, never
        the 404 that used to send `/api/profile` to the console on every page load."""
        app = App(RULESETS, library.RUNS_DIR)
        assert app.get("/api/me") == {
            "signedIn": False, "handle": None, "admin": False, "accounts": False,
        }

    def test_reusing_an_address_is_off_on_windows_and_on_everywhere_else(self) -> None:
        """The one place this project disagrees with `http.server`'s defaults.

        `SO_REUSEADDR` is two behaviours under one name. On Windows it lets a second
        process bind a port another process is *listening* on, so both stay alive,
        neither reports anything, and only one is ever reached; on Linux it reuses a
        socket left in TIME_WAIT and a live listener still refuses. A constant is the
        whole fix, so a constant is what is pinned -- and the two arms below prove the
        halves it is for, the refusal and the rebind.
        """
        assert Server.allow_reuse_address == (os.name != "nt")

    def test_a_second_server_on_a_busy_port_is_refused(self) -> None:
        """Two servers, one port -- and no external port is involved.

        Deterministic on purpose: the first server takes an ephemeral port with
        `--port 0` and holds it for the length of the test, so the port the second bind
        is refused on cannot be free, busy for another reason, or a different port than
        the one being tested. Both platforms refuse a bind over a live listener, for
        different reasons -- Windows because `allow_reuse_address` is off there, Linux
        because `SO_REUSEADDR` reuses a TIME_WAIT and never a listener -- so this arm is
        not Windows-only, and on Linux it is what holds the reuse fix to that limit.
        """
        app = App(RULESETS, library.RUNS_DIR)
        first = Server(("127.0.0.1", 0), handler_for(app))
        port = first.server_port
        threading.Thread(target=first.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        try:
            with pytest.raises(OSError):
                Server(("127.0.0.1", port), handler_for(app)).server_close()
            # The refusal is the point, but so is the survivor: the first server is
            # still the one on the port, which is exactly what the old behaviour could
            # not promise.
            with urlopen(f"http://127.0.0.1:{port}/") as response:
                assert b"<title>Chess variant sandbox</title>" in response.read()
        finally:
            first.shutdown()
            first.server_close()

    @pytest.mark.timeout(60)
    def test_serve_says_what_to_do_about_a_busy_port(self) -> None:
        """A refusal a person can act on, and `serve` returns rather than blocking.

        `PortInUse` is a `ValueError` because `cli.main` prints those as one line and
        exits 2; a busy port is a thing the user did, not a stack trace.

        The timeout is the arm's own safety net rather than decoration: the failure
        this test guards against is `serve` *succeeding* on a busy port, and a `serve`
        that succeeds calls `serve_forever`. Without a limit, a regression here hangs
        the suite instead of failing it.
        """
        app = App(RULESETS, library.RUNS_DIR)
        holder = Server(("127.0.0.1", 0), handler_for(app))
        port = holder.server_port
        threading.Thread(target=holder.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        try:
            with pytest.raises(PortInUse) as raised:
                serve(host="127.0.0.1", port=port, ruleset_dir=RULESETS)
        finally:
            holder.shutdown()
            holder.server_close()
        message = str(raised.value)
        assert isinstance(raised.value, ValueError)
        assert str(port) in message and "--port 0" in message
        assert "already in use" in message

    @pytest.mark.timeout(30)
    def test_a_restart_does_not_have_to_wait_for_the_port(self) -> None:
        """Bind, serve, stop, bind again, with no pause anywhere in it.

        The request is what makes this arm mean anything. `Connection: close` hands the
        close to the *server*, so what it leaves behind is a socket in TIME_WAIT whose
        **local** address is the listening port -- the state that used to refuse the
        rebind, and that cost the deployed site 65 s per restart (measured on the box
        2026-09-11). Without a served request the port is clean and the rebind would
        succeed on any platform under any option, which would prove nothing.

        Windows and Linux both pass, for different reasons: Windows never let a
        connected socket hold the listening port, and Linux now sets `SO_REUSEADDR`.
        """
        app = App(RULESETS, library.RUNS_DIR)
        first = Server(("127.0.0.1", 0), handler_for(app))
        port = first.server_port
        threading.Thread(target=first.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        connection = HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request("GET", "/", headers={"Connection": "close"})
        assert connection.getresponse().read()
        connection.close()
        first.shutdown()
        first.server_close()

        started = time.perf_counter()
        # The bind is the assertion. An OSError here is the minute of 502s come back.
        second = Server(("127.0.0.1", port), handler_for(app))
        second.server_close()
        # A bind is granted or refused at once; the clock only catches a retry loop, which
        # would wait out TIME_WAIT's 60 s. Ten seconds is far below that and far above
        # what a busy machine adds; one second was not, beside three other workers (Q377c).
        assert time.perf_counter() - started < 10.0

    @pytest.mark.timeout(30, func_only=True)
    def test_one_connection_serves_several_requests(self, warm_library: None) -> None:
        """HTTP/1.1, and the Content-Length discipline that is what makes it safe.

        A click is several requests, and each was its own connection and its own socket
        left on the port: 40 requests cost 63 ms and left 43 sockets at HTTP/1.0,
        against 12 ms and 2 kept alive (measured 2026-09-11). Every response goes
        through one `_send` that always sets Content-Length, and a reused connection is
        exactly where a response without one stops being untidy and starts
        desynchronising the next request -- so this asks one connection three times and
        checks the socket it asked down is still the one it opened.

        The listing is warmed first (`warm_library`), and the timeout mark covers only
        this function. Unwarmed, the first listing in an xdist worker took 13 s, longer
        than the socket's 10 s, and the test failed in 2 full runs of 4 (Q415). The timeout
        bounds a kept-alive connection answering. What a cold listing costs is not what
        this test claims.
        """
        app = App(RULESETS, library.RUNS_DIR)
        server = Server(("127.0.0.1", 0), handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            opened = None
            for path in ("/", "/api/library", "/"):
                connection.request("GET", path)
                response = connection.getresponse()
                body = response.read()
                assert response.status == 200 and body
                assert response.version == 11
                assert response.getheader("Content-Length") == str(len(body))
                assert response.getheader("Connection") != "close"
                opened = opened or connection.sock
                # `http.client` drops its socket and opens another the moment a server
                # closes on it, so one socket across three requests is the claim.
                assert connection.sock is opened
        finally:
            connection.close()
            server.shutdown()
            server.server_close()

    def test_an_unknown_endpoint_is_a_404_not_a_stack_trace(self) -> None:
        app = App(RULESETS, library.RUNS_DIR)
        with pytest.raises(KeyError):
            app.get("/api/nonsense")
        with pytest.raises(KeyError):
            app.get("/api/game/deadbeef")

    def test_a_seat_that_is_not_an_agent_is_refused(self, chess_game) -> None:
        """Seats come from a fixed roster, so this endpoint cannot become a way to push
        an arbitrary string into the agent factory."""
        app = App(RULESETS, library.RUNS_DIR)
        with pytest.raises(ValueError):
            app.post("/api/session", {"hash": chess_game.hash, "seats": ["human", "nonsense"]})


# B1's layout (`docs/archive/plan-client-compute.md`, *What B1 built*), with a few bytes a file in
# place of Pyodide's thirteen megabytes. The digest is invented: the route reads the
# directory's name from the manifest and never checks it against the wheel.
RUNTIME = "pyodide-0.29.5"
WHEEL = "wheel-0123456789ab/chess_variant_sandbox-0.1.0-cp313-cp313-pyemscripten_2025_0_wasm32.whl"
ARTEFACT = {
    f"{RUNTIME}/pyodide.mjs": b"export const loadPyodide = () => 0;\n",
    f"{RUNTIME}/pyodide.asm.js": b"var Module = {};\n",
    f"{RUNTIME}/pyodide.asm.wasm": b"\x00asm\x01\x00\x00\x00",
    f"{RUNTIME}/python_stdlib.zip": b"PK\x05\x06" + bytes(18),
    f"{RUNTIME}/pyodide-lock.json": b'{"packages": {}}\n',
    WHEEL: b"PK\x05\x06" + bytes(18),
}


def build_artefact(root: Path, files: dict[str, bytes] = ARTEFACT, **extra: object) -> Path:
    """A `build/browser/` under `root`, and its manifest in `manifest.py`'s shape."""
    import hashlib

    for name, body in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(body)
    manifest = {
        "schema": 1,
        "runtime": RUNTIME,
        "wheel": WHEEL,
        "files": [
            {"path": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            for name, body in sorted(files.items())
        ],
        **extra,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


class TestTheBrowserBuildIsServed:
    """`docs/archive/plan-client-compute.md` B4: the browser build and the worker, served locally.

    `/browser/<path>` answers from `build/browser/`, and only what its manifest names. The
    wasm goes out as `application/wasm`, which `WebAssembly.instantiateStreaming` insists
    on. Everything under the two directories the manifest names is a year and
    `immutable`, because a new Pyodide or a new wheel is a new directory, and the manifest
    is `no-store`, because it is how a page learns which directories are current. None of
    it needs a real build: B1's layout in a temporary directory is what the route reads.
    """

    @staticmethod
    def _get(app: App, *paths: str) -> list[tuple[int, dict[str, str], bytes]]:
        """Each path asked of a running server, as status, headers and body."""
        server = Server(("127.0.0.1", 0), handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        answers = []
        try:
            for path in paths:
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                try:
                    connection.request("GET", path)
                    response = connection.getresponse()
                    body = response.read()
                    answers.append((response.status, dict(response.getheaders()), body))
                finally:
                    connection.close()
        finally:
            server.shutdown()
            server.server_close()
        return answers

    @staticmethod
    def _app(browser: Path) -> App:
        return App(RULESETS, library.RUNS_DIR, browser_dir=browser)

    def test_the_manifest_is_served_and_never_kept(self, tmp_path: Path) -> None:
        root = build_artefact(tmp_path / "browser")
        [(status, headers, body)] = self._get(self._app(root), "/browser/manifest.json")
        assert status == 200
        assert headers["Content-Type"] == "application/json"
        assert headers["Cache-Control"] == "no-store"
        assert body == (root / "manifest.json").read_bytes()
        assert json.loads(body)["runtime"] == RUNTIME

    def test_the_wasm_goes_out_as_application_wasm_for_a_year(self, tmp_path: Path) -> None:
        root = build_artefact(tmp_path / "browser")
        [(status, headers, body)] = self._get(
            self._app(root), f"/browser/{RUNTIME}/pyodide.asm.wasm"
        )
        assert status == 200
        # Exactly this, with no charset: `instantiateStreaming` compares the type.
        assert headers["Content-Type"] == "application/wasm"
        assert headers["Cache-Control"] == "public, max-age=31536000, immutable"
        assert headers["Content-Length"] == str(len(body))
        assert body == ARTEFACT[f"{RUNTIME}/pyodide.asm.wasm"]

    def test_every_file_under_the_two_named_directories_is_immutable(
        self, tmp_path: Path
    ) -> None:
        root = build_artefact(tmp_path / "browser")
        names = sorted(ARTEFACT)
        answers = self._get(self._app(root), *(f"/browser/{name}" for name in names))
        types = {}
        for name, (status, headers, body) in zip(names, answers, strict=True):
            assert status == 200, name
            assert body == ARTEFACT[name], name
            assert headers["Cache-Control"] == "public, max-age=31536000, immutable", name
            types[name.rsplit(".", 1)[-1]] = headers["Content-Type"]
        assert types == {
            "mjs": "text/javascript",
            "js": "text/javascript",
            "wasm": "application/wasm",
            "zip": "application/zip",
            "json": "application/json",
            "whl": "application/zip",
        }

    def test_a_file_outside_the_named_directories_is_not_immutable(
        self, tmp_path: Path
    ) -> None:
        """Only the manifest's two directories carry a version in their name."""
        files = {**ARTEFACT, "notes/readme.txt": b"moves with every build\n"}
        root = build_artefact(tmp_path / "browser", files)
        [(status, headers, _)] = self._get(self._app(root), "/browser/notes/readme.txt")
        assert status == 200
        assert headers["Cache-Control"] == "no-store"
        assert headers["Content-Type"] == "application/octet-stream"

    def test_a_path_that_climbs_out_of_the_build_answers_404(self, tmp_path: Path) -> None:
        """Every spelling of `..`, a directory, and a file the manifest does not name.

        The last arm is a manifest that itself names a path outside: `manifest.py` never
        writes one, and the route still checks where the name lands.
        """
        (tmp_path / "secret.txt").write_text("not the build's", encoding="utf-8")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "secret.txt").write_text("not the build's", encoding="utf-8")
        root = build_artefact(tmp_path / "build" / "browser")
        (root / "stray.txt").write_text("dropped in, never built", encoding="utf-8")
        asked = (
            "/browser/../secret.txt",
            "/browser/../../secret.txt",
            "/browser/%2e%2e/secret.txt",
            "/browser/..%2fsecret.txt",
            "/browser/..%5csecret.txt",
            f"/browser/{RUNTIME}/../../secret.txt",
            f"/browser/{RUNTIME}/%2e%2e/%2e%2e/secret.txt",
            "/browser/%2fsecret.txt",
            "/browser/",
            f"/browser/{RUNTIME}",
            f"/browser/{RUNTIME}/",
            "/browser/stray.txt",
            f"/browser/{RUNTIME}/nothing.js",
        )
        answers = self._get(self._app(root), *asked)
        for path, (status, _, body) in zip(asked, answers, strict=True):
            assert status == 404, path
            assert b"not the build's" not in body and b"never built" not in body, path

        outside = build_artefact(
            tmp_path / "other",
            {**ARTEFACT, "../secret.txt": b"not the build's"},
        )
        [(status, _, body)] = self._get(self._app(outside), "/browser/../secret.txt")
        assert status == 404
        assert b"not the build's" not in body

    def test_with_no_build_the_manifest_is_nothing_and_the_rest_404(
        self, tmp_path: Path
    ) -> None:
        """No directory, an empty one, and one a failed build left without a manifest.

        The last is what `build.sh` leaves when `manifest.py` refuses: the files are
        there and unchecked, so none of them is served. The manifest itself is a quiet
        204, never kept: the designer asks for it on every open, and a 404 there was a
        console error on every page load of a machine with no build (B10).
        """
        failed = tmp_path / "failed"
        build_artefact(failed)
        (failed / "manifest.json").unlink()
        (tmp_path / "empty").mkdir()
        for root in (tmp_path / "absent", tmp_path / "empty", failed):
            (manifest, (status, headers, body)) = self._get(
                self._app(root),
                "/browser/manifest.json",
                f"/browser/{RUNTIME}/pyodide.asm.wasm",
            )
            assert manifest[0] == 204 and manifest[2] == b"", root
            assert manifest[1]["Cache-Control"] == "no-store"
            assert status == 404, root
            assert headers["Content-Type"] == "application/json"
            assert "build.sh makes one" in json.loads(body)["error"]
            # The answer names no directory on the machine: the site answers it too.
            assert str(tmp_path) not in body.decode("utf-8")

    def test_a_manifest_that_is_not_one_serves_nothing(self, tmp_path: Path) -> None:
        root = build_artefact(tmp_path / "browser")
        (root / "manifest.json").write_text("[1, 2", encoding="utf-8")
        answers = self._get(
            self._app(root), "/browser/manifest.json", f"/browser/{RUNTIME}/pyodide.mjs"
        )
        assert [status for status, _, _ in answers] == [204, 404]

    def test_the_default_is_where_the_build_script_writes(self) -> None:
        """`serve` reads `build/browser/` under the working directory, as it reads
        `rulesets/` and `runs/`, and that is the directory `build.sh` fills."""
        repository = Path(__file__).resolve().parents[1]
        if not (repository / "deploy").is_dir():
            pytest.skip("the public export carries no deploy/ (docs/reference/public-fork.md)")
        assert App(RULESETS, library.RUNS_DIR).browser_dir == Path("build") / "browser"
        script = (repository / "deploy" / "pyodide" / "build.sh").read_text(encoding="utf-8")
        assert 'ARTEFACT="$REPO/build/browser"' in script

    @staticmethod
    def _started(
        monkeypatch: pytest.MonkeyPatch, *argv: str, site: bool = False
    ) -> App:
        """The app a `serve` or `site serve` command line builds, caught at the bind.

        `listen` is the first thing either command does with the app, so replacing it with
        a recorder that stops the command gets the real app, built from the real options,
        without a socket, a warm-up or (for the site) a database connection.
        """
        from chess_variant_sandbox import cli
        from chess_variant_sandbox.site import server as site_module

        caught: list[App] = []

        class Stop(Exception):
            pass

        def listen(app: App, host: str, port: int) -> None:
            caught.append(app)
            raise Stop

        monkeypatch.setattr(server_module, "listen", listen)
        monkeypatch.setattr(site_module, "listen", listen)
        monkeypatch.setenv("CHESS_VARIANT_SANDBOX_DATABASE_URL", "postgresql:///unused")
        for name in ("ADMIN_PASSWORD", "ADMIN_HANDLE"):
            monkeypatch.delenv(name, raising=False)
        args = cli._build_parser().parse_args(["site", "serve", *argv] if site else ["serve", *argv])
        with pytest.raises(Stop):
            args.handler(args)
        return caught[0]

    @pytest.mark.parametrize("site", [False, True], ids=["serve", "site serve"])
    def test_the_commands_serve_the_build_from_browser_or_by_default_build_browser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: bool
    ) -> None:
        """G12e. `--browser DIR` is where `/browser/*` is answered from, on `serve` and on
        `site serve`; without it the build is `build/browser/` under the working directory,
        as it was before the option, so a scratch site started from a worktree could be
        given the main checkout's build instead of a junction to it."""
        from chess_variant_sandbox import cli

        assert cli.BROWSER_DIR == server_module.BROWSER_DIR

        default = self._started(monkeypatch, site=site)
        assert default.browser_dir == Path("build") / "browser"

        root = build_artefact(tmp_path / "elsewhere" / "browser")
        app = self._started(monkeypatch, "--browser", str(root), site=site)
        assert app.browser_dir == root
        [(status, _, body)] = self._get(app, "/browser/manifest.json")
        assert status == 200
        assert body == (root / "manifest.json").read_bytes()
        # The default directory is not consulted once the option names another.
        missing = self._started(monkeypatch, "--browser", str(tmp_path / "absent"), site=site)
        assert self._get(missing, "/browser/manifest.json")[0][0] == 204

    def test_the_worker_script_is_served_from_the_package(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """200 and `no-store` when the package carries it: its name carries no version."""
        import chess_variant_sandbox.web

        package = Path(chess_variant_sandbox.web.__file__).resolve().parent
        worker = server_module.WORKER_FILE
        assert worker == package / "static" / "measure-worker.js"
        script = tmp_path / "measure-worker.js"
        script.write_bytes(b"self.onmessage = () => {};\n")
        monkeypatch.setattr(server_module, "WORKER_FILE", script)
        [(status, headers, body)] = self._get(
            self._app(tmp_path / "absent"), "/static/measure-worker.js"
        )
        assert status == 200
        assert headers["Content-Type"] == "text/javascript"
        assert headers["Cache-Control"] == "no-store"
        assert body == script.read_bytes()

    def test_the_worker_script_answers_404_while_there_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server_module, "WORKER_FILE", tmp_path / "measure-worker.js")
        [(status, headers, body)] = self._get(
            self._app(tmp_path / "absent"), "/static/measure-worker.js"
        )
        assert status == 404
        assert headers["Content-Type"] == "application/json"
        assert json.loads(body)["error"]

    def test_the_packages_own_worker_is_what_the_route_answers(self, tmp_path: Path) -> None:
        """Whichever state the package is in: 404 before B3 writes the file, its bytes
        after."""
        [(status, _, body)] = self._get(
            self._app(tmp_path / "absent"), "/static/measure-worker.js"
        )
        if server_module.WORKER_FILE.is_file():
            assert (status, body) == (200, server_module.WORKER_FILE.read_bytes())
        else:
            assert status == 404


class TestThePageIsStillJavaScript:
    """Hundreds of kilobytes of untyped JavaScript, once a Python string literal.

    It is plain files now (Q513), each parsed on its own in `tests/test_page_files.py`;
    this parses them joined, as `page_text.PAGE` reads them. Before the move `ruff` and
    `mypy` saw a `str`, the tests above read it with `index` and `in`, and nothing
    anywhere checked that it parsed. A stray backtick or an unbalanced brace ships
    as a blank page and shows up the first time somebody opens the app. Node is present
    on any machine that has ever built a front end and this skips where it is not, so the
    check is free where it works and silent where it cannot.
    """

    def test_the_page_script_parses(self) -> None:
        import re

        from page_text import PAGE

        scripts = re.findall(r"<script[^>]*>(.*?)</script>", PAGE, re.S)
        assert scripts, "the page has no script block, which is itself the bug"
        done = node_session.run("\n".join(scripts), check=True)
        assert done.returncode == 0, done.stderr

    def test_a_wild_piece_is_drawn_as_neither_players(self) -> None:
        """Schema 9's third owner, on the play board and on the designer preview.

        Board bytes are all that cross, so both renderers read the owner off one --
        `floor((value - 1) / sideStride)`, which is 2 for a wild piece -- and neither
        may index an icon pair with it: a pair is one drawing per army and has no third
        entry. The page is a string and nothing type-checks it, so this reads the string.
        """
        from page_text import PAGE

        assert "--p2:" in PAGE and "--p2-ink:" in PAGE
        # Once in the play board's `drawPiece`, once in the designer preview's. Generation 42
        # (N4c): the duck, a third-owner piece too, is the one that wears its type's drawing,
        # and only by the first seat's entry (`duck ? 0 : player`), never the owner's.
        assert PAGE.count("'var(--p2)'") == 2
        assert PAGE.count("player > 1") == 2
        assert PAGE.count("[duck ? 0 : player]") == 1 and PAGE.count("pair[duck ? 0 :") == 1


class TestTheLoadingIndicator:
    """One `pending()`, shown at every transition that waits on a slow server call.

    Opening a game from the library, going back to the library from a game, and
    starting a new game -- from the game page, from the board's new-game button or
    from its play-the-other-side button -- each wait on one server call that can take
    several seconds warm on this machine, worse cold, worse still on the box. They call
    the same function the moment they are clicked, before anything is awaited, and `fail`
    -- the page's one error path -- clears it so a failed fetch never leaves it spinning
    over a page that is never going to arrive.

    Named `pending` rather than `busy`: board.py already uses `busy` as a local variable
    in a dozen animation functions for how long a move takes to draw, a wholly different
    kind of waiting, and a page-level `function busy()` collided with it head-on --
    `_watch_ply`'s driver script declares `const busy = animateChange(...)` at the top
    level and a `const` cannot share a name with a function declaration in the same
    scope. Caught by the full suite, not by `node --check`, which only parses one view's
    worth of script at a time.
    """

    def test_each_transition_shows_it_before_its_one_fetch(self) -> None:
        from page_text import PAGE

        # The library card: opening a game after selecting it.
        assert "pending('Loading the game…'); go('#/game/' + card.dataset.hash); };" in PAGE
        # The game page's crumb: back to the library.
        assert "pending('Loading the library…'); go('#/'); };" in PAGE
        # The game page's start button: a new game, called before the session POST.
        start = PAGE.index("    start.onclick = () => {")
        body = PAGE[start:PAGE.index("document.getElementById('swap').onclick", start)]
        assert "pending('Starting the game…');" in body
        assert body.index("pending(") < body.index("api('/api/session', body)")

    def test_the_library_says_it_is_loading_too(self) -> None:
        """`docs/archive/plan-ux.md` C1. A cold server warms the listing for twenty seconds, and
        the landing page was a header and a footer for all of it."""
        from page_text import PAGE

        start = PAGE.index("async function showLibrary()")
        body = PAGE[start:PAGE.index("function sortValue(", start)]
        assert "pending('Loading the library…');" in body
        assert body.index("pending(") < body.index("api('/api/library')")

    def test_the_two_mid_game_buttons_show_it_too(self) -> None:
        """New game and play the other side hit the same session call as the start button.

        Both go through the board's `restart`, which is where `pending` fires, so one
        click shows it once whichever button was pressed and whichever route the game
        takes -- a saved ruleset's POST or a draft's scratch play.
        """
        from page_text import PAGE

        start = PAGE.index("const restart = (seats, follows = true) =>")
        body = PAGE[start:PAGE.index("document.getElementById('flip').onclick", start)]
        assert "pending('Starting the game…');" in body
        assert body.index("pending(") < body.index("playDraft(seats)")
        assert body.index("pending(") < body.index("api('/api/session', body)")
        # Every button routes through it, so none needs a `pending` of its own.
        assert "if (again) again.onclick = () => restart(snapshot.seats);" in body
        assert "restart([snapshot.seats[1], snapshot.seats[0]]);" in body
        assert "restart([WATCH_AGENT, WATCH_AGENT], false);" in body
        assert body.count("pending(") == 1


class TestTheRematchControl:
    """Box C1 of `docs/archive/plan-meta-game.md`, on the page's side.

    The board's `restart` is the rematch control, and the store records what it starts as
    a rematch of the game it was pressed on (`matches.rematch_of`). What it sends is the
    match id and nothing about the player, because the server already knows who is asking
    and the match it names already holds both seats.
    """

    def test_it_names_the_game_it_was_pressed_on(self) -> None:
        from page_text import PAGE

        start = PAGE.index("const restart = (seats, follows = true) =>")
        body = PAGE[start:PAGE.index("document.getElementById('flip').onclick", start)]
        # The whole body, so a later field about the person would have to break this.
        assert "const body = { hash: rules.hash, seats };" in body
        assert "if (follows && !watched) body.rematch_of = snapshot.id;" in body

    def test_a_game_nobody_played_names_nothing(self) -> None:
        """Q262's watch. Two computer players is nobody's game, so the board it starts
        follows nothing and a watch started from a board follows nothing either."""
        from page_text import PAGE

        start = PAGE.index("const watched = !snapshot.seats.includes('human');")
        body = PAGE[start:PAGE.index("document.getElementById('flip').onclick", start)]
        assert "if (follows && !watched) body.rematch_of = snapshot.id;" in body
        assert "restart([WATCH_AGENT, WATCH_AGENT], false);" in body

    def test_starting_from_the_game_page_sends_nothing(self) -> None:
        """A match started any other way has the column null, which is what makes a count
        over it a number. The game page's start button is the other way."""
        from page_text import PAGE

        start = PAGE.index("const body = { hash, seats };")
        body = PAGE[start:PAGE.index("api('/api/session', body)", start)]
        assert "rematch" not in body

    def test_the_indicator_is_the_pages_own_busy_idiom(self) -> None:
        """`.spin` is already how the board says an agent is thinking (`maybeAgent`).

        `pending()` reuses it rather than inventing a second way to say "working" --
        and removes what it showed through `view`, the same element the error box used to
        be prepended to, so the two are cleared and shown the same way. `fail` has since
        moved its box (Q264); clearing the spinner is still its first act.
        """
        from page_text import PAGE

        assert '.busy {' in PAGE  # its own rule, not a bare reuse of .error's
        start = PAGE.index("function pending(")
        body = PAGE[start:PAGE.index("function clearPending(", start)]
        assert '<span class="spin"></span>' in body
        assert "view.prepend(box)" in body
        assert "clearPending()" in PAGE[PAGE.index("function fail("):start]

    def test_a_failed_fetch_clears_it_and_shows_the_error_instead(self) -> None:
        result = _pending_then_fail("() => Promise.reject(new Error('offline'))")
        assert result["afterPending"] == {"count": 1, "className": "busy"}
        # The spinner went and nothing took its place in the flow: a failure nobody's
        # click caused is fixed to the viewport instead (Q264).
        assert result["view"] == []
        assert result["afterFail"]["count"] == 1
        assert result["afterFail"]["classes"] == ["error toast"]
        # The page's sentence, never the browser's (Q659, `tests/test_page_requests.py`).
        assert result["afterFail"]["text"] == [
            "The application did not answer. Try again in a moment."]

    def test_a_non_2xx_response_clears_it_the_same_way(self) -> None:
        """`api()` treats a non-ok response as a failure too -- a 500, not just offline."""
        fetch_stub = (
            "() => Promise.resolve({ ok: false, statusText: 'Server Error', "
            "json: () => Promise.resolve({ error: 'the core is stale' }) })"
        )
        result = _pending_then_fail(fetch_stub)
        assert result["afterPending"] == {"count": 1, "className": "busy"}
        assert result["view"] == []
        assert result["afterFail"]["count"] == 1
        assert result["afterFail"]["classes"] == ["error toast"]
        assert result["afterFail"]["text"] == ["the core is stale"]


def _pending_then_fail(fetch_stub: str) -> dict:
    """Show the loading indicator, then run a failing `api()` call through `fail`, in node.

    Stands in for whichever of the three transitions called `pending()`: what is under
    test is only that `fail` clears it, so `pending`, `api` and `fail` are exercised
    directly rather than the router and the click handlers around them. `view` is a
    small fake element that actually tracks its children, because the thing being
    checked is which boxes are on it before and after -- a stub that swallows
    `prepend`/`remove` the way the animation and agent-reply stubs do would hide the
    exact bug this guards against.
    """
    import json as _json
    import re

    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = f"""
      function __node() {{
        const node = {{
          className: '', _html: '', _text: '', children: [], parent: null,
          get innerHTML() {{ return node._html; }},
          set innerHTML(v) {{ node._html = v; }},
          get textContent() {{ return node._text; }},
          set textContent(v) {{ node._text = v; }},
          prepend(child) {{ child.parent = node; node.children.unshift(child); }},
          appendChild(child) {{ child.parent = node; node.children.push(child); }},
          // The header's links go on through `append`, which the boot line reaches
          // before this runner does; `TestTheWayInToTheSitesOwnPages` is what tests it.
          append(...kids) {{ kids.forEach(kid => node.children.push(kid)); }},
          remove() {{
            if (node.parent) {{
              node.parent.children = node.parent.children.filter(c => c !== node);
              node.parent = null;
            }}
          }},
          querySelector() {{ return null; }}, querySelectorAll() {{ return []; }},
          addEventListener() {{}}, setAttribute() {{}}, closest() {{ return null; }},
          classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }}, style: {{}},
        }};
        return node;
      }}
      const VIEW = __node(), BODY = __node();
      const __toasts = () => BODY.children.filter(c => c.className === 'error toast');
      global.document = {{
        getElementById: id => (id === 'view' ? VIEW : __node()),
        querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {{}},
        createElement: () => __node(), body: BODY,
      }};
      global.window = {{ matchMedia: () => ({{ matches: false, addEventListener: () => {{}} }}),
                        addEventListener: () => {{}}, scrollTo: () => {{}} }};
      global.location = {{ pathname: '/game', hash: '' }};
      // The boot's `identify()` is never answered, so it never reaches `render()` and the
      // runner below is the only thing drawing (Q659: `identify` goes through `api` too).
      const __stub = {fetch_stub};
      global.fetch = (path, options) =>
        path === '/api/me' ? new Promise(() => {{}}) : __stub(path, options);
    """
    runner = """
      API_RETRY_PAUSE = 0;
      pending('Loading the game…');
      const afterPending = { count: VIEW.children.length,
                              className: VIEW.children[0] && VIEW.children[0].className };
      api('/api/library').catch(fail).then(() => {
        console.log(JSON.stringify({
          afterPending: afterPending,
          view: VIEW.children.map(c => c.className),
          // `.bubble` is on the body too, put there by the router's own boot.
          afterFail: { count: __toasts().length,
                       classes: __toasts().map(c => c.className),
                       text: __toasts().map(c => c.textContent) },
        }));
      });
    """
    done = node_session.run(stub + script.group(1) + "\n" + runner)
    assert done.returncode == 0, done.stderr
    return _json.loads(done.stdout)


class TestAnErrorLandsWhereTheReaderIsLooking:
    """Q264, from the user's inbox of 2026-09-14: *"The error messages landing on top of
    the site, when you're scrolled down are easy to miss."*

    Every error went to the top of `view`, which is the top of the page. On a game page
    or a warm library that is several screens above whatever the reader was doing, so a
    refusal they had just asked for was written somewhere they were not looking.

    There are now two places and one `fail` that picks between them. A request a person
    made from a control -- a move, a take-back, a start, a sweep -- is answered on that
    control's own line. Everything else, which is a route that would not draw or a poll
    nobody asked for, is fixed to the viewport, where a scroll cannot carry it off.
    """

    def test_a_control_gets_its_answer_on_its_own_line(self) -> None:
        seen = _error_paths()
        assert seen["besideRow"] == ["", "error beside"]
        assert seen["besideText"] == ["that seat is taken"]

    def test_pressing_a_refused_control_twice_leaves_one_answer(self) -> None:
        """One refusal, twice -- not a column of them growing under the button."""
        seen = _error_paths()
        assert seen["besideRow"].count("error beside") == 1

    def test_an_error_nobody_asked_for_is_fixed_to_the_viewport(self) -> None:
        seen = _error_paths()
        assert seen["toastClasses"] == ["error toast"]
        assert seen["toastText"] == ["the connection dropped"]

    def test_only_one_of_them_is_up_at_a_time(self) -> None:
        """The newest is the one still true, and two stacked would cover the page."""
        seen = _error_paths()
        assert seen["afterSecond"] == 1

    def test_it_can_be_dismissed(self) -> None:
        """Fixed and undismissable would be a box sitting over the page for good."""
        seen = _error_paths()
        assert seen["afterDismiss"] == 0

    def test_a_new_screen_takes_it_with_it(self) -> None:
        """It lives on `document.body`, so redrawing `view` does not remove it; `render`
        does, because the error belongs to the screen that raised it."""
        seen = _error_paths()
        assert seen["afterRender"] == 0

    def test_the_fixed_one_is_fixed_and_readable_on_a_phone(self) -> None:
        """375 px wide with a 12 px gutter either side, so it is a box and not a strip."""
        from page_text import PAGE

        rule = PAGE[PAGE.index(".error.toast {"):PAGE.index(".error.toast > button")]
        assert "position: fixed" in rule
        assert "bottom: 12px" in rule and "left: 12px" in rule and "right: 12px" in rule
        # It floats over the page, so it cannot borrow the page's background.
        assert "background: var(--panel)" in rule

    def test_the_one_beside_a_control_never_covers_it(self) -> None:
        """It is in the flow, on a line of its own under the control's row, rather than
        positioned over anything."""
        from page_text import PAGE

        rule = PAGE[PAGE.index(".error.beside {"):PAGE.index(".error.toast {")]
        assert "flex: 1 0 100%" in rule
        assert "position" not in rule

    def test_the_two_paths_are_the_one_mechanism(self) -> None:
        """A later page calls `fail` and gets one of the two; it does not build a third."""
        from page_text import PAGE

        start = PAGE.index("function fail(error, near)")
        body = PAGE[start:PAGE.index("function clearToast(", start)]
        assert "return control ? beside(control, words) : toast(words);" in body
        # Two places an error box is built, both inside `fail`'s own helpers.
        assert "className = 'error'" not in PAGE
        assert PAGE.count("className = 'error beside'") == 1
        assert PAGE.count("className = 'error toast'") == 1

    def test_the_controls_that_name_themselves(self) -> None:
        """One call site per page, each the request a person most obviously asked for."""
        from page_text import PAGE

        for named in ("fail(error, 'undo')", "fail(error, 'start')", "fail(error, button)"):
            assert named in PAGE, named

    def test_nothing_is_put_beside_the_status(self) -> None:
        """Q725: a box beside the status went into the strip and pushed the board down.
        A move's refusal and a machine's failure are toasts; the strip's height is fixed."""
        from page_text import PAGE

        assert "fail(error, 'status')" not in PAGE and "fail(error, status)" not in PAGE
        assert "'status')," not in PAGE.replace("getElementById('status'),", "")
        start = PAGE.index("function send(move)")
        body = PAGE[start:PAGE.index("\n}\n", start)]
        assert body.count("fail(error);") == 2 and "fail(new Error(" in body

    def test_the_designers_save_still_answers_under_its_own_button(self) -> None:
        """It writes its refusal into the saved-state box rather than calling `fail`,
        because the sentence has to survive a redraw of that panel. Beside the control
        either way, which is what Q264 asks of it."""
        from page_text import PAGE

        start = PAGE.index("function saveDraft(")
        body = PAGE[start:PAGE.index("function playDraft(", start)]
        assert 'DRAFT_SAVED = `<div class="error">' in body
        assert "box.innerHTML = DRAFT_SAVED;" in body


def _error_paths() -> dict:
    """Drive `fail` down both of its paths in node, on a DOM that tracks parents.

    `closest` and `querySelectorAll` are what `beside` uses to find the control's line
    and to clear the last answer off it, so the stub implements both against real
    children rather than returning nothing the way the view stubs do.
    """
    import json as _json
    import re

    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      function __node(cls) {
        const node = {
          className: cls || '', children: [], parent: null, _text: '', _html: '',
          style: {}, dataset: {}, hidden: false,
          get innerHTML() { return node._html; },
          set innerHTML(v) { node._html = String(v); },
          get textContent() { return node._text; },
          set textContent(v) { node._text = String(v); },
          append(...kids) { kids.forEach(kid => { kid.parent = node;
                                                  node.children.push(kid); }); },
          appendChild(kid) { kid.parent = node; node.children.push(kid); return kid; },
          prepend(kid) { kid.parent = node; node.children.unshift(kid); },
          remove() {
            if (node.parent) {
              node.parent.children = node.parent.children.filter(c => c !== node);
              node.parent = null;
            }
          },
          closest(selector) {
            const want = selector.split(',').map(s => s.trim().slice(1));
            for (let at = node; at; at = at.parent) {
              if (want.some(w => at.className.split(' ').includes(w))) return at;
            }
            return null;
          },
          querySelectorAll(selector) {
            const want = selector.trim().slice(1).split('.');
            return node.children.filter(
              c => want.every(w => c.className.split(' ').includes(w)));
          },
          querySelector() { return null; },
          addEventListener() {}, setAttribute() {},
          classList: { toggle() {}, add() {}, remove() {} },
        };
        return node;
      }
      const BODY = __node('body'), VIEW = __node('view'), ROW = __node('row'),
            UNDO = __node('');
      ROW.append(UNDO);
      const __KNOWN = { view: VIEW, undo: UNDO };
      global.document = {
        getElementById: id => __KNOWN[id] || (__KNOWN[id] = __node('')),
        querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {},
        createElement: () => __node(''), body: BODY,
      };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, innerWidth: 375, scrollY: 0,
                        scrollTo: () => {} };
      global.location = { pathname: '/game', hash: '' };
      // Nothing answers, so the boot's `identify()` never reaches `render()` and the
      // runner below is the only thing driving the page.
      global.fetch = () => new Promise(() => {});
    """
    runner = """
      const toasts = () => BODY.children.filter(c => c.className === 'error toast');
      fail(new Error('that seat is taken'), 'undo');
      fail(new Error('that seat is taken'), 'undo');
      const besideRow = ROW.children.map(c => c.className);
      const besideText = ROW.children.filter(c => c.className === 'error beside')
        .map(c => c.textContent);
      fail(new Error('the connection dropped'));
      const toastClasses = toasts().map(c => c.className);
      const toastText = toasts().map(c => c.textContent);
      fail(new Error('and the server went away'));
      const afterSecond = toasts().length;
      toasts()[0].children[0].onclick();
      const afterDismiss = toasts().length;
      fail(new Error('gone on the next screen'));
      render();
      console.log(JSON.stringify({
        besideRow, besideText, toastClasses, toastText, afterSecond, afterDismiss,
        afterRender: toasts().length,
      }));
    """
    done = node_session.run(stub + script.group(1) + "\n" + runner)
    assert done.returncode == 0, done.stderr
    return _json.loads(done.stdout)


class TestEveryRouteStartsAtTheTop:
    """`docs/archive/plan-ux.md` C2. The router scrolls the window, and nothing else.

    A hash change replaces the view and left the window where the last one had been
    scrolled to, so a game opened from the bottom of the library arrived somewhere in
    the middle of its own page -- past its name and past the play panel. Only the window
    moves: the board's move list and the tab panes scroll inside themselves, and a
    window scroll does not touch either.
    """

    HASH = "abc123def456"

    def test_drawing_a_game_scrolls_the_window_up(self) -> None:
        seen = _page_on(False, f"#/game/{self.HASH}",
                        _game_payload(self.HASH, site=False))
        assert seen["errors"] == []
        assert [0, 0] in seen["scrolled"]

    def test_the_library_does_the_same(self) -> None:
        seen = _page_on(False, "", _game_payload(self.HASH, site=False))
        assert seen["errors"] == []
        assert [0, 0] in seen["scrolled"]

    def test_only_the_window_is_scrolled(self) -> None:
        """The inner scrollers keep their own position, so nothing else is reset."""
        from page_text import PAGE

        start = PAGE.index("function render() {")
        body = PAGE[start:PAGE.index("showLibrary().catch(fail);\n}", start)]
        assert body.count("scrollTo") == 1
        assert "window.scrollTo(0, 0);" in body


class TestTheLandingAsksForNothingTheSiteWithholds:
    """`docs/archive/plan-ux.md` C1's companion: the survey was asked for on every landing.

    `/api/scoring` is a withheld route, so `api` refused it before it was sent and the
    listing caught the refusal and drew no panel -- correct, and still one 404 in the
    console every time somebody opened the site. The question is now asked only where it
    is served, and `offered()` is what decides, because that is the one place the
    withheld table is read.
    """

    HASH = "abc123def456"

    def test_the_site_never_asks_for_the_survey(self) -> None:
        seen = _page_on(True, "", _game_payload(self.HASH, site=True))
        assert seen["errors"] == []
        assert "GET /api/library" in seen["fetched"]
        assert not [asked for asked in seen["asked"] if "/api/scoring" in asked]

    def test_locally_it_is_still_asked_for(self) -> None:
        """The panel is the local application's way to fix forty stale scores."""
        seen = _page_on(False, "", _game_payload(self.HASH, site=False))
        assert seen["errors"] == []
        assert "GET /api/scoring" in seen["asked"]

    def test_the_decision_reads_the_withheld_table(self) -> None:
        """`offered()` rather than a check of this view's own, which would be a second
        copy of the table and would drift from `site/routes.py` without saying so."""
        from page_text import PAGE

        start = PAGE.index("async function showLibrary()")
        body = PAGE[start:PAGE.index("function sortValue(", start)]
        assert "offered('GET', '/api/scoring')" in body


class TestTheWayInToTheSitesOwnPages:
    """Signing in, a profile, the credits and the privacy notice, linked from the page.

    All five pages existed and nothing under `web/page/` pointed at any of them, so the
    way to one was typing the address. The header's corner now shows *sign in* to a
    visitor and the reader's own handle to somebody signed in, and every page carries a
    footer with the two prose pages in it.

    Gated on whether the server has accounts at all, which is what separates the site
    from the local application: the local one has no store, serves none of the five
    pages, and would have been given a header offering a link that 404s. `/api/me`
    (Q199) answers both the same way, 200 either way, and says `accounts` itself rather
    than leaving the page to read a status code -- the local application's `accounts`
    is always `false`, and the site's is always `true`, signed in or not.
    """

    def test_a_visitor_is_offered_the_way_in(self) -> None:
        drawn = _header_after(
            "() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve("
            "{ signedIn: false, handle: null, admin: false, accounts: true,"
            " players: true, round: true }) })"
        )
        # L7 (`docs/archive/plan-launch.md` §4): what an account adds, in one line beside it.
        # LP0: the reason is its own span, which a phone's corner drops (`style.py`), so
        # there the corner reads *sign in* whole rather than *… count*.
        signin, why = drawn["who"]
        assert signin == {"text": "sign in", "href": "/signin"}
        assert (why["tag"], why["className"], why["_text"]) == (
            "span", "accountWhy", " to make your games count")
        assert drawn["foot"] == [
            {"text": "games", "href": "/game"},
            " · ",
            {"text": "about", "href": "/"},
            " · ",
            {"text": "the score", "href": "#/score"},
            " · ",
            {"text": "what's new", "href": "#/whats-new"},
            " · ",
            {"text": "the round", "href": "#/round"},
            " · ",
            {"text": "players", "href": "/players"},
            " · ",
            {"text": "suggestions", "href": "/suggestions"},
            " · ",
            {"text": "credits", "href": "/credits"},
            " · ",
            {"text": "privacy", "href": "/privacy"},
            " · ",
            {"text": "terms", "href": "/terms"},
            " · ",
            {"text": "contact", "href": "https://github.com/nabfrew"},
        ]

    def test_a_phone_shortens_the_visitors_corner_to_sign_in(self) -> None:
        """LP0: at 375 the corner was cut from the left to *… count*. Inside the phone's
        rules the reason is dropped, and only there: a desk keeps the whole line."""
        import re

        from chess_variant_sandbox.web.page import style

        rule = "header.top #who .accountWhy { display: none; }"
        assert style.RULES.count(rule) == 1
        before = style.RULES[:style.RULES.index(rule)]
        phone = before.rindex("@media (max-width: 760px)")
        assert before.count("{", phone) - before.count("}", phone) == 1, "inside the media rule"
        assert not re.search(r"\.accountWhy\s*\{[^}]*display:\s*none",
                             style.RULES[:phone])

    def test_players_is_linked_only_once_the_list_starts(self) -> None:
        """L3 of `docs/archive/plan-launch.md` §3.3: below five people `/api/me` says
        `players: false` and the footer has no way to a page that lists nobody; from five
        on it does. The same answer for every reader, so the link never comes and goes
        with who is reading."""
        def foot(players: str) -> list[object]:
            return _header_after(
                "() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve("
                "{ signedIn: true, handle: 'ada', admin: false, accounts: true,"
                f" players: {players} }}) }})"
            )["foot"]
        assert {"text": "players", "href": "/players"} not in foot("false")
        assert {"text": "players", "href": "/players"} in foot("true")

    def test_the_round_is_linked_only_while_it_has_an_entrant(self) -> None:
        """L4 of `docs/archive/plan-launch.md` §3.4: `/api/me` says `round` while the latest round
        has an entrant, and only then does the footer announce it."""
        def foot(entered: str) -> list[object]:
            return _header_after(
                "() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve("
                "{ signedIn: false, handle: null, admin: false, accounts: true,"
                f" round: {entered} }}) }})"
            )["foot"]
        assert {"text": "the round", "href": "#/round"} not in foot("false")
        assert {"text": "the round", "href": "#/round"} in foot("true")

    def test_a_signed_in_person_is_shown_their_handle(self) -> None:
        drawn = _header_after(
            "() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve("
            "{ signedIn: true, handle: 'ada', admin: false, accounts: true }) })"
        )
        # Q731: a button to your page, named for where it goes and for whom.
        assert drawn["who"] == [_your_page("ada")]

    def test_a_server_with_no_accounts_is_offered_nothing_it_does_not_have(self) -> None:
        """The local application serves none of the five pages, so a header linking to
        one would link to a 404. The footer keeps *what's new*, which is a route inside
        the application and a file in the tree, and drops the two pages the store's
        server has."""
        drawn = _header_after(
            "() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve("
            "{ signedIn: false, handle: null, admin: false, accounts: false }) })"
        )
        assert drawn["who"] == []
        assert drawn["foot"] == [{"text": "games", "href": "/game"}, " · ",
                                 {"text": "about", "href": "/"}, " · ",
                                 {"text": "the score", "href": "#/score"}, " · ",
                                 {"text": "what's new", "href": "#/whats-new"}]

    def test_the_handle_is_a_text_node_and_never_markup(self) -> None:
        """A handle is a name somebody chose for themselves. `_header_after`'s header and
        footer throw on `innerHTML`, so this passing is the claim: the links are built as
        elements with `textContent`, and a handle full of tags is a handle full of tags.
        """
        drawn = _header_after(
            "() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve("
            "{ signedIn: true, handle: '<img src=x onerror=alert(1)>', admin: false,"
            " accounts: true }) })"
        )
        assert drawn["who"] == [_your_page("<img src=x onerror=alert(1)>")]

    def test_only_a_maintainer_is_shown_the_way_to_the_admin_page(self) -> None:
        """`/admin` is a page a `users.role` of `admin` reaches and nobody else does, so
        a link to it in everybody's header would be an invitation to a 404 -- and a list
        of what the site has that they cannot see."""
        signed_in = (
            "() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve("
            "{ signedIn: true, handle: 'ada', admin: false, accounts: true }) })"
        )
        drawn = _header_after(signed_in, admin="true")
        assert drawn["who"] == [
            {"text": "Maintenance", "href": "/admin"},
            " · ",
            _your_page("ada"),
        ]
        assert _header_after(signed_in, admin="false")["who"] == [_your_page("ada")]

    def test_an_answer_that_does_not_say_leaves_the_flag_where_it_was(self) -> None:
        """The local application has no roles at all and sends the key nowhere, so a
        `noteAdmin` that took any falsy value for a *no* would be reading an absence as
        an answer. Both payloads that do say are the site's."""
        drawn = _header_after(
            "() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve("
            "{ signedIn: true, handle: 'ada', accounts: true }) })",
            admin="undefined",
        )
        assert drawn["who"] == [_your_page("ada")]

        from page_text import PAGE

        assert "if (typeof said !== 'boolean' || said === ADMIN) return;" in PAGE
        # Both senders, and nothing else: the listing's block and the game page's.
        assert PAGE.count("noteAdmin(SITE && SITE.admin);") == 2
        assert "noteAdmin(game.standing && game.standing.admin);" in PAGE

    def test_the_header_and_the_footer_are_outside_the_view(self) -> None:
        """Every view reassigns `view.innerHTML`, so anything drawn inside it is drawn
        again per page. These are painted once, when the one identity call lands."""
        from page_text import PAGE

        assert '<span class="muted small" id="who"></span>' in PAGE
        assert PAGE.index('id="who"') < PAGE.index('<div id="view">')
        assert PAGE.index('<div id="view">') < PAGE.index('id="foot"')
        boot = PAGE[PAGE.index("identify().then(() => {"):]
        assert boot.index("paintIdentity();") < boot.index("render();")

    def test_the_mode_control_and_the_header_agree_about_who_is_signed_in(self) -> None:
        """Two readers of one fact, from two places: the game page's mode control reads
        `standing.signedIn` off the game it is drawing, and the header reads the profile
        route. A visitor offered a rated game they cannot start, or a handle in the
        corner over a sentence telling them to sign in, would be the page contradicting
        itself on one screen."""
        from page_text import PAGE

        assert "standing.signedIn === false" in PAGE
        start = PAGE.index("async function identify(")
        body = PAGE[start:PAGE.index("\nfunction paintIdentity(", start)]
        assert "api('/api/me')" in body
        assert "ACCOUNTS = data.accounts === true;" in body
        assert "VIEWER = data.handle || null;" in body and "VIEWER_MARK = data.mark" in body


def _your_page(handle: str) -> dict:
    """The header's button to your own page as `_header_after` reports it (Q731): its
    accessible name, where it goes, its visible words leaf by leaf, and its class."""
    return {"text": f"Your page, signed in as {handle}", "href": "/profile",
            "shown": f"{handle}|your page| ›", "kind": "memberButton"}


def _header_after(fetch_stub: str, admin: str = "undefined") -> dict:
    """Run `identify()` and `paintIdentity()` in node, and report what they built.

    The header and the footer are real enough to track their children and to *refuse*
    `innerHTML`, which is what makes the text-node claim a test rather than a comment.
    Everything else the page reaches for at load is the permissive stub `_in_page` uses,
    and every fetch but the identity one never resolves -- the router runs at load and
    would otherwise draw a library out of a profile.

    `admin` is JavaScript for what a payload said about the reader's role, handed to
    `noteAdmin` the way the listing and the game page hand it theirs.
    """
    import json as _json
    import re

    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = f"""
      const el = new Proxy({{}}, {{ get: (t, k) => (k === 'style' ? {{}} : () => null),
                                   set: () => true }});
      function __strict() {{
        const node = {{
          children: [], _text: '', href: '', tag: 'span',
          get textContent() {{ return node._text; }},
          set textContent(v) {{ node._text = String(v); node.children = []; }},
          set innerHTML(v) {{ throw new Error('this element must not be given markup'); }},
          append(...kids) {{ kids.forEach(kid => node.children.push(kid)); }},
          attrs: {{}}, setAttribute(k, v) {{ node.attrs[k] = String(v); }},
        }};
        return node;
      }}
      function __anchor() {{
        const node = {{
          tag: 'a', href: '', _text: '', children: [], attrs: {{}}, className: '',
          get textContent() {{ return node._text; }},
          set textContent(v) {{ node._text = String(v); node.children = []; }},
          set innerHTML(v) {{ throw new Error('a link must not be given markup'); }},
          append(...kids) {{ kids.forEach(kid => node.children.push(kid)); }},
          setAttribute(k, v) {{ node.attrs[k] = String(v); }},
        }};
        return node;
      }}
      const WHO = __strict(), FOOT = __strict();
      global.document = {{
        getElementById: id => (id === 'who' ? WHO : id === 'foot' ? FOOT : el),
        querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {{}},
        createElement: tag => (tag === 'a' ? __anchor() : tag === 'span' ? __strict() : el),
        body: el,
      }};
      global.window = {{ matchMedia: () => ({{ matches: false, addEventListener: () => {{}} }}),
                        addEventListener: () => {{}}, innerWidth: 1200, scrollTo: () => {{}} }};
      global.location = {{ pathname: '/game', hash: '' }};
      global.fetch = path =>
        path === '/api/me' ? ({fetch_stub})() : new Promise(() => {{}});
    """
    runner = f"""
      const leaves = node => typeof node === 'string' ? [node]
        : node.children.length ? node.children.flatMap(leaves) : [node._text];
      const seen = node => node.tag !== 'a' ? node
        : node.children.length ? {{ text: node.attrs['aria-label'], href: node.href,
                                    shown: node.children.flatMap(leaves).join('|'),
                                    kind: node.className }}
        : {{ text: node._text, href: node.href }};
      identify().then(() => {{
        noteAdmin({admin});
        paintIdentity();
        console.log(JSON.stringify({{ who: WHO.children.map(seen),
                                      foot: FOOT.children.map(seen) }}));
      }});
    """
    done = node_session.run(stub + script.group(1) + "\n" + runner)
    assert done.returncode == 0, done.stderr
    return _json.loads(done.stdout)


class TestTheSiteDrawsOnlyWhatItServes:
    """A route the site withholds is neither a button nor a request there.

    The site answers a withheld route with a 404 *no such endpoint*, and the game page
    drew five buttons that met it (`docs/archive/plan-ux.md` A1, A3). `offered` reads the site's
    table once `/api/me` has said which server this is, `api` refuses a withheld route
    before it is sent, and the game page draws nothing that calls one. Locally every
    route is served, so every button is drawn.
    """

    HASH = "abc123def456"
    IDS = ("rate", "measureIt", "buildBot", "evolve", "promoteIt")
    WORDS = ("work them out again", "work out how hard", "build an opponent",
             "Let a model try", "taken again", "yardstick")

    def test_the_page_carries_the_sites_withheld_table(self) -> None:
        import re

        from page_text import PAGE

        from chess_variant_sandbox.site import routes

        listed = re.search(r"const WITHHELD = \[(.*?)\];", PAGE, re.S)
        assert listed is not None
        assert set(re.findall(r"'([A-Z]+ /[^']*)'", listed.group(1))) == {
            f"{route.method} {route.path}" for route in routes.WITHHELD
        }

    def test_the_site_draws_no_button_it_would_refuse(self) -> None:
        drawn = _page_on(True, f"#/game/{self.HASH}", _game_payload(self.HASH, site=True))
        assert drawn["errors"] == []
        html = drawn["html"]
        assert 'id="start"' in html and 'id="fork"' in html
        for name in self.IDS:
            assert f'id="{name}"' not in html, name
        for words in self.WORDS:
            assert words not in html, words
        assert 'id="keeping"' not in html and "How you do against them" not in html
        # The opponents nobody has rated are one sentence under the leaderboard (L2), with
        # no reason the site withholds, since nothing there can be done about it (G15).
        said = " ".join(html.split())
        assert ("<b>random</b>, <b>trade-up</b>, <b>own piece values</b> have no rating"
                " on this game yet, so they are not on it.") in said
        assert "not in the picker when this table" not in said

    def test_locally_every_button_is_still_drawn(self) -> None:
        drawn = _page_on(False, f"#/game/{self.HASH}", _game_payload(self.HASH, site=False))
        assert drawn["errors"] == []
        for name in self.IDS:
            assert f'id="{name}"' in drawn["html"], name
        assert "How you do against them" in drawn["html"]

    def test_the_site_sends_no_request_to_a_withheld_route(self) -> None:
        """On a game page and on landing.

        The game page does not even ask `api`. The library still asks for the survey,
        and `api` refuses it before it is sent, so on landing the network is what is
        checked. The local runs are the control: they send the anchor, the measurement
        poll and the survey, so a harness that could not see those fails here first.
        """
        game = f"GET /api/game/{self.HASH}"
        for site in (True, False):
            payload = _game_payload(self.HASH, site=site)
            for route in (f"#/game/{self.HASH}", ""):
                seen = _page_on(site, route, payload)
                asked = seen["asked"] if route else []
                refused = _withheld(seen["fetched"] + asked)
                if site:
                    assert refused == [], (route, refused)
                else:
                    assert refused, route
                if route:
                    assert seen["fetched"].count(game) == 1, seen["fetched"]
                else:
                    assert "GET /api/library" in seen["fetched"]


@pytest.fixture(scope="module")
def answers() -> dict:
    """What the designer is told on landing: the kit, a real check of the first template,
    and a standing that is saved and unmeasured, where measure and nominate are drawn."""
    from chess_variant_sandbox.web import candidacy, cards, designing

    kit = {"templates": designing.templates(), "vocabulary": designing.vocabulary(),
           "cards": cards.card_model(), "rungs": candidacy.rungs(), "nominations": []}
    spec = kit["templates"][0]["spec"]
    check = designing.check(spec)
    digest = check["hash"]
    return {
        "/api/design": kit,
        "/api/design/check": check,
        "/api/design/moves": designing.move_map(spec, 0),
        f"/api/design/standing/{digest}": {
            "hash": digest, "saved": True, "stage": "candidate",
            "measured": {"state": "unscored"}, "measuring": None, "nominated": False,
            "blocker": None, "build": "", "rungs": kit["rungs"], "lineage": None,
        },
        "/api/design/screen": {"hash": digest, "passed": True, "seconds": 0.1,
                               "reasons": []},
    }


class TestTheSiteDesignerDrawsOnlyWhatItServes:
    """The designer's withheld routes are neither a button nor a request on the site.

    Measure, nominate, an icon upload and the opponent builder's save each met the site's
    404. Where a button would have been, a sentence says why. Locally every one is drawn,
    and the local run is the control that shows the harness can see them.
    """

    HASH = "abc123def456"

    def _designer(self, site: bool, answers: dict) -> dict:
        return _page_on(site, "#/design", _game_payload(self.HASH, site=site), answers,
                        landed='data-rung="save" data-state="done"')

    def test_the_site_designer_draws_no_button_it_would_refuse(self, answers) -> None:
        site, local = self._designer(True, answers), self._designer(False, answers)
        assert site["errors"] == [] and local["errors"] == []
        for marker in ('id="dMeasure"', 'id="dNominate"', 'id="dNominateNote"', "data-upload="):
            assert marker not in site["every"], marker
            assert marker in local["every"], marker
        said, local_said = _said(site["every"]), _said(local["every"])
        for words in ("does not measure designed games itself yet",
                      "does not take your own drawings yet"):
            assert words in said and words not in local_said, words
        assert "Below, you can measure it" not in said and "Below, you can measure it" in local_said
        # D23 (Q563): the withheld sentence names measuring only. The site's shortlist has a
        # working *nominate it*, and the site is no place to point at another copy.
        assert "measure or nominate" not in said and "copy you run" not in said
        # What the site does serve is still offered.
        for kept in ('id="dSave"', 'id="dPlay"', 'id="dWatch"'):
            assert kept in site["every"], kept

    def test_the_site_designer_sends_no_request_to_a_withheld_route(self, answers) -> None:
        site = self._designer(True, answers)
        assert _withheld(site["fetched"] + site["asked"]) == []
        assert "POST /api/design/check" in site["fetched"]
        assert any(asked.startswith("GET /api/design/standing/") for asked in site["fetched"])

    def test_the_local_opponent_builder_offers_its_save(self, answers) -> None:
        seen = _page_on(False, f"#/opponent/{self.HASH}", _game_payload(self.HASH, site=False),
                        answers, landed='id="botSummary"')
        assert seen["errors"] == []
        assert 'id="botSave"' in seen["every"] and 'id="botPlay"' in seen["every"]

    def test_the_site_answers_the_opponent_builder_with_the_game(self, answers) -> None:
        """D47 (`archive/plan-understanding.md` §3.3): the site keeps no opponents, so a typed
        `#/opponent/<hash>` draws the game's page, never sliders it could not keep."""
        seen = _page_on(True, f"#/opponent/{self.HASH}", _game_payload(self.HASH, site=True),
                        answers)
        assert seen["errors"] == []
        assert 'id="start"' in seen["html"]
        for marker in ('id="botSummary"', 'id="botSave"', 'id="buildBot"'):
            assert marker not in seen["every"], marker
        assert _withheld(seen["fetched"] + seen["asked"]) == []


class TestTheRefusingSentencesSayWhatIs:
    """`docs/archive/plan-launch.md` §3.5 (L5, Q563): a game with no rated computer players is
    *practice only*, said in the words a player uses, never *sandbox mode* or a
    *leaderboard* it does not have (B17)."""

    def test_the_unrated_tag_says_practice_only(self) -> None:
        tag = _in_page("", "unbeatenTag({ rated: false })")
        assert ">unrated</span>" in tag
        assert "Practice only:" in tag
        for word in ("sandbox", "leaderboard"):
            assert word not in tag.lower(), word


class TestBusyIsSaidBeforeItRefuses:
    """`docs/archive/plan-launch.md` §5.2 (L11, Q566): when every strong seat is taken, the
    opponent list says *busy* beside the strongest player before anybody picks it; *hard*
    is the opponent a site game page offers first; and an answer slower than three seconds
    says the computer is busy with other games and the move is safe. No countdown, no
    queue place."""

    HASH = "abc123def456"
    HARDEST = "minimax:nodes=200000"

    def _select(self, site: bool, busy: dict | None = None) -> str:
        import re

        payload = _game_payload(self.HASH, site=site)
        if busy is not None:
            payload["standing"]["busy"] = busy
        html = _page_on(site, f"#/game/{self.HASH}", payload)["html"]
        found = re.search(r'<select id="seat1">(.*?)</select>', html, re.S)
        assert found is not None
        return " ".join(found.group(1).split())

    def test_the_strongest_reads_busy_only_when_the_server_says_so(self) -> None:
        import re

        said = "busy, back in a few minutes"
        free = self._select(True, {})
        assert said not in free
        busy = self._select(True, {self.HARDEST: said})
        option = re.search(rf'<option value="{self.HARDEST}"[^>]*>([^<]*)</option>', busy)
        assert option is not None and option.group(1).strip() == f"hardest \u00b7 {said}"
        assert busy.count(said) == 1

    def test_a_visitor_reads_sign_in_to_play_beside_what_needs_an_account(self) -> None:
        """L15 (`docs/archive/plan-launch.md` §3.6): the server's `signInFor` words go after the
        player's name, as *busy* does, and win over *busy* for a visitor."""
        import re

        payload = _game_payload(self.HASH, site=True)
        payload["standing"]["busy"] = {self.HARDEST: "busy, back in a few minutes"}
        payload["standing"]["signInFor"] = {self.HARDEST: "sign in to play"}
        html = _page_on(True, f"#/game/{self.HASH}", payload)["html"]
        found = re.search(r'<select id="seat1">(.*?)</select>', html, re.S)
        assert found is not None
        menu = " ".join(found.group(1).split())
        option = re.search(rf'<option value="{self.HARDEST}"[^>]*>([^<]*)</option>', menu)
        assert option is not None and option.group(1).strip() == "hardest \u00b7 sign in to play"
        assert "busy, back in a few minutes" not in menu
        hard = re.search(r'<option value="minimax:nodes=6400"[^>]*>([^<]*)</option>', menu)
        assert hard is not None and "sign in" not in hard.group(1)

    def test_the_site_offers_hard_first_and_the_local_application_medium(self) -> None:
        assert 'value="minimax:nodes=6400" selected' in self._select(True)
        assert 'value="minimax:nodes=1600" selected' in self._select(False)

    def test_a_slow_answer_says_the_computer_is_busy_and_nothing_grows(self) -> None:
        setup = """
          SESSION = { id: 's1', sideToMove: 1, seats: ['human', 'minimax:nodes=200000'],
                      ruleset: { sides: ['white', 'black'] } };
        """
        quick = _in_page(setup, "thinkingHtml()")
        slow = _in_page(setup + "AGENT_SLOW = 's1';", "thinkingHtml()")
        other = _in_page(setup + "AGENT_SLOW = 's0';", "thinkingHtml()")
        line = "The computer is busy with other games; your move is safe."
        assert line not in quick and line not in other
        assert line in slow
        assert _in_page("", "SLOW_AFTER") == 3000

    def test_the_slow_line_is_the_sites_alone(self) -> None:
        """The local application runs on its owner's computer, where there are no other
        people's games to be busy with."""
        from page_text import PAGE

        start = PAGE.index("function maybeAgent()")
        body = PAGE[start:PAGE.index("\n}\n", start)]
        assert "if (ACCOUNTS) setTimeout(" in body


class TestTheScorePanelInPlainWords:
    """One sentence per state, the reason behind the (i) (`docs/archive/plan-ux.md` A2).

    The stale panel printed *stale (replay): the core or the agents changed* and then
    *It has not been measured yet* about a game that had been measured.
    """

    HASH = "abc123def456"
    CODE = ("core", "agents", "replay", "analysis pass")

    @classmethod
    def _panel(cls, site: bool, **change: object) -> str:
        import re

        payload = {**_game_payload(cls.HASH, site=site), **change}
        html = _page_on(site, f"#/game/{cls.HASH}", payload)["html"]
        panel = html[html.index("How it scored"):]
        panel = panel[:panel.index('id="keeping"')] if 'id="keeping"' in panel else panel
        # Behind the (i) is where the maintainer's words are allowed, which is the
        # tooltip and the same text on the button for the popover a tap opens. Neither
        # is on screen until somebody asks for it.
        hidden = re.sub(r'(title|data-more)="[^"]*"', "", panel)
        return " ".join(hidden.split())

    def test_a_stale_score_is_one_plain_sentence(self) -> None:
        import re

        for site in (True, False):
            panel = self._panel(site)
            assert "This game's score is out of date" in panel
            for word in self.CODE:
                assert not re.search(rf"\b{word}\b", panel), (site, word)
            assert "not been measured" not in panel and "Nobody has measured" not in panel

    def test_only_a_game_never_measured_says_so(self) -> None:
        for site in (True, False):
            panel = self._panel(site, staleness=None)
            assert ("Nobody has measured this one yet" in panel) is not site
            assert ("No score yet." in panel) is site
            assert "out of date" not in panel

    def test_the_site_says_no_score_and_not_why(self) -> None:
        """G24 (`docs/archive/plan-understanding.md` §3, Q563): the site measures nothing itself
        yet, so a game with no score says *No score yet.* and nothing about a tournament
        it withholds. The local application still says what measuring plays."""
        site = self._panel(True, staleness=None)
        assert "tournament" not in site and "does not offer" not in site
        local = self._panel(False, staleness=None)
        assert "standard opponents playing" in local

    def test_the_reason_travels_in_the_tooltip(self) -> None:
        """On the local application, where the reason is the maintainer's to act on. The
        site's (i) says what out of date means and nothing of why (G20)."""
        html = _page_on(False, f"#/game/{self.HASH}",
                        _game_payload(self.HASH, site=False))["html"]
        assert "stale (replay): the core or the agents changed" in html
        site = _page_on(True, f"#/game/{self.HASH}", _game_payload(self.HASH, site=True))["html"]
        assert "stale (replay)" not in site and 'data-gloss="stale"' in site
        # The unrated players are named under the leaderboard, and nothing says why (G15).
        assert ("have no rating on this game yet, so they are not on it."
                in " ".join(site.split()))
        assert "does not offer yet" not in site

    def test_the_reason_can_be_read_without_a_hover(self) -> None:
        """A title is a hover and a phone has no hover, so the reason was on the site
        and unreadable there (`docs/archive/plan-ux.md` C3). The (i) carries it as well, and
        the popover a tap opens prints it under the glossary's own sentence."""
        html = _page_on(False, f"#/game/{self.HASH}",
                        _game_payload(self.HASH, site=False))["html"]
        assert 'data-more="stale (replay): the core or the agents changed"' in html
        shown = _in_page("", "glossBubble('stale', 'because the core changed')")
        assert "Out of date" in shown and "because the core changed" in shown
        # An (i) with nothing of its own to add says only what the glossary says.
        assert _in_page("", "glossBubble('stale', '')").count("<p>") == 1
        assert "data-more" not in _in_page("", "info('stale')")


class TestTheOpponentMenuIsTrimmed:
    """The rungs first, everything else under *more* (`docs/archive/plan-ux.md` A5)."""

    def test_the_first_eight_entries_are_you_and_the_seven_rungs(self) -> None:
        import re

        from chess_variant_sandbox.web import opponents

        roster = _game_payload("abc123def456", site=True)["opponents"]
        html = _in_page(
            f"const OPTIONS = {json.dumps(roster)};",
            "seatSelect('seat1', OPTIONS, 'human', null)",
        )
        values = re.findall(r'<option value="([^"]*)"', html)
        assert values[:8] == [agent for agent, _ in opponents.LADDER]
        assert len(values) == len(roster)
        labels = re.findall(r'<optgroup label="([^"]*)"', html)
        assert labels[0] == opponents.LADDER_GROUP == "computer players, weakest first"
        assert labels[1:] and all(label.startswith("more") for label in labels[1:])
        # "you" is a seat rather than a rung, so it sits above both groups.
        assert html.index('value="human"') < html.index("<optgroup")

    def test_the_learned_player_sits_under_more_and_holds_the_game_to_practice(self) -> None:
        """Q683. A practice-only player is offered under *more* by its name, and choosing
        it is what `practiceOnlyIn` finds; the line under the mode menu then says so."""
        import re

        from chess_variant_sandbox.web import opponents

        learned = opponents.Opponent(
            agent="minimax:net=aaaaaaaaaaaa,nodes=6400", label="x", group="that taught itself",
            name=opponents.LEARNED_NAME, store="learned", practice_only=True)
        roster = [entry.as_dict() for entry in [*opponents.ladder(), learned]]
        html = _in_page(
            f"ROSTER = {json.dumps(roster)};",
            "seatSelect('seat1', ROSTER, 'human', null)",
        )
        assert '<optgroup label="more — that taught itself">' in html
        assert re.search(r'value="minimax:net=aaaaaaaaaaaa,nodes=6400"\s*>'
                         r'the learned player \(experimental\)<', html)
        found = _in_page(
            f"ROSTER = {json.dumps(roster)};",
            "[practiceOnlyIn(['human', 'minimax:net=aaaaaaaaaaaa,nodes=6400']).name,"
            " practiceOnlyIn(['human', 'minimax:nodes=6400']),"
            " practiceOnlySays(ROSTER[ROSTER.length - 1])]",
        )
        assert found[0] == opponents.LEARNED_NAME and found[1] is None
        assert found[2].startswith("A game against the learned player (experimental) is "
                                   "always practice")

    def test_on_the_site_choosing_the_learned_player_starts_a_practice_game(self) -> None:
        """Q683. With *rated* chosen, picking the learned player sets the menu to practice
        and holds it there, the line under it says why, and the start sends practice;
        picking another player gives the menu back."""
        from chess_variant_sandbox.web import opponents

        learned = "minimax:net=aaaaaaaaaaaa,nodes=6400"
        payload = _game_payload("abc123def456", site=True)
        payload["opponents"] = [*payload["opponents"], opponents.Opponent(
            agent=learned, label="x", group="that taught itself", name=opponents.LEARNED_NAME,
            store="learned", practice_only=True).as_dict()]
        press = f"""
          const mode = document.getElementById('mode'), seat = document.getElementById('seat1');
          document.getElementById('seat0').value = 'human';
          mode.value = 'leaderboard'; mode.onchange();
          seat.value = {json.dumps(learned)}; seat.onchange();
          window.__held = [mode.value, mode.disabled, document.getElementById('modeSays')
            .textContent, document.getElementById('start').textContent];
          document.getElementById('start').onclick();
          seat.value = 'minimax:nodes=6400'; seat.onchange();
          window.__freed = [mode.disabled];
          document.getElementById('view')._html += '<i id="held">'
            + JSON.stringify([window.__held, window.__freed]) + '</i>';
        """
        report = _page_on(True, "#/game/abc123def456", payload, press=press)
        import re

        held, freed = json.loads(re.search(r'<i id="held">(.*?)</i>', report["html"]).group(1))
        assert held[0] == "sandbox" and held[1] is True
        assert held[2].startswith("A game against the learned player (experimental) is always")
        assert held[3] == "start a practice game against the learned player (experimental)"
        assert freed == [False]
        (body,) = [sent for path, sent in report["sent"] if path == "/api/session"]
        assert body["mode"] == "sandbox" and body["seats"] == ["human", learned]

    def test_the_hardest_rung_makes_no_time_claim(self) -> None:
        from chess_variant_sandbox.web import opponents

        hardest = dict(opponents.LADDER)["minimax:nodes=200000"]
        assert hardest.endswith("the slowest") and "s a move" not in hardest

    def test_each_rung_leads_with_its_word_then_the_seat_panels_plain_words(self) -> None:
        """F7, P2-P5. The name first, then what it does in the very words the seat panel
        prints under a seat (`play.strength`), so the picker and the board say one thing."""
        from chess_variant_sandbox.web import opponents
        from chess_variant_sandbox.web.play import strength

        for entry in opponents.ladder():
            assert entry.label.startswith(entry.name)
            if entry.agent == "human":
                continue
            assert entry.about.startswith(strength(entry.agent)), entry
            assert entry.label == f"{entry.name} · {entry.about}"
            for machine in ("searches", "lookahead", "playouts", "nodes", " - "):
                assert machine not in entry.label, (machine, entry.label)
        named = {entry.agent: entry.label for entry in opponents.ladder()}
        assert named["minimax:nodes=400"] == "easy · looks at 400 positions before each move"
        assert named["greedy"] == "greedy · takes what it can see, one turn ahead"
        assert named["mcts:nodes=1600"] == (
            "tree search · tries 1,600 random games before each move"
        )

    def test_a_player_nobody_names_is_the_computer(self) -> None:
        """P15, P16, S8: an agent string is never a name."""
        from chess_variant_sandbox.web import opponents

        assert opponents.name_of("minimax:nodes=1600") == "medium"
        assert opponents.name_of("minimax:h=deadbeefcafe,nodes=6400") == "the computer"
        said = _in_page("ROSTER = []; OPPONENTS = [];",
                        "[shortName('minimax:nodes=1600'), shortName('human')]")
        assert said == ["the computer", "you"]

    OPTIONS = [
        {"agent": "human", "label": "you", "name": "you", "about": "", "store": "ladder"},
        {"agent": "minimax:nodes=6400", "name": "hard", "store": "ladder",
         "label": "hard · looks at 6,400 positions before each move",
         "about": "looks at 6,400 positions before each move",
         "group": "computer players, weakest first"},
    ]

    def _options(self, setup: str, game: object) -> list[str]:
        html = _in_page(
            f"{setup} const OPTIONS = {json.dumps(self.OPTIONS)};",
            f"seatSelect('seat1', OPTIONS, 'human', {json.dumps({'kind': 'picker', 'standings': [{'name': 'minimax:nodes=6400', 'rating': 446}]})}, {json.dumps(game)})",
        )
        return [" ".join(text.split()) for text in re.findall(r">([^<]*)</option>", html)]

    def test_on_the_site_an_option_carries_the_one_scale_people_are_rated_on(self) -> None:
        """§2.3, P6: *hard · rated 1946 here*, never the tournament's signed number."""
        game = {"standing": {"siteRatings": {"minimax:nodes=6400": 1946.4}}}
        shown = self._options("ACCOUNTS = true;", game)
        assert shown == ["you", "hard · rated 1946 here"]

    def test_signed_in_an_option_says_how_far_it_sits_from_you(self) -> None:
        """The server's distance (`yourGaps`), worded: never *(-785)*."""
        for gap, words in ((240.4, "about 240 above you"), (-785.2, "about 785 below you"),
                           (0.2, "about equal to you")):
            game = {"standing": {"siteRatings": {"minimax:nodes=6400": 1946.0},
                                 "yourGaps": {"minimax:nodes=6400": gap}}}
            shown = self._options("ACCOUNTS = true;", game)
            assert shown[1] == f"hard · {words}"
            assert "(" not in shown[1] and "−" not in shown[1]

    def test_the_local_application_keeps_its_tournament_number(self) -> None:
        shown = self._options("ACCOUNTS = false;", None)
        assert shown == ["you", "hard (+446)"]

    def test_a_priced_opponent_says_what_it_does_in_a_players_words(self) -> None:
        """P9-P11. What a searched valuation beat, by the rung's name, never *the
        built-in objective* or *the derived table*."""
        from types import SimpleNamespace

        from chess_variant_sandbox.web import opponents

        valued = SimpleNamespace(piece_ids=("N", "B"), ratios=(1.2, 1.0),
                                 accept_score=0.78, accept_games=64)
        said = opponents._valued_plan(valued)
        assert said.startswith("Plays like ") and "values N at 1.20×" in said
        assert "78% of the points against" in said and "tuned by trial games" in said
        for machine in ("objective", "derived table", "hill-climbing", "budget"):
            assert machine not in said

    def test_the_plain_words_go_under_the_menu(self) -> None:
        """F7: the strength in plain words under the name, where a phone's menu does
        not cut it off."""
        said = _in_page(f"ROSTER = {json.dumps(self.OPTIONS)};",
                        "planFor('minimax:nodes=6400')")
        assert said == "Looks at 6,400 positions before each move."


class TestOneRatingScaleOnTheSite:
    """One computer player, one number on the site (`docs/reference/plain-words.md` §3.3; the
    audit's G11 and G14), and since L2 (`docs/archive/plan-leaderboard.md`) that number is a row of the game's
    leaderboard.

    The opponent table said *hardest +802* and the Unbeaten panel said *rated 2302* about
    the same player, because the table is the tournament's own scale centred on zero and
    the site's is that plus 1500, where people are rated. D2 of `docs/archive/plan-ux.md`
    kept both and labelled them; a visitor needs one, so the site shows the scale people
    are rated on, the one that moves, and the local application keeps the tournament's.
    What these hold is that no number appears without its scale, that the site never
    prints the tournament's number, and that an absent number is an absent row or a
    sentence rather than an invented one.

    The fixture is coherent on purpose: the top rung is +600 in the tournament and 2100 on
    the site, and the weakest rung is on no site table at all.
    """

    HASH = "abc123def456"
    COLUMN = 'data-gloss="ratedHere"'
    """The site's rating column's (i)."""

    @classmethod
    def _page(cls, site: bool, **change: object) -> str:
        """The game page with the maintainer's words in the tooltips stripped out."""
        import re

        payload = {**_game_payload(cls.HASH, site=site), **change}
        html = _page_on(site, f"#/game/{cls.HASH}", payload)["html"]
        return re.sub(r'title="[^"]*"', "", html)

    @staticmethod
    def _strength(html: str) -> str:
        start = html.index('id="leaderboard"' if 'id="leaderboard"' in html
                           else "How hard each opponent is")
        return html[start:html.index("How it scored")]

    @staticmethod
    def _unbeaten(html: str) -> str:
        return html[html.index(">Unbeaten"):html.index('id="voting"')]

    @staticmethod
    def _words(markup: str) -> str:
        """What a reader reads. The (i) buttons go first, so their own letter does not
        land in the middle of the sentence they sit in."""
        import re

        plain = re.sub(r'<button class="info".*?</button>', "", markup, flags=re.S)
        return " ".join(re.sub(r"<[^>]*>", " ", plain).split())

    @staticmethod
    def _rows(markup: str) -> list[str]:
        import re

        body = markup[markup.index("<tbody>"):markup.index("</tbody>")]
        return re.findall(r"<tr.*?</tr>", body, re.S)

    def test_the_site_has_one_column_and_one_sentence(self) -> None:
        table = self._strength(self._page(True))
        assert ">Leaderboard" in table and 'data-gloss="leaderboard"' in table
        assert self.COLUMN in table
        assert "in the tournament" not in table and 'data-gloss="ladderScale"' not in table
        said = self._words(table)
        assert ("Every computer player on this game, strongest first, on the scale people"
                " are rated on.") in said
        assert "+600" not in table

    def test_one_opponent_carries_its_site_number_alone(self) -> None:
        rows = self._rows(self._strength(self._page(True)))
        assert any("<b>2100</b>" in row for row in rows)
        assert not any("+600" in row for row in rows)

    def test_the_unbeaten_panel_says_the_same_one_number(self) -> None:
        html = self._page(True)
        said = self._words(self._unbeaten(html))
        assert "Its strongest computer player is rated 2100 here." in said
        assert "tournament" not in said and "+600" not in said

    def test_an_opponent_the_site_has_not_seeded_is_no_row(self) -> None:
        """Never a row with a blank or an invented number: it is named under the table."""
        table = self._strength(self._page(True))
        rows = self._rows(table)
        assert len(rows) == 6 and not any(">random<" in row for row in rows)
        assert all("<b>" in row for row in rows)
        assert "<b>random</b>, <b>trade-up</b>" in table

    def test_a_game_with_no_site_ratings_has_no_column_and_says_so(self) -> None:
        """§2.5: the names, weakest first, no numbers, and the sentence for a library game."""
        standing = {**_game_payload(self.HASH, site=True)["standing"], "rated": False,
                    "siteRatings": {}, "leaderboard": {
                        "rated": False, "rows": [], "people": 0, "unrated": [],
                        "waiting": ["random", "greedy", "easy"], "library": True}}
        table = self._strength(self._page(True, standing=standing))
        said = self._words(table)
        assert ("This game's computer players have not been rated yet, so every game here"
                " is practice for now. They are rated on our own machine; this usually takes"
                " a day.") in said
        assert "its computer players, weakest first random greedy easy" in said
        assert self.COLUMN not in table and "+600" not in table and "<b>" not in table

    def test_the_local_application_shows_one_scale_and_says_why(self) -> None:
        table = self._strength(self._page(False))
        assert ">in the tournament" in table
        assert "keeps no accounts and rates no people" in self._words(table)
        assert self.COLUMN not in table and "<b>2100</b>" not in table
        assert "<b>+600</b>" in table and "Leaderboard" not in table

    def test_an_absent_ladder_file_leaves_the_leaderboard_as_the_server_sent_it(
        self,
    ) -> None:
        """The leaderboard is the site's table, so a stale or missing tournament file on
        disk changes none of it."""
        html = self._page(True, pickerRatings=None, ratings=None)
        table = self._strength(html)
        assert "Nobody has been rated on this game yet" not in self._words(table)
        assert "in the tournament" not in table and "<b>2100</b>" in table
        said = self._words(self._unbeaten(html))
        assert "rated 2100 here" in said and "in the tournament" not in said


class TestFiveStars:
    """A4 of `docs/archive/plan-ux.md`. The sort key, the mean and the count existed.

    The stars did not. `POST /api/vote` has been served since task 1.5 and nothing ever
    called it, so the library offered *star rating* over a library nobody could vote in.
    What these hold is that a signed-in person can vote and sees their own vote back, that
    the mean and the count afterwards are the server's answer rather than any arithmetic
    of the page's, that a visitor is told how to join in instead of being given a control
    the server would refuse, and that the local application, which has no accounts, draws
    none of it.
    """

    HASH = "abc123def456"

    @classmethod
    def _drawn(cls, site: bool = True, vote: dict | None = None, **rest: object) -> dict:
        """The game page, with `vote` merged into the standing the server sent."""
        payload = _game_payload(cls.HASH, site=site)
        if vote is not None:
            payload["standing"] = {**payload["standing"], **vote}
        return _page_on(site, f"#/game/{cls.HASH}", payload, **rest)  # type: ignore[arg-type]

    @staticmethod
    def _panel(html: str) -> str:
        start = html.index('id="voting"')
        return html[start:html.index('id="leaderboard"', start)]

    @staticmethod
    def _row(html: str) -> str:
        start = html.index('id="stars"')
        return html[start:html.index("</div>", start)]

    def test_a_member_gets_five_stars_to_press(self) -> None:
        panel = self._panel(self._drawn()["html"])
        for n in range(1, 6):
            assert f'id="star{n}"' in panel
        assert 'aria-label="1 star"' in panel and 'aria-label="5 stars"' in panel
        assert "Press a star to rate it." in panel
        assert "Nobody has rated this one yet" not in panel

    def test_a_vote_already_cast_comes_back_with_the_mean(self) -> None:
        """What a member sees after voting and reloading: the server sends the row back."""
        panel = self._panel(self._drawn(
            vote={"yourVote": {"stars": 4}, "stars": 4.25, "votes": 3})["html"])
        assert "You gave it 4 out of 5." in panel
        assert "&#9733; 4.3 from 3 votes." in panel
        row = self._row(panel)
        assert row.count("&#9733;") == 4 and row.count("&#9734;") == 1
        assert row.count('aria-pressed="true"') == 4

    def test_a_visitor_is_told_how_to_join_in_rather_than_given_a_control(self) -> None:
        panel = self._panel(self._drawn(
            vote={"signedIn": False, "stars": 4.25, "votes": 3})["html"])
        assert 'id="star1"' not in panel
        assert "&#9733; 4.3 from 3 votes. Sign in to rate it yourself." in panel
        empty = self._panel(self._drawn(vote={"signedIn": False})["html"])
        assert "Nobody has rated this one yet. Sign in to be the first." in empty

    def test_the_local_application_draws_none_of_it(self) -> None:
        """No accounts, so nobody to be and no vote to record."""
        html = self._drawn(site=False)["html"]
        assert "What players make of it" not in html and 'id="star1"' not in html

    def test_pressing_a_star_sends_that_vote_and_nothing_else(self) -> None:
        """And the panel then reads the answer, which is two people's votes rather than
        the one just cast -- a page doing its own arithmetic would say 4.0 from 1."""
        drawn = self._drawn(
            answers={"/api/vote": {"hash": self.HASH, "stars": 4.5, "votes": 2}},
            press="document.getElementById('star4').onclick();",
        )
        assert drawn["sent"] == [["/api/vote", {"hash": self.HASH, "stars": 4, "tags": []}]]
        assert "You gave it 4 out of 5." in drawn["every"]
        assert "&#9733; 4.5 from 2 votes." in drawn["every"]

    TAGS: list[dict[str, str]] = [  # noqa: RUF012 - read only
        {"tag": "again", "words": "I'd play it again"},
        {"tag": "clever", "words": "a clever rule"},
        {"tag": "new", "words": "feels new"},
        {"tag": "fast", "words": "over too fast"},
    ]

    def test_the_tags_come_once_there_are_stars_and_there_is_nowhere_to_write(self) -> None:
        """Q599: up to three fixed tags, chosen by tapping, and no text field anywhere in
        the panel -- a vote holds no words of the voter's (the user, 2026-10-02)."""
        unvoted = self._panel(self._drawn(vote={"voteTags": self.TAGS, "mostTags": 3})["html"])
        assert 'id="voteTags"' not in unvoted
        voted = self._panel(self._drawn(vote={
            "voteTags": self.TAGS, "mostTags": 3,
            "yourVote": {"stars": 4, "tags": ["again", "clever", "new"]},
            "stars": 4.0, "votes": 1,
        })["html"])
        assert 'id="voteTags"' in voted and "up to 3" in voted
        for entry in self.TAGS:
            assert f'data-tag="{entry["tag"]}"' in voted
        # Three chosen: the fourth cannot be.
        assert voted.count('aria-pressed="true"') == 4 + 3
        assert re.search(r'data-tag="fast"\s+aria-pressed="false" disabled', voted)
        assert 'id="voteTag-again"' in voted
        for field in ("<textarea", 'type="text"', "<input", "contenteditable"):
            assert field not in voted, field

    def test_everybody_reads_the_counts_and_nobody_reads_who(self) -> None:
        counts = [{"tag": "again", "words": "I'd play it again", "count": 2}]
        visitor = self._panel(self._drawn(vote={
            "signedIn": False, "voteTags": self.TAGS, "tagCounts": counts,
            "stars": 4.0, "votes": 2,
        })["html"])
        assert 'id="tagCounts"' in visitor and "&middot; 2" in visitor
        assert 'id="voteTags"' not in visitor

    def test_pressing_a_tag_sends_the_stars_and_the_tags_and_nothing_else(self) -> None:
        drawn = self._drawn(
            vote={"voteTags": self.TAGS, "mostTags": 3,
                  "yourVote": {"stars": 4, "tags": ["again"]}, "stars": 4.0, "votes": 1},
            answers={"/api/vote": {"hash": self.HASH, "stars": 4.0, "votes": 1,
                                   "tags": ["again", "new"], "tagCounts": []}},
            press="document.getElementById('voteTag-new').onclick();",
        )
        assert drawn["sent"] == [
            ["/api/vote", {"hash": self.HASH, "stars": 4, "tags": ["again", "new"]}]
        ]

    def test_a_refused_vote_answers_beside_the_star_that_was_pressed(self) -> None:
        """Q264's rule, on this control: the sentence lands where the reader is looking."""
        from page_text import PAGE

        start = PAGE.index("function wireVoting(")
        assert "fail(error, button)" in PAGE[start:PAGE.index("function paintStars(")]

    def test_the_card_and_the_page_show_the_same_number(self) -> None:
        """Both read `stars` and `votes` off the same query (`site/browse.py::columns`)."""
        card, panel = _in_page("", "[peopleHtml({ stars: 4.25, votes: 3, humanGames: 0 }),"
                                   " voteWords({ stars: 4.25, votes: 3, signedIn: false })]")
        assert "&#9733; 4.3" in card and "3 votes" in card
        assert "&#9733; 4.3 from 3 votes." in panel

    def test_the_mean_is_printed_in_one_place(self) -> None:
        """It used to be a clause in the Unbeaten panel, which is about the ladder."""
        html = self._drawn(vote={"stars": 4.25, "votes": 3})["html"]
        assert html.count("from 3 votes") == 1

    def test_a_star_is_big_enough_to_press_with_a_thumb(self) -> None:
        row = self._row(self._panel(self._drawn()["html"]))
        assert "u-minw-40" in row and "u-minh-40" in row


class TestTheGamePagesThreeLines:
    """Box B1 of `docs/archive/plan-meta-game.md`, drawn beside the tag.

    *First played by*, *the highest-rated person here*, and everyone who has beaten the
    top with the date and the seat, in order. Each is a record about people rather than
    about the game, so what these hold is that every line names its rule, that a line with
    nothing to say is absent rather than blank (test 7 of `docs/reference/no-dark-patterns.md`,
    and `docs/archive/plan-meta-game.md` §2.1's *absent is not zero*), that the order and the rating are the server's rather
    than the page's, and that the local application, which keeps no accounts, draws none
    of them.
    """

    HASH = "abc123def456"
    WHEN = "2026-09-14T12:00:00"

    @classmethod
    def _drawn(cls, site: bool = True, **standing: object) -> str:
        payload = _game_payload(cls.HASH, site=site)
        if standing:
            payload["standing"] = {**payload["standing"], **standing}
        html = _page_on(site, f"#/game/{cls.HASH}", payload)["html"]
        end = html.index("What players make of it") if site else len(html)
        start = max(html.rfind(">Unbeaten", 0, end),
                    html.rfind(">First to beat the top", 0, end))
        return html[start:end] if start > 0 else html

    @staticmethod
    def _words(markup: str) -> str:
        """What a reader reads, with the (i) buttons taken out of the sentences.

        Bold is closed up rather than spaced, because a name in bold is read as part of
        the sentence it sits in and *ada , 14 Sep* is an artefact of the stripping.
        """
        import re

        plain = re.sub(r'<button class="info".*?</button>', "", markup, flags=re.S)
        # A name linked to its member's page (Q760) is read as the name, closed up too.
        plain = re.sub(r"</?a\b[^>]*>", "", plain)
        return " ".join(re.sub(r"<[^>]*>", " ", plain.replace("<b>", "").replace("</b>", "")).split())

    @classmethod
    def _cleared(cls, *clears: object) -> str:
        return cls._drawn(
            unbeaten=False,
            firstClear={"handle": "ada", "when": cls.WHEN, "seat": 0},
            clearedBy=list(clears),
        )

    # ------------------------------------------------------------- first played by

    def test_the_first_human_game_is_a_dated_line_with_its_rule(self) -> None:
        panel = self._drawn(firstPlayed={"handles": ["ada"], "when": self.WHEN})
        assert "First played by ada, 2026-09-14." in self._words(panel)
        assert 'data-gloss="firstPlayed"' in panel

    def test_a_game_nobody_has_played_says_nothing_about_who_played_it(self) -> None:
        assert "First played by" not in self._drawn()

    def test_a_game_between_two_people_names_both(self) -> None:
        panel = self._drawn(firstPlayed={"handles": ["ada", "grace"], "when": self.WHEN})
        assert "First played by ada and grace, 2026-09-14." in self._words(panel)

    # -------------------------------------------------------- highest-rated person

    def test_the_highest_rated_person_is_named_with_their_rating(self) -> None:
        panel = self._drawn(highest={"handles": ["ada"], "rating": 1712.4})
        assert "The highest-rated person here is ada, rated 1712 here." in self._words(panel)
        assert 'data-gloss="highestPerson"' in panel

    def test_everybody_level_at_the_top_holds_it_together(self) -> None:
        panel = self._drawn(highest={"handles": ["ada", "grace"], "rating": 1700})
        assert "The highest-rated people here are ada and grace" in self._words(panel)

    def test_a_table_with_nobody_established_shows_no_holder(self) -> None:
        """Nobody is behind and nobody is shown as last: the line is simply absent."""
        assert "highest-rated person" not in self._drawn()

    def test_the_holder_is_whoever_the_server_says(self) -> None:
        """The page never sorts ratings itself. `siteRatings` holds the computer players
        and no people at all, so a page working this out would have nothing to read."""
        from page_text import PAGE

        start = PAGE.index("function highestHtml(")
        assert "standing.highest" in PAGE[start:PAGE.index("function rungHtml(")]
        assert "siteRatings" not in PAGE[start:PAGE.index("function rungHtml(")]

    # --------------------------------------------------------------- the clears

    def test_everyone_who_has_beaten_the_top_is_listed_with_a_date_and_a_seat(
        self,
    ) -> None:
        panel = self._cleared(
            {"handle": "ada", "when": self.WHEN, "seat": 0},
            {"handle": "grace", "when": "2026-09-15T09:00:00", "seat": 1},
        )
        said = self._words(panel)
        assert "2 people have beaten it, in the order they did." in said
        assert "ada, 2026-09-14, playing first" in said
        assert "grace, 2026-09-15, playing second" in said
        assert 'data-gloss="clears"' in panel

    def test_the_list_is_drawn_in_the_order_the_server_sent(self) -> None:
        """A clear is decided in finish order, and the page keeps that order rather than
        making one of its own."""
        panel = self._cleared(
            {"handle": "grace", "when": "2026-09-15T09:00:00", "seat": 1},
            {"handle": "ada", "when": self.WHEN, "seat": 0},
        )
        said = self._words(panel)
        assert said.index("grace") < said.index("ada, 2026-09-14")

    def test_an_unbeaten_game_lists_nobody(self) -> None:
        panel = self._drawn()
        assert "have beaten it" not in panel and "<ol" not in panel

    def test_one_clear_draws_no_list_because_the_sentence_above_is_one(self) -> None:
        panel = self._cleared({"handle": "ada", "when": self.WHEN, "seat": 0})
        said = self._words(panel)
        assert "First beaten by ada on 2026-09-14, playing first." in said
        assert "have beaten it" not in said and "<ol" not in panel

    # ----------------------------------------------------------------- and locally

    def test_the_local_application_draws_none_of_them(self) -> None:
        """No accounts, so nobody to be first, nobody to be highest and nobody to clear."""
        html = self._drawn(site=False)
        for words in ("First played by", "highest-rated person", "have beaten it"):
            assert words not in html


def _badges_drawn(held: list[dict]) -> dict:
    """Run `wireBadges` over a DOM that keeps its children, and read back what it built.

    A stub that swallowed `append` would hide the whole of what is under test here: the
    lines are text nodes rather than markup, so what they say is in the tree and not in
    an `innerHTML` a substring check could read.
    """
    import json as _json
    import re

    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      function __node(tag) {
        const node = {
          tag: tag, className: '', _html: '', _text: '', children: [],
          dataset: {}, style: {}, attributes: {}, title: '',
          get innerHTML() { return node._html; },
          set innerHTML(v) { throw new Error('a badge must not be given markup'); },
          get textContent() { return node._text; },
          set textContent(v) { node._text = String(v); },
          append(...kids) { kids.forEach(kid => node.children.push(kid)); },
          appendChild(kid) { node.children.push(kid); return kid; },
          replaceChildren(...kids) { node.children = kids; },
          setAttribute(name, value) { node.attributes[name] = value; },
          addEventListener() {}, querySelector() { return null; },
          querySelectorAll() { return []; }, closest() { return null; },
          classList: { toggle() {}, add() {}, remove() {} },
        };
        return node;
      }
      const BOX = __node('div');
      global.document = {
        getElementById: id => (id === 'gameBadges' ? BOX : __node('div')),
        querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {},
        createElement: tag => __node(tag), createTextNode: text => ({ text: text }),
        body: __node('body'),
      };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
      global.location = { pathname: '/game', hash: '' };
      global.fetch = () => new Promise(() => {});
    """
    runner = f"""
      // What a reader reads, with the (i) buttons left out of the sentence, as
      // `TestTheGamePagesThreeLines._words` takes them out of the markup.
      const __said = kid => kid.text !== undefined ? kid.text
        : kid.className === 'info' ? ''
        : (kid.children.length ? kid.children.map(__said).join('') : kid._text);
      wireBadges({{ standing: {{ badges: {_json.dumps(held)} }} }});
      console.log(JSON.stringify({{
        lines: BOX.children.map(__said),
        marks: BOX.children.flatMap(line => line.children
          .filter(kid => kid.className === 'info')
          .map(kid => ({{ gloss: kid.dataset.gloss, more: kid.dataset.more }}))),
      }}));
    """
    done = node_session.run(stub + script.group(1) + "\n" + runner)
    assert done.returncode == 0, done.stderr
    return _json.loads(done.stdout)


class TestTheGamePagesBadges:
    """Box E4 of `docs/archive/plan-meta-game.md`. A badge on this game, under B1's lines.

    Everything a line says comes from the server -- the name, the sentence and the rule
    out of `site/badges.BADGES`, the figures out of the row -- so the page works nothing
    out. The rule goes behind the (i), which is how every other term here explains
    itself, and the lines are text nodes rather than markup.
    """

    CROWD = {
        "kind": "crowd-pleaser", "tier": None, "name": "Crowd pleaser",
        "sentence": "People chose this game and rated it highly.",
        "rule": "Five different people have finished a game on it that counts.",
        "quoted": "4.4 stars from 9 people and 12 people who had played it",
        "when": "2026-10-03T12:00:00+00:00", "game": "abc123def456",
    }

    def test_a_crowd_pleaser_reads_as_a_dated_quotation(self) -> None:
        drawn = _badges_drawn([self.CROWD])
        assert drawn["lines"] == [
            "Crowd pleaser, 2026-10-03. People chose this game and rated it highly."
            " On that day, 4.4 stars from 9 people and 12 people who had played it."
        ]

    def test_the_rule_sentence_is_behind_the_info_button(self) -> None:
        """§3's seventh test, in the idiom this page explains every term in."""
        drawn = _badges_drawn([self.CROWD])
        assert drawn["marks"] == [{"gloss": "badge", "more": self.CROWD["rule"]}]

    def test_a_game_with_no_badge_draws_nothing_at_all(self) -> None:
        """No heading, no empty slot, no greyed list (§3, test 5)."""
        assert _badges_drawn([])["lines"] == []

    def test_nothing_is_given_markup(self) -> None:
        """The DOM above throws on `innerHTML`, so this passing is the claim: every
        line is built out of text nodes."""
        hostile = {**self.CROWD, "name": "<img src=x onerror=alert(1)>",
                   "sentence": "\"'><b>", "rule": "<script>", "quoted": "<i>"}
        drawn = _badges_drawn([hostile])
        assert drawn["lines"][0].startswith("<img src=x onerror=alert(1)>")

    def test_the_panel_holds_a_box_for_them_whether_or_not_any_are_held(self) -> None:
        from page_text import PAGE

        assert PAGE.count('id="gameBadges"') == 1
        assert "wireBadges(game);" in PAGE

    def test_a_score_badge_sits_above_a_panel_that_may_read_out_of_date(self) -> None:
        """Both statements are true. The badge quotes what the index said on the day it
        was earned; the panel reports what can be read now, and `_game_payload`'s
        measurement is stale."""
        html = TestTheGamePagesThreeLines._drawn()
        assert 'id="gameBadges"' in html
        assert "Out of date" in _page_on(
            True, f"#/game/{TestTheGamePagesThreeLines.HASH}",
            _game_payload(TestTheGamePagesThreeLines.HASH, site=True),
        )["html"]


def _withheld(requests: list[str]) -> list[str]:
    """The requests among these that name a route the site withholds."""
    from chess_variant_sandbox.site import routes

    return [
        asked for asked in requests
        if any(
            route.matches(asked.split(" ")[0], routes.split(asked.split(" ")[1].split("?")[0]))
            for route in routes.WITHHELD
        )
    ]


def _game_payload(digest: str, site: bool) -> dict:
    """One game as `/api/game` answers it: a stale score, a picker table, two unrated."""
    from chess_variant_sandbox.web import opponents

    roster = [entry.as_dict() for entry in opponents.ladder()] + [
        {"agent": "minimax:h=aaaaaaaaaaaa,nodes=6400", "label": "trade-up",
         "group": "plans, written by hand", "plan": "trade pieces", "store": "written"},
        {"agent": "minimax:values=bbbbbbbbbbbb,nodes=6400", "label": "own piece values",
         "group": "with this game's own piece values", "plan": "prices the pieces",
         "store": "valuations"},
    ]
    rated = [entry["agent"] for entry in roster[1:-2]]
    payload = {
        "hash": digest, "name": "a test game", "description": "for the page", "board": "8x8",
        "pieceTypes": 2, "ruleMass": 10, "axes": [], "pieces": [], "ruleLines": [],
        "setupLines": [], "verdict": None, "provenance": {"origin": "designed"},
        "sides": ["white", "black"], "armies": [[], []], "opponents": roster,
        "fairyCheck": {"engine": None}, "fairyTranslates": False, "fairyReasons": [],
        "playable": True, "score": None, "scoreGuide": None, "build": "",
        "staleness": {"current": False, "needs": "replay",
                      "summary": "stale (replay): the core or the agents changed"},
        "keeping": {"stage": "archived", "roster": ["random"], "gamesPerPairing": 4},
        "ratings": None,
        "pickerRatings": {"games": 40, "standings": [
            {"name": agent, "rating": 100 * index, "games": 20}
            for index, agent in enumerate(rated)
        ]},
    }
    if site:
        payload["standing"] = {
            "rated": True, "top": [rated[-1]], "topRating": 2100, "unbeaten": True,
            "clears": 0, "votes": 0, "signedIn": True, "admin": False,
            "yourRecord": {"state": "untried"},
            # The site's scale is the tournament's plus 1500 (`site/rating.py`). The
            # weakest rung is left off, because a player the site has not seeded is what
            # the table's empty cell is for.
            "siteRatings": {
                agent: 100 * index + 1500.0
                for index, agent in enumerate(rated) if index
            },
            # L2: the same table as the leaderboard draws it (`site/leaderboard.py`),
            # strongest first, nobody yet; the weakest rung and the two stored players
            # are offered with no row.
            "leaderboard": {
                "rated": True, "people": 0, "library": True, "waiting": [],
                "rows": [
                    {"name": roster[index + 1]["name"], "computer": True,
                     "rating": 100 * index + 1500, "settling": False, "games": 0,
                     "top": index == len(rated) - 1, "gone": False, "you": False}
                    for index in reversed(range(1, len(rated)))
                ],
                "unrated": [roster[1]["name"], "trade-up", "own piece values"],
            },
        }
    return payload


def _page_on(
    accounts: bool,
    route: str,
    game: dict,
    answers: dict | None = None,
    landed: str = 'id="(start|newGame)"',
    press: str = "",
) -> dict:
    """Boot the whole page at `route` in node, on the site or locally, and report.

    `fetch` answers `/api/me`, the game, the listing and any `answers` by path, query
    ignored, and 404s everything else the way the site does. An element exists only once
    some drawn markup names its id, so a button that was not drawn cannot be wired either.
    The run ends once `landed` matches the drawn markup. Returns the view's markup, all
    markup drawn into any element (`every`), every request that reached `fetch`, every
    POST body (`sent`), every call made to `api`, and every error `fail` showed.

    `press` is JavaScript run once the view has landed and before the report, for a
    control whose answer redraws part of the page.
    """
    import json as _json
    import re

    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    answers = {
        "/api/me": {"signedIn": True, "handle": "ada", "admin": False, "accounts": accounts},
        f"/api/game/{game['hash']}": game,
        "/api/library": {"games": [], "opponents": game["opponents"],
                         "coverage": {"axes": []}, "metrics": None,
                         **({"site": {"signedIn": True, "admin": False, "sorts": [],
                                      "filters": []}} if accounts else {})},
        **(answers or {}),
    }
    stub = f"""
      const __fetched = [], __sent = [], __errors = [],
            __LANDED = new RegExp({_json.dumps(landed)});
      function __el(id) {{
        const node = {{
          id: id, _html: '', value: '', style: {{}}, dataset: {{}}, hidden: false,
          textContent: '', className: '',
          get innerHTML() {{ return node._html; }},
          set innerHTML(v) {{ node._html = String(v); }},
          append() {{}}, remove() {{}}, addEventListener() {{}},
          // `fail` shows a box beside the control that raised, or, with no control to
          // sit beside, a fixed one on the body. Both are read here, so a route that
          // fails with nothing on screen to attach to is still visible to a test.
          appendChild(child) {{
            if (String(child.className).includes('error')) __errors.push(child.textContent);
          }},
          setAttribute() {{}}, replaceChildren() {{}},
          querySelector(selector) {{
            const named = /^#([\\w-]+)$/.exec(selector);
            return named && node._html.includes('id="' + named[1] + '"')
              ? __byId(named[1]) : null;
          }},
          querySelectorAll() {{ return []; }}, closest() {{ return null; }},
          classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
          prepend(child) {{ if (child.className === 'error') __errors.push(child.textContent); }},
        }};
        return node;
      }}
      const VIEW = __el('view'), __ELEMENTS = {{ view: VIEW }};
      const __every = () => Object.values(__ELEMENTS).map(node => node._html).join('\\n');
      function __byId(id) {{
        return __ELEMENTS[id]
          || (['who', 'foot', 'tagline', 'home'].includes(id)
              || __every().includes('id="' + id + '"') ? (__ELEMENTS[id] = __el(id)) : null);
      }}
      global.document = {{
        getElementById: __byId,
        querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {{}},
        createElement: () => __el(''), createTextNode: () => __el('#text'),
        body: __el('body'),
      }};
      global.window = {{ matchMedia: () => ({{ matches: false, addEventListener: () => {{}} }}),
                        addEventListener: () => {{}}, innerWidth: 1200, scrollY: 0,
                        __scrolled: [],
                        scrollTo: (x, y) => window.__scrolled.push([x, y]) }};
      global.location = {{ pathname: '/game', hash: {_json.dumps(route)} }};
      const __answers = {_json.dumps(answers)};
      global.fetch = (path, options) => {{
        __fetched.push(((options && options.method) || 'GET') + ' ' + path);
        if (options && options.body) __sent.push([path, JSON.parse(options.body)]);
        const key = path.split('?')[0];
        const known = Object.prototype.hasOwnProperty.call(__answers, key);
        return Promise.resolve({{ ok: known, status: known ? 200 : 404, statusText: 'Not Found',
          json: () => Promise.resolve(known ? __answers[key]
                                            : {{ error: 'no such endpoint: ' + path }}) }});
      }};
    """
    runner = """
      const __asked = [], __api = api;
      api = (path, body) => {
        __asked.push((body === undefined ? 'GET ' : 'POST ') + path);
        return __api(path, body);
      };
      const __landed = () => __LANDED.test(__every()) || __errors.length;
      (function wait(tries) {
        if (!__landed() && tries < 300) return setTimeout(() => wait(tries + 1), 10);
        __PRESS();
        setTimeout(() => console.log(JSON.stringify({
          html: VIEW._html, every: __every(), fetched: __fetched, sent: __sent,
          asked: __asked, errors: __errors, scrolled: window.__scrolled,
        })), 50);
      })(0);
    """
    runner = f"const __PRESS = () => {{ {press} }};\n" + runner
    done = node_session.run(stub + script.group(1) + "\n" + runner)
    assert done.returncode == 0, done.stderr
    return _json.loads(done.stdout)


class TestAFaultInThePageIsSaidPlainly:
    """Q682: a JavaScript error thrown inside a success handler on the game page reaches
    `fail` like a refusal. It used to be shown as the engine's message (*Cannot read
    properties of undefined*); now the reader gets a plain sentence and the console the
    detail, while a sentence somebody wrote is still shown as it is."""

    HASH = "abc123def456"

    def test_a_fault_in_a_success_handler_shows_a_sentence(self) -> None:
        press = ("api('/api/game/" + self.HASH + "').then(game => game.nothing.here)"
                 ".catch(fail);"
                 " Promise.reject(new Error('That move is not legal here.')).catch(fail);")
        for site in (True, False):
            seen = _page_on(site, f"#/game/{self.HASH}", _game_payload(self.HASH, site=site),
                            press=press)
            said = " ".join(seen["errors"])
            assert "Cannot read" not in said and "TypeError" not in said, said
            assert "Something went wrong on this page." in said, said
            assert "That move is not legal here." in said, said

    def test_only_an_error_the_browser_raised_is_reworded(self) -> None:
        shown = json.loads(_in_the_page("", "JSON.stringify(["
            "errorWords(new Error('The server said no.')),"
            "errorWords(new RangeError('Invalid array length')),"
            "errorWords('a string'), PAGE_FAULT])"))
        assert shown[0] == "The server said no."
        assert shown[1] == shown[3] and "Invalid" not in shown[1]
        assert shown[2] == "a string"


class TestWhatReachesThePage:
    """§6 and task 1.10: every string the page did not write reaches it escaped or as text.

    A handle, a game's name and description, a spec field typed into the designer, an
    error that echoes input, an icon's `data:` URI. The page builds its views as markup
    strings, so the defence is `esc`, and `esc` has to be safe inside a quoted attribute
    as well as between tags.
    """

    # Quotes to leave an attribute, an angle bracket to leave text, and a tag that runs.
    HOSTILE = "\"'><img src=x onerror=alert(1)>"

    def test_every_markup_assignment_is_counted(self) -> None:
        """A new `innerHTML`, `outerHTML` or `insertAdjacentHTML` fails here until
        somebody has checked that what it assigns is escaped, and moved the number."""
        import re

        from page_text import PAGE

        sites = re.findall(
            r"\.(?:innerHTML|outerHTML)\s*=(?!=)|insertAdjacentHTML\(|document\.write\(",
            PAGE,
        )
        # 77 since Q456: B5's panel in the designer, measuring on this computer, redraws
        # itself in place. Every name, hash, sentence and error it interpolates goes through
        # `esc`; the rest is counts, and numbers printed through `Number(...)`.
        #
        # 78 since Q473: the map's local view writes its bar over the drawing -- the
        # game's name through `esc`, the rest counts and constants.
        #
        # 79 since Q489: the designer redraws *which game this is* with its rungs, so a
        # saved draft's row reads the fresh measurement. `identityHtml` again, whose
        # names and hashes go through `esc`.
        #
        # 81 since Q503a: the designer's *Publish* button writes what the press said
        # beside it, a success or a refusal, each sentence through `esc`.
        #
        # 82 since Q503d: the game page's *report this game* line redraws itself after
        # the press, the reason's label through `esc` and the rest a constant.
        #
        # 83 since Q503b: the library's tabs redraw themselves when one is pressed, each
        # label through `esc` and each count through `Number(...)`.
        #
        # 84 since Q503f: *Nominate* redraws the game page's *Where it stands* from the
        # server's answer (`whereHtml`), every word and sentence through `esc`.
        #
        # 87 since Q503h: the round page (`round.js`) writes its checkout sentence, a
        # constant, and the round itself, every name, handle and reason through `esc`;
        # the game page's yes and no redraw themselves after a vote (`roundHtml`).
        #
        # 89 since Q518: the designer's track answers its own *nominate it* beside the
        # button, the server's sentence or the refusal, each through `esc`.
        #
        # 91 since Q577: the play frame's status strip carries the *in check* pill and a
        # note under the line, so it is written as markup in both of `drawStatus`'s
        # branches; every side, seat and note in it goes through `esc`.
        #
        # 92 since Q570: a game between two people states its rules under the moves
        # (`betweenRulesHtml`), each sentence the server's through `esc`.
        #
        # 97 since Q571: *Play a person* on a game's page redraws itself after a tick
        # (`playPersonHtml`) and writes its challenge sheet, the sheet's answer and its
        # emptying; the landing page's *A challenge for you* redraws after a decline
        # (`forYouHtml`). Every handle, name and sentence goes through `esc`.
        #
        # 105 since Q731: *Play a person* redraws its *a friend who isn't a member yet*
        # box after an invitation is sent, the server's two sentences through `esc`.
        #
        # 104 since Q718: the landing page's opening board writes each game's still board
        # (`thumbHtml`, the cards' own) and its caption, the name and the server's sentence
        # through `esc`.
        #
        # 101 since Q587 (V4): the play frame writes the square names outside the board
        # (`drawCoords`: each a square's own name through `esc`, placed by numbers) and
        # the ended banner over it (`drawEnded`: the verdict's two halves and the board's
        # own rematch word through `esc`, the rest constants), and empties each.
        #
        # 76 since Q396: the optional description hint joins the save's message under
        # the save button, and every sentence it quotes goes through `esc`.
        #
        # 75 since Q363: `#/map` draws the explorer, so the old map's shell and the
        # write of its body went with `showMap`.
        #
        # 77 since Q359: the explorer writes its legend, frame and list of games under
        # its shell, where every name goes through `esc` and the rest is the legend the
        # map already draws.
        #
        # 76 since Q358: the explorer writes its own shell and, where the vendored
        # library does not load, the sentence that says so. Both are constants with
        # nothing interpolated into them.
        #
        # 111 since Q769: the chooser of where a strike's survivor lands left the hand
        # (one site) for the promotion's card, which writes it through the same three.
        # Its square names (`landingName`) and the struck piece's word go through `esc`.
        #
        # 112 since Q765: the choice of what a promoting piece becomes moved from the hand
        # (one site) to its own card over the board (three: the card, the bar shown while
        # looking at the board, and emptying it). Each piece's name and the mover's word go
        # through `esc`, each drawing through `drawPiece`, which escapes its href and letter.
        #
        # 110 since Q742: the measuring control's place under the rungs is redrawn when it
        # moves into or out of *Publish* (`hereAwayHtml`, the page's own markup), and a
        # press of *Publish* writes the server's sentence again after the panel is redrawn,
        # through `esc`.
        #
        # 108 since Q741: the landing page's *Made by members* shelf, each game's name,
        # maker and hash through `esc`, the maker's mark the DOM-built `markHtml`.
        #
        # 107 since Q733: the slot under a board between two people (`drawCorrSend`),
        # emptied once the game is over and otherwise its words and buttons, the other
        # player's handle and the held move's words through `esc`.
        #
        # 102 since Q619 (K3): a tapped piece's sentences under the board
        # (`drawPieceSays`), every name and sentence through `esc`.
        #
        # 74 before that, since Q307's stage 5: the designer's own play panel writes a spinner while a
        # game is starting and the refusal plus the invitation when one cannot, the
        # refusal through `esc` and the rest a constant. Before it, Q306's *and now try*
        # panel, Q305's library palette and rule menu, Q304's workbench board column,
        # Q345's changelog shell and Q342's map. Every name, sentence and hash any of
        # them interpolates goes through `esc`; the rest is counts and coordinates the
        # server computed.
        # 114 since Q788 (counted by Q728): the landing page's *New in the library* shelf and
        # its picks row write each game's thumbnail and score pill (`thumbHtml`, `scoreTag`)
        # and the shelf's cards, every name, hash and sentence through `esc`.
        # 118 since Q772b: the portal exit chooser's bar (`drawExits`), and three layers laid
        # over the board without redrawing it (the pair link, the lit exit, the board's
        # tier), every square name and sentence through `esc`.
        # 124 since Q772c: the designer's portals (`portals.js`): the pairs' cards redrawn in
        # place, a check's sentences, drawings and tray row put on them (five), and the
        # keyboard's ring over the board in *play it here* (one, a constant shape). Every
        # square name, side name and server sentence through `esc`.
        # 125 since Q836: the swirl a piece passes through a portal under (`portalSwirl`),
        # two circles of numbers and a colour token, nothing from outside the page.
        # 126 since Q830b: what became of a rule file (`ruleFileSaid`), its name, the
        # validator's lines and the draft's name each through `esc`.
        # 125 since Q838d: the pair card's *how they look* row is gone, and the check no
        # longer redraws it in place (`pairLooksHtml`).
        # 126 since Q843d: the ripple a piece passes through a fold under (`foldRipple`), four
        # circles of numbers and colour tokens, nothing from outside the page.
        # 127 since Q851: the tray's line under the board redrawn with the tray
        # (`paletteNoteBodyHtml`), the piece's name and side through `esc`.
        # 130 since Q918c: a game in rounds writes its strip (`roundStatusHtml`) and the
        # slot under the board twice (`drawRoundSend`), every side, seat, move text and
        # server sentence through `esc`.
        # 132 since Q935c: a spy game's slot while it is played by turns (`drawSpySend`),
        # emptied once the game is over and otherwise the phase in words, each side's name
        # through `esc`.
        # 135 since Q605 (R9b): *Proven positions* draws the picked position in place
        # (`wireProven`): a constant *Drawing the position…*, `provenAtHtml`'s board, moves
        # and sentences (every move text and server sentence through `esc`), and emptied on
        # a refusal.
        #
        # 137 since Q943b: an ordered round's reveal draws a lost move's arrow (numbers only,
        # `lostArrowHtml`) and an unplayed move's struck ghost (`drawPiece` and numbers).
        # 138 since Q638: *full screen* turns into *leave full screen* and back
        # (`setBoardFull`), its icon and its constant words (`fullWords`).
        # 139 since Q995b: how the players take turns redrawn in place (`drawTurns`), its
        # three constant ways and the server's sentences of what changed, each through
        # `esc`.
        # 143 since Q993b: the proven positions' own page (`proven.js`) replaces the game
        # page's three (`wireProven`) with seven: the view drawn and emptied before its wait,
        # the local application's constant sentence, the list redrawn (`provenListHtml`) and
        # the frame drawn, emptied, or a constant *Drawing the position…* (`provenFrameHtml`).
        # Every name, side, move text and server sentence through `esc`; the rest numbers.
        # 147 since Q995c: *find a rule* redraws the menu's options (`ruleOptionsHtml`) and
        # the line beside it (`ruleFoundHtml`) as the box is typed in, the chosen rule's
        # sentence and count (`ruleChosenHtml`), and a save under a borrowed name asks for
        # one (`nameAskHtml`). Every title, sentence and name through `esc`; the rest numbers.
        # 149 since Q995d: *every piece rests* redrawn in place (`drawRestAll`): its menu
        # (`restAllHtml`, constant words from `spec.plain` and numbers) and the sentence
        # under it (`restSaidHtml`), every piece name and server sentence through `esc`.
        # 151 since Q999c: what a piece sees laid over the board (`paintSight`, `sightHtml`:
        # numbers, squares and constant tokens), and the designer's refusals under a piece's
        # lines of sight (`sightProblemsHtml`, each server sentence through `esc`).
        # 155 since Q557: the piece palette under a piece (`drawingsPanelHtml`), its chips
        # (`drawingKindsHtml`) and its list (`drawingListHtml`) redrawn in place as it opens,
        # is searched and has a chip pressed. Every name, slug, sentence and picture's data
        # URI through `esc`; the rest counts and constants.
        # 157 since PF: the ready-made pieces' chips (`readyKindsHtml`) and list
        # (`readyListHtml`) redrawn in place (`redrawReady`). Every name, slug, reading and
        # picture's data URI through `esc`; the rest counts and constants.
        # 153 since PF4: the drawing palette, the ready pieces' panel and the library games'
        # panel are one picker. Seven sites become three: the picker drawn where it stands
        # (`redrawPicker`), and its chips and list redrawn in place (`redrawPickList`). Every
        # name, slug, sentence and picture's data URI through `esc`; the line under a tile is
        # built as text nodes (`placePickAbout`).
        # 155 since N4c: the duck's strip in the hand (`duckTrayHtml`: constant words, the
        # pass's move number and the duck's drawing) and the duck card's words redrawn after a
        # check (`drawDuckSays`, each server sentence through `esc`).
        # 161 since LP4: the landing's hero boards and the story's boards (`heroStill`,
        # `drawStoryFrame`: the still board's own markup), the hand under a board and its
        # swap in the pinned tray (`storyTray`: piece and side names through `esc`, counts),
        # the ladder and its people's-scale sentences (names and sentences through `esc`),
        # and the blast's fireball (`blastFire`: numbers and token names), with the board's
        # ghosts and swirls moved from `board.js` to `motion.js` unchanged.
        assert len(sites) == 161

    def test_esc_is_safe_inside_a_quoted_attribute(self) -> None:
        escaped = _in_page("", "esc(`\"'<>&`) + '|' + esc(null) + '|' + esc(0)")
        assert escaped == "&quot;&#39;&lt;&gt;&amp;||0"

    def test_the_page_has_one_escaper(self) -> None:
        """The designer grew its own attribute escaper when `esc` ended attributes early.
        With `esc` safe everywhere, a second one is a place to forget the first."""
        from page_text import PAGE

        assert PAGE.count("const esc = ") == 1
        assert "function cattr(" not in PAGE

    def test_a_hostile_string_never_reaches_the_markup_raw(self) -> None:
        """Every view's builders, fed an API payload whose every string is hostile.

        The escaped form of the payload is `&quot;&#39;&gt;&lt;img ...`. A raw `<img`
        means text was not escaped, and the raw quote pair means an attribute can be
        closed early -- which is what an `esc` that left quotes alone let through.
        """
        drawn = _in_page(_HOSTILE_SETUP.replace("__HOSTILE__", json.dumps(self.HOSTILE)),
                         _HOSTILE_VIEWS)
        assert set(drawn) >= {"card", "standing", "designerPiece", "cardField"}
        for name, html in drawn.items():
            assert html, f"{name} drew nothing, so it tested nothing"
            assert self.HOSTILE not in html, name
            assert "<img src=x" not in html, name
            assert "\"'" not in html, name


_HOSTILE_SETUP = """
const H = __HOSTILE__;
const ICON = { href: 'data:image/svg+xml,' + H, rotate: false };
const game = {
  hash: H, name: H, description: H, board: H, source: H, pieceTypes: 2, descent: 1,
  cellBins: [H], family: { hash: H, name: H }, index: null, designIndex: null,
  staleness: { current: false, summary: H }, playable: false, playableReasons: [H],
  fairyTranslates: false, fairyReasons: [H], rated: true, unbeaten: false, clears: 2,
  exhibit: H,
  firstClear: { handle: H, when: '2026-09-01', seat: 0 }, sides: [H, H],
  armies: [[{ name: H, id: H, count: 1 }], []],
  standing: { rated: true, top: [H], unbeaten: false, clears: 2, votes: 0,
              firstClear: { handle: H, when: '2026-09-01', seat: 1 }, yourRecord: null,
              firstPlayed: { handles: [H], when: '2026-09-01' },
              highest: { handles: [H], rating: 1700 },
              clearedBy: [{ handle: H, when: '2026-09-01', seat: 1 },
                          { handle: H, when: '2026-09-02', seat: 0 }] },
};
const piece = { name: H, id: H, moves: [H], notes: [H], royal: true, icons: [ICON] };
ROSTER = [{ agent: H, label: H, group: H, plan: H }];
COVERAGE = { axes: [{ axis: H, asks: H, bins: [{ bin: H, means: H }] }] };
BROWSE.text = H;
SESSION = { seats: [H, H], ruleset: { icons: [[ICON]], pieceIds: [H], nTypes: 1,
            sides: [H, H] } };
DRAFT = { name: H, description: H, board: { width: H, height: H },
  pieces: [{ id: H, name: H, value: H,
             moves: [{ kind: 'leap', vectors: [[1, 2]], range: H, min_range: H,
                       condition: { from_rank: 1 } }] }],
  setup: { placements: [{ piece: H, squares: [H] }] } };
DRAFT_CHECK = { preview: { icons: [[ICON]], sides: [H, H] } };
DRAFT_PARENT = { name: H, hash: H, index: null, staleness: { current: false, needs: H } };
"""

_HOSTILE_VIEWS = """({
  card: cardHtml(game),
  scoreTag: scoreTag(game),
  playableTag: playableTag(game),
  unbeatenTag: unbeatenTag(game),
  controls: controlsHtml(),
  scoring: scoringHtml({ games: [{ name: H, state: 'stale', needs: 'replay', reason: H }],
    stale: 1, absent: 0, current: 0, rescore: 0, replay: 1, roster: [],
    gamesPerPairing: 4, weights: H, build: H, sweep: null }),
  pieces: piecesHtml(piece, { index: 0 }),
  sides: sidePicker(game, ROSTER, null),
  fairy: fairyNote(game, []),
  standing: standingHtml(game),
  voting: votingHtml(game),
  verdict: verdictHtml({ verdict: H, strengths: [H], weaknesses: [H],
                         suspected_degeneracy: H }),
  board: drawPiece({ player: 0, type: 0 }, 0, 0, 10) + seatHtml(0),
  letter: (SESSION.ruleset.icons = [], drawPiece({ player: 0, type: 0 }, 0, 0, 10)),
  designerPiece: pieceEditor(DRAFT.pieces[0], 0),
  designerMove: moveEditor(DRAFT.pieces[0].moves[0], 0, 0),
  placement: placementEditor(DRAFT.setup.placements[0], 0),
  palette: paletteHtml(),
  identity: identityHtml({ hash: H, schemaVersion: 3, settledVersion: true,
                           declaredVersion: H }),
  cardField: fieldHtml({ control: 'text', key: 'x', label: H, says: H }, H, ['x'],
                       'rules', []),
  cardMenu: fieldHtml({ control: 'menu', key: 'y', label: H, says: H,
                        options: [{ value: H, label: H, says: H }] }, H, ['y'], 'rules', []),
})"""


class TestHeadIsGetWithoutTheBody:
    """Q792: ``HEAD /`` answered 501 (the stdlib's answer for a method with no handler).
    A HEAD is the GET's status and headers and no body."""

    def test_head_answers_like_get_and_sends_no_body(self) -> None:
        server = Server(("127.0.0.1", 0), handler_for(App(RULESETS, library.RUNS_DIR)))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        try:
            for path in ("/", "/api/ruleset-does-not-exist"):
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                connection.request("GET", path)
                got = connection.getresponse()
                body = got.read()
                connection.request("HEAD", path)
                head = connection.getresponse()
                # Read on the same keep-alive connection: a body left on the wire would
                # break the next request, so the connection must still be usable.
                assert head.read() == b""
                assert head.status == got.status
                assert head.getheader("Content-Type") == got.getheader("Content-Type")
                assert head.getheader("Content-Length") == str(len(body))
                assert head.getheader("Content-Security-Policy") == got.getheader(
                    "Content-Security-Policy"
                )
                connection.request("GET", path)
                again = connection.getresponse()
                assert again.status == got.status
                assert again.read() == body
                connection.close()
            assert got.status in (200, 404)
        finally:
            server.shutdown()
            server.server_close()

    def test_the_page_itself_is_a_200(self) -> None:
        server = Server(("127.0.0.1", 0), handler_for(App(RULESETS, library.RUNS_DIR)))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            connection.request("HEAD", "/")
            head = connection.getresponse()
            assert head.status == 200
            assert int(head.getheader("Content-Length") or 0) > 0
            assert head.read() == b""
            connection.close()
        finally:
            server.shutdown()
            server.server_close()


class TestTheCrossSiteHeader:
    def test_a_post_without_the_header_is_refused_before_dispatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The local application too. It is bound to localhost, but a page on any site
        can still POST to it, and `/api/run` spends money."""
        from chess_variant_sandbox.web.server import REQUEST_HEADER, REQUEST_VALUE

        reached: list[str] = []
        app = App(RULESETS, library.RUNS_DIR)
        monkeypatch.setattr(app, "post", lambda path, body: reached.append(path) or {})
        server = Server(("127.0.0.1", 0), handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        try:
            forged = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            forged.request("POST", "/api/run", body=b'{"hash": "x"}',
                           headers={"Content-Type": "text/plain"})
            assert forged.getresponse().status == 403
            forged.close()
            assert reached == []

            ours = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            ours.request("POST", "/api/run", body=b'{"hash": "x"}',
                         headers={"Content-Type": "application/json",
                                  REQUEST_HEADER: REQUEST_VALUE})
            assert ours.getresponse().status == 200
            ours.close()
            assert reached == ["/api/run"]
        finally:
            server.shutdown()
            server.server_close()

    def test_both_senders_carry_it(self) -> None:
        """The value the server compares against, so a rename of one cannot miss the other."""
        from page_text import PAGE, text

        from chess_variant_sandbox.web.server import REQUEST_VALUE

        sent = f"'X-Requested-With': '{REQUEST_VALUE}'"
        assert sent in PAGE
        assert sent in text("/static/site/post.js")
        assert PAGE.count("method: 'POST'") == 1


_ANIMATION_STUB = """
  let __made = 0;
  const __ANIM = [];
  function __node(label, tag) {
    const attrs = {};
    const node = {
      label: label, tag: tag, style: {}, _html: '',
      setAttribute(k, v) { attrs[k] = String(v); },
      remove() {}, querySelector() { return null; },
      get innerHTML() { return node._html; },
      set innerHTML(v) { node._html = v; },
      animate(frames, opts) {
        __ANIM.push({
          label: label, tag: tag, html: node._html,
          attrs: JSON.parse(JSON.stringify(attrs)),
          delay: (opts && opts.delay) || 0,
          duration: (opts && opts.duration) || 0,
          frames: JSON.stringify(frames),
        });
        return { onfinish: null };
      },
    };
    return node;
  }
  const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                             set: () => true });
  global.document = { getElementById: () => el, querySelector: () => null,
                      querySelectorAll: () => [], addEventListener: () => {},
                      createElement: () => el, body: el,
                      createElementNS: (ns, tag) => __node('made' + (++__made), tag) };
  global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                    addEventListener: () => {}, scrollTo: () => {} };
  global.location = { pathname: '/game', hash: '' };
  global.fetch = () => new Promise(() => {});
  function __svg(board) {
    const here = new Map();
    board.forEach((value, square) => { if (value) here.set(square, __node('sq' + square, 'g')); });
    return {
      appendChild() {},
      querySelector(selector) {
        const found = /data-piece="(\\d+)"/.exec(selector);
        return found && here.has(Number(found[1])) ? here.get(Number(found[1])) : null;
      },
    };
  }
"""


def _watch_ply(rules: dict, before: list, after: list, last: dict | None) -> dict:
    """Play one ply through the page's own animation, in node, and report what moved.

    The page is a string, so the alternative is asserting on its source -- which cannot
    tell a piece animated from its own square from one animated from the square another
    piece was just taken on, and that distinction is the whole of Q69. Everything the
    animation touches is stubbed except its arithmetic: `animate` records the delay, the
    duration and the frames it was handed, and the returned list is the ply as a viewer
    would see it, in milliseconds. Skipped where node is absent rather than approximated.
    """
    import json as _json
    import re

    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    call = f"""
      SESSION = {{ ruleset: {_json.dumps(rules)}, board: {_json.dumps(after)},
                   lastMove: {_json.dumps(last)}, seats: ['a', 'b'], moves: [] }};
      const before = {_json.dumps(before)};
      const cell = 100 / Math.max({rules["width"]}, {rules["height"]});
      const beats = plyBeats(SESSION.board, SESSION.lastMove);
      // The frame `drawBoard` would have drawn in, built the way it builds it, so a
      // wrapping ruleset reaches the animation as the board on screen rather than as a
      // flat one the crossing code would never run for.
      const frame = boardView({rules["width"]}, {rules["height"]},
        wrapAxes(SESSION.ruleset.wrap), SHIFT, FLIPPED);
      const busy = animateChange(__svg(SESSION.board), {{ board: before }}, cell,
        {rules["width"]}, {rules["height"]}, frame, beats);
      console.log(JSON.stringify({{ beats: beats, busy: busy, animations: __ANIM }}));
    """
    done = node_session.run(_ANIMATION_STUB + script.group(1) + "\n" + call)
    assert done.returncode == 0, done.stderr
    return _json.loads(done.stdout)


def _agent_replies(scenario: str, setup: str = "", trigger: str = "maybeAgent();") -> dict:
    """Run the page's own `maybeAgent` against a reply that arrives late, in node.

    `api` and `applySnapshot` are the two things stubbed, so what is under test is the
    chain `maybeAgent` builds and nothing else. Returns the ids of the snapshots that
    reached the board, and how many times a move was asked for -- a game that is stopped
    has to stop *asking*, which is a thing the applied list cannot see.

    `setup` runs before the chain is started and `trigger` is what starts it, so the same
    harness answers both halves of pausing: that a paused game asks for nothing, and that
    resuming asks through the one function that checks the reply still belongs to the
    game on screen.
    """
    import json as _json
    import re

    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      const applied = [];
      const anything = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                                       set: () => true });
      let board = { innerHTML: '', append() {} };
      global.document = { getElementById: () => board, querySelector: () => null,
                          querySelectorAll: () => [], addEventListener: () => {},
                          createElement: () => anything, body: anything };
      global.window = { matchMedia: () => ({ matches: true, addEventListener: () => {} }),
                        addEventListener: () => {}, scrollTo: () => {} };
      global.location = { pathname: '/game', hash: '' };
      let settle = null;
      global.fetch = () => new Promise(() => {});
    """
    runner = f"""
      let asked = 0;
      api = () => new Promise(resolve => {{ asked += 1; settle = resolve; }});
      applySnapshot = snapshot => {{ applied.push(snapshot.id); }};
      drawBoard = () => {{}};
      drawPane = () => {{}};
      SESSION = {{ id: 'first', awaitingAgent: true, seats: ['a', 'b'], sideToMove: 0,
                   history: [], board: [], ruleset: {{}} }};
      {setup}
      {trigger}
      {scenario}
      if (settle) settle({{ id: 'first', awaitingAgent: false }});
      setTimeout(() => console.log(JSON.stringify({{ applied: applied, asked: asked }})), 20);
    """
    done = node_session.run(stub + script.group(1) + "\n" + runner)
    assert done.returncode == 0, done.stderr
    return _json.loads(done.stdout)


class TestAnAbandonedGameStopsPlaying:
    """Found while watching a machine game for Q69, and it is the same complaint.

    A machine game is a chain: paint the reply, ask for the next move, paint that. So
    *replacing* the game starts a second chain rather than ending the first, and nothing
    checked which game a reply had been asked of. `new game` and `play the other side`
    both do it, and what lands on the board is two games taking alternate plies --
    watched here as `first@53, second@1, first@54, second@2`. Pieces from a position
    twenty moves back appear among the pieces of the new one and vanish again next ply,
    which from the near side of the screen is a board that captures things for no reason.

    The other half is the page moving on entirely: `#status` is gone, and writing
    "thinking" into it threw on every reply, which is an error box on a page nobody is
    looking at and a request loop nobody asked for.
    """

    def test_a_reply_to_a_replaced_game_is_dropped(self) -> None:
        replies = _agent_replies("SESSION = { id: 'second', awaitingAgent: false };")
        assert replies["applied"] == []

    def test_a_reply_to_the_game_on_screen_is_kept(self) -> None:
        assert _agent_replies("")["applied"] == ["first"]

    def test_a_reply_with_no_board_to_draw_on_is_dropped(self) -> None:
        assert _agent_replies("board = null;")["applied"] == []


def _ply(engine, texts):
    """Play a line from the opening, and hand back its last ply as the page gets it.

    `last` carries what the wild-piece phase did, because a real snapshot's does: the
    session puts the core's own account of the phase into the history entry the page
    reads as `lastMove`.
    """
    from chess_variant_sandbox.engine.moves import move_from, move_to

    state = engine.initial_state()
    before: list[int] = []
    last: dict = {}
    for text in texts:
        move = engine.find_move(state, text)
        before = list(state.board)
        state, phase = engine.apply_with_phase(state, move)
        last = {
            "from": move_from(move),
            "to": move_to(move),
            "phase": [step.as_dict() for step in phase],
        }
    return before, list(state.board), last


def _ply_from(engine, state, move):
    """One ply from a hand-built position, as the page gets it.

    `_ply` plays a line from the opening; these are the two pairings no opening reaches.
    """
    from chess_variant_sandbox.engine.moves import move_from, move_to

    before = list(state.board)
    after, phase = engine.apply_with_phase(state, move)
    last = {
        "from": move_from(move),
        "to": move_to(move),
        "phase": [step.as_dict() for step in phase],
    }
    return before, list(after.board), last


def _one_boar(chess_spec, route, captures=None):
    """Chess plus a single wild boar on `route`, unmirrored."""
    import copy

    spec = copy.deepcopy(chess_spec)
    spec["schema_version"] = 9
    spec["name"] = "boar"
    piece = {
        "id": "W",
        "name": "boar",
        "value": 3,
        "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}],
    }
    if captures is not None:
        piece["captures"] = captures
    spec["pieces"].append(piece)
    spec["rules"]["npcs"] = [{"piece": "W", "route": route, "mirror": False}]
    return spec


def _struck_ghosts(watched):
    """The pieces drawn on the board that the position no longer has, as they were drawn.

    A ghost is made by `createElementNS`, so the stub labels it `made<n>`; its `html` is
    whatever `drawPiece` put in it, which with no icons is the piece's own letter.
    """
    return [
        a for a in watched["animations"]
        if a["tag"] == "g" and a["label"][:4] == "made" and a["delay"] > 0
    ]


class TestAWildPiecesTurn:
    """Q69, reported by the user watching a machine game: a boar taken by nothing.

    A ruleset with `rules.npcs` runs a wild-piece phase **after** the mover's move and
    inside the same `make`, so one submitted move makes two board changes and the second
    one can be a capture with no player piece moving to that square. The board animated
    the ply as a single diff, which had two consequences and the user saw both.

    The sequencing one: the mover's piece and the boar travelled at the same instant, so
    nothing on screen said the boar's step was a *consequence* of the move.

    The attribution one, which is the "captured by nothing": two boars of one type share
    a board byte, so when a pawn took the boar on c3 while the other stepped c6-c5, the
    arrival on c5 was matched -- by value, first match wins, ascending -- to c3. A boar
    flew out of the square it had just been eaten on, the pawn's capture got no strike at
    all, and the boar that really moved was knocked off c6 by nothing at all. One ply in
    twenty-four of a boar game, and it is the ply the game turns on.
    """

    @staticmethod
    def _rules(engine):
        return {
            "width": engine.spec.width, "height": engine.spec.height,
            "nTypes": engine.spec.n_types, "sideStride": engine.spec.side_stride,
            "nFacings": engine.spec.n_facings, "icons": [],
            "pieceIds": list(engine.spec.piece_ids),
            "facingAngles": [list(row) for row in engine.spec.facing_headings()],
        }

    def test_the_page_is_told_what_the_phase_did(self, boar_game, chess_game) -> None:
        """Q74: the core reports the phase, and the page stops working it out.

        The page used to receive the compiled route table and pair a departure with an
        arrival off it. That was exact for every shape a committed ruleset reaches and
        still a reconstruction -- so the session now carries `Game::npc_log` itself, one
        `(from, to, eaten)` per wild piece that walked, in the history entry the page
        reads as `lastMove`. Nothing is sent for a ruleset with no wild pieces, and the
        route table is not sent at all any more.
        """
        session = Sessions().start(boar_game, ("random", "random"))
        engine = session.engine
        move = engine.find_move(session.state, "Pd2-d4")
        phase = session.play(move)["lastMove"]["phase"]
        # c3 -> c4 and c6 -> c5: both boars, one step each, and nothing eaten.
        assert phase == [{"from": 18, "to": 26, "eaten": 0},
                         {"from": 42, "to": 34, "eaten": 0}]

        assert "wild" not in Sessions().start(chess_game, ("random", "random")).snapshot()[
            "ruleset"
        ]
        plain = Sessions().start(chess_game, ("random", "random"))
        plain.play(plain.engine.find_move(plain.state, "Pd2-d4"))
        assert plain.snapshot()["lastMove"]["phase"] == []

    def test_the_phase_is_its_own_beat(self, boar_game) -> None:
        """One ply, two beats, and the second one is the phase.

        `plyBeats` is the model the rest of this rests on and the seam a viewer stepping
        through a game reads: beat *n* of a ply is a position, not a fragment of an
        animation.
        """
        engine = Engine(boar_game.spec)
        before, after, last = _ply(engine, ["Pd2xc3"])
        watched = _watch_ply(self._rules(engine), before, after, last)

        beats = watched["beats"]
        assert [beat["kind"] for beat in beats] == ["move", "phase"]
        # c6 -> c5: the boar that survived, and the *only* one that moved.
        assert beats[1]["steps"] == [{"from": 42, "to": 34, "eaten": 0}]
        assert beats[1]["board"] == after
        # The position between them: the pawn has taken the boar on c3 and the other
        # boar has not moved yet. Nothing else in the game is in a position like it.
        assert beats[0]["board"][18] == before[11], "the pawn stands on c3"
        assert beats[0]["board"][42] == before[42], "the second boar is still on c6"
        assert beats[0]["board"][34] == 0, "and has not stepped to c5 yet"

    def test_the_capture_is_shown_where_it_happened(self, boar_game) -> None:
        """The bug, in the one place it is visible: which square the ring is struck on.

        Under the old pairing the boar taken on c3 was animated as *travelling* to c5, so
        it got no strike, and the strike that was drawn landed on c6 -- a square the pawn
        never went near, from no direction, on a piece nothing had touched. There is one
        capture in this ply and it happened on c3.
        """
        engine = Engine(boar_game.spec)
        before, after, last = _ply(engine, ["Pd2xc3"])
        watched = _watch_ply(self._rules(engine), before, after, last)

        cell = 100 / 8

        def centre(square):
            return (round(square % 8 * cell + cell / 2, 3),
                    round((7 - square // 8) * cell + cell / 2, 3))

        rings = [
            (round(float(a["attrs"]["cx"]), 3), round(float(a["attrs"]["cy"]), 3))
            for a in watched["animations"] if a["tag"] == "circle"
        ]
        assert rings == [centre(18)], "one capture, on c3, where the pawn took the boar"
        assert centre(42) not in rings, "and nothing struck the boar that walked away"

    def test_the_boar_waits_for_the_move_it_answers(self, boar_game) -> None:
        """The sequencing half: the phase does not start until the mover has landed.

        Both changes used to begin at the same instant, which is why the user could not
        tell which piece had taken what. A wild piece is drawn on the square it *ends*
        on, so holding it still means a delayed travel whose first frame is its origin --
        `fill: 'backwards'` -- and not a piece that pops into place.
        """
        engine = Engine(boar_game.spec)
        before, after, last = _ply(engine, ["Pd2xc3"])
        watched = _watch_ply(self._rules(engine), before, after, last)

        moved = {a["label"]: a for a in watched["animations"] if a["tag"] == "g"}
        pawn = moved["sq18"]
        boar = moved["sq34"]
        assert pawn["delay"] == 0
        assert boar["delay"] >= pawn["duration"], "the boar steps after the pawn lands"
        assert boar["delay"] > pawn["duration"], "with a beat between them"
        # And the whole ply is reported as busy for as long as it actually runs, or the
        # next snapshot wipes the phase before anybody has seen it.
        assert watched["busy"] >= boar["delay"] + boar["duration"]

    def test_a_piece_the_boar_eats_is_struck_where_it_arrived(self, boar_game) -> None:
        """The mover walks into the boar's next square and the boar eats it there.

        A real line: after `Qd1-d3` the boar steps onto d3 and takes the queen. The board
        draws the boar on d3, so the queen has no element of its own -- and the honest
        picture is one piece that travels to d3 and is then struck there, rather than a
        queen that vanishes at d1 and a boar that appears at d3.
        """
        engine = Engine(boar_game.spec)
        before, after, last = _ply(
            engine,
            ["Pd2-d4", "Pd7-d6", "Bc1-f4", "Pa7-a6", "Ph2-h3", "Pe7-e5", "Qd1-d3"],
        )
        watched = _watch_ply(self._rules(engine), before, after, last)

        beats = watched["beats"]
        assert [beat["kind"] for beat in beats] == ["move", "phase"]
        step = next(s for s in beats[1]["steps"] if s["to"] == 19)  # d3
        assert step["eaten"] == before[3], "the queen, eaten where she landed"
        assert beats[0]["board"][19] == before[3], "and standing there between the beats"

        # One ghost, travelling to d3 and struck there: not two separate events.
        ghosts = [a for a in watched["animations"] if a["tag"] == "g" and a["label"][:4] == "made"]
        travelled = [a for a in ghosts if "translate" in a["frames"] and a["delay"] == 0]
        assert len(travelled) == 1
        struck = [a for a in ghosts if a["label"] == travelled[0]["label"] and a["delay"] > 0]
        assert struck, "the same ghost is what the boar knocks off d3"

    def test_a_pawn_eaten_where_it_promoted_is_the_piece_it_became(self, chess_spec) -> None:
        """Q74's first guess, and the core settles it.

        A pawn promotes onto the boar's next square and the boar eats the queen standing
        there. Diffing two positions cannot see the queen at all -- the ply begins with a
        pawn and ends with a boar -- so the page used to draw the pawn being knocked off
        a square it had already left as a pawn. `Game::npc_log` holds the byte the
        destination carried at the instant the boar wrote over it, which is the queen.
        """
        from chess_variant_sandbox.engine.compile import WILD
        from chess_variant_sandbox.engine.moves import (
            MOVE_PROMOTION,
            move_from,
            move_promotion,
            move_to,
            move_type,
        )

        engine = Engine(_one_boar(chess_spec, ["e6", "f7", "f8"]))
        state = engine.make_state(
            {"a1": ("K", 0), "a8": ("K", 1), "e7": ("P", 0), "f8": ("N", 1),
             "f7": ("W", WILD)},
            moved=("e7",),
        )
        types = list(engine.spec.piece_ids)
        move = next(  # e7 takes on f8 and promotes to a queen
            m for m in engine.legal_moves(state)
            if move_from(m) == 52 and move_to(m) == 61
            and move_type(m) == MOVE_PROMOTION
            and move_promotion(m) % engine.spec.n_types == types.index("Q")
        )
        before, after, last = _ply_from(engine, state, move)

        queen = engine.spec.encode_piece(0, types.index("Q"))
        assert queen != engine.spec.encode_piece(0, types.index("P"))
        assert last["phase"] == [{"from": 53, "to": 61, "eaten": queen}]

        watched = _watch_ply(self._rules(engine), before, after, last)
        # Between the beats the queen stands on f8, so that is the piece the board
        # travels there and the boar then knocks off it.
        assert watched["beats"][0]["board"][61] == queen
        assert any(">Q</text>" in ghost["html"] for ghost in _struck_ghosts(watched))
        assert not any(">P</text>" in ghost["html"] for ghost in _struck_ghosts(watched))

    def test_a_castling_rook_eaten_in_transit_is_taken_rather_than_lost(
        self, chess_spec
    ) -> None:
        """Q74's second guess. White castles and the boar eats the rook on f1.

        Nothing stood on f1 before the move, so the square the page read for a victim was
        empty and the rook simply stopped existing -- no strike, no ring, a piece quietly
        gone from a board it had moved across. The core's `eaten` is the rook, on the
        square the castling had just put it.

        The boar may take a rook and not a king, which is why castling is legal at all: a
        wild piece threatens the square it steps to next, and f1 is a square this king
        crosses.
        """
        from chess_variant_sandbox.engine.compile import WILD
        from chess_variant_sandbox.engine.moves import MOVE_CASTLE, move_type

        engine = Engine(_one_boar(chess_spec, ["f3", "f2", "f1"], captures=["R"]))
        state = engine.make_state(
            {"e1": ("K", 0), "h1": ("R", 0), "a8": ("K", 1), "f2": ("W", WILD)}
        )
        move = next(m for m in engine.legal_moves(state) if move_type(m) == MOVE_CASTLE)
        before, after, last = _ply_from(engine, state, move)

        rook = engine.spec.encode_piece(0, list(engine.spec.piece_ids).index("R"))
        assert before[5] == 0, "nothing stands on f1 until the castling puts the rook there"
        assert last["phase"] == [{"from": 13, "to": 5, "eaten": rook}]

        watched = _watch_ply(self._rules(engine), before, after, last)
        assert watched["beats"][0]["board"][5] == rook, "the rook, between the beats"
        cell = 100 / 8
        rings = [
            (round(float(a["attrs"]["cx"]), 3), round(float(a["attrs"]["cy"]), 3))
            for a in watched["animations"] if a["tag"] == "circle"
        ]
        assert rings == [(round(5 % 8 * cell + cell / 2, 3),
                          round(7 * cell + cell / 2, 3))], "one capture, struck on f1"
        assert any(">R</text>" in ghost["html"] for ghost in _struck_ghosts(watched))


def _lean(snapshot: dict) -> dict:
    """One snapshot as the page would receive it, minus the two big fields nothing here
    reads.

    Copied, not referenced: `snapshot()` hands out the session's own history list, so a
    snapshot kept in Python goes on growing with the game. The page never sees that
    because every snapshot is serialised on its way out, which is what this reproduces.
    """
    import json as _json

    lean = _json.loads(_json.dumps(snapshot))
    lean["moves"] = []
    lean["ruleset"]["icons"] = []
    return lean


def _watched_game(game, moves: list[str]) -> list[dict]:
    """The snapshots a page is handed while a game is played, the opening one first.

    Each one is copied as it is taken, because that is when it would have gone over the
    wire: kept by reference, they would all share the history list and every one of them
    would report the finished game.
    """
    session = Sessions().start(game, ("random", "random"))
    engine = session.engine
    out = [_lean(session.snapshot())]
    for text in moves:
        out.append(_lean(session.play(engine.find_move(session.state, text))))
    return out


def _review(snapshots: list[dict], script: str) -> dict:
    """Step through a game with the page's own timeline and cursor, in node.

    The snapshots are the ones a real `Session` handed out, so every position in the
    timeline is one the engine produced rather than one the test wrote. Drawing is
    stubbed to record *what it was handed* -- `position()`, `tint()`, the marks and the
    review line -- because that is the seam stepping is answerable at: a step is right
    when the drawing is given the beat the reader asked for. Skipped where node is
    absent rather than approximated.
    """
    import json as _json
    import re

    from page_text import PAGE

    source = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert source is not None
    runner = f"""
      const seen = [];
      drawBoard = () => seen.push({{
        index: cursorAt(), reviewing: AT !== null, paused: PAUSED,
        board: position().slice(), colour: tint().slice(),
        last: AT ? AT.lastMove : SESSION.lastMove, steps: PHASE,
        line: TIMELINE.length > 1 ? beatLine(TIMELINE[cursorAt()], cursorAt()) : '',
      }});
      drawPane = () => {{}};
      drawReview = () => {{}};
      drawStatus = () => {{}};
      for (const snapshot of {_json.dumps(snapshots)}) {{
        SESSION = snapshot;
        record(snapshot);
      }}
      {script}
      console.log(JSON.stringify({{
        seen: seen, paused: PAUSED,
        timeline: TIMELINE.map(entry => ({{
          kind: entry.kind, plies: entry.plies, board: entry.board,
          steps: entry.steps, text: entry.text, hands: entry.hands }})),
      }}));
    """
    done = node_session.run(_ANIMATION_STUB + source.group(1) + "\n" + runner)
    assert done.returncode == 0, done.stderr
    return _json.loads(done.stdout)


class TestSteppingThroughAGame:
    """Q70, asked by the user: *pausing and forwards and backwards through the moves*.

    He found the last two defects on this surface by watching a machine game go past,
    which is a poor way to find anything -- by the time you know you wanted to look at
    something, it has gone. So this is not a convenience: it is the tool that makes
    reviewing a game possible at all.

    Two properties carry it. **A step is a beat**, so a game with a wild piece can be
    stopped between the mover's move and the phase the rules run after it -- the split
    Q69 made visible, which a per-ply stepper would hide again. And **a step renders a
    position that already exists**: the timeline holds the boards `plyBeats` produced,
    stepping is an array index, and nothing anywhere undoes a move to get back to one.
    That is what makes the way back and the way forward agree, and it is what keeps this
    inside the rule that the page may hold a position and ask the engine questions but
    may not answer them.
    """

    def test_a_step_stops_between_the_move_and_the_wild_pieces_turn(self, boar_game) -> None:
        """The whole reason the unit is a beat.

        `Pd2xc3` is one submitted move and two board changes: the pawn takes the boar on
        c3, and then the phase steps the other boar c6-c5. Stepping back one from the end
        has to land *between* them -- pawn on c3, the second boar still on c6 -- which is
        a position no ply of this game ends on and which the page can only show because
        `plyBeats` kept it.
        """
        snapshots = _watched_game(boar_game, ["Pd2xc3"])
        opening = snapshots[0]["board"]
        watched = _review(snapshots, "stepBy(-1); stepBy(1);")

        assert [beat["kind"] for beat in watched["timeline"]] == ["start", "move", "phase"]
        between, phase = watched["seen"]
        assert between["index"] == 1 and between["reviewing"] is True
        assert between["board"][18] == opening[11], "the pawn stands on c3"
        assert between["board"][42] == opening[42], "and the second boar is still on c6"
        assert between["board"][34] == 0, "it has not stepped to c5 yet"
        # The trail belongs to the phase, so the beat before it does not carry one.
        assert between["steps"] is None
        # And the beat after it does, and says in words what it is.
        assert phase["steps"] == [{"from": 42, "to": 34, "eaten": 0}]
        assert phase["board"] == snapshots[-1]["board"]
        assert "and then the wild pieces move" in phase["line"]
        assert between["line"].startswith("after 1.")

    def test_stepping_back_and_forward_again_shows_the_same_thing(self, boar_game) -> None:
        """The classic defect here is a backward path that differs from the forward one.

        It cannot happen, and this is the reason rather than a spot check: every beat is
        an array the engine produced, kept, and a step is an index into it. So the walk
        below goes back to the opening, forward, back again and forward to the end,
        visiting most beats several times from both directions -- and every visit to a
        beat is asserted to hand the drawing the same board, the same last-move marks,
        the same wild-piece trail and the same line of text.
        """
        snapshots = _watched_game(
            boar_game, ["Pd2-d4", "Pd7-d6", "Bc1-f4", "Pa7-a6", "Ph2-h3", "Pe7-e5", "Qd1-d3"]
        )
        walk = """
          while (cursorAt() > 0) stepBy(-1);
          while (cursorAt() < TIMELINE.length - 1) stepBy(1);
          stepBy(-1); stepBy(-1); stepBy(1);
          while (cursorAt() > 0) stepBy(-1);
          while (cursorAt() < TIMELINE.length - 1) stepBy(1);
        """
        watched = _review(snapshots, walk)

        frames: dict[int, list[dict]] = {}
        for frame in watched["seen"]:
            frames.setdefault(frame["index"], []).append(frame)
        assert min(frames) == 0, "the walk reached the opening position"
        assert max(frames) == len(watched["timeline"]) - 1, "and came back to the end"
        assert sum(1 for seen in frames.values() if len(seen) > 1) > 5, (
            "a walk that never revisits a beat proves nothing about revisiting one"
        )
        for index, seen in sorted(frames.items()):
            first = seen[0]
            for again in seen[1:]:
                assert again["board"] == first["board"], f"beat {index} changed on the way back"
                assert again["last"] == first["last"]
                assert again["steps"] == first["steps"]
                assert again["line"] == first["line"]
            # And what it shows is the beat itself, not merely something stable.
            assert first["board"] == watched["timeline"][index]["board"]

    def test_the_end_of_the_timeline_is_the_game_itself(self, chess_game) -> None:
        """Standing on the last beat is following the game, not reviewing it.

        The distinction is what lets the status line and the verdict stay true: they
        describe the position the game is in, and the only position the *game* is in is
        the last one. Anything earlier is reviewed, and says so.
        """
        snapshots = _watched_game(chess_game, ["Pe2-e4", "Pe7-e5", "Ng1-f3"])
        watched = _review(snapshots, "stepBy(-1); stepBy(-1); stepBy(1); stepBy(1);")

        indexes = [frame["index"] for frame in watched["seen"]]
        assert indexes == [2, 1, 2, 3]
        assert [frame["reviewing"] for frame in watched["seen"]] == [True, True, True, False]
        # Stepping stops the machine wherever it stops the board, including at the end.
        assert watched["paused"] is True
        assert watched["seen"][-1]["board"] == snapshots[-1]["board"]

    def test_the_game_goes_on_being_recorded_while_it_is_read(self, chess_game) -> None:
        """A reply that lands while somebody is reading move two is still a move.

        `record` reads the position a ply started from off the timeline rather than off
        the board, so where the reader happens to be standing cannot change what gets
        written down -- and the beat that arrives is reachable by stepping forward to it.
        """
        import json as _json

        snapshots = _watched_game(chess_game, ["Pe2-e4", "Pe7-e5", "Ng1-f3"])
        arrives = _json.dumps(snapshots[3])
        watched = _review(
            snapshots[:3],
            # What `applySnapshot` does when a reply lands, in the order it does it.
            f"stepBy(-1); stepBy(-1); SESSION = {arrives}; record(SESSION);"
            " stepBy(1); stepBy(1); stepBy(1);",
        )
        assert len(watched["timeline"]) == 4
        # Reading move one did not drag the board forward when the reply landed...
        assert [frame["index"] for frame in watched["seen"]] == [1, 0, 1, 2, 3]
        # ... and the ply that arrived is a beat you can step on to.
        assert watched["seen"][-1]["board"] == snapshots[3]["board"]

    def test_a_take_back_shortens_the_game_it_is_taken_back_from(self, chess_game) -> None:
        """The plies after the position being restored are no longer part of the game.

        Leaving them in the timeline would offer a reader a "forward" into a line nobody
        played, which is worse than not offering one at all.
        """
        snapshots = _watched_game(chess_game, ["Pe2-e4", "Pe7-e5", "Ng1-f3", "Nb8-c6"])
        assert len(_review(snapshots, "")["timeline"]) == 5
        # The take-back's own snapshot is the earlier position, handed over again.
        import json as _json

        back = _json.dumps(snapshots[2])
        watched = _review(snapshots, f"record({back});")
        assert [beat["plies"] for beat in watched["timeline"]] == [0, 1, 2]


class TestAPausedGameStopsAsking:
    """A machine game is a chain and nothing else: it runs for as long as something keeps
    asking for the next move. There is no timer to stop and no loop to break, so pausing
    is one thing -- `maybeAgent` declining to ask -- and resuming is the same function
    being called again.

    Which matters more than it sounds, because of the defect fixed beside this one: two
    machine games once took alternate plies on one board, because replacing a game
    started a second chain rather than ending the first. Resuming must not do that, and
    it cannot, because it starts no chain of its own -- it goes through the one function
    that checks the reply still belongs to the game on screen.
    """

    def test_a_paused_game_asks_for_nothing(self) -> None:
        stopped = _agent_replies("", setup="PAUSED = true;")
        assert stopped["asked"] == 0, "a paused game must not ask for another move"
        assert stopped["applied"] == []

    def test_resuming_asks_again(self) -> None:
        going = _agent_replies("", setup="PAUSED = true;", trigger="resumeGame();")
        assert going["asked"] == 1
        assert going["applied"] == ["first"]

    def test_resuming_does_not_revive_a_game_the_page_has_moved_on_from(self) -> None:
        """The liveness check is `maybeAgent`'s, and resuming has to go through it."""
        gone = _agent_replies(
            "SESSION = { id: 'second', awaitingAgent: false };",
            setup="PAUSED = true;",
            trigger="resumeGame();",
        )
        assert gone["applied"] == []


def _move_list(snapshots: list[dict], script: str = "") -> list[tuple[list[str], str]]:
    """The play board's move list as the page draws it, one `(classes, text)` per row.

    The snapshots go through `record` exactly as a live page takes them, so a phase row
    names its wild piece from the walkers the real timeline found. `script` runs before
    the list is drawn -- a step back, say -- and drawing the board is stubbed, because
    the pane is the thing under test. Skipped where node is absent.
    """
    import html
    import json as _json

    setup = f"""
      const pane = {{ innerHTML: '', className: '', children: null, scrollTop: 0 }};
      document.getElementById = id => (id === 'pane' ? pane : null);
      drawBoard = () => {{}};
      drawReview = () => {{}};
      drawStatus = () => {{}};
      for (const snapshot of {_json.dumps(snapshots)}) {{
        SESSION = snapshot;
        record(snapshot);
      }}
      TAB = 'moves';
      {script}
      drawPane();
    """
    drawn = _in_page(setup, "pane.innerHTML")
    rows = re.findall(r'<div(?: class="([^"]*)")?>(.*?)</div>', drawn, re.S)
    return [
        (classes.split(), " ".join(html.unescape(re.sub(r"<[^>]+>", "", text)).split()))
        for classes, text in rows
    ]


def _played_game(game, plies: int, seed: int = 0) -> list[dict]:
    """The snapshots a page is handed while two random players play `plies` plies."""
    session = Sessions().start(game, ("random", "random"), seed=seed)
    out = [_lean(session.snapshot())]
    for _ in range(plies):
        snapshot = session.play_agent()
        if snapshot["outcome"]:
            break
        out.append(_lean(snapshot))
    return out


def _wild_games() -> list[Path]:
    """Every committed or archived ruleset with wild pieces in it."""
    return sorted(
        path
        for folder in (RULESETS, library.ARCHIVE_DIR)
        for path in folder.glob("*.json")
        if (json.loads(path.read_text(encoding="utf-8")).get("rules") or {}).get("npcs")
    )


class TestTheMoveListSaysWhatTheWildPiecesDid:
    """Q75. A ply in a wild-piece game is two beats, and the move list named one of them.

    The board animates the phase and the review steps onto it (Q69, Q70), but the list a
    reader goes back to afterwards said only *Pe7-e6* -- so the ply a wolves game turns on,
    a wolf eating a bishop, left no line anywhere. Now the phase is a row of its own under
    its move, worded from the core's own account of it (`lastMove.phase`, Q74): no engine
    call and nothing re-derived, and a game without wild pieces draws the list it always
    drew.
    """

    def test_the_phase_is_a_row_under_its_move(self, boar_game) -> None:
        """Both boars step after `Pd2-d4`: one row for the move, one for the phase, and
        the second boar's step shares the name the first one gave."""
        rows = _move_list(_watched_game(boar_game, ["Pd2-d4"]))
        assert rows == [
            (["stripe"], "1. Pd2-d4"),
            (["stripe", "phase"], "and the boar on c3 steps to c4 and c6 to c5"),
        ]

    def test_a_capture_names_what_was_taken_and_where(self, boar_game) -> None:
        """The line the list most needs: the boar steps onto d3 and eats the queen that
        had just moved there. `eaten` is the board byte the core overwrote, so it is the
        queen by name, and it is said in full rather than folded into a run of steps."""
        rows = _move_list(
            _watched_game(
                boar_game,
                ["Pd2-d4", "Pd7-d6", "Bc1-f4", "Pa7-a6", "Ph2-h3", "Pe7-e5", "Qd1-d3"],
            )
        )
        moves = [text for classes, text in rows if "phase" not in classes]
        assert moves[-1] == "4. Qd1-d3"
        classes, said = rows[-1]
        assert "phase" in classes
        assert "the boar on d4 takes the queen on d3" in said, said
        # One row per beat: seven moves, and every one of them was answered by a phase.
        assert len(rows) == 14
        assert sum("phase" in c for c, _ in rows) == 7
        # And a phase that took something is not drawn as quietly as a patrol: exactly
        # the rows that say *takes*, which in this line are three of the seven.
        took = [text for c, text in rows if "took" in c]
        assert took == [text for c, text in rows if "phase" in c and " takes " in text]
        assert said in took and len(took) == 3

    def test_the_highlight_follows_the_beat_not_the_ply(self, boar_game) -> None:
        """Stepping back one from the end stands between the move and the phase, so the
        move's row is the one lit; one further forward and it is the phase's."""
        snapshots = _watched_game(boar_game, ["Pd2-d4", "Pd7-d6"])
        between = _move_list(snapshots, "stepBy(-1);")
        lit = [text for classes, text in between if "on" in classes]
        assert lit == ["… Pd7-d6"]
        after = _move_list(snapshots, "stepBy(-1); stepBy(1);")
        # The last beat is the game as it stands, which is following it, not reviewing.
        assert [text for classes, text in after if "on" in classes] == []
        earlier = _move_list(snapshots, "stepBy(-2);")
        assert [text for classes, text in earlier if "on" in classes] == [
            "and the boar on c3 steps to c4 and c6 to c5"
        ]

    def test_an_ordinary_game_draws_the_list_it_always_drew(self, chess_game) -> None:
        """No wild piece, no phase row: one row per ply, the move's text, striped by ply
        exactly as the old odd-row rule striped it."""
        texts = ["Pe2-e4", "Pe7-e5", "Ng1-f3", "Nb8-c6", "Bf1-c4"]
        rows = _move_list(_watched_game(chess_game, texts))
        assert rows == [
            (["stripe"], "1. Pe2-e4"),
            ([], "… Pe7-e5"),
            (["stripe"], "2. Ng1-f3"),
            ([], "… Nb8-c6"),
            (["stripe"], "3. Bf1-c4"),
        ]
        stepped = _move_list(_watched_game(chess_game, texts), "stepBy(-1);")
        assert [text for classes, text in stepped if "on" in classes] == ["… Nb8-c6"]

    def test_the_list_does_not_invent_a_walker_it_was_not_told(self) -> None:
        """A ply the timeline never split has its steps but no walkers, and says *a wild
        piece* rather than guessing which piece that was."""
        said = _in_page(
            "SESSION = { ruleset: { width: 8, nTypes: 7, pieceNames: ['pawn'] } };",
            "phaseText([{ from: 16, to: 24, eaten: 0 }], null)",
        )
        assert said == "and the wild piece on a3 steps to a4"

    @pytest.mark.parametrize("path", _wild_games(), ids=lambda path: path.stem)
    def test_every_wild_piece_game_gets_a_row_per_phase(self, path: Path) -> None:
        """Every committed or archived game with wild pieces, forty random plies each:
        each ply whose phase moved anything has exactly one row for it, straight after
        its move, naming the walker by the spec's own name for it."""
        spec = load_spec(path)
        game = library.Game(
            hash=spec_hash(spec), name=path.stem, description="", path=path,
            source="archived", spec=spec,
        )
        snapshots = _played_game(game, 40, seed=7)
        history = snapshots[-1]["history"]
        rows = _move_list(snapshots)

        expected: list[str] = []
        for move in history:
            # A duck move (generation 42, N4c; its entry carries `pass`) is written on its
            # turn's line, after the piece move, so it opens no row of its own; the phase,
            # which runs after the duck half, follows that line.
            if "pass" not in move:
                expected.append("move")
            if move["phase"]:
                expected.append("phase")
        assert ["phase" if "phase" in c else "move" for c, _ in rows] == expected
        assert "phase" in expected, "forty plies and no wild piece moved"

        engine = Engine(spec)
        npcs = engine.spec.npcs
        assert npcs is not None
        wild = {engine.spec.piece_names[t] for t in npcs.piece if t >= 0}
        for classes, said in rows:
            if "phase" not in classes:
                continue
            # A phase that opens with a copy (generation 41, N3d) names the new piece first:
            # "and a new hydra grows on c6 from the one on d5" (N7's hydra, the first library
            # game with a split, met this).
            if said.startswith("and a new "):
                assert "wild piece" not in said, said
                assert any(f"a new {name} grows on " in said for name in wild), said
                continue
            assert said.startswith("and the "), said
            assert "wild piece" not in said, said
            assert any(f"the {name} on " in said for name in wild), said


def _board_legend(rules: dict, zones: bool = True) -> str:
    """The line the board draws under a position with these rules, in node."""
    import json as _json

    setup = f"""
      const legend = {{ innerHTML: '' }};
      document.getElementById = id => (id === 'legend' ? legend : null);
      SESSION = {{ ruleset: {_json.dumps(rules)} }};
      ZONES = {'true' if zones else 'false'};
      drawLegend(squareMarks(SESSION.ruleset, {{ zone: !ZONES }}));
    """
    return _in_page(setup, "legend.innerHTML")


def _review_bar(seats: list[str], outcome: dict | None = None) -> dict:
    """The replay bar the board builds under a game with these seats, in node."""
    import json as _json

    setup = f"""
      const nodes = {{}};
      const make = () => ({{ innerHTML: '', textContent: '', title: '',
                             dataset: {{}}, disabled: false, hidden: false }});
      document.getElementById = id => (nodes[id] || (nodes[id] = make()));
      SESSION = {{ id: 'g', seats: {_json.dumps(seats)},
                   outcome: {_json.dumps(outcome)}, history: [], ruleset: {{}} }};
      TIMELINE_FOR = 'g';
      TIMELINE = [{{ kind: 'start', plies: 0, text: '' }},
                  {{ kind: 'ply', plies: 1, text: 'Pe2-e4' }}];
      drawReview();
    """
    return _in_page(
        setup,
        "({ markup: nodes.review.innerHTML, pauseHidden: nodes.pause.hidden,"
        " where: nodes.where.textContent })",
    )


def _ruleset_block(game) -> dict:
    """The ruleset the page is handed for this game, as a real session sends it."""
    return _lean(Sessions().start(game, ("human", "random")).snapshot())["ruleset"]


class TestTheBoardKnowsItsMode:
    """B1 of `docs/archive/plan-ux.md`. A rated game and a sandbox game looked identical.

    The board drew a take-back the store would refuse, offered no way to give a game up,
    and said nothing about which of the two a person was in -- so the first news that a
    result counted was a banner explaining why a move could not be taken back (§2.2).

    The mode is the store's answer and nothing the page works out: it travels on the
    snapshot, which is rebuilt from the row for every move, reply, take-back and restart.
    """

    def test_the_snapshot_carries_the_mode_and_the_local_application_has_none(
        self, chess_game
    ) -> None:
        """`web/play.py` is the local application's, which keeps nothing and has no
        modes. It sends the field empty rather than guessing at one."""
        assert Sessions().start(chess_game, ("human", "random")).snapshot()["mode"] == ""

    def test_the_mark_says_which_kind_of_game_this_is(self) -> None:
        rated = _in_page("", "modeHtml({ mode: 'leaderboard' })")
        assert ">rated<" in rated and "this result is rated" in rated
        loose = _in_page("", "modeHtml({ mode: 'sandbox' })")
        assert ">sandbox<" in loose and "nothing here counts" in loose
        # No modes at all is no mark: one that is always the same says nothing.
        assert _in_page("", "modeHtml({ mode: '' })") == ""

    def test_a_rated_game_draws_no_take_back_and_offers_a_resignation(self) -> None:
        from page_text import PAGE

        assert "const rated = mode === 'leaderboard';" in PAGE
        assert "${frameHtml(inline, snapshot.mode, watched, between, (!!snapshot.round || !!snapshot.spy) && !watched, !!snapshot.puzzle)}" in PAGE
        # The frame's control row, directly under the board (Q577): a rated game's has
        # no take-back and has a resignation, the local application's (no mode) the
        # other way round, and a game nobody sits at offers nobody's resignation. A
        # sandbox game has both (Q578, `TestASandboxGameCanBeEnded`).
        rated = _in_page("", "frameHtml(false, 'leaderboard', false)")
        loose = _in_page("", "frameHtml(false, '', false)")
        watched = _in_page("", "frameHtml(false, 'leaderboard', true)")
        controls = rated[rated.index('data-frame="controls"'):]
        assert 'id="resign"' in controls and 'id="undo"' not in controls
        assert 'id="undo"' in loose and 'id="resign"' not in loose
        assert 'id="resign"' not in watched
        # And the seats say so, whichever mode it is.
        assert "${modeHtml(snapshot)}" in PAGE

    def test_resigning_takes_two_presses_and_disarms_itself(self) -> None:
        """A finish that moves the rating is not a thing to do by a stray click."""
        from page_text import PAGE

        start = PAGE.index("const resign = document.getElementById('resign');")
        body = PAGE[start:PAGE.index("root.querySelectorAll('[data-tab]')", start)]
        assert "if (resign.dataset.armed !== 'yes') {" in body
        assert "sayOn(resign, 'really resign?');" in body
        assert "setTimeout(disarm, ARM_HOLD);" in body
        assert "resign.onblur = disarm;" in body
        assert "api('/api/session/' + snapshot.id + '/resign', {})" in body

    def test_an_armed_finish_waits_long_enough_to_be_read(self) -> None:
        """First-visit study N1: *really resign?* reverted after six seconds, before two
        readers' second press. Both buttons that end a game hold for `ARM_HOLD`, at least
        fifteen seconds, and no other timer disarms them."""
        from page_text import PAGE

        hold = _in_page("", "ARM_HOLD")
        assert hold >= 15000
        assert PAGE.count("setTimeout(disarm, ARM_HOLD);") == 1
        assert "sayOn(cancel, 'cancel the game'); }, ARM_HOLD);" in PAGE
        assert "6000)" not in PAGE.replace("16000)", "")

    def test_a_rematch_is_played_in_the_mode_it_was_pressed_in(self) -> None:
        """Q249 sent no mode, so the server's default applied and a rematch of a rated
        game was a sandbox game -- the board stopped counting without saying so."""
        from page_text import PAGE

        start = PAGE.index("const restart = (seats, follows = true) =>")
        body = PAGE[start:PAGE.index("document.getElementById('back').onclick", start)]
        assert "if (snapshot.mode && seats.includes('human')) body.mode = snapshot.mode;" in body

    def test_a_watch_is_started_without_one(self) -> None:
        """Nobody plays a watch, so there is nothing to rate and the store refuses a
        rated one. The seats are what say so, on both controls that start one."""
        from page_text import PAGE

        start = PAGE.index("const restart = (seats, follows = true) =>")
        body = PAGE[start:PAGE.index("document.getElementById('back').onclick", start)]
        assert "seats.includes('human')" in body, "the mode goes with the seats"
        page_start = PAGE.index("document.getElementById('watchIt').onclick")
        watch = PAGE[page_start:PAGE.index("fail(error, 'watchIt')", page_start)]
        assert "seats: [WATCH_AGENT, WATCH_AGENT]" in watch
        assert "mode" not in watch

    def test_the_game_page_has_the_watch_button_its_menus_could_already_reach(self) -> None:
        """Two computer players was reachable by setting both seat menus and pressing
        start, which is a thing you have to know rather than a thing you can see."""
        from page_text import PAGE

        assert '<button id="watchIt" title="${esc(watchTitle())}">' in PAGE

    def test_a_watch_says_who_is_playing_wherever_it_is_started(self) -> None:
        """Q265's sentence was the designer's alone, because only the draft route
        produced it. The board is one board, so it draws the one the server sent."""
        from page_text import PAGE

        start = PAGE.index("async function showBoard(snapshot, how) {")
        body = PAGE[start:PAGE.index("function applySnapshot(", start)]
        assert "noteWatching(snapshot);" in body


class TestTheReplayBarInAHumanGame:
    """B3 of `docs/archive/plan-ux.md`: no *pause* where a person holds a seat.

    A game you are playing already waits for you, so a pause button offers to stop
    something that is not running -- and it reads as a second kind of turn. Stepping back
    still stops the machine, because a reply landing on a position the reader has stepped
    away from would drag the board forward; the way on is *latest*.
    """

    def test_a_game_with_a_person_in_it_has_four_buttons(self) -> None:
        bar = _review_bar(["human", "minimax:nodes=400"])
        assert bar["markup"].count("<button") == 5
        assert bar["pauseHidden"] is True, "four of the five are drawn"
        assert bar["where"].startswith("after 1.")

    def test_a_game_nobody_sits_at_keeps_it(self) -> None:
        assert _review_bar(["random", "random"])["pauseHidden"] is False

    def test_a_finished_game_has_nothing_to_stop(self) -> None:
        assert _review_bar(["random", "random"], {"winner": 0, "reason": "mate"})[
            "pauseHidden"
        ] is True

    def test_coming_back_to_the_last_position_starts_the_game_again(
        self, chess_game
    ) -> None:
        """Without this the four-button bar would be a trap: stepping back stops the
        machine and there would be no control left to start it."""
        snapshots = _watched_game(chess_game, ["Pe2-e4", "Pe7-e5", "Ng1-f3"])
        for snapshot in snapshots:
            snapshot["seats"] = ["human", "random"]
        walk = "maybeAgent = () => {}; stepBy(-1); stepBy(-1); stepBy(1); stepBy(1);"
        watched = _review(snapshots, walk)

        assert watched["paused"] is False
        assert [frame["reviewing"] for frame in watched["seen"]] == [True, True, True, False]
        assert watched["seen"][-1]["board"] == snapshots[-1]["board"]

    def test_a_watched_game_stays_stopped_where_the_reader_stopped_it(
        self, chess_game
    ) -> None:
        """The other half of it: nobody is playing, so arriving at the end is arriving
        at a position, not a decision to play on. `resume` is what plays on."""
        snapshots = _watched_game(chess_game, ["Pe2-e4", "Pe7-e5", "Ng1-f3"])
        watched = _review(snapshots, "stepBy(-1); stepBy(1);")
        assert watched["paused"] is True


class TestTheLegendUnderTheBoard:
    """B4 of `docs/archive/plan-ux.md`: one line naming the marks this board carries.

    A ring is not self-explanatory and a board is not a place to ask. Only the kinds
    present are named, so the line is about the game on screen rather than about boards
    in general.
    """

    def test_chess_names_the_promotion_zone_and_nothing_else(self, chess_game) -> None:
        said = _board_legend(_ruleset_block(chess_game))
        assert "rose hatching" in said and "promotes on" in said
        assert "teal ring" not in said and "blue line" not in said
        # Whose zone is whose, now the zone has no beads (Q577), by the board's own
        # rank names off the compiled zones.
        assert "white&#39;s on rank 8, black&#39;s on rank 1" in said
        # The key is the mark itself, drawn small, and the entry names its kind.
        assert 'data-legend="zone"' in said and "url(#mark-hatch-zone-key)" in said

    def test_shogi_names_its_three_ranks(self, games) -> None:
        game = next(game for game in games if game.name == "shogi")
        said = _board_legend(_ruleset_block(game))
        assert "ranks 7-9" in said and "ranks 1-3" in said, said

    def test_dobutsu_names_the_square_that_wins_the_game(self, games) -> None:
        game = next(game for game in games if game.name == "dobutsu")
        said = _board_legend(_ruleset_block(game))
        assert "teal ring" in said and "wins the game" in said
        assert "rose hatching" in said, "dobutsu promotes too"

    def test_a_one_way_seam_is_named_where_there_is_one(self, games) -> None:
        # In the Library since Q512, so listed with the archive.
        library_games = library.discover(RULESETS, library.RUNS_DIR, corpus.ARCHIVE_DIR)
        game = next(game for game in library_games if game.name == "one-way cylinder chess")
        assert "blue line" in _board_legend(_ruleset_block(game))
        assert "blue line" not in _board_legend(_ruleset_block(next(
            game for game in games if game.name == "chess")))

    def test_a_board_with_no_marks_gets_no_line(self, games) -> None:
        game = next(game for game in games if game.name == "dull")
        assert _board_legend(_ruleset_block(game)) == ""

    def test_the_line_follows_what_the_board_is_drawing(self, chess_game) -> None:
        """The promotion zone is on a toggle, so a reader who puts it away is not told
        about a ring that is no longer there."""
        assert _board_legend(_ruleset_block(chess_game), zones=False) == ""


def _full_game(game, texts: list[str], seats=("human", "human")) -> list[dict]:
    """Snapshots as the page receives them, legal moves and all, one per move played."""
    import json as _json

    session = Sessions().start(game, seats)
    engine = session.engine
    out = [_json.loads(_json.dumps(session.snapshot()))]
    for text in texts:
        out.append(_json.loads(_json.dumps(session.play(engine.find_move(session.state, text)))))
    for snapshot in out:
        snapshot["ruleset"]["icons"] = []
    return out


def _frame(snapshots: list[dict], script: str, expression: str):
    """The play frame's state after these snapshots, in node: the timeline recorded as a
    live page records it, `script` run, then `expression` read. The status strip is a fake
    element whose markup is kept; drawing the board is stubbed."""
    import json as _json

    setup = f"""
      const strip = {{ classList: {{ on: false, toggle(_, on) {{ this.on = on; }} }} }};
      const status = {{ innerHTML: '', textContent: '', closest: () => strip }};
      document.getElementById = id => (id === 'status' ? status : null);
      drawBoard = () => {{}};
      for (const snapshot of {_json.dumps(snapshots)}) {{
        SESSION = snapshot;
        record(snapshot);
      }}
      {script}
    """
    return _in_page(setup, expression)


class TestTheBoardSaysCheckAndTheEnd:
    """Q577, the first-visit study's B1 (4 of 5 readers): check was a grey *- in check*
    under the board, the move list had no sign for it, the result was below the screen,
    and a click on a piece with nowhere to go did nothing at all. Three readers thought
    the game was still going or the board frozen.

    Every fact here is the engine's: `inCheck`, each move's `check` (asked of the core as
    the move is played), the outcome. The page only finds where to draw it.
    """

    def test_each_move_carries_the_engines_check_and_the_ruleset_its_royal_flags(
        self, chess_game
    ) -> None:
        """Chess, for a short line everybody can check by eye: 1. e4 f5 2. Qh5+."""
        snapshots = _full_game(chess_game, ["Pe2-e4", "Pf7-f5", "Qd1-h5"])
        last = snapshots[-1]
        assert [move["check"] for move in last["history"]] == [False, False, True]
        assert last["inCheck"] is True
        engine = Sessions().start(chess_game, ("human", "human")).engine
        royal = last["ruleset"]["royal"]
        assert [i for i, flag in enumerate(royal) if flag] == sorted(engine.spec.royal_types)

    def test_the_checked_royal_is_marked_and_the_strip_says_so_in_chess(
        self, chess_game
    ) -> None:
        snapshots = _full_game(chess_game, ["Pe2-e4", "Pf7-f5", "Qd1-h5"])
        seen = _frame(snapshots, "drawStatus();", """({
          squares: checkSquares(),
          marks: squareMarks(Object.assign({}, SESSION.ruleset, { checked: checkSquares() }))[60],
          strip: status.innerHTML,
        })""")
        assert seen["squares"] == [60], "black's king, on e8"
        assert seen["marks"][0] == {"kind": "check", "sides": []}
        assert "in check" in seen["strip"] and 'class="pill alarm"' in seen["strip"]
        assert "black" in seen["strip"] and "to move" in seen["strip"]

    def test_a_position_stepped_back_to_draws_its_own_check(self, chess_game) -> None:
        snapshots = _full_game(chess_game, ["Pe2-e4", "Pf7-f5", "Qd1-h5", "Pg7-g6"])
        seen = _frame(snapshots, "PAUSED = true; CURSOR = 3; AT = TIMELINE[3];",
                      "[checkSquares(), (CURSOR = null, AT = null, checkSquares())]")
        assert seen == [[60], []], "in check after Qh5, out of it after g6"

    def test_checkmate_ends_the_list_on_its_sign_and_the_result(self, chess_game) -> None:
        """Chess's shortest mate, 1. f3 e5 2. g4 Qh4#."""
        snapshots = _full_game(chess_game, ["Pf2-f3", "Pe7-e5", "Pg2-g4", "Qd8-h4"])
        assert snapshots[-1]["outcome"] == {"winner": 1, "reason": "checkmate"}
        seen = _frame(snapshots, "drawStatus();", """({
          rows: moveRows().map(r => [r.kind, r.html]),
          squares: checkSquares(),
          strip: status.textContent,
          ended: strip.classList.on,
        })""")
        kinds = [kind for kind, _ in seen["rows"]]
        assert kinds == ["move"] * 4 + ["result"]
        assert ">#</span>" in seen["rows"][3][1], "the mating move carries #"
        assert ">+</span>" not in "".join(html for _, html in seen["rows"])
        assert "wins by checkmate" in seen["rows"][4][1]
        assert seen["squares"] == [4], "white's king, mated on e1, keeps its mark"
        assert seen["strip"] == "Black (you) wins by checkmate."
        assert seen["ended"] is True, "the strip above the board becomes the verdict"

    def test_the_end_is_said_in_words_not_the_engines_code(self) -> None:
        said = _in_page(
            "SESSION = { seats: ['human', 'random'], ruleset: { sides: ['white', 'black'] } };",
            "[verdictText({ winner: 0, reason: 'royal_captured' }),"
            " verdictText({ winner: null, reason: 'repetition' }),"
            " verdictText({ winner: 1, reason: 'something_new' }),"
            " verdictText({ winner: 0, reason: 'extinction' }),"
            " verdictText({ winner: 0, reason: 'bare_royal' })]",
        )
        assert said[0].endswith("wins by capturing the royal piece.")
        assert said[1] == "Drawn by repetition."
        # P19: an unknown reason is still said, and never as its key.
        assert said[2].endswith("wins by the rules."), "an unknown reason is still said"
        assert said[3].endswith("wins by wiping out a whole kind of piece.")
        assert said[4].endswith("wins by leaving the royal piece alone.")

    def test_every_reason_the_core_has_is_in_words(self) -> None:
        """P19. Each `Reason::as_str` of `rust/src/game.rs` has its words, so the
        fallback is for a reason added later, never one there now."""
        from page_text import PAGE

        source = (Path(__file__).resolve().parent.parent / "rust" / "src" / "game.rs")
        reasons = set(re.findall(r'Reason::\w+ => "(\w+)"', source.read_text("utf-8")))
        assert reasons, "the core's reasons were found"
        start = PAGE.index("const REASON_WORDS = {")
        table = PAGE[start:PAGE.index("};", start)]
        for reason in reasons:
            assert f"{reason}: '" in table, reason

    def test_a_piece_with_no_legal_move_says_so(self, chess_game) -> None:
        """After 1. e4 f5 2. Qh5+ black's only move is g6, so the knight on b8 has none;
        the page reads that off the engine's list and the check off `inCheck`."""
        snapshots = _full_game(chess_game, ["Pe2-e4", "Pf7-f5", "Qd1-h5"])
        seen = _frame(snapshots, "clickSquare(57); drawStatus();",
                      "[NOTE, status.innerHTML, (clickSquare(54), NOTE)]")
        assert seen[0] == "that knight has no legal move - your side is in check"
        assert "that knight has no legal move" in seen[1]
        assert seen[2] == "", "a piece that can move clears the line"

    def test_a_side_with_two_royal_pieces_is_not_marked(self) -> None:
        """Which of two royals is attacked is the engine's to say, and it says only that
        one is: the strip says *in check*, the board draws no square."""
        seen = _in_page(
            "SESSION = { inCheck: true, sideToMove: 0, board: [1, 0, 1, 0],"
            " ruleset: { royal: [true], nTypes: 1, sideStride: 1 } }; AT = null;",
            "[checkSquares(), (SESSION.board = [1, 0, 0, 0], checkSquares())]",
        )
        assert seen == [[], [0]]


class TestThePlayFrame:
    """Q577, the study's F2 and F1: the controls and the legend were under the board,
    off a 900-pixel screen. The frame is status, board, controls, legend, in that order
    (plan-visual.md §3.3, as V4 laid it out; Q577 had the legend above the board), in
    both chromes, and every slot says what it is (`data-frame`) for the look check.
    `fitBoard` holds all four in the first screen."""

    def test_the_slots_are_in_order_status_board_controls_legend(self) -> None:
        html = _in_page("", "frameHtml(false, 'leaderboard', false)")
        order = [html.index(f'data-frame="{slot}"')
                 for slot in ("status", "board", "controls", "legend")]
        assert order == sorted(order)
        controls = html[html.index('data-frame="controls"'):]
        for control in ('id="flip"', 'id="shift"', 'id="resign"'):
            assert control in controls, control

    def test_the_status_strip_is_never_hidden_and_sized_once_per_game(self) -> None:
        """Q725: the strip keeps one height for the whole game, so the board never moves
        as the status changes. A game between two people keeps a line more from the start
        (its class is in the markup, never added later), and `fitBoard` is the one place
        that marks a strip narrow, once a size, from its width."""
        from page_text import PAGE

        plain = _in_page("", "frameHtml(false, 'sandbox', false)")
        assert '<div class="row frame-status" data-frame="status">' in plain
        between = _in_page("", "frameHtml(false, 'leaderboard', false, "
                                "{you: 0, mayCancel: false})")
        assert '<div class="row frame-status between" data-frame="status">' in between
        # Two fits mark it: `fitBoard`, and its branch for the board alone (Q638, `fitFull`),
        # each from the strip's width in the layout it lays out.
        assert PAGE.count("classList.toggle('narrow'") == 2
        for fit in ("function fitBoard()", "function fitFull("):
            start = PAGE.index(fit)
            assert "classList.toggle('narrow'" in PAGE[start:PAGE.index("\n}\n", start)]
        assert "const STRIP_NARROW = 560;" in PAGE
        # Nothing hides the strip or gives it a height of its own.
        for hidden in ("strip.hidden", "strip.style.display", "strip.style.height",
                       "status.style.height", "status.style.minHeight"):
            assert hidden not in PAGE, hidden

    def test_both_chromes_draw_the_one_frame(self) -> None:
        from page_text import PAGE

        assert PAGE.count("${frameHtml(inline, snapshot.mode, watched, between, (!!snapshot.round || !!snapshot.spy) && !watched, !!snapshot.puzzle)}") == 1
        assert PAGE.count('<svg id="board"') == 1
        inline = _in_page("", "frameHtml(true, '', false)")
        assert 'id="stopPlay"' in inline and 'data-frame="controls"' in inline

    def test_the_bar_has_an_icon_over_each_word_and_keeps_the_words(self) -> None:
        """§3.3: at 375 the bar is icons with a word under each; the words are the ones
        the bar had (lower case, Q548's to change), each in a span the arming buttons
        rewrite so the icon stays."""
        html = _in_page("", "frameHtml(true, 'sandbox', false)")
        for name, words in (("stopPlay", "back to editing"), ("undo", "take back"),
                            ("flip", "flip"), ("resign", "resign")):
            button = html[html.index(f'id="{name}"'):]
            button = button[:button.index("</button>")]
            assert 'class="ico"' in button and f'<span class="w">{words}</span>' in button, name
        between = _in_page("", "frameHtml(false, 'sandbox', false, { you: 0, mayCancel: true })")
        for words in ("offer a draw", "cancel the game", "rematch"):
            assert f'<span class="w">{words}</span>' in between, words

    def test_an_armed_button_changes_its_word_and_keeps_its_icon(self) -> None:
        seen = _in_page("", """(() => {
          const word = { textContent: 'resign' };
          const button = { textContent: 'x', querySelector: () => word };
          sayOn(button, 'really resign?');
          const bare = { textContent: 'old' };
          sayOn(bare, 'new');
          return [word.textContent, button.textContent, bare.textContent];
        })()""")
        assert seen == ["really resign?", "x", "new"]

    def test_the_board_carries_a_layer_for_its_names_and_a_banner(self) -> None:
        html = _in_page("", "frameHtml(false, '', false)")
        board = html[html.index('class="board-frame"'):html.index('data-frame="controls"')]
        assert 'id="coords"' in board and 'id="ended" hidden' in board

    def test_the_banner_says_the_verdict_in_its_two_halves(self) -> None:
        """The banner's heading and reason are the strip's own sentence, cut in two: no
        word of the verdict is new."""
        setup = "SESSION = { seats: ['human', 'random'], ruleset: { sides: ['white', 'black'] } };"
        seen = _in_page(setup, """[
          [{ winner: 0, reason: 'checkmate' }, { winner: null, reason: 'repetition' },
           { winner: null, reason: 'cancelled' }].map(o => [verdictParts(o), verdictText(o)]),
          resultFor({ winner: 0, reason: 'checkmate' }),
          resultFor({ winner: 1, reason: 'checkmate' }),
          resultFor({ winner: null, reason: 'repetition' })]""")
        (mate, mate_text), (drawn, drawn_text), (off, off_text) = seen[0]
        assert mate == {"head": "White (you) wins", "how": "by checkmate"}
        assert mate_text == "White (you) wins by checkmate."
        assert drawn == {"head": "Drawn", "how": "by repetition"} and drawn_text == "Drawn by repetition."
        assert off["how"] == "" and off_text == off["head"]
        assert seen[1:] == ["won", "lost", ""]

    def test_nobody_or_both_at_the_board_is_nobodys_win(self) -> None:
        for seats in ("['human', 'human']", "['random', 'random']"):
            setup = f"SESSION = {{ seats: {seats}, ruleset: {{ sides: ['white', 'black'] }} }};"
            assert _in_page(setup, "resultFor({ winner: 0, reason: 'checkmate' })") == ""
        between = ("SESSION = { seats: ['human', 'human'], ruleset: { sides: ['white', 'black'] },"
                   " correspondence: { you: 1, players: ['ada', 'bo'] } };")
        assert _in_page(between, "resultFor({ winner: 1, reason: 'checkmate' })") == "won"

    def test_starting_another_game_is_not_in_the_control_row(self) -> None:
        html = _in_page("", "frameHtml(false, '', false)")
        for other in ('id="again"', 'id="swap"', 'id="watch"'):
            assert other not in html


class TestASandboxGameCanBeEnded:
    """Q578, the first-visit study's F12: a sandbox game had take back and new game and no
    way to close a lost game with a result. Its control row now carries the resignation a
    rated game has (the store records it and rates nothing: `test_site_matches.py`)."""

    def test_a_sandbox_game_offers_both_a_take_back_and_a_resignation(self) -> None:
        html = _in_page("", "frameHtml(false, 'sandbox', false)")
        controls = html[html.index('data-frame="controls"'):]
        assert 'id="undo"' in controls and 'id="resign"' in controls
        assert "a sandbox game counts for nothing" in controls
        assert "your rating moves" not in controls

    def test_a_rated_resignation_still_says_the_rating_moves(self) -> None:
        html = _in_page("", "frameHtml(false, 'leaderboard', false)")
        assert "your rating moves" in html and "counts for nothing" not in html

    def test_nobody_at_the_board_and_no_store_offer_none(self) -> None:
        """A watch is nobody's to give up; the local application has no store to record
        a resignation in, so its board is as it was."""
        assert 'id="resign"' not in _in_page("", "frameHtml(false, 'sandbox', true)")
        assert 'id="resign"' not in _in_page("", "frameHtml(false, '', false)")


class TestTheStartButtonSaysWhatItStarts:
    """Q578, the study's F4: *start a game* said nothing about whether the game would
    count or who was opposite, a reader's first game was sandbox without its noticing, and
    the rated option's sentence was cut off at 375px."""

    ROSTER = ("ROSTER = [{ agent: 'minimax:nodes=1600', name: 'medium',"
              " label: 'medium · looks at 1,600 positions before each move' }];")

    def test_the_words_name_the_mode_and_the_opponent(self) -> None:
        def said(seats: list[str], mode: str) -> str:
            return _in_page(self.ROSTER, f"startWords({json.dumps(seats)}, {json.dumps(mode)})")

        human_first = ["human", "minimax:nodes=1600"]
        assert said(human_first, "leaderboard") == "start a rated game against medium"
        assert said(human_first, "sandbox") == "start a practice game against medium"
        # The local application has no modes, so it names the opponent alone.
        assert said(human_first[::-1], "") == "start a game against medium"
        assert said(["human", "human"], "") == "start a game for two at this board"
        assert said(["minimax:nodes=1600"] * 2, "") == (
            "start a game: medium against medium"
        )

    def test_the_menu_is_two_short_options_and_a_line_that_wraps(self) -> None:
        payload = _game_payload("abc123def456", site=True)
        drawn = _in_page("", f"modePicker({json.dumps(payload)})")
        options = re.findall(r"<option[^>]*>([^<]*)</option>", drawn)
        assert options == ["no &mdash; practice", "yes &mdash; rated"]
        assert 'class="u-maxw-100p"' in drawn
        # The sentence is under the menu, where it can wrap; sandbox until chosen.
        assert 'id="modeSays"' in drawn
        assert "This is a practice game" in drawn
        assert '<option value="sandbox" selected>' in drawn

    def test_the_last_choice_is_remembered(self) -> None:
        payload = _game_payload("abc123def456", site=True)
        kept = ("window.localStorage = { getItem: key => key === 'cvs-play-mode'"
                " ? 'leaderboard' : null, setItem: () => {} };")
        drawn = _in_page(kept, f"modePicker({json.dumps(payload)})")
        assert '<option value="leaderboard" selected>' in drawn
        assert "This one is rated, no undo" in drawn
        # Anything else stored, or storage refused, is the default.
        odd = "window.localStorage = { getItem: () => 'nonsense', setItem: () => {} };"
        assert '<option value="sandbox" selected>' in _in_page(
            odd, f"modePicker({json.dumps(payload)})")
        refused = "Object.defineProperty(window, 'localStorage', { get() { throw 1; } });"
        assert '<option value="sandbox" selected>' in _in_page(
            refused, f"modePicker({json.dumps(payload)})")

    def test_pressing_start_sends_the_mode_names_it_and_keeps_it(self) -> None:
        digest = "abc123def456"
        press = """
          window.localStorage = { kept: {}, getItem(k) { return this.kept[k] || null; },
                                  setItem(k, v) { this.kept[k] = v; } };
          const mode = document.getElementById('mode');
          mode.value = 'leaderboard';
          document.getElementById('seat0').value = 'human';
          document.getElementById('seat1').value = 'minimax:nodes=1600';
          mode.onchange();
          VIEW._html += '<i id="said">' + document.getElementById('start').textContent
            + '|' + document.getElementById('modeSays').textContent + '</i>';
          document.getElementById('start').onclick();
          VIEW._html += '<i id="kept">' + window.localStorage.kept['cvs-play-mode'] + '</i>';
        """
        report = _page_on(True, f"#/game/{digest}", _game_payload(digest, site=True),
                          press=press)
        said = re.search(r'<i id="said">([^<]*)</i>', report["html"]).group(1)
        assert said.startswith("start a rated game against ")
        assert "This one is rated, no undo" in said
        sent = [body for path, body in report["sent"] if path == "/api/session"]
        assert sent and sent[0]["mode"] == "leaderboard"
        assert '<i id="kept">leaderboard</i>' in report["html"]


class TestAVisitorIsToldTheGameIsNotKept:
    """Q578, the study's F5: two readers played signed out, made an account, and met a
    profile reading *0 games tried* that nothing had warned them of."""

    def test_the_signed_out_panel_says_so_and_the_signed_in_one_does_not(self) -> None:
        payload = _game_payload("abc123def456", site=True)
        out = {**payload, "standing": {**payload["standing"], "signedIn": False}}
        assert "isn't kept" in _said(_in_page("", f"modePicker({json.dumps(out)})"))
        unrated = {**out, "standing": {**out["standing"], "rated": False}}
        assert "isn't kept" in _said(_in_page("", f"modePicker({json.dumps(unrated)})"))
        assert "isn't kept" not in _in_page("", f"modePicker({json.dumps(payload)})")
        # The local application sends no standing and says nothing about accounts.
        local = _game_payload("abc123def456", site=False)
        assert _in_page("", f"modePicker({json.dumps(local)})") == ""


class TestADraftsLinkCanBeCopied:
    """Q578, the study's B3: a draft's link already works and is the one way anybody else
    reaches it; neither the save's answer nor the draft's page offered it."""

    def test_the_save_answer_carries_the_copy_button(self) -> None:
        said = _in_page(_kit(), "savedHtml(" + _save_answer() + ")")
        assert 'data-copy-link="abc123abc123"' in said

    def test_the_link_is_the_routers_game_address_on_this_origin(self) -> None:
        setup = "location.origin = 'https://example.test';"
        assert _in_page(setup, "gameLink('abc123')") == "https://example.test/game/abc123"
        from page_text import CORE

        assert "parts[0] === 'game'" in CORE

    def test_one_handler_serves_every_copy_button(self) -> None:
        from page_text import PAGE

        assert PAGE.count("closest('[data-copy-link]')") == 1
        assert "navigator.clipboard.writeText(link)" in PAGE
        render = PAGE[PAGE.index("function render() {"):]
        assert render.index("wireCopyLinks();") < render.index("\n}")


class TestATemplatesSentenceIsNotTheDrafts:
    """Q578, the study's F9: a draft saved from a template kept the template's sentence,
    so the library described the designer's game as the template."""

    def test_a_template_starts_with_no_description(self) -> None:
        spec = {"name": "a game with a king", "description": "the template's sentence"}
        assert _in_page("", f"fromTemplate({json.dumps(spec)}).description") == ""
        assert _in_page("", f"fromTemplate({json.dumps(spec)}).name") == spec["name"]

    def test_the_field_asks_for_one(self) -> None:
        from page_text import PAGE

        field = PAGE[PAGE.index('<textarea id="dDesc"'):]
        field = field[:field.index("</textarea>")]
        assert 'placeholder="What is this game?' in field


class TestWhenTheAgentFails:
    """A failing agent has to say so, in the one place a person is looking.

    Reported from a real session: every agent move raised, the page put an error box at
    the top, and the status underneath the board went on saying "thinking" with a spinner
    for as long as anybody watched. `onSquare` refuses human input while `awaitingAgent`
    is set, so the game was unrecoverable *and* the screen said it was working.
    """

    def test_the_page_stops_saying_thinking(self) -> None:
        """The page is a string and nothing type-checks it, so this reads the string.

        Narrow on purpose: it asserts that the failure path of the agent call clears the
        status and offers a way back, which is the whole content of the fix.
        """
        from page_text import PAGE

        start = PAGE.index("function maybeAgent()")
        body = PAGE[start:PAGE.index("function pieceOf(", start)]
        assert ".catch(error =>" in body, "the failure path must be handled, not swallowed"
        assert "could not move" in body, "the status must say what happened"
        assert "try that move again" in body, "a stalled game needs a way out"

    def test_the_library_can_be_narrowed_by_the_outside_engine(self) -> None:
        """The control, the filter it drives, and the state it reads.

        Three separate places in three modules -- the select in `browse`, the default in
        `core`, the reset in `browse` -- and a filter missing any one of them is a menu
        that does nothing, silently, which is the failure this file exists to catch.
        """
        from page_text import PAGE

        assert "<select id=\"fairy\">" in PAGE
        assert "BROWSE.fairy === 'yes' && game.fairyTranslates !== true" in PAGE
        assert "BROWSE.fairy === 'no' && game.fairyTranslates !== false" in PAGE
        # Offered, counted as a filter, and cleared with the rest.
        assert "'stage', 'scored', 'playable', 'fairy'" in PAGE
        assert "BROWSE.fairy = '';" in PAGE
        assert "(BROWSE.fairy ? 1 : 0)" in PAGE

    def test_the_game_page_says_why_the_outside_engine_is_absent(self) -> None:
        """Three ways of being absent, and the page must not guess between them.

        Rules with no counterpart will never be playable there whatever anybody installs;
        rules that translate on a machine with no engine want a download; rules that
        translate but that the installed build plays differently are a fact about that
        build. The page used to collapse the last two into one sentence because nothing
        on it started an engine -- but the roster had, and its verdict was memoised and
        dropped. Now `game.fairyCheck` is that verdict and each branch reads it.
        """
        from page_text import PAGE

        start = PAGE.index("function fairyNote(")
        body = PAGE[start:PAGE.index("function pieceArt(", start)]
        # Absent only when it is absent: the picker itself is the authority.
        assert "seats || []).some(o => o.store === 'fairy')" in body
        assert "game.fairyCheck" in body
        assert "no counterpart over there" in body
        assert "no Fairy-Stockfish is" in body and "installed on this machine" in body
        # The build is named and the verification's own words and position are shown.
        assert "check.ok === false" in body
        assert "${esc(check.engine)}" in body
        assert "check.reasons" in body and "check.fen" in body
        assert "did not play the same game" in body

    def test_the_card_tag_says_the_rules_translate_and_not_that_it_plays(self) -> None:
        """The tag's text is the one place the listing's claim is read at a glance.

        It said "Fairy-Stockfish" bare, and a reader took that as "plays it", opened
        `sardines`, and found no such opponent: the 2021 release cannot promote a custom
        pawn. The tag was answering `supports` and had not said so.
        """
        from page_text import PAGE

        start = PAGE.index("function fairyTag(")
        body = PAGE[start:PAGE.index("\n}\n", start)]
        assert 'game.fairyTranslates !== true' in body
        assert ">translates to Fairy-Stockfish</span>" in body
        assert '">Fairy-Stockfish</span>' not in body

    def test_the_leaderboard_names_the_opponents_it_has_not_rated(self) -> None:
        """A picker table is a snapshot of the roster at rating time; the roster is live.

        Fairy-Stockfish on chess: offered in the picker, absent from "how hard each
        opponent is", with no sign it had ever been asked about, because the table was
        rated before the engine joined and the panel drew the rows it had. Both
        directions of the difference are named now -- and only for the picker's own
        table, because the rated ladder's rows are the canonical rungs, which are
        not picker entries and would all read as "no longer offered".
        """
        from page_text import PAGE

        start = PAGE.index("function strengthHtml(")
        body = PAGE[start:PAGE.index("\n}\n", start)]
        assert "const picker = !!(table && table.kind === 'picker')" in body
        # A practice-only player (Q683) is never rated, so it is never "not rated yet".
        assert ("ROSTER.filter(o => o.agent !== 'human' && !o.practiceOnly"
                " && !rated.has(o.agent))") in body
        assert "not rated yet, because" in body and "not in the picker when this table" in body
        assert "not rated yet)" in body  # the button counts them
        assert "(no longer available)" in body
        assert "picker && !entry" in body
        # The ladder's table keeps its own one-sentence account of what it lacks.
        assert "The ones with a plan are not in it" in body

    def test_the_menu_says_the_rules_translate_and_not_that_it_plays(self) -> None:
        """The one wording that would be a lie, in the one place a person reads it."""
        from page_text import PAGE

        start = PAGE.index("<select id=\"fairy\">")
        menu = PAGE[start:PAGE.index("</select>", start)]
        assert "the rules translate" in menu
        assert "no counterpart" in menu
        assert "plays" not in menu and "playable" not in menu

    def test_the_stale_build_error_names_the_rebuild(self) -> None:
        """The one shape of failure that actually happened, and the message it earns.

        The agent tuple is the one thing about the boundary both sides have to agree on
        from memory, so a `_core` compiled before a slot was added rejects every agent --
        deep inside a request, with a message about tuple lengths and nothing about
        builds. `uv sync` does not notice that `rust/` changed, which is exactly how a
        tree gets into that state.
        """
        from chess_variant_sandbox.tournament.play import _STALE_CORE

        rendered = _STALE_CORE.format(error="expected tuple of length 6, got 7")
        assert "expected tuple of length 6, got 7" in rendered
        assert "--reinstall-package chess-variant-sandbox" in rendered
        assert "maturin develop" in rendered


class TestTheBoardAnimation:
    """Also read off the page string, and for the same reason: nothing type-checks it.

    Two properties, neither of them a number anybody should have to find by playing. Every
    duration the board uses comes from one table -- so "slow it down" is one edit and not
    a hunt through six literals. And a capture happens when the capturing piece *arrives*
    rather than when it sets off, which is the defect this replaced: the victim faded over
    260ms while the move itself took up to 420ms, so on anything longer than a step the
    piece being taken was gone before anything reached it, and a capture read as a piece
    quietly giving up rather than as a piece being taken.
    """

    def animation(self) -> str:
        from page_text import PAGE

        # LP4: the table and the motion itself are `motion.js`, which every page that draws a
        # game moving shares; the play board's diff of two positions is `board.js`'s.
        start = PAGE.index("const PACE = {")
        motion = PAGE[start:PAGE.index("\n}\n", PAGE.index("function portalFlight(")) + 3]
        board = PAGE.index("// " + "-" * 63 + " the animation")
        # And a round's reveal (Q918c, `simultaneous.js`), which reads the same table.
        rounds = PAGE.index("function animateRound(")
        return motion + PAGE[board:PAGE.index("the side panel", board)] + PAGE[
            rounds:PAGE.index("\n}\n", rounds)]

    def pace(self) -> dict[str, float]:
        import re

        table = self.animation()
        table = table[: table.index("};")]
        return {name: float(value)
                for name, value in re.findall(r"(\w+):\s*([\d.]+)", table)}

    def test_every_duration_is_named(self) -> None:
        """A literal here is a timing nobody can find. The table is the only place a
        millisecond is stated, and it has no entries nothing reads."""
        import re

        body = self.animation()
        assert re.search(r"duration:\s*[\d.]", body) is None, "a duration escaped the table"
        assert re.search(r"delay:\s*[\d.]", body) is None, "a delay escaped the table"
        used = set(re.findall(r"PACE\.(\w+)", body))
        assert used == set(self.pace()), "the timing table and its readers disagree"

    def test_the_strike_lands_while_the_move_is_still_happening(self) -> None:
        """`impact` is a share of the travel, so it has to be inside it. At 1.0 the
        victim reacts after the capturing piece has stopped, which is the sequence the
        eye reads as two separate events rather than as one."""
        pace = self.pace()
        assert 0 < pace["impact"] < 1
        assert pace["travelMin"] < pace["travelMax"]
        assert pace["strike"] > 0 and pace["burst"] > 0

    def test_the_capture_waits_for_the_piece_that_causes_it(self) -> None:
        """The arrival time is the travel's own duration rather than a second guess at
        it, and both halves of the capture are delayed by it -- including the ring, which
        is the half that reads at a glance. The explosion animation has the same
        property (LP4, the landing mock-up's blast): the fireball over the area waits for the
        blow, and so does each piece it took, which burns a breath after it."""
        body = self.animation()
        assert "return duration;" in body, "travel has to say how long it took"
        assert "const flight = travel(" in body and "flight * PACE.impact" in body
        assert body.count("delay: blow.at") == 2, "the capture's two halves wait"
        assert "const start = blow.at + PACE.blastLag;" in body, "a piece the blast took waits"
        assert "blows.has(center) ? blows.get(center).at : lastBlow" in body, "so does its fire"
        # A victim with no attacker of its own -- an en passant pawn, a bystander in an
        # explosion -- still dies of the move that killed it rather than instantly.
        assert "blows.get(sq) || { at: lastBlow, dx: 0, dy: 0 }" in body

    def test_a_move_nobody_asked_to_see_waits_for_the_one_before_it(self) -> None:
        """Reported as "it only works when black captures", which is exactly how it looks
        from the near side of the screen.

        A draw replaces the board's contents outright, so it destroys whatever the last
        move was still animating. There is one draw a person does not ask for -- the
        agent's reply -- and measured against this roster an agent answers in 43-542ms,
        while a move takes 630ms to travel and a capture's strike does not *start* until
        the capturing piece is 80% of the way there. So a person's captures were wiped
        before any part of them had been drawn and the agent's played out in full, because
        nothing redraws the board on a person's turn. The wait is on the paint and not on
        the request, so the agent still thinks while the board finishes moving.
        """
        from page_text import PAGE

        # The animation says how long it will take, and the draw records it. `beats` is
        # what `previous` became: it is worked out from the previous position and is null
        # in exactly the draws that had nothing to animate.
        assert "return busy;" in self.animation()
        assert "BUSY_UNTIL = beats\n    ? performance.now() + animateChange(" in PAGE
        assert "setTimeout(paint, Math.max(0, BUSY_UNTIL - performance.now()))" in PAGE

        agent = PAGE.index("function maybeAgent()")
        assert "whenSettled(() => { answered(); if (mine()) applySnapshot(s); })" in (
            PAGE[agent:agent + 2400]), \
            "the agent's reply is the draw that waits"
        # A person's own move is not one of them: it is the reason they clicked.
        send = PAGE.index("function send(move)")
        assert "then(s => applySnapshot(s))" in PAGE[send:send + 400]

    def test_nothing_animated_means_nothing_to_wait_for(self) -> None:
        """Every early return in `animateChange` has to report a duration, or the board
        is held busy on the value of a function that returned nothing. `undefined` there
        is not a small mistake: `BUSY_UNTIL` becomes `NaN`, `Math.max(0, NaN)` is `NaN`,
        and `setTimeout` fires immediately -- which is the bug, silently back."""
        import re

        body = self.animation()
        head = body[body.index("function animateChange("):body.index("function animatePhase(")]
        # `\b` here was a *backspace* rather than a word boundary -- an escape a raw
        # string does not have -- so the pattern was `\\x08return` and matched nothing
        # in any page. The guard has been vacuous since it was written.
        assert re.search(r"\breturn\s*;", head) is None, "an early return says nothing"
        # `animateChange`'s one, a board that changed shape (reduced motion returns
        # `portalStill`'s duration, Q836), and the trip through a portal's three: an end
        # with no glyph to pulse, and a move `portalStill` has no crossfade for, twice; and
        # since Q843d the trip through a fold's one, a fold with no pool to pulse. Since LP4
        # the end with no glyph is `motion.js`'s `portalPulse`, shared with the landing.
        assert head.count("return 0;") == 4
        pulse = body[body.index("function portalPulse("):]
        assert "if (!glyph || typeof glyph.animate !== 'function') return 0;" in pulse
        # `animateMove`'s three: nothing changed, a pure rotation, and the end of it --
        # all through the one expression that builds what the phase reads back.
        assert head.count("return nothing();") == 3
        assert "const nothing = () => ({ busy, landed, doomed });" in head

    def test_stillness_is_still_honoured(self) -> None:
        """A capture that flashes a ring is exactly what `prefers-reduced-motion` is
        asking not to be shown."""
        from page_text import PAGE

        start = PAGE.index("function animateChange(")
        # Q836: with motion reduced nothing moves; a trip through a portal alone gets a
        # crossfade, opacity only (`tests/test_portal_travel.py::TestReducedMotion`).
        assert ("if (STILL.matches) return portalStill(svg, before, after, last, cell, view);"
                in PAGE[start:start + 900])


class TestWhatThePageIsTold:
    """The contract between the API and the one page that reads it.

    The page is a string in `web/page/` and nothing type-checks the field names it
    reaches for, so a rename here is silent until somebody opens a tab. These name the
    fields the interface actually depends on.
    """

    def test_a_cell_travels_as_words_as_well_as_an_identifier(self, chess_game) -> None:
        """`1-1-0-0-1-0-0` is what the archive files a game under, and it is what the
        library showed a reader. Both now travel: the label is an identifier and the
        bins are the same facts in English. It grows a component when the grid grows an
        axis -- schema 3 added the capture axis and schema 5 the one for a board with no
        edges -- and every earlier label reads as that axis's bin 0, which is what makes
        an axis inert on everything already filed."""
        from chess_variant_sandbox.loop.grid import bins_of, descriptor

        card = chess_game.summary()
        assert card["cell"] == "1-1-0-0-1-0-0-0"
        assert card["cellBins"] == list(bins_of(descriptor(chess_game.spec)))
        assert card["cellBins"][3] == "mate"

        axes = library.detail(chess_game)["axes"]
        assert [a["axis"] for a in axes] == ["board", "army", "hand", "ending",
                                             "rules", "capture", "edges", "taking"]
        assert all(a["asks"] and a["bin"] and a["means"] for a in axes)

    def test_the_library_carries_the_legend_once(self, games) -> None:
        """A cell means nothing without it, and it is the same for every ruleset."""
        from chess_variant_sandbox.loop import grid

        report = library.coverage(games)
        assert len(report["axes"]) == len(grid.AXES)
        assert all(len(axis["bins"]) >= 2 for axis in report["axes"])
        assert all(b["means"] for axis in report["axes"] for b in axis["bins"])
        for cell in report["occupied"]:
            assert len(cell["bins"]) == len(report["axes"])

    def test_the_rules_arrive_as_parts_and_as_a_block(self, chess_game) -> None:
        """The block is what a model reads; the parts are what a player reads next to
        the board. Same sentences, or the two readers disagree."""
        detail = library.detail(chess_game)
        assert detail["rules"].startswith("chess - a two-player game")
        assert detail["ruleLines"] and detail["setupLines"]
        for line in detail["ruleLines"] + detail["setupLines"]:
            assert line in detail["rules"]

    def test_a_piece_carries_what_it_does_and_what_it_looks_like(self, chess_game) -> None:
        detail = library.detail(chess_game)
        pawn = next(p for p in detail["pieces"] if p["id"] == "P")
        king = next(p for p in detail["pieces"] if p["id"] == "K")
        assert pawn["name"] == "pawn"
        assert len(pawn["moves"]) == 3          # one step, double step, capture
        assert king["royal"] is True and pawn["royal"] is False
        # A newcomer reads the sentence; the vectors it was written from travel beside
        # it, one per move, for the (i) the page puts on the end of each line.
        assert pawn["moves"][0] == "one square forward, not to take"
        assert len(pawn["descriptors"]) == len(pawn["moves"])
        assert pawn["descriptors"][0] == (
            "may only move along (+0,+1), exactly 1 square, path must be clear"
        )
        # Both seats, because a single-colour set tells the players apart by rotating.
        assert len(pawn["icons"]) == 2
        assert all(icon and icon["href"].startswith("data:") for icon in pawn["icons"])

    def test_the_score_is_explained_under_the_weights_that_produced_it(self) -> None:
        """Not under the current default. A score records the version it was computed
        under and versions are never edited, so explaining a v3 number with v5's curves
        would describe a scoring rule that number was never subject to -- and the old
        versions are exactly the ones still sitting in ``runs/``."""
        from chess_variant_sandbox.metrics.scoring import DEFAULT_WEIGHTS

        guide = library.score_guide({"weights_version": "v3", "total": 0.5})
        assert guide is not None
        assert guide["version"] == "v3" != DEFAULT_WEIGHTS
        # Every component the page will print a row for has something to say about it.
        for name, entry in guide["components"].items():
            assert entry["what"] and entry["how"] and entry["reading"], name
            assert entry["curve"]["kind"] and entry["curve"]["points"]
            assert entry["family"] in guide["families"]
        assert guide["howItWorks"] and guide["gates"]

    def test_a_score_from_a_version_this_checkout_lost_still_shows(self) -> None:
        """The number is real even when the explanation is gone. Absent guide, not a
        crash and not a made-up one."""
        assert library.score_guide({"weights_version": "v99", "total": 0.5}) is None
        assert library.score_guide(None) is None

    def test_an_unscored_game_asks_for_no_guide(self, tmp_path) -> None:
        spec = load_spec(RULESETS / "gardner.json")
        game = library.Game(hash=spec_hash(spec), name="g", description="",
                            path=RULESETS / "gardner.json", source="committed", spec=spec)
        assert library.detail(game)["scoreGuide"] is None


class TestTheBoardIsToldTheWholeGame:
    def test_territory_reaches_the_board(self) -> None:
        """Three committed rulesets are *won* on a count of claimed squares, and the
        board drew none of it -- so the score of the game being played was the one thing
        not on screen. The rule travels too, because a colouring that decides nothing
        and a colouring that decides everything look identical."""
        spec = load_spec(RULESETS / "rps-v2.json")
        game = library.Game(hash=spec_hash(spec), name="rps territory", description="",
                            path=RULESETS / "rps-v2.json", source="committed", spec=spec)
        session = Sessions().start(game, ("human", "human"))
        snapshot = session.snapshot()

        area = spec["board"]["width"] * spec["board"]["height"]
        assert len(snapshot["colour"]) == area
        assert snapshot["ruleset"]["territory"] == {"mode": "permanent", "win_at": 41}
        assert not any(snapshot["colour"])

        after = session.play(snapshot["moves"][0]["move"])
        assert sum(1 for c in after["colour"] if c) == 1

    def test_a_board_without_territory_says_so_by_absence(self, chess_game) -> None:
        snapshot = Sessions().start(chess_game, ("human", "human")).snapshot()
        assert snapshot["colour"] == []
        assert snapshot["ruleset"]["territory"] is None
        assert snapshot["remaining"] == 1

    def test_a_longer_turn_reaches_the_board(self) -> None:
        """`rules.turn` has no committed ruleset, so it gets a probe -- and a player
        owed a second move has to be told, or the board looks like it ignored a click."""
        import copy

        spec = copy.deepcopy(load_spec(RULESETS / "gardner.json"))
        spec["schema_version"] = 4
        spec["rules"]["turn"] = {"moves": 2}
        game = library.Game(hash=spec_hash(spec), name="two moves", description="",
                            path=Path("<in-memory>"), source="committed", spec=spec)
        session = Sessions().start(game, ("human", "human"))
        assert session.snapshot()["remaining"] == 2
        after = session.play(session.snapshot()["moves"][0]["move"])
        assert after["remaining"] == 1
        assert after["sideToMove"] == 0, "the same player is still to move"


class TestSides:
    """Which side you play, which is a real choice and used to be an unlabelled one.

    Two anonymous dropdowns could always express it -- seat yourself second and put an
    agent first -- and never said which seat was which side, or what either side was.
    On a symmetric game that costs the first move. On horde one seat is thirty-six pawns
    and the other is an ordinary army, and choosing between them is choosing the game.
    """

    def test_a_person_can_take_the_second_seat(self, horde) -> None:
        """The agent has the move before the person does, and the person is player 1."""
        session = Sessions().start(horde, ("minimax:nodes=400", "human"), seed=3)
        opening = session.snapshot()
        assert opening["sideToMove"] == 0
        assert opening["awaitingAgent"] is True, "the person is not on the move"

        replied = session.play_agent()
        assert replied["sideToMove"] == 1
        assert replied["seats"][replied["sideToMove"]] == "human"
        assert replied["awaitingAgent"] is False
        assert replied["moves"], "the person was given nothing to play"

        after = session.play(replied["moves"][0]["move"])
        assert after["history"][-1]["player"] == 1
        assert after["history"][-1]["by"] == "human"

    def test_taking_back_from_the_second_seat_stops_where_you_moved(self, horde) -> None:
        """Undo walks back to a position a person was asked about, and the first such
        position in this seating is *after* the agent's opening move, not before it."""
        session = Sessions().start(horde, ("minimax:nodes=400", "human"), seed=3)
        session.play_agent()
        session.play(session.snapshot()["moves"][0]["move"])
        session.play_agent()

        back = session.undo()
        assert back["seats"][back["sideToMove"]] == "human"
        assert len(back["history"]) == 1, "the agent's opening move is still on the board"

    def test_the_armies_say_what_each_side_is(self, horde) -> None:
        """The measurement the picker shows, and the reason it shows it."""
        first, second = library.armies(horde)
        assert [piece["id"] for piece in first] == ["P"]
        assert first[0]["count"] == 36
        assert {piece["id"] for piece in second} == set("PRNBQK")
        assert sum(piece["count"] for piece in second) == 16
        assert first != second

    def test_a_derived_army_is_read_off_the_board(self, chess_game) -> None:
        """Chess states one army and mirrors the other; shogi rotates its second.

        Both are absent from ``setup.placements``, so an army summary built from the
        file would have shown one side of those games and nothing of the other.
        """
        first, second = library.armies(chess_game)
        assert first == second
        assert sum(piece["count"] for piece in first) == 16
        # The piece's own name where the spec gives one, its letter where it does not,
        # which is the fallback the rest of the interface already draws.
        assert first[0]["id"] == "P" and first[0]["name"] == "pawn"
        # Most numerous first, so the line reads as an army rather than as an inventory.
        assert [piece["count"] for piece in first] == sorted(
            (piece["count"] for piece in first), reverse=True
        )

    def test_a_wild_piece_is_in_neither_army(self, boar_game) -> None:
        """Q41: ``armies`` indexed ``counts[(value - 1) // stride]`` into two rows.

        Schema 9 put a third owner on the board, so the expression yields 2 and the page
        raised for any ruleset with ``rules.npcs``. The row is counted now and then
        dropped, because the question the picker asks is *which seat do I take* and a
        boar stands on the same squares whichever seat that is -- a third army offered
        beside two seats would read as a side nobody can play.
        """
        armies = library.armies(boar_game)
        assert len(armies) == 2
        first, second = armies
        assert sum(piece["count"] for piece in first) == 16
        assert first == second, "chess mirrors its army and a boar belongs to neither"
        assert not any(piece["id"] == "W" for side in armies for piece in side)
        # And the boar is not hidden from the page: it is a declared piece with a
        # drawing and a description, said where every other piece's is.
        assert any(piece["id"] == "W" for piece in library.pieces(boar_game))

    def test_the_sides_are_named_after_what_the_board_draws(self, games) -> None:
        """The two colour words are the palette's claim, not chess's.

        A single-colour set tells the players apart by turning a piece around, so both
        armies are the same colour and a player told they are black would look for black
        pieces and find the opponent's as well.
        """
        by_name = {game.name: game for game in games}
        assert library.sides(by_name["chess"]) == ("white", "black")
        assert library.sides(by_name["horde"]) == ("white", "black")

        shogi = by_name["shogi"]
        icons = icons_for(library.engine_for(shogi.spec).spec.piece_names)
        if all(pair[1] is None for pair in icons):
            pytest.skip("the icon library is optional and this checkout has not fetched it")
        assert library.sides(shogi) == ("first", "second")

    def test_a_session_carries_the_names_the_page_will_use(self, chess_game) -> None:
        """The page cannot work them out: it is the *server* that chose the icons."""
        snapshot = Sessions().start(chess_game, ("human", "human")).snapshot()
        assert snapshot["ruleset"]["sides"] == ["white", "black"]

    def test_a_session_carries_the_squares_a_piece_promotes_on(self, chess_game) -> None:
        """Compiled and per player. The spec says ``"last"`` once from the first
        player's side; what reaches the board is the pair of bands."""
        snapshot = Sessions().start(chess_game, ("human", "human")).snapshot()
        assert snapshot["ruleset"]["zones"] == [list(range(56, 64)), list(range(8))]


class TestTheBoardTurnsAround:
    """Playing the second seat down a board with your own pieces at the far end is
    playing it upside down. The page is a string and nothing type-checks it, so these
    read the string -- narrowly, for the two things that make the flip work at all.
    """

    def test_there_is_one_coordinate_mapping(self) -> None:
        """`drawBoard` places the squares and `animateChange` flies pieces between them.

        They computed the same coordinate separately, which is harmless while the board
        only ever faces one way -- and sends every piece off towards the corner it used
        to be in the moment it does not.
        """
        from page_text import PAGE

        assert PAGE.count("function squareXY(") == 1
        assert PAGE.count("H - 1 -") == 1, (
            "the rank flip belongs in squareXY and nowhere else"
        )

    def test_the_board_can_be_turned_by_hand(self) -> None:
        """Two agents playing each other have no seat to orient to, and a person
        watching still wants to see it from the other end."""
        from page_text import PAGE

        assert 'id="flip"' in PAGE
        assert "FLIPPED = !FLIPPED" in PAGE


class TestTheBoardWithoutEdges:
    """A wrapping board privileges no boundary, so the page offers the reader a frame.

    The geometry is `viz/wrapped.py`, shared with the offline replay and tested there.
    What is the *page's* is the row that moves the view, the keys that do the same, and
    the animation reaching for the step rather than for the two squares.
    """

    @staticmethod
    def _rules(engine) -> dict:
        from chess_variant_sandbox.viz.wrapped import wrap_payload

        return {
            "width": engine.spec.width, "height": engine.spec.height,
            "nTypes": engine.spec.n_types, "sideStride": engine.spec.side_stride,
            "nFacings": engine.spec.n_facings, "icons": [],
            "pieceIds": list(engine.spec.piece_ids),
            "facingAngles": [list(row) for row in engine.spec.facing_headings()],
            "wrap": wrap_payload(engine.spec.wrap),
        }

    def test_the_view_shifts_by_a_row_at_a_time_and_says_where_it_is(self) -> None:
        """Buttons for the axes that wrap, and shift with an arrow for the same -- the
        bare arrows step through the game, and reading one is the commoner job."""
        from page_text import PAGE

        assert 'id="shift"' in PAGE and "function drawShift(" in PAGE
        assert "SHIFT = { files: SHIFT.files + files" in PAGE
        assert "event.shiftKey && event.key.indexOf('Arrow') === 0" in PAGE
        # The sentence under the board, which is what stops the outline reading as a
        # claim about where the board really begins.
        assert "'showing from '" in PAGE
        # P18: the control's words are a player's, not the frame's.
        assert "slide the board" in PAGE and "shift the view" not in PAGE

    def test_a_square_that_ends_the_game_is_marked_inside_the_cell(self) -> None:
        """The marker goes in the cell rather than over the board, which is what puts
        it on a ghost copy at that ring's paleness and moves it with the view shift.

        Its map comes off the session's compiled rule, so it belongs to the game on
        screen; the drawing itself is `viz/wrapped.py`, shared with the replay.
        """
        from page_text import PAGE

        assert (
            "const SQUARE_MARKS = squareMarks(Object.assign({}, rules, { checked: checkSquares() }),"
            in PAGE
        )
        assert "if (SQUARE_MARKS[square]) out += squareBadge(" in PAGE
        assert PAGE.count("function squareBadge(") == 1, (
            "the badge is the shared module's, not a second drawing of it"
        )

    def test_a_one_way_seam_is_drawn_where_it_really_is(self) -> None:
        """A seam somebody may cross one way and not the other has a position of its
        own, so the board draws it and the outline stands down on that axis. Both are
        `viz/wrapped.py`'s and tested there; what is the page's is handing them the
        compiled rule and drawing the two in that order."""
        from page_text import PAGE

        assert (
            "outlineHtml(cell, FRAME, rules.wrap) + barrierHtml(cell, FRAME, rules.wrap)"
            in PAGE
        )
        assert PAGE.count("function barrierHtml(") == 1, (
            "the seam is the shared module's, not a second drawing of it"
        )

    def test_the_promotion_zone_is_a_mark_behind_a_toggle(self) -> None:
        """A band of squares nothing else on the board shows, on every game that has
        one, wrapped or not -- so its row is its own and not the wrapping board's.

        The toggle is the reader's, which is why it is remembered in the browser rather
        than sent anywhere, and why the board reads `ZONES` on every draw.
        """
        from page_text import PAGE

        assert 'id="marks"' in PAGE and "function drawMarks(" in PAGE
        # Since Q772b the row also holds a portal game's pair-link toggle, so it is empty
        # only where there is neither a promotion nor a portal.
        assert "if (!zones && !links) { row.innerHTML = ''" in PAGE, (
            "a game with no promotion gets no button"
        )
        assert "(zones ? zoneToggleHtml(ZONES) : '')" in PAGE
        assert "let ZONES = zonesShown();" in PAGE
        assert "rememberZones(ZONES);" in PAGE
        # Three things the page remembers for its reader, each with one read and one write
        # and no fourth way in: this toggle, the play panel's last mode (Q578, F4), and
        # that the designer's note to a phone has been said (Q545); and since K3 (Q619)
        # the games this browser has started, so a first game seats the gentlest; and since
        # Q733 a move picked between two people and not sent, kept per game until it is
        # sent or put back; and since Q772b whether the reader has portal pair links on.
        assert PAGE.count("window.localStorage") == 13
        assert PAGE.count("window.localStorage.getItem(LINKS_KEY") == 1
        assert PAGE.count("window.localStorage.setItem(LINKS_KEY") == 1
        assert PAGE.count("window.localStorage.setItem(HELD_KEY") == 1
        assert PAGE.count("window.localStorage.getItem(HELD_KEY") == 1
        assert PAGE.count("window.localStorage.removeItem(HELD_KEY") == 1
        assert PAGE.count("window.localStorage.getItem(PLAYED_KEY") == 1
        assert PAGE.count("window.localStorage.setItem(PLAYED_KEY") == 1
        assert PAGE.count("window.localStorage.getItem(PHONE_NOTE_KEY") == 1
        assert PAGE.count("window.localStorage.setItem(PHONE_NOTE_KEY") == 1
        assert PAGE.count("window.localStorage.getItem(ZONE_KEY") == 1
        assert PAGE.count("window.localStorage.setItem(ZONE_KEY") == 1
        assert PAGE.count("window.localStorage.getItem(MODE_KEY") == 1
        assert PAGE.count("window.localStorage.setItem(MODE_KEY") == 1
        assert "try { return window.localStorage.getItem(ZONE_KEY)" in PAGE
        assert "try { window.localStorage.setItem(ZONE_KEY" in PAGE

    def test_a_piece_crossing_the_seam_arrives_from_outside_the_far_edge(self) -> None:
        """The bug this replaces: a knight leaving by the a-file was drawn sliding back
        across the middle of the board, which is the one thing it did not do.

        One-way cylinder chess, where White leaves by the a-file and comes back on the
        h-file. `Nb1-h2` is two files left and one rank up, through the seam, so the
        piece the board draws on h2 flies in from two files *right* of it -- the copy of
        b1 one repeat along -- and a second copy walks out of the board at b1.
        """
        engine = Engine(load_spec(library.ARCHIVE_DIR / "onewaycylinder.json"))
        before, after, last = _ply(engine, ["Ph2-h4", "Pa7-a6", "Nb1-h2"])
        watched = _watch_ply(self._rules(engine), before, after, last)

        travels = [a for a in watched["animations"] if "translate" in a["frames"]]
        assert len(travels) == 2, "the knight arriving, and the copy of it leaving"
        # Two squares right and one down, on a board whose square is 12.5 viewBox units:
        # the copy of b1 that lies one repeat past the h-file.
        for animation in travels:
            assert animation["frames"].startswith(
                '[{"transform":"translate(25px, 12.5px)"}'
            ), "both copies make the same journey, one repeat apart"
        assert {a["label"] for a in travels} == {"sq15", "made1"}, (
            "the piece on h2 is the one that arrives; the one that leaves is a copy"
        )

        # And they hand over, so there is never a knight at each end of the board.
        fades = [a["frames"] for a in watched["animations"] if "opacity" in a["frames"]]
        assert len(fades) == 2
        assert any(f.startswith('[{"opacity":0}') for f in fades), "one fades in"
        assert any(f.startswith('[{"opacity":1}') for f in fades), "the other fades out"

    def test_an_ordinary_move_on_the_same_board_is_left_alone(self) -> None:
        """The seam is only for the moves that cross it. `Ph2-h4` is one piece, one
        travel, and no copy of anything."""
        engine = Engine(load_spec(library.ARCHIVE_DIR / "onewaycylinder.json"))
        before, after, last = _ply(engine, ["Ph2-h4"])
        watched = _watch_ply(self._rules(engine), before, after, last)

        assert [a["label"] for a in watched["animations"]] == ["sq31"]
        assert "opacity" not in watched["animations"][0]["frames"]


class TestTheOpponentsYouCanPick:
    """The roster is per-ruleset now, and that is the whole reason it exists.

    A node budget is the same player on every board -- which is what lets the index be a
    scale -- and a rule of thumb is not: `agents/features.py` normalises by board size,
    so one digest is not one player on two boards. A module-level tuple could express the
    first and never the second.
    """

    def test_a_ruleset_with_a_store_offers_its_rules_of_thumb(self, chess_game) -> None:
        from chess_variant_sandbox.web import opponents

        roster = opponents.for_game(chess_game.hash)
        groups = {entry.group for entry in roster}
        assert opponents.LADDER_GROUP in groups
        # Named off the constant rather than restated, so renaming a group in the picker
        # is not also a test edit -- what is being held down is that the store's
        # heuristics reach the roster at all.
        assert groups & {group for _, group, _ in opponents.SOURCES}
        # Every one of them says, in English, what it is trying to do. That is what the
        # `plan` field is for and why a heuristic carries prose at all.
        for entry in roster:
            if entry.store != "ladder":
                assert entry.plan, f"{entry.label} has no plan"

    def test_a_searched_valuation_is_offered_only_when_it_was_accepted(
        self, tmp_path, chess_game
    ) -> None:
        """A refused search is the built-in objective underneath.

        A refused search resolves to two empty tuples, so an agent naming one *is*
        `minimax:nodes=6400`. Offering it would hide a negative result behind a label,
        which is the one thing the store exists to make impossible.
        """
        from chess_variant_sandbox.agents import valuations as store_module
        from chess_variant_sandbox.agents.evaluation import Evaluator
        from chess_variant_sandbox.agents.features import builtin_vector, feature_context
        from chess_variant_sandbox.agents.valuations import SEARCH, Valuation, store_path
        from chess_variant_sandbox.engine import Engine
        from chess_variant_sandbox.web import opponents

        compiled = Engine(chess_game.spec).spec
        derived = tuple(Evaluator.for_spec(compiled).values)
        # `valued_for` looks for the *current* procedure, so this has to be filed under
        # it -- and loading registers it, so the registry is restored afterwards rather
        # than left holding a fake under the real procedure's key.
        key = (SEARCH.digest, compiled.hash)
        before = store_module._VALUATIONS.pop(key, None)

        def store(accepted: bool) -> None:
            store_module._VALUATIONS.pop(key, None)
            Valuation(
                ruleset_hash=compiled.hash,
                procedure=SEARCH.digest,
                piece_ids=tuple(compiled.piece_ids),
                vector=tuple(v * 2 for v in derived),
                derived=derived,
                weights=builtin_vector(feature_context(compiled)),
                accepted=accepted,
                accept_score=0.7 if accepted else 0.4,
                accept_error=0.04,
                accept_games=96,
            ).save(store_path(compiled.hash, SEARCH.digest, tmp_path))

        try:
            self._check_valued(opponents, compiled.hash, tmp_path, store)
        finally:
            store_module._VALUATIONS.pop(key, None)
            if before is not None:
                store_module._VALUATIONS[key] = before

    @staticmethod
    def _check_valued(opponents, ruleset_hash, tmp_path, store) -> None:

        store(accepted=False)
        assert not any(
            "values=" in entry.agent
            for entry in opponents.valued_for(ruleset_hash, tmp_path)
        )

        store(accepted=True)
        offered = opponents.valued_for(ruleset_hash, tmp_path)
        assert len(offered) == 1
        assert "values=" in offered[0].agent
        # It says what it did, in English and per piece.
        assert "what the rules imply" in offered[0].plan

    # -- Q142: a store from another vocabulary is refused, by name, with its repair ----
    #
    # The choice was between refusing a stale store and playing it as the built-in
    # objective with a note. Refused, because the fallback is an agent string naming one
    # player while another plays under it, and the on-demand rating would file that
    # player's Elo under the stored name. These build their stores under a hash no ruleset
    # has, so nothing a real game resolves is touched, and unregister what they load.

    STALE = "0123456789ab"

    def _stale_valuation(self, root: Path, accepted: bool) -> None:
        from chess_variant_sandbox.agents.valuations import SEARCH, Valuation, store_path

        Valuation(
            ruleset_hash=self.STALE,
            procedure=SEARCH.digest,
            piece_ids=("a", "b"),
            vector=(2.0, 0.0),
            derived=(1.0, 0.0),
            # What `builtin_vector` wrote before a vocabulary bump: one term short.
            weights=(0.1,) * (len(VOCABULARY) - 1),
            accepted=accepted,
            accept_score=0.7,
            accept_error=0.04,
            accept_games=96,
        ).save(store_path(self.STALE, SEARCH.digest, root))

    @pytest.fixture
    def unregister(self):
        from chess_variant_sandbox.agents import valuations

        yield
        registry = valuations._VALUATIONS
        for key in [key for key in registry if key[1] == self.STALE]:
            del registry[key]

    def test_a_searched_table_from_another_vocabulary_is_refused_with_its_repair(
        self, tmp_path, unregister
    ) -> None:
        """Crazyhouse's accepted table, which the core met as 'minimax has no native
        implementation' the first time anybody rated the picker or played it."""
        from chess_variant_sandbox.agents.valuations import SEARCH, store_path
        from chess_variant_sandbox.web import opponents

        values = tmp_path / "valuations"
        self._stale_valuation(values, accepted=True)
        roster = opponents.for_game(self.STALE, tmp_path / "none", valuations=values)
        assert not any("values=" in entry.agent for entry in roster)

        ruleset = RULESETS / "chess.json"
        said = opponents.refused_stores(self.STALE, ruleset, valuations=values)
        assert len(said) == 1, said
        sentence = said[0]
        # What is stale: the opponent by its picker name, the file, both lengths.
        assert f"'{opponents.VALUED_LABEL}' is not offered" in sentence
        assert store_path(self.STALE, SEARCH.digest, values).as_posix() in sentence
        assert f"carries {len(VOCABULARY) - 1} weights" in sentence
        assert f"has {len(VOCABULARY)} terms" in sentence
        # And the command that repairs it, naming this ruleset's file.
        assert "`chess-variant-sandbox values rulesets/chess.json --force`" in sentence
        # One sentence: a single full stop, at the end.
        assert sentence.count(". ") == 0 and sentence.endswith(".")

    def test_a_stale_store_that_was_refused_anyway_is_not_news(
        self, tmp_path, unregister
    ) -> None:
        """A refused search was never offered, so its length says nothing about the picker."""
        from chess_variant_sandbox.web import opponents

        values = tmp_path / "valuations"
        self._stale_valuation(values, accepted=False)
        assert opponents.refused_stores(self.STALE, None, valuations=values) == []

    def test_the_game_page_carries_what_its_roster_refused(self, games) -> None:
        """The payload's sentences are the roster's own verdict, from the same stores,
        and nothing they name is in the picker beside them.

        On the game the user met it on while it has a searched table on disk, stale or
        not: a re-search that clears it leaves an empty list, which has to agree just
        the same. One game, because a detail payload costs most of a second.
        """
        from chess_variant_sandbox.web import opponents

        searched = {path.name.split("-")[0] for path in Path("valuations").glob("*.json")}
        candidates = sorted(
            (game for game in games if game.hash in searched),
            key=lambda game: (game.name != "crazyhouse", game.name),
        )
        app = App(RULESETS, library.RUNS_DIR)
        checked = 0
        for game in candidates[:1]:
            checked += 1
            payload = app.get(f"/api/game/{game.hash}")
            said = payload["refusedStores"]
            assert said == opponents.refused_stores(game.hash, game.path), game.name
            offered = {entry["label"].split(" (")[0] for entry in payload["opponents"]}
            for sentence in said:
                name = sentence.split("'")[1]
                assert name not in offered, (game.name, sentence)
        assert checked, "no game with a searched table was found to check"

    def test_the_strength_panel_says_what_the_roster_refused(self) -> None:
        """Where the rating used to fail, both with a table and without one."""
        from page_text import PAGE

        panel = PAGE[PAGE.index("function strengthHtml"):PAGE.index("function withCommands")]
        assert "game.refusedStores" in panel
        assert panel.count("+ refused + button") == 2
        # Escaped before the command is set as code, so a sentence is never markup.
        helper = PAGE[PAGE.index("function withCommands"):]
        assert helper.index("esc(line)") < helper.index("u-ff-mono")

    def test_every_stored_objective_offered_on_a_committed_game_is_one_the_core_builds(
        self, games
    ) -> None:
        """The regression net for Q142 itself: ask the core, not a mirror of its check.

        `build_agent` above resolves the Python oracle, which never compared a vector's
        length to anything, so a picker entry the core refuses passed every test here.
        One move at one node per stored objective is enough to reach `agent_spec`.
        """
        from chess_variant_sandbox.agents import native_spec
        from chess_variant_sandbox.engine.engine import _pack
        from chess_variant_sandbox.tournament.play import eval_values
        from chess_variant_sandbox.web import opponents

        asked = 0
        for game in games:
            if game.path.parent != RULESETS:
                continue
            stored = [
                entry for entry in opponents.for_game(game.hash)
                if entry.store == "valuations"
            ]
            if not stored:
                continue
            engine = Engine(game.spec)
            state = _pack(engine.initial_state())
            for entry in stored:
                agent = native_spec(entry.agent, game.hash)
                assert agent is not None, entry.agent
                engine.core.select_move(
                    state, (agent[0], 1, *agent[2:]), eval_values(engine.spec), 0
                )
                asked += 1
        assert asked, "no committed ruleset offers a stored objective to check"

    def test_the_anchor_is_not_offered_twice(self, chess_game) -> None:
        """Index 0 of every set is the built-in evaluator's own vector.

        Offering it beside `minimax:nodes=6400` would be the same player under two
        names, which is exactly the confusion content addressing exists to prevent.
        """
        from chess_variant_sandbox.agents.heuristics import load_set
        from chess_variant_sandbox.web import opponents

        stored = load_set(chess_game.hash)
        assert stored is not None
        anchor = stored.heuristics[0].digest
        assert not any(anchor in entry.agent for entry in opponents.for_game(chess_game.hash))

    def test_every_offered_agent_can_actually_be_built(self, chess_game) -> None:
        """Listing an opponent and being able to play it are the same act.

        `resolve` refuses an unknown digest rather than defaulting to one, so an agent
        string naming a heuristic cannot be built until its store has been loaded. If
        listing did not register, every one of these would raise.
        """
        from chess_variant_sandbox.web import opponents

        engine = Engine(chess_game.spec)
        for entry in opponents.for_game(chess_game.hash):
            if entry.agent == opponents.HUMAN:
                continue
            parse_agent(entry.agent)
            assert build_agent(entry.agent, engine) is not None

    def test_a_ruleset_with_no_store_still_offers_the_ladder(self, tmp_path) -> None:
        """Absent is reported as absent: no heuristics is not an error."""
        from chess_variant_sandbox.web import opponents

        roster = opponents.for_game("nosuchruleset", tmp_path)
        assert [entry.agent for entry in roster] == [agent for agent, _ in opponents.LADDER]

    def test_a_heuristic_seat_starts_a_session(self, tmp_path, chess_game) -> None:
        from chess_variant_sandbox.web import opponents

        app = App(RULESETS, tmp_path)
        seat = next(
            entry.agent
            for entry in opponents.for_game(chess_game.hash)
            if entry.store != "ladder"
        )
        snapshot = app.post("/api/session", {"hash": chess_game.hash, "seats": ["human", seat]})
        assert snapshot["seats"] == ["human", seat]
        with pytest.raises(ValueError):
            app.post("/api/session", {"hash": chess_game.hash,
                                      "seats": ["human", "minimax:h=deadbeefcafe,nodes=6400"]})


class TestAStaleScoreIsNotAScore:
    """The bug this whole mechanism was built for, held down at the level it happened.

    `antichess` displayed as `0.0, gated, -0.6 Elo per doubling` beside a freshly played,
    cleanly monotonic ladder spanning 1,428 Elo. Neither number was wrong. They were two
    different measurements: the verdict came from a year-old run scored under `v5`, the
    ladder from a tournament played that morning, and the library picked them up in two
    independent scans -- the score by newest weights version, the ladder by biggest
    roster -- with nothing requiring them to land in the same directory.

    So the fix is structural rather than a comparison added at the end. One file holds
    the score, the ladder and the basis both were taken under; a reader gets all three or
    none. These tests pin the "none" half, because the "all three" half is the easy one.
    """

    def _world(self, tmp_path, **basis_overrides):
        from chess_variant_sandbox import corpus

        spec = load_spec(RULESETS / "gardner.json")
        digest = spec_hash(spec)
        (tmp_path / "rulesets").mkdir()
        (tmp_path / "rulesets" / "g.json").write_text(json.dumps(spec), encoding="utf-8")

        out = corpus.measurement_dir(digest, tmp_path / "runs")
        out.mkdir(parents=True, exist_ok=True)
        basis = {**corpus.current_basis(digest, spec=spec).as_dict(), **basis_overrides}
        (out / "canonical.json").write_text(json.dumps({
            "hash": digest, "name": "gardner", "index": 0.76, "sd": 0.02,
            "viable": True, "violations": [], "weights_version": "v8",
            "agents": list(corpus.CANONICAL_AGENTS), "games": 256,
            "families": {}, "basis": basis,
            "seeds": [0], "totals_used": 1, "runs": _one_run(0.76, 0.02),
            "standings": [
                {"name": "random", "rating": -700.0},
                {"name": "minimax:nodes=6400", "rating": 620.0},
            ],
        }), encoding="utf-8")

        found = library.discover(tmp_path / "rulesets", tmp_path / "runs")
        return next(game for game in found if game.hash == digest)

    def test_a_current_measurement_shows_its_score_and_its_ladder(self, tmp_path) -> None:
        game = self._world(tmp_path)
        assert game.summary()["index"] == 0.76
        assert game.ratings is not None
        assert len(game.ratings["standings"]) == 2
        assert game.summary()["staleness"]["current"] is True

    def test_a_stale_one_shows_its_number_marked_provisional_and_no_ladder(
        self, tmp_path
    ) -> None:
        """The ladder goes with the score as a *current* thing, and that is the point.

        EV2 (`docs/archive/plan-evolving-score.md` §5.1): a stale score's last number is shown,
        marked provisional with its revision and date, never as current. The Elo table is
        still withheld: "how hard is this opponent" is answered by a table fitted under an
        engine that no longer exists, and a ladder is a comparison.
        """
        game = self._world(tmp_path, engine="0000000000000000-0000000000000000")
        card = game.summary()
        assert card["index"] == 0.76
        assert card["status"] == "provisional"
        assert card["revision"] and card["measuredAt"]
        assert card["staleness"]["current"] is False
        assert game.ratings is None
        assert game.current_score is None
        assert game.stale

    def test_the_reason_travels_with_the_card(self, tmp_path) -> None:
        """Absent and stale are different news, and the card distinguishes them.

        No number and no staleness means "never measured"; a provisional number with a
        reason means "measured, but not under the code you are running". The repairs are
        different and so are the sentences.
        """
        stale = self._world(tmp_path, weights_version="v5").summary()
        assert stale["index"] == 0.76 and stale["status"] == "provisional"
        assert stale["revision"] == "v5"
        assert "v5" in stale["staleness"]["summary"]
        assert stale["staleness"]["needs"] == "rescore"

    def test_a_measurement_of_a_different_ruleset_is_never_borrowed(self, tmp_path) -> None:
        """The hash is in the basis too, so a directory holding somebody else's
        measurement is refused rather than read as this game's -- not even as a
        provisional number, since it was never this game's estimate."""
        game = self._world(tmp_path, ruleset_hash="somethingelse")
        assert game.summary()["index"] is None
        assert game.status == "none" and game.summary()["status"] == "none"
        assert "different ruleset" in game.summary()["staleness"]["summary"]


class TestTwoRatingTablesThatMustNotMix:
    """The separation that keeps a per-ruleset player out of a cross-ruleset scale.

    `metrics.components` filters heuristic names, but that is defence in depth and not
    the argument: `rating_spread` reads the *whole* rating table, so no name filter can
    reach it. The guarantee is that the two corpora are separate files, and these tests
    are what hold that.
    """

    def test_the_stored_ladder_table_contains_no_heuristic(self, warm_library) -> None:
        """Read off the tree's own `runs/`: the main checkout's has the tables, a worktree's
        none, and there it skips rather than passing on nothing (Q745). That the picker's
        table cannot reach this one is the two tests below, which need no `runs/`.
        Discovered here, since the module's `games` reads no `runs/` (Q995f)."""
        standings = [
            (game.name, standing)
            for game in library.discover(RULESETS, library.RUNS_DIR)
            for standing in (game.ratings or {}).get("standings", [])
        ]
        if not standings:
            pytest.skip("no game in this tree's runs/ has a stored ladder table")
        for name, standing in standings:
            assert "h=" not in standing["name"], f"{name} rates a heuristic"

    def test_the_pickers_table_is_not_a_tournament_file(self, tmp_path) -> None:
        """The library finds the rated table by globbing for `tournament.json`.

        A second file by that name anywhere under `runs/` would be picked up as one, so
        the picker's table is named differently and this is the check that says so.
        """
        from chess_variant_sandbox.web.opponents import stored_path

        assert stored_path("abc123", tmp_path).name != "tournament.json"
        assert stored_path("abc123", tmp_path).name == "abc123.json"

    def test_a_pickers_table_is_invisible_to_the_library(self, tmp_path) -> None:
        from chess_variant_sandbox.web.opponents import stored_path

        path = stored_path("4c1922ffae26", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "ruleset_hash": "4c1922ffae26",
            "standings": [{"name": "minimax:h=abc,nodes=6400", "rating": 900.0, "games": 20}],
        }), encoding="utf-8")
        # Invisible by construction now rather than by a name check: the library reads
        # `canonical.json` and nothing else, so a file of standings under any other name
        # is not a measurement and cannot be mistaken for one.
        assert library._measurements(tmp_path) == {}

    def test_a_listing_reads_each_listed_games_measurement_once(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q874: a kept score (`previous/<n>/`) names its game's hash again, and a game that
        left the library leaves its files behind. Neither may cost a lookup: the first is
        the same answer twice, and the second went to `corpus.find`, 60 ms a file, for a
        measurement `discover` never used -- 19 s of a listing in the main checkout."""
        rulesets = tmp_path / "rulesets"
        rulesets.mkdir()
        shutil.copy(RULESETS / "dull.json", rulesets / "dull.json")
        digest = spec_hash(load_spec(rulesets / "dull.json"))
        runs = tmp_path / "runs"
        for directory in (runs / digest, runs / digest / "previous" / "1",
                          runs / "gone", runs / "gone" / "previous" / "1"):
            directory.mkdir(parents=True)
            named = digest if digest in directory.parts else "0123456789ab"
            (directory / "canonical.json").write_text(
                json.dumps({"hash": named}), encoding="utf-8")

        asked: list[str] = []
        real = corpus.read_measurement

        def counted(found: str, *args: object, **kwargs: object) -> object:
            asked.append(found)
            return real(found, *args, **kwargs)  # type: ignore[arg-type]

        def refused(*args: object, **kwargs: object) -> object:
            raise AssertionError("a listing looked a hash up with corpus.find")

        monkeypatch.setattr(corpus, "read_measurement", counted)
        monkeypatch.setattr(corpus, "find", refused)
        [game] = library.discover(rulesets, runs)
        assert game.hash == digest
        assert asked == [digest]


class TestTheListingWalksOnlyTheGamesOwnDirectories:
    """Q885: a cold listing walked the whole of `runs/`, 10,760 directories in the main
    checkout and 1.7 s, for the 126 measurements directly under it. It now lists `runs/`,
    each game's own `runs/<hash>/` and `runs/<hash>-<seed>/`, and asks every other
    directory there for two files by name. These hold that the answer did not move and
    that the rest of the tree is never opened."""

    CAMPAIGN = "replay-2026-10-13"

    def _library(self, tmp_path: Path) -> tuple[Path, Path, str, str]:
        """Two games: one measured where a measurement lives now, one only in the old
        single-run place, with a verdict beside a seed run and one in a design run."""
        rulesets = tmp_path / "rulesets"
        rulesets.mkdir()
        digests = []
        for name in ("dull", "gardner"):
            shutil.copy(RULESETS / f"{name}.json", rulesets / f"{name}.json")
            digests.append(spec_hash(load_spec(rulesets / f"{name}.json")))
        measured, old = digests
        runs = tmp_path / "runs"
        spec = load_spec(rulesets / "dull.json")
        self._measurement(corpus.measurement_dir(measured, runs), measured, 0.5,
                          corpus.current_basis(measured, spec=spec).as_dict())
        self._measurement(corpus.run_dir(old, 0, runs), old, 0.4, None)
        self._verdict(corpus.run_dir(measured, 1000, runs), measured, "Beside a seed.")
        self._verdict(runs / "designed", old, "Beside a design run.")
        return rulesets, runs, measured, old

    @staticmethod
    def _measurement(directory: Path, digest: str, index: float, basis: dict | None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        body: dict = {"hash": digest, "name": "fixture", "index": index, "sd": 0.02,
                      "viable": True, "violations": [], "weights_version": "v12",
                      "families": {}, "standings": [], "games": 4}
        if basis is not None:
            body |= {"seeds": [0], "totals_used": 1, "runs": _one_run(index, 0.02),
                     "basis": basis}
        (directory / "canonical.json").write_text(json.dumps(body), encoding="utf-8")

    @staticmethod
    def _verdict(directory: Path, digest: str, words: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "verdict.json").write_text(json.dumps({
            "ruleset_hash": digest, "verdict": words, "strengths": [], "weaknesses": [],
            "suspected_degeneracy": "", "would_play_again": True,
        }), encoding="utf-8")

    def _clutter(self, runs: Path, measured: str, old: str) -> list[Path]:
        """What history and campaigns leave behind, each naming a listed game again with a
        different number, and a game that left the library. Returns the directories a
        listing has no business opening, which is all but the last."""
        kept = runs / measured / "previous" / "1"
        replaced = runs / self.CAMPAIGN / "replaced" / measured
        campaign_run = runs / self.CAMPAIGN / f"{old}-0"
        deep = runs / "q999" / "schema6" / old
        retired = runs / "0123456789ab"
        self._measurement(kept, measured, 0.11, None)
        self._measurement(replaced, measured, 0.22, None)
        self._measurement(campaign_run, old, 0.33, None)
        self._measurement(deep, old, 0.44, None)
        self._measurement(retired, "0123456789ab", 0.55, None)
        (campaign_run / "games.jsonl").write_text("", encoding="utf-8")
        return [kept, replaced, campaign_run, deep]

    @staticmethod
    def _answer(rulesets: Path, runs: Path) -> list[tuple]:
        library._LISTINGS.clear()
        return [
            (game.hash, game.source, game.status, game.verdict, game.score,
             game.measurement.directory if game.measurement else None,
             game.staleness, game.history, game.premise)
            for game in library.discover(rulesets, runs)
        ]

    def test_the_clutter_moves_no_answer(self, tmp_path, monkeypatch) -> None:
        """The same listing with and without everything a campaign leaves under `runs/`.
        On a tree with nothing but the games' own directories the old walk and this one
        read the same files, so this is the old answer too."""
        monkeypatch.setattr(library, "_LISTINGS", {})
        rulesets, runs, measured, old = self._library(tmp_path)
        clean = self._answer(rulesets, runs)
        by_hash = {row[0]: row for row in clean}
        assert by_hash[measured][2] == "current" and by_hash[measured][4]["index"] == 0.5
        assert by_hash[measured][3]["verdict"] == "Beside a seed."
        assert by_hash[old][5] == corpus.run_dir(old, 0, runs)
        assert by_hash[old][3]["verdict"] == "Beside a design run."

        self._clutter(runs, measured, old)
        assert self._answer(rulesets, runs) == clean

    def test_the_walk_never_enters_a_campaign_or_a_kept_score(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(library, "_LISTINGS", {})
        rulesets, runs, measured, old = self._library(tmp_path)
        unvisited = self._clutter(runs, measured, old)
        unvisited += [runs / self.CAMPAIGN, runs / self.CAMPAIGN / "replaced",
                      runs / "q999", runs / "designed", runs / measured / "previous"]

        listed: list[Path] = []
        real_scandir, real_listdir = os.scandir, os.listdir

        def scandir(path: object = ".") -> object:
            listed.append(Path(os.fsdecode(path)).resolve())  # type: ignore[arg-type]
            return real_scandir(path)  # type: ignore[call-overload]

        def listdir(path: object = ".") -> object:
            listed.append(Path(os.fsdecode(path)).resolve())  # type: ignore[arg-type]
            return real_listdir(path)  # type: ignore[call-overload]

        def walk(*args: object, **kwargs: object) -> object:
            raise AssertionError("a listing walked a tree")

        monkeypatch.setattr(os, "scandir", scandir)
        monkeypatch.setattr(os, "listdir", listdir)
        monkeypatch.setattr(os, "walk", walk)
        library.discover(rulesets, runs)
        monkeypatch.undo()

        assert runs.resolve() in listed
        assert corpus.measurement_dir(measured, runs).resolve() in listed
        entered = sorted({path for path in listed if path.is_relative_to(runs.resolve())})
        for directory in unvisited:
            assert directory.resolve() not in entered, f"the listing opened {directory}"
        for path in entered:
            assert path == runs.resolve() or path.parent == runs.resolve(), path


class TestTheSpineSchedule:
    """Who plays whom when the picker is rated, and why it is not a round robin.

    A round robin over seventeen opponents makes `minimax:nodes=200000` -- about a fifth
    of a second a move -- play sixty-four games rather than twenty, and it is the one
    that decides the wall clock. The spine bounds that by its own length.
    """

    def test_challengers_meet_the_spine_and_never_each_other(self) -> None:
        from chess_variant_sandbox.tournament.runner import schedule
        from chess_variant_sandbox.web.opponents import SPINE, _SpineConfig

        spine = SPINE
        challengers = ("minimax:nodes=200000", "minimax:h=aaaaaaaaaaaa,nodes=6400")
        config = _SpineConfig(
            agents=(*spine, *challengers),
            games_per_pairing=2,
            include_self_play=False,
            spine=spine,
        )
        pairs = {(first, second) for _, first, second in schedule(config)}
        for one in challengers:
            for other in challengers:
                assert (one, other) not in pairs
            for anchor in spine:
                assert (one, anchor) in pairs and (anchor, one) in pairs
        # And the spine still plays itself, which is what anchors the scale.
        assert (spine[0], spine[1]) in pairs

    def test_the_spine_reaches_the_top_of_the_range(self) -> None:
        """A player who beats every anchor every time has no determined rating.

        With a spine topping out at 1,600 nodes, `minimax:nodes=200000` and a good rule
        of thumb would both sweep it and come back tied at whatever the fit's ceiling
        is. 6,400 nodes is cheap and loses often enough to place them.
        """
        from chess_variant_sandbox.web.opponents import SPINE

        assert "minimax:nodes=6400" in SPINE


class TestTheRatingRunKeepsTheCoresGamesInTheCore:
    """One player the core cannot run used to send the whole schedule to Python.

    Fairy-Stockfish has no native form, and a schedule with it in used to play every
    game in the Python loop, one at a time: seven minutes on crazyhouse and chess where
    the core's own players alone take one. Q432 split the picker into two schedules
    here; Q439 moved the split into the runner, which sends the games whose seats both
    have a native form to the core and keeps only the outside player's in the loop. So
    the picker is one schedule again (Q446), and the runner tells the run its count as
    games come back. The stand-in here is a Python agent, because it is the missing
    native form that matters and not the binary, which the suite does not need.
    """

    @staticmethod
    def _outsider(monkeypatch: pytest.MonkeyPatch, on_move=None) -> None:
        import random as _random

        from chess_variant_sandbox.agents import SearchStats, base
        from chess_variant_sandbox.engine import GameState

        class Lowest:
            def __init__(self) -> None:
                self.name = "outsider"
                self.stats = SearchStats()

            def select_move(
                self, state: GameState, moves: list[int], rng: _random.Random
            ) -> int:
                self.stats.moves += 1
                if on_move is not None:
                    on_move()
                return min(moves)

        monkeypatch.setitem(base._REGISTRY, "outsider", lambda engine, options: Lowest())

    @staticmethod
    def _rate(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        agents: tuple[str, ...],
        ratings=None,
    ):
        """Rate a gardner picker of `agents`, and every schedule it handed the runner."""
        from chess_variant_sandbox.web import opponents

        spec = load_spec(RULESETS / "gardner.json")
        roster = [opponents.Opponent(agent=agent, label=agent, group="") for agent in agents]
        schedules = []
        real = opponents.run_tournament

        def spy(spec, config, *args, **kwargs):
            schedules.append(config)
            return real(spec, config, *args, **kwargs)

        monkeypatch.setattr(opponents, "run_tournament", spy)
        run = (ratings or opponents.Ratings(tmp_path)).start(
            spec, spec_hash(spec), "gardner", roster, games=1
        )
        deadline = time.monotonic() + 60
        while run.state == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert run.state == "finished", run.error
        return run, schedules

    @pytest.mark.parametrize("outside", [True, False], ids=["with-outsider", "core-only"])
    def test_the_whole_picker_is_one_schedule(
        self, tmp_path, monkeypatch, outside: bool
    ) -> None:
        """One schedule numbers every seed, so no two games share one, and it is the
        schedule the run promised."""
        from chess_variant_sandbox.tournament.runner import schedule

        self._outsider(monkeypatch)
        agents = ("random", "greedy", "minimax:nodes=400", *(("outsider",) if outside else ()))
        run, schedules = self._rate(tmp_path, monkeypatch, agents)
        assert len(schedules) == 1
        assert schedules[0].agents == agents
        played = schedule(schedules[0])
        assert len({seed for seed, _, _ in played}) == len(played)
        assert run.total == len(played) == run.played == run.table["games"]
        assert {row["name"] for row in run.table["standings"]} == set(agents)

    def test_the_count_moves_while_the_outside_engines_games_play(
        self, tmp_path, monkeypatch
    ) -> None:
        """The page's middle count, *N are done, and the other M are Fairy-Stockfish's*.

        With one `run_tournament` call and nothing to call back to, `played` would stay
        0 until every game was back. The runner tells it once when the core's games are
        back and once per outside game after that, so each outside game starts with the
        core's games and the outside games before it counted.
        """
        from chess_variant_sandbox.tournament.runner import schedule
        from chess_variant_sandbox.web import opponents

        spec = load_spec(RULESETS / "gardner.json")
        ratings = opponents.Ratings(tmp_path)
        seen: list[int] = []
        self._outsider(
            monkeypatch, on_move=lambda: seen.append(ratings.get(spec_hash(spec)).played)
        )
        run, schedules = self._rate(
            tmp_path, monkeypatch, ("random", "greedy", "minimax:nodes=400", "outsider"),
            ratings,
        )
        played = schedule(schedules[0])
        core = sum("outsider" not in (first, second) for _, first, second in played)
        assert 0 < core < run.total
        # Seen from inside each outside game: the core's games are all counted before
        # the first of them, and every one after counts the one before it.
        assert seen == sorted(seen)
        assert set(seen) == set(range(core, run.total))
        assert run.played == run.total


class TestTheRateButtonSaysHowLongItTakes:
    """Q432. The button said *a minute or so* and a picker with Fairy-Stockfish took seven.

    Measured on the 16-core box after the split above: without the outside engine 4s
    (dobutsu) to 147s (courier), with it 71s (gardner) to 562s (courier). The page says
    so, and says it again while the run is going, where the wait actually happens.
    """

    HASH = "abc123def456"
    FAIRY = {"agent": "fairystockfish:nodes=200000", "label": "Fairy-Stockfish (200,000 nodes)",
             "group": "an engine somebody else wrote", "plan": "an outside engine",
             "store": "fairy"}

    def _said(self, fairy: bool, press: str = "", answers: dict | None = None) -> str:
        """The words of the strength panel, or of everything drawn once `press` ran."""
        payload = _game_payload(self.HASH, site=False)
        if fairy:
            payload["opponents"] = [*payload["opponents"], self.FAIRY]
        drawn = _page_on(False, f"#/game/{self.HASH}", payload, answers=answers, press=press)
        scales = TestOneRatingScaleOnTheSite
        return scales._words(drawn["every"] if press else scales._strength(drawn["html"]))

    def test_the_old_promise_is_gone(self) -> None:
        from page_text import PAGE

        assert "a minute or so" not in PAGE

    def test_our_own_players_take_up_to_three_minutes(self) -> None:
        said = self._said(fairy=False)
        assert ("plays them against each other, which takes under a minute on a small "
                "board and up to three on a large one") in said
        assert "Fairy-Stockfish" not in said

    def test_with_fairy_stockfish_it_is_up_to_ten_and_says_why(self) -> None:
        said = self._said(fairy=True)
        assert ("plays them against each other, which takes a minute on a small board "
                "and up to ten on a large one, because Fairy-Stockfish's games are played "
                "one at a time") in said

    def _running(self, played: int) -> str:
        running = {"state": "running", "total": 200, "played": played, "seconds": 75.4}
        return self._said(
            fairy=True,
            answers={"/api/rate": {**running, "played": 0, "seconds": 0},
                     f"/api/rating/{self.HASH}": running},
            # The poll's interval is cleared once it has drawn, or node would wait on it.
            press="document.getElementById('rate').onclick();"
                  "setTimeout(() => { clearInterval(RATING_TIMER); RATING_TIMER = null; }, 30);",
        )

    def test_the_wait_is_repeated_while_it_runs(self) -> None:
        said = self._running(played=0)
        assert ("It is playing 200 games, 75s so far. It takes a minute on a small board and "
                "up to ten on a large one, because Fairy-Stockfish's games are played one at "
                "a time.") in said

    def test_once_the_core_is_done_it_says_what_is_left_and_why_once(self) -> None:
        said = self._running(played=180)
        assert ("180 are done, and the other 20 are Fairy-Stockfish's, one at a time. All "
                "told it takes a minute on a small board and up to ten on a large one.") in said
        # `every` still holds the first draw, button and all; the message is its own box.
        message = said[said.index("It is playing"):]
        assert message.count("one at a time") == 1


class TestWhatBecomesOfACandidate:
    """The three actions, through the API a browser actually calls.

    The ladder itself is `tests/test_corpus.py`'s subject. What these check is the wiring:
    that the page is told which rung a game is on and why it cannot climb, that the
    browser cannot put something in the corpus the command line would refuse, and that
    the library notices when a game moves.
    """

    def _world(self, tmp_path, rulesets: Path = RULESETS):
        """A design run with one candidate, and empty corpus directories beside it."""
        import json as _json

        from chess_variant_sandbox.loop.archive import Archive, Entry
        from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS

        spec = load_spec(RULESETS / "gardner.json")
        spec = {k: v for k, v in spec.items() if k not in ("appearance", "provenance")}
        spec["name"] = "a candidate"
        # A *mechanical* change, because a rename is not a new game: the hash covers
        # mechanics only, so a renamed gardner is gardner and `find` rightly reports it
        # as already committed.
        spec["rules"] = {**spec["rules"], "max_plies": 199}
        digest = spec_hash(spec)

        runs = tmp_path / "runs"
        archive = Archive(runs / "design")
        archive.record(
            Entry(hash=digest, name="a candidate", generation=1, parent=None,
                  status="evaluated", index=0.5, viable=True, detail={"theme": "a theme"}),
            spec=spec,
        )
        app = App(rulesets, runs, archive_dir=tmp_path / "archive")

        def measured(viable: bool = True) -> None:
            from chess_variant_sandbox import corpus

            out = corpus.measurement_dir(digest, runs)
            out.mkdir(parents=True, exist_ok=True)
            (out / "canonical.json").write_text(_json.dumps({
                "hash": digest, "name": "a candidate",
                "agents": list(corpus.CANONICAL_AGENTS), "games": 256,
                "index": 0.8, "sd": 0.02, "viable": viable,
                "violations": [] if viable else ["quick_kill_rate=0.9 above maximum 0.5"],
                # Four runs, because this fixture stands in for a game about to be
                # promoted and that rung asks for the whole library measurement.
                "seeds": list(corpus.LIBRARY_SEEDS), "totals_used": 4,
                "runs": _runs_at(corpus.LIBRARY_SEEDS, 0.8, 0.02, viable),
                "weights_version": DEFAULT_WEIGHTS, "families": {},
                "standings": [{"name": "random", "rating": -700.0}],
                # Stamped as a measurement taken now would be. Without this the fixture
                # would be exercising the stale path in every test that uses it.
                "basis": corpus.current_basis(
                    digest, spec=spec, seeds=corpus.LIBRARY_SEEDS
                ).as_dict(),
            }), encoding="utf-8")

        return app, digest, measured

    def test_the_page_is_told_which_rung_it_is_on_and_why_it_cannot_climb(self, tmp_path) -> None:
        app, digest, _ = self._world(tmp_path)
        keeping = app.get(f"/api/game/{digest}")["keeping"]
        assert keeping["stage"] == "candidate"
        assert keeping["measurement"] is None
        # The refusal, in words, rather than a button that is merely dead.
        assert "chess-variant-sandbox test" in keeping["blocker"]
        # And what "tested" would even mean, which is the whole point of the rung.
        assert keeping["roster"] == list(__import__("chess_variant_sandbox").corpus.CANONICAL_AGENTS)

    def test_archiving_moves_it_and_the_library_says_so(self, tmp_path) -> None:
        app, digest, _ = self._world(tmp_path)
        kept = app.post("/api/archive", {"hash": digest, "name": "Kept", "note": "worth it"})

        assert kept["stage"] == "archived"
        assert Path(kept["path"]).exists()
        # The library is re-read, so the panel that offered the action is rebuilt from
        # where the game now lives rather than from where it was.
        assert app.get(f"/api/game/{digest}")["keeping"]["stage"] == "archived"
        assert next(g for g in app.refresh() if g.hash == digest).source == "archived"

    def test_the_browser_cannot_promote_what_the_command_line_refuses(self, tmp_path) -> None:
        app, digest, measured = self._world(tmp_path)

        with pytest.raises(ValueError, match="chess-variant-sandbox test"):
            app.post("/api/promote", {"hash": digest, "reason": "because"})

        measured(viable=False)
        with pytest.raises(ValueError, match="not viable"):
            app.post("/api/promote", {"hash": digest, "reason": "because"})

        measured(viable=True)
        with pytest.raises(ValueError, match="needs a reason"):
            app.post("/api/promote", {"hash": digest, "reason": "  "})

    def test_a_promoted_game_reads_as_committed_and_offers_nothing_further(self, tmp_path) -> None:
        """A promotion writes into the app's `rulesets/`, so this one gets a copy of it.

        It used to write into the checkout's own and unlink the file after, which under
        `-n 4` put a forty-third ruleset in front of every other worker's corpus census
        while it ran (Q377c). The copy holds every committed game, so the app sees the
        same corpus and the same refusals; only the directory is this test's.
        """
        corpus_copy = tmp_path / "rulesets"
        corpus_copy.mkdir()
        for committed in RULESETS.glob("*.json"):
            shutil.copyfile(committed, corpus_copy / committed.name)
        app, digest, measured = self._world(tmp_path, rulesets=corpus_copy)
        measured()
        promoted = app.post(
            "/api/promote",
            {"hash": digest, "name": "Kept", "reason": "it pins down a thing"},
        )
        assert Path(promoted["path"]).parent == corpus_copy
        assert promoted["stage"] == "committed"
        keeping = app.get(f"/api/game/{digest}")["keeping"]
        assert keeping["stage"] == "committed"
        assert "already in" in keeping["blocker"]

    def test_a_measurement_is_reported_before_it_is_asked_for(self, tmp_path) -> None:
        """The GET answers "absent" rather than 404, so a page can render the panel
        without knowing whether anyone has ever pressed the button."""
        app, digest, measured = self._world(tmp_path)
        assert app.get(f"/api/test/{digest}")["state"] == "absent"
        measured()
        state = app.get(f"/api/test/{digest}")
        assert state["state"] == "finished"
        assert state["result"]["index"] == 0.8


class TestWhereTheHumanSits:
    """Stage 29: the only measurement in this project that is not the ladder judging itself.

    Everything else compares agents to agents. These hold down the three properties that
    make a person's results usable as an anchor without turning them into a target:
    recording is automatic so a win is no more likely to be kept than a loss; an
    abandoned game leaves a row so the one remaining bias is visible; and placing the
    person on the ladder does not move the ladder.
    """

    def _app(self, tmp_path) -> App:
        return App(RULESETS, tmp_path)

    def _play_out(self, app: App, digest: str, opponent: str) -> str:
        """Start a human-vs-agent session and play it to a verdict, always first legal."""
        snapshot = app.post("/api/session", {"hash": digest, "seats": ["human", opponent]})
        session_id = snapshot["id"]
        for _ in range(400):
            if snapshot["outcome"] is not None:
                break
            if snapshot["awaitingAgent"]:
                snapshot = app.post(f"/api/session/{session_id}/agent", {})
                continue
            snapshot = app.post(
                f"/api/session/{session_id}/move", {"move": snapshot["moves"][0]["move"]}
            )
        return session_id

    def test_a_started_game_is_recorded_before_it_finishes(self, tmp_path, chess_game) -> None:
        """The bias automatic recording cannot remove, made visible rather than hidden.

        A person losing can close the tab. Nothing can stop that; what can be done is to
        write the row when the game *starts*, so "games begun and never finished" is a
        number a reader can weigh the scores against.
        """
        app = self._app(tmp_path)
        app.post("/api/session", {"hash": chess_game.hash, "seats": ["human", "random"]})
        summary = app.get(f"/api/human/{chess_game.hash}")
        assert summary["started"] == 1
        assert summary["finished"] == 0
        assert summary["abandoned"] == 1
        assert summary["score"] is None

    def test_a_finished_game_is_recorded_without_being_asked(self, tmp_path) -> None:
        app = self._app(tmp_path)
        game = next(g for g in app.refresh() if g.name == "gardner")
        self._play_out(app, game.hash, "random")

        summary = app.get(f"/api/human/{game.hash}")
        assert summary["finished"] == 1
        assert summary["abandoned"] == 0
        assert summary["score"] in (0.0, 0.5, 1.0)
        assert summary["byOpponent"][0]["agent"] == "random"

    def test_a_result_asked_for_twice_is_recorded_once(self, tmp_path) -> None:
        """A finished position answers the same way every time it is looked at.

        The page discovers a game is over by looking at the position it got back, so the
        write happens on whichever action ended it -- and an agent seat that has nothing
        left to play answers the same terminal snapshot again. Dropping the repeat when
        it is written keeps the file a log of what happened rather than of how often it
        was looked at.
        """
        from chess_variant_sandbox.web.human import HumanRecord

        record = HumanRecord(tmp_path)
        for _ in range(3):
            record.note_result(
                session="s1", ruleset_hash="h", ruleset_name="n", opponent="random",
                human_seat=0, seed=0, plies=10, winner=0, reason="checkmate",
            )
        games = record.games("h")
        assert len(games) == 1 and games[0].result == 1.0

    def test_the_result_is_the_persons_score_from_their_own_seat(self, tmp_path) -> None:
        from chess_variant_sandbox.web.human import HumanRecord

        record = HumanRecord(tmp_path)
        record.note_result(session="a", ruleset_hash="h", ruleset_name="n",
                           opponent="random", human_seat=1, seed=0, plies=4,
                           winner=0, reason="checkmate")
        record.note_result(session="b", ruleset_hash="h", ruleset_name="n",
                           opponent="random", human_seat=1, seed=0, plies=4,
                           winner=None, reason="stalemate")
        assert [g.result for g in record.games("h")] == [0.0, 0.5]

    def test_two_agents_playing_each_other_is_not_a_human_game(self, tmp_path) -> None:
        app = self._app(tmp_path)
        game = next(g for g in app.refresh() if g.name == "gardner")
        self._play_out(app, game.hash, "random")
        app.post("/api/session", {"hash": game.hash, "seats": ["random", "greedy"]})
        assert app.get(f"/api/human/{game.hash}")["started"] == 1

    def test_the_anchor_places_the_person_without_moving_the_ladder(self) -> None:
        """29.3 in code: every opponent keeps the rating it already had.

        `rate()` would re-fit everybody, and then the anchor and the thing it anchors
        would be one measurement wearing two names. This is a single-parameter fit
        against fixed ratings, so a person's results cannot change what they are being
        compared against.
        """
        from chess_variant_sandbox.web.human import anchor_rating

        ratings = {"weak": -300.0, "middling": 0.0, "strong": 300.0}
        even = anchor_rating(
            [("weak", 1.0), ("middling", 0.5), ("strong", 0.0)] * 4, dict(ratings)
        )
        assert even is not None
        assert -80.0 < even["rating"] < 80.0
        assert even["top"] == "strong"
        # The ratings dict handed in is not consulted for anything but reading, and the
        # ladder it describes is unchanged afterwards.
        assert ratings == {"weak": -300.0, "middling": 0.0, "strong": 300.0}

    def test_beating_everything_is_reported_as_unbounded_rather_than_as_a_number(self) -> None:
        from chess_variant_sandbox.web.human import anchor_rating

        ratings = {"weak": -300.0, "strong": 300.0}
        swept = anchor_rating([("weak", 1.0), ("strong", 1.0)] * 20, ratings)
        assert swept is not None
        assert swept["bounded"] is False
        assert swept["swept"] == "up"

    def test_the_direction_of_a_sweep_is_not_read_off_the_mean_score(self) -> None:
        """The mean covers games against opponents the ladder has no rating for.

        Beat an unrated opponent, lose to a rated one, and the average says 0.5 while
        the fit saw a clean loss. An interface that drew the arrow from the average
        would point the wrong way at exactly the person the anchor exists to place.
        """
        from chess_variant_sandbox.web.human import anchor_rating

        ratings = {"strong": 300.0}
        swept = anchor_rating([("unrated", 1.0), ("strong", 0.0)], ratings)
        assert swept is not None
        assert swept["games"] == 1
        assert swept["bounded"] is False
        assert swept["swept"] == "down"

    def test_a_placed_player_is_swept_in_neither_direction(self) -> None:
        from chess_variant_sandbox.web.human import anchor_rating

        placed = anchor_rating([("weak", 1.0), ("strong", 0.0)], {"weak": -300.0, "strong": 300.0})
        assert placed is not None
        assert placed["bounded"] is True
        assert placed["swept"] == ""

    def test_a_ruleset_with_no_rated_picker_reports_games_and_no_anchor(
        self, tmp_path
    ) -> None:
        """Two different absences, and conflating them would read as "unrated person"."""
        app = self._app(tmp_path)
        game = next(g for g in app.refresh() if g.name == "gardner")
        self._play_out(app, game.hash, "random")
        summary = app.get(f"/api/human/{game.hash}")
        assert summary["finished"] == 1
        assert summary["anchor"] is None

    def test_the_gap_to_the_top_rung_is_what_the_anchor_is_for(self) -> None:
        """A roster whose top rung is below a mediocre player on one game and above one
        on another is a roster whose rungs are not the same rung -- stage 27's claim,
        from the side the ladder cannot see."""
        from chess_variant_sandbox.web.human import anchor_rating

        anchor = anchor_rating(
            [("strong", 1.0)] * 6 + [("strong", 0.0)] * 6,
            {"weak": -300.0, "strong": 300.0},
        )
        assert anchor is not None
        assert anchor["top"] == "strong"
        assert anchor["top_above_human"] == pytest.approx(300.0 - anchor["rating"], abs=0.2)
        # Whose expectation it is. The field is the *person's* expected score against
        # the top rung, and a sentence that read it the other way round would tell a
        # player they were winning against the thing that is beating them.
        assert anchor["expected_against_top"] < 0.5
        assert anchor["expected_against_top"] == pytest.approx(
            1.0 / (1.0 + 10.0 ** (anchor["top_above_human"] / 400.0)), abs=0.002
        )

    def test_the_expectation_flips_when_the_person_is_the_stronger_one(self) -> None:
        from chess_variant_sandbox.web.human import anchor_rating

        anchor = anchor_rating([("weak", 1.0)] * 12, {"weak": -300.0})
        assert anchor is not None
        assert anchor["top_above_human"] < 0.0
        assert anchor["expected_against_top"] > 0.5

    def test_nothing_under_metrics_reads_the_human_store(self) -> None:
        """29.3, enforced rather than promised.

        One player's results are a calibration anchor and a sanity check, never a
        target. The mutant corpus already caught this project fitting an instrument to a
        label that turned out to be wrong, and the cheapest guarantee against doing it
        again is that the scorer cannot see the file at all.
        """
        import chess_variant_sandbox.metrics as metrics_package

        root = Path(metrics_package.__file__).parent
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            assert "web.human" not in source and "HumanRecord" not in source, path


def test_the_archive_is_read_only_when_it_is_named(tmp_path) -> None:
    """A caller that names its own directories must not silently read one it did not.

    `archive/` arrived after several callers had already named a ruleset directory and a
    runs directory, and defaulting it meant those callers began reading the real
    repository's archive into a temporary world. That is the shape of bug that makes a
    test pass or fail depending on what somebody kept last week.
    """
    spec = load_spec(RULESETS / "gardner.json")
    spec["rules"] = {**spec["rules"], "max_plies": 198}
    (tmp_path / "rulesets").mkdir()
    (tmp_path / "archive").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "archive" / "kept.json").write_text(json.dumps(spec), encoding="utf-8")

    unnamed = library.discover(tmp_path / "rulesets", tmp_path / "runs")
    assert unnamed == []

    named = library.discover(
        tmp_path / "rulesets", tmp_path / "runs", tmp_path / "archive"
    )
    assert [game.source for game in named] == ["archived"]


class TestManagingTheScores:
    """Taking a measurement again, from the one screen a person is actually looking at.

    Every score in `runs/` goes stale the moment the engine, the agents, the metrics or
    the weights move, and the library already said so on every card. What it could not do
    was anything about it: the repair lived entirely in `chess-variant-sandbox measure`, and for a
    committed ruleset the page offered no measurement control at all -- the panel about
    keeping a game said "nothing further to do here", which was true of promoting it and
    false of measuring it.

    So these hold two things down: the survey the panel is built from says the same thing
    the command line does, and a sweep is honest about what became of every game in it.
    """

    @staticmethod
    def _game(tmp_path: Path, source: str = "gardner") -> tuple[Path, str]:
        """A committed ruleset of its own, so a test never writes into `rulesets/`.

        Copied whole rather than renamed: the spec hash covers mechanics only, so three
        copies of one game under three names are one game, and a sweep over them would
        be a sweep over one item.
        """
        spec = load_spec(RULESETS / f"{source}.json")
        rulesets = tmp_path / "rulesets"
        rulesets.mkdir(exist_ok=True)
        (rulesets / f"{source}.json").write_text(json.dumps(spec), encoding="utf-8")
        return rulesets, spec_hash(spec)

    @staticmethod
    @pytest.fixture
    def built(monkeypatch):
        """A core that is not behind its sources, whatever this checkout's is.

        The refusal itself is asserted below. Every other test here is about the queue
        rather than the build, and a suite that only passes on a freshly rebuilt
        extension is a suite that fails for a reason it is not about.
        """
        monkeypatch.setattr(server_module, "stale_build", lambda: None)

    def _app(self, tmp_path: Path, source: str = "gardner") -> tuple[App, str]:
        rulesets, digest = self._game(tmp_path, source)
        (tmp_path / "runs").mkdir(exist_ok=True)
        (tmp_path / "archive").mkdir(exist_ok=True)
        return App(rulesets, tmp_path / "runs", archive_dir=tmp_path / "archive"), digest

    def _current(self, tmp_path: Path, digest: str):
        """The basis a measurement of this game taken right now would carry.

        The spec is looked up rather than passed, because these tests hold a hash and
        the directories it lives in -- and without it the tables fingerprint is `""`,
        which is the "this checkout cannot compile that game" reading rather than the
        "measured just now" one every caller here wants.
        """
        candidate = corpus.find(
            digest, runs_dir=tmp_path / "runs", ruleset_dir=tmp_path / "rulesets",
            archive_dir=tmp_path / "archive",
        )
        return corpus.current_basis(digest, spec=candidate.spec)

    def _store(self, tmp_path: Path, digest: str, basis) -> None:
        """Write a measurement resting on `basis`, the way `corpus._store` does."""
        out = corpus.measurement_dir(digest, tmp_path / "runs")
        out.mkdir(parents=True, exist_ok=True)
        (out / "canonical.json").write_text(json.dumps({
            "hash": digest, "name": "surveyed", "index": 0.5, "sd": 0.02,
            "viable": True, "violations": [], "weights_version": basis.weights_version,
            "families": {}, "standings": [], "games": 4,
            "seeds": [0], "totals_used": 1, "runs": _one_run(0.5, 0.02),
            "basis": basis.as_dict(),
        }), encoding="utf-8")

    def test_the_survey_says_what_is_out_of_date_and_what_that_costs(self, tmp_path) -> None:
        """Three states, because they need three different amounts of work.

        Never measured and measured-under-other-code are different news, and a weights
        bump is seconds where a changed engine is minutes. A panel that showed one number
        would be hiding what its own button is about to spend.
        """
        app, digest = self._app(tmp_path)

        scoring = app.get("/api/scoring")
        assert [g["state"] for g in scoring["games"]] == ["absent"]
        assert (scoring["absent"], scoring["current"], scoring["stale"]) == (1, 0, 0)

        self._store(tmp_path, digest, self._current(tmp_path, digest))
        scoring = app.get("/api/scoring")
        assert scoring["current"] == 1 and scoring["games"][0]["needs"] == ""

        # Same games, scored under weights this checkout no longer uses: the records
        # stand, so the repair is a re-score rather than a replay.
        older = self._current(tmp_path, digest)
        self._store(tmp_path, digest, replace(older, weights_version="v0"))
        scoring = app.get("/api/scoring")
        assert scoring["stale"] == 1 and scoring["rescore"] == 1
        assert "v0" in scoring["games"][0]["reason"]
        # And the number itself does not travel while it is out of date.
        assert scoring["games"][0]["index"] is None

    def test_the_third_repair_is_counted_as_itself(self, tmp_path) -> None:
        """`Staleness.needs` gives three answers and this panel counted two of them.

        A `reanalyse` -- the pass that reads the games changed, the games did not -- fell
        into neither count. So the panel offering to measure N games described fewer than
        N, and the one repair that is cheaper than a replay was the one with no number
        next to it.
        """
        app, digest = self._app(tmp_path)
        # The pass moved and nothing else did, which is the whole of a `reanalyse`.
        self._store(tmp_path, digest, replace(self._current(tmp_path, digest),
                                              analysis="0" * 16))

        scoring = app.get("/api/scoring")
        assert scoring["games"][0]["needs"] == "reanalyse"
        assert (scoring["rescore"], scoring["reanalyse"], scoring["replay"]) == (0, 1, 0)
        # And the games behind it still stand, which is what makes it not a replay.
        assert scoring["stale"] == 1

    def test_the_page_names_the_third_repair_rather_than_reading_it_as_a_replay(self) -> None:
        """The three screens that said "play them again" about games nothing invalidated.

        Asserted against the page source because there is no JavaScript runtime here.
        The two-way ternaries are pinned as absent by their exact old text: a third
        branch added beside them would pass a positive check and still leave the bug.
        """
        from page_text import PAGE

        assert "g.needs === 'rescore' ? 'scoring again' : 'playing again'" not in PAGE
        assert "stale.needs === 'rescore' ? 'score it again'" not in PAGE

        assert "reanalyse: 'analysing and scoring again'" in PAGE
        assert "reanalyse: ['analyse and score it again'," in PAGE
        assert "s.reanalyse" in PAGE

    def test_the_page_and_the_command_line_agree_about_what_is_stale(self, tmp_path) -> None:
        """One survey behind both, rather than two loops that happen to agree."""
        app, digest = self._app(tmp_path)
        statuses = corpus.survey(
            runs_dir=tmp_path / "runs",
            ruleset_dir=tmp_path / "rulesets",
            archive_dir=tmp_path / "archive",
        )
        assert [status.as_dict() for status in statuses] == app.get("/api/scoring")["games"]

    def test_a_measurement_reaches_the_page_that_started_it(self, tmp_path) -> None:
        """The bug in `docs/archive/bugs-resolved.md`: measure from a game's page and nothing appears.

        The page was right and the server was not. `pollTest` sees the run finish, drops
        its own copy of the game and asks `/api/game/<hash>` again -- and `App.game`
        answered out of `_games`, the map built by the last `refresh`, whose `Game`
        carries the measurement it was discovered with and memoises its score on the
        instance. So the panel that had just spent minutes measuring re-rendered as
        "nobody has measured this one", and walking back to the library -- which calls
        `refresh` on every listing -- showed the score, which is what made it look like a
        page bug.

        Written as two reads of one endpoint across one write, because that is the shape
        of what the person did: ask, measure, ask again.
        """
        app, digest = self._app(tmp_path)

        before = app.get(f"/api/game/{digest}")
        assert before["score"] is None and before["staleness"] is None

        # What the measurement the button starts leaves behind.
        self._store(tmp_path, digest, self._current(tmp_path, digest))

        after = app.get(f"/api/game/{digest}")
        assert after["score"] is not None, "the page that started it must see the result"
        assert after["score"]["index"] == 0.5
        assert after["index"] == 0.5
        assert after["staleness"]["current"] is True

        # And the two screens say the same thing in the same second, which is the half
        # the reporter could see: the library was already right about this game.
        card = next(g for g in app.get("/api/library")["games"] if g["hash"] == digest)
        assert card["index"] == after["index"]

    def test_the_game_page_sees_a_finished_measurement_without_rescanning_the_corpus(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Performance item 20: the same freshness as the test above, at a bounded cost.

        `App.game()` used to answer this by calling `refresh()` -- a full `discover()`
        over the whole of `runs/` -- so that a just-finished measurement would never be
        served out of the stale snapshot the bug in `docs/bugs.md` describes. That kept
        the freshness and paid for every *other* game's `canonical.json` to touch one.

        So this asserts both halves at once: `library.discover` is not called again to
        open the game a second time (the bound), and the score it reports is the one the
        measurement just wrote (the freshness) -- proving the second call is not simply
        trusting the first call's snapshot, which would report the old ``score: None``.
        """
        app, digest = self._app(tmp_path)

        before = app.get(f"/api/game/{digest}")
        assert before["score"] is None

        self._store(tmp_path, digest, self._current(tmp_path, digest))

        calls = []
        original = library.discover

        def counting(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(library, "discover", counting)

        after = app.get(f"/api/game/{digest}")
        assert after["score"] is not None, "a snapshot trusted rather than re-read would still say None"
        assert after["score"]["index"] == 0.5
        assert after["staleness"]["current"] is True
        assert calls == [], "the game page must not re-read the whole corpus to see one game's new score"

    def test_the_game_page_sees_a_verdict_written_after_it_was_first_opened(
        self, tmp_path,
    ) -> None:
        """The verdict travels the same way the score does.

        `App.game()`'s targeted re-read (`library.reread_one`) checks this one game's
        own measurement directory for `verdict.json` rather than the whole of `runs/`,
        so a verdict written there after the page was first opened must still be seen on
        the next open -- the same freshness `test_a_verdict_is_prose_and_is_absent_until_one_is_written`
        holds for a full `discover()`, held here for the cheap path instead.
        """
        app, digest = self._app(tmp_path)

        before = app.get(f"/api/game/{digest}")
        assert before["verdict"] is None

        measurement_dir = corpus.measurement_dir(digest, tmp_path / "runs")
        measurement_dir.mkdir(parents=True, exist_ok=True)
        (measurement_dir / "verdict.json").write_text(json.dumps({
            "ruleset_hash": digest,
            "verdict": "A small game that knows what it is.",
            "strengths": ["short"], "weaknesses": [], "suspected_degeneracy": "",
            "would_play_again": True,
        }), encoding="utf-8")

        after = app.get(f"/api/game/{digest}")
        assert after["verdict"] is not None
        assert after["verdict"]["verdict"].startswith("A small game")

    def test_a_verdict_beside_one_of_the_seed_runs_is_found_without_a_walk(
        self, tmp_path,
    ) -> None:
        """`chess-variant-sandbox verdict --run <run>` is handed a tournament directory, and for a
        library game that is one of its four seeded `runs/<hash>-<seed>/` directories as
        often as its `runs/<hash>/` measurement directory. `_verdict_for` checks every
        directory this one hash owns -- never the rest of `runs/` -- so it must find a
        verdict filed under any of them.
        """
        from chess_variant_sandbox.web import library

        digest = "deadbeefcafe"
        seed_dir = corpus.run_dir(digest, 1000, tmp_path / "runs")
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "verdict.json").write_text(json.dumps({
            "ruleset_hash": digest,
            "verdict": "Filed beside the seed-1000 tournament.",
            "strengths": [], "weaknesses": [], "suspected_degeneracy": "",
            "would_play_again": False,
        }), encoding="utf-8")

        found = library._verdict_for(digest, tmp_path / "runs")
        assert found is not None
        assert found["verdict"] == "Filed beside the seed-1000 tournament."

        # And a verdict about a different hash, sitting in one of these directories by
        # coincidence, must never answer for this one.
        assert library._verdict_for("00000000000", tmp_path / "runs") is None

    def test_one_view_of_a_game_cannot_disagree_with_itself(self, tmp_path) -> None:
        """`keeping` reads the measurement off disk and `score` reads it off the `Game`.

        With a cached `Game` behind the second, one response carried
        `keeping.measurement.index` from the run that had just finished and `score: null`
        beside it -- the same JSON answering "measured" and "never measured" at once.
        That inconsistency is the tell, so it is pinned rather than left to the first
        assertion, which would still pass if the endpoint went back to reading disk once
        per key.
        """
        app, digest = self._app(tmp_path)
        self._store(tmp_path, digest, self._current(tmp_path, digest))

        view = app.get(f"/api/game/{digest}")
        assert view["keeping"]["measurement"]["index"] == view["score"]["index"]

    def test_a_sweep_reports_what_became_of_every_game_in_it(self, tmp_path, monkeypatch, built) -> None:
        """Including the one that failed, which does not stop the others.

        A sweep that said "finished" over a silent failure would be worse than no sweep:
        the whole point of it is to be able to trust that the scores are current.
        """
        app, first = self._app(tmp_path)
        _, second = self._game(tmp_path, "dull")
        _, third = self._game(tmp_path, "dobutsu")

        def refresh(candidate, **kwargs):
            if candidate.hash == second:
                raise corpus.CorpusError("no games to re-score")
            self._store(tmp_path, candidate.hash,
                        corpus.current_basis(candidate.hash, spec=candidate.spec))
            return corpus.read_measurement(candidate.hash, tmp_path / "runs")

        monkeypatch.setattr(corpus, "refresh", refresh)
        app.post("/api/scoring", {"scope": "stale"})
        sweep = _finished(app)

        assert sweep["state"] == "finished" and sweep["total"] == 3
        by_hash = {item["hash"]: item for item in sweep["items"]}
        assert by_hash[first]["state"] == "finished"
        assert by_hash[third]["state"] == "finished", "one failure must not stop the rest"
        assert by_hash[second]["state"] == "failed"
        assert "no games to re-score" in by_hash[second]["error"]
        assert sweep["failed"] == 1

    def _store_on(self, tmp_path: Path, digest: str, basis) -> None:
        """A measurement whose summary names the seeds its basis names, four of them."""
        out = corpus.measurement_dir(digest, tmp_path / "runs")
        out.mkdir(parents=True, exist_ok=True)
        (out / "canonical.json").write_text(json.dumps({
            "hash": digest, "name": "surveyed", "index": 0.5, "sd": 0.02,
            "viable": True, "violations": [], "weights_version": basis.weights_version,
            "families": {}, "standings": [], "games": 4 * len(basis.seeds),
            "seeds": list(basis.seeds), "totals_used": len(basis.seeds),
            "runs": _runs_at(basis.seeds, 0.5, 0.02), "basis": basis.as_dict(),
        }), encoding="utf-8")

    def test_a_finished_item_carries_the_runs_its_number_rests_on(
        self, tmp_path, monkeypatch, built
    ) -> None:
        """Q339. The item says how many runs its number is the mean of, itself.

        It used to carry the headline alone, and the page borrowed a count off the
        survey's row for the same game -- so a game the survey had no row for printed a
        number with no count beside it.
        """
        app, digest = self._app(tmp_path)

        def refresh(candidate, **kwargs):
            self._store_on(tmp_path, candidate.hash,
                           corpus.current_basis(candidate.hash, spec=candidate.spec))
            return corpus.read_measurement(candidate.hash, tmp_path / "runs",
                                           spec=candidate.spec)

        monkeypatch.setattr(corpus, "refresh", refresh)
        app.post("/api/scoring", {"scope": "stale"})
        (item,) = _finished(app)["items"]
        assert item["hash"] == digest and item["state"] == "finished"
        assert item["index"] == 0.5 and item["viable"] is True
        assert item["seeds"] == list(corpus.LIBRARY_SEEDS)

    def test_an_item_whose_measurement_is_not_current_keeps_no_number(self) -> None:
        """A reader that cannot match the basis shows no number, the sweep included."""
        item = keeping.SweepItem(ruleset_hash="h", name="g", needs="replay")
        item.finish({"index": 0.5, "viable": True, "seeds": [0],
                     "staleness": {"current": False, "needs": "rescore"}})
        assert item.state == "finished"
        assert (item.index, item.viable, item.seeds) == (None, None, ())
        assert item.as_dict()["seeds"] == []

    def test_an_item_whose_measurement_names_no_seeds_keeps_no_number(self) -> None:
        for seeds in (None, [], "0"):
            item = keeping.SweepItem(ruleset_hash="h", name="g", needs="replay")
            item.finish({"index": 0.5, "viable": True, "seeds": seeds,
                         "staleness": {"current": True}})
            assert (item.index, item.seeds) == (None, ()), seeds

    def test_a_current_item_keeps_its_number_and_its_seeds_together(self) -> None:
        item = keeping.SweepItem(ruleset_hash="h", name="g", needs="replay")
        item.finish({"index": 0.5, "viable": False, "seeds": [0, 1000, 2000, 3000],
                     "staleness": {"current": True}})
        assert item.as_dict()["index"] == 0.5 and item.as_dict()["viable"] is False
        assert item.as_dict()["seeds"] == [0, 1000, 2000, 3000]

    def test_the_panel_prints_each_finished_item_with_its_own_count(self) -> None:
        """In the pill's words, and *no score* where the item holds no number."""
        items = [
            {"hash": "a", "name": "alpha", "needs": "replay", "state": "finished",
             "error": "", "index": 0.5, "viable": True, "seeds": [0, 1, 2, 3], "seconds": 1},
            {"hash": "b", "name": "beta", "needs": "replay", "state": "finished",
             "error": "", "index": 0.25, "viable": True, "seeds": [0], "seconds": 1},
            {"hash": "c", "name": "gamma", "needs": "replay", "state": "finished",
             "error": "", "index": None, "viable": None, "seeds": [], "seconds": 1},
            {"hash": "d", "name": "delta", "needs": "replay", "state": "finished",
             "error": "", "index": 0.75, "viable": True, "seeds": [], "seconds": 1},
        ]
        sweep = {"state": "finished", "seconds": 4, "total": 4, "done": 4, "failed": 0,
                 "current": None, "items": items}
        drawn = _in_page("", f"sweepHtml({json.dumps(sweep)})")
        assert 'alpha 0.500 <span class="muted">4 runs</span>' in drawn
        assert 'beta 0.250 <span class="muted">1 run</span>' in drawn
        assert "gamma no score" in drawn
        # A number that arrived without its count is not printed without it.
        assert "delta no score" in drawn and "0.750" not in drawn

    def test_stopping_leaves_the_games_it_never_started_alone(self, tmp_path, monkeypatch, built) -> None:
        """Stop means "do not start the next one", not "kill this tournament".

        A measurement killed half way leaves a `runs/` directory that looks measured and
        is not, which is the failure the whole basis mechanism exists to stop making.
        """
        app, _ = self._app(tmp_path)
        self._game(tmp_path, "dull")
        self._game(tmp_path, "dobutsu")
        started = threading.Event()
        release = threading.Event()

        def refresh(candidate, **kwargs):
            started.set()
            release.wait(10)
            self._store(tmp_path, candidate.hash,
                        corpus.current_basis(candidate.hash, spec=candidate.spec))
            return corpus.read_measurement(candidate.hash, tmp_path / "runs")

        monkeypatch.setattr(corpus, "refresh", refresh)
        app.post("/api/scoring", {"scope": "stale"})
        assert started.wait(10)
        app.post("/api/scoring/stop", {})
        release.set()
        sweep = _finished(app)

        assert sweep["state"] == "stopped"
        states = [item["state"] for item in sweep["items"]]
        # The one it was on finished; the rest were never begun.
        assert states.count("finished") == 1 and states.count("skipped") == 2

    def test_two_sweeps_cannot_run_at_once(self, tmp_path, monkeypatch, built) -> None:
        """They would be two threads writing the same `runs/<hash>-0` directories."""
        app, digest = self._app(tmp_path)
        release = threading.Event()

        def refresh(candidate, **kwargs):
            release.wait(10)
            self._store(tmp_path, candidate.hash,
                        corpus.current_basis(candidate.hash, spec=candidate.spec))
            return corpus.read_measurement(candidate.hash, tmp_path / "runs")

        monkeypatch.setattr(corpus, "refresh", refresh)
        first = app.post("/api/scoring", {"scope": "stale"})
        second = app.post("/api/scoring", {"scope": "all"})
        assert second["scope"] == first["scope"] == "stale", "the second must not start"
        release.set()
        _finished(app)

    def test_a_sweep_with_nothing_to_do_says_so(self, tmp_path) -> None:
        """Rather than spending a tournament to discover it. `all` still replays."""
        app, digest = self._app(tmp_path)
        self._store(tmp_path, digest, self._current(tmp_path, digest))
        with pytest.raises(ValueError):
            app.post("/api/scoring", {"scope": "stale"})
        with pytest.raises(ValueError):
            app.post("/api/scoring", {"scope": "sideways"})

    def test_a_sweep_is_refused_while_the_core_is_behind_its_sources(
        self, tmp_path, monkeypatch
    ) -> None:
        """Before the hours, not after them.

        A core built from older sources does not fail -- it plays, with the previous
        build's rules -- so the sweep would take every measurement, write it, and stamp
        each one with a basis describing source it was not built from. `corpus.measure`
        refuses for this reason and so does the button.
        """
        app, _ = self._app(tmp_path)
        monkeypatch.setattr(server_module, "stale_build", lambda: "rebuild the core")
        with pytest.raises(ValueError, match="rebuild the core"):
            app.post("/api/scoring", {"scope": "stale"})
        assert app.get("/api/scoring")["build"] == "rebuild the core"

    def test_a_committed_ruleset_can_be_measured_again_from_its_own_page(self) -> None:
        """The page is a string and nothing type-checks it, so this reads the string.

        `chess` is committed, so the keeping panel returns early -- which used to mean
        the only screen showing its score had no way to take it again.
        """
        from page_text import PAGE

        assert "measuringHtml(game)" in PAGE, "the Measured panel offers the control"
        assert "id=\"measureIt\"" in PAGE
        assert "id=\"testIt\"" not in PAGE, "the old duplicate button is gone"
        keeping = PAGE[PAGE.index("function keepingHtml"):PAGE.index("function wireKeeping")]
        assert "measureIt" not in keeping, "measuring belongs beside the score"
        assert "sweepStale" in PAGE and "sweepAll" in PAGE and "stopSweep" in PAGE

    def test_a_stale_build_greys_the_buttons_out_and_does_not_remove_them(self) -> None:
        """The one state the panel is *for* must not be the state with no button in it.

        A core behind its sources is the commonest reason a measurement cannot be taken,
        and the first version of this panel answered it by hiding both buttons and
        printing a shell command -- which turned the screen that exists to fix stale
        scores back into a screen that only reports them. The buttons stay, disabled,
        carrying the reason.
        """
        rendered = _render_scoring({
            "games": [{"name": "chess", "state": "absent", "needs": "replay",
                       "reason": "never measured"}],
            "current": 0, "stale": 0, "absent": 1, "rescore": 0, "replay": 1,
            "roster": ["random"], "gamesPerPairing": 4, "weights": "v8",
            "build": "the loaded core was built from different sources", "sweep": None,
        })
        assert 'id="sweepStale"' in rendered and 'id="sweepAll"' in rendered
        assert "disabled" in rendered
        assert "the loaded core was built from different sources" in rendered

        working = _render_scoring({
            "games": [{"name": "chess", "state": "absent", "needs": "replay",
                       "reason": "never measured"}],
            "current": 0, "stale": 0, "absent": 1, "rescore": 0, "replay": 1,
            "roster": ["random"], "gamesPerPairing": 4, "weights": "v8",
            "build": "", "sweep": None,
        })
        assert 'id="sweepStale"' in working and "disabled" not in working

    def test_the_page_is_valid_javascript(self) -> None:
        """The page grew a hundred lines of it, and a syntax error is a blank screen."""
        import re

        from page_text import PAGE

        script = re.search(r"<script>(.*)</script>", PAGE, re.S)
        assert script is not None
        done = node_session.run(script.group(1), check=True)
        assert done.returncode == 0, done.stderr


def _render_scoring(payload: dict) -> str:
    """Run the page's own `scoringHtml` over a payload, in node.

    The page is a string and nothing type-checks it, so the alternative is asserting on
    substrings of the source -- which cannot tell a button that is rendered from one
    inside a branch that never runs. Skipped where node is absent rather than
    approximated.
    """
    import json as _json
    import re

    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                                 set: () => true });
      global.document = { getElementById: () => el, querySelector: () => null,
                          querySelectorAll: () => [], addEventListener: () => {},
                          createElement: () => el, body: el };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, scrollTo: () => {} };
      global.location = { pathname: '/game', hash: '' };
      global.fetch = () => new Promise(() => {});
    """
    call = "console.log(scoringHtml(" + _json.dumps(payload) + "));"
    done = node_session.run(stub + script.group(1) + chr(10) + call)
    assert done.returncode == 0, done.stderr
    return done.stdout


def _finished(app: App, timeout: float = 15.0) -> dict:
    """Wait for the sweep to stop running, and hand back what it says it did."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sweep = app.get("/api/scoring")["sweep"]
        if sweep is not None and sweep["state"] != "running":
            return sweep
        time.sleep(0.05)
    raise AssertionError("the sweep never finished")


class TestBuildingAGameByHand:
    """The editor may fill in a form the schema already accepts, and nothing else.

    Every judgement about a draft is `spec/schema.py`'s or `loop/gates.py`'s, asked
    rather than restated -- so a game built in a browser is refused for the same reasons,
    in the same words, as one a model proposed.
    """

    def test_every_template_is_a_game_that_already_plays(self) -> None:
        """The editor's first draw is a board with pieces on it, not a list of errors.

        A blank spec is not a starting point -- it fails validation for having no
        pieces -- so each template is a complete ruleset that compiles, passes the cheap
        gate, and does not collide with anything already in the corpus. The last of those
        matters: the hash covers mechanics only, so a template that *was* `dull` would be
        unsaveable for the least obvious reason available.
        """
        from chess_variant_sandbox.web import designing

        known = {
            spec_hash(load_spec(path))
            for directory in (library.RULESET_DIR, library.ARCHIVE_DIR)
            for path in Path(directory).glob("*.json")
        }
        for template in designing.templates():
            report = designing.check(template["spec"])
            assert report["valid"], (template["key"], report["errors"])
            assert report["playable"] is True, (template["key"], report["playableReasons"])
            assert report["hash"] not in known, f"{template['key']} is a game we have"

    def test_every_template_passes_the_screen(self) -> None:
        """A starting point a searching player cannot win is not a starting point.

        The designer shows the rungs beside the draft, so a template that fails one puts
        a red mark on the page before anybody has changed a rule. The skirmish template
        did exactly that -- a searching player won none of its eight games, because both
        sides could decline contact forever and draw on the no-progress clock -- and
        `must_capture` is the rule that fixed it (Q220).

        Seeds beyond the screen's own 0 because a template that only passes at one seed
        passes by luck. Eight games at two small budgets, four seeds, three templates:
        about a second in total.
        """
        from chess_variant_sandbox.loop.screen import screen
        from chess_variant_sandbox.web import designing

        for template in designing.templates():
            spec = designing.settled(template["spec"])
            for seed in range(4):
                result = screen(spec, seed=seed)
                assert result.passed, (template["key"], seed, result.reasons, result.stats)

    def test_a_draft_that_is_not_a_ruleset_is_refused_rather_than_saved(
        self, tmp_path
    ) -> None:
        """`validate_spec` *returns* its problems rather than raising them.

        A bare call to it therefore reads like a check and is a no-op, which let
        `{"name": "broken"}` into the archive -- where it became a library card whose
        every panel failed on a spec with no board.
        """
        from chess_variant_sandbox.web import designing

        app = App(RULESETS, tmp_path / "runs", tmp_path / "heur", library.ARCHIVE_DIR)
        drafts = [
            {"name": "broken"},
            {},
            {"name": "x", "board": {"width": 4, "height": 4},
             "pieces": [], "setup": {"placements": []}},
        ]
        for draft in drafts:
            with pytest.raises(ValueError):
                app.post("/api/design/save", {"spec": draft})

        report = designing.check({"name": "x"})
        assert report["valid"] is False
        # One problem per line: a list of six things wrong with a draft is a list.
        assert len(report["errors"]) > 1
        assert any("board" in error for error in report["errors"])

    def test_a_saved_game_is_a_candidate_and_not_a_corpus_entry(self, tmp_path) -> None:
        """Building a game must not be a way into the calibration corpus.

        It lands under `runs/`, which git ignores, and climbs the same two rungs as
        anything a design run proposed -- and `corpus` refuses to promote it for the same
        reason it refuses any unmeasured candidate.
        """
        from chess_variant_sandbox.web import designing

        runs = tmp_path / "runs"
        app = App(RULESETS, runs, tmp_path / "heur", library.ARCHIVE_DIR)
        spec = json.loads(json.dumps(designing.templates()[1]["spec"]))
        spec["name"] = "a hand-built game"
        kept = app.post("/api/design/save", {"spec": spec})
        assert kept["created"] is True
        assert Path(kept["path"]).exists()

        game = next(g for g in app.refresh() if g.hash == kept["hash"])
        # Not "evolved": a mutation and a drawing are both candidates in a run directory
        # and are not the same thing to a reader.
        assert game.source == "designed"
        assert game.summary()["index"] is None
        state = library.keeping(game, runs, RULESETS, tmp_path / "archive")
        assert state["stage"] == "candidate"
        assert state["blocker"], "an unmeasured candidate cannot be promoted"

    def test_saving_a_rename_is_not_a_second_game(self, tmp_path) -> None:
        """The hash covers mechanics only, which is the rule that stops a design run
        spending a tournament on a proposal that only renamed its parent."""
        from chess_variant_sandbox.web import designing

        app = App(RULESETS, tmp_path / "runs", tmp_path / "heur", library.ARCHIVE_DIR)
        spec = json.loads(json.dumps(designing.templates()[0]["spec"]))
        first = app.post("/api/design/save", {"spec": spec})
        spec["name"] = "something else entirely"
        second = app.post("/api/design/save", {"spec": spec})
        assert second["hash"] == first["hash"]
        assert second["created"] is False

    def test_a_draft_saved_without_a_description_gets_one_when_saved_again(
        self, tmp_path
    ) -> None:
        """Q595: the same rules saved with new words are the same game with those words.
        It used to answer *Already saved* and keep the empty description for ever."""
        from chess_variant_sandbox.web import designing

        runs = tmp_path / "runs"
        app = App(RULESETS, runs, tmp_path / "heur", library.ARCHIVE_DIR)
        spec = json.loads(json.dumps(designing.templates()[0]["spec"]))
        spec["name"], spec["description"] = "a draft", ""
        first = app.post("/api/design/save", {"spec": spec})
        assert (first["restated"], first["unchanged"]) == ([], [])

        spec["description"] = "Now it says what it is."
        second = app.post("/api/design/save", {"spec": spec})

        assert second["hash"] == first["hash"] and second["created"] is False
        assert (second["restated"], second["unchanged"]) == (["description"], [])
        game = next(g for g in app.refresh() if g.hash == first["hash"])
        assert (game.name, game.description) == ("a draft", "Now it says what it is.")
        lineage = (runs / "designed" / "lineage.jsonl").read_text(encoding="utf-8")
        assert len(lineage.splitlines()) == 1, "a new description is not a new entry"

        spec["name"] = "a better name"
        third = app.post("/api/design/save", {"spec": spec})
        assert (third["restated"], third["name"]) == (["name"], "a better name")
        game = next(g for g in app.refresh() if g.hash == first["hash"])
        assert (game.name, game.description) == ("a better name", "Now it says what it is.")

        again = app.post("/api/design/save", {"spec": spec})
        assert (again["restated"], again["unchanged"]) == ([], [])

    def test_a_save_told_not_to_restate_keeps_the_words_and_names_them(
        self, tmp_path
    ) -> None:
        """The site's half (Q595): rules somebody else saved first, or a published game,
        keep their words; the answer names the ones the save asked for and did not get."""
        from chess_variant_sandbox.web import designing

        runs = tmp_path / "runs"
        spec = json.loads(json.dumps(designing.templates()[0]["spec"]))
        spec["name"], spec["description"] = "a draft", "as published"
        first = designing.save(spec, runs_dir=runs)
        spec["name"], spec["description"] = "renamed", "other words"

        kept = designing.save(spec, runs_dir=runs, restate=False)

        assert kept["hash"] == first["hash"] and kept["name"] == "a draft"
        assert (kept["restated"], kept["unchanged"]) == ([], ["name", "description"])
        on_disk = json.loads(Path(kept["path"]).read_text(encoding="utf-8"))
        assert (on_disk["name"], on_disk["description"]) == ("a draft", "as published")

    def test_a_hand_built_game_can_be_played_at_once(self, tmp_path) -> None:
        from chess_variant_sandbox.web import designing

        app = App(RULESETS, tmp_path / "runs", tmp_path / "heur", library.ARCHIVE_DIR)
        kept = app.post("/api/design/save", {"spec": designing.templates()[2]["spec"]})
        snapshot = app.post(
            "/api/session", {"hash": kept["hash"], "seats": ["human", "minimax:nodes=400"]}
        )
        assert snapshot["moves"], "a saved game with no legal opening move"

    # ------------------------------------------------ a fork that changed nothing

    def test_a_fork_that_changed_no_rule_is_refused_by_name(self, tmp_path) -> None:
        """The worst behaviour the editor had, and the clearest thing it can now say.

        *Make one like this* on chess gives you `chess variant`; the hash covers mechanics
        only, so saving it unchanged wrote a candidate under chess's own hash --
        and `library.discover` resolves a hash collision in favour of the committed file,
        so the page said "Saved as chess variant", `created: true`, and *open its page*
        went to chess. The candidate was invisible for ever and nothing warned.
        """
        app = App(RULESETS, tmp_path / "runs", tmp_path / "heur", library.ARCHIVE_DIR)
        chess = app.get("/api/design/4c1922ffae26")
        fork = json.loads(json.dumps(chess["spec"]))
        fork["name"] = chess["name"] + " variant"
        fork["description"] = ""
        fork.pop("provenance", None)
        # The premise of the bug, asserted rather than assumed.
        assert spec_hash(fork) == chess["hash"]

        with pytest.raises(ValueError) as refused:
            app.post("/api/design/save", {"spec": fork, "parent": chess["hash"]})
        said = str(refused.value)
        assert "Nothing you changed is a rule" in said
        assert chess["hash"] not in said, "a game is never named by its hash"
        # By *name*: a person who renamed a game and watched it disappear was told
        # nothing, and a hash is not what they called it.
        assert "chess" in said
        assert not (tmp_path / "runs" / "designed").exists(), "it was written anyway"

    def test_a_changed_rule_makes_a_different_game_that_saves(self, tmp_path) -> None:
        """The other side of the refusal: it must not stand in the way of a real fork."""
        app = App(RULESETS, tmp_path / "runs", tmp_path / "heur", library.ARCHIVE_DIR)
        chess = app.get("/api/design/4c1922ffae26")
        fork = json.loads(json.dumps(chess["spec"]))
        fork["name"] = "wider chess"
        fork["board"]["width"] = 9
        kept = app.post("/api/design/save", {"spec": fork, "parent": chess["hash"]})
        assert kept["created"] is True
        assert kept["hash"] != chess["hash"]

    def test_a_draft_opened_from_a_game_is_told_what_its_measurement_costs(self) -> None:
        """Both hashes and the consequence, before anything has been changed.

        The number is `Game.current_score`'s, so the one policy about a stale measurement
        holds here too. Since EV2 the game's own page shows a stale number marked
        *provisional*, but a draft's parent is one of the places a provisional number never
        reaches (`docs/archive/plan-evolving-score.md` §3, `test_provisional_estimate`): the editor
        withholds it and says *out of date* from the staleness it carries. The assertion
        holds whether chess's library measurement is current or stale today.
        """
        app = App(RULESETS, library.RUNS_DIR, "heuristics", library.ARCHIVE_DIR)
        opened = app.get("/api/design/4c1922ffae26")
        assert opened["hash"] == "4c1922ffae26"
        assert opened["name"] == "chess"
        assert opened["stage"] == "committed"
        assert "index" in opened and "staleness" in opened
        assert opened["roster"], "'measured' means nothing without saying against what"
        game = app.game("4c1922ffae26")
        assert opened["index"] == (game.current_score or {}).get("index")
        if game.status == library.PROVISIONAL:
            assert opened["index"] is None, "a provisional number never reaches a parent"
            assert opened["staleness"] and not opened["staleness"]["current"]

    # ------------------------------------------------ the version stamp is settled

    def test_a_later_schemas_rule_needs_no_hand_edited_stamp(self) -> None:
        """The trap: the templates declare 2, 1 and 2, twenty-two of the twenty-six rule
        keys are reachable only through the JSON box, and most of them arrived after
        schema 2. Pasting one was answered with a good sentence about a number nobody
        should have to know."""
        from chess_variant_sandbox.web import designing

        draft = json.loads(json.dumps(designing.templates()[0]["spec"]))
        draft["schema_version"] = 2
        draft["rules"]["obstacles"] = {"squares": ["c3", "d4"]}  # schema 6's rule

        report = designing.check(draft)
        assert report["valid"] is True, report["errors"]
        # Since Q1008c no stamp is read and none is reported: the name ignores it.
        assert "schemaVersion" not in report
        assert report["hash"] == spec_hash({**draft, "schema_version": 6})
        assert any("walls" in line for line in report["ruleLines"])

    def test_no_ruleset_in_the_library_is_refused_for_its_version_stamp(self) -> None:
        """The exit criterion, taken over the corpus rather than over a hand-written list.

        Every rule key any committed or archived ruleset uses, stamped 1 and stamped 9,
        has to come back valid -- and at the same hash either way, because where the
        settling *ends* is what makes two people who guessed differently about the stamp
        build one game instead of two.
        """
        from chess_variant_sandbox.spec import validate_spec
        from chess_variant_sandbox.web import designing

        paths = sorted(
            list(Path(RULESETS).glob("*.json"))
            + list(Path(library.ARCHIVE_DIR).glob("*.json"))
        )
        assert len(paths) > 40, "the corpus this test rests on has gone missing"
        seen: set[str] = set()
        for path in paths:
            spec = load_spec(path)
            seen.update(spec.get("rules", {}))
            settled = [
                designing.settled({**spec, "schema_version": stamp}) for stamp in (1, 9)
            ]
            for stamped in settled:
                assert validate_spec(stamped) == [], (path.name, stamped["schema_version"])
            assert spec_hash(settled[0]) == spec_hash(settled[1]), path.name

        from chess_variant_sandbox.spec.schema import _RULES_KEYS

        # What the corpus could not cover, so the next reader knows the bound rather than
        # having to re-derive it. The hand-built half is the test below.
        # `contacts` left this set on 2026-09-07, when Q101's `archive/succession` became
        # the first library spec to use one (`schema-showcase-coverage.md` row 30).
        # `promote_on_entry` left it on 2026-10-09, when Q787's `archive/shogi-drop-promoted`
        # became the first library spec to use it (row 21); the hand-built arm below keeps it.
        # `portals` (schema 19, Q771) joined it, and left on 2026-10-10 with Q772a's
        # showcases (`schema-showcase-coverage.md` rows 53 to 55).
        # `simultaneous` (schema 26, Q918a) joined it; Q918e's showcases take it out (rows 64
        # to 67).
        # `simultaneous` left the set with Q918e's four round showcases (2026-10-15).
        # `bare_royal` left it on 2026-10-21, when Q994p's `archive/the-golem-well`, a
        # model's design from RULES.md alone, became the first library spec to use it.
        # `duck` (generation 42, N4b) joined it, and left on 2026-10-26 with N6's
        # `archive/the-shepherds-duck`.
        assert _RULES_KEYS - seen == {
            "checkmate", "turn",
        }, "the rule keys nothing in the library uses have changed"

    def test_the_rule_keys_nothing_in_the_library_uses_are_not_refused_either(
        self,
    ) -> None:
        """The four nothing in the library uses, built by hand for that reason.

        A key may still be refused for a *coherence* reason -- `promote_on_entry` needs
        somewhere to enter play from -- and that is not this test's business. What may
        never happen is a refusal naming the version stamp, on a draft the editor settles.

        `contacts` is the fifth case and is kept although `archive/succession` now uses
        one: the question here is about a *draft in the editor*, where a two-piece
        template's only possible contact is a pawn meeting a pawn, and the library arm
        above cannot ask it.
        """
        from chess_variant_sandbox.spec import validate_spec
        from chess_variant_sandbox.web import designing

        base = json.loads(json.dumps(designing.templates()[1]["spec"]))  # a king and a pawn
        cases = {
            "bare_royal": "loss",
            "checkmate": "win",
            "turn": {"moves": 2},
            "promote_on_entry": True,
            # Schema 10 (Q19a). The template is a king and a pawn, so the one contact
            # the army can have is a pawn meeting a pawn.
            "contacts": {
                "immune": ["K"],
                "links": [{"mover": "P", "onto": "P", "mover_to": "K"}],
            },
            # Schema 19 (Q771): a pair moved like a portal piece the draft gains below.
            "portals": [{"piece": "O", "orange": "c3", "blue": "d4"}],
            # Generation 42 (N4b): a duck the draft gains below, in a game won by taking.
            "duck": {"piece": "U"},
        }
        for key, value in cases.items():
            draft = json.loads(json.dumps(base))
            draft["rules"][key] = value
            if key == "promote_on_entry":
                # It promotes a piece that *enters play*, so there has to be a way in.
                draft["rules"]["drops"] = {"from_captures": True}
            if key == "portals":
                draft["pieces"].append({"id": "O", "name": "portal", "value": 0, "moves": []})
            if key == "duck":
                draft["pieces"].append({"id": "U", "name": "duck", "value": 0, "moves": []})
                draft["rules"]["win"] = "royal_capture"
            problems = validate_spec(designing.settled(draft))
            assert not any("schema" in problem for problem in problems), (key, problems)

    # ------------------------------------------------ trying it before keeping it

    def test_a_draft_can_be_played_without_being_kept(self, tmp_path) -> None:
        """The loop that matters most, and it was closed by a write to the archive.

        *save and play* saved first, because starting a session took a hash and looked it
        up in the library. The only honest answer to "is this rule interesting" is playing
        it, so ten moves of an idea you throw away used to leave a candidate behind.
        """
        from chess_variant_sandbox.web import designing

        runs = tmp_path / "runs"
        app = App(RULESETS, runs, tmp_path / "heur", library.ARCHIVE_DIR)
        draft = json.loads(json.dumps(designing.templates()[0]["spec"]))
        draft["name"] = "an idea"
        draft["rules"]["obstacles"] = {"squares": ["c3"]}  # and a stamp it never declared

        snapshot = app.post(
            "/api/design/play", {"spec": draft, "seats": ["human", "minimax:nodes=400"]}
        )
        assert snapshot["ruleset"]["scratch"] is True
        assert snapshot["moves"], "a draft with no legal opening move"
        # The rules panel travels with it, because the board's tabs fetch by hash and a
        # draft has no hash the library can resolve.
        assert any("walls" in line for line in snapshot["detail"]["ruleLines"])

        played = app.post(
            f"/api/session/{snapshot['id']}/move", {"move": snapshot["moves"][0]["move"]}
        )
        assert played["history"], "the move did not land"
        app.post(f"/api/session/{snapshot['id']}/agent", {})

        assert not runs.exists() or not list(runs.rglob("*.json*")), (
            "playing a draft wrote something to runs/"
        )

    def test_a_scratch_game_leaves_no_row_in_the_human_record(self, tmp_path) -> None:
        """`web/human.py` is keyed by ruleset hash and read back by hash, so a game
        against rules nobody kept would leave a row nothing can ever name -- and it would
        move where the anchor says the ladder sits."""
        from chess_variant_sandbox.web import designing

        runs = tmp_path / "runs"
        app = App(RULESETS, runs, tmp_path / "heur", library.ARCHIVE_DIR)
        draft = json.loads(json.dumps(designing.templates()[0]["spec"]))
        snapshot = app.post(
            "/api/design/play", {"spec": draft, "seats": ["human", "minimax:nodes=400"]}
        )
        app.post(
            f"/api/session/{snapshot['id']}/move", {"move": snapshot["moves"][0]["move"]}
        )
        assert app.get("/api/human")["rulesets"] == []
        assert app.humans.games(snapshot["ruleset"]["hash"]) == []

    def test_a_draft_that_is_not_a_ruleset_cannot_be_played(self, tmp_path) -> None:
        app = App(RULESETS, tmp_path / "runs", tmp_path / "heur", library.ARCHIVE_DIR)
        with pytest.raises(ValueError):
            app.post("/api/design/play", {"spec": {"name": "broken"}})

    def test_a_draft_is_offered_design_steps_and_keeps_none_of_them(
        self, tmp_path
    ) -> None:
        """Stage 4's *and now try*, on the draft in hand rather than on a saved game.

        One route answers both questions the GET twin answers by query -- which steps
        apply, and one of them applied -- and the draft travels in the body, because the
        thing in the editor is usually not a game the library could name.
        """
        from chess_variant_sandbox.web import designing

        runs = tmp_path / "runs"
        app = App(RULESETS, runs, tmp_path / "heur", library.ARCHIVE_DIR)
        draft = json.loads(json.dumps(designing.templates()[1]["spec"]))
        draft["name"] = "an idea"

        listed = app.post("/api/design/steps", {"spec": draft})
        assert listed["why"] == ""
        assert len(listed["steps"]) > 5
        step = listed["steps"][0]

        made = app.post(
            "/api/design/steps",
            {"spec": draft, "step": step["operator"], "seed": step["seed"]},
        )["remix"]
        assert made["words"] == step["words"]
        assert made["openingMoves"] > 0
        # The draft's own name survives a press, and the game that came out is a
        # different one.
        assert made["spec"]["name"] == "an idea"
        assert made["hash"] != spec_hash(designing.settled(draft))
        assert not runs.exists() or not list(runs.rglob("*.json*")), (
            "a design step wrote something to runs/"
        )

    def test_a_seed_a_page_invented_is_refused_rather_than_guessed_at(
        self, tmp_path
    ) -> None:
        from chess_variant_sandbox.web import designing

        app = App(RULESETS, tmp_path / "runs", tmp_path / "heur", library.ARCHIVE_DIR)
        draft = designing.templates()[0]["spec"]
        with pytest.raises(ValueError, match="is not one"):
            app.post(
                "/api/design/steps",
                {"spec": draft, "step": "join-the-edges", "seed": "soon"},
            )

    def test_a_saved_draft_is_either_end_of_a_gap(self, tmp_path) -> None:
        """*Make the game in between* on a person's own game (Q354).

        `?between=` used to take only the games the map draws, so a draft was no end of a
        gap even on the machine of the person who made it. The local application has one
        person and every draft under `runs/` is theirs, so both ends answer here; the
        site narrows them by its store (`tests/test_site_links.py`).
        """
        from chess_variant_sandbox.spec import relate

        dull = load_spec(RULESETS / "dull.json")
        draft = json.loads(json.dumps(dull))
        draft["name"] = "dull, longer and harsher"
        draft["rules"]["max_plies"] = 80
        draft["rules"]["stalemate"] = "loss"
        app = App(RULESETS, tmp_path / "runs", tmp_path / "heur", library.ARCHIVE_DIR)
        kept = app.post("/api/design/save", {"spec": draft})
        assert app.game(kept["hash"]).source == "designed"

        for path, other in ((kept["hash"], spec_hash(dull)), (spec_hash(dull), kept["hash"])):
            made = app.get(f"/api/design/{path}", {"between": other})["between"]
            assert "spec" in made, made.get("why")
            assert relate.units(made["spec"], dull) == 1
            assert relate.units(made["spec"], draft) == 1

    # ------------------------------------------- the game in between, on a draft's page

    @staticmethod
    def _dull_with(name: str, **rules: object) -> dict:
        spec = json.loads(json.dumps(load_spec(RULESETS / "dull.json")))
        spec["name"] = name
        spec["rules"].update(rules)
        return spec

    def _two_game_library(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> App:
        """`dull` and a game one rule from it as the whole library, in a tree of its own.

        Its own tree so the nearest library game is a fact of the fixture rather than of
        whatever the corpus holds this week. Fairy-Stockfish is told to be absent: its
        verdict is a second engine's handshake per game, half a second each, and the game
        page's opponent list is nothing these tests ask about.
        """
        from chess_variant_sandbox import fairy

        monkeypatch.setattr(fairy, "available", lambda: False)
        monkeypatch.setattr(fairy, "verdict", lambda spec: None)
        rulesets = tmp_path / "rulesets"
        rulesets.mkdir()
        (tmp_path / "archive").mkdir()
        for stem, spec in (
            ("dull", load_spec(RULESETS / "dull.json")),
            ("dull_shorter", self._dull_with("dull, shorter", max_plies=40)),
        ):
            (rulesets / f"{stem}.json").write_text(json.dumps(spec), encoding="utf-8")
        return App(rulesets, tmp_path / "runs", tmp_path / "heur", tmp_path / "archive")

    def test_a_draft_s_page_offers_the_game_between_it_and_the_nearest_library_game(
        self, tmp_path, monkeypatch
    ) -> None:
        """Q429. The map offers *make the game in between* on a gap and draws no drafts,
        so the game between a draft and the library had a route and no page. A draft's own
        page carries the library game most like it -- the first of its own *games like
        this one* (M5: the map's likeness) -- and the press is the map's own request.

        A second draft one rule away is named nowhere on the page: the games like this one
        are the map's games, and the other end of a gap is a *library* game, which is what
        the map's gaps join.
        """
        from chess_variant_sandbox.spec import relate

        app = self._two_game_library(tmp_path, monkeypatch)
        draft = self._dull_with("dull, longer and harsher", max_plies=80, stalemate="loss")
        kept = app.post("/api/design/save", {"spec": draft})
        beside = app.post(
            "/api/design/save",
            {"spec": self._dull_with("dull, longest", max_plies=90, stalemate="loss")},
        )

        page = app.get(f"/api/game/{kept['hash']}")
        near = page["between"]

        assert beside["hash"] not in {link["hash"] for link in page["links"]}
        assert all(link["source"] in server_module.MAPPED for link in page["links"])
        assert (page["onMap"], page["community"]) == (False, None)
        first = page["links"][0]
        assert (near["hash"], near["likeness"], near["units"]) == (
            first["hash"], first["likeness"], first["units"]
        )
        assert near["units"] == 2 and "distance" not in near
        assert {near["from"], near["taken"]} == {kept["hash"], first["hash"]}
        other = app.game(first["hash"]).spec

        made = app.get(f"/api/design/{near['from']}", {"between": near["taken"]})["between"]
        assert made["words"] == near["words"]
        assert relate.units(made["spec"], other) == 1
        assert relate.units(made["spec"], draft) == 1

    def test_one_rule_from_the_library_has_nothing_in_between(
        self, tmp_path, monkeypatch
    ) -> None:
        """A draft one step from a library game is most drafts, and two games one rule
        apart have no third between them. The page is told so in those words, and is
        given nothing to press."""
        app = self._two_game_library(tmp_path, monkeypatch)
        kept = app.post(
            "/api/design/save", {"spec": self._dull_with("dull, longer", max_plies=80)}
        )

        near = app.get(f"/api/game/{kept['hash']}")["between"]

        assert near["units"] == 1
        assert near["why"] == (
            f"{near['name']} is one rule from this game, so no game stands between them."
        )
        assert not {"from", "taken", "words"} & set(near)

    def test_a_library_game_s_page_is_sent_none(self, tmp_path, monkeypatch) -> None:
        """Its gaps are on the map, where the button already is."""
        app = self._two_game_library(tmp_path, monkeypatch)
        dull = spec_hash(load_spec(RULESETS / "dull.json"))

        assert app.get(f"/api/game/{dull}")["between"] is None

    def test_the_page_draws_what_the_server_sent_and_nothing_else(self) -> None:
        """No button without an answer, the button with one, a sentence instead of the
        button where there is no game in between, and the words for what *most like* was
        measured by: the map's likeness, the same as the games like this one."""
        mine = {"hash": "d1", "name": "my draft", "between": None}
        assert _in_the_page("", f"betweenHtml({json.dumps(mine)})").strip() == ""

        near = {"hash": "l1", "name": "dull", "units": 2, "likeness": 0.5,
                "from": "d1", "taken": "l1", "words": "how long a game may run"}
        drawn = _in_the_page("", f"betweenHtml({json.dumps({**mine, 'between': near})})")
        assert 'id="between"' in drawn and "make the game in between" in drawn
        assert "<b>my draft</b> with how long a game may run taken from" in drawn
        assert "most like this one is <b>dull</b>, 0.500 alike" in drawn
        assert "2 rules apart" in drawn and 'data-gloss="alike"' in drawn
        assert "nearest by how many rules" not in drawn and "overall likeness" not in drawn

        why = {"hash": "l1", "name": "dull", "units": 1, "likeness": 0.7,
               "why": "dull is one rule from this game, so no game stands between them."}
        drawn = _in_the_page("", f"betweenHtml({json.dumps({**mine, 'between': why})})")
        assert "no game stands between them" in drawn and "one rule apart" in drawn
        assert 'id="between"' not in drawn

    def test_the_press_asks_the_map_s_route_and_opens_the_map_s_editor(self) -> None:
        """What the button does, run in node: the map's request, `from` in the path and
        `taken` in the query, and what comes back handed to `mapOpenDraft` -- or, where
        the server finds nothing after all, its sentence beside the button and no editor."""
        setup = """
          const asked = [], opened = [], said = [];
          const button = { disabled: false, closest: () => null,
                           querySelectorAll: () => [], append: box => said.push(box.text) };
          document.getElementById = id => (id === 'between' ? button : el);
          document.createElement = () => ({ setAttribute() {}, remove() {},
                                            set textContent(words) { this.text = words; } });
          let reply = { hash: 'l1', between: { spec: { name: 'made' } } };
          api = path => { asked.push(path); return Promise.resolve(reply); };
          mapOpenDraft = (answer, made) => opened.push([answer.hash, made.spec.name]);
          wireBetween({ hash: 'd1', name: 'mine', between: { hash: 'l1', name: 'dull',
            units: 2, likeness: 0.5, from: 'l1', taken: 'd1', words: 'w' } });
          button.onclick();
          setTimeout(() => {
            reply = { hash: 'l1', between: { why: 'nothing after all' } };
            button.onclick();
            setTimeout(() => console.log(JSON.stringify(
              { asked, opened, said, disabled: button.disabled })), 0);
          }, 0);
        """
        printed = _in_the_page(setup, "''").strip().splitlines()[-1]
        seen = json.loads(printed)
        assert seen["asked"] == ["/api/design/l1?between=d1"] * 2
        assert seen["opened"] == [["l1", "made"]]
        assert seen["said"] == ["nothing after all"]
        assert seen["disabled"] is False, "a refused press can be pressed again"

        from page_text import PAGE

        start = PAGE.index("async function showGame(")
        shown = PAGE[start : PAGE.index("\n}\n", start)]
        assert "${betweenHtml(game)}" in shown and "wireBetween(game);" in shown

    def test_the_move_map_says_which_piece_it_stood_up(self) -> None:
        """So the editor can draw the map on an empty board without inventing a board
        byte. The encoding is player-major and carries the piece's initial facing; a
        second opinion about it is how two drawings of one piece come to disagree."""
        from chess_variant_sandbox.web import designing

        spec = load_spec(RULESETS / "chess.json")
        knight = designing.move_map(spec, 1, square=36)
        engine = Engine(spec)
        compiled = engine.spec
        assert knight["value"] == compiled.encode_piece(0, 1, compiled.facing_initial[1])

    def test_the_move_map_is_the_engines_answer_and_not_the_pages(self) -> None:
        """A form full of `[[1, 2]]` cannot say whether that is the piece you meant.

        Checked against chess because published expectations exist for it: a pawn on e5
        may go to one square and take on two others, and those are three different
        squares -- which is exactly the distinction the three sets are for.
        """
        from chess_variant_sandbox.web import designing

        spec = load_spec(RULESETS / "chess.json")
        pawn = designing.move_map(spec, 0, square=36)  # e5
        assert pawn["moves"] == [44]
        assert sorted(pawn["captureOnly"]) == [43, 45]
        assert pawn["moveOnly"] == [44]

        knight = designing.move_map(spec, 1, square=36)
        assert len(knight["moves"]) == 8
        assert knight["moveOnly"] == [] and knight["captureOnly"] == []

    def test_an_uploaded_drawing_travels_with_the_spec_and_not_in_the_hash(self) -> None:
        """`appearance.icons` maps a piece to a string the schema leaves free, so an
        inlined drawing needed no schema change -- and appearance is outside
        `canonical.MECHANIC_KEYS`, so a game does not become a different game because its
        knight got a new picture."""
        from chess_variant_sandbox.spec import validate_spec
        from chess_variant_sandbox.web import designing

        drawn = designing.icon_data_uri(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
            '<circle cx="5" cy="5" r="4"/></svg>'
        )
        spec = json.loads(json.dumps(designing.templates()[0]["spec"]))
        before = spec_hash(spec)
        spec["appearance"] = {"icons": {"S": drawn}}
        assert validate_spec(spec) == []
        assert spec_hash(spec) == before

        art = icons_for(["soldier"], ["S"], spec["appearance"]["icons"])
        assert art[0][0]["href"] == drawn
        # One upload serves both seats, turned round for the second -- exactly what a
        # single-colour library set does, and what makes the sides "first" and "second".
        assert art[0][1]["rotate"] is True

    def test_a_drawing_that_is_not_one_is_refused(self) -> None:
        from chess_variant_sandbox.web import designing

        with pytest.raises(ValueError):
            designing.icon_data_uri("not an svg at all")
        with pytest.raises(ValueError):
            designing.icon_data_uri("<svg><script>alert(1)</script></svg>")
        with pytest.raises(ValueError):
            designing.icon_data_uri("<svg>" + "x" * 70_000 + "</svg>")

    def test_the_size_refusal_is_in_plain_words(self) -> None:
        from chess_variant_sandbox.web import designing

        with pytest.raises(ValueError) as caught:
            designing.icon_data_uri("<svg>" + "x" * 70_000 + "</svg>")
        assert "the spec" not in str(caught.value)
        assert "the rules have to stay a file" in str(caught.value)


class TestBuildingAnOpponent:
    """A heuristic is a name, a sentence and fourteen numbers, which is why this can be a
    form at all. The refusals are `Heuristic`'s own."""

    def test_what_is_built_can_be_played(self, tmp_path, chess_game) -> None:
        heuristics = tmp_path / "heur"
        heuristics.mkdir()
        app = App(RULESETS, tmp_path / "runs", heuristics, library.ARCHIVE_DIR)
        before = len(app.roster(chess_game.hash))
        built = app.post("/api/opponent", {
            "hash": chess_game.hash, "name": "chase the king",
            "plan": "Hunt the king and never mind the pieces.",
            "weights": _weights(material=0.2, royal_exposure=1.0),
        })
        roster = app.roster(chess_game.hash)
        assert len(roster) == before + 1
        mine = next(entry for entry in roster if entry.agent == built["agent"])
        assert mine.label == "chase the king"
        # Listing an opponent and being able to build it are the same act.
        assert build_agent(mine.agent, Engine(chess_game.spec)) is not None

        snapshot = app.post("/api/session", {"hash": chess_game.hash,
                                             "seats": ["human", built["agent"]]})
        snapshot = app.post(f"/api/session/{snapshot['id']}/move",
                            {"move": snapshot["moves"][0]["move"]})
        snapshot = app.post(f"/api/session/{snapshot['id']}/agent", {})
        assert snapshot["history"][-1]["by"] == built["agent"]

        app.post("/api/opponent/forget",
                 {"hash": chess_game.hash, "digest": built["digest"]})
        assert len(app.roster(chess_game.hash)) == before

    def test_the_first_one_built_is_not_swallowed_by_the_anchor_rule(
        self, tmp_path, chess_game
    ) -> None:
        """Index 0 of a *generated* set is the built-in evaluator's own vector and is
        skipped. A set somebody built by hand has no anchor, so skipping its first entry
        would silently drop the first opponent they ever made."""
        heuristics = tmp_path / "heur"
        heuristics.mkdir()
        app = App(RULESETS, tmp_path / "runs", heuristics, library.ARCHIVE_DIR)
        built = app.post("/api/opponent", {
            "hash": chess_game.hash, "name": "the only one", "plan": "",
            "weights": _MATERIAL_ONLY,
        })
        agents = [entry.agent for entry in app.roster(chess_game.hash)]
        assert built["agent"] in agents

    @staticmethod
    def _a_written_plan(heuristics: Path, ruleset_hash: str) -> tuple[Heuristic, list[float]]:
        """A written store for the game, and the same plan's weights at twice the scale.

        One digest, other raw numbers: two players under one name (Q433). Index 0 is the
        anchor slot the picker skips, so the plan sits at index 1 and is offered.
        """
        plan = Heuristic("written plan", "", tuple(_weights(material=1.0, mobility=0.29)))
        anchor = Heuristic("anchor", "", tuple(_MATERIAL_ONLY))
        HeuristicSet(ruleset_hash, (anchor, plan)).save(store_path(ruleset_hash, heuristics))
        doubled = _weights(material=2.0, mobility=0.58)
        assert Heuristic("mine", "", tuple(doubled)).digest == plan.digest
        return plan, doubled

    @pytest.mark.parametrize("listed_first", [True, False], ids=["listed", "not-listed"])
    def test_a_second_spelling_of_an_offered_plan_is_refused_before_it_is_written(
        self, tmp_path, chess_game, listed_first: bool
    ) -> None:
        """The error names both, and `custom/` is never written (Q433).

        Refusing only at registration wrote the file first, and a store that clashes is
        refused whole by the picker. Listed first, the written plan is registered in the
        process; not listed, it is still the store the picker registers before `custom/`.
        """
        heuristics = tmp_path / "heur"
        heuristics.mkdir()
        app = App(RULESETS, tmp_path / "runs", heuristics, library.ARCHIVE_DIR)
        plan, doubled = self._a_written_plan(heuristics, chess_game.hash)
        if listed_first:
            assert any(entry.label == plan.name for entry in app.roster(chess_game.hash))
        with pytest.raises(ValueError) as refused:
            app.post("/api/opponent", {"hash": chess_game.hash, "name": "mine",
                                       "plan": "", "weights": doubled})
        assert "'mine'" in str(refused.value) and "'written plan'" in str(refused.value)
        assert not (heuristics / "custom").exists()
        assert any(entry.label == plan.name for entry in app.roster(chess_game.hash))

    def test_an_older_clash_in_the_store_is_refused_not_overwritten(
        self, tmp_path, chess_game
    ) -> None:
        """A `custom/` store written before Q433 can hold a second spelling already.

        Building beside it would add a player the picker never offers, since the picker
        refuses a clashing store whole, and reading the store as empty would overwrite
        every opponent in it. So the build is refused, naming the old entry, and the file
        is left as it was.
        """
        heuristics = tmp_path / "heur"
        heuristics.mkdir()
        app = App(RULESETS, tmp_path / "runs", heuristics, library.ARCHIVE_DIR)
        _plan, doubled = self._a_written_plan(heuristics, chess_game.hash)
        kept = Heuristic("kept", "", tuple(_weights(material=1.0, centrality=0.41)))
        older = Heuristic("older", "", tuple(doubled))
        path = store_path(chess_game.hash, heuristics / "custom")
        HeuristicSet(chess_game.hash, (kept, older), model="built by hand").save(path)
        before = path.read_bytes()
        app.roster(chess_game.hash)
        with pytest.raises(ValueError) as refused:
            app.post("/api/opponent", {"hash": chess_game.hash, "name": "new", "plan": "",
                                       "weights": _weights(material=1.0, centrality=0.9)})
        assert "'older'" in str(refused.value) and "'written plan'" in str(refused.value)
        assert path.read_bytes() == before

    def test_an_older_clash_leaves_the_picker_alone_and_says_so(
        self, tmp_path, chess_game, warm_map, warm_chess_reading
    ) -> None:
        """Q446. Only the clashing entry is left out, and the game page names it.

        `for_game` used to load `custom/` through `load_set`, which refuses a clashing
        store whole, so one opponent built before Q433 took every opponent built for the
        game out of the picker. The file itself is left as it was: skipping the entry in
        `load_set` instead would have let `forget_opponent` save the shorter set.
        """
        from chess_variant_sandbox.agents.heuristics import resolve

        heuristics = tmp_path / "heur"
        heuristics.mkdir()
        app = App(RULESETS, tmp_path / "runs", heuristics, library.ARCHIVE_DIR)
        plan, doubled = self._a_written_plan(heuristics, chess_game.hash)
        kept = Heuristic("kept", "", tuple(_weights(material=1.0, centrality=0.41)))
        older = Heuristic("older", "", tuple(doubled))
        after = Heuristic("after it", "", tuple(_weights(material=1.0, mobility=0.9)))
        path = store_path(chess_game.hash, heuristics / "custom")
        HeuristicSet(chess_game.hash, (kept, older, after), model="built by hand").save(path)
        before = path.read_bytes()

        roster = app.roster(chess_game.hash)
        built = [entry.label for entry in roster if entry.store == "custom"]
        assert built == ["kept", "after it"]
        # The digest the old entry shares is offered once, as the plan that holds it,
        # and resolves to that plan's own numbers.
        shared = [entry for entry in roster if f"h={plan.digest}," in entry.agent]
        assert [(entry.label, entry.store) for entry in shared] == [("written plan", "written")]
        assert resolve(plan.digest).weights == plan.weights
        engine = Engine(chess_game.spec)
        for entry in roster:
            if entry.store == "custom":
                assert build_agent(entry.agent, engine) is not None

        # Through the server, so the page is told about the store it was given. Only
        # the sentences about stores in `heuristics`: chess's valuation is the
        # committed one and says what it says.
        said = app.get(f"/api/game/{chess_game.hash}")["refusedStores"]
        assert [line for line in said if heuristics.as_posix() in line] == [
            "The opponent 'older' is not offered because it has the balance of 'written "
            "plan' with other exact numbers, and the engine plays the exact numbers, so one "
            "name would call two players; it is still stored in "
            f"`{path.as_posix()}`, and taking it out of that file is the repair."
        ]
        assert path.read_bytes() == before

    def test_a_clash_inside_one_store_leaves_out_only_the_second_spelling(
        self, tmp_path, chess_game
    ) -> None:
        """Two entries of one `custom/` store can share a digest too; the first holds it."""
        from chess_variant_sandbox.web import opponents

        heuristics = tmp_path / "heur"
        first = Heuristic("first", "", tuple(_weights(material=1.0, mobility=0.37)))
        second = Heuristic("second", "", tuple(_weights(material=3.0, mobility=1.11)))
        assert first.digest == second.digest and first.weights != second.weights
        HeuristicSet(chess_game.hash, (first, second), model="built by hand").save(
            store_path(chess_game.hash, heuristics / "custom")
        )
        built = [
            entry.label for entry in opponents.for_game(chess_game.hash, heuristics)
            if entry.store == "custom"
        ]
        assert built == ["first"]
        said = opponents.refused_stores(
            chess_game.hash, None, valuations=tmp_path / "values", heuristics=heuristics
        )
        assert len(said) == 1
        assert said[0].startswith("The opponent 'second' is not offered because it has "
                                  "the balance of 'first' with other exact numbers")

    def test_forgetting_survives_a_clash_inside_the_store(
        self, tmp_path, chess_game
    ) -> None:
        """`load_set` registers a whole file in one call and refuses it whole for a
        clash (Q433); through it, a `custom/` store holding one from before Q433 could
        never forget anything again, the entry actually asked for included (Q449)."""
        from chess_variant_sandbox.web import designing

        heuristics = tmp_path / "heur"
        first = Heuristic("first", "", tuple(_weights(material=1.0, advance=0.5)))
        second = Heuristic("second", "", tuple(_weights(material=2.0, advance=1.0)))
        assert first.digest == second.digest and first.weights != second.weights
        path = store_path(chess_game.hash, heuristics / "custom")
        HeuristicSet(chess_game.hash, (first, second), model="built by hand").save(path)

        removed = designing.forget_opponent(chess_game.hash, first.digest, heuristics)
        assert removed == {"kept": 1}
        left = HeuristicSet.load(path)
        assert [h.name for h in left.heuristics] == ["second"]

        # The digest names both spellings; the second is gone the same way.
        removed = designing.forget_opponent(chess_game.hash, second.digest, heuristics)
        assert removed == {"kept": 0}
        assert not path.exists()

    def test_the_same_numbers_under_another_name_still_build(
        self, tmp_path, chess_game
    ) -> None:
        """Identity is the numbers, so a written plan rebuilt exactly is no clash."""
        heuristics = tmp_path / "heur"
        heuristics.mkdir()
        app = App(RULESETS, tmp_path / "runs", heuristics, library.ARCHIVE_DIR)
        plan, _doubled = self._a_written_plan(heuristics, chess_game.hash)
        app.roster(chess_game.hash)
        built = app.post("/api/opponent", {"hash": chess_game.hash, "name": "copied",
                                           "plan": "", "weights": list(plan.weights)})
        assert built["digest"] == plan.digest

    def test_a_vector_that_is_not_a_player_is_refused(self, tmp_path, chess_game) -> None:
        heuristics = tmp_path / "heur"
        heuristics.mkdir()
        app = App(RULESETS, tmp_path / "runs", heuristics, library.ARCHIVE_DIR)
        for weights in (_ZEROS, _ZEROS[:-1], [99.0] + [0.0] * (len(VOCABULARY) - 1)):
            with pytest.raises(ValueError):
                app.post("/api/opponent", {"hash": chess_game.hash, "name": "x",
                                           "plan": "", "weights": weights})
        with pytest.raises(ValueError):
            app.post("/api/opponent", {"hash": chess_game.hash, "name": "  ",
                                       "plan": "", "weights": _MATERIAL_ONLY})

    def test_nothing_built_here_reaches_the_measured_store(
        self, tmp_path, chess_game
    ) -> None:
        """`heuristic_headroom` is a median over a *set*, so what is in the set moves a
        committed measurement. An opponent somebody invented while playing goes to a
        third store beside `model/`, which only the picker reads."""
        from chess_variant_sandbox.agents.heuristics import load_set

        heuristics = tmp_path / "heur"
        heuristics.mkdir()
        app = App(RULESETS, tmp_path / "runs", heuristics, library.ARCHIVE_DIR)
        app.post("/api/opponent", {"hash": chess_game.hash, "name": "mine", "plan": "",
                                   "weights": _MATERIAL_ONLY})
        assert (heuristics / "custom" / f"{chess_game.hash}.json").exists()
        assert not (heuristics / f"{chess_game.hash}.json").exists()
        assert load_set(chess_game.hash, heuristics) is None

    def test_which_sliders_do_anything_is_read_off_the_rules(self) -> None:
        """Six of the fourteen terms read a rule most rulesets do not declare, and a
        slider that silently does nothing is worse than one that is not offered."""
        from chess_variant_sandbox.web import designing

        chess = designing.live_terms(load_spec(RULESETS / "chess.json"))
        assert "material" in chess and "promotion_occupancy" in chess
        # `check_limit` is an int and reads 0 where the rule is absent, so an
        # `is not None` test marked this live on every ruleset in the corpus.
        assert "check_margin" not in chess
        assert "check_margin" in designing.live_terms(
            load_spec(RULESETS / "threecheck.json"))
        assert "hand_material" in designing.live_terms(
            load_spec(RULESETS / "crazyhouse.json"))
        assert "reach_progress" in designing.live_terms(
            load_spec(RULESETS / "racingkings.json"))
        assert "territory_margin" in designing.live_terms(
            load_spec(RULESETS / "rps-v2.json"))

    # -- stage 30C: what the pieces are worth to it -------------------------

    def test_what_each_piece_is_worth_to_the_rules_is_offered(self) -> None:
        """The second half of the panel, and the reason it is per-game.

        The weight vocabulary is fixed and comes with the design kit; a valuation is as
        long as the ruleset has pieces, so it comes with the ruleset. A royal piece is
        listed and marked rather than dropped, because `_NEEDS` already established that
        a control which cannot do anything should say why.
        """
        from chess_variant_sandbox.web import designing

        rows = designing.piece_values(load_spec(RULESETS / "chess.json"))
        assert [row["id"] for row in rows] == ["P", "N", "B", "R", "Q", "K"]
        assert rows[0]["derived"] == 1.0
        assert not rows[0]["royal"] and rows[0]["needs"] == ""
        king = rows[-1]
        assert king["royal"] and king["derived"] == 0.0
        assert "ends the game" in king["needs"]
        # An icon, so the panel can name a piece by its picture rather than by a letter.
        assert all(row["icon"] for row in rows)

    def test_a_slider_is_a_multiple_of_what_the_rules_imply(
        self, tmp_path, chess_game
    ) -> None:
        """Multipliers in, absolute values out, on the derived table's scale.

        A stored opponent must not change its mind when `_measured_values` does, which is
        the whole point of content-addressing a player -- so the file holds a number and
        not "five times". And the scale rule makes it a reallocation: raising the knight
        lowers everything else, and the mean over non-royal pieces is where it was.
        """
        from chess_variant_sandbox.agents.evaluation import Evaluator, mean_abs_non_royal
        from chess_variant_sandbox.agents.heuristics import load_set

        heuristics = tmp_path / "heur"
        heuristics.mkdir()
        app = App(RULESETS, tmp_path / "runs", heuristics, library.ARCHIVE_DIR)
        app.post("/api/opponent", {
            "hash": chess_game.hash, "name": "knight fancier", "plan": "",
            "weights": _MATERIAL_ONLY,
            "values": [1, 5, 1, 1, 1, 1],
        })
        compiled = Engine(chess_game.spec).spec
        stored = load_set(chess_game.hash, heuristics / "custom", compiled.piece_ids)
        assert stored is not None
        values = stored.heuristics[0].values
        derived = Evaluator.for_spec(compiled).values

        assert len(values) == compiled.n_types
        assert values[compiled.royal_types[0]] == 0.0
        # A fixed budget, reallocated: what the slider says is exact in the *ratios*,
        # and the absolute numbers all come down to pay for it. Asserting the knight's
        # own value rose fivefold would be asserting a volume knob, which is the thing
        # the scale rule exists to rule out.
        assert values[1] / values[0] == pytest.approx(5.0 * derived[1] / derived[0])
        assert values[0] < derived[0]
        assert mean_abs_non_royal(values, compiled.royal_types) == pytest.approx(
            mean_abs_non_royal(derived, compiled.royal_types), rel=1e-9
        )
        # Positional, so the file says which piece each slot is.
        assert stored.piece_ids == compiled.piece_ids

    def test_moving_every_slider_the_same_way_is_no_opinion_at_all(
        self, tmp_path, chess_game
    ) -> None:
        """The panel says so, so it has to be true of the *name* as well as of the play.

        An override is a reallocation of a fixed budget, so every slider at five is the
        player with no override. Storing a table equal to the derived one instead would
        give that player a second name, because the digest covers the values.
        """
        from chess_variant_sandbox.agents.heuristics import load_set

        heuristics = tmp_path / "heur"
        heuristics.mkdir()
        app = App(RULESETS, tmp_path / "runs", heuristics, library.ARCHIVE_DIR)
        plain = app.post("/api/opponent", {
            "hash": chess_game.hash, "name": "no opinion", "plan": "",
            "weights": _MATERIAL_ONLY,
        })
        louder = app.post("/api/opponent", {
            "hash": chess_game.hash, "name": "louder", "plan": "",
            "weights": _MATERIAL_ONLY,
            "values": [5, 5, 5, 5, 5, 5],
        })
        assert louder["digest"] == plain["digest"]
        stored = load_set(chess_game.hash, heuristics / "custom")
        assert stored is not None
        assert len(stored.heuristics) == 1
        assert stored.heuristics[0].values == ()

    def test_a_priced_royal_is_refused_rather_than_ignored(
        self, tmp_path, chess_game
    ) -> None:
        """The slider is disabled, so a value arriving for one came from somewhere else.

        Ignored rather than trusted, and the *table* is what carries the refusal: the
        panel cannot offer the control, so a refusal naming it would name something the
        person never saw.
        """
        from chess_variant_sandbox.agents.heuristics import load_set

        heuristics = tmp_path / "heur"
        heuristics.mkdir()
        app = App(RULESETS, tmp_path / "runs", heuristics, library.ARCHIVE_DIR)
        app.post("/api/opponent", {
            "hash": chess_game.hash, "name": "regicide", "plan": "",
            "weights": _MATERIAL_ONLY,
            "values": [1, 1, 1, 1, 1, 9],
        })
        compiled = Engine(chess_game.spec).spec
        stored = load_set(chess_game.hash, heuristics / "custom", compiled.piece_ids)
        assert stored is not None
        # The royal multiplier was dropped, which leaves every other slider at one --
        # and that is the player with no opinion.
        assert stored.heuristics[0].values == ()

    def test_a_valuation_of_the_wrong_length_is_refused(
        self, tmp_path, chess_game
    ) -> None:
        heuristics = tmp_path / "heur"
        heuristics.mkdir()
        app = App(RULESETS, tmp_path / "runs", heuristics, library.ARCHIVE_DIR)
        with pytest.raises(ValueError, match="have to line up"):
            app.post("/api/opponent", {
                "hash": chess_game.hash, "name": "x", "plan": "",
                "weights": _MATERIAL_ONLY, "values": [1, 2, 3],
            })

    def test_the_game_page_carries_what_the_pieces_are_worth(self, chess_game) -> None:
        """The panel reads it off the game, beside `liveTerms`, for the same reason."""
        detail = library.detail(chess_game)
        rows = detail["pieceValues"]
        assert [row["id"] for row in rows] == ["P", "N", "B", "R", "Q", "K"]
        assert rows[-1]["royal"]

    def test_the_schema_note_says_how_wide_a_valuation_may_be(self) -> None:
        from chess_variant_sandbox.web import designing

        note = designing.schema_note()
        assert note["maxPieces"] == 24
        assert note["valueRange"] == [0.1, 10.0]


def _in_page(setup: str, expression: str):
    """Run an expression against the page's own functions, in node.

    The sibling of `_render_scoring`, and the same argument: the page is a string that
    nothing type-checks, so the alternative is asserting on substrings of the source --
    which cannot tell a filter that works from one that returns everything. The stub is a
    fake DOM good enough for the page to load; the functions under test here are pure
    over LIBRARY, COVERAGE and BROWSE.
    """
    import json as _json
    import re

    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                                 set: () => true });
      global.document = { getElementById: () => el, querySelector: () => null,
                          querySelectorAll: () => [], addEventListener: () => {},
                          createElement: () => el, body: el };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
      global.location = { pathname: '/game', hash: '' };
      global.fetch = () => new Promise(() => {});
    """
    body = stub + script.group(1) + "\n" + setup + "\n"
    body += "console.log(JSON.stringify(" + expression + "));"
    done = node_session.run(body)
    assert done.returncode == 0, done.stderr
    return _json.loads(done.stdout)


# Two axes is enough to catch the join this is really about: a card carries its bins as a
# *list*, in the grid's order, and the filter has to find which position an axis sits at.
# Off by one and every filter silently answers about the wrong question.
_FIXTURE = """
COVERAGE = { axes: [
  { axis: 'board', asks: 'how big', bins: [
      { bin: 'small', means: '' }, { bin: 'large', means: '' } ] },
  { axis: 'ending', asks: 'how won', bins: [
      { bin: 'mate', means: '' }, { bin: 'ground', means: '' } ] },
] };
METRICS = { families: { depth: { title: 'depth', means: '', weight: 1 } },
            components: { skill_gradient: { title: 'skill gradient', means: '', weight: 1 } } };
LIBRARY = [
  { hash: 'aaa', name: 'alpha', description: '', source: 'committed',
    board: '5x5', pieceTypes: 2, ruleMass: 10, squares: 25, descent: 0,
    cell: '0-0', cellBins: ['small', 'mate'], index: 0.5, viable: true, playable: true,
    staleness: null, families: { depth: 0.5 }, components: { skill_gradient: 0.5 },
    family: { hash: 'aaa', name: 'alpha' }, run: '', theme: '' },
  { hash: 'bbb', name: 'beta', description: '', source: 'evolved',
    board: '9x9', pieceTypes: 3, ruleMass: 20, squares: 81, descent: 1,
    cell: '1-0', cellBins: ['large', 'mate'], index: 0.9, viable: true, playable: true,
    staleness: null, families: { depth: 0.9 }, components: { skill_gradient: 0.9 },
    family: { hash: 'aaa', name: 'alpha' }, run: 'r1', theme: '' },
  { hash: 'ccc', name: 'gamma', description: '', source: 'archived',
    board: '9x9', pieceTypes: 4, ruleMass: 30, squares: 81, descent: 0,
    cell: '1-1', cellBins: ['large', 'ground'], index: null, viable: null, playable: false,
    staleness: null, families: {}, components: {},
    family: { hash: 'ccc', name: 'gamma' }, run: '', theme: '' },
];
"""


class TestTheGamePageOfARetiredGame:
    """Q303, the page's half. A retired game keeps its page and loses its board.

    The user, 2026-09-18, asked for the page to stay and for a delete button on it. So
    the play panel becomes a sentence saying why there is no board, the name carries a
    label, and everything else on the page -- the rules, the pieces, the score, the votes
    and what people played -- is drawn exactly as before.

    `standing` is the site's answer and the local application sends none, so every game
    it draws takes the ordinary path. Nothing here is worked out from an absent spec: the
    server says *retired* and says who may delete.
    """

    HASH = "abc123def456"

    @classmethod
    def _drawn(cls, press: str = "", **standing: object) -> dict:
        payload = _game_payload(cls.HASH, site=True)
        payload["standing"] = {**payload["standing"], **standing}
        return _page_on(
            True,
            f"#/game/{cls.HASH}",
            payload,
            answers={
                f"/api/game/{cls.HASH}/delete": {
                    "hash": cls.HASH, "was": "a test game", "name": "deleted game",
                },
            },
            landed=r'id="(start|deleteGame)"|>Retired<',
            press=press,
        )

    def test_a_retired_game_offers_no_way_to_start_one(self) -> None:
        html = self._drawn(retired=True)["html"]
        assert ">Retired<" in html
        for control in ('id="start"', 'id="swap"', 'id="watchIt"', 'id="mode"'):
            assert control not in html, control
        assert "no new game can be started here" in html

    def test_everything_else_on_the_page_is_still_drawn(self) -> None:
        """The point of keeping the page. The rules and the record are what it was for."""
        html = self._drawn(retired=True)["html"]
        for panel in ("The rules", "The pieces", "How it scored", "Where it came from"):
            assert panel in html, panel

    def test_a_game_that_is_still_in_the_corpus_keeps_its_play_panel(self) -> None:
        html = self._drawn()["html"]
        assert 'id="start"' in html
        assert ">Retired<" not in html

    def test_the_name_carries_the_label(self) -> None:
        assert '<span class="tag">retired</span>' in self._drawn(retired=True)["html"]

    def test_the_delete_button_is_drawn_only_where_the_server_says(self) -> None:
        assert 'id="deleteGame"' not in self._drawn(retired=True)["html"]
        assert 'id="deleteGame"' in self._drawn(retired=True, canDelete=True)["html"]

    def test_pressing_it_confirms_once_and_then_deletes(self) -> None:
        report = self._drawn(
            retired=True,
            canDelete=True,
            press="global.window.confirm = () => true;"
                  " document.getElementById('deleteGame').onclick();",
        )
        assert [path for path, _ in report["sent"]] == [f"/api/game/{self.HASH}/delete"]
        assert report["errors"] == []

    def test_saying_no_deletes_nothing(self) -> None:
        report = self._drawn(
            retired=True,
            canDelete=True,
            press="global.window.confirm = () => false;"
                  " document.getElementById('deleteGame').onclick();",
        )
        assert report["sent"] == []


class TestTheGamePageOnAPhone:
    """`docs/archive/plan-ux.md` C6. At 375px *start a game* was four screens down.

    One column, and the game page put the play panel second: eight kind-cards, six
    pieces, the rules and the provenance came first. The rule that reorders it reads the
    two columns apart from the board page's, which shares `.split` and is already in the
    right order -- so this checks the premise it rests on as well as the rule itself.
    """

    def test_the_play_column_is_first_under_860(self) -> None:
        """V6 (`plan-visual.md` §3.4) puts the play panel first in the markup, so it is
        first in one column with no rule reordering it, and the rest of the game follows
        in its sections."""
        from page_text import DETAIL, PAGE

        assert DETAIL.index('<div class="gamePlay">') < DETAIL.index('<div class="gameSide">')
        assert DETAIL.index('<div class="gameSide">') < DETAIL.index('<div class="gameRest">')
        rule = PAGE[PAGE.index(".gameTop, .gameRest {"):]
        rule = rule[rule.index("@media (max-width: 860px) {"):]
        assert ".gameTop, .gameRest { grid-template-columns: minmax(0, 1fr); }" in rule

    def test_nothing_on_it_scrolls_sideways(self) -> None:
        """Measured at 375px on `hierarchy chess`, which overflowed by 43px.

        A grid item is at least as wide as its own content unless it is told otherwise,
        so one long piece description set the column's width and the page scrolled
        sideways; and an opponent named by its parameters
        (`minimax:h=98f172aeac49,nodes=1600`) is one word to a browser, which set the
        rating table's.
        """
        from page_text import PAGE

        rule = PAGE[PAGE.index("@media (max-width: 860px) {"):]
        rule = rule[:rule.index("svg#board {")]
        assert ".split > * { min-width: 0; }" in rule
        assert "table.ratings td, table.ratings th { overflow-wrap: anywhere; }" in PAGE

    def test_the_name_comes_before_the_play_column(self) -> None:
        """§2.6 asks for the name first. C6 put the play column first and left the name
        inside the other one, so a phone read `all games`, the play panel, and only then
        which game this was. The name is above both columns now, in either layout."""
        from page_text import DETAIL

        assert DETAIL.index("<h2 ") < DETAIL.index('<div class="gameTop"')

    def test_the_two_pages_split_columns_are_still_told_apart(self) -> None:
        """The game page no longer shares `.split` with the board page (V6): its play
        column is first in its own markup. The board page keeps `.split`, its board first
        and its panel second."""
        from page_text import BOARD, DETAIL, PAGE

        game = DETAIL[DETAIL.index('<div class="gameTop">'):]
        # The play column opens with B2's *carry on* block, which draws nothing unless
        # this person left a rated game running here, then the start position (K1), and
        # then the play panel.
        assert ('<div class="gamePlay">\n        ${carryOnHtml(game)}\n        ${openingHtml(game)}'
                '\n        <div class="panel">') in game
        # That panel's heading is *Play* for an ordinary game and *Retired* for one whose
        # rules have left the corpus.
        assert '<h3 class="u-mt-0">Play</h3>' in game
        board = BOARD[BOARD.index('<div class="split'):]
        # The panel opens on the two seats, one above the other since Q288 put how hard
        # each one thinks under its name; stacked, on a phone, the tabs lead it and the
        # seats follow (Q728).
        assert ('<div class="panel boardSide">\n        <div class="seatLines">\n'
                '          <div>${seatHtml(0)}</div>') in board
        assert ".boardSide > .seatLines { order: 1;" in PAGE
        narrow = PAGE.index("@media (max-width: 860px) {\n    .split { grid-template-columns: 1fr; }")
        assert narrow < PAGE.index(".boardSide { display: flex;") < narrow + 1200
        # Q307. The designer plays a draft on this same markup with no second column at
        # all, which is what `.alone` says -- so the selector has no `nth-child(2)` to
        # turn on there and the game page's rule cannot reorder the editor's board. Q993b:
        # nor the proven positions' page, whose list stands beside the board instead.
        assert "${inline || proven ? '' : `<div class=\"panel boardSide\">" in board


class TestAnOpenGameIsFoundAgain:
    """B2 of `docs/archive/plan-ux.md`. A reload of `#/play/<id>` showed the library.

    A game in progress used to live in this tab's memory alone, so the route was only an
    address while the session it named was the one in hand. On the site the moves are a
    row, so the position can be rebuilt -- and the route now fetches it. Closing the tab
    on a rated game and opening the address again draws the board it left.

    The failure it replaces is silent in the worst way: the URL is right, the server has
    the game, and the page answers with the listing.
    """

    HASH = "abc123def456"

    def _session(self, chess_game) -> dict:
        found = _lean(Sessions().start(chess_game, ("human", "random")).snapshot())
        found["mode"] = "leaderboard"
        return found

    def _opened(self, found: dict, chess_game, known: bool = True) -> dict:
        return _page_on(
            True,
            "#/play/" + found["id"],
            _game_payload(chess_game.hash, site=True),
            answers={"/api/session/" + found["id"]: found} if known else {},
            landed=r'id="board"|id="newGame"',
        )

    def test_a_fresh_load_fetches_the_position_and_draws_the_board(
        self, chess_game
    ) -> None:
        found = self._session(chess_game)
        report = self._opened(found, chess_game)
        assert "GET /api/session/" + found["id"] in report["fetched"]
        assert 'id="board"' in report["html"], "the board, not the listing"
        assert report["errors"] == []

    def test_a_rated_game_reloaded_is_still_a_rated_game(self, chess_game) -> None:
        """The snapshot carries the mode, so what comes back is the board B1 draws --
        the mark, no take-back, a resignation -- and not a sandbox board."""
        html = self._opened(self._session(chess_game), chess_game)["html"]
        assert ">rated<" in html
        assert 'id="undo"' not in html
        assert 'id="resign"' in html

    def test_an_id_the_server_does_not_know_lands_in_the_library_with_the_reason(
        self, chess_game
    ) -> None:
        """A local application keeps its sessions in memory, so every reload lands here,
        and so does a game whose rules the store will not open. A blank page with a toast
        is not an answer; the listing with the server's sentence over it is."""
        report = self._opened(self._session(chess_game), chess_game, known=False)
        assert 'id="newGame"' in report["every"], "the library is drawn"
        assert any("no such endpoint" in said for said in report["errors"])

    def test_the_session_in_hand_is_drawn_without_a_request(self) -> None:
        """The tab's own game answers its own address; only another way in costs a
        round trip."""
        from page_text import CORE

        branch = CORE[CORE.index("if (parts[0] === 'play' && parts[1]) {"):]
        branch = branch[:branch.index("\n  }")]
        assert "if (SESSION && SESSION.id === parts[1]) return showBoard(SESSION);" in branch
        assert "api('/api/session/' + parts[1]).then(showBoard)" in branch


class TestTheGamePageOffersAnOpenGameBack:
    """B2's other half. The store held the open match and no page named it.

    `standing.yourOpen` is the server's answer -- this person's unfinished rated games on
    this game -- and the block draws one entry each. It sits above the play panel because
    *start a game* in leaderboard mode resigns whatever is already running on this game.
    """

    HASH = "abc123def456"

    OPEN = {
        "id": "sessionid01", "game": HASH, "opponent": "minimax:nodes=1600", "seat": 0,
        "plies": 12, "started": "2026-09-17T09:00:00+00:00",
        "moved": "2026-09-18T11:30:00+00:00",
    }

    @classmethod
    def _drawn(cls, **standing: object) -> str:
        payload = _game_payload(cls.HASH, site=True)
        payload["standing"] = {**payload["standing"], **standing}
        return _page_on(
            True, f"#/game/{cls.HASH}", payload, landed=r'id="(start|carry)"|>Retired<'
        )["html"]

    def test_an_open_game_is_offered_back_with_what_tells_two_apart(self) -> None:
        html = self._drawn(yourOpen=[self.OPEN])
        assert ">Carry on</h3>" in html
        assert 'data-carry="sessionid01"' in html
        assert "playing first" in html and "12 turns played" in html
        assert "last moved 2026-09-18" in html

    def test_it_is_above_the_play_panel(self) -> None:
        """Because starting another rated game here is what resigns this one."""
        html = self._drawn(yourOpen=[self.OPEN])
        assert html.index(">Carry on</h3>") < html.index('id="start"')

    def test_the_seat_is_read_off_the_entry(self) -> None:
        html = self._drawn(yourOpen=[{**self.OPEN, "seat": 1}])
        assert "playing second" in html and "playing first" not in html

    def test_a_game_nobody_has_moved_in_says_so_rather_than_a_blank(self) -> None:
        html = self._drawn(yourOpen=[{**self.OPEN, "plies": 0, "moved": None}])
        assert "0 turns played, not moved yet" in html

    def test_nothing_is_drawn_when_nothing_is_running(self) -> None:
        """No heading, no empty box. Absent is not a nought."""
        assert ">Carry on</h3>" not in self._drawn(yourOpen=[])
        assert ">Carry on</h3>" not in self._drawn()

    def test_the_local_application_draws_none_of_it(self) -> None:
        """It sends no standing at all, and keeps no rows to have left running."""
        html = _page_on(
            False, f"#/game/{self.HASH}", _game_payload(self.HASH, site=False)
        )["html"]
        assert ">Carry on</h3>" not in html

    def test_a_retired_game_still_offers_the_game_back(self) -> None:
        """The rules have left the corpus and the row still holds them, so the match can
        be finished. No new game can be started on it, which is the panel below."""
        html = self._drawn(retired=True, yourOpen=[self.OPEN])
        assert ">Carry on</h3>" in html and 'data-carry="sessionid01"' in html
        assert ">Retired</h3>" in html and 'id="start"' not in html

    def test_pressing_it_opens_the_board_at_that_address(self) -> None:
        from page_text import PAGE

        assert "go('#/play/' + button.dataset.carry);" in PAGE


def _select(html: str, ident: str) -> str:
    """One `<select>` out of the library's controls, by id."""
    start = html.index(f'<select id="{ident}">')
    return html[start:html.index("</select>", start)]


# What the store adds to the listing's orderings, in the server's own words
# (`site/server.py`). Two keys, and the page names them among its own four.
_SITE_SORTS = """
SITE = { sorts: [{ key: 'stars', label: 'star rating' },
                 { key: 'humanGames', label: 'human games played' }] };
ACCOUNTS = true;
"""


class TestTheOrderByMenuInTwoTiers:
    """`docs/archive/plan-ux.md` C4. Twenty-six entries, and the first six were the file's.

    The menu was the scoring formula's table of contents with *name* on top of it. The
    keys somebody choosing a game to play has a reason to use come first and ungrouped
    (four, and five since Q826 put *date added* among them); the eighteen families and components sit in one group under a heading
    that says what they are.
    """

    def test_the_first_five_entries_are_the_five(self) -> None:
        """*Date added* is a player's key too: the user asked on 2026-10-10 for games to be
        sortable by the day they were made, and the *new in the library* shelf opens the
        list ordered by it (Q826), so it sits in the first tier, after the score."""
        import re

        menu = _select(_in_page(_FIXTURE + _SITE_SORTS, "controlsHtml()"), "sort")
        top = menu[:menu.index("<optgroup")]
        assert re.findall(r'<option value="([^"]+)"', top) == [
            "name", "index", "added", "stars", "humanGames"
        ]
        assert "date added" in top
        assert "star rating" in top and "human games played" in top

    def test_a_server_with_no_store_offers_the_three_it_has(self) -> None:
        """Every library file carries the day it joined (`provenance.added`), so a checkout
        with no store still orders by it; the store's two columns are what it lacks."""
        import re

        menu = _select(_in_page(_FIXTURE, "controlsHtml()"), "sort")
        top = menu[:menu.index("<optgroup")]
        assert re.findall(r'<option value="([^"]+)"', top) == ["name", "index", "added"]

    def test_the_score_is_one_group_and_the_rest_another(self) -> None:
        import re

        menu = _select(_in_page(_FIXTURE + _SITE_SORTS, "controlsHtml()"), "sort")
        assert re.findall(r'<optgroup label="([^"]+)"', menu) == [
            "more about the game itself", "by one part of the score"
        ]
        parts = menu[menu.index('<optgroup label="by one part of the score"'):]
        assert 'value="family:depth"' in parts
        assert 'value="component:skill_gradient"' in parts
        # And nothing about a measurement is left in the tier above it.
        rest = menu[menu.index('<optgroup label="more about'):menu.index("by one part")]
        assert "family:" not in rest and "component:" not in rest
        assert 'value="squares"' in rest and 'value="ruleMass"' in rest

    def test_the_site_does_not_offer_the_maintainers_two_groupings(self) -> None:
        """Where a file is kept and which design run wrote it are questions about this
        repository, and the site is not a view of a repository."""
        import re

        locally = _select(_in_page(_FIXTURE, "controlsHtml()"), "group")
        on_site = _select(_in_page(_FIXTURE + _SITE_SORTS, "controlsHtml()"), "group")
        assert re.findall(r'<option value="([^"]+)"', locally) == [
            "none", "family", "cell", "source", "run"
        ]
        assert re.findall(r'<option value="([^"]+)"', on_site) == [
            "none", "family", "cell"
        ]


class TestATagIsReadByTappingIt:
    """First-visit study F3: the line above the cards says *tap or point at one*, and on a
    phone a tap on a tag was a tap on its card, which opened the game. Where nothing can
    hover, a tap on a tag opens its sentence in the page's one bubble, ahead of the card;
    where a pointer can hover, the tag is left to its `title` and the card."""

    SETUP = """
      let HOVERLESS = true;
      const LISTEN = [];
      document.addEventListener = (kind, fn, capture) =>
        LISTEN.push({ kind, fn, capture: !!capture });
      const BUBBLE = { hidden: true, dataset: {}, innerHTML: '', style: {},
                       contains: () => false };
      document.createElement = () => BUBBLE;
      document.body = { appendChild() {} };
      window.matchMedia = query => ({ matches: query === '(hover: none)' && HOVERLESS,
                                      addEventListener() {} });
      wireInfo.done = false;
      wireInfo();
      const TAG = { title: 'Practice only: <b>', getBoundingClientRect: () => ({ left: 10, bottom: 20 }) };
      const tap = () => {
        const seen = { stopped: false };
        LISTEN.find(one => one.capture).fn({
          target: { closest: sel => (sel === '.tag[title]' ? TAG : null) },
          preventDefault() {}, stopPropagation() { seen.stopped = true; } });
        return seen.stopped;
      };
    """

    def test_a_tap_on_a_phone_reads_the_tag(self) -> None:
        assert _in_page(self.SETUP, "[tap(), BUBBLE.hidden, BUBBLE.innerHTML]") == [
            True, False, "<p>Practice only: &lt;b&gt;</p>"]
        # A second tap on the same tag puts it away.
        assert _in_page(self.SETUP + "tap();", "[tap(), BUBBLE.hidden]") == [True, True]

    def test_a_pointer_that_hovers_is_left_to_the_card(self) -> None:
        assert _in_page(self.SETUP + "HOVERLESS = false;",
                        "[tap(), BUBBLE.hidden]") == [False, True]

    def test_the_line_says_tap(self) -> None:
        from page_text import PAGE

        assert "Tap or point at one for what it means." in PAGE


class TestTheSiteOpensOnItsOwnJudgement:
    """`docs/archive/plan-ux.md` C5, and D1 as the user decided it on 2026-09-17: by score, best
    first. Alphabetical opened on *almost chess, antichess, atomic, bastion* and said
    nothing about which of seventy games is worth an evening.

    A game whose score is out of date has no number to sort on, because the corpus rule
    withholds a stale number rather than showing it, so it sorts last beside the games
    nobody has measured. The dull anchors sort last on their own low numbers.
    """

    HASH = "abc123def456"

    def test_the_site_opens_by_score_and_here_by_name(self) -> None:
        assert _in_page(_FIXTURE + _SITE_SORTS + "settleOrder();",
                        "[BROWSE.sort, BROWSE.descending]") == ["index", True]
        assert _in_page(_FIXTURE + "settleOrder();",
                        "[BROWSE.sort, BROWSE.descending]") == ["name", False]

    def test_a_list_with_no_score_opens_by_name(self) -> None:
        """First-visit study N5: *order by score* over a list in which no game has a
        score orders by nothing, so such a list opens by name. One scored game is enough
        for the score to be the order again."""
        unscored = "LIBRARY.forEach(game => { game.index = null; });"
        assert _in_page(_FIXTURE + _SITE_SORTS + unscored + "settleOrder();",
                        "[BROWSE.sort, BROWSE.descending]") == ["name", False]
        one = unscored + "LIBRARY[1].index = 0.4;"
        assert _in_page(_FIXTURE + _SITE_SORTS + one + "settleOrder();",
                        "[BROWSE.sort, BROWSE.descending]") == ["index", True]
        # A tab is the same: the one it opens in is its own, or name where nothing in it
        # has a value for it.
        assert _in_page(_FIXTURE + unscored,
                        "[openingSort('index', LIBRARY), openingSort('squares', LIBRARY)]"
                        ) == ["name", "squares"]

    def test_the_order_is_the_readers_after_that(self) -> None:
        """The listing is re-read whenever somebody comes back to it, and a default that
        reapplied itself would undo the menu every time."""
        kept = _in_page(
            _FIXTURE + _SITE_SORTS + "settleOrder(); BROWSE.sort = 'name';"
            " BROWSE.descending = false; settleOrder();",
            "[BROWSE.sort, BROWSE.descending]",
        )
        assert kept == ["name", False]

    def test_best_first_and_the_unmeasured_last(self) -> None:
        order = _in_page(_FIXTURE + _SITE_SORTS + "settleOrder();",
                         "ordered(narrowed()).map(g => g.name)")
        assert order == ["beta", "alpha", "gamma"]

    def test_the_listing_opens_that_way_on_the_site(self) -> None:
        """End to end, through `showLibrary`, so the decision is wired and not only
        written."""
        scored = {
            "hash": "aaa", "name": "alpha", "description": "", "source": "committed",
            "board": "5x5", "pieceTypes": 2, "ruleMass": 10, "squares": 25, "descent": 0,
            "cell": "0-0", "cellBins": [], "index": 0.5, "viable": True, "playable": True,
            "staleness": None, "families": {}, "components": {},
            "family": {"hash": "aaa", "name": "alpha"}, "run": "", "theme": "",
        }
        for games, sort, way in (([scored], "index", "high to low"),
                                 ([], "name", "low to high")):
            listing = {
                "games": games, "opponents": [], "coverage": {"axes": []}, "metrics": None,
                "site": {"signedIn": False, "admin": False, "filters": [], "sorts": [
                    {"sort": "stars", "key": "stars", "label": "star rating"},
                    {"sort": "human_games", "key": "humanGames",
                     "label": "human games played"},
                ]},
            }
            seen = _page_on(True, "", _game_payload(self.HASH, site=True),
                            {"/api/library": listing})
            assert seen["errors"] == []
            menu = _select(seen["html"], "sort")
            # N5: a listing with no score to sort on opens by name.
            assert f'<option value="{sort}" selected>' in menu, sort
            assert way in seen["html"], sort


# Every pill a card can carry, on one game, so the audit of what they explain has all
# six to read. Six is the whole set: the tag, this reader's record, whether anybody has
# played it, the outside engine, the cheap gate and the score.
_EVERY_PILL = """
Object.assign(LIBRARY[0], {
  rated: true, unbeaten: true, top: ['minimax:nodes=6400'], untried: true, humanGames: 0,
  fairyTranslates: true, playable: false, playableReasons: ['no side can win'],
});
"""


class TestFindingOneGameAmongFifty:
    """Filtering, ordering and grouping, run as the page runs them.

    The filters are the behaviour grid's own axes, which means a card carries its answers
    as a list in the grid's order and the filter has to look up which position an axis is
    at. That join is the one thing here that can be wrong without looking wrong.
    """

    def test_a_bin_selects_exactly_the_games_that_answer_that_way(self) -> None:
        chosen = _in_page(
            _FIXTURE + "BROWSE.bins = { ending: ['ground'] };",
            "narrowed().map(g => g.name)",
        )
        assert chosen == ["gamma"]

        both = _in_page(
            _FIXTURE + "BROWSE.bins = { board: ['large'], ending: ['mate'] };",
            "narrowed().map(g => g.name)",
        )
        assert both == ["beta"], "two axes narrow together rather than widening"

        either = _in_page(
            _FIXTURE + "BROWSE.bins = { board: ['small', 'large'] };",
            "narrowed().map(g => g.name)",
        )
        assert either == ["alpha", "beta", "gamma"], "bins within one axis are alternatives"

    def test_an_untouched_filter_costs_nothing(self) -> None:
        assert _in_page(_FIXTURE, "narrowed().map(g => g.name)") == [
            "alpha", "beta", "gamma"
        ]
        assert _in_page(_FIXTURE + "BROWSE.bins = { board: [] };",
                        "narrowed().map(g => g.name)") == ["alpha", "beta", "gamma"]

    def test_the_other_filters(self) -> None:
        assert _in_page(_FIXTURE + "BROWSE.stage = 'archived';",
                        "narrowed().map(g => g.name)") == ["gamma"]
        assert _in_page(_FIXTURE + "BROWSE.playable = 'no';",
                        "narrowed().map(g => g.name)") == ["gamma"]
        assert _in_page(_FIXTURE + "BROWSE.scored = 'yes';",
                        "narrowed().map(g => g.name)") == ["alpha", "beta"]
        assert _in_page(_FIXTURE + "BROWSE.scored = 'no';",
                        "narrowed().map(g => g.name)") == ["gamma"]
        assert _in_page(_FIXTURE + "BROWSE.text = 'GAMM';",
                        "narrowed().map(g => g.name)") == ["gamma"], "search is case-blind"

    def test_unmeasured_sorts_last_in_both_directions(self) -> None:
        """"Unscored" is not "scored zero".

        A descending sort that put every unmeasured game at the top would be answering a
        different question from the one the menu asks, and it is the direction somebody
        looking for the best game would press first.
        """
        down = _in_page(
            _FIXTURE + "BROWSE.sort = 'component:skill_gradient'; BROWSE.descending = true;",
            "ordered(narrowed()).map(g => g.name)",
        )
        up = _in_page(
            _FIXTURE + "BROWSE.sort = 'component:skill_gradient'; BROWSE.descending = false;",
            "ordered(narrowed()).map(g => g.name)",
        )
        assert down == ["beta", "alpha", "gamma"]
        assert up == ["alpha", "beta", "gamma"]

    def test_a_family_gathers_a_parent_with_its_revisions(self) -> None:
        """The grouping this was worth doing for: forty cards in alphabetical order is
        exactly the arrangement that hides which of them are revisions of one idea."""
        groups = _in_page(
            _FIXTURE + "BROWSE.group = 'family';",
            "grouped(ordered(narrowed())).map(g => [g.title, g.games.map(x => x.name)])",
        )
        assert groups[0] == ["alpha", ["alpha", "beta"]]
        # A family of one is not a family: they are gathered at the end under what they
        # are, rather than each taking a heading of its own.
        assert groups[1][0] == "on their own"
        assert groups[1][1] == ["gamma"]

    def test_a_card_says_every_word_it_uses(self) -> None:
        """`docs/archive/plan-ux.md` C3. A pill is a word from this project's own vocabulary,
        and a word with nothing behind it is the maintainer talking to themselves."""
        import re

        drawn = _in_page(_FIXTURE + _EVERY_PILL, "cardHtml(LIBRARY[0])")
        pills = re.findall(r"<span class=\"tag[^\"]*\"[^>]*>", drawn)
        assert len(pills) == 6, pills
        for pill in pills:
            assert "title=" in pill, pill

    def test_the_pills_are_always_a_row_of_their_own(self) -> None:
        """Beside the name they fitted next to a short one and dropped under a long one,
        so the same card was two layouts depending on what the game was called."""
        from page_text import PAGE

        long_name = _in_page(
            _FIXTURE + _EVERY_PILL + "LIBRARY[0].name = 'a rather long name indeed';",
            "cardHtml(LIBRARY[0])",
        )
        short = _in_page(_FIXTURE + _EVERY_PILL, "cardHtml(LIBRARY[0])")
        for drawn in (short, long_name):
            assert '<div class="row pills">' in drawn
            assert drawn.index("</h2>") < drawn.index('class="row pills"')
        assert ".card .pills {" in PAGE

    def test_a_visitor_is_not_told_where_the_file_is_kept(self) -> None:
        """On the site every game is simply here. *Yardstick* is a word about the
        calibration corpus and there is nothing a visitor can do with it."""
        locally = _in_page(_FIXTURE, "[cardHtml(LIBRARY[0]), controlsHtml()]")
        on_site = _in_page(_FIXTURE + "ACCOUNTS = true;",
                           "[cardHtml(LIBRARY[0]), controlsHtml()]")
        assert "yardstick" in locally[0] and 'id="stage"' in locally[1]
        assert "yardstick" not in on_site[0] and "archived" not in on_site[0]
        assert 'id="stage"' not in on_site[1]
        # The rest of the line is what the game is, and it stays.
        assert "5x5" in on_site[0] and "2 piece types" in on_site[0]

    def test_the_legend_names_what_the_pills_are(self) -> None:
        legend = _in_page(_FIXTURE + "ACCOUNTS = true;", "pillLegendHtml()")
        said = " ".join(legend.split())
        assert "The words under a game's name say" in said
        assert "beaten its strongest computer player" in said
        assert 'data-gloss="index"' in legend and 'data-gloss="unbeaten"' in legend
        # Locally there is no store, so no tag to explain and no claim that there is.
        here = _in_page(_FIXTURE, "pillLegendHtml()")
        assert "beaten its strongest computer player" not in " ".join(here.split())
        assert 'data-gloss="committed"' in here

    def test_the_criterion_is_drawn_on_the_card_when_it_is_the_order(self) -> None:
        """A ranking whose criterion is invisible is one you have to take on trust."""
        shown = _in_page(
            _FIXTURE + "BROWSE.sort = 'component:skill_gradient';",
            "[sortedByHtml(LIBRARY[0]), sortedByHtml(LIBRARY[2]), "
            "(BROWSE.sort = 'name', sortedByHtml(LIBRARY[0]))]",
        )
        assert "skill gradient" in shown[0] and "0.50" in shown[0]
        assert "not measured" in shown[1], "a game with no number says so"
        assert shown[2] == "", "nothing is drawn when the order is not a measurement"


# One draft, one board, one move map. Small enough to reason about by hand: a 3x3 board
# with two pieces on it, and a piece standing alone in the middle that may move to a1 and
# take on a3.
_DRAFT_FIXTURE = """
DRAFT = { name: 'a draft', pieces: [{ id: 'A', name: 'ay' }, { id: 'B', name: 'bee' }] };
DRAFT_PIECE = 0;
const PREVIEW = { width: 3, height: 3, board: [0, 0, 0, 0, 1, 0, 2, 0, 0],
  obstacles: [], pieceIds: ['A', 'B'], pieceNames: ['ay', 'bee'], nTypes: 2,
  sideStride: 2, facingAngles: [[0], [180]], icons: [], sides: ['first', 'second'] };
const MAP = { from: 4, value: 1, width: 3, height: 3, moves: [0], takes: [6],
  moveOnly: [0], captureOnly: [6], obstacles: [] };
"""


def _said(html: str) -> str:
    """A rendered panel as a reader sees it: one line of words. The sentences live in
    template literals that wrap where the source wraps, so a substring assertion against
    the raw string is really an assertion about where a line break fell."""
    return " ".join(html.split())


class TestWhatTheDesignerDrawsAndSays:
    """The two claims the editor makes about a draft, checked by running the page's own
    functions rather than by reading the source for substrings.

    Both used to be wrong in the same way: they said something that was not true of the
    thing on screen. The map was drawn on top of the opening position, so its dots and
    rings sat among pieces the map is not about; and nothing at all said which game the
    draft *was*, so a fork that changed no rule looked like a new one.
    """

    def test_each_mode_draws_the_board_its_question_needs(self) -> None:
        """One board in the middle, and three questions asked of it. *Set up* and *where
        it goes* are the opening position, because both are about the game as it stands;
        *teach a move* empties the board under the piece, because what a descriptor means
        is a fact about an empty board and pieces standing around it cannot show it."""
        drawn = _in_page(
            _DRAFT_FIXTURE,
            "(() => { DRAFT_MAP = MAP;"
            " const setup = benchBoardHtml(PREVIEW, 'setup');"
            " const teach = benchBoardHtml(PREVIEW, 'teach');"
            " DRAFT_MAP = Object.assign({ live: true }, MAP);"
            " const goes = benchBoardHtml(PREVIEW, 'goes');"
            " const count = (html, what) => html.split(what).length - 1;"
            " return { setupPieces: count(setup, '<text'),"
            "   setupMarks: count(setup, 'var(--hl-sel)') + count(setup, 'var(--dot)'),"
            "   teachPieces: count(teach, '<text'),"
            "   teachDot: count(teach, 'var(--dot)'),"
            "   teachRing: count(teach, 'var(--bad)'),"
            "   goesPieces: count(goes, '<text'),"
            "   goesDot: count(goes, 'var(--dot)') }; })()",
        )
        # Set up is the plain opening position: two pieces where the setup put them, and
        # not one mark among them.
        assert drawn["setupPieces"] == 2 and drawn["setupMarks"] == 0
        # Teaching empties it under the one piece the engine stood up.
        assert drawn["teachPieces"] == 1
        assert drawn["teachDot"] >= 1 and drawn["teachRing"] >= 1
        # And *where it goes* keeps the army and marks the answer among it.
        assert drawn["goesPieces"] == 2 and drawn["goesDot"] >= 1

    def test_the_lone_board_draws_the_piece_the_engine_stood_up(self) -> None:
        """`move_map` sends the board byte back rather than leaving the page to work one
        out: the encoding is player-major and carries the piece's initial facing, and a
        second opinion about it is how two drawings of one piece come to disagree."""
        drawn = _in_page(
            _DRAFT_FIXTURE + "MAP.value = 2; DRAFT_MAP = MAP;",
            "benchBoardHtml(PREVIEW, 'teach')",
        )
        assert ">B</text>" in drawn, "the lone board drew a piece the map is not about"
        assert ">A</text>" not in drawn, "the lone board kept a piece the map is not about"

    def test_an_empty_board_answer_is_never_drawn_on_the_real_position(self) -> None:
        """A mode switch redraws before the fetch it started has come back. Drawing the
        answer anyway would put a bishop's whole star straight through its own pawns."""
        drawn = _in_page(
            _DRAFT_FIXTURE + "DRAFT_MAP = MAP;", "benchBoardHtml(PREVIEW, 'goes')"
        )
        assert "var(--dot)" not in drawn and "var(--bad)" not in drawn

    def test_a_fork_is_told_it_is_still_its_parent(self) -> None:
        said = _said(_in_page(
            _DRAFT_FIXTURE
            + "DRAFT_PARENT = { hash: 'abc123abc123', name: 'chess', index: 0.84,"
            "  seeds: [0, 1000, 2000, 3000], roster: ['random', 'greedy'],"
            "  staleness: { current: true } };",
            "identityHtml({ hash: 'abc123abc123', schemaVersion: 1 })",
        ))
        assert "abc123abc123" in said
        assert "the same game" in said
        assert "Nothing you have changed is a rule" in said and "chess" in said
        assert "0.84" in said, "the parent's number, so the cost of changing is on screen"
        # And what it rests on, which `/api/design/<hash>` did not send before Q339.
        assert "mean of four runs" in said

    def test_a_real_fork_is_told_the_measurement_does_not_travel(self) -> None:
        said = _said(_in_page(
            _DRAFT_FIXTURE
            + "DRAFT_PARENT = { hash: 'abc123abc123', name: 'chess', index: 0.84,"
            "  roster: ['random'], staleness: null };",
            "identityHtml({ hash: 'ffffffffffff', schemaVersion: 1 })",
        ))
        # Both hashes, and what the difference between them costs.
        assert "abc123abc123" in said and "ffffffffffff" in said
        assert "keeps its own measurement" in said
        assert "no measurement of its own" in said

    def test_a_stale_parent_is_not_given_a_number(self) -> None:
        """The one policy about a stale score holds here too: it is withheld, and the
        repair is named. A number fitted under an engine that no longer exists is not a
        smaller number, it is a number about a different program."""
        said = _said(_in_page(
            _DRAFT_FIXTURE
            + "DRAFT_PARENT = { hash: 'abc123abc123', name: 'chess', index: null,"
            "  roster: [], staleness: { current: false, needs: 'replay' } };",
            "identityHtml({ hash: 'ffffffffffff', schemaVersion: 1 })",
        ))
        assert "out of date" in said and "replay" in said
        assert "0." not in said.split("A game is identified")[0], "a stale number shown"

def _kit() -> str:
    """The rungs the designer already holds, as the page holds them.

    The names the save's sentence uses are the ladder's own, read from the same answer
    the rung list below the button is drawn from, so the test is against those names
    rather than against a second copy written here.
    """
    from chess_variant_sandbox.web import candidacy

    return "DESIGN_KIT = " + json.dumps({"rungs": candidacy.rungs()}) + ";"


def _save_answer(**fields) -> str:
    kept = {"created": True, "name": "a draft somebody kept", "hash": "abc123abc123"}
    return json.dumps({**kept, **fields})


def _saved_box(answer: dict, ok: bool = True) -> dict:
    """Press the save button in node, and read the box beside it.

    `_in_page`'s stub swallows every assignment, and what is under test is exactly an
    assignment -- the sentence the save puts in `dSaved`. So `dSaved` is a real object
    here and everything else is that stub. The fetch that follows a save never lands,
    which is the point of the second test below: the sentence is on screen before the
    lineage and the rungs are asked for, and stays there whatever they answer.
    """
    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                                 set: () => true });
      const BOX = { innerHTML: '', scrollIntoView: () => {} };
      global.document = { getElementById: id => (id === 'dSaved' ? BOX : el),
                          querySelector: () => null, querySelectorAll: () => [],
                          addEventListener: () => {}, createElement: () => el, body: el };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {},
                        // Two columns: the verdict and its buttons are in the panel.
                        getComputedStyle: () => ({ position: 'static' }) };
      global.location = { pathname: '/game', hash: '' };
      global.fetch = path => path.indexOf('/api/design/save') === 0
        ? Promise.resolve({ ok: __OK__, statusText: 'Bad Request',
                            json: () => Promise.resolve(__ANSWER__) })
        : new Promise(() => {});
    """.replace("__OK__", "true" if ok else "false").replace(
        "__ANSWER__", json.dumps(answer)
    )
    runner = """
      DRAFT = { name: 'a draft somebody kept' };
      saveDraft();
      setTimeout(() => console.log(JSON.stringify(
        { box: BOX.innerHTML, kept: DRAFT_SAVED })), 0);
    """
    done = node_session.run(stub + script.group(1) + "\n" + _kit() + runner)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


class TestWhatASaveSaysWhereTheClickWas:
    """Q275. After a save the designer says what the save did, beside the button.

    Three things, each of them the save's own answer rather than the page's opinion:
    whether the draft was kept, how far the playability check got (`rung`, `rungPassed`),
    and whether it was listed and why (`listed`, `said`). The site answers all three. The
    local application has no store and runs no check on a save, so its answer carries
    only the first and the page draws only that sentence.

    Before this the page said "Saved as X" and nothing else, so a draft that was kept but
    left off the listing read exactly like one anybody could open.
    """

    def test_a_listed_save_names_the_rungs_and_says_it_is_listed(self) -> None:
        from chess_variant_sandbox.site import drafts

        row = drafts.Kept("abc123abc123", drafts.PUBLIC, created=True, revoked=False)
        listed = drafts.why(row, {"passed": True, "reasons": []})
        said = _said(_in_page(_kit(), "savedHtml(" + _save_answer(
            listed=True, visibility="public", said=listed,
            rung="screen", rungPassed=True) + ")"))

        assert "Saved as <b>a draft somebody kept</b>" in said
        # How many steps the check ran (D26); the ladder below names each one.
        assert "ran all three of its steps" in said and "passed every one" in said
        assert "rung" not in said
        assert listed in said, "the server's own sentence about the listing"

    def test_an_unlisted_save_says_which_rung_stopped_it_and_why_it_is_not_listed(
        self,
    ) -> None:
        from chess_variant_sandbox.site import drafts

        reason = "the first player has no legal move from the starting position"
        row = drafts.Kept("abc123abc123", drafts.UNLISTED, created=True, revoked=False)
        unlisted = drafts.why(row, {"passed": False, "reasons": [reason]})
        said = _said(_in_page(_kit(), "savedHtml(" + _save_answer(
            listed=False, visibility="unlisted", said=unlisted,
            rung="smoke", rungPassed=False) + ")"))

        # Kept, so it still says so -- and then what it did not pass, by name.
        assert "Saved as <b>a draft somebody kept</b>" in said
        assert "got as far as <b>it plays and ends</b>" in said
        assert "did not pass there" in said
        assert "a searching player can win it" not in said, "a rung that never ran"
        # Q503a: every save keeps a draft, and this one cannot be published yet.
        assert "only people with its link" in said and reason in said
        assert "cannot be published" in said

    def test_a_resave_with_new_words_says_it_saved_them(self) -> None:
        """Q595: not *Already saved*, which read as though the description were dropped."""
        said = _said(_in_page(_kit(), "savedHtml(" + _save_answer(
            created=False, restated=["description"], unchanged=[]) + ")"))
        assert "Saved the new description of <b>a draft somebody kept</b>" in said
        assert "same game" in said and "Already saved" not in said
        both = _said(_in_page(_kit(), "savedHtml(" + _save_answer(
            created=False, restated=["name", "description"], unchanged=[]) + ")"))
        assert "Saved the new name and description of" in both

    def test_a_resave_whose_words_stay_says_why(self) -> None:
        from chess_variant_sandbox.site import drafts

        said = _said(_in_page(_kit(), "savedHtml(" + _save_answer(
            created=False, restated=[], unchanged=["description"],
            keptAsIs=drafts.AS_PUBLISHED) + ")"))
        assert "Already saved as <b>a draft somebody kept</b>" in said
        assert drafts.AS_PUBLISHED in said
        # The same words with nothing asked of them say nothing about why.
        plain = _said(_in_page(_kit(), "savedHtml(" + _save_answer(
            created=False, restated=[], unchanged=[], keptAsIs=drafts.AS_PUBLISHED) + ")"))
        assert drafts.AS_PUBLISHED not in plain

    def test_the_publish_panel_is_off_and_says_publishing_opens_soon(self) -> None:
        """Q503a. The designer draws `drafts.standing`'s panel as it comes: the check
        ticked, the reading not, and the button disabled with the reason beside it."""
        from chess_variant_sandbox.site import drafts

        panel = {
            "level": "Draft", "mine": True, "open": False, "publishable": True,
            "says": "Saved as a draft <b>", "waiting": drafts.NEEDS_A_READING,
            "ticks": [{"met": True, "says": "it passes the playability check"},
                      {"met": False, "says": drafts.READING_TICK}],
        }
        drawn = _said(_in_page(_kit(), "publishHtml(" + json.dumps({"publish": panel}) + ")"))

        assert "Where this game stands: Draft" in drawn
        assert 'id="dPublishGo" class="primary" disabled' in drawn
        # `esc` writes an apostrophe as `&#39;`, and the sentence has one ("can't").
        assert drafts.NEEDS_A_READING.replace("'", "&#39;") in drawn
        assert "&#10003;</span>it passes" in drawn and "&#10007;</span>a score measured on" in drawn
        assert "Saved as a draft &lt;b&gt;" in drawn, "the server's sentence is escaped"

    def test_the_publish_panel_shows_the_terms_beside_an_unticked_box(self) -> None:
        """Q722. The terms' one sentence beside a box, unticked, linking the whole terms;
        the button stays off until the box is ticked even when both ticks are met, and
        says so. The version a tick stands for rides on the box, for the press to send."""
        from chess_variant_sandbox.site import terms

        panel = {
            "level": "Draft", "mine": True, "open": True, "publishable": True,
            "says": "", "waiting": "", "ticks": [],
            "terms": {"version": terms.VERSION, "says": terms.DEDICATION,
                      "href": terms.PATH},
        }
        drawn = _said(_in_page(_kit(), "publishHtml(" + json.dumps({"publish": panel}) + ")"))

        assert 'id="dTermsAgree"' in drawn and f'data-version="{terms.VERSION}"' in drawn
        assert "checked" not in drawn
        assert terms.DEDICATION.replace("'", "&#39;") in drawn and 'href="/terms"' in drawn
        assert 'id="dPublishGo" class="primary" disabled' in drawn
        assert 'data-open="yes"' in drawn and "tick the box to agree to the terms first" in drawn
        # No box without the button.
        panel["publishable"] = False
        drawn = _said(_in_page(_kit(), "publishHtml(" + json.dumps({"publish": panel}) + ")"))
        assert "dTermsAgree" not in drawn

    def test_a_press_that_waits_shows_waiting_for_verification_and_no_button(self) -> None:
        """Q742. A press the site took while it checks the score: the button and the box
        give way to *Waiting for verification* and the server's sentence, which says the
        publish completes by itself once the score has been checked."""
        from chess_variant_sandbox.site import drafts

        panel = {
            "level": "Draft", "mine": True, "open": False, "publishable": True,
            "says": "", "waiting": drafts.WAITING, "waitsForCheck": True, "stopped": "",
            "ticks": [{"met": True, "says": "it passes the playability check"},
                      {"met": True, "says": drafts.READING_CHECKING}],
            "terms": {"version": 1, "says": "x", "href": "/terms"},
        }
        drawn = _said(_in_page(_kit(), "publishHtml(" + json.dumps({"publish": panel}) + ")"))

        assert 'id="dPublishWaiting"' in drawn and "Waiting for verification" in drawn
        assert drafts.WAITING in drawn
        assert "dPublishGo" not in drawn and "dTermsAgree" not in drawn

    def test_a_publish_called_off_says_why_above_the_button(self) -> None:
        from chess_variant_sandbox.site import drafts

        why = drafts.CALLED_OFF + drafts.OUT_OF_TIME
        panel = {
            "level": "Draft", "mine": True, "open": False, "publishable": True,
            "says": "", "waiting": drafts.NEEDS_A_READING, "waitsForCheck": False,
            "stopped": why, "ticks": [],
        }
        drawn = _said(_in_page(_kit(), "publishHtml(" + json.dumps({"publish": panel}) + ")"))

        assert 'id="dPublishStopped"' in drawn and why.replace("'", "&#39;") in drawn
        assert drawn.index("dPublishStopped") < drawn.index('id="dPublishGo"')

    def test_the_measuring_control_sits_under_the_line_that_asks_for_a_score(self) -> None:
        """Q742, the user: *make sure the measurement button is next to where the
        measurement is demanded*. On the site, wherever the designer's *Publish* button is
        drawn, the measuring control is inside that panel: the line asking for a score,
        then *measure it here*, then the box and the button. It is not drawn a second time
        under the rungs, and for anybody else it keeps its old place."""
        from chess_variant_sandbox.site import drafts

        panel = {
            "level": "Draft", "mine": True, "open": False, "publishable": True,
            "says": "", "waiting": drafts.NEEDS_A_READING, "waitsForCheck": False,
            "stopped": "",
            "ticks": [{"met": True, "says": "it passes the playability check"},
                      {"met": False, "says": drafts.READING_TICK}],
            "terms": {"version": 1, "says": "x", "href": "/terms"},
        }
        setup = _kit() + (
            " ACCOUNTS = true; VIEWER = 7;"
            " HERE_MANIFEST = { schema: 1, runtime: 'pyodide-0', wheel: 'wheel-0/x.whl' };"
            " DRAFT_CHECK = { hash: 'abc123abc123', valid: true, playable: true };"
            " DRAFT_STANDING = { hash: 'abc123abc123', publish: " + json.dumps(panel) + " };"
        )
        drawn = _said(_in_page(setup, "publishHtml(DRAFT_STANDING, DRAFT_CHECK)"))
        asked = drawn.index(drafts.NEEDS_A_READING)
        assert asked < drawn.index('id="dHere" data-in="panel"') < drawn.index('id="dHereGo"')
        assert drawn.index('id="dHereGo"') < drawn.index('id="dTermsAgree"')
        assert drawn.index('id="dHereGo"') < drawn.index('id="dPublishGo"')
        assert _in_page(setup, "hereAwayHtml(DRAFT_CHECK)") == ""

        # Somebody else's draft, or the local application: no button, and the control
        # stays where it was, under the rungs.
        theirs = setup + " DRAFT_STANDING.publish.publishable = false;"
        assert 'id="dHere"' not in _in_page(theirs, "publishHtml(DRAFT_STANDING, DRAFT_CHECK)")
        assert 'id="dHereGo"' in _in_page(theirs, "hereAwayHtml(DRAFT_CHECK)")
        local = setup + " ACCOUNTS = false;"
        assert 'id="dHereGo"' in _in_page(local, "hereAwayHtml(DRAFT_CHECK)")

    def test_the_publish_panel_offers_no_button_to_anybody_else(self) -> None:
        panel = {"level": "Draft", "mine": False, "open": False, "ticks": [],
                 "says": "only its designer can publish it", "waiting": ""}
        drawn = _said(_in_page(_kit(), "publishHtml(" + json.dumps({"publish": panel}) + ")"))

        assert "only its designer can publish it" in drawn
        assert "dPublishGo" not in drawn
        assert _in_page(_kit(), "publishHtml({})") == ""

    def test_a_local_save_has_no_listing_and_no_rung_to_report(self) -> None:

        """The local application's answer is `designing.save`'s and carries neither, so
        the page says what it knows and invents nothing about a store it has not got."""
        said = _said(_in_page(_kit(), "savedHtml(" + _save_answer() + ")"))

        assert "Saved as <b>a draft somebody kept</b>" in said
        assert "playability check" not in said and "listed" not in said

    def test_the_sentence_lands_in_the_box_beside_the_button(self) -> None:
        box = _saved_box({"created": True, "name": "a draft somebody kept",
                          "hash": "abc123abc123", "listed": True, "said": "Listed.",
                          "rung": "screen", "rungPassed": True})

        assert "Saved as <b>a draft somebody kept</b>" in _said(box["box"])
        assert "Listed." in _said(box["box"])
        # Kept as the last thing the save said, so a redraw of the panel shows it again.
        assert box["kept"] == box["box"]

    def test_a_refused_save_puts_its_sentence_in_the_same_place(self) -> None:
        """A 400 from the check's wall clock (`site/screening.refusal`) or from a draft
        that is not a ruleset yet. It is a plain sentence and it goes where the save's
        own sentence would have gone, not into the page-wide error box."""
        from chess_variant_sandbox.site import screening

        refusal = screening.refusal(screening.CHECK_SECONDS)
        box = _saved_box({"error": refusal}, ok=False)

        assert 'class="error"' in box["box"]
        assert refusal in _said(box["box"])
        assert "Saved" not in box["box"]
        assert box["kept"] == box["box"]


def _page_agent(name: str) -> str:
    """One of the designer's seat constants, read off the page it is declared in.

    The page is a string, so the alternative is writing the agent out here as well --
    and then a test that passes says only that two copies of a budget agree.
    """
    from page_text import DESIGN

    found = re.search(rf"const {name} = '([^']+)'", DESIGN)
    assert found is not None, f"{name} is not declared in the designer"
    return found.group(1)


def _all_facing_draft() -> dict:
    """The draft Q265 came in on, shrunk to the template it started from.

    A chess-shaped game in which every piece but the king points somewhere: the pawn
    steps and takes as it always did, and the rook goes forward and sideways but not
    back, which is what makes a facing mean anything -- a piece whose moves are
    symmetric under the turn is refused by the compiler, because turning it would be a
    pass.

    What matters about it is the *width*: a facing multiplies a position's legal moves
    by nearly four and most of what it adds is turns, so a node budget that searched two
    plies of chess searches one of this.
    """
    from chess_variant_sandbox.web import designing

    draft = json.loads(json.dumps(designing.templates()[1]["spec"]))
    for piece in draft["pieces"]:
        if piece["id"] == "R":
            piece["moves"] = [{"kind": "slide", "vectors": [[0, 1], [1, 0], [-1, 0]]}]
        if not piece.get("royal"):
            piece["facing"] = {"axes": 4}
    draft["setup"]["placements"] = [
        placement if placement["piece"] == "K" else {**placement, "facing": "forward"}
        for placement in draft["setup"]["placements"]
    ]
    draft["name"] = "everything points somewhere"
    return designing.settled(draft)


def _watch_press() -> dict:
    """Press *watch it play itself* in node, and read what it asked the server for.

    The seats are the point: a budget is part of an agent's identity, so which rung the
    watch seats is a decision the page makes and a test can hold. Everything the panel
    draws is stubbed except the wiring under the button.
    """
    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                                 set: () => true });
      const side = { innerHTML: '' };
      const wired = {};
      global.document = {
        getElementById: id => {
          if (id === 'verdictSide') return side;
          if (!wired[id]) wired[id] = { onclick: null, style: {}, innerHTML: '' };
          return wired[id];
        },
        querySelector: () => null, querySelectorAll: () => [],
        addEventListener: () => {}, createElement: () => el, body: el };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {},
                        getComputedStyle: () => ({ position: 'static' }) };
      global.location = { pathname: '/game', hash: '' };
      global.fetch = () => new Promise(() => {});
    """
    runner = """
      DRAFT = { name: 'a draft', pieces: [] };
      DRAFT_MAP = null; DRAWING = null;
      DRAFT_CHECK = { valid: true, playable: true, openingMoves: 20, ruleLines: [] };
      drawBoardSide = () => {}; identityHtml = () => ''; wireBoards = () => {};
      let asked = null;
      api = (path, body) => { asked = { path, body }; return new Promise(() => {}); };
      drawVerdict();
      wired.dWatch.onclick();
      console.log(JSON.stringify({ asked, title: side.innerHTML }));
    """
    done = node_session.run(stub + script.group(1) + "\n" + runner)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


class TestWhoPlaysTheWatch:
    """Q265. *Watch it play itself* seats a player, and the page says which one.

    The user built a chess-shaped draft in which every piece carried a facing and
    reported that the watch *"just played random nonsense moves"*. It was the player.
    The watch seated `minimax:nodes=400`, the weakest searching rung on the ladder, and
    a facing makes a position nearly four times wider -- so the search finished one ply
    and nearly half its moves went on turning a piece, which from the other side of the
    screen is indistinguishable from rules that cannot be played.

    Two things are held here. The watch seats the strongest rung that stays cheap enough
    to run a whole unattended game on a shared box, and the board says which player is
    playing, so weak play by a weak player can be told from a broken game.
    """

    def test_the_watch_seats_the_strongest_rung_that_stays_cheap(self) -> None:
        """A rung, not a budget somebody liked. A budget is part of an agent's identity,
        so a seat at 25,600 nodes would be a player nothing in the project has rated.

        *Cheap* is the site's own line and not a stopwatch here: `quota.STRONG_NODES` is
        the budget above which a seat is rationed, because one move there costs
        0.24-0.67 s and a watch plays a whole game unattended. So the watch takes the
        largest searching rung below that line -- 6-28 ms a move on the heaviest
        committed rulesets, 12 ms on the draft Q265 came in on (2026-09-17, this
        machine).
        """
        from chess_variant_sandbox.site import quota
        from chess_variant_sandbox.web import opponents

        watch = _page_agent("WATCH_AGENT")
        searches = {
            agent: int(agent.split("nodes=")[1])
            for agent, _ in opponents.LADDER
            if agent.startswith("minimax:nodes=")
        }
        assert watch in searches, "the watch seats a player nobody rated"
        assert searches[watch] < quota.STRONG_NODES, (
            "the watch seats a budget the site rations; a whole unattended game of it "
            "costs the box a core"
        )
        assert searches[watch] == max(
            nodes for nodes in searches.values() if nodes < quota.STRONG_NODES
        ), "there is a stronger rung that is still cheap; the watch should seat it"

    def test_the_button_asks_for_that_rung_on_both_sides(self) -> None:
        pressed = _watch_press()
        watch = _page_agent("WATCH_AGENT")

        assert pressed["asked"]["path"] == "/api/design/play"
        assert pressed["asked"]["body"]["seats"] == [watch, watch]
        # And the button says what it will do before it is pressed, with the budget it
        # is about to seat rather than a number written out beside it.
        nodes = int(watch.split("nodes=")[1])
        assert f"looking at {nodes:,} positions before each move" in _said(pressed["title"])

    def test_a_board_offers_the_same_watch(self) -> None:
        """Q262. The site lost computer-against-computer play when a match became a row,
        so the board carries the designer's control -- one constant and one rung, and a
        game nobody played follows nothing."""
        from page_text import PAGE

        start = PAGE.index("async function showBoard(snapshot, how)")
        body = PAGE[start:PAGE.index("function applySnapshot(", start)]
        assert "watch it play itself" in body
        assert "restart([WATCH_AGENT, WATCH_AGENT], false);" in body
        # Not offered on a board that is already a watch, where `new game` is that button.
        at = body.index('<button id="watch"')
        assert "watched ? ''" in body[at - 40 : at]

    def test_the_route_names_the_player_in_the_pickers_own_words(
        self, tmp_path: Path
    ) -> None:
        """The sentence is the server's, off the ladder's names and plain words, so a rung
        relabelled there is relabelled on the board rather than in two places."""
        from chess_variant_sandbox.web import opponents

        watch = _page_agent("WATCH_AGENT")
        rung = next(entry for entry in opponents.ladder() if entry.agent == watch)
        name, explains = rung.name, rung.about
        assert explains, "a searching rung says what it does"
        app = App(RULESETS, tmp_path, tmp_path / "heur", library.ARCHIVE_DIR)

        watched = app.post(
            "/api/design/play", {"spec": _all_facing_draft(), "seats": [watch, watch]}
        )
        assert name in watched["watching"] and explains in watched["watching"]
        assert "Both sides" in watched["watching"]

        # A game somebody is sitting at needs no such sentence: they chose the opponent,
        # and the panel beside the board names it.
        played = app.post(
            "/api/design/play",
            {"spec": _all_facing_draft(), "seats": ["human", watch]},
        )
        assert played["watching"] == ""

    def test_the_note_names_the_player_and_how_much_it_can_see(self) -> None:
        """What the board carries during a watch. The server's sentence, then why the
        play may still be thin -- which is the question an unattended game raises and
        the one nothing on the page used to answer."""
        watch = _page_agent("WATCH_AGENT")
        lines = _in_page(
            f"const WATCHED = {json.dumps([watch, watch])};",
            "[watchingLine({ watching: 'Both sides are played by hard (looks at 6,400"
            " positions before each move).', seats: WATCHED, moves: [1, 2, 3] }),"
            " watchingLine({ watching: '', seats: WATCHED, moves: [] }),"
            " watchingLine({ watching: 'random (plays anything legal) is playing hard"
            " (looks at 6,400 positions before each move).', seats: ['random', WATCHED[0]],"
            " moves: [1] })]",
        )
        assert (
            "Both sides are played by hard (looks at 6,400 positions before each move)."
            in lines[0]
        )
        # P21: in a visitor's words, with no *budget* and no *searching player*.
        assert "strongest computer player quick enough to watch" in lines[0]
        assert "3 legal moves" in lines[0]
        assert "budget" not in lines[0] and "searching" not in lines[0]
        # Nothing to say about a game a person is playing, and no claim about a player
        # the designer did not choose.
        assert lines[1] == ""
        assert "strongest computer player" not in lines[2]

    def test_the_watch_searches_two_plies_where_the_old_seat_searched_one(self) -> None:
        """The substance of the fix, on the draft it was reported on.

        Depth is what a budget buys and it is a function of the game: the same 400 nodes
        that see two plies of a template see one of a draft where every piece can turn.
        Asked of the core through `Engine`, one position at a time, exactly as the watch
        asks it -- `web/play.py::play_agent` calls `choose_move` per move.
        """
        from chess_variant_sandbox.tournament.play import eval_values, native_spec

        draft = _all_facing_draft()
        engine = Engine(draft)
        opening = engine.legal_moves(engine.initial_state())
        assert len(opening) > 30, "a draft this narrow does not show what Q265 was about"

        def depths(agent: str) -> list[int]:
            native = native_spec(agent, engine.spec.hash)
            values = eval_values(engine.spec)
            state = engine.initial_state()
            found = []
            for _ in range(8):
                move, depth = engine.select_move_depth(state, native, values, state.ply)
                assert move is not None
                found.append(depth)
                state = engine.apply(state, move)
            return found

        watched = depths(_page_agent("WATCH_AGENT"))
        weakest = depths("minimax:nodes=400")
        assert min(watched) >= 2, watched
        assert sum(watched) / len(watched) >= sum(weakest) / len(weakest) + 0.75, (
            watched, weakest
        )


class TestTheBoardSaysHowHardTheMachineThinks:
    """Q288. The seat panel read *hard vs hard*, which names a rung and says nothing
    about it, and `play_agent` threw away how far each reply looked.

    Two halves, and they are different kinds of fact. The strength is the seat's
    identity -- a node budget, fixed when the game starts -- so it is said once, beside
    the seat. The depth is what one search reached in one position, which only the core
    knows, so it travels with the move it was reported for.
    """

    def test_a_seat_is_described_by_the_budget_it_plays_at(self, chess_game) -> None:
        from chess_variant_sandbox.web import opponents
        from chess_variant_sandbox.web.play import strength

        snapshot = Sessions().start(chess_game, ("human", "minimax:nodes=6400")).snapshot()
        assert snapshot["strength"] == ["", "looks at 6,400 positions before each move"]

        # Every rung the picker offers says something, and two budgets under one kind
        # of search are two different sentences -- which is the point of saying it.
        said = {agent: strength(agent) for agent, _ in opponents.LADDER}
        assert said.pop("human") == ""
        assert all(said.values()), said
        assert len(set(said.values())) == len(said), said
        for agent, words in said.items():
            if "nodes=" in agent:
                assert f"{int(agent.split('nodes=')[1]):,}" in words, (agent, words)

        # A rule of thumb thinks at its own budget, and the words do not need its
        # store: the digest is who it is, not how hard it looks.
        assert strength("minimax:h=deadbeefcafe,nodes=6400") == said["minimax:nodes=6400"]

    def test_a_seat_with_no_budget_written_is_described_at_the_one_it_gets(self) -> None:
        """Off the core's own parameters for the kind, so a default moved there moves
        here, and nothing is said about a player this does not know."""
        from chess_variant_sandbox.agents import native_spec
        from chess_variant_sandbox.web.play import strength

        default = native_spec("minimax")
        assert default is not None
        assert f"{default[1]:,} positions" in strength("minimax")
        assert strength("nonsense") == ""

    def test_the_panel_draws_it_under_the_seat(self) -> None:
        drawn = _in_page(
            "SESSION = { seats: ['human', 'minimax:nodes=6400'], ruleset: { sides:"
            " ['white', 'black'] }, strength: ['', 'looks at 6,400 positions before"
            " each move'] };",
            "[seatHtml(0), seatHtml(1), (SESSION.strength = undefined, seatHtml(1))]",
        )
        assert "looks at 6,400 positions before each move" in _said(drawn[1])
        # A person has no line under the name, and a snapshot from before the field
        # draws the name alone rather than failing.
        assert "<div" not in drawn[0]
        assert "<div" not in drawn[2] and "black" in drawn[2]

    def test_a_machine_move_keeps_how_far_it_looked(self, chess_game) -> None:
        """The depth is the core's, asked through `Engine` in the position the reply
        was chosen in, and the move beside it is the one `choose_move` would play."""
        from chess_variant_sandbox.tournament.play import eval_values, native_spec

        agent = "minimax:nodes=6400"
        session = Sessions().start(chess_game, ("human", agent), seed=3)
        session.play(session.snapshot()["moves"][0]["move"])
        engine = Engine(chess_game.spec)
        asked = session.state
        seed = 3 + asked.ply
        expected, depth = engine.select_move_depth(
            asked, native_spec(agent, engine.spec.hash), eval_values(engine.spec), seed
        )
        assert expected == choose_move(engine, agent, asked, seed=seed)

        replied = session.play_agent()
        person, machine = replied["history"]
        assert person["depth"] is None
        assert machine["text"] == engine.describe_move(asked, expected)
        assert machine["depth"] == depth and depth >= 2, depth

    def test_a_player_that_does_not_look_ahead_reports_no_depth(self, chess_game) -> None:
        """The core reports zero for a player that does not deepen, and from a searching
        player zero would mean it played blind -- so nothing is passed on for them."""
        session = Sessions().start(chess_game, ("greedy", "mcts:nodes=50"), seed=1)
        session.play_agent()
        replied = session.play_agent()
        assert [move["depth"] for move in replied["history"]] == [None, None]

    def test_a_replayed_game_says_what_the_played_one_said(self, chess_game) -> None:
        """Q424: `replay` is how the site's stored game comes back, and each move takes
        the depth stored beside it rather than being searched again. What it is handed is
        a record, so a person's move takes none whatever it is handed, and a move the
        position does not allow is refused as `play` refuses it."""
        played = Sessions().start(chess_game, ("human", "minimax:nodes=400"), seed=5)
        played.play(played.snapshot()["moves"][0]["move"])
        played.play_agent()
        assert played.depths[0] is None and isinstance(played.depths[1], int)

        rebuilt = Sessions().start(chess_game, ("human", "minimax:nodes=400"), seed=5)
        for move, depth in zip(played.played, played.depths, strict=True):
            rebuilt.replay(move, depth)
        assert rebuilt.history == played.history
        assert rebuilt.snapshot()["history"] == played.snapshot()["history"]

        # A store that kept nothing gives a board with nothing beside any move.
        bare = Sessions().start(chess_game, ("human", "minimax:nodes=400"), seed=5)
        for move in played.played:
            bare.replay(move)
        assert bare.depths == [None, None]

        told = Sessions().start(chess_game, ("human", "minimax:nodes=400"), seed=5)
        told.replay(played.played[0], 7)
        assert told.depths == [None], "a person's move was given a depth"
        with pytest.raises(ValueError, match="not legal"):
            told.replay(played.played[0])

    def test_the_moves_list_shows_it_beside_the_move(self) -> None:
        setup = """
          const pane = { innerHTML: '', className: '', children: null, scrollTop: 0 };
          document.getElementById = id => (id === 'pane' ? pane : null);
          TAB = 'moves';
          SESSION = { ruleset: { hash: 'x' }, history: [
            { ply: 0, text: 'Pe2-e4', by: 'human', depth: null },
            { ply: 1, text: 'Pe7-e5', by: 'minimax:nodes=6400', depth: 4 },
            { ply: 2, text: 'Ng1-f3', by: 'human', depth: null },
            { ply: 3, text: 'Nb8-c6', by: 'minimax:nodes=6400', depth: 1 },
            { ply: 4, text: 'Bf1-c4', by: 'human', depth: null },
            { ply: 5, text: 'Ra8-b8', by: 'minimax:nodes=400', depth: 0 } ] };
        """
        rows = _in_page(
            setup, "(drawPane(), pane.innerHTML.split('</div>').slice(0, 6))"
        )
        assert "looked" not in rows[0] and "ahead" not in rows[0]
        # P14: in turns, the unit the picker's greedy player is described in.
        assert "Pe7-e5" in rows[1] and "looked 4 turns ahead" in rows[1]
        assert "looked 1 turn ahead" in rows[3]
        assert "played without looking ahead" in rows[5]
        # The tooltip says what a turn is, and that the picker's positions are what
        # bought them (F7: the label and the move list read as one thing).
        assert "a turn is one player&#39;s move" in rows[1]
        assert "positions it looks at before each move" in rows[1]

    def test_a_rebuilt_game_keeps_the_depths_the_page_was_told(self) -> None:
        """The site stores each depth beside its move since Q424, but a row from before
        that kept none, so a page open across the deploy is sent the rebuilt game without
        them. The page carries a depth forward to the same move of the same game, and to
        nothing else."""
        kept = _in_page(
            "",
            """(() => {
              const told = { id: 'a', history: [
                { ply: 0, text: 'x', by: 'human', depth: null },
                { ply: 1, text: 'y', by: 'minimax:nodes=6400', depth: 3 } ] };
              const rebuilt = () => ({ id: 'a', history: [
                { ply: 0, text: 'x', by: 'human', depth: null },
                { ply: 1, text: 'y', by: 'minimax:nodes=6400', depth: null },
                { ply: 2, text: 'z', by: 'minimax:nodes=6400', depth: 2 } ] });
              const same = rebuilt(); keepDepths(told, same);
              const other = Object.assign(rebuilt(), { id: 'b' }); keepDepths(told, other);
              const moved = rebuilt(); moved.history[1].text = 'w'; keepDepths(told, moved);
              return [same.history.map(m => m.depth), other.history[1].depth,
                      moved.history[1].depth];
            })()""",
        )
        assert kept == [[None, 3, 2], None, None]
        from page_text import PAGE

        start = PAGE.index("function applySnapshot(")
        assert "keepDepths(SESSION, snapshot);" in PAGE[start : start + 200]


class TestOnlyADirectionalPieceIsTurned:
    """What the board actually draws, by running its own `drawPiece` in node.

    The board used to turn every one of the second player's icons, because it read
    `facingAngles[player]` for every piece and that table is `[[0], [180]]` wherever
    nothing faces. `directional` is the engine's per-piece answer to which types the
    compiler gave a facing, and the page reads it rather than deciding anything itself.
    """

    # Two types with four facings apiece: type 0 has a facing, type 1 has none.
    # `facingAngles` is `chess-oriented-pawns`': the second player's are reflected, so
    # the same facing index turns the other way round on screen.
    RULESET = (
        "SESSION = { ruleset: { nTypes: 2, sideStride: 8, pieceIds: ['P', 'R'],"
        " facingAngles: [[0, 90, 180, 270], [180, 90, 0, 270]],"
        " directional: [true, false],"
        " icons: [[{ href: 'p0' }, { href: 'p1' }], [{ href: 'r0' }, { href: 'r1' }]] } };"
    )

    @staticmethod
    def _angle(drawn: str) -> int | None:
        found = re.search(r"rotate\((-?[\d.]+)", drawn)
        return None if found is None else int(float(found.group(1)))

    def _drawn(self, player: int, kind: int, facing: int = 0) -> str:
        return _in_page(
            self.RULESET,
            f"drawPiece({{ player: {player}, type: {kind}, facing: {facing} }}, 0, 0, 10)",
        )

    def test_a_piece_with_no_facing_is_upright_for_both_seats(self) -> None:
        """The rule the user asked for. An icon arrives pointing up and stays there."""
        assert self._angle(self._drawn(0, 1)) is None
        assert self._angle(self._drawn(1, 1)) is None

    def test_a_piece_with_a_facing_points_where_it_faces(self) -> None:
        """Zero is up relative to the board, and the heading comes off `facingAngles`.
        Facing 1 is east for the first player; the second player reads it reflected."""
        assert self._angle(self._drawn(0, 0)) is None
        assert self._angle(self._drawn(0, 0, facing=1)) == 90
        assert self._angle(self._drawn(1, 0)) == 180

    def test_a_single_colour_set_still_marks_its_second_player(self) -> None:
        """Shogi and dobutsu tell the two armies apart by turning the tile round, and
        that is ownership rather than heading -- so it survives, on the pieces with no
        facing. `side_names` calls those sides first and second for the same reason.
        """
        ruleset = self.RULESET.replace("{ href: 'r1' }", "{ href: 'r1', rotate: true }")
        drawn = _in_page(
            ruleset, "drawPiece({ player: 1, type: 1, facing: 0 }, 0, 0, 10)"
        )
        assert self._angle(drawn) == 180

    def test_the_designer_preview_reads_the_same_list(self) -> None:
        """Two renderers, one rule. The editor's preview draws a board of its own, and a
        rule the play board keeps and the preview does not is a rule that is half true.

        Read per drawing function rather than counted over the whole page: the same list
        also decides whether a piece gets one upload button or one per seat, and that is
        a third reader with nothing to say about which way a piece is turned.
        """
        from page_text import PAGE

        # The still draws a piece in `stillPieceHtml` since Q843d, which an echo in a fold
        # draws too.
        for name in ("function drawPiece(", "function stillPieceHtml("):
            start = PAGE.index(name)
            body = PAGE[start:PAGE.index("\nfunction ", start + 1)]
            assert body.count("directional || [])[") == 1, name


class TestAnUploadOntoAPieceDrawnPerPlayer:
    """A facing piece is two drawings, and an upload used to write one string over both.

    `appearance.icons[id]` is a library name, or a *pair* of them, one per player --
    which a piece with a facing needs, because a piece that points cannot also carry its
    side by being drawn upside down. The upload wrote a single data URI into that slot,
    so a drawing meant for one seat became the drawing for both and the pair the game
    needed was gone, silently.

    The editor now offers one upload button per seat wherever a piece is drawn per
    player -- it has a facing, or its entry is already a pair because somebody typed two
    names -- and an upload writes only that seat's element.
    """

    # A pikeman that points, already wearing a pair, and a king that does not.
    FIXTURE = """
    DRAFT = { pieces: [{ id: 'P', name: 'pikeman' }, { id: 'K', name: 'king' }],
              appearance: { icons: { P: ['spear-left', 'spear-right'] } } };
    DRAFT_CHECK = { preview: { directional: [true, false], sides: ['white', 'black'],
      icons: [[{ href: 'a' }, { href: 'b' }], [{ href: 'c' }, { href: 'c' }]] } };
    """

    def test_a_piece_drawn_per_player_gets_one_button_per_seat(self) -> None:
        shown = _in_page(self.FIXTURE, "[uploadButtonsHtml(0), uploadButtonsHtml(1)]")
        assert shown[0].count("<button") == 2
        assert 'data-seat="0"' in shown[0] and 'data-seat="1"' in shown[0]
        # Named as the board names the seats -- white and black here, first and second
        # for a single-colour set -- rather than as player one and player two.
        assert "white" in shown[0] and "black" in shown[0]
        # One drawing serves a piece with no facing, so it keeps the single button.
        assert shown[1].count("<button") == 1 and "data-seat" not in shown[1]

    def test_the_upload_lands_on_the_chosen_seat_and_leaves_the_other(self) -> None:
        kept = _in_page(
            self.FIXTURE,
            "[seatPair(DRAFT.appearance.icons.P, DRAFT.pieces[0], 1, 'data:svg,new'),"
            " seatPair(DRAFT.appearance.icons.P, DRAFT.pieces[0], 0, 'data:svg,new')]",
        )
        assert kept[0] == ["spear-left", "data:svg,new"]
        assert kept[1] == ["data:svg,new", "spear-right"]

    def test_a_piece_with_no_pair_yet_grows_one_without_losing_the_other_seat(
        self,
    ) -> None:
        """A facing piece that holds one name, or none, has to become a pair without the
        seat nobody touched losing its picture. It is seeded with what that seat already
        wears -- the typed name, or the piece's own name, which is what the icon lookup
        falls back to when there is no entry at all."""
        grown = _in_page(
            self.FIXTURE,
            "[seatPair('knight', DRAFT.pieces[0], 0, 'data:svg,new'),"
            " seatPair(undefined, DRAFT.pieces[0], 1, 'data:svg,new')]",
        )
        assert grown[0] == ["data:svg,new", "knight"]
        assert grown[1] == ["pikeman", "data:svg,new"]

    def test_one_drawing_for_both_is_still_written_as_one_string(self) -> None:
        """Almost every entry is a string, and an upload with no seat named keeps it one
        -- the pair is for the pieces that need it, not a shape every piece grows into."""
        from page_text import PAGE

        start = PAGE.index("function uploadIcon(")
        body = PAGE[start:PAGE.index("\nfunction ", start + 1)]
        assert "seat === null" in body
        assert "? answer.href : seatPair(held, piece, seat, answer.href)" in body

    def test_the_slot_shows_both_drawings_where_the_seats_differ(self) -> None:
        """Otherwise an upload onto the second seat lands in the spec and shows nowhere,
        and the reader has no way to tell it worked."""
        shown = _in_page(self.FIXTURE, "[pieceIconHtml(0), pieceIconHtml(1)]")
        assert shown[0].count("<img") == 2
        assert 'src="a"' in shown[0] and 'src="b"' in shown[0]
        assert shown[1].count("<img") == 1

    def test_typing_a_second_name_splits_the_button_without_redrawing_the_form(
        self,
    ) -> None:
        """The icon field turns a piece paired while it is being typed into, and
        redrawing the form on a keystroke takes the focus out of it. The check's own
        refresh rebuilds the button box instead, and rewires what it rebuilt."""
        from page_text import PAGE

        start = PAGE.index("function checkDraft(")
        body = PAGE[start:PAGE.index("\nfunction ", start + 1)]
        assert "uploadButtonsHtml(Number(box.dataset.uploads))" in body
        assert "wireUploads(box)" in body
        assert "drawDesigner()" not in body


class TestTheDesignerPreviewDrawsWhatTheBoardDraws:
    """The still preview drew a bare grid -- no goal ring, no promotion band, no seam.

    Not because it disagreed with the play board about any of them, but because its
    payload carried none of the three compiled fields the board reads: `squareMarks` had
    nothing to mark and there was no wrap rule to find a seam in. So a draft whose whole
    point is a racing goal or a promotion band showed neither until it was played.
    """

    def test_the_check_sends_the_three_compiled_fields(self) -> None:
        from chess_variant_sandbox.web import designing

        preview = designing.check(load_spec(Path(RULESETS) / "chess.json"))["preview"]
        # Chess promotes on the far rank each way, so the two bands are its two ends.
        assert preview["zones"] is not None
        assert preview["zones"][0] and preview["zones"][1]
        assert preview["zones"][0] != preview["zones"][1]
        # A game with edges and no square to reach says so with None, which is what both
        # renderers test to skip the whole of the drawing.
        assert preview["goals"] is None and preview["wrap"] is None

    def test_a_game_with_promotion_shows_the_band(self) -> None:
        drawn = _in_page(
            _DRAFT_FIXTURE,
            "[boardPreview(Object.assign({}, PREVIEW, { zones: [[6, 7, 8], [0, 1, 2]] }),"
            " null), boardPreview(PREVIEW, null)]",
        )
        # One hatched cell per square of the band, filled from the one pattern the
        # still defines in the zone's colour (Q577: ground, not a ring with beads).
        assert drawn[0].count('data-mark="zone"') == 6
        assert drawn[0].count("var(--zone)") == 1
        assert "var(--zone)" not in drawn[1], "a game with no promotion gets no band"

    def test_a_goal_square_and_a_seam_are_drawn_too(self) -> None:
        drawn = _in_page(
            _DRAFT_FIXTURE,
            "boardPreview(Object.assign({}, PREVIEW, { goals: [[8], [0]],"
            " wrap: { files: [[true, false], [true, false]], ranks: [] } }), null)",
        )
        assert "var(--goal)" in drawn
        # A seam one side may cross one way only is a fact of the game, so the preview
        # draws the line rather than leaving it to be inferred.
        assert "var(--seam)" in drawn
        # And opens its frame, because the seam's arrows point out of the board and a
        # still preview has no ghost rings to hold them.
        assert 'viewBox="0 0' not in drawn

    def test_the_reader_s_promotion_toggle_reaches_the_preview(self) -> None:
        """One stored key across the play board, the replay and this: a reader who put
        the band away finds it away here too. The preview offers no control of its own --
        a still of a draft is not where a promotion band gets put away."""
        from page_text import PAGE

        start = PAGE.index("function boardPreview(")
        body = PAGE[start:PAGE.index("\nfunction ", start + 1)]
        assert "squareMarks(preview, { zone: !zonesShown() })" in body
        assert "zoneToggleHtml" not in body


def test_the_board_the_editor_draws_stays_on_screen() -> None:
    """The single most useful thing the editor draws used to leave the screen as soon as
    you scrolled to the piece you were editing. Held by CSS, so this reads the CSS -- the
    behaviour itself was checked in a browser."""
    from page_text import PAGE

    assert ".designer .stick { position: sticky;" in PAGE
    assert "align-self: start;" in PAGE[PAGE.index(".designer .stick") :][:120]


def test_the_sticking_columns_stick_only_beside_a_form_column() -> None:
    """Q490. From 901 to 1199 px the bench puts the board and the verdict side by side
    and runs the form full width under them, so a column that sticks there slides down
    over the form: at 1024 px, every panel open, 122 of its 127 controls came to rest
    under the board or the verdict, *add it* among them. The columns stick only at the
    width the form gets its own column, which is one media query, read here by name.
    Walked in a browser at 375, 800, 901, 1024, 1199, 1200 and 1440 px, scrolling each
    of the form's controls to mid-screen and asking what is on top of it: nothing is."""
    from page_text import PAGE

    at = PAGE.index(".designer .stick {")
    sticks_at = PAGE[PAGE.rindex("@media", 0, at) :].split("{", 1)[0].strip()
    three = PAGE.index(".designer.bench { grid-template-columns: minmax(280px, 350px)")
    three_at = PAGE[PAGE.rindex("@media", 0, three) :].split("{", 1)[0].strip()
    assert sticks_at == three_at == "@media (min-width: 1200px)"
    # The one rule that places the form under the other two ends below that width.
    under = PAGE.index(".designer.bench > #form { order: 3; grid-column: 1 / -1; }")
    assert PAGE[PAGE.rindex("@media", 0, under) :].startswith(
        "@media (min-width: 901px) and (max-width: 1199px)")
    assert PAGE.count(".designer .stick {") == 1


class TestEveryMoveRowControlCarriesItsLabel:
    """A6. The row was four menus and three unlabelled number boxes, and the boxes read
    *far*, *1* and *any* -- placeholders that look like values already entered, with the
    meaning of each in a tooltip. Nothing on it said which of a direction's two numbers
    is sideways, which is what walk 1.1 of the plan stopped on twice."""

    def _row(self) -> str:
        return _in_page(
            "extrasHtml = () => '';",
            "moveEditor({ kind: 'slide', vectors: [[0, 1]], range: 2 }, 0, 0)",
        )

    def test_no_control_stands_without_one(self) -> None:
        row = self._row()
        # Seven controls, seven labels, and every one of them inside a label.
        assert row.count("class=\"moveField\"") == 7
        assert row.count("data-mfield=") == 7
        for label in ("how it travels", "turned", "direction, sideways then forward",
                      "how far", "no nearer than", "only from rank", "and it may"):
            assert f"<span>{label}</span>" in row, label

    def test_the_direction_box_says_which_number_is_sideways(self) -> None:
        row = _said(self._row())
        assert "sideways then forward" in row
        assert "0,1 is one square forward and 1,0 is one to the right" in row

    def test_a_placeholder_no_longer_reads_as_a_value(self) -> None:
        """*far* in a number box is a word somebody may have typed. What the blank means
        is said in the label above it now, so the placeholder can say what it does."""
        assert 'placeholder="to the edge"' in self._row()


class TestOneBoardInTwoPlaces:
    """Q307. `showBoard` draws into whatever it is given, so the designer can play a
    draft on the board in the column it edits in without a second copy of the board
    existing anywhere.

    The two differ only in chrome. A page board is the whole screen and carries the
    crumb out of the game and the panel of rules, pieces and moves beside it; an inline
    board sits beside an editor already showing all three, so it carries neither and
    gives that width to the board. Everything below the chrome -- the drawing, the
    animation, the review row, the hand, the status -- is one function either way.
    """

    def _show(self) -> str:
        from page_text import PAGE

        start = PAGE.index("async function showBoard(snapshot, how) {")
        return PAGE[start : PAGE.index("function applySnapshot(", start)]

    def test_there_is_one_board_markup(self) -> None:
        from page_text import PAGE

        assert PAGE.count('<svg id="board"') == 1

    def test_the_mount_is_what_says_which_of_the_two_it_is(self) -> None:
        body = self._show()
        # Q993b: a third chrome, the proven positions' page, is a mount with `proven`.
        assert "const proven = (how && how.mount && how.proven) || null;" in body
        assert "const inline = !!(how && how.mount) && !proven;" in body
        assert "const root = how && how.mount ? how.mount : view;" in body
        assert "root.innerHTML = `" in body
        assert "view.innerHTML" not in body

    def test_the_page_board_keeps_its_crumb_and_the_panel_beside_it(self) -> None:
        body = self._show()
        assert '<button class="crumb" id="back">' in body
        # The tab it opens on is `TAB` (K3: a first game opens on the rules), moves first.
        assert "${['moves', 'rules', 'pieces'].map(tab => `<button data-tab=" in body
        assert "TAB = OPEN_TAB || 'moves';" in body

    def test_an_inline_board_carries_neither(self) -> None:
        body = self._show()
        # Both are written as the `inline` arm of one conditional rather than as a
        # second template, so a control added to the board is added to every board.
        assert "${inline || proven ? '' : `<div class=\"panel boardSide\">" in body
        assert 'id="stopPlay"' in body

    def test_the_crumb_is_wired_by_the_chrome_and_never_by_presence(self) -> None:
        """The designer has a crumb of its own called `back`. An inline board that
        looked one up would find it and rewire the editor's own way out of the page."""
        body = self._show()
        at = body.index("document.getElementById('back').onclick")
        assert "if (inline) document.getElementById('stopPlay').onclick" in body
        assert body[at - 6 : at].strip() == "else"

    def test_the_board_is_given_up_in_one_place(self) -> None:
        """A view that takes the board away drops the session, the timeline and the
        review cursor together -- `maybeAgent` then drops a reply still on its way,
        because the session it was asked of is no longer the one on screen. What it
        clears is checked by running it, in `tests/test_designer.py`."""
        from page_text import PAGE

        from chess_variant_sandbox.web.page import SCRIPT_DIR as PAGE_SCRIPTS

        assert PAGE.count("function leaveBoard(") == 1
        # Its callers: the designer, and (Q993b) the proven positions' page, which gives a
        # try up when another position is chosen or the view is drawn again.
        callers = {name: (PAGE_SCRIPTS / f"{name}.js").read_text(encoding="utf-8")
                   .count("leaveBoard();") for name in ("design", "proven")}
        assert callers == {"design": 1, "proven": 2}
        assert PAGE.count("leaveBoard();") == sum(callers.values())


def test_the_sticky_board_covers_no_panel_below_it() -> None:
    """A sticky panel paints over every sibling after it, so the panels under the board
    had their buttons covered while the page scrolled. A whole *column* sticks instead,
    and each of the two that do holds its own panels and nothing else's. Checked in a
    browser by clicking the info button of *Which game this is* on a long form."""
    from page_text import PAGE

    assert '<div id="boardSide" class="stick"></div>' in PAGE
    assert '<div id="verdictSide" class="stick"></div>' in PAGE
    assert 'class="panel stick"' not in PAGE
    assert PAGE.count('class="stick"') == 2


class TestTheDesignersFootBarOnAPhone:
    """Q300. Stacked, the verdict column sits under the whole form, which put the
    pass/fail sentence and *save it*, *play it* and *watch it* at the end of a 9,500 px
    page -- *play it, without saving* was at y=7,657 at 375 px. They are one element
    now, and the stylesheet fixes that element to the foot of the screen.

    Held here: the element exists in both of the verdict's states, it carries the three
    buttons, only the narrow breakpoint moves it, and the page gives back the height a
    fixed bar takes out of the flow. The behaviour was walked in a browser at 375x812
    and 375x400, on a draft opened from *almost chess* with *make one like this*: the
    bar is 166 px with the sentence and 40% of the screen at its tallest, the page
    scrolls sideways at neither size, and all three buttons answer.
    """

    def _verdict(self) -> str:
        from page_text import PAGE

        start = PAGE.index("function drawVerdict(")
        return PAGE[start : PAGE.index("\nfunction ", start + 1)]

    def test_the_sentence_and_the_buttons_are_one_element(self) -> None:
        """The earlier worker on this box stopped here: no rule over the stylesheet can
        lift three elements out of the middle of a panel, so the panel gives them up."""
        verdict = self._verdict()
        # The last of the two bars: the first is the one a draft that is not a game yet
        # gets, which carries the errors and no buttons at all.
        bar = verdict[verdict.rindex('<div class="footbar" id="dActions">') :]
        # The bar's own close, which is the one at the panel's indent.
        bar = bar[: bar.index("\n      </div>")]
        for button in ('id="dSave"', 'id="dPlay"', 'id="dWatch"'):
            assert button in bar, button
        assert "Playing it keeps nothing" not in bar, "the bar carries the buttons alone"
        # Two states, two bars: a draft that is not a game yet still says so at the foot.
        assert verdict.count('class="footbar" id="dActions"') == 2

    def test_only_the_narrow_breakpoint_moves_it(self) -> None:
        from page_text import PAGE

        at = PAGE.index(".designer .footbar {")
        assert "position: fixed" in PAGE[at : at + 140]
        assert PAGE[PAGE.rindex("@media", 0, at) :].startswith("@media (max-width: 900px)")

    def test_the_page_takes_back_what_a_fixed_bar_costs_it(self) -> None:
        """A fixed bar is out of the flow, so without this the last panel of the editor
        sits under it. Exactly its height, measured; and the width it happens at is the
        stylesheet's alone, which is what reading `position` back off the bar buys."""
        from page_text import DESIGN, PAGE

        start = PAGE.index("function reserveFootbar(")
        body = PAGE[start : PAGE.index("\nfunction ", start + 1)]
        assert "offsetHeight" in body and "paddingBottom" in body
        # Every exit from the verdict, including the one that draws no bar at all.
        assert self._verdict().count("reserveFootbar()") == 3
        assert "900" not in DESIGN, "the breakpoint is written in the stylesheet"

    def test_a_save_says_what_it_did_where_the_press_was(self) -> None:
        """The button is in the bar and the answer lands in the panel behind it, which
        on a phone is several screens away. Both answers, kept and refused."""
        from page_text import PAGE

        start = PAGE.index("function saveDraft(")
        body = PAGE[start : PAGE.index("\nfunction ", start + 1)]
        assert body.count("showSaved(box)") == 2


def test_clicking_a_control_in_a_piece_card_keeps_the_cursor() -> None:
    """A click anywhere in a piece card selects that piece. The handler used to redraw
    the whole form, which replaced the input the same click had just focused, so no
    field in a piece card could be typed into after clicking it. Checked in a browser
    at 600px and 375px with real clicks. The highlight now moves in place."""
    from page_text import PAGE

    start = PAGE.index("form.querySelectorAll('[data-piece]').forEach")
    body = PAGE[start : PAGE.index("wireUploads(form);", start)]
    assert "drawDesigner()" not in body
    assert "classList.toggle('on', on)" in body


def _designer_forms(specs: dict[str, dict]) -> dict[str, dict]:
    """Draw the designer's form for each spec, in node, and report what it built.

    `drawDesigner` is the one place a spec becomes the editor's form, so running it is
    what "the game opens in the designer" means as a test. The stub is a DOM good enough
    for the form to be built and wired; the HTML is the thing under test. One node
    process for the whole corpus, because sixty starts is a minute and this is not slow.
    Returns `{name: {"html": ...}}`, or `{"error": ...}` for a spec that threw.
    """
    import json as _json
    import re

    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      function __el() {
        return { dataset: {}, tagName: 'DIV', value: '', checked: false, style: {},
                 innerHTML: '', querySelector: () => __el(), querySelectorAll: () => [],
                 addEventListener: () => {}, appendChild: () => {}, remove: () => {} };
      }
      const FORM = __el();
      global.document = { getElementById: (id) => (id === 'form' ? FORM : __el()),
                          querySelector: () => __el(), querySelectorAll: () => [],
                          addEventListener: () => {}, createElement: () => __el(),
                          body: __el(), createElementNS: () => __el() };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
      global.location = { pathname: '/game', hash: '' };
      global.fetch = () => new Promise(() => {});
    """
    run = """
      const OUT = {};
      for (const [name, spec] of Object.entries(SPECS)) {
        DRAFT = spec;
        DESIGN_KIT = { templates: [] };
        DRAFT_CHECK = null;
        DRAFT_PIECE = 0;
        FORM.innerHTML = '';
        try { drawDesigner(); OUT[name] = { html: FORM.innerHTML }; }
        catch (error) { OUT[name] = { error: String(error) }; }
      }
      console.log(JSON.stringify(OUT));
    """
    body = stub + script.group(1) + "\nconst SPECS = " + _json.dumps(specs) + ";\n" + run
    done = node_session.run(body)
    assert done.returncode == 0, done.stderr
    return _json.loads(done.stdout)


def _icon_fields(html: str) -> list[str]:
    """The value of every icon field the form drew, in piece order."""
    found = re.findall(r'data-field="icon" data-index="(\d+)"[^>]*value="([^"]*)"', html)
    return [value for _, value in sorted(found, key=lambda pair: int(pair[0]))]


class TestEveryGameWeHaveOpensInTheDesigner:
    """The designer is where a reader forks a game, so a game it refuses to open is a
    game that cannot be forked.

    `appearance.icons[id]` has had two spellings since schema 8 -- a name, or a pair of
    names one per player, which is what a piece with a facing needs because a piece that
    points cannot also carry its side by being drawn upside down. The field read the
    first spelling and called `startsWith` on the second, so both weathervanes threw
    `named.startsWith is not a function` before the form was drawn. Nothing caught it
    because every other committed game names one icon per piece.
    """

    def test_every_committed_and_archived_spec_draws_a_form(self) -> None:
        paths = sorted(Path(RULESETS).glob("*.json")) + sorted(
            Path(library.ARCHIVE_DIR).glob("*.json")
        )
        assert len(paths) > 40, "the corpus is not where this test thinks it is"
        specs = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in paths}
        drawn = _designer_forms(specs)
        broken = {name: got["error"] for name, got in drawn.items() if "error" in got}
        assert not broken, broken
        # A form, not an empty string returned by a stub that silently did nothing.
        for name, got in drawn.items():
            assert 'data-field="icon"' in got["html"], name

    def test_a_drawing_per_player_reaches_the_field_as_two_names(self) -> None:
        """What the pair looks like on screen, and what a single name still looks like.

        The field takes the pair back the way it shows it, so the two names are separated
        by a space and nothing else. Two identical names are one name, because that is
        what a reader typed and what the spec means.
        """
        piece = {
            "id": "P",
            "name": "pawn",
            "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1}],
        }
        cases = {
            "pair": ["knight", "cannon"],
            "single": "knight",
            "same-twice": ["knight", "knight"],
            "uploaded": "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
            "uploaded-pair": [
                "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
                "data:image/svg+xml;base64,PHN2Zz4gPC9zdmc+",
            ],
        }
        specs = {
            name: {
                "schema_version": 8,
                "name": name,
                "board": {"width": 4, "height": 4},
                "pieces": [piece],
                "appearance": {"icons": {"P": icon}},
            }
            for name, icon in cases.items()
        }
        drawn = _designer_forms(specs)
        shown = {name: _icon_fields(got["html"])[0] for name, got in drawn.items()}
        assert shown["pair"] == "knight cannon"
        assert shown["single"] == "knight"
        assert shown["same-twice"] == "knight"
        # An uploaded drawing has no name to show, either spelling of it.
        assert shown["uploaded"] == ""
        assert shown["uploaded-pair"] == ""

    def test_the_field_reads_and_writes_the_pair(self) -> None:
        """The page is a string, so this pins both halves of the round trip: the read
        knows the array, and the write puts one back rather than a name with a space in
        it."""
        from page_text import PAGE

        assert "Array.isArray(named)" in PAGE
        assert "DRAFT.appearance.icons[piece.id] = typed;" in PAGE


class TestHowManyRunsTheBrowserAsksFor:
    """`/api/test` takes `seeds`, so a panel can ask for the rung it needs.

    The designer's own measure rung stays at one run and labels it. The panel that offers
    promotion asks for four, because `corpus.promotion_blocker` demands four and a button
    offering a rung the next one refuses would be a trap.
    """

    def test_the_two_words_are_the_seed_sets_corpus_names(self) -> None:
        from chess_variant_sandbox import corpus
        from chess_variant_sandbox.web.server import _seed_set

        assert _seed_set("one") == corpus.ONE_SEED
        assert _seed_set("four") == corpus.LIBRARY_SEEDS

    def test_anything_else_means_whatever_the_rung_expects(self) -> None:
        """A browser cannot ask for a seed set the project does not use."""
        from chess_variant_sandbox.web.server import _seed_set

        for asked in (None, "", "three", "0,1000", 4, "FOUR"):
            assert _seed_set(asked) is None

    def test_what_was_asked_for_reaches_the_refresh(self, tmp_path, monkeypatch) -> None:
        from chess_variant_sandbox import corpus
        from chess_variant_sandbox.web.keeping import Measurements

        asked: list[object] = []

        class _Done:
            def as_dict(self) -> dict:
                return {"index": 0.5}

        def refresh(candidate, runs_dir=None, report=None, force=False, seeds=None):
            asked.append(seeds)
            return _Done()

        monkeypatch.setattr(corpus, "refresh", refresh)
        candidate = corpus.Candidate(
            hash="abc123def456", name="a draft", spec={}, path=tmp_path,
            stage="candidate",
        )

        measurements = Measurements(tmp_path)
        run = measurements.start(candidate, seeds=corpus.LIBRARY_SEEDS)
        for _ in range(2000):
            if run.state != "running":
                break
            time.sleep(0.002)

        assert run.state == "finished"
        assert asked == [corpus.LIBRARY_SEEDS]


# ------------------------------------------------ measuring on this computer (B5)

# The page's coordinator under Node, against the worker that ships and a stand-in Pyodide
# (`docs/archive/plan-client-compute.md` §8, B5). `run_request` waits as long as its request says,
# so units finish out of the order they were dealt in, and answers with the request; the
# gathering half, `gather_request`, is scripted, because nothing here runs Python. The real
# half is held natively by `TestTheGatheringHalf` and in a browser by the plan's record. A
# wheel whose name says `old` has no `gather_request`, as a wheel built before Q458 has not.
# The scripted gathering half's upload: a number past 2**53 in it, which a `JSON.parse`
# on the way would have rounded.
_HERE_UPLOAD = '{"schema":1,"phases":{"tournament":"1234567890123456789\\n"}}'

_HERE_PYODIDE = r"""
export async function loadPyodide(options) {
    const module = { HEAP8: new Int8Array(new ArrayBuffer(65536)) };
    let installed = "";
    return {
        version: "0.0.0-stand-in",
        _module: module,
        async loadPackage(wheel) { installed = wheel; },
        pyimport(name) {
            return {
                gather_request: installed.includes("wheel-old") ? undefined : scripted(name),
                run_request(text) {
                    const asked = JSON.parse(text);
                    if (asked.boom) throw new Error("SliceError: boom");
                    const until = Date.now() + asked.wait;
                    while (Date.now() < until) { /* a unit that takes a while */ }
                    return "reply to " + text + "\n";
                },
            };
        },
    };
}

function unit(phase, part, parts, wait, boom) {
    return { phase, part, name: `${phase}-${part + 1}-of-${parts}`, games: 2,
             plays: phase === "tournament",
             request: JSON.stringify(boom ? { phase, part, wait, boom } : { phase, part, wait }) };
}

function scripted(module) {
    const seen = [];
    return (text) => {
        const asked = JSON.parse(text);
        if (asked.state === null) {
            seen.push({ spec: asked.spec, module });
            return JSON.stringify({ state: "one", step: "tournament", result: null,
                units: [0, 1, 2, 3, 4].map((part) =>
                    unit("tournament", part, 5, (4 - part) * 25, !!asked.spec.boom && part === 3)) });
        }
        seen.push({ state: asked.state, replies: asked.replies });
        if (asked.state === "one") {
            return JSON.stringify({ state: "two", step: "reading", result: null, units: [
                unit("analysis", 0, 3, 40), unit("analysis", 1, 3, 0), unit("analysis", 2, 3, 15),
                unit("blunder", 0, 2, 25), unit("blunder", 1, 2, 0)] });
        }
        return JSON.stringify({ step: "done", units: [], result: { index: 0.5, hash: "abc", seen },
                                upload: '{"schema":1,"phases":{"tournament":"1234567890123456789\\n"}}' });
    };
}
"""

# What starts the page's workers under Node: a browser `Worker`'s face on `worker_threads`,
# the page's `MEASURE` run in this context, and one measurement a worker count.
_HERE_HOST = r"""
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { pathToFileURL } from "node:url";
import { Worker as NodeWorker } from "node:worker_threads";

const [shim, script, artefact, measure] = process.argv.slice(2);
let started = [];
globalThis.Worker = class {
    constructor(url) {
        this.url = url;
        this.sent = [];
        started.push(this);
        this.inner = new NodeWorker(pathToFileURL(shim),
            { workerData: { script: pathToFileURL(script).href } });
        this.inner.on("message", (data) => this.onmessage({ data }));
        this.inner.on("error", (error) => this.onerror({ message: error.message }));
    }
    postMessage(message) { this.sent.push(message); this.inner.postMessage(message); }
    terminate() { this.inner.terminate(); }
};
globalThis.document = { getElementById: () => null };
globalThis.location = { href: "http://127.0.0.1/" };
let wheel = "wheel-0123456789ab/x.whl";
let fetched = 0;
globalThis.fetch = async () => {
    fetched += 1;
    return { ok: true, status: 200, json: async () =>
        ({ schema: 1, runtime: "pyodide-0.0.0", wheel }) };
};
// The page's `api`, scripted: what the site answers, in order, and what it was asked.
let asked = [], answers = [];
globalThis.api = async (path, body) => {
    asked.push([path, body === undefined ? null : body]);
    const next = answers.shift();
    if (next instanceof Error) throw next;
    return next;
};
vm.runInThisContext("let DRAFT_HASH = 'abc', DRAFT_CHECK = null, DRAFT = null, DRAFT_STANDING = null;\n"
    + "let ACCOUNTS = false, VIEWER = null;\n"
    + readFileSync(measure, "utf8"));
vm.runInThisContext(`HERE_BASE = ${JSON.stringify(pathToFileURL(artefact).href + "/")};`);

const out = {};
for (const [label, spec, count, served] of [
        ["1", { name: "a draft" }, 1], ["3", { name: "a draft" }, 3],
        ["broken", { name: "a draft", boom: true }, 2],
        ["old", { name: "a draft" }, 2, "wheel-old/x.whl"]]) {
    started = [];
    wheel = served || "wheel-0123456789ab/x.whl";
    await vm.runInThisContext(`hereMeasure(${JSON.stringify(spec)}, ${count})`);
    const run = vm.runInThisContext("HERE");
    out[label] = {
        state: run.state, error: run.error, result: run.result, games: run.games,
        said: run.said, heaps: run.heaps, loads: run.loads.length,
        urls: started.map((worker) => worker.url),
        sent: started.map((worker) => worker.sent.map((message) => message.kind)),
        load: started.map((worker) => ({
            runtime: worker.sent[0].runtime, wheel: worker.sent[0].wheel,
            gathers: worker.sent[0].gathers === undefined ? null : worker.sent[0].gathers,
            keys: Object.keys(worker.sent[0]).sort() })),
        asked,
    };
}

// On the site, signed in (B10): the finished measurement is sent and followed.
wheel = "wheel-0123456789ab/x.whl";
vm.runInThisContext("ACCOUNTS = true; VIEWER = 'ada'; HERE_POLL = 0;"
    + " DRAFT_STANDING = { hash: 'abc', saved: true, publish: { mine: true } };");
const kept = { index: 0.81094, viable: true, state: "legal", sample: null, yours: true,
               seeds: [0], weights: "v11" };
for (const [label, script] of [
        ["kept", [{ job: 7, game: "abc", state: "queued" },
                  { job: 7, game: "abc", state: "running" },
                  { job: 7, game: "abc", state: "done", seconds: 12.5, result: kept }]],
        ["stale", [{ game: "abc", state: "refused",
                     refusal: { name: "stale-build", sentence: "its games were played by a core built from other sources" } }]],
        ["unsaved", [new Error("no game with the hash abc is saved on the site")]],
        // Q682: a fault in the page's own handler, and a worker's message, are never shown
        // as they are.
        ["fault", [new TypeError("Cannot read properties of undefined (reading 'job')")]],
        ["worker", [globalThis.workerError("PythonError: Traceback (most recent call last)")]]]) {
    started = [];
    asked = [];
    answers = script;
    const before = fetched;
    await vm.runInThisContext(`hereMeasure({ name: "a draft" }, 1)`);
    const run = vm.runInThisContext("HERE");
    out[label] = { state: run.state, error: run.error, check: run.check, asked, fetched: fetched - before,
                   upload: run.upload };
}

// Q1004: the standing says whether the site could keep it. Unsaved rules: the score is
// held and nothing is sent. Rules the person can never make theirs: nothing is sent. Rules
// changed while it ran: the standing of the game it measured is asked for.
for (const [label, standing, script] of [
        ["held", { hash: "abc", saved: false, publish: null }, []],
        ["library", { hash: "abc", saved: true, publish: null }, []],
        ["theirs", { hash: "abc", saved: true, publish: { mine: false } }, []],
        ["moved", { hash: "def", saved: true, publish: { mine: true } },
         [{ hash: "abc", saved: false, publish: null }]]]) {
    asked = [];
    answers = script;
    vm.runInThisContext(`DRAFT_STANDING = ${JSON.stringify(standing)};`);
    await vm.runInThisContext(`hereMeasure({ name: "a draft" }, 1)`);
    const run = vm.runInThisContext("HERE");
    out[label] = { state: run.state, check: run.check, asked };
}
// The held score once a save makes the game this person's: a standing of another game, or
// of the game still unsaved, sends nothing; the saved one sends the upload as it came, once.
vm.runInThisContext(`DRAFT_STANDING = { hash: "abc", saved: false, publish: null };`);
await vm.runInThisContext(`hereMeasure({ name: "a draft" }, 1)`);
asked = [];
answers = [{ job: 8, game: "abc", state: "done", seconds: 3, result: kept }];
const stood = [];
for (const standing of [{ hash: "def", saved: true, publish: { mine: true } },
                        { hash: "abc", saved: false, publish: null },
                        { hash: "abc", saved: true, publish: { mine: true } },
                        { hash: "abc", saved: true, publish: { mine: true } }]) {
    const sent = vm.runInThisContext(`hereStood(${JSON.stringify(standing)})`);
    stood.push(sent === null ? null : "sent");
    if (sent) await sent;
}
out.saved = { stood, asked, check: vm.runInThisContext("HERE.check") };
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def here_page() -> dict:
    """What `_HERE_HOST` saw, run once for the module's nine readers.

    A fixture rather than a memo (Q749): a real node and its worker threads, 3 s of wall
    clock, is a cost the nine share, and `tests/duration_guard.py` charges a shared
    fixture to no one test.
    """
    return _run_here_host()


def _run_here_host() -> dict:
    import subprocess
    import tempfile

    from page_text import MEASURE, PAGE

    found = shutil.which("node")
    if found is None:
        pytest.skip("node is not installed")
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        (root / "pyodide-0.0.0").mkdir()
        (root / "pyodide-0.0.0" / "pyodide.mjs").write_text(_HERE_PYODIDE, encoding="utf-8")
        # The page's one way of wording a failure (`core.js`), which the view leans on.
        start = PAGE.index("const PAGE_FAULT")
        words = PAGE[start:PAGE.index("\n}\n", PAGE.index("function errorWords", start)) + 3]
        (root / "measure.js").write_text(words + MEASURE, encoding="utf-8")
        (root / "host.mjs").write_text(_HERE_HOST, encoding="utf-8")
        done = subprocess.run(
            [
                found, str(root / "host.mjs"),
                str(repository / "tests" / "browser" / "node-worker.mjs"),
                str(repository / "chess_variant_sandbox" / "web" / "static" / "measure-worker.js"),
                str(root), str(root / "measure.js"),
            ],
            capture_output=True, timeout=120,
        )
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    out: dict = json.loads(done.stdout.decode("utf-8"))
    return out


class TestMeasuringOnThisComputer:
    """B5's view, `web/page/measure.py`'s `MEASURE`, in the designer.

    What a browser does with it is the plan's record (§8, B5): chess measured from the
    designer at one worker and at four. What is held here is what can be held without
    one -- where the view sits, when it offers anything, how many workers it starts, that
    it posts nothing and parses nothing but the gathering half's envelope, what the
    finished number looks like, and, under Node against the worker that ships, that every
    unit is dealt once and every reply lands in its unit's place whatever order the
    workers finish in.
    """

    def test_the_designer_carries_the_view(self) -> None:
        from page_text import DESIGN, MEASURE, PAGE

        assert MEASURE in DESIGN and PAGE.count(MEASURE) == 1
        # Q1004: every fresh standing is offered to a score held for want of a save.
        start = PAGE.index("function loadStanding(hash)")
        assert "hereStood(standing);" in PAGE[start:PAGE.index("function runScreen(", start)]
        assert PAGE.count('<div id="dHere">${hereHtml(check)}</div>') == 1
        start = PAGE.index("async function showDesigner(from)")
        assert "hereLookup();" in PAGE[start:PAGE.index("function benchMode(", start)]
        start = PAGE.index("function drawVerdict()")
        body = PAGE[start:PAGE.index("function footbarFixed()", start)]
        assert body.index("wireLadder();") < body.index("wireHere();")

    def test_it_posts_only_the_upload_and_parses_only_the_gathering_halfs_envelope(
        self,
    ) -> None:
        """It sends one thing, the gathering half's upload as the string it arrived as,
        to the site's own route (B10), and a record never passes through `JSON.parse`
        (§3)."""
        from page_text import MEASURE

        assert MEASURE.count("api(") == 3
        # Q1004: and, before sending, asks whether the game it measured is saved.
        assert "await api('/api/design/standing/' + hash)" in MEASURE
        assert "api('/api/provisional', { game: run.result.hash, upload: run.upload })" in MEASURE
        assert "api('/api/provisional/' + answer.job)" in MEASURE
        assert "run.upload = typeof answer.upload === 'string' ? answer.upload : '';" in MEASURE
        assert MEASURE.count("fetch(") == 1
        assert "fetch(hereUrl('manifest.json'), { cache: 'no-store' })" in MEASURE
        assert MEASURE.count("JSON.parse(") == 1 and "return JSON.parse(reply.text);" in MEASURE
        # The worker is B3's file, served by B4, and nothing else is started.
        assert MEASURE.count("new Worker(") == 1 and "HERE_WORKER = '/static/measure-worker.js'" in MEASURE

    def test_the_page_carries_no_python(self) -> None:
        """The gathering half is the wheel's `slices.gather_request` (Q458): a worker is told
        it gathers, and is sent no source to run."""
        from page_text import MEASURE

        assert "if (gathers) load.gathers = true;" in MEASURE
        for python in ("gatherer", "GATHERER", "from __future__", "import ", "def "):
            assert python not in MEASURE, python
        assert "</script" not in MEASURE

    def test_one_core_is_left_and_four_workers_is_the_most(self) -> None:
        shown = _in_the_page(
            "",
            "JSON.stringify([1, 2, 3, 4, 5, 8, 16, 0, undefined].map(n => {"
            " global.navigator = { hardwareConcurrency: n }; return hereWorkerCap(); }))",
        )
        assert json.loads(shown) == [1, 1, 2, 3, 4, 4, 4, 1, 1]

    def test_it_offers_nothing_without_a_manifest(self) -> None:
        """No artefact served, or none this page reads, is no offer at all -- not a button
        that fails. The same for a draft that is not a game, or one that never finishes."""
        shown = _in_the_page(
            "const good = { valid: true, playable: true };"
            " const said = [];"
            " HERE_MANIFEST = null; said.push(hereHtml(good));"
            " HERE_MANIFEST = { schema: 1, runtime: 'pyodide-0', wheel: 'wheel-0/x.whl' };"
            " said.push(hereHtml(good)); said.push(hereHtml({ valid: false }));"
            " said.push(hereHtml({ valid: true, playable: false })); said.push(hereHtml(null));",
            "JSON.stringify(said)",
        )
        absent, offered, broken, stuck, unchecked = json.loads(shown)
        assert absent == "" and broken == "" and stuck == "" and unchecked == ""
        assert 'id="dHereGo"' in offered and "measure it here" in offered
        assert 'id="dHereWorkers"' in offered

    def test_the_finished_number_is_four_decimals_under_its_label(self) -> None:
        setup = """
          DRAFT_CHECK = { hash: 'abc123def456', valid: true };
          HERE = { state: 'done', workers: 4, seconds: 125, heaps: [25165824, 30212096],
            times: [['loading', 3.4], ['tournament', 20], ['gathering', 1.5], ['gathering', 2]],
            result: { hash: 'abc123def456', name: 'a draft', index: 0.81094, viable: true,
              violations: [], weights: 'v11', games: 256,
              components: [{ key: 'skill', title: 'Skill decides it', score: 0.912,
                             what: 'How reliably the stronger player wins.' }] } };
          const said = [hereHtml(DRAFT_CHECK)];
          DRAFT_CHECK = { hash: 'fedcba654321', valid: true };
          said.push(hereHtml(DRAFT_CHECK));
        """
        now, moved = json.loads(_in_the_page(setup, "JSON.stringify(said)"))
        assert ">0.8109</div>" in now
        assert "measured on your computer" in now and "as one run of 256 games" in now
        assert "<b>Skill decides it</b> 0.91 &mdash; How reliably the stronger player wins." in now
        assert "Nothing has checked it." in now and "sent nowhere" in now
        # The steps in the order they run, and the gathers between them added up at the end.
        assert (
            "2 min 5 s in all, on four cores: loading the engine into this browser, 3.4 s;"
            " playing the tournament, 20.0 s;"
            " putting the pieces together, 3.5 s." in now
        )
        assert "at most 29 MiB" in now
        assert "the rules have changed" not in now and "the rules have changed" in moved

    def test_every_unit_is_dealt_once_and_every_reply_lands_in_its_place(self, here_page: dict) -> None:
        out = here_page
        one, three = out["1"], out["3"]
        for run, count in ((one, 1), (three, 3)):
            assert run["state"] == "done", run["error"]
            assert len(run["urls"]) == count == run["loads"] == len(run["heaps"])
            assert all(url == "/static/measure-worker.js" for url in run["urls"])
            assert all(heap >= 65536 for heap in run["heaps"])
            # Only tournament units play, two games each.
            assert run["games"] == 10
        # Units finished out of order at three workers, and the gathering half was handed
        # the same replies, in part order, as at one.
        assert one["result"] == three["result"]
        seen = one["result"]["seen"]
        assert seen[0]["spec"] == {"name": "a draft"}

        def replies(phase: str, waits: list[int]) -> list[str]:
            return [
                "reply to " + json.dumps(
                    {"phase": phase, "part": part, "wait": wait}, separators=(",", ":")
                ) + "\n"
                for part, wait in enumerate(waits)
            ]

        assert seen[1]["replies"] == {"tournament": replies("tournament", [100, 75, 50, 25, 0])}
        assert seen[2]["replies"] == {
            "analysis": replies("analysis", [40, 0, 15]),
            "blunder": replies("blunder", [25, 0]),
        }

    def test_one_worker_gathers_and_every_worker_is_loaded_from_the_manifest(self, here_page: dict) -> None:
        three = here_page["3"]
        gatherer, *players = three["load"]
        assert gatherer["gathers"] is True
        assert gatherer["keys"] == ["gathers", "kind", "runtime", "wheel"]
        # The step came from the wheel's own module, the one `run_request` comes from.
        seen = three["result"]["seen"][0]
        assert seen["module"] == "chess_variant_sandbox.tournament.slices"
        assert all(load["gathers"] is None for load in players)
        assert all(load["keys"] == ["kind", "runtime", "wheel"] for load in players)
        for load in three["load"]:
            assert load["runtime"].startswith("file:") and load["runtime"].endswith("/pyodide-0.0.0/")
            assert load["wheel"].endswith("/wheel-0123456789ab/x.whl")
        first, *others = three["sent"]
        assert first[0] == "load" and first.count("gather") == 3
        assert all(sent[0] == "load" and "gather" not in sent for sent in others)
        assert {kind for sent in three["sent"] for kind in sent} == {"load", "gather", "unit"}

    def test_off_the_site_nothing_is_sent(self, here_page: dict) -> None:
        out = here_page
        for label in ("1", "3", "broken", "old"):
            assert out[label]["asked"] == [], label
        # The upload was handed over all the same, and kept as the string it came as.
        assert out["kept"]["upload"] == _HERE_UPLOAD

    def test_on_the_site_the_upload_goes_as_it_came_and_the_job_is_followed(self, here_page: dict) -> None:
        """B10: the gathering half's upload, never parsed, to the site's route; then the
        job asked about until the server has walked it, and its answer kept whole."""
        kept = here_page["kept"]
        assert kept["state"] == "done"
        assert kept["asked"] == [
            ["/api/provisional", {"game": "abc", "upload": _HERE_UPLOAD}],
            ["/api/provisional/7", None],
            ["/api/provisional/7", None],
        ]
        assert kept["check"]["state"] == "done" and kept["check"]["seconds"] == 12.5
        assert kept["check"]["result"]["index"] == 0.81094
        assert kept["fetched"] == 1

    def test_a_stale_build_fetches_the_manifest_again(self, here_page: dict) -> None:
        stale = here_page["stale"]
        assert stale["check"]["refusal"]["name"] == "stale-build"
        assert len(stale["asked"]) == 1
        # Once for the measurement, and once more because the site refused its engine.
        assert stale["fetched"] == 2

    def test_a_send_that_fails_keeps_the_sentence(self, here_page: dict) -> None:
        unsaved = here_page["unsaved"]
        assert unsaved["check"] == {
            "state": "failed", "error": "no game with the hash abc is saved on the site",
        }

    def test_a_send_that_faults_says_so_in_a_sentence(self, here_page: dict) -> None:
        """Q682: a `TypeError` in the page's own handler, or a worker's message, reaches
        the view as a plain clause, never as the engine's or Python's words."""
        fault, worker = here_page["fault"], here_page["worker"]
        assert fault["check"] == {
            "state": "failed",
            "error": "something went wrong on this page; reloading it usually puts it right"}
        assert worker["check"]["state"] == "failed"
        assert worker["check"]["error"].startswith("measuring in this browser ran into a fault")
        for said in (fault["check"]["error"], worker["check"]["error"]):
            assert "Cannot read" not in said and "Traceback" not in said, said

    def test_the_servers_number_and_label_replace_the_browsers(self) -> None:
        setup = """
          ACCOUNTS = true; VIEWER = 'ada';
          DRAFT_CHECK = { hash: 'abc123def456', valid: true };
          const result = { hash: 'abc123def456', name: 'a draft', index: 0.75, viable: true,
            violations: [], weights: 'v11', games: 256,
            components: [{ key: 'skill', title: 'Skill decides it', score: 0.912, what: '' }] };
          const said = {};
          const base = { state: 'done', workers: 1, seconds: 60, heaps: [], times: [],
                         result, upload: 'u' };
          for (const [label, check] of Object.entries({
              none: null,
              queued: { state: 'queued', job: 7 },
              kept: { state: 'done', job: 7, seconds: 12.5, result: { index: 0.81094,
                viable: true, state: 'legal', sample: null, yours: true, seeds: [0],
                weights: 'v11' } },
              stale: { state: 'refused', refusal: { name: 'stale-build', sentence: 'x' } },
              refused: { state: 'refused', job: 7,
                         refusal: { name: 'wrong-outcome', sentence: 'game 2 ends otherwise' } },
              gone: { state: 'done', job: 7, result: null, superseded: 'superseded' } })) {
            HERE = { ...base, check };
            said[label] = hereHtml(DRAFT_CHECK);
          }
          ACCOUNTS = false; VIEWER = null; HERE = { ...base, check: null };
          said.local = hereHtml(DRAFT_CHECK);
        """
        said = json.loads(_in_the_page(setup, "JSON.stringify(said)"))
        mine = ">0.7500</div>"
        assert mine in said["none"] and "go to the site" not in said["none"]
        assert mine in said["queued"] and "The site is about to check it" in said["queued"]
        kept = said["kept"]
        assert ">0.8109</div>" in kept and mine not in kept
        assert "measured on your computer, not yet confirmed" in kept
        assert "Checked by the site in 12.5 s." in kept and "Only you can see this number." in kept
        assert "as your browser scored it" in kept
        assert "newer engine" in said["stale"] and 'id="dHereSend"' not in said["stale"]
        assert "game 2 ends otherwise" in said["refused"] and 'id="dHereSend"' in said["refused"]
        assert "a newer score for this game replaced it" in said["gone"]
        assert "sent nowhere" in said["local"] and "Sign in" not in said["local"]

    def test_an_unsaved_game_holds_its_score_and_a_save_sends_it(self, here_page: dict) -> None:
        """Q1004, the user: a measurement of a game nobody saved was sent, refused, and
        lost. Now nothing is sent while the game is unsaved; the score is held on the page,
        and once a save's standing says the game is this person's, the upload goes as it
        came -- nothing measured again -- and only once."""
        held = here_page["held"]
        assert held["state"] == "done" and held["asked"] == []
        assert held["check"] == {"state": "unsaved"}
        saved = here_page["saved"]
        assert saved["stood"] == [None, None, "sent", None]
        assert saved["asked"] == [["/api/provisional", {"game": "abc", "upload": _HERE_UPLOAD}]]
        assert saved["check"]["state"] == "done" and saved["check"]["result"]["index"] == 0.81094

    def test_a_game_no_save_can_make_yours_sends_nothing(self, here_page: dict) -> None:
        for label, words in (("library", "already a library game"),
                             ("theirs", "Somebody else saved these rules first")):
            out = here_page[label]
            assert out["asked"] == [], label
            assert out["check"]["state"] == "unkept" and words in out["check"]["why"], label

    def test_rules_changed_while_it_ran_are_asked_about_by_their_own_hash(self, here_page: dict) -> None:
        moved = here_page["moved"]
        assert moved["asked"] == [["/api/design/standing/abc", None]]
        assert moved["check"] == {"state": "unsaved"}

    def test_measuring_here_is_off_up_front_where_nothing_could_be_kept(self) -> None:
        """Q1004: the button is drawn and does not work, with what turns it on, for a
        visitor and for rules no save of this person's can make theirs; it works for their
        own saved game and for rules nobody saved, which say the save keeps the score."""
        setup = """
          HERE_MANIFEST = { schema: 1, runtime: 'pyodide-0', wheel: 'wheel-0/x.whl' };
          const check = { hash: 'abc', valid: true, playable: true };
          const said = {};
          ACCOUNTS = true; VIEWER = null; DRAFT_STANDING = null;
          said.visitor = hereHtml(check);
          VIEWER = 'ada';
          for (const [label, standing] of Object.entries({
              unknown: null,
              unsaved: { hash: 'abc', saved: false, publish: null },
              mine: { hash: 'abc', saved: true, publish: { mine: true } },
              library: { hash: 'abc', saved: true, publish: null },
              theirs: { hash: 'abc', saved: true, publish: { mine: false } } })) {
            DRAFT_STANDING = standing;
            said[label] = hereHtml(check);
          }
          ACCOUNTS = false; VIEWER = null; DRAFT_STANDING = null;
          said.local = hereHtml(check);
        """
        said = json.loads(_in_the_page(setup, "JSON.stringify(said)"))
        off = {"visitor": "Sign in and save this game",
               "library": "already a library game", "theirs": "Somebody else saved"}
        for label, words in off.items():
            drawn = " ".join(said[label].split())
            assert '<button id="dHereGo" disabled' in drawn, label
            assert 'id="dHereOff"' in drawn and words in drawn, label
            assert "you can measure" in drawn, label
            assert 'id="dHereWorkers"' not in drawn, label
        for label in ("unknown", "unsaved", "mine", "local"):
            drawn = " ".join(said[label].split())
            assert 'id="dHereGo" class="primary">' in drawn and "disabled" not in drawn, label
            assert 'id="dHereOff"' not in drawn, label
        assert "This game is not saved yet: when the measurement ends, save it" in " ".join(
            said["unsaved"].split())
        assert "not saved yet" not in said["mine"] and "not saved yet" not in said["local"]

    def test_a_held_score_offers_the_save_while_the_rules_are_the_ones_it_measured(self) -> None:
        setup = """
          ACCOUNTS = true; VIEWER = 'ada';
          DRAFT_HASH = 'abc'; DRAFT_CHECK = { hash: 'abc', valid: true };
          HERE = { state: 'done', workers: 1, seconds: 60, heaps: [], times: [], upload: 'u',
            check: { state: 'unsaved' },
            result: { hash: 'abc', name: 'a draft', index: 0.75, viable: true,
                      violations: [], weights: 'v11', games: 256, components: [] } };
          const said = { here: hereHtml(DRAFT_CHECK) };
          DRAFT_HASH = 'def'; DRAFT_CHECK = { hash: 'def', valid: true };
          said.moved = hereHtml(DRAFT_CHECK);
          HERE.check = { state: 'unkept', why: HERE_OFF.theirs };
          said.unkept = hereHtml(DRAFT_CHECK);
        """
        said = {k: " ".join(v.split()) for k, v in
                json.loads(_in_the_page(setup, "JSON.stringify(said)")).items()}
        here, moved, unkept = said["here"], said["moved"], said["unkept"]
        assert ">0.7500</div>" in here and "measured on your computer" in here
        assert 'id="dHereSave"' in here and "save it and keep the score" in here
        assert "nothing is measured again" in here
        assert 'id="dHereSave"' not in moved and "put the rules back as they were" in moved
        assert 'id="dHereSave"' not in unkept and "It was not sent: Somebody else" in unkept
        assert 'id="dHereSend"' not in here + moved + unkept

    def test_a_unit_that_fails_ends_the_measurement_with_its_reason(self, here_page: dict) -> None:
        """The run keeps the worker's own message for whoever maintains the site; the
        visitor reads a plain clause instead (Q728)."""
        broken = here_page["broken"]
        assert broken["state"] == "failed"
        assert "boom" in broken["error"]
        assert broken["result"] is None
        assert broken["said"].startswith("measuring in this browser ran into a fault")
        assert "boom" not in broken["said"]
        setup = """HERE = { state: 'failed', error: 'PythonError: boom', said: 'a plain clause',
                         started: 0 };"""
        shown = json.loads(_in_the_page(setup, "JSON.stringify(hereRunHtml(HERE))"))
        assert "It stopped before it finished: a plain clause" in shown and "boom" not in shown

    def test_a_wheel_that_cannot_gather_is_refused_at_load_by_name(self, here_page: dict) -> None:
        """A wheel built before Q458 has no `gather_request`. The measurement stops before
        any unit is dealt, and keeps what is wrong and what rebuilds it for the maintainer."""
        old = here_page["old"]
        assert old["state"] == "failed" and old["result"] is None
        assert "no slices.gather_request" in old["error"] and "build.sh" in old["error"]
        # What is wrong and what rebuilds it is the maintainer's; a visitor sees neither.
        assert "gather_request" not in old["said"] and "build.sh" not in old["said"]
        assert "ran into a fault" in old["said"]
        assert old["games"] == 0
        assert all(sent == ["load"] for sent in old["sent"])


@pytest.fixture(scope="module")
def dull() -> dict:
    return load_spec(RULESETS / "dull.json")


class TestTheGatheringHalf:
    """`slices.gather_request`, natively: what the one worker that gathers runs.

    A step is text in and text out, so it is held here on the native core. It is the
    wheel's own code, so what the suite runs natively is what the browser runs (Q458). The
    whole measurement through it, against `slices.measure_in_slices`, is the slow test
    below.
    """

    @staticmethod
    def _first(spec: dict) -> dict:
        from chess_variant_sandbox.tournament import slices

        answer: dict = json.loads(slices.gather_request(json.dumps({"spec": spec, "state": None})))
        return answer

    def test_the_first_step_deals_the_tournament_in_the_fixed_cut(self, dull) -> None:
        from chess_variant_sandbox.tournament import runner, slices

        answer = self._first(dull)
        assert answer["step"] == slices.TOURNAMENT and answer["result"] is None
        assert isinstance(answer["state"], str)
        units = answer["units"]
        assert len(units) == slices.CUT
        assert [unit["part"] for unit in units] == list(range(slices.CUT))
        assert all(unit["phase"] == slices.TOURNAMENT and unit["plays"] for unit in units)
        covered: list[int] = []
        for shown in units:
            request = json.loads(shown["request"])
            assert "seed" not in request["unit"]
            unit = slices.Unit.from_dict(request["unit"])
            assert unit.name == shown["name"] and len(unit.games) == shown["games"]
            covered.extend(unit.games)
        assert covered == list(range(len(runner.schedule(slices.canonical_config()))))

    def test_the_draft_is_settled_before_anything_is_dealt(self, dull) -> None:
        """The number is for the game the designer's check hashed (`designing.settled`)."""
        from chess_variant_sandbox.spec.schema import SCHEMA_VERSION
        from chess_variant_sandbox.web.designing import settled

        draft = json.loads(json.dumps(dull))
        draft["schema_version"] = 1
        # Tidied into the current spelling, which writes the current stamp (Q1008c).
        assert settled(json.loads(json.dumps(draft)))["schema_version"] == SCHEMA_VERSION
        answer = self._first(draft)
        dealt = json.loads(answer["units"][0]["request"])["spec"]
        assert dealt["schema_version"] == settled(draft)["schema_version"]
        assert spec_hash(dealt) == spec_hash(settled(draft)) == spec_hash(dull)

    def test_a_missing_reply_is_refused_by_name(self, dull) -> None:
        from chess_variant_sandbox.tournament import slices

        state = self._first(dull)["state"]
        short = {"tournament": ["x"] * (slices.CUT - 1)}
        with pytest.raises(slices.SliceError, match="missing tournament-16-of-16"):
            slices.gather_request(json.dumps({"state": state, "replies": short}))
        holed: list[str | None] = [""] * slices.CUT
        holed[2] = None
        with pytest.raises(slices.SliceError, match="missing tournament-3-of-16"):
            slices.gather_request(
                json.dumps({"state": state, "replies": {"tournament": holed}})
            )

    def test_a_step_it_does_not_know_is_refused(self) -> None:
        from chess_variant_sandbox.tournament import slices

        state = json.dumps({"spec": {}, "cut": 16, "step": "later", "dealt": {}})
        with pytest.raises(slices.SliceError, match="no step after 'later'"):
            slices.gather_request(json.dumps({"state": state, "replies": {}}))

    def test_the_page_module_holds_no_copy_of_the_gather(self) -> None:
        """One copy of the gathering half, in the wheel. `measure.py` keeps only `drive`,
        the native stand-in for the page, and that goes through `gather_request` too."""
        import inspect

        from chess_variant_sandbox.web.page import measure

        for gone in ("GATHERER", "gather", "_advance", "_deal", "_joined", "_result", "CUT"):
            assert not hasattr(measure, gone), gone
        assert "slices.gather_request(" in inspect.getsource(measure.drive)


@pytest.mark.slow
def test_the_gathering_half_measures_what_measure_in_slices_does(tmp_path) -> None:
    """A whole measurement through `slices.gather_request`, as the page drives it, against
    B0's reference on the same spec. Two steps run: the tournament, and the reading step
    with the score."""
    from chess_variant_sandbox import corpus
    from chess_variant_sandbox.tournament import slices

    spec = load_spec(RULESETS / "dull.json")
    answer = json.loads(slices.gather_request(json.dumps({"spec": spec, "state": None})))
    steps = [answer["step"]]
    while answer["result"] is None:
        replies: dict[str, list[str]] = {}
        for unit in answer["units"]:
            replies.setdefault(unit["phase"], []).append(slices.run_request(unit["request"]))
        answer = json.loads(
            slices.gather_request(json.dumps({"state": answer["state"], "replies": replies}))
        )
        steps.append(answer["step"])
    assert steps == ["tournament", "reading", "done"]

    driven = answer["result"]
    reference = slices.measure_in_slices(spec, tmp_path, slices=4)
    stored = json.loads(
        (tmp_path / f"index-{reference.result.weights_version}.json").read_text(encoding="utf-8")
    )
    assert driven["index"] == reference.index
    assert {part["key"]: part["score"] for part in driven["components"]} == stored["scores"]
    assert "fit" not in driven
    assert driven["games"] == len(reference.records)
    current = corpus.current_basis(spec_hash(spec), spec=spec, seeds=corpus.ONE_SEED)
    assert driven["basis"] == current.as_dict()

    # B10: the upload the last step wrote is tier one's slim form, stamped as played in a
    # browser, and the server's own walk and score of it give the page's number.
    from chess_variant_sandbox.measurement import BROWSER, Basis
    from chess_variant_sandbox.site import legality

    sent = legality.read_upload(answer["upload"])
    stamp = Basis.from_dict(dict(sent.basis or {}))
    assert stamp is not None and stamp.played == BROWSER and not stamp.checked
    assert stamp == Basis.from_dict(driven["basis"]).in_browser()  # type: ignore[union-attr]
    scored = legality.score(legality.check(spec, sent, current), current.weights_version)
    assert scored.headline == driven["index"]


class TestAHexagonReachesEveryBoardApartFromItsWalls:
    """Q281b: the play board, the designer's preview and the page itself draw a
    `board.shape: "hexagon"` game as one -- the corners travel as `absent`, not as
    walls, and every board loop skips them. The drawing is `tests/test_viz.py`'s."""

    NAMES = ("hex-glinski-hexagon", "hex-glinski-flanks", "hex-glinski-flanks-one-way")

    @staticmethod
    def _spec(name: str) -> dict:
        from retired_specs import REBUILT

        if name in REBUILT:  # retired by Q870 for `pinwheel`; rebuilt from the hexagon
            return REBUILT[name]()
        path = Path(__file__).resolve().parents[1] / "archive" / f"{name}.json"
        spec: dict = json.loads(path.read_text(encoding="utf-8"))
        return spec

    @pytest.mark.parametrize("name", NAMES)
    def test_the_play_board_and_the_preview_send_the_corners_apart(self, name: str) -> None:
        from chess_variant_sandbox.web import designing

        spec = self._spec(name)
        game = library.Game(hash=spec_hash(spec), name=name, description="",
                            source="archive", path=Path(f"{name}.json"), spec=spec)
        snapshot = Sessions().start(game, ("human", "human")).snapshot()
        preview = designing.check(spec)["preview"]
        for sent in (snapshot, preview):
            assert (sent["obstacles"], len(sent["absent"])) == ([], 30)
        joined = name != "hex-glinski-hexagon"
        for wrap in (snapshot["ruleset"]["wrap"], preview["wrap"]):
            assert (wrap is not None and len(wrap["copies"]) == 6) is joined

    def test_a_square_board_sends_no_absent_cells(self, chess_game) -> None:
        snapshot = Sessions().start(chess_game, ("human", "human")).snapshot()
        assert (snapshot["obstacles"], snapshot["absent"]) == ([], [])
        assert snapshot["ruleset"]["wrap"] is None

    def test_every_board_loop_skips_an_absent_cell(self) -> None:
        from page_text import PAGE

        assert "boardCut(SESSION.absent, rules.wrap)" in PAGE
        assert "boardCut(preview.absent, preview.wrap)" in PAGE
        assert PAGE.count("if (cellAbsent(square, FRAME)) continue;") == 1
        # The still board's loop, and the names set outside the board (`edgeLabels`, V4).
        assert PAGE.count("if (cellAbsent(square, view)) continue;") == 2
        # The designer's ground-finder treats a corner as no ground at all.
        assert "(preview.obstacles || []).concat(preview.absent || [])" in PAGE


def test_leaving_the_map_before_its_first_answer_draws_nothing() -> None:
    """Q281c2, seen in a browser: open the map on a cold server, go to a game before
    `/api/graph` answers, and the late answer wrote the map's tagline over the game's
    header and raised *Cannot set properties of null* into the game page. `showExplorer`
    now stops after each await once its own body is no longer on the page.
    """
    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      const __nodes = {};
      const __node = () => new Proxy({ style: {}, classList: { add() {}, remove() {},
        toggle() {} } }, { get: (t, k) => (k in t ? t[k] : () => []),
                          set: (t, k, v) => { t[k] = v; return true; } });
      let __gone = false;
      global.document = {
        getElementById: id => (id === 'exploreBody' && __gone ? null
          : (__nodes[id] || (__nodes[id] = __node()))),
        querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {},
        createElement: () => __node(), body: __node(), head: __node() };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, scrollTo: () => {}, innerWidth: 1200 };
      global.location = { hash: '#/map' };
      global.fetch = () => new Promise(() => {});
    """
    runner = """
      let __answer;
      api = path => (path === '/api/graph'
        ? new Promise(done => { __answer = done; }) : Promise.resolve(null));
      document.getElementById('tagline').textContent = 'the game page';
      showExplorer().then(() => {
        console.log(JSON.stringify(document.getElementById('tagline').textContent));
      }, error => console.log('threw ' + error.message));
      __gone = true;
      __answer({ nodes: [], similar: [], communities: [], levels: [] });
    """
    done = node_session.run(stub + script.group(1) + "\n" + runner)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == '"the game page"'


def test_a_view_that_writes_no_summary_shows_the_served_tagline() -> None:
    """Q281c2, seen in a browser: after the map, a game's page and the designer still
    said *71 games, 93 pairs alike enough for a line* under the title, because only the
    library, the map and what's new write the header's second line. `render` puts back
    the line the page was served with before any view draws.
    """
    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      const __nodes = {};
      const __node = () => new Proxy({ style: {}, textContent: 'as served',
        classList: { add() {}, remove() {}, toggle() {} } },
        { get: (t, k) => (k in t ? t[k] : () => []),
          set: (t, k, v) => { t[k] = v; return true; } });
      global.document = {
        getElementById: id => (__nodes[id] || (__nodes[id] = __node())),
        querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {},
        createElement: () => __node(), body: __node(), head: __node() };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, scrollTo: () => {}, innerWidth: 1200 };
      global.location = { pathname: '/game', hash: '' };
      global.fetch = () => new Promise(() => {});
    """
    runner = """
      api = () => new Promise(() => {});
      document.getElementById('tagline').textContent = '71 games, 93 pairs alike';
      location.hash = '#/game/abc123def456';
      render();
      console.log(JSON.stringify(document.getElementById('tagline').textContent));
    """
    done = node_session.run(stub + script.group(1) + "\n" + runner)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == '"as served"'


# The board's sight, stubbed: an IntersectionObserver whose callback the test calls, and
# the page's one `visibilitychange` listener, caught so the test can fire it.
_SIGHT = """
  let sight = null, heard = {};
  global.IntersectionObserver = function (callback) {
    sight = callback; this.observe = () => {}; this.disconnect = () => {};
  };
  document.addEventListener = (kind, listener) => { heard[kind] = listener; };
  watchBoardSight({ querySelector: () => ({}) });
"""


class TestAWatchNobodySeesWaits:
    """Q541: a watch in the designer's bench asked the engine 14 times in 8 s with its
    board scrolled off screen. The chain waits while the board is out of the viewport or
    the tab is in the background, and starts again through `maybeAgent` when it is back."""

    def test_a_board_out_of_sight_asks_nothing(self) -> None:
        replies = _agent_replies("", _SIGHT + "sight([{ isIntersecting: false }]);")
        assert replies["asked"] == 0

    def test_a_board_back_in_sight_asks_again(self) -> None:
        replies = _agent_replies("sight([{ isIntersecting: true }]);",
                                 _SIGHT + "sight([{ isIntersecting: false }]);")
        assert replies == {"applied": ["first"], "asked": 1}

    def test_a_tab_in_the_background_asks_nothing_until_it_is_back(self) -> None:
        hidden = _SIGHT + "document.visibilityState = 'hidden';"
        assert _agent_replies("", hidden)["asked"] == 0
        back = _agent_replies(
            "document.visibilityState = 'visible'; heard.visibilitychange();", hidden)
        assert back == {"applied": ["first"], "asked": 1}

    def test_coming_back_while_an_answer_is_on_its_way_asks_once(self) -> None:
        replies = _agent_replies(
            "sight([{ isIntersecting: false }]); sight([{ isIntersecting: true }]);",
            _SIGHT,
        )
        assert replies["asked"] == 1

    def test_the_person_s_own_pause_is_left_alone(self) -> None:
        """Out of sight is not a pause: the pause button's word is not changed by it, and
        a game the person paused stays paused when the board comes back."""
        replies = _agent_replies("sight([{ isIntersecting: true }]);",
                                 _SIGHT + "PAUSED = true; sight([{ isIntersecting: false }]);")
        assert replies["asked"] == 0
