"""The behaviour grid, checked against the games we already have.

The descriptors decide what competes with what, so a grid that cannot tell two real
games apart quietly merges them and the search keeps one where it should keep both.
Validating it against the committed corpus is cheap, has a real chance of failing, and
did fail on the first four axes -- which is why there is a fifth.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import pytest

from chess_variant_sandbox.loop import grid
from chess_variant_sandbox.loop.archive import Archive, Entry
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.transform import clear_facing, strip_rule

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
CORPUS = sorted(path.stem for path in RULESETS.glob("*.json"))


@pytest.fixture(scope="module")
def cells() -> dict[str, tuple[int, ...]]:
    return {
        name: grid.descriptor(load_spec(RULESETS / f"{name}.json")) for name in CORPUS
    }


class TestDescriptors:
    def test_the_three_great_traditions_are_three_different_games(self, cells) -> None:
        """The check the plan named, and the cheapest thing that could have gone wrong."""
        assert len({cells["chess"], cells["shogi"], cells["xiangqi"]}) == 3

    def test_the_corpus_spreads_across_the_grid(self, cells) -> None:
        counts = Counter(cells.values())
        assert len(counts) >= 10, f"{len(CORPUS)} rulesets in {len(counts)} cells"
        # The first four axes put eight of the twenty-two -- the whole 8x8, six-piece,
        # mate-ending neighbourhood -- into one cell, which is most of the region a
        # search over chess-like games explores. The fifth axis took that to five, and
        # five was the ceiling while the corpus held twenty-two.
        #
        # **A count, not a share, and the history is what settles it.** A cell holds one
        # elite, so six games in a cell means five discarded whatever the corpus size --
        # and the two states anyone has actually judged do not even order by share: five
        # of twenty-two was accepted when the fifth axis went in (23%), and seven of
        # *thirty-three* forced the seventh (21%), a smaller share of a bigger corpus.
        # The corpus doubling to forty-two is not licence to let a cell grow with it.
        #
        # Six, because it is the largest ceiling that still fires on those wrapped
        # boards -- the last time the grid was genuinely blind rather than merely full.
        # `chess-oriented-pawns` took the chess cell from five to six on its own, and
        # one game arriving is not the event a new axis answers; every axis here was
        # added when a *batch* sharing an unseen feature collapsed into one cell.
        #
        # **It left again on 2026-09-07** (Q98), when the phantom `en_passant` came off
        # it: two special rules rather than three puts it in the `1-1-0-0-0-...` cell
        # beside `bastion`, `makruk`, `sardines` and `shatranj`, so the chess cell is
        # five again and the worst cell in the corpus is five. The ceiling is untouched
        # by that -- a game moving between two cells is not evidence about either.
        #
        # So when this fails, the question is which of the two it is. If the answer is a
        # facing rule, `chess_variant_sandbox/loop/grid.py`'s module docstring has already made that
        # call and says what to do.
        worst, crowd = counts.most_common(1)[0]
        assert crowd <= 6, (
            f"{crowd} of {len(CORPUS)} rulesets share cell {grid.label(worst)}: "
            f"{sorted(name for name, cell in cells.items() if cell == worst)}"
        )

    def test_a_cell_is_known_from_the_spec_alone(self, cells) -> None:
        """No games are played to find out where a candidate belongs, which is what makes
        a cell something a proposal can be *aimed* at."""
        spec = load_spec(RULESETS / "chess.json")
        assert grid.descriptor(spec) == cells["chess"]
        assert grid.label(cells["chess"]) == "1-1-0-0-1-0-0-0"
        assert grid.parse(grid.label(cells["chess"])) == cells["chess"]

    def test_every_cell_can_be_described_in_english(self, cells) -> None:
        """The description is what a model is given, so it has to read as a brief."""
        for name in ("chess", "shogi", "dobutsu", "racingkings"):
            text = grid.describe_cell(cells[name])
            assert "board of" in text
            assert "won by" in text
            assert "_" not in text and "cell" not in text

    def test_a_drop_ruleset_is_filed_apart_from_one_without(self, cells) -> None:
        assert cells["crazyhouse"] != cells["chess"]
        assert cells["shogi"][2] == 1 and cells["chess"][2] == 0

    def test_a_restricted_capture_rule_is_a_different_kind_of_game(self, cells) -> None:
        """The axis schema 3 added, and the only one that separates a *kind* of game
        rather than a size of one.

        Every ruleset written before the capture matrix sits in bin 0, so the axis
        doubled the grid without moving anything already in it.
        """
        assert cells["rps-v1"][5] == 1
        assert cells["intransitive"][5] == 1
        assert all(
            cells[name][5] == 0
            for name in CORPUS
            if not name.startswith("rps-") and name != "intransitive"
        )
        # ...and it is not merely the board or the army doing the work: hierarchy chess
        # is chess's board and chess's army, filed apart from chess.
        assert cells["rps-hierarchy-chess"] != cells["chess"]

    def test_a_facing_rule_is_deliberately_not_yet_a_kind_of_game(self, cells) -> None:
        """The one merge in this grid that is on purpose, asserted so it is not quiet.

        Schema 8's `piece.facing` has the argument for a ninth axis and not the
        operational test -- two facing specs exist and each is filed with the game it
        was made from. `grid`'s module docstring holds the reasoning and the trigger:
        when a second *committed* facing ruleset lands, add the axis and delete this.

        **Asserted by taking the facing off, not by comparing two games** (Q98). This
        used to read `cells["chess-oriented-pawns"] == cells["chess"]`, which stopped
        being true when that ruleset lost a phantom `en_passant` in the change that
        began refusing one: it now carries two special rules against chess's three and
        is filed apart -- for the rule it no longer has, and not for the facing. The
        claim under test is that *turning a piece* moves nothing, so turning a piece is
        what the test moves, and the second assertion pins the alternative explanation
        shut by putting chess itself in the same cell once its en passant goes.
        """
        turned = load_spec(RULESETS / "chess-oriented-pawns.json")
        plain = clear_facing(copy.deepcopy(turned), "P")
        assert grid.descriptor(plain) == grid.descriptor(turned)
        assert grid.descriptor(turned) == cells["chess-oriented-pawns"]
        # It is the missing rule that files it apart from chess, not the orientation:
        # the `rules` axis counts `spec["rules"]`, and a facing is a piece property
        # counted apart in `complexity.facing_pieces`.
        assert (
            grid.descriptor(strip_rule(copy.deepcopy(load_spec(RULESETS / "chess.json")),
                                       "en_passant"))
            == cells["chess-oriented-pawns"]
        )
        assert cells["chess-oriented-pawns"][4] == 0 and cells["chess"][4] == 1

    @pytest.mark.parametrize(
        ("key", "dead", "live"),
        [
            (
                "strike",
                {"effect": "promote", "position": "stay", "immune": ["K"],
                 "chain": [{"from": "K", "to": "Q"}]},
                {"effect": "promote", "position": "stay", "immune": ["K"],
                 "chain": [{"from": "P", "to": "N"}]},
            ),
            (
                "contacts",
                {"position": "stay", "immune": ["K"],
                 "links": [{"mover": "P", "onto": "K", "mover_to": "Q"},
                           {"mover": "N", "onto": "K", "onto_to": "Q"}]},
                {"position": "stay", "immune": ["K"],
                 "links": [{"mover": "P", "onto": "P", "mover_to": "Q"}]},
            ),
        ],
    )
    def test_a_capture_rule_the_compiler_drops_is_filed_with_chess(
        self, cells, key: str, dead: dict, live: dict
    ) -> None:
        """Q417: *transforms* is what the compiler keeps, in either spelling.

        Every link of each dead rule is about taking the immune king, which neither
        spelling ever reaches, so the rule compiles to chess's own tables and is filed
        with chess. The grid used to read any non-``capture`` chain and any table with a
        link as transforming. The live neighbour is one axis over, and the live table
        changes only the mover, which is still a rule though nothing survives.
        """
        from chess_variant_sandbox.engine.compile import compile_spec
        from chess_variant_sandbox.spec import validate_spec

        taking = [axis.name for axis in grid.AXES].index("taking")
        for rule, kept in ((dead, False), (live, True)):
            spec = load_spec(RULESETS / "chess.json")
            spec["schema_version"] = 10
            spec["rules"][key] = copy.deepcopy(rule)
            if kept:
                assert validate_spec(spec) == []
                assert compile_spec(spec).strike is not None
            else:
                # Refused since Q538 as stating nothing, and still filed with chess by a
                # reader that meets it before the validator does.
                from chess_variant_sandbox.spec.schema import PLAYED_AS_CAPTURE

                (refusal,) = validate_spec(spec)
                assert PLAYED_AS_CAPTURE in refusal, refusal
            cell = list(cells["chess"])
            cell[taking] = 1 if kept else 0
            assert grid.descriptor(spec) == tuple(cell), (key, kept)

    def test_a_game_is_filed_under_the_ending_that_distinguishes_it(self, cells) -> None:
        """King of the hill has both a reach condition and checkmate, and what makes it
        king of the hill is the arrival."""
        assert cells["kingofthehill"][3] == 2
        assert cells["chess"][3] == 0
        assert cells["threecheck"][3] == 3


class TestGrid:
    def _entry(self, digest: str, cell: str, index: float, *, viable: bool = True) -> Entry:
        return Entry(
            hash=digest, name=digest, generation=0, parent=None, status="evaluated",
            index=index, objective=index, viable=viable, cell=cell,
            index_ci_low=index - 0.2, index_ci_high=index + 0.2,
        )

    def test_one_elite_per_cell(self) -> None:
        built = grid.build([
            self._entry("a", "1-1-0-0-1-0-0", 0.50),
            self._entry("b", "1-1-0-0-1-0-0", 0.62),
            self._entry("c", "0-1-0-0-0-0-0", 0.31),
        ])
        assert built.filled == 2
        assert built.elite(grid.parse("1-1-0-0-1-0-0")).hash == "b"
        assert built.elite(grid.parse("0-1-0-0-0-0-0")).hash == "c"

    def test_a_gated_candidate_never_holds_a_cell(self) -> None:
        """A non-viable ruleset is not a low-scoring one -- it failed a gate, and letting
        it hold a cell would send the next generation to build on a broken foundation."""
        built = grid.build([
            self._entry("broken", "1-1-0-0-1-0-0", 0.90, viable=False),
            self._entry("fine", "1-1-0-0-1-0-0", 0.40),
        ])
        assert built.elite(grid.parse("1-1-0-0-1-0-0")).hash == "fine"

    def test_the_judge_decides_a_cell_the_index_cannot(self) -> None:
        """Inside a cell, the order of the three tests is the whole design."""
        from chess_variant_sandbox.llm.compare import Comparison
        from chess_variant_sandbox.loop.preference import fit

        entries = [self._entry("a", "1-1-0-0-1-0-0", 0.50), self._entry("b", "1-1-0-0-1-0-0", 0.52)]
        # Overlapping intervals: the index has no opinion about which of these is better.
        assert not entries[1].beats(entries[0])

        preference = fit([
            Comparison(first="a", second="b", winner="a", agreed=True) for _ in range(8)
        ])
        assert grid.build(entries).elite(grid.parse("1-1-0-0-1-0-0")).hash == "b"
        assert grid.build(entries, prefer=preference).elite(
            grid.parse("1-1-0-0-1-0-0")
        ).hash == "a"

    def test_an_index_that_does_separate_outranks_the_judge(self) -> None:
        """Measurement first: an opinion breaks a tie, it does not overturn evidence."""
        from chess_variant_sandbox.llm.compare import Comparison
        from chess_variant_sandbox.loop.preference import fit

        weak = Entry(hash="a", name="a", generation=0, parent=None, status="evaluated",
                     index=0.20, objective=0.20, viable=True, cell="1-1-0-0-1-0-0",
                     index_ci_low=0.18, index_ci_high=0.22)
        strong = Entry(hash="b", name="b", generation=0, parent=None, status="evaluated",
                       index=0.80, objective=0.80, viable=True, cell="1-1-0-0-1-0-0",
                       index_ci_low=0.78, index_ci_high=0.82)
        preference = fit([
            Comparison(first="a", second="b", winner="a", agreed=True) for _ in range(8)
        ])
        assert grid.build([weak, strong], prefer=preference).elite(
            grid.parse("1-1-0-0-1-0-0")
        ).hash == "b"

    def test_empty_neighbours_are_one_step_away(self) -> None:
        built = grid.build([self._entry("a", "1-1-0-0-1-0-0", 0.5)])
        neighbours = built.empty_neighbours(grid.parse("1-1-0-0-1-0-0"))

        assert grid.parse("1-1-1-0-1-0-0") in neighbours, "one axis changed"
        assert grid.parse("2-2-0-0-1-0-0") not in neighbours, "two axes changed"
        assert grid.parse("1-1-0-0-1-0-0") not in neighbours, "the cell is not its own neighbour"
        assert all(sum(1 for x, y in zip(n, grid.parse("1-1-0-0-1-0-0"), strict=True)
                       if x != y) == 1 for n in neighbours)

    def test_a_filled_neighbour_is_not_a_target(self) -> None:
        built = grid.build([
            self._entry("a", "1-1-0-0-1-0-0", 0.5),
            self._entry("b", "1-1-1-0-1-0-0", 0.5),
        ])
        assert grid.parse("1-1-1-0-1-0-0") not in built.empty_neighbours(grid.parse("1-1-0-0-1-0-0"))

    def test_the_archive_reports_its_grid(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        archive.record(self._entry("a", "1-1-0-0-1-0-0", 0.5))
        archive.record(self._entry("b", "0-1-0-0-0-0-0", 0.4))

        assert Archive(tmp_path).grid().filled == 2
        assert "kinds of game found" in archive.summary()

    def test_an_archive_written_before_the_grid_still_loads(self, tmp_path) -> None:
        """Entries with no cell are simply not in the grid, rather than crashing it."""
        archive = Archive(tmp_path)
        archive.record(Entry(hash="old", name="old", generation=0, parent=None,
                             status="evaluated", index=0.5, objective=0.5, viable=True))
        assert archive.grid().filled == 0
        assert archive.leaderboard()[0].hash == "old"
