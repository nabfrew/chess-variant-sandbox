"""Tier one and tier two on a game in rounds, a spy game and a game with lines of sight (Q1000).

`site/legality.py` walked games that alternate one move at a time, and nothing else. A
round game's upload (schema 26, measured in the designer's browser since Q998) failed its
walk with the core's `ValueError` about `apply_round`, and the job was marked failed; a spy
game's (schema 27) failed the same way; and a game with a line of sight (schema 30) passed,
but stored none of the sight tallies the native pass writes (`analysis.SIGHT_SERIES`), so
the spot check then found the stored analysis short of its own replay. The claims here:

* **each kind passes the walk and is stored as the native measurement would store it**:
  every file the browser gathered -- `games.jsonl` with its per-ply series written twice a
  round, `analysis.json` with its round, spy and sight tallies, the index -- is the
  server's, to the byte (`TestAnUploadOfEachKind`);
* **the walk reads a round and a spy step as the pass does**, field by field, and draws the
  same blunder trials (`TestTheWalkReadsEachKindAsThePassDoes`);
* **a forgery in a round or a spy step is refused by name**, with the game and the ply
  (`TestTheRefusalsInSteps`);
* **the spot check replays each kind and finds it equal**, without a database
  (`TestTheSpotCheckOfEachKind`) and through the job itself (`TestTheJobOfEachKind`, which
  needs a real Postgres and skips with its reason without one).

Every upload is made the way a browser makes one (`test_site_legality.measure_as_a_browser`):
each phase's units through B0's request text, gathered in part order. The specs are the
small test games of their schemas, cut to forty plies.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import site_store
from test_site_legality import Measured, lines, measure_as_a_browser

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.measurement import LEGAL, Basis
from chess_variant_sandbox.site import db, legality, spotcheck
from chess_variant_sandbox.site.legality import Refused
from chess_variant_sandbox.site.provisional import SPOT_CHECKED
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.tournament import analysis as analysis_pass
from chess_variant_sandbox.tournament import slices
from chess_variant_sandbox.tournament.analysis import CorpusAnalysis
from chess_variant_sandbox.tournament.play import play_game

HERE = Path(__file__).resolve().parent
FILES = ("games.jsonl", "tournament.json", "analysis.json")

ROUND_PLAYERS = ("random", "greedy", "matrix:nodes=100")
TURN_PLAYERS = ("random", "greedy", "minimax:nodes=100")

# Each kind: its test spec, its players, and the tallies its `analysis.json` carries.
KINDS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "rounds": ("_simultaneous_gardner", ROUND_PLAYERS, ("rounds",)),
    "spy": ("_spy_small", TURN_PLAYERS, ("rounds", "spy")),
    "sight": ("_sight_contacts", TURN_PLAYERS, ("sight",)),
    "sight-in-rounds": ("_sight_rounds", ROUND_PLAYERS, ("rounds", "sight")),
    "sight-with-a-spy": ("_sight_spy", TURN_PLAYERS, ("rounds", "spy", "sight")),
}
TALLIES = ("rounds", "spy", "sight")
STEPPED = ("rounds", "spy")


def spec_of(kind: str) -> dict[str, Any]:
    name = KINDS[kind][0]
    spec: dict[str, Any] = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    # A spy game's turns are probed in Python (`analysis._score_roots`), so its games are
    # cut shorter, to keep a spot check's replay of them under the fast suite's limit.
    spec["rules"]["max_plies"] = 26 if "spy" in kind else 40
    return spec


def roster_of(kind: str) -> Any:
    # A spy game's graded games are its dearest to analyse, so it plays one a pairing.
    return slices.canonical_config(
        agents=KINDS[kind][1], games_per_pairing=1 if "spy" in kind else 2, seed=0, workers=1
    )


_MEASURED: dict[str, Measured] = {}


@pytest.fixture(scope="module", params=list(KINDS))
def measured(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Measured:
    """Each kind, measured once as a browser would."""
    kind = str(request.param)
    if kind not in _MEASURED:
        _MEASURED[kind] = measure_as_a_browser(
            spec_of(kind), roster_of(kind), tmp_path_factory.mktemp(f"browser-{kind}")
        )
    return _MEASURED[kind]


def kind_of(measured: Measured) -> str:
    return next(kind for kind in KINDS if spec_of(kind)["name"] == measured.spec["name"])


def refused(measured: Measured, text: str) -> Refused:
    with pytest.raises(Refused) as caught:
        measured.check(text)
    return caught.value


# ------------------------------------------------------------------ the honest case


class TestAnUploadOfEachKind:
    def test_the_measurement_exercises_what_it_is_here_for(self, measured: Measured) -> None:
        """Games long enough to be sampled and trials drawn, so every reading is walked."""
        assert len(measured.rows(slices.BLUNDER)) >= 2
        for row in measured.rows(slices.ANALYSIS):
            for tally in TALLIES:
                assert (tally in row) == (tally in KINDS[kind_of(measured)][2]), tally

    def test_it_passes_and_scores_to_the_browsers_number(self, measured: Measured) -> None:
        scored = legality.score(measured.check(measured.upload()), measured.current.weights_version)
        assert not scored.unscored
        assert scored.headline == measured.index
        assert scored.viable == measured.viable

    def test_every_file_is_the_browsers_to_the_byte(self, measured: Measured) -> None:
        """The series a round writes twice, and every tally the native pass writes."""
        scored = legality.score(measured.check(measured.upload()), measured.current.weights_version)
        index = f"index-{measured.current.weights_version}.json"
        assert set(scored.files) == {*FILES, index}
        for name in (*FILES, index):
            assert scored.files[name] == measured.files[name], name

    def test_forged_tallies_are_ignored(self, measured: Measured) -> None:
        """The tallies are functions of the positions: the walk's stand, the sent are dropped."""
        rows = measured.rows(slices.ANALYSIS)
        for row in rows:
            for tally in TALLIES:
                if isinstance(row.get(tally), dict):
                    row[tally] = {key: 0 for key in row[tally]}
        text = measured.upload(analysis=lines(rows))
        scored = legality.score(measured.check(text), measured.current.weights_version)
        assert scored.files["analysis.json"] == measured.files["analysis.json"]
        assert scored.headline == measured.index


# ------------------------------------------------------------------------ the walk


class TestTheWalkReadsEachKindAsThePassDoes:
    @pytest.mark.parametrize("kind", list(KINDS))
    def test_series_analysis_and_trials_match_the_core_and_the_pass(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Random games beside searching ones, because a random player loses a spy, meets
        a piece and walks into a sight where a searching one may never bother.

        The probe of a turn is stood in for by a flat one: `near_best` is the one field the
        walk does not rebuild, so its values are not compared, and which plies are sampled
        does not depend on them. A round's reading stays the pass's, since a round game's
        trial is drawn from it."""
        monkeypatch.setattr(
            analysis_pass, "_score_roots", lambda engine, evaluator, state, legal: [0.0] * len(legal)
        )
        spec = spec_of(kind)
        engine = Engine(spec)
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        epsilon = analysis_pass._epsilon_for(engine)
        searching = KINDS[kind][1][2]
        pairs = [("greedy", searching), (searching, "greedy")] + [("random", "random")] * 4
        records = [play_game(engine, pair, seed) for seed, pair in enumerate(pairs)]
        trials = 0
        for index, record in enumerate(records):
            walked = legality.walk(
                engine, evaluator, record, phase=slices.TOURNAMENT, game=index, series=True
            )
            assert walked.record.as_dict() == record.as_dict()
            own = analysis_pass._analyse_game(engine, evaluator, record, epsilon)
            assert replace(walked.positions, near_best=own.near_best) == own
            trial = analysis_pass._one_blunder(
                engine, record, analysis_pass.blunder_rng(0, record)
            )
            side = legality.trial_side(engine, walked, 0)
            assert (side is None) == (trial is None)
            if trial is not None:
                trials += 1
                assert side == trial.blundering_side
        assert trials, "a trial is drawn somewhere, so the drawing is read"

    def test_a_round_reads_both_seats_and_a_spy_game_its_phases(self) -> None:
        engine = Engine(spec_of("sight-with-a-spy"))
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        found: set[str] = set()
        for seed in range(12):
            record = play_game(engine, ("random", "random"), seed)
            spy = legality.walk(
                engine, evaluator, record, phase=slices.TOURNAMENT, game=0
            ).positions.spy
            assert spy is not None
            found |= {key for key in ("announced_rounds", "blind_rounds") if spy[key]}
            if spy["alternating_plies"]:
                found.add("alternating_plies")
        assert found == {"alternating_plies", "announced_rounds", "blind_rounds"}


# -------------------------------------------------------------------- the refusals


def a_long_game(measured: Measured) -> int:
    return next(
        i for i, row in enumerate(measured.rows(slices.TOURNAMENT)) if len(row["moves"]) > 10
    )


@pytest.fixture(scope="module", params=STEPPED)
def stepped(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Measured:
    kind = str(request.param)
    if kind not in _MEASURED:
        _MEASURED[kind] = measure_as_a_browser(
            spec_of(kind), roster_of(kind), tmp_path_factory.mktemp(f"browser-{kind}")
        )
    return _MEASURED[kind]


class TestTheRefusalsInSteps:
    """A round's two words and a spy game's steps are walked the core's way, and each
    forgery is refused by its own name, naming the game and the ply."""

    def test_an_illegal_word_in_a_step(self, stepped: Measured) -> None:
        rows = stepped.rows(slices.TOURNAMENT)
        game = a_long_game(stepped)
        rows[game]["moves"][9] = 0
        caught = refused(stepped, stepped.upload(tournament=lines(rows)))
        assert caught.name == legality.ILLEGAL_MOVE
        assert caught.game == game and caught.ply == 9
        assert f"game {game} " in caught.sentence

    def test_a_record_cut_inside_a_round(self, stepped: Measured) -> None:
        """A game's record ends half way through a round: one word of two."""
        rows = stepped.rows(slices.TOURNAMENT)
        engine = Engine(stepped.spec)
        game, cut = next(
            (index, ply)
            for index, row in enumerate(rows)
            for ply in [_first_round(engine, row["moves"])]
            if ply is not None
        )
        rows[game]["moves"] = rows[game]["moves"][: cut + 1]
        caught = refused(stepped, stepped.upload(tournament=lines(rows)))
        assert caught.name == legality.UNFINISHED
        assert caught.game == game and caught.ply == cut + 1
        assert "round" in caught.sentence

    def test_steps_after_the_end(self, stepped: Measured) -> None:
        rows = stepped.rows(slices.TOURNAMENT)
        game = a_long_game(stepped)
        rows[game]["moves"] = rows[game]["moves"] + rows[game]["moves"][-2:]
        caught = refused(stepped, stepped.upload(tournament=lines(rows)))
        assert caught.name == legality.AFTER_TERMINAL
        assert caught.game == game and caught.ply == len(stepped.rows(slices.TOURNAMENT)[game]["moves"])

    def test_a_wrong_outcome(self, stepped: Measured) -> None:
        rows = stepped.rows(slices.TOURNAMENT)
        rows[1]["winner"] = 0 if rows[1]["winner"] is None else None
        caught = refused(stepped, stepped.upload(tournament=lines(rows)))
        assert caught.name == legality.WRONG_OUTCOME and caught.game == 1

    def test_trials_the_pass_would_not_draw(self, stepped: Measured) -> None:
        rows = stepped.rows(slices.BLUNDER)
        caught = refused(stepped, stepped.upload(blunder=lines(rows[1:])))
        assert caught.name == legality.BLUNDER


def _first_round(engine: Engine, moves: list[int]) -> int | None:
    """The ply a record's first round starts on, asked of the core, or `None`."""
    state = engine.initial_state()
    ply = 0
    while ply < len(moves):
        if engine.is_round(state):
            return ply
        state = engine.apply(state, moves[ply])
        ply += 1
    return None


# ------------------------------------------------------------------ the spot check


class _Unlimited:
    def check(self) -> None:
        return None


class TestTheSpotCheckOfEachKind:
    def test_its_replay_of_what_tier_one_stored_is_equal(
        self, measured: Measured, tmp_path: Path
    ) -> None:
        """Every game, every analysis and every trial replayed on the server's own core."""
        scored = legality.score(measured.check(measured.upload()), measured.current.weights_version)
        for name, text in scored.files.items():
            (tmp_path / name).write_text(text, encoding="utf-8", newline="")
        stored = spotcheck._Stored.read(tmp_path)
        picks = [g for unit in slices.blunder_units(stored.records, seed=0) for g in unit.games]
        # Every trial, and every game but the graded ones after the first: the analysis of a
        # game is its dearest replay, and the stored analyses are held to the browser's,
        # byte for byte, above.
        graded = [index for index, _ in slices._graded(stored.records)]
        games = sorted(set(range(len(stored.records))) - set(graded[1:]) | set(picks[:1]))
        sample = spotcheck.Sample(games=tuple(games), trials=tuple(picks))
        roster = replace(measured.roster, workers=1)
        said = spotcheck._replay(measured.spec, roster, stored, sample, _Unlimited())  # type: ignore[arg-type]
        assert said.endswith("equal")
        assert isinstance(CorpusAnalysis.load(tmp_path / "analysis.json"), CorpusAnalysis)


# -------------------------------------------------------------------------- the job


@dataclass
class Store:
    connection: Any
    runs: Path
    game: str
    designer: int
    measured: Measured

    def basis(self, game: str, spec: dict[str, Any]) -> Basis:
        assert spec_hash(spec) == game
        return self.measured.current


@pytest.fixture
def store(tmp_path: Path, measured: Measured) -> Iterator[Store]:
    url, schema = site_store.emptied("no database to queue legality jobs in")
    connection = db.connect(url)
    digest = spec_hash(measured.spec)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {schema}")
            cursor.execute(
                "insert into users (handle, email) values (%s, %s) returning id",
                ("ada", "ada@example.org"),
            )
            designer = int(cursor.fetchone()[0])
            cursor.execute(
                "insert into games (hash, spec, name, owner, visibility, origin)"
                " values (%s, %s::jsonb, %s, %s, 'public', 'designed')",
                (digest, json.dumps(measured.spec), measured.spec["name"], designer),
            )
        connection.commit()
        (tmp_path / "runs").mkdir()
        yield Store(connection, tmp_path / "runs", digest, designer, measured)
    finally:
        connection.rollback()
        connection.close()


class TestTheJobOfEachKind:
    def test_an_honest_upload_is_legal_then_spot_checked_with_every_file_kept(
        self, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The job the page's upload files: tier one passes it, tier two replays it equal.

        The check draws three games and two trials here rather than its 32 and 6, to keep a
        spy game's analyses inside the fast suite's limit; what it draws is still its seal's."""
        monkeypatch.setattr(spotcheck, "GAMES", 3)
        monkeypatch.setattr(spotcheck, "TRIALS", 2)
        job = legality.file(
            store.connection, store.runs, game=store.game, submitter=store.designer,
            text=store.measured.upload(), basis=store.basis,
        )
        store.connection.commit()
        finished = legality.run(store.connection, store.runs, job, basis=store.basis)
        store.connection.commit()
        assert finished.state == legality.DONE, finished.error
        result = finished.result
        assert result is not None and result.state == LEGAL
        assert result.headline == store.measured.index
        where = store.runs / "provisional" / str(result.id)
        for name in FILES:
            assert (where / name).read_text(encoding="utf-8") == store.measured.files[name], name

        assert finished.sample is not None
        assert spotcheck.claim(store.connection) == finished.sample
        checked = spotcheck.run(
            store.connection, store.runs, finished.sample, "server", store.basis
        )
        store.connection.commit()
        assert checked.state == spotcheck.DONE, checked.said
        assert checked.said.startswith("spot-checked")
        assert checked.result is not None and checked.result.state == SPOT_CHECKED
