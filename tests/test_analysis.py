"""The engine-derived second pass.

These measurements exist because the index reads persisted records and never touches a
live engine, and three of the things most worth knowing about a game need positions.
The pass computes them once, writes them next to the records, and the scorer reads the
file -- so the tests that matter are about that boundary holding, and about the
measurements meaning what their names say.
"""

from __future__ import annotations

import pytest

from chess_variant_sandbox.metrics.components import _normalised_entropy, compute_components
from chess_variant_sandbox.tournament.analysis import (
    BLUNDER_FRACTION,
    BLUNDER_SAMPLE,
    BlunderTrial,
    CorpusAnalysis,
    analyse_corpus,
)
from chess_variant_sandbox.tournament.play import play_game
from chess_variant_sandbox.tournament.ratings import rate


@pytest.fixture(scope="module")
def corpus(chess_spec, chess):
    """A small real corpus, played once and reused. Tournaments are the expensive part."""
    # `threats=0`, the search before Q502i: on the search after it, the four games' blunder
    # trials happened to come out the same at run seeds 0 and 17, which is what
    # `test_a_different_run_seed_moves_the_trials` needs them not to do.
    player = "minimax:nodes=200,threats=0"
    records = [play_game(chess, (player, player), seed) for seed in range(4)]
    return chess_spec, records


@pytest.fixture(scope="module")
def analysed(corpus) -> CorpusAnalysis:
    """The corpus's second pass with no blunder trials, read by three tests (Q377)."""
    spec, records = corpus
    return analyse_corpus(spec, records, blunder_sample=0, workers=1)


@pytest.fixture(scope="module")
def analysed_with_a_trial(corpus) -> CorpusAnalysis:
    """The same with one blunder trial, read by two tests."""
    spec, records = corpus
    return analyse_corpus(spec, records, blunder_sample=1, workers=1)


class TestSecondPass:
    def test_analyses_only_games_that_were_actually_chosen(self, chess_spec, chess) -> None:
        """Random movers are excluded: their move list says nothing about the game."""
        records = [
            play_game(chess, ("random", "random"), 0),
            play_game(chess, ("minimax:nodes=200", "minimax:nodes=200"), 1),
        ]
        analysis = analyse_corpus(chess_spec, records, blunder_sample=0, workers=1)
        assert len(analysis.games) == 1
        assert "random" not in analysis.games[0].agents

    def test_counts_a_move_for_every_ply(self, corpus, analysed) -> None:
        _, records = corpus
        for game, record in zip(analysed.games, records, strict=False):
            assert sum(game.moves_by_type) == record.plies

    def test_never_reports_more_near_best_moves_than_there_were_moves(self, analysed) -> None:
        for game in analysed.games:
            assert len(game.near_best) == len(game.legal_counts)
            for near, legal in zip(game.near_best, game.legal_counts, strict=True):
                assert 1 <= near <= legal

    def test_round_trips_through_disk(self, analysed_with_a_trial, tmp_path) -> None:
        """The persisted file is the interface the scorer reads, so it has to survive."""
        analysis = analysed_with_a_trial
        path = analysis.save(tmp_path / "analysis.json")
        restored = CorpusAnalysis.load(path)
        assert [g.as_dict() for g in restored.games] == [g.as_dict() for g in analysis.games]
        assert [b.as_dict() for b in restored.blunders] == [
            b.as_dict() for b in analysis.blunders
        ]

    def test_position_keys_are_the_cores_keys_read_back_from_disk(
        self, corpus, analysed, chess, tmp_path
    ) -> None:
        """One key per ply, as the engine's own replay reports it, surviving the file."""
        _, records = corpus
        restored = CorpusAnalysis.load(analysed.save(tmp_path / "analysis.json"))

        for game, record in zip(restored.games, records, strict=True):
            state = chess.initial_state()
            replayed = []
            for move in record.moves:
                state = chess.apply(state, move)
                replayed.append(state.key)
            assert game.position_keys == replayed
            assert len(game.position_keys) == record.plies

    def test_an_analysis_written_before_position_keys_still_loads(
        self, corpus, tmp_path
    ) -> None:
        import json

        spec, records = corpus
        data = analyse_corpus(spec, records[:1], blunder_sample=0, workers=1).as_dict()
        for game in data["games"]:
            del game["position_keys"]
        path = tmp_path / "analysis.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        restored = CorpusAnalysis.load(path)
        assert restored.games and all(g.position_keys == [] for g in restored.games)

    @pytest.mark.slow
    def test_blunder_trial_diverges_from_the_game_it_came_from(self, corpus) -> None:
        spec, records = corpus
        analysis = analyse_corpus(spec, records, blunder_sample=2, seed=3, workers=1)
        assert analysis.blunders
        for trial in analysis.blunders:
            source = next(r for r in records if r.seed == trial.seed)
            assert trial.ply == int(source.plies * BLUNDER_FRACTION)
            assert trial.blundering_side in (0, 1)


class TestDerivedComponents:
    """The measurements the second pass exists to make possible."""

    def test_components_are_absent_without_an_analysis(self, corpus) -> None:
        """An old corpus is still a valid corpus.

        Missing is reported as missing. A weights version that wants one of these will
        raise by name in ``score_all`` rather than scoring an absent number as zero,
        which would read as "this game is terrible at something we never measured".
        """
        spec, records = corpus
        measured = compute_components(spec, records, rate(records))
        for name in ("decision_criticality", "piece_utilisation", "blunder_recoverability"):
            assert name not in measured.raw

    def test_components_appear_with_one(self, corpus, analysed_with_a_trial) -> None:
        spec, records = corpus
        measured = compute_components(spec, records, rate(records), analysed_with_a_trial)
        assert 0.0 <= measured.raw["decision_criticality"] <= 1.0
        assert 0.0 <= measured.raw["piece_utilisation"] <= 1.0
        assert 0.0 <= measured.raw["blunder_recoverability"] <= 1.0

    def test_utilisation_is_one_when_every_type_moves_equally(self) -> None:
        assert _normalised_entropy([10, 10, 10, 10]) == pytest.approx(1.0)

    def test_utilisation_falls_when_a_type_never_moves(self) -> None:
        """The failure this measures: material added for novelty that nobody uses."""
        used = _normalised_entropy([10, 10, 10, 10])
        idle = _normalised_entropy([30, 8, 1, 1])
        assert idle < used

    def test_utilisation_ignores_how_much_material_there_is(self) -> None:
        """Normalising by log(n) is what makes three types comparable with nine."""
        assert _normalised_entropy([5, 5, 5]) == pytest.approx(
            _normalised_entropy([5] * 9)
        )


class TestBlunderReading:
    """The two ends of the blunder measurement are different failures, not one scale."""

    def _trial(self, blundering: int, new_winner: int | None) -> BlunderTrial:
        return BlunderTrial(
            seed=0,
            agents=("a", "a"),
            ply=10,
            original_winner=0,
            blundering_side=blundering,
            new_winner=new_winner,
        )

    def test_losing_after_the_blunder_is_not_recovery(self) -> None:
        assert not self._trial(blundering=0, new_winner=1).recovered

    def test_holding_the_draw_counts_as_recovery(self) -> None:
        assert self._trial(blundering=0, new_winner=None).recovered

    def test_still_winning_counts_as_recovery(self) -> None:
        assert self._trial(blundering=0, new_winner=0).recovered


class TestTheSampleSizeIsOneNumber:
    """`--blunders` is the campaign's sample or the run it rewrites is not comparable.

    `analyse` writes `analysis.json` in place, and `measurement`'s basis records the spec,
    the weights, the roster and the sources -- not how many blunder trials the file behind
    the score was drawn on. So a hand-run `chess-variant-sandbox analyse` at a different count replaces a
    campaign's `blunder_recoverability` with a number taken another way and leaves nothing
    that says so. The CLI default was 20 against `BLUNDER_SAMPLE`'s 24 for as long as both
    existed, which is what a whole afternoon of `v5` archaeology eventually found
    (`docs/archive/bugs-resolved.md`, the `v5` entry).

    Equal values are not the point; one *name* is. These tests fail on a copy that happens
    to agree as well as on one that has drifted.
    """

    def test_the_cli_default_is_the_analysis_constant(self) -> None:
        from chess_variant_sandbox import cli

        assert cli.BLUNDER_SAMPLE == BLUNDER_SAMPLE
        args = cli._build_parser().parse_args(["analyse", "runs/whatever"])
        assert args.blunders == BLUNDER_SAMPLE

    def test_the_default_reads_that_name_rather_than_a_literal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Move the constant and the flag moves with it, or the flag is a copy."""
        from chess_variant_sandbox import cli

        monkeypatch.setattr(cli, "BLUNDER_SAMPLE", BLUNDER_SAMPLE + 7)
        args = cli._build_parser().parse_args(["analyse", "runs/whatever"])
        assert args.blunders == BLUNDER_SAMPLE + 7

    def test_the_campaign_path_takes_the_same_number(self) -> None:
        """The third place the sample could drift: `analyse_corpus`' own default, which
        every campaign path calls without passing one."""
        import inspect

        parameter = inspect.signature(analyse_corpus).parameters["blunder_sample"]
        assert parameter.default == BLUNDER_SAMPLE


def test_blunder_sample_is_48() -> None:
    """Repair E, `v11` A5 (Q388): twice the trials, priced in
    `docs/reference/ladder-statistic.md` §4 and §6.

    `BLUNDER_SAMPLE` moved from 24 to 48 -- the index's run-to-run sd at × 0.948 of
    today's, and `blunder_recoverability`'s K from 1.31 to about 1.8. This pins the value
    itself, since the tests above pin only that every caller reads the same name; a copy
    that agreed with a drifted constant would still pass them.
    """
    assert BLUNDER_SAMPLE == 48


class TestOnePositionCalls:
    """The core's answers to a single position, against the Python they replaced.

    The analysis pass is a search over one sampled position at a time, and the boundary
    could not serve that shape until ``root_scores`` existed -- so the pass ran the
    Python probe, at four times the cost, using a *different* implementation from the
    one that played the games it was measuring. These hold the replacement to producing
    the same numbers.
    """

    def test_the_core_probe_matches_the_python_probe(self, chess, chess_spec) -> None:
        from chess_variant_sandbox.agents.evaluation import Evaluator
        from chess_variant_sandbox.tournament.analysis import (
            PROBE_MAX_DEPTH,
            PROBE_NODES,
            _root_scores,
        )
        from chess_variant_sandbox.tournament.play import eval_values

        evaluator = Evaluator.for_engine(chess, mobility_weight=0.0)
        values = eval_values(chess.spec)
        record = play_game(chess, ("minimax:nodes=200", "minimax:nodes=200"), 5)

        state = chess.initial_state()
        compared = 0
        for index, move in enumerate(record.moves[:24]):
            if index % 8 == 0:
                legal = chess.legal_moves(state)
                if len(legal) > 1:
                    native = chess.root_scores(state, values, PROBE_NODES, PROBE_MAX_DEPTH)
                    assert native == _root_scores(chess, evaluator, state, legal)
                    compared += 1
            state = chess.apply(state, move)
        assert compared >= 2

    def test_one_move_agrees_with_the_agent_that_would_have_played_it(self, chess) -> None:
        """The point of the call: the opponent you play is the player the index rated."""
        from chess_variant_sandbox.agents import build_agent
        from chess_variant_sandbox.tournament.play import choose_move, seat_rng

        state = chess.initial_state()
        for ply, agent in enumerate(("greedy", "minimax:nodes=200", "random", "greedy")):
            legal = chess.legal_moves(state)
            native = choose_move(chess, agent, state, seed=7 + ply)
            oracle = build_agent(agent, chess).select_move(state, legal, seat_rng(7 + ply, 0))
            assert native == oracle
            state = chess.apply(state, native)

    def test_finishing_from_the_start_is_the_whole_game(self, chess) -> None:
        """``play_from`` and ``play_game`` are the same loop, so they must not diverge."""
        from chess_variant_sandbox.tournament.play import finish_game

        played = play_game(chess, ("minimax:nodes=200", "greedy"), 11)
        resumed = finish_game(chess, ("minimax:nodes=200", "greedy"), chess.initial_state(), 11)
        assert resumed.moves == played.moves
        assert (resumed.winner, resumed.reason) == (played.winner, played.reason)

    def test_a_resumed_game_matches_the_python_loop(self, chess) -> None:
        """The oracle for ``play_from``: the Python game loop over the same position."""
        from chess_variant_sandbox.tournament.play import _finish_game_python, finish_game

        state = chess.initial_state()
        for move in play_game(chess, ("greedy", "greedy"), 2).moves[:10]:
            state = chess.apply(state, move)

        native = finish_game(chess, ("minimax:nodes=200", "greedy"), state, 4)
        python = _finish_game_python(chess, ("minimax:nodes=200", "greedy"), state, 4)
        assert native.moves == python.moves
        assert (native.winner, native.reason) == (python.winner, python.reason)

    def test_no_legal_move_is_reported_as_no_move(self, chess) -> None:
        """A terminal position is a question for ``terminal``, not an error here."""
        from chess_variant_sandbox.tournament.play import choose_move

        # Walk a game to its end, then ask.
        record = play_game(chess, ("greedy", "greedy"), 1)
        state = chess.initial_state()
        for move in record.moves:
            state = chess.apply(state, move)
        if not chess.legal_moves(state):
            assert choose_move(chess, "greedy", state) is None


class TestWorkerIndependence:
    """A measurement that depends on how the work was divided is a bug.

    The blunder trials used to draw their forced move from one stream walked in order
    when serial, and from a per-record stream when parallel, so the same corpus analysed
    two ways picked different moves and could score differently -- the same class of bug
    as a tournament whose result depends on its thread count.
    """

    def test_the_forced_move_is_fixed_by_the_record_not_by_the_order(
        self, corpus, chess
    ) -> None:
        """Every trial's stream is a pure function of (seed, record).

        Asserted on the derivation rather than by running a worker pool, because that is
        what actually has to hold: given it, the serial loop and the pool cannot disagree
        however the games are split between them.
        """
        from chess_variant_sandbox.tournament.analysis import _one_blunder, blunder_rng

        spec, records = corpus
        analysis = analyse_corpus(spec, records, blunder_sample=4, seed=0, workers=1)
        assert analysis.blunders

        by_seed = {trial.seed: trial for trial in analysis.blunders}
        for record in reversed(records):
            if record.seed not in by_seed:
                continue
            # Same record, same seed, computed in the opposite order: same trial.
            again = _one_blunder(chess, record, blunder_rng(0, record))
            assert again is not None
            assert again.as_dict() == by_seed[record.seed].as_dict()

    def test_position_keys_do_not_depend_on_the_worker_count(
        self, chess_spec, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Serial and pooled analyses of one corpus persist identical keys.

        Eight games is the pool's threshold, so two workers really split the work.
        Greedy players on the smallest committed chess-shaped board keep it short.

        A pool that dies falls back to one process and says so, and then the two arms
        would agree without a pool ever having run -- which is what a pool nested under
        an xdist worker could do unnoticed. So the test also asserts it did not (Q377c).
        """
        from pathlib import Path

        from chess_variant_sandbox.engine import Engine
        from chess_variant_sandbox.spec.canonical import load_spec

        rulesets = Path(__file__).resolve().parents[1] / "rulesets"
        spec = load_spec(rulesets / "gardner.json")
        engine = Engine(spec)
        records = [play_game(engine, ("greedy", "greedy"), seed) for seed in range(8)]

        serial = analyse_corpus(spec, records, blunder_sample=0, workers=1)
        capsys.readouterr()
        pooled = analyse_corpus(spec, records, blunder_sample=0, workers=2)
        assert "falling back" not in capsys.readouterr().out, "the pool died; nothing was pooled"
        keys = [g.position_keys for g in serial.games]
        assert all(keys)
        assert [g.position_keys for g in pooled.games] == keys

    def test_a_different_run_seed_moves_the_trials(self, corpus) -> None:
        """The counterpart: the stream is fixed by the seed, not constant."""
        spec, records = corpus
        first = analyse_corpus(spec, records, blunder_sample=4, seed=0, workers=1)
        second = analyse_corpus(spec, records, blunder_sample=4, seed=17, workers=1)
        assert first.as_dict() != second.as_dict()
