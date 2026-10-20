"""`Basis.players`: the agents are read by what they hand the core, not by their source (Q631).

`docs/reference/compute-economy.md` §2.3. Before this, `agents/*.py` sat in the `engine` source set, so
any code edit to any agent module -- `nets.py`, `heuristics.py`, `valuations.py`, the
fit-making half of `fitted.py` (since removed, Q558), none of which a rated roster
reaches -- staled the whole library by replay. batch-0925 paid 91 minutes for 0 file
differences in 280 runs.

Both directions are held here, against specs:

* an edit to every agent module but the probe's ruler moves no field the library's stamp
  is compared on, and the stamp still reads current;
* a change to what a rated rung is handed -- one seat's parameter, or the piece values --
  moves `players` and stales the stamp by replay.
"""

from __future__ import annotations

import json
import shutil

import pytest

from chess_variant_sandbox import corpus, measurement
from chess_variant_sandbox.measurement import Basis
from chess_variant_sandbox.spec import spec_hash

# A committed spec, on which every canonical rung resolves.
SPEC_PATH = measurement.ROOT / "rulesets" / "gardner.json"

_CACHED = (
    measurement.source_fingerprint,
    measurement.bytes_fingerprint,
    measurement.compile_source_key,
    measurement.values_source_key,
    measurement.code_fingerprint,
)


def _spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _forget(sources: bool = False) -> None:
    """A fresh process's view of the answers, and of the sources too where asked."""
    if sources:
        for cached in _CACHED:
            cached.cache_clear()
    measurement._TABLES.clear()
    measurement._DISK.update(path=None, key=None, tables={})
    measurement._VALUES.clear()
    measurement._VALUES_DISK.update(path=None, key=None, tables={})
    measurement._LAST_COMPILED.clear()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv(measurement.TABLES_CACHE_ENV, str(tmp_path / "cache" / "tables.json"))
    _forget()
    yield
    monkeypatch.undo()
    _forget()


def _stamp(spec: dict) -> Basis:
    return corpus.current_basis(spec_hash(spec), spec=spec)


class TestAnAgentsEditThatHandsTheCoreNothingNewStalesNothing:
    def test_no_compared_source_set_reads_an_agent_module_but_the_ruler(self) -> None:
        """The fast form of the test below: the sets themselves.

        `agents` is read only where the players cannot be, and the compile key only keys a
        cache whose answers are the compiler's own.
        """
        for kind in ("engine", "metrics", "analysis"):
            read = {
                path.name for path in measurement.fingerprinted(kind)
                if path.parent.name == "agents"
            }
            assert read <= {"evaluation.py"}, kind

    @pytest.mark.slow
    def test_every_agent_module_but_the_ruler_edited_moves_no_compared_field(
        self, tmp_path, monkeypatch
    ) -> None:
        """A code edit, not a comment, to each module; the stamp's compared fields hold.

        The tree is copied and edited on disk, which is what every source fingerprint
        reads. The modules already imported are the ones that play, and they did not
        change, so this proves the other half: no source set the stamp compares reads
        `agents/` any more. `TestAChangeToTheHandoffStalesIt` proves the digest does read
        what the agents hand the core.
        """
        spec = _spec()
        before = _stamp(spec)
        assert before.players

        copy = tmp_path / "tree"
        package = measurement.ROOT / "chess_variant_sandbox"
        for path in package.rglob("*.py"):
            target = copy / path.relative_to(measurement.ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        edited = []
        for path in sorted((copy / "chess_variant_sandbox" / "agents").glob("*.py")):
            if path.name == "evaluation.py":
                continue  # the probe's ruler: the analysis pass reads it (below)
            path.write_text(
                path.read_text(encoding="utf-8") + "\n_Q631_EDIT = 1\n", encoding="utf-8"
            )
            edited.append(path.name)
        assert {"nets.py", "heuristics.py", "valuations.py"} <= set(edited)

        monkeypatch.setattr(measurement, "ROOT", copy)
        _forget(sources=True)
        try:
            after = _stamp(spec)
        finally:
            monkeypatch.undo()
            _forget(sources=True)

        assert after.agents_source != before.agents_source  # the edit was seen
        for field in ("engine", "metrics", "analysis", "tables", "players", "net"):
            assert getattr(after, field) == getattr(before, field), field
        assert before.stale_against(after).current

    def test_the_probes_ruler_is_read_by_the_analysis_pass_and_not_the_engine(self) -> None:
        """`agents/evaluation.py` left `engine` with the rest; the pass reads it, so an edit
        to it is a re-analysis (and, as with the rest of the pass, a re-score)."""
        ruler = measurement.ROOT / "chess_variant_sandbox" / "agents" / "evaluation.py"
        assert ruler in measurement.fingerprinted("analysis")
        assert ruler in measurement.fingerprinted("metrics")
        assert ruler not in measurement.fingerprinted("engine")
        assert not any(
            path.parent.name == "agents" for path in measurement.fingerprinted("engine")
        )


class TestAChangeToTheHandoffStalesIt:
    def test_one_rated_rungs_parameter_moves_it(self, monkeypatch) -> None:
        """The immunity weight a minimax rung is sent, changed in the module that sends it."""
        from chess_variant_sandbox.agents import minimax

        spec = _spec()
        before = _stamp(spec)
        monkeypatch.setattr(minimax, "IMMUNE_PIECES", minimax.IMMUNE_PIECES + 0.5)
        _forget()
        after = _stamp(spec)
        assert after.players != before.players
        assert after.engine == before.engine
        stale = before.stale_against(after)
        assert stale.needs == "replay"
        assert "the players the core is handed changed since it was played" in stale.reasons

    def test_the_piece_values_move_it(self, tmp_path, monkeypatch) -> None:
        from chess_variant_sandbox.tournament import play

        spec = _spec()
        before = _stamp(spec)
        real = play.eval_values

        def doubled(compiled):
            return [2.0 * value for value in real(compiled)]

        monkeypatch.setattr(play, "eval_values", doubled)
        # A patch is not a source edit, so the values cache's key cannot see it: start cold.
        monkeypatch.setenv(measurement.TABLES_CACHE_ENV, str(tmp_path / "cold" / "tables.json"))
        _forget()
        assert before.stale_against(_stamp(spec)).needs == "replay"

    def test_a_different_ruleset_is_a_different_handoff(self) -> None:
        other = json.loads((measurement.ROOT / "rulesets" / "dull.json").read_text())
        assert measurement.players_fingerprint(other, corpus.CANONICAL_AGENTS) != (
            measurement.players_fingerprint(_spec(), corpus.CANONICAL_AGENTS)
        )


class TestWhereThePlayersCannotBeRead:
    """`""` taken now: the agents' source decides, which is the rule before Q631."""

    def test_a_seat_that_does_not_resolve_cannot_say(self) -> None:
        # A heuristic digest no store holds, as on the server for a visitor's design.
        assert measurement.players_fingerprint(_spec(), ("minimax:h=000000000000,nodes=100",)) == ""

    def test_a_seat_with_no_native_form_cannot_say(self, monkeypatch) -> None:
        import chess_variant_sandbox.agents as agents

        real = agents.native_spec
        monkeypatch.setattr(
            agents, "native_spec",
            lambda agent, digest="": None if agent == "greedy" else real(agent, digest),
        )
        assert measurement.players_fingerprint(_spec(), ("random", "greedy")) == ""

    def test_the_source_decides_then(self) -> None:
        stored = Basis.from_dict(_stamp(_spec()).as_dict())
        assert stored is not None
        assert stored.players and stored.agents_source
        blind = Basis.current(
            stored.ruleset_hash, stored.weights_version, stored.agents,
            stored.games_per_pairing, stored.seeds, tables=stored.tables,
            verdict_rule=stored.verdict_rule,
        )
        assert blind.players == ""
        assert stored.stale_against(blind).current
        moved = Basis(**{**_fields(blind), "agents_source": "moved"})
        stale = stored.stale_against(moved)
        assert stale.needs == "replay"
        assert any("cannot be read on both sides" in reason for reason in stale.reasons)


def _fields(basis: Basis) -> dict:
    from dataclasses import fields

    return {f.name: getattr(basis, f.name) for f in fields(basis)}


class TestTheStamp:
    def test_it_round_trips(self) -> None:
        stamp = _stamp(_spec())
        assert Basis.from_dict(stamp.as_dict()) == stamp
        assert stamp.as_dict()["players"] == stamp.players

    def test_a_stamp_from_before_the_field_is_a_replay(self) -> None:
        stamp = _stamp(_spec()).as_dict()
        del stamp["players"], stamp["agents_source"]
        old = Basis.from_dict(stamp)
        assert old is not None
        stale = old.stale_against(_stamp(_spec()))
        assert stale.needs == "replay"
        assert "it does not record what the core was handed to play" in stale.reasons

    def test_a_check_that_finds_nothing_changed_compiles_once(self, monkeypatch) -> None:
        """`tables` and the values share one compile, and the disk cache spares the next."""
        from chess_variant_sandbox.engine import compile as compile_module

        calls = []
        real = compile_module.compile_spec

        def counted(spec):
            calls.append(1)
            return real(spec)

        monkeypatch.setattr(compile_module, "compile_spec", counted)
        from chess_variant_sandbox.tournament import play

        monkeypatch.setattr(play, "_EVAL_VALUES", {})
        first = _stamp(_spec())
        assert len(calls) == 1
        _forget()
        assert _stamp(_spec()) == first
        assert len(calls) == 1


def test_the_values_cache_moves_with_the_tables_cache(tmp_path) -> None:
    """The values cache sits beside the tables cache, so one variable moves both."""
    assert measurement.values_cache_path().parent == measurement.tables_cache_path().parent
    assert measurement.tables_cache_path().is_relative_to(tmp_path)
